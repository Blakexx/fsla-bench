"""Recurrent (decode-step) benchmark — forward only.

The autoregressive-inference regime: one token at a time, no pairwise content Gram, so the
chunked form's shared-Gram savings do NOT apply. The prediction is bandwidth PARITY at matched
state (state bytes are equal by construction): RoLA realized as virtual-heads over the canonical
`fused_recurrent_simple_gla` should match the wide monolith on the same kernel, with vh adding
only broadcast/materialization overhead. No new kernel surface — every arm runs canonical fla.

Decode has no backward, so this reports forward ms/token + recurrent-state MiB only:
  variant, nc, arm, ms_per_tok, state_mib
arms: rola-vh (global + kappa combine) and the matched wide monolith, for RLA + scalar-GLA.

`--check` asserts step-by-step decode == the verified chunked routed kernel on the same inputs
(rel < 3e-2 in bf16) before trusting the timings. `--bench` produces the table (idle GPU).

Run:  python -m rola_bench.perf.recurrent --check        # correctness
      python -m rola_bench.perf.recurrent --bench [--out DIR]
"""
import argparse
from pathlib import Path

import torch

from fla_rola.ops.simple_gla import chunk_simple_gla
from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla_rola.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from fla_rola.ops.linear_attn import fused_recurrent_linear_attn
from fla_rola.modules.feature_map import HedgehogFeatureMap

from . import _common as C


def make_inputs(L, nc, gla, seed=0):
    g = torch.Generator(device=C.DEV).manual_seed(seed)
    t = lambda *s: torch.randn(*s, device=C.DEV, generator=g, dtype=torch.float32)
    q, k = t(C.B, L, C.H, C.DQK).abs(), t(C.B, L, C.H, C.DQK).abs()
    v = t(C.B, L, C.H, C.DV)
    r = torch.softmax(t(C.B, L, C.H, nc), -1)
    w = torch.softmax(t(C.B, L, C.H, nc), -1)
    ld = (-torch.rand(C.B, L, C.H, nc, device=C.DEV, generator=g) * 0.5).clamp(min=-2.5) if gla else None
    return q, k, v, r, w, ld


def vh_expand(q, k, v, w, ld, nc):
    """[B,L,H,*] -> virtual-head [B,L,H*nc,*]; v carries the write gate and an appended den column."""
    Bq, L = q.shape[0], q.shape[1]
    qv = q.unsqueeze(3).expand(Bq, L, C.H, nc, C.DQK).reshape(Bq, L, C.H * nc, C.DQK)
    kv = k.unsqueeze(3).expand(Bq, L, C.H, nc, C.DQK).reshape(Bq, L, C.H * nc, C.DQK)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)  # den column
    vv = (v1.unsqueeze(3) * w.unsqueeze(-1)).reshape(Bq, L, C.H * nc, C.DV + 1)
    gv = ld.reshape(Bq, L, C.H * nc).float() if ld is not None else None
    return qv, kv, vv, gv


def vh_combine(o_aug, r, nc, norm, kappa=0.5, eps=1e-5):
    """o_aug:[B,L,H*nc,DV+1] per-state (num | den) readouts -> combined [B,L,H,DV]."""
    Bq, L = o_aug.shape[0], o_aug.shape[1]
    o = o_aug.view(Bq, L, C.H, nc, C.DV + 1)
    num, den = o[..., :C.DV], o[..., C.DV]
    if norm == "kappa":
        r = r * (den.abs() + eps).pow(-kappa)
    elif norm == "per_state":
        r = r / (den.abs() + eps)
    o_num = (num * r.unsqueeze(-1)).sum(3)
    o_den = (den * r).sum(3)
    return o_num / (o_den.unsqueeze(-1) + eps)


def decode_routed_step(qt, kt, vt, rt, wt, ldt, state, nc, norm):
    """One decode step (L=1), canonical fused_recurrent under the hood."""
    qv, kv, vv, gv = vh_expand(qt, kt, vt, wt, ldt, nc)
    o, state = fused_recurrent_simple_gla(qv, kv, vv, g=gv, scale=1.0,
                                          initial_state=state, output_final_state=True)
    return vh_combine(o.float(), rt, nc, norm), state


def decode_monolith_step(qt, kt, vt, state, g=None):
    o, state = fused_recurrent_simple_gla(qt, kt, vt, g=g, scale=1.0,
                                          initial_state=state, output_final_state=True)
    return o, state


def decode_gdn_step(qt, kt, vt, gt, beta, state):
    """GatedDeltaNet decode step (L=1). use_qk_l2norm_in_kernel normalizes q/k internally."""
    o, state = fused_recurrent_gated_delta_rule(qt, kt, vt, g=gt, beta=beta, scale=1.0,
                                                initial_state=state, output_final_state=True,
                                                use_qk_l2norm_in_kernel=True)
    return o, state


def decode_hedgehog_step(qt, kt, vt, fmap, state):
    """Hedgehog decode step (L=1): feature map on q/k, then normalized linear-attn recurrence."""
    o, state = fused_recurrent_linear_attn(fmap(qt), fmap(kt), vt, scale=1.0,
                                           initial_state=state, output_final_state=True, normalize=True)
    return o, state


def check(nc=16, L=64, gla=True, norm="global"):
    """Step-by-step decode must equal the verified chunked routed kernel on the same inputs."""
    q, k, v, r, w, ld = make_inputs(L, nc, gla)
    qv, kv, vv, gv = vh_expand(q, k, v, w, ld, nc)
    o_ref = vh_combine(chunk_simple_gla(qv, kv, vv, g=gv, scale=1.0)[0].float(), r, nc, norm)
    state, outs = None, []
    for ti in range(L):
        sl = slice(ti, ti + 1)
        o_t, state = decode_routed_step(q[:, sl], k[:, sl], v[:, sl], r[:, sl], w[:, sl],
                                        ld[:, sl] if ld is not None else None, state, nc, norm)
        outs.append(o_t)
    o_dec = torch.cat(outs, 1)
    rel = (o_dec - o_ref).abs().max().item() / (o_ref.abs().max().item() + 1e-9)
    ok = rel < 3e-2
    print(f"{'decode==chunked ' + ('GLA' if gla else 'RLA') + f' nc={nc} norm={norm}':48s} "
          f"rel {rel:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def run_check():
    ok = True
    for gla in (False, True):
        for nc in (16, 64):
            for norm in ("global", "kappa", "per_state"):
                ok &= check(nc=nc, gla=gla, norm=norm)
    print("ALL PASS" if ok else "FAILURES")
    return ok


def _mib(*dims):
    n = 1
    for d in dims:
        n *= d
    return round(n * 4 / 2 ** 20, 1)


def _time_cell(meta, fn, steps, warmup, rows):
    """Time one decode cell; on OOM / any error record a sentinel ms_per_tok + recover (one bad
    shape never aborts the whole sweep). meta carries variant/nc/shape/arm/state_mib."""
    try:
        ms = C.time_forward(fn, steps, warmup)
        rows.append({**meta, "ms_per_tok": round(ms, 3)})
        print(rows[-1], flush=True)
    except torch.cuda.OutOfMemoryError:
        rows.append({**meta, "ms_per_tok": "OOM"}); torch.cuda.empty_cache()
        print(f"OOM  {meta}", flush=True)
    except Exception as e:  # noqa: BLE001
        rows.append({**meta, "ms_per_tok": "ERR"}); torch.cuda.empty_cache()
        print(f"ERR  {meta}: {type(e).__name__}: {e}", flush=True)


def bench(steps, warmup, out):
    """Decode ms/token + recurrent-state MiB, swept over the full nc scan AND shape (wide/square/
    heads), mirroring the chunked bench's arm set. State is NOT threaded across reps: decode cost is
    step-invariant for these kernels, so a representative single step is timed; state_mib reports the
    recurrent footprint. Cells the shape solve declares infeasible (gdn head_dim>256, heads nc>32)
    are skipped. Based has no fused_recurrent kernel in the fork (chunk-only) -> not benched here."""
    rows = []
    cdt = lambda x: x.to(C.DT) if x is not None else None
    rnd = lambda *s, gen=None: torch.randn(*s, device=C.DEV, generator=gen)
    print(f"# recurrent decode | {C.gpu_name()} | B={C.B} H={C.H} d_qk=d_v={C.DQK} bf16 | forward only "
          f"| ms/token (median of {steps}) | shapes wide/square/heads, nc {tuple(C.NCS)}")

    # ---- RoLA-family (native) + monolith (shape sweep): RLA & GLA inner kernels ----
    for gla in (False, True):
        var = "GLA" if gla else "RLA"
        for nc in C.NCS:
            q, k, v, r, w, ld = make_inputs(1, nc, gla, seed=1)
            for arm, norm in (("rola-vh global", "global"), ("rola-vh kappa", "kappa")):
                state = torch.zeros(C.B, C.H * nc, C.DQK, C.DV + 1, device=C.DEV, dtype=torch.float32)
                meta = dict(variant=var, nc=nc, shape="native", arm=arm,
                            state_mib=_mib(C.B, C.H * nc, C.DQK, C.DV + 1))
                _time_cell(meta, lambda norm=norm, st=state: decode_routed_step(
                    cdt(q), cdt(k), cdt(v), r, w, cdt(ld), st, nc, norm), steps, warmup, rows)
            # matched monolith, swept over shape (one dense state per head; GLA adds a scalar gate).
            for shape in ("wide", "square", "heads"):
                d = C.shaped_dims("lin1", shape, nc)
                if d is None:
                    print(f"skip monolith {shape} nc{nc} (infeasible)", flush=True); continue
                H, dk, dv = d["H"], d["d_k"], d["d_v"]
                gm = torch.Generator(device=C.DEV).manual_seed(2)
                qm, km = rnd(C.B, 1, H, dk, gen=gm).abs().to(C.DT), rnd(C.B, 1, H, dk, gen=gm).abs().to(C.DT)
                vm = rnd(C.B, 1, H, dv, gen=gm).to(C.DT)
                gate = torch.nn.functional.logsigmoid(rnd(C.B, 1, H, gen=gm).float()).to(C.DT) if gla else None
                state_m = torch.zeros(C.B, H, dk, dv, device=C.DEV, dtype=torch.float32)
                meta = dict(variant=var, nc=nc, shape=shape, arm="monolith", state_mib=_mib(C.B, H, dk, dv))
                _time_cell(meta, lambda gate=gate, st=state_m: decode_monolith_step(
                    qm, km, vm, st, g=(gate.float() if gate is not None else None)), steps, warmup, rows)

    # ---- cross-architecture decode (own; no RLA/GLA split): gdn + hedgehog × shape ----
    for nc in C.NCS:
        for shape in ("wide", "square", "heads"):
            dg = C.shaped_dims("gdn", shape, nc)
            if dg is not None:
                H, dk, dv = dg["H"], dg["d_k"], dg["d_v"]
                gm = torch.Generator(device=C.DEV).manual_seed(3)
                q1, k1 = rnd(C.B, 1, H, dk, gen=gm).to(C.DT), rnd(C.B, 1, H, dk, gen=gm).to(C.DT)
                v1 = rnd(C.B, 1, H, dv, gen=gm).to(C.DT)
                beta = torch.rand(C.B, 1, H, device=C.DEV, generator=gm).sigmoid().to(C.DT)
                ldg = torch.nn.functional.logsigmoid(rnd(C.B, 1, H, gen=gm).float()).to(C.DT)
                st = torch.zeros(C.B, H, dk, dv, device=C.DEV, dtype=torch.float32)
                meta = dict(variant="own", nc=nc, shape=shape, arm="gdn", state_mib=_mib(C.B, H, dk, dv))
                _time_cell(meta, lambda: decode_gdn_step(q1, k1, v1, ldg, beta, st), steps, warmup, rows)
            dh = C.shaped_dims("lin2", shape, nc)
            if dh is not None:
                H, dk, dv = dh["H"], dh["d_k"], dh["d_v"]
                gm = torch.Generator(device=C.DEV).manual_seed(4)
                q1, k1 = rnd(C.B, 1, H, dk, gen=gm).to(C.DT), rnd(C.B, 1, H, dk, gen=gm).to(C.DT)
                v1 = rnd(C.B, 1, H, dv, gen=gm).to(C.DT)
                fmap = HedgehogFeatureMap(head_dim=dk).to(C.DEV, C.DT)
                feat = fmap(q1).shape[-1]                       # doubled feature dim
                st = torch.zeros(C.B, H, feat, dv, device=C.DEV, dtype=torch.float32)
                meta = dict(variant="own", nc=nc, shape=shape, arm="hedgehog", state_mib=_mib(C.B, H, feat, dv))
                _time_cell(meta, lambda: decode_hedgehog_step(q1, k1, v1, fmap, st), steps, warmup, rows)

    out = Path(out)
    C.write_csv(out / "recurrent_decode.csv", rows)
    print(f"\nWROTE {out}/recurrent_decode.csv  [{C.gpu_name()}]")


# --------------------------------------------------------------------------------------------
# END-TO-END decode: the FUSED single-token layer fwd = in_proj(d_model -> q/k/v/…) -> recurrent
# kernel -> out_proj, timed as one region (forward only; decode has no backward). d_model=DM.
# Mirrors chunked.bench_e2e but for the decode path; same arm set as bench().
# --------------------------------------------------------------------------------------------
def bench_e2e(steps, warmup, out, DM=512):
    import torch.nn as nn
    rows = []
    def lin(i, o): return nn.Linear(i, o, bias=False).to(C.DEV, C.DT)
    def t1(*s): return torch.randn(*s, device=C.DEV, dtype=C.DT)
    print(f"\n# recurrent E2E decode (fused layer, fwd only) | {C.gpu_name()} | d_model={DM} "
          f"| ms/token (median of {steps})")

    def time_cell(meta, fwd):
        try:
            ms = C.time_forward(fwd, steps, warmup)
            rows.append({**meta, "e2e_ms_per_tok": round(ms, 3)}); print(rows[-1], flush=True)
        except torch.cuda.OutOfMemoryError:
            rows.append({**meta, "e2e_ms_per_tok": "OOM"}); torch.cuda.empty_cache(); print(f"OOM {meta}", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({**meta, "e2e_ms_per_tok": "ERR"}); torch.cuda.empty_cache(); print(f"ERR {meta}: {type(e).__name__}: {e}", flush=True)

    for gla in (False, True):
        var = "GLA" if gla else "RLA"
        for nc in C.NCS:
            B = C.B
            x = t1(B, 1, DM)
            # rola-vh (native): project q/k/v/r/w(/ld), decode, out_proj
            qp, kp, vp = lin(DM, C.H*C.DQK), lin(DM, C.H*C.DQK), lin(DM, C.H*C.DV)
            rp, wp = lin(DM, C.H*nc), lin(DM, C.H*nc)
            ldp = lin(DM, C.H*nc) if gla else None
            op = lin(C.H*C.DV, DM)
            for arm, norm in (("rola-vh global", "global"), ("rola-vh kappa", "kappa")):
                state = torch.zeros(B, C.H*nc, C.DQK, C.DV+1, device=C.DEV, dtype=torch.float32)
                def fwd(norm=norm, st=state):
                    q = qp(x).view(B, 1, C.H, C.DQK); k = kp(x).view(B, 1, C.H, C.DQK); v = vp(x).view(B, 1, C.H, C.DV)
                    r = torch.softmax(rp(x).view(B, 1, C.H, nc).float(), -1).to(C.DT)
                    w = torch.softmax(wp(x).view(B, 1, C.H, nc).float(), -1).to(C.DT)
                    ld = ldp(x).view(B, 1, C.H, nc) if ldp is not None else None
                    o, _ = decode_routed_step(q, k, v, r, w, ld, st, nc, norm)
                    return op(o.reshape(B, 1, -1).to(C.DT))
                time_cell(dict(variant=var, nc=nc, shape="native", arm=arm), fwd)
            # monolith shapes
            for shape in ("wide", "square", "heads"):
                d = C.shaped_dims("lin1", shape, nc)
                if d is None: continue
                H, dk, dv = d["H"], d["d_k"], d["d_v"]
                mqp, mkp, mvp = lin(DM, H*dk), lin(DM, H*dk), lin(DM, H*dv)
                mgp = lin(DM, H) if gla else None; mop = lin(H*dv, DM)
                st = torch.zeros(B, H, dk, dv, device=C.DEV, dtype=torch.float32)
                def fwd(H=H, dk=dk, dv=dv, mqp=mqp, mkp=mkp, mvp=mvp, mgp=mgp, mop=mop, st=st):
                    q = mqp(x).view(B, 1, H, dk).abs(); k = mkp(x).view(B, 1, H, dk).abs(); v = mvp(x).view(B, 1, H, dv)
                    g = torch.nn.functional.logsigmoid(mgp(x).view(B, 1, H).float()).to(C.DT) if mgp is not None else None
                    o, _ = decode_monolith_step(q, k, v, st, g=g)
                    return mop(o.reshape(B, 1, -1).to(C.DT))
                time_cell(dict(variant=var, nc=nc, shape=shape, arm="monolith"), fwd)
    # cross-arch (own): gdn + hedgehog × shape
    for nc in C.NCS:
        B = C.B; x = t1(B, 1, DM)
        for shape in ("wide", "square", "heads"):
            dg = C.shaped_dims("gdn", shape, nc)
            if dg is not None:
                H, dk, dv = dg["H"], dg["d_k"], dg["d_v"]
                gqp, gkp, gvp = lin(DM, H*dk), lin(DM, H*dk), lin(DM, H*dv)
                gbp, ggp, gop = lin(DM, H), lin(DM, H), lin(H*dv, DM)
                st = torch.zeros(B, H, dk, dv, device=C.DEV, dtype=torch.float32)
                def fwd(H=H, dk=dk, dv=dv, gqp=gqp, gkp=gkp, gvp=gvp, gbp=gbp, ggp=ggp, gop=gop, st=st):
                    q = gqp(x).view(B, 1, H, dk); k = gkp(x).view(B, 1, H, dk); v = gvp(x).view(B, 1, H, dv)
                    beta = gbp(x).view(B, 1, H).sigmoid().to(C.DT)
                    ldg = torch.nn.functional.logsigmoid(ggp(x).view(B, 1, H).float()).to(C.DT)
                    o, _ = decode_gdn_step(q, k, v, ldg, beta, st)
                    return gop(o.reshape(B, 1, -1).to(C.DT))
                time_cell(dict(variant="own", nc=nc, shape=shape, arm="gdn"), fwd)
            dh = C.shaped_dims("lin2", shape, nc)
            if dh is not None:
                H, dk, dv = dh["H"], dh["d_k"], dh["d_v"]
                hqp, hkp, hvp = lin(DM, H*dk), lin(DM, H*dk), lin(DM, H*dv)
                fmap = HedgehogFeatureMap(head_dim=dk).to(C.DEV, C.DT); feat = fmap(t1(1, 1, H, dk)).shape[-1]
                hop = lin(H*dv, DM); st = torch.zeros(B, H, feat, dv, device=C.DEV, dtype=torch.float32)
                def fwd(H=H, dk=dk, dv=dv, hqp=hqp, hkp=hkp, hvp=hvp, fmap=fmap, hop=hop, st=st):
                    q = hqp(x).view(B, 1, H, dk); k = hkp(x).view(B, 1, H, dk); v = hvp(x).view(B, 1, H, dv)
                    o, _ = decode_hedgehog_step(q, k, v, fmap, st)
                    return hop(o.reshape(B, 1, -1).to(C.DT))
                time_cell(dict(variant="own", nc=nc, shape=shape, arm="hedgehog"), fwd)
    C.write_csv(Path(out) / "recurrent_e2e.csv", rows)
    print(f"WROTE {Path(out)}/recurrent_e2e.csv  [{C.gpu_name()}]")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Recurrent (decode, forward-only) benchmark.")
    ap.add_argument("--check", action="store_true", help="assert step-decode == chunked, then exit")
    ap.add_argument("--bench", action="store_true", help="produce the ms/token table")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.require_cuda()
    if a.check:
        raise SystemExit(0 if run_check() else 1)
    if a.bench:
        bench(a.steps, a.warmup, a.out)
    else:
        ap.error("pass --check or --bench")


if __name__ == "__main__":
    main()
