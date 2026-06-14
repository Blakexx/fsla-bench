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


def decode_monolith_step(qt, kt, vt, state):
    o, state = fused_recurrent_simple_gla(qt, kt, vt, scale=1.0,
                                          initial_state=state, output_final_state=True)
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


def bench(steps, warmup, out):
    rows = []
    cdt = lambda x: x.to(C.DT) if x is not None else None
    print(f"# recurrent decode | {C.gpu_name()} | B={C.B} H={C.H} d_qk=d_v={C.DQK} bf16 "
          f"| forward only | ms/token (median of {steps})")
    for gla in (False, True):
        var = "GLA" if gla else "RLA"
        for nc in C.NCS:
            q, k, v, r, w, ld = make_inputs(1, nc, gla, seed=1)
            # state intentionally NOT threaded across reps: decode cost is step-invariant for these
            # kernels, so we time a representative single step (state stays at step 0), not a full
            # rollout. state_mib reports the recurrent-state footprint, not a sequence.
            for arm, norm in (("rola-vh global", "global"), ("rola-vh kappa", "kappa")):
                state = torch.zeros(C.B, C.H * nc, C.DQK, C.DV + 1, device=C.DEV, dtype=torch.float32)
                f = lambda norm=norm: decode_routed_step(cdt(q), cdt(k), cdt(v), r, w, cdt(ld), state, nc, norm)
                ms = C.time_forward(f, steps, warmup)
                rows.append(dict(variant=var, nc=nc, arm=arm, ms_per_tok=round(ms, 3),
                                 state_mib=round(state.numel() * 4 / 2 ** 20, 1)))
                print(rows[-1], flush=True)
            # matched wide monolith: d_qk' = nc*d_qk, same state floats (+den col to match).
            # seeded for byte-reproducibility (forward-only; no --check on this arm).
            dqkw = nc * C.DQK
            gm = torch.Generator(device=C.DEV).manual_seed(2)
            rm = lambda *s: torch.randn(*s, device=C.DEV, generator=gm)
            qm = rm(C.B, 1, C.H, dqkw).abs().to(C.DT)
            km = rm(C.B, 1, C.H, dqkw).abs().to(C.DT)
            vm = rm(C.B, 1, C.H, C.DV + 1).to(C.DT)
            state_m = torch.zeros(C.B, C.H, dqkw, C.DV + 1, device=C.DEV, dtype=torch.float32)
            f = lambda: decode_monolith_step(qm, km, vm, state_m)
            ms = C.time_forward(f, steps, warmup)
            rows.append(dict(variant=var, nc=nc, arm="monolith wide", ms_per_tok=round(ms, 3),
                             state_mib=round(state_m.numel() * 4 / 2 ** 20, 1)))
            print(rows[-1], flush=True)
    out = Path(out)
    C.write_csv(out / "recurrent_decode.csv", rows)
    print(f"\nWROTE {out}/recurrent_decode.csv  [{C.gpu_name()}]")


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
