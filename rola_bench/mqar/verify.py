"""In-image verification for the fla_rola fork (run as a one-off GPU job after image build).

(1) REGRESSION: fla_rola unrouted == canonical fla (fwd+grads, simple_gla/gla/linear) — proves the
    baselines' package is untouched and the fork's canonical path is identical.
(2) ROUTING: fla_rola routed Triton path == the fork's own torch reference (_rola_chunk_core),
    fwd + all grads, RLA + GLA, nc=16/64 — proves the Triton kernels work on this GPU/driver.
Exit 0 = green. No dependency on the local-only oracle package."""
import sys
import torch
import fla.ops.simple_gla, fla.ops.gla, fla.ops.linear_attn  # populate fla.ops (wheel __init__ is a namespace stub)
import fla
from rola_bench import engine
fla_rola = engine.import_engine()   # the engine package, through the one seam

TOL_R, TOL_T = 1e-4, 2e-2


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def fwd_bwd(call, ins, gkeys, dO):
    leaves = {k: (v.detach().requires_grad_() if k in gkeys else v) for k, v in ins.items()}
    o = call(leaves)
    g = torch.autograd.grad(o, [leaves[k] for k in gkeys], dO)
    return o.detach(), g


def main():
    ok = True
    g = lambda *s: torch.randn(*s, device='cuda', dtype=torch.float32)
    # (1) regression
    torch.manual_seed(0)
    ins = dict(q=g(2, 256, 4, 16), k=g(2, 256, 4, 16), v=g(2, 256, 4, 16),
               gs=torch.nn.functional.logsigmoid(g(2, 256, 4)),
               gk=torch.nn.functional.logsigmoid(g(2, 256, 4, 16)))
    for name, call, keys in [
            ('simple_gla', lambda m: (lambda i: m.ops.simple_gla.chunk_simple_gla(i['q'], i['k'], i['v'], i['gs'], scale=1.0)[0]), ['q', 'k', 'v', 'gs']),
            ('gla', lambda m: (lambda i: m.ops.gla.chunk_gla(i['q'], i['k'], i['v'], i['gk'], scale=1.0)[0]), ['q', 'k', 'v', 'gk']),
            ('linear', lambda m: (lambda i: m.ops.linear_attn.fused_chunk_linear_attn(i['q'], i['k'], i['v'], normalize=False, scale=1.0)[0]), ['q', 'k', 'v'])]:
        dO = torch.randn_like(ins['v'])
        oc, gc = fwd_bwd(call(fla), ins, keys, dO)
        of, gf = fwd_bwd(call(fla_rola), ins, keys, dO)
        r = max(rel(of, oc), max(rel(a, b) for a, b in zip(gf, gc)))
        print(f"regression {name:11s} {r:.1e} {'PASS' if r < TOL_R else 'FAIL'}")
        ok &= r < TOL_R
    # (2) routed Triton vs the fork's torch reference (un-normalized readout, norm='raw')
    from rola_bench.engine import chunk_rola
    from rola_bench.engine import _rola_chunk_core
    for variant in ('RLA', 'GLA'):
        for nc in (16, 64):
            torch.manual_seed(nc)
            B, L, H, K, V = 2, 256, 2, 12, 12   # workload shapes; sm75 envelope is dv<=15 (BV=16) at BG=16
            q, k, v = g(B, L, H, K), g(B, L, H, K), g(B, L, H, V)
            r_ = torch.softmax(g(B, L, H, nc), -1); w_ = torch.softmax(g(B, L, H, nc), -1)
            ld = (1 - w_ * (1 - torch.sigmoid(g(B, L, H, nc)))).clamp(min=1e-6).log().clamp(min=-2.5) \
                if variant == 'GLA' else None
            leaves = [t.requires_grad_() for t in [q, k, v, r_, w_] + ([ld] if ld is not None else [])]
            o_t = chunk_rola(q, k, v, r=r_, w=w_, g=ld, norm='raw', scale=1.0)
            dO = torch.randn_like(o_t)
            g_t = torch.autograd.grad(o_t, leaves, dO, retain_graph=False)
            fold = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
            o_e = _rola_chunk_core(fold(q), fold(k), fold(v), fold(w_), fold(r_),
                                   fold(ld) if ld is not None else None, 64)
            o_e = o_e.view(B, H, L, V).permute(0, 2, 1, 3)
            g_e = torch.autograd.grad(o_e, leaves, dO)
            rr = max(rel(o_t, o_e), max(rel(a, b) for a, b in zip(g_t, g_e)))
            print(f"routing {variant} nc={nc:<4d} {rr:.1e} {'PASS' if rr < TOL_T else 'FAIL'}")
            ok &= rr < TOL_T
    # (3) per-state denominator kernels (kappa) vs in-script dense oracle: additive + decayed
    from rola_bench.engine import rola_perstate_den_triton, rola_perstate_den_gla_triton
    for variant in ('RLA', 'GLA'):
        for nc in (16, 64):
            torch.manual_seed(100 + nc)
            B, L, K = 4, 256, 12
            q_, k_ = g(B, L, K).abs(), g(B, L, K).abs()
            w_ = torch.softmax(g(B, L, nc), -1)
            ld = (-torch.rand(B, L, nc, device='cuda')).clamp(min=-2.4) if variant == 'GLA' else None
            G = torch.einsum('bid,bjd->bij', q_, k_)
            m = torch.tril(torch.ones(L, L, device='cuda', dtype=torch.bool))
            if ld is None:
                d_ref = (G[..., None] * w_[:, None] * m[None, ..., None]).sum(2)
                d = rola_perstate_den_triton(q_, k_, w_)
            else:
                Lam = ld.cumsum(1)
                delta = Lam[:, :, None] - Lam[:, None]
                # mask BEFORE exp: anti-causal deltas are large-positive and overflow fp32
                dec = torch.exp(torch.where(m[None, ..., None], delta, torch.full_like(delta, -1e30)))
                d_ref = (G[..., None] * dec * w_[:, None]).sum(2)
                d = rola_perstate_den_gla_triton(q_, k_, w_, ld)
            rr = rel(d, d_ref)
            print(f"den {variant} nc={nc:<4d} {rr:.1e} {'PASS' if rr < TOL_T else 'FAIL'}")
            ok &= rr < TOL_T
    # baseline-layers gate: a missing fla.layers silently broke ALL rla/hedgehog/gdn
    # baselines on the fleet (fla-core has fla.ops but not fla.layers). This catches it.
    try:
        # `from fla import layers` binds `layers`, NOT `fla` -- importing `fla.layers` here
        # would make `fla` a function-local and UnboundLocalError the earlier regression code.
        from fla import layers as _fla_layers
        for _cls in ('LinearAttention', 'GatedDeltaNet'):
            assert hasattr(_fla_layers, _cls), f'fla.layers.{_cls} missing'
        print('baseline-layers  fla.layers + LinearAttention/GatedDeltaNet  PASS')
    except Exception as _e:
        print(f'baseline-layers  FAIL  {_e!r}'); ok = False
    print("VERIFY", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
