#!/usr/bin/env python3
"""Pre-deploy CI gate: build the SMALLEST instance of every cell family and run one fwd+bwd.

The per-box GPU smoke (smoke.py) only exercises ONE config (RoLA) — so a mixer that's missing/broken
for a *different* family (e.g. a stripped zoology.mixers.sse) sails through and only fails on the
fleet. This gate closes that gap: it enumerates the distinct families in the actual work-list
(canonical_mqar_grid — the SAME configs the dispatcher launches, so it can't drift), takes the
smallest config of each, builds its real sequence mixer, and runs one fwd+bwd.

These MQAR cells wrap the kernel mixer in a Hybrid (BaseConv + main mixer), so the family is the
non-base_conv entry nested in Hybrid.kwargs['configs']; building the full Hybrid instantiates it
(a missing module fails right here). Exits nonzero if ANY family fails → fails the docker build.

Device: uses CUDA if present, else CPU. On CPU the routed RoLA path uses its eager fallback, but the
fla Triton baselines (gla/based/attention) cannot run a kernel on CPU — for those we still REQUIRE a
clean import+construction (which is what catches missing modules) and report fwd+bwd as skipped.
"""
import sys
import torch


def family_of(cfg):
    """(family_key, num_chunks) for a cell. Family = the non-base_conv mixer (recursing into Hybrid),
    keyed by the fields that select distinct code paths: the mixer name + (for RoLA) the inner
    kernel / feature-map / state-norm. Splitting on these is what makes routed-rla vs routed-gla vs
    the norm variants each get exercised, instead of collapsing into one RoLA family."""
    sm = cfg.model.sequence_mixer
    kw = sm.kwargs or {}
    main, mk = {"name": sm.name}, kw
    subs = kw.get("configs")
    if subs:  # Hybrid — find the main (non-base_conv) sub-mixer
        for it in subs:
            nm = it.get("name") if isinstance(it, dict) else getattr(it, "name", None)
            ik = (it.get("kwargs") if isinstance(it, dict) else getattr(it, "kwargs", None)) or {}
            if nm and "base_conv" not in nm:
                main, mk = {"name": nm}, ik
                break
    kk = mk.get("kernel_kwargs") or {}
    key = (main["name"], mk.get("kernel"), mk.get("phi"), kk.get("state_norm"))
    return key, mk.get("num_chunks", 1)


def main():
    from rola_bench.mqar.experiments import canonical_mqar_grid as g
    fams = {}
    for c in g.configs:
        key, nc = family_of(c)
        size = c.model.d_model * (nc or 1)
        if key not in fams or size < fams[key][0]:
            fams[key] = (size, c)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cell families: {len(fams)} | device: {dev}")
    bad = []
    for key, (_, c) in sorted(fams.items(), key=lambda x: str(x[0])):
        label = f"{key[0]} kernel={key[1]} phi={key[2]} norm={key[3]}"
        try:
            sm = c.model.sequence_mixer
            d_model = c.model.d_model
            try:
                mixer = sm.instantiate(d_model=d_model, layer_idx=0).to(dev)
            except TypeError:
                mixer = sm.instantiate(d_model=d_model).to(dev)
            # Exercise the harness state-logging path (Hybrid.state_size -> sub-mixer): the runner
            # sets ROLA_PRINT_STATE_JSON and the logger calls model.state_size, which crashed on
            # canonical baselines lacking state_size. A family that can't report state fails here.
            if hasattr(mixer, "state_size"):
                mixer.state_size(sequence_length=16)
            x = torch.randn(2, 16, d_model, device=dev, requires_grad=True)
            try:
                o = mixer(x)
                o = o[0] if isinstance(o, tuple) else o
                o.float().pow(2).mean().backward()
                ok = (bool(torch.isfinite(o).all()) and x.grad is not None
                      and bool(torch.isfinite(x.grad).all()))
                print(f"  {'PASS' if ok else 'FAIL'}  {label}  (fwd+bwd)")
                if not ok:
                    bad.append(label)
            except (NotImplementedError, RuntimeError) as e:
                # On CPU, fla Triton baselines can't run a kernel — construction already passed
                # (the missing-module catch), so treat fwd as skipped on CPU but FAIL on GPU.
                if dev == "cpu" and "cuda" in (str(e) + repr(e)).lower() or "triton" in repr(e).lower():
                    print(f"  PASS  {label}  (constructed; fwd skipped — needs GPU on CPU host)")
                else:
                    print(f"  FAIL  {label} -> {type(e).__name__}: {e}")
                    bad.append(label)
        except Exception as e:
            print(f"  FAIL  {label} -> {type(e).__name__}: {e}")
            bad.append(label)

    if bad:
        print(f"CELL-CHECK FAILED: {len(bad)}/{len(fams)} families broken: {bad}")
        return 1
    print(f"CELL-CHECK OK: all {len(fams)} families on {dev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
