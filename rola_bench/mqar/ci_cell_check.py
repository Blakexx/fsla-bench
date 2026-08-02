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
import os
import sys
import torch


def family_of(cfg):
    """(family_key, N) for a cell. Family = the non-base_conv mixer (recursing into Hybrid), keyed by
    the fields that select distinct CODE PATHS.

    RE-KEYED 2026-08-02 (Phase 1.6). The pre-V3 key was `(name, kernel, phi, kernel_kwargs.
    state_norm)` and `num_chunks` — four fields of which three no longer exist on any config this
    repo emits, so every routed cell would have collapsed into ONE family `(name, None, None, None)`
    and the gate would have exercised a single RoLA path while reporting full coverage.

    The V3 key is `(name, decay, state_norm, routing signature)`, where the routing signature is the
    per-level `(tied, read, write, alpha)` tuple list — that IS the structural axis that selects
    between `IndependentRouting` and `UnionRouting` and between the entmax orders, i.e. the thing
    that makes ARM 1, ARM 2, ARM 3 and the D1 control four distinct kernel paths rather than one."""
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
    rows = mk.get("level_routing_configs") or []
    signature = tuple(
        (r.get("tied"), r.get("read"), r.get("write"), r.get("alpha")) if isinstance(r, dict)
        else (r.tied, r.read, r.write, r.alpha)
        for r in (rows if isinstance(rows, (list, tuple)) else [rows]))
    key = (main["name"], mk.get("decay"), mk.get("state_norm"), signature)
    levels = mk.get("states_per_level")
    if isinstance(levels, (list, tuple)):
        N = 1
        for b in levels:
            N *= int(b)
    elif levels is None:
        N = 1
    else:
        N = int(levels) ** int(mk.get("levels_per_head", 1))
    return key, N


def main():
    # The SAME work-list the dispatcher launches, so this gate cannot drift from it.
    # (Was `experiments.canonical_mqar_grid`, a module deleted when the specs became YAML.)
    from rola_bench.mqar.run_rla_sweep import load_configs
    g_configs, _ = load_configs(os.environ.get("CI_CELL_CONFIG", "canonical"))
    fams = {}
    for c in g_configs:
        key, nc = family_of(c)
        size = c.model.d_model * (nc or 1)
        if key not in fams or size < fams[key][0]:
            fams[key] = (size, c)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cell families: {len(fams)} | device: {dev}")
    bad, skipped = [], []
    for key, (_, c) in sorted(fams.items(), key=lambda x: str(x[0])):
        label = f"{key[0]} decay={key[1]} norm={key[2]} routing={key[3]}"
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
                # fla Triton baselines can't run a kernel on a CPU host. Construction already passed
                # (the missing-module catch), so the family is BUILDABLE — but its kernel was NOT
                # exercised. Record it as SKIPPED (not PASS) so a CPU-only CI host can't silently
                # greenlight unrun kernels (#87). A Triton/CUDA RuntimeError on a GPU host is a real
                # FAIL. (Parenthesized condition — the old `A and B or C` mis-classified GPU Triton
                # errors as skips.)
                msg = (str(e) + repr(e)).lower()
                if dev == "cpu" and ("cuda" in msg or "triton" in msg):
                    print(f"  SKIP  {label}  (constructed; kernel NOT exercised — needs GPU)")
                    skipped.append(label)
                else:
                    print(f"  FAIL  {label} -> {type(e).__name__}: {e}")
                    bad.append(label)
        except Exception as e:
            print(f"  FAIL  {label} -> {type(e).__name__}: {e}")
            bad.append(label)

    if bad:
        print(f"CELL-CHECK FAILED: {len(bad)}/{len(fams)} families broken: {bad}")
        return 1
    if skipped and not os.environ.get("CI_ALLOW_CPU_SKIP"):
        print(f"CELL-CHECK INCOMPLETE: {len(skipped)}/{len(fams)} families were NOT exercised on this "
              f"{dev} host (their kernels need a GPU): {skipped}. This gate must run on a GPU builder "
              f"to validate them — a CPU host only checks construction. Set CI_ALLOW_CPU_SKIP=1 to "
              f"accept construction-only validation. Failing loud (#87).")
        return 1
    extra = f"  ({len(skipped)} construction-only — CI_ALLOW_CPU_SKIP set)" if skipped else ""
    print(f"CELL-CHECK OK: all {len(fams)} families on {dev}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
