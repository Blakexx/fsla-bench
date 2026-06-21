"""Perf (kernel-efficiency) job builder, spec-driven on the global layer.

`perf_job(config)` reads rola_bench/perf/experiments/<config>.yaml (benches + the runner knobs +
fleet). One run_id = one WHOLE bench module — the headline numbers are drift-free ratios from
round-robin interleaving all impls inside ONE process, so we don't split per-arm. No checkpoints.
Results: results/perf/<config>/<cell>.json.  Launch: launch.py perf default
"""
from rola_bench.jobs import make_job
from rola_bench.spec import load, env_from


def perf_job(config="default"):
    spec = load("perf", config)
    benches = spec["benches"]
    knobs = {k: spec[k] for k in ("reps", "ncs", "decode_steps", "decode_warmup") if k in spec}
    box_env = env_from("PERF", knobs)
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {**box_env, **fleet.get("env", {})}
    return make_job("perf", config, lambda: [f"perf-{b}" for b in benches],
                    "python -u -m rola_bench.perf.run_perf", fleet)
