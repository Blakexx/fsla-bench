"""Scaling perf job builder, spec-driven on the global layer.

`perf_job(config)` reads rola_bench/perf/experiments/<config>.yaml. Each bench entry maps to one
scaling run_id (`perf-scaling-{train,prefill,decode}`), with layer arms interleaved round-robin
inside the runner. Results: results/perf/<config>/<cell>.json. Launch:
`launch.py perf scaling_chunked`.
"""
from rola_bench.jobs import make_job
from rola_bench.spec import load, env_from


def perf_job(config="default"):
    spec = load("perf", config)
    benches = spec["benches"]
    knobs = {k: spec[k] for k in ("reps",) if k in spec}
    box_env = env_from("PERF", knobs)
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {**box_env, **fleet.get("env", {})}
    return make_job("perf", config, lambda: [f"perf-{b}" for b in benches],
                    "python -u -m rola_bench.perf.run_perf", fleet)
