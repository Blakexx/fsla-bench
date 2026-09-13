"""Scaling perf job builder, spec-driven on the global layer.

`perf_job(config)` reads rola_bench/perf/experiments/<config>.yaml. Each bench entry maps to one
scaling run_id (`perf-scaling-{train,prefill,decode}`), with layer arms interleaved round-robin
inside the runner. Results: `rola_results` at perf/<config>. Launch:
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
    fleet["env"] = {"PERF_CONFIG": config, **box_env, **fleet.get("env", {})}

    def _work_list():
        ids = []
        for bench in benches:
            if bench == "profile-graph":
                ids.extend(f"perf-profile-graph-{p['name']}" for p in spec.get("profiles", []))
            elif bench == "engine-bench":
                ids.extend(f"perf-engine-bench-{c['name']}" for c in spec.get("engine_cells", []))
            else:
                ids.append(f"perf-{bench}")
        return ids

    return make_job("perf", config, _work_list,
                    "python -u -m rola_bench.perf.run_perf", fleet)
