"""Perf (kernel-efficiency) job builder, on the global layer.

Granularity: one run_id = one WHOLE bench module — the headline numbers are drift-free *ratios* from
round-robin interleaving all impls inside ONE process (perf/_common.round_robin), so splitting per-arm
would destroy that. PERF_BENCHES restricts the set. No checkpoints. Results: results/perf/<config>/.

Launch: launch.py perf default
"""
import os

from rola_bench.jobs import make_job

ALL_BENCHES = ["perf-chunked", "perf-recurrent"]


def perf_job(config="default"):
    def work_list():
        sel = os.environ.get("PERF_BENCHES")
        if sel:
            keep = {f"perf-{s.strip()}" for s in sel.split(",")}
            return [b for b in ALL_BENCHES if b in keep]
        return list(ALL_BENCHES)
    fleet = {
        # Perf wants a clean, declared GPU; default to the A100 headline card (override OFFER_QUERY).
        "offer_query": os.environ.get(
            "OFFER_QUERY", "gpu_name=A100_PCIE num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
        "idle_timeout": 900,
        "batch": 1,                  # one whole bench per box (round-robin integrity lives in the run)
        "per_batch_timeout": 3600,
        "env": {"PERF_REPS": os.environ.get("PERF_REPS", "15")},
    }
    return make_job("perf", config, work_list, "python -u -m rola_bench.perf.run_perf", fleet)
