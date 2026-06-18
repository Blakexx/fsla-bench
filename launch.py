#!/usr/bin/env python3
"""Launch a RoLA benchmark on a fleet of rented GPUs.

    python launch.py mqar --boxes 6      # the canonical MQAR grid sweep
    python launch.py lm   --boxes 4      # language-modeling arms (scaffold; see rola_bench.lm.job)

This is the suite's orchestration entrypoint: each benchmark module (rola_bench.mqar / rola_bench.lm)
exports a `fleet.Job`; we look one up by name and hand it to the generic `fleet` Dispatcher, which
owns provisioning, HTTP control, result/checkpoint sync, and idle-free teardown.
"""
import argparse, importlib, os
# The dispatcher is a pure orchestrator — it builds job configs and HTTP-coordinates remote boxes;
# ALL training/benchmarking runs on the cloud GPUs, never locally. But importing the job configs
# pulls in torch/rola, which would grab a ~1GB local CUDA context for nothing. Hide the GPU so the
# dispatcher stays CPU-only (verified: work_list still builds). Overridable if ever needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from fleet import Dispatcher

# benchmark name -> (module, attribute) exporting a fleet.Job
JOBS = {
    "mqar": ("rola_bench.mqar.job", "mqar_job"),
    "lm":   ("rola_bench.lm.job", "lm_job"),
    "perf": ("rola_bench.perf.job", "perf_job"),     # kernel-efficiency (whole-bench cells, no ckpt)
    "similarity": ("rola_bench.similarity.job", "sim_job"),  # post-hoc similarity-matrix eval (rank+sharpness), best-per-cell ckpts
    "seed": ("rola_bench.mqar.seed_job", "seed_job"),  # seed-confirmation: best cell per (kernel,rung) at extra seeds
    "glakappa": ("rola_bench.mqar.glakappa_job", "glakappa_job"),  # routed-GLA re-run with kappa (matches RLA kappa-asym)
    "rlabias": ("rola_bench.mqar.rlabias_job", "rlabias_job"),     # router-bias ablation (nc128 RLA-kappa, bias={0,1}x2 seeds)
    "dmodel64": ("rola_bench.mqar.dmodel64_job", "dmodel64_job"),  # d_model=64 nc256 RLA-kappa, 5-LR sweep (task #111)
    "d64base": ("rola_bench.mqar.dmodel64base_job", "d64base_job"),  # d_model=64 nc256 matched baselines (RLA-wide, Hedgehog-square)
    "normsym": ("rola_bench.mqar.normsym_job", "normsym_job"),       # norm x sym equivalence ablation (72 cells, #96)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=sorted(JOBS))
    ap.add_argument("--boxes", type=int, default=6)
    ap.add_argument("--batch", type=int, default=None, help="override run_ids per box batch")
    a = ap.parse_args()
    mod, attr = JOBS[a.benchmark]
    job = getattr(importlib.import_module(mod), attr)
    if a.batch:
        job.batch = a.batch
    Dispatcher(job, boxes=a.boxes).run()


if __name__ == "__main__":
    main()
