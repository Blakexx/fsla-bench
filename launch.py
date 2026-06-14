#!/usr/bin/env python3
"""Launch a RoLA benchmark on a fleet of rented GPUs.

    python launch.py mqar --boxes 6      # the canonical MQAR grid sweep
    python launch.py lm   --boxes 4      # language-modeling arms (scaffold; see rola_bench.lm.job)

This is the suite's orchestration entrypoint: each benchmark module (rola_bench.mqar / rola_bench.lm)
exports a `fleet.Job`; we look one up by name and hand it to the generic `fleet` Dispatcher, which
owns provisioning, HTTP control, result/checkpoint sync, and idle-free teardown.
"""
import argparse, importlib
from fleet import Dispatcher

# benchmark name -> (module, attribute) exporting a fleet.Job
JOBS = {
    "mqar": ("rola_bench.mqar.job", "mqar_job"),
    "lm":   ("rola_bench.lm.job", "lm_job"),
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
