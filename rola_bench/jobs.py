"""Global config-driven job factory — one builder for every benchmark.

A fleet `Job` is almost entirely boilerplate; the only real per-experiment knobs are the work-list
(run_ids), the box-side run command, and a few fleet settings (batch / timeout / extra env). This
assembles the Job from those three, so benchmarks don't each hand-roll a `*_job.py`.

Result store is uniform: results/<bench>/<config>/<run_id>.jsonl (the fleet dispatcher writes that
dir/run_id format; `done_ids` reads it back for resume).
"""
import os

from fleet import Job

from rola_bench import results as R

RESULTS_ROOT = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/results")
RUNTIME_ROOT = os.environ.get("ROLA_BENCH_RUNTIME", "/home/blake/rola-bench/runtime")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/d/rola_data/fleet_ckpts")
DEFAULT_IMAGE = "blakeresearch/rola-bench:v17"
DEFAULT_OFFER = "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500"


def make_job(bench, config, work_list, run_cmd, fleet=None):
    """Build a fleet.Job for (bench, config). `fleet` is the experiment's launch knobs
    (image / offer_query / batch / per_batch_timeout / idle_timeout / env), all optional."""
    fleet = fleet or {}
    cfg_dir = os.path.join(RESULTS_ROOT, bench, config)            # results store (dir/run_id)
    runtime = os.path.join(RUNTIME_ROOT, f"{bench}-{config}")
    return Job(
        name=f"{bench}-{config}",
        image=os.environ.get("ROLA_BENCH_IMAGE", fleet.get("image", DEFAULT_IMAGE)),
        offer_query=os.environ.get("OFFER_QUERY", fleet.get("offer_query", DEFAULT_OFFER)),
        results_path=cfg_dir,
        ckpt_dir=CKPT_DIR,
        work_list=work_list,
        done_ids=lambda: R.done_ids(cfg_dir),
        env={"FLEET_RUN_CMD": run_cmd, **fleet.get("env", {})},
        bad_hosts=os.path.join(runtime, "bad_hosts.txt"),
        idle_timeout=fleet.get("idle_timeout", 1200),
        batch=fleet.get("batch", 1),
        per_batch_timeout=fleet.get("per_batch_timeout", 18000),
    )
