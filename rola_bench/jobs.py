"""Global config-driven job factory — one builder for every benchmark.

A fleet `Job` is almost entirely boilerplate; the only real per-experiment knobs are the work-list
(run_ids), the box-side run command, and a few fleet settings. This assembles the Job from those,
so benchmarks don't each hand-roll a `*_job.py`.

Result store is uniform: results/<bench>/<config>/<run_id>.jsonl (the fleet dispatcher writes that
dir/run_id format; `done_ids` reads it back for resume).

`fleet` (a dict, usually a spec's `fleet:` block) supplies simple knobs: image / offer_query / batch
/ per_batch_timeout / idle_timeout / env, plus any other plain fleet.Job field (no_kill,
pull_in_progress, live_pull_every, post_timeout, ...). `**job_kwargs` is for bench callables that
can't live in a yaml (payload_for) or path overrides (ckpt_dir, state_path).
"""
import os

from fleet import Job

from rola_bench import results as R

RESULTS_ROOT = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/results")
RUNTIME_ROOT = os.environ.get("ROLA_BENCH_RUNTIME", "/home/blake/rola-bench/runtime")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/e/rola/rola_data/fleet_ckpts")
DEFAULT_IMAGE = "blakeresearch/rola-bench:v27"   # v27: DROP the warm pre-pass entirely (it was ~95% serial autotune-do_bench, net-negative; cells self-warm via WARMUP reps). v26: affinity-aware warm worker count. v25: scaling-bench fixes (nc->even-pow2 snap for flat|square|tree, infeasible-shape SKIP)
DEFAULT_OFFER = "gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500 inet_down_cost<0.01 inet_up_cost<0.01"  # cap bandwidth $/GB — we pull a big image + checkpoints


def make_job(bench, config, work_list, run_cmd, fleet=None, **job_kwargs):
    fleet = dict(fleet or {})
    cfg_dir = os.path.join(RESULTS_ROOT, bench, config)            # results store (dir/run_id)
    runtime = os.path.join(RUNTIME_ROOT, f"{bench}-{config}")
    spec = dict(
        name=fleet.pop("name", f"{bench}-{config}"),
        image=os.environ.get("ROLA_BENCH_IMAGE", fleet.pop("image", DEFAULT_IMAGE)),
        offer_query=os.environ.get("OFFER_QUERY", fleet.pop("offer_query", DEFAULT_OFFER)),
        results_path=cfg_dir,
        ckpt_dir=job_kwargs.pop("ckpt_dir", CKPT_DIR),
        work_list=work_list,
        done_ids=lambda: R.done_ids(cfg_dir),
        env={"FLEET_RUN_CMD": run_cmd, **fleet.pop("env", {})},
        bad_hosts=os.path.join(runtime, "bad_hosts.txt"),
        state_path=os.path.join(runtime, "hosts.json"),
        idle_timeout=fleet.pop("idle_timeout", 1200),
        batch=fleet.pop("batch", 1),
        per_batch_timeout=fleet.pop("per_batch_timeout", 18000),
    )
    spec.update(fleet)         # remaining plain fleet.Job fields (no_kill, pull_in_progress, ...)
    spec.update(job_kwargs)    # bench callables / path overrides (payload_for, ckpt_dir already popped)
    return Job(**spec)
