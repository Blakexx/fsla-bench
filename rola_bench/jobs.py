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

OFFER QUERIES ARE DIALECT-BOUND. `offer_query` is written in the ACTIVE provider's dialect (declared
as `offer_query_dialect` in fleet.semantics) — vast's filter language and CloudRift's key=value
catalogue selector are not interchangeable, and CloudRift's parser IGNORES unknown keys, so a vast
query silently degrades to "any 1-GPU SKU" instead of erroring. So a spec may give `offer_query` as
either a plain string (taken to be `vast_filter`, which is what every existing spec is written in)
or a `{dialect: query}` mapping; asking for a dialect the active provider does not speak is a loud
error, never a silent downgrade.
"""
import os

from fleet import Job, provider, semantics_of

from rola_bench import results as R

RESULTS_ROOT = os.environ.get("ROLA_BENCH_RESULTS", "/home/blake/rola-bench/results")
RUNTIME_ROOT = os.environ.get("ROLA_BENCH_RUNTIME", "/home/blake/rola-bench/runtime")
CKPT_DIR = os.environ.get("ROLA_BENCH_CKPTS", "/mnt/e/rola/rola_data/fleet_ckpts")
DEFAULT_IMAGE = "blakeresearch/rola-bench:v40"   # ⚠ keep in sync with the experiments yamls (paper_v1 pins v40); bumps are manual   # v27: DROP the warm pre-pass entirely (it was ~95% serial autotune-do_bench, net-negative; cells self-warm via WARMUP reps). v26: affinity-aware warm worker count. v25: scaling-bench fixes (nc->even-pow2 snap for flat|square|tree, infeasible-shape SKIP)
# Default offer query PER DIALECT. The vast filter caps bandwidth $/GB (we pull a big image +
# checkpoints) and screens on reliability — both are marketplace concepts with no analogue on a
# provider-owned catalogue, where you pick a SKU and the only real question is whether a node is free.
DEFAULT_OFFER = {
    "vast_filter": ("gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500 "
                    "inet_down_cost<0.01 inet_up_cost<0.01"),
    "cloudrift_kv": "gpu_name=RTX_4090 num_gpus=1 public_ip=true",
}


def offer_query_for(spec_query, dialect):
    """Pick the offer query written in the ACTIVE provider's dialect.

    `spec_query` is None (use the default), a {dialect: query} mapping, or a plain string — the last
    being every existing spec, all of which are vast filter strings. Handing a vast filter to a
    catalogue provider is the quiet failure this guards: unknown keys are ignored, so you get *a*
    box, just not the GPU you asked for.
    """
    if isinstance(spec_query, dict):
        if dialect not in spec_query:
            raise ValueError(
                f"this spec's offer_query is declared for dialect(s) {sorted(spec_query)}, but the "
                f"active provider speaks {dialect!r}. Add a {dialect!r} entry to the spec's "
                f"fleet.offer_query mapping.")
        return spec_query[dialect]
    if spec_query is not None:
        if dialect != "vast_filter":
            raise ValueError(
                f"this spec's offer_query is a plain string, which means the vast filter dialect, "
                f"but the active provider speaks {dialect!r}. Replace it with a mapping, e.g. "
                f"offer_query: {{vast_filter: '...', {dialect}: '...'}} — a vast filter handed to a "
                f"catalogue provider is silently ignored, not rejected.")
        return spec_query
    try:
        return DEFAULT_OFFER[dialect]
    except KeyError:
        raise ValueError(f"no default offer query for dialect {dialect!r}; "
                         f"set one in rola_bench.jobs.DEFAULT_OFFER or in the spec") from None


def make_job(bench, config, work_list, run_cmd, fleet=None, **job_kwargs):
    fleet = dict(fleet or {})
    dialect = semantics_of(provider()).offer_query_dialect
    cfg_dir = os.path.join(RESULTS_ROOT, bench, config)            # results store (dir/run_id)
    runtime = os.path.join(RUNTIME_ROOT, f"{bench}-{config}")
    spec = dict(
        name=fleet.pop("name", f"{bench}-{config}"),
        image=os.environ.get("ROLA_BENCH_IMAGE", fleet.pop("image", DEFAULT_IMAGE)),
        offer_query=os.environ.get("OFFER_QUERY") or offer_query_for(fleet.pop("offer_query", None),
                                                                     dialect),
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
