# rola-bench

The RoLA **benchmark suite + orchestration**. It defines *what* to run — four benchmarks expressed as
declarative specs over one config-driven runner — and hands the work to the generic
[`fleet`](https://github.com/Blakexx/fleet-dev) package, which owns *how* (rent spot GPUs, control over
HTTP, sync results, idle-free teardown).

This is the top of the dependency chain: `rola-bench ──▶ {rola, fla-rola, zoology, fleet}`. Part of the
RoLA project (Blake Bottum, 2026).

## The spec-driven model

An experiment is a `<bench> <config>` pair. The *config* is a declarative YAML committed under
`rola_bench/<bench>/experiments/<config>.yaml` — the single source of truth for that run (grid axes,
hyperparameters, and a `fleet:` block for dispatch knobs). There are no per-experiment Python scripts:
the spec loader (`rola_bench/spec.py`) reads the YAML, each benchmark's `job.py` expands it into a
work-list of cell ids and a box-side run command, and the global job factory (`rola_bench/jobs.py`,
`make_job`) assembles a `fleet.Job` from those. `launch.py` looks the builder up by name and runs the
dispatcher.

```bash
python launch.py <bench> <config> [--boxes N] [--batch N]

python launch.py mqar canonical --boxes 6     # the canonical matched-state MQAR grid
python launch.py mqar normsym                 # any MQAR spec, by name
python launch.py mqar seed_confirm            # seed-confirmation
python launch.py perf default                 # kernel-efficiency e2e fwd/bwd
python launch.py similarity main              # post-hoc routing/rank audit
python launch.py lm   default                 # language modeling
```

`launch.py` runs CPU-only (it hides the GPU): it builds job configs and HTTP-coordinates the boxes —
**all training and benchmarking runs on the rented GPUs, never locally.** The dispatcher reuses cells
already complete in the result store, requeues a dead box's owed cells, and tears every box down on
drain.

## The four benchmarks

| Bench (`rola_bench/<bench>/`) | Measures | Spec(s) (`experiments/`) |
|---|---|---|
| **`mqar`** | **Accuracy** on Multi-Query Associative Recall — the matched-state grid (RoLA vs. canonical baselines across nc rungs, LRs, seeds), scored per difficulty slice. | `canonical.yaml`, `normsym.yaml`, `dmodel64.yaml`, `router_bias.yaml`, `qknorm_ablation.yaml`, `seed_confirm.py` |
| **`lm`** | **Language modeling** — perplexity / downstream evals for RoLA arms vs. attention + GDN baselines, trained with the HF stack (resume-safe checkpoints). | `default.yaml` |
| **`perf`** | **Kernel efficiency** — end-to-end forward / forward+backward wall-clock and activation memory of the canonical layers on the LM training shape, round-robin interleaved for drift-free ratios. | `default.yaml` |
| **`similarity`** | **Routing audit** — post-hoc rank, sharpness, and state-usage of the trained attention matrices (best-per-cell checkpoints pulled from the MQAR store), to confirm routing genuinely spreads load across states. | `main.yaml` |

Each benchmark dir carries its spec(s) in `experiments/`, a `job.py` (the `<bench>_job(config)` builder),
a box-side runner (`run_rla_sweep.py` / `run_lm.py` / `run_perf.py` / `run_sim.py`), and CPU/smoke
verification helpers.

## Result store

Uniform, one file per cell — parallel writers never contend and resume is just "does this cell exist
with an ok result":

```
results/<bench>/<config>/<cell>.json     # one JSON object per cell (grid point incl. seed)
```

`rola_bench/results.py` is the only reader/writer (`write_cell`, `read_cell`, `done_ids`,
`all_records`); re-running a cell overwrites it (latest wins). The store is checked into the repo so the
paper's tables are reproducible; `runtime/` and `fleet_ckpts` are operational state (gitignored).

## Dependencies

```
rola-bench ──▶ rola        (model + kernels)
           ──▶ fla-rola    (routed FLA fork — the model source of truth)
           ──▶ zoology     (MQAR baselines + the RoLAMixer bridge)   [mqar only]
           ──▶ fleet       (generic GPU orchestration)
           ──▶ fla-core, transformers, …                            (third-party stack)
```

Real package edges — no `sys.path` hacks, no vendored copies (see `pyproject.toml`). MQAR configs
reference `zoology.mixers.rola.RoLAMixer` (RoLA is a first-class zoology mixer); the LM benchmark
imports `fla_rola.models.rola` (registers RoLA with the HF auto-classes).

## Images

| Image | Dockerfile | Benchmark |
|-------|------------|-----------|
| `blakeresearch/rola-bench:v19` | `docker/mqar.Dockerfile` | MQAR / perf / similarity |
| `blakeresearch/rola-lm:v1`     | `docker/lm.Dockerfile`   | LM |

Both build `FROM blakeresearch/fleet-base` and set `FLEET_RUN_CMD` (+ `FLEET_VERIFY_CMD`) so
`fleet.box_server` knows how to run and self-verify a batch. `docker/build.sh <target> [--push]
[--prune]` stages clean sources (no `.git`/data/results) → lean, publication-safe images.
