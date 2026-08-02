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

## Suite entry points (`rola_bench/suites/`)

Lightweight construct-gates + drivers on top of the fleet machinery above — NOT a second dispatch
system. Each declares its pinned config list + seeds + output schema, and supports `--smoke`
(construct everything, run ONE tiny step/measurement, no training / no full benchmark):

| Suite | Smoke command | Real results land in | Locked numbers |
|---|---|---|---|
| `rola_bench.suites.mqar` | `python -m rola_bench.suites.mqar --smoke` | `results/mqar/<config>/<cell>.json` (via `launch.py mqar <config>`) | `rola_bench/mqar/instances.py` (`ROLA_INSTANCES`) + `rola_bench.named_configs` (flagship) |
| `rola_bench.suites.lm` | `python -m rola_bench.suites.lm --smoke` | `results/lm/<dataset>/*.json` (via `launch.py lm <config>`) | `rola_bench/lm/arms.py` (`ARMS`) |
| `rola_bench.suites.perf` | `flock /tmp/rola_gpu.lock python -m rola_bench.suites.perf --smoke` | `results/perf/*.csv` (`scaling.py`) + JSON lines (`engine_bench.py`/`decode_bench.py --out`) | `rola_bench.named_configs`; fleet configs `experiments/{paper_v1,engine_regression}.yaml` |

### The flagship named configs (`rola_bench/named_configs.py`)

**RoLA-base** is the pinned, published-flagship RoLA architecture the suites default to: square
topology (`levels_per_head=2, states_per_level=sqrt(nc)`, nc=256), `router_bias=True`,
`kernel='gla_scalar'`, `state_norm='kappa'` (explicit — never the layer default; see rationale in
`named_configs.py`), `use_short_conv=True`, UNTIED, per-level heterogeneous routing — write is SPARSE
at level 0 (coarse, all the §7 skip perf) + DENSE at level 1 (fine, within-region blending), read is
DENSE at both levels (dead-end-free by construction). **RoLA-base-rla** is the same with `kernel='rla'`
(the LM ablation, no decay). **RoLA-base-global** is the same with `state_norm='global'` (κ≡0 fixed) —
the kappa-vs-global ablation row for the paper. MQAR variants drop the short conv (convs confound
short-offset recall). **RoLA-tied** (perf suite only) is the tied+sparse+sparse shared-solve max-perf
variant.

Per-level heterogeneous activation needed the C8 shim-lift to execute on the engine —
`NAMED_CONFIGS[...].build_kwargs` constructs the real config, and `construct()` construct-gates it: if
the engine raises the specific "per-level heterogeneous routing activation" `NotImplementedError`, it
falls back to the runnable **RoLA-base-sw** / **RoLA-base-sw-rla** / **RoLA-base-sw-global** (both
levels write='sparse', uniform activation) and reports `requires_c8=True` — never a silent
substitution. **C8 has since landed** on the engine tip this migration verified against (confirmed via
`--smoke`, `requires_c8=False` for every named config as of this writing) — the gate/fallback code
stays in place as a no-op safety net in case a future engine checkout regresses it.

The HF LM path carries `level_routing_configs` through `RoLAConfig` and the named-instance boundary.
The field accepts a broadcast routing config dict for homogeneous levels or an explicit per-level list;
the latter is used by the flagship's heterogeneous sparse-coarse/dense-fine routing.

### Perf bench files (`rola_bench/perf/`)

- **`scaling.py`** — the mature architecture-comparison harness (3 regimes: train/prefill/decode; IQR;
  OOM/ERR sentinels; state-MiB; round-robin). `paper_v1`'s numbers come from HERE: attn/SDPA, the
  flagship (`build_rola_named` — `rola-base`/`rola-base-rla`, construct-gated), the RoLA-vs-monolith-
  vs-attn scaling shape sweep (`rola`/`monolith` kinds, unchanged), and REAL external `fla_rola.layers`
  baselines (`gla`/`gdn`/`deltanet`/`gsa`, `_build_fla_baseline` — canonical geometry mirroring
  `rola_bench.lm.arms`'s `GDN_GEOM`/`GLA_GEOM`, not the (d,nc) sweep axis). The decode arm's
  `SCALE_DECODE_GRAPHED=1` env adds a `-graphed` cell variant per RoLA arm (the C9 `RoLADecodeGraph`
  pattern, `_decode_layer_sample_graphed`) alongside the eager cell — eager and graphed are cells of
  the SAME bench, not a separate harness.
- **`engine_bench.py`** — cross-engine RoLA layer bench (ENGINEERING/merge-decision only, not paper
  material — see `engine_regression.yaml`): `--worktree` points at any
  `flash-linear-attention-rola-cuda-p0` checkout (rola / rola-triton / rola-cute / …), unset (fleet
  default) resolves `fla_rola` from the image's pip install; warms and times fwd-only and fwd+bwd as
  separate loops (Triton autotunes per grad-mode); median-of-20 (`--reps`) via
  `rola_bench.perf._common.sample_stats` (same convention as `scaling.py`); OOM/ERR sentinel rows
  (reuses `scaling._INFEASIBLE`); one JSON line per cell.
- **`decode_bench.py`** — a THIN named entry point for one ad hoc eager-vs-graphed decode sample; the
  real measurement lives in `scaling.py` (`_decode_layer_sample`/`_decode_layer_sample_graphed`) — this
  script does not reimplement a second timing discipline.
- **`METHODOLOGY.md`** — the measurement rules every perf script here follows (incl. which numbers are
  citable: local 3080Ti = engineering gates, A100 fleet = citable).

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
