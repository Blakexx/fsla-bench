# fsla-bench

FSLA's benchmarks and its local measurement suite. The benchmark (MQAR recall) is a declarative spec that the generic
[`fleet`](https://github.com/Blakexx/fleet) package runs on rented GPUs. The measurement suite times two fsla commits
against each other and the attention reference, and diffs their surfaces, on this machine. Every result goes through fsla-devtools' store, `fsla_devtools.results`.

The language-modeling, layer-performance and similarity benchmarks are archived at the tag
`archive/pre-rebuild-2026-09-14`: layer performance is rebuilt on the measurement suite's timing sessions, and
language modeling and similarity return when FSLA trains.

fsla-bench is a library consumer like any other. It builds FSLA only through fla (our fork of it), whose FSLA layer and HF model
are a thin wrapper over fsla. It never reaches into fsla's kernels:

```
fsla-bench ──▶ fla           the fork's fsla branch: the FSLA layer and HF model (over fsla), and the baseline layers
           ──▶ zoology       MQAR's model and baselines (the fork's fsla-pins); its FSLAMixer wraps fla's FSLA
           ──▶ chromatic-recall  the recall tasks (presence, graph, cue consistency) and the launcher MQAR runs through
           ──▶ fleet         GPU orchestration
           ──▶ fsla-devtools the result store's code (`fsla_devtools.results`), writing into fsla-results, the data
           ──▶ fsla          (through fla; the measurement suite measures fsla commits as bundles, each built by its own pins)
```

## Layout

```
fsla_bench/
  models/fsla.py        the FSLA arms: wirings and cells, built through fla; state read off the layer
  fleet/                python -m fsla_bench.fleet (the launcher), jobs.py (Job factory + result sink),
                        spec.py (spec loader), verify.py (the box boot gate)
  mqar/                 job.py, run.py (box side), smoke.py, build_configs.py, local_grid.py (this machine's card),
                        realized_state.py, analysis/graph.py, experiments/*.yaml
  measure/              the measurement suite's groups and attention reference (measure/README.md)
measure/                the measurement suite's targets: sessions, null gates, memory passes and diffs over the bundles
tests/<area>/           one directory per area above
docker/                 the fleet images
```

## The FSLA arms (`fsla_bench/models/fsla.py`)

A **wiring** is one of fsla's own level spellings per routing level, outermost first (`fsla.dense_routing`,
`fsla.split_routing`, `fsla.tied_routing`). A **cell** is a wiring at a state count N per head:
every level's width (the uniform factorization b**D = N unless widths are given), its levels as fsla builds them, a
decay source as fsla's JSON (`None`, or `{"type": "LearnedDecay", ...}`) and the route projection's `bias`. The
canonical MQAR slate is `fsla-arm1-densread-sparsewrite`, `fsla-arm3-levelsplit` and the one-level
control `fsla-d1-dense`.

Every bench builds a cell one way: `mixer_config` for zoology, or `layer` for a bare fla layer, both passing fsla's
objects as their JSON (`fla.layers.fsla.encode`). It reads a built cell's state back off the layer (`state_floats`),
never from a formula. Whatever
fsla refuses, every build refuses with fsla's own error. Today that is prefill while its kernel is rebuilt and
training before its native backward, so every FSLA arm of every bench fails until fsla builds those passes. The
baseline arms run.

## The benchmarks

An experiment is `<bench> <config>`: `fsla_bench/<bench>/experiments/<config>.yaml` is its single source of truth (the
grid, the hyperparameters, and a `fleet:` block of dispatch knobs). The bench's `job.py` expands it into cell ids and
a box-side command, and `fsla_bench.fleet.jobs.make_job` assembles the `fleet.Job`.

| Bench | Measures | Specs |
|---|---|---|
| `mqar` | Recall accuracy at matched realized state: FSLA's wirings against zoology's baselines across N, L, learning rates and seeds. | `canonical`, `law_grid`, `graph_grid`, `presence_grid`, `cue_consistency`, `ood_recall`, `separation`, `dmodel64`, `router_bias`; `seed_confirm.py` (re-runs the winners at extra seeds) |

```bash
bazel run //:fleet -- <fsla checkout> mqar canonical --boxes 6
```

The launcher runs CPU-only: it builds the job and coordinates the boxes over HTTP, and every run happens on the rented
GPUs. It reads this machine's environment:

| Variable | Meaning |
|---|---|
| `FSLA_BENCH_CKPTS` | where pulled MQAR checkpoints live (required by jobs that pull checkpoints; no default) |
| `FSLA_BENCH_RUNTIME` | fleet operational state (bad hosts, host state); default `fleet-runtime/` under the dev config's `host.build_cache`, never in the checkout |
| `FSLA_BENCH_IMAGE`, `OFFER_QUERY`, `FLEET_PROVIDER` | override the spec's image, offer query and provider |

## Local runs

Everything runs through Bazel (`bazel run`, its arguments after `--`, in this checkout), on one of the workspace's two
environments. Each is built once from its lock into the machine's shared store (`MODULE.bazel`), with the packages no
index serves pinned by their public commits:

| Environment | Its stack | Its interpreter |
|---|---|---|
| `bench_env` | torch 2.14 (CUDA 13.0, fsla's), the fla fork's fsla branch, zoology, chromatic-recall, fleet, fsla_devtools (`requirements.lock`) | `bazel run //:bench_python -- <args>` |
| `research_env` | the baselines' stack: torch 2.6 (CUDA 12.4), flash-attn's release wheel, upstream fla 0.4.2, zoology, chromatic-recall, fleet (`research.lock`) | `bazel run //:research_python -- <args>` |

fsla is never installed: a run that imports it (the arms registry does, so the launcher, the MQAR runs and the model
tests) takes a fsla checkout as its first argument, with that checkout's library built (`bazel run //:library` there),
and puts it on the path after this one. The MQAR smoke builds every arm on this machine's GPU and runs one step:

```bash
bazel run //:mqar_smoke -- <fsla checkout>
```

An MQAR-family grid small enough for this card runs one tier at a time through the fleet's own cell runner, and the
graph and presence grids are read by their analysis:

```bash
GRID_TIERS=decoupling bazel run //:mqar_local_grid -- <fsla checkout> --config graph_grid --dry-run
bazel run //:bench_python -- -m fsla_bench.mqar.analysis.graph --config graph_grid
```

The measurement suite is Bazel targets (`measure/`, over fsla-devtools' measurement rules): two fsla BUNDLES -- the
`candidate`, the fsla checkout a run names, and the `reference`, a commit `MODULE.bazel` pins -- each built in its own
workspace by its own pins (fsla-devtools' `bundle` tag), their timing entries beside fsla-bench's attention reference in
one interleaved session per group of central cells and arm set, each with its null gate, a memory pass a group, and the
diffs of the surfaces both commits export; every result stored. See
[`fsla_bench/measure/README.md`](fsla_bench/measure/README.md):

```bash
bazel run //measure:session_layer-alt-entmax1.5-64x64-T2048_entmax_solve --repo_env=FSLA_CANDIDATE=<fsla checkout>
```

Local numbers are engineering gates. Citable numbers come from rented runs.

## Results

Every result is stored through `fsla_devtools.results` (fsla-devtools' store), in the data repository fsla-results (the
checkout the dev config's `store.root` names), at the label of the target that produced it (`fsla_bench/results.py`):
the fleet's rows at `@fsla_bench//:fleet`, the local MQAR grid's beside them at `@fsla_bench//:mqar_local_grid`, and the
measurement suite at each storing target's own label (`@fsla_bench//measure:session_<group>_<arms>`, `null_...`,
`memory_<group>`, `store_diff_<surface>`). A fleet job plugs the store into fleet's `store_result` sink and `done_ids`:
each pulled row is a sample whose semantics are (bench, config, cell, image), stored once however often it is pulled
again, and a cell is done when the current image's semantics have a sample with no error, so an image bump runs the
grid again; the local grid's semantics name its checkouts instead of an image. Failures are samples too. The readers
(`fsla_bench.mqar.analysis.graph`, the seed confirmation's winners) take both labels' samples of one bench and config,
the newest per run id.

## Tests and the gate

Tests run per area. The GPU-heavy suites are never run unscoped:

```bash
bazel run //:pytest -- <fsla checkout> tests/models
bazel run //:pytest -- <fsla checkout> tests/mqar
```

The gate is `bazel test //:gate`: ruff with fsla's rule set and its E30x spacing family over `fsla_bench` and `tests`,
the records' writers and readers on a temporary store (`tests.results.test_records`), the declaration check, and the doc
references (`//:docrefs`,
fsla-devtools' `fsla_devtools.docrefs`: every citation these docs make of the tree pinned in `references.lock.json` and
read against its target), on the bench environment. The commit hook (`.githooks/pre-commit` -> `bazel/bazel --commit-gate`, fsla-devtools' commit gate) runs
it on the commit as git will write it.

Bazel is fsla's pinned bazelisk (a fsla checkout's `python3 tools/bootstrap.py` links it), and it runs through this
checkout's front door (`bazel/bazel`), which runs fsla-devtools' from the machine's fsla-devtools checkout (the dev
config's `workspace.devtools`, else `fsla-devtools` beside this checkout).

## Publishing

This repository is developed in public: `Blakexx/fsla-bench` is the development repository, with its whole history. The
commit and push hooks run fsla-devtools' tripwire over what a change adds, under `.github/declarations.json`.

## Images

| Image | Dockerfile | Benchmarks |
|---|---|---|
| `blakeresearch/fsla-bench` | `docker/mqar.Dockerfile` | MQAR |

It builds `FROM blakeresearch/fleet-base` and sets `FLEET_RUN_CMD` and `FLEET_VERIFY_CMD`
(`python -m fsla_bench.fleet.verify`). `docker/build.sh mqar [--push] [--prune]` stages clean sources from the
checkouts named by `FSLA_FLA_SRC`, `FSLA_ZOO_SRC` (the fork's `fsla-pins`) and `FSLA_RECALL_SRC` (chromatic-recall), each at its pinned commit. The images do not install fsla yet, and fla does not import
without it, so no current image builds. The fleet base's torch is a CUDA 12.8 build.
