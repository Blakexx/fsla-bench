# `rola_bench.suite` — rola checkouts measured by their own instruments, every result kept

The local measurement tier of rola-bench. A run takes the checkouts to measure, builds a graph of measurement nodes on
them, runs what is not already stored, and keeps every result in the measurements store (`store.root` in rola's dev
config, `<store>/suite/`). Standard library only: it runs from any rola venv or the dev container without fleet,
zoology or transformers. Rented, citable runs stay on the fleet tier (`launch.py`, `rola_bench/perf/METHODOLOGY.md`);
this tier is the engineering gate.

    python -m rola_bench.suite plan --target worktree:PATH --reference worktree:PATH,label:master,schedule:box/sparse-g32 --cells gate
    python -m rola_bench.suite run  --target ... [--reference ...] [--modules carry,timing] [--cells all|gate|a,b]
                                    [--subjects all|a,b] [--repeat] [--force]
    python -m rola_bench.suite show carry.phases --store <store root>

## Targets and arms

A target is `worktree:PATH[,venv:PATH][,label:NAME][,schedule:POLICY]`. The venv defaults to the sibling `venv-<name>`.
The schedule is the carry family's order policy the target's binary takes: `first`, or `DENSE/SPARSE` split on the
cell's name (master: `box/sparse-g32`).

Every node runs the target checkout's OWN instrument (rola's `tools/*.py --json`) from its own directory under its own
venv. A checkout is measured by its own definitions, the way rola's probe runs each binary's own worker.

A timing run's arms are explicit. The `--target` is the `subject`, and each `--reference` is a `reference` timed in the
same interleaved invocation. Latency is only comparable within one session, so the session records every arm's raw
times, role, commit and schedule. Ratios between arms are computed by the reader, and only within a session.

## Keys, completion, repeats

A node is one module on one unit (a cell, or a subject at a cell). Its key is sha256 over:
- the module and unit;
- the target's BINARY: sha256 of its built extension;
- the INSTRUMENT: its file and every repository file it imports, found by walking the imports, plus the data files and
  directories the module names (the cell registry, a budget, the kernel source a compile reads);
- the ENVIRONMENT, read from the machine: GPU and driver, ptxas, torch, Nsight Compute, the SM clock target;
- the module's parameters;
- for each node it depends on, that node's key and output digest.

A change anywhere re-keys exactly the nodes that depend on it, and a revert finds the old key's result again.

- A key with an ok record is complete and does not run.
- A failure is recorded with its error and retried on the next run. Nodes depending on it are blocked, not recorded.
- `--repeat` adds a sample to complete nodes of repeatable modules (every measurement, not the static SASS or register
  reads).
- `--force` runs every selected node again.

## Records

`<store>/suite/<module>/<key>.json` holds the semantics that made the key, `ok`, the error of the last failure, the
target's label, commit and dirtiness (`meta`), and the samples. Each sample is an output file `<key>.<n>.out.json`, when
it ran, and how long it took. Outputs are the instruments' raw JSON. Nothing relational is stored: no baseline ratio, no
delta between commits.

## Modules

| module | unit | instrument (rola) | raw output |
|---|---|---|---|
| `carry.sass` | the target | `tools/sass_gate.py` | SASS signatures of every carry kernel function, per cubin |
| `carry.registers` | arm 0 | `tools/life_ranges.py --arm 0` | peak live registers, by region |
| `carry.phases` | carry cell | `tools/phase_ledger.py` | cycles a warp a window per phase, per warp |
| `carry.counters` | carry cell | `tools/pipe_counters.py` | the profiler's pipe and resource counters, launch totals |
| `carry.census` | carry cell | `tools/stall_census.py` | stall samples by component and reason, phase and wavefront census |
| `carry.timeline` | carry cell | `tools/pipe_timeline.py` | the pipes' PM-sampled series over one launch |
| `timing.session` | subject @ cell [@ calls=N] | `tools/probe_cells.py` | every arm's round medians, clock, stamps, session id |

`--cells gate` is the kernel's first four gate cells (`GATE_CELLS`). Timing covers every subject of the target's bench
roster, at every call count it declares (`bench.subjects.Subject.calls`: the sequence as N carried calls), on every cell
`bench.subjects.applicable` admits for it; `--cells` narrows the carry cells, a layer subject takes every layer cell it
applies to. Each arm runs the unit as its own checkout spells it (`target.Lane`): a reference from before the call count
names a multi-call unit as a bench of its own (`PRE_COUNT_BENCHES`).

## Tests

`python -m unittest rola_bench.suite.test_engine` checks the engine's contract on fake nodes, with no GPU.
