# `fsla_bench.measure` — fsla checkouts and the libraries they are compared against, measured as one declared build

The local measurement tier of fsla-bench. The build is fsla-devtools' declared build system (`fsla_devtools.build`),
its sessions are its timing system's (`fsla_devtools.timing`), and every input is a central cell
(`fsla_devtools.cells`). The root is `declare.py` at the repository's root: it loads each fsla checkout's own
`declare.py` by path, composes them with this package's attention reference and groups, and declares the sessions, the
memory pass and the stores. This package defines no measurement of fsla's.

    bazel run //:suite -- plan declare.py:suite --arg target=worktree:PATH --only 'session/L1024*'
    bazel run //:suite -- run  declare.py:suite --arg target=worktree:PATH[,venv:PATH][,label:NAME] \
        [--arg references='worktree:PATH,label:master;worktree:PATH'] [--arg rounds=8] [--arg reps=11] \
        [--arg warmup=10] [--arg store_root=DIR] [--only GLOB]... [--skip GLOB]... [--force]
    bazel run //:suite -- run  declare.py:diffs --arg target=... --arg references=...   # the dual run

THE ROOT DECLARES EVERYTHING and takes no selector: which of it a build runs is pruned by LABEL at the CLI, which
knows nothing of cells, groups or parts -- `--only 'tip/side/oracle'` runs that side and what it needs;
`--skip '*/side/*'` drops the sides and everything that reads them; `--only 'session/*'` runs every session.

`//:suite` runs the build system on the bench environment, in this checkout; each fsla checkout runs in its own venv
(`venv:PATH`, by default the `venv-<checkout>` fsla's `bazel run //:dev_env` writes beside it).

## Checkouts

A checkout is a fsla worktree and the venv that runs it (`venv-<name>` beside it unless `venv:` names one); its label
scopes its targets (`tip/binary`) and defaults to its directory's name. The `target` is the first checkout, the
`references` the others. Each checkout's `declare.py` declares, in that checkout's venv and directory: its build
(cached while its binary stands), its machine facts, its timing registrations on the layer cells under each construction
(`entmax_solve@layer=C`, `layer_prefill@layer=C`, `decode_step@layer=C` and the op `layer_op@layer=C`), its clock reader and its DIFF SIDES (`SURFACES`: the fp64 oracle, the producer reference, the carry kernel, each over
the cells of its kind and tier) and its own kernel-vs-oracle diff. A checkout's instruments are its own Bazel targets, which this suite does not run. For each surface and each
reference, the root declares ONE DIFF (`fsla_devtools.diff`) of the two checkouts' sides under THIS ROOT'S rule for
that surface (`RULES`: the oracle bit-identical, the producer support-equal, a kernel per slot at twice the
reassociation its checkout states in `NUMERICS`, exact quantities exact) -- the oracle pair's dual run, and kernel-vs-kernel conformance -- stored at `diff/<surface>`;
`diffs` is the group over them. A worktree without a `declare.py` predates the declaration API and is not
compared.

`attention.py` is fsla-bench's own entry: `flash`, causal attention through torch's forced flash backend on every QKV
cell of every group, registered in the target's venv. Where no session can be declared (a rented card with no fsla
checkout built), `python -m fsla_bench.measure.attention CELL [--reps N] [--warmup N] [--param k=v]` builds the entry
on a central cell and reads its own stopwatch: one JSON line of what it built and its calls' milliseconds.

## Groups and sessions (`groups.py`)

A GROUP is a selection of central cells and its relation, as code: what it holds (`L1024-N65536-dv64`: 1024 tokens,
value width 64, FSLA's state N = 65536, attention one bf16 head), the parameters its cells share, and the arm sets it
times together. `check` refuses a group whose cells break its claim. `gate` selects `GATE`.

For each selected group and arm set the root declares one SESSION (`measure_timing`): every checkout's registration of
those arms, and the attention entry, on the group's cells. Every entry is built before any call (a barrier); then the
warmup, then rounds of reps, each rep every entry once in a fresh random order, under the GPU lock and the clock lock
with the target's clock reader proving the clock before and after, an untimed reset before every call. An entry that
cannot run on a cell (a binary without the arm, a setup that does not fit) is recorded as that member's failure and the
session goes on; a session that cannot run at all fails the build, and the server's stop still runs. One MEMORY pass
(`measure_memory`) takes each entry on each cell alone. One NULL GATE (`measure_null_gate`) times the target's
`layer_op` on each layer group's cell against copies of itself in second workers: where the per-rep ratios put
one outside their interquartile range, a worker's bias is found, and a ratio across checkouts' workers is read beside it.

## Records

A checkout's own diff at `fsla/<diff>`, the cross-checkout diffs at `diff/<surface>`, sessions at `timing/session`, the memory pass at `timing/memory`, the null gate at
`timing/null`, through `fsla_devtools.results`. A store target appends a run-stamped sample to the record its source's semantics
key -- the executor, its parameters, the cells' records and the draw, and every dependency's key and output -- so a
session of the same
checkouts' binaries on the same cells adds a sample to one record, and a session run from fsla's own root with the same
entries shares it. A session's samples are raw and ordered (round, rep, position); which checkout is the reference for a
ratio is chosen when the records are read, within a session.

## Tests

`bazel test //:suite_contract` (`python -m unittest tests.measure.test_suite`, in the gate) checks the groups against the central cells and the root's declarations
on fake checkouts, with no GPU; fsla-devtools' own tests (its `test_declared_build` and `test_timing`) check the build
and timing systems.
