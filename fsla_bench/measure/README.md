# `fsla_bench.measure` — two fsla commits and the libraries they are compared against, measured in one session

The local measurement tier of fsla-bench. Its targets are `measure/` at the repository's root, over fsla-devtools'
measurement rules (`bazel/measure.bzl`): every session is fsla-devtools' session program (`fsla_devtools.timing`), and
every input is a central cell (`fsla_devtools.cells`). This package holds the groups (`groups.py`) and the attention
reference (`attention.py`); it defines no measurement of fsla's.

    bazel run //measure:session_<group>_<arms> --repo_env=FSLA_CANDIDATE=<fsla checkout>   # one session, stored
    bazel run //measure:null_<group>_<arms> --repo_env=FSLA_CANDIDATE=<fsla checkout>      # its null gate
    bazel run //measure:memory_<group> --repo_env=FSLA_CANDIDATE=<fsla checkout>           # each member alone
    bazel run //measure:session_<...> --repo_env=FSLA_CANDIDATE=... -- --no-store          # measure, store nothing
    bazel test //measure:diffs --repo_env=FSLA_CANDIDATE=<fsla checkout>                   # the verdicts across commits
    bazel query //measure:all --repo_env=FSLA_CANDIDATE=<fsla checkout>                    # every target

## The bundles

A timing compares only inside the session that interleaved it, so the two commits are entries of ONE run.
`MODULE.bazel` names two BUNDLES (fsla-devtools' `bundle` tag): `candidate`, the fsla checkout a run names
(`--repo_env=FSLA_CANDIDATE=PATH`: a module names no machine path), built in place with the machine's fsla-devtools, and
`reference`, a commit of fsla, checked out under the machine's store and built by its own pins; moving it is moving a
pin. Each fetch builds what the commit's `//measure:all` holds of timing entries, its clock reader and its diff sides,
and copies it into the repository: `@reference//:entmax_solve_<cell>` is fsla's entry of that commit, its bundle (its
Python, its library, its build record) the commit's own. A record names each member by that label, its bundle's digest
and its stamped commit. A commit older than fsla-devtools' `timing_entry` rule is no bundle.

`attention.py` is fsla-bench's own entry: `flash`, causal attention through torch's forced flash backend, a timing entry
on every QKV cell a group names (`flash_<cell>`, and each of a group's own arms, `flash@T=4096_<cell>`), in the bench's
bundle (`//measure:bundle`). Where no session can be declared (a rented card with no fsla bundle built), `python -m
fsla_bench.measure.attention CELL [--reps N] [--warmup N] [--param k=v]` builds the entry on a central cell and reads its
own stopwatch: one JSON line of what it built and its calls' milliseconds.

## Groups and sessions (`groups.py`, `measure/sessions.bzl`)

A GROUP is a selection of central cells and its relation, as code: what it holds (`L1024-N65536-dv64`: 1024 tokens,
value width 64, FSLA's state N = 65536, attention one bf16 head), the parameters its cells share, and the arm sets it
times together. `check` refuses a group whose cells break its claim; Bazel reads the groups through the bench's
`groups` extension (`bazel/groups.bzl`, `described()`), so a broken group fails the fetch. `GATE` names the groups whose
sessions `//measure:gate` builds.

For each group and arm set, one SESSION (`session_<group>_<arms>`): every bundle's entry of those arms on the group's
cells and the attention reference's, the candidate's clock reader proving the clock. An FSLA arm is a LABEL: a group
that names an arm the candidate exports no entry of fails the analysis; a reference contributes the entries it exports,
since a commit may predate an arm. Every entry is set up before any call (a barrier), each in its bundle's worker on its
bundle's interpreter; then the warm-up, then rounds of reps, each rep every entry once in a fresh random order, under
the GPU lock and the clock lock, an untimed reset before every call. An entry that cannot run on a cell (a binary
without the arm, a setup that does not fit) is recorded as that member's failure and the session goes on. Its NULL
GATE (`null_<group>_<arms>`) times the same members against copies of themselves in second workers: where the per-rep
ratios put one outside their interquartile range, a worker's bias is found, and the verdict does not judge a pair
whose gate found one. One MEMORY pass a group (`memory_<group>`) takes each member alone. Every session claims
`unlocked`: the rented card's container declares no clock lock, and a host that declares one still locks and proves it.

## The diffs

For each diff surface both bundles export (`side_oracle`, `side_producer`, ...) and this bench holds a rule for
(`RULES` in `measure/sessions.bzl`: the oracle bit-identical, the producer support-equal; the carry kernel's has none
until a commit states its reassociation), one verdict across the two commits (`diff_<surface>`), its gate
(`diff_<surface>_test`) and its store (`store_diff_<surface>`). They are manual -- a verdict produces both commits'
sides, the fp64 oracle's included -- and `//measure:diffs` runs every gate.

## Records

Each target that stores keeps its samples at its own label (`records/fsla_bench/measure/<target>/<run>.json` in the data
repository): a session at `@fsla_bench//measure:session_<group>_<arms>`, its null gate at `null_<group>_<arms>`, a memory
pass at `memory_<group>` (each stored by the session program), a cross-commit verdict at `store_diff_<surface>`, through
`fsla_devtools.results`. A
session's samples are raw and ordered (round, rep, position); which bundle is the reference for a ratio is chosen when
the records are read (`python -m fsla_devtools.results verdict --reference @reference`), within a session.

## Tests

The groups are checked against the central cells at every fetch of `@bench_groups`; fsla-devtools' own tests check the
session program, the bundle export and the diff (`test_session`, `test_export`, `test_diff`).
