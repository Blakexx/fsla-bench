"""THE BENCH'S MEASUREMENTS, declared from its groups (`@bench_groups//:groups.bzl`, `fsla_bench/measure/groups.py`) over
the fsla bundles (`@candidate`, `@reference`: fsla-devtools' `bundle` tag) and the bench's own attention reference.

For each group and each arm set it times together, ONE SESSION (`session_<group>_<arms>`) over every bundle's entry of
those arms on the group's cells and the attention reference's, the candidate's clock reader proving the clock; its
NULL GATE (`null_<group>_<arms>`) over the same members, so every pair the session compares is read beside what the
gate found; one MEMORY pass a group (`memory_<group>`). An FSLA arm is a label: an arm the candidate does not export
fails this package's analysis (audit D4); a reference contributes the entries it exports, since a commit may predate an
arm. For each diff surface both bundles export and this bench holds a rule for (`RULES`), one verdict across the two
commits (`diff_<surface>`), its gate and its store; they are manual, run on purpose (`bazel test //measure:diffs`),
since a verdict produces both commits' sides. Which bundle is the reference for a ratio is chosen when the records are
read (`python -m fsla_devtools.results verdict --reference @reference`).
"""

load("@bench_groups//:groups.bzl", "GATE", "GROUPS")
load("@candidate//:exports.bzl", CANDIDATE = "ENTRIES")
load("@fsla_cells//:cells.bzl", "CELLS")
load("@reference//:exports.bzl", REFERENCE = "ENTRIES")
load("//bazel:defs.bzl", "diff_test", "diff_verdict", "interleaved_timing", "measure_memory", "results_store", "timing_entry")

#: the fsla bundles a session holds, by their repositories
BUNDLES = {"@candidate": CANDIDATE, "@reference": REFERENCE}

#: THE RULES A SURFACE IS COMPARED UNDER ACROSS COMMITS. The oracle is BIT-IDENTICAL: both commits run fp64 on identical
#: inputs, so a rewrite that only nearly reproduces the old one has changed the definition. The producer is
#: SUPPORT-EQUAL: its declared purpose was to replace a bisection with a closed form, so agreement to 1e-13 is the result
#: and the support set is the claim with teeth. The carry kernel's is per slot at twice the reassociation its commit
#: measured, which no commit states yet: it has no rule here until one does.
RULES = {
    "oracle": ("bit-identical", {}),
    "producer": ("support-equal", {"rtol": 0.0, "atol": 1e-13}),
}

def _qkv(group):
    return [c for c in group["cells"] if CELLS[c]["kind"] == "qkv"]

def _flash_arms(group):
    """The attention reference's arms a group times: `flash`, and the group's own (`flash@T=...`), with parameters."""
    return dict([("flash", {})] + [(arm, params) for arm, params in group["flash_arms"]])

def flash_entries():
    """The attention reference on every QKV cell a group names it on, an entry an arm and a cell."""
    declared = {}
    for group in GROUPS.values():
        for arm, params in _flash_arms(group).items():
            for cell in _qkv(group):
                declared["%s_%s" % (arm, cell)] = (arm, params, cell)
    for name, (_arm, params, cell) in sorted(declared.items()):
        timing_entry(
            name = name,
            cells = ["@fsla_cells//:" + cell],
            executor = "fsla_bench.measure.attention:flash",
            params = params,
        )

def _fsla_members(bundle, entries, arm, cells):
    return ["%s//:%s" % (bundle, name) for name, e in sorted(entries.items())
            if e["kind"] == "timed" and e["params"].get("arm") == arm and e["cells"][0] in cells]

def _members(group, arms):
    flash, members = _flash_arms(group), []
    for arm in arms:
        if arm in flash:
            members += [":%s_%s" % (arm, cell) for cell in _qkv(group)]
            continue
        if not [e for e in CANDIDATE.values() if e["params"].get("arm") == arm]:
            fail("group %s times %s, which the candidate exports no entry of" % (group["name"], arm))
        for bundle, entries in BUNDLES.items():
            members += _fsla_members(bundle, entries, arm, group["cells"])
    return members

def sessions():
    """Each group's sessions, their null gates and its memory pass; `gate`, the gate's groups' sessions."""
    gate = []
    for name, group in sorted(GROUPS.items()):
        everyone = []
        for arms in group["together"]:
            members = _members(group, arms)
            if not members:
                continue
            target = "%s_%s" % (name, "+".join(arms))
            interleaved_timing(
                name = "session_" + target,
                claim = "unlocked",
                clock = "@candidate//:clock",
                entries = members,
            )
            interleaved_timing(
                name = "null_" + target,
                claim = "unlocked",
                clock = "@candidate//:clock",
                entries = members,
                null_gate = True,
            )
            everyone += [m for m in members if m not in everyone]
            if name in GATE:
                gate.append(":session_" + target)
        if everyone:
            measure_memory(
                name = "memory_" + name,
                entries = everyone,
            )
    native.filegroup(
        name = "gate",
        srcs = gate,
    )

def diffs():
    """Each surface both bundles export a side of, compared under `RULES`: its verdict, its gate and its store."""
    tests = []
    for name, side in sorted(CANDIDATE.items()):
        surface = name.removeprefix("side_")
        if side["kind"] != "side" or surface not in RULES or name not in REFERENCE:
            continue
        strategy, params = RULES[surface]
        diff_verdict(
            name = "diff_" + surface,
            devices = "@host_gpu//:identity.txt" if side["device"] else None,
            left = "@candidate//:" + name,
            minimum = len(side["cells"]),
            params = params,
            right = "@reference//:" + name,
            strategy = strategy,
            tags = ["manual"],
        )
        diff_test(
            name = "diff_%s_test" % surface,
            tags = ["manual"],
            verdict = ":diff_" + surface,
        )
        results_store(
            name = "store_diff_" + surface,
            src = ":diff_" + surface,
            tags = ["manual"],
        )
        tests.append(":diff_%s_test" % surface)
    native.test_suite(
        name = "diffs",
        tags = ["manual"],
        tests = tests,
    )
