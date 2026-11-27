"""THE GROUPS AS STARLARK: fsla-bench's module extension whose one repository (`@bench_groups//:groups.bzl`) holds every
group of `fsla_bench/measure/groups.py` -- which central cells launch together, the claim they hold, the arm sets timed
in one session -- checked against the central registry, for the sessions' macro (`measure/sessions.bzl`). The file is
run on the machine's python3 with fsla-devtools' package on its path (the registry is standard library only); a group
that breaks its claim, or names a cell the registry lacks, fails the fetch.

    groups = use_extension("//bazel:groups.bzl", "groups")
    use_repo(groups, "bench_groups")
"""

_GROUPS_BZL = """\
\"\"\"GENERATED from fsla_bench/measure/groups.py (`described()`): every group by name, its fields; the gate's groups.\"\"\"

GROUPS = {groups}

GATE = {gate}
"""

def _groups_impl(rctx):
    devtools = rctx.path(rctx.attr._devtools).dirname
    groups = rctx.path(rctx.attr._groups)
    #: re-read when the groups or the registry they are checked against move
    rctx.watch(groups)
    rctx.watch_tree(devtools.get_child("fsla_devtools").get_child("cells"))
    found = rctx.execute([rctx.which("python3"), "-B", "-S", "-c", "\n".join([
        "import importlib.util, json, sys",
        "sys.path.insert(0, sys.argv[1])",
        "spec = importlib.util.spec_from_file_location('groups', sys.argv[2])",
        "groups = sys.modules['groups'] = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(groups)",
        "d = groups.described()",
        "print(json.dumps({'groups': repr(d['GROUPS']), 'gate': repr(d['GATE'])}))",
    ]), str(devtools), str(groups)])
    if found.return_code != 0:
        fail("fsla_bench/measure/groups.py did not describe its groups:\n" + found.stderr)
    described = json.decode(found.stdout)
    rctx.file("groups.bzl", _GROUPS_BZL.format(groups = described["groups"], gate = described["gate"]))
    rctx.file("BUILD.bazel", 'exports_files(["groups.bzl"], visibility = ["//visibility:public"])\n')

_groups_repo = repository_rule(
    implementation = _groups_impl,
    attrs = {
        "_devtools": attr.label(default = "@fsla_devtools//:pyproject.toml"),
        "_groups": attr.label(default = "//:fsla_bench/measure/groups.py"),
    },
    doc = "fsla-bench's groups as Starlark, checked against the central registry.",
)

def _groups_ext_impl(_mctx):
    _groups_repo(name = "bench_groups")

groups = module_extension(
    implementation = _groups_ext_impl,
    doc = "fsla-bench's groups (`@bench_groups//:groups.bzl`).",
)
