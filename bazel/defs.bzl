"""fsla-bench's rules: fsla-devtools' generic rules, each handed this workspace's environment (`bench_env`) and tracked
tree, the doc references (`docrefs_test`), and the runs on its two environments (`bench_python`, `research_python`)."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bench_env//:flags.bzl", "VENV")
load("@bench_tree//:tree.bzl", "TRACKED")
load("@fsla_devtools//bazel:rules.bzl", _docrefs_test = "docrefs_test", _tree_part = "tree_part", _tree_test = "tree_test")
load("@research_env//:flags.bzl", RESEARCH_VENV = "VENV")

def tree_test(**kwargs):
    """A check over the tree's tracked files, the bench environment's tools on PATH."""
    _tree_test(venv = VENV, **kwargs)

def tree_part(parts = {}):
    """This package's part of the tree: `tracked` and `tracked_python`."""
    _tree_part(TRACKED, parts)

def docrefs_test(**kwargs):
    """The doc references (fsla-devtools' `fsla_devtools.docrefs`), on the bench environment's interpreter, in the
    checkout."""
    _docrefs_test(python = Label("@bench_env//:python"), tool = Label("@docrefs//:docrefs.py"),
                  anchor = Label("//:MODULE.bazel"), **kwargs)

_PYTHON_RUN = """\
#!/bin/bash
# an environment's interpreter, in the checkout (fsla-bench's python_run){fsla_note}
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
{fsla}cd "$checkout"
export PYTHONPATH="$checkout{pythonpath}"
exec {venv}/bin/python {module} "$@"
"""

#: the run's first argument: the fsla checkout it imports, its library built (`bazel run //:library` there)
_FSLA = """fsla="${1:?the fsla checkout this run imports, its library built (bazel run //:library there)}"
shift
fsla="$(cd "${BUILD_WORKING_DIRECTORY:-.}" && cd "$fsla" && pwd)"
"""

def _python_run_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _PYTHON_RUN.format(
        fsla_note = ": its first argument the fsla checkout it imports" if ctx.attr.fsla else "",
        fsla = _FSLA if ctx.attr.fsla else "",
        pythonpath = ":$fsla" if ctx.attr.fsla else "",
        venv = shell.quote(ctx.attr.venv),
        module = ("-m " + shell.quote(ctx.attr.module)) if ctx.attr.module else "",
    ), is_executable = True)
    return [DefaultInfo(executable = script)]

python_run = rule(
    implementation = _python_run_impl,
    executable = True,
    attrs = {
        "venv": attr.string(mandatory = True, doc = "the environment's store venv"),
        "module": attr.string(doc = "the module it runs; none runs the interpreter on the run's arguments"),
        "fsla": attr.bool(doc = "the run imports fsla: its first argument is the fsla checkout, on the path after this " +
                                "one (both hold a `declare.py`)"),
    },
    doc = "A run on an environment's interpreter in the checkout (`bazel run`; its arguments follow `--`), the checkout " +
          "importable, and a fsla checkout after it where the run imports fsla.",
)

def bench_python(**kwargs):
    """A run on the bench environment (the fork's stack)."""
    python_run(venv = VENV, **kwargs)

def research_python(**kwargs):
    """A run on the research environment (the baselines' stack)."""
    python_run(venv = RESEARCH_VENV, **kwargs)
