"""Gate for the engine import seam (`rola_bench.engine`).

The seam is only worth anything if it is the ONLY door. These tests enforce that: a new
`from fla_rola... import ...` added anywhere in the package fails here, which is the whole point —
the coming `fla_rola` -> `rola` rename must stay a one-line change in `engine.py`.

Also gates the seam's two behavioural promises: it resolves nothing at import time (several call
sites import inside functions specifically to keep Triton off the module-import path), and an
unknown symbol says what to do rather than raising a bare AttributeError.

CPU-only; the source scan touches no engine at all.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

from rola_bench import engine

_PKG = pathlib.Path(__file__).resolve().parent

# Files allowed to import the engine package directly.
#   engine.py           — is the seam.
#   perf/engine_bench.py — its V2-era WIP is parked at
#       rola-scratch/workflows/20260802-thunder/engine_bench_v2era.patch and that patch carries the
#       import line as diff CONTEXT, so repointing it would stop the patch applying. Repoint when
#       the patch lands or is dropped. Explicitly not paper material; retires with the V2 line.
#   lm/flame/           — a vendored third-party trainer fork, not ours to edit.
_EXEMPT = {"engine.py", "perf/engine_bench.py"}
_VENDORED = "lm/flame/"


def _package_sources():
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG).as_posix()
        if rel.startswith(_VENDORED) or rel in _EXEMPT:
            continue
        yield rel, path


def _engine_imports(path):
    """Every `import fla_rola...` / `from fla_rola... import ...` in one file, by AST (not grep —
    a mention inside a docstring or a comment is documentation, not a coupling)."""
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == engine.ENGINE_PACKAGE or alias.name.startswith(
                        engine.ENGINE_PACKAGE + "."):
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == engine.ENGINE_PACKAGE or node.module.startswith(
                    engine.ENGINE_PACKAGE + "."):
                hits.append((node.lineno, f"from {node.module} import ..."))
    return hits


def test_the_seam_is_the_only_door_to_the_engine():
    offenders = []
    for rel, path in _package_sources():
        for lineno, what in _engine_imports(path):
            offenders.append(f"{rel}:{lineno}  {what}")
    assert not offenders, (
        "these files reach the engine package directly instead of through `rola_bench.engine`, "
        "which is what the coming fla_rola->rola rename has to edit one file to survive:\n  "
        + "\n  ".join(offenders))


def test_the_scan_actually_scans_something():
    """A guard on the guard: if the walk or the exemption list silently matched everything, the
    test above would pass vacuously."""
    files = list(_package_sources())
    assert len(files) > 20, f"only {len(files)} files scanned"
    assert any(rel == "named_configs.py" for rel, _ in files)


def test_seam_import_does_not_pull_in_the_engine():
    """Laziness is load-bearing: importing the seam must not import the engine (and therefore must
    not import Triton), or the CPU-only config gates would need a CUDA driver to collect.
    Checked in a SUBPROCESS — an in-process check would be contaminated by this session's own
    engine imports."""
    code = (
        "import sys; import rola_bench.engine as e; "
        f"assert '{engine.ENGINE_PACKAGE}' not in sys.modules, 'seam import pulled in the engine'; "
        "print('lazy')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(_PKG.parent))
    assert out.returncode == 0, out.stderr
    assert "lazy" in out.stdout


@pytest.mark.parametrize("name", ["RoLA", "LevelRoutingConfig", "RoLAConfig",
                                  "routing_factor_levels", "naive_rola_v3", "EntmaxActivation"])
def test_load_bearing_symbols_resolve_through_the_seam(name):
    assert getattr(engine, name) is not None
    assert name in dir(engine)


def test_unknown_symbol_says_what_to_do():
    with pytest.raises(AttributeError, match="_SYMBOLS"):
        engine.no_such_engine_symbol


def test_import_engine_returns_the_package_and_submodules():
    assert engine.import_engine().__name__ == engine.ENGINE_PACKAGE
    assert engine.import_engine("layers").__name__ == f"{engine.ENGINE_PACKAGE}.layers"
