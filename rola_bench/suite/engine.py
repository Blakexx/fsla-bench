"""THE ENGINE: a DAG of measurement nodes, each keyed by what its result depends on, each result kept.

A NODE is one action on one unit: a module (what it runs and on what), its unit (a cell, a subject at a cell), and the
nodes it depends on. Its KEY is sha256 over the node's semantics -- the module, the unit, the identity of every input the
module declares (a binary, an instrument's code, the environment) -- and, for each dependency, that node's key and the
digest of its output. So a change anywhere re-keys exactly the nodes downstream of it, and a revert finds the old key's
result again.

A RECORD is the store's file for one key, `<store>/suite/<module>/<key>.json`: the semantics that made the key, whether
the node succeeded, its error when it did not, and its SAMPLES -- every time the node ran under that key, the raw output,
when and how long. The output itself is `<key>.<n>.out.json` beside it. A key with an ok record is COMPLETE and does not
run again; a failed record is retried; `repeat` adds a sample to a complete key of a repeatable module. Nothing derived
from two records -- a ratio, a delta between commits -- is ever stored: that is the reader's.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Node:
    """One action on one unit. `identity` is every input the action depends on besides its dependency nodes, already
    reduced to hashes (a binary's sha256, an instrument's code key, the environment key); `run(outputs, dest)` writes the
    raw output to `dest`, reading its dependencies' outputs from `outputs`."""

    module: str
    unit: str
    identity: dict
    run: Callable[[dict[str, Path], Path], None]
    deps: tuple[str, ...] = ()
    repeatable: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.module}@{self.unit}" if self.unit else self.module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(nodes: list[Node]) -> list[Node]:
    """Dependencies first; refuses a dependency that is not a node of the graph, and a cycle."""
    by_name = {n.name: n for n in nodes}
    done: list[Node] = []
    state: dict[str, str] = {}

    def visit(n: Node) -> None:
        if state.get(n.name) == "done":
            return
        if state.get(n.name) == "open":
            raise SystemExit(f"a dependency cycle through {n.name}")
        state[n.name] = "open"
        for d in n.deps:
            if d not in by_name:
                raise SystemExit(f"{n.name} depends on {d}, which is not in the graph")
            visit(by_name[d])
        state[n.name] = "done"
        done.append(n)

    for n in nodes:
        visit(n)
    return done


@dataclass
class Outcome:
    name: str
    key: str
    status: str
    wall_s: float = 0.0
    samples: int = 0
    error: str = ""


class Store:
    """The suite's records under `<root>/suite/`."""

    def __init__(self, root: Path):
        self.root = root / "suite"

    def paths(self, module: str, key: str) -> tuple[Path, Path]:
        base = self.root / module
        return base / f"{key}.json", base

    def load(self, module: str, key: str) -> dict | None:
        path, _ = self.paths(module, key)
        return json.loads(path.read_text()) if path.exists() else None

    def output(self, module: str, key: str, record: dict) -> Path:
        _, base = self.paths(module, key)
        return base / record["samples"][-1]["output"]

    def save(self, module: str, key: str, record: dict) -> None:
        path, base = self.paths(module, key)
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        tmp.replace(path)


def key_of(node: Node, dep_keys: dict[str, tuple[str, str]]) -> tuple[str, dict]:
    semantics = {"module": node.module, "unit": node.unit, "identity": node.identity,
                 "deps": {d: {"key": k, "output": o} for d, (k, o) in sorted(dep_keys.items())}}
    return hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()[:24], semantics


def run(nodes: list[Node], store: Store, *, repeat: bool = False, force: bool = False, dry: bool = False,
        log: Callable[[str], None] = print) -> list[Outcome]:
    """Walk the graph in dependency order; run what is not complete (or, with `repeat`, what is repeatable; with `force`,
    everything); keep every result. A node whose dependency failed or was not run is BLOCKED, and stays unrecorded."""
    outputs: dict[str, Path] = {}
    dep_keys: dict[str, tuple[str, str]] = {}
    outcomes: list[Outcome] = []
    pending: set[str] = set()
    for node in order(nodes):
        missing = [d for d in node.deps if d not in outputs]
        if dry and missing and all(d in pending for d in missing):
            pending.add(node.name)
            outcomes.append(Outcome(node.name, "", "pending"))
            log(f"pending   {node.name} (after {', '.join(missing)})")
            continue
        if missing:
            outcomes.append(Outcome(node.name, "", "blocked", error=f"not run: {', '.join(missing)}"))
            log(f"blocked   {node.name} (no result for {', '.join(missing)})")
            continue
        key, semantics = key_of(node, {d: dep_keys[d] for d in node.deps})
        record = store.load(node.module, key)
        complete = bool(record and record.get("ok"))
        if complete and not force and not (repeat and node.repeatable):
            outputs[node.name] = store.output(node.module, key, record)
            dep_keys[node.name] = (key, digest(outputs[node.name]))
            outcomes.append(Outcome(node.name, key, "complete", samples=len(record["samples"])))
            log(f"complete  {node.name} [{key[:12]}] {len(record['samples'])} sample(s)")
            continue
        if dry:
            pending.add(node.name)
            outcomes.append(Outcome(node.name, key, "pending"))
            log(f"pending   {node.name} [{key[:12]}]" + (" (failed before)" if record else ""))
            continue
        samples = record["samples"] if complete else []
        _, base = store.paths(node.module, key)
        base.mkdir(parents=True, exist_ok=True)
        dest = base / f"{key}.{len(samples)}.out.json"
        started = time.time()
        try:
            node.run({d: outputs[d] for d in node.deps}, dest)
            if not dest.exists():
                raise RuntimeError("the action wrote no output")
        except (Exception, SystemExit) as ex:
            wall = time.time() - started
            store.save(node.module, key, {"ok": complete, "key": key, "semantics": semantics, "meta": node.meta,
                                          "samples": samples, "error": f"{type(ex).__name__}: {ex}"[-2000:],
                                          "trace": traceback.format_exc()[-3000:], "failed_utc": _utc()})
            outcomes.append(Outcome(node.name, key, "failed", wall, len(samples), str(ex)[-300:]))
            log(f"FAILED    {node.name} [{key[:12]}] {wall:.1f}s: {str(ex)[-200:]}")
            if complete:
                outputs[node.name] = store.output(node.module, key, record)
                dep_keys[node.name] = (key, digest(outputs[node.name]))
            continue
        wall = time.time() - started
        samples = [*samples, {"output": dest.name, "utc": _utc(), "wall_s": round(wall, 2)}]
        store.save(node.module, key, {"ok": True, "key": key, "semantics": semantics, "meta": node.meta, "samples": samples})
        outputs[node.name] = dest
        dep_keys[node.name] = (key, digest(dest))
        outcomes.append(Outcome(node.name, key, "ran", wall, len(samples)))
        log(f"ran       {node.name} [{key[:12]}] {wall:.1f}s" + (f", sample {len(samples)}" if node.repeatable else ""))
    return outcomes


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
