"""Uniform result store for every benchmark (mqar / lm / perf / similarity).

Layout, no exceptions:
    results/<bench>/<config>/<run_id>.jsonl
      - outer dir   = the config name
      - inner file  = the run_id
      - records     = that run's cell row(s)  (append-only; a re-run adds a row)

A run_id is the filename, so resume is just "does <run_id>.jsonl have an ok record". One file per
cell => parallel writers never contend. This module is the ONLY way results are written/read.
"""
import glob
import json
import os


def cell_file(config_dir, run_id):
    return os.path.join(config_dir, f"{run_id}.jsonl")


def write_cell(config_dir, run_id, record):
    """Append one record to results/<config>/<run_id>.jsonl (creating the config dir)."""
    os.makedirs(config_dir, exist_ok=True)
    with open(cell_file(config_dir, run_id), "a") as fh:
        fh.write(json.dumps(record) + "\n")


def done_ids(config_dir):
    """run_ids (inner-file basenames) that have at least one ok record."""
    done = set()
    for f in glob.glob(os.path.join(config_dir, "*.jsonl")):
        try:
            if any(json.loads(l).get("ok") for l in open(f) if l.strip()):
                done.add(os.path.basename(f)[:-len(".jsonl")])
        except Exception:
            pass
    return done


def all_records(config_dir):
    """Every record under a config dir (sorted by run_id), newest row per file last."""
    for f in sorted(glob.glob(os.path.join(config_dir, "*.jsonl"))):
        for l in open(f):
            try:
                yield json.loads(l)
            except Exception:
                pass
