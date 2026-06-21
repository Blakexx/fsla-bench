"""Uniform result store: results/<bench>/<config>/<cell>.json — one JSON object per cell.

  outer dir = config (the experiment)
  inner file = the cell id  (one grid point incl seed; the dedup unit — the code/zoology field is
               still named `run_id`, but conceptually it is the cell)
  contents   = a single JSON object (the cell's result)

One file per cell => parallel writers never contend, and resume is just "does <cell>.json exist with
an ok result". (A single-record .jsonl was pointless; a shared .jsonl would contend — so: one .json
per cell, overwritten on re-run so the latest result wins.) This module is the only writer/reader.
"""
import glob
import json
import os


def cell_file(config_dir, cell_id):
    return os.path.join(config_dir, f"{cell_id}.json")


def write_cell(config_dir, cell_id, record):
    """Write one cell's result object to results/<config>/<cell>.json (overwrite — latest wins)."""
    os.makedirs(config_dir, exist_ok=True)
    with open(cell_file(config_dir, cell_id), "w") as fh:
        json.dump(record, fh)


def read_cell(config_dir, cell_id):
    p = cell_file(config_dir, cell_id)
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def done_ids(config_dir):
    """Cell ids (inner-file basenames) whose result object is ok."""
    done = set()
    for f in glob.glob(os.path.join(config_dir, "*.json")):
        try:
            if json.load(open(f)).get("ok"):
                done.add(os.path.basename(f)[:-len(".json")])
        except Exception:
            pass
    return done


def all_records(config_dir):
    """Every cell's result object under a config dir (sorted by cell id)."""
    for f in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
        try:
            yield json.load(open(f))
        except Exception:
            pass
