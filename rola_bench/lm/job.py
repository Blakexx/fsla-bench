"""LM benchmark job builder — iso-backbone RoLA-vs-attention, spec-driven on the global layer.

`lm_job(config)` reads rola_bench/lm/experiments/<config>.yaml (arms × seeds + backbone/geometry/
budget + fleet) and builds the fleet.Job via rola_bench.jobs.make_job. The spec's backbone/geometry/
budget sections are flattened to the LM_* box env the runner (run_lm.py) + arms.py read, so the spec
file is the single source of truth. Results: results/lm/<config>/<cell>.json (cell = <arm>_s<seed>).
LM keeps preemption-safe checkpoint resume (payload_for + pull_in_progress + no_kill).

Launch: launch.py lm default
"""
import os

from rola_bench.jobs import make_job
from rola_bench.spec import load, env_from

CKPT_DIR = os.environ.get("ROLA_LM_CKPTS", "/mnt/d/rola_data/lm_ckpts")


def _payload_for(run_id):
    """Resume payload = the last in-progress checkpoint tar pulled for this cell (untarred + resumed
    on the box). None on first run."""
    p = os.path.join(CKPT_DIR, run_id + ".pt")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read()


def lm_job(config="default"):
    spec = load("lm", config)
    arms, seeds = spec["arms"], spec["seeds"]
    box_env = env_from("LM", spec.get("backbone"), spec.get("geometry"), spec.get("budget"))
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {**box_env, **fleet.get("env", {})}
    return make_job("lm", config, lambda: [f"{a}_s{s}" for a in arms for s in seeds],
                    "python run_lm.py", fleet, ckpt_dir=CKPT_DIR, payload_for=_payload_for)
