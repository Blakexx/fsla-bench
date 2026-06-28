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

CKPT_ROOT = os.environ.get("ROLA_LM_CKPTS", "/mnt/e/rola/rola_data/lm_ckpts")


def lm_job(config="default"):
    spec = load("lm", config)
    arms, seeds = spec["arms"], spec["seeds"]
    box_env = env_from("LM", spec.get("backbone"), spec.get("geometry"), spec.get("budget"))
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {**box_env, **fleet.get("env", {})}
    # PER-CONFIG ckpt dir: cell run_ids are <arm>_s<seed> with NO corpus, so two corpora share names
    # (rola-rla-kappa-asym_s1337 …). Isolating by config prevents one corpus's resume checkpoint from
    # overwriting another's in a shared dir (the run_corpus_suite.sh did this via ROLA_LM_CKPTS=…_$name).
    ckpt_dir = os.path.join(CKPT_ROOT, config)

    def _payload_for(run_id):  # resume payload = this cell's last in-progress ckpt tar; None on first run
        p = os.path.join(ckpt_dir, run_id + ".pt")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return fh.read()

    return make_job("lm", config, lambda: [f"{a}_s{s}" for a in arms for s in seeds],
                    "python run_lm.py", fleet, ckpt_dir=ckpt_dir, payload_for=_payload_for)
