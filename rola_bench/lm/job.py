"""LM benchmark job builder — iso-backbone RoLA-vs-attention, spec-driven on the global layer.

`lm_job(config)` reads rola_bench/lm/experiments/<config>.yaml (arms × seeds + backbone/geometry/
budget + fleet) and builds the fleet.Job via rola_bench.jobs.make_job. The spec's backbone/geometry/
budget sections are flattened to the LM_* box env the runner (run_lm.py) + arms.py read, so the spec
file is the single source of truth. Results: results/lm/<config>/<cell>.json (cell = <arm>_s<seed>).
LM keeps preemption-safe checkpoint resume (payload_for + pull_in_progress + no_kill).

Launch: launch.py lm default
"""
import os
import re

from rola_bench.jobs import make_job
from rola_bench.spec import load, env_from

CKPT_ROOT = os.environ.get("ROLA_LM_CKPTS", "/mnt/e/rola/rola_data/lm_ckpts")


def _rola_overrides(arm):
    if "__" not in arm:
        return {}
    _, suffix = arm.split("__", 1)
    out = {}
    for tok in suffix.split("_"):
        m = re.fullmatch(r"(nh|dv|dqk|nc)(\d+)", tok)
        if m:
            out[{"nh": "rola_nh", "dv": "rola_dv", "dqk": "rola_dqk", "nc": "rola_nc"}[m.group(1)]] = int(m.group(2))
    return out


def _active_state_floats(arm, hidden, geometry):
    if arm.startswith("rola-"):
        g = {**geometry, **_rola_overrides(arm)}
        nh, nc, dqk, dv = (int(g["rola_nh"]), int(g["rola_nc"]), int(g["rola_dqk"]), int(g["rola_dv"]))
        return nh * nc * dqk * dv
    if arm == "gdn":
        hd = int(geometry.get("gdn_head_dim", 256))
        return int(geometry.get("gdn_nh", 8)) * hd * int(hd * float(geometry.get("gdn_expand_v", 2.0)))
    if arm == "gdn-isostate":
        hd = int(geometry.get("gdniso_head_dim", geometry.get("gdn_head_dim", 256)))
        return int(geometry.get("gdniso_nh", geometry.get("gdn_nh", 8))) * hd * int(
            hd * float(geometry.get("gdniso_expand_v", geometry.get("gdn_expand_v", 2.0))))
    if arm == "gla":
        nh = int(geometry.get("gla_nh", 4))
        return int(hidden * float(geometry.get("gla_ek", 2.0))) * int(hidden * float(geometry.get("gla_ev", 8.0))) // nh
    if arm == "gla-isostate":
        nh = int(geometry.get("glaiso_nh", geometry.get("gla_nh", 4)))
        return int(hidden * float(geometry.get("glaiso_ek", geometry.get("gla_ek", 2.0)))) * int(
            hidden * float(geometry.get("glaiso_ev", geometry.get("gla_ev", 8.0)))) // nh
    return None


def _validate_half_kv_states(spec):
    """Active bounded LM arms must use state = block_size * hidden = half of attention KV cache."""
    backbone = spec.get("backbone") or {}
    geometry = spec.get("geometry") or {}
    budget = spec.get("budget") or {}
    hidden = int(backbone.get("hidden", 512))
    block_size = int(budget.get("block_size", 2048))
    target = hidden * block_size
    bad = []
    for arm in spec["arms"]:
        state = _active_state_floats(arm, hidden, geometry)
        if state is None:
            continue
        if state != target:
            bad.append(f"{arm}: state={state}, want={target}")
        if arm.startswith("rola-"):
            g = {**geometry, **_rola_overrides(arm)}
            x_axis = int(g["rola_nc"]) * int(g["rola_dqk"])
            out_width = int(g["rola_nh"]) * int(g["rola_dv"])
            if x_axis != block_size:
                bad.append(f"{arm}: rola_nc*rola_dqk={x_axis}, want block_size={block_size}")
            if out_width != hidden:
                bad.append(f"{arm}: rola_nh*rola_dv={out_width}, want hidden={hidden}")
    if bad:
        raise ValueError("LM half-KV state validation failed: " + "; ".join(bad))


def lm_job(config="default"):
    spec = load("lm", config)
    _validate_half_kv_states(spec)
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
