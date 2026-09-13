"""LM benchmark job builder — iso-backbone RoLA-vs-attention, spec-driven on the global layer.

`lm_job(config)` reads rola_bench/lm/experiments/<config>.yaml (arms × seeds + backbone/geometry/
budget + fleet) and builds the fleet.Job via rola_bench.fleet.jobs.make_job. The spec's backbone/geometry/
budget sections are flattened to the LM_* box env the runner (run.py) + arms.py read, so the spec
file is the single source of truth. Results: `rola_results` at lm/<config> (cell = <arm>_s<seed>).
LM keeps preemption-safe checkpoint resume (payload_for + pull_in_progress + no_kill).

Launch: rola_bench/fleet/__main__.py lm default
"""
import os
import re

from rola_bench.fleet.jobs import make_job
from rola_bench.fleet.spec import load, env_from

CKPT_ROOT = os.environ.get("ROLA_LM_CKPTS", "/mnt/e/rola/rola_data/lm_ckpts")


def _rola_overrides(arm):
    if "__" not in arm:
        return {}
    _, suffix = arm.split("__", 1)
    out = {}
    for tok in suffix.split("_"):
        # `dqk` deleted with the V2 Q/K accounting (F1); it names nothing on the V3 layer. Mirrors
        # `rola_bench.lm.arms._STRUCT_KEYS`, which raises on an unrecognized token for the same
        # reason: a silently-dropped geometry override publishes a cell nobody specified.
        m = re.fullmatch(r"(nh|dv|nc)(\d+)", tok)
        if m:
            out[{"nh": "rola_nh", "dv": "rola_dv", "nc": "rola_nc"}[m.group(1)]] = int(m.group(2))
    return out


def _active_state_floats(arm, hidden, geometry):
    """Recurrent-state floats/layer for a SPEC arm, from the spec's own geometry block.

    RoLA (F1, 2026-08-02): `H * N * d_v`. V3's per-head state is `[N, d_v]`; the `rola_dqk` factor
    this used to carry was the V2 matrix-state shape and overstated the RoLA arms' state by exactly
    that factor (16x at the old defaults), which is what made the old state validator pass on
    cells that were nowhere near the baselines. `rola_dqk` is still READ from the spec if present
    only to REFUSE it — a spec that still sets it is a spec written against the deleted architecture,
    and silently ignoring the key would repeat the F3 failure mode.

    `gdn-isostate` / `gla-isostate` are gone (F4): they were byte-identical to `gdn`/`gla`.
    """
    if arm.startswith("rola-"):
        g = {**geometry, **_rola_overrides(arm)}
        if "rola_dqk" in g:
            raise ValueError(
                f"spec geometry sets rola_dqk={g['rola_dqk']!r}, but the V3 RoLA architecture owns "
                "no d_qk at all (its state is [N, d_v] per head). Re-pin this spec's RoLA geometry "
                "to (rola_nh, rola_nc, rola_dv) — see rola_bench.lm.arms, finding F1.")
        return int(g["rola_nh"]) * int(g["rola_nc"]) * int(g["rola_dv"])
    if arm == "gdn":
        hd = int(geometry.get("gdn_head_dim", 256))
        return int(geometry.get("gdn_nh", 8)) * hd * int(hd * float(geometry.get("gdn_expand_v", 2.0)))
    if arm == "gla":
        nh = int(geometry.get("gla_nh", 4))
        return int(hidden * float(geometry.get("gla_ek", 2.0))) * int(hidden * float(geometry.get("gla_ev", 8.0))) // nh
    return None


def _validate_matched_states(spec):
    """Every BOUNDED arm in a spec must carry the SAME recurrent state — that, and only that, is
    what makes an LM row a matched-state comparison.

    Was `_validate_half_kv_states` (2026-08-02): it required
    `state == hidden * block_size` = half of attention's KV cache at the training context. That
    target is retired with the user ruling of the same day — the LM tier's job is
    trains-stably-at-standard-scale, and the CAPACITY claim moved to the MQAR `N ∝ L` grid, so
    sizing every arm to a context-derived capacity number is no longer the point (and it is what
    pushed the baselines to 8x the standard head geometry). The arms are now sized to TEXTBOOK
    `d_k = d_v = 128 @ H = 8` in `rola_bench.lm.arms`; what a SPEC still has to guarantee is that
    its own geometry block does not break the match between them.

    Also gone with it: the two per-RoLA side conditions (`rola_nc*rola_dqk == block_size`, written
    in the deleted `d_qk`, and `rola_nh*rola_dv == hidden`, a no-output-bottleneck rule the current
    flagship deliberately crosses at `H*d_v = 2 x d_model`, exactly where GDN already sits).
    Attention is skipped throughout: it has no bounded state to match (`_active_state_floats`
    returns None).
    """
    backbone = spec.get("backbone") or {}
    geometry = spec.get("geometry") or {}
    hidden = int(backbone.get("hidden", 512))
    states = {}
    for arm in spec["arms"]:
        state = _active_state_floats(arm, hidden, geometry)
        if state is not None:
            states[arm] = state
    if len(set(states.values())) > 1:
        detail = "; ".join(f"{a}: state={s}" for a, s in sorted(states.items()))
        raise ValueError(
            "LM matched-state validation failed — the bounded arms of this spec do not carry the "
            f"same recurrent state, so the comparison is not matched: {detail}")


def lm_job(config="default"):
    spec = load("lm", config)
    _validate_matched_states(spec)
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
                    "python run.py", fleet, ckpt_dir=ckpt_dir, payload_for=_payload_for)
