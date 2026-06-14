"""CPU-only structural tests for the perf arm-registry (no GPU, no kernel calls).

Run with the project venv (no GPU needed):

    PYTHONPATH=. /mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python \
        rola_bench/perf/test_perf_cpu.py

These assert the registry is well-formed and the matched-state sizing holds the recurrent-state
float count per head equal across architectures (nc*d_qk*d_v). They do NOT run any CUDA kernel.
"""
from rola_bench.perf import chunked as ch
from rola_bench.perf import _common as C


def test_registry_wellformed():
    names = [a["name"] for a in ch.ARMS]
    assert len(names) == len(set(names)), "arm names must be unique"
    for a in ch.ARMS:
        assert a["variants"] in ("rla_gla", "own"), a["name"]
        assert callable(a["make_inputs"]) and callable(a["run"])
        assert isinstance(a["grad_keys"], list) and a["grad_keys"]
        assert a["projection_cfg"] is None or callable(a["projection_cfg"])
    # the tex column lists must reference real arms
    for impl in ch.TEX_ROLA_IMPLS + ch.TEX_OWN_IMPLS:
        assert impl in names, impl
    # projection fallbacks must point at arms that DO have a projection_cfg
    cfg_names = {a["name"] for a in ch.ARMS if a["projection_cfg"] is not None}
    for src, dst in ch.PROJ_FALLBACK.items():
        assert dst in cfg_names, (src, dst)
    print("test_registry_wellformed PASS")


def test_matched_state():
    """Each cross-arch arm's recurrent-state float count per head equals nc*d_qk*d_v
    (exactly for GDN/Hedgehog; within the integer-rounding of the Taylor formula for Based)."""
    for nc in C.NCS:
        tgt = ch.TOTAL_STATE(nc)
        g = ch.gdn_dims(nc)
        assert g["d_k"] * g["d_v"] == tgt, ("gdn", nc, g)
        h = ch.hedgehog_dims(nc)
        assert 2 * h["d_k"] * h["d_v"] == tgt, ("hedgehog", nc, h)
        b = ch.based_dims(nc)
        feat = 1 + b["K"] + b["K"] ** 2
        # Based state = feat * V; V is round(tgt/feat), so |state - tgt| < feat.
        assert abs(feat * b["d_v"] - tgt) < feat, ("based", nc, b, feat * b["d_v"], tgt)
        assert b["K"] <= 16, ("based K must respect the kernel cap", b)
    print("test_matched_state PASS")


def test_leaves_passthrough():
    """_leaves marks grad on tensor keys and passes non-tensor entries (e.g. a feature-map
    module) through untouched."""
    import torch
    t = dict(q=torch.randn(2, 3), const=torch.randn(2, 3), mod=object())
    out = ch._leaves(t, ["q"])
    assert out["q"].requires_grad and not out["const"].requires_grad
    assert out["mod"] is t["mod"]
    print("test_leaves_passthrough PASS")


if __name__ == "__main__":
    test_registry_wellformed()
    test_matched_state()
    test_leaves_passthrough()
    print("ALL PASS")
