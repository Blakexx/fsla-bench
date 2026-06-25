"""CPU-only structural tests for the perf e2e arm-registry (no GPU, no kernel calls).

Run with the project venv (no GPU needed):

    PYTHONPATH=. /mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python \
        rola_bench/perf/test_perf_cpu.py

These assert the e2e registry is well-formed. They do NOT instantiate any module or run any CUDA
kernel (build_e2e needs a GPU; it is exercised by the real bench run, not here).
"""
from rola_bench.perf.scratch import chunked as ch


def test_e2e_registry_wellformed():
    names = [a["name"] for a in ch.E2E_ARMS]
    assert len(names) == len(set(names)), "e2e arm names must be unique"
    for a in ch.E2E_ARMS:
        assert a["variants"] in ("rla_gla", "own"), a["name"]
        # method: None for the routed RoLA arm, else a baseline-method string ('dispatch' for
        # monolith, which picks rla/gla at build time).
        assert a["method"] is None or isinstance(a["method"], str), a["name"]
        assert isinstance(a["shapes"], list) and a["shapes"], a["name"]
    # exactly one routed RoLA arm (method=None), and it sweeps the native shape only
    rola = [a for a in ch.E2E_ARMS if a["method"] is None]
    assert len(rola) == 1 and rola[0]["name"] == "rola", rola
    assert rola[0]["shapes"] == ["native"], rola[0]
    print("test_e2e_registry_wellformed PASS")


if __name__ == "__main__":
    test_e2e_registry_wellformed()
    print("ALL PASS")
