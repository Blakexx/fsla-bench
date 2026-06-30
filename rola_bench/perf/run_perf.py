"""Box-side runner for the perf fleet job (FLEET_RUN_CMD -> this module).

Reads RUN_IDS (comma list of bench modules) from the env, runs each WHOLE bench on this box's GPU
preserving the in-process round-robin, and appends one ndjson summary row per run_id to
$FLEET_RESULTS — {"run_id", "ok", "gpu", <bench rows>}. The dispatcher's done_ids keys off that ok
row; the full per-cell table lives inside it (and in the CSVs the benches also write for the paper).

Mirrors mqar/run_rla_sweep.py's contract: never abort the batch on one cell's failure; write an
ok=False row with the error so the dispatcher records it rather than silently dropping it.
"""
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from rola_bench.spec import load

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
REPS = int(os.environ.get("PERF_REPS", "15"))


def _append(row):
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run_scaling(out_dir, which):
    """One bench-TYPE of the RoLA-vs-attn scaling sweep (which ∈ {train,prefill,decode}). Split per
    type so each is an independent fleet run_id (perf-scaling-<type>) — you can launch just one (e.g.
    scaling-decode) by listing only it in the spec's `benches`. Axes from SCALE_* env."""
    from rola_bench.perf import scaling as sc
    import rola_bench.perf._common as C
    C.REPS = REPS
    # Per-bench-type FRONTIER ceilings — it's a timing bench (not training), so L can go far past any
    # trainable length. decode reaches 1M (RoLA flat vs attn's linear-growing KV — the headline win
    # plot); prefill to 128k (attn O(L^2) blowup vs RoLA O(L)); training caps at 64k (backward +
    # nb_tile is memory-bound beyond that — graceful OOM sentinels handle overflow). Override with
    # SCALE_LENS_<TYPE> or a blanket SCALE_LENS.
    _DEFAULT_LENS = {"train": "1024,4096,16384,65536",
                     "prefill": "1024,4096,16384,65536,131072",
                     "decode": "1024,4096,16384,65536,262144,1048576"}
    lens_env = (os.environ.get(f"SCALE_LENS_{which.upper()}")
                or os.environ.get("SCALE_LENS") or _DEFAULT_LENS[which])
    lens = [int(v) for v in lens_env.split(",") if v.strip()]
    ncs = [int(v) for v in os.environ.get("SCALE_NCS", "16,64,256,1024").split(",") if v.strip()]
    dqks = [int(v) for v in os.environ.get("SCALE_DQKS", "16,64,128").split(",") if v.strip()]
    routings = [r for r in os.environ.get("SCALE_ROUTINGS", "flat,square,tree").split(",") if r.strip()]
    rows = []
    sc.sweep([which], lens, ncs, dqks, rows, routings=routings)
    if rows:
        C.write_csv(Path(out_dir) / f"scaling_{which}.csv", rows)
    return {"scaling": rows, "bench": which}


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "", "false", "False", "no", "No")


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v) not in ("0", "", "false", "False", "no", "No")


def _profile_from_env():
    return {
        "arm": os.environ.get("PROFILE_ARM", "rola-rla"),
        "L": int(os.environ.get("PROFILE_L", "65536")),
        "batch": int(os.environ.get("PROFILE_B", os.environ.get("PERF_B", "2"))),
        "d_qk": int(os.environ.get("PROFILE_DQK", "16")),
        "nc": int(os.environ.get("PROFILE_NC", "64")),
        "routing": os.environ.get("PROFILE_ROUTING", "square"),
        "warmup": int(os.environ.get("PROFILE_WARMUP", "2")),
        "reps": int(os.environ.get("PROFILE_REPS", "1")),
        "seed": int(os.environ.get("PROFILE_SEED", "0")),
        "torch_profiler": _env_bool("PROFILE_TORCH_PROFILER", False),
        "with_stack": _env_bool("PROFILE_WITH_STACK", False),
        "top": int(os.environ.get("PROFILE_TOP", "30")),
    }


def _profile_from_spec(run_id):
    prefix = "perf-profile-graph-"
    if not run_id.startswith(prefix):
        return _profile_from_env()
    config = os.environ.get("PERF_CONFIG", "profile_graph")
    name = run_id[len(prefix):]
    spec = load("perf", config)
    for profile in spec.get("profiles", []):
        if profile.get("name") == name:
            return profile
    raise KeyError(f"profile {name!r} not found in perf config {config!r}")


def run_profile_graph(out_dir, run_id="perf-profile-graph"):
    """One diagnostic graph profile on the box GPU.

    Axes normally come from the perf YAML's `profiles:` list. PROFILE_* env vars remain as an ad hoc
    fallback for local/manual runs.
    """
    from rola_bench.perf import profile_graph as pg
    profile = _profile_from_spec(run_id)
    args = SimpleNamespace(
        arm=profile.get("arm", "rola-rla"),
        L=int(profile.get("L", 65536)),
        batch=int(profile.get("batch", os.environ.get("PERF_B", "2"))),
        d_qk=int(profile.get("d_qk", profile.get("dqk", 16))),
        nc=int(profile.get("nc", 64)),
        routing=profile.get("routing", "square"),
        warmup=int(profile.get("warmup", 2)),
        reps=int(profile.get("reps", 1)),
        seed=int(profile.get("seed", 0)),
        torch_profiler=_as_bool(profile.get("torch_profiler", False)),
        with_stack=_as_bool(profile.get("with_stack", False)),
        top=int(profile.get("top", 30)),
        deep_profile=profile.get("deep_profile", os.environ.get("ROLA_DEEP_PROFILE", "none")),
        deep_kernel_name=profile.get("deep_kernel_name", os.environ.get("ROLA_DEEP_KERNEL", "_kappa_fwd_chunk")),
        deep_launch_skip=int(profile.get("deep_launch_skip", os.environ.get("ROLA_DEEP_LAUNCH_SKIP", 16))),
        deep_launch_count=int(profile.get("deep_launch_count", os.environ.get("ROLA_DEEP_LAUNCH_COUNT", 1))),
        deep_ncu_set=profile.get("deep_ncu_set", os.environ.get("ROLA_DEEP_NCU_SET", "full")),
        deep_warmup=int(profile.get("deep_warmup", os.environ.get("ROLA_DEEP_WARMUP", 1))),
        deep_timeout=int(profile.get("deep_timeout", os.environ.get("ROLA_DEEP_TIMEOUT", 900))),
        deep_child=False,
        out=Path(out_dir) / "profile_graph",
    )
    return {"bench": "profile-graph", "profile": profile, "profile_graph": pg.run(args)}


BENCHES = {
    "perf-scaling-train": lambda o: run_scaling(o, "train"),
    "perf-scaling-prefill": lambda o: run_scaling(o, "prefill"),
    "perf-scaling-decode": lambda o: run_scaling(o, "decode"),
}


def main():
    import torch
    import rola_bench.perf._common as C
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    # Capture CUDA status EXPLICITLY. Previously require_cuda() raised SystemExit on a failed init
    # (e.g. driver/runtime mismatch -> "named symbol not found"), which the per-bench except didn't
    # catch -> run_perf exited with EMPTY results and no recorded error (the A100 mystery). Now a
    # CUDA-init failure becomes a VISIBLE, synced failure row carrying the actual error string.
    cuda_ok = torch.cuda.is_available()
    cuda_err = ""
    if not cuda_ok:
        try:
            torch.cuda.init()
        except Exception as e:  # noqa: BLE001
            cuda_err = f"{type(e).__name__}: {e}"
    gpu = torch.cuda.get_device_name() if cuda_ok else "NO_CUDA"
    out_dir = str(C.RESULTS_DIR)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        if not cuda_ok:
            _append({"run_id": rid, "ok": False, "gpu": gpu,
                     "error": f"cuda unavailable: {cuda_err or 'is_available()=False'}"})
            print(f"[perf] {rid} FAILED: cuda unavailable ({cuda_err})", flush=True)
            continue
        fn = BENCHES.get(rid)
        if fn is None and rid.startswith("perf-profile-graph"):
            fn = lambda o, _rid=rid: run_profile_graph(o, _rid)
        if fn is None:
            _append({"run_id": rid, "ok": False, "gpu": gpu, "error": "unknown perf bench id"})
            print(f"[perf] unknown bench {rid!r}", flush=True)
            continue
        try:
            print(f"[perf] running {rid} on {gpu} (reps={REPS})", flush=True)
            data = fn(out_dir)
            _append({"run_id": rid, "ok": True, "gpu": gpu, "reps": REPS, **data})
            print(f"[perf] {rid} OK", flush=True)
        except Exception as e:  # noqa: BLE001 -- one bench's failure must not drop the rest
            _append({"run_id": rid, "ok": False, "gpu": gpu,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-12000:]})
            print(f"[perf] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
