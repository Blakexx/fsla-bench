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

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
REPS = int(os.environ.get("PERF_REPS", "15"))


def _append(row):
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run_chunked(out_dir):
    """Full chunked fwd+bwd sweep (kernel efficiency + projection cost), round-robin in-process."""
    from rola_bench.perf import chunked as ch
    import rola_bench.perf._common as C
    rows, proj, e2e = [], [], []
    ch.bench_kernel(REPS, rows)            # stage-2 kernel (isolated)
    ch.bench_projections(REPS, proj)       # stage-1 projections (isolated)
    ch.bench_e2e(REPS, e2e)                # FUSED end-to-end layer (fwd + fwd+bwd)
    # also write the canonical CSV/TeX (paper artifacts) next to the dispatch jsonl
    if rows:
        C.write_csv(Path(out_dir) / "kernel_efficiency.csv", rows)
    if proj:
        C.write_csv(Path(out_dir) / "projection_cost.csv", proj)
    if e2e:
        C.write_csv(Path(out_dir) / "kernel_e2e.csv", e2e)
    if rows and proj:
        ch.write_tex(Path(out_dir), rows, proj, REPS)
    return {"kernel": rows, "projection": proj, "e2e": e2e}


def run_recurrent(out_dir):
    """Full recurrent (decode, forward-only) sweep."""
    from rola_bench.perf import recurrent as rc
    steps = int(os.environ.get("PERF_DECODE_STEPS", "200"))
    warm = int(os.environ.get("PERF_DECODE_WARMUP", "50"))
    rc.bench(steps=steps, warmup=warm, out=out_dir)        # decode decomposition (kernel only)
    rc.bench_e2e(steps=steps, warmup=warm, out=out_dir)    # FUSED end-to-end decode (fwd only)
    return {"decode": _read_csv(Path(out_dir) / "recurrent_decode.csv"),
            "e2e": _read_csv(Path(out_dir) / "recurrent_e2e.csv")}


BENCHES = {"perf-chunked": run_chunked, "perf-recurrent": run_recurrent}


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
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
            print(f"[perf] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
