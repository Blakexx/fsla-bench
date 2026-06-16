"""The kernel-efficiency perf benchmark as a `fleet` Job.

Perf differs from the MQAR sweep in granularity. The chunked bench's headline numbers are
*ratios* between impls, made drift-free by ROUND-ROBIN interleaving all impls inside ONE process
(see perf/_common.round_robin). Splitting it per-arm across boxes would destroy that interleaving,
so the fleet "cell" here is a WHOLE bench run on one isolated GPU: one run_id = one bench module.
That still parallelises chunked vs recurrent across two boxes and — the real point — produces
reproducible numbers on a clean, declared cloud GPU rather than a shared workstation.

No checkpoints are involved (perf builds fresh models), so this job ships no payloads — it uses the
plain run-ids contract, exactly like mqar. The box image supplies the GPU + fla/fla-rola kernels;
FLEET_RUN_CMD (set via env below) points the box at run_perf.py.
"""
import glob, json, os
from fleet import Job

RESULTS_DIR = os.environ.get("ROLA_BENCH_PERF_RESULTS", "/home/blake/rola-bench/results/perf")  # scientific output
RUNTIME_DIR = os.environ.get("ROLA_BENCH_PERF_RUNTIME", "/home/blake/rola-bench/runtime/perf")  # fleet ops state — gitignored
IMAGE = os.environ.get("ROLA_BENCH_IMAGE", "blakeresearch/rola-bench:v5")  # v5 = perf shape×full-nc sweep (wide/square/heads, nc 2..256)

# One run_id per bench module. PERF_BENCHES restricts the set (e.g. PERF_BENCHES=chunked).
ALL_BENCHES = ["perf-chunked", "perf-recurrent"]


def work_list():
    sel = os.environ.get("PERF_BENCHES")
    if sel:
        keep = {f"perf-{s.strip()}" for s in sel.split(",")}
        return [b for b in ALL_BENCHES if b in keep]
    return list(ALL_BENCHES)


def done_ids():
    """run_ids with an ok summary row in any perf results ndjson."""
    have = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")):
        try:
            for line in open(f):
                j = json.loads(line)
                if j.get("ok") and j.get("run_id"):
                    have.add(j["run_id"])
        except Exception:
            pass
    return have


perf_job = Job(
    name="rola-perf",
    image=IMAGE,
    # Perf wants a clean, declared GPU. Default to the LM-training GPU class (A100) for the
    # headline table; override OFFER_QUERY to bench a different card (e.g. RTX_4090).
    offer_query=os.environ.get(
        "OFFER_QUERY", "gpu_name=A100_PCIE num_gpus=1 rentable=true reliability>0.97 inet_down>500"),
    results_path=os.path.join(RESULTS_DIR, "perf_dispatch.jsonl"),
    ckpt_dir=os.path.join(RUNTIME_DIR, "_unused_ckpts"),   # perf saves no checkpoints
    work_list=work_list,
    done_ids=done_ids,
    env={"FLEET_RUN_CMD": "python -u -m rola_bench.perf.run_perf",
         "PERF_REPS": os.environ.get("PERF_REPS", "15")},
    bad_hosts=os.path.join(RUNTIME_DIR, "bad_hosts.txt"),
    idle_timeout=900,
    batch=1,                  # one whole bench per box (round-robin integrity lives inside the run)
    per_batch_timeout=3600,   # a full chunked/recurrent sweep is minutes; 1h is generous headroom
)
