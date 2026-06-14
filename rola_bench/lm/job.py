"""The LM benchmark as a `fleet` Job — SCAFFOLD.

Structure parity with rola_bench.mqar.job so the suite orchestrates both benchmarks identically
(`python launch.py lm`). The shape is real; the box-side contract is not yet complete:

  TODO before LM can actually dispatch on the fleet — train_lm.py must, per arm, append an ndjson
  row {"run_id": <arm>, "ok": bool, ...} to $FLEET_RESULTS and write its checkpoint to
  $FLEET_CKPT_DIR/<arm>.pt (the same contract run_rla_sweep.py already satisfies for MQAR). The LM
  image's FLEET_RUN_CMD (docker/lm.Dockerfile) then runs train_lm for the RUN_IDS arm(s).

Until that lands, this Job describes the intended work-list but `launch.py lm` should be treated as
not-yet-live. The MQAR benchmark is the live fleet path.
"""
import glob, json, os
from fleet import Job

# One fleet run_id per LM arm; comma-separated override via LM_ARMS.
ARMS = [a for a in os.environ.get("LM_ARMS", "rola-rla-kappa-asym-125m").split(",") if a]
RESULTS_DIR = os.environ.get("ROLA_LM_RESULTS", "/home/blake/rola-bench/lm_results")
CKPT_DIR = os.environ.get("ROLA_LM_CKPTS", "/mnt/d/rola_data/lm_ckpts")
IMAGE = os.environ.get("ROLA_LM_IMAGE", "blakeresearch/rola-lm:v1")


def work_list():
    return list(ARMS)


def done_ids():
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


lm_job = Job(
    name="rola-lm",
    image=IMAGE,
    offer_query="gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500",
    results_path=os.path.join(RESULTS_DIR, "results_dispatch.jsonl"),
    ckpt_dir=CKPT_DIR,
    work_list=work_list,
    done_ids=done_ids,
    idle_timeout=1800,
    batch=1,
)
