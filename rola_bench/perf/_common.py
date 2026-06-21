"""Shared protocol for the RoLA kernel-efficiency benchmarks (chunked + recurrent).

Design goals: high-quality, reproducible, canonical numbers that anyone can regenerate.

  * Fixed shapes mirroring LM training (B=8, H=8, L=1024, d_qk=d_v=16, bf16) and a fixed
    seed per cell, so a re-run reproduces the table.
  * ROUND-ROBIN interleaving: every timing round cycles through ALL impls before the next
    round, so clock/thermal drift biases each impl equally and between-impl ratios are
    drift-free (the headline numbers are ratios).
  * CUDA-event timing, warmup excluded, median + IQR over the kept rounds.
  * Honest activation accounting: bytes RETAINED for backward (saved tensors, deduped by
    storage data_ptr) via `saved_tensors_hooks` — catches saved input leaves a naive
    "post-forward allocated delta" misses. Peak via `reset_peak_memory_stats` per round.

Imports are clean: `fla` (canonical) and `fla_rola` (the fork) are pip-installed; no
sys.path hacks. Run on an IDLE GPU for the memory/timing numbers to be meaningful.
"""
import csv
import os
import statistics
import time
from pathlib import Path

import torch

# Benchmark shapes. DEFAULT = the canonical LM-training shape (the headline efficiency table).
# These are module globals so the arms (chunked.mk / mk_gdn / ...) read them at call time; the
# per-cell loop overrides them via set_shape() to time at each MQAR cell's exact dims (task #98 —
# cost-vs-accuracy), with no arm-signature changes. nc is passed explicitly (not a global).
B = int(os.environ.get("PERF_B", "8"))       # batch; drop (e.g. 2) for the long-L scaling sweep so big L fits
H, L, DQK, DV = 8, 1024, 16, 16
# nc rungs to bench. DEFAULT = {16,64,256} (the similarity-rung sample) — each (nc,shape) is its own
# Triton autotune, so 3 rungs instead of the full 8 cuts ~2.6x of the autotune cost. Override with
# PERF_NCS (e.g. "2,4,8,16,32,64,128,256") for the full state scan. NOT a silent cap — explicit env.
NCS = tuple(int(x) for x in os.environ.get("PERF_NCS", "16,64,256").split(",") if x.strip())
# Sequence-length scan for the SCALING sweep (RoLA O(L) vs attention O(L^2) + nc sublinearity).
# DEFAULT = [L] (single length -> the matched-state table is unchanged). Override with PERF_LENS.
LENS = tuple(int(x) for x in os.environ.get("PERF_LENS", str(L)).split(",") if x.strip())
DEV, DT = "cuda", torch.bfloat16


def set_shape(B=None, H=None, L=None, DQK=None, DV=None):
    """Override the active benchmark shape (per-cell). Only the named dims change; others keep
    their current value. Returns the prior shape so callers can restore it."""
    import rola_bench.perf._common as _c
    prior = (_c.B, _c.H, _c.L, _c.DQK, _c.DV)
    if B is not None: _c.B = B
    if H is not None: _c.H = H
    if L is not None: _c.L = L
    if DQK is not None: _c.DQK = DQK
    if DV is not None: _c.DV = DV
    return prior

# --------------------------------------------------------------------------------------------
# Matched-state SHAPED dims (wide / square / heads), shared by the chunked + recurrent benches.
# Mirrors rola_bench/mqar/experiments/canonical_baselines.baseline_cell in the perf base shape:
# total recurrent floats S = H*nc*d_qk*d_v are held constant across shapes; returns (H, d_k, d_v)
# or None when the accuracy grid omits the cell (gdn head_dim>256; head-scaling at nc>32).
#   kind: 'lin1' = dimension-preserving feature map (monolith RLA/GLA),  fm=1
#         'lin2' = feature-doubling map (Hedgehog softmax([x,-x])),       fm=2
#         'gdn'  = delta-rule; head_dim (d_k) kernel-capped at 256
# Reads the live module globals (H/DQK/DV) so set_shape() composes; the head-count base is H.
GDN_HEAD_CAP = 256       # chunk/recurrent gated_delta_rule head_dim ceiling
HEADS_NC_CAP = 32        # head-scaling OOMs beyond ~H140 at batch 128 (accuracy grid cap)


def shaped_dims(kind, shape, nc):
    S = H * nc * DQK * DV
    fm = 2 if kind == "lin2" else 1
    cap = GDN_HEAD_CAP if kind == "gdn" else None
    if shape == "wide":                                  # H * fm * d_k * DV = S (widen keys)
        dk = max(1, round(S / (H * DV * fm)))
        return None if (cap and dk > cap) else dict(H=H, d_k=dk, d_v=DV)
    if shape == "square":                                # H * fm * d * d = S
        d = max(1, round((S / (H * fm)) ** 0.5))
        return None if (cap and d > cap) else dict(H=H, d_k=d, d_v=d)
    if shape == "heads":                                 # h * fm * DV * DV = S (scale head count)
        if nc > HEADS_NC_CAP:
            return None
        h = max(1, round(S / (DV * DV * fm)))
        return dict(H=h, d_k=DV, d_v=DV)
    return None


# Default reps; override on the CLI. WARM rounds are discarded before timing.
REPS, WARMUP = 15, 4

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "perf"


def median(xs):
    return statistics.median(xs)


def iqr(xs):
    """Inter-quartile range; 0 for <2 samples (statistics.quantiles needs >=2)."""
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("perf benchmarks require a CUDA GPU (torch.cuda.is_available() is False)")


class ActivationProbe:
    """Context manager recording bytes retained for backward (deduped by storage)."""

    def __init__(self):
        self.saved = {}

    def __enter__(self):
        def pack(x):
            if x.is_cuda:
                self.saved[x.untyped_storage().data_ptr()] = x.untyped_storage().nbytes()
            return x
        self._hook = torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x)
        self._hook.__enter__()
        return self

    def __exit__(self, *a):
        return self._hook.__exit__(*a)

    @property
    def mib(self):
        return sum(self.saved.values()) / 2 ** 20


def fwd_bwd_rep(fn, leaves):
    """One forward+backward timing rep. Returns (fwd_ms, fwd_bwd_ms, peak_mib, act_mib).

    `fn(leaves) -> output`; `leaves` is the dict of (possibly grad-requiring) input tensors.
    Grads are cleared after each rep so memory accounting is per-rep.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    probe = ActivationProbe()
    with probe:
        e0.record()
        o = fn(leaves)
        e1.record()
    torch.cuda.synchronize()
    act = probe.mib
    (o.float().sum()).backward()
    e2.record()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    for v in leaves.values():
        if isinstance(v, torch.Tensor) and v.grad is not None:
            v.grad = None
        elif isinstance(v, torch.nn.Module):
            # Non-tensor leaves can be learnable sub-modules (e.g. the Hedgehog feature map's
            # Linear). Their param grads ACCUMULATE across reps if not cleared, inflating the
            # baseline allocation and skewing the per-rep peak delta. Zero them too.
            v.zero_grad(set_to_none=True)
    return e0.elapsed_time(e1), e0.elapsed_time(e2), peak, act


def round_robin(impls, run_one, reps=REPS, warmup=WARMUP):
    """Interleave `impls` across rounds so drift hits every impl equally.

    impls   : list of (name, payload). `run_one(name, payload)` does ONE rep and returns a tuple.
    returns : {name: [rep_tuple, ...]} with the first `warmup` rounds dropped.
    """
    data = {name: [] for name, _ in impls}
    for r in range(warmup + reps):
        for name, payload in impls:
            rec = run_one(name, payload)
            if r >= warmup:
                data[name].append(rec)
    return data


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def time_forward(fn, steps, warmup):
    """Median wall-clock ms over `steps` forward-only calls (decode/recurrent path)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3
