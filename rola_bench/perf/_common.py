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
import statistics
import time
from pathlib import Path

import torch

# Canonical benchmark shapes (match the LM training config).
B, H, L, DQK, DV = 8, 8, 1024, 16, 16
NCS = (16, 64, 256)
DEV, DT = "cuda", torch.bfloat16

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
