"""Diagnostic computation-graph profiler for one RoLA scaling arm.

This is not a reportable benchmark table. It is a reproducible investigation tool for
answering "where did this cell spend time and memory?" using the same layer builders
and arm names as the scaling benchmark.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from rola_bench.perf import _common as C


os.environ.setdefault("HF_HOME", "/tmp/rola-bench-hf")

_SCALING = None
_CHUNK_ROLA_ROUTED = None


def _scaling():
    global _SCALING
    if _SCALING is None:
        from rola_bench.perf import scaling as scaling_mod
        _SCALING = scaling_mod
    return _SCALING


def _chunk_rola_routed():
    global _CHUNK_ROLA_ROUTED
    if _CHUNK_ROLA_ROUTED is None:
        from fla_rola.ops.rola import chunk_rola_routed
        _CHUNK_ROLA_ROUTED = chunk_rola_routed
    return _CHUNK_ROLA_ROUTED


@dataclass
class Phase:
    name: str
    ms: float
    iqr_ms: float
    samples_ms: list[float]
    stats_ms: dict[str, Any]
    alloc_delta_mib: float
    peak_delta_mib: float
    peak_samples_mib: list[float]
    peak_stats_mib: dict[str, Any]
    end_alloc_mib: float
    outputs: list[dict[str, Any]]


def _sync():
    torch.cuda.synchronize()


def _round(x: float, ndigits: int = 3) -> float:
    return round(float(x), ndigits)


def _mib(nbytes: int | float) -> float:
    return float(nbytes) / 2 ** 20


def _tensor_info(x: Any) -> list[dict[str, Any]]:
    out = []
    items = x if isinstance(x, (tuple, list)) else (x,)
    for i, t in enumerate(items):
        if not isinstance(t, torch.Tensor):
            continue
        out.append({
            "index": i,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "mib": _round(_mib(t.numel() * t.element_size())),
        })
    return out


def _event_ms(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    _sync()
    return start.elapsed_time(end), value


def _timing_stats(samples: list[float]) -> tuple[float, float]:
    return C.median(samples), C.iqr(samples)


def _measure_phase(name: str, fn, reps: int = 1) -> tuple[Phase, Any]:
    samples: list[float] = []
    peaks: list[float] = []
    alloc_delta = end_alloc = base = 0
    value = None

    def wrapped():
        with record_function(f"profile_graph::{name}"):
            return fn()

    value = wrapped()
    _sync()
    for _ in range(max(1, reps)):
        _sync()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ms, value = _event_ms(wrapped)
        end_alloc = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        samples.append(float(ms))
        peaks.append(_mib(peak - base))
    med_ms, iqr_ms = _timing_stats(samples)
    peak_stats = C.sample_stats(peaks)
    phase = Phase(
        name=name,
        ms=_round(med_ms),
        iqr_ms=_round(iqr_ms),
        samples_ms=[_round(x) for x in samples],
        stats_ms=C.sample_stats(samples),
        alloc_delta_mib=_round(_mib(end_alloc - base)),
        peak_delta_mib=_round(max(peaks) if peaks else 0.0),
        peak_samples_mib=[_round(x) for x in peaks],
        peak_stats_mib=peak_stats,
        end_alloc_mib=_round(_mib(end_alloc)),
        outputs=_tensor_info(value),
    )
    return phase, value


def _git_rev(path: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _module_git_rev(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        path = Path(mod.__file__).resolve()
    except Exception:
        return None
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return _git_rev(str(parent))
    return None


def _module_forward(module, x):
    o = module(x)
    return o[0] if isinstance(o, tuple) else o


def _whole_forward_profile(module, x, reps: int = 1) -> dict[str, Any]:
    samples: list[float] = []
    peaks: list[float] = []
    y = None
    with torch.inference_mode():
        for _ in range(max(1, reps)):
            _sync()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ms, y = _event_ms(lambda: _module_forward(module, x))
            peak = torch.cuda.max_memory_allocated()
            samples.append(float(ms))
            peaks.append(_mib(peak - base))
    med_ms, iqr_ms = _timing_stats(samples)
    return {
        "ms": _round(med_ms),
        "iqr_ms": _round(iqr_ms),
        "samples_ms": [_round(x) for x in samples],
        "stats_ms": C.sample_stats(samples),
        "peak_delta_mib": _round(max(peaks) if peaks else 0.0),
        "peak_samples_mib": [_round(x) for x in peaks],
        "peak_stats_mib": C.sample_stats(peaks),
        "output": _tensor_info(y),
    }


def _runtime_metadata() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "device_index": torch.cuda.current_device(),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory_mib": _round(_mib(props.total_memory)),
        "multiprocessor_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dtype": str(C.DT).replace("torch.", ""),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "rola_bench_rev": _git_rev("/home/blake/rola-bench"),
        "fla_rola_rev": _module_git_rev("fla_rola"),
    }


def _torch_profile(module, x, out_dir: Path, top: int, with_stack: bool) -> dict[str, Any]:
    trace_path = out_dir / "trace.json.gz"
    with torch.inference_mode():
        _sync()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
        ) as prof:
            with record_function("profile_graph::whole_forward"):
                _module_forward(module, x)
        _sync()
        peak = torch.cuda.max_memory_allocated()
    raw_trace = trace_path.with_suffix(".json")
    prof.export_chrome_trace(str(raw_trace))
    with raw_trace.open("rb") as src, gzip.open(trace_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    raw_trace.unlink()
    table_cuda = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=top)
    table_mem = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=top)
    (out_dir / "profiler_cuda.txt").write_text(table_cuda)
    (out_dir / "profiler_memory.txt").write_text(table_mem)
    trace_bytes = trace_path.read_bytes()
    return {
        "trace_encoding": "chrome_trace_json_gzip_base64",
        "trace_json_gz_base64": base64.b64encode(trace_bytes).decode("ascii"),
        "trace_json_gz_bytes": len(trace_bytes),
        "cuda_table": table_cuda,
        "memory_table": table_mem,
        "peak_delta_mib": _round(_mib(peak - base)),
        "top_cuda": _top_profiler_rows(prof, "self_cuda_time_total", top),
        "top_memory": _top_profiler_rows(prof, "self_cuda_memory_usage", top),
    }


def _evt_us(evt, name: str) -> float:
    aliases = {
        "self_cuda_time_total": ("self_cuda_time_total", "self_device_time_total"),
        "cuda_time_total": ("cuda_time_total", "device_time_total"),
    }
    for attr in aliases.get(name, (name,)):
        if hasattr(evt, attr):
            return float(getattr(evt, attr))
    return 0.0


def _evt_mem(evt, name: str) -> float:
    aliases = {
        "self_cuda_memory_usage": ("self_cuda_memory_usage", "self_device_memory_usage"),
        "cuda_memory_usage": ("cuda_memory_usage", "device_memory_usage"),
    }
    for attr in aliases.get(name, (name,)):
        if hasattr(evt, attr):
            return float(getattr(evt, attr))
    return 0.0


def _top_profiler_rows(prof, sort_by: str, limit: int) -> list[dict[str, Any]]:
    def key(evt):
        if "memory" in sort_by:
            return _evt_mem(evt, sort_by)
        return _evt_us(evt, sort_by)

    rows = []
    for evt in sorted(prof.key_averages(), key=key, reverse=True)[:limit]:
        rows.append({
            "key": evt.key,
            "count": evt.count,
            "self_cpu_ms": _round(evt.self_cpu_time_total / 1000.0),
            "cpu_total_ms": _round(evt.cpu_time_total / 1000.0),
            "self_cuda_ms": _round(_evt_us(evt, "self_cuda_time_total") / 1000.0),
            "cuda_total_ms": _round(_evt_us(evt, "cuda_time_total") / 1000.0),
            "self_cuda_mem_mib": _round(_mib(_evt_mem(evt, "self_cuda_memory_usage"))),
            "cuda_mem_mib": _round(_mib(_evt_mem(evt, "cuda_memory_usage"))),
        })
    return rows


def _full_text(s: str) -> str:
    return s


def _find_tool(names: tuple[str, ...], extra_globs: tuple[str, ...] = ()) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path

    for pattern in extra_globs:
        candidates = Path("/").glob(pattern.lstrip("/"))
        for path in sorted(candidates, key=lambda p: str(p), reverse=True):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    return None


def _child_profile_cmd(args, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "rola_bench.perf.profile_graph",
        "--arm", args.arm,
        "--L", str(args.L),
        "--batch", str(args.batch),
        "--d-qk", str(args.d_qk),
        "--nc", str(args.nc),
        "--routing", args.routing,
        "--warmup", str(args.deep_warmup),
        "--reps", "1",
        "--seed", str(args.seed),
        "--top", str(args.top),
        "--out", str(out_dir / "deep_child"),
        "--deep-child",
    ]
    return cmd


def _run_deep_command(name: str, cmd: list[str], out_dir: Path, timeout: int) -> dict[str, Any]:
    log = out_dir / f"{name}.log"
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ, "ROLA_PROFILE_DEEP_CHILD": "1"},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr = proc.stdout, proc.stderr
        log.write_text(stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""))
        return {
            "tool": name,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": _round(time.time() - started),
            "command": cmd,
            "stdout": _full_text(stdout),
            "stderr": _full_text(stderr),
            "log_text": _full_text(log.read_text()),
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        log.write_text(stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""))
        return {
            "tool": name,
            "ok": False,
            "returncode": None,
            "seconds": _round(time.time() - started),
            "command": cmd,
            "error": f"timeout after {timeout}s",
            "stdout": _full_text(stdout),
            "stderr": _full_text(stderr),
            "log_text": _full_text(log.read_text()),
        }


_NCU_METRIC_LABELS = {
    "Duration": "duration",
    "Memory Throughput": "memory_throughput",
    "DRAM Throughput": "dram_throughput",
    "L1/TEX Cache Throughput": "l1_tex_throughput",
    "L2 Cache Throughput": "l2_throughput",
    "LSU": "lsu",
    "Grid Size": "grid_size",
    "Block Size": "block_size",
    "Registers Per Thread": "registers_per_thread",
    "Dynamic Shared Memory Per Block": "dynamic_smem_per_block",
    "Static Shared Memory Per Block": "static_smem_per_block",
    "Waves Per SM": "waves_per_sm",
    "Theoretical Occupancy": "theoretical_occupancy",
    "Achieved Occupancy": "achieved_occupancy",
    "Block Limit Registers": "block_limit_registers",
    "Block Limit Shared Mem": "block_limit_smem",
    "Block Limit Warps": "block_limit_warps",
    "Block Limit Blocks": "block_limit_blocks",
    "Stall Wait": "stall_wait",
    "Stall Long Scoreboard": "stall_long_scoreboard",
    "Stall Math Pipe Throttle": "stall_math_pipe_throttle",
    "Stall Barrier": "stall_barrier",
    "Stall MIO Throttle": "stall_mio_throttle",
    "Stall Short Scoreboard": "stall_short_scoreboard",
}


def _ncu_metric_suffix(line: str) -> str:
    if "%" in line:
        return "pct"
    if "Gbyte/s" in line:
        return "gb_s"
    if "Kbyte/block" in line:
        return "kib_per_block"
    if "byte/block" in line:
        return "bytes_per_block"
    if "register/thread" in line:
        return "regs_per_thread"
    if "thread" in line:
        return "threads"
    if "cycle" in line:
        return "cycles"
    if "usecond" in line or " us " in f" {line} ":
        return "us"
    if " ms " in f" {line} ":
        return "ms"
    if "block" in line:
        return "blocks"
    if "inst" in line:
        return "inst"
    return "value"


def _parse_ncu_details(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    number = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw in text.splitlines():
        line = raw.strip()
        for label, base_key in _NCU_METRIC_LABELS.items():
            if not line.startswith(label):
                continue
            vals = number.findall(line)
            if not vals:
                continue
            key = f"{base_key}_{_ncu_metric_suffix(line)}"
            metrics.setdefault(key, _round(float(vals[-1])))
    return metrics


def _ncu_report_prefix(out_dir: Path, kernel: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", kernel.strip("_")).strip("_")
    return out_dir / f"ncu_{stem or 'kernel'}"


def _attach_ncu_details(info: dict[str, Any], ncu: str, report_prefix: Path, out_dir: Path, timeout: int) -> None:
    report_file = Path(f"{report_prefix}.ncu-rep")
    info["report_file"] = str(report_file)
    if not info.get("ok") or not report_file.exists():
        return

    cmd = [
        ncu,
        "--import", str(report_file),
        "--page", "details",
        "--print-details", "all",
    ]
    log = out_dir / "ncu_details.log"
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log.write_text(proc.stdout + ("\n\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""))
        info.update({
            "details_ok": proc.returncode == 0,
            "details_returncode": proc.returncode,
            "details_seconds": _round(time.time() - started),
            "details_command": cmd,
            "details_text": _full_text(proc.stdout),
            "details_stderr": _full_text(proc.stderr),
        })
        if proc.returncode == 0:
            info["parsed_metrics"] = _parse_ncu_details(proc.stdout)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        log.write_text(stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""))
        info.update({
            "details_ok": False,
            "details_returncode": None,
            "details_seconds": _round(time.time() - started),
            "details_command": cmd,
            "details_error": f"timeout after {timeout}s",
            "details_text": _full_text(stdout),
            "details_stderr": _full_text(stderr),
        })


def _deep_profile(args, out_dir: Path) -> dict[str, Any]:
    """Optional NVIDIA-tool profile.

    PyTorch profiler stops at the Triton launch boundary. Nsight Systems explains launch ordering,
    gaps, and memory timeline; Nsight Compute samples one/few matching Triton kernel launches and reports
    occupancy, memory, instruction, and stall metrics. Both are opt-in because they can be slow and may
    require host profiler permissions.
    """
    mode = getattr(args, "deep_profile", "none")
    if mode in (None, "", "none") or getattr(args, "deep_child", False):
        return {"enabled": False}
    out = {"enabled": True, "mode": mode, "tools": {}}
    child = _child_profile_cmd(args, out_dir)
    timeout = int(getattr(args, "deep_timeout", 900))

    if mode in ("nsys", "both"):
        nsys = _find_tool(
            ("nsys",),
            (
                "/opt/nvidia/nsight-systems/*/bin/nsys",
                "/opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys",
                "/usr/local/cuda-*/bin/nsys",
            ),
        )
        if nsys is None:
            out["tools"]["nsys"] = {"ok": False, "error": "nsys not found on PATH or standard install paths"}
        else:
            report = out_dir / "nsys_profile"
            cmd = [
                nsys, "profile",
                "--force-overwrite=true",
                "--stats=true",
                "--trace=cuda,nvtx,osrt",
                "--cuda-memory-usage=true",
                "--sample=none",
                "-o", str(report),
                *child,
            ]
            out["tools"]["nsys"] = _run_deep_command("nsys", cmd, out_dir, timeout)

    if mode in ("ncu", "both"):
        ncu = _find_tool(
            ("ncu", "nv-nsight-cu-cli"),
            (
                "/opt/nvidia/nsight-compute/*/ncu",
                "/usr/local/cuda-*/bin/ncu",
            ),
        )
        if ncu is None:
            out["tools"]["ncu"] = {
                "ok": False,
                "error": "ncu/nv-nsight-cu-cli not found on PATH or standard install paths",
            }
        else:
            kernel = getattr(args, "deep_kernel_name", "_kappa_fwd_scan_cb_atomic")
            report = _ncu_report_prefix(out_dir, kernel)
            cmd = [
                ncu,
                "--force-overwrite",
                "--target-processes", "all",
                "--kernel-name-base", "function",
                "--kernel-name", f"regex:{kernel}",
                "--launch-skip", str(getattr(args, "deep_launch_skip", 16)),
                "--launch-count", str(getattr(args, "deep_launch_count", 1)),
                "--set", getattr(args, "deep_ncu_set", "full"),
                "--export", str(report),
                *child,
            ]
            info = _run_deep_command("ncu", cmd, out_dir, timeout)
            if "No kernels were profiled" in info.get("log_text", ""):
                info["ok"] = False
                info["error"] = (
                    f"ncu matched no kernels for regex:{kernel}; see available kernels in log_text"
                )
            _attach_ncu_details(info, ncu, report, out_dir, timeout)
            if info.get("ok") and not info.get("details_ok"):
                info["ok"] = False
                info["error"] = "ncu report was not imported successfully; no details were parsed"
            if info.get("ok") and not info.get("parsed_metrics"):
                info["ok"] = False
                info["error"] = "ncu produced no parsed metrics; kernel match/details are not actionable"
            out["tools"]["ncu"] = info
    return out


def _build_arm(arm: str, L: int, d: int, nc: int, routing: str):
    sc = _scaling()
    C.set_shape(L=L, DQK=d, DV=d)
    if arm == "attn":
        return sc._CausalMHA(sc.DM, C.H).to(C.DEV, C.DT)
    if arm == "rola-rla":
        return sc.build_rola(False, nc, d, routing=routing)
    if arm == "rola-gla":
        return sc.build_rola(True, nc, d, routing=routing)
    if arm == "monolith-rla":
        return sc.build_monolith(False, nc, d)
    if arm == "monolith-gla":
        return sc.build_monolith(True, nc, d)
    raise ValueError(f"unknown arm {arm!r}")


def _analytic_sizes(module, arm: str, B: int, L: int, d: int, nc: int) -> dict[str, Any]:
    sc = _scaling()
    dtype_bytes = torch.tensor([], dtype=C.DT).element_size()
    H = C.H
    dm = sc.DM
    sizes = {
        "input_x_mib": _round(_mib(B * L * dm * dtype_bytes)),
        "q_mib": _round(_mib(B * L * H * d * dtype_bytes)),
        "k_mib": _round(_mib(B * L * H * d * dtype_bytes)),
        "v_mib": _round(_mib(B * L * H * d * dtype_bytes)),
        "attention_kv_cache_mib_bf16": _round(_mib(B * L * 2 * dm * dtype_bytes)),
    }
    if arm.startswith("rola-"):
        D = module.route_D
        b = module.route_b
        one_logits = B * L * H * D * b * dtype_bytes
        router_count = 1 if module.read_W is None else 2
        working_state = B * H * nc * d * (d + 1) * 4
        persistent_layer_state = H * nc * d * (d + 1) * 4
        sizes.update({
            "route_D": D,
            "route_b": b,
            "router_logits_materialized": False,
            "one_router_logits_equivalent_mib": _round(_mib(one_logits)),
            "all_router_logits_equivalent_mib": _round(_mib(one_logits * router_count)),
            "router_logits_count": router_count,
            "kernel_working_state_mib_fp32": _round(_mib(working_state)),
            "persistent_layer_state_mib_fp32": _round(_mib(persistent_layer_state)),
            "state_x_pct_physical": sc._state_x_pct(nc, d, L),
        })
    return sizes


def _ledger_item(name: str, shape: tuple[int, ...], dtype: str, bytes_per_el: int, note: str = ""):
    numel = 1
    for s in shape:
        numel *= int(s)
    return {
        "name": name,
        "shape": list(shape),
        "dtype": dtype,
        "mib": _round(_mib(numel * bytes_per_el)),
        "note": note,
    }


def _buffer_ledger(module, arm: str, B: int, L: int, d: int, nc: int) -> dict[str, Any]:
    """Expected major HBM buffers for the layer and routed forward.

    This is an analytic ledger, not a CUDA allocator trace. Its job is to show whether measured peak is
    explained by known model-sized tensors or by hidden kernel/profiler/transient workspace.
    """
    sc = _scaling()
    H = C.H
    dm = sc.DM
    bf16 = torch.tensor([], dtype=C.DT).element_size()
    fp32 = 4
    items = [
        _ledger_item("input_x", (B, L, dm), "bf16", bf16, "Layer input held live by caller."),
        _ledger_item("q", (B, L, H, d), "bf16", bf16, "Projected query."),
        _ledger_item("k", (B, L, H, d), "bf16", bf16, "Projected key."),
        _ledger_item("v", (B, L, H, d), "bf16", bf16, "Projected value."),
        _ledger_item("q_feature", (B, L, H, d), "bf16", bf16, "Feature-mapped query."),
        _ledger_item("k_feature", (B, L, H, d), "bf16", bf16, "Feature-mapped key."),
        _ledger_item("layer_output", (B, L, dm), "bf16", bf16, "Output after o_proj."),
    ]
    routed = []
    if arm.startswith("rola-"):
        D = module.route_D
        b = module.route_b
        BH = B * H
        route_count = 1 if module.read_W is None else 2
        if getattr(module, "use_short_conv", False):
            items.append(_ledger_item("write_route_stream", (B, L, dm), "bf16", bf16,
                                      "Short-conv write-router input stream."))
            if module.read_W is not None:
                items.append(_ledger_item("read_route_stream", (B, L, dm), "bf16", bf16,
                                          "Short-conv read-router input stream."))
            if module.kernel == "gla_scalar":
                items.append(_ledger_item("decay_alpha_stream", (B, L, H), "bf16", bf16,
                                          "Short-conv scalar decay stream used as h_g."))
        items.append(_ledger_item("router_logits_equivalent_avoided",
                                  (route_count, B, L, H, D, b), "bf16", bf16,
                                  "Equivalent full logits if materialized; production streams tiles inside the scan."))
        if module.state_norm == "kappa":
            items.append(_ledger_item("kappa", (B, L, H, 1), "bf16", bf16, "Per-token kappa exponent."))
        if module.kernel == "gla_scalar":
            items.append(_ledger_item("gla_alpha_logits", (B, L, H), "bf16", bf16, "Per-head scalar decay logits."))
        routed = [
            _ledger_item("kappa_num", (BH, L, d), "fp32", fp32,
                         "Full-length numerator buffer, normalized in-place after _kappa_routed_fwd."),
            _ledger_item("kappa_den", (BH, L), "fp32", fp32, "Full-length denominator buffer in _kappa_routed_fwd."),
            _ledger_item("Sval", (BH, nc, d, d), "fp32", fp32, "Running value state inside _kappa_routed_fwd."),
            _ledger_item("Sden", (BH, nc, d), "fp32", fp32, "Running denominator state inside _kappa_routed_fwd."),
            _ledger_item("routed_out_bf16", (BH, L, d), "bf16", bf16, "Returned routed output before o_proj."),
        ]
    total_live = sum(x["mib"] for x in items)
    total_routed = sum(x["mib"] for x in routed)
    return {
        "major_live_tensors": items,
        "routed_internal_tensors": routed,
        "major_live_total_mib": _round(total_live),
        "routed_internal_total_mib": _round(total_routed),
    }


def _phase_by_name(phases: list[Phase], name: str) -> Phase | None:
    for phase in phases:
        if phase.name == name:
            return phase
    return None


def _diagnostics(whole: dict[str, Any], phases: list[Phase], ledger: dict[str, Any],
                 profiler_info: dict[str, Any] | None) -> dict[str, Any]:
    routed = _phase_by_name(phases, "chunk_rola_routed")
    routed_peak = routed.peak_delta_mib if routed is not None else None
    routed_known = ledger.get("routed_internal_total_mib", 0.0)
    out = {
        "whole_forward_peak_mib": whole.get("peak_delta_mib"),
        "routed_phase_peak_mib": routed_peak,
        "known_routed_internal_mib": routed_known,
    }
    if routed_peak is not None:
        out["unattributed_routed_peak_mib"] = _round(max(0.0, routed_peak - routed_known))
    if profiler_info is not None:
        out["profiler_peak_mib"] = profiler_info.get("peak_delta_mib")
        out["profiler_peak_over_whole_mib"] = _round(
            profiler_info.get("peak_delta_mib", 0.0) - whole.get("peak_delta_mib", 0.0))
    return out


def _profile_rola_phases(module, x, args) -> list[Phase]:
    phases: list[Phase] = []
    chunk_rola_routed = _chunk_rola_routed()
    B, L, _ = x.shape
    H = module.num_heads
    use_short_conv = bool(getattr(module, "use_short_conv", False))

    def add(name, fn):
        phase, value = _measure_phase(name, fn, reps=args.reps)
        phases.append(phase)
        return value

    def conv_value(y):
        return y[0] if isinstance(y, tuple) else y

    with torch.inference_mode():
        q, k, v = add("qkv_proj", lambda: (module.q_proj(x), module.k_proj(x), module.v_proj(x)))
        if use_short_conv:
            q = add("q_short_conv", lambda: conv_value(module.q_conv1d(q)))
            k = add("k_short_conv", lambda: conv_value(module.k_conv1d(k)))
            v = add("v_short_conv", lambda: conv_value(module.v_conv1d(v)))
        q = q.reshape(B, L, H, module.proj_qk)
        k = k.reshape(B, L, H, module.proj_qk)
        v = v.reshape(B, L, H, module.head_v_dim)
        if module.qk_norm:
            q, k = add("qk_norm", lambda: (module.q_norm(q), module.k_norm(k)))
        qf, kf = add("feature_map", lambda: module._feature_map(q, k))
        if module.state_norm == "kappa":
            kappa_logits = add("kappa_logits", lambda: module.w_kappa(x))
            if use_short_conv:
                kappa_logits = add("kappa_short_conv", lambda: conv_value(module.kappa_conv1d(kappa_logits)))
            kap = add("kappa_sigmoid", lambda: torch.sigmoid(kappa_logits).view(B, L, H, 1))
        else:
            kap = None
        if module.kernel == "gla_scalar":
            alpha_logits = add("gla_alpha_logits", lambda: module.w_g(x))
            if use_short_conv:
                alpha_logits = add("gla_alpha_short_conv", lambda: conv_value(module.g_conv1d(alpha_logits)))
            alpha = add("gla_alpha_sigmoid", lambda: torch.sigmoid(alpha_logits).view(B, L, H))
            Wg = module._unit_w_g.to(device=x.device, dtype=torch.float32) if use_short_conv else module.w_g.weight
        else:
            alpha_logits = None
            alpha = None
            Wg = None
        if use_short_conv:
            wx = add("write_route_short_conv", lambda: conv_value(module.write_route_conv1d(x)))
            if module.read_W is None:
                rx = wx
            else:
                rx = add("read_route_short_conv", lambda: conv_value(module.read_route_conv1d(x)))
        else:
            wx = x
            rx = x
        needs_router_logits = module.router_zloss_coef > 0.0 or module.state_norm == "raw"
        if needs_router_logits:
            wl = add("write_router_logits_required", lambda: module._factor_logits(wx, module.write_W, module.write_b))
            if module.read_W is None:
                rl = wl
            else:
                rl = add("read_router_logits_required", lambda: module._factor_logits(rx, module.read_W, module.read_b))
            module._stash_zloss(wl, rl)
        else:
            wl = rl = None
            module._router_aux = None
        h_read = rx.unsqueeze(2).expand(B, L, H, module.hidden_size)
        h_write = wx.unsqueeze(2).expand(B, L, H, module.hidden_size)
        h_decay = (alpha_logits.view(B, L, H, 1) if (module.kernel == "gla_scalar" and use_short_conv)
                   else x.unsqueeze(2).expand(B, L, H, module.hidden_size))
        out = add("chunk_rola_routed", lambda: chunk_rola_routed(
            qf, kf, v, h_read, module.write_W if module.read_W is None else module.read_W, module.write_W,
            module.route_D, module.route_b, norm=module.state_norm, kappa=kap, scale=1.0,
            b_r=(module.write_b if module.read_W is None else module.read_b),
            b_w=module.write_b, Wg=Wg, wl=wl, rl=rl, alpha=alpha, h_w=h_write, h_g=h_decay))
        if alpha_logits is not None:
            # Keep the tensor live until after the routed phase, matching the real forward's lifetime.
            _ = alpha_logits
        add("o_proj", lambda: module.o_proj(out.reshape(B, L, H * module.head_v_dim)))
    return phases


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RoLA Graph Profile",
        "",
        "## Cell",
        "",
        f"- arm: `{summary['cell']['arm']}`",
        f"- L: `{summary['cell']['L']}`",
        f"- batch: `{summary['cell']['B']}`",
        f"- d_qk=d_v: `{summary['cell']['d_qk']}`",
        f"- nc: `{summary['cell']['nc']}`",
        f"- routing: `{summary['cell']['routing']}`",
        f"- dtype: `{summary['runtime']['dtype']}`",
        f"- gpu: `{summary['runtime']['gpu']}`",
        "",
        "## Whole Forward",
        "",
        f"- time: `{summary['whole_forward']['ms']} ms`",
        f"- IQR: `{summary['whole_forward'].get('iqr_ms', 0.0)} ms`",
        f"- IQR %: `{summary['whole_forward'].get('stats_ms', {}).get('iqr_pct', 0.0)}`",
        f"- peak delta: `{summary['whole_forward']['peak_delta_mib']} MiB`",
        "",
        "## Phase High-Water",
        "",
        "| phase | median ms | IQR ms | IQR % | alloc delta MiB | peak delta MiB | output MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in summary.get("phases", []):
        out_mib = sum(o.get("mib", 0.0) for o in p.get("outputs", []))
        lines.append(
            f"| `{p['name']}` | {p['ms']} | {p.get('iqr_ms', 0.0)} | "
            f"{p.get('stats_ms', {}).get('iqr_pct', 0.0)} | {p['alloc_delta_mib']} | "
            f"{p['peak_delta_mib']} | {round(out_mib, 3)} |"
        )
    analytic = summary.get("analytic", {})
    if analytic:
        lines.extend(["", "## Analytic Tensor Pressure", ""])
        for k, v in analytic.items():
            lines.append(f"- `{k}`: `{v}`")
    if "all_router_logits_equivalent_mib" in analytic:
        kv = analytic["attention_kv_cache_mib_bf16"]
        router = analytic["all_router_logits_equivalent_mib"]
        state = analytic["kernel_working_state_mib_fp32"]
        lines.extend([
            "",
            "## Read",
            "",
            f"- Full router logits would be `{router}` MiB versus `{state}` MiB of fp32 working recurrent state.",
            f"- Attention KV cache at this cell would be `{kv}` MiB in bf16.",
            "- The production chunk path streams routing tiles inside the scan, so this equivalent size should "
            "not appear as live memory unless raw mode or router z-loss explicitly requires logits.",
        ])
    diagnostics = summary.get("diagnostics", {})
    if diagnostics:
        lines.extend(["", "## Diagnostics", ""])
        for k, v in diagnostics.items():
            lines.append(f"- `{k}`: `{v}`")
    ledger = summary.get("buffer_ledger", {})
    if ledger:
        lines.extend(["", "## Routed Buffer Ledger", ""])
        lines.append(f"- `major_live_total_mib`: `{ledger.get('major_live_total_mib')}`")
        lines.append(f"- `routed_internal_total_mib`: `{ledger.get('routed_internal_total_mib')}`")
        lines.append("")
        lines.append("| buffer | MiB | dtype | shape |")
        lines.append("|---|---:|---|---|")
        for item in ledger.get("routed_internal_tensors", []):
            lines.append(f"| `{item['name']}` | {item['mib']} | `{item['dtype']}` | `{item['shape']}` |")
    if summary.get("torch_profiler"):
        lines.extend(["", "## Profiler Artifacts", ""])
        for k, v in summary["torch_profiler"].items():
            if k.startswith("top_"):
                continue
            if k == "trace_json_gz_base64":
                nbytes = summary["torch_profiler"].get("trace_json_gz_bytes", 0)
                lines.append(f"- `{k}`: `<embedded {nbytes} bytes gzip/base64>`")
                continue
            if k in ("cuda_table", "memory_table"):
                lines.append(f"- `{k}`: `<embedded text table, {len(v)} chars>`")
                continue
            lines.append(f"- `{k}`: `{v}`")
    if summary.get("deep_profile", {}).get("enabled"):
        lines.extend(["", "## Deep Profile", ""])
        deep = summary["deep_profile"]
        lines.append(f"- `mode`: `{deep.get('mode')}`")
        for name, info in deep.get("tools", {}).items():
            lines.append(f"- `{name}`: `ok={info.get('ok')}` `returncode={info.get('returncode')}`")
            if info.get("error"):
                lines.append(f"  - error: `{info['error']}`")
            if info.get("log_text"):
                lines.append(f"  - `log_text`: `<embedded {len(info['log_text'])} chars>`")
            if info.get("details_text"):
                lines.append(f"  - `details_text`: `<embedded {len(info['details_text'])} chars>`")
    lines.append("")
    return "\n".join(lines)


def _stdout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Small console copy; summary.json/result payload remains fully self-contained."""
    out = dict(summary)
    prof = out.get("torch_profiler")
    if isinstance(prof, dict):
        prof = dict(prof)
        if "trace_json_gz_base64" in prof:
            prof["trace_json_gz_base64"] = f"<embedded {prof.get('trace_json_gz_bytes', 0)} bytes gzip/base64>"
        for key in ("cuda_table", "memory_table"):
            if key in prof:
                prof[key] = f"<embedded text table, {len(prof[key])} chars>"
        out["torch_profiler"] = prof
    deep = out.get("deep_profile")
    if isinstance(deep, dict):
        deep = dict(deep)
        tools = {}
        for name, info in deep.get("tools", {}).items():
            info = dict(info)
            for key in ("stdout", "stderr", "log_text", "details_text", "details_stderr"):
                if key in info:
                    info[key] = f"<embedded {len(info[key])} chars>"
            tools[name] = info
        deep["tools"] = tools
        out["deep_profile"] = deep
    return out


def run(args) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("profile_graph requires CUDA")
    sc = _scaling()
    C.B = args.batch
    C.REPS = args.reps
    torch.manual_seed(args.seed)
    module = _build_arm(args.arm, args.L, args.d_qk, args.nc, args.routing)
    if module is None:
        raise SystemExit(f"arm {args.arm} is infeasible at nc={args.nc}, d_qk={args.d_qk}")
    module.eval()
    x = torch.randn(args.batch, args.L, sc.DM, device=C.DEV, dtype=C.DT)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _module_forward(module, x)
        _sync()

    out_dir = args.out / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = _whole_forward_profile(module, x, reps=args.reps)
    phases = _profile_rola_phases(module, x, args) if args.arm.startswith("rola-") else []
    profiler_info = None
    if args.torch_profiler:
        profiler_info = _torch_profile(module, x, out_dir, args.top, args.with_stack)
    ledger = _buffer_ledger(module, args.arm, args.batch, args.L, args.d_qk, args.nc)

    summary = {
        "cell": {
            "arm": args.arm,
            "L": args.L,
            "B": args.batch,
            "d_qk": args.d_qk,
            "nc": args.nc,
            "routing": args.routing,
            "reps": args.reps,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "runtime": _runtime_metadata(),
        "measurement": {
            "timing": "CUDA events",
            "whole_forward_reps": args.reps,
            "phase_reps": args.reps,
            "warmup": args.warmup,
            "headline": "median with IQR; raw samples retained in summary.json",
            "phase_note": "Phase timings are diagnostic and measured after the layer has been decomposed.",
        },
        "whole_forward": whole,
        "phases": [asdict(p) for p in phases],
        "analytic": _analytic_sizes(module, args.arm, args.batch, args.L, args.d_qk, args.nc),
        "buffer_ledger": ledger,
    }
    if profiler_info is not None:
        summary["torch_profiler"] = profiler_info
    deep_info = _deep_profile(args, out_dir)
    if deep_info.get("enabled"):
        summary["deep_profile"] = deep_info
    summary["diagnostics"] = _diagnostics(whole, phases, ledger, profiler_info)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(_markdown_report(summary))
    print(json.dumps(_stdout_summary(summary), indent=2))
    print(f"\nWROTE {out_dir}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Profile one RoLA scaling graph arm.")
    p.add_argument("--arm", choices=("rola-rla", "rola-gla", "monolith-rla", "monolith-gla", "attn"),
                   default="rola-rla")
    p.add_argument("--L", type=int, default=65536)
    p.add_argument("--batch", type=int, default=int(os.environ.get("PERF_B", "2")))
    p.add_argument("--d-qk", type=int, default=16)
    p.add_argument("--nc", type=int, default=64)
    p.add_argument("--routing", choices=("flat", "square", "tree"), default="square")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--deep-profile", choices=("none", "nsys", "ncu", "both"),
                   default=os.environ.get("ROLA_DEEP_PROFILE", "none"),
                   help="Optional NVIDIA profiler pass around a child run. nsys gives timeline/launch/memory; "
                        "ncu samples matching Triton kernels for hardware counters.")
    p.add_argument("--deep-kernel-name", default=os.environ.get("ROLA_DEEP_KERNEL", "_kappa_fwd_scan_cb_atomic"))
    p.add_argument("--deep-launch-skip", type=int, default=int(os.environ.get("ROLA_DEEP_LAUNCH_SKIP", "16")))
    p.add_argument("--deep-launch-count", type=int, default=int(os.environ.get("ROLA_DEEP_LAUNCH_COUNT", "1")))
    p.add_argument("--deep-ncu-set", default=os.environ.get("ROLA_DEEP_NCU_SET", "full"))
    p.add_argument("--deep-warmup", type=int, default=int(os.environ.get("ROLA_DEEP_WARMUP", "1")))
    p.add_argument("--deep-timeout", type=int, default=int(os.environ.get("ROLA_DEEP_TIMEOUT", "900")))
    p.add_argument("--deep-child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=Path(os.environ.get(
        "ROLA_PROFILE_OUT", str(C.RESULTS_DIR / "profile_graph"))))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
