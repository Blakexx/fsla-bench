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
import shutil
import subprocess
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
    alloc_delta_mib: float
    peak_delta_mib: float
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


def _measure_phase(name: str, fn) -> tuple[Phase, Any]:
    _sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    def wrapped():
        with record_function(f"profile_graph::{name}"):
            return fn()

    ms, value = _event_ms(wrapped)
    end_alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    phase = Phase(
        name=name,
        ms=_round(ms),
        alloc_delta_mib=_round(_mib(end_alloc - base)),
        peak_delta_mib=_round(_mib(peak - base)),
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


def _module_forward(module, x):
    o = module(x)
    return o[0] if isinstance(o, tuple) else o


def _whole_forward_profile(module, x) -> dict[str, Any]:
    with torch.inference_mode():
        _sync()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ms, y = _event_ms(lambda: _module_forward(module, x))
        peak = torch.cuda.max_memory_allocated()
    return {
        "ms": _round(ms),
        "peak_delta_mib": _round(_mib(peak - base)),
        "output": _tensor_info(y),
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
            "one_router_logits_mib": _round(_mib(one_logits)),
            "all_router_logits_mib": _round(_mib(one_logits * router_count)),
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
        for name in ("write_router_logits", "read_router_logits")[:route_count]:
            items.append(_ledger_item(name, (B, L, H, D, b), "bf16", bf16, "Materialized factor logits."))
        if module.state_norm == "kappa":
            items.append(_ledger_item("kappa", (B, L, H, 1), "bf16", bf16, "Per-token kappa exponent."))
        if module.kernel == "gla_scalar":
            items.append(_ledger_item("gla_alpha_logits", (B, L, H), "bf16", bf16, "Per-head scalar decay logits."))
        routed = [
            _ledger_item("kappa_num", (BH, L, d), "fp32", fp32, "Full-length numerator buffer in _kappa_routed_fwd."),
            _ledger_item("kappa_den", (BH, L), "fp32", fp32, "Full-length denominator buffer in _kappa_routed_fwd."),
            _ledger_item("Sval", (BH, nc, d, d), "fp32", fp32, "Running value state inside _kappa_routed_fwd."),
            _ledger_item("Sden", (BH, nc, d), "fp32", fp32, "Running denominator state inside _kappa_routed_fwd."),
            _ledger_item("routed_out_bf16", (BH, L, d), "bf16", bf16, "Returned routed output before o_proj."),
        ]
        if module.state_norm == "kappa":
            routed.append(_ledger_item(
                "normalized_out_temp", (BH, L, d), "fp32", fp32,
                "Temporary fp32 num/den result before casting back to bf16."))
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

    def add(name, fn):
        phase, value = _measure_phase(name, fn)
        phases.append(phase)
        return value

    with torch.inference_mode():
        q, k, v = add("qkv_proj", lambda: (module.q_proj(x), module.k_proj(x), module.v_proj(x)))
        q = q.reshape(B, L, H, module.proj_qk)
        k = k.reshape(B, L, H, module.proj_qk)
        v = v.reshape(B, L, H, module.head_v_dim)
        if module.qk_norm:
            q, k = add("qk_norm", lambda: (module.q_norm(q), module.k_norm(k)))
        qf, kf = add("feature_map", lambda: module._feature_map(q, k))
        if module.state_norm == "kappa":
            kap = add("kappa_logits_sigmoid", lambda: torch.sigmoid(module.w_kappa(x)).view(B, L, H, 1))
        else:
            kap = None
        if module.kernel == "gla_scalar":
            alpha_logits = add("gla_alpha_logits", lambda: module.w_g(x))
            Wg = module.w_g.weight
            h = x.unsqueeze(2).expand(B, L, H, module.hidden_size)
        else:
            alpha_logits = None
            Wg = None
            h = x.unsqueeze(2).expand(B, L, H, module.hidden_size)
        wl = add("write_router_logits", lambda: module._factor_logits(x, module.write_W, module.write_b))
        if module.read_W is None:
            rl = wl
        else:
            rl = add("read_router_logits", lambda: module._factor_logits(x, module.read_W, module.read_b))
        module._stash_zloss(wl, rl)
        out = add("chunk_rola_routed", lambda: chunk_rola_routed(
            qf, kf, v, h, module.write_W if module.read_W is None else module.read_W, module.write_W,
            module.route_D, module.route_b, norm=module.state_norm, kappa=kap, scale=1.0,
            b_r=(module.write_b if module.read_W is None else module.read_b),
            b_w=module.write_b, Wg=Wg, wl=wl, rl=rl))
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
        f"- peak delta: `{summary['whole_forward']['peak_delta_mib']} MiB`",
        "",
        "## Phase High-Water",
        "",
        "| phase | ms | alloc delta MiB | peak delta MiB | output MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in summary.get("phases", []):
        out_mib = sum(o.get("mib", 0.0) for o in p.get("outputs", []))
        lines.append(
            f"| `{p['name']}` | {p['ms']} | {p['alloc_delta_mib']} | "
            f"{p['peak_delta_mib']} | {round(out_mib, 3)} |"
        )
    analytic = summary.get("analytic", {})
    if analytic:
        lines.extend(["", "## Analytic Tensor Pressure", ""])
        for k, v in analytic.items():
            lines.append(f"- `{k}`: `{v}`")
    if "all_router_logits_mib" in analytic:
        kv = analytic["attention_kv_cache_mib_bf16"]
        router = analytic["all_router_logits_mib"]
        state = analytic["kernel_working_state_mib_fp32"]
        lines.extend([
            "",
            "## Read",
            "",
            f"- Router logits are `{router}` MiB versus `{state}` MiB of fp32 working recurrent state.",
            f"- Attention KV cache at this cell would be `{kv}` MiB in bf16.",
            "- If peak tracks router logits more than recurrent state, the fix is tiled/progressive "
            "router-logit production rather than changing the state formula.",
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
    whole = _whole_forward_profile(module, x)
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
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(C.DT).replace("torch.", ""),
            "rola_bench_rev": _git_rev("/home/blake/rola-bench"),
            "fla_rola_rev": _git_rev("/home/blake/flash-linear-attention-rola"),
        },
        "whole_forward": whole,
        "phases": [asdict(p) for p in phases],
        "analytic": _analytic_sizes(module, args.arm, args.batch, args.L, args.d_qk, args.nc),
        "buffer_ledger": ledger,
    }
    if profiler_info is not None:
        summary["torch_profiler"] = profiler_info
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
    p.add_argument("--out", type=Path, default=Path(os.environ.get(
        "ROLA_PROFILE_OUT", str(C.RESULTS_DIR / "profile_graph"))))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
