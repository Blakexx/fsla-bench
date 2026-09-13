"""Routing-distribution figures from the `routing` similarity probe (run_sim._probe_routing).

Reproducible by construction: a figure is a pure function of the committed probe rows — no model,
no GPU. Reads a results jsonl, renders one figure per RoLA cell into results/figures/.

RoLA routing is PER-HEAD, so we never average over a fixed state index (heads specialize to
different states, which would cancel). Instead:
  * Panel 1 (rank-frequency): align EACH head by its own WRITE rank (rank 1 = that head's
    most-written state), then show the head median +/- 10-90% band for write and for read-at-
    write-rank. Peaks line up, per-head sparsity is preserved, one curve pair.
  * Panel 2 (scatter): one point per (head, state) — fully per-head, no aggregation — write vs
    read mass. The "written XOR read" L-shape shows read avoids the heavily-written states.

Usage:
  python -m rola_bench.similarity.plot_routing [results.jsonl] [--out DIR]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DEFAULT_SRC = os.path.join(HERE, "..", "..", "results", "similarity", "routing_dispatch.jsonl")
DEFAULT_OUT = os.path.join(HERE, "..", "..", "results", "figures")


def routing_rows(path):
    out = []
    for line in open(path):
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("ok") and j.get("probe") == "routing":
            out.append(j)
    return out


def _vectors(row):
    for v in row["per_slice"].values():
        if "vectors" in v:
            return v["vectors"]
    return None


def plot_cell(row, axes):
    """Render one cell's (write-rank, scatter) panels into axes=(ax_rank, ax_scatter)."""
    nc = row["num_chunks"]
    v = _vectors(row)
    if v is None:
        return False
    W = np.asarray(v["write_mass"], dtype=float)   # [H, nc]
    R = np.asarray(v["read_mass"], dtype=float)     # [H, nc]
    H = W.shape[0]

    ax = axes[0]
    order = np.argsort(-W, axis=1)                  # per-head write ranking
    Ws = np.take_along_axis(W, order, axis=1)       # write sorted desc per head
    Rw = np.take_along_axis(R, order, axis=1)       # read reindexed by the SAME write ranking
    rank = np.arange(1, nc + 1)
    for arr, color, lab in [(Ws, "tab:blue", "write"), (Rw, "tab:orange", "read @ write-rank")]:
        ax.plot(rank, np.median(arr, 0), color=color, lw=2, label=lab)
        ax.fill_between(rank, np.percentile(arr, 10, 0), np.percentile(arr, 90, 0), color=color, alpha=0.2)
    ax.axhline(1.0 / nc, ls="--", c="gray", lw=1, label="uniform (1/nc)")
    ax.set_yscale("log")
    ax.set_xlabel("state rank (by each head's WRITE mass)")
    ax.set_ylabel("mean routing mass")
    ax.set_title(f"nc={nc}: per-head write-rank  (median, 10-90% band over {H} heads)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(W.ravel() + 1e-7, R.ravel() + 1e-7, s=7, alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    lo, hi = 1e-6, 1.0
    ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("write mass  (per head x state)")
    ax.set_ylabel("read mass")
    ax.set_title(f"nc={nc}: write vs read, all {H} heads pooled")
    return True


def plot_joint(row, src_path, out_dir):
    """Render the per-head JOINT write x read co-occurrence heatmap M[i,j]=<write_i read_j>, decoded
    from the base64 blob IN the row (generic — no external files). Rows sorted by write marginal,
    cols by read marginal, so high-write/high-read sit top-left; LogNorm color. One panel per head."""
    from matplotlib.colors import LogNorm
    from rola_bench.similarity.run import _unb64_f16
    v = _vectors(row)
    if not v or "joint_b64" not in v:
        return None
    M = _unb64_f16(v["joint_b64"], v["joint_shape"])      # [H,nc,nc]
    wm = np.asarray(v["write_mass"], dtype=float)          # [H,nc]
    rm = np.asarray(v["read_mass"], dtype=float)
    H, nc, _ = M.shape
    cols = int(np.ceil(H ** 0.5)); rows = int(np.ceil(H / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), squeeze=False)
    vmax = float(M.max())
    for h in range(H):
        wo = np.argsort(-wm[h])                               # ONE ordering (write mass) for BOTH axes,
        Mh = M[h][np.ix_(wo, wo)]                             # so index i = same physical state on both ->
        ax = axes[h // cols][h % cols]                        # diagonal = "write to i AND read from i".
        im = ax.imshow(Mh + 1e-12, aspect="auto", origin="upper",
                       norm=LogNorm(vmin=max(1e-9, vmax * 1e-5), vmax=vmax), cmap="magma")
        ax.plot([0, nc], [0, nc], ls=":", c="cyan", lw=0.8)   # self-pairing diagonal
        ax.set_xlim(0, nc); ax.set_ylim(nc, 0)
        ax.set_title(f"head {h}"); ax.set_xlabel("READ state (ordered by write mass)")
        ax.set_ylabel("WRITE state (ordered by write mass)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    nc_ = row["num_chunks"]
    fig.suptitle(f"nc={nc_}: joint write x read  M[i,j]=<write_i . read_j>  (both axes by WRITE mass; "
                 f"left cols = most-written states — are they read?)", y=1.0)
    fig.tight_layout()
    out = os.path.join(out_dir, f"routing_joint_nc{nc_}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def plot_pmi(row, out_dir, mass_floor=1e-3):
    """PMI map: log( M[i,j] / (w[i] r[j]) ) — the joint with the marginal-product baseline divided
    out, so ONLY deviations from per-token independence show. ~0 (white) everywhere => write/read
    are independent per token; a bright diagonal => same-state recall; off-diagonal bands => handoffs.
    States below `mass_floor` on either marginal are masked (PMI is undefined/explosive there)."""
    from matplotlib.colors import TwoSlopeNorm
    from rola_bench.similarity.run import _unb64_f16
    v = _vectors(row)
    if not v or "joint_b64" not in v:
        return None
    M = _unb64_f16(v["joint_b64"], v["joint_shape"])      # [H,nc,nc]
    wm = np.asarray(v["write_mass"], dtype=float); rm = np.asarray(v["read_mass"], dtype=float)
    H, nc, _ = M.shape
    cols = int(np.ceil(H ** 0.5)); rows = int(np.ceil(H / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), squeeze=False)
    for h in range(H):
        w, r = wm[h], rm[h]
        baseline = np.outer(w, r)                         # independence prediction w_i r_j
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(np.where((M[h] > 0) & (baseline > 0), M[h] / baseline, np.nan))
        keep = (w >= mass_floor)[:, None] & (r >= mass_floor)[None, :]   # mask explosive low-mass cells
        pmi = np.where(keep, pmi, np.nan)
        wo = np.argsort(-w)                               # both axes by write mass (consistent ordering)
        P = pmi[np.ix_(wo, wo)]
        finite = P[np.isfinite(P)]
        lim = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
        lim = max(lim, 1e-3)
        ax = axes[h // cols][h % cols]
        im = ax.imshow(P, aspect="auto", origin="upper", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
        ax.plot([0, nc], [0, nc], ls=":", c="k", lw=0.6)
        ax.set_xlim(0, nc); ax.set_ylim(nc, 0)
        ax.set_title(f"head {h}"); ax.set_xlabel("READ state (by write mass)")
        ax.set_ylabel("WRITE state (by write mass)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="PMI (log M/(w·r))")
    nc_ = row["num_chunks"]
    fig.suptitle(f"nc={nc_}: PMI = log( M / (write_marg x read_marg) ) — deviation from independence "
                 f"(red>0 co-occur, blue<0 avoid, white~0 independent; masked below mass {mass_floor})", y=1.0)
    fig.tight_layout()
    out = os.path.join(out_dir, f"routing_pmi_nc{nc_}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    rows = sorted(routing_rows(a.src), key=lambda r: -r["num_chunks"])
    if not rows:
        raise SystemExit(f"no ok routing rows with vectors in {a.src}")
    os.makedirs(a.out, exist_ok=True)
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        plot_cell(row, axes[i])
    fig.tight_layout()
    out = os.path.join(a.out, "routing_dist.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  ({len(rows)} cell(s): {[r['run_id'].split('::')[0] for r in rows]})")
    # Per-cell joint write x read heatmaps (one file per cell — they're 2D, don't stack well).
    for row in rows:
        jp = plot_joint(row, a.src, a.out)
        if jp:
            print(f"wrote {jp}")
        pp = plot_pmi(row, a.out)
        if pp:
            print(f"wrote {pp}")


if __name__ == "__main__":
    main()
