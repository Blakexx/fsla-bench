"""Generic, checkpoint-driven post-hoc RANK evaluator (rola-bench task #85).

This is the standalone successor to the rank diagnostic that currently lives INLINE in
the model library `rola.py` (gated by env ROLA_MEASURE_RANK, emitting `RANK_JSON {...}`
lines from the kernel forward). Task #84 will remove that inline probe; this module
replaces it.

Why a separate module: the inline probe can only run during a live forward pass of a
RoLA kernel. We want to (a) measure rank POST-HOC on a saved checkpoint, (b) measure it
for ANY model — softmax attention, RoLA, or any linear-attention variant — not just
RoLA, and (c) keep it decoupled from the training/kernel runtime so it survives #84.

Genericity is achieved with an `attention_extractor` callback under ONE rule: it returns the
model's FINAL attention matrix M — the operator in `O = M @ V`, with the model's own
normalization applied. This single rule encompasses every architecture and needs no
rank-invariance argument:
  - softmax attention            -> M = softmax(scores)            (already normalized)
  - unnormalized linear attn     -> M = phi(Q) phi(K)ᵀ ∘ (read writeᵀ)   (no division)
  - normalized linear attn / RoLA-> M = row-normalized of the above (the denominator division)
The metric core SVDs whatever M the extractor returns. See `to_final_attention` for the
normalize/causal handling shared by the linear-attention family.

--------------------------------------------------------------------------------------
METRIC PROVENANCE — mirrored EXACTLY from rola.py (commit 8f7b918, the reference impl):

  `_rank_stats`            rola.py lines 311-338   -> rank_stats() / _rank_stats_from_sv()
  `_effective_attention_rank` rola.py lines 342-401 -> the per-slice SVD loop + spectrum
       (the per-(seq,head) loop @ 374-381, _rank_stats call @ 382, spectrum @ 385-400)
  RoLA effective weight W  rola.py line 376         -> _rola_attn_from_features()
       `W = (q[bi,:,hi] @ k[bi,:,hi].T) * (r[bi,:,hi] @ w[bi,:,hi].T)`

The numerical definitions below are copied verbatim from those lines (not reinvented):
  - numerical rank at tol t:  #{ sigma > t * sigma_max }            (rola.py:330)
  - Roy & Vetterli effective rank: exp(-sum p log p), p = sigma/sum(sigma)  (rola.py:333-334)
  - participation ratio rank:  (sum sigma)^2 / sum(sigma^2)         (rola.py:335)
  - stable rank:  sum(sigma^2) / sigma_max^2  ==  ||W||_F^2 / ||W||_2^2  (rola.py:337)
The UNMASKED W is the measured object (rola.py:353-360); the causal mask only inflates
rank and carries no rank signal, so it is off by default here too.
--------------------------------------------------------------------------------------

Import-light: torch only. No training stack, no rola.py runtime dependency for the core
metrics. The RoLA adapter (build_rola_attention_extractor) imports rola lazily, only when
you actually use it, so the metric core stays usable without rola installed.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch


# Default numerical-rank tolerances — identical to rola.py:342 (_effective_attention_rank
# default `tols=(1e-1, 1e-2, 1e-3, 1e-4)`).
DEFAULT_TOLS: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)


# ======================================================================================
# Core metric — mirrors rola.py `_rank_stats` (lines 311-338).
# ======================================================================================
def _rank_stats_from_sv(sv: torch.Tensor, l: int, tols: Sequence[float], prefix: str = "") -> dict:
    """Rank statistics from singular values sv:[N,l] (sorted desc per row), over the
    N = b*H per-(sequence,head) slices. VERBATIM port of rola.py `_rank_stats`
    (rola.py:311-338). Every estimator is emitted both as the legacy mean and as the
    full sorted per-slice distribution `*_dist` (the paper's barrier claim is
    distributional — a mean over slices can report a rank no slice realizes).

    Estimators (rola.py:329-337):
      numerical rank #{sigma > tol*sigma_max} at each tol;
      Roy & Vetterli (2007) effective rank exp(H(sigma/sum sigma));
      participation ratio (sum sigma)^2 / sum(sigma^2);
      stable rank sum(sigma^2) / sigma_max^2.
    """
    smax = sv[..., :1].clamp_min(1e-20)                       # rola.py:322
    out: dict = {}

    def put(name: str, vals: torch.Tensor) -> None:           # rola.py:325-327
        out[f"{prefix}{name}"] = round(vals.mean().item(), 2)
        out[f"{prefix}{name}_dist"] = [round(v, 2) for v in vals.sort().values.tolist()]

    for t in tols:                                            # rola.py:329-330
        put(f"rank_{t:.0e}", (sv > t * smax).sum(-1).float())
    # Roy & Vetterli effective rank: exp(Shannon entropy, nats, of the L1-normalized
    # singular-value distribution). xlogy gives the 0*log0 -> 0 convention. (rola.py:331-334)
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-20)
    put("eff_rank", torch.exp(-torch.special.xlogy(p, p).sum(-1)))
    put("pr_rank", sv.sum(-1) ** 2 / (sv ** 2).sum(-1).clamp_min(1e-20))   # rola.py:335
    # stable (numerical) rank ||W||_F^2 / sigma_max^2 = sum sigma^2 / sigma_max^2;
    # threshold-free, in [1, rank]. (rola.py:336-337)
    put("stable_rank", (sv ** 2).sum(-1) / smax.squeeze(-1) ** 2)
    return out


@torch.no_grad()
def rank_stats(
    attn: torch.Tensor,
    tols: Sequence[float] = DEFAULT_TOLS,
    max_b: int = 4,
    max_l: int = 4096,
    emit_spectrum: bool = True,
) -> dict:
    """Numerical / stable / effective rank of an attention matrix.

    `attn` is the (implied) attention weight, shape [B,H,L,L] or [L,L] or [B,L,L]. It is
    SVD'd one L x L slice at a time (loop over batch x head) — identical math to a batched
    SVD but with peak memory = one L x L matrix, matching rola.py's per-slice loop
    (rola.py:369-381), which exists because the batched cuSOLVER workspace OOMs / corrupts
    the heavy cell and produced anomalous non-monotone rank.

    Returns the same dict shape as rola.py's `_effective_attention_rank`: the `_rank_stats`
    keys (rank_1e-01 ... stable_rank, each + `_dist`) plus the spectrum quantiles
    (spec_idx / spec_p10 / spec_p50 / spec_p90), sv_ratio_128/256, seq_len, n_slices.

    Definitions are copied verbatim from rola.py:311-401 (see module docstring). The
    measured object is the UNMASKED matrix exactly as the caller supplies it — no causal
    mask applied (rola.py:353-360: the mask only inflates rank, carries no rank signal).
    """
    if attn.dim() == 2:           # [L,L]   -> single slice
        attn = attn[None, None]
    elif attn.dim() == 3:         # [B,L,L] -> one head
        attn = attn[:, None]
    elif attn.dim() != 4:
        raise ValueError(f"attn must be [L,L], [B,L,L] or [B,H,L,L]; got shape {tuple(attn.shape)}")

    B, H, L, L2 = attn.shape
    if L != L2:
        raise ValueError(f"attention matrix must be square in its last two dims; got {L}x{L2}")
    b = min(max_b, B)
    l = min(max_l, L)

    # Per-slice SVD loop (rola.py:373-381). Subsample to [:b, :, :l, :l] up front so the
    # singular-value tensor is l-wide (matches rola.py q[:b,:l] subsampling at line 366).
    svs = []
    diff_l, effrac_l, t1_l = [], [], []                       # per-row SHARPNESS (selection peakiness)
    for bi in range(b):
        for hi in range(H):
            W = attn[bi, hi, :l, :l].float()
            svs.append(torch.linalg.svdvals(W))               # rola.py:378
            # Each row = query i's weighting over keys. Renormalize to a selection distribution
            # (clamp negatives->0 [no-op for our non-negative similarity], divide by row sum) ->
            # SCALE-INVARIANT (comparable normalized vs unnormalized). Then normalize the non-
            # uniformity by the number of AVAILABLE keys K (causal: i+1; full: L) -> a clean [0,1]
            # "diffuseness" comparable across seq-len / causal position. rank = how many UNIQUE
            # blends; sharpness = how PEAKED each blend is (orthogonal axes).
            is_causal = W.triu(1).abs().max().item() < 1e-8
            A = W.clamp(min=0)
            P = A / A.sum(-1, keepdim=True).clamp_min(1e-20)   # [l, l] row-stochastic
            ent = -(P * P.clamp_min(1e-20).log()).sum(-1)      # nats, per row
            ipr = 1.0 / P.pow(2).sum(-1).clamp_min(1e-20)      # effective #keys blended, per row
            Kav = (torch.arange(1, l + 1, device=W.device).float() if is_causal
                   else torch.full((l,), float(l), device=W.device))  # available keys per row
            m = Kav >= 2                                       # drop trivial rows (<2 available keys)
            if m.any():
                diff_l.append(ent[m] / Kav[m].log())           # diffuseness in [0,1]: 0=one-hot,1=uniform
                effrac_l.append(ipr[m] / Kav[m])               # eff-keys as a FRACTION of available
                t1_l.append(P[m].amax(-1))                     # top-1 selection mass
    sv = torch.stack(svs)                                     # [b*H, l]   rola.py:381

    out = _rank_stats_from_sv(sv, l, tols)                    # rola.py:382

    # SHARPNESS panel (L-normalized, scale-free -> comparable across models/seq-len/normalization).
    # Complements rank: a full-rank matrix can still be diffuse (diffuseness~1), which rank alone
    # can't see; together they pin down "many sharp distinct selections" = a (partial) permutation.
    if diff_l:
        D = torch.cat(diff_l); E = torch.cat(effrac_l); T = torch.cat(t1_l)
        out["sharp_diffuseness_mean"] = round(D.mean().item(), 4)   # 0=one-hot select, 1=uniform blend
        out["sharp_diffuseness_p90"] = round(torch.quantile(D, 0.9).item(), 4)  # most-diffuse decile
        out["sharp_eff_key_frac_mean"] = round(E.mean().item(), 4)  # eff #keys / available keys
        out["sharp_top1_mean"] = round(T.mean().item(), 4)          # mean top-1 mass

    if emit_spectrum:
        # Spectrum sigma_i/sigma_max at log-spaced indices, quantiles over slices — the
        # "supply thins" plot of the paper's section 4. (rola.py:385-398)
        smax = sv[..., :1].clamp_min(1e-20)                   # rola.py:385
        idx = sorted({0, 1} | {2 ** i for i in range(1, l.bit_length())} | {l - 1})
        idx = [i for i in idx if i < l]
        rel = (sv / smax)[:, idx]
        qs = torch.quantile(rel, torch.tensor([0.1, 0.5, 0.9], device=rel.device), dim=0)
        out["spec_idx"] = idx
        out["spec_p10"] = [round(v, 5) for v in qs[0].tolist()]
        out["spec_p50"] = [round(v, 5) for v in qs[1].tolist()]
        out["spec_p90"] = [round(v, 5) for v in qs[2].tolist()]
        out["sv_ratio_128"] = round((sv[..., min(127, l - 1)] / smax.squeeze(-1)).mean().item(), 4)
        out["sv_ratio_256"] = round((sv[..., min(255, l - 1)] / smax.squeeze(-1)).mean().item(), 4)
    out["seq_len"] = l                                        # rola.py:399
    out["n_slices"] = sv.shape[0]                             # rola.py:400
    return out


# ======================================================================================
# RoLA effective-weight assembly — mirrors rola.py:376.
# ======================================================================================
def rola_attn_from_features(
    qf: torch.Tensor,
    kf: torch.Tensor,
    read_gates: torch.Tensor,
    write_gates: torch.Tensor,
) -> torch.Tensor:
    """Assemble the per-head RoLA effective weight W = (phi(Q) phi(K)^T) ∘ (read write^T).

    Inputs are the model-layout tensors EXACTLY as `_effective_attention_rank` receives
    them (rola.py:363): qf,kf:[B,L,H,d_qk]  read_gates,write_gates:[B,L,H,nc]. The Hadamard
    product is rola.py:376:
        W = (q[bi,:,hi] @ k[bi,:,hi].T) * (r[bi,:,hi] @ w[bi,:,hi].T)   # [L,L]
    Returns the full [B,H,L,L] stack (this evaluator's metric core loops the SVD itself, so
    we hand it the batched tensor rather than rola.py's pre-looped list).

    NOTE: pass the SAME tensors the inline probe would see — i.e. qf/kf AFTER the feature
    map (elu+1 / hedgehog / ...), and read_gates AFTER any kappa/per_state rescaling. The
    RoLA adapter below does exactly that.
    """
    G = torch.einsum("blhd,bmhd->bhlm", qf, kf)       # (phi(Q) phi(K)^T) per head
    R = torch.einsum("blhc,bmhc->bhlm", read_gates, write_gates)  # (read write^T) per head
    return G * R                                       # Hadamard — rola.py:376


def to_final_attention(
    W: torch.Tensor,
    normalize: bool,
    causal: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reduce a raw effective-weight matrix W[B,H,L,L] to the FINAL attention matrix the model
    applies — the operator M in `O = M @ V`.

    This is the single rule that unifies every architecture's rank measurement, with NO
    rank-invariance argument required: each model's extractor returns the matrix it actually
    uses, and we SVD that.
      - softmax attention      -> the extractor returns softmax(scores) directly (already final).
      - unnormalized linear attn (RoLA `raw`) -> normalize=False  (M = W).
      - normalized linear attn / RoLA global|kappa|per_state -> normalize=True (row-normalized:
        M_ij = W_ij / sum_j W_ij, i.e. the linear-attention denominator division).

    `causal` (default False) is an ORTHOGONAL knob: the literal operator is lower-triangular, but
    the causal mask inflates rank and carries no rank signal, so the rank-law object is the
    unmasked matrix (matches the reference). Flip it on only to inspect the literal causal operator.

    We deliberately SVD the operator each model actually uses; the numerical / stable / effective
    rank estimators and the spectrum reflect its true singular-value distribution — we make NO
    rank-invariance argument. `eps` is ADDITIVE (matches rola.py's `/(den+1e-5)`); for the positive
    feature maps in use (elu+1, taylor, softmax-based) rowsums are strictly positive so this is
    well-conditioned. (`causal=True` is a labeled inspection mode — masked rows can be degenerate —
    not the rank-law object.)"""
    if causal:
        W = W.tril()                                   # j<=i; future masked to 0
    if normalize:
        W = W / (W.sum(-1, keepdim=True) + eps)        # linear-attention denominator (additive eps)
    return W


# ======================================================================================
# Generic checkpoint-driven driver.
# ======================================================================================
def make_eval_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 1234,
    device: str = "cpu",
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """A FIXED, seeded eval batch so ranks are comparable across cells/models.

    Returns token ids [B,L] in [0, vocab_size). Use the SAME (batch_size, seq_len,
    vocab_size, seed) for every model you want to compare — that is the whole point of
    pinning the batch (rola.py's inline probe measured on whatever live batch flowed
    through; here we control it so cross-model numbers are apples-to-apples)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    return ids.to(device=device, dtype=dtype)


@torch.no_grad()
def evaluate_rank(
    checkpoint_path: Optional[str],
    model_builder: Callable[[], torch.nn.Module],
    attention_extractor: Callable[[torch.nn.Module, object], torch.Tensor],
    eval_batch: object,
    *,
    map_location: str = "cpu",
    state_dict_key: Optional[str] = None,
    strict: bool = True,
    tols: Sequence[float] = DEFAULT_TOLS,
    max_b: int = 4,
    max_l: int = 4096,
    emit_spectrum: bool = True,
) -> dict:
    """Load a checkpoint into a freshly-built model, extract its attention on a fixed
    batch, and return rank_stats.

    Args:
      checkpoint_path: path to a torch-saved checkpoint, or None to evaluate the model as
        built by model_builder (e.g. for a toy / randomly-initialized sanity run).
      model_builder: zero-arg fn returning the model (architecture only; weights loaded
        here). Keeping construction in a callback is what makes this generic — the driver
        never needs to know the model class.
      attention_extractor: fn(model, eval_batch) -> attn[B,H,L,L] (or [L,L]/[B,L,L]). The
        ONLY model-specific seam. Softmax models return their softmax weights; RoLA /
        linear-attn return the implied W. See build_rola_attention_extractor.
      eval_batch: passed straight to the extractor; pin it (make_eval_batch) for
        cross-model comparability.
      state_dict_key: if the checkpoint is a dict wrapping the weights under a key (e.g.
        "model", "state_dict", "model_state_dict"), name it. If None, the driver tries a
        few common keys, else treats the loaded object as the state_dict itself.
      strict: passed to load_state_dict.

    Returns the rank_stats dict (same shape as rola.py's RANK_JSON payload, minus the
    nc/d_qk/d_model/epoch/seqlen annotations the caller adds).
    """
    model = model_builder()
    if checkpoint_path is not None:
        sd = _load_state_dict(checkpoint_path, map_location, state_dict_key)
        model.load_state_dict(sd, strict=strict)
    model.eval()
    attn = attention_extractor(model, eval_batch)
    if not torch.is_tensor(attn):
        raise TypeError(f"attention_extractor must return a torch.Tensor; got {type(attn)}")
    return rank_stats(attn, tols=tols, max_b=max_b, max_l=max_l, emit_spectrum=emit_spectrum)


def _load_state_dict(path: str, map_location: str, state_dict_key: Optional[str]) -> dict:
    """Load a checkpoint and dig out the state_dict. Fails LOUD if it can't find one
    (per the project's no-silent-fallbacks principle)."""
    # weights_only=False permits arbitrary pickled objects — TRUSTED LOCAL checkpoints only
    # (our own fleet ckpts). Never point this at untrusted files (arbitrary-code-execution risk).
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if state_dict_key is not None:
        if state_dict_key not in obj:
            raise KeyError(f"state_dict_key {state_dict_key!r} not in checkpoint keys {list(obj.keys())}")
        return obj[state_dict_key]
    if isinstance(obj, dict):
        # Already a bare state_dict? (values are tensors)
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        for k in ("model", "state_dict", "model_state_dict", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        raise KeyError(
            f"could not locate a state_dict in checkpoint {path!r}; top-level keys were "
            f"{list(obj.keys())}. Pass state_dict_key=... explicitly."
        )
    raise TypeError(f"checkpoint {path!r} loaded as {type(obj)}, not a dict; pass state_dict_key=...")


# ======================================================================================
# Per-model attention adapter for RoLA.
# ======================================================================================
def build_rola_attention_extractor(
    layer_index: int = 0,
    normalize: Optional[bool] = None,
    causal: bool = False,
) -> Callable:
    """Return an `attention_extractor(model, eval_batch) -> M[B,H,L,L]` for a RoLA model, where
    M is the FINAL attention matrix (see `to_final_attention`).

    Genericity boundary: the driver is model-agnostic; THIS function knows RoLA's
    internals. It reproduces, OUTSIDE the kernel forward, exactly the pipeline that feeds
    rola.py's inline probe (rola.py:669-693 for RLA / 800-822 for GLA, then 758 / 891):

        q,k,v   = w_q(x), w_k(x), w_v(x)         (+ optional short conv)   rola.py:1069-1080
        qf,kf   = kernel._feature_map(q,k)        (elu+1 / hedgehog / ...)  rola.py:663-667
        w_g,r_g = mixer._route(x)                 (softmax read/write)      rola.py:1055-1061
        [kappa/per_state] rescale read_gates by the per-state denominator   rola.py:672-693
        W       = (qf phi-gram) ∘ (read write^T)                            rola.py:376

    The probe measures the UN-DECAYED gram for GLA too (rola.py:879-882: the scalar decay is
    a rank-preserving two-sided diagonal rescale), so this extractor does NOT apply decay —
    matching the reference.

    `model` may be a bare RoLA mixer, or any module exposing the target RoLA mixer; we
    locate the `layer_index`-th submodule that quacks like a RoLA mixer (has w_q / kernel /
    write_router). `eval_batch` may be token ids [B,L] (we embed via a discovered embedding
    when the model isn't a bare mixer) OR a ready hidden-state tensor [B,L,d_model].
    """
    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        mixer = _find_rola_mixer(model, layer_index)
        x = _to_hidden_states(model, mixer, eval_batch)   # [B,L,d_model] on model's device/dtype
        qf, kf, read_gates, write_gates = _rola_features(mixer, x)
        W = rola_attn_from_features(qf, kf, read_gates, write_gates)
        # FINAL matrix: row-normalize iff the kernel normalizes (state_norm != 'raw'); the
        # additive RLA kernel always normalizes (global/per_state/kappa), GLA normalizes unless
        # 'raw'. Override with `normalize=`. Decay is omitted (rank-preserving diagonal rescale).
        norm = normalize
        if norm is None:
            norm = getattr(mixer.kernel, "state_norm", "global") != "raw"
        return to_final_attention(W, normalize=norm, causal=causal)

    return extractor


# ======================================================================================
# Baseline adapters — each returns the model's FINAL attention matrix M[B,H,L,L], computed
# from THAT layer's OWN projections / feature maps / normalization (faithful, not guessed).
# All share the `(layer_index, causal)` contract and route through `to_final_attention`.
# ======================================================================================
def _find_layer(model: torch.nn.Module, layer_index: int, predicate, what: str) -> torch.nn.Module:
    """Locate the `layer_index`-th submodule satisfying `predicate` (or `model` itself). Fails loud."""
    if predicate(model):
        return model
    hits = [m for m in model.modules() if predicate(m)]
    if not hits:
        raise ValueError(f"no {what} found in the model")
    if layer_index >= len(hits):
        raise IndexError(f"layer_index={layer_index} but only {len(hits)} {what}(s) found")
    return hits[layer_index]


def _is_fla_linear_attn(m: torch.nn.Module) -> bool:
    return all(hasattr(m, a) for a in ("q_proj", "k_proj", "feature_map_q", "feature_map_k",
                                       "do_feature_map_norm", "head_k_dim"))


def build_fla_linear_attn_extractor(layer_index: int = 0, causal: bool = False,
                                    eps: float = 1e-10) -> Callable:
    """FINAL-matrix extractor for an fla `LinearAttention` layer (RLA=`elu`, Hedgehog=`hedgehog`,
    t2r, dpfp, ...). Faithful: uses the layer's OWN q/k projections, feature-map modules, the
    `norm_q`/`norm_k` pre-norms (eps 1e-4, fla layer), and `do_feature_map_norm` for the readout
    denominator (eps 1e-10, fla's `normalize_output`). Mirrors `LinearAttention.forward` exactly
    (rearrange to `head_k_dim`, kv-group repeat, feature map, optional q/k norm) — only the causal
    chunk kernel `O=M@V` is replaced by the explicit `M`. `causal` reproduces the kernel's
    lower-triangular structure (the literal operator); off = the unmasked content object."""
    from einops import rearrange, repeat

    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        layer = _find_layer(model, layer_index, _is_fla_linear_attn, "fla LinearAttention layer")
        x = _to_hidden_states(model, layer, eval_batch)
        d = layer.head_k_dim
        q = rearrange(layer.q_proj(x), "... (h d) -> ... h d", d=d)
        g = getattr(layer, "num_kv_groups", 1)
        if g > 1:
            k = repeat(layer.k_proj(x), "... (h d) -> ... (h g) d", d=d, g=g)
        else:
            k = rearrange(layer.k_proj(x), "... (h d) -> ... h d", d=d)
        qf = layer.feature_map_q(q)
        kf = layer.feature_map_k(k)
        if getattr(layer, "norm_q", False):
            qf = qf / (qf.sum(-1, True) + 1e-4)
        if getattr(layer, "norm_k", False):
            kf = kf / (kf.sum(-1, True) + 1e-4)
        W = torch.einsum("blhd,bmhd->bhlm", qf, kf)
        return to_final_attention(W, normalize=layer.do_feature_map_norm, causal=causal, eps=eps)

    return extractor


def _is_mha(m: torch.nn.Module) -> bool:
    return all(hasattr(m, a) for a in ("Wqkv", "head_dim", "num_heads", "inner_attn"))


def build_softmax_attention_extractor(layer_index: int = 0, causal: bool = False) -> Callable:
    """FINAL-matrix extractor for a zoology MHA layer: M = softmax(QKᵀ/sqrt(d) [+ causal mask]).
    Softmax IS the normalization, so M is returned directly (not via to_final_attention). Faithful
    to zoology `SelfAttention` (same 1/sqrt(head_dim) scale, same `bthd,bshd->bhts` einsum).
    `causal=True` reproduces the model's literal lower-triangular operator (zoology always masks);
    `causal=False` is the unmasked softmax pattern (the rank-law content object)."""
    from einops import rearrange

    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        layer = _find_layer(model, layer_index, _is_mha, "MHA layer")
        x = _to_hidden_states(model, layer, eval_batch)
        qkv = rearrange(layer.Wqkv(x), "... (three h d) -> ... three h d", three=3, d=layer.head_dim)
        q, k, _v = qkv.unbind(2)                       # each [B,L,H,head_dim]
        scores = torch.einsum("bthd,bshd->bhts", q, k * (q.shape[-1] ** -0.5))   # [B,H,L,L]
        if causal:
            L = scores.shape[-1]
            scores = scores + torch.triu(
                torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype), 1)
        return torch.softmax(scores, dim=-1, dtype=torch.float32)

    return extractor


def _is_gla(m: torch.nn.Module) -> bool:
    # zoology GatedLinearAttention: has feature_map_fn (fla LinearAttention has feature_map_q/k).
    return (hasattr(m, "feature_map_fn") and hasattr(m, "q_proj") and hasattr(m, "head_k_dim")
            and not hasattr(m, "feature_map_q"))


def build_gla_extractor(layer_index: int = 0, causal: bool = False, eps: float = 1e-5) -> Callable:
    """FINAL content-matrix extractor for a zoology `GatedLinearAttention` layer. UNMASKED object is
    the content gram φ(q)φ(k)ᵀ (φ = the layer's own `feature_map_fn`, default identity); GLA's
    per-key-dim forget-gate decay is a rank-preserving two-sided diagonal (omitted unmasked, exactly
    as the RoLA-GLA reference does), and the output gate / output norm act on O channel-wise (do not
    enter M). GLA is unnormalized → `normalize=False`. The CAUSAL decayed operator is obtained
    separately via the kernel-probe extractor (faithful, no analytic decay re-derivation)."""
    from einops import rearrange, repeat

    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        layer = _find_layer(model, layer_index, _is_gla, "zoology GatedLinearAttention layer")
        x = _to_hidden_states(model, layer, eval_batch)
        d = layer.head_k_dim
        q = rearrange(layer.q_proj(x), "... (h d) -> ... h d", d=d)
        g = getattr(layer, "num_kv_groups", 1)
        if g > 1:
            k = repeat(layer.k_proj(x), "... (h d) -> ... (h g) d", d=d, g=g)
        else:
            k = rearrange(layer.k_proj(x), "... (h d) -> ... h d", d=d)
        if layer.feature_map_fn is not None:
            q = layer.feature_map_fn(q)
            k = layer.feature_map_fn(k)
        W = torch.einsum("blhd,bmhd->bhlm", q, k)
        return to_final_attention(W, normalize=False, causal=causal, eps=eps)

    return extractor


def _is_based(m: torch.nn.Module) -> bool:
    return all(hasattr(m, a) for a in ("proj_q", "proj_k", "feature_map", "feature_dim"))


def build_based_extractor(layer_index: int = 0, causal: bool = False) -> Callable:
    """FINAL-matrix extractor for a zoology `Based` layer: φ = the layer's OWN taylor-exp
    `feature_map`; M = row-normalized φ(q)φ(k)ᵀ (Based's readout is normalized, eps = layer.eps,
    1e-12). Faithful: uses `proj_q`/`proj_k` + the layer's feature_map module verbatim."""
    from einops import rearrange

    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        layer = _find_layer(model, layer_index, _is_based, "zoology Based layer")
        x = _to_hidden_states(model, layer, eval_batch)
        fd = layer.feature_dim
        q = rearrange(layer.proj_q(x), "... (h d) -> ... h d", d=fd)
        k = rearrange(layer.proj_k(x), "... (h d) -> ... h d", d=fd)
        qf = layer.feature_map(q)            # taylor-exp expansion on the last dim
        kf = layer.feature_map(k)
        W = torch.einsum("blhe,bmhe->bhlm", qf, kf)
        return to_final_attention(W, normalize=True, causal=causal, eps=getattr(layer, "eps", 1e-12))

    return extractor


def _is_gdn(m: torch.nn.Module) -> bool:
    # fla GatedDeltaNet: distinctive A_log (decay) + b_proj (beta) + a_proj (gate).
    return all(hasattr(m, a) for a in ("q_proj", "k_proj", "b_proj", "A_log", "a_proj", "head_k_dim"))


def build_gdn_extractor(layer_index: int = 0, causal: bool = True) -> Callable:
    """FINAL-matrix extractor for an fla `GatedDeltaNet` layer, via the kernel-PROBE.

    The (gated) delta rule has NO content gram — its implied token-mixing matrix is
        M_tj = q_t^T ( prod_{j<l<=t} (I - beta_l k_l k_l^T) * decay ) beta_j k_j ,
    but the delta-rule output is LINEAR in v, so O = M @ V. We recover M exactly by running the
    REAL kernel on v = identity (one value-channel per key position): O|_{V=I} = M. Faithful (uses
    `chunk_gated_delta_rule` verbatim, same `use_qk_l2norm_in_kernel=True` as the layer), no
    re-derivation of the WY/delta recurrence. GDN is inherently causal — there is no unmasked
    content object — so `causal` is ignored (always the causal delta operator); GDN has no entry
    in an 'unmasked rank-law' table. Requires CUDA (Triton kernel)."""
    import torch.nn.functional as F
    from einops import rearrange, repeat

    def extractor(model: torch.nn.Module, eval_batch) -> torch.Tensor:
        from fla_rola.ops.gated_delta_rule import chunk_gated_delta_rule
        layer = _find_layer(model, layer_index, _is_gdn, "fla GatedDeltaNet layer")
        if getattr(layer, "use_short_conv", False):
            raise NotImplementedError("GDN extractor: use_short_conv=True not supported (baselines use it off)")
        x = _to_hidden_states(model, layer, eval_batch)
        Bx, Lx, _ = x.shape
        dk = layer.head_k_dim
        q = rearrange(F.silu(layer.q_proj(x)), "... (h d) -> ... h d", d=dk)
        k = rearrange(F.silu(layer.k_proj(x)), "... (h d) -> ... h d", d=dk)
        g = getattr(layer, "num_v_heads", layer.num_heads) // layer.num_heads
        if g > 1:
            q = repeat(q, "... h d -> ... (h g) d", g=g)
            k = repeat(k, "... h d -> ... (h g) d", g=g)
        Hv = q.shape[-2]
        beta = layer.b_proj(x).sigmoid()
        if getattr(layer, "allow_neg_eigval", False):
            beta = beta * 2.0
        gate = -layer.A_log.float().exp() * F.softplus(layer.a_proj(x).float() + layer.dt_bias)
        # v = identity over (position, value-channel): O[b,t,h,j] = M[b,h,t,j]
        v = torch.eye(Lx, device=x.device, dtype=q.dtype)[None, :, None, :].expand(Bx, Lx, Hv, Lx).contiguous()
        o, _ = chunk_gated_delta_rule(q, k, v, g=gate.to(q.dtype), beta=beta.to(q.dtype),
                                      use_qk_l2norm_in_kernel=True, output_final_state=False)
        return o.permute(0, 2, 1, 3).contiguous()      # [B,L,H,L] -> [B,H,L,L] = M (causal delta operator)

    return extractor


def _rola_features(mixer: torch.nn.Module, x: torch.Tensor):
    """Replicate the front half of RoLA.forward + the kernel's feature-map / gate-rescale,
    returning (qf, kf, read_gates, write_gates) in [B,L,H,*] — the exact 4 tensors the
    inline probe SVDs (rola.py:758 / 891)."""
    import torch.nn.functional as F

    B, L, _ = x.shape
    H = mixer.n_heads
    kernel = mixer.kernel
    # VirtualHeadGLAKernel also has `_log_decay` but its forward is the virtual-head (non-shared-gram)
    # path — the W reconstructed below would not match it. Fail loud rather than silently mis-measure.
    if type(kernel).__name__ == "VirtualHeadGLAKernel":
        raise NotImplementedError(
            "rank extraction for VirtualHeadGLAKernel is unsupported (virtual-head, non-shared-gram); "
            "its effective weight differs from the shared-gram form reconstructed here.")

    q = mixer.w_q(x); k = mixer.w_k(x); v = mixer.w_v(x)
    if getattr(mixer, "use_short_conv", False):
        # Causal depthwise conv + SiLU, mirroring rola.py:1070-1077.
        def _cconv(xx, conv):
            xx = xx.transpose(1, 2)
            xx = F.pad(xx, (mixer.conv_pad, 0))
            xx = conv(xx)
            return F.silu(xx.transpose(1, 2))
        q = _cconv(q, mixer.q_conv); k = _cconv(k, mixer.k_conv); v = _cconv(v, mixer.v_conv)
    q = q.view(B, L, H, mixer.proj_qk)
    k = k.view(B, L, H, mixer.proj_qk)
    v = v.view(B, L, H, mixer.d_v)

    write_gates, read_gates = mixer._route(x)                 # rola.py:1055-1061

    # Feature map — kernel-specific. AdditiveKernel exposes _feature_map (rola.py:663);
    # ScalarGLAKernel applies elu+1 inline (rola.py:802). Cover both.
    if hasattr(kernel, "_feature_map"):
        qf, kf = kernel._feature_map(q, k)                    # rola.py:663-667
    else:
        qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0               # rola.py:802 (GLA path)

    # kappa / per_state rescale the read gates by the per-state denominator BEFORE the
    # combine, and the probe SVDs the rescaled gates. Reproduce on the torch (non-Triton)
    # path — CPU-safe, exact. CRITICAL: the GLA kernel uses the DECAYED per-state mass
    # (_rola_gla_perstate_den, needs ld); only the additive (RLA) kernel uses the undecayed
    # den. Mirrors rola.py ScalarGLAKernel.forward:659-679 vs AdditiveKernel.forward:575-594.
    state_norm = getattr(kernel, "state_norm", "global")
    if state_norm in ("kappa", "per_state"):
        if hasattr(kernel, "_log_decay"):                     # ScalarGLAKernel — decayed mass
            from rola import _rola_gla_perstate_den
            ld = kernel._log_decay(x, write_gates)            # rola.py:650-654
            d = _rola_gla_perstate_den(qf, kf, write_gates, ld)   # rola.py:674
        else:                                                 # AdditiveKernel (RLA) — undecayed
            from rola import _rola_perstate_den
            d = _rola_perstate_den(qf, kf, write_gates)       # rola.py:592
        if state_norm == "kappa":
            kap = torch.sigmoid(kernel.w_kappa(x)).view(B, L, mixer.n_heads, 1)  # rola.py:676
            read_gates = read_gates * (d + 1e-5).pow(-kap)
        else:
            read_gates = read_gates / (d + 1e-5)              # rola.py:679
    return qf, kf, read_gates, write_gates


def _find_rola_mixer(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    """Locate the target RoLA mixer. A bare mixer is returned directly; otherwise the
    `layer_index`-th submodule that looks like a RoLA mixer (has w_q + kernel +
    write_router) is returned. Fails loud if none / not enough found."""
    if _is_rola_mixer(model):
        return model
    mixers = [m for m in model.modules() if _is_rola_mixer(m)]
    if not mixers:
        raise ValueError(
            "no RoLA mixer found in the model (looking for a submodule with "
            "w_q + kernel + write_router). Pass a bare mixer or a model containing one."
        )
    if layer_index >= len(mixers):
        raise IndexError(f"layer_index={layer_index} but only {len(mixers)} RoLA mixer(s) found")
    return mixers[layer_index]


def _is_rola_mixer(m: torch.nn.Module) -> bool:
    return all(hasattr(m, a) for a in ("w_q", "w_k", "kernel", "write_router", "_route"))


def _model_dim(mixer: torch.nn.Module) -> int:
    """The mixer's input width — RoLA/MHA/Based expose `d_model`, fla LinearAttention/GLA expose
    `hidden_size`; fall back to the first Linear's in_features. (Generic across the baselines.)"""
    for a in ("d_model", "hidden_size"):
        if hasattr(mixer, a) and isinstance(getattr(mixer, a), int):
            return getattr(mixer, a)
    for m in mixer.modules():
        if isinstance(m, torch.nn.Linear):
            return m.in_features
    raise ValueError(f"could not determine the model dim for {type(mixer).__name__}")


def _to_hidden_states(model: torch.nn.Module, mixer: torch.nn.Module, eval_batch) -> torch.Tensor:
    """Turn `eval_batch` into a [B,L,d] hidden-state tensor on the mixer's device/dtype (d =
    `_model_dim(mixer)`, i.e. d_model or hidden_size).

    - If eval_batch is already a float tensor with last dim == d, use it.
    - Else treat it as token ids and embed via a discovered nn.Embedding whose dim matches d.
      If none is found (e.g. a bare mixer with no embedding), fail loud.
    """
    dm = _model_dim(mixer)
    p = next(mixer.parameters())
    dev, dt = p.device, p.dtype
    if torch.is_tensor(eval_batch) and eval_batch.is_floating_point() \
            and eval_batch.dim() == 3 and eval_batch.shape[-1] == dm:
        return eval_batch.to(device=dev, dtype=dt)
    # token ids -> embed
    if not torch.is_tensor(eval_batch):
        raise TypeError(f"eval_batch must be a tensor (ids [B,L] or hidden [B,L,d]); got {type(eval_batch)}")
    emb = _find_embedding(model, dm)
    if emb is None:
        raise ValueError(
            f"eval_batch looks like token ids but no nn.Embedding with embedding_dim == {dm} "
            "was found. For a bare mixer, pass hidden states [B,L,d] directly."
        )
    return emb(eval_batch.to(device=dev, dtype=torch.long)).to(device=dev, dtype=dt)


def _find_embedding(model: torch.nn.Module, d_model: int):
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding) and m.embedding_dim == d_model:
            return m
    return None
