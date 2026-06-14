"""Generic, checkpoint-driven post-hoc RANK evaluator (rola-bench task #85).

This is the standalone successor to the rank diagnostic that currently lives INLINE in
the model library `rola.py` (gated by env ROLA_MEASURE_RANK, emitting `RANK_JSON {...}`
lines from the kernel forward). Task #84 will remove that inline probe; this module
replaces it.

Why a separate module: the inline probe can only run during a live forward pass of a
RoLA kernel. We want to (a) measure rank POST-HOC on a saved checkpoint, (b) measure it
for ANY model — softmax attention, RoLA, or any linear-attention variant — not just
RoLA, and (c) keep it decoupled from the training/kernel runtime so it survives #84.

Genericity is achieved with an `attention_extractor` callback: the caller provides a fn
`(model, batch) -> attn[B,H,L,L]` returning whatever the model's (implied) attention
matrix is. A softmax model returns its softmax weights; a linear-attention / RoLA model
returns the implied effective weight W = (Q Kᵀ) ∘ (read · writeᵀ). The metric math then
treats every model identically.

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
    max_b: int = 64,
    max_l: int = 8192,
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
    for bi in range(b):
        for hi in range(H):
            W = attn[bi, hi, :l, :l].float()
            svs.append(torch.linalg.svdvals(W))               # rola.py:378
    sv = torch.stack(svs)                                     # [b*H, l]   rola.py:381

    out = _rank_stats_from_sv(sv, l, tols)                    # rola.py:382

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
    max_b: int = 64,
    max_l: int = 8192,
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
def build_rola_attention_extractor(layer_index: int = 0) -> Callable:
    """Return an `attention_extractor(model, eval_batch) -> W[B,H,L,L]` for a RoLA model.

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
        return rola_attn_from_features(qf, kf, read_gates, write_gates)

    return extractor


def _rola_features(mixer: torch.nn.Module, x: torch.Tensor):
    """Replicate the front half of RoLA.forward + the kernel's feature-map / gate-rescale,
    returning (qf, kf, read_gates, write_gates) in [B,L,H,*] — the exact 4 tensors the
    inline probe SVDs (rola.py:758 / 891)."""
    import torch.nn.functional as F

    B, L, _ = x.shape
    H = mixer.n_heads
    kernel = mixer.kernel

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
    # combine, and the probe SVDs the rescaled gates (rola.py:758 sees the rescaled
    # read_gates). Reproduce on the torch (non-Triton) path — CPU-safe, exact.
    state_norm = getattr(kernel, "state_norm", "global")
    if state_norm in ("kappa", "per_state"):
        from rola import _rola_perstate_den
        d = _rola_perstate_den(qf, kf, write_gates)           # rola.py:688
        if state_norm == "kappa":
            kap = torch.sigmoid(kernel.w_kappa(x)).view(B, L, mixer.n_heads, 1)  # rola.py:690-691
            read_gates = read_gates * (d + 1e-5).pow(-kap)
        else:
            read_gates = read_gates / (d + 1e-5)              # rola.py:693
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


def _to_hidden_states(model: torch.nn.Module, mixer: torch.nn.Module, eval_batch) -> torch.Tensor:
    """Turn `eval_batch` into a [B,L,d_model] hidden-state tensor on the mixer's device/dtype.

    - If eval_batch is already a float tensor with last dim == mixer.d_model, use it.
    - Else treat it as token ids and embed via a discovered nn.Embedding whose dim matches
      d_model. If none is found (e.g. a bare mixer with no embedding), fail loud — the
      caller should pass hidden states directly for a bare mixer.
    """
    p = next(mixer.parameters())
    dev, dt = p.device, p.dtype
    if torch.is_tensor(eval_batch) and eval_batch.is_floating_point() \
            and eval_batch.dim() == 3 and eval_batch.shape[-1] == mixer.d_model:
        return eval_batch.to(device=dev, dtype=dt)
    # token ids -> embed
    if not torch.is_tensor(eval_batch):
        raise TypeError(f"eval_batch must be a tensor (ids [B,L] or hidden [B,L,d_model]); got {type(eval_batch)}")
    emb = _find_embedding(model, mixer.d_model)
    if emb is None:
        raise ValueError(
            "eval_batch looks like token ids but no nn.Embedding with embedding_dim == "
            f"{mixer.d_model} was found. For a bare mixer, pass hidden states "
            "[B,L,d_model] directly."
        )
    return emb(eval_batch.to(device=dev, dtype=torch.long)).to(device=dev, dtype=dt)


def _find_embedding(model: torch.nn.Module, d_model: int):
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding) and m.embedding_dim == d_model:
            return m
    return None
