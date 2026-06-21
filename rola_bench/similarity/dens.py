"""Eager per-state denominators used by the routing/similarity evaluator to reconstruct the
read-gate rescale r̃ = r·(d+ε)^{-κ(x)} (the KAPPA / per_state norm). (Relocated from the deleted
CLA rola.py — fla_rola is the model source of truth; these are eval-only references.)
"""
import torch
import torch.nn.functional as F


def _rola_perstate_den(qf, kf, wg, chunk=64):
    """Per-state denominator d_i^c = sum_{j<=i} w_j^c (φ(q_i)·φ(k_j)) — the mass each state
    contributes to token i's global partition function. [B,L,H,*] in → [B,L,H,nc] out."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
    q, k, w = fold(qf), fold(kf), fold(wg)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); w = F.pad(w, (0, 0, 0, pad))
    Lp = L + pad; n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk); kc = k.view(B * H, n, chunk, dqk); wc = w.view(B * H, n, chunk, nc)
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.einsum('bnij,bnjc->bnic', G * causal, wc)
    KZ = torch.einsum('bnjc,bnjd->bncd', wc, kc)                  # per-chunk z increments
    Z = torch.cumsum(KZ, dim=1) - KZ                              # exclusive prefix
    inter = torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)                # [B,L,H,nc]


def _rola_gla_perstate_den(qf, kf, wg, ld, chunk=64):
    """Per-state denominator UNDER per-state log-decay ld:
    d_i^c = sum_{j<=i} (φq_i·φk_j) w_j^c e^{Λ_ic-Λ_jc}, Λ = inclusive cumsum(ld).
    [B,L,H,*] in → [B,L,H,nc] out (decay absorbed chunk-locally, decayed cross-chunk carry)."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
    q, k, w, g = fold(qf), fold(kf), fold(wg), fold(ld)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        w = F.pad(w, (0, 0, 0, pad)); g = F.pad(g, (0, 0, 0, pad))
    Lp = L + pad; n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk); kc = k.view(B * H, n, chunk, dqk)
    wc = w.view(B * H, n, chunk, nc); gc = g.view(B * H, n, chunk, nc)
    a = gc.cumsum(2)                                              # chunk-local Λ [b,n,t,c]
    Lam = a[:, :, -1, :]                                         # chunk decay totals [b,n,c]
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.exp(a) * torch.einsum('bnij,bnjc->bnic', G * causal, wc * torch.exp(-a))
    w_end = wc * torch.exp(Lam.unsqueeze(2) - a)                  # writes decayed to chunk end
    KZ = torch.einsum('bnjc,bnjd->bncd', w_end, kc)              # per-chunk carry increments
    acc = torch.zeros(B * H, nc, dqk, device=q.device, dtype=q.dtype)
    Zs = []
    for i in range(n):                                           # decayed exclusive prefix
        Zs.append(acc)
        acc = torch.exp(Lam[:, i]).unsqueeze(-1) * acc + KZ[:, i]
    Z = torch.stack(Zs, 1)                                       # [b,n,c,d] state at chunk start
    inter = torch.exp(a) * torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)               # [B,L,H,nc]
