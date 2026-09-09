"""Independent CPU truth implementations for standalone Mamba2 ChunkScan.

This module intentionally imports neither TileLang, Triton, nor Mamba.  It
accepts the exact seven-input FP16 contract frozen from TileLang v0.1.5 and
computes the result primarily in FP32 before converting the final output to
FP16.
"""

from typing import Dict, NamedTuple

import torch


class ChunkScanShape(NamedTuple):
    batch: int
    seqlen: int
    nchunks: int
    chunk_size: int
    ngroups: int
    nheads: int
    headdim: int
    dstate: int


def _validate_inputs(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> ChunkScanShape:
    tensors = {
        "cb": cb,
        "x": x,
        "dt": dt,
        "dA_cumsum": dA_cumsum,
        "C": C,
        "prev_states": prev_states,
        "D": D,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor, got device {tensor.device}")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must use compact C-contiguous layout")

    expected_ranks = {
        "cb": 5,
        "x": 4,
        "dt": 4,
        "dA_cumsum": 4,
        "C": 4,
        "prev_states": 5,
    }
    for name, rank in expected_ranks.items():
        if tensors[name].ndim != rank:
            raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensors[name].shape)}")

    batch, nchunks, ngroups, chunk_q, chunk_s = cb.shape
    if chunk_q != chunk_s:
        raise ValueError(f"cb's two chunk axes must match, got {chunk_q} and {chunk_s}")
    chunk_size = chunk_q
    x_batch, seqlen, nheads, headdim = x.shape
    dt_batch, dt_heads, dt_chunks, dt_chunk_size = dt.shape

    dimensions = {
        "batch": batch,
        "seqlen": seqlen,
        "nchunks": nchunks,
        "chunk_size": chunk_size,
        "ngroups": ngroups,
        "nheads": nheads,
        "headdim": headdim,
    }
    non_positive = {name: value for name, value in dimensions.items() if value <= 0}
    if non_positive:
        raise ValueError(f"all dimensions must be positive, got {non_positive}")

    if x_batch != batch:
        raise ValueError(f"x batch must be {batch}, got {x_batch}")
    if (dt_batch, dt_heads, dt_chunks, dt_chunk_size) != (
        batch,
        nheads,
        nchunks,
        chunk_size,
    ):
        raise ValueError(
            "dt must have shape "
            f"{(batch, nheads, nchunks, chunk_size)}, got {tuple(dt.shape)}"
        )
    if seqlen != nchunks * chunk_size:
        raise ValueError(
            "seqlen must equal nchunks * chunk_size; "
            f"got {seqlen} != {nchunks} * {chunk_size}"
        )
    if nheads % ngroups != 0:
        raise ValueError(f"nheads must be divisible by ngroups; got {nheads} and {ngroups}")

    if dA_cumsum.shape != dt.shape:
        raise ValueError(
            f"dA_cumsum must have shape {tuple(dt.shape)}, got {tuple(dA_cumsum.shape)}"
        )
    if C.shape[:3] != (batch, seqlen, ngroups):
        raise ValueError(
            "C must begin with shape "
            f"{(batch, seqlen, ngroups)}, got {tuple(C.shape)}"
        )
    dstate = C.shape[3]
    if dstate <= 0:
        raise ValueError(f"dstate must be positive, got {dstate}")
    expected_states = (batch, nchunks, nheads, headdim, dstate)
    if prev_states.shape != expected_states:
        raise ValueError(
            f"prev_states must have shape {expected_states}, got {tuple(prev_states.shape)}"
        )
    if D.shape != (nheads,):
        raise ValueError(
            "D must have the TileLang paper-kernel shape "
            f"{(nheads,)}, got {tuple(D.shape)}"
        )

    return ChunkScanShape(
        batch=batch,
        seqlen=seqlen,
        nchunks=nchunks,
        chunk_size=chunk_size,
        ngroups=ngroups,
        nheads=nheads,
        headdim=headdim,
        dstate=dstate,
    )


def chunk_scan_reference_loop(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Straightforward loop oracle that mirrors the scalar mathematical sum."""

    shape = _validate_inputs(cb, x, dt, dA_cumsum, C, prev_states, D)
    cb32 = cb.float()
    x32 = x.float()
    dt32 = dt.float()
    dA32 = dA_cumsum.float()
    C32 = C.float()
    states32 = prev_states.float()
    D32 = D.float()
    output = torch.zeros(
        (shape.batch, shape.seqlen, shape.nheads, shape.headdim),
        dtype=torch.float32,
        device="cpu",
    )
    heads_per_group = shape.nheads // shape.ngroups

    for b in range(shape.batch):
        for c in range(shape.nchunks):
            for q in range(shape.chunk_size):
                out_position = c * shape.chunk_size + q
                for h in range(shape.nheads):
                    group = h // heads_per_group
                    a_q = dA32[b, h, c, q]

                    # Term 1: contribution from the state entering this chunk.
                    state_term = torch.zeros(shape.headdim, dtype=torch.float32)
                    state_decay = torch.exp(a_q)
                    for p in range(shape.headdim):
                        state_sum = torch.zeros((), dtype=torch.float32)
                        for n in range(shape.dstate):
                            state_sum += (
                                C32[b, out_position, group, n]
                                * states32[b, c, h, p, n]
                            )
                        state_term[p] = state_decay * state_sum

                    # Term 2: lower-triangular scan within this same chunk.
                    scan_term = torch.zeros(shape.headdim, dtype=torch.float32)
                    for s in range(q + 1):
                        in_position = c * shape.chunk_size + s
                        weight = (
                            cb32[b, c, group, q, s]
                            * torch.exp(a_q - dA32[b, h, c, s])
                            * dt32[b, h, c, s]
                        )
                        for p in range(shape.headdim):
                            scan_term[p] += weight * x32[b, in_position, h, p]

                    # Term 3: per-head D*x residual at the output position.
                    residual_term = D32[h] * x32[b, out_position, h, :]
                    output[b, out_position, h, :] = state_term + scan_term + residual_term

    return output.to(dtype=x.dtype).contiguous()


def chunk_scan_reference(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Vectorized CPU oracle for the frozen standalone `_chunk_scan_fwd` ABI."""

    shape = _validate_inputs(cb, x, dt, dA_cumsum, C, prev_states, D)
    cb32 = cb.float()
    x32 = x.float()
    dt32 = dt.float()
    dA32 = dA_cumsum.float()
    C32 = C.float()
    states32 = prev_states.float()
    D32 = D.float()

    heads_per_group = shape.nheads // shape.ngroups
    head_to_group = torch.arange(shape.nheads, device="cpu") // heads_per_group

    # C_by_head: [B,S,H,N], C_chunks: [B,Ck,L,H,N]
    C_by_head = C32[:, :, head_to_group, :]
    C_chunks = C_by_head.reshape(
        shape.batch, shape.nchunks, shape.chunk_size, shape.nheads, shape.dstate
    )
    state_term = torch.einsum("bcqhn,bchpn->bcqhp", C_chunks, states32)
    state_decay = torch.exp(dA32.permute(0, 2, 3, 1)).unsqueeze(-1)
    state_term = state_term * state_decay

    # scores: [B,Ck,H,q,s].  Masking is applied explicitly before contraction.
    cb_by_head = cb32[:, :, head_to_group, :, :]
    decay = torch.exp(dA32[..., :, None] - dA32[..., None, :])
    scores = cb_by_head * decay.permute(0, 2, 1, 3, 4)
    scores = scores * dt32.permute(0, 2, 1, 3).unsqueeze(-2)
    causal_mask = torch.tril(
        torch.ones((shape.chunk_size, shape.chunk_size), dtype=torch.bool, device="cpu")
    )
    scores = scores.masked_fill(~causal_mask.view(1, 1, 1, shape.chunk_size, shape.chunk_size), 0.0)
    x_chunks = x32.reshape(
        shape.batch, shape.nchunks, shape.chunk_size, shape.nheads, shape.headdim
    )
    scan_term = torch.einsum("bchqs,bcshp->bcqhp", scores, x_chunks)

    residual_term = x_chunks * D32.view(1, 1, 1, shape.nheads, 1)
    output32 = state_term + scan_term + residual_term
    return output32.reshape(shape.batch, shape.seqlen, shape.nheads, shape.headdim).to(
        dtype=x.dtype
    ).contiguous()


def comparison_stats(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    """Return explicit numerical diagnostics for two same-shaped tensors."""

    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    actual32 = actual.float()
    expected32 = expected.float()
    abs_diff = (actual32 - expected32).abs()
    relative = abs_diff / expected32.abs().clamp_min(1e-12)
    return {
        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "max_rel_diff": float(relative.max().item()) if relative.numel() else 0.0,
        "nan_count": int(torch.isnan(actual32).sum().item()),
        "inf_count": int(torch.isinf(actual32).sum().item()),
    }


__all__ = [
    "chunk_scan_reference",
    "chunk_scan_reference_loop",
    "comparison_stats",
]
