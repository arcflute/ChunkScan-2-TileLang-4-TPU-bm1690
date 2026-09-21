"""Logical inputs and compact task-major ABI helpers for P9.2."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

from tpu_demo.mamba2_chunk_scan.chunk_scan_multicore_s1_p9 import (
    CHUNK_SIZE,
    DSTATE,
    HEADDIM,
    OUTPUT_GUARD_ROWS,
    ChunkScanMulticoreConfig,
    physical_shapes,
)


Inputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
SENTINEL = -777.0


def _random_tensor(
    shape: Iterable[int],
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    value = (torch.rand(tuple(shape), generator=generator) * 2.0 - 1.0)
    return (value * scale).to(torch.float16).contiguous()


def make_inputs(
    config: ChunkScanMulticoreConfig,
    seed: int,
) -> Inputs:
    """Create deterministic logical Mamba2 ChunkScan inputs."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    cb = _random_tensor(
        (
            config.batch,
            config.nchunks,
            config.ngroups,
            CHUNK_SIZE,
            CHUNK_SIZE,
        ),
        generator,
        0.20,
    )
    x = _random_tensor(
        (config.batch, config.seqlen, config.nheads, HEADDIM),
        generator,
        0.20,
    )
    dt32 = (
        torch.rand(
            (
                config.batch,
                config.nheads,
                config.nchunks,
                CHUNK_SIZE,
            ),
            generator=generator,
        )
        * 0.10
        + 0.05
    )
    dt = dt32.to(torch.float16).contiguous()
    a = -(
        torch.rand((config.nheads,), generator=generator) * 0.15
        + 0.05
    )
    dA_cumsum = torch.cumsum(
        dt.float() * a.view(1, config.nheads, 1, 1),
        dim=-1,
    ).to(torch.float16).contiguous()
    C = _random_tensor(
        (
            config.batch,
            config.seqlen,
            config.ngroups,
            DSTATE,
        ),
        generator,
        0.10,
    )
    prev_states = _random_tensor(
        (
            config.batch,
            config.nchunks,
            config.nheads,
            HEADDIM,
            DSTATE,
        ),
        generator,
        0.10,
    )
    D = torch.linspace(
        0.125,
        0.25,
        config.nheads,
        dtype=torch.float32,
    ).to(torch.float16).contiguous()
    return cb, x, dt, dA_cumsum, C, prev_states, D


def variant_inputs(base: Inputs, variant: str) -> Inputs:
    """Create the established state/scan/residual/all-terms variants."""

    cb, x, dt, dA, C, states, D = (tensor.clone() for tensor in base)
    if variant == "state_only":
        cb.zero_()
        x.zero_()
        D.zero_()
    elif variant == "scan_only":
        states.zero_()
        D.zero_()
    elif variant == "residual_only":
        cb.zero_()
        states.zero_()
    elif variant != "all_terms":
        raise ValueError(variant)
    return cb, x, dt, dA, C, states, D


def poison_upper_triangle(inputs: Inputs) -> Inputs:
    """Replace only causally invisible CB entries with a large value."""

    cb, *remaining = (tensor.clone() for tensor in inputs)
    upper = torch.triu(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool),
        diagonal=1,
    ).view(1, 1, 1, CHUNK_SIZE, CHUNK_SIZE)
    cb = torch.where(upper, torch.full_like(cb, 8.0), cb).contiguous()
    return (cb, *remaining)


def pack_inputs(
    config: ChunkScanMulticoreConfig,
    inputs: Inputs,
) -> Tuple[torch.Tensor, ...]:
    """Pack logical tensors in `(batch, chunk, head)` task order."""

    cb, x, dt, dA, C, states, D = inputs
    batch = config.batch
    chunks = config.nchunks
    heads = config.nheads

    cb_task = (
        cb[:, :, 0]
        .unsqueeze(2)
        .expand(batch, chunks, heads, CHUNK_SIZE, CHUNK_SIZE)
        .contiguous()
        .view(-1, CHUNK_SIZE)
    )
    x_task = (
        x.view(batch, chunks, CHUNK_SIZE, heads, HEADDIM)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(-1, HEADDIM)
    )
    dt_task = dt.permute(0, 2, 1, 3).contiguous().view(-1, CHUNK_SIZE)
    dA_task = dA.permute(0, 2, 1, 3).contiguous().view(-1, CHUNK_SIZE)
    C_task = (
        C[:, :, 0]
        .view(batch, chunks, CHUNK_SIZE, DSTATE)
        .unsqueeze(2)
        .expand(batch, chunks, heads, CHUNK_SIZE, DSTATE)
        .contiguous()
        .view(-1, DSTATE)
    )
    states_task = states.contiguous().view(-1, DSTATE)
    D_task = (
        D.view(1, 1, heads, 1)
        .expand(batch, chunks, heads, 1)
        .contiguous()
        .view(-1, 1)
    )

    packed = (
        cb_task,
        x_task,
        dt_task,
        dA_task,
        C_task,
        states_task,
        D_task,
    )
    shapes = physical_shapes(config)
    names = (
        "cb",
        "x",
        "dt",
        "dA_cumsum",
        "C",
        "prev_states",
        "D",
    )
    for name, tensor in zip(names, packed):
        if tuple(tensor.shape) != shapes[name]:
            raise AssertionError(
                f"{name}: packed shape {tuple(tensor.shape)} "
                f"!= {shapes[name]}"
            )
        if tensor.dtype != torch.float16 or not tensor.is_contiguous():
            raise AssertionError(f"{name}: invalid packed dtype/layout")
    return packed


def make_physical_output(
    config: ChunkScanMulticoreConfig,
) -> torch.Tensor:
    return torch.full(
        physical_shapes(config)["out"],
        SENTINEL,
        dtype=torch.float16,
    )


def guards_unchanged(output: torch.Tensor) -> bool:
    return bool(
        torch.all(output[:OUTPUT_GUARD_ROWS] == SENTINEL).item()
        and torch.all(output[-OUTPUT_GUARD_ROWS:] == SENTINEL).item()
    )


def payload_has_no_sentinel(output: torch.Tensor) -> bool:
    payload = output[OUTPUT_GUARD_ROWS:-OUTPUT_GUARD_ROWS]
    return bool(torch.all(payload != SENTINEL).item())


def unpack_output(
    config: ChunkScanMulticoreConfig,
    physical: torch.Tensor,
) -> torch.Tensor:
    """Restore logical `[B,S,H,P]` output from guarded task-major rows."""

    expected_shape = physical_shapes(config)["out"]
    if tuple(physical.shape) != expected_shape:
        raise ValueError(
            f"physical output shape {tuple(physical.shape)} != "
            f"{expected_shape}"
        )
    payload = physical[OUTPUT_GUARD_ROWS:-OUTPUT_GUARD_ROWS]
    return (
        payload.view(
            config.batch,
            config.nchunks,
            config.nheads,
            CHUNK_SIZE,
            HEADDIM,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            config.batch,
            config.seqlen,
            config.nheads,
            HEADDIM,
        )
        .contiguous()
    )


def logical_task_max_abs(
    config: ChunkScanMulticoreConfig,
    output: torch.Tensor,
) -> torch.Tensor:
    """Return one maximum absolute value for every `(B,Ck,H)` task."""

    task_major = (
        output.view(
            config.batch,
            config.nchunks,
            CHUNK_SIZE,
            config.nheads,
            HEADDIM,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(config.task_count, CHUNK_SIZE, HEADDIM)
    )
    return task_major.float().abs().amax(dim=(1, 2))


__all__ = [
    "Inputs",
    "SENTINEL",
    "guards_unchanged",
    "logical_task_max_abs",
    "make_inputs",
    "make_physical_output",
    "pack_inputs",
    "payload_has_no_sentinel",
    "poison_upper_triangle",
    "unpack_output",
    "variant_inputs",
]
