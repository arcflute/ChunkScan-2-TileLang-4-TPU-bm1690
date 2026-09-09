"""CPU-only tests for the standalone Mamba2 ChunkScan oracle."""

from typing import Dict, Tuple

import pytest
import torch

from tpu_demo.mamba2_chunk_scan.configs import CPU_TEST_CONFIGS, TPU_SMOKE_CONFIG
from tpu_demo.mamba2_chunk_scan.reference import (
    chunk_scan_reference,
    chunk_scan_reference_loop,
    comparison_stats,
)


FP32_ORACLE_ATOL = 1e-5
FP32_ORACLE_RTOL = 1e-5
FP16_OUTPUT_ATOL = 1e-2
FP16_OUTPUT_RTOL = 1e-2


Inputs = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _random_tensor(shape: Tuple[int, ...], generator: torch.Generator, scale: float) -> torch.Tensor:
    value = (torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * scale
    return value.to(torch.float16).contiguous()


def _make_inputs(config: Dict[str, int], seed: int = 20250909) -> Inputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = config["batch"]
    seqlen = config["seqlen"]
    chunk_size = config["chunk_size"]
    nchunks = config.get("nchunks", seqlen // chunk_size)
    ngroups = config["ngroups"]
    nheads = config["nheads"]
    headdim = config["headdim"]
    dstate = config["dstate"]

    cb = _random_tensor((batch, nchunks, ngroups, chunk_size, chunk_size), generator, 0.08)
    x = _random_tensor((batch, seqlen, nheads, headdim), generator, 0.12)
    dt32 = torch.rand((batch, nheads, nchunks, chunk_size), generator=generator) * 0.04 + 0.01
    dt = dt32.to(torch.float16).contiguous()
    # Physical Mamba2-like construction: dt > 0, A < 0, and the cumulative
    # value resets at each chunk because cumsum is over the last axis only.
    A = -(torch.rand((nheads,), generator=generator) * 0.15 + 0.05)
    dA_cumsum = torch.cumsum(dt.float() * A.view(1, nheads, 1, 1), dim=-1)
    dA_cumsum = dA_cumsum.to(torch.float16).contiguous()
    C = _random_tensor((batch, seqlen, ngroups, dstate), generator, 0.10)
    prev_states = _random_tensor((batch, nchunks, nheads, headdim, dstate), generator, 0.10)
    D = _random_tensor((nheads,), generator, 0.20)
    return cb, x, dt, dA_cumsum, C, prev_states, D


def _replace(inputs: Inputs, **updates: torch.Tensor) -> Inputs:
    names = ("cb", "x", "dt", "dA_cumsum", "C", "prev_states", "D")
    values = dict(zip(names, inputs))
    values.update(updates)
    return tuple(values[name] for name in names)  # type: ignore[return-value]


def _assert_oracles_match(inputs: Inputs) -> None:
    loop = chunk_scan_reference_loop(*inputs)
    vectorized = chunk_scan_reference(*inputs)
    torch.testing.assert_close(
        vectorized,
        loop,
        atol=FP32_ORACLE_ATOL,
        rtol=FP32_ORACLE_RTOL,
    )
    stats = comparison_stats(vectorized, loop)
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0


def test_loop_and_vectorized_oracles_match_first() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=1)
    _assert_oracles_match(inputs)


def test_state_only() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=2)
    cb, x, _, _, _, _, D = inputs
    inputs = _replace(inputs, cb=torch.zeros_like(cb), x=torch.zeros_like(x), D=torch.zeros_like(D))
    _assert_oracles_match(inputs)
    assert torch.count_nonzero(chunk_scan_reference(*inputs)) > 0


def test_scan_only() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=3)
    *_, prev_states, D = inputs
    inputs = _replace(inputs, prev_states=torch.zeros_like(prev_states), D=torch.zeros_like(D))
    _assert_oracles_match(inputs)
    assert torch.count_nonzero(chunk_scan_reference(*inputs)) > 0


def test_residual_only() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=4)
    cb, x, _, _, _, prev_states, _ = inputs
    inputs = _replace(inputs, cb=torch.zeros_like(cb), prev_states=torch.zeros_like(prev_states))
    actual = chunk_scan_reference(*inputs)
    expected = (x.float() * inputs[-1].float().view(1, 1, -1, 1)).to(torch.float16)
    torch.testing.assert_close(actual, expected, atol=FP16_OUTPUT_ATOL, rtol=FP16_OUTPUT_RTOL)
    _assert_oracles_match(inputs)


def test_all_terms_and_numerical_statistics() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=5)
    loop = chunk_scan_reference_loop(*inputs)
    vectorized = chunk_scan_reference(*inputs)
    torch.testing.assert_close(
        vectorized, loop, atol=FP32_ORACLE_ATOL, rtol=FP32_ORACLE_RTOL
    )
    stats = comparison_stats(vectorized, loop)
    assert stats == {
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "nan_count": 0,
        "inf_count": 0,
    }


def test_causal_mask_ignores_large_upper_triangle() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=6)
    cb = inputs[0]
    chunk_size = cb.shape[-1]
    upper = torch.triu(torch.ones((chunk_size, chunk_size), dtype=torch.bool), diagonal=1)
    poisoned_cb = torch.where(upper.view(1, 1, 1, chunk_size, chunk_size), 1000.0, cb.float())
    poisoned_inputs = _replace(inputs, cb=poisoned_cb.to(torch.float16).contiguous())
    expected = chunk_scan_reference(*inputs)
    actual = chunk_scan_reference(*poisoned_inputs)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_two_chunks_use_distinct_prev_states() -> None:
    config = CPU_TEST_CONFIGS["scalar_two_chunk"]
    inputs = _make_inputs(config, seed=7)
    cb, x, dt, dA, C, prev_states, D = inputs
    cb.zero_()
    x.zero_()
    dA.zero_()
    C.fill_(1.0)
    prev_states[0, 0, 0, 0, 0] = 2.0
    prev_states[0, 1, 0, 0, 0] = 7.0
    D.zero_()
    actual = chunk_scan_reference(cb, x, dt, dA, C, prev_states, D)
    expected = torch.tensor([2.0, 7.0], dtype=torch.float16).view(1, 2, 1, 1)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_head_group_mapping() -> None:
    config = {
        "batch": 1,
        "seqlen": 1,
        "chunk_size": 1,
        "ngroups": 2,
        "nheads": 4,
        "headdim": 1,
        "dstate": 1,
    }
    cb, x, dt, dA, C, prev_states, D = _make_inputs(config, seed=8)
    cb.zero_()
    x.zero_()
    dA.zero_()
    C[0, 0, 0, 0] = 3.0
    C[0, 0, 1, 0] = 11.0
    prev_states.fill_(1.0)
    D.zero_()
    actual = chunk_scan_reference(cb, x, dt, dA, C, prev_states, D)
    expected = torch.tensor([3.0, 3.0, 11.0, 11.0], dtype=torch.float16).view(1, 1, 4, 1)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_zero_inputs_produce_zero_output() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=9)
    zeros = tuple(torch.zeros_like(tensor) for tensor in inputs)
    actual = chunk_scan_reference(*zeros)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert not torch.isnan(actual).any()
    assert not torch.isinf(actual).any()


def test_dtype_and_shape_with_project_tpu_smoke_shape() -> None:
    inputs = _make_inputs(TPU_SMOKE_CONFIG, seed=10)
    actual = chunk_scan_reference(*inputs)
    assert actual.shape == (
        TPU_SMOKE_CONFIG["batch"],
        TPU_SMOKE_CONFIG["seqlen"],
        TPU_SMOKE_CONFIG["nheads"],
        TPU_SMOKE_CONFIG["headdim"],
    )
    assert actual.dtype == torch.float16
    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    stats = comparison_stats(actual, actual)
    assert stats["max_abs_diff"] <= FP16_OUTPUT_ATOL
    assert stats["max_rel_diff"] <= FP16_OUTPUT_RTOL
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0


def test_invalid_sequence_chunk_relation() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=11)
    x = inputs[1][:, :-1].contiguous()
    C = inputs[4][:, :-1].contiguous()
    invalid = _replace(inputs, x=x, C=C)
    with pytest.raises(ValueError, match=r"seqlen must equal nchunks \* chunk_size"):
        chunk_scan_reference(*invalid)


def test_invalid_head_group_relation() -> None:
    config = {
        "batch": 1,
        "seqlen": 2,
        "chunk_size": 1,
        "ngroups": 2,
        "nheads": 3,
        "headdim": 1,
        "dstate": 1,
    }
    inputs = _make_inputs(config, seed=12)
    with pytest.raises(ValueError, match="nheads must be divisible by ngroups"):
        chunk_scan_reference(*inputs)


def test_invalid_D_shape() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=13)
    nheads = inputs[1].shape[2]
    headdim = inputs[1].shape[3]
    invalid_D = torch.zeros((nheads, headdim), dtype=torch.float16)
    invalid = _replace(inputs, D=invalid_D)
    with pytest.raises(ValueError, match="D must have the TileLang paper-kernel shape"):
        chunk_scan_reference(*invalid)


def test_invalid_tensor_shape_and_dtype_report_clear_errors() -> None:
    inputs = _make_inputs(CPU_TEST_CONFIGS["two_chunk_unit"], seed=14)
    bad_states = inputs[5][..., :-1].contiguous()
    with pytest.raises(ValueError, match="prev_states must have shape"):
        chunk_scan_reference(*_replace(inputs, prev_states=bad_states))

    with pytest.raises(TypeError, match="x must have dtype torch.float16"):
        chunk_scan_reference(*_replace(inputs, x=inputs[1].float().contiguous()))
