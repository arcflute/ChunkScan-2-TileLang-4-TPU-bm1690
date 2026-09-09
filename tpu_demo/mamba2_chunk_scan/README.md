# Standalone Mamba2 ChunkScan on BM1690

This directory starts the BM1690 reproduction of the exact standalone Mamba2
`_chunk_scan_fwd` operator evaluated by PipeThreader.  The fixed source of truth
is TileLang v0.1.5's
`examples/linear_attention/example_mamba_chunk_scan.py`, pinned at commit
`a32009bf1e314b514c07389123648ba19009f3a5`.  The contract is cross-checked
against `state-spaces/mamba`'s `mamba_ssm/ops/triton/ssd_chunk_scan.py`.

## Scope boundary

The standalone operator consumes already materialized `cb`, `dA_cumsum`, and
`prev_states`.  It computes all three terms of the published ChunkScan kernel:

1. the contribution from the state entering each chunk;
2. the causal scan within each chunk;
3. the `D * x` residual.

It is not `mamba_chunk_scan_combined`.  This project stage intentionally does
not compute ChunkCumsum, ChunkState, StatePassing, `C @ B^T`, final state, a
`z`/SiLU gate, `seq_idx`, variable-length sequences, or tail chunks.

## Frozen input/output overview

With `B` batch elements, `Ck` chunks, chunk size `L`, `G` groups, `H` heads,
head dimension `P`, state dimension `N`, and `S = Ck * L`:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `cb` | `[B, Ck, G, L, L]` | `torch.float16` |
| `x` | `[B, S, H, P]` | `torch.float16` |
| `dt` | `[B, H, Ck, L]` | `torch.float16` |
| `dA_cumsum` | `[B, H, Ck, L]` | `torch.float16` |
| `C` | `[B, S, G, N]` | `torch.float16` |
| `prev_states` | `[B, Ck, H, P, N]` | `torch.float16` |
| `D` | `[H]` | `torch.float16` |
| output | `[B, S, H, P]` | `torch.float16` |

All external tensors are frozen as compact C-contiguous tensors for this
reproduction.  See `source_contract.md` for the exact indexing, strides, math,
and differences from the broader official Triton wrapper.

## Stages

- **A0 (this directory now):** freeze the source contract and provide an
  independent CPU/PyTorch loop oracle, a vectorized equivalent, and unit tests.
- **A1 (not implemented yet):** write a serial BM1690 TPU kernel matching this
  exact contract.
- **A2 (not implemented yet):** validate A1 with cmodel, including numerical
  results, generated source, addresses, layouts, and LMEM use.

A0 has no TileLang, Triton, Mamba package, GPU, cmodel, or compiled-extension
dependency.  It uses PyTorch only for CPU tensor operations.

## Run the CPU tests

From the repository root:

```bash
/root/autodl-tmp/pipethreader-envs/chunkscan-a0-py310/bin/python \
  -m pytest -v tpu_demo/mamba2_chunk_scan/test_reference.py
```

No TPU or cmodel validation has been performed in A0.  All BM1690 performance
and hardware-overlap conclusions remain **UNKNOWN**.
