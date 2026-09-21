# Standalone Mamba2 ChunkScan for the BM1690 cmodel and hardware

This directory starts the BM1690 reproduction of the exact standalone Mamba2
`_chunk_scan_fwd` operator evaluated by PipeThreader.  The fixed source of truth
is TileLang v0.1.5's
`examples/linear_attention/example_mamba_chunk_scan.py`, pinned at commit
`a32009bf1e314b514c07389123648ba19009f3a5`.  The contract is cross-checked
against `state-spaces/mamba`'s `mamba_ssm/ops/triton/ssd_chunk_scan.py`.

The P0-P7 reproduction was closed on the BM1690 CPU cmodel.  A separate
single-core BM1690 PCIe route has since compiled and run S0, S1, S2, S3,
and P6 on physical hardware.  The goals are functional correctness and
auditable pipeline code structure, not latency, throughput, speedup,
hardware utilization, or proof of physical engine overlap.

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

The authoritative staged plan is in `IMPLEMENTATION_PLAN.md`.  The frozen P3
task graph and reduction axis are documented in `stask_mapping.md`; the P3
closure procedure is in `P3_IMPLEMENTATION_GUIDE.md`, the closed S2 sequence
is in `P4_S2_IMPLEMENTATION_GUIDE.md`, the closed S3 sequence is in
`P4_S3_IMPLEMENTATION_GUIDE.md`, the closed six-entry matrix procedure is in
`P5_IMPLEMENTATION_GUIDE.md`, and the closed compiler-automation procedure is
in `P6_IMPLEMENTATION_GUIDE.md`.  The P7 single-entry procedure is in
`P7_IMPLEMENTATION_GUIDE.md`; its verified result is recorded in
`P7_CLOSURE_REPORT.md`.

- **A0 — complete:** source contract, loop oracle, vectorized oracle, and CPU
  tests are available.
- **P0 — complete:** preserved the W8A16 evidence and made the ChunkScan build
  independent from the W8A16 worktree.
- **P1 — complete:** TPU broadcast, exponential, causal mask, GEMM, cast, and
  residual primitives pass the complete project cmodel regression.
- **P2 — complete:** the serial standalone ChunkScan TPU kernel passes all
  three mathematical terms and the complete causal cmodel gate.
- **P3 — complete:** the TPU task mapping and serial K16 reduction-tiled S1
  control pass direct S1-to-S0 and CPU-oracle cmodel validation.
- **P4 — complete:** P4.1 S2 and P4.2 paper-ordered S3 both pass their cmodel
  numerical gates and raw-source structural gates.
- **P5 — complete:** the frozen six-entry tiling/stage/order matrix records
  three accepted and three explicitly rejected outcomes, without performance
  ranking.
- **P6 — complete:** the explicit sProgram contract is consumed by the
  compiler, all seven planner tests pass, generated-pipeline evidence matches
  the manual S3 reference, and the complete cmodel and protection gates pass.
- **P7 — complete for this worktree's cmodel scope:** the single entry reran
  A0-P6 and archived its logs, source hashes, and evidence index.  Broader
  Mamba2 integration remains separate; the single-core BM1690 device route
  is recorded below.

P1.1 isolated the broadcast requirement: W-axis expansion may use a zero W
stride, whereas C-axis expansion must be materialized by the BM1690 NPU
broadcast instruction.  The explicit-broadcast replacement has passed cmodel
from this worktree with zero error for column, row, scalar, and difference
cases.  The FP32 natural-exponential vector and matrix paths have also passed
cmodel.  P1.3 validated exact lower-triangular causal materialization of
the FP16 `cb` tile.  That probe has passed exactly after correcting local-C
subregion addressing.  P1.4 now validates the historical and scan GEMMs.  An
installed project cmodel probe passed both required FP16-input, FP32-output
GEMMs exactly.  P1.5 validates local dtype conversion and the materialized
dynamic-scalar `D * x` residual path for positive and negative D values.  The
complete P1.1-P1.5 regression now passes, so P1 is closed.  A reported P1.1
failure log came from the older implicit-C-stride implementation; the current
project broadcast probe passes exactly with two explicit NPU broadcasts and
no C-axis zero stride.

P2 now replaces the bring-up draft with a one-core, single-buffer serial S0.
The full candidate has passed isolated cmodel prevalidation for state-only,
scan-only, positive- and negative-D residual-only, all-terms, and causal
upper-triangle poisoning.  Its maximum absolute error is
`6.103515625e-05`, and its generated source uses three explicit NPU broadcasts
with no C-axis zero stride or parallel markers.  The unchanged candidate is
installed in the project worktree and has reproduced the same complete `PASS`
through `run_cmodel.sh`, so P2 is closed.

The downstream pipeline stages keep two forms of generated source.  The raw
BM1690 target source is checked for pipeline regions, task order, reduction
loop structure, synchronization information, and versioned LMEM buffers.  The
cmodel adapter intentionally strips `tpu_parallel_start/end` before compiling
the emulator library, so cmodel output validates only serial functional
equivalence.  The separate PCIe device route retains the raw markers and
validates numerical correctness on hardware; it still does not establish
physical engine overlap or a performance advantage.

## Run the CPU tests

From the repository root:

```bash
python -m pytest -v tpu_demo/mamba2_chunk_scan/test_reference.py
```

A0 through P6 have passed their project-worktree gates.  P4.2 adds the useful
dense `cb` load to the manual two-stage pipeline, retains S2 as its direct
control, and verifies the emitted sProg-B ordering from raw target code.  P5
then closes the fixed configuration matrix with three accepted and three
explicitly rejected outcomes.  P6 closes the compiler-generated explicit
sProg-B route with planner unit tests, deterministic compiler rejection gates,
manual-S3 structural equivalence, and cmodel numerical equivalence.

P7's verified run `20260920T065556Z-4383` completed all 14 sequential gates.
Its result and per-step logs are under `artifacts/p7/`; see
`P7_CLOSURE_REPORT.md` for the independent acceptance audit.  This is a
worktree-local cmodel closure, not a cross-server Git reproduction or a real
BM1690E/SG2260E hardware result.

## BM1690 hardware validation

The separate PCIe route uses `main_template_device.cpp` and the real
`libtpuv7_rt.so`.  `test_chunk_scan_device_s0.py` passed two smoke cases;
`test_chunk_scan_device_pipeline.py` passed six cases each for S1, S2, S3,
and P6 at `B=G=H=1, S=128, Ck=2, L=64, P=64, N=128` on one core.
The device source has no pipeline marker for S1 and one matched marker pair
for each of S2, S3, and P6.  See `BM1690_REPO_HANDOFF.md` for the tested
environment, commands, result audit, and exact limitations.

These results do not establish performance, physical GDMA/BDC overlap,
eight-core scaling, other shapes, or BM1690e/SG2260e compatibility.

The staged follow-up for trustworthy single-core benchmarking, shape
parameterization, BM1690 1/2/4/8-core correctness, scaling measurements, and
final tuning is documented in
`P8_P10_BM1690_PERFORMANCE_MULTICORE_PLAN.md`.

P8 has since passed its seven-round physical BM1690 measurement gate.  P9
starts with the required 1/2/4/8-core runtime work-item probe; its commands and
acceptance checks are in `P9_BM1690_MULTICORE_IMPLEMENTATION_GUIDE.md`.
