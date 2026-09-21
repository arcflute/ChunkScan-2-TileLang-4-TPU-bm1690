# TileLang-TPU Mamba2 ChunkScan implementation plan

Plan scope revised on 2026-09-13: all required execution and validation use
the CPU-hosted BM1690 cmodel; real-accelerator evaluation is not a project
stage.

Historical note: the P0–P7 plan below was completed under that cmodel-only
scope.  The project subsequently completed a separate single-core BM1690
hardware correctness bring-up.  The approved P8–P10 hardware performance and
multi-core continuation is maintained in
`P8_P10_BM1690_PERFORMANCE_MULTICORE_PLAN.md`; it does not retroactively turn
P0–P7 cmodel evidence into hardware performance evidence.

## P0–P7 historical invariants

1. The W8A16 worktree is a read-only historical baseline.
2. ChunkScan must build and run from its own TileLang and TVM source tree.
3. The first target is the standalone seven-input `_chunk_scan_fwd`.
4. Serial correctness must pass before pipeline scheduling begins.
5. The complete experiment uses the BM1690 cmodel and never requires a real
   TPU or GPU.
6. Cmodel execution proves functional equivalence only.  Raw BM1690 target
   source proves lowering structure only; neither is performance evidence.
7. Pipeline source annotations and buffer versioning do not prove physical
   overlap, concurrency, latency reduction, or speedup on real hardware.
8. No stage ranks candidates using cmodel wall-clock time.
9. Every stage must pass its gate before the next stage starts.

## Experimental boundary

The deliverable is a BM1690-targeted TileLang-TPU reproduction that compiles,
lowers, and executes correctly through the CPU-hosted cmodel.  It contains two
separate kinds of evidence:

1. cmodel numerical evidence that the serial and pipelined programs preserve
   the frozen ChunkScan semantics;
2. raw target-code evidence that the requested reduction tiling, task order,
   pipeline stages, synchronization structure, and buffer versions were
   emitted before cmodel adaptation.

The cmodel adapter deliberately removes `tpu_parallel_start/end` before it
compiles the emulator library.  Consequently, a pipelined candidate must
archive both the raw target source and the cmodel-executed source.  Correct
cmodel output cannot by itself establish that parallel markers, synchronization
or overlapping engines would behave correctly on a physical BM1690.

Real-device latency, throughput, utilization, speedup, physical GDMA/BDC
overlap, contention, and multi-core scaling are outside the project scope and
must remain unclaimed.

## Collaboration rule

- Codex directly maintains Markdown documentation in this directory.
- The user manually applies Python, C++, shell, and other source-code changes
  from the before/after diffs or complete replacement files supplied by Codex.
- Codex verifies the resulting worktree and test evidence before advancing the
  current stage.

## Stages

| Stage | Objective | Required gate |
| --- | --- | --- |
| A0 | Freeze source contract and CPU oracle | CPU loop/vectorized tests pass |
| P0 | Freeze W8A16 and isolate the toolchain | All loaded TileLang/TVM paths belong to this repository |
| P1 | Validate TPU semantic primitives | Broadcast, exp, mask, GEMM, cast and residual microprobes pass cmodel |
| P2 | Build serial S0 ChunkScan | State, scan, residual and all-terms cases pass cmodel |
| P3 | Freeze the TPU sTask mapping and serial reduction-tiled S1 | S1 matches S0/oracle and exposes a legal multi-iteration reduction loop |
| P4 | Implement manual paper-inspired S2/S3 schedules | Every schedule passes cmodel and its raw target structure passes static checks |
| P5 | Cover legal tiling and pipeline configurations | Each chosen candidate passes or is rejected with an explicit dependency/LMEM/lowering reason |
| P6 | Automate the designated pipeline lowering | Compiler-generated structure matches the designated manual reference and passes cmodel |
| P7 | Close the standalone reproduction | All evidence is reproducible from one cmodel-only entry and the final scope statement passes review |

## Current stage

P6 and P7 are closed for this worktree's CPU-hosted BM1690 cmodel scope.
P6's explicit sProgram contract is consumed by `PipelinePlanning`; all seven
planner tests pass; the generated raw structure matches the designated manual
S3 reference; its positive, negative, LMEM, and cmodel numerical gates pass.
P7's single entry reran A0-P6 in 14 sequential steps, refreshed every stage
result, and indexed source and evidence hashes.  The independent P7 audit is
in `P7_CLOSURE_REPORT.md`.  Real TPU execution and cross-server Git
reproduction have not been validated.

P1 is complete.  The project-worktree P1.1-P1.5 regression passed in one
continuous run: rank-2 broadcasting, FP32 natural exponential, causal-mask
materialization, both GEMMs, and cast plus dynamic `D*x` residual all exited
successfully and reported `PASS`.

Completed substage: P1.1, rank-2 column, row, and scalar broadcasting.

Completed substage: P1.2, FP32 natural exponential for the `[64, 1]`
historical-state vector and `[64, 64]` scan-decay matrix.

Completed substage: P1.3, exact lower-triangular causal materialization for the
FP16 `[64, 64]` `cb` tile and correct local-C subregion addressing.

Completed substage: P1.4, the two FP16-input, FP32-accumulation GEMMs:
historical `[64,128] x [64,128]^T -> [64,64]` and scan
`[64,64] x [64,64] -> [64,64]`.

Completed substage: P1.5, local FP16-to-FP32 and FP32-to-FP16
conversion together with the dynamic scalar `D * x` residual path.  The
scalar must be expanded along W into `[1, 64]`, then materialized across NPU
lanes into `[64, 64]`; implicit C-axis zero-stride broadcasting remains
forbidden.

P1.5 acceptance contract:

- load FP16 `x[64,64]` and dynamic FP16 `D[1,1]` from global memory, then
  convert both to FP32 in local memory;
- expand `D[1,1]` along W into an FP32 `[1,64]` row, then use one explicit
  NPU broadcast to materialize FP32 `D_matrix[64,64]`;
- compute `residual_fp32 = x_fp32 * D_matrix`, then convert that result back
  to FP16;
- check positive `D=0.703125` and negative `D=-1.375`; the materialized D
  matrix, FP32 residual, and FP16 residual must each match PyTorch exactly;
- generated source must contain exactly three `tpu_bdc_cast` sites (two
  FP16-to-FP32 and one FP32-to-FP16), one fill, one add, one multiply, one
  NPU broadcast, two S2L copies, three L2S copies, zero C-axis zero-stride
  sites, and one W-axis zero-stride site.

The installed project probe passed with zero mismatches and zero absolute
error for all six numerical checks.  Its generated-source counts exactly
match the contract above.  P1.5 and the complete P1 primitive stage are now
closed.  Cmodel timing is not a BM1690 performance result.

P2 serial S0 acceptance contract:

- retain the seven-input logical ABI plus one output and the frozen smoke
  shape `B=G=H=1`, `Ck=2`, `L=P=64`, `N=128`, `S=128`;
- use only one core, one buffer version, a serial two-chunk loop, no
  `tpu_parallel_start/end`, and no PipeThreader overlap annotation;
- expose compact `dt` and `dA_cumsum` as legal physical `[2,64]` row views;
  construct the required local dA column with one bounded `T.serial(64)` GDMA
  scatter instead of reading beyond a `[128,1]` buffer boundary;
- compute the historical-state GEMM, causal scan GEMM, and dynamic `D*x`
  residual in FP32 accumulation, then convert once to the FP16 output;
- materialize row/scalar broadcasts explicitly and require zero generated
  C-axis zero-stride sites;
- validate state-only, scan-only, positive-D residual-only, negative-D
  residual-only, and all-terms cases against the CPU oracle at `atol=rtol=1e-2`;
- require both chunks to carry nonzero reference signals and require poisoning
  every CB upper-triangle entry to leave the clean device output bitwise
  unchanged;
- archive the toolchain identity, physical views, lowered TIR, generated C,
  runtime sources, source metrics, numerical results, and the explicit
  statement that cmodel timing is not hardware performance.

The complete P2 candidate has passed isolated prevalidation.  Its largest
absolute error is `6.103515625e-05` in the all-terms case; positive and
negative residual-only cases are exact; the poisoned and clean device outputs
are bitwise identical.  The generated source contains one historical
transpose-B GEMM, one scan GEMM, two FP32 exponential sites, three explicit
NPU broadcasts, three legal W-axis zero strides, no C-axis zero stride, and no
parallel markers.  The parsed static LMEM address upper bound is
`82176 / 262144` bytes.  The candidate is now installed unchanged in the
project worktree and the complete project `run_cmodel.sh` entry has produced a
`PASS` result with the same numerical and source-structure evidence.  P2 is
closed.

## Downstream cmodel-only stages

### P3: sTask mapping and serial reduction-tiled S1

P3 replaces the removed real-BM1690 baseline stage.  It introduces no
parallelism.  Its purpose is to translate the paper's ChunkScan running example
into an explicit TPU task graph and to create the reduction loop required by a
meaningful software pipeline.

The paper partitions the scan GEMM reduction axis rather than merely renaming
the outer chunk loop.  P3 must therefore:

- preserve S0 as the full-tile serial correctness control;
- derive and archive the exact data-dependency graph for the causal scan hot
  loop, including `LoadY` (`cb`, `dA`, and `dt`), decay/scale preprocessing,
  `LoadX`, and GEMM accumulation;
- map data movement to the BM1690 GDMA class and exponential, elementwise,
  broadcast, cast, and GEMM work to the BDC class; BDC operations remain
  ordered because this project does not claim that they are independent
  concurrent engines;
- build S1 as a serial reduction-tiled implementation with FP32 accumulation
  and no pipeline markers;
- use a legal reduction tile configuration with enough iterations to contain
  a nonempty prologue, steady state, and epilogue once S2/S3 are introduced;
- if the fixed `L=64` smoke shape cannot provide at least four legal reduction
  iterations, add a separate pipeline-conformance shape while retaining the
  original P2 smoke case unchanged;
- validate S1 against both S0 and the CPU oracle for the isolated scan term and
  the complete seven-input operator, including the causal-poisoning case;
- archive the selected tile shape, loop extent, dependency edges, engine-class
  mapping, TIR, raw source, source metrics, and LMEM allocation.

P3 closes only when S1 is numerically correct, contains the expected serial
reduction loop, stays within LMEM, and contains no
`tpu_parallel_start/end`.  S1 cmodel time is ignored.

The frozen P3 mapping is recorded in `stask_mapping.md`.  The selected S1
configuration is `L=64`, `K_TILE=16`, and four serial reduction iterations, so
the original P2 smoke shape is sufficient and no extra conformance shape is
needed.  The installed S1 has passed the project cmodel gate.  S0 and S1 ran
in separate worker processes and their saved outputs were compared directly
and against the same CPU oracle.  The all-terms S1-to-oracle maximum absolute
error is `6.103515625e-05`; the scan-only S1-to-S0 maximum absolute difference
is `1.9073486328125e-06`; positive and negative residual cases are exact; and
the clean and upper-triangle-poisoned outputs are bitwise identical.  Raw and
cmodel source are identical, the expected four-iteration reduction loop occurs
once, parallel-marker counts are zero, and the parsed static LMEM upper bound
is `69632 / 262144` bytes.  The protected P2 S0 regression was run again after
P3 and remains `PASS`.  P3 is closed.

### P4: manual paper-inspired pipeline schedules

P4 constructs the following controlled variants:

| Variant | Contract |
| --- | --- |
| S0 | Original one-core, single-buffer, full-tile serial ChunkScan |
| S1 | Reduction-tiled serial control with no pipeline markers |
| S2 | Two-stage Load/Compute schedule with versioned producer buffers |
| S3 | Explicit paper-inspired schedule that prioritizes `LoadY`, then schedules decay/scale preprocessing with `LoadX`, followed by GEMM accumulation |

S2 and S3 use TileLang pipeline scheduling primitives.  Before P6,
`PipelinePlanning` regenerated the planning annotations before software
pipeline injection, so frontend `order`/`stage` arguments were not accepted as
proof of the final schedule.  The P4 manual variants therefore express the
intended order through their source statement order and enforce the actual
order on raw target code.  P6 now supplies the stable compiler representation:
explicit order/stage are validated, preserved as standard pipeline annotations,
and consumed before software-pipeline injection.

Each candidate must satisfy two independent gates.

Numerical gate:

- run through the cmodel after the adapter removes the parallel markers;
- match the CPU oracle and S1 for state-only, scan-only, positive- and
  negative-residual-only, all-terms, both-chunks-nonzero, and causal-poisoning
  cases;
- retain FP32 accumulation and the frozen FP16 external ABI.

Structural gate on raw BM1690 target source:

- S0 and S1 contain no pipeline region;
- S2 and S3 contain the expected balanced pipeline region or regions;
- the reduction loop has identifiable prologue, nonempty steady state, and
  epilogue;
- producer values that cross stage boundaries have distinct static LMEM buffer
  versions, while values that do not cross a boundary are not duplicated
  without justification;
- task order and dependency/synchronization structure match the frozen S2/S3
  schedule contracts;
- instruction-family counts, buffer addresses, loop extents, and LMEM upper
  bound pass automated checks;
- the archived cmodel-executed source contains no parallel markers, recording
  the adapter transformation explicitly.

P4 does not compare candidate wall-clock times and does not select a fastest
variant.  Passing P4 means that the intended pipeline arrangement is expressed
and lowered while preserving serial semantics; it does not mean that physical
overlap or speedup has been observed.

P4.1 S2 keeps the four-iteration `K=16` axis frozen by P3 and changes its loop
to `T.Pipelined(..., num_stages=2)`.  The current TPU rewriter double-buffers
the global-to-local `dA`, `dt`, and `x` producers.  Because it cannot version a
conditionally selected causal `cb` source, S2 first materializes four immutable
lower-triangular `cb` K16 slices outside the pipeline and selects among them
inside the BDC work.  This lowering limitation is part of the evidence and is
not hidden by a compiler modification.

The installed S2 passes S1/oracle numerical comparison and causal poisoning.
Its raw source has one balanced pipeline region, a two-iteration steady loop,
two static versions of each `dA`, `dt`, and `x` producer, zero C-axis
zero-stride sites, and a parsed LMEM upper bound of `69632 / 262144` bytes.
The cmodel-adapted source has zero parallel markers.  The protected S1
regression also remains `PASS`, so P4.1 is closed.

P4.2 S3 keeps the same K16-by-four, two-stage loop but moves a useful dense
`cb` global-to-local load into the pipeline.  To preserve causal semantics
without unsupported cross-NPU-lane mask construction, S3 materializes one
strict-upper correction matrix per chunk and computes
`causal_cb = dense_cb - upper_cb` inside the pipeline.  The dense `cb`, `dA`,
`dt`, and `x` producers all lower to two static versions.

The installed S3 passes S2/oracle comparison for every term,
negative `D`, all terms, and upper-triangle poisoning; every S3-to-S2 maximum
absolute difference is zero.  Raw evidence contains one balanced pipeline
region, prologue producer counts of two each, steady producer counts of one
each, no epilogue loads, and scan-GEMM counts of zero/one/two across
prologue/steady/epilogue.  The steady future-load order is
`LoadY.cb -> LoadY.dA -> LoadY.dt -> LoadX`, the raw LMEM bound is
`69632 / 262144` bytes, and cmodel source contains no parallel markers.  The
project result is `PASS`, so P4.2 is closed.

### P5: dependency- and LMEM-aware configuration coverage

The paper jointly explores task partitioning and pipeline scheduling using a
hardware profiler.  That performance-guided optimization cannot be reproduced
in this cmodel-only project.  P5 instead freezes the following six-entry
matrix:

| Configuration | Expected outcome | Gate |
| --- | --- | --- |
| S2, K16, two stages | accepted | closed P4.1 control |
| sProg-B, K16, two stages | accepted | closed P4.2 manual reference |
| sProg-B, K16, three stages | accepted | triple-buffer raw structure and cmodel correctness |
| sProg-A, K16, two stages | rejected | requested LoadX-first is not preserved in raw TPU target order |
| sProg-B, K32, two stages | rejected | two iterations cannot form a nonempty two-stage steady state |
| sProg-B, K16, four stages | rejected | four iterations cannot form a nonempty four-stage steady state |

For every declared configuration, P5 must record one of two outcomes:

1. accepted: lowering succeeds, dependency and LMEM checks pass, raw source
   matches the intended structure, and cmodel correctness passes;
2. rejected: the exact dependency, LMEM, unsupported-lowering, insufficient
   loop-depth, or numerical reason is archived.

The three-stage accepted candidate must expose three versions of the effective
`cb`, `dA`, `dt`, and `x` producers.  With four reduction iterations, its
single steady iteration is compiler-unrolled; the raw gate therefore checks a
balanced parallel region and `3/1/0` producer placement across
prologue/steady/epilogue rather than requiring a textual one-iteration loop.

The installed P5 candidate and matrix test pass: all three-stage-to-S3
maximum absolute differences are zero, causal poisoning is bitwise invariant,
and the static LMEM bound remains `69632 / 262144` bytes.  The sProg-A probe
demonstrates that the current TPU planner rewrites its requested LoadX-first
source into LoadY-first raw order, so it is rejected rather than mislabeled as
implemented.  The project matrix contains exactly three accepted and three
explicitly rejected configurations and reports `PASS`, so P5 is closed.

P5 must not call any candidate "best", "faster", "optimal", or "preferred by
performance".  The later reference candidate is designated by paper fidelity,
structural coverage, and reproducibility, not cmodel timing.

### P6: compiler automation without a performance policy

P6 moves the designated manual S3 structure into reusable TileLang-TPU
compiler machinery.  The automation scope is:

- represent or infer the selected task stages and dependency edges;
- produce the required `order`, `stage`, synchronization, and grouping data;
- perform buffer versioning for values crossing pipeline stages;
- generate prologue, steady-state, and epilogue structure;
- reject insufficient loop depth, dependency violations, and LMEM overflow;
- emit an auditable schedule manifest with the generated target source.

The compiler-generated candidate must match the designated manual reference in
task order, stage partition, versioned buffers, pipeline regions, instruction
families, and cmodel numerical output.  This is rule-based lowering automation,
not a reproduction of PipeThreader's profile-guided optimal-schedule search.

### P7: standalone reproduction closure

P7 packages the finished standalone experiment.  One top-level cmodel-only
entry must reproduce:

- toolchain identity and frozen source contract;
- A0 CPU tests and P1 primitive regression;
- P2 S0, P3 S1, and P4 manual S2/S3 numerical and structural checks;
- the P5 configuration outcome table;
- the P6 compiler-generated reference comparison;
- all TIR, raw target source, cmodel-executed source, manifests, LMEM metrics,
  and numerical summaries.

The final report must state that the project demonstrates a BM1690-targeted
TileLang-TPU ChunkScan implementation and cmodel-validated pipeline lowering.
It must also state that real-TPU execution, performance, physical overlap, and
full-model Mamba2 integration were not evaluated.  Broader
`mamba_chunk_scan_combined` or model integration is a separate optional route,
not a gate for this standalone reproduction.

P7 closed on 2026-09-20 in run `20260920T065556Z-4383`: all 14 sequential
gates passed, all stage results were refreshed, and the source/evidence index
was verified.  See `P7_CLOSURE_REPORT.md` for the independent audit and exact
scope limits.

P1.4 acceptance contract:

- the historical-state case consumes two FP16 `[64, 128]` tiles and computes
  `left @ right.T` into one FP32 `[64, 64]` accumulation tile;
- the causal-scan case consumes two FP16 `[64, 64]` tiles and computes
  `left @ right` into one FP32 `[64, 64]` accumulation tile;
- both FP32 output tiles are explicitly zero-filled before their single GEMM;
- both paths are compared with PyTorch matmul after converting the FP16 inputs
  to FP32, at `rtol=1e-4`, `atol=1e-4`;
- generated source must contain exactly one `tpu_bdc_fp_mm_R_trans`, one
  ordinary `tpu_bdc_fp_mm`, two FP32 fills, four S2L copies, and two L2S
  copies, with FP16 inputs and FP32 outputs on both GEMM sites.

The installed project probe passed both cases with maximum absolute and
relative error `0.0`.  Generated source contains exactly one
`tpu_bdc_fp_mm_R_trans`, one ordinary `tpu_bdc_fp_mm`, two FP32 fills, four
S2L copies, two L2S copies, and two FP16-input/FP32-output GEMM sites.  A final
rerun of P1.1, P1.2, P1.3, and P1.4 passed after the new runner entry was
installed.  P1.4 is complete.  Cmodel timing remains correctness-only
evidence and is not used as a BM1690 performance claim.

P1.3 exposed a second C-axis layout issue.  A zero-filled full local tile plus
64 static row-prefix S2L copies generated the intended instruction count, but
only row zero was populated: 2079 of 2080 lower-triangle elements mismatched.
The local-region address calculation had treated logical C as linear LMEM via
`c * stride.c`; on BM1690, logical C selects an NPU lane first and advances
within a lane only after every `NPU_NUM` channels.

The candidate correction computes a local C subregion byte offset as:

```text
(c % NPU_NUM) * LOCAL_MEM_SIZE
+ (c / NPU_NUM) * stride.c * dtype_bytes
```

With that correction, the project-worktree cmodel probe produced an exact
64x64 lower triangle with one fill, 64 statically unrolled S2L copies, one L2S
copy, and no runtime loop.  A separate isolated boundary probe passed local C
offsets `0, 1, 63, 64, 127`, covering both lane selection and the next C group.
The P1.1 and P1.2 probes were rerun after rebuilding the corrected compiler and
remained passing.  P1.3 is complete.

P1.2 acceptance contract:

- the vector input covers `[-8, 0]` and represents the historical-state
  `exp(dA_q)` factor;
- the matrix is constructed as `dA[q] - dA[k]` and covers `[-8, 8]` before
  the causal mask is applied;
- both paths are compared with `torch.exp` at `rtol=1e-5`, `atol=1e-7`;
- generated source must contain exactly two coefficient loads, two table
  loads, two FP32 exponential sites, two S2L copies, and two L2S copies.

The project-worktree probe passed cmodel: vector maximum absolute error
`5.960464477539063e-08`, matrix maximum absolute error
`3.0517578125e-05`, and maximum relative error
`1.1486029194429648e-07` for both.  Generated source contains exactly the
required two coefficient loads, two table loads, two FP32 exponential sites,
two S2L copies, and two L2S copies.  P1.2 is complete.

P1.1 has established the required implementation split:

- logical N-axis broadcast (`[M, 1] -> [M, N]`) is valid through a zero PPL
  W stride;
- logical M-axis broadcast (`[1, N] -> [M, N]`) is not valid through a zero
  PPL C stride on BM1690 and must use `tpu_bdc_npu_bcast` explicitly;
- a dynamic scalar (`[1, 1] -> [M, N]`) must first expand along W into a
  one-row tile, then use the explicit NPU broadcast.

The original implicit-C-stride probe failed after isolating the fault.  The
replacement was then rebuilt and passed from the project worktree with zero
error in the column, row, scalar, and difference cases.  Generated source has
zero C-stride sites, two W-stride broadcast sites, and two explicit NPU
broadcast sites.  A subsequent project-worktree regression rerun again passed
all four cases with the same structural counts.  Output from an older
pre-fix run may show three `ppl.add` calls and unsafe C zero strides; it is not
evidence for the current source.  After such an older failure log was
reported, the current project entry was rerun again and produced the expected
two `tpu_bdc_npu_bcast` sites, zero C-axis zero strides, and exact PASS results
for all four cases.  P1.1 is complete.

Downstream broadcast inventory for the serial draft, to be applied when the
P2 serial kernel is revised after all P1 primitive gates close:

- `historical *= dA_col` and `decay_scores += dA_col` are logical N-axis
  broadcasts and may retain the validated zero-W-stride elementwise path;
- `decay_scores -= dA_row` and `decay_scores *= dt_row` are logical M-axis
  broadcasts and require an explicitly materialized NPU-broadcast matrix;
- `scan_or_residual = x * D` starts from a dynamic `[1, 1]` value and requires
  W expansion to `[1, HEADDIM]` followed by explicit NPU broadcast to
  `[CHUNK_SIZE, HEADDIM]`.

## Evidence required for every kernel candidate

- repository and TVM commits;
- dirty/clean state;
- logical ABI and physical views;
- shape, dtype and tile parameters;
- lowered host and device TIR;
- raw generated BM1690 target source before cmodel marker removal;
- the source actually compiled for cmodel after adaptation;
- LMEM allocation metrics;
- pipeline stage/order/dependency manifest and versioned-buffer addresses when
  applicable;
- term-isolated and all-terms correctness;
- execution environment identity for CPU reference and cmodel;
- an explicit statement that cmodel wall-clock time is diagnostic only and is
  not used to compare schedules or support performance claims.
