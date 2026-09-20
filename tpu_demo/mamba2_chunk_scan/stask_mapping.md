# P3 ChunkScan sTask mapping

This document freezes the software-task decomposition used by the P3 serial
reduction-tiled control and the later P4 manual pipeline schedules.  It is a
structural contract for BM1690 lowering and cmodel validation, not evidence of
physical overlap or performance.

## Pipeline axis

The pipeline axis is the source-position reduction axis `s` of the causal scan
GEMM.  For the frozen smoke shape, `L = 64` and P3 selects:

- reduction tile `K_TILE = 16`;
- reduction iterations `K_TILES = 4`;
- one core and one buffer version;
- a serial `T.serial(4)` loop with no `tpu_parallel_start/end` markers.

This is distinct from the outer two-chunk loop.  The outer chunk loop remains
serial and is not the axis that P4 will software-pipeline.

## Task graph

| sTask | Inputs loaded or consumed | Operation class | BM1690 engine class | Output |
| --- | --- | --- | --- | --- |
| `LoadY.cb` | `cb[q, s]` sub-tile | global-to-local copy plus legal causal materialization/correction | GDMA then BDC where correction is needed | `cb_tile[64,16]` |
| `LoadY.dA` | `dA_cumsum[q]`, `dA_cumsum[s]` | global-to-local copy and cast | GDMA then BDC | FP32 query/source decay vectors |
| `LoadY.dt` | `dt[s]` | global-to-local copy and cast | GDMA then BDC | FP32 source scale vector |
| `DecayScale` | local `cb`, `dA[q]`, `dA[s]`, `dt[s]` | subtract, natural exponential, explicit broadcast, multiply, cast | BDC | weighted FP16 scan matrix tile |
| `LoadX` | `x[s, p]` | global-to-local copy | GDMA | `x_tile[16,64]` |
| `MMA` | weighted scan matrix and `x_tile` | FP16-input GEMM and FP32 accumulation add | BDC | updated FP32 scan accumulator |

The current PPL path places exponential, elementwise work, casts, broadcasts,
and GEMM in the BDC class.  Therefore their relative order is a dependency
constraint; the project does not model them as separate concurrent engines.

## Dependency edges

The required edges for reduction iteration `k` are:

```text
LoadY.cb  ───────────────────────────────┐
LoadY.dA ──> DecayScale.exp ────────────┤
LoadY.dt  ───────────────────────────────┤
                                           v
                         DecayScale.mul -> MMA.gemm -> MMA.fp32_add[k]
LoadX ────────────────────────────────────^                |
                                                             v
                                                   MMA.fp32_add[k+1]
```

Equivalently, the frozen per-iteration serial order is:

```text
LoadY -> DecayScale -> LoadX -> MMA
```

The accumulation edge from iteration `k` to iteration `k + 1` must always be
preserved.  P4 may overlap independent loads with earlier computation only
after assigning distinct buffer versions and retaining every edge above.

## Causal-tile contract

For reduction block `k`, the source range is `[16*k, 16*(k+1))`.  The value
entering `DecayScale` and GEMM at query row `q` must retain only positions
`s <= q`; no upper-triangle global value may influence the result.

S1 and S2 implement this contract by zero-filling four local K16 blocks and
copying only legal row prefixes.  Their four static causal copy groups contain
64, 48, 32, and 16 row-prefix copy sites respectively.  S3 instead prepares
one strict-upper correction matrix, loads the useful dense K16 `cb` tile as a
versioned pipeline producer, and computes `causal_cb = dense_cb - upper_cb`
inside the pipeline.  Both representations are subject to the same poisoning
gate.

The causal-poison test replaces every upper-triangle `cb` entry with `8.0` and
requires the clean and poisoned cmodel outputs to be bitwise identical.

## Work outside the reduction pipeline

The following complete-operator work remains outside the scan reduction loop:

- the historical-state contribution `C @ prev_states^T`, including its decay;
- the dynamic residual `D * x`;
- the final FP32 additions, FP32-to-FP16 conversion, and global store.

These terms still participate in all numerical gates.  Keeping them outside
the P3 reduction loop prevents the later pipeline experiment from changing the
standalone seven-input ChunkScan contract.

## P3 structural gate

The serial S1 lowering must contain exactly one four-iteration `k_blk` loop,
one ordinary scan GEMM site, one historical transpose-B GEMM site, no parallel
markers, no C-axis zero-stride broadcast, and a static local-memory upper bound
not exceeding 262144 bytes.  The frozen candidate uses 69632 bytes according
to the conservative source parser.

The already closed S0 result is first checked as prerequisite evidence.  S0
and S1 then run sequentially in separate cmodel worker processes because two
generated adapters are not isolated reliably in one Python process.  Their
saved outputs are compared directly with each other and independently with the
same CPU oracle, frozen operator semantics, and seed.

Any cmodel timing printed by the adapter is ignored.
