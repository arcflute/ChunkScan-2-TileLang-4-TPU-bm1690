# Standalone `_chunk_scan_fwd` source contract

## 1. Sources and operator identity

The sole TileLang source of truth for this contract is:

```text
/root/autodl-tmp/pipethreader-references/tilelang-v0.1.5/
examples/linear_attention/example_mamba_chunk_scan.py
commit: a32009bf1e314b514c07389123648ba19009f3a5
tag: v0.1.5
```

Relevant definitions are:

- `chunk_scan_triton`, lines 13-16: calls Mamba `_chunk_scan_fwd`;
- `ref_program`, lines 19-63: mathematical reference;
- `chunk_scan_fwd`, lines 85-217: TileLang kernel generator;
- `main` ABI, lines 93-101;
- state contribution, lines 138-153;
- causal intra-chunk scan, lines 155-179;
- residual and output, lines 181-195.

The corresponding official implementation used for cross-checking is:

```text
/root/autodl-tmp/pipethreader-references/mamba-reference/
mamba_ssm/ops/triton/ssd_chunk_scan.py
observed commit: e9594ce1c732d97440f0332fdc43170a2294dbfa
```

Relevant definitions are `_chunk_scan_fwd_kernel` at line 49 and
`_chunk_scan_fwd` at line 1259.  This A0 contract deliberately follows the
narrower TileLang v0.1.5 ABI rather than every optional feature accepted by the
official wrapper.

## 2. Dimensions and exact ABI

Symbols:

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `S` | sequence length |
| `Ck` | number of chunks |
| `L` | chunk size |
| `G` | number of groups |
| `H` | number of heads |
| `P` | head dimension |
| `N` | state dimension (`dstate`) |

The frozen relation is `S = Ck * L`.  All dimensions are positive and `H` must
be divisible by `G`.

The seven inputs and one output are:

| # | Name | Logical shape | Exact TileLang v0.1.5 dtype |
| --- | --- | --- | --- |
| 0 | `cb` | `[B, Ck, G, L, L]` | FP16 |
| 1 | `x` | `[B, S, H, P]` | FP16 |
| 2 | `dt` | `[B, H, Ck, L]` | FP16 |
| 3 | `dA_cumsum` | `[B, H, Ck, L]` | FP16 |
| 4 | `C` | `[B, S, G, N]` | FP16 |
| 5 | `prev_states` | `[B, Ck, H, P, N]` | FP16 |
| 6 | `D` | `[H]` | FP16 |
| 7 | output | `[B, S, H, P]` | FP16 |

These dtypes come directly from lines 86-87 and 94-101: every ABI tensor uses
`dtype = "float16"`, while local accumulators use `accum_dtype = "float"`.
`D` is required by the TileLang `main` function and is exactly one-dimensional,
even though the broader official wrapper allows `D=None` or `[H, P]`.

## 3. Logical layout and strides

The TileLang ABI declares shaped tensors but has no explicit stride arguments.
It therefore does not expose the official wrapper's arbitrary input strides.
For this reproduction, the external ABI is frozen to compact C-contiguous
logical layout in the exact axis order above.  Expected element strides are:

| Tensor | Axis order | C-contiguous element strides |
| --- | --- | --- |
| `cb` | `B,Ck,G,q,s` | `(Ck*G*L*L, G*L*L, L*L, L, 1)` |
| `x` | `B,S,H,P` | `(S*H*P, H*P, P, 1)` |
| `dt` | `B,H,Ck,s` | `(H*Ck*L, Ck*L, L, 1)` |
| `dA_cumsum` | `B,H,Ck,s` | `(H*Ck*L, Ck*L, L, 1)` |
| `C` | `B,S,G,N` | `(S*G*N, G*N, N, 1)` |
| `prev_states` | `B,Ck,H,P,N` | `(Ck*H*P*N, H*P*N, P*N, N, 1)` |
| `D` | `H` | `(1,)` |
| output | `B,S,H,P` | `(S*H*P, H*P, P, 1)` |

The official Triton wrapper forwards real tensor strides.  In particular, for
the physically `[B,H,Ck,L]` tensors it passes `stride(2)` as the kernel's chunk
stride and `stride(1)` as its head stride (official lines 1294-1295).  That
stride-generic behavior is not added to the frozen TileLang ABI.

The swizzled layouts at TileLang lines 132-136 apply only to selected GPU shared
buffers.  They do not change the external tensor layout or the mathematical
contract.

## 4. Index mapping

For batch `b`, chunk `c`, output position `q` within a chunk, source position
`s`, head `h`, head channel `p`, and state coordinate `n`:

```text
global output position = c * L + q
global source position = c * L + s
head-to-group ratio     = H / G
group(h)                = floor(h / (H / G))
```

Thus consecutive groups of `H/G` heads share the same `cb` and `C` group.  This
matches the TileLang indices `bz // (nheads // ngroups)` at lines 147 and 159,
and the official `pid_h // nheads_ngroups_ratio` at lines 83 and 87.

The GPU launch mapping in the fixed TileLang source is:

```text
bz = head h
by = c * B + b
batch_idx = by % B
chunk_idx = by // B
bx = combined output-position tile and headdim tile
```

## 5. Mathematics

Let:

```text
a_q = dA_cumsum[b,h,c,q]
a_s = dA_cumsum[b,h,c,s]
g   = group(h)
```

The output at `(b, c*L+q, h, p)` is the sum of three explicit terms.

### Historical-state contribution

```text
state[b,c,q,h,p] =
    exp(a_q) * sum_n(
        C[b,c*L+q,g,n] * prev_states[b,c,h,p,n]
    )
```

`prev_states[b,c]` is the state entering chunk `c`.  Index zero contains the
initial state.  The standalone kernel neither produces these states nor returns
a final state.

### Causal intra-chunk scan

```text
scan[b,c,q,h,p] = sum over s satisfying 0 <= s <= q of (
    cb[b,c,g,q,s]
    * exp(a_q - a_s)
    * dt[b,h,c,s]
    * x[b,c*L+s,h,p]
)
```

The causal condition is exactly `q >= s`.  TileLang implements it at lines
173-175; entries above the diagonal must have no effect, regardless of their
stored value.

### Residual

```text
residual[b,c,q,h,p] = D[h] * x[b,c*L+q,h,p]
```

The final result is `state + scan + residual`.

## 6. Precision

The external ABI and output are FP16.  The TileLang accumulator `acc_o`, decay
fragments, `dt` fragments, scale, `D`, and residual fragment are declared with
`accum_dtype = "float"` (FP32).  GEMM contributions accumulate into `acc_o`.
The final store converts the accumulator to the FP16 output buffer.

The fixed TileLang kernel expresses natural exponential as:

```text
exp2(value * 1.44269504)
```

where `1.44269504 = log2(e)`.  The A0 oracle evaluates the equivalent natural
`exp` in FP32 and converts only the final result to FP16.

## 7. Differences from official `_chunk_scan_fwd`

| Area | TileLang v0.1.5 paper kernel | Official Mamba wrapper/kernel |
| --- | --- | --- |
| Dtype | all ABI tensors fixed FP16 | output follows `x`; kernel uses input element types and FP32 intermediates |
| Strides | no stride parameters; this project freezes compact layout | forwards tensor strides explicitly |
| `D` | required `[H]` | optional `[H]` or `[H,P]` |
| `z` gate | absent | optional SiLU gate and optional pre-gate output |
| `seq_idx` | absent | optional sequence-boundary masking |
| Tail chunk | unsupported; `S=Ck*L` | masked partial final chunk supported |
| Decay | `exp2((a_q-a_s)*log2(e))` | `exp(min(a_q-a_s, 0))` in current kernel |
| `cb` | already supplied | already supplied by `_chunk_scan_fwd`; constructed elsewhere in combined forward |
| State output | none | standalone also returns no final state; combined path handles it elsewhere |

For physically valid Mamba2 inputs, `dt > 0` and `A < 0`, so `dA_cumsum` is
non-increasing within a chunk.  Under the causal condition `q >= s`,
`a_q-a_s <= 0`; the official clamp therefore does not change valid arithmetic.

## 8. Explicitly unsupported in this version

- ChunkCumsum and construction of `dA_cumsum` inside the operator;
- ChunkState and construction of `prev_states`;
- StatePassing, initial-state propagation, and final-state output;
- construction of `cb` through `C @ B^T`;
- `z`/SiLU gate;
- `seq_idx` and packed/variable-length sequences;
- partial/tail chunks;
- `D=None` and two-dimensional `D[H,P]`;
- non-FP16 ABI tensors;
- non-contiguous external layouts;
- complete `mamba_chunk_scan_combined` or a complete Mamba2 model.
