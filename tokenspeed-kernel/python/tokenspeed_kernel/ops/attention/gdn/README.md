# GDN kernels

## Ascend eager baseline

The `torch_npu` solution registers `ascend_eager_gdn_chunk_prefill` and
`ascend_eager_gdn_decode_step` through `tokenspeed-kernel-npu`. It executes
the complete delta-rule recurrence using NPU tensor operations. Q/K
normalization keeps each intermediate in the activation dtype before casting
to FP32, and the beta sigmoid keeps the gate-input dtype. This deliberately
matches the actual BF16 entry points of
[Transformers 5.12.0](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)'s
`torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`. The
existing GPU fused decode normalizes directly in FP32; its rounding differs
from this reference-aligned Ascend baseline. Recurrent updates use FP32 and
outputs round to the activation dtype. State uses the shared
`[rows, value_heads, value_dim, key_dim]` layout and must be FP32.
BF16 state, speculative verification,
intermediate checkpoints, and graph capture are not supported. Run the
engine with `--enforce-eager --mamba-ssm-dtype float32`.

Prefill accepts packed variable-length sequences and reads their cumulative
lengths on the host once per call. Each request uses 64-token blocks with
FP32 matrix operations and contiguous internal `[batch, heads, key, value]`
state. Tail blocks are zero-padded to 64, including for short requests. The
triangular correction runs its 63 row updates across all blocks together,
then carries state between blocks. This follows the operation order of the
official reference, reducing launch growth for long prompts; short prompts
still pay the fixed triangular-correction cost. Both the host metadata
synchronization and NPU launch overhead belong to the measured baseline.
No activation or state arithmetic runs on the CPU. Empty sequences preserve
their initial states and the initial-state tensor remains read-only.

The block algorithm is adapted under Apache-2.0 in
`tokenspeed_kernel_npu/thirdparty/qwen35_gdn.py`; its source attribution and
full license are included in `tokenspeed-kernel-npu/THIRDPARTYNOTICES`.
The package does not import Transformers at runtime. Prefill log decay uses
the public `gdn_gating` boundary and NPU FP32 `exp` / `softplus` operations;
its explicit softplus `beta` and `threshold` arguments are preserved. Other
platforms retain their existing gating implementation.

Decode gathers all source rows before writing any destination row. Negative
read indices supply zero state and negative write indices suppress writes;
padding never redirects a write to the cache's null row. Write indices must
be distinct, as guaranteed by the existing scheduler ownership contract.
Q/K/V and state-pool views may be strided. The scheduler retains ownership
of state allocation, prefix reuse, request isolation, and reclamation.
Decode converts gathered state to contiguous K-first layout and reduces
over the key axis in the same order as the reference, then writes K-last
state back to the shared pool.

`tokenspeed-kernel-npu/test/test_gdn.py` compares NPU outputs and states to
an independent CPU FP64 matrix recurrence. It covers the Qwen3.5-0.8B
16-head, 128-by-128 state geometry, asymmetric dimensions, grouped heads,
empty and boundary-length sequences, continuation, cross-request source
and destination exchanges, padding, and multi-step decode.
Additional tests compare BF16/FP32 outputs and states exactly with the
Transformers 5.12.0 chunk/recurrent operations on the same NPU, including
63/64/65/129/512-token lengths, nonzero states, and asymmetric dimensions.
Changing a forward's chunk boundary or comparing chunk prefill with recurrent
decode changes FP32 association order. Each is checked against the same
independent FP64 oracle at the existing tolerances; bitwise identity is
required for the same operation order and for isolation/padding behavior.

The numerical checks separate input normalization from recurrence. With
CANN 9.0 and seed 20261022, the CPU and NPU BF16 square, sum, and epsilon
addition were identical. BF16 `rsqrt` differed by at most one ulp: for
key dimension 7, 3/8 Q inverses and 1/8 K inverses differed; for dimension
16, 1/258 differed in each; all 128 inverses per Q/K matched at the model's
key dimension 128. The final BF16 multiply can turn this into a two-ulp
normalized component difference (maximum absolute difference 0.0078125).
Casting through FP32 before `rsqrt` did not remove the CPU/NPU difference.
For every differing sample, the NPU value matched FP64 `rsqrt` rounded to
BF16; the CPU BF16 approximation was the differing value.
The test reports these measurements, bounds `rsqrt` to one ulp, and checks
the final multiplication independently. It then checks the FP32 state
against an independent FP64 matrix recurrence fed the same quantized Q/K
with `rtol=2e-5, atol=2e-6`. A separate full-output comparison retains raw
CPU normalization. This avoids treating hardware rounding as recurrent
arithmetic error. Whole-model evaluation separately records whether its HF
cache uses the stock model-dtype recurrent state or an explicit FP32 state
allocation; changing that cache policy does not change HF model equations.

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Decode and prefill explicitly select FlashInfer's CuTe backend. FlashInfer
0.7.0's automatic backend selection can use Cake GDN for supported shapes,
which bypasses the PDL-wrapped device bodies. Regression tests cover both
power-of-two and other head groupings and reject calls into that alternate path.

FlashInfer 0.7.0 also persists CuTe-DSL kernels to disk. PDL artifacts use a
separate `tokenspeed_pdl_` module namespace so an ordinary kernel cannot satisfy
a PDL cache lookup, or vice versa. Cache invalidation includes all upstream
source files plus the local PDL and GDN adapter sources. Installed FlashInfer
modules and the process-wide cache configuration remain untouched.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Chain verification

ReplaySSM chains use chunked verification on Blackwell for 13–16 tokens and
at least 24 (request, value-head) states. Other shapes and platforms retain
step-by-step verification. `test/runtime/test_gdn_verify_policy.py` checks this
selection and its boundaries without launching device kernels.

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request, and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.

## Accepted-prefix replay

`triton_gdn_replay_commit` commits accepted prefixes for all layers in one
portable Gluon launch. Each CTA owns a layer, request, value head, and V tile.
Keys and gates are prepared in fixed chunks, with K/V staged in shared memory.

State updates stay in FP32 until the final write, and rejected tokens are
masked during loading. Aligned state accesses are vectorized, with fallbacks
for unaligned rows and arbitrary row strides. Batch size, window length, and
payload layer stride remain runtime scalars, so serving shape changes reuse
the compiled kernel.
