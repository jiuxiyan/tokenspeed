# Ascend operator contracts

The runtime imports these implementations through the registered public
`tokenspeed-kernel` operations. This package owns the vendor implementation;
the runtime prologue still owns every K/V cache write.

## Qwen3.5 attention

BF16 Q/K/V with 8 query heads, 2 K/V heads and head dimension 256 use the
explicit Qwen3.5 eager arithmetic boundaries: BF16 QK matmul, BF16 scale,
BF16 additive causal mask, FP32 softmax rounded to BF16, and BF16 AV matmul.
The default scale is `1/sqrt(head_dim)`; an explicitly supplied scale is
preserved. This matches the installed Transformers Qwen3.5 eager operation on
the same NPU, including its intermediate rounding. It is complete dense
attention; it does not truncate history or replace causal attention.

Packed prefill partitions requests using host cumulative lengths. Paged
extend and decode gather only the pages needed for each visible cache length,
in logical page order, then remove the final page's invalid suffix. Causal
extend aligns query position zero to `kv_length - query_length`. Noncausal
extend attends every visible cache row. Decode accepts one query per request.
Empty requests and extra padded output rows remain zero and do not read
invalid page addresses. Inputs and K/V caches are not modified.

This geometry is an eager correctness path. Paged lengths are copied to the
host for per-request allocation; graph capture is explicitly rejected. It
uses O(query length * key length) score/probability storage, explicitly
expands grouped K/V heads, and launches operations per request. These costs
must be included in performance results. Other geometries retain the native
CANN fused attention behavior; uncached head-256 shapes use per-request BSH
because CANN 9.0 rejects that head dimension in uncached TND.

## Qwen3.5 partial RoPE

BF16 NeoX head-256/rotary-64 inputs use an activation-dtype cos/sin table and
round each multiplication before the addition, matching Qwen3.5's explicit
BF16 tensor operations. The unrotated 192 channels are copied exactly. Both
in-place operation and separate output buffers are supported. The composite
prologue resolves M-RoPE position rows into a flat selected table before this
leaf is called. Other shapes and dtypes retain native fused rotation.

## Verification

`test_qwen35_attention.py` retains independent CPU FP32 references and tests
the public attention dispatcher against the official same-device BF16 eager
implementation with zero tolerance. Coverage includes packed variable-length
requests, random physical page order, partial pages, invalid-tail sentinels,
zero-length and padded requests, causal and noncausal extend, custom scales,
and both RoPE output modes. `test_mha.py` covers the existing native geometry,
including graph replay with updated cache lengths.
