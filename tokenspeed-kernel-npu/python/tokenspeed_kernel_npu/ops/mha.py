# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def _is_qwen35_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.shape[1:] == (8, 256)
        and k.shape[-2:] == v.shape[-2:] == (2, 256)
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
    )


def _qwen35_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Complete Qwen3.5 attention with the model's BF16 eager boundaries.

    QK and scale each return BF16, softmax computes in FP32 and returns BF16,
    and AV returns BF16. This explicitly supported geometry uses O(Q*K)
    temporary storage, unlike native FIA. No activation moves to the CPU.
    """
    if not q.shape[0] or not k.shape[0]:
        return q.new_zeros(q.shape)
    if is_causal and q.shape[0] > k.shape[0]:
        raise ValueError("causal queries must be a suffix of cached K/V")
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).repeat_interleave(4, dim=0).unsqueeze(0)
    value = v.transpose(0, 1).repeat_interleave(4, dim=0).unsqueeze(0)
    scores = torch.matmul(query, key.transpose(-1, -2)) * _scale(q, softmax_scale)
    if is_causal:
        query_positions = (
            torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
        )
        key_positions = torch.arange(k.shape[0], device=q.device)
        blocked = key_positions[None, :] > query_positions[:, None]
        mask = torch.zeros_like(blocked, dtype=q.dtype).masked_fill(
            blocked, torch.finfo(q.dtype).min
        )
        scores = scores + mask
    probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    return torch.matmul(probabilities, value).squeeze(0).transpose(0, 1).contiguous()


def _validate_qwen35_boundaries(boundaries: list[int], tokens: int) -> None:
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] > tokens
        or any(start > end for start, end in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("query boundaries must partition valid input tokens")


def _qwen35_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    boundaries: list[int],
    softmax_scale: float | None,
) -> torch.Tensor:
    _validate_qwen35_boundaries(boundaries, q.shape[0])
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("prefill q/k/v token counts must match")
    output = torch.zeros_like(q)
    for start, end in zip(boundaries, boundaries[1:]):
        output[start:end] = _qwen35_attention(
            q[start:end],
            k[start:end],
            v[start:end],
            is_causal=True,
            softmax_scale=softmax_scale,
        )
    return output


def _qwen35_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    boundaries: list[int],
    is_causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Gather every valid K/V row, preserving page order and request isolation.

    Host length metadata is read for eager allocation. This path is not graph
    capturable; the Qwen3.5 NPU runtime already rejects graph mode.
    """
    if torch.npu.is_current_stream_capturing():
        raise NotImplementedError("Qwen3.5 BF16 attention requires eager execution")
    _validate_qwen35_boundaries(boundaries, q.shape[0])
    lengths = cache_seqlens.cpu().tolist()
    if len(lengths) != len(boundaries) - 1 or page_table.shape[0] != len(lengths):
        raise ValueError("cache metadata must have one row per request")
    output = torch.zeros_like(q)
    page_size = k_cache.shape[1]
    for request, (start, end, length) in enumerate(
        zip(boundaries, boundaries[1:], lengths)
    ):
        if length < 0 or length > page_table.shape[1] * page_size:
            raise ValueError("cache sequence length exceeds page table capacity")
        if start == end or length == 0:
            continue
        pages = page_table[request, : (length + page_size - 1) // page_size].long()
        key = k_cache[pages].reshape(-1, 2, 256)[:length]
        value = v_cache[pages].reshape(-1, 2, 256)[:length]
        output[start:end] = _qwen35_attention(
            q[start:end],
            key,
            value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
        )
    return output


def _mha_prefill_bsh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    boundaries: list[int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Run complete causal attention per packed request in CANN's BSH layout.

    CANN 9.0's uncached TND operator rejects a 256-wide head. BSH supports
    this geometry. Host boundaries already belong to the public prefill ABI;
    only tensor layout conversion and one attention launch per request are
    added. The compressed causal mask has constant size, including long input.
    """
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] > q.shape[0]
        or any(start > end for start, end in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("prefill boundaries must partition valid input tokens")
    if k.shape[0] != q.shape[0] or v.shape[:2] != k.shape[:2]:
        raise ValueError("prefill q/k/v token counts and k/v head counts must match")
    output = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        if length == 0:
            continue
        result, _ = torch_npu.npu_fused_infer_attention_score(
            q[start:end].reshape(1, length, -1).contiguous(),
            k[start:end].reshape(1, length, -1).contiguous(),
            v[start:end].reshape(1, length, -1).contiguous(),
            # A one-token BSH call selects CANN's decode implementation;
            # it needs no causal mask because its only key is visible.
            atten_mask=None if length == 1 else _causal_mask(q.device),
            actual_seq_lengths=[length],
            actual_seq_lengths_kv=[length],
            num_heads=q.shape[1],
            num_key_value_heads=k.shape[1],
            scale=_scale(q, softmax_scale),
            input_layout="BSH",
            sparse_mode=0 if length == 1 else 2,
        )
        output[start:end].copy_(result.reshape(length, q.shape[1], v.shape[-1]))
    return output


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens, max_seqlen
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if _is_qwen35_bf16(q, k, v):
        return _qwen35_prefill(q, k, v, cu_seqlens_cpu, softmax_scale)
    if q.shape[-1] == 256:
        return _mha_prefill_bsh(q, k, v, cu_seqlens_cpu, softmax_scale)
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=cu_seqlens_cpu[1:],
        actual_seq_lengths_kv=cu_seqlens_cpu[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=2,
    )
    return output


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    if _is_qwen35_bf16(q, k_cache, v_cache):
        if torch.npu.is_current_stream_capturing():
            raise NotImplementedError("Qwen3.5 BF16 attention requires eager execution")
        return _qwen35_paged_attention(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            boundaries=cu_seqlens_q.cpu().tolist(),
            is_causal=is_causal,
            softmax_scale=softmax_scale,
        )

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return output


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run one-query decode over a paged K/V cache.

    Native geometries support graph capture. Qwen3.5's BF16 Q8/KV2/D256
    eager arithmetic path reads host length metadata and rejects capture.
    """
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if max_seqlen_q != 1:
        raise NotImplementedError("Ascend MHA decode supports one query per request")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    batch_size = cache_seqlens.shape[0]
    if _is_qwen35_bf16(q, k_cache, v_cache):
        return _qwen35_paged_attention(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            boundaries=list(range(batch_size + 1)),
            is_causal=False,
            softmax_scale=softmax_scale,
        )
    actual_seq_lengths_kv = (
        [1] * batch_size if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, 1, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        block_size=k_cache.shape[1],
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
