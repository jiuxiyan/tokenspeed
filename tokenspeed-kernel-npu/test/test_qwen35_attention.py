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

"""Qwen3.5-0.8B attention geometry against independent CPU FP32 references."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel_npu.ops.rotary_embedding import apply_rope

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Qwen3.5 attention tests require an NPU"
)

Q_HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM, PAGE_SIZE = 8, 2, 256, 64, 128


def _random(shape, generator):
    return torch.randn(shape, generator=generator, dtype=torch.float32).bfloat16()


def _attention_reference(q, k, v, *, causal):
    """CPU FP32 GQA with explicit head replication and causal positions."""
    assert q.device.type == k.device.type == v.device.type == "cpu"
    q = q.float().transpose(0, 1)
    k = k.float().repeat_interleave(Q_HEADS // KV_HEADS, dim=1).transpose(0, 1)
    v = v.float().repeat_interleave(Q_HEADS // KV_HEADS, dim=1).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(HEAD_DIM)
    if causal:
        # A cached extension occupies the last q_length positions in K/V.
        query_positions = torch.arange(q.shape[1]) + k.shape[1] - q.shape[1]
        key_positions = torch.arange(k.shape[1])
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores.masked_fill_(~allowed[None, :, :], -float("inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1)


def _check(actual, expected, record_property, name):
    actual = actual.detach().cpu().float()
    expected = expected.float()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    difference = actual - expected
    record_property(f"{name}_max_abs", difference.abs().max().item())
    record_property(f"{name}_rmse", difference.square().mean().sqrt().item())
    record_property(
        f"{name}_relative_l2",
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item(),
    )
    # This is the existing Ascend BF16 attention/RoPE test criterion. The
    # independent reference retains FP32 probabilities and FP32 arithmetic.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("lengths", [(1,), (3, 5), (127, 129)])
def test_qwen35_prefill(lengths, record_property):
    generator = torch.Generator().manual_seed(20261023)
    total = sum(lengths)
    q = _random((total, Q_HEADS, HEAD_DIM), generator)
    k = _random((total, KV_HEADS, HEAD_DIM), generator)
    v = _random(k.shape, generator)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    actual = mha_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        torch.tensor(boundaries, dtype=torch.int32, device="npu"),
        boundaries,
        max(lengths),
    )
    expected = torch.cat(
        [
            _attention_reference(q[start:end], k[start:end], v[start:end], causal=True)
            for start, end in zip(boundaries, boundaries[1:])
        ]
    )
    _check(actual, expected, record_property, "prefill")


def _paged_fixture(lengths, generator):
    """Independent CPU sequences placed into non-contiguous physical pages.

    Invalid positions contain large finite sentinels, so attending beyond a
    supplied sequence length changes the result rather than looking like zeros.
    """
    pages_per_request = (max(lengths) + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = len(lengths) * pages_per_request
    permutation = torch.randperm(num_pages, generator=generator)
    page_table = permutation.reshape(len(lengths), pages_per_request).int()
    k_cache = torch.full(
        (num_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM), 23.0, dtype=torch.bfloat16
    )
    v_cache = torch.full_like(k_cache, -29.0)
    sequences = []
    for request, length in enumerate(lengths):
        k = _random((length, KV_HEADS, HEAD_DIM), generator)
        v = _random(k.shape, generator)
        sequences.append((k, v))
        for logical_page in range((length + PAGE_SIZE - 1) // PAGE_SIZE):
            start = logical_page * PAGE_SIZE
            count = min(PAGE_SIZE, length - start)
            physical_page = int(page_table[request, logical_page])
            k_cache[physical_page, :count] = k[start : start + count]
            v_cache[physical_page, :count] = v[start : start + count]
    return k_cache, v_cache, page_table, sequences


@pytest.mark.parametrize("lengths", [(1, 63, 64, 65), (127, 128, 129, 257)])
def test_qwen35_paged_decode(lengths, record_property):
    generator = torch.Generator().manual_seed(20261024)
    q = _random((len(lengths), Q_HEADS, HEAD_DIM), generator)
    k_cache, v_cache, page_table, sequences = _paged_fixture(lengths, generator)
    actual = mha_decode_with_kvcache(
        q.npu(),
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        max_seqlen_k=max(lengths),
    )
    expected = torch.cat(
        [
            _attention_reference(q[i : i + 1], k, v, causal=False)
            for i, (k, v) in enumerate(sequences)
        ]
    )
    _check(actual, expected, record_property, "decode")


@pytest.mark.parametrize(
    "lengths,q_lengths", [((65, 129), (3, 5)), ((129, 257), (1, 3))]
)
def test_qwen35_paged_extend(lengths, q_lengths, record_property):
    generator = torch.Generator().manual_seed(20261025)
    q = _random((sum(q_lengths), Q_HEADS, HEAD_DIM), generator)
    k_cache, v_cache, page_table, sequences = _paged_fixture(lengths, generator)
    cumulative_q, cumulative_kv = [0], [0]
    for q_length, length in zip(q_lengths, lengths):
        cumulative_q.append(cumulative_q[-1] + q_length)
        cumulative_kv.append(cumulative_kv[-1] + length)
    actual = mha_extend_with_kvcache(
        q.npu(),
        torch.tensor(cumulative_q, dtype=torch.int32, device="npu"),
        torch.tensor(cumulative_kv, dtype=torch.int32, device="npu"),
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        max(q_lengths),
        max(lengths),
        is_causal=True,
    )
    expected = torch.cat(
        [
            _attention_reference(q[start:end], k, v, causal=True)
            for start, end, (k, v) in zip(cumulative_q, cumulative_q[1:], sequences)
        ]
    )
    _check(actual, expected, record_property, "extend")


def _rope_reference(values, table, positions, heads):
    """CPU FP32 partial NeoX RoPE; table itself remains FP32 here."""
    source = values.float().reshape(len(positions), heads, HEAD_DIM)
    cosine, sine = table[positions].chunk(2, dim=-1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    first = source[..., : ROTARY_DIM // 2]
    second = source[..., ROTARY_DIM // 2 : ROTARY_DIM]
    rotated = torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    )
    return torch.cat((rotated, source[..., ROTARY_DIM:]), dim=-1).reshape(values.shape)


@pytest.mark.parametrize("positions", [(0,), (0, 1, 63, 64, 127, 128, 129, 511)])
@pytest.mark.parametrize("output_buffers", [False, True])
def test_qwen35_partial_rope(positions, output_buffers, record_property):
    generator = torch.Generator().manual_seed(20261026)
    positions = torch.tensor(positions, dtype=torch.int64)
    inverse_frequency = 1.0 / (
        10_000_000.0 ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM)
    )
    angles = torch.outer(
        torch.arange(int(positions.max()) + 1).float(), inverse_frequency
    )
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    packed = _random((len(positions), (Q_HEADS + KV_HEADS) * HEAD_DIM), generator)
    q_cpu, k_cpu = packed.split((Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM), dim=-1)
    q, k = packed.npu().split((Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM), dim=-1)
    q_before, k_before = q.clone(), k.clone()
    q_out = torch.empty_like(q) if output_buffers else None
    k_out = torch.empty_like(k) if output_buffers else None
    apply_rope(
        positions=positions.npu(),
        q=q,
        k=k,
        head_size=HEAD_DIM,
        cos_sin_cache=table.npu(),
        is_neox=True,
        q_rope_out=q_out,
        k_rope_out=k_out,
    )
    if output_buffers:
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
        torch.testing.assert_close(k, k_before, atol=0, rtol=0)
    for name, output, original, heads in (
        ("rope_q", q if q_out is None else q_out, q_cpu, Q_HEADS),
        ("rope_k", k if k_out is None else k_out, k_cpu, KV_HEADS),
    ):
        expected = _rope_reference(original, table, positions, heads)
        _check(output, expected, record_property, name)
        # The 192 nonrotary channels must be bitwise unchanged.
        torch.testing.assert_close(
            output.cpu().reshape(len(positions), heads, HEAD_DIM)[..., ROTARY_DIM:],
            original.reshape(len(positions), heads, HEAD_DIM)[..., ROTARY_DIM:],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("positions", [(0,), (0, 1, 63, 64, 127, 128, 129, 511)])
@pytest.mark.parametrize("output_buffers", [False, True])
def test_qwen35_rope_hf_bfloat16_boundary(positions, output_buffers):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    generator = torch.Generator().manual_seed(20261029)
    position_ids = torch.tensor(positions, dtype=torch.int64, device="npu")
    inverse = 1.0 / (
        10_000_000.0 ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM)
    )
    angles = torch.outer(torch.arange(max(positions) + 1).float(), inverse)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).npu()
    packed = _random((len(positions), (Q_HEADS + KV_HEADS) * HEAD_DIM), generator).npu()
    q, k = packed.split((Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM), dim=-1)
    q_before, k_before = q.clone(), k.clone()
    cosine, sine = table[position_ids].bfloat16().chunk(2, dim=-1)
    cosine = torch.cat((cosine, cosine), dim=-1).unsqueeze(0)
    sine = torch.cat((sine, sine), dim=-1).unsqueeze(0)
    expected_q, expected_k = apply_rotary_pos_emb(
        q.reshape(len(positions), Q_HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        k.reshape(len(positions), KV_HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        cosine,
        sine,
    )
    q_out = torch.empty_like(q) if output_buffers else None
    k_out = torch.empty_like(k) if output_buffers else None
    apply_rope(
        positions=position_ids,
        q=q,
        k=k,
        head_size=HEAD_DIM,
        cos_sin_cache=table,
        is_neox=True,
        q_rope_out=q_out,
        k_rope_out=k_out,
    )
    for actual, expected in (
        (q if q_out is None else q_out, expected_q),
        (k if k_out is None else k_out, expected_k),
    ):
        torch.testing.assert_close(
            actual,
            expected.squeeze(0).transpose(0, 1).reshape(actual.shape),
            atol=0,
            rtol=0,
        )
    if output_buffers:
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
        torch.testing.assert_close(k, k_before, atol=0, rtol=0)


def _hf_attention(q, k, v, *, causal, scale):
    """Run the installed official HF implementation on the same NPU inputs."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        eager_attention_forward,
    )

    with torch.device("meta"):
        module = Qwen3_5Attention(
            Qwen3_5TextConfig(
                hidden_size=1024,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=256,
            ),
            layer_idx=3,
        ).eval()
    mask = None
    if causal:
        mask = torch.full(
            (q.shape[0], k.shape[0]),
            torch.finfo(q.dtype).min,
            dtype=q.dtype,
            device=q.device,
        ).triu(diagonal=k.shape[0] - q.shape[0] + 1)[None, None]
    result, _ = eager_attention_forward(
        module,
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        mask,
        scaling=scale,
        dropout=0.0,
    )
    return result.squeeze(0)


@pytest.mark.parametrize("lengths", [(1,), (0, 3, 5), (127, 129)])
@pytest.mark.parametrize("scale", [0.0625, 0.071])
def test_qwen35_prefill_hf_boundary(lengths, scale):
    from tokenspeed_kernel.ops.attention.mha import mha_prefill as public_prefill

    generator = torch.Generator().manual_seed(20261030)
    total = sum(lengths)
    q = _random((total + 2, Q_HEADS, HEAD_DIM), generator).npu()
    k = _random((total + 2, KV_HEADS, HEAD_DIM), generator).npu()
    v = _random(k.shape, generator).npu()
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    actual = public_prefill(
        q,
        k,
        v,
        torch.tensor(boundaries, dtype=torch.int32, device="npu"),
        boundaries,
        max(lengths),
        softmax_scale=scale,
        solution="torch_npu",
    )
    expected = torch.zeros_like(q)
    for start, end in zip(boundaries, boundaries[1:]):
        if start < end:
            expected[start:end] = _hf_attention(
                q[start:end],
                k[start:end],
                v[start:end],
                causal=True,
                scale=scale,
            )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("lengths", [(0, 1, 63, 64, 65), (127, 128, 129, 257)])
def test_qwen35_decode_hf_boundary(lengths):
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache as public_decode,
    )

    generator = torch.Generator().manual_seed(20261031)
    q = _random((len(lengths), Q_HEADS, HEAD_DIM), generator).npu()
    k_cache, v_cache, page_table, sequences = _paged_fixture(lengths, generator)
    # A padded row has no valid page addresses. It must not read any cache.
    for i, length in enumerate(lengths):
        if length == 0:
            page_table[i].fill_(-1)
        else:
            used_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
            page_table[i, used_pages:].fill_(2**30)
    actual = public_decode(
        q,
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        max_seqlen_k=max(lengths),
        max_seqlen_q=1,
        solution="torch_npu",
    )
    expected = torch.zeros_like(q)
    for i, (k, v) in enumerate(sequences):
        if k.shape[0]:
            expected[i : i + 1] = _hf_attention(
                q[i : i + 1], k.npu(), v.npu(), causal=False, scale=0.0625
            )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qwen35_decode_null_page_padding_and_readonly_cache():
    """Runtime padding clamps its cache length to one and selects null page 0."""
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache as public_decode,
    )

    generator = torch.Generator().manual_seed(20261102)
    q = _random((3, Q_HEADS, HEAD_DIM), generator).npu()
    k, v, table, sequences = _paged_fixture((65, 129), generator)
    k_cache = torch.cat((torch.zeros_like(k[:1]), k), dim=0).npu()
    v_cache = torch.cat((torch.zeros_like(v[:1]), v), dim=0).npu()
    table = table + 1
    page_table = torch.stack((table[0], torch.zeros_like(table[0]), table[1])).npu()
    before = [value.clone() for value in (q, k_cache, v_cache, page_table)]
    actual = public_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        torch.tensor([65, 1, 129], dtype=torch.int32, device="npu"),
        max_seqlen_k=129,
        max_seqlen_q=1,
        solution="torch_npu",
    )
    expected = torch.zeros_like(q)
    for i, (key, value) in zip((0, 2), sequences):
        expected[i : i + 1] = _hf_attention(
            q[i : i + 1],
            key.npu(),
            value.npu(),
            causal=False,
            scale=0.0625,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for original, value in zip(before, (q, k_cache, v_cache, page_table)):
        torch.testing.assert_close(original, value, atol=0, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "lengths,q_lengths", [((0, 65, 129), (0, 3, 5)), ((129, 257), (1, 3))]
)
def test_qwen35_extend_hf_boundary(lengths, q_lengths, causal):
    from tokenspeed_kernel.ops.attention.mha import (
        mha_extend_with_kvcache as public_extend,
    )

    generator = torch.Generator().manual_seed(20261101)
    q = _random((sum(q_lengths) + 2, Q_HEADS, HEAD_DIM), generator).npu()
    k_cache, v_cache, page_table, sequences = _paged_fixture(lengths, generator)
    cumulative_q, cumulative_kv = [0], [0]
    for q_length, length in zip(q_lengths, lengths):
        cumulative_q.append(cumulative_q[-1] + q_length)
        cumulative_kv.append(cumulative_kv[-1] + length)
    actual = public_extend(
        q,
        torch.tensor(cumulative_q, dtype=torch.int32, device="npu"),
        torch.tensor(cumulative_kv, dtype=torch.int32, device="npu"),
        k_cache.npu(),
        v_cache.npu(),
        page_table.npu(),
        torch.tensor(lengths, dtype=torch.int32, device="npu"),
        max(q_lengths),
        max(lengths),
        is_causal=causal,
        solution="torch_npu",
    )
    expected = torch.zeros_like(q)
    for start, end, (k, v) in zip(cumulative_q, cumulative_q[1:], sequences):
        if start < end:
            expected[start:end] = _hf_attention(
                q[start:end],
                k.npu(),
                v.npu(),
                causal=causal,
                scale=0.0625,
            )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
