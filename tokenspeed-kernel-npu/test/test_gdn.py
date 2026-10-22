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

"""Compare Ascend GDN to an independent float64 matrix-form recurrence."""

import json

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops.gdn import _prepare_qk, gdn_decode, gdn_prefill

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)


def _reference(q, k, v, g, beta, state, lengths, *, normalize, scale):
    # CPU float64 reference uses the full transition matrix, not the tested
    # rank-one prediction/update algorithm. Inputs are already quantized.
    q, k = q.detach().cpu(), k.detach().cpu()
    if normalize:
        # Match the reference model's visible activation-dtype rounding before
        # checking the recurrent arithmetic independently in float64.
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k, v, g, beta, state = [
        x.detach().cpu().double() for x in (q, k, v, g, beta, state)
    ]
    output = torch.empty_like(v)
    final = state.clone()
    offset = 0
    for request, length in enumerate(lengths):
        for head in range(v.shape[2]):
            source_head = head // (v.shape[2] // q.shape[2])
            h = state[request, head].transpose(-1, -2).clone()
            for token in range(offset, offset + length):
                query, key = q[0, token, source_head], k[0, token, source_head]
                gate = beta[0, token, head]
                transition = torch.eye(
                    key.numel(), dtype=torch.float64
                ) - gate * torch.outer(key, key)
                h = g[0, token, head].exp() * (transition @ h) + gate * torch.outer(
                    key, v[0, token, head]
                )
                output[0, token, head] = (query * scale) @ h
            final[request, head] = h.transpose(-1, -2)
        offset += length
    return output, final


def _inputs(lengths, heads, value_heads, key_dim, value_dim, dtype):
    generator = torch.Generator().manual_seed(20261022)
    total = sum(lengths)
    q = torch.randn(1, total, heads, key_dim, generator=generator).to(dtype)
    k = torch.randn(q.shape, generator=generator).to(dtype)
    v = torch.randn(1, total, value_heads, value_dim, generator=generator).to(dtype)
    g = -torch.rand(1, total, value_heads, generator=generator)
    beta = torch.rand(g.shape, generator=generator)
    state = (
        torch.randn(len(lengths), value_heads, value_dim, key_dim, generator=generator)
        * 0.1
    )
    return [x.npu() for x in (q, k, v, g, beta, state)]


def _prefill(q, k, v, g, beta, state, lengths, *, normalize=True):
    boundaries = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="npu"
    )
    return gdn_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        cu_seqlens=boundaries,
        qk_l2norm=normalize,
        output_final_state=True,
        output_h=False,
    )


def _assert_quantized_reference(
    q, k, v, g, beta, initial, lengths, actual, state, *, normalize, scale
):
    normalized_q, normalized_k = _prepare_qk(q, k, q.shape[2], normalize=normalize)
    expected, expected_state = _reference(
        normalized_q,
        normalized_k,
        v,
        g,
        beta,
        initial,
        lengths,
        normalize=False,
        scale=scale,
    )
    torch.testing.assert_close(
        state.cpu().double(), expected_state, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.to(actual.dtype).float(),
        rtol=1e-4 if actual.dtype == torch.float32 else 8e-3,
        atol=2e-6 if actual.dtype == torch.float32 else 1e-3,
    )


@pytest.mark.parametrize(
    "lengths,geometry",
    [
        ([1, 0, 3], (2, 4, 7, 5)),
        ([3, 5], (16, 16, 128, 128)),
        ([63, 1, 65], (2, 2, 16, 12)),
    ],
)
def test_bf16_qk_normalization_rounding(lengths, geometry, record_property):
    q, k, *_ = _inputs(lengths, *geometry, torch.bfloat16)
    normalized = _prepare_qk(q, k, q.shape[2], normalize=True)
    for name, tensor, actual in zip(("q", "k"), (q, k), normalized):
        cpu = tensor.cpu()
        cpu_square, npu_square = cpu * cpu, tensor * tensor
        cpu_sum = cpu_square.sum(-1, keepdim=True)
        npu_sum = npu_square.sum(-1, keepdim=True)
        cpu_norm, npu_norm = cpu_sum + 1e-6, npu_sum + 1e-6
        for expected, result in (
            (cpu_square, npu_square),
            (cpu_sum, npu_sum),
            (cpu_norm, npu_norm),
        ):
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        cpu_inverse, npu_inverse = cpu_norm.rsqrt(), npu_norm.rsqrt().cpu()
        # Both operands are positive BF16, so adjacent bit patterns are one ulp.
        inverse_ulps = (
            cpu_inverse.view(torch.int16).int() - npu_inverse.view(torch.int16).int()
        ).abs()
        assert inverse_ulps.max().item() <= 1
        # Independently check the product with the actual quantized inverse;
        # a one-ulp rsqrt difference can become two ulps after BF16 multiply.
        expected = cpu * npu_inverse
        torch.testing.assert_close(actual.cpu(), expected.float(), rtol=0, atol=0)
        cpu_normalized = cpu * cpu_inverse
        normalized_ulps = (
            expected.view(torch.int16).int() - cpu_normalized.view(torch.int16).int()
        ).abs()
        metrics = {
            "key_dim": geometry[2],
            "role": name,
            "rsqrt_mismatch_count": int(torch.count_nonzero(inverse_ulps)),
            "rsqrt_total": inverse_ulps.numel(),
            "rsqrt_max_ulp": int(inverse_ulps.max()),
            "normalized_max_ulp": int(normalized_ulps.max()),
            "normalized_max_abs": float(
                (expected.float() - cpu_normalized.float()).abs().max()
            ),
        }
        record_property(f"bf16_norm_{name}", json.dumps(metrics))
        print(f"GDN BF16 normalization: {json.dumps(metrics)}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "lengths,geometry",
    [
        ([1, 0, 3], (2, 4, 7, 5)),
        ([3, 5], (16, 16, 128, 128)),
        ([63, 1, 65], (2, 2, 16, 12)),
    ],
)
def test_prefill_float64_reference(dtype, lengths, geometry):
    q, k, v, g, beta, initial = _inputs(lengths, *geometry, dtype)
    before = initial.clone()
    actual, state = _prefill(q, k, v, g, beta, initial, lengths)
    # Isolate FP32 recurrence from hardware-dependent BF16 rsqrt rounding,
    # which is constrained separately above. Keep the raw CPU-model output
    # comparison as an additional end-to-end numerical check below.
    normalized_q, normalized_k = _prepare_qk(q, k, q.shape[2], normalize=True)
    _, expected_state = _reference(
        normalized_q,
        normalized_k,
        v,
        g,
        beta,
        initial,
        lengths,
        normalize=False,
        scale=q.shape[-1] ** -0.5,
    )
    expected, _ = _reference(
        q, k, v, g, beta, initial, lengths, normalize=True, scale=q.shape[-1] ** -0.5
    )
    # FP32 accumulation is checked independently of the final BF16 rounding.
    torch.testing.assert_close(
        state.cpu().double(), expected_state, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.to(dtype).float(),
        rtol=1e-4 if dtype == torch.float32 else 8e-3,
        atol=2e-6 if dtype == torch.float32 else 1e-3,
    )
    torch.testing.assert_close(initial, before, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_prefill_continuation_and_request_isolation(normalize):
    q, k, v, g, beta, initial = _inputs([3, 5], 2, 4, 8, 6, torch.bfloat16)
    whole, state = _prefill(q, k, v, g, beta, initial, [3, 5], normalize=normalize)
    offset = 0
    for request, length in enumerate([3, 5]):
        parts = [x[:, offset : offset + length] for x in (q, k, v, g, beta)]
        first, carry = _prefill(
            *(x[:, :1] for x in parts),
            initial[request : request + 1],
            [1],
            normalize=normalize,
        )
        rest, last = _prefill(
            *(x[:, 1:] for x in parts), carry, [length - 1], normalize=normalize
        )
        # A new forward changes chunk boundaries, as in the official reference.
        # Check both associations against the same independent FP64 recurrence.
        for output, final in (
            (torch.cat((first, rest), dim=1), last),
            (whole[:, offset : offset + length], state[request : request + 1]),
        ):
            _assert_quantized_reference(
                *parts,
                initial[request : request + 1],
                [length],
                output,
                final,
                normalize=normalize,
                scale=q.shape[-1] ** -0.5,
            )
        alone, alone_state = _prefill(
            *parts, initial[request : request + 1], [length], normalize=normalize
        )
        torch.testing.assert_close(
            alone, whole[:, offset : offset + length], rtol=0, atol=0
        )
        torch.testing.assert_close(alone_state[0], state[request], rtol=0, atol=0)
        offset += length


def test_decode_strided_dual_indices_padding_and_reference():
    generator = torch.Generator().manual_seed(17)
    batch, heads, key_dim, value_dim = 3, 2, 7, 5
    packed = (
        torch.randn(batch, 1, heads, key_dim * 2, generator=generator).bfloat16().npu()
    )
    q, k = packed[..., :key_dim], packed[..., key_dim:]
    v = torch.randn(batch, 1, heads, value_dim, generator=generator).bfloat16().npu()
    a = torch.randn(batch, 1, heads, generator=generator).bfloat16().npu()
    b = torch.randn(batch, 1, heads, generator=generator).bfloat16().npu()
    A_log = torch.tensor([-0.2, 0.3], device="npu")
    dt_bias = torch.tensor([-1.0, 0.7], device="npu")
    pool = torch.randn(5, heads, value_dim, key_dim, generator=generator).npu()
    before = pool.clone()
    reads = torch.tensor([1, 2, -1], dtype=torch.int32, device="npu")
    writes = torch.tensor([2, 1, -1], dtype=torch.int32, device="npu")
    actual = gdn_decode(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool,
        initial_state_indices=reads,
        scale=None,
        output_state_indices=writes,
        use_qk_l2norm=True,
    )
    log_decay = -A_log.cpu().double().exp() * F.softplus(
        a.cpu().double() + dt_bias.cpu().double()
    )
    beta = b.cpu().sigmoid().double()
    initial = torch.stack((before[1], before[2], torch.zeros_like(before[0])))
    normalized_q, normalized_k = _prepare_qk(q, k, q.shape[2], normalize=True)
    _, state = _reference(
        normalized_q.transpose(0, 1),
        normalized_k.transpose(0, 1),
        v.transpose(0, 1),
        log_decay.transpose(0, 1),
        beta.transpose(0, 1),
        initial,
        [1, 1, 1],
        normalize=False,
        scale=key_dim**-0.5,
    )
    expected, _ = _reference(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        log_decay.transpose(0, 1),
        beta.transpose(0, 1),
        initial,
        [1, 1, 1],
        normalize=True,
        scale=key_dim**-0.5,
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.transpose(0, 1).bfloat16().float(),
        rtol=8e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(pool[2].cpu().double(), state[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pool[1].cpu().double(), state[1], rtol=2e-5, atol=2e-6)
    for row in (0, 3, 4):
        torch.testing.assert_close(pool[row], before[row], rtol=0, atol=0)


def test_multistep_decode_matches_prefill():
    q, k, v, _, _, initial = _inputs([9], 2, 2, 8, 6, torch.bfloat16)
    generator = torch.Generator().manual_seed(9)
    a = torch.randn(1, 9, 2, generator=generator).bfloat16().npu()
    b = torch.randn(1, 9, 2, generator=generator).bfloat16().npu()
    A_log = torch.tensor([-0.4, 0.1], device="npu")
    dt_bias = torch.tensor([0.2, -0.3], device="npu")
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)
    whole, final = _prefill(q, k, v, g, b.sigmoid(), initial, [9])
    pool = torch.zeros(3, *initial.shape[1:], device="npu")
    pool[1].copy_(initial[0])
    read = torch.tensor([1], dtype=torch.int32, device="npu")
    write = torch.tensor([2], dtype=torch.int32, device="npu")
    outputs = []
    for token in range(9):
        outputs.append(
            gdn_decode(
                q[:, token : token + 1],
                k[:, token : token + 1],
                v[:, token : token + 1],
                A_log=A_log,
                a=a[:, token : token + 1],
                dt_bias=dt_bias,
                b=b[:, token : token + 1],
                initial_state=pool,
                initial_state_indices=read,
                scale=None,
                output_state_indices=write,
                use_qk_l2norm=True,
            )
        )
        read, write = write, read
    for output, state in (
        (torch.cat(outputs, dim=1), pool[read.long()]),
        (whole, final),
    ):
        _assert_quantized_reference(
            q,
            k,
            v,
            g,
            b.sigmoid(),
            initial,
            [9],
            output,
            state,
            normalize=True,
            scale=q.shape[-1] ** -0.5,
        )
    assert torch.count_nonzero(pool[0]).item() == 0


def test_prefill_padding_does_not_change_state():
    q, k, v, g, beta, initial = _inputs([3], 2, 2, 8, 6, torch.bfloat16)
    expected, expected_state = _prefill(q, k, v, g, beta, initial, [3])
    padded = []
    for tensor in (q, k, v, g, beta):
        tail = torch.full_like(tensor[:, :1], float("nan"))
        padded.append(torch.cat((tensor, tail), dim=1))
    actual, state = _prefill(*padded, initial, [3])
    torch.testing.assert_close(actual[:, :3], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 3:]).item() == 0
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_decode_negative_indices_do_not_read_nan_null_state():
    q, k, v, _, _, state = _inputs([1], 2, 2, 8, 6, torch.bfloat16)
    state.fill_(float("nan"))
    before = state.clone()
    padding = torch.tensor([-1], dtype=torch.int32, device="npu")
    output = gdn_decode(
        q,
        k,
        v,
        A_log=torch.zeros(2, device="npu"),
        a=torch.zeros(1, 1, 2, device="npu"),
        dt_bias=torch.zeros(2, device="npu"),
        b=torch.zeros(1, 1, 2, device="npu"),
        initial_state=state,
        initial_state_indices=padding,
        scale=None,
        output_state_indices=padding,
        use_qk_l2norm=True,
    )
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(state, before, rtol=0, atol=0, equal_nan=True)


def test_public_dispatch_and_explicit_unsupported_mode():
    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

    q, k, v, g, beta, state = _inputs([1], 2, 2, 8, 6, torch.bfloat16)
    boundaries = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=state,
        cu_seqlens=boundaries,
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
        solution="torch_npu",
    )
    expected, expected_state = _prefill(q, k, v, g, beta, state, [1])
    torch.testing.assert_close(result.out, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.final_state, expected_state, rtol=0, atol=0)
    index = torch.tensor([0], dtype=torch.int32, device="npu")
    decoded = gdn_decode_step(
        q,
        k,
        v,
        A_log=torch.zeros(2, device="npu"),
        a=torch.zeros(1, 1, 2, device="npu"),
        dt_bias=torch.zeros(2, device="npu"),
        b=torch.zeros(1, 1, 2, device="npu"),
        initial_state=state.clone(),
        initial_state_indices=index,
        solution="torch_npu",
    )
    assert decoded.device.type == "npu"
    with pytest.raises(NotImplementedError, match="checkpoints"):
        gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=state,
            cu_seqlens=boundaries,
            qk_l2norm=True,
            output_final_state=True,
            output_h=True,
        )
    with pytest.raises(ValueError, match="FP32 recurrent"):
        _prefill(q, k, v, g, beta, state.bfloat16(), [1])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "lengths,geometry,dtype",
    [
        ([1], (2, 4, 7, 5), torch.bfloat16),
        ([16], (16, 16, 128, 128), torch.bfloat16),
        ([63, 0, 65], (2, 2, 16, 12), torch.bfloat16),
        ([64, 128], (1, 1, 8, 6), torch.bfloat16),
        ([129], (2, 2, 16, 12), torch.float32),
        ([512], (1, 1, 8, 6), torch.bfloat16),
    ],
)
def test_prefill_matches_hf512_chunk_order(lengths, geometry, dtype, normalize):
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    q, k, v, g, beta, initial = _inputs(lengths, *geometry, dtype)
    if not normalize:
        q.mul_(0.1)
        k.mul_(0.1)
    actual, state = _prefill(q, k, v, g, beta, initial, lengths, normalize=normalize)
    repeats = v.shape[2] // q.shape[2]
    offset = 0
    for request, length in enumerate(lengths):
        if length == 0:
            torch.testing.assert_close(state[request], initial[request], rtol=0, atol=0)
            continue
        stop = offset + length
        expected, final = hf.torch_chunk_gated_delta_rule(
            q[:, offset:stop].repeat_interleave(repeats, dim=2),
            k[:, offset:stop].repeat_interleave(repeats, dim=2),
            v[:, offset:stop],
            g[:, offset:stop],
            beta[:, offset:stop],
            initial_state=initial[request : request + 1].transpose(-1, -2).contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=normalize,
        )
        torch.testing.assert_close(actual[:, offset:stop], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            state[request], final[0].transpose(-1, -2), rtol=0, atol=0
        )
        offset = stop
    _assert_quantized_reference(
        q,
        k,
        v,
        g,
        beta,
        initial,
        lengths,
        actual,
        state,
        normalize=normalize,
        scale=q.shape[-1] ** -0.5,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_decode_matches_hf512_recurrent_order(dtype):
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    q, k, v, _, _, initial = _inputs([5], 2, 4, 16, 12, dtype)
    generator = torch.Generator().manual_seed(21)
    a = torch.randn(1, 5, 4, generator=generator).to(dtype).npu()
    b = torch.randn(1, 5, 4, generator=generator).to(dtype).npu()
    A_log = torch.tensor([-0.2, 0.0, 0.1, 0.3], device="npu")
    dt_bias = torch.tensor([-1.0, 0.2, 0.5, 0.7], device="npu")
    state = initial.clone()
    expected_state = initial.transpose(-1, -2).contiguous()
    index = torch.zeros(1, dtype=torch.int32, device="npu")
    for token in range(q.shape[1]):
        qs, ks, vs, aa, bb = [x[:, token : token + 1] for x in (q, k, v, a, b)]
        actual = gdn_decode(
            qs,
            ks,
            vs,
            A_log=A_log,
            a=aa,
            dt_bias=dt_bias,
            b=bb,
            initial_state=state,
            initial_state_indices=index,
            scale=None,
            output_state_indices=None,
            use_qk_l2norm=True,
        )
        expected, expected_state = hf.torch_recurrent_gated_delta_rule(
            qs.repeat_interleave(2, dim=2),
            ks.repeat_interleave(2, dim=2),
            vs,
            -A_log.float().exp() * F.softplus(aa.float() + dt_bias),
            bb.sigmoid(),
            initial_state=expected_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            state, expected_state.transpose(-1, -2), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (0.7, 1.25)])
def test_gating_public_and_runtime_preserve_parameters(dtype, beta, threshold):
    from tokenspeed_kernel.ops.attention.gdn import gdn_gating

    from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

    a = torch.linspace(-20, 20, 40, device="npu", dtype=dtype).reshape(5, 8)[:, ::2]
    A_log = torch.tensor([-0.4, 0.0, 0.1, 0.5], device="npu")
    dt_bias = torch.tensor([-1.0, 0.2, 0.5, 0.7], device="npu", dtype=dtype)
    expected = -A_log.float().exp() * F.softplus(
        a.float() + dt_bias, beta=beta, threshold=threshold
    )
    for operation in (gdn_gating, fused_gdn_gating):
        actual = operation(A_log, a, dt_bias, beta=beta, threshold=threshold)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, 0.37])
def test_prefill_custom_scale_and_optional_final_state(scale):
    q, k, v, g, beta, initial = _inputs([65], 2, 2, 8, 6, torch.float32)
    before = initial.clone()
    actual, final = gdn_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial,
        cu_seqlens=torch.tensor([0, 65], dtype=torch.int32, device="npu"),
        qk_l2norm=True,
        output_final_state=False,
        output_h=False,
    )
    assert final is None
    normalized_q, normalized_k = _prepare_qk(q, k, q.shape[2], normalize=True)
    expected, _ = _reference(
        normalized_q,
        normalized_k,
        v,
        g,
        beta,
        initial,
        [65],
        normalize=False,
        scale=scale,
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)


def test_prefill_all_empty_requests():
    q, k, v, g, beta, initial = _inputs([0, 0], 2, 4, 8, 6, torch.bfloat16)
    actual, state = _prefill(q, k, v, g, beta, initial, [0, 0])
    assert actual.shape == v.shape
    torch.testing.assert_close(state, initial, rtol=0, atol=0)
