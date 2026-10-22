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

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.activation import qwen35_sigmoid_mul, sigmoid_mul
from tokenspeed_kernel.ops.layernorm import (
    gated_rmsnorm,
    gemma_rmsnorm,
    qk_rmsnorm,
    qwen35_gated_rmsnorm,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Qwen3.5 norm/gate tests require an NPU"
)


def _close(actual, expected):
    tolerance = 1e-2 if actual.dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        actual.cpu(), expected.to(actual.dtype), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("rows", [0, 1, 7, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gemma_rmsnorm(rows, dtype, with_residual):
    torch.manual_seed(11)
    x = torch.randn(rows, 1024, dtype=dtype)
    weight = torch.randn(1024, dtype=dtype) * 0.2
    residual = torch.randn_like(x) if with_residual else None
    values = x if residual is None else x + residual
    expected = F.rms_norm(values.double(), (1024,), eps=1e-6)
    expected *= 1 + weight.double()
    x_npu = x.npu()
    residual_npu = None if residual is None else residual.npu()
    result = gemma_rmsnorm(x_npu, weight.npu(), 1e-6, residual_npu)
    if with_residual:
        result, updated = result
        assert updated is residual_npu
        torch.testing.assert_close(updated.cpu(), values, atol=0, rtol=0)
    _close(result, expected)


def test_gemma_weight_offset_does_not_round_to_bfloat16():
    # This offset disappears if 1 + weight is materialized in BF16.
    x = torch.ones(1, 1024, dtype=torch.float32, device="npu")
    weight = torch.full((1024,), 2**-10, dtype=torch.bfloat16, device="npu")
    expected = torch.full((1, 1024), (1 + 2**-10) / (1 + 1e-6) ** 0.5)
    torch.testing.assert_close(
        gemma_rmsnorm(x, weight, 1e-6).cpu(), expected, atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("rows", [0, 1, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qk_gemma_rmsnorm_strided_projection(rows, dtype):
    torch.manual_seed(12)
    packed = torch.randn(rows, 10 * 256, dtype=dtype)
    weights = [torch.randn(256, dtype=dtype) * 0.2 for _ in range(2)]
    q_cpu, k_cpu = packed.split([8 * 256, 2 * 256], dim=-1)
    q, k = packed.npu().split([8 * 256, 2 * 256], dim=-1)
    result = qk_rmsnorm(
        q, k, weights[0].npu(), weights[1].npu(), 1e-6, weight_offset=1.0
    )
    for actual, values, weight, heads in zip(result, [q_cpu, k_cpu], weights, [8, 2]):
        expected = F.rms_norm(
            values.double().reshape(rows, heads, 256), (256,), eps=1e-6
        )
        expected = (expected * (1 + weight.double())).reshape(values.shape)
        _close(actual, expected)


@pytest.mark.parametrize("rows", [0, 1, 3, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shared_weight", [False, True])
@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
def test_gated_rmsnorm_strided(
    rows, dtype, shared_weight, norm_before_gate, sigmoid_gate
):
    torch.manual_seed(13)
    width, group = 16 * 128, 128
    packed = torch.randn(rows, width * 2, dtype=dtype)
    x_cpu, z_cpu = packed.split(width, dim=-1)
    x, z = packed.npu().split(width, dim=-1)
    weight = torch.randn(group if shared_weight else width, dtype=dtype)
    grouped = x_cpu.double().reshape(rows, 16, group)
    gate = z_cpu.double().sigmoid()
    if not sigmoid_gate:
        gate *= z_cpu.double()
    if not norm_before_gate:
        grouped = grouped * gate.reshape(rows, 16, group)
    expected = F.rms_norm(grouped, (group,), eps=1e-6).reshape(rows, width)
    expected *= weight.double().repeat(16) if shared_weight else weight.double()
    if norm_before_gate:
        expected *= gate
    result = gated_rmsnorm(
        x,
        weight.npu(),
        z,
        eps=1e-6,
        group_size=group,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )
    _close(result, expected)


@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_sigmoid_gate_strided_head_view(rows, dtype, with_bias):
    torch.manual_seed(14)
    x = torch.randn(rows, 8 * 256, dtype=dtype)
    packed = torch.randn(rows, 8, 2 * 256, dtype=dtype)
    gate_cpu = packed[..., 256:]
    gate = packed.npu()[..., 256:]
    bias = torch.randn(8, dtype=dtype) if with_bias else None
    values = gate_cpu.double()
    if bias is not None:
        values = values + bias.double().reshape(1, 8, 1)
    expected = x.double() * values.sigmoid().reshape(x.shape)
    x_npu = x.npu()
    result = sigmoid_mul(x_npu, gate, None if bias is None else bias.npu())
    assert result is x_npu
    _close(result, expected)


def test_runtime_gemma_boundary_and_weight_loader():
    from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

    torch.manual_seed(15)
    norm = GemmaRMSNorm(1024).to(device="npu", dtype=torch.bfloat16)
    weight = torch.randn(1024, dtype=torch.bfloat16) * 0.2
    norm._weight_loader(norm.weight, weight.npu())
    x = torch.randn(3, 1024, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    expected_residual = x + residual
    expected = F.rms_norm(expected_residual.double(), (1024,), eps=1e-6)
    expected *= 1 + weight.double()
    with torch.inference_mode():
        result, fp8, updated = norm.add_norm_with_fp8(x.npu(), residual.npu(), None)
        result_fp4, fp4, _ = norm.add_norm_with_fp4(x.npu(), residual.npu(), None)
    assert fp8 is None and fp4 is None
    _close(result, expected)
    _close(result_fp4, expected)
    torch.testing.assert_close(updated.cpu(), expected_residual, atol=0, rtol=0)


def test_gated_rmsnorm_without_gate_and_unsupported_fp8():
    torch.manual_seed(16)
    x = torch.randn(2, 128, dtype=torch.float32, device="npu")
    weight = torch.randn(128, dtype=torch.float32, device="npu")
    kwargs = dict(
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        sigmoid_gate=False,
        weights_independent=True,
    )
    result = gated_rmsnorm(x, weight, None, fp8_scale=None, **kwargs)
    expected = F.rms_norm(x.cpu().double(), (128,), eps=1e-6) * weight.cpu().double()
    _close(result, expected)
    with pytest.raises(ValueError, match="FP8"):
        gated_rmsnorm(x, weight, None, fp8_scale=x.new_ones(()), **kwargs)


def test_sigmoid_gate_2d_and_invalid_shape():
    x = torch.linspace(-2, 2, 32, device="npu").reshape(2, 16)
    gate = torch.linspace(2, -2, 64, device="npu").reshape(2, 32)[:, :16]
    expected = x.cpu().double() * gate.cpu().double().sigmoid()
    _close(sigmoid_mul(x, gate), expected)
    with pytest.raises(ValueError, match="shape"):
        sigmoid_mul(torch.empty(0, 16, device="npu"), torch.empty(0, 8, device="npu"))


@pytest.mark.parametrize("rows", [0, 1, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qwen35_gated_norm_hf_rounding(rows, dtype):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    torch.manual_seed(27)
    packed = torch.randn(rows, 4096, dtype=dtype, device="npu")
    x, gate = packed.split(2048, dim=-1)
    reference = Qwen3_5RMSNormGated(128).to(device="npu", dtype=dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.randn_like(reference.weight))
        expected = reference(x.reshape(rows, 16, 128), gate.reshape(rows, 16, 128))
        actual = qwen35_gated_rmsnorm(
            x, reference.weight, gate, eps=1e-6, group_size=128
        )
    # On one device the explicit HF boundaries must match exactly; a fused
    # final-only cast can pass a loose tolerance while changing model logits.
    torch.testing.assert_close(actual, expected.reshape(x.shape), atol=0, rtol=0)
    if rows and dtype == torch.bfloat16:
        fused = gated_rmsnorm(
            x,
            reference.weight,
            gate,
            eps=1e-6,
            group_size=128,
            norm_before_gate=True,
            sigmoid_gate=False,
            fp8_scale=None,
            weights_independent=True,
        )
        assert torch.count_nonzero(fused != actual).item() > 0


@pytest.mark.parametrize("rows", [0, 1, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qwen35_sigmoid_gate_hf_rounding(rows, dtype):
    torch.manual_seed(28)
    x = torch.randn(rows, 2048, dtype=dtype, device="npu")
    packed = torch.randn(rows, 8, 512, dtype=dtype, device="npu")
    gate = packed[..., 256:]
    # The official attention has two statements at activation precision:
    # sigmoid(gate), followed by the output product.
    rounded_sigmoid = gate.sigmoid()
    expected = x * rounded_sigmoid.reshape(x.shape)
    fused = sigmoid_mul(x.clone(), gate)
    actual = qwen35_sigmoid_mul(x, gate)
    assert actual is x
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if rows and dtype == torch.bfloat16:
        assert torch.count_nonzero(fused != actual).item() > 0
