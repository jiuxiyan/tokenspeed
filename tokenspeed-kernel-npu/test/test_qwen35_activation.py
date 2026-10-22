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

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.activation import qwen35_silu_and_mul

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Qwen3.5 SwiGLU tests require an NPU"
)


def _reference(x):
    gate, up = x.cpu().double().chunk(2, dim=-1)
    rounded_silu = (gate * gate.sigmoid()).to(x.dtype).double()
    return (rounded_silu * up).to(x.dtype)


def _assert_reference(actual, x):
    # Independent FP64 arithmetic preserves the required intermediate rounding.
    tolerance = 8e-3 if x.dtype == torch.bfloat16 else 1e-3
    if x.dtype == torch.float32:
        tolerance = 3e-6
    torch.testing.assert_close(actual.cpu(), _reference(x), rtol=tolerance, atol=1e-6)
    # HF executes the SiLU primitive in activation dtype on the same device.
    gate, up = x.chunk(2, dim=-1)
    hf = torch.nn.SiLU()(gate) * up
    torch.testing.assert_close(actual, hf, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape", [(0, 7168), (1, 7168), (7, 7168), (65, 32), (2, 3, 64)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_qwen35_swiglu_shapes_and_dtypes(shape, dtype):
    torch.manual_seed(20261022)
    x = torch.randn(shape, dtype=dtype, device="npu")
    before = x.clone()
    result = qwen35_silu_and_mul(x)
    assert result.shape == x.shape[:-1] + (x.shape[-1] // 2,)
    assert result.dtype == x.dtype and result.device == x.device
    _assert_reference(result, x)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["rows", "columns", "transpose"])
def test_qwen35_swiglu_strided_input_and_output(layout):
    torch.manual_seed(28)
    if layout == "rows":
        x = torch.randn(14, 7168, dtype=torch.bfloat16, device="npu")[::2]
    elif layout == "columns":
        x = torch.randn(7, 14336, dtype=torch.bfloat16, device="npu")[:, ::2]
    else:
        x = torch.randn(7168, 7, dtype=torch.bfloat16, device="npu").T
    before = x.clone()
    storage = torch.full((7, 7168), 19.0, dtype=x.dtype, device=x.device)
    out = storage[:, ::2]
    result = qwen35_silu_and_mul(x, out=out)
    assert result is out
    _assert_reference(result, x)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(
        storage[:, 1::2], torch.full_like(out, 19), rtol=0, atol=0
    )


def test_qwen35_swiglu_runtime_preserves_intermediate_bf16_rounding():
    from tokenspeed.runtime.layers.activation import Qwen35SiluAndMul

    torch.manual_seed(29)
    x = torch.randn(7, 7168, dtype=torch.bfloat16, device="npu")
    result = Qwen35SiluAndMul()(x)
    _assert_reference(result, x)
    gate, up = x.chunk(2, dim=-1)
    fused = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    # A broad allclose alone would permit accidentally restoring fused FP32.
    assert torch.count_nonzero(result != fused).item() > 0


def test_qwen35_swiglu_empty_out_and_input_alias():
    x = torch.empty(0, 7168, dtype=torch.bfloat16, device="npu")
    out = torch.empty(0, 3584, dtype=x.dtype, device=x.device)
    assert qwen35_silu_and_mul(x, out=out) is out
    x = torch.linspace(-3, 3, 128, device="npu").bfloat16().reshape(2, 64)
    expected = qwen35_silu_and_mul(x)
    alias = x[:, :32]
    assert qwen35_silu_and_mul(x, out=alias) is alias
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["odd", "scalar", "integer", "out_shape", "out_dtype"])
def test_qwen35_swiglu_rejects_invalid_contract(kind):
    x = torch.ones(2, 64, dtype=torch.bfloat16, device="npu")
    out = None
    if kind == "odd":
        x = x[:, :-1]
    elif kind == "scalar":
        x = x[0, 0]
    elif kind == "integer":
        x = x.long()
    elif kind == "out_shape":
        out = torch.empty_like(x)
    else:
        out = torch.empty(2, 32, dtype=torch.float32, device="npu")
    with pytest.raises(ValueError):
        qwen35_silu_and_mul(x, out=out)
