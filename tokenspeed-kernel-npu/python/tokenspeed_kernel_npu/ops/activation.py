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

"""Ascend fused activation operators."""

from functools import lru_cache

import torch
import torch_npu


def sigmoid_mul(
    x: torch.Tensor, gate: torch.Tensor, head_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Update contiguous ``[tokens, width]`` x with an FP32 sigmoid gate.

    Gate may be a row-strided 2D tensor or a 3D packed-projection head view.
    Optional bias contains one value per head. The result rounds once into x.
    """
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D tensor")
    if gate.ndim not in (2, 3) or gate.stride(-1) != 1:
        raise ValueError("gate must be 2D or 3D with unit inner stride")
    if x.dtype != gate.dtype or x.device != gate.device:
        raise ValueError("x and gate must share a dtype and device")
    gate_width = gate.shape[1] if gate.ndim == 2 else gate.shape[1] * gate.shape[2]
    if gate.shape[0] != x.shape[0] or gate_width != x.shape[1]:
        raise ValueError("gate shape must match x's tokens and width")
    values = gate.float()
    if head_bias is not None:
        if (
            head_bias.ndim != 1
            or not head_bias.is_contiguous()
            or head_bias.numel() == 0
            or head_bias.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or head_bias.device != x.device
        ):
            raise ValueError(
                "head_bias must be a floating contiguous vector on x's device"
            )
        heads = head_bias.numel()
        if x.shape[-1] % heads or (gate.ndim == 3 and gate.shape[1] != heads):
            raise ValueError("head_bias must contain one value per gate head")
        values = values.reshape(x.shape[0], heads, x.shape[-1] // heads)
        values = values + head_bias.float().reshape(1, heads, 1)
    if x.numel() == 0:
        return x
    result = x.float() * values.reshape(x.shape).sigmoid()
    x.copy_(result.to(x.dtype))
    return x


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply SwiGLU to last-dimension [gate, up], optionally writing ``out``."""
    if x.numel() == 0:
        if out is not None:
            return out
        return x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,))
    result = torch_npu.npu_swiglu(x, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def qwen35_silu_and_mul(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply Qwen3.5 SwiGLU with SiLU rounded before the up multiplication."""
    if x.ndim == 0 or x.shape[-1] % 2:
        raise ValueError("SwiGLU requires an even last dimension for [gate, up]")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Qwen3.5 SwiGLU requires BF16, FP16 or FP32 inputs")
    if x.device.type != "npu":
        raise ValueError("Qwen3.5 Ascend SwiGLU requires NPU inputs")
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    if out is not None and (
        out.shape != output_shape or out.dtype != x.dtype or out.device != x.device
    ):
        raise ValueError("out must match the output shape, dtype and device")
    if x.numel() == 0:
        return x.new_empty(output_shape) if out is None else out
    gate, up = x.chunk(2, dim=-1)
    # HF materializes SiLU in the activation dtype before the second product.
    # Native SwiGLU and an FP32 fused expression omit this rounding boundary.
    result = torch.nn.functional.silu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def qwen35_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply Qwen3.5's sigmoid rounded to gate dtype before multiplying x."""
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D tensor")
    if gate.ndim not in (2, 3) or gate.stride(-1) != 1:
        raise ValueError("gate must be 2D or 3D with unit inner stride")
    if x.dtype != gate.dtype or x.device != gate.device:
        raise ValueError("x and gate must share a dtype and device")
    gate_width = gate.shape[1] if gate.ndim == 2 else gate.shape[1] * gate.shape[2]
    if gate.shape[0] != x.shape[0] or gate_width != x.shape[1]:
        raise ValueError("gate shape must match x's tokens and width")
    x.mul_(gate.sigmoid().reshape(x.shape))
    return x


def dequant_silu_and_mul_quant(x, weight_scale, activation_scale):
    """Dequantize INT32 [gate,up], apply SwiGLU and dynamically quantize rows."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, scales = torch_npu.npu_dequant_swiglu_quant(
        x.reshape(-1, x.shape[-1]),
        weight_scale=weight_scale,
        activation_scale=activation_scale.reshape(-1),
        activate_left=True,
        quant_mode=1,
    )
    return (
        values.reshape(x.shape[:-1] + (x.shape[-1] // 2,)),
        scales.reshape(x.shape[:-1]),
    )


@lru_cache(maxsize=None)
def _unit_smooth_scale(device: torch.device, width: int) -> torch.Tensor:
    # CANN 9.0's SwiGluQuant reads this optional input unconditionally.
    # Keep the identity tensor alive across graph capture/replay; do not create
    # a Fill kernel on every activation call.
    return torch.ones((1, width), device=device, dtype=torch.float32)


def silu_and_mul_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused floating SwiGLU + INT8 quantization with dequantization scales."""
    if x.numel() == 0:
        return (
            x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=torch.int8),
            x.new_empty(x.shape[:-1], dtype=torch.float32),
        )
    values, inverse_scale = torch_npu.npu_swiglu_quant(
        x,
        smooth_scales=_unit_smooth_scale(x.device, x.shape[-1] // 2),
        activate_left=True,
        quant_mode=1,
    )
    # Unlike DynamicQuant/DequantSwigluQuant this op returns 127/amax.
    return values, inverse_scale.reciprocal()
