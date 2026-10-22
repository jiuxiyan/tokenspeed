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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Normalize with FP32 ``1 + weight`` and update an optional residual.

    The residual sum is rounded to the activation dtype before normalization,
    matching the unfused model boundary. All arithmetic stays on x's device.
    """
    if x.numel() == 0:
        result = torch.empty_like(x)
        return result if residual is None else (result, residual)
    if residual is not None:
        residual.copy_(x + residual)
        values = residual.float()
    else:
        values = x.float()
    scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    result = (values * scale * (1.0 + weight.float())).to(x.dtype)
    return result if residual is None else (result, residual)


def gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """FP32 grouped norm and gate, rounded once to the activation dtype.

    Weight may span the whole row or be shared across groups. NPU operations
    use the current stream; ``weights_independent`` only affects CUDA PDL.
    """
    del weights_independent
    if fp8_scale is not None:
        raise ValueError("Ascend gated RMSNorm does not support FP8 output")
    width = x.shape[-1]
    if group_size <= 0 or width % group_size:
        raise ValueError("input width must be divisible by a positive group_size")
    if weight.shape not in ((width,), (group_size,)):
        raise ValueError("weight must span the row or one normalization group")
    if z is not None and z.shape != x.shape:
        raise ValueError("gate shape must match the input")
    if x.numel() == 0:
        return torch.empty_like(x)
    grouped_shape = x.shape[:-1] + (width // group_size, group_size)
    values = x.float().reshape(grouped_shape)
    gate = None
    if z is not None:
        gate_input = z.float().reshape(grouped_shape)
        gate = gate_input.sigmoid()
        if not sigmoid_gate:
            gate = gate_input * gate
        if not norm_before_gate:
            values = values * gate
    scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    result = (values * scale).reshape(x.shape)
    affine = weight.float()
    if weight.numel() != width:
        affine = affine.repeat(width // group_size)
    result = result * affine
    if gate is not None and norm_before_gate:
        result = result * gate.reshape(x.shape)
    return result.to(x.dtype)


def qwen35_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    *,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """Qwen3.5 HF norm with activation-dtype norm and affine boundaries.

    The normalized input rounds before multiplying the weight, and that
    product rounds before the FP32 SiLU gate. These two boundaries differ
    from the generic fused gated RMSNorm contract.
    """
    if group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("input width must be divisible by a positive group_size")
    if weight.shape != (group_size,) or z.shape != x.shape:
        raise ValueError("Qwen3.5 requires per-head weights and a matching gate")
    if x.numel() == 0:
        return torch.empty_like(x)
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    affine = normalized.to(x.dtype) * weight
    gated = affine.reshape(x.shape) * torch.nn.functional.silu(z.float())
    return gated.to(x.dtype)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to query and key tensors."""
    del enable_pdl
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_shape, k_shape = q.shape, k.shape
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    q_scale = torch.rsqrt(q_heads.float().square().mean(-1, keepdim=True) + eps)
    k_scale = torch.rsqrt(k_heads.float().square().mean(-1, keepdim=True) + eps)
    q_out = (q_heads * q_scale.to(q.dtype) * q_weight).reshape(q_shape)
    k_out = (k_heads * k_scale.to(k.dtype) * k_weight).reshape(k_shape)
    return q_out, k_out


__all__ = [
    "gated_rmsnorm",
    "gemma_rmsnorm",
    "qk_rmsnorm",
    "qwen35_gated_rmsnorm",
    "rmsnorm",
]
