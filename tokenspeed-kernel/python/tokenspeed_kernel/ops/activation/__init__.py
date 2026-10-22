# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    relu2,
)
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn


def sigmoid_mul(
    x: torch.Tensor, gate: torch.Tensor, head_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Multiply x in place by sigmoid(gate), optionally with per-head bias.

    x is contiguous ``[tokens, width]``; gate may be a strided
    ``[tokens, heads, head_dim]`` view or a matching 2D tensor.
    """
    if current_platform().is_npu:
        from tokenspeed_kernel.ops.activation.ascend import sigmoid_mul as impl
    else:
        from tokenspeed_kernel.ops.activation.triton import sigmoid_mul as impl
    return impl(x, gate, head_bias=head_bias)


def qwen35_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Ascend Qwen3.5 output gate with activation-dtype sigmoid rounding."""
    if not current_platform().is_npu:
        raise NotImplementedError("Qwen3.5 staged sigmoid gate is an Ascend entry")
    from tokenspeed_kernel.ops.activation.ascend import qwen35_sigmoid_mul as impl

    return impl(x, gate)


def qwen35_silu_and_mul(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Ascend Qwen3.5 SwiGLU with activation-dtype intermediate rounding.

    Args:
        x: NPU BF16, FP16 or FP32 tensor ``[..., gate | up]`` with even width.
            Leading dimensions and strides are preserved logically.
        out: Optional output with the same leading shape, half the last width,
            and the same dtype and device as x. Strided outputs are supported.

    Returns:
        ``SiLU(gate) * up`` with SiLU rounded to x.dtype before multiplication,
        written into and returning out when supplied. The input is not changed
        unless the caller explicitly aliases it with out.
    """
    if not current_platform().is_npu:
        raise NotImplementedError("Qwen3.5 staged SwiGLU is an Ascend entry")
    from tokenspeed_kernel.ops.activation.ascend import qwen35_silu_and_mul as impl

    return impl(x, out)


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    Positive ``limit`` values use the portable Triton implementation because
    the CUDA implementation does not expose the checkpoint's clamp semantics.

    """
    if current_platform().is_npu and limit is None:
        from tokenspeed_kernel.ops.activation.ascend import silu_and_mul as npu_swiglu

        return npu_swiglu(x, out)
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=pdl_enabled())


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
    )


__all__ = [
    "qwen35_sigmoid_mul",
    "qwen35_silu_and_mul",
    "add3",
    "relu2",
    "sigmoid_mul",
    "silu_and_mul",
    "situ_and_mul",
]
