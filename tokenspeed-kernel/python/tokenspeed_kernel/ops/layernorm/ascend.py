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

"""Ascend layernorm kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_npu.ops.layernorm import gated_rmsnorm as _npu_gated_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import gemma_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm as _npu_qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import qwen35_gated_rmsnorm, rmsnorm

__all__ = [
    "ascend_gated_rmsnorm",
    "gemma_rmsnorm",
    "qk_rmsnorm",
    "qwen35_gated_rmsnorm",
    "rmsnorm",
]


@register_kernel(
    "layernorm",
    "gated_rmsnorm",
    name="ascend_gated_rmsnorm",
    solution="torch_npu",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
    traits={"fp8_out": frozenset({False})},
)
def ascend_gated_rmsnorm(
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
    """Ascend implementation of the public grouped gated RMSNorm contract."""
    return _npu_gated_rmsnorm(
        x,
        weight,
        z,
        eps,
        group_size,
        norm_before_gate,
        sigmoid_gate,
        fp8_scale,
        weights_independent,
    )


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head norm; Gemma's weight offset is formed and applied in FP32."""
    if weight_offset:
        head_dim = q_weight.numel()
        if k_weight.numel() != head_dim or head_dim == 0:
            raise ValueError(
                "q and k weights must have the same positive head dimension"
            )
        if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
            raise ValueError("q and k widths must be divisible by the head dimension")
        if q.numel() == 0 and k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)
        results = []
        for values, weight in ((q, q_weight), (k, k_weight)):
            heads = values.float().reshape(
                *values.shape[:-1], values.shape[-1] // head_dim, head_dim
            )
            scale = torch.rsqrt(heads.square().mean(-1, keepdim=True) + eps)
            results.append(
                (heads * scale * (weight.float() + weight_offset))
                .reshape(values.shape)
                .to(values.dtype)
            )
        return results[0], results[1]
    return _npu_qk_rmsnorm(q, k, q_weight, k_weight, eps)
