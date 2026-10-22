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

"""Ascend rotary embedding operator."""

from __future__ import annotations

import torch
import torch_npu


def apply_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
) -> None:
    """Apply RoPE to Q and K, writing into outputs or the inputs in place.

    Qwen3.5's BF16 head256/rotary64 NeoX geometry preserves the two BF16
    products and their BF16 sum, matching the model's explicit HF boundary.
    Other geometries retain the native fused operator.
    """
    if (
        head_size == 256
        and cos_sin_cache.shape[-1] == 64
        and q.dtype == k.dtype == torch.bfloat16
        and is_neox
    ):
        if positions.numel() == 0:
            return
        table = cos_sin_cache[positions.long()].to(q.dtype)
        cos_half, sin_half = table.chunk(2, dim=-1)
        cosine = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
        sine = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)

        def rotate(values: torch.Tensor) -> torch.Tensor:
            heads = values.reshape(positions.numel(), -1, head_size)
            rotary = heads[..., :64]
            rotate_half = torch.cat((-rotary[..., 32:], rotary[..., :32]), dim=-1)
            result = rotary * cosine + rotate_half * sine
            return torch.cat((result, heads[..., 64:]), dim=-1).reshape(values.shape)

        q_out, k_out = rotate(q), rotate(k)
        (q if q_rope_out is None else q_rope_out).copy_(q_out)
        (k if k_rope_out is None else k_rope_out).copy_(k_out)
        return
    q_out, k_out = torch_npu.npu_mrope(
        positions,
        q,
        k,
        cos_sin_cache.to(q.dtype),
        head_size,
        mrope_section=[0, 0, 0],
        rotary_mode="half" if is_neox else "interleave",
    )
    (q if q_rope_out is None else q_rope_out).copy_(q_out)
    (k if k_rope_out is None else k_rope_out).copy_(k_out)


__all__ = ["apply_rope"]
