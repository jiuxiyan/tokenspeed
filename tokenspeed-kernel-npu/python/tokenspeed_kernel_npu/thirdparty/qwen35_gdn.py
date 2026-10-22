# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 LightSeek Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapt the Transformers 5.12.0 Qwen3.5 torch chunk GDN operation order.

Source: transformers/models/qwen3_5/modeling_qwen3_5.py at v5.12.0,
``torch_chunk_gated_delta_rule``. Changes: normalization and input dtype
conversion belong to the caller; scale is explicit, chunks are fixed at 64,
and this helper always returns FP32 output and a K-first final state.
There is no Transformers runtime dependency. See THIRDPARTYNOTICES.
"""

import torch
import torch.nn.functional as F


def chunk_gdn(query, key, value, g, beta, initial_state, *, scale: float):
    """Evaluate one nonempty sequence using FP32 [B,H,K,V] state."""
    chunk_size = 64
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous() for tensor in (query, key, value, beta, g)
    ]
    sequence_length = query.shape[2]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        tensor.reshape(
            tensor.shape[0], tensor.shape[1], -1, chunk_size, tensor.shape[-1]
        )
        for tensor in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # Preserve the reference's triangular solve and its FP32 reduction order.
    # The 63 row updates run across all heads and chunks together.
    for row_index in range(1, chunk_size):
        row = attn[..., row_index, :row_index].clone()
        sub = attn[..., :row_index, :row_index].clone()
        attn[..., row_index, :row_index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = initial_state
    output = torch.zeros_like(value)
    for chunk in range(query.shape[2]):
        q_i, k_i, v_i = query[:, :, chunk], key[:, :, chunk], value[:, :, chunk]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, chunk]
        v_prime = k_cumdecay[:, :, chunk] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, chunk, :, None].exp()) @ state
        output[:, :, chunk] = attn_inter + attn @ v_new
        state = (
            state * g[:, :, chunk, -1, None, None].exp()
            + (
                k_i * (g[:, :, chunk, -1, None] - g[:, :, chunk]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    output = output.reshape(output.shape[0], output.shape[1], -1, output.shape[-1])
    return output[:, :, :sequence_length].transpose(1, 2).contiguous(), state
