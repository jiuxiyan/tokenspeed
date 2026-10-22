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

"""Explicit, instance-local recurrent-cache precision for the HF reference.

Only recurrent-state allocation changes. Official HF model equations,
conv-state updates, KV-cache updates, and recurrent-state copy remain intact.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from transformers.utils import is_torchdynamo_compiling


class Float32RecurrentCacheLayer(LinearAttentionLayer):
    """Keep conv state in the model dtype and recurrent state in FP32."""

    def lazy_initialization(self, conv_states=None, recurrent_states=None) -> None:
        if conv_states is not None:
            super().lazy_initialization(conv_states=conv_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(
                recurrent_states, dtype=torch.float32, device=recurrent_states.device
            )
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.recurrent_states)
            self.is_recurrent_states_initialized = True


def make_reference_cache(config, *, state_dtype: str) -> DynamicCache:
    """Create an independent official cache with an explicit state policy.

    ``model`` preserves stock HF cache allocation. ``float32`` replaces only
    this cache instance's linear-attention layers before the first forward,
    preserving the first FP32 final state without an intervening BF16 cast.
    """
    if state_dtype not in ("model", "float32"):
        raise ValueError(f"unknown reference recurrent-state dtype: {state_dtype}")
    cache = DynamicCache(config=config)
    if state_dtype == "float32":
        for index, layer in enumerate(cache.layers):
            if type(layer) is LinearAttentionLayer:
                cache.layers[index] = Float32RecurrentCacheLayer(config)
            elif isinstance(layer, LinearAttentionLayer):
                raise TypeError(
                    "mixed linear/full cache layers require a dedicated precision policy"
                )
    return cache


def cache_dtype_manifest(
    cache, *, state_dtype: str, model_dtype: torch.dtype
) -> list[dict]:
    """Check actual initialized cache tensors and report every layer's dtype."""
    expected_state = torch.float32 if state_dtype == "float32" else model_dtype
    records = []
    for index, layer in enumerate(cache.layers):
        record = {"layer": index}
        if isinstance(layer, LinearAttentionLayer):
            if (
                not layer.is_conv_states_initialized
                or not layer.is_recurrent_states_initialized
            ):
                raise AssertionError(f"linear cache layer {index} was not initialized")
            if (
                layer.conv_states.dtype != model_dtype
                or layer.recurrent_states.dtype != expected_state
            ):
                raise AssertionError(
                    f"linear cache layer {index} has unexpected dtypes"
                )
            record.update(
                conv=str(layer.conv_states.dtype),
                recurrent=str(layer.recurrent_states.dtype),
            )
        else:
            if layer.keys.dtype != model_dtype or layer.values.dtype != model_dtype:
                raise AssertionError(f"KV cache layer {index} has unexpected dtypes")
            record.update(key=str(layer.keys.dtype), value=str(layer.values.dtype))
        records.append(record)
    return records
