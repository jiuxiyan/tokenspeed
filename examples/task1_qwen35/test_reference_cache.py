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

"""CPU checks of explicit official-cache allocation; no NPU or weights needed."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "requires PyTorch and Transformers",
)
class ReferenceCacheTests(unittest.TestCase):
    def make_config(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        return Qwen3_5TextConfig(
            hidden_size=32,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )

    def test_model_policy_preserves_stock_bf16_recurrent_rounding(self):
        import torch
        from transformers.cache_utils import DynamicCache

        from examples.task1_qwen35.reference_cache import make_reference_cache

        config = self.make_config()
        reference = make_reference_cache(config, state_dtype="model")
        stock = DynamicCache(config=config)
        conv = torch.randn(1, 16, 4).bfloat16()
        state = torch.randn(1, 2, 8, 8)
        for cache in (reference, stock):
            cache.update_conv_state(conv, 0)
            cache.update_recurrent_state(state, 0)
            self.assertEqual(cache.layers[0].recurrent_states.dtype, torch.bfloat16)
            torch.testing.assert_close(
                cache.layers[0].recurrent_states, state.bfloat16(), rtol=0, atol=0
            )
        torch.testing.assert_close(
            reference.layers[0].recurrent_states,
            stock.layers[0].recurrent_states,
            rtol=0,
            atol=0,
        )

    def test_float32_policy_retains_first_and_later_state_without_touching_conv_kv(
        self,
    ):
        import torch

        from examples.task1_qwen35.reference_cache import (
            cache_dtype_manifest,
            make_reference_cache,
        )

        cache = make_reference_cache(self.make_config(), state_dtype="float32")
        conv = torch.arange(64).reshape(1, 16, 4).bfloat16()
        first_state = torch.full((1, 2, 8, 8), 1.0 + 2.0**-12)
        cache.update_conv_state(conv, 0)
        cache.update_recurrent_state(first_state, 0)
        self.assertEqual(cache.layers[0].recurrent_states.dtype, torch.float32)
        torch.testing.assert_close(
            cache.layers[0].recurrent_states, first_state, rtol=0, atol=0
        )
        self.assertFalse(
            torch.equal(
                cache.layers[0].recurrent_states, first_state.bfloat16().float()
            )
        )
        address = cache.layers[0].recurrent_states.data_ptr()
        next_state = first_state + 2.0**-14
        cache.update_recurrent_state(next_state, 0)
        self.assertEqual(cache.layers[0].recurrent_states.data_ptr(), address)
        torch.testing.assert_close(
            cache.layers[0].recurrent_states, next_state, rtol=0, atol=0
        )
        torch.testing.assert_close(cache.layers[0].conv_states, conv, rtol=0, atol=0)
        key, value = (
            torch.randn(1, 2, 3, 8).bfloat16(),
            torch.randn(1, 2, 3, 8).bfloat16(),
        )
        cache.update(key, value, 1)
        torch.testing.assert_close(cache.layers[1].keys, key, rtol=0, atol=0)
        torch.testing.assert_close(cache.layers[1].values, value, rtol=0, atol=0)
        audit = cache_dtype_manifest(
            cache, state_dtype="float32", model_dtype=torch.bfloat16
        )
        self.assertEqual(
            audit[0],
            {"layer": 0, "conv": "torch.bfloat16", "recurrent": "torch.float32"},
        )
        self.assertEqual(
            audit[1], {"layer": 1, "key": "torch.bfloat16", "value": "torch.bfloat16"}
        )

    def test_policies_are_independent_instances_and_do_not_change_global_mapping(self):
        import torch
        from transformers.cache_utils import DynamicCache, LinearAttentionLayer

        from examples.task1_qwen35.reference_cache import make_reference_cache

        config = self.make_config()
        fp32 = make_reference_cache(config, state_dtype="float32")
        model = make_reference_cache(config, state_dtype="model")
        fresh_stock = DynamicCache(config=config)
        self.assertIs(type(model.layers[0]), LinearAttentionLayer)
        self.assertIs(type(fresh_stock.layers[0]), LinearAttentionLayer)
        conv = torch.zeros(1, 16, 4, dtype=torch.bfloat16)
        for cache in (fp32, model, fresh_stock):
            cache.update_conv_state(conv, 0)
            cache.update_recurrent_state(torch.ones(1, 2, 8, 8), 0)
        fp32.layers[0].recurrent_states.zero_()
        self.assertTrue(torch.all(model.layers[0].recurrent_states == 1))
        self.assertTrue(torch.all(fresh_stock.layers[0].recurrent_states == 1))
        self.assertEqual(model.layers[0].recurrent_states.dtype, torch.bfloat16)
        self.assertEqual(fresh_stock.layers[0].recurrent_states.dtype, torch.bfloat16)

    def test_unknown_policy_is_rejected(self):
        from examples.task1_qwen35.reference_cache import make_reference_cache

        with self.assertRaises(ValueError):
            make_reference_cache(self.make_config(), state_dtype="automatic")


if __name__ == "__main__":
    unittest.main()
