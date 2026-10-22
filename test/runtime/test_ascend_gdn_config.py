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

"""Qwen GDN rejects unsupported Ascend modes before cache allocation."""

import pytest

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5Config
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture
def launch(tmp_path):
    checkpoint = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config={
            "num_hidden_layers": 4,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        },
    )
    checkpoint.save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        language_model_only=True,
        enforce_eager=True,
        mamba_ssm_dtype="float32",
    )
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        dtype=args.dtype,
        server_args=args,
    )
    # This is boot-time validation only: changing the resolved launch device
    # exercises the NPU contract without allocating NPU tensors.
    args.device = "npu"
    return args, model


def test_ascend_gdn_accepts_eager_fp32_state(launch):
    args, model = launch
    spec = LinearAttnConfig.generate(args, model)
    assert spec.layer_ids == (0, 1, 2)
    assert spec.temporal_state_shape == (16, 128, 128)


def test_ascend_gdn_rejects_graph_mode(launch):
    args, model = launch
    args.enforce_eager = False
    with pytest.raises(ValueError, match="requires --enforce-eager"):
        LinearAttnConfig.generate(args, model)


@pytest.mark.parametrize("algorithm", ["MTP", "EAGLE3", "DUMMY"])
def test_ascend_gdn_rejects_speculative_decoding(launch, algorithm):
    args, model = launch
    args.speculative_algorithm = algorithm
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        LinearAttnConfig.generate(args, model)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_ascend_gdn_rejects_non_fp32_state(launch, dtype):
    args, model = launch
    args.mamba_ssm_dtype = dtype
    with pytest.raises(ValueError, match="requires --mamba-ssm-dtype float32"):
        LinearAttnConfig.generate(args, model)


def test_other_devices_keep_existing_gdn_modes(launch):
    args, model = launch
    args.device = "cuda"
    args.enforce_eager = False
    args.speculative_algorithm = "MTP"
    args.mamba_ssm_dtype = "bfloat16"
    assert LinearAttnConfig.generate(args, model).temporal_state_shape == (16, 128, 128)


def test_full_attention_only_has_no_gdn_restrictions(launch):
    args, model = launch
    model.hf_text_config.full_attention_interval = 1
    args.enforce_eager = False
    args.speculative_algorithm = "MTP"
    args.mamba_ssm_dtype = "bfloat16"
    assert LinearAttnConfig.generate(args, model) is None
