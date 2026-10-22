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

"""Causal convolution boundaries and rolling state on Ascend."""

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn, causal_conv1d_update

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")


def test_bfloat16_products_are_promoted_before_accumulation():
    # (129/128)^2 - 130/128 = 1/16384. A BF16-rounded product
    # incorrectly cancels to zero, even with a subsequent FP32 accumulator.
    channels = 64
    x = torch.tensor([1.0078125, 1.0, 0.0, 0.0], dtype=torch.bfloat16).repeat(
        channels, 1
    )
    weight = torch.tensor(
        [1.0078125, -1.015625, 0.0, 0.0], dtype=torch.bfloat16
    ).repeat(channels, 1)
    state = torch.zeros(2, channels, 3, dtype=torch.bfloat16, device="npu")
    indices = torch.tensor([1], dtype=torch.int32, device="npu")
    boundaries = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    metadata = build_causal_conv1d_prefill_metadata(boundaries, torch.tensor([4]), 64)
    output = causal_conv1d_fn(
        x.T.contiguous().npu().T,
        weight.npu(),
        None,
        state,
        boundaries,
        indices,
        torch.tensor([False], device="npu"),
        None,
        prefill_metadata=metadata,
    )
    expected = torch.full((channels,), 1 / 16384, dtype=torch.bfloat16)
    torch.testing.assert_close(output[:, -1].cpu(), expected, atol=0, rtol=0)
    state[1].copy_(x[:, :3])
    output = causal_conv1d_update(
        x[:, 3:].T.npu(),
        state,
        weight.npu(),
        activation=None,
        conv_state_indices=indices,
        parent_indices=None,
    )
    torch.testing.assert_close(output[0].cpu(), expected, atol=0, rtol=0)


def reference(x, w, initial):
    history = torch.cat((initial.float(), x.float()), dim=-1)
    result = sum(
        history[:, tap : tap + x.shape[1]] * w[:, tap : tap + 1].float()
        for tap in range(4)
    )
    # Official HF portable path materializes BF16 convolution before SiLU.
    return F.silu(result.to(x.dtype).float()).to(x.dtype), history[:, -3:].to(x.dtype)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 63, 64, 65, 128, 129])
@pytest.mark.parametrize("resumed", [False, True])
def test_prefill_and_decode_window(length, resumed):
    torch.manual_seed(42)
    channels = 6144
    x = torch.randn(channels, length).bfloat16()
    w = (torch.randn(channels, 4) * 0.2).bfloat16()
    initial = (
        torch.randn(channels, 3).bfloat16()
        if resumed
        else torch.zeros(channels, 3, dtype=torch.bfloat16)
    )
    state = torch.randn(4, channels, 3, dtype=torch.bfloat16, device="npu")
    state[1].copy_(initial)
    untouched = state[[0, 2, 3]].cpu().clone()
    cu = torch.tensor([0, length], dtype=torch.int32, device="npu")
    idx = torch.tensor([1], dtype=torch.int32, device="npu")
    metadata = build_causal_conv1d_prefill_metadata(cu, torch.tensor([length]), 64)
    # Runtime projects [T,C] and passes a transposed, noncontiguous [C,T].
    device_x = x.T.contiguous().npu().T
    actual = causal_conv1d_fn(
        device_x,
        w.npu(),
        None,
        state,
        cu,
        idx,
        torch.tensor([resumed], device="npu"),
        "silu",
        prefill_metadata=metadata,
    )
    expected, history = reference(x, w, initial)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(state[1].cpu(), history, atol=0, rtol=0)
    torch.testing.assert_close(state[[0, 2, 3]].cpu(), untouched, atol=0, rtol=0)
    for step in range(5):
        token = torch.randn(channels, 1).bfloat16()
        expected, history = reference(token, w, history)
        actual = causal_conv1d_update(
            token.T.npu(),
            state,
            w.npu(),
            activation="silu",
            conv_state_indices=idx,
            parent_indices=None,
        )
        torch.testing.assert_close(actual.cpu().T, expected, atol=0.01, rtol=0.01)
        torch.testing.assert_close(state[1].cpu(), history, atol=0, rtol=0)
    torch.testing.assert_close(state[[0, 2, 3]].cpu(), untouched, atol=0, rtol=0)
