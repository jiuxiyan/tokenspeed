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

"""Register the eager Ascend GDN baseline behind the public kernel API."""

import torch
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode, gdn_gating, gdn_prefill

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_eager_gdn_chunk_prefill",
        solution="torch_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16, torch.float32}
        ),
        priority=Priority.PORTABLE,
        traits={"output_h": frozenset({False}), "qk_l2norm": frozenset({False, True})},
    )
    def ascend_eager_gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        *,
        scale,
        initial_state,
        cu_seqlens,
        qk_l2norm,
        output_final_state,
        output_h,
    ):
        output, final_state = gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
            output_h=output_h,
        )
        return GdnChunkPrefillResult(out=output, final_state=final_state)

    register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_eager_gdn_decode_step",
        solution="torch_npu",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16, torch.float32}
        ),
        priority=Priority.PORTABLE,
        traits={"strided_qkv": frozenset({False, True})},
    )(gdn_decode)
