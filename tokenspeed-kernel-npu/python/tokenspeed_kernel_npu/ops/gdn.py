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

"""Eager Ascend Gated DeltaNet with FP32, K-last recurrent state.

This correctness baseline executes every tensor operation on the NPU. Prefill
reads sequence boundaries on the host; decode uses device indices. It does not
implement graph capture, speculative verification, or recurrent checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.thirdparty.qwen35_gdn import chunk_gdn


def _validate_inputs(q, k, v, state) -> None:
    if q.device.type != "npu":
        raise ValueError("Ascend GDN requires NPU tensors")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Ascend eager GDN requires --enforce-eager")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("GDN expects q/k [B,T,H,K] and v [B,T,Hv,V]")
    if q.shape[2] == 0 or q.shape[3] == 0 or v.shape[2] == 0 or v.shape[3] == 0:
        raise ValueError("head dimensions must be positive")
    if v.shape[:2] != q.shape[:2] or v.shape[2] % q.shape[2]:
        raise ValueError("value heads must be a multiple of query/key heads")
    if any(x.device != q.device for x in (k, v, state)):
        raise ValueError("GDN compute tensors must share the NPU device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("GDN activations must be BF16, FP16 or FP32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("GDN q/k/v must have the same dtype")
    if state.dtype != torch.float32:
        raise ValueError("Ascend eager GDN requires FP32 recurrent state")
    if state.ndim != 4 or state.shape[1:] != (v.shape[2], v.shape[3], q.shape[3]):
        raise ValueError("GDN state must be K-last [rows,Hv,V,K]")


def _prepare_qk(q, k, value_heads: int, *, normalize: bool):
    if normalize:
        # Match Transformers 5.12's torch chunk/recurrent entry points:
        # normalize before their FP32 cast, retaining each input-dtype result.
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = q.float(), k.float()
    repeats = value_heads // q.shape[2]
    if repeats != 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    return q, k


def _step(q, k, v, decay, beta, state, scale: float):
    # Internally use contiguous K-first state, matching the reference's sum(-2).
    decayed = state * decay[..., None, None]
    prediction = (decayed * k.unsqueeze(-1)).sum(-2)
    delta = (v - prediction) * beta.unsqueeze(-1)
    updated = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (updated * (q * scale).unsqueeze(-1)).sum(-2)
    return output, updated


def gdn_gating(A_log, a, dt_bias, *, beta: float, threshold: float):
    """Compute FP32 log decay with the reference's staged softplus operations."""
    if a.device.type != "npu" or any(x.device != a.device for x in (A_log, dt_bias)):
        raise ValueError("Ascend GDN gating requires tensors on one NPU")
    if a.ndim < 1 or A_log.shape != a.shape[-1:] or dt_bias.shape != A_log.shape:
        raise ValueError("GDN gating parameters must match the final head dimension")
    return -A_log.float().exp() * F.softplus(
        a.float() + dt_bias.float(), beta=beta, threshold=threshold
    )


@torch.no_grad()
def gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool,
    output_final_state: bool,
    output_h: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process packed sequences in 64-token blocks and return final states.

    Inputs q/k have shape [1,T,H,K], v [1,T,Hv,V], log decay g and beta
    [1,T,Hv], initial state [N,Hv,V,K], and boundaries [N+1]. The initial
    state is read-only. Empty sequences preserve it; trailing padded tokens
    produce zero output without updating state. Recurrent arithmetic uses
    FP32. Q/K normalization retains the input dtype, matching the Hugging Face
    torch reference; the output also rounds to the activation dtype.
    ``output_h`` must be False: intermediate speculative checkpoints are not
    part of this eager baseline.
    """
    _validate_inputs(q, k, v, initial_state)
    if output_h:
        raise NotImplementedError("Ascend eager GDN does not return checkpoints")
    if q.shape[0] != 1 or cu_seqlens.ndim != 1:
        raise ValueError(
            "prefill requires packed batch dimension 1 and vector boundaries"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("sequence boundaries must be integer tensors")
    if g.shape != v.shape[:3] or beta.shape != g.shape:
        raise ValueError("prefill gates must have shape [1,T,Hv]")
    if g.device != q.device or beta.device != q.device:
        raise ValueError("GDN gates must share the NPU device")
    boundaries = cu_seqlens.detach().cpu().tolist()
    if (
        len(boundaries) != initial_state.shape[0] + 1
        or not boundaries
        or boundaries[0] != 0
        or boundaries[-1] > q.shape[1]
        or any(a > b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("prefill boundaries must partition valid input tokens")
    qf, kf = _prepare_qk(q, k, v.shape[2], normalize=qk_l2norm)
    vf, gf, betaf = v.float(), g.float(), beta.float()
    output = torch.zeros_like(v)
    final_state = initial_state.clone()
    actual_scale = q.shape[-1] ** -0.5 if scale is None else scale
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        state = initial_state[request : request + 1].transpose(-1, -2).contiguous()
        value, state = chunk_gdn(
            qf[:, start:end],
            kf[:, start:end],
            vf[:, start:end],
            gf[:, start:end],
            betaf[:, start:end],
            state,
            scale=actual_scale,
        )
        output[:, start:end].copy_(value)
        final_state[request].copy_(state[0].transpose(-1, -2))
    return output, final_state if output_final_state else None


@torch.no_grad()
def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    output_state_indices: torch.Tensor | None,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """One decode token with independent read/write state-pool indices.

    q/k/v are [B,1,H,K]/[B,1,Hv,V]; a/b are [B,1,Hv]. A_log and dt_bias
    have Hv elements. The FP32 K-last pool is mutated only at nonnegative
    write indices. A negative read starts from zero and a negative write
    leaves the pool untouched. Distinct write indices are required, as in
    the scheduler's state ownership contract. Reads are gathered before any
    write, including when different requests exchange source/destination rows.
    """
    _validate_inputs(q, k, v, initial_state)
    if q.shape[1] != 1:
        raise ValueError("Ascend GDN decode supports exactly one token per request")
    batch, heads = q.shape[0], v.shape[2]
    if a.shape != (batch, 1, heads) or b.shape != a.shape:
        raise ValueError("decode gates must have shape [B,1,Hv]")
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError("A_log and dt_bias must have shape [Hv]")
    writes = (
        initial_state_indices if output_state_indices is None else output_state_indices
    )
    for indices in (initial_state_indices, writes):
        if indices.shape != (batch,) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("decode state indices must be integer vectors of length B")
    if any(
        x.device != q.device
        for x in (A_log, a, dt_bias, b, initial_state_indices, writes)
    ):
        raise ValueError("GDN parameters, gates and indices must share the NPU device")
    if initial_state.shape[0] == 0:
        raise ValueError("decode needs a nonempty state pool")
    qf, kf = _prepare_qk(q, k, heads, normalize=use_qk_l2norm)
    reads = initial_state_indices.long()
    state = initial_state.index_select(0, reads.clamp_min(0))
    # where (rather than multiplication) also masks stale NaNs in the null row.
    state = torch.where(reads[:, None, None, None] >= 0, state, torch.zeros_like(state))
    state = state.transpose(-1, -2).contiguous()
    decay = gdn_gating(A_log, a[:, 0], dt_bias, beta=1.0, threshold=20.0).exp()
    beta = b[:, 0].sigmoid().float()
    output, updated = _step(
        qf[:, 0],
        kf[:, 0],
        v[:, 0].float(),
        decay,
        beta,
        state,
        q.shape[-1] ** -0.5 if scale is None else scale,
    )
    # Boolean selection is intentionally eager. It never redirects padding to
    # pool row zero, whose contents must remain unchanged for fresh requests.
    live = writes >= 0
    initial_state.index_copy_(0, writes[live].long(), updated[live].transpose(-1, -2))
    return output.unsqueeze(1).to(q.dtype)
