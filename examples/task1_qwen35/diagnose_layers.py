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

"""Read-only layer localization on a fixed TokenSpeed trajectory.

Diagnostic only: hooks copy intermediate tensors to CPU and are unsuitable
for timing. No production implementation or reference arithmetic is changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_installed = False


def install_engine_hooks():
    global _installed
    if _installed:
        return
    _installed = True
    import torch
    import torch_npu

    import tokenspeed.runtime.models.qwen3_5 as ts
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    directory = Path(os.environ["TASK1_LAYER_DIAG_DIR"])
    selected = set(json.loads(os.environ["TASK1_LAYER_DIAG_STEPS"]))
    state = {"active": False, "layer": -1, "values": {}}
    registered = False
    step = -1
    original_forward = ModelRunner.forward
    original_input = ts._input_norm
    original_post = ts._post_attn_norm
    original_gated = ts.qwen35_gated_rmsnorm
    original_fia = torch_npu.npu_fused_infer_attention_score
    original_rope = torch_npu.npu_mrope
    import tokenspeed_kernel.ops.embedding.ascend as rope_registration
    import tokenspeed_kernel_npu.ops.mha as npu_mha

    import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba

    original_staged_attention = getattr(npu_mha, "_qwen35_attention", None)
    original_leaf_rope = rope_registration._apply_rope
    original_gdn_prefill = mamba.gdn_chunk_prefill
    original_gdn_decode = mamba.gdn_decode_step

    def save(key, value):
        if state["active"]:
            if isinstance(value, tuple):
                value = value[0]
            state["values"][key] = value.detach().cpu().clone()

    def input_norm(manager, norm, hidden, residual, *args, **kwargs):
        state["layer"] = manager.layer_id
        out = original_input(manager, norm, hidden, residual, *args, **kwargs)
        save(f"layers.{manager.layer_id}.input", out[2])
        save(f"layers.{manager.layer_id}.input_norm", out[0])
        return out

    def post_norm(manager, norm, hidden, residual, *args, **kwargs):
        save(f"layers.{manager.layer_id}.attn", hidden)
        out = original_post(manager, norm, hidden, residual, *args, **kwargs)
        save(f"layers.{manager.layer_id}.post_residual", out[2])
        save(f"layers.{manager.layer_id}.post_norm", out[0])
        return out

    def gated(x, weight, z, **kwargs):
        save(f"layers.{state['layer']}.gdn_norm_input", x)
        save(f"layers.{state['layer']}.gdn_gate", z)
        out = original_gated(x, weight, z, **kwargs)
        save(f"layers.{state['layer']}.gdn_norm", out)
        return out

    def attention(query, key, value, **kwargs):
        if state["active"]:
            prefix = f"layers.{state['layer']}"
            heads, kv_heads = kwargs["num_heads"], kwargs["num_key_value_heads"]
            dimension = (
                query.numel() // (query.shape[0] * heads)
                if kwargs["input_layout"] == "TND"
                else query.shape[-1] // heads
            )
            save(prefix + ".mha_q", query.reshape(-1, heads, dimension))
            if "block_table" in kwargs:
                length = int(kwargs["actual_seq_lengths_kv"][0])
                page_size = kwargs["block_size"]
                pages = kwargs["block_table"][
                    0, : (length + page_size - 1) // page_size
                ].long()
                keys = key[pages].reshape(-1, kv_heads, dimension)[:length]
                values = value[pages].reshape(-1, kv_heads, dimension)[:length]
            else:
                keys = key.reshape(-1, kv_heads, dimension)
                values = value.reshape(-1, kv_heads, dimension)
            save(prefix + ".mha_k", keys)
            save(prefix + ".mha_v", values)
        out = original_fia(query, key, value, **kwargs)
        save(f"layers.{state['layer']}.mha_output", out[0])
        return out

    def rope(positions, q, k, table, head_size, **kwargs):
        prefix = f"layers.{state['layer']}"
        save(prefix + ".rope_q_input", q)
        save(prefix + ".rope_k_input", k)
        if state["active"]:
            # Text positions are one shared sequence; retain only used rows.
            text_positions = positions if positions.ndim == 1 else positions[0]
            save(prefix + ".rope_table", table[text_positions.reshape(-1).long()])
        out = original_rope(positions, q, k, table, head_size, **kwargs)
        save(prefix + ".rope_q_output", out[0])
        save(prefix + ".rope_k_output", out[1])
        return out

    def staged_attention(q, k, v, **kwargs):
        prefix = f"layers.{state['layer']}"
        save(prefix + ".mha_q", q)
        save(prefix + ".mha_k", k)
        save(prefix + ".mha_v", v)
        out = original_staged_attention(q, k, v, **kwargs)
        save(prefix + ".mha_output", out)
        return out

    def leaf_rope(**kwargs):
        prefix = f"layers.{state['layer']}"
        save(prefix + ".rope_q_input", kwargs["q"])
        save(prefix + ".rope_k_input", kwargs["k"])
        if state["active"]:
            table = kwargs["cos_sin_cache"][kwargs["positions"].long()]
            save(prefix + ".rope_table_fp32", table)
            save(prefix + ".rope_table", table.to(kwargs["q"].dtype))
        original_leaf_rope(**kwargs)
        save(
            prefix + ".rope_q_output",
            kwargs["q"] if kwargs.get("q_rope_out") is None else kwargs["q_rope_out"],
        )
        save(
            prefix + ".rope_k_output",
            kwargs["k"] if kwargs.get("k_rope_out") is None else kwargs["k_rope_out"],
        )

    def capture_gdn(q, k, v, g, beta, initial):
        prefix = f"layers.{state['layer']}"
        for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            save(prefix + ".gdn_" + name, value)
        save(prefix + ".gdn_initial_state", initial.transpose(-1, -2))

    def gdn_prefill(q, k, v, g, beta, **kwargs):
        capture_gdn(q, k, v, g, beta, kwargs["initial_state"])
        out = original_gdn_prefill(q, k, v, g, beta, **kwargs)
        save(
            f"layers.{state['layer']}.gdn_final_state",
            out.final_state.transpose(-1, -2),
        )
        save(f"layers.{state['layer']}.gdn_core_output", out.out)
        return out

    def gdn_decode(**kwargs):
        if state["active"]:
            pool = kwargs["initial_state"]
            reads = kwargs["initial_state_indices"].long()
            initial = pool[reads.clamp_min(0)]
            initial = torch.where(
                (reads >= 0)[:, None, None, None], initial, torch.zeros_like(initial)
            )
            g = -kwargs["A_log"].float().exp() * torch.nn.functional.softplus(
                kwargs["a"].float() + kwargs["dt_bias"].float()
            )
            capture_gdn(
                kwargs["q"], kwargs["k"], kwargs["v"], g, kwargs["b"].sigmoid(), initial
            )
            save(f"layers.{state['layer']}.gdn_a", kwargs["a"])
            save(f"layers.{state['layer']}.gdn_b", kwargs["b"])
        out = original_gdn_decode(**kwargs)
        if state["active"]:
            writes = kwargs["output_state_indices"].long()
            save(
                f"layers.{state['layer']}.gdn_final_state",
                pool[writes].transpose(-1, -2),
            )
            save(f"layers.{state['layer']}.gdn_core_output", out)
        return out

    def output_hook(key):
        return lambda module, args, output: save(key, output)

    def input_hook(key):
        return lambda module, args: save(key, args[0])

    def layer_hook(key):
        def capture(module, args, output):
            if state["active"]:
                save(key, output[0] + output[1])

        return capture

    def forward(self, ctx, ids, positions, *args, **kwargs):
        nonlocal step, registered
        if ctx.forward_mode.is_idle() or not (directory / "request_active").exists():
            return original_forward(self, ctx, ids, positions, *args, **kwargs)
        if not registered:
            model = self.model.model
            model.norm.register_forward_hook(output_hook("final_norm"))
            for i, layer in enumerate(model.layers):
                prefix = f"layers.{i}"
                layer.register_forward_hook(layer_hook(prefix + ".output"))
                for name, key in (
                    ("gate_up_proj", "gate_up"),
                    ("act_fn", "mlp_activation"),
                    ("down_proj", "mlp"),
                ):
                    getattr(layer.mlp, name).register_forward_hook(
                        output_hook(prefix + "." + key)
                    )
                layer.mlp.gate_up_proj.register_forward_pre_hook(
                    input_hook(prefix + ".mlp_input")
                )
                attn = layer.linear_attn if hasattr(layer, "linear_attn") else layer
                for name in (
                    "in_proj_qkvzba",
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "qkv_proj",
                ):
                    if hasattr(attn, name):
                        getattr(attn, name).register_forward_hook(
                            output_hook(prefix + "." + name)
                        )
            registered = True
        if ctx.forward_mode.is_extend():
            if step >= 0:
                raise RuntimeError("layer diagnostics accepts one request")
            step = 0
        else:
            step += 1
        state["active"] = step in selected
        state["values"] = {}
        out = original_forward(self, ctx, ids, positions, *args, **kwargs)
        if state["active"]:
            torch.save(
                {
                    "input_ids": ids.cpu(),
                    "positions": positions.cpu(),
                    "values": state["values"],
                    "logits": out.next_token_logits.cpu(),
                },
                directory / f"engine_step{step:03d}.pt",
            )
        state["active"] = False
        return out

    ts._input_norm = input_norm
    ts._post_attn_norm = post_norm
    ts.qwen35_gated_rmsnorm = gated
    torch_npu.npu_fused_infer_attention_score = attention
    torch_npu.npu_mrope = rope
    if original_staged_attention is not None:
        npu_mha._qwen35_attention = staged_attention
    rope_registration._apply_rope = leaf_rope
    mamba.gdn_chunk_prefill = gdn_prefill
    mamba.gdn_decode_step = gdn_decode
    ModelRunner.forward = forward


if os.environ.get("TASK1_LAYER_DIAG_DIR"):
    install_engine_hooks()


def metrics(actual, expected):
    import torch

    x, y = actual.float().reshape(-1), expected.float().reshape(-1)
    if x.shape != y.shape:
        raise AssertionError(f"shape mismatch {actual.shape} vs {expected.shape}")
    difference = x - y
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / y.norm().clamp_min(1e-12)).item(),
        "rmse": difference.square().mean().sqrt().item(),
        "mismatch_count": int(torch.count_nonzero(difference)),
        "numel": x.numel(),
    }


def engine(args):
    import torch
    import torch_npu

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TASK1_LAYER_DIAG_DIR"] = str(args.output.resolve())
    os.environ["TASK1_LAYER_DIAG_STEPS"] = json.dumps(args.steps)
    os.environ.pop("TASK1_WORKER_CONTROL", None)
    install_engine_hooks()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    from tokenspeed.runtime.entrypoints.engine import Engine

    fixture = json.loads((args.baseline / f"{args.label}.json").read_text())
    config = json.loads((args.baseline / "manifest.json").read_text())["engine_config"]
    outputs = []
    with Engine(**config) as model:
        (args.output / "request_active").write_text("one diagnostic request\n")
        for chunk in model.generate(
            input_ids=fixture["input_ids"],
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": max(args.steps) + 1,
                "ignore_eos": True,
            },
            stream=True,
        ):
            outputs.extend(chunk["output_ids"])
    if outputs != fixture["output_ids"][: len(outputs)]:
        raise AssertionError("diagnostic hooks changed the original greedy trajectory")
    (args.output / "fixture.json").write_text(json.dumps(fixture, indent=2))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"steps": args.steps, "output_ids_match": True}))


def reference(args):
    import torch
    import torch_npu

    torch.set_num_threads(8)
    torch.npu.set_device(0)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    hf.causal_conv1d_fn = hf.causal_conv1d_update = None
    hf.chunk_gated_delta_rule = hf.fused_recurrent_gated_delta_rule = None
    hf.FusedRMSNormGated = None
    hf.is_fast_path_available = False
    config = json.loads((args.output / "config.json").read_text())
    fixture = json.loads((args.output / "fixture.json").read_text())
    model = (
        hf.Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .npu()
    )
    layers = model.model.language_model.layers
    state = {"active": False, "values": {}}

    def save(key, value):
        if state["active"]:
            if isinstance(value, tuple):
                value = value[0]
            state["values"][key] = value.detach().cpu().clone()

    def output_hook(key):
        return lambda module, inputs, output: save(key, output)

    def input_hook(key):
        return lambda module, inputs: save(key, inputs[0])

    def norm_input_hook(prefix):
        def capture(module, inputs):
            save(prefix + ".gdn_norm_input", inputs[0])
            save(prefix + ".gdn_gate", inputs[1])

        return capture

    for i, layer in enumerate(layers):
        prefix = f"layers.{i}"
        layer.register_forward_hook(output_hook(prefix + ".output"))
        layer.input_layernorm.register_forward_pre_hook(input_hook(prefix + ".input"))
        layer.input_layernorm.register_forward_hook(output_hook(prefix + ".input_norm"))
        layer.post_attention_layernorm.register_forward_pre_hook(
            input_hook(prefix + ".post_residual")
        )
        layer.post_attention_layernorm.register_forward_hook(
            output_hook(prefix + ".post_norm")
        )
        layer.mlp.register_forward_pre_hook(input_hook(prefix + ".mlp_input"))
        layer.mlp.register_forward_hook(output_hook(prefix + ".mlp"))
        layer.mlp.down_proj.register_forward_pre_hook(
            input_hook(prefix + ".mlp_activation")
        )
        layer.mlp.gate_proj.register_forward_hook(output_hook(prefix + ".gate_proj"))
        layer.mlp.up_proj.register_forward_hook(output_hook(prefix + ".up_proj"))
        attn = (
            layer.linear_attn
            if layer.layer_type == "linear_attention"
            else layer.self_attn
        )
        attn.register_forward_hook(output_hook(prefix + ".attn"))
        if layer.layer_type == "linear_attention":
            attn.norm.register_forward_pre_hook(norm_input_hook(prefix))
            attn.norm.register_forward_hook(output_hook(prefix + ".gdn_norm"))
    model.model.language_model.norm.register_forward_hook(output_hook("final_norm"))
    from examples.task1_qwen35.reference_cache import make_reference_cache

    cache = make_reference_cache(model.config, state_dtype=args.reference_state_dtype)
    reports = []
    policy = {
        "matmul_allow_hf32": torch.npu.matmul.allow_hf32,
        "matmul_cube_math_type": str(torch.npu.matmul.cube_math_type),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "reference_state_dtype": args.reference_state_dtype,
    }
    (args.output / "npu_precision_policy.json").write_text(json.dumps(policy, indent=2))

    def hf_gdn_capture(original, prefix):
        def invoke(q, k, v, **kwargs):
            for name, value in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("g", kwargs["g"]),
                ("beta", kwargs["beta"]),
            ):
                save(prefix + ".gdn_" + name, value)
            out = original(q, k, v, **kwargs)
            initial = kwargs["initial_state"]
            save(
                prefix + ".gdn_initial_state",
                torch.zeros_like(out[1]) if initial is None else initial,
            )
            save(prefix + ".gdn_final_state", out[1])
            save(prefix + ".gdn_core_output", out[0])
            return out

        return invoke

    for i, layer in enumerate(layers):
        if layer.layer_type == "linear_attention":
            attn = layer.linear_attn
            prefix = f"layers.{i}"
            attn.chunk_gated_delta_rule = hf_gdn_capture(
                attn.chunk_gated_delta_rule, prefix
            )
            attn.recurrent_gated_delta_rule = hf_gdn_capture(
                attn.recurrent_gated_delta_rule, prefix
            )
            for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"):
                getattr(attn, name).register_forward_hook(
                    output_hook(prefix + "." + name)
                )

    def hf_rope(module, inputs, output):
        save("rope_cos", output[0])
        save("rope_sin", output[1])
        save("rope_inv_freq", module.inv_freq)

    model.model.language_model.rotary_emb.register_forward_hook(hf_rope)
    with torch.inference_mode():
        for step in range(max(args.steps) + 1):
            state["active"] = step in args.steps
            state["values"] = {}
            ids = (
                fixture["input_ids"] if step == 0 else [fixture["output_ids"][step - 1]]
            )
            output = model(
                input_ids=torch.tensor([ids], dtype=torch.long, device="npu"),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            if step not in args.steps:
                continue
            state["active"] = False
            actual = torch.load(
                args.output / f"engine_step{step:03d}.pt", weights_only=True
            )
            if actual["input_ids"].reshape(-1).tolist() != ids:
                raise AssertionError("different trajectory input")
            reference_values = state["values"]
            torch.save(
                {"values": reference_values, "logits": output.logits.cpu()},
                args.output / f"hf_step{step:03d}.pt",
            )
            record = {
                "step": step,
                "logits": metrics(actual["logits"], output.logits.cpu()),
                "layers": [],
                "same_input": [],
            }
            for i, layer in enumerate(layers):
                prefix = f"layers.{i}"
                row = {"layer": i, "type": layer.layer_type}
                for point in (
                    "input",
                    "input_norm",
                    "attn",
                    "post_residual",
                    "post_norm",
                    "mlp_activation",
                    "mlp",
                    "output",
                    "gdn_norm_input",
                    "gdn_gate",
                    "gdn_norm",
                    "gdn_q",
                    "gdn_k",
                    "gdn_v",
                    "gdn_g",
                    "gdn_beta",
                    "gdn_initial_state",
                    "gdn_final_state",
                    "gdn_core_output",
                ):
                    key = prefix + "." + point
                    if key in actual["values"] and key in reference_values:
                        row[point] = metrics(
                            actual["values"][key], reference_values[key]
                        )
                record["layers"].append(row)
                values = actual["values"]
                x = values[prefix + ".mlp_input"].npu().unsqueeze(0)
                gate_up = values[prefix + ".gate_up"].npu().unsqueeze(0)
                gate, up = gate_up.chunk(2, -1)
                staged = torch.nn.functional.silu(gate) * up
                hf_gu = torch.cat((layer.mlp.gate_proj(x), layer.mlp.up_proj(x)), -1)
                same = {
                    "layer": i,
                    "input_norm": metrics(
                        values[prefix + ".input_norm"],
                        layer.input_layernorm(values[prefix + ".input"].npu()).cpu(),
                    ),
                    "post_norm": metrics(
                        values[prefix + ".post_norm"],
                        layer.post_attention_layernorm(
                            values[prefix + ".post_residual"].npu()
                        ).cpu(),
                    ),
                    "gate_up_gemm": metrics(gate_up.cpu(), hf_gu.cpu()),
                    "swiglu": metrics(values[prefix + ".mlp_activation"], staged.cpu()),
                    "down_proj": metrics(
                        values[prefix + ".mlp"],
                        layer.mlp.down_proj(
                            values[prefix + ".mlp_activation"].npu()
                        ).cpu(),
                    ),
                    "whole_mlp": metrics(values[prefix + ".mlp"], layer.mlp(x).cpu()),
                }
                if prefix + ".gdn_q" in values:
                    gq, gk, gv, gg, gb = [
                        values[prefix + ".gdn_" + name].npu()
                        for name in ("q", "k", "v", "g", "beta")
                    ]
                    initial = values[prefix + ".gdn_initial_state"].npu()
                    raw_gq, raw_gk = gq, gk
                    if gq.shape[2] != gv.shape[2]:
                        gq = gq.repeat_interleave(gv.shape[2] // gq.shape[2], dim=2)
                        gk = gk.repeat_interleave(gv.shape[2] // gk.shape[2], dim=2)
                    formula = (
                        hf.torch_chunk_gated_delta_rule
                        if step == 0
                        else hf.torch_recurrent_gated_delta_rule
                    )
                    same_core, same_final = formula(
                        gq,
                        gk,
                        gv,
                        g=gg,
                        beta=gb,
                        initial_state=initial,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    attn = layer.linear_attn
                    ts_norm_input = values[prefix + ".input_norm"].npu().unsqueeze(0)
                    projections = torch.cat(
                        (
                            attn.in_proj_qkv(ts_norm_input),
                            attn.in_proj_z(ts_norm_input),
                            attn.in_proj_b(ts_norm_input),
                            attn.in_proj_a(ts_norm_input),
                        ),
                        -1,
                    )
                    same["gdn_projection_same_input"] = metrics(
                        values[prefix + ".in_proj_qkvzba"], projections.cpu()
                    )
                    ts_a = values[prefix + ".in_proj_qkvzba"].npu()[
                        ..., -attn.num_v_heads :
                    ]
                    same_g = -attn.A_log.float().exp() * torch.nn.functional.softplus(
                        ts_a.float() + attn.dt_bias
                    )
                    same["gdn_g_same_input"] = metrics(gg.cpu(), same_g.cpu())
                    if step == 1:
                        previous_state = torch.load(
                            args.output / "engine_step000.pt", weights_only=True
                        )["values"][prefix + ".gdn_final_state"]
                        same["prefill_state_to_decode_initial"] = metrics(
                            values[prefix + ".gdn_initial_state"], previous_state
                        )
                    projected = values[prefix + ".in_proj_qkvzba"].npu()[
                        ..., : attn.conv_dim
                    ]
                    if step == 0:
                        conv_expected = torch.nn.functional.silu(
                            attn.conv1d(projected.transpose(0, 1)[None])[
                                ..., : projected.shape[0]
                            ]
                        ).transpose(1, 2)
                    else:
                        previous = torch.load(
                            args.output / "engine_step000.pt", weights_only=True
                        )["values"][prefix + ".in_proj_qkvzba"].npu()[
                            ..., : attn.conv_dim
                        ]
                        context = torch.cat((previous[-3:], projected), dim=0)
                        conv_expected = torch.nn.functional.silu(
                            torch.nn.functional.conv1d(
                                context.transpose(0, 1)[None],
                                attn.conv1d.weight,
                                groups=attn.conv_dim,
                            )
                        ).transpose(1, 2)
                    conv_actual = torch.cat(
                        (raw_gq.flatten(2), raw_gk.flatten(2), gv.flatten(2)), -1
                    )
                    same["gdn_conv_same_input"] = metrics(
                        conv_actual.cpu(), conv_expected.cpu()
                    )
                    same["gdn_out_proj_same_input"] = metrics(
                        values[prefix + ".attn"],
                        attn.out_proj(
                            values[prefix + ".gdn_norm"]
                            .npu()
                            .reshape(1, -1, attn.value_dim)
                        ).cpu(),
                    )
                    same["gdn_same_input_core"] = metrics(
                        values[prefix + ".gdn_core_output"], same_core.cpu()
                    )
                    same["gdn_same_input_final_state"] = metrics(
                        values[prefix + ".gdn_final_state"], same_final.cpu()
                    )
                    same["gdn_norm_same_input"] = metrics(
                        values[prefix + ".gdn_norm"],
                        layer.linear_attn.norm(
                            values[prefix + ".gdn_norm_input"].npu().reshape(-1, 128),
                            values[prefix + ".gdn_gate"].npu().reshape(-1, 128),
                        ).cpu(),
                    )
                if prefix + ".mha_q" in values:
                    q = values[prefix + ".mha_q"].npu().transpose(0, 1)
                    k = (
                        values[prefix + ".mha_k"]
                        .npu()
                        .repeat_interleave(4, dim=1)
                        .transpose(0, 1)
                    )
                    v = (
                        values[prefix + ".mha_v"]
                        .npu()
                        .repeat_interleave(4, dim=1)
                        .transpose(0, 1)
                    )
                    allowed = (
                        torch.arange(k.shape[1], device="npu")[None, :]
                        <= (
                            torch.arange(q.shape[1], device="npu")
                            + k.shape[1]
                            - q.shape[1]
                        )[:, None]
                    )
                    scores = torch.matmul(q, k.transpose(-1, -2)) * 0.0625
                    scores = scores.masked_fill(
                        ~allowed[None], torch.finfo(scores.dtype).min
                    )
                    probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
                    hf_attention = torch.matmul(probabilities, v).transpose(0, 1)
                    scores32 = (
                        torch.matmul(q.float(), k.float().transpose(-1, -2)) * 0.0625
                    )
                    scores32 = scores32.masked_fill(~allowed[None], -float("inf"))
                    full32_attention = (
                        torch.matmul(scores32.softmax(-1), v.float())
                        .transpose(0, 1)
                        .to(q.dtype)
                    )
                    same["fia_vs_hf_eager"] = metrics(
                        values[prefix + ".mha_output"], hf_attention.cpu()
                    )
                    same["fia_vs_fp32"] = metrics(
                        values[prefix + ".mha_output"], full32_attention.cpu()
                    )
                if prefix + ".rope_table" in values:
                    table = values[prefix + ".rope_table"].npu()
                    hf_table = torch.cat(
                        (
                            reference_values["rope_cos"][0, :, :32],
                            reference_values["rope_sin"][0, :, :32],
                        ),
                        -1,
                    )
                    same["rope_table_vs_hf"] = metrics(table.cpu(), hf_table)
                    same["hf_inv_freq_dtype"] = str(
                        reference_values["rope_inv_freq"].dtype
                    )
                    cos, sin = table.chunk(2, -1)
                    for kind, heads in (("q", 8), ("k", 2)):
                        raw = (
                            values[prefix + f".rope_{kind}_input"]
                            .npu()
                            .reshape(-1, heads, 256)
                        )
                        cosine, sine = cos[:, None, :], sin[:, None, :]
                        rotary = cos.shape[-1] * 2
                        first, second = (
                            raw[..., : rotary // 2],
                            raw[..., rotary // 2 : rotary],
                        )
                        staged = torch.cat(
                            (
                                first * cosine - second * sine,
                                second * cosine + first * sine,
                                raw[..., rotary:],
                            ),
                            -1,
                        )
                        fp32 = torch.cat(
                            (
                                first.float() * cosine.float()
                                - second.float() * sine.float(),
                                second.float() * cosine.float()
                                + first.float() * sine.float(),
                                raw[..., rotary:].float(),
                            ),
                            -1,
                        ).to(raw.dtype)
                        same[f"rope_{kind}_vs_hf_staged"] = metrics(
                            values[prefix + f".rope_{kind}_output"], staged.cpu()
                        )
                        same[f"rope_{kind}_vs_fp32"] = metrics(
                            values[prefix + f".rope_{kind}_output"], fp32.cpu()
                        )
                record["same_input"].append(same)
            reports.append(record)
            (args.output / "layer_report.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "logits": record["logits"],
                        "first_layers": record["layers"][:4],
                        "first_same_input": record["same_input"][:4],
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("engine", "reference"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--reference-state-dtype", choices=("model", "float32"), required=True
    )
    args = parser.parse_args()
    (engine if args.mode == "engine" else reference)(args)


if __name__ == "__main__":
    main()
