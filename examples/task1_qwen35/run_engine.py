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

"""Run the original TokenSpeed Engine with exact token fixtures on one NPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from examples.task1_qwen35.common import (
    chat_input_ids,
    environment,
    exact_prompt,
    ids_hash,
    npu_snapshot,
    save_json,
    stream_timings,
    timing_summary,
)

# multiprocessing.spawn imports this file before it starts each worker. The
# wrapper is therefore installed on the worker's model class, not a frontend
# copy. The timing mode never sets this variable.
if os.environ.get("TASK1_WORKER_CONTROL"):
    from examples.task1_qwen35.worker_observer import install

    install(os.environ["TASK1_WORKER_CONTROL"])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "chat", "benchmark", "correctness", "diagnostic"),
        required=True,
    )
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--correctness-lengths", type=int, nargs="+", default=[1, 16, 128, 512]
    )
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument(
        "--profile", action="store_true", help="NPU profiler in diagnostic mode only"
    )
    args = parser.parse_args()
    if args.profile and args.mode != "diagnostic":
        parser.error("--profile is only valid with --mode diagnostic")
    if (
        min(
            args.input_lengths
            + args.output_lengths
            + args.correctness_lengths
            + [args.decode_steps]
        )
        < 1
    ):
        parser.error("all token lengths must be positive")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    if (
        max(args.input_lengths + args.correctness_lengths)
        + max(args.output_lengths + [args.decode_steps])
        > args.max_model_len
    ):
        parser.error("input plus output length exceeds max-model-len")
    return args


def request(
    engine,
    tokenizer,
    ids: list[int],
    length: int,
    label: str,
    control: Path | None,
    directory: Path,
    *,
    mode: str,
    profile: bool,
    fixed_length: bool,
) -> dict:
    if control is not None:
        save_json(
            control,
            {
                "active": True,
                "label": label,
                "directory": str(directory / "worker"),
                "capture_logits": mode == "correctness",
                "profile": profile,
                "generation_limit": length,
            },
        )
    events = []
    started = time.perf_counter()
    final = None
    try:
        for chunk in engine.generate(
            input_ids=ids,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": length,
                "ignore_eos": fixed_length,
            },
            stream=True,
        ):
            events.append(
                {"time": time.perf_counter(), "new_ids": list(chunk["output_ids"])}
            )
            final = chunk
        finished = time.perf_counter()
    finally:
        if control is not None:
            save_json(control, {"active": False})
    if final is None:
        raise AssertionError("engine returned no response")
    actual_tokens = sum(len(event["new_ids"]) for event in events)
    if not fixed_length and not 0 < actual_tokens <= length:
        raise AssertionError(
            f"expected 1..{length} chat tokens, received {actual_tokens}"
        )
    result = stream_timings(
        events, started, finished, length if fixed_length else actual_tokens
    )
    result.update(
        {
            "label": label,
            "input_ids": ids,
            "input_sha256": ids_hash(ids),
            "input_tokens": len(ids),
            "generation_tokens": actual_tokens,
            "generation_limit": length,
            "fixed_length": fixed_length,
            "text": tokenizer.decode(result["output_ids"], skip_special_tokens=False),
            "meta_info": final["meta_info"],
            "frames": [
                {"at_ms": (e["time"] - started) * 1000, "new_ids": e["new_ids"]}
                for e in events
            ],
        }
    )
    if final["meta_info"]["prompt_tokens"] != len(ids):
        raise AssertionError("engine prompt token count differs from fixture")
    if final["meta_info"].get("cached_tokens", 0) != 0:
        raise AssertionError("prefix-cache hits invalidate the uncached baseline")
    save_json(directory / f"{label}.json", result)
    return result


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"choose a fresh output directory: {output}")
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer

    if torch.npu.device_count() != 1:
        raise RuntimeError(
            "set ASCEND_RT_VISIBLE_DEVICES to exactly one chip before launch"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    control = None
    if args.mode in ("correctness", "diagnostic"):
        control = output / "worker_control.json"
        save_json(control, {"active": False})
        os.environ["TASK1_WORKER_CONTROL"] = str(control)
        from examples.task1_qwen35.worker_observer import install

        install(str(control))
    else:
        if os.environ.get("TASK1_WORKER_CONTROL"):
            raise RuntimeError("unset TASK1_WORKER_CONTROL for primary timing runs")

    from tokenspeed.runtime.entrypoints.engine import Engine

    config = dict(
        model=args.model,
        device="npu",
        dtype="bfloat16",
        mamba_ssm_dtype="float32",
        language_model_only=True,
        attention_backend="mha",
        sampling_backend="greedy",
        world_size=1,
        nnodes=1,
        nprocs_per_node=1,
        attn_tp_size=1,
        data_parallel_size=1,
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_pdl=True,
        disable_autotune=True,
        enable_prefix_caching=False,
        disable_kvstore=True,
        grammar_backend="none",
        retraction_snapshot_ratio=0.0,
        max_model_len=args.max_model_len,
        max_total_tokens=args.max_total_tokens,
        max_num_seqs=1,
        chunked_prefill_size=max(args.input_lengths + args.correctness_lengths),
        prefix_granularity=128,
        max_cudagraph_capture_size=1,
        stream_interval=1,
        stream_output=True,
        skip_tokenizer_init=False,
        enable_log_request_stats=True,
        log_level="info",
        seed=20261022,
    )
    manifest = {
        "mode": args.mode,
        "environment": environment(),
        "engine_config": config,
        "npu_before": npu_snapshot(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "measurement": {
            "ttft": "Frontend request invocation to receipt of first token; includes IPC/scheduling/H2D/model/sampling/detokenization; excludes prior tokenization and model loading.",
            "tpot": "First-to-last token receipt divided by G-1; frontend delivery timing. A coalesced first frame is reported as unavailable, not an exact TTFT.",
            "prefill": "Worker model forward is separately recorded in diagnostic mode; engine request stats are also emitted to the run log.",
            "memory": "Device-wide npu-smi snapshots include background occupancy; worker diagnostics report actual worker allocator values. Frontend allocator is not used.",
            "diagnostics": "Worker synchronization, CPU copies and profiler are disabled in benchmark mode. Diagnostic timings are not benchmark timings.",
        },
        "cases": [],
        "summaries": [],
    }
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    with Engine(**config) as engine:
        manifest["engine_startup_seconds"] = time.perf_counter() - started
        manifest["npu_after_load"] = npu_snapshot()
        if args.mode == "benchmark":
            for input_length in args.input_lengths:
                ids = exact_prompt(tokenizer, input_length, 0)
                for generation in args.output_lengths:
                    samples = []
                    for repeat in range(args.warmups + args.repeats):
                        warm = repeat < args.warmups
                        label = f"i{input_length}_o{generation}_{'warm' if warm else 'run'}{repeat:02d}"
                        result = request(
                            engine,
                            tokenizer,
                            ids,
                            generation,
                            label,
                            None,
                            output,
                            mode=args.mode,
                            profile=False,
                            fixed_length=True,
                        )
                        result["warmup"] = warm
                        manifest["cases"].append({"label": label, "warmup": warm})
                        if not warm:
                            samples.append(result)
                        save_json(output / "manifest.json", manifest)
                    manifest["summaries"].append(
                        {
                            "input_tokens": input_length,
                            "output_tokens": generation,
                            **timing_summary(samples),
                        }
                    )
                    manifest[f"npu_after_i{input_length}_o{generation}"] = (
                        npu_snapshot()
                    )
                    save_json(output / "manifest.json", manifest)
        elif args.mode == "chat":
            prompts = (
                (
                    "chat_chinese",
                    "请用两句话解释大语言模型推理中的 Prefill 和 Decode 分别做什么。",
                ),
                (
                    "chat_english",
                    "In two sentences, explain why a language model keeps a key-value cache during generation.",
                ),
                (
                    "chat_code",
                    "Write a Python function add(a, b) that returns their sum. Output only the code.",
                ),
                ("chat_arithmetic", "计算 17×6+8，只回答最终整数。"),
            )
            manifest["chat_template"] = {
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            for label, prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                ids = chat_input_ids(tokenizer, messages)
                if len(ids) + args.decode_steps > args.max_model_len:
                    raise ValueError(f"{label} exceeds max-model-len")
                result = request(
                    engine,
                    tokenizer,
                    ids,
                    args.decode_steps,
                    label,
                    None,
                    output,
                    mode=args.mode,
                    profile=False,
                    fixed_length=False,
                )
                result["messages"] = messages
                result["response_text"] = tokenizer.decode(
                    result["output_ids"], skip_special_tokens=True
                )
                save_json(output / f"{label}.json", result)
                manifest["cases"].append({"label": label, "warmup": False})
                save_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            "label": label,
                            "prompt": prompt,
                            "response": result["response_text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        else:
            lengths = (
                args.correctness_lengths
                if args.mode == "correctness"
                else args.input_lengths
            )
            generation = args.decode_steps
            for input_length in lengths:
                if args.mode == "diagnostic":
                    for warmup in range(args.warmups):
                        label = f"diagnostic_warm_i{input_length}_{warmup:02d}"
                        request(
                            engine,
                            tokenizer,
                            exact_prompt(tokenizer, input_length, 0),
                            generation,
                            label,
                            None,
                            output,
                            mode=args.mode,
                            profile=False,
                            fixed_length=True,
                        )
                        manifest["cases"].append({"label": label, "warmup": True})
                        save_json(output / "manifest.json", manifest)
                label = f"length_{input_length}"
                result = request(
                    engine,
                    tokenizer,
                    exact_prompt(tokenizer, input_length, 0),
                    generation,
                    label,
                    control,
                    output,
                    mode=args.mode,
                    profile=args.profile,
                    fixed_length=True,
                )
                manifest["cases"].append({"label": label, "warmup": False})
                save_json(output / "manifest.json", manifest)
            if args.mode in ("correctness", "smoke"):
                isolated = []
                for label, length, variant in (
                    ("A1", 32, 0),
                    ("B", 65, 1),
                    ("A2", 32, 0),
                ):
                    isolated.append(
                        request(
                            engine,
                            tokenizer,
                            exact_prompt(tokenizer, length, variant),
                            generation,
                            label,
                            control,
                            output,
                            mode=args.mode,
                            profile=False,
                            fixed_length=True,
                        )
                    )
                    manifest["cases"].append({"label": label, "warmup": False})
                same = isolated[0]["output_ids"] == isolated[2]["output_ids"]
                manifest["request_isolation"] = {
                    "sequence": ["A1", "B", "A2"],
                    "identical_output_ids": same,
                    "max_num_seqs": 1,
                    "prefix_caching": False,
                }
                save_json(output / "manifest.json", manifest)
                if not same:
                    raise AssertionError(
                        "A-B-A request lifecycle changed A's generated tokens"
                    )
        manifest["npu_before_shutdown"] = npu_snapshot()
    manifest["completed"] = True
    save_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "summaries": manifest["summaries"],
                "completed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
