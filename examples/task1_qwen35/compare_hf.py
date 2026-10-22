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

"""Teacher-force the engine trajectory through the official HF implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from examples.task1_qwen35.common import environment, ids_hash, save_json


def compare_logits(actual, reference) -> dict:
    import torch

    x, y = actual.float().reshape(-1), reference.float().reshape(-1)
    if x.shape != y.shape:
        raise AssertionError(f"vocabulary shapes differ: {x.shape} != {y.shape}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise AssertionError("nonfinite logits")
    difference = x - y
    norm = torch.linalg.vector_norm(y).clamp_min(1e-12)
    top = torch.topk(y, 2)
    return {
        "vocabulary_size": x.numel(),
        "all_finite": True,
        "max_abs": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / norm).item(),
        "cosine": torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
        "engine_top1": x.argmax().item(),
        "reference_top1": y.argmax().item(),
        "top1_agrees": bool(x.argmax() == y.argmax()),
        "reference_top1_margin": (top.values[0] - top.values[1]).item(),
        "reference_logprob_engine_top1": y.log_softmax(-1)[x.argmax()].item(),
        "engine_top1_logprob": x.log_softmax(-1)[x.argmax()].item(),
    }


def validate_snapshot(saved: dict, fixture: dict, step: int, vocab_size: int) -> None:
    """Reject a mismatched trajectory before comparing numerical values."""
    import torch

    ids = fixture["input_ids"] if step == 0 else [fixture["output_ids"][step - 1]]
    expected_positions = (
        list(range(len(ids))) if step == 0 else [len(fixture["input_ids"]) + step - 1]
    )
    if saved["input_ids"].reshape(-1).tolist() != ids:
        raise AssertionError(
            f"step {step}: observed model input differs from teacher-forced IDs"
        )
    positions = saved["positions"]
    # Text M-RoPE uses three identical position rows; ordinary RoPE uses one.
    if positions.ndim not in (1, 2) or positions.shape[-1] != len(ids):
        raise AssertionError(
            f"step {step}: unexpected position shape {positions.shape}"
        )
    if any(
        row != expected_positions for row in positions.reshape(-1, len(ids)).tolist()
    ):
        raise AssertionError(
            f"step {step}: observed positions differ from the text trajectory"
        )
    expected_phase = "extend" if step == 0 else "decode"
    if saved["phase"] != expected_phase:
        raise AssertionError(
            f"step {step}: expected {expected_phase}, got {saved['phase']}"
        )
    if list(saved["logits"].shape) != [1, vocab_size]:
        raise AssertionError(f"step {step}: expected batch-one full-vocabulary logits")
    if not torch.isfinite(saved["logits"]).all():
        raise AssertionError(f"step {step}: nonfinite engine logits")
    if saved["logits"].argmax(-1).item() != fixture["output_ids"][step]:
        raise AssertionError(
            f"step {step}: emitted greedy token differs from observed logits argmax"
        )


def summarize_metrics(metrics: list[dict]) -> dict:
    if not metrics:
        raise AssertionError("no logits were compared")
    return {
        "steps": len(metrics),
        "max_abs": max(m["max_abs"] for m in metrics),
        "max_relative_l2": max(m["relative_l2"] for m in metrics),
        "min_cosine": min(m["cosine"] for m in metrics),
        "top1_agreement": sum(m["top1_agrees"] for m in metrics) / len(metrics),
        "top1_mismatches": [
            {
                key: m[key]
                for key in (
                    "label",
                    "step",
                    "phase",
                    "engine_top1",
                    "reference_top1",
                    "reference_top1_margin",
                )
            }
            for m in metrics
            if not m["top1_agrees"]
        ],
    }


def select_snapshots(
    run: Path, label: str, fixture: dict, previous_label: str | None
) -> tuple[list[Path], list[dict]]:
    """Recognize one audited previous-request overlap decode in legacy runs."""
    import torch

    snapshots = sorted((run / "worker" / label).glob("*.pt"))
    expected_count = fixture["generation_tokens"]
    excluded = []
    if len(snapshots) == expected_count + 1 and previous_label is not None:
        saved = torch.load(snapshots[0], map_location="cpu", weights_only=True)
        previous = json.loads((run / f"{previous_label}.json").read_text())
        expected_ids = [previous["output_ids"][-1]]
        expected_position = len(previous["input_ids"]) + len(previous["output_ids"]) - 1
        positions = saved["positions"]
        is_previous_unused_decode = (
            saved["phase"] == "decode"
            and saved["input_ids"].reshape(-1).tolist() == expected_ids
            and positions.ndim in (1, 2)
            and positions.shape[-1] == 1
            and all(
                row == [expected_position] for row in positions.reshape(-1, 1).tolist()
            )
        )
        if is_previous_unused_decode:
            excluded.append(
                {
                    "snapshot": str(snapshots[0]),
                    "phase": saved["phase"],
                    "input_ids": saved["input_ids"].tolist(),
                    "positions": positions.tolist(),
                    "previous_label": previous_label,
                    "expected_previous_last_output_id": expected_ids[0],
                    "expected_previous_next_position": expected_position,
                    "reason": "Legacy control-label race: one in-flight overlap decode consumes the previous request's last emitted token after that request completed, before this request's first extend. It does not produce an emitted token for the current request.",
                }
            )
            snapshots = snapshots[1:]
    if len(snapshots) != expected_count:
        raise AssertionError(
            f"{label}: expected {expected_count} request forwards, found {len(snapshots)}; unrecognized extra/missing snapshots are not excluded"
        )
    return snapshots, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument(
        "--reference-state-dtype",
        choices=("model", "float32"),
        required=True,
        help="Explicit HF recurrent-cache policy: stock model dtype or FP32 state; conv/KV remain in --dtype",
    )
    parser.add_argument("--max-abs", type=float, required=True)
    parser.add_argument("--max-relative-l2", type=float, required=True)
    parser.add_argument("--min-cosine", type=float, required=True)
    parser.add_argument("--min-top1-agreement", type=float, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Inspect an incomplete run; cannot qualify as protocol acceptance",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a fresh report path: {args.output}")
    if (
        args.max_abs < 0
        or args.max_relative_l2 < 0
        or not -1 <= args.min_cosine <= 1
        or not 0 <= args.min_top1_agreement <= 1
    ):
        parser.error("invalid numerical thresholds")
    manifest = json.loads((args.engine_results / "manifest.json").read_text())
    if manifest["mode"] != "correctness":
        raise ValueError("run_engine --mode correctness must produce the logits first")
    if not manifest.get("completed", False) and not args.allow_partial:
        raise ValueError(
            "engine run is incomplete; --allow-partial is for diagnosis only"
        )
    if Path(manifest["engine_config"]["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("reference and engine must use the same local checkpoint")
    available = [entry["label"] for entry in manifest["cases"]]
    labels = args.labels or available
    if len(set(labels)) != len(labels) or not set(labels) <= set(available):
        raise ValueError("labels must be unique cases from the engine manifest")
    import torch

    if args.device == "npu":
        import torch_npu  # noqa: F401

        if torch.npu.device_count() != 1:
            raise RuntimeError("reference also requires exactly one visible chip")
        torch.npu.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    from examples.task1_qwen35.reference_cache import (
        cache_dtype_manifest,
        make_reference_cache,
    )

    # Explicitly select the official portable implementation, even when FLA or
    # causal-conv extensions happen to be installed. No TokenSpeed ops enter it.
    hf.causal_conv1d_fn = None
    hf.causal_conv1d_update = None
    hf.chunk_gated_delta_rule = None
    hf.fused_recurrent_gated_delta_rule = None
    hf.FusedRMSNormGated = None
    hf.is_fast_path_available = False
    model = (
        hf.Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model,
            dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .to(args.device)
    )
    vocab_size = model.config.text_config.vocab_size
    report = {
        "reference": "official Transformers Qwen3_5ForConditionalGeneration, eager attention, portable PyTorch conv/GDN/norm",
        "environment": environment(),
        "device": args.device,
        "dtype": args.dtype,
        "reference_state_dtype": args.reference_state_dtype,
        "comparison_kind": (
            "matched_fp32_recurrent_state"
            if args.reference_state_dtype == "float32"
            else "stock_hf_compatibility"
        ),
        "reference_cache_policy": (
            "Official HF equations with instance-local FP32 recurrent-cache allocation from the first prefill; official update copy is unchanged. Conv and KV caches retain the model dtype."
            if args.reference_state_dtype == "float32"
            else "Unmodified official HF DynamicCache: recurrent state uses the conv/model dtype; FP32 recurrence results round at each cache update."
        ),
        "engine_results": str(args.engine_results.resolve()),
        "engine_environment": manifest["environment"],
        "engine_manifest_sha256": hashlib.sha256(
            (args.engine_results / "manifest.json").read_bytes()
        ).hexdigest(),
        "engine_run_completed": manifest.get("completed", False),
        "selected_labels": labels,
        "logit_mask": "None: every entry of the model's exact vocabulary is compared and must be finite. Causal attention masks are implemented independently inside each model; no padding or nonfinite logits are silently removed.",
        "thresholds": {
            "max_abs": args.max_abs,
            "relative_l2": args.max_relative_l2,
            "cosine_min": args.min_cosine,
            "top1_agreement_min": args.min_top1_agreement,
        },
        "precision_note": "The reference uses official HF portable arithmetic. Recurrent cache precision is an explicit reported experimental variable; all numerical thresholds remain fixed. Stock-HF compatibility and matched-FP32-state acceptance are reported separately.",
        "cases": [],
    }
    all_metrics = []
    selected_snapshots = {}
    with torch.inference_mode():
        for label in labels:
            fixture = json.loads((args.engine_results / f"{label}.json").read_text())
            if ids_hash(fixture["input_ids"]) != fixture["input_sha256"]:
                raise AssertionError(f"{label}: input fixture hash mismatch")
            if len(fixture["output_ids"]) != fixture["generation_tokens"]:
                raise AssertionError(f"{label}: generated token count mismatch")
            index = available.index(label)
            previous_label = available[index - 1] if index else None
            snapshots, excluded = select_snapshots(
                args.engine_results, label, fixture, previous_label
            )
            selected_snapshots[label] = snapshots
            cache = make_reference_cache(
                model.config, state_dtype=args.reference_state_dtype
            )
            case = {
                "label": label,
                "input_tokens": len(fixture["input_ids"]),
                "input_sha256": fixture["input_sha256"],
                "output_tokens": fixture["generation_tokens"],
                "decode_steps": fixture["generation_tokens"] - 1,
                "excluded_previous_request_forwards": excluded,
                "cache_dtype_verified_steps": 0,
                "steps": [],
            }
            for step, snapshot in enumerate(snapshots):
                saved = torch.load(snapshot, map_location="cpu", weights_only=True)
                validate_snapshot(saved, fixture, step, vocab_size)
                ids = (
                    fixture["input_ids"]
                    if step == 0
                    else [fixture["output_ids"][step - 1]]
                )
                output = model(
                    input_ids=torch.tensor([ids], device=args.device, dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
                dtype_manifest = cache_dtype_manifest(
                    cache,
                    state_dtype=args.reference_state_dtype,
                    model_dtype=getattr(torch, args.dtype),
                )
                if step == 0:
                    case["cache_dtypes_by_layer"] = dtype_manifest
                elif dtype_manifest != case["cache_dtypes_by_layer"]:
                    raise AssertionError("reference cache dtypes changed during decode")
                case["cache_dtype_verified_steps"] += 1
                metrics = compare_logits(saved["logits"][0], output.logits[0, -1].cpu())
                metrics["step"] = step
                metrics["label"] = label
                metrics["phase"] = saved["phase"]
                metrics["pass"] = (
                    metrics["max_abs"] <= args.max_abs
                    and metrics["relative_l2"] <= args.max_relative_l2
                    and metrics["cosine"] >= args.min_cosine
                )
                case["steps"].append(metrics)
                all_metrics.append(metrics)
            case["summary"] = summarize_metrics(case["steps"])
            report["cases"].append(case)
            save_json(args.output, report)
    report["summary"] = summarize_metrics(all_metrics)
    report["summary"]["pass"] = (
        all(m["pass"] for m in all_metrics)
        and report["summary"]["top1_agreement"] >= args.min_top1_agreement
    )
    if "A1" in labels and "A2" in labels:
        a1 = json.loads((args.engine_results / "A1.json").read_text())
        a2 = json.loads((args.engine_results / "A2.json").read_text())
        if a1["input_ids"] != a2["input_ids"] or a1["output_ids"] != a2["output_ids"]:
            raise AssertionError("A-B-A input or output IDs differ")
        first = selected_snapshots["A1"]
        second = selected_snapshots["A2"]
        if len(first) != len(second):
            raise AssertionError("A-B-A forward counts differ")
        errors = []
        for a, b in zip(first, second):
            x = torch.load(a, map_location="cpu", weights_only=True)["logits"]
            y = torch.load(b, map_location="cpu", weights_only=True)["logits"]
            errors.append((x - y).abs().max().item())
        report["isolation_logits"] = {
            "max_abs_by_step": errors,
            "exact_equal": all(e == 0 for e in errors),
        }
        report["summary"]["pass"] = (
            report["summary"]["pass"] and report["isolation_logits"]["exact_equal"]
        )
    required = {f"length_{length}" for length in (1, 16, 128, 512)} | {"A1", "B", "A2"}
    isolation = manifest.get("request_isolation", {})
    protocol_scope = (
        manifest.get("completed", False)
        and required <= set(labels)
        and all(
            case["output_tokens"] == 32
            for case in report["cases"]
            if case["label"] in required
        )
        and isolation.get("sequence") == ["A1", "B", "A2"]
        and isolation.get("identical_output_ids") is True
        and isolation.get("max_num_seqs") == 1
        and isolation.get("prefix_caching") is False
        and args.dtype == "bfloat16"
        and args.reference_state_dtype == "float32"
        and report["environment"]["packages"]["transformers"] == "5.12.0"
        and args.max_abs == 0.5
        and args.max_relative_l2 == 0.02
        and args.min_cosine == 0.999
        and args.min_top1_agreement == 0.95
    )
    report["summary"]["protocol_scope_complete"] = protocol_scope
    report["summary"]["protocol_pass"] = protocol_scope and report["summary"]["pass"]
    save_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
