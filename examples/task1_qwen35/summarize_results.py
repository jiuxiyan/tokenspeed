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

"""Produce reviewable JSON, CSV and Markdown from completed Task1 raw runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from examples.task1_qwen35.common import ids_hash, save_json, timing_summary


def file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_benchmark(run: Path, *, allow_partial: bool) -> dict:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["mode"] != "benchmark":
        raise ValueError(
            "main performance table requires an uninstrumented benchmark run"
        )
    if not manifest.get("completed", False) and not allow_partial:
        raise ValueError(
            "benchmark is incomplete; use --allow-partial only for progress inspection"
        )
    groups = {}
    labels = set()
    sources = [file_identity(manifest_path)]
    for entry in manifest["cases"]:
        label = entry["label"]
        if label in labels:
            raise ValueError(f"duplicate case: {label}")
        labels.add(label)
        path = run / f"{label}.json"
        sample = json.loads(path.read_text())
        sources.append(file_identity(path))
        if (
            len(sample["input_ids"]) != sample["input_tokens"]
            or ids_hash(sample["input_ids"]) != sample["input_sha256"]
        ):
            raise ValueError(f"{label}: inconsistent input fixture")
        generated = len(sample["output_ids"])
        if (
            not sample["fixed_length"]
            or generated != sample["generation_tokens"]
            or generated != sample["generation_limit"]
        ):
            raise ValueError(f"{label}: fixed generation workload was not completed")
        if sample["decode_steps"] != generated - 1:
            raise ValueError(f"{label}: wrong decode-step denominator")
        if sample["meta_info"].get("cached_tokens", 0) != 0:
            raise ValueError(f"{label}: prefix-cache hit in an uncached baseline")
        if not sample["ttft_is_exact"] and (
            sample["ttft_ms"] is not None or sample["tpot_ms"] is not None
        ):
            raise ValueError(
                f"{label}: coalesced first frame incorrectly marked as exact"
            )
        key = (sample["input_tokens"], generated)
        group = groups.setdefault(key, {"warmups": [], "samples": []})
        group["warmups" if entry["warmup"] else "samples"].append(sample)
    rows = []
    for (input_tokens, output_tokens), group in sorted(groups.items()):
        all_samples = group["warmups"] + group["samples"]
        if len({sample["input_sha256"] for sample in all_samples}) != 1:
            raise ValueError("same-shape requests used different input fixtures")
        if manifest.get("completed", False) and (
            len(group["warmups"]) != manifest["warmups"]
            or len(group["samples"]) != manifest["repeats"]
        ):
            raise ValueError("completed run has missing warmups or repetitions")
        rows.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "decode_steps": output_tokens - 1,
                "warmups": len(group["warmups"]),
                "measured_requests": len(group["samples"]),
                "input_sha256": all_samples[0]["input_sha256"],
                "sample_labels": [sample["label"] for sample in group["samples"]],
                "warmup_labels": [sample["label"] for sample in group["warmups"]],
                **timing_summary(group["samples"]),
            }
        )
    protocol_complete = (
        manifest.get("completed", False)
        and set(groups) == {(i, o) for i in (32, 512) for o in (32, 128)}
        and manifest["warmups"] == 2
        and manifest["repeats"] == 5
        and all(row["ttft_ms"]["n"] == 5 and row["tpot_ms"]["n"] == 5 for row in rows)
    )
    return {
        "run": str(run.resolve()),
        "completed": manifest.get("completed", False),
        "protocol_matrix_complete": protocol_complete,
        "environment": manifest["environment"],
        "engine_config": manifest["engine_config"],
        "engine_startup_seconds": manifest.get("engine_startup_seconds"),
        "measurement": manifest["measurement"],
        "rows": rows,
        "sources": sources,
        "device_hbm_snapshots": {
            key: value for key, value in manifest.items() if key.startswith("npu_")
        },
    }


def read_diagnostics(run: Path) -> dict:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["mode"] != "diagnostic" or not manifest.get("completed", False):
        raise ValueError(
            "memory/model-time summaries require a completed diagnostic run"
        )
    rows = []
    sources = [file_identity(manifest_path)]
    for entry in manifest["cases"]:
        if entry["warmup"]:
            continue
        label = entry["label"]
        fixture_path = run / f"{label}.json"
        fixture = json.loads(fixture_path.read_text())
        sources.append(file_identity(fixture_path))
        groups = {}
        paths = sorted((run / "worker" / label).glob("*.json"))
        if len(paths) != fixture["generation_tokens"]:
            raise ValueError(f"{label}: missing worker forward observations")
        for path in paths:
            item = json.loads(path.read_text())
            sources.append(file_identity(path))
            groups.setdefault((item["phase"], item["profiled"]), []).append(item)
        for (phase, profiled), samples in sorted(groups.items()):
            durations = [sample["model_forward_ms"] for sample in samples]
            rows.append(
                {
                    "label": label,
                    "input_tokens": fixture["input_tokens"],
                    "output_tokens": fixture["generation_tokens"],
                    "phase": phase,
                    "profiled": profiled,
                    "forwards": len(samples),
                    "model_forward_ms_median": (
                        None if profiled else statistics.median(durations)
                    ),
                    "model_forward_ms_min": None if profiled else min(durations),
                    "model_forward_ms_max": None if profiled else max(durations),
                    "allocated_before_first_bytes": samples[0]["allocated_before"],
                    "reserved_before_first_bytes": samples[0]["reserved_before"],
                    "allocated_after_max_bytes": max(
                        sample["allocated_after"] for sample in samples
                    ),
                    "reserved_after_max_bytes": max(
                        sample["reserved_after"] for sample in samples
                    ),
                    "peak_allocated_bytes": max(
                        sample["max_allocated"] for sample in samples
                    ),
                    "peak_reserved_bytes": max(
                        sample["max_reserved"] for sample in samples
                    ),
                }
            )
    result = {
        "run": str(run.resolve()),
        "environment": manifest["environment"],
        "engine_config": manifest["engine_config"],
        "rows": rows,
        "sources": sources,
        "scope": "Actual worker model forwards; synchronized before/after each forward. Peak resets per forward, so loading and scheduler temporary peaks are not measured. Profiled rows remain separate and their timings are excluded.",
    }
    audit_path = run / "weight_audit.json"
    if audit_path.exists():
        result["weight_audit"] = json.loads(audit_path.read_text())
        result["sources"].append(file_identity(audit_path))
    return result


def format_ms(stats: dict) -> str:
    if stats["median"] is None:
        return f"N/A (n={stats['n']})"
    return f"{stats['median']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}] (n={stats['n']})"


def markdown_report(report: dict) -> str:
    benchmark = report["benchmark"]
    lines = [
        "# Task1 measured results",
        "",
        f"Benchmark: `{benchmark['run']}`. Completed: {benchmark['completed']}. Full 2×2 / 2-warmup / 5-repeat protocol: {benchmark['protocol_matrix_complete']}.",
        "",
        "Latency values are median [min, max] in ms. These are frontend delivery measurements; model loading and tokenization were completed beforehand. Decode uses G−1 steps. N/A is retained for an unobservable exact first-token time.",
        "",
        "| Input | Output G | Decode steps | Warmups | Measured | TTFT ms | TPOT ms | Total ms |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in benchmark["rows"]:
        lines.append(
            f"| {row['input_tokens']} | {row['output_tokens']} | {row['decode_steps']} | {row['warmups']} | {row['measured_requests']} | {format_ms(row['ttft_ms'])} | {format_ms(row['tpot_ms'])} | {format_ms(row['total_ms'])} |"
        )
    lines += [
        "",
        f"Engine startup: {benchmark['engine_startup_seconds']} seconds. Code revision: `{benchmark['environment'].get('commit')}`. Actual source hash: `{benchmark['environment'].get('source_manifest_sha256')}`.",
        "",
    ]
    for diagnostic in report["diagnostics"]:
        lines += [
            f"## Separate worker diagnostics: `{diagnostic['run']}`",
            "",
            diagnostic["scope"],
            "",
            "Memory below is GiB (2^30 bytes), including resident weights/cache. It is not a whole-lifecycle peak. `extend` is Prefill; output length is shown because diagnostics may be shorter than the main matrix.",
            "",
            "| Input | Output | Phase | Profiled | Forwards | Model median ms | Before allocated | Peak allocated | Peak reserved |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in diagnostic["rows"]:
            duration = row["model_forward_ms_median"]
            duration_text = "excluded" if duration is None else f"{duration:.3f}"
            lines.append(
                f"| {row['input_tokens']} | {row['output_tokens']} | {row['phase']} | {row['profiled']} | {row['forwards']} | {duration_text} | {row['allocated_before_first_bytes'] / 2**30:.3f} | {row['peak_allocated_bytes'] / 2**30:.3f} | {row['peak_reserved_bytes'] / 2**30:.3f} |"
            )
        lines.append("")
        if "weight_audit" in diagnostic:
            audit = diagnostic["weight_audit"]
            lines += [
                f"Weight audit: {audit['checkpoint_tensor_count']} checkpoint tensors; all text parameters covered: {audit['all_text_parameters_covered']}; missing: `{audit['missing_text_parameters']}`.",
                "",
            ]
    if "reference" in report:
        lines += [
            "## Independent numerical comparison",
            "",
            "```json",
            json.dumps(report["reference"]["summary"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "Raw device-wide npu-smi snapshots are preserved in `summary.json`; they include background occupancy and are not continuous peak measurements. Source paths and SHA256 values are preserved with every raw input. No diagnostic timings are mixed into the main latency table.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, nargs="*", default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh summary directory"
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a new or empty summary directory")
    report = {
        "benchmark": read_benchmark(args.benchmark, allow_partial=args.allow_partial),
        "diagnostics": [read_diagnostics(path) for path in args.diagnostic],
    }
    if args.reference is not None:
        report["reference"] = json.loads(args.reference.read_text())
        report["reference_source"] = file_identity(args.reference)
    save_json(args.output / "summary.json", report)
    (args.output / "summary.md").write_text(markdown_report(report))
    with (args.output / "benchmark.csv").open("w", newline="") as stream:
        fields = [
            "input_tokens",
            "output_tokens",
            "decode_steps",
            "warmups",
            "measured_requests",
        ]
        fields += [
            f"{metric}_{stat}"
            for metric in ("ttft_ms", "tpot_ms", "total_ms")
            for stat in ("median", "min", "max", "n")
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["benchmark"]["rows"]:
            flat = {key: row[key] for key in fields[:5]}
            flat.update(
                {
                    f"{metric}_{stat}": value
                    for metric in ("ttft_ms", "tpot_ms", "total_ms")
                    for stat, value in row[metric].items()
                }
            )
            writer.writerow(flat)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol_matrix_complete": report["benchmark"][
                    "protocol_matrix_complete"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
