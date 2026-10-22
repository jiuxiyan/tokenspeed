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

"""Read torch_npu 2.9 profiler exports without importing torch or using a device."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from examples.task1_qwen35.common import save_json

KERNEL_FIELDS = (
    "Device_id",
    "Name",
    "Type",
    "Accelerator Core",
    "Start Time(us)",
    "Duration(us)",
)
OPERATOR_FIELDS = (
    "Name",
    "Host Self Duration(us)",
    "Host Total Duration(us)",
    "Device Self Duration(us)",
    "Device Total Duration(us)",
)


def number(value) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid profiler number: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"nonfinite profiler number: {value!r}")
    return result


def interval_summary(intervals: list[tuple[Decimal, Decimal]]) -> dict:
    """Union intervals without treating simultaneous kernels as extra busy time."""
    merged = []
    cumulative = Decimal(0)
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative profiler duration")
        cumulative += end - start
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum((end - start for start, end in merged), Decimal(0))
    span = merged[-1][1] - merged[0][0] if merged else Decimal(0)
    gaps = [right[0] - left[1] for left, right in zip(merged, merged[1:])]
    return {
        "event_count": len(intervals),
        "summed_duration_us": float(cumulative),
        "union_busy_us": float(busy),
        "span_us": float(span),
        "span_start_us": str(merged[0][0]) if merged else None,
        "span_end_us": str(merged[-1][1]) if merged else None,
        "gap_us": float(span - busy),
        "largest_gap_us": float(max(gaps, default=Decimal(0))),
        "busy_fraction_within_selected_event_span": (
            float(busy / span) if span else None
        ),
        "overlap_multiplicity_us": float(cumulative - busy),
        "merged_intervals_us": [[str(start), str(end)] for start, end in merged],
    }


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name}: missing confirmed schema fields {sorted(missing)}"
            )
        return list(reader)


def kernel_summary(path: Path) -> dict:
    groups = {}
    devices = {}
    for row in read_csv(path, KERNEL_FIELDS):
        device, kind, core = row["Device_id"], row["Type"], row["Accelerator Core"]
        start, duration = number(row["Start Time(us)"]), number(row["Duration(us)"])
        if duration < 0:
            raise ValueError("negative kernel duration")
        key = (device, kind, core)
        group = groups.setdefault(
            key, {"count": 0, "duration": Decimal(0), "names": set()}
        )
        group["count"] += 1
        group["duration"] += duration
        group["names"].add(row["Name"])
        devices.setdefault(device, []).append((start, start + duration))
    rows = [
        {
            "device_id": device,
            "type": kind,
            "accelerator_core": core,
            "count": group["count"],
            "distinct_names": len(group["names"]),
            "summed_device_duration_us": float(group["duration"]),
        }
        for (device, kind, core), group in groups.items()
    ]
    rows.sort(key=lambda row: row["summed_device_duration_us"], reverse=True)
    return {
        "kernel_event_count": sum(row["count"] for row in rows),
        "distinct_type_count": len({row["type"] for row in rows}),
        "types": rows,
        "devices": {
            device: interval_summary(items) for device, items in devices.items()
        },
        "scope": "CSV task intervals grouped by exported device/type/core. Summed durations may overlap. The union is busy time within observed task intervals, not chip utilization or profiler-on wall time.",
    }


def operator_summary(path: Path) -> dict:
    groups = {}
    for row in read_csv(path, OPERATOR_FIELDS):
        group = groups.setdefault(
            row["Name"],
            {"count": 0, **{field: Decimal(0) for field in OPERATOR_FIELDS[1:]}},
        )
        group["count"] += 1
        for field in OPERATOR_FIELDS[1:]:
            value = number(row[field])
            if value < 0:
                raise ValueError("negative operator duration")
            group[field] += value
    rows = [
        {
            "name": name,
            "count": values["count"],
            **{field: float(values[field]) for field in OPERATOR_FIELDS[1:]},
        }
        for name, values in groups.items()
    ]
    rows.sort(key=lambda row: row["Device Self Duration(us)"], reverse=True)
    return {
        "operators": rows,
        "scope": "Framework operator attribution from the installed exporter. Total includes child operators and is not additive across hierarchy. Even Self device durations are sums of related kernels, not interval unions or utilization.",
    }


def event_key(event: dict) -> tuple[str, str, Decimal]:
    return str(event["pid"]), str(event["tid"]), number(event["ts"])


def trace_summary(path: Path, *, selection: str, device_pids: list[str]) -> dict:
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        raise ValueError(
            "installed torch_npu trace_view schema is an event list; unsupported root shape"
        )
    complete = []
    index = {}
    inventory = {}
    flows = {}
    for event in events:
        pid = str(event.get("pid", ""))
        process = inventory.setdefault(
            pid, {"metadata": [], "complete_events": 0, "categories": Counter()}
        )
        if event.get("ph") == "M" and event.get("name") in (
            "process_name",
            "process_labels",
        ):
            process["metadata"].append(
                {"name": event["name"], "args": event.get("args", {})}
            )
        if event.get("cat") == "HostToDevice" and event.get("ph") in ("s", "f"):
            flows.setdefault(str(event["id"]), {})[event["ph"]] = event
        if event.get("ph") != "X":
            continue
        start, duration = number(event["ts"]), number(event["dur"])
        if duration < 0:
            raise ValueError("negative trace duration")
        item = {"event": event, "interval": (start, start + duration)}
        complete.append(item)
        index.setdefault(event_key(event), []).append(item)
        process["complete_events"] += 1
        process["categories"][str(event.get("cat", ""))] += 1
    unmatched = []
    if selection == "host-to-device":
        if device_pids:
            raise ValueError("device-pids applies only to explicit-pid selection")
        selected_keys = set()
        for flow_id, flow in flows.items():
            if not {"s", "f"} <= flow.keys():
                unmatched.append({"flow_id": flow_id, "reason": "unpaired flow"})
                continue
            key = event_key(flow["f"])
            matches = index.get(key, [])
            if len(matches) != 1:
                unmatched.append(
                    {
                        "flow_id": flow_id,
                        "reason": f"{len(matches)} complete events at endpoint",
                        "pid": key[0],
                        "tid": key[1],
                        "ts_us": str(key[2]),
                    }
                )
                continue
            selected_keys.add(key)
        selected = [index[key][0] for key in sorted(selected_keys)]
        scope = "Only X events uniquely matched to paired HostToDevice flow endpoints, using the installed CANNFileParser rule. Unlinked device activity may be omitted; this is a selected-event busy-time lower bound, not total hardware utilization."
    elif selection == "explicit-pid":
        if not device_pids:
            raise ValueError(
                "explicit-pid selection requires --device-pids from reviewed process metadata"
            )
        unknown = set(device_pids) - inventory.keys()
        if unknown:
            raise ValueError(f"unknown trace pids: {sorted(unknown)}")
        selected = [
            item for item in complete if str(item["event"]["pid"]) in device_pids
        ]
        scope = "All X events for explicitly selected trace process IDs. Inspect process metadata to ensure these are device timelines; nested annotations are removed only by interval union, not classified as kernels."
    else:
        raise ValueError(f"unsupported trace selection: {selection}")
    whole = interval_summary([item["interval"] for item in complete])
    busy = interval_summary([item["interval"] for item in selected])
    by_pid = {}
    kinds = {}
    for item in selected:
        event = item["event"]
        by_pid.setdefault(str(event["pid"]), []).append(item["interval"])
        key = (
            str(event["pid"]),
            str(event.get("cat", "")),
            str(event.get("args", {}).get("Task Type", "")),
        )
        group = kinds.setdefault(key, {"count": 0, "duration": Decimal(0)})
        group["count"] += 1
        group["duration"] += item["interval"][1] - item["interval"][0]
    return {
        "selection": selection,
        "scope": scope,
        "processes": inventory,
        "selected_device_events_found": bool(selected),
        "whole_trace_complete_event_span_us": whole["span_us"],
        "whole_trace_span_start_us": whole["span_start_us"],
        "whole_trace_span_end_us": whole["span_end_us"],
        "selected": busy,
        "selected_busy_fraction_within_whole_trace_span": (
            busy["union_busy_us"] / whole["span_us"] if whole["span_us"] else None
        ),
        "by_pid": {pid: interval_summary(items) for pid, items in by_pid.items()},
        "unmatched_flows": unmatched,
        "categories": [
            {
                "pid": pid,
                "category": category,
                "task_type": task_type,
                "count": values["count"],
                "summed_device_duration_us": float(values["duration"]),
            }
            for (pid, category, task_type), values in kinds.items()
        ],
        "interpretation": "The whole trace span is the earliest-to-latest positive-duration X event, not necessarily profiler start/stop. Busy fractions are timeline occupancy, not AI Core occupancy. Summed kernel duration divided by span can exceed 100% and is never called utilization. CSV and trace clock alignment is not assumed.",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="One ASCEND_PROFILER_OUTPUT directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh analysis directory"
    )
    parser.add_argument(
        "--trace-selection", choices=("host-to-device", "explicit-pid"), required=True
    )
    parser.add_argument("--device-pids", nargs="+", default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a fresh or empty output directory")
    report = {
        "schema": "torch_npu 2.9.0.post2 installed sources; see profiler_schema.json",
        "source_directory": str(args.directory.resolve()),
        "sources": [],
        "missing_exports": [],
    }
    exports = (
        ("kernel_details.csv", "kernels", kernel_summary),
        ("operator_details.csv", "operators", operator_summary),
    )
    for name, key, reader in exports:
        path = args.directory / name
        if path.is_file():
            report[key] = reader(path)
        else:
            report["missing_exports"].append(name)
    trace = args.directory / "trace_view.json"
    if trace.is_file():
        report["trace"] = trace_summary(
            trace, selection=args.trace_selection, device_pids=args.device_pids
        )
    else:
        report["missing_exports"].append(trace.name)
    for name in ("kernel_details.csv", "operator_details.csv", "trace_view.json"):
        path = args.directory / name
        if path.is_file():
            report["sources"].append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not report["sources"]:
        raise FileNotFoundError(
            "no confirmed profiler exports found; no metrics invented"
        )
    save_json(args.output / "profile_summary.json", report)
    if "kernels" in report:
        write_csv(args.output / "kernel_types.csv", report["kernels"]["types"])
    if "operators" in report:
        write_csv(args.output / "operator_totals.csv", report["operators"]["operators"])
    if "trace" in report:
        write_csv(args.output / "trace_categories.csv", report["trace"]["categories"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "missing_exports": report["missing_exports"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
