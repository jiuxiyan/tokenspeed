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

"""Host-only experiment bookkeeping shared by runners and tests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def code_identity(root: Path) -> dict:
    """Identify both a Git checkout and a source archive with BASE_COMMIT."""
    root = root.resolve()
    commit = None
    commit_source = "unavailable"
    dirty = None
    diff_hash = None
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == root:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            commit = revision.stdout.strip()
            commit_source = "git_HEAD"
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            dirty = status.stdout
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            diff_hash = hashlib.sha256(diff.stdout).hexdigest()
    except FileNotFoundError:
        pass
    if commit is None and (root / "BASE_COMMIT").is_file():
        commit = (root / "BASE_COMMIT").read_text().strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
            raise ValueError(
                "BASE_COMMIT must contain one full 40-character Git revision"
            )
        commit_source = "BASE_COMMIT"
    # Hash actual source files as well: a revision alone does not describe local
    # adaptations, and Git's diff does not include newly added files.
    files = {}
    suffixes = {".py", ".toml", ".sh", ".cpp", ".h", ".hpp", ".c", ".cmake", ".txt"}
    source_roots = (
        "python",
        "tokenspeed-kernel/python",
        "tokenspeed-kernel-npu/python",
        "tokenspeed-scheduler",
        "examples/task1_qwen35",
    )
    for source in source_roots:
        for path in sorted((root / source).rglob("*")):
            relative = path.relative_to(root)
            if any(
                part in {".git", "__pycache__", ".venv", "build"}
                for part in relative.parts
            ):
                continue
            if path.is_file() and (
                path.suffix in suffixes or path.name == "CMakeLists.txt"
            ):
                files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "commit": commit,
        "commit_source": commit_source,
        "working_tree_status": dirty,
        "git_diff_sha256": diff_hash,
        "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_file_count": len(files),
        "source_files_sha256": files,
    }


def environment() -> dict:
    versions = {}
    for name in (
        "torch",
        "torch-npu",
        "transformers",
        "triton-ascend",
        "tokenspeed-scheduler",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        **code_identity(Path(__file__).resolve().parents[2]),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }


def npu_snapshot() -> dict:
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def chat_input_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Render chat as plain token IDs suitable for the Engine's IPC payload."""
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise TypeError("chat template must return a flat list of integer token IDs")
    return ids


def exact_prompt(tokenizer, length: int, variant: int) -> list[int]:
    if length < 1:
        raise ValueError("input length must be positive")
    texts = (
        "智能计算系统需要验证模型的计算结果和运行效率。请解释为什么状态更新必须正确。",
        "A small language model processes a sequence of tokens. Explain memory and computation clearly. ",
    )
    seed = tokenizer.encode(texts[variant % len(texts)], add_special_tokens=False)
    if not seed:
        raise ValueError("tokenizer produced an empty fixture")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def timing_summary(samples: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = [s[key] for s in samples if s[key] is not None]
        return {
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(values),
        }

    return {key: stats(key) for key in ("ttft_ms", "tpot_ms", "total_ms")}


def stream_timings(
    events: list[dict], start: float, end: float, expected_tokens: int
) -> dict:
    nonempty = [event for event in events if event["new_ids"]]
    ids = [token for event in nonempty for token in event["new_ids"]]
    if len(ids) != expected_tokens:
        raise AssertionError(
            f"expected {expected_tokens} generated tokens, received {len(ids)}"
        )
    first = nonempty[0]["time"]
    last = nonempty[-1]["time"]
    # If IPC coalesces the first frame, it cannot identify the first token's time.
    exact_first = len(nonempty[0]["new_ids"]) == 1
    intervals = []
    for previous, current in zip(nonempty, nonempty[1:]):
        intervals.append(
            {
                "elapsed_ms": (current["time"] - previous["time"]) * 1000,
                "new_tokens": len(current["new_ids"]),
            }
        )
    return {
        "output_ids": ids,
        "first_frame_tokens": len(nonempty[0]["new_ids"]),
        "ttft_is_exact": exact_first,
        "ttft_ms": (first - start) * 1000 if exact_first else None,
        "first_frame_ms": (first - start) * 1000,
        "decode_steps": expected_tokens - 1,
        "decode_window_ms": (last - first) * 1000 if exact_first else None,
        "tpot_ms": (
            (last - first) * 1000 / (expected_tokens - 1)
            if exact_first and expected_tokens > 1
            else None
        ),
        "total_ms": (end - start) * 1000,
        "delivery_intervals": intervals,
    }
