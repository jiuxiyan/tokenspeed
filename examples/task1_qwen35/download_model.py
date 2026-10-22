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

"""Download revision-pinned Qwen3.5-0.8B files from its official ModelScope repo.

Every existing file is hashed in bounded chunks. Downloads replace their final
path only after both size and SHA256 pass; failures leave a .partial file.
A subsequent run restarts an incomplete download rather than trusting its bytes.
This utility uses only Python's standard library and does not initialize an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

_REPOSITORY_URL = "https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B/repo"
_CHUNK_BYTES = 8 * 1024 * 1024


def load_manifest(path: Path) -> list[dict]:
    """Read the fixed flat-file list and reject unpinned or ambiguous entries."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty list of file entries")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "Path",
            "Revision",
            "Sha256",
            "Size",
        }:
            raise ValueError(
                "each manifest entry needs Path, Revision, Sha256 and Size"
            )
        name = entry["Path"]
        if (
            not isinstance(name, str)
            or not name
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or name.endswith(".partial")
            or name in seen
        ):
            raise ValueError(f"invalid or duplicate model filename: {name!r}")
        if not isinstance(entry["Revision"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", entry["Revision"]
        ):
            raise ValueError(f"{name}: Revision must be a fixed 40-character commit")
        if not isinstance(entry["Sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["Sha256"]
        ):
            raise ValueError(f"{name}: invalid SHA256")
        if type(entry["Size"]) is not int or entry["Size"] < 0:
            raise ValueError(f"{name}: Size must be a nonnegative integer")
        seen.add(name)
    return entries


def file_sha256(path: Path) -> str:
    """Hash a file without loading the model weights into host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(entry: dict, destination: Path) -> None:
    """Verify or download one file, preserving its final path on failure."""
    name = entry["Path"]
    target = destination / name
    if target.exists():
        existing_sha256 = file_sha256(target)
        if (
            target.stat().st_size == entry["Size"]
            and existing_sha256 == entry["Sha256"]
        ):
            print(f"verified existing: {name} ({entry['Size']} bytes)", flush=True)
            return

    query = urllib.parse.urlencode({"Revision": entry["Revision"], "FilePath": name})
    partial = target.with_name(f"{target.name}.partial")
    if partial.is_symlink():
        raise ValueError(f"partial download must not be a symlink: {partial}")
    digest = hashlib.sha256()
    downloaded_size = 0
    print(f"downloading: {name} at {entry['Revision']}", flush=True)
    with urllib.request.urlopen(f"{_REPOSITORY_URL}?{query}", timeout=120) as response:
        with partial.open("wb") as output:
            while chunk := response.read(_CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
                downloaded_size += len(chunk)
    actual_sha256 = digest.hexdigest()
    if downloaded_size != entry["Size"] or actual_sha256 != entry["Sha256"]:
        raise RuntimeError(
            f"verification failed for {name}: got {downloaded_size} bytes / "
            f"{actual_sha256}, expected {entry['Size']} bytes / {entry['Sha256']}; "
            f"unverified bytes remain at {partial}"
        )
    partial.replace(target)
    print(
        f"verified download: {name} ({downloaded_size} bytes) {actual_sha256}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("model_manifest.json"),
        help="fixed file manifest; defaults to model_manifest.json beside this script",
    )
    args = parser.parse_args()
    entries = load_manifest(args.manifest)
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        download_file(entry, destination)
    print(f"ALL {len(entries)} FILES VERIFIED: {destination}", flush=True)


if __name__ == "__main__":
    main()
