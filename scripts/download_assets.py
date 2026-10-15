"""Fetch the public, pinned LIBERO simulation assets with per-file verification.

Uses only the Python standard library.  It never loads checkpoint pickle data or
executes downloaded code.  A completed manifest records all hashes and licenses
remain in their upstream repositories.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.parse
import urllib.request


REPOSITORY = "lerobot/libero-assets"
REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"


def _hashes(path: Path, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest(), git_sha1.hexdigest()


def download_file(entry: dict, output: Path) -> dict:
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid remote asset path")
    target = output.joinpath(*relative.parts)
    size = entry["size"]
    expected = entry.get("lfs", {}).get("oid") or entry["oid"]
    algorithm = "sha256" if entry.get("lfs") else "git_blob_sha1"

    def verify(path):
        if not path.exists() or path.stat().st_size != size:
            return None
        sha256, git_sha1 = _hashes(path, size)
        actual = sha256 if algorithm == "sha256" else git_sha1
        if actual != expected:
            return None
        return {"path": entry["path"], "bytes": size, "sha256": sha256,
                "upstream_hash_algorithm": algorithm, "upstream_hash": expected}

    result = verify(target)
    if result is not None:
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{urllib.parse.quote(entry['path'], safe='/')}"
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "EmbodiedResearch/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            result = verify(temporary)
            if result is None:
                raise ValueError(f"Asset failed size/hash verification: {entry['path']}")
            temporary.replace(target)
            return result
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 10))
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "libero")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&limit=1000"
    entries = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        url = None
        for segment in link.split(","):
            if 'rel="next"' in segment:
                url = segment.split("<", 1)[1].split(">", 1)[0]
    files = [entry for entry in entries if entry["type"] == "file"]
    results = []
    print(f"Pinned asset snapshot: {len(files)} files, {sum(item['size'] for item in files):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_file, entry, args.output) for entry in files]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 50 == 0 or len(results) == len(files):
                print(f"Verified {len(results)}/{len(files)} assets", flush=True)
    manifest = {"repository": REPOSITORY, "repository_type": "dataset", "revision": REVISION,
                "file_count": len(results), "total_bytes": sum(item["bytes"] for item in results),
                "files": sorted(results, key=lambda item: item["path"])}
    (args.output / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {args.output / 'asset_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
