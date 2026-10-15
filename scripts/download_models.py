"""Download only pinned, public model artifacts; never deserialize pickle or fetch code."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "minerva": {
        "repo": "k1000dai/MINERVA",
        "revision": "1b4fb1743f00a7d8eb87c7059c446447907d12bf",
        "prefix": "t05_l1_0.54M/",
    },
    "smolvla": {
        "repo": "HuggingFaceVLA/smolvla_libero",
        "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "prefix": "",
    },
    "smolvlm_assets": {
        "repo": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        "prefix": "",
    },
}


def _request(url: str, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "EmbodiedResearch/0.1", **headers}),
        timeout=90,
    )


def _hashes(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def _valid(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    sha, git = _hashes(path)
    return sha == entry["lfs"]["oid"] if entry.get("lfs") else git == entry["oid"]


def download_file(url: str, destination: Path, entry: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _valid(destination, entry):
        partial = destination.with_name(destination.name + ".part")
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > entry["size"]:
                    offset = 0
                if offset == entry["size"] and _valid(partial, entry):
                    partial.replace(destination)
                    break
                if offset == entry["size"]:
                    offset = 0  # A complete but corrupt partial must restart.
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with _request(url, **headers) as response:
                    resume = offset > 0 and response.status == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server returned an unexpected resume offset")
                    with partial.open("ab" if resume else "wb") as stream:
                        last_report = time.monotonic()
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            if time.monotonic() - last_report > 15:
                                print(f"{destination.name}: {stream.tell():,}/{entry['size']:,} bytes", flush=True)
                                last_report = time.monotonic()
                if not _valid(partial, entry):
                    raise RuntimeError(f"Size or upstream hash mismatch for {destination.name}")
                partial.replace(destination)
                break
            except Exception as error:
                if attempt == 4:
                    raise
                print(f"Retry {attempt + 1}: {destination.name}: {error}", flush=True)
                time.sleep(min(2**attempt, 8))
        else:
            raise RuntimeError(f"Download did not complete: {destination}")
    sha, git = _hashes(destination)
    print(f"verified {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
    return {"file": destination.name, "size": destination.stat().st_size, "sha256": sha,
            "git_blob_sha1": git, "upstream_oid": entry["oid"],
            "upstream_lfs_sha256": entry.get("lfs", {}).get("oid"), "url": url}


def download_model(name: str, models: Path) -> dict:
    spec = SPECS[name]
    repo, revision, prefix = spec["repo"], spec["revision"], spec["prefix"]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}/{prefix}?expand=true"
    with _request(url) as response:
        entries = json.load(response)
    entries = [entry for entry in entries if entry["type"] == "file"]
    if prefix:
        with _request(f"https://huggingface.co/api/models/{repo}/tree/{revision}?expand=true") as response:
            root_entries = json.load(response)
        entries.extend(e for e in root_entries if e["path"] == "README.md" and e["type"] == "file")
    if name == "smolvlm_assets":
        entries = [e for e in entries if e["path"].endswith((".json", ".txt", ".jinja"))]
    else:
        entries = [e for e in entries if e["path"].endswith((".json", ".safetensors", ".md"))]
    if not entries or (name != "smolvlm_assets" and not any(e["path"].endswith("model.safetensors") for e in entries)):
        raise RuntimeError(f"Missing expected artifacts for {name}")
    destination = models / name
    records = []
    for entry in entries:
        relative = entry["path"].removeprefix(prefix)
        if Path(relative).name != relative:
            raise ValueError(f"Unexpected nested artifact path: {relative}")
        if name == "smolvlm_assets" and entry["size"] > 15_000_000:
            raise ValueError("Refusing an unexpectedly large backbone metadata artifact")
        resolved = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(entry['path'])}"
        records.append(download_file(resolved, destination / relative, entry))
    manifest = {"schema_version": 1, "name": name, **spec, "files": records,
                "downloaded_unix": time.time(), "verified_against_upstream": True}
    manifest_path = destination / "download_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--only", choices=list(SPECS), action="append")
    args = parser.parse_args()
    for name in args.only or list(SPECS):
        download_model(name, args.models.resolve())


if __name__ == "__main__":
    main()
