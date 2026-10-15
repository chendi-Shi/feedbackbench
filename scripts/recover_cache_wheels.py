"""Recover original compatible pip HTTP wheel bodies, verified against PyPI.

No cache content is modified and no environment package is installed.  ZIP
metadata is inspected as data only; recovered bytes must match an official PyPI
release SHA256 and filename before entering the wheelhouse.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def _no_wmi(*args, **kwargs):
    raise OSError("Use the native Windows metadata fallback")


if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
    platform._wmi_query = _no_wmi

from packaging.tags import parse_tag, sys_tags  # noqa: E402

_WRITE_LOCK = threading.Lock()


def inspect_wheel(path: Path, supported: set) -> dict | None:
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"PK\x03\x04":
                return None
        with zipfile.ZipFile(path) as archive:
            wheel_paths = [name for name in archive.namelist() if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")]
            if len(wheel_paths) != 1:
                return None
            wheel_path = wheel_paths[0]
            wheel = BytesParser().parsebytes(archive.read(wheel_path))
            tags = set()
            for tag in wheel.get_all("Tag", []):
                tags.update(parse_tag(tag))
            if not tags.intersection(supported):
                return None
            metadata = BytesParser().parsebytes(archive.read(wheel_path.rsplit("/", 1)[0] + "/METADATA"))
            name, version = metadata["Name"], metadata["Version"]
            if not name or not version or name.lower() in {"torch", "torchvision", "torchaudio"}:
                return None
            return {"cache_path": str(path), "name": name, "version": version,
                    "size": path.stat().st_size, "wheel_tags": sorted(str(tag) for tag in tags)}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def recover(entry: dict, output: Path) -> dict:
    source = Path(entry["cache_path"])
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    url = f"https://pypi.org/pypi/{urllib.parse.quote(entry['name'], safe='')}/{urllib.parse.quote(entry['version'], safe='')}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            release = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as error:
        return {**entry, "status": "not_recovered", "reason": f"PyPI verification unavailable: {error}", "sha256": digest}
    matches = [item for item in release.get("urls", [])
               if item.get("packagetype") == "bdist_wheel" and item.get("digests", {}).get("sha256") == digest]
    if len(matches) != 1:
        return {**entry, "status": "not_recovered", "reason": "No unique official PyPI SHA256 match", "sha256": digest}
    official = matches[0]
    filename = official["filename"]
    if Path(filename).name != filename or not filename.endswith(".whl") or official["size"] != entry["size"]:
        raise ValueError("Unexpected official wheel metadata")
    target = output / filename
    with _WRITE_LOCK:
        if target.exists():
            method = "existing_verified"
        else:
            try:
                os.link(source, target)
                method = "hardlink"
            except FileExistsError:
                method = "existing_verified"
            except OSError:
                temporary = output / f".{filename}.{os.getpid()}.recovery-part"
                shutil.copyfile(source, temporary)
                if not target.exists():
                    try:
                        temporary.rename(target)
                    except FileExistsError:
                        pass
                if temporary.exists():
                    temporary.unlink()
                method = "copy"
        with target.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"Wheelhouse contains different bytes for {filename}")
    # Test the original ZIP central directory and each entry's CRC without extracting.
    with zipfile.ZipFile(target) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt wheel ZIP entry: {filename}:{corrupt}")
    return {**entry, "status": "recovered", "filename": filename, "sha256": digest,
            "source_url": official["url"], "method": method}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(os.environ["LOCALAPPDATA"]) / "pip" / "Cache" / "http-v2")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "wheelhouse")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    supported = set(sys_tags())
    entries = []
    scanned = 0
    for path in args.cache.rglob("*.body"):
        scanned += 1
        entry = inspect_wheel(path, supported)
        if entry:
            entries.append(entry)
    print(f"Scanned {scanned} HTTP bodies; found {len(entries)} compatible candidate wheels", flush=True)
    # Numeric/scientific wheels first so the parent can use them while other small
    # packages are still being verified.
    priority = {"scipy": 0, "numpy": 1, "numba": 2, "llvmlite": 3}
    entries.sort(key=lambda item: (priority.get(item["name"].lower(), 10), -item["size"]))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(recover, entry, args.output): entry for entry in entries}
        for future in as_completed(pending):
            try:
                record = future.result()
            except Exception as error:
                record = {**pending[future], "status": "not_recovered", "reason": repr(error)}
            records.append(record)
            if record["status"] == "recovered":
                print(f"Recovered {record['filename']} ({record['size']:,} bytes)", flush=True)
    manifest = {"python": sys.version, "cache": str(args.cache.resolve()), "scanned_bodies": scanned,
                "candidate_wheels": len(entries), "recovered_wheels": sum(item["status"] == "recovered" for item in records),
                "unique_recovered_wheels": len({item["filename"] for item in records if item["status"] == "recovered"}),
                "records": sorted(records, key=lambda item: (item["name"].lower(), item["version"]))}
    (args.output / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified recovery complete: {manifest['recovered_wheels']} wheels", flush=True)


if __name__ == "__main__":
    main()
