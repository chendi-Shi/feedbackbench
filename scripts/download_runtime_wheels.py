"""Download pinned wheels through mirrors, verified against official PyPI metadata.

This script never installs packages. Supply exact name==version pins or a pip
--report JSON. Source distributions, unpinned requests and unknown digests fail.
The default supported interpreter is CPython 3.12 / Windows AMD64.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRRORS = ("https://pypi.tuna.tsinghua.edu.cn", "https://mirrors.ustc.edu.cn/pypi")


def get_url(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "EmbodiedResearch-WheelVerifier/0.1"}),
        timeout=60,
    )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_score(filename: str) -> int:
    if not filename.endswith(".whl"):
        return -1
    if filename.endswith("-cp312-cp312-win_amd64.whl"):
        return 100
    if filename.endswith("-cp312-abi3-win_amd64.whl"):
        return 95
    if filename.endswith("-abi3-win_amd64.whl"):
        python_tag = filename.rsplit("-", 3)[1]
        if python_tag.startswith("cp3") and python_tag[3:].isdigit() and int(python_tag[3:]) <= 12:
            return 70 + int(python_tag[3:])
    if filename.endswith("-py3-none-win_amd64.whl"):
        return 65
    if filename.endswith(("-py3-none-any.whl", "-py2.py3-none-any.whl")):
        return 50
    return -1


def download(pin: str, directory: Path, requested_filename: str | None = None) -> dict:
    if pin.count("==") != 1:
        raise ValueError(f"An exact name==version pin is required: {pin}")
    name, version = pin.split("==")
    metadata_url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{urllib.parse.quote(version)}/json"
    with get_url(metadata_url) as response:
        metadata = json.load(response)
    candidates = [item for item in metadata["urls"] if wheel_score(item["filename"]) >= 0]
    if requested_filename:
        candidates = [item for item in candidates if item["filename"] == requested_filename]
    if not candidates:
        raise ValueError(f"No compatible CPython 3.12 Windows AMD64 wheel for {pin}")
    entry = max(candidates, key=lambda item: wheel_score(item["filename"]))
    filename = entry["filename"]
    if Path(filename).name != filename:
        raise ValueError("Invalid wheel filename")
    expected_hash = entry["digests"]["sha256"]
    expected_size = entry["size"]
    path = directory / filename
    temporary = path.with_name(filename + ".mirror.part")
    started = time.monotonic()
    source = "existing verified wheel"

    def valid(candidate: Path) -> bool:
        return candidate.is_file() and candidate.stat().st_size == expected_size and digest(candidate) == expected_hash

    if not valid(path):
        package_path = urllib.parse.urlparse(entry["url"]).path
        urls = [mirror + package_path for mirror in MIRRORS] + [entry["url"]]
        last_error = None
        for source in urls:
            try:
                with get_url(source) as response, temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                if not valid(temporary):
                    raise ValueError("Downloaded bytes do not match official PyPI size/SHA256")
                # Another cache-recovery worker may have produced the same file.
                if valid(path):
                    temporary.unlink()
                else:
                    temporary.replace(path)
                break
            except Exception as error:
                last_error = error
                print(f"retry source for {filename}: {error}", flush=True)
        else:
            raise RuntimeError(f"All trusted sources failed for {filename}") from last_error
    record = {"name": name, "version": version, "filename": filename, "size": expected_size,
              "sha256": expected_hash, "official_metadata_url": metadata_url,
              "official_file_url": entry["url"], "download_source": source,
              "elapsed_seconds": time.monotonic() - started, "verified": True}
    provenance = path.with_name(filename + ".provenance.json")
    provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFIED {filename} {expected_size:,} bytes {record['elapsed_seconds']:.2f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--report", type=Path, help="Existing pip install --dry-run --report file")
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts" / "wheelhouse")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("Use 1–4 concurrent downloads")
    requests = [(pin, None) for pin in args.package]
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        for item in report["install"]:
            metadata = item["metadata"]
            download_url = item.get("download_info", {}).get("url", "")
            filename = urllib.parse.unquote(urllib.parse.urlparse(download_url).path.rsplit("/", 1)[-1])
            if not filename.endswith(".whl"):
                print(f"SKIP non-wheel requirement: {metadata['name']}", flush=True)
                continue
            requests.append((f"{metadata['name']}=={metadata['version']}", filename))
    requests = list(dict.fromkeys(requests))
    if not requests:
        parser.error("Provide --package or --report")
    args.directory.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, pin, args.directory, filename): pin for pin, filename in requests}
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
