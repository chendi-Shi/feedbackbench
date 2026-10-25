"""Prepare one real official LIBERO demonstration probe without Torch/models.

Example: .venv/Scripts/python.exe -B scripts/prepare_action_adaptation_data.py --download
Metadata only: add --metadata-only (no HDF5 download or numpy/h5py import).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_research.action_demo_data import (  # noqa: E402
    MAX_DOWNLOAD_BYTES, SOURCE_CODE_URL, download_file,
    inspect_source, prepare_example, sha256_file, write_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/action_adaptation/probe_v1")
    parser.add_argument("--download", action="store_true", help="Fetch the one pinned HDF5 if absent")
    parser.add_argument("--metadata-only", action="store_true", help="Inspect pinned metadata only; no arrays/models")
    args = parser.parse_args(argv)
    if args.download and args.metadata_only:
        parser.error("--download and --metadata-only are mutually exclusive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = inspect_source()
    write_json(output / "source_metadata.json", source)
    if args.metadata_only:
        print(json.dumps(source, indent=2))
        return 0
    # Preserve the exact official source used to motivate the explicit +1 probe
    # index. It is evidence, never imported/executed. Bound all network bytes.
    evidence_path = output / "official_create_dataset.py.txt"
    evidence_bytes_downloaded = 0
    if not evidence_path.exists():
        with urllib.request.urlopen(SOURCE_CODE_URL, timeout=60) as response:
            code = response.read(256 * 1024 + 1)
        if len(code) > 256 * 1024:
            raise ValueError("Official source evidence exceeds 256 KiB cap")
        with evidence_path.open("xb") as stream:
            stream.write(code)
        evidence_bytes_downloaded = len(code)
    source["source_code_evidence_file"] = evidence_path.name
    source["source_code_evidence_sha256"] = sha256_file(evidence_path)
    # Keep the original long upstream path in the manifest, not in a Windows
    # local filename under this already long workspace root.
    source_path = output / "task0.hdf5"
    if args.download:
        download = download_file(
            source_path, source,
            byte_limit=MAX_DOWNLOAD_BYTES - source["metadata_response_bytes"] - evidence_bytes_downloaded,
        )
    elif not source_path.exists():
        raise FileNotFoundError("Pinned demonstration is absent; use --download to fetch the one reviewed file")
    else:
        download = {"downloaded_bytes": 0, "reused_verified_file": True}
    source["download"] = download
    source["download"]["metadata_bytes"] = source["metadata_response_bytes"]
    source["download"]["source_code_bytes"] = evidence_bytes_downloaded
    source["download"]["total_payload_bytes_this_run"] = (
        download["downloaded_bytes"] + source["metadata_response_bytes"] + evidence_bytes_downloaded
    )
    result = prepare_example(source_path, output, source=source)
    print(json.dumps({"status": result["status"], "metadata": str(output / "metadata.json"),
                      "example": str(output / "example.npz"),
                      "source_bytes": source["bytes"], "download": source["download"],
                      "alignment_replay_verified": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
