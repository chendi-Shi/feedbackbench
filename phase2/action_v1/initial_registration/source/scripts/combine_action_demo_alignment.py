"""Bind four completed single-source alignment receipts without replaying again."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine(manifest_path, receipt_paths):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    sources = manifest["sources"]
    expected = {s["suite"]: s for s in sources}
    if set(expected) != {"libero_spatial", "libero_object", "libero_goal", "libero_10"} or len(sources) != 4:
        raise ValueError("Exactly four fixed source suites required")
    if len(receipt_paths) != 4:
        raise ValueError("Four complete individual receipts required")
    records, receipts, offsets, scripts, observed = [], [], set(), set(), set()
    for path in map(Path, receipt_paths):
        receipt = json.loads(path.read_text())
        if (receipt["status"] != "passed" or receipt.get("scope") != "observation_action_alignment"
                or receipt.get("dynamics_replay_verified") is not False
                or receipt.get("diagnostic_revision") != 3 or len(receipt["records"]) != 1
                or receipt["source_manifest_sha256"] != manifest_hash):
            raise ValueError("Receipt is incomplete, wrong revision, or bound to another manifest")
        record = receipt["records"][0]
        suite = record["suite"]
        if suite not in expected or suite in observed:
            raise ValueError("Duplicate or unexpected source suite")
        source = expected[suite]
        if any(record.get(key) != source.get(key) for key in ("sha256", "bytes", "path", "task_uid", "task_id")):
            raise ValueError("Source identity/hash mismatch")
        if record["status"] != "passed" or record["replay"]["status"] != "passed":
            raise ValueError("Individual source checks did not pass")
        if len(record["demonstrations"]) != 50 or len(record["replay"]["samples"]) != 5:
            raise ValueError("Expected all 50 demonstrations and 5 fixed reconstruction samples")
        for key in ("v1_receipt", "prior_observation_receipt"):
            reference = receipt[key]
            reference_path = Path(reference["path"])
            if not reference_path.is_absolute():
                reference_path = ROOT / reference_path
            if not reference_path.is_file() or digest(reference_path) != reference["sha256"]:
                raise ValueError("Preserved prior diagnostic receipt is missing or changed")
        offsets.add(receipt["action_offset"])
        offsets.add(record["action_offset"])
        scripts.add(receipt["script_sha256"])
        observed.add(suite)
        records.append(copy.deepcopy(record))
        receipts.append({"suite": suite, "path": str(path), "sha256": digest(path),
                         "v1_receipt": receipt["v1_receipt"],
                         "prior_observation_receipt": receipt["prior_observation_receipt"]})
    if len(offsets) != 1 or not offsets.issubset({0, 1}) or len(scripts) != 1:
        raise ValueError("Sources do not share one action offset and validator revision")
    return {"schema": "feedbackbench.action_alignment_study.r3", "status": "passed",
            "scope": "observation_action_alignment", "coverage": "four_source_study",
            "diagnostic_revision": 3, "action_offset": next(iter(offsets)),
            "dynamics_replay_verified": False, "source_manifest_sha256": manifest_hash,
            "validator_script_sha256": next(iter(scripts)), "records": records,
            "single_source_receipts": receipts,
            "limitations": "Verifies observation/state/action indexing and original sensor timing; "
                           "does not claim complete dynamics reproduction or unseen-task generalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/action_adaptation/study_v1/source_manifest.json")
    parser.add_argument("--reports", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/action_adaptation/study_v1/alignment_r3.json")
    args = parser.parse_args()
    result = combine(args.source_manifest, args.reports)
    result["combiner_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "action_offset": result["action_offset"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
