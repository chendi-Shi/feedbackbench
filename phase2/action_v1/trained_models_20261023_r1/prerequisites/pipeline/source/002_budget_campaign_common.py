"""Small, model-free evidence and deadline helpers for the compute campaign."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs/learned_compute/v4"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def deadline_reached(protocol, now=None):
    return (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(protocol["deadline_utc"].replace("Z", "+00:00"))


def digest(path):
    path = Path(path)
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path.absolute()))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def case_directory(case):
    return hashlib.sha256(case["case_id"].encode()).hexdigest()[:16]


def owned_path(campaign, *parts):
    campaign = Path(campaign).resolve()
    path = campaign.joinpath(*parts).resolve()
    if path == campaign or not path.is_relative_to(campaign):
        raise ValueError("expected an owned child of the campaign")
    return path


def fingerprint(protocol):
    return canonical_hash({key: value for key, value in protocol.items() if key != "campaign_fingerprint"})


def verify_protocol(campaign):
    campaign = Path(campaign).resolve()
    protocol = read_json(campaign / "protocol.json")
    if protocol["campaign_fingerprint"] != fingerprint(protocol):
        raise ValueError("campaign protocol fingerprint changed")
    for relative, expected in protocol["sources"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError(f"frozen source changed: {relative}; create a new protocol revision")
    for relative, expected in protocol["input_manifests"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"input manifest changed: {relative}")
    for source in protocol["official_sources"]:
        if digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("official task mapping/order source changed")
    import importlib.metadata
    for name, version in protocol["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"runtime package version changed: {name}")
    return protocol


def seal_episode(directory):
    directory = Path(directory)
    if (directory / "receipt.json").exists():
        raise FileExistsError("receipt already exists")
    files = {path.relative_to(directory).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(directory.rglob("*")) if path.is_file() and not path.name.endswith(".tmp")}
    required = {"metadata.json", "result.json", "trajectory.npz", "samples.npz"}
    if not required <= files.keys():
        raise ValueError("cannot seal an incomplete episode")
    write_json(directory / "receipt.json", {"schema": "feedbackbench.budget_episode_receipt.v1",
               "status": "complete", "sealed_at": utc_now(), "files": files})


def verify_episode(directory, case=None, protocol=None, strict=False):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    if receipt.get("schema") != "feedbackbench.budget_episode_receipt.v1" or receipt.get("status") != "complete":
        raise ValueError("not a completed budget episode receipt")
    names = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} - {"receipt.json"}
    if names != receipt["files"].keys():
        raise ValueError("receipt inventory mismatch")
    for name, record in receipt["files"].items():
        path = owned_path(directory, name)
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"receipt digest mismatch: {name}")
    metadata, result = read_json(directory / "metadata.json"), read_json(directory / "result.json")
    if not result.get("complete") or not (result.get("terminated") or result.get("truncated")):
        raise ValueError("episode did not reach an official terminal condition")
    if case and (metadata["case_id"] != case["case_id"] or metadata["task_key"] != case["task_uid"]):
        raise ValueError("completed case identity mismatch")
    if protocol and metadata["campaign_fingerprint"] != protocol["campaign_fingerprint"]:
        raise ValueError("episode belongs to a different campaign")
    if strict:
        import numpy as np
        if not case or not protocol or metadata["case"] != case or metadata["split"] != case["split"]:
            raise ValueError("episode full case metadata mismatch")
        phase = metadata["phase"]
        method = None if phase == "collection" else next(
            row for row in protocol["evaluation"]["methods"] if row["method_id"] == case["method_id"])
        if metadata["method"] != method:
            raise ValueError("episode method differs from the planned method")
        if phase == "evaluation":
            campaign = directory.parents[1] if directory.parent.name == "evaluation" else directory.parents[3]
            binding = metadata["trained_binding"]
            if binding["manifest_sha256"] != digest(campaign / "trained_manifest.json"):
                raise ValueError("episode trained-controller manifest binding changed")
            if method["family"] != "fixed":
                variant = method["feature_mode"] if method["family"] == "learned" else "full"
                expected_path = f"trained/{variant}_seed{method['seed']}/selector.pt"
                if (binding["selector"]["path"] != expected_path
                        or binding["selector"]["sha256"] != digest(owned_path(campaign, expected_path))):
                    raise ValueError("episode selected controller binding mismatch")
        horizon = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[case["suite"]]
        if (result["horizon"] != horizon or not 0 < result["steps"] <= horizon
                or (result["truncated"] and not result["terminated"] and result["steps"] != horizon)):
            raise ValueError("episode ended before its official horizon without termination")
        calls = [read_json(path) for path in sorted((directory / "decisions").glob("*.json"))]
        if len(calls) != result["decisions"] or sum(row["executed_steps"] for row in calls) != result["steps"]:
            raise ValueError("decision and executed-action accounting mismatch")
        for index, row in enumerate(calls):
            expected_nfe = 14 if phase == "collection" else row["budget"]
            if row["decision"] != index or row["budget"] not in (2, 4, 10) or row["actual_nfe"] != expected_nfe:
                raise ValueError("per-decision NFE accounting mismatch")
        if sum(row["actual_nfe"] for row in calls) != result["actual_nfe"]:
            raise ValueError("total NFE accounting mismatch")
        if {str(k): sum(row["budget"] == k for row in calls) for k in (2, 4, 10)} != result["budget_frequencies"]:
            raise ValueError("budget frequency accounting mismatch")
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            if (data["actions"].shape != (result["steps"], 7) or data["states"].shape != (result["steps"], 8)
                    or not np.isfinite(data["actions"]).all() or not np.isfinite(data["states"]).all()):
                raise ValueError("executed trajectory does not match the complete result")
    return metadata, result
