"""Lightweight audit helpers for the independently registered action study."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from budget_campaign_common_v5 import write_json

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = "2026-10-30T06:08:00Z"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(relative, base=ROOT):
    base = Path(base).resolve()
    result = (base / relative).resolve()
    if result == base or not result.is_relative_to(base):
        raise ValueError(f"Path escapes its audited root: {relative}")
    return result


def deadline_reached():
    return datetime.now(timezone.utc) >= datetime.fromisoformat(DEADLINE.replace("Z", "+00:00"))


def verify_protocol(path, expected_sha256, *, sources=True, inputs=False):
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError("Frozen action protocol changed")
    protocol = read(path)
    if protocol.get("schema") != "feedbackbench.action_adaptation_train.v1" or protocol.get("frozen") is not True:
        raise ValueError("Expected a frozen independent action-training protocol")
    if protocol.get("deadline_utc") != DEADLINE:
        raise ValueError("Original study deadline cannot be extended")
    # Bind normalization/tokenizer metadata as well as model tensors. PolicyAdapter
    # verifies each file against these pinned manifests before model construction.
    model_files = protocol["model"]["files"]
    model_root = resolve(protocol["model"]["models_dir"])
    for name in ("smolvla", "smolvlm_assets"):
        manifest_path = model_root / name / "download_manifest.json"
        relative = manifest_path.relative_to(ROOT).as_posix()
        if relative not in model_files or digest(manifest_path) != model_files[relative]:
            raise ValueError("Pinned model/preprocessor manifest changed")
        for record in read(manifest_path)["files"]:
            relative = (manifest_path.parent / record["file"]).relative_to(ROOT).as_posix()
            if model_files.get(relative) != record["sha256"]:
                raise ValueError("Protocol does not bind every original model/preprocessor file")
    def safe_id(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value) is not None
    if not protocol.get("runs") or not all(safe_id(name) for name in protocol["runs"]):
        raise ValueError("Unsafe training run identifier")
    evaluation = protocol["evaluation"]
    method_ids = [method["method_id"] for method in evaluation["methods"]]
    case_ids = [case["case_id"] for case in evaluation["cases"]]
    if (not all(safe_id(name) for name in method_ids + case_ids)
            or len(set(method_ids)) != len(method_ids) or len(set(case_ids)) != len(case_ids)
            or any(case["method_id"] not in method_ids for case in evaluation["cases"])):
        raise ValueError("Unsafe or duplicated evaluation identifiers")
    if sources:
        if not protocol.get("source_files"):
            raise ValueError("Source binding is empty")
        for relative, expected in protocol["source_files"].items():
            if digest(resolve(relative)) != expected:
                raise ValueError(f"Frozen action source changed: {relative}")
    if inputs:
        for relative, expected in model_files.items():
            if digest(resolve(relative)) != expected:
                raise ValueError(f"Frozen model/preprocessor input changed: {relative}")
        records = [protocol["data"]["manifest"], protocol["data"]["alignment_evidence"]]
        records += [run["schedule"] for run in protocol["runs"].values()]
        for record in records:
            if digest(resolve(record["path"])) != record["sha256"]:
                raise ValueError(f"Frozen input changed: {record['path']}")
        manifest = read(resolve(protocol["data"]["manifest"]["path"]))
        unique = {demo["path"]: demo["sha256"] for demo in manifest["demos"]}
        for relative, expected in unique.items():
            if digest(resolve(relative)) != expected:
                raise ValueError(f"Expert source changed: {relative}")
    return protocol


def seal(directory, binding):
    """Publish a receipt last. Partial attempts never get a success receipt."""
    directory = Path(directory)
    if (directory / "receipt.json").exists():
        raise FileExistsError("Refusing to overwrite a sealed result")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    write_json(directory / "receipt.json", {"schema": "feedbackbench.action_result_receipt.v1",
                                            "sealed_at": now(), "binding": binding, "files": files})
    return digest(directory / "receipt.json")


def verify_seal(directory, binding):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    if receipt.get("binding") != binding or not receipt.get("files"):
        raise ValueError("Sealed result binding mismatch")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(receipt["files"]) | {"receipt.json"}:
        raise ValueError("Sealed result file set changed")
    for relative, record in receipt["files"].items():
        path = resolve(relative, directory)
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Sealed evidence changed: {relative}")
    return receipt
