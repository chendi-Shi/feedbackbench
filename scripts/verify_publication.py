"""Verify publication bytes and explicitly declared redacted diagnostic bindings.

This is an integrity check of the exported evidence, not a model/simulator run or
a replacement for numerical validation of the original local experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{64}\Z")
RUN_DIAGNOSTIC = re.compile(r"outputs/frontier/(?:vlajepa_isolated_closed_loop|vlajepa_long_horizon|vlajepa_recovery)/runs/run_[0-9]{4}\.json\Z")
REDACTION_FIELD = re.compile(r"compatibility\.patches\[([0-9]+)\]\.changes\[([0-9]+)\]\.after\Z")
PRIVATE_PATH = re.compile(r'''(?i)(?:\b[A-Z]:[\\/]+Users[\\/]+|/(?:home|Users)/)[^"'<>\r\n]+''')
REDACTION_TOKEN = "<REDACTED_LOCAL_PATH>"
REDACTION_KIND = "compatibility_patch_local_path"
ISOLATED = "outputs/frontier/vlajepa_isolated_closed_loop"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def _safe_name(name):
    require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name,
            "Invalid publication path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(part not in {"", ".", ".."} for part in name.split("/")),
            "Escaping or noncanonical publication path: " + name)
    return path


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def redact_run_diagnostic(name, raw):
    """Exporter helper: alter only private paths in compatibility patch strings."""
    require(bool(RUN_DIAGNOSTIC.fullmatch(name)), "Run diagnostic is outside the redaction allowlist")
    original = _json(raw)
    cleaned, fields = deepcopy(original), []
    for i, patch in enumerate(cleaned.get("compatibility", {}).get("patches", [])):
        for j, change in enumerate(patch.get("changes", [])):
            value = change.get("after")
            if not isinstance(value, str):
                continue
            replacement = PRIVATE_PATH.sub(REDACTION_TOKEN, value)
            if replacement != value:
                change["after"] = replacement
                fields.append(f"compatibility.patches[{i}].changes[{j}].after")
    encoded = json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    require(not PRIVATE_PATH.search(encoded), "Private path outside allowed compatibility patch fields: " + name)
    if not fields:
        return raw, False, None
    return encoded.encode("utf-8"), True, {"kind": REDACTION_KIND, "fields": fields}


def _validate_redaction(name, entry, data):
    require(bool(RUN_DIAGNOSTIC.fullmatch(name)), "Source-hash redaction mapping is outside the diagnostic allowlist")
    descriptor = entry.get("redaction")
    require(isinstance(descriptor, dict) and set(descriptor) == {"kind", "fields"}
            and descriptor["kind"] == REDACTION_KIND, "Missing or invalid run diagnostic redaction descriptor")
    fields = descriptor["fields"]
    require(isinstance(fields, list) and bool(fields) and all(isinstance(item, str) for item in fields)
            and len(fields) == len(set(fields)), "Invalid diagnostic redaction fields")
    require(entry.get("redacted") is True and bool(SHA.fullmatch(str(entry.get("source_sha256", ""))))
            and entry["source_sha256"] != entry["sha256"]
            and type(entry.get("source_bytes")) is int and entry["source_bytes"] > 0,
            "Invalid original/exported diagnostic identities")
    marked = []

    def walk(value, field=""):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, field + "." + key if field else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{field}[{index}]")
        elif isinstance(value, str) and REDACTION_TOKEN in value:
            marked.append(field)

    walk(data)
    require(set(marked) == set(fields), "Diagnostic redaction fields do not match exported markers")
    for field in fields:
        match = REDACTION_FIELD.fullmatch(field)
        require(match is not None, "Redaction altered a field outside compatibility patch strings")
        try:
            value = data["compatibility"]["patches"][int(match[1])]["changes"][int(match[2])]["after"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Declared redaction field is absent") from error
        require(isinstance(value, str) and REDACTION_TOKEN in value, "Missing diagnostic redaction marker")
    require(not PRIVATE_PATH.search(json.dumps(data, ensure_ascii=False)), "Exported diagnostic retains a private path")


def verify_publication(root):
    root = Path(root).resolve()
    manifest_path = root / "PUBLICATION_MANIFEST.json"
    manifest = _json(manifest_path.read_bytes())
    require(manifest.get("schema") == "feedbackbench.github_publication.v1", "Unexpected publication manifest schema")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Missing publication file inventory")
    actual, paths, redactions = {}, {}, {}

    def path_for(name):
        _safe_name(name)
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), "Missing or escaping publication file: " + name)
        return path

    for name, entry in files.items():
        require(name != "PUBLICATION_MANIFEST.json", "Manifest cannot include itself in its file inventory")
        path = path_for(name)
        require(isinstance(entry, dict) and bool(SHA.fullmatch(str(entry.get("sha256", ""))))
                and type(entry.get("bytes")) is int and entry["bytes"] >= 0, "Invalid manifest file metadata: " + name)
        checksum = _sha(path)
        require(path.stat().st_size == entry["bytes"] and checksum == entry["sha256"], "Publication byte/hash mismatch: " + name)
        actual[name], paths[name] = checksum, path
        if "redaction" in entry or RUN_DIAGNOSTIC.fullmatch(name) and entry.get("redacted") is True:
            _validate_redaction(name, entry, _json(path.read_bytes()))
            redactions[name] = entry

    def read(name):
        require(name in files, "Required file is absent from publication manifest: " + name)
        return _json(paths[name].read_bytes())

    def binding(name, expected, *, permit_redaction=False):
        _safe_name(name)
        require(name in files and isinstance(expected, str) and bool(SHA.fullmatch(expected)), "Invalid or missing evidence binding: " + name)
        if permit_redaction and name in redactions:
            require(expected == files[name]["source_sha256"], "Original diagnostic source binding differs: " + name)
            return {"path": name, "expected_sha256": expected, "exported_sha256": actual[name],
                    "mode": "source_binding_via_redaction_manifest", "redacted_fields": files[name]["redaction"]["fields"]}
        require(actual[name] == expected, "Evidence byte hash differs: " + name)
        return {"path": name, "expected_sha256": expected, "exported_sha256": actual[name], "mode": "byte_verified"}

    source_checks = manifest.get("frozen_source_checks")
    require(isinstance(source_checks, dict), "Missing frozen source inventory")
    checked_sources = {}
    for name, count in source_checks.items():
        protocol = read(name)
        sources = protocol.get("source_sha256", protocol.get("sources"))
        require(type(count) is int and count > 0 and isinstance(sources, dict) and len(sources) == count,
                "Frozen source inventory/count differs: " + name)
        for source, expected in sources.items():
            binding(source, expected)
        checked_sources[name] = count

    receipts, receipt_files = [], 0
    for name in sorted(files):
        if not name.endswith("/receipt.json") or "/episodes/" not in name:
            continue
        receipt = read(name)
        contents = receipt.get("files")
        require(isinstance(contents, dict) and bool(contents), "Empty or invalid episode receipt: " + name)
        parent = PurePosixPath(name).parent
        require(all(isinstance(item, str) and _safe_name(item).name == item and item != "receipt.json" for item in contents),
                "Episode receipt contains a nonlocal filename")
        entries = list(paths[name].parent.iterdir())
        require(all(item.is_file() for item in entries), "Episode artifact directory contains an unexpected subdirectory: " + name)
        present = {item.name for item in entries if item.name != "receipt.json"}
        require(present == set(contents), "Episode artifact inventory differs: " + name)
        for filename, expected in contents.items():
            binding(str(parent / filename), expected)
            receipt_files += 1
        receipts.append(name)

    worker_bindings, complete_workers, identities = [], set(), set()
    worker_names = sorted(name for name in files if name.startswith(ISOLATED + "/workers/") and re.fullmatch(r"worker_[0-9]{4}\.json", PurePosixPath(name).name))
    protocol_name = ISOLATED + "/protocol.json"
    if protocol_name in files:
        protocol = read(protocol_name)
        cases = {row["id"]: row for row in protocol["spec"]["cases"]}
        sealed = {PurePosixPath(name).parent.name for name in receipts if name.startswith(ISOLATED + "/episodes/")}
        for name in worker_names:
            worker = read(name)
            require(worker.get("schema") == "feedbackbench.vlajepa_isolated_worker.v1"
                    and worker.get("protocol_fingerprint") == protocol["fingerprint"], "Worker protocol/schema differs")
            binding("scripts/run_vlajepa_isolated.py", worker.get("script_sha256"))
            pid, created = worker.get("pid"), worker.get("process_create_time")
            require(type(pid) is int and pid > 0 and type(created) in (int, float) and math.isfinite(created) and created > 0,
                    "Invalid worker process identity")
            require((pid, created) not in identities, "Worker process identity reused")
            identities.add((pid, created))
            require(worker.get("status") in {"started", "complete", "blocked_preflight", "failed"}, "Invalid worker status")
            before, case_id = worker["statuses_before"], worker["case_id"]
            require(set(before) == set(cases) and all(value in {"complete", "missing"} for value in before.values())
                    and before.get(case_id) == "missing" and worker["completed_before"] == sum(value == "complete" for value in before.values()),
                    "Worker initial case inventory differs")
            runs = worker.get("run_records", [])
            require(isinstance(runs, list), "Invalid worker run inventory")
            for reference in runs:
                require(reference["path"].startswith(ISOLATED + "/runs/") and RUN_DIAGNOSTIC.fullmatch(reference["path"]),
                        "Worker references an unrelated run diagnostic")
                worker_bindings.append({**binding(reference["path"], reference["sha256"], permit_redaction=True), "worker": name})
            if worker["status"] == "complete":
                require(worker["statuses_after"] == {**before, case_id: "complete"}
                        and worker["completed_after"] == worker["completed_before"] + 1
                        and worker["newly_completed"] == [case_id] and case_id not in complete_workers,
                        "Worker must complete exactly one distinct case")
                require(case_id in sealed and len(runs) == 1, "Completed worker lacks one sealed case/run")
                result = read(ISOLATED + "/episodes/" + case_id + "/result.json")
                require(result["id"] == case_id and result["model_info"]["run_record"] == runs[0]["path"],
                        "Sealed result references another worker run")
                complete_workers.add(case_id)
        require(complete_workers == sealed, "Sealed isolated episode lacks a complete independent worker")
    else:
        require(not worker_names, "Worker evidence lacks its protocol")

    return {"schema": "feedbackbench.publication_verification.v1", "status": "passed",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": _sha(manifest_path), "files_verified": len(files),
            "frozen_source_checks": checked_sources, "episode_receipts_verified": len(receipts),
            "receipt_files_verified": receipt_files, "isolated_workers_verified": len(worker_names),
            "isolated_worker_run_bindings": worker_bindings,
            "redacted_diagnostics": [{"path": name, "source_sha256": entry["source_sha256"],
                "exported_sha256": entry["sha256"], "mode": "source_binding_via_redaction_manifest",
                "redacted_fields": entry["redaction"]["fields"]} for name, entry in sorted(redactions.items())],
            "limitations": ["All published file bytes, frozen sources and episode receipt hashes were recomputed.",
                "For allowlisted redacted run diagnostics, original source hashes are declarations in PUBLICATION_MANIFEST.json; they are not hashes of the exported bytes.",
                "Original worker/report/audit references remain unchanged. The absent original private strings cannot be independently reconstructed from the export.",
                "This integrity check does not reexecute a model or simulator and does not recompute numerical rollout semantics.",
                "The original isolated worker validator requires original/newly reproduced local diagnostics; use this explicit publication verifier for redacted export bindings."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_publication(args.root)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        manifest = _json((args.root.resolve() / "PUBLICATION_MANIFEST.json").read_bytes())
        protected = {args.root.resolve() / name for name in manifest["files"]} | {args.root.resolve() / "PUBLICATION_MANIFEST.json"}
        require(output not in protected, "Verification output must not overwrite published evidence")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
