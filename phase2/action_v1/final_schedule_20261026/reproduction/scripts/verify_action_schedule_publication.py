"""Read-only audit of the finished scheduling revision and its public projection.

The original v1 verifier remains responsible for adapters, traces and statistics.
This layer binds the revision, every preserved attempt and the unknown outcome.
Original hashes identify provenance; redacted private bytes are not re-created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import verify_action_publication as base

SCHEMA = "feedbackbench.action_schedule_publication.v1"
PROJECTION = "feedbackbench.action_schedule_projection.v1"
PREVIOUS_COMMIT = "d494e26b55c206b7b7c89524edc93abf68f55ebd"
DEADLINE = "2026-10-30T06:08:00Z"
SOURCES = {"scripts/run_action_schedule_r1.py", "scripts/evaluate_action_schedule_r1.py",
           "scripts/report_action_schedule_r1.py", "tests/test_action_schedule_r1.py"}
FORBIDDEN = {"owner", "pid", "created", "owned_children", "identities", "identities_checked",
             "command", "traceback", "stdout", "stderr", "owner_pid", "owner_create_time"}
require, sha, json_bytes = base.require, base.sha, base.json_bytes


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid source SHA256")
    return value


def privacy(value):
    """Reject process identities and absolute private paths in structured exports."""
    if isinstance(value, dict):
        require(not (set(value) & FORBIDDEN), "Private process/log field in scheduling projection")
        for key, child in value.items():
            privacy(key)
            privacy(child)
    elif isinstance(value, list):
        for child in value:
            privacy(child)
    elif isinstance(value, str):
        require(not re.search(r"[A-Za-z]:[/\\]|\\\\|/(?:Users|home)/", value),
                "Absolute private path in scheduling evidence")


def validate_schedule(contents, manifest):
    """Validate the scheduling sub-schema; caller separately validates base v1."""
    link = manifest.get("scheduling")
    require(isinstance(link, dict) and link.get("schema") == SCHEMA, "Scheduling publication binding is required")
    records = manifest["files"]
    used = set()

    def document(name):
        base.safe_name(name)
        require(name in contents and name in records, "Missing scheduling evidence")
        raw = contents[name]
        require(sha(raw) == records[name]["sha256"] and len(raw) == records[name]["bytes"], "Scheduling bytes changed")
        value = json.loads(raw)
        privacy(value)
        used.add(name)
        return value

    def projected(name, expected=None, origin=None):
        value = document(name)
        require(set(value) == {"schema", "source_sha256", "omitted_fields", "full_private_evidence_verified", "data"}
                and value["schema"] == PROJECTION and value["full_private_evidence_verified"] is False,
                "Projection misrepresented as complete original evidence")
        source = hash_value(value["source_sha256"])
        require(records[name].get("projection") is True and records[name].get("source_sha256") == source,
                "Projection source declaration changed")
        require(expected is None or expected == source, "Projection refers to another original file")
        require(origin is None or records[name].get("origin") == origin, "Projection original path differs")
        require(isinstance(value["omitted_fields"], list), "Projection must declare omissions")
        return value["data"]

    index = document(link["index"])
    require(sha(contents[link["index"]]) == link["index_sha256"] and index.get("schema") == SCHEMA,
            "Scheduling index changed")
    revision_sha = hash_value(index["revision_sha256"])
    require(revision_sha == link["revision_sha256"] and index["previous_publication_commit"] == PREVIOUS_COMMIT
            and manifest["baseline"]["commit"] == PREVIOUS_COMMIT, "Scheduling/history binding differs")
    require(index["full_private_seals_verifiable"] is False, "False full-private-seal claim")
    protocol = json.loads(contents["protocol.json"])
    report = json.loads(contents["report.json"])
    evidence = json.loads(contents["evidence_index.json"])
    revision = projected(index["protocol"], revision_sha)
    parent = revision["parent"]
    parent_path = base.safe_name(parent["campaign"])
    revision_path = parent_path + "/scheduling_r1/protocol.json"
    require(records[index["protocol"]]["origin"] == revision_path, "Revision projection original path differs")
    require(revision.get("schema") == "feedbackbench.action_schedule_revision.v1" and revision.get("frozen") is True
            and revision["deadline_utc"] == protocol["deadline_utc"] == DEADLINE,
            "Revision deadline or identity changed")
    require(parent["protocol_sha256"] == manifest["protocol_sha256"] == report["protocol_sha256"]
            and parent["trained_manifest_sha256"] == report["trained_manifest_sha256"], "Original scientific binding changed")
    case_ids = [case["case_id"] for case in protocol["evaluation"]["cases"]]
    require(len(case_ids) == len(set(case_ids)) == 280 and revision["case_ids"] == case_ids,
            "Original 280-case order changed")
    require(revision["attempt_policy"] == {"max_attempts_per_case": 3,
        "resource_exhaustion": "missing_and_continue", "non_resource_failure": "needs_attention"}, "Attempt policy changed")
    require(revision["resources"] == {"start_available_gib": 6, "stable_seconds": 60, "sample_interval_seconds": 5,
        "runtime_available_gib": .75, "free_disk_gib": 4,
        "evaluation_max_seconds": protocol["resources"]["evaluation_max_seconds"]}, "Resource schedule changed")
    require(revision["evaluation_planned"] == 280 and revision["maximum_sealed_total"] == 279,
            "Scheduling completion scope changed")
    require(set(revision["source_files"]) == SOURCES, "Scheduling source inventory differs")
    all_sources = {**revision["source_files"], **revision.get("supporting_files", {})}
    require(set(index["sources"]) == set(all_sources), "Missing scheduling snapshot source")
    for original, expected in all_sources.items():
        base.safe_name(original)
        public = index["sources"][original]
        require(public.startswith("scheduling/source/") and sha(contents[public]) == expected
                and records[public]["source_sha256"] == expected and records[public]["projection"] is False,
                "Frozen scheduling source substituted")
        used.add(public)

    inventory = projected(index["prior_inventory"], revision["prior_inventory"]["sha256"], revision["prior_inventory"]["path"])
    original_files = inventory["files"]
    require(set(inventory["archive_files"]) == set(original_files), "Prior archive declaration differs")
    for name, expected in original_files.items():
        base.safe_name(name)
        hash_value(expected)
        base.safe_name(inventory["archive_files"][name])
        require(inventory["archive_files"][name].startswith(parent_path + "/scheduling_r1/original/"), "Prior archive escapes revision")
    for name, expected in protocol["source_files"].items():
        require(original_files.get(name) == expected, "Original frozen source missing from prior evidence")
    for name, expected in (("protocol.json", manifest["protocol_sha256"]),
                           ("trained_manifest.json", report["trained_manifest_sha256"])):
        require(original_files.get(parent_path + "/" + name) == expected, "Original parent evidence omitted")
    authorization = document(index["authorization"])
    require(sha(contents[index["authorization"]]) == revision["authorization"]["sha256"]
            and authorization.get("schema") == "feedbackbench.action_schedule_authorization.v1"
            and bool(authorization.get("user_message")), "Explicit authorization differs")
    proposal = document(index["proposal"])
    proposal_sha = sha(contents[index["proposal"]])
    require(proposal_sha == authorization["accepted_proposal_sha256"]
            and proposal_sha in original_files.values(), "Accepted proposal is absent from historical evidence")
    require(proposal["parent_protocol_sha256"] == parent["protocol_sha256"]
            and proposal["parent_trained_manifest_sha256"] == parent["trained_manifest_sha256"]
            and proposal["changes"]["denominator"] == 280 and proposal["changes"]["maximum_sealed_total"] == 279
            and proposal["unchanged"]["attempt_cap_per_case"] == 3 and proposal["unchanged"]["deadline_utc"] == DEADLINE,
            "Accepted proposal scientific scope differs")

    rows = index["cases"]
    require([row["case_id"] for row in rows] == case_ids, "Case audit inventory differs")
    completed, missing = [], []
    audit = []
    missing_details = []
    for row in rows:
        case_id = row["case_id"]
        attempts = row["attempts"]
        require(1 <= len(attempts) <= 3 and row["attempt_count"] == len(attempts), "Attempt cap/count differs")
        predecessor = {}
        for ordinal, attempt in enumerate(attempts, 1):
            output = f"{parent_path}/evaluation/{case_id}/attempt_{ordinal:03d}"
            launch = output + ".launch"
            require(attempt["attempt_number"] == ordinal and attempt["output"] == output and attempt["launch"] == launch,
                    "Attempt ordinal or path changed")
            files = attempt["file_hashes"]
            require(bool(files) and all(name.startswith(output + "/") or name.startswith(launch + "/") for name in files),
                    "Attempt file inventory escapes its scope")
            for name, expected in files.items():
                base.safe_name(name)
                hash_value(expected)
            outcome_origin, status_origin = launch + "/outcome.json", output + "/status.json"
            outcome = projected(attempt["outcome"], files[outcome_origin], outcome_origin)
            status = projected(attempt["status"], files[status_origin], status_origin)
            require(outcome["state"] == "worker_exited" and outcome["protocol_sha256"] == parent["protocol_sha256"]
                    and outcome["current"] == f"evaluation/{case_id}/attempt_{ordinal:03d}", "Outcome binding differs")
            require(not status.get("error") or status["error"] in {
                "ResourceStop('start_memory_threshold')", "ResourceStop('runtime_memory_reserve')", "ResourceStop('disk_reserve')"},
                "Non-resource worker failure relabeled")
            historical = attempt["historical"]
            require(type(historical) is bool, "Invalid historical classification")
            if historical:
                require(all(original_files.get(name) == expected for name, expected in files.items()),
                        "Attempt is falsely presented as historical")
                require(attempt["authorization"] is None and attempt["schedule_binding"] is None,
                        "Historical attempt unexpectedly uses new authorization")
            else:
                auth_origin = f"{parent_path}/scheduling_r1/authorizations/{case_id}_attempt_{ordinal:03d}.json"
                auth = projected(attempt["authorization"], outcome["launch_authorization_sha256"], auth_origin)
                require(outcome["revision_sha256"] == revision_sha and outcome["launch_authorization_path"] == auth_origin,
                        "Outcome revision/authorization changed")
                expected = {"schema": "feedbackbench.action_schedule_authorization.v1", "revision_path": revision_path,
                    "revision_sha256": revision_sha, "parent_protocol_sha256": parent["protocol_sha256"],
                    "trained_manifest_sha256": parent["trained_manifest_sha256"], "case_id": case_id,
                    "attempt_number": ordinal, "output": output, "launch": launch, "prior_attempt_files": predecessor}
                require(all(auth.get(key) == value for key, value in expected.items()), "Authorization or predecessor evidence changed")
            classification = attempt["classification"]
            if classification == "resource_interrupted":
                require(attempt["receipt"] is None and attempt["schedule_binding"] is None
                        and all(output + "/" + name not in files for name in ("receipt.json", "report.json", "terminal_observed.json")),
                        "Terminal/sealed evidence relabeled as a resource interruption")
                reason = attempt["resource_reason"]
                external = outcome.get("interruption")
                require(status["status"] in ("starting", "running", "resource_stopped"), "Failed worker classified as resource stop")
                if external in ("memory_reserve", "disk_reserve"):
                    require(reason == external, "External resource reason differs")
                else:
                    require(external is None and outcome["worker_exit_code"] == 75
                            and status["status"] == "resource_stopped"
                            and reason in ("start_memory_threshold", "runtime_memory_reserve", "disk_reserve")
                            and status.get("error") == f"ResourceStop({reason!r})", "Exit code is not evidence of resource interruption")
            else:
                require(classification == "completed" and ordinal == len(attempts), "Completed attempt was retried")
                require(case_id in evidence["evaluation"], "Completed case missing its sealed public result")
                public = evidence["evaluation"][case_id]
                require(attempt["receipt"] == public["source_receipt"]
                        and files[output + "/receipt.json"] == sha(contents[public["source_receipt"]]), "Attempt source receipt differs")
                receipt = json.loads(contents[public["source_receipt"]])
                for member, record in receipt["files"].items():
                    require(files.get(output + "/" + member) == record["sha256"], "Attempt inventory differs from original seal")
                if not historical:
                    binding = document(attempt["schedule_binding"])
                    require(sha(contents[attempt["schedule_binding"]]) == receipt["files"]["schedule_binding.json"]["sha256"],
                            "Scheduling binding omitted or changed within original seal")
                    expected = {"schema": "feedbackbench.action_schedule_binding.v1", "revision_path": revision_path,
                        "revision_sha256": revision_sha, "parent_protocol_sha256": parent["protocol_sha256"],
                        "trained_manifest_sha256": parent["trained_manifest_sha256"], "case_id": case_id,
                        "attempt_number": ordinal, "authorization_path": auth_origin,
                        "authorization_sha256": outcome["launch_authorization_sha256"]}
                    require(binding == expected, "Sealed scheduling dual binding differs")
            predecessor.update(files)
        if row["classification"] == "completed":
            require(attempts[-1]["classification"] == "completed", "Case completion lacks a final seal")
            completed.append(case_id)
        else:
            require(row["classification"] == "resource_missing" and len(attempts) == 3
                    and all(a["classification"] == "resource_interrupted" for a in attempts), "Missing case is not three resource attempts")
            missing.append(case_id)
            missing_details.append({"case_id": case_id, "official_success": None, "attempt_count": 3,
                                    "reason": "three_resource_attempts_exhausted"})
        audit.append({"case_id": case_id, "classification": row["classification"], "attempt_count": len(attempts)})
    require(len(completed) == 279 and len(missing) == 1 and set(completed) == set(evidence["evaluation"])
            and missing == revision["preserved_missing_case_ids"] == report["missing_cases"], "Finished 279+1 inventory differs")
    schedule_report = projected(index["report"])
    for key in ("protocol_sha256", "trained_manifest_sha256", "status", "training_completed", "training_planned",
                "evaluation_completed", "evaluation_planned", "rates", "contrasts", "missing_cases", "sealed_receipts", "claims"):
        require(schedule_report[key] == report[key], "Scheduling report disagrees with scientific base report")
    require(report["status"] == "partial" and report["training_completed"] == report["training_planned"] == 6
            and schedule_report["evaluation_pending"] == 0 and schedule_report["evaluation_resource_missing"] == 1
            and schedule_report["pending_cases"] == [] and schedule_report["case_audit_errors"] == []
            and schedule_report["scheduling_finished"] is True and schedule_report["original_study_complete"] is False
            and schedule_report["scheduling_revision_sha256"] == revision_sha
            and schedule_report["scheduling_revision_id"] == revision["revision_id"] and schedule_report["case_audit"] == audit,
            "False scheduling or study completion claim")
    public_missing = [{key: row[key] for key in ("case_id", "official_success", "attempt_count", "reason")}
                      for row in schedule_report["resource_missing_cases"]]
    require(public_missing == missing_details, "Unknown case changed to failure or disappeared")
    for name in contents:
        if name.startswith("scheduling/") and name.endswith(".json"):
            require(name in used, "Unreferenced scheduling JSON could hide private evidence")
    return index


def validate_payload(contents):
    manifest = base.validate_payload(contents)
    validate_schedule(contents, manifest)
    return manifest


def verify_snapshot(directory, expected_sha256):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""), "Explicit manifest SHA256 required")
    root = base.long_path(base.normal_path(directory))
    raw = base.stable_bytes(root / "PUBLICATION_MANIFEST.json", expected_sha256)
    manifest = json.loads(raw)
    records = manifest["files"]
    require(len(records) <= 5000 and len({name.casefold() for name in records}) == len(records), "Invalid public inventory")
    require(all(type(r["bytes"]) is int and 0 <= r["bytes"] <= base.MAX_BYTES for r in records.values())
            and sum(r["bytes"] for r in records.values()) <= base.MAX_TOTAL_BYTES, "Snapshot exceeds bounded audit memory")
    contents = {name: base.stable_bytes(base.child(root, name), record["sha256"]) for name, record in records.items()}
    contents["PUBLICATION_MANIFEST.json"] = raw
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(contents), "Snapshot file inventory differs")
    validate_payload(contents)
    return {"status": "verified", "manifest_sha256": expected_sha256, "study_status": "partial",
        "scheduling_revision_sha256": manifest["scheduling"]["revision_sha256"], "scheduling_finished": True,
        "evaluation_completed": 279, "evaluation_resource_missing": 1, "evaluation_pending": 0,
        "evaluation_planned": 280, "files": len(contents), "writes": 0, "model_loaded": False,
        "original_full_private_seals_verified": False, "historical_git_checkout_verified": False,
        "scope": "Base public v1 evidence plus scheduling projections, all attempts, original-source SHA declarations and dual bindings."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_snapshot(args.snapshot, args.manifest_sha256)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
