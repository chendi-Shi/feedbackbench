"""Small synthetic scheduling audits. No model, simulator, Git or real evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".runtime"))
import verify_action_schedule_publication as verifier


@pytest.fixture
def publisher():
    return pytest.importorskip("publish_action_schedule", reason="Private exporter is intentionally absent from public snapshots")


def synthetic():
    contents, records = {}, {}
    def add(name, value, *, source=None, origin=None, projection=False):
        raw = value if isinstance(value, bytes) else verifier.json_bytes(value)
        contents[name] = raw
        records[name] = {"sha256": verifier.sha(raw), "bytes": len(raw), "source_sha256": source or verifier.sha(raw),
                         "origin": origin, "projection": projection, "category": "synthetic"}
        return name
    def project(name, value, source, origin):
        return add(name, {"schema": verifier.PROJECTION, "source_sha256": source, "omitted_fields": ["owner"],
            "full_private_evidence_verified": False, "data": value}, source=source, origin=origin, projection=True)
    parent = "outputs/action_adaptation/study_v1"
    revision_path = parent + "/scheduling_r1/protocol.json"
    revision_sha, trained_sha, inventory_sha = "1" * 64, "2" * 64, "3" * 64
    case_ids = [f"case_{n:03d}" for n in range(280)]
    original_code = add("source/scripts/original.py", b"# original synthetic source\n")
    source_sha = verifier.sha(contents[original_code])
    protocol = {"deadline_utc": verifier.DEADLINE, "resources": {"evaluation_max_seconds": 7200},
        "source_files": {"scripts/original.py": source_sha}, "runs": {f"run{n}": {} for n in range(6)},
        "evaluation": {"cases": [{"case_id": case_id} for case_id in case_ids]}}
    add("protocol.json", protocol)
    protocol_sha = verifier.sha(contents["protocol.json"])
    prior = {"scripts/original.py": source_sha, parent + "/protocol.json": protocol_sha,
             parent + "/trained_manifest.json": trained_sha}
    source_files, source_index = {}, {}
    for original in sorted(verifier.SOURCES):
        target = add("scheduling/source/" + original, ("# " + original + "\n").encode())
        source_index[original], source_files[original] = target, verifier.sha(contents[target])
    proposal = {"schema": "feedbackbench.proposed_execution_revision.v1", "parent_protocol_sha256": protocol_sha,
        "parent_trained_manifest_sha256": trained_sha, "changes": {"denominator": 280, "maximum_sealed_total": 279},
        "unchanged": {"attempt_cap_per_case": 3, "deadline_utc": verifier.DEADLINE}}
    add("scheduling/proposal.json", proposal)
    proposal_sha = verifier.sha(contents["scheduling/proposal.json"])
    prior["private/proposal.json"] = proposal_sha
    add("scheduling/authorization.json", {"schema": "feedbackbench.action_schedule_authorization.v1",
        "accepted_proposal_sha256": proposal_sha, "user_message": "continue"})
    authorization_sha = verifier.sha(contents["scheduling/authorization.json"])
    revision = {"schema": "feedbackbench.action_schedule_revision.v1", "revision_id": "study_v1_schedule_r1", "frozen": True,
        "deadline_utc": verifier.DEADLINE, "parent": {"campaign": parent, "protocol_sha256": protocol_sha,
        "trained_manifest_sha256": trained_sha}, "case_ids": case_ids, "evaluation_planned": 280,
        "maximum_sealed_total": 279, "source_files": source_files, "supporting_files": {},
        "preserved_missing_case_ids": ["case_001"],
        "attempt_policy": {"max_attempts_per_case": 3, "resource_exhaustion": "missing_and_continue", "non_resource_failure": "needs_attention"},
        "resources": {"start_available_gib": 6, "stable_seconds": 60, "sample_interval_seconds": 5,
            "runtime_available_gib": .75, "free_disk_gib": 4, "evaluation_max_seconds": 7200},
        "prior_inventory": {"path": parent + "/scheduling_r1/prior_inventory.json", "sha256": inventory_sha},
        "authorization": {"path": parent + "/scheduling_r1/authorization.json", "sha256": authorization_sha}}
    project("scheduling/protocol.json", revision, revision_sha, revision_path)
    rows, evidence, receipts, audit = [], {}, [], []
    for index, case_id in enumerate(case_ids):
        attempts, predecessors = [], {}
        count = 2 if index == 0 else 3 if index == 1 else 1
        for ordinal in range(1, count + 1):
            historical = index < 2
            complete = index != 1 and ordinal == count
            output = f"{parent}/evaluation/{case_id}/attempt_{ordinal:03d}"
            launch = output + ".launch"
            prefix = f"scheduling/attempts/c{index:03d}/a{ordinal}"
            private = lambda label: verifier.sha(f"{output}:{label}".encode())
            status = {"status": "complete" if complete else "running", "error": None}
            outcome = {"state": "worker_exited", "worker_exit_code": 0 if complete else 1,
                "interruption": None if complete else "memory_reserve", "protocol_sha256": protocol_sha,
                "current": f"evaluation/{case_id}/attempt_{ordinal:03d}"}
            files = {output + "/status.json": private("status"), launch + "/outcome.json": private("outcome"),
                     launch + "/stdout.log": private("stdout"), launch + "/stderr.log": private("stderr")}
            auth_target = binding_target = receipt_target = None
            if not historical:
                auth_origin = f"{parent}/scheduling_r1/authorizations/{case_id}_attempt_{ordinal:03d}.json"
                auth_sha = private("authorization")
                auth = {"schema": "feedbackbench.action_schedule_authorization.v1", "revision_path": revision_path,
                    "revision_sha256": revision_sha, "parent_protocol_sha256": protocol_sha,
                    "trained_manifest_sha256": trained_sha, "case_id": case_id, "attempt_number": ordinal,
                    "output": output, "launch": launch, "prior_attempt_files": dict(predecessors)}
                auth_target = project(prefix + "/authorization.json", auth, auth_sha, auth_origin)
                outcome.update(revision_sha256=revision_sha, launch_authorization_path=auth_origin,
                               launch_authorization_sha256=auth_sha)
            if complete:
                public = f"evaluation/c{index:03d}"
                report_target = add(public + "/report.json", {"official_success": index % 2 == 0})
                files[output + "/report.json"] = verifier.sha(contents[report_target])
                members = {"report.json": {"sha256": files[output + "/report.json"]},
                           "status.json": {"sha256": files[output + "/status.json"]}}
                if not historical:
                    binding = {"schema": "feedbackbench.action_schedule_binding.v1", "revision_path": revision_path,
                        "revision_sha256": revision_sha, "parent_protocol_sha256": protocol_sha,
                        "trained_manifest_sha256": trained_sha, "case_id": case_id, "attempt_number": ordinal,
                        "authorization_path": auth_origin, "authorization_sha256": auth_sha}
                    binding_target = add(public + "/schedule_binding.json", binding)
                    files[output + "/schedule_binding.json"] = verifier.sha(contents[binding_target])
                    members["schedule_binding.json"] = {"sha256": files[output + "/schedule_binding.json"]}
                receipt_target = add(public + "/source_receipt.json", {"files": members})
                files[output + "/receipt.json"] = verifier.sha(contents[receipt_target])
                evidence[case_id] = {"report": report_target, "source_receipt": receipt_target}
                receipts.append({"path": output + "/receipt.json", "sha256": files[output + "/receipt.json"]})
            outcome_target = project(prefix + "/outcome.json", outcome, files[launch + "/outcome.json"], launch + "/outcome.json")
            status_target = project(prefix + "/status.json", status, files[output + "/status.json"], output + "/status.json")
            if historical:
                prior.update(files)
            attempts.append({"attempt_number": ordinal, "classification": "completed" if complete else "resource_interrupted",
                "historical": historical, "output": output, "launch": launch, "file_hashes": files,
                "outcome": outcome_target, "status": status_target, "authorization": auth_target,
                "schedule_binding": binding_target, "receipt": receipt_target,
                "resource_reason": None if complete else "memory_reserve"})
            predecessors.update(files)
        category = "resource_missing" if index == 1 else "completed"
        rows.append({"case_id": case_id, "classification": category, "attempt_count": count, "attempts": attempts})
        audit.append({"case_id": case_id, "classification": category, "attempt_count": count})
    project("scheduling/prior_inventory.json", {"files": prior,
        "archive_files": {name: parent + "/scheduling_r1/original/" + expected for name, expected in prior.items()}},
        inventory_sha, revision["prior_inventory"]["path"])
    report = {"protocol_sha256": protocol_sha, "trained_manifest_sha256": trained_sha, "status": "partial",
        "training_completed": 6, "training_planned": 6, "evaluation_completed": 279, "evaluation_planned": 280,
        "rates": [], "contrasts": {"clean": {"status": "pending_complete_pairs", "estimate": None}},
        "missing_cases": ["case_001"], "sealed_receipts": receipts, "claims": {}}
    add("report.json", report)
    schedule_report = dict(report, evaluation_pending=0, evaluation_resource_missing=1, pending_cases=[], case_audit_errors=[],
        scheduling_finished=True, original_study_complete=False, scheduling_revision_sha256=revision_sha,
        scheduling_revision_id=revision["revision_id"], case_audit=audit,
        resource_missing_cases=[{"case_id": "case_001", "official_success": None, "attempt_count": 3,
                                 "reason": "three_resource_attempts_exhausted"}])
    project("scheduling/report.json", schedule_report, "4" * 64, parent + "/scheduling_r1/report.json")
    add("evidence_index.json", {"training": {}, "evaluation": evidence})
    add("scheduling/index.json", {"schema": verifier.SCHEMA, "revision_sha256": revision_sha,
        "previous_publication_commit": verifier.PREVIOUS_COMMIT, "protocol": "scheduling/protocol.json",
        "prior_inventory": "scheduling/prior_inventory.json", "authorization": "scheduling/authorization.json",
        "proposal": "scheduling/proposal.json", "sources": source_index, "report": "scheduling/report.json",
        "cases": rows, "full_private_seals_verifiable": False})
    manifest = {"schema": verifier.base.SCHEMA, "baseline": {"commit": verifier.PREVIOUS_COMMIT},
        "protocol_sha256": protocol_sha, "files": records, "scheduling": {"schema": verifier.SCHEMA,
        "index": "scheduling/index.json", "index_sha256": verifier.sha(contents["scheduling/index.json"]), "revision_sha256": revision_sha}}
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    return contents


def change(contents, name, operation):
    value = json.loads(contents[name])
    operation(value)
    contents[name] = verifier.json_bytes(value)
    manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
    manifest["files"][name].update(sha256=verifier.sha(contents[name]), bytes=len(contents[name]))
    if name == "scheduling/index.json":
        manifest["scheduling"]["index_sha256"] = verifier.sha(contents[name])
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)


def audit(contents):
    return verifier.validate_schedule(contents, json.loads(contents["PUBLICATION_MANIFEST.json"]))


def test_finished_queue_is_279_sealed_plus_unknown_not_complete():
    contents = synthetic()
    before = copy.deepcopy(contents)
    index = audit(contents)
    assert len(index["cases"]) == 280
    assert contents == before
    missing = json.loads(contents["scheduling/report.json"])["data"]["resource_missing_cases"][0]
    assert missing["official_success"] is None and missing["attempt_count"] == 3


@pytest.mark.parametrize("mutation", ["fourth", "drop_prior", "historical_forgery", "missing_binding", "ordinal", "wrong_case"])
def test_attempt_inventory_and_historical_boundaries(mutation):
    contents = synthetic()
    def mutate(index):
        if mutation == "fourth":
            index["cases"][1]["attempts"].append(copy.deepcopy(index["cases"][1]["attempts"][-1]))
            index["cases"][1]["attempt_count"] = 4
        elif mutation == "drop_prior":
            index["cases"][0]["attempts"].pop(0)
            index["cases"][0]["attempt_count"] = 1
        elif mutation == "historical_forgery":
            index["cases"][2]["attempts"][0]["historical"] = True
        elif mutation == "missing_binding":
            index["cases"][2]["attempts"][0]["schedule_binding"] = "missing.json"
        elif mutation == "ordinal":
            index["cases"][2]["attempts"][0]["attempt_number"] = 2
        else:
            index["cases"][2]["case_id"] = "other_case"
    change(contents, "scheduling/index.json", mutate)
    with pytest.raises((ValueError, KeyError)):
        audit(contents)


@pytest.mark.parametrize("mutation", ["unknown_false", "complete", "denominator", "pair_estimate", "pending", "audit_error"])
def test_report_cannot_relabel_unknown_or_claim_completion(mutation):
    contents = synthetic()
    def mutate(value):
        report = value["data"]
        if mutation == "unknown_false":
            report["resource_missing_cases"][0]["official_success"] = False
        elif mutation == "complete":
            report["original_study_complete"] = True
        elif mutation == "denominator":
            report["evaluation_planned"] = 279
        elif mutation == "pair_estimate":
            report["contrasts"]["clean"]["estimate"] = 1
        elif mutation == "pending":
            report["evaluation_pending"] = 1
        else:
            report["case_audit_errors"] = [{"error": "unsealed"}]
    change(contents, "scheduling/report.json", mutate)
    with pytest.raises(ValueError):
        audit(contents)


@pytest.mark.parametrize("mutation", ["revision", "parent", "prior_files", "owner", "absolute_path"])
def test_authorization_dual_binding_and_privacy(mutation):
    contents = synthetic()
    def mutate(value):
        auth = value["data"]
        if mutation == "revision":
            auth["revision_sha256"] = "f" * 64
        elif mutation == "parent":
            auth["parent_protocol_sha256"] = "f" * 64
        elif mutation == "prior_files":
            auth["prior_attempt_files"] = {"other/file": "f" * 64}
        elif mutation == "owner":
            auth["owner"] = {"pid": 1, "created": 2}
        else:
            auth["output"] = "C:/Users/private/project"
    change(contents, "scheduling/attempts/c002/a1/authorization.json", mutate)
    with pytest.raises(ValueError):
        audit(contents)


@pytest.mark.parametrize("mutation", ["io_error", "reason", "terminal", "source_sha", "full_seal"])
def test_resource_failure_and_projection_claims(mutation):
    contents = synthetic()
    name = "scheduling/attempts/c001/a1/status.json"
    if mutation == "terminal":
        change(contents, "scheduling/index.json", lambda value: value["cases"][1]["attempts"][0]["file_hashes"].update(
            {"outputs/action_adaptation/study_v1/evaluation/case_001/attempt_001/terminal_observed.json": "f" * 64}))
    elif mutation == "reason":
        change(contents, "scheduling/attempts/c001/a1/outcome.json", lambda value: value["data"].update(interruption="time_limit"))
    else:
        def mutate(value):
            if mutation == "io_error":
                value["data"].update(status="failed", error="OSError('disk full')")
            elif mutation == "source_sha":
                value["source_sha256"] = "f" * 64
            else:
                value["full_private_evidence_verified"] = True
        change(contents, name, mutate)
    with pytest.raises(ValueError):
        audit(contents)


@pytest.mark.parametrize("mutation", ["source", "deadline", "resource_threshold", "old_inventory", "missing_schedule"])
def test_frozen_sources_scope_and_required_schedule_layer(mutation):
    contents = synthetic()
    if mutation == "source":
        name = "scheduling/source/scripts/run_action_schedule_r1.py"
        contents[name] = b"# substituted source\n"
    elif mutation == "old_inventory":
        change(contents, "scheduling/prior_inventory.json", lambda value: value["data"]["files"].update({"scripts/original.py": "f" * 64}))
    elif mutation == "missing_schedule":
        manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
        del manifest["scheduling"]
        contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    else:
        def mutate(value):
            if mutation == "deadline":
                value["data"]["deadline_utc"] = "2027-01-01T00:00:00Z"
            else:
                value["data"]["resources"]["start_available_gib"] = 1
        change(contents, "scheduling/protocol.json", mutate)
    with pytest.raises(ValueError):
        audit(contents)


def test_new_validator_calls_original_scientific_validator_first(monkeypatch):
    contents, called = synthetic(), []
    def base_check(value):
        called.append("base")
        return json.loads(value["PUBLICATION_MANIFEST.json"])
    monkeypatch.setattr(verifier.base, "validate_payload", base_check)
    assert verifier.validate_payload(contents)["scheduling"]["schema"] == verifier.SCHEMA
    assert called == ["base"]
    def refused(value):
        raise ValueError("Original statistics do not verify")
    monkeypatch.setattr(verifier.base, "validate_payload", refused)
    with pytest.raises(ValueError, match="Original statistics"):
        verifier.validate_payload(contents)


def test_old_v1_snapshot_is_not_accepted_as_schedule_only_by_manifest_hash(monkeypatch):
    monkeypatch.setattr(verifier.base, "validate_payload", lambda value: {"files": {}})
    with pytest.raises(ValueError, match="Scheduling publication binding"):
        verifier.validate_payload({})


def test_readonly_relocated_snapshot_inventory_and_external_hash(tmp_path, monkeypatch):
    contents = synthetic()
    monkeypatch.setattr(verifier.base, "validate_payload", lambda value: json.loads(value["PUBLICATION_MANIFEST.json"]))
    for name, raw in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    expected = verifier.sha(contents["PUBLICATION_MANIFEST.json"])
    result = verifier.verify_snapshot(tmp_path, expected)
    assert result["study_status"] == "partial" and result["scheduling_finished"] is True
    assert result["writes"] == 0 and result["original_full_private_seals_verified"] is False
    with pytest.raises(ValueError):
        verifier.verify_snapshot(tmp_path, "0" * 64)
    (tmp_path / "extra.log").write_text("not allowed")
    with pytest.raises(ValueError, match="inventory"):
        verifier.verify_snapshot(tmp_path, expected)


def test_new_publisher_requires_explicit_revision_before_capture(monkeypatch, publisher):
    monkeypatch.setattr(publisher, "capture", lambda *args: pytest.fail("Missing revision must not reach capture"))
    with pytest.raises(ValueError):
        publisher.main(["--freeze", "--snapshot-id", "example"])


def test_new_public_sources_pass_the_private_exporter_privacy_scan(publisher):
    for name in ("scripts/verify_action_schedule_publication.py", "tests/test_verify_action_schedule_publication.py"):
        publisher.base.scan_private((ROOT / name).read_bytes(), name)


@pytest.mark.parametrize("committed", [False, True])
def test_git_history_accepts_before_and_after_only_additive_commit(monkeypatch, committed, publisher):
    manifest = {"baseline": {"commit": verifier.PREVIOUS_COMMIT, "historical_commit": "old",
                 "historical_manifest_sha256": "a" * 64, "historical_files": 12},
                "namespace": "phase2/action_v1/new", "files": {"report.json": {}}}
    monkeypatch.setattr(publisher.base, "baseline", lambda **kwargs: manifest["baseline"])
    def git(*args):
        if "--name-status" in args and committed:
            return b"A\0phase2/action_v1/new/report.json\0A\0phase2/action_v1/new/PUBLICATION_MANIFEST.json\0"
        return b""
    monkeypatch.setattr(publisher, "git", git)
    publisher.verify_git_preservation(manifest)
    monkeypatch.setattr(publisher, "git", lambda *args: b"M\0old/manifest.json\0" if "--name-status" in args else b"")
    with pytest.raises(ValueError, match="Historical Git"):
        publisher.verify_git_preservation(manifest)


def test_public_tests_collect_without_private_runtime(tmp_path):
    for name in ("scripts/verify_action_publication.py", "scripts/verify_action_schedule_publication.py",
                 "tests/test_verify_action_schedule_publication.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q",
        "tests/test_verify_action_schedule_publication.py"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests collected" in result.stdout
