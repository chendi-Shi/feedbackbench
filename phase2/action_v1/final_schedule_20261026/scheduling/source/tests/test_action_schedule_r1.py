"""Revision scheduling boundaries using small synthetic evidence and fake time.

No test loads a policy/simulator, starts a real subprocess, registers a campaign,
or edits the historical study. Existing scientific sources remain untouched.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import action_campaign_common as common

SPEC = importlib.util.spec_from_file_location("action_schedule_r1_under_test", SCRIPTS / "run_action_schedule_r1.py")
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)

GIB = 1024 ** 3
PARENT_SHA = "a" * 64
TRAINED_SHA = "b" * 64
CASE = {"case_id": "libero_spatial_t0_i5_clean_lora_s17", "method_id": "lora_s17",
        "suite": "libero_spatial", "task_id": 0, "task_uid": "libero_spatial:0",
        "language": "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "initial_state_id": 5, "condition": "clean"}


@pytest.fixture(autouse=True)
def sandbox_root(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule, "ROOT", tmp_path)
    monkeypatch.setattr(schedule, "AREA", tmp_path / "outputs/action_adaptation")
    monkeypatch.setattr(schedule, "resolve", lambda name, base=None: common.resolve(name, base or tmp_path))


@pytest.fixture
def case_dir(tmp_path):
    return tmp_path / "evaluation" / CASE["case_id"]


def gate(memory=lambda tick: 6 * GIB, disk=lambda tick: 4 * GIB,
         deadline=lambda tick: False, paused=lambda tick: False):
    clock, waits, updates = [0.0], [], []
    def wait(seconds):
        assert seconds == 5
        waits.append(seconds)
        clock[0] += seconds
        assert clock[0] <= 200, "Resource gate failed to terminate under deterministic samples"
    result = schedule.stable_resource_gate(
        sample=lambda: {"available_bytes": memory(clock[0]), "disk_free_bytes": disk(clock[0])},
        clock=lambda: clock[0], wait=wait, check_deadline=lambda: deadline(clock[0]),
        publish=lambda **fields: updates.append(fields), paused=lambda: paused(clock[0]))
    return result, clock[0], waits, updates


def test_exact_threshold_requires_full_continuous_minute_even_from_clock_zero():
    result, elapsed, waits, _ = gate()
    assert result is True and elapsed == 60 and len(waits) == 12


def test_memory_drop_by_one_byte_restarts_entire_minute():
    result, elapsed, _, _ = gate(memory=lambda tick: 6 * GIB - 1 if tick == 30 else 6 * GIB)
    assert result is True and elapsed == 95


def test_disk_drop_also_restarts_continuous_window():
    result, elapsed, _, _ = gate(disk=lambda tick: 4 * GIB - 1 if tick == 55 else 4 * GIB)
    assert result is True and elapsed == 120


def test_deadline_and_pause_prevent_launch_even_with_enough_memory():
    assert gate(deadline=lambda tick: tick >= 20)[:2] == (False, 20)
    assert gate(paused=lambda tick: tick >= 20)[:2] == (False, 20)
    assert gate(deadline=lambda tick: True)[:2] == (False, 0)


def test_resource_status_io_failure_is_not_silently_treated_as_ready():
    def fail(**fields):
        raise OSError("cannot persist resource readiness")
    with pytest.raises(OSError):
        schedule.stable_resource_gate(sample=lambda: {"available_bytes": 6 * GIB, "disk_free_bytes": 4 * GIB},
            clock=lambda: 0, wait=lambda seconds: None, check_deadline=lambda: False, publish=fail)


def resource_attempt(case_dir, number, *, reason="runtime_memory_reserve", interruption=None, exit_code=75):
    output = case_dir / f"attempt_{number:03d}"
    launch = output.with_name(output.name + ".launch")
    output.mkdir(parents=True)
    launch.mkdir()
    common.write_json(output / "status.json", {"status": "resource_stopped", "error": f"ResourceStop('{reason}')",
        "traceback": "Traceback (most recent call last):\n  synthetic resource check\nResourceStop: " + reason + "\n"})
    common.write_json(launch / "outcome.json", {"state": "worker_exited", "worker_exit_code": exit_code,
        "interruption": interruption, "owned_children": [], "protocol_sha256": PARENT_SHA,
        "current": output.relative_to(case_dir.parent.parent).as_posix()})
    (launch / "stdout.log").write_text("")
    (launch / "stderr.log").write_text("")
    return output, launch


def inspect(case_dir):
    return schedule.inspect_case(case_dir, CASE, PARENT_SHA, TRAINED_SHA)


def test_three_resource_attempts_are_missing_and_a_fourth_is_not_permitted(case_dir):
    for number in (1, 2, 3):
        resource_attempt(case_dir, number)
    result = inspect(case_dir)
    assert result["classification"] == "resource_missing" and result["attempt_count"] == 3
    resource_attempt(case_dir, 4)
    with pytest.raises((ValueError, RuntimeError)):
        inspect(case_dir)


def test_resource_attempt_below_limit_stays_pending(case_dir):
    for number in (1, 2):
        resource_attempt(case_dir, number)
    result = inspect(case_dir)
    assert result["classification"] == "pending" and result["attempt_count"] == 2


@pytest.mark.parametrize("reason", ["start_memory_threshold", "runtime_memory_reserve", "disk_reserve"])
def test_known_resource_stop_with_its_traceback_is_recognized(case_dir, reason):
    resource_attempt(case_dir, 1, reason=reason)
    assert inspect(case_dir)["classification"] == "pending"


@pytest.mark.parametrize("reason", ["deadline", "time_limit", "some_other_error"])
def test_exit_75_with_non_resource_reason_requires_diagnosis(case_dir, reason):
    resource_attempt(case_dir, 1, reason=reason)
    with pytest.raises((ValueError, RuntimeError)):
        inspect(case_dir)


@pytest.mark.parametrize("fault", ["worker_io", "launch_error", "persistence_error", "stderr_io", "missing_outcome"])
def test_resource_exit_code_cannot_hide_io_or_missing_evidence(case_dir, fault):
    output, launch = resource_attempt(case_dir, 1)
    if fault == "worker_io":
        common.write_json(output / "status.json", {"status": "failed", "error": "OSError('disk write failed')"})
    elif fault == "launch_error":
        common.write_json(launch / "outcome.json", {"state": "needs_attention", "worker_exit_code": 75,
                                                   "interruption": "memory_reserve", "error": "write failed"})
    elif fault == "persistence_error":
        (output / "persistence_failure.json").write_text('{"error": "write failed"}')
    elif fault == "stderr_io":
        (launch / "stderr.log").write_text("Traceback (most recent call last):\nOSError: disk write failed\n")
    else:
        (launch / "outcome.json").unlink()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        inspect(case_dir)


@pytest.mark.parametrize("artifact", ["terminal_observed.json", "report.json"])
def test_terminal_or_report_without_seal_is_never_retryable(case_dir, artifact):
    output, _ = resource_attempt(case_dir, 1)
    common.write_json(output / artifact, {"status": "complete", "official_success": False})
    with pytest.raises((ValueError, RuntimeError)):
        inspect(case_dir)


def sealed_case(case_dir, success):
    output, launch = resource_attempt(case_dir, 1, exit_code=0)
    (output / "status.json").unlink()
    result = {"status": "complete", "official_success": success, "case": CASE,
              "protocol_sha256": PARENT_SHA, "trained_manifest_sha256": TRAINED_SHA,
              "steps": 280, "horizon": 280, "terminated": False, "truncated": True,
              "NFE": 280, "decision_count": 28}
    common.write_json(output / "report.json", result)
    common.seal(output, {"protocol_sha256": PARENT_SHA, "trained_manifest_sha256": TRAINED_SHA,
                         "case_id": CASE["case_id"]})
    return output


@pytest.mark.parametrize("success", [False, True])
def test_sealed_success_and_failure_are_both_completed_and_unchanged(case_dir, success):
    output = sealed_case(case_dir, success)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    result = inspect(case_dir)
    assert result["classification"] == "completed"
    assert result["result"]["official_success"] is success
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_sealed_report_tampering_is_not_reclassified_or_retried(case_dir):
    output = sealed_case(case_dir, False)
    report = common.read(output / "report.json")
    report["official_success"] = True
    common.write_json(output / "report.json", report)
    with pytest.raises(ValueError):
        inspect(case_dir)


def test_incomplete_report_without_seal_also_stops(case_dir):
    output, _ = resource_attempt(case_dir, 1)
    common.write_json(output / "report.json", {"status": "partial"})
    with pytest.raises((ValueError, RuntimeError)):
        inspect(case_dir)


@pytest.mark.parametrize("fault", ["failed", "persistence_error", "traceback_io"])
def test_supervisor_memory_signal_cannot_mask_worker_io_failure(case_dir, fault):
    output, _ = resource_attempt(case_dir, 1, interruption="memory_reserve")
    status = common.read(output / "status.json")
    if fault == "failed":
        status.update(status="failed", error="OSError('write failed')")
    elif fault == "persistence_error":
        status["persistence_error"] = "PermissionError('denied')"
    else:
        status["traceback"] = "Traceback (most recent call last):\nOSError: disk full\n"
    common.write_json(output / "status.json", status)
    with pytest.raises((ValueError, RuntimeError)):
        inspect(case_dir)


@pytest.mark.parametrize("interruption", ["memory_reserve", "disk_reserve"])
def test_clean_supervisor_resource_interrupt_is_retryable(case_dir, interruption):
    resource_attempt(case_dir, 1, interruption=interruption, exit_code=1)
    assert inspect(case_dir)["classification"] == "pending"


def test_completed_episode_may_never_be_followed_by_another_attempt(case_dir):
    sealed_case(case_dir, False)
    resource_attempt(case_dir, 2)
    with pytest.raises(ValueError):
        inspect(case_dir)


def test_attempt_number_gap_requires_audit(case_dir):
    resource_attempt(case_dir, 2)
    with pytest.raises(ValueError):
        inspect(case_dir)


@pytest.fixture
def revision_fixture(tmp_path, monkeypatch):
    """Small exact-byte original archive; base protocol checker is isolated."""
    parent = schedule.AREA / "study_v1"
    revision_dir = parent / "scheduling_r1"
    revision_dir.mkdir(parents=True)
    protocol = {"resources": {"evaluation_max_seconds": 7200, "runtime_available_gib": .75,
                              "free_disk_gib": 4},
                "runs": {"lora_s17": {}},
                "evaluation": {"cases": [dict(CASE)] + [dict(CASE, case_id=f"case_{n:03d}") for n in range(1, 280)]},
                "source_files": {}}
    for number in range(41):
        path = tmp_path / f"original_sources/frozen_{number:02d}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"# frozen scientific source {number}\n")
        protocol["source_files"][path.relative_to(tmp_path).as_posix()] = common.digest(path)
    common.write_json(parent / "protocol.json", protocol)
    parent_sha = common.digest(parent / "protocol.json")
    (parent / "protocol.sha256").write_text(parent_sha + "\n")
    common.write_json(parent / "trained_manifest.json", {"protocol_sha256": parent_sha, "runs": {"lora_s17": {}}})
    for name in ("status.json", "report.json"):
        common.write_json(parent / name, {"state": "frozen"})
    (parent / "process_ownership.jsonl").write_text("")
    originals = [parent / name for name in ("protocol.json", "protocol.sha256", "trained_manifest.json",
                                          "status.json", "report.json", "process_ownership.jsonl")]
    originals += [tmp_path / name for name in protocol["source_files"]]
    proposal_path = tmp_path / "proposal.json"
    common.write_json(proposal_path, {"schema": "feedbackbench.proposed_execution_revision.v1",
        "parent_protocol_sha256": parent_sha,
        "parent_trained_manifest_sha256": common.digest(parent / "trained_manifest.json"),
        "changes": {"denominator": 280, "maximum_sealed_total": 279},
        "unchanged": {"attempt_cap_per_case": 3, "deadline_utc": common.DEADLINE}})
    originals.append(proposal_path)
    files, archives = {}, {}
    for original in originals:
        name, sha = original.relative_to(tmp_path).as_posix(), common.digest(original)
        archive = revision_dir / "original" / sha
        archive.parent.mkdir(exist_ok=True)
        archive.write_bytes(original.read_bytes())
        files[name], archives[name] = sha, archive.relative_to(tmp_path).as_posix()
    inventory_path = revision_dir / "prior_inventory.json"
    common.write_json(inventory_path, {"schema": "feedbackbench.action_schedule_prior_inventory.v1",
                                     "files": files, "archive_files": archives})
    authorization = revision_dir / "authorization.json"
    common.write_json(authorization, {"schema": "feedbackbench.action_schedule_authorization.v1",
        "user_message": "continue approved revision", "accepted_proposal_sha256": common.digest(proposal_path)})
    sources = {}
    for name in ("scripts/run_action_schedule_r1.py", "scripts/evaluate_action_schedule_r1.py",
                 "scripts/report_action_schedule_r1.py", "tests/test_action_schedule_r1.py"):
        source = tmp_path / name
        source.parent.mkdir(exist_ok=True)
        source.write_text("# synthetic scheduling source\n")
        relative = source.relative_to(tmp_path).as_posix()
        sources[relative] = common.digest(source)
        snapshot = revision_dir / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    revision = {"schema": schedule.SCHEMA, "frozen": True, "revision_id": "study_v1_schedule_r1",
        "deadline_utc": common.DEADLINE, "parent": {"campaign": parent.relative_to(tmp_path).as_posix(),
            "protocol_sha256": parent_sha, "trained_manifest_sha256": common.digest(parent / "trained_manifest.json")},
        "source_files": sources, "resources": {"start_available_gib": 6, "stable_seconds": 60,
            "sample_interval_seconds": 5, "runtime_available_gib": .75, "free_disk_gib": 4, "evaluation_max_seconds": 7200},
        "attempt_policy": {"max_attempts_per_case": 3, "resource_exhaustion": "missing_and_continue",
                           "non_resource_failure": "needs_attention"},
        "case_ids": [case["case_id"] for case in protocol["evaluation"]["cases"]], "preserved_missing_case_ids": [],
        "prior_inventory": {"path": inventory_path.relative_to(tmp_path).as_posix(), "sha256": common.digest(inventory_path)},
        "authorization": {"path": authorization.relative_to(tmp_path).as_posix(), "sha256": common.digest(authorization)},
        "launcher_records": []}
    path = revision_dir / "protocol.json"
    common.write_json(path, revision)
    def verify_parent(target, fingerprint, **kwargs):
        assert Path(target) == parent / "protocol.json"
        if common.digest(target) != fingerprint:
            raise ValueError("Parent bytes changed")
        for name, expected in protocol["source_files"].items():
            if common.digest(tmp_path / name) != expected:
                raise ValueError("Frozen scientific source changed")
        return protocol
    monkeypatch.setattr(schedule, "verify_protocol", verify_parent)
    return SimpleNamespace(root=tmp_path, parent=parent, path=path, revision=revision,
        sha=common.digest(path), protocol=protocol, inventory=files, archives=archives,
        authorization=authorization)


def test_revision_verification_leaves_all_41_original_sources_and_parent_bytes_unchanged(revision_fixture):
    fixture = revision_fixture
    before = {name: (fixture.root / name).read_bytes() for name in fixture.inventory}
    revision, parent, protocol = schedule.verify_revision(fixture.path, fixture.sha)
    assert revision == fixture.revision and parent == fixture.parent and protocol == fixture.protocol
    assert len(protocol["source_files"]) == 41
    assert before == {name: (fixture.root / name).read_bytes() for name in fixture.inventory}


@pytest.mark.parametrize("target", ["parent_status", "parent_source", "source_snapshot", "archive", "authorization"])
def test_changed_historical_or_authorization_evidence_refuses_revision(revision_fixture, target):
    fixture = revision_fixture
    if target == "parent_status":
        path = fixture.parent / "status.json"
    elif target == "parent_source":
        path = fixture.root / next(iter(fixture.protocol["source_files"]))
    elif target == "source_snapshot":
        path = fixture.path.parent / "source_snapshot/scripts/run_action_schedule_r1.py"
    elif target == "archive":
        path = fixture.root / next(iter(fixture.archives.values()))
    else:
        path = fixture.authorization
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises((ValueError, RuntimeError)):
        schedule.verify_revision(fixture.path, fixture.sha)


@pytest.mark.parametrize("change", ["deadline", "attempts", "resources", "case_order", "parent_escape"])
def test_rehashed_revision_cannot_change_approved_scope(revision_fixture, change):
    fixture = revision_fixture
    revision = fixture.revision
    if change == "deadline":
        revision["deadline_utc"] = "2026-11-30T06:08:00Z"
    elif change == "attempts":
        revision["attempt_policy"]["max_attempts_per_case"] = 4
    elif change == "resources":
        revision["resources"]["stable_seconds"] = 5
    elif change == "case_order":
        revision["case_ids"] = list(reversed(revision["case_ids"]))
    else:
        revision["parent"]["campaign"] = "../unrelated_project"
    common.write_json(fixture.path, revision)
    with pytest.raises((ValueError, RuntimeError)):
        schedule.verify_revision(fixture.path, common.digest(fixture.path))


@pytest.fixture
def launch_fixture(revision_fixture, monkeypatch):
    fixture = revision_fixture
    owner = {"pid": 12345, "created": 100.25}
    output = fixture.parent / "evaluation" / CASE["case_id"] / "attempt_001"
    launch = output.with_name(output.name + ".launch")
    launch.mkdir(parents=True)
    for filename in ("stdout.log", "stderr.log"):
        (launch / filename).write_text("")
    auth_path = schedule.authorization_path(fixture.path, CASE["case_id"], output.name)
    auth_path.parent.mkdir()
    auth = {"schema": schedule.AUTH_SCHEMA, "revision_sha256": fixture.sha,
        "revision_path": schedule.relative(fixture.path), "parent_protocol_sha256": fixture.revision["parent"]["protocol_sha256"],
        "trained_manifest_sha256": fixture.revision["parent"]["trained_manifest_sha256"],
        "case_id": CASE["case_id"], "attempt_number": 1, "output": schedule.relative(output),
        "launch": schedule.relative(launch), "owner": owner, "prior_attempt_files": {}}
    common.write_json(auth_path, auth)
    common.write_json(schedule.AREA / "worker.lock", {"owner": owner,
        "revision_sha256": fixture.sha, "revision": schedule.relative(fixture.path.parent),
        "campaign": schedule.relative(fixture.parent)})
    common.write_json(fixture.path.parent / "status.json", {"owner": owner,
        "revision_sha256": fixture.sha, "current": output.relative_to(fixture.parent).as_posix(), "state": "running",
        "launch_authorization_path": schedule.relative(auth_path), "launch_authorization_sha256": common.digest(auth_path)})
    ancestor = SimpleNamespace(pid=owner["pid"], create_time=lambda: owner["created"])
    monkeypatch.setattr(schedule, "matching", lambda record: ancestor if record == owner else None)
    monkeypatch.setattr(schedule.psutil, "Process", lambda: SimpleNamespace(parents=lambda: [ancestor]))
    monkeypatch.setattr(schedule, "deadline_reached", lambda: False)
    fixture.output, fixture.auth_path, fixture.auth, fixture.owner = output, auth_path, auth, owner
    return fixture


def authorize(fixture, **kwargs):
    return schedule.verify_launch_authorization(fixture.path, fixture.sha, CASE["case_id"],
        fixture.output, fixture.auth_path, **kwargs)


def test_authorized_descendant_has_both_parent_and_revision_binding(launch_fixture):
    fixture = launch_fixture
    result = authorize(fixture)
    assert result["binding"]["revision_sha256"] == fixture.sha
    assert result["binding"]["parent_protocol_sha256"] == fixture.revision["parent"]["protocol_sha256"]
    assert result["binding"]["trained_manifest_sha256"] == fixture.revision["parent"]["trained_manifest_sha256"]
    assert result["binding"]["authorization_sha256"] == common.digest(fixture.auth_path)


@pytest.mark.parametrize("field", ["revision_sha256", "parent_protocol_sha256", "trained_manifest_sha256", "case_id", "output", "launch"])
def test_cross_revision_protocol_case_or_path_authorization_is_rejected(launch_fixture, field):
    fixture = launch_fixture
    auth = common.read(fixture.auth_path)
    auth[field] = "unrelated"
    common.write_json(fixture.auth_path, auth)
    with pytest.raises(ValueError):
        authorize(fixture)


@pytest.mark.parametrize("fault", ["pid_reused", "not_ancestor", "wrong_lock", "wrong_current", "deadline", "already_exists", "outside_parent"])
def test_live_authorization_requires_exact_process_time_lock_and_path(launch_fixture, monkeypatch, fault):
    fixture = launch_fixture
    if fault == "pid_reused":
        stale = SimpleNamespace(pid=fixture.owner["pid"], create_time=lambda: fixture.owner["created"] + .01)
        monkeypatch.setattr(schedule.psutil, "Process", lambda: SimpleNamespace(parents=lambda: [stale]))
    elif fault == "not_ancestor":
        monkeypatch.setattr(schedule.psutil, "Process", lambda: SimpleNamespace(parents=lambda: []))
    elif fault == "wrong_lock":
        lock = common.read(schedule.AREA / "worker.lock")
        lock["owner"] = dict(fixture.owner, created=999)
        common.write_json(schedule.AREA / "worker.lock", lock)
    elif fault == "wrong_current":
        status = common.read(fixture.path.parent / "status.json")
        status["current"] = "evaluation/another_case/attempt_001"
        common.write_json(fixture.path.parent / "status.json", status)
    elif fault == "deadline":
        monkeypatch.setattr(schedule, "deadline_reached", lambda: True)
    elif fault == "already_exists":
        fixture.output.mkdir()
    else:
        fixture.output = fixture.root / "other_project" / "attempt_001"
    with pytest.raises(ValueError):
        authorize(fixture)


def test_execute_skips_three_resources_and_both_sealed_outcomes_without_launching(revision_fixture, monkeypatch):
    fixture = revision_fixture
    cases = fixture.protocol["evaluation"]["cases"][:3]
    fixture.protocol["evaluation"]["cases"] = cases
    categories = ["resource_missing", "completed", "completed"]
    monkeypatch.setattr(schedule, "completed_training", lambda *args: {"lora_s17": {}})
    monkeypatch.setattr(schedule, "inspect_case", lambda path, case, *args: {
        "classification": categories[cases.index(case)], "attempt_count": 3 if case == cases[0] else 1,
        "result": {"official_success": case == cases[2]}})
    monkeypatch.setattr(schedule.ScheduleSupervisor, "report", lambda self: None)
    monkeypatch.setattr(schedule.ScheduleSupervisor, "record", lambda *args: None)
    supervisor = schedule.ScheduleSupervisor(fixture.path, fixture.sha, fixture.revision, fixture.parent, fixture.protocol)
    monkeypatch.setattr(supervisor, "ready", lambda: pytest.fail("Completed/missing cases must not request a new worker"))
    monkeypatch.setattr(supervisor, "child", lambda *args: pytest.fail("No fourth attempt or completed-case rerun"))
    before = {name: (fixture.root / name).read_bytes() for name in fixture.inventory}
    supervisor.execute()
    assert supervisor.state["evaluation_completed"] == 2
    assert supervisor.state["state"] == "partial_resource_missing"
    assert supervisor.state["resource_missing_cases"] == [cases[0]["case_id"]]
    assert before == {name: (fixture.root / name).read_bytes() for name in fixture.inventory}


def test_deadline_after_resource_wait_prevents_authorization_and_child(revision_fixture, monkeypatch):
    fixture = revision_fixture
    fixture.protocol["evaluation"]["cases"] = fixture.protocol["evaluation"]["cases"][:1]
    monkeypatch.setattr(schedule, "completed_training", lambda *args: {"lora_s17": {}})
    monkeypatch.setattr(schedule, "inspect_case", lambda *args: {"classification": "pending", "attempt_count": 0})
    monkeypatch.setattr(schedule, "verify_revision", lambda *args: None)
    monkeypatch.setattr(schedule, "deadline_reached", lambda: True)
    monkeypatch.setattr(schedule.ScheduleSupervisor, "report", lambda self: None)
    monkeypatch.setattr(schedule.ScheduleSupervisor, "record", lambda *args: None)
    supervisor = schedule.ScheduleSupervisor(fixture.path, fixture.sha, fixture.revision, fixture.parent, fixture.protocol)
    monkeypatch.setattr(supervisor, "ready", lambda: True)
    monkeypatch.setattr(supervisor, "child", lambda *args: pytest.fail("Deadline must prohibit worker launch"))
    with pytest.raises(ValueError):
        supervisor.execute()
    assert not (fixture.path.parent / "authorizations").exists()


def test_report_retains_unknown_in_280_denominator_and_withholds_pairs(revision_fixture, monkeypatch):
    import report_action_schedule_r1 as reporting
    from report_action_campaign import paired_summary
    fixture = revision_fixture
    monkeypatch.setattr(reporting, "verify_revision", lambda *args: (fixture.revision, fixture.parent, fixture.protocol))
    def inspect_report(path, case, *args):
        missing = case == fixture.protocol["evaluation"]["cases"][0]
        return {"classification": "resource_missing" if missing else "completed",
                "attempt_count": 3 if missing else 1, "attempts": [], "missing_reason": "three_resource_attempts_exhausted"}
    monkeypatch.setattr(reporting, "inspect_case", inspect_report)
    # Exercise the real scientific pair-withholding function: a missing LoRA
    # result must not turn into False and therefore produce a numerical effect.
    contrast = paired_summary({}, "lora", "frozen", "clean")
    monkeypatch.setattr(reporting, "build_parent_report", lambda *args: {
        "status": "partial", "training_completed": 6, "evaluation_completed": 279,
        "evaluation_planned": 280, "rates": [{"completed": 279, "planned": 280, "failures": 100}],
        "contrasts": {"lora_minus_frozen_clean": contrast}, "limitations": []})
    result = reporting.build_report(fixture.path, fixture.sha)
    assert result["status"] == "partial" and result["original_study_complete"] is False
    assert result["scheduling_finished"] is True and result["evaluation_pending"] == 0
    assert result["evaluation_completed"] == 279 and result["evaluation_planned"] == 280
    assert result["resource_missing_cases"][0]["official_success"] is None
    assert result["rates"][0]["failures"] == 100
    assert result["contrasts"]["lora_minus_frozen_clean"] == {"status": "pending_complete_pairs", "estimate": None}


def test_report_and_scheduler_import_without_eager_cycle():
    import report_action_schedule_r1 as reporting
    import run_action_schedule_r1 as production_schedule
    assert callable(reporting.build_report) and callable(production_schedule.ScheduleSupervisor.report)


def test_missing_prior_file_inventory_cannot_authorize_a_worker(launch_fixture):
    fixture = launch_fixture
    auth = common.read(fixture.auth_path)
    del auth["prior_attempt_files"]
    common.write_json(fixture.auth_path, auth)
    with pytest.raises(ValueError):
        authorize(fixture)


def test_authorization_byte_change_even_with_same_owner_and_paths_is_rejected(launch_fixture):
    fixture = launch_fixture
    before = fixture.auth_path.read_bytes()
    # Whitespace changes JSON bytes without changing any parsed authorization.
    fixture.auth_path.write_bytes(before.replace(b"  ", b" \t", 1))
    assert len(fixture.auth_path.read_bytes()) == len(before)
    assert common.read(fixture.auth_path) == fixture.auth
    with pytest.raises(ValueError):
        authorize(fixture)


def test_prior_attempt_added_or_modified_file_invalidates_authorization(launch_fixture):
    fixture = launch_fixture
    old = fixture.output
    old.mkdir()
    common.write_json(old / "status.json", {"status": "resource_stopped"})
    common.write_json(old.with_name(old.name + ".launch") / "outcome.json", {"state": "worker_exited"})
    new = old.with_name("attempt_002")
    before = schedule.prior_attempt_files(new)
    common.write_json(old / "later_diagnostic.json", {"error": "new evidence"})
    after = schedule.prior_attempt_files(new)
    assert set(before) < set(after)
    common.write_json(old / "status.json", {"status": "failed"})
    assert after[schedule.relative(old / "status.json")] != schedule.prior_attempt_files(new)[schedule.relative(old / "status.json")]


@pytest.mark.parametrize("fault", ["owned_child_alive", "other_campaign", "owner_alive", "unknown_revision"])
def test_resume_never_archives_live_or_other_campaign_global_lock(revision_fixture, monkeypatch, fault):
    fixture = revision_fixture
    current, old_owner, old_child = ({"pid": 9001, "created": 101}, {"pid": 9002, "created": 102},
                                      {"pid": 9003, "created": 103})
    lock = {"owner": old_owner, "campaign": schedule.relative(fixture.parent), "owned_children": [old_child]}
    if fault == "other_campaign":
        lock["campaign"] = "outputs/action_adaptation/another_study"
    elif fault == "unknown_revision":
        lock.update(revision="outputs/action_adaptation/study_v1/other_revision", revision_sha256="e" * 64)
    common.write_json(schedule.AREA / "worker.lock", lock)
    before = (schedule.AREA / "worker.lock").read_bytes()
    monkeypatch.setattr(schedule, "verify_revision", lambda *args, **kwargs: (fixture.revision, fixture.parent, fixture.protocol))
    monkeypatch.setattr(schedule, "_owned_records", lambda *args: [])
    monkeypatch.setattr(schedule, "deadline_reached", lambda: False)
    monkeypatch.setattr(schedule, "identity", lambda pid: current)
    monkeypatch.setattr(schedule.psutil, "Process", lambda: SimpleNamespace(parents=lambda: []))
    live = old_child if fault == "owned_child_alive" else old_owner if fault == "owner_alive" else None
    monkeypatch.setattr(schedule, "matching", lambda record: object() if record == live else None)
    monkeypatch.setattr(schedule, "ScheduleSupervisor", lambda *args: pytest.fail("Unsafe lock must prevent supervisor"))
    with pytest.raises(ValueError):
        schedule.main(["supervise", "--revision", str(fixture.path), "--revision-sha256", fixture.sha, "--resume"])
    assert (schedule.AREA / "worker.lock").read_bytes() == before
    assert not (fixture.path.parent / "lock_history").exists()


def test_deadline_reached_during_final_gate_sample_cannot_return_ready():
    checks = [False] * 13 + [True]
    tick = [0]
    result = schedule.stable_resource_gate(sample=lambda: {"available_bytes": 6 * GIB, "disk_free_bytes": 4 * GIB},
        clock=lambda: tick[0], wait=lambda seconds: tick.__setitem__(0, tick[0] + seconds),
        check_deadline=lambda: checks.pop(0), publish=lambda **fields: None)
    assert result is False and tick[0] == 60


def test_unsampled_suspend_time_does_not_count_as_continuous_resource_stability():
    tick, updates = [0], []
    def wait(seconds):
        tick[0] += 65 if tick[0] == 25 else seconds
        assert tick[0] <= 150
    result = schedule.stable_resource_gate(sample=lambda: {"available_bytes": 6 * GIB, "disk_free_bytes": 4 * GIB},
        clock=lambda: tick[0], wait=wait, check_deadline=lambda: False,
        publish=lambda **fields: updates.append((tick[0], fields["stable_seconds"])))
    assert result is True and tick[0] == 150
    assert (90, 0) in updates


@pytest.mark.parametrize("success", [False, True])
def test_worker_decorates_original_seal_with_dual_binding_without_model(launch_fixture, monkeypatch, success):
    import evaluate_action_schedule_r1 as worker
    fixture = launch_fixture
    monkeypatch.setattr(worker, "verify_revision", lambda *args: (fixture.revision, fixture.parent, fixture.protocol))
    monkeypatch.setattr(worker, "verify_launch_authorization", schedule.verify_launch_authorization)
    original = SimpleNamespace(seal=common.seal)
    expected = {"protocol_sha256": fixture.revision["parent"]["protocol_sha256"],
                "trained_manifest_sha256": fixture.revision["parent"]["trained_manifest_sha256"], "case_id": CASE["case_id"]}
    def frozen_main(argv):
        assert argv[argv.index("--protocol-sha256") + 1] == expected["protocol_sha256"]
        assert argv[argv.index("--trained-manifest-sha256") + 1] == expected["trained_manifest_sha256"]
        fixture.output.mkdir()
        common.write_json(fixture.output / "report.json", {"status": "complete", "case": CASE,
            "official_success": success, "protocol_sha256": expected["protocol_sha256"],
            "trained_manifest_sha256": expected["trained_manifest_sha256"]})
        original.seal(fixture.output, expected)
        return 0
    original.main = frozen_main
    monkeypatch.setitem(sys.modules, "evaluate_action_adaptation", original)
    assert worker.main(["--revision", str(fixture.path), "--revision-sha256", fixture.sha,
        "--case-id", CASE["case_id"], "--output", str(fixture.output), "--authorization", str(fixture.auth_path)]) == 0
    assert original.seal is common.seal
    receipt = common.verify_seal(fixture.output, expected)
    assert "schedule_binding.json" in receipt["files"]
    binding = common.read(fixture.output / "schedule_binding.json")
    assert binding["revision_sha256"] == fixture.sha and binding["parent_protocol_sha256"] == expected["protocol_sha256"]
    launch = fixture.output.with_name(fixture.output.name + ".launch")
    common.write_json(launch / "outcome.json", {"state": "worker_exited", "owner": fixture.owner,
        "revision_sha256": fixture.sha, "protocol_sha256": expected["protocol_sha256"],
        "current": fixture.output.relative_to(fixture.parent).as_posix(), "worker_exit_code": 0,
        "launch_authorization_path": schedule.relative(fixture.auth_path),
        "launch_authorization_sha256": common.digest(fixture.auth_path)})
    result = schedule.inspect_case(fixture.output.parent, CASE, expected["protocol_sha256"],
        expected["trained_manifest_sha256"], fixture.path, fixture.sha)
    assert result["classification"] == "completed" and result["result"]["official_success"] is success


def test_worker_restores_original_seal_if_original_raises(launch_fixture, monkeypatch):
    import evaluate_action_schedule_r1 as worker
    fixture = launch_fixture
    monkeypatch.setattr(worker, "verify_revision", lambda *args: (fixture.revision, fixture.parent, fixture.protocol))
    monkeypatch.setattr(worker, "verify_launch_authorization", schedule.verify_launch_authorization)
    def fail(argv):
        raise OSError("synthetic original worker persistence failure")
    original = SimpleNamespace(seal=common.seal, main=fail)
    monkeypatch.setitem(sys.modules, "evaluate_action_adaptation", original)
    with pytest.raises(OSError):
        worker.main(["--revision", str(fixture.path), "--revision-sha256", fixture.sha,
            "--case-id", CASE["case_id"], "--output", str(fixture.output), "--authorization", str(fixture.auth_path)])
    assert original.seal is common.seal and not fixture.output.exists()


def test_report_unsealed_diagnostic_stays_unknown_and_cannot_authorize_retry(revision_fixture, monkeypatch):
    import report_action_schedule_r1 as reporting
    fixture = revision_fixture
    monkeypatch.setattr(reporting, "verify_revision", lambda *args: (fixture.revision, fixture.parent, fixture.protocol))
    def inspect_report(path, case, *args):
        if case == fixture.protocol["evaluation"]["cases"][0]:
            raise OSError("terminal report has no durable seal")
        return {"classification": "pending", "attempt_count": 0}
    monkeypatch.setattr(reporting, "inspect_case", inspect_report)
    monkeypatch.setattr(reporting, "build_parent_report", lambda *args: {
        "status": "partial", "evaluation_completed": 0, "evaluation_planned": 280, "limitations": []})
    result = reporting.build_report(fixture.path, fixture.sha)
    assert result["evaluation_completed"] == 0 and result["evaluation_pending"] == 280
    assert result["scheduling_finished"] is False and result["original_study_complete"] is False
    assert result["case_audit_errors"][0]["official_success"] is None
    assert result["case_audit_errors"][0]["automatic_retry_permitted"] is False


def test_report_rejects_scientific_parent_count_inconsistent_with_strict_seals(revision_fixture, monkeypatch):
    import report_action_schedule_r1 as reporting
    fixture = revision_fixture
    monkeypatch.setattr(reporting, "verify_revision", lambda *args: (fixture.revision, fixture.parent, fixture.protocol))
    monkeypatch.setattr(reporting, "inspect_case", lambda *args: {"classification": "pending", "attempt_count": 0})
    monkeypatch.setattr(reporting, "build_parent_report", lambda *args: {
        "status": "partial", "evaluation_completed": 1, "evaluation_planned": 280, "limitations": []})
    with pytest.raises(ValueError):
        reporting.build_report(fixture.path, fixture.sha)
