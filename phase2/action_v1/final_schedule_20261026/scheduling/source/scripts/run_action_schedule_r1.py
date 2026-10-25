"""Independent, evidence-preserving scheduling revision for the action study.

The original scientific protocol, worker, checkpoints and completed attempts
remain unchanged. Only this supervisor's queue and more conservative start
gate differ. Exhausted resource attempts remain missing in the original plan.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import uuid4

import psutil

from action_campaign_common import (DEADLINE, ROOT, deadline_reached, digest, now, read,
                                    resolve, verify_protocol, verify_seal, write_json)
from run_action_campaign import (AREA, Supervisor, completed_training, identities,
                                 refuse_unsealed_complete)
from run_action_adaptation_probe import identity, matching

GIB = 1024 ** 3
SCHEMA = "feedbackbench.action_schedule_revision.v1"
AUTH_SCHEMA = "feedbackbench.action_schedule_authorization.v1"
BINDING_SCHEMA = "feedbackbench.action_schedule_binding.v1"
RESOURCE_REASONS = {"memory_reserve", "disk_reserve"}
WORKER_RESOURCE_REASONS = {"start_memory_threshold", "runtime_memory_reserve", "disk_reserve"}
ATTEMPT = re.compile(r"attempt_([0-9]{3})(\.launch)?$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _envelope(path, expected_sha):
    path = Path(path).resolve()
    require(path.is_relative_to(AREA.resolve()) and path.name == "protocol.json",
            "Scheduling protocol must be a new protocol.json below the action area")
    require(digest(path) == expected_sha, "Scheduling revision changed")
    revision = read(path)
    require(revision.get("schema") == SCHEMA and revision.get("frozen") is True,
            "Expected a frozen scheduling revision")
    require(revision.get("deadline_utc") == DEADLINE, "Original deadline cannot be extended")
    parent = resolve(revision["parent"]["campaign"])
    require(parent.is_relative_to(AREA.resolve()) and parent != AREA.resolve()
            and parent != path.parent, "Invalid parent campaign")
    inventory_record = revision["prior_inventory"]
    inventory_path = resolve(inventory_record["path"])
    require(digest(inventory_path) == inventory_record["sha256"], "Historical inventory changed")
    inventory = read(inventory_path)
    require(isinstance(inventory.get("schema"), str) and isinstance(inventory.get("files"), dict)
            and bool(inventory["files"]), "Historical inventory is empty")
    return revision, parent, inventory["files"]


def verify_revision(path, expected_sha, *, inputs=False):
    """Verify original inputs plus this revision; never mutate either protocol."""
    revision, parent, inventory = _envelope(path, expected_sha)
    parent_binding = revision["parent"]
    protocol = verify_protocol(parent / "protocol.json", parent_binding["protocol_sha256"], inputs=inputs)
    require((parent / "protocol.sha256").read_text(encoding="utf-8").strip()
            == parent_binding["protocol_sha256"], "Parent protocol SHA sidecar differs")
    require(digest(parent / "trained_manifest.json") == parent_binding["trained_manifest_sha256"],
            "Frozen trained manifest changed")
    trained = read(parent / "trained_manifest.json")
    require(trained.get("protocol_sha256") == parent_binding["protocol_sha256"]
            and set(trained.get("runs", {})) == set(protocol["runs"]), "All original training runs must be frozen")
    resources = revision["resources"]
    require(resources == {"start_available_gib": 6, "stable_seconds": 60, "sample_interval_seconds": 5,
                          "runtime_available_gib": .75, "free_disk_gib": 4,
                          "evaluation_max_seconds": protocol["resources"]["evaluation_max_seconds"]},
            "Scheduling resources differ from the approved revision")
    require(protocol["resources"]["runtime_available_gib"] == .75
            and protocol["resources"]["free_disk_gib"] == 4
            and 0 < resources["evaluation_max_seconds"] <= 7200,
            "Original runtime guards must remain unchanged")
    require(revision["attempt_policy"] == {"max_attempts_per_case": 3,
            "resource_exhaustion": "missing_and_continue", "non_resource_failure": "needs_attention"},
            "Attempt policy differs from the approved revision")
    case_ids = [case["case_id"] for case in protocol["evaluation"]["cases"]]
    require(revision["case_ids"] == case_ids and len(case_ids) == 280,
            "Original case order and all 280 planned cases must be retained")
    preserved = revision["preserved_missing_case_ids"]
    require(isinstance(preserved, list) and len(preserved) == len(set(preserved))
            and all(case in case_ids for case in preserved), "Invalid preserved missing cases")
    sources = revision.get("source_files", {})
    require({"scripts/run_action_schedule_r1.py", "scripts/evaluate_action_schedule_r1.py",
             "scripts/report_action_schedule_r1.py", "tests/test_action_schedule_r1.py"} <= set(sources),
            "Scheduling implementation is not source-bound")
    for name, expected in sources.items():
        require(digest(resolve(name)) == expected, f"Scheduling source changed: {name}")
    snapshot_root = Path(path).resolve().parent / "source_snapshot"
    for name, expected in sources.items():
        require(digest(resolve(name, snapshot_root)) == expected, f"Scheduling source snapshot changed: {name}")
    required = {relative(parent / name) for name in
                ("protocol.json", "protocol.sha256", "trained_manifest.json", "status.json", "report.json", "process_ownership.jsonl")}
    require(required <= set(inventory), "Historical parent state is not completely inventoried")
    for name, expected in inventory.items():
        require(digest(resolve(name)) == expected, f"Historical evidence changed: {name}")
    inventory_document = read(resolve(revision["prior_inventory"]["path"]))
    archived = inventory_document.get("archive_files", {})
    require(set(archived) == set(inventory), "Historical archive inventory differs")
    archive_root = Path(path).resolve().parent / "original"
    for name, saved in archived.items():
        archived_path = resolve(saved)
        require(archived_path.is_relative_to(archive_root)
                and digest(archived_path) == inventory[name], f"Historical archived evidence changed: {name}")
    authorization = revision["authorization"]
    authorization_file = resolve(authorization["path"])
    require(authorization_file == Path(path).resolve().parent / "authorization.json"
            and digest(authorization_file) == authorization["sha256"], "User authorization record changed")
    decision = read(authorization_file)
    require(decision.get("schema") == AUTH_SCHEMA and bool(str(decision.get("user_message", "")).strip()),
            "Explicit user authorization is missing")
    proposal_paths = [name for name, sha in inventory.items() if sha == decision.get("accepted_proposal_sha256")]
    require(len(proposal_paths) == 1, "Accepted proposal must be uniquely bound in historical evidence")
    proposal = read(resolve(proposal_paths[0]))
    require(proposal.get("schema") == "feedbackbench.proposed_execution_revision.v1"
            and proposal.get("parent_protocol_sha256") == parent_binding["protocol_sha256"]
            and proposal.get("parent_trained_manifest_sha256") == parent_binding["trained_manifest_sha256"]
            and proposal["changes"]["denominator"] == 280 and proposal["changes"]["maximum_sealed_total"] == 279
            and proposal["unchanged"]["attempt_cap_per_case"] == 3
            and proposal["unchanged"]["deadline_utc"] == DEADLINE,
            "Accepted proposal targets another study or attempt/deadline policy")
    for record in revision.get("launcher_records", []):
        require(digest(resolve(record["path"])) == record["sha256"], "Historical launcher record changed")
        for record_identity in record["identities"]:
            _valid_identity(record_identity)
    return revision, parent, protocol


def _valid_identity(record):
    require(isinstance(record, dict) and set(record) == {"pid", "created"}
            and type(record["pid"]) is int and record["pid"] > 0
            and type(record["created"]) in (int, float) and math.isfinite(record["created"])
            and record["created"] > 0,
            "Malformed process identity")
    return record


def authorization_path(revision_path, case_id, attempt_name):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", case_id) is not None
            and re.fullmatch(r"attempt_[0-9]{3}", attempt_name) is not None, "Unsafe authorization name")
    return Path(revision_path).resolve().parent / "authorizations" / f"{case_id}_{attempt_name}.json"


def prior_attempt_files(output):
    """Bind every predecessor's full preserved file set before authorizing a retry."""
    output = Path(output).resolve()
    ordinal = int(output.name[-3:])
    files = {}
    for number in range(1, ordinal):
        previous = output.parent / f"attempt_{number:03d}"
        launch = previous.with_name(previous.name + ".launch")
        require(previous.is_dir() and launch.is_dir() and (launch / "outcome.json").is_file(),
                "Predecessor attempt has no durable outcome")
        for path in sorted([*previous.rglob("*"), *launch.rglob("*")]):
            if path.is_file():
                files[relative(path)] = digest(path)
    return files


def _authorization(revision_path, revision_sha, case_id, output, authorization):
    revision, parent, _ = _envelope(revision_path, revision_sha)
    output = Path(output).resolve()
    expected_output = parent / "evaluation" / case_id / output.name
    require(output == expected_output and re.fullmatch(r"attempt_00[123]", output.name) is not None,
            "Attempt output is not the exact original campaign case path")
    require(case_id in revision["case_ids"] and case_id not in revision["preserved_missing_case_ids"],
            "Unregistered or permanently missing case cannot launch")
    auth_path = Path(authorization).resolve()
    require(auth_path == authorization_path(revision_path, case_id, output.name), "Authorization path differs")
    auth = read(auth_path)
    expected = {"schema": AUTH_SCHEMA, "revision_sha256": revision_sha,
                "revision_path": relative(revision_path), "parent_protocol_sha256": revision["parent"]["protocol_sha256"],
                "trained_manifest_sha256": revision["parent"]["trained_manifest_sha256"],
                "case_id": case_id, "attempt_number": int(output.name[-3:]),
                "output": relative(output), "launch": relative(output.with_name(output.name + ".launch"))}
    require(all(auth.get(key) == value for key, value in expected.items()), "Launch authorization binding differs")
    require(auth.get("prior_attempt_files") == prior_attempt_files(output),
            "Prior attempt file inventory differs from launch authorization")
    _valid_identity(auth["owner"])
    return revision, parent, auth, digest(auth_path)


def verify_launch_authorization(revision_path, revision_sha, case_id, output, authorization, *, require_live=True):
    """Verify path/immutable authorization and, at launch, the live supervisor chain."""
    revision, parent, auth, auth_sha = _authorization(revision_path, revision_sha, case_id, output, authorization)
    output = Path(output).resolve()
    launch = output.with_name(output.name + ".launch")
    require(launch.is_dir() and (launch / "stdout.log").is_file() and (launch / "stderr.log").is_file(),
            "Only a supervised launch may invoke the revision worker")
    if require_live:
        require(not output.exists(), "Attempt output already exists")
        require(not deadline_reached(), "Original deadline has elapsed")
        lock = read(AREA / "worker.lock")
        require(lock.get("owner") == auth["owner"]
                and lock.get("revision_sha256") == revision_sha
                and lock.get("revision") == relative(Path(revision_path).parent)
                and lock.get("campaign") == relative(parent), "Global single-worker lock does not authorize this revision")
        require(matching(auth["owner"]) is not None, "Authorized supervisor is no longer alive")
        ancestors = [{"pid": process.pid, "created": process.create_time()} for process in psutil.Process().parents()]
        require(auth["owner"] in ancestors, "Worker must descend from the exact authorized supervisor")
        state = read(Path(revision_path).parent / "status.json")
        require(state.get("owner") == auth["owner"] and state.get("revision_sha256") == revision_sha
                and state.get("current") == output.relative_to(parent).as_posix()
                and state.get("launch_authorization_path") == relative(authorization)
                and state.get("launch_authorization_sha256") == auth_sha
                and state.get("state") in ("launching", "running"), "Supervisor current attempt differs")
    return {"authorization": auth, "authorization_sha256": auth_sha,
            "binding": {"schema": BINDING_SCHEMA, "revision_path": relative(revision_path),
                        "revision_sha256": revision_sha, "parent_protocol_sha256": revision["parent"]["protocol_sha256"],
                        "trained_manifest_sha256": revision["parent"]["trained_manifest_sha256"],
                        "case_id": case_id, "attempt_number": auth["attempt_number"],
                        "authorization_path": relative(authorization), "authorization_sha256": auth_sha}}


def _files_below(directory):
    return {relative(p) for p in directory.rglob("*") if p.is_file()} if directory.exists() else set()


def _resource_attempt(output, launch, outcome):
    require(not outcome.get("error") and not outcome.get("persistence_error")
            and outcome.get("state") == "worker_exited", "Worker outcome is not a clean recorded exit")
    require(not list(output.glob("persistence_failure*")) and not list(launch.glob("persistence_failure*")),
            "Persistence failure requires diagnosis, not a resource retry")
    status = read(output / "status.json")
    accepted = {f"ResourceStop({reason!r})": reason for reason in WORKER_RESOURCE_REASONS}
    require(status.get("status") in ("starting", "running", "resource_stopped")
            and not status.get("persistence_error")
            and (not status.get("error") or status["error"] in accepted),
            "Worker failure cannot be relabeled as an external resource stop")
    if status.get("traceback"):
        require(status.get("error") in accepted
                and not any(marker in status["traceback"] for marker in
                            ("OSError:", "PermissionError:", "IOError:", "persistence_error")),
                "Non-resource exception history requires attention")
    for filename in ("stderr.log", "stdout.log"):
        path = launch / filename
        if path.exists():
            log = path.read_text(encoding="utf-8", errors="replace")
            require("Traceback (most recent call last)" not in log and '"persistence_error"' not in log,
                    "Worker exception or persistence diagnostic requires attention")
    reason = outcome.get("interruption")
    if reason in RESOURCE_REASONS:
        return reason
    require(reason is None and outcome.get("worker_exit_code") == 75, "Non-resource worker exit cannot be retried")
    require(status.get("status") == "resource_stopped" and status.get("error") in accepted,
            "Exit 75 is not proof of a permitted resource interruption")
    return accepted[status["error"]]


def inspect_case(case_dir, case, fingerprint, manifest_sha, revision_path=None, revision_sha=None):
    """Classify all preserved attempts. Ambiguous/non-resource evidence fails closed."""
    case_dir = Path(case_dir).resolve()
    case_id = case["case_id"]
    require(case_dir.name == case_id, "Case directory differs from the registered case")
    inventory = None
    if revision_path is not None:
        revision, parent, inventory = _envelope(revision_path, revision_sha)
        require(case_dir == parent / "evaluation" / case_id
                and fingerprint == revision["parent"]["protocol_sha256"]
                and manifest_sha == revision["parent"]["trained_manifest_sha256"], "Case audit binding differs")
    refuse_unsealed_complete(case_dir)
    entries = list(case_dir.iterdir()) if case_dir.exists() else []
    require(all(p.is_dir() and ATTEMPT.fullmatch(p.name) for p in entries), "Unexpected case evidence entry")
    numbers = sorted({int(ATTEMPT.fullmatch(p.name).group(1)) for p in entries})
    require(numbers == list(range(1, len(numbers) + 1)) and len(numbers) <= 3,
            "Attempt identities are nonconsecutive or exceed the frozen maximum")
    attempts, completed = [], []
    for number in numbers:
        output = case_dir / f"attempt_{number:03d}"
        launch = output.with_name(output.name + ".launch")
        require(output.is_dir() and launch.is_dir() and (launch / "outcome.json").is_file(),
                "Attempt has no durable completed launch outcome; audit before continuing")
        require(not (output / "report.json").exists() or (output / "receipt.json").exists(),
                "A report without a receipt requires diagnosis and cannot be rerun")
        files = _files_below(output) | _files_below(launch)
        historical = False
        schedule = None
        if inventory is not None:
            historical = bool(files & set(inventory))
            if historical:
                require(files <= set(inventory), "Historical attempt gained unbound files")
                for name in files:
                    require(digest(resolve(name)) == inventory[name], "Historical attempt changed")
            else:
                schedule = verify_launch_authorization(revision_path, revision_sha, case_id, output,
                    authorization_path(revision_path, case_id, output.name), require_live=False)
        outcome = read(launch / "outcome.json")
        require(outcome.get("protocol_sha256") == fingerprint
                and outcome.get("current") == output.relative_to(case_dir.parent.parent).as_posix(),
                "Attempt outcome targets another original protocol/case")
        if schedule:
            require(outcome.get("revision_sha256") == revision_sha
                    and outcome.get("owner") == schedule["authorization"]["owner"]
                    and outcome.get("launch_authorization_path") == schedule["binding"]["authorization_path"]
                    and outcome.get("launch_authorization_sha256") == schedule["authorization_sha256"],
                    "Attempt owner/revision/authorization changed")
        record = {"attempt_number": number, "output": relative(output), "launch": relative(launch),
                  "outcome_sha256": digest(launch / "outcome.json"), "historical": historical}
        if (output / "receipt.json").exists():
            binding = {"protocol_sha256": fingerprint, "trained_manifest_sha256": manifest_sha, "case_id": case_id}
            verify_seal(output, binding)
            result = read(output / "report.json")
            require(result.get("status") == "complete" and result.get("case") == case
                    and type(result.get("official_success")) is bool
                    and result.get("protocol_sha256") == fingerprint
                    and result.get("trained_manifest_sha256") == manifest_sha, "Sealed report binding differs")
            if schedule:
                require(read(output / "schedule_binding.json") == schedule["binding"], "Sealed scheduling binding differs")
            record.update(classification="completed", receipt_path=relative(output / "receipt.json"),
                          receipt_sha256=digest(output / "receipt.json"))
            completed.append((record, result))
        else:
            reason = _resource_attempt(output, launch, outcome)
            record.update(classification="resource_interrupted", resource_reason=reason)
        attempts.append(record)
    require(len(completed) <= 1, "Duplicate complete cases cannot select a favorable result")
    if completed:
        require(inventory is None or case_id not in revision["preserved_missing_case_ids"],
                "Permanently missing historical case cannot acquire a new result")
        record, result = completed[0]
        require(record["attempt_number"] == len(attempts), "A completed episode was retried")
        return {"case_id": case_id, "classification": "completed", "attempt_count": len(attempts),
                "attempts": attempts, "result": result, "receipt_path": record["receipt_path"],
                "receipt_sha256": record["receipt_sha256"]}
    classification = "resource_missing" if len(attempts) == 3 else "pending"
    if inventory is not None and case_id in revision["preserved_missing_case_ids"]:
        require(classification == "resource_missing" and all(a["historical"] for a in attempts),
                "Registered permanently missing case does not match three historical resource attempts")
    return {"case_id": case_id, "classification": classification, "attempt_count": len(attempts),
            "attempts": attempts, "missing_reason": "three_resource_attempts_exhausted" if classification == "resource_missing" else None}


def stable_resource_gate(*, sample, clock, wait, check_deadline, publish, paused=lambda: False):
    """Require 6 GiB/4 GiB for a continuous 60 seconds, sampled every five seconds."""
    stable_since = None
    previous_tick = None
    while True:
        if check_deadline():
            publish(state="partial_deadline")
            return False
        if paused():
            publish(state="paused_between_workers")
            return False
        resources = sample()
        tick = clock()
        require(math.isfinite(tick), "Invalid resource-gate monotonic clock")
        if previous_tick is not None and not 0 <= tick - previous_tick <= 10:
            # A suspend/long unsampled interval is not a demonstrated continuous
            # minute. Normal five-second sampling permits scheduling jitter.
            stable_since = None
        previous_tick = tick
        ready = resources["available_bytes"] >= 6 * GIB and resources["disk_free_bytes"] >= 4 * GIB
        stable_since = (tick if stable_since is None else stable_since) if ready else None
        duration = tick - stable_since if stable_since is not None else 0
        publish(state="waiting_resources", available_bytes=resources["available_bytes"],
                disk_free_bytes=resources["disk_free_bytes"], stable_seconds=duration)
        if ready and duration >= 60:
            return not check_deadline() and not paused()
        wait(5)


class ScheduleSupervisor(Supervisor):
    def __init__(self, revision_path, revision_sha, revision, parent, protocol):
        self.revision_path, self.revision_sha = Path(revision_path).resolve(), revision_sha
        self.revision_dir, self.revision = self.revision_path.parent, revision
        super().__init__(parent, revision["parent"]["protocol_sha256"], protocol)
        self.state.update(schema="feedbackbench.action_schedule_supervisor.v1", revision_sha256=revision_sha,
                          revision_path=relative(self.revision_path), training_completed=len(protocol["runs"]),
                          evaluation_planned=len(protocol["evaluation"]["cases"]), resource_missing_cases=[])

    def record(self, record, role):
        with (self.revision_dir / "process_ownership.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "role": role, "identity": record}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def save(self, **extra):
        self.state.update(updated_at=now(), **extra)
        write_json(self.revision_dir / "status.json", self.state)

    def report(self):
        from report_action_schedule_r1 import build_report
        write_json(self.revision_dir / "report.json", build_report(self.revision_path, self.revision_sha))

    def ready(self):
        self.report()
        return stable_resource_gate(sample=lambda: {"available_bytes": psutil.virtual_memory().available,
                                      "disk_free_bytes": shutil.disk_usage(ROOT).free},
            clock=time.monotonic, wait=time.sleep, check_deadline=deadline_reached, publish=self.save,
            paused=lambda: (self.revision_dir / "PAUSE_REQUEST.json").exists())

    def execute(self):
        verified = completed_training(self.campaign, self.protocol, self.fingerprint)
        require(read(self.campaign / "trained_manifest.json")["runs"] == verified
                and len(verified) == len(self.protocol["runs"]), "All original final checkpoints must remain complete")
        manifest_sha = self.revision["parent"]["trained_manifest_sha256"]
        completed = 0
        missing = []
        self.report()
        for case in self.protocol["evaluation"]["cases"]:
            case_dir = self.campaign / "evaluation" / case["case_id"]
            while True:
                inspected = inspect_case(case_dir, case, self.fingerprint, manifest_sha, self.revision_path, self.revision_sha)
                if inspected["classification"] == "completed":
                    completed += 1
                    self.save(state="between_workers", evaluation_completed=completed, current=None)
                    self.report()
                    break
                if inspected["classification"] == "resource_missing":
                    missing.append(case["case_id"])
                    self.save(state="between_workers", resource_missing_cases=missing, current=None)
                    self.report()
                    break
                if not self.ready():
                    self.report()
                    return
                verify_revision(self.revision_path, self.revision_sha)
                require(not deadline_reached(), "Original deadline elapsed before launch")
                fresh = inspect_case(case_dir, case, self.fingerprint, manifest_sha, self.revision_path, self.revision_sha)
                require(fresh == inspected, "Case evidence changed during the resource wait")
                # Hash verification can take time. If capacity fell since the
                # continuous gate, restart that gate before consuming an attempt.
                if psutil.virtual_memory().available < 6 * GIB or shutil.disk_usage(ROOT).free < 4 * GIB:
                    continue
                output = case_dir / f"attempt_{fresh['attempt_count'] + 1:03d}"
                auth_path = authorization_path(self.revision_path, case["case_id"], output.name)
                auth_path.parent.mkdir(exist_ok=True)
                auth = {"schema": AUTH_SCHEMA, "revision_path": relative(self.revision_path),
                        "revision_sha256": self.revision_sha, "parent_protocol_sha256": self.fingerprint,
                        "trained_manifest_sha256": manifest_sha, "case_id": case["case_id"],
                        "attempt_number": fresh["attempt_count"] + 1, "owner": self.state["owner"],
                        "output": relative(output), "launch": relative(output.with_name(output.name + ".launch")),
                        "authorized_at": now(), "prior_attempt_files": prior_attempt_files(output)}
                with auth_path.open("x", encoding="utf-8") as stream:
                    json.dump(auth, stream, sort_keys=True, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_action_schedule_r1.py"),
                           "--revision", str(self.revision_path), "--revision-sha256", self.revision_sha,
                           "--case-id", case["case_id"], "--output", str(output), "--authorization", str(auth_path)]
                for key in ("error", "reason", "interruption", "terminated", "terminated_in_finally", "worker_exit_code"):
                    self.state.pop(key, None)
                self.state.update(launch_authorization_path=relative(auth_path),
                                  launch_authorization_sha256=digest(auth_path))
                code = self.child(command, output, self.revision["resources"]["evaluation_max_seconds"])
                if code == 3:
                    outcome = read(output.with_name(output.name + ".launch") / "outcome.json")
                    expired = outcome.get("interruption") == "deadline" or deadline_reached()
                    self.save(state="partial_deadline" if expired else "needs_attention",
                              reason="original_deadline" if expired else "evaluation_time_limit", ended_at=now())
                    self.report()
                    return
                # Reclassify evidence, not exit code alone. A terminal episode
                # without a receipt or any non-resource failure stops the queue.
                inspect_case(case_dir, case, self.fingerprint, manifest_sha, self.revision_path, self.revision_sha)
        self.save(state="partial_resource_missing" if missing else "complete", current=None,
                  evaluation_completed=completed, resource_missing_cases=missing, ended_at=now())
        self.report()


def _owned_records(revision_path, revision, parent):
    records = identities(parent) + identities(Path(revision_path).parent)
    inventory = read(resolve(revision["prior_inventory"]["path"]))
    records.extend(_valid_identity(record) for record in inventory.get("identities_checked", []))
    for record in revision.get("launcher_records", []):
        records.extend(record["identities"])
    launch = Path(revision_path).parent / "launch.json"
    if launch.exists():
        launched = read(launch)
        for key in ("owner", "launcher", "child"):
            if isinstance(launched.get(key), dict):
                records.append(_valid_identity(launched[key]))
    return list({(r["pid"], r["created"]): _valid_identity(r) for r in records}.values())


def _lock_identities(value):
    records = []
    if isinstance(value, dict):
        if "pid" in value and "created" in value:
            records.append(_valid_identity({"pid": value["pid"], "created": value["created"]}))
        for item in value.values():
            records.extend(_lock_identities(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_lock_identities(item))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "verify", "supervise"))
    parser.add_argument("--revision", required=True, type=Path)
    parser.add_argument("--revision-sha256", required=True)
    parser.add_argument("--resume", action="store_true", help="Archive an audited dead global owner lock")
    args = parser.parse_args(argv)
    revision_path = args.revision.resolve()
    revision, parent, protocol = verify_revision(revision_path, args.revision_sha256,
                                                 inputs=args.command in ("verify", "supervise"))
    records = _owned_records(revision_path, revision, parent)
    if args.command == "status":
        value = read(revision_path.parent / "status.json") if (revision_path.parent / "status.json").exists() else {"state": "registered"}
        value["live_owned_processes"] = [r for r in records if matching(r) is not None]
        print(json.dumps(value, indent=2))
        return 0
    if args.command == "verify":
        print(json.dumps({"verified": True, "revision_sha256": args.revision_sha256}))
        return 0
    if deadline_reached():
        print(json.dumps({"state": "deadline_reached", "launch_permitted": False}))
        return 3
    current = identity(os.getpid())
    ancestry = [{"pid": p.pid, "created": p.create_time()} for p in psutil.Process().parents()]
    launcher_path = revision_path.parent / "launch.json"
    current_launchers = _lock_identities(read(launcher_path)) if launcher_path.exists() else []
    permitted_ancestors = [r for r in current_launchers if r in ancestry]
    # Historical owners/children must be dead. Only explicitly recorded current
    # Windows launcher ancestors may coexist with this supervisor.
    require(not [r for r in records if matching(r) is not None and r != current and r not in permitted_ancestors],
            "An owned process is already alive; duplicate launch refused")
    lock = AREA / "worker.lock"
    if lock.exists():
        previous = read(lock)
        known_revision = ("revision" not in previous and "revision_sha256" not in previous) or (
            previous.get("revision") == relative(revision_path.parent)
            and previous.get("revision_sha256") == args.revision_sha256)
        require(args.resume and previous.get("campaign") in (relative(parent), str(parent.relative_to(ROOT)))
                and known_revision
                and matching(_valid_identity(previous["owner"])) is None
                and not any(matching(record) is not None for record in _lock_identities(previous)),
                "Existing global lock requires an audited dead same-campaign owner")
        history = revision_path.parent / "lock_history"
        history.mkdir(exist_ok=True)
        lock.rename(history / (uuid4().hex + ".json"))
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"owner": current, "campaign": relative(parent), "revision": relative(revision_path.parent),
                   "revision_sha256": args.revision_sha256}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    supervisor = None
    try:
        supervisor = ScheduleSupervisor(revision_path, args.revision_sha256, revision, parent, protocol)
        supervisor.execute()
    except BaseException as error:
        state = supervisor.state if supervisor else {"owner": current, "revision_sha256": args.revision_sha256}
        state.update(state="needs_attention", error=repr(error), ended_at=now())
        if supervisor is not None:
            try:
                supervisor.report()
            except BaseException as report_error:
                state["report_error"] = repr(report_error)
        try:
            write_json(revision_path.parent / "status.json", state)
        except BaseException as persistence_error:
            with (revision_path.parent / ("persistence_failure_" + uuid4().hex[:12] + ".json")).open("x", encoding="utf-8") as stream:
                json.dump({"state": state, "persistence_error": repr(persistence_error)}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        raise
    print(json.dumps(supervisor.state, indent=2))
    return 0 if supervisor.state["state"] in ("complete", "partial_resource_missing") else 3


if __name__ == "__main__":
    raise SystemExit(main())
