"""Durable single-worker collection -> training -> held-out evaluation campaign.

The supervisor imports no learning stack. A fresh process owns each episode.
Completed failures are terminal evidence, never retried to improve success rates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from budget_campaign_common_v5 import (DEFAULT_CAMPAIGN, ROOT, canonical_hash, case_directory,
    deadline_reached, digest, fingerprint, owned_path, read_json, utc_now,
    verify_episode, verify_protocol, write_json)
from budget_protocol import make_protocol
from run import environment_python

SOURCES = (
    "scripts/budget_protocol.py", "scripts/budget_campaign_common.py",
    "scripts/run_budget_campaign.py", "scripts/budget_episode_worker.py", "scripts/budget_report.py",
    "scripts/train_budget_selector.py", "scripts/validate_budget_sampler_real.py",
    "scripts/libero_arena_budget.py", "scripts/run.py",
    "scripts/budget_startup_memory.py",
    "scripts/budget_campaign_common_v5.py", "scripts/run_budget_campaign_v5.py",
    "scripts/budget_episode_worker_v5.py", "tests/test_budget_io_v5.py",
    "tests/test_budget_monitor_v5.py",
    "src/embodied_research/budget_sampler.py", "src/embodied_research/budget_learning.py",
    "tests/test_budget_learning.py",
    "src/embodied_research/cache_sampler.py", "src/embodied_research/action_cache.py",
    "src/embodied_research/policies.py", "src/embodied_research/simulator.py",
    "src/embodied_research/runtime.py", "src/embodied_research/__init__.py",
    "docs/LEARNED_COMPUTE_PROTOCOL.md",
    "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    "vendor/MINERVA/src/lerobot/envs/libero.py",
)


def freeze(campaign, parity_path):
    campaign = Path(campaign).resolve()
    if (campaign / "protocol.json").exists():
        raise FileExistsError("campaign is already frozen; never overwrite a protocol")
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError("freeze requires an empty campaign directory")
    parity = read_json(parity_path)
    if parity.get("status") != "passed":
        raise ValueError("a passed real-checkpoint 2/4/10 parity receipt is required")
    for relative, expected in parity["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"parity source/input changed: {relative}")
    for name, record in parity["files"].items():
        if digest(owned_path(Path(parity_path).parent, name)) != record["sha256"]:
            raise ValueError(f"parity evidence changed: {name}")
    protocol = make_protocol(ROOT)
    if deadline_reached(protocol):
        raise RuntimeError("research deadline has passed")
    protocol.update(status="frozen", frozen_at=utc_now(), sources={name: digest(ROOT / name) for name in SOURCES})
    # Bind the actual package sources used below the public policy adapter.
    source_paths = set(SOURCES)
    for module in ("lerobot/processor/pipeline.py", "lerobot/policies/factory.py"):
        path = ROOT / "vendor/MINERVA/src" / module
        if path.is_file():
            source_paths.add(path.relative_to(ROOT).as_posix())
    protocol["sources"] = {name: digest(ROOT / name) for name in sorted(source_paths)}
    protocol["source_binding"] = {"status": "frozen_before_collection", "sources": sorted(source_paths)}
    protocol["input_manifests"] = {name: digest(ROOT / name) for name in (
        "models/smolvla/download_manifest.json", "models/smolvlm_assets/download_manifest.json",
        "assets/libero/asset_manifest.json", "requirements-runtime.in", "constraints-runtime.txt")}
    protocol["runtime"] = {"threads": 2, "interop_threads": 1, "arena_bytes": 256 * 1024**2,
                           "arena_native_warning_policy": "reject", "chunk_execution_steps": 10,
                           "image_size": 360, "model_dtype": "float32", "model_frozen": True,
                           "same_initial_noise_per_current_decision": True,
                           "random_distribution_source": "each full seed's calibration weighted frequencies"}
    protocol["resource_limits"]["status"] = "frozen"
    protocol["resource_limits"]["initial_min_system_available_bytes"] = int(5.25 * 1024**3)
    protocol["runtime"]["startup_memory_reclaim"] = "once after model load and before simulator construction, own process only; costs included in episode wall time; subsequent page faults included in inference wall time"
    protocol["campaign_revision"] = 5
    previous = ROOT / "outputs/learned_compute/v4"
    prior_protocol = verify_protocol(previous)
    prior_completed = []
    for case in prior_protocol["collection"]["cases"]:
        destination = previous / "collection" / case_directory(case)
        candidates = ([destination] if destination.exists() else
                      sorted((previous / "attempts/collection" / case_directory(case)).glob("*/receipt.json")))
        if len(candidates) > 1:
            raise ValueError("multiple prior sealed attempts for one case")
        if candidates:
            episode = candidates[0] if candidates[0].is_dir() else candidates[0].parent
            _, result = verify_episode(episode, case, prior_protocol, strict=True)
            prior_completed.append({"case_id": case["case_id"], "steps": result["steps"],
                                    "receipt_sha256": digest(episode / "receipt.json"),
                                    "path": episode.relative_to(ROOT).as_posix()})
    if len(prior_completed) != 3:
        raise ValueError("v5 supersedes exactly three verified v4 collection episodes")
    protocol["supersedes"] = {
        "campaign": "outputs/learned_compute/v4", "protocol_sha256": digest(previous / "protocol.json"),
        "collection_episodes_completed": len(prior_completed),
        "executed_policy_actions": sum(row["steps"] for row in prior_completed),
        "completed_episodes": prior_completed,
        "prior_collection_included": False,
        "reason": "v4 supervisor stopped on a Windows atomic status replacement denial; its third worker sealed valid evidence after the parent exited. v5 adds bounded atomic replacement retries and owned-process cleanup after persistent monitoring I/O failure. All three v4 trajectories and sources are preserved independently, not pooled.",
        "algorithm_and_splits_changed": False,
    }
    protocol["runtime"]["io_failure_policy"] = "bounded atomic JSON replacement retry; persistent supervisor status/control write errors stop owned worker cooperatively, retain monitoring through existing grace, reap exact process identities, and require attention without automatic episode retry"
    protocol["resource_limits"].update(max_attempts_per_case=3, retry_wait_seconds=600,
                                       resource_stop_grace_seconds=180,
                                       minimum_free_disk_bytes=10 * 1024**3)
    protocol["training"].update(hidden_dim=64, max_epochs=100, batch_size=256, learning_rate=.001,
                                weight_decay=.0001, risk_scales=[.05, .05, .1], residual_quantile=.9,
                                thresholds=[0, .25, .5, .75, 1, 1.5, 2, 3, 4, 8, 16],
                                mean_calibration_budget_limit=4,
                                target_units=["normalized_translation_rmse", "normalized_rotation_rmse",
                                              "postprocessed_gripper_sign_disagreement"],
                                reference_semantics="10-step model action is an imitation reference, not ground truth")
    protocol["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "numpy", "transformers", "mujoco", "robosuite", "hf-libero", "psutil")}
    protocol["parity"] = {"report_sha256": digest(parity_path), "report": parity}
    protocol["campaign_fingerprint"] = fingerprint(protocol)
    campaign.mkdir(parents=True, exist_ok=True)
    for name in source_paths:
        destination = campaign / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(campaign / "protocol.json", protocol)
    write_json(campaign / "status.json", {"state": "ready", "created_at": utc_now(),
               "campaign_fingerprint": protocol["campaign_fingerprint"], "counts": protocol["counts"]})
    return {"campaign": str(campaign), "fingerprint": protocol["campaign_fingerprint"], "counts": protocol["counts"]}


def process_identity(pid=None):
    import psutil
    process = psutil.Process(pid)
    return {"pid": process.pid, "created": process.create_time()}


def is_alive(identity):
    import psutil
    import math
    if (not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 0
            or type(identity.get("created")) not in (int, float) or not math.isfinite(identity["created"])):
        raise ValueError("invalid persisted process identity; manual diagnosis required")
    try:
        process = psutil.Process(identity["pid"])
        return process.create_time() == identity["created"] and process.is_running()
    except psutil.NoSuchProcess:
        return False


def recorded_owned_processes(campaign):
    """Append-only ownership receipts survive a failed status replacement."""
    identities = {}
    for path in sorted((Path(campaign) / "process_ownership").glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("empty process ownership receipt; manual diagnosis required")
        for line in lines:
            record = json.loads(line)
            if not record["owned_processes"]:
                raise ValueError("empty process ownership identity snapshot; manual diagnosis required")
            for identity in record["owned_processes"]:
                # is_alive validates both fields and fails closed on bad data.
                key = (identity["pid"], identity["created"])
                identities[key] = identity
    return list(identities.values())


def fresh_diagnostic(campaign, record):
    """Use a new exclusive file when replacement of an existing path failed."""
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = owned_path(campaign, "diagnostics", f"io-{os.getpid()}-{time.time_ns()}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path.relative_to(campaign).as_posix()


@contextmanager
def campaign_lock(campaign):
    lock = owned_path(campaign, "supervisor.lock")
    identity = process_identity()
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)
    except FileExistsError:
        old = read_json(lock)
        if is_alive(old):
            raise RuntimeError(f"campaign already has an active supervisor PID {old['pid']}")
        status_path = Path(campaign) / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        known_children = status.get("child_processes", [])
        if status.get("child"):
            known_children = [status["child"], *known_children]
        if any(is_alive(child) for child in known_children):
            raise RuntimeError("previous supervisor died but its child is still alive; do not duplicate it")
        stale = owned_path(campaign, "locks", f"{time.time_ns()}.json")
        stale.parent.mkdir(parents=True, exist_ok=True)
        lock.rename(stale)
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)
    try:
        # Also protect restarts after an exception removed the lock while a
        # deliberately un-killed surviving child is still recorded in status.
        status_path = Path(campaign) / "status.json"
        previous_status = read_json(status_path) if status_path.exists() else {}
        children = list(previous_status.get("child_processes", []))
        if previous_status.get("child"):
            children.append(previous_status["child"])
        children.extend(recorded_owned_processes(campaign))
        if any(is_alive(child) for child in children):
            raise RuntimeError("previous worker is still alive; do not launch a duplicate")
        yield identity
    finally:
        if lock.exists() and read_json(lock) == identity:
            lock.unlink()


def status_write(campaign, identity, state, **details):
    write_json(Path(campaign) / "status.json", {"state": state, "updated_at": utc_now(),
               "supervisor": identity, **details})


def complete_cases(campaign, protocol, phase):
    complete = {}
    for case in protocol[phase]["cases"]:
        destination = owned_path(campaign, phase, case_directory(case))
        if destination.exists():
            _, result = verify_episode(destination, case, protocol, strict=True)
            complete[case["case_id"]] = result
    return complete


def freeze_trained(campaign, protocol):
    trained = Path(campaign) / "trained"
    summary = read_json(trained / "summary.json")
    binding = read_json(trained / "binding.json")
    if (summary.get("status") != "complete" or summary.get("binding") != binding
            or binding["campaign_fingerprint"] != protocol["campaign_fingerprint"]):
        raise ValueError("training did not finish against this frozen campaign")
    checkpoints = list(trained.glob("*/selector.pt"))
    expected = {(variant, seed) for variant in protocol["training"]["feature_modes"]
                for seed in protocol["training"]["seeds"]}
    records = summary["models"]
    if len(checkpoints) != 9 or len(records) != 9 or {(r["variant"], r["seed"]) for r in records} != expected:
        raise ValueError("all 9 seed/ablation checkpoints must exist before opening test evaluation")
    for record in records:
        if digest(owned_path(trained, record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("trained checkpoint does not match completed training summary")
    for relative, record in binding["sources"].items():
        if digest(ROOT / relative) != record["sha256"]:
            raise ValueError("training source binding changed")
    if canonical_hash(read_json(trained / "config.json")) != binding["config_sha256"]:
        raise ValueError("training configuration binding changed")
    data_manifest = read_json(trained / "data_manifest.json")
    if (canonical_hash({key: value for key, value in data_manifest.items() if key != "fingerprint"})
            != binding["data_fingerprint"] or data_manifest["fingerprint"] != binding["data_fingerprint"]):
        raise ValueError("training input manifest binding changed")
    expected_cases = {case["case_id"] for case in protocol["collection"]["cases"]}
    if {row["case_id"] for row in data_manifest["episodes"]} != expected_cases or len(data_manifest["episodes"]) != len(expected_cases):
        raise ValueError("training input cases differ from the complete declared collection")
    for row in data_manifest["episodes"]:
        if digest(owned_path(campaign, "collection", row["relative_directory"], "receipt.json")) != row["receipt_sha256"]:
            raise ValueError("training receipt binding changed")
    files = {path.relative_to(campaign).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(trained.rglob("*")) if path.is_file()}
    manifest = {"schema": "feedbackbench.budget_trained_manifest.v1", "frozen_at": utc_now(),
                "campaign_fingerprint": protocol["campaign_fingerprint"], "files": files,
                "controller_count": len(checkpoints), "test_opened_after_freeze": True}
    target = Path(campaign) / "trained_manifest.json"
    if target.exists():
        raise FileExistsError("trained checkpoint manifest already frozen")
    write_json(target, manifest)
    return summary


def monitor_child(campaign, protocol, identity, command, phase, run_path, log_path, attempt=None, case=None):
    import psutil
    campaign = Path(campaign)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ownership_path = owned_path(campaign, "process_ownership", run_path.stem + ".jsonl")
    ownership_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    stopped = None
    reason = None
    io_errors = []
    survivors = []
    peak_rss = 0
    minimum_free = psutil.virtual_memory().available
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # The append-only ownership file is opened before creating a model worker.
    # It provides a second identity record if atomic status replacement fails.
    with log_path.open("ab", buffering=0) as log, ownership_path.open("xb", buffering=0) as ownership:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=creationflags)
        child_identity = process_identity(child.pid)
        process = psutil.Process(child.pid)
        known = {(child_identity["pid"], child_identity["created"]): process}
        recorded = set()

        def identities():
            return [{"pid": key[0], "created": key[1]} for key in known]

        def io_failure(error, operation):
            nonlocal reason
            reason = "supervisor_io_failure"
            record = {"operation": operation, "error_type": type(error).__name__,
                      "error": str(error), "winerror": getattr(error, "winerror", None),
                      "at": utc_now(), "owned_processes": identities(),
                      "phase": phase, "case_id": None if case is None else case["case_id"],
                      "supervisor": identity, "child": child_identity,
                      "action_required": True}
            io_errors.append(record)
            try:
                log.write((json.dumps({"supervisor_io_failure": record}, sort_keys=True) + "\n").encode("utf-8"))
                os.fsync(log.fileno())
            except OSError as log_error:
                record["fallback_log_error"] = str(log_error)
            try:
                record["diagnostic"] = fresh_diagnostic(campaign, record)
            except OSError as diagnostic_error:
                record["fallback_diagnostic_error"] = str(diagnostic_error)

        while True:
            # Windows venv python.exe is a redirector; the model runs in a
            # descendant interpreter. Follow only this owned process tree and
            # retain PID+creation identities even if its launcher exits first.
            for root_key, root_handle in list(known.items()):
                if is_alive({"pid": root_key[0], "created": root_key[1]}):
                    try:
                        for descendant in root_handle.children(recursive=True):
                            key = (descendant.pid, descendant.create_time())
                            known[key] = descendant
                    except psutil.NoSuchProcess:
                        pass
            if set(known) != recorded:
                try:
                    ownership.write((json.dumps({"owned_processes": identities(), "at": utc_now()},
                                                sort_keys=True) + "\n").encode("utf-8"))
                    os.fsync(ownership.fileno())
                except OSError as error:
                    io_failure(error, "process_ownership_append")
                recorded = set(known)
            active = [(key, handle) for key, handle in known.items()
                      if is_alive({"pid": key[0], "created": key[1]})]
            if child.poll() is not None and not active:
                break
            free = psutil.virtual_memory().available
            rss = 0
            for key, handle in active:
                try:
                    rss += handle.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_rss, minimum_free = max(peak_rss, rss), min(minimum_free, free)
            if deadline_reached(protocol) and not io_errors:
                reason = "deadline_reached"
            elif rss > protocol["resource_limits"]["max_process_rss_bytes"] or free < protocol["resource_limits"]["min_system_available_bytes"]:
                reason = reason or "resource_guard"
            if reason and stopped is None:
                stopped = time.monotonic()
                if attempt is not None:
                    try:
                        write_json(owned_path(campaign, "controls", attempt.parent.parent.name,
                                   attempt.parent.name, attempt.name + ".json"),
                                   {"reason": reason, "requested_at": utc_now()})
                    except OSError as error:
                        io_failure(error, "control_write")
            if not io_errors:
                try:
                    status_write(campaign, identity, phase, child=child_identity,
                                 child_processes=[{"pid": key[0], "created": key[1]} for key, _ in active],
                                 case_id=None if case is None else case["case_id"],
                                 attempt=None if attempt is None else str(attempt.relative_to(campaign)),
                                 child_rss_bytes=rss, available_bytes=free,
                                 elapsed_seconds=time.monotonic() - started, stop_reason=reason)
                except OSError as error:
                    io_failure(error, "status_write")
                    # Keep the monitor alive, but do not repeatedly write the
                    # blocked status path. Request an ordinary cooperative stop.
                    if stopped is None:
                        stopped = time.monotonic()
                        if attempt is not None:
                            try:
                                write_json(owned_path(campaign, "controls", attempt.parent.parent.name,
                                           attempt.parent.name, attempt.name + ".json"),
                                           {"reason": reason, "requested_at": utc_now()})
                            except OSError as control_error:
                                io_failure(control_error, "control_write")
            if stopped is not None and time.monotonic() - stopped >= protocol["resource_limits"]["resource_stop_grace_seconds"]:
                # Descendants first. is_alive binds both PID and creation time;
                # no process-name matching or unrelated-project termination.
                for key, handle in reversed(active):
                    if is_alive({"pid": key[0], "created": key[1]}):
                        try:
                            handle.terminate()
                        except psutil.NoSuchProcess:
                            pass
                        except psutil.AccessDenied as error:
                            io_failure(error, "owned_process_terminate")
                _, survivors = psutil.wait_procs([handle for _, handle in active], timeout=10)
                if survivors:
                    io_failure(RuntimeError("owned worker survived termination; do not launch another worker"),
                               "owned_process_reap")
                break
            time.sleep(5)
        exit_code = child.poll() if survivors else child.wait()
        audit = {"phase": phase, "case_id": None if case is None else case["case_id"],
                 "child": child_identity, "exit_code": exit_code, "stop_reason": reason,
                 "owned_processes": identities(), "children_reaped": not survivors,
                 "rss_scope": "sum of the owned launcher and observed interpreter descendants",
                 "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss,
                 "minimum_available_bytes": minimum_free, "ended_at": utc_now(),
                 "io_errors": io_errors, "action_required": bool(io_errors),
                 "process_ownership": ownership_path.relative_to(campaign).as_posix()}
        try:
            write_json(run_path, audit)
        except OSError as error:
            io_failure(error, "run_audit_write")
            audit.update(stop_reason=reason, action_required=True)
            try:
                audit["fallback_audit"] = fresh_diagnostic(campaign, audit)
            except OSError as fallback_error:
                audit["fallback_audit_error"] = str(fallback_error)
    return audit


def require_io_attention(campaign, identity, audit):
    """Stop the queue after an I/O fault; a sealed episode remains recoverable."""
    record = {"state": "needs_attention", "reason": "supervisor_io_failure", "audit": audit,
              "supervisor": identity, "updated_at": utc_now(), "action_required": True}
    try:
        status_write(campaign, identity, "needs_attention", reason="supervisor_io_failure", audit=audit)
    except OSError as error:
        record["status_write_error"] = str(error)
        record["primary_status_is_stale"] = True
    try:
        record["diagnostic"] = fresh_diagnostic(campaign, record)
    except OSError as error:
        record["diagnostic_write_error"] = str(error)
    try:
        record["report"] = build_report(campaign)
    except OSError as error:
        record["report_write_error"] = str(error)
    return record


def build_report(campaign):
    from budget_report import generate_report
    report = generate_report(campaign)
    write_json(Path(campaign) / "report.json", report)
    return report


def training_attempt_ordinals(campaign):
    """Count reserved runs even when a stopped child never created trained/."""
    campaign = Path(campaign)
    ordinals = set()
    for directory, pattern in (("runs", "training-*.json"),
                               ("process_ownership", "training-*.jsonl"),
                               ("logs", "training-*.log")):
        for path in (campaign / directory).glob(pattern):
            suffix = path.stem.removeprefix("training-")
            if not suffix.isdecimal() or int(suffix) < 1:
                raise ValueError("invalid persisted training attempt identity")
            ordinals.add(int(suffix))
    for path in (campaign / "training_attempts").glob("*"):
        if not path.is_dir() or not path.name.isdecimal() or int(path.name) < 1:
            raise ValueError("invalid archived training attempt identity")
        ordinals.add(int(path.name))
    return ordinals


def supervise(campaign, max_episodes=None):
    import psutil
    campaign = Path(campaign).resolve()
    protocol = verify_protocol(campaign)
    python = str(environment_python(ROOT))
    completed_here = 0
    with campaign_lock(campaign) as identity:
        while True:
            verify_protocol(campaign)
            if deadline_reached(protocol):
                status_write(campaign, identity, "deadline_reached")
                return build_report(campaign)
            if (campaign / "PAUSE_REQUEST.json").exists():
                status_write(campaign, identity, "paused_by_request")
                return build_report(campaign)
            collected = complete_cases(campaign, protocol, "collection")
            evaluated = complete_cases(campaign, protocol, "evaluation")
            if len(collected) < len(protocol["collection"]["cases"]):
                phase = "collection"
                case = next(row for row in protocol[phase]["cases"] if row["case_id"] not in collected)
            elif not (campaign / "trained_manifest.json").exists():
                phase, case = "training", None
            elif len(evaluated) < len(protocol["evaluation"]["cases"]):
                phase = "evaluation"
                case = next(row for row in protocol[phase]["cases"] if row["case_id"] not in evaluated)
            else:
                status_write(campaign, identity, "complete", collection=len(collected), evaluation=len(evaluated))
                return build_report(campaign)
            available = psutil.virtual_memory().available
            minimum = protocol["resource_limits"]["initial_min_system_available_bytes"] if phase != "training" else int(1.5 * 1024**3)
            if available < minimum or shutil.disk_usage(campaign).free < protocol["resource_limits"]["minimum_free_disk_bytes"]:
                status_write(campaign, identity, "waiting_resources", next_phase=phase,
                             available_bytes=available, required_bytes=minimum,
                             collection=len(collected), evaluation=len(evaluated))
                time.sleep(30)
                continue
            if phase == "training":
                training_ordinals = training_attempt_ordinals(campaign)
                if (campaign / "trained").exists():
                    summary_path = campaign / "trained/summary.json"
                    if summary_path.exists() and read_json(summary_path).get("status") == "complete":
                        freeze_trained(campaign, protocol)
                        continue
                    # Resume by preserving the interrupted fitting attempt and
                    # deterministically rerunning all nine models, with test closed.
                    archives = owned_path(campaign, "training_attempts")
                    archives.mkdir(exist_ok=True)
                    number = max(training_ordinals, default=1)
                    owned_path(campaign, "trained").rename(owned_path(campaign, "training_attempts", f"{number:03d}"))
                    training_ordinals.add(number)
                training_ordinal = max(training_ordinals, default=0) + 1
                if training_ordinal > protocol["resource_limits"]["max_attempts_per_case"]:
                    status_write(campaign, identity, "needs_attention", reason="training attempt limit reached")
                    return build_report(campaign)
                command = [python, "scripts/train_budget_selector.py", "--data", str(campaign / "collection"),
                           "--output", str(campaign / "trained"), "--epochs", "100", "--batch-size", "256",
                           "--learning-rate", ".001", "--weight-decay", ".0001", "--risk-scales", ".05", ".05", ".1",
                           "--residual-quantile", ".9", "--thresholds", "0", ".25", ".5", ".75", "1", "1.5", "2", "3", "4", "8", "16"]
                audit = monitor_child(campaign, protocol, identity, command, phase,
                    campaign / "runs" / f"training-{training_ordinal:03d}.json",
                    campaign / "logs" / f"training-{training_ordinal:03d}.log")
                if audit.get("action_required"):
                    return require_io_attention(campaign, identity, audit)
                if audit["exit_code"] != 0 or audit["stop_reason"]:
                    if audit["stop_reason"]:
                        continue
                    status_write(campaign, identity, "needs_attention", reason="training interrupted or failed", audit=audit)
                    return build_report(campaign)
                freeze_trained(campaign, protocol)
                build_report(campaign)
                continue
            task = next(row for row in protocol["tasks"] if row["task_uid"] == case["task_uid"])
            if digest(ROOT / task["path"]) != task["bddl_sha"] or digest(ROOT / task["initial_path"]) != task["hash"]:
                raise ValueError("official task/initial-state input bytes changed")
            case_root = owned_path(campaign, "attempts", phase, case_directory(case))
            case_root.mkdir(parents=True, exist_ok=True)
            attempts = sorted(path for path in case_root.iterdir() if path.is_dir())
            # Recover a fully sealed episode if the supervisor died before move.
            recovered = [path for path in attempts if (path / "receipt.json").exists()]
            if recovered:
                if len(recovered) != 1:
                    raise RuntimeError("multiple completed attempts are not allowed")
                attempt = recovered[0]
                verify_episode(attempt, case, protocol, strict=True)
            else:
                if len(attempts) >= protocol["resource_limits"]["max_attempts_per_case"]:
                    status_write(campaign, identity, "needs_attention", reason="attempt limit reached", case_id=case["case_id"])
                    return build_report(campaign)
                attempt = owned_path(campaign, "attempts", phase, case_directory(case), f"{len(attempts) + 1:03d}")
                attempt.mkdir()
                tag = f"{phase}-{case_directory(case)}-{len(attempts) + 1:03d}"
                command = [python, "scripts/budget_episode_worker_v5.py", "--campaign", str(campaign), "--phase", phase,
                           "--case-id", case["case_id"], "--attempt", str(attempt)]
                audit = monitor_child(campaign, protocol, identity, command, phase,
                    campaign / "runs" / f"{tag}.json", campaign / "logs" / f"{tag}.log", attempt, case)
                if audit.get("action_required"):
                    return require_io_attention(campaign, identity, audit)
                if audit["exit_code"] != 0 or audit["stop_reason"]:
                    if audit["exit_code"] == 3 or audit["stop_reason"]:
                        # Preserve all partial evidence; never count interruption as task failure.
                        status_write(campaign, identity, "waiting_after_resource_stop", audit=audit)
                        for _ in range(protocol["resource_limits"]["retry_wait_seconds"] // 10):
                            if deadline_reached(protocol) or (campaign / "PAUSE_REQUEST.json").exists():
                                break
                            time.sleep(10)
                        continue
                    status_write(campaign, identity, "needs_attention", reason="episode worker failed", audit=audit)
                    return build_report(campaign)
                verify_episode(attempt, case, protocol, strict=True)
            destination = owned_path(campaign, phase, case_directory(case))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("never overwrite completed evidence")
            # Both absolute paths were checked as descendants of this campaign.
            attempt.rename(destination)
            completed_here += 1
            build_report(campaign)
            if max_episodes is not None and completed_here >= max_episodes:
                status_write(campaign, identity, "bounded_run_complete", completed_here=completed_here)
                return build_report(campaign)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "supervise", "status", "report", "verify"))
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.command == "freeze":
        if args.parity is None:
            parser.error("freeze requires --parity")
        result = freeze(args.campaign, args.parity)
    elif args.command == "supervise":
        try:
            result = supervise(args.campaign, args.max_episodes)
        except BaseException as error:
            # Do not replace another supervisor's status on lock rejection.
            write_json(args.campaign / "supervisor_error.json", {"error": str(error), "at": utc_now(),
                       "traceback": traceback.format_exc()})
            raise
    elif args.command == "status":
        result = read_json(args.campaign / "status.json")
        result["supervisor_alive"] = bool(result.get("supervisor") and is_alive(result["supervisor"]))
        result["child_alive"] = bool(result.get("child") and is_alive(result["child"]))
        recorded = recorded_owned_processes(args.campaign)
        result["known_owned_process_count"] = len(recorded)
        result["live_owned_processes"] = [identity for identity in recorded if is_alive(identity)]
    elif args.command == "report":
        result = build_report(args.campaign)
    else:
        protocol = verify_protocol(args.campaign)
        result = {phase: len(complete_cases(args.campaign, protocol, phase)) for phase in ("collection", "evaluation")}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
