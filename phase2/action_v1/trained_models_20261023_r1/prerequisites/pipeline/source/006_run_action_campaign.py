"""One supervised CPU worker for the independent expert-action adaptation study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import psutil

from action_campaign_common import ROOT, deadline_reached, digest, now, read, resolve, verify_protocol, verify_seal, write_json
from run_action_adaptation_probe import collect_owned, identity, matching, stop_owned

GIB = 1024 ** 3
AREA = ROOT / "outputs/action_adaptation"


def checkpoint_receipt(path, protocol_sha256, run_id):
    receipt = read(path / "receipt.json")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    import hashlib
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != receipt.get("receipt_sha256"):
        raise ValueError("Training checkpoint receipt changed")
    if receipt.get("schema") != "feedbackbench.action_adaptation_checkpoint.v1":
        raise ValueError("Unknown checkpoint schema")
    binding_json = json.dumps(receipt["binding"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(binding_json.encode()).hexdigest() != receipt.get("binding_sha256"):
        raise ValueError("Checkpoint input binding changed")
    if receipt["binding"]["protocol_sha256"] != protocol_sha256 or receipt["binding"]["run_id"] != run_id:
        raise ValueError("Training checkpoint belongs to another run/protocol")
    if set(receipt.get("files", {})) != {"adapter.npz", "optimizer.npz", "rng.npz", "state.json", "history.json"}:
        raise ValueError("Incomplete training checkpoint")
    actual = {p.name for p in path.iterdir() if p.is_file()}
    if actual != set(receipt["files"]) | {"receipt.json"}:
        raise ValueError("Training checkpoint file set changed")
    for name, expected in receipt["files"].items():
        if digest(resolve(name, path)) != expected:
            raise ValueError(f"Training checkpoint changed: {name}")
    return receipt


def identities(campaign):
    result = []
    def add(record):
        if record is not None and record not in result:
            if set(record) != {"pid", "created"} or type(record["pid"]) is not int or type(record["created"]) not in (int, float):
                raise ValueError("Malformed owned process identity")
            result.append(record)
    journal = campaign / "process_ownership.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)["identity"]
            add(record)
    paths = [campaign / "status.json", *campaign.glob("persistence_failure_*.json"),
             *campaign.glob("training/*/attempt_*.launch/outcome.json"),
             *campaign.glob("evaluation/*/attempt_*.launch/outcome.json")]
    for path in paths:
        if path.exists():
            record = read(path)
            if isinstance(record.get("state"), dict):
                record = record["state"]
            add(record.get("owner"))
            add(record.get("child"))
            for child in record.get("owned_children", []):
                add(child)
    return result


def status(campaign):
    state = read(campaign / "status.json") if (campaign / "status.json").exists() else {"state": "registered"}
    state["live_owned_processes"] = [r for r in identities(campaign) if matching(r) is not None]
    return state


def completed_training(campaign, protocol, fingerprint):
    completed = {}
    for run_id, run in protocol["runs"].items():
        for report_path in sorted((campaign / "training" / run_id).glob("attempt_*/report.json")):
            report = read(report_path)
            if report.get("status") != "complete":
                continue
            if run_id in completed:
                raise ValueError("Duplicate completed training evidence")
            if report.get("protocol_sha256") != fingerprint or report.get("run_id") != run_id:
                raise ValueError("Training report binding mismatch")
            if report["completed_steps"] != run["steps"]:
                raise ValueError("Completed training has wrong step count")
            checkpoint = resolve(report["final_checkpoint"]["path"], report_path.parent)
            receipt = checkpoint_receipt(checkpoint, fingerprint, run_id)
            if receipt["completed_steps"] != run["steps"]:
                raise ValueError("Final checkpoint step count mismatch")
            adapter = resolve(report["final_checkpoint"]["adapter"], checkpoint)
            if digest(adapter) != report["final_checkpoint"]["sha256"]:
                raise ValueError("Final adapter binding mismatch")
            completed[run_id] = {"mode": run["mode"], "seed": run["initialization_seed"],
                "adapter": {"path": adapter.relative_to(ROOT).as_posix(), "sha256": digest(adapter)},
                "receipt": {"path": (checkpoint / "receipt.json").relative_to(ROOT).as_posix(),
                            "sha256": digest(checkpoint / "receipt.json")},
                "report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": digest(report_path)},
                "base_fingerprint": report["base_fingerprint"], "completed_steps": report["completed_steps"]}
    return completed


def latest_checkpoint(run_dir, fingerprint, run_id):
    candidates = []
    for path in run_dir.glob("attempt_*/checkpoints/step_*/receipt.json"):
        if re.fullmatch(r"step_\d{6}", path.parent.name) is None:
            continue
        receipt = checkpoint_receipt(path.parent, fingerprint, run_id)
        candidates.append((receipt["completed_steps"], str(path.parent), path.parent))
    return max(candidates)[2] if candidates else None


def refuse_unsealed_complete(case_dir):
    for path in case_dir.glob("attempt_*/terminal_observed.json"):
        if not (path.parent / "receipt.json").exists():
            raise RuntimeError("Terminal evaluation observed but not sealed; audit existing evidence, never rerun the case")
    for path in case_dir.glob("attempt_*/report.json"):
        if read(path).get("status") == "complete" and not (path.parent / "receipt.json").exists():
            raise RuntimeError("Complete evaluation evidence is not sealed; audit/seal existing evidence, never rerun the case")


class Supervisor:
    def __init__(self, campaign, fingerprint, protocol):
        self.campaign, self.fingerprint, self.protocol = campaign, fingerprint, protocol
        self.state = {"schema": "feedbackbench.action_supervisor.v1", "state": "starting",
                      "owner": identity(os.getpid()), "protocol_sha256": fingerprint, "started_at": now(),
                      "owned_children": [], "current": None, "training_completed": 0, "evaluation_completed": 0}
        self.record(self.state["owner"], "supervisor")

    def record(self, record, role):
        with (self.campaign / "process_ownership.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "role": role, "identity": record}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def save(self, **extra):
        self.state.update(updated_at=now(), **extra)
        write_json(self.campaign / "status.json", self.state)

    def ready(self):
        while not deadline_reached():
            if (self.campaign / "PAUSE_REQUEST.json").exists():
                self.save(state="paused_between_workers")
                return False
            if psutil.virtual_memory().available >= 4.25 * GIB and shutil.disk_usage(ROOT).free >= 4 * GIB:
                return True
            self.save(state="waiting_resources", available_gib=psutil.virtual_memory().available / GIB)
            time.sleep(30)
        self.save(state="partial_deadline")
        return False

    def child(self, command, output, max_seconds):
        logs = output.with_name(output.name + ".launch")
        logs.mkdir(parents=True, exist_ok=False)
        verify_protocol(self.campaign / "protocol.json", self.fingerprint)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
        self.save(state="launching", current=output.relative_to(self.campaign).as_posix(), owned_children=[])
        process = None
        reason = None
        try:
            with (logs / "stdout.log").open("x", encoding="utf-8") as stdout, (logs / "stderr.log").open("x", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                child = identity(process.pid)
                self.state["owned_children"].append(child)
                self.record(child, "worker")
                self.save(state="running", command=command, worker_started_at=now())
                started = time.monotonic()
                last_saved = started
                with (logs / "resources.jsonl").open("x", encoding="utf-8") as resources:
                    while process.poll() is None:
                        before = list(self.state["owned_children"])
                        collect_owned(child, self.state["owned_children"])
                        for record in self.state["owned_children"]:
                            if record not in before:
                                self.record(record, "descendant")
                        free = psutil.virtual_memory().available
                        resources.write(json.dumps({"at": now(), "available_bytes": free}) + "\n")
                        resources.flush()
                        reason = ("deadline" if deadline_reached() else "time_limit" if time.monotonic() - started >= max_seconds
                                  else "memory_reserve" if free < .75 * GIB else "disk_reserve" if shutil.disk_usage(ROOT).free < 4 * GIB else None)
                        if reason:
                            break
                        if time.monotonic() - last_saved >= 5:
                            self.save()
                            last_saved = time.monotonic()
                        time.sleep(.5)
                if reason:
                    self.state["terminated"] = stop_owned(self.state["owned_children"])
                code = process.wait(timeout=10)
                self.save(state="worker_exited", worker_exit_code=code, interruption=reason)
                write_json(logs / "outcome.json", dict(self.state))
                return 3 if reason in ("deadline", "time_limit") else 75 if reason else code
        finally:
            # Storage faults, exceptions and unexpected launcher exits cannot leave a known worker running.
            if process is not None:
                if not self.state["owned_children"] and process.poll() is None:
                    self.state["owned_children"].append(identity(process.pid))
                if self.state["owned_children"]:
                    collect_owned(self.state["owned_children"][0], self.state["owned_children"])
                if any(matching(r) is not None for r in self.state["owned_children"]):
                    self.state["terminated_in_finally"] = stop_owned(self.state["owned_children"])
                    self.save(state="needs_attention", reason="remaining_owned_children")
                    raise RuntimeError("Worker left owned descendants; stopped and retained evidence")

    def execute(self):
        protocol_path = self.campaign / "protocol.json"
        common = ["--protocol", str(protocol_path), "--protocol-sha256", self.fingerprint]
        completed = completed_training(self.campaign, self.protocol, self.fingerprint)
        self.save(training_completed=len(completed))
        for run_id in self.protocol["runs"]:
            if run_id in completed:
                continue
            run_dir = self.campaign / "training" / run_id
            while True:
                if not self.ready():
                    return
                attempt_number = len(list(run_dir.glob("attempt_*.launch"))) + 1
                if attempt_number > 3:
                    raise RuntimeError(f"Three preserved attempts exhausted for {run_id}")
                output = run_dir / f"attempt_{attempt_number:03d}"
                resume = latest_checkpoint(run_dir, self.fingerprint, run_id)
                command = [sys.executable, "-B", str(ROOT / "scripts/train_action_adaptation.py"), *common,
                           "--run-id", run_id, "--output", str(output)]
                if resume:
                    command += ["--resume", str(resume)]
                code = self.child(command, output, self.protocol["resources"]["max_seconds"])
                completed = completed_training(self.campaign, self.protocol, self.fingerprint)
                if run_id in completed:
                    self.save(training_completed=len(completed))
                    break
                if deadline_reached() or code == 3:
                    self.save(state="partial_deadline" if deadline_reached() else "needs_attention", reason="training_time_limit")
                    return
                if code != 75:
                    raise RuntimeError(f"Training worker failed ({code}); diagnosis required")
                self.save(state="waiting_resources", reason="preserved_resource_interruption")
                time.sleep(300)
        manifest_path = self.campaign / "trained_manifest.json"
        manifest = {"schema": "feedbackbench.action_trained_manifest.v1", "protocol_sha256": self.fingerprint, "runs": completed}
        if manifest_path.exists():
            if read(manifest_path) != manifest:
                raise ValueError("Frozen trained manifest differs from verified final checkpoints")
        else:
            write_json(manifest_path, manifest)
        manifest_sha = digest(manifest_path)
        evaluated = 0
        for case in self.protocol["evaluation"]["cases"]:
            case_dir = self.campaign / "evaluation" / case["case_id"]
            refuse_unsealed_complete(case_dir)
            binding = {"protocol_sha256": self.fingerprint, "trained_manifest_sha256": manifest_sha, "case_id": case["case_id"]}
            sealed = list(case_dir.glob("attempt_*/receipt.json"))
            if len(sealed) > 1:
                raise ValueError("Duplicate completed evaluation case")
            if sealed:
                verify_seal(sealed[0].parent, binding)
                evaluated += 1
                continue
            while True:
                if not self.ready():
                    return
                attempt_number = len(list(case_dir.glob("attempt_*.launch"))) + 1
                if attempt_number > 3:
                    raise RuntimeError(f"Three preserved attempts exhausted for {case['case_id']}")
                output = case_dir / f"attempt_{attempt_number:03d}"
                command = [sys.executable, "-B", str(ROOT / "scripts/evaluate_action_adaptation.py"), *common,
                           "--trained-manifest", str(manifest_path), "--trained-manifest-sha256", manifest_sha,
                           "--case-id", case["case_id"], "--output", str(output)]
                code = self.child(command, output, self.protocol["resources"]["evaluation_max_seconds"])
                if (output / "receipt.json").exists():
                    verify_seal(output, binding)
                    evaluated += 1
                    self.save(evaluation_completed=evaluated)
                    break
                refuse_unsealed_complete(case_dir)
                if deadline_reached() or code == 3:
                    self.save(state="partial_deadline" if deadline_reached() else "needs_attention", reason="evaluation_time_limit")
                    return
                if code != 75:
                    raise RuntimeError(f"Evaluation failed ({code}); diagnosis required")
                self.save(state="waiting_resources", reason="preserved_resource_interruption")
                time.sleep(300)
        self.save(state="complete", evaluation_completed=evaluated, current=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "verify", "supervise"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Archive only an audited dead owner lock")
    args = parser.parse_args(argv)
    campaign = args.campaign.resolve()
    if not campaign.is_relative_to(AREA.resolve()) or campaign == AREA.resolve():
        parser.error("Campaign must be below outputs/action_adaptation")
    if args.command == "status":
        print(json.dumps(status(campaign), indent=2))
        return 0
    fingerprint = (campaign / "protocol.sha256").read_text(encoding="utf-8").strip()
    protocol = verify_protocol(campaign / "protocol.json", fingerprint, inputs=True)
    if args.command == "verify":
        print(json.dumps({"verified": True, "protocol_sha256": fingerprint}))
        return 0
    if deadline_reached():
        print(json.dumps({"state": "deadline_reached", "launch_permitted": False}))
        return 3
    lock = AREA / "worker.lock"
    previous = status(campaign)
    if previous["live_owned_processes"]:
        raise RuntimeError("An owned process is already alive; duplicate launch refused")
    if previous["state"] == "complete":
        print(json.dumps(previous))
        return 0
    if lock.exists():
        record = read(lock)
        if not args.resume or record.get("campaign") != str(campaign.relative_to(ROOT)) or matching(record["owner"]) is not None:
            raise RuntimeError("Existing global worker lock requires identity audit; no worker launched")
        archive = campaign / "lock_history"
        archive.mkdir(exist_ok=True)
        lock.rename(archive / (uuid4().hex + ".json"))
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"owner": identity(os.getpid()), "campaign": str(campaign.relative_to(ROOT))}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    supervisor = Supervisor(campaign, fingerprint, protocol)
    try:
        supervisor.execute()
        from report_action_campaign import build_report
        write_json(campaign / "report.json", build_report(campaign))
    except BaseException as error:
        supervisor.state.update(state="needs_attention", error=repr(error), ended_at=now())
        try:
            supervisor.save()
        except BaseException as persistence_error:
            with (campaign / ("persistence_failure_" + uuid4().hex + ".json")).open("x", encoding="utf-8") as stream:
                json.dump({"state": supervisor.state, "persistence_error": repr(persistence_error)}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        raise
    print(json.dumps(supervisor.state, indent=2))
    return 0 if supervisor.state["state"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
