"""Bounded, single-worker supervisor for development-only action-training probes.

This does not resume the completed budget campaign or change any other process.
Every invocation preserves a separate launch directory, source copy and outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import psutil

from budget_campaign_common_v5 import write_json

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "outputs/action_adaptation"
DEADLINE = datetime(2026, 10, 30, 6, 8, tzinfo=timezone.utc)
MIN_START_GIB = 4.25
MIN_RUNNING_GIB = 0.75
MIN_DISK_GIB = 4.0
MAX_SECONDS = 1200
GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(pid):
    return {"pid": pid, "created": psutil.Process(pid).create_time()}


def matching(record):
    try:
        process = psutil.Process(record["pid"])
        return process if process.create_time() == record["created"] and process.is_running() else None
    except psutil.NoSuchProcess:
        return None


def collect_owned(child, records):
    queue = [child, *records]
    visited = set()
    while queue:
        root = queue.pop(0)
        key = (root["pid"], root["created"])
        if key in visited:
            continue
        visited.add(key)
        parent = matching(root)
        if parent is None:
            continue
        try:
            children = parent.children(recursive=False)
        except psutil.NoSuchProcess:
            continue
        for process in children:
            try:
                record = {"pid": process.pid, "created": process.create_time()}
                if not process.is_running() or process.ppid() != parent.pid or matching(record) is None:
                    continue
            except psutil.NoSuchProcess:
                continue
            if record not in records:
                records.append(record)
            queue.append(record)


def stop_owned(records):
    """Use both PID and creation time; never target a process by name."""
    stopped = []
    processes = []
    for record in reversed(records):
        process = matching(record)
        if process is not None:
            try:
                process.terminate()
                processes.append(process)
                stopped.append(record)
            except psutil.NoSuchProcess:
                pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        record = {"pid": process.pid, "created": process.create_time()}
        if record in records and matching(record) is not None:
            process.kill()
    psutil.wait_procs(alive, timeout=5)
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    example = args.example.resolve()
    if not output.is_relative_to(AREA.resolve()) or output == AREA.resolve():
        parser.error("output must be a new child of outputs/action_adaptation")
    if output.exists() or not example.is_file():
        parser.error("output already exists or example does not exist")
    launch = output.with_name(output.name + ".launch")
    launch.mkdir(parents=True, exist_ok=False)
    state = {"schema": "feedbackbench.action_probe_launch.v1", "state": "preflight",
             "started_at": now(), "owner": identity(os.getpid()), "child": None,
             "owned_children": [], "output": str(output.relative_to(ROOT)),
             "example": str(example), "example_sha256": digest(example),
             "deadline_utc": DEADLINE.isoformat(), "start_memory_gib": MIN_START_GIB,
             "runtime_memory_gib": MIN_RUNNING_GIB, "max_seconds": MAX_SECONDS,
             "development_only": True, "closed_loop_evaluation": False}
    state_path = launch / "status.json"
    write_json(state_path, state)
    lock = AREA / "worker.lock"
    process = None
    lock_acquired = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"owner": state["owner"], "launch": str(launch.relative_to(ROOT))}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        lock_acquired = True
        available = psutil.virtual_memory().available / GIB
        state["available_before_start_gib"] = available
        if datetime.now(timezone.utc) >= DEADLINE:
            state["state"] = "deadline_reached"
            return 3
        if available < MIN_START_GIB or shutil.disk_usage(ROOT).free / GIB < MIN_DISK_GIB:
            state["state"] = "blocked_preflight"
            return 3
        sources = ["scripts/run_action_adaptation_probe.py", "scripts/probe_action_adaptation.py",
                   "scripts/budget_campaign_common.py", "scripts/budget_campaign_common_v5.py",
                   "src/embodied_research/action_adaptation.py", "src/embodied_research/policies.py",
                   "src/embodied_research/action_demo_data.py", "scripts/prepare_action_adaptation_data.py",
                   "plans/action_adaptation_probe_v1.json",
                   "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
                   "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py"]
        state["source_files"] = {}
        for name in sources:
            source = ROOT / name
            destination = launch / "source_snapshot" / Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            state["source_files"][name] = digest(destination)
        state["source_revision_sha256"] = hashlib.sha256(
            json.dumps(state["source_files"], sort_keys=True).encode()).hexdigest()
        state["example_metadata_sha256"] = digest(example.with_name("metadata.json"))
        command = [sys.executable, "-B", str(ROOT / "scripts/probe_action_adaptation.py"),
                   "--example", str(example), "--output", str(output), "--mode", "lora",
                   "--rank", "8", "--steps", "3"]
        state["command"] = command
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           WANDB_MODE="disabled", HF_HUB_DISABLE_TELEMETRY="1")
        write_json(state_path, state)
        if any(digest(ROOT / name) != expected for name, expected in state["source_files"].items()):
            raise RuntimeError("Live probe source changed after snapshot; no worker launched")
        with (launch / "worker.stdout.log").open("x", encoding="utf-8") as stdout, \
                (launch / "worker.stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            state["child"] = identity(process.pid)
            state["owned_children"].append(state["child"])
            state.update(state="running", worker_started_at=now())
            write_json(state_path, state)
            started = time.monotonic()
            last_write = started
            with (launch / "resources.jsonl").open("x", encoding="utf-8") as samples:
                while process.poll() is None:
                    collect_owned(state["child"], state["owned_children"])
                    free = psutil.virtual_memory().available / GIB
                    samples.write(json.dumps({"at": now(), "available_gib": free,
                                              "owned_children": state["owned_children"]}) + "\n")
                    samples.flush()
                    reason = ("deadline_reached" if datetime.now(timezone.utc) >= DEADLINE else
                              "time_limit" if time.monotonic() - started > MAX_SECONDS else
                              "memory_reserve" if free < MIN_RUNNING_GIB else None)
                    if reason:
                        state.update(state="interrupted", reason=reason)
                        break
                    if time.monotonic() - last_write >= 10:
                        write_json(state_path, state)
                        last_write = time.monotonic()
                    time.sleep(.5)
            if process.poll() is None:
                state["terminated_owned"] = stop_owned(state["owned_children"])
            state["worker_exit_code"] = process.wait(timeout=10)
            if state["state"] == "running":
                state["state"] = "worker_exited" if process.returncode == 0 else "needs_attention"
            # A zero process exit is not automatically a successful training probe.
            if (output / "report.json").is_file():
                state["worker_report_sha256"] = digest(output / "report.json")
    except BaseException as error:
        state.update(state="needs_attention", error=f"{type(error).__name__}: {error}")
        if process is not None:
            if state["child"] is None:
                # A freshly launched child still has this Popen handle; record it before cleanup.
                try:
                    state["child"] = identity(process.pid)
                    state["owned_children"].append(state["child"])
                except psutil.NoSuchProcess:
                    pass
            if state["child"]:
                collect_owned(state["child"], state["owned_children"])
            state["terminated_owned"] = stop_owned(state["owned_children"])
            process.wait(timeout=10)
        raise
    finally:
        state["ended_at"] = now()
        if any(matching(r) is not None for r in state["owned_children"]):
            state["final_cleanup"] = stop_owned(state["owned_children"])
            state.update(state="needs_attention", reason="owned_child_remaining_after_worker")
        state["live_owned_children"] = [r for r in state["owned_children"] if matching(r) is not None]
        state["lock_acquired"] = lock_acquired
        try:
            write_json(state_path, state)
        except BaseException as error:
            state.update(state="needs_attention", persistence_error=repr(error))
            try:
                with (launch / ("persistence_failure_" + uuid4().hex + ".json")).open("x", encoding="utf-8") as stream:
                    json.dump(state, stream, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException as fallback_error:
                print(json.dumps({"fallback_error": repr(fallback_error), "state": state}), file=sys.stderr)
            raise
        # Keep the lock as audit. It must be archived only after owner/children are verified dead.
        print(json.dumps(state, ensure_ascii=True))
    return 0 if state["state"] == "worker_exited" else 2


if __name__ == "__main__":
    raise SystemExit(main())
