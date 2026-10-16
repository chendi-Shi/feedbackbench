"""Finish the frozen four-episode stage in the foreground, with bounded memory waiting.

This entry point does not import a model or change the frozen experiment. It
never restarts partial episodes, removes locks, or signals another process.
Each invocation writes new operator logs under .runtime/closed_loop_completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/frontier/vlajepa_closed_loop"
RUNNER = "scripts/run_vlajepa_closed_loop.py"
LAUNCHER = "scripts/run.py"
REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


class StatusWriteError(RuntimeError):
    """The normal operator status could not be persisted; its old value is stale."""


def replace_with_retry(source, destination, *, replace=None, sleep=None):
    """Retry only short Windows sharing/access denials; preserve other failures."""
    replace = os.replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(len(REPLACE_DELAYS) + 1):
        try:
            replace(source, destination)
            return
        except PermissionError as error:
            sharing_denial = getattr(error, "winerror", None) in {5, 32, 33} or error.errno == errno.EACCES
            if not sharing_denial or attempt == len(REPLACE_DELAYS):
                raise
            sleep(REPLACE_DELAYS[attempt])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_file(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and not Path(name).drive and "\\" not in name and ":" not in name
            and ".." not in Path(name).parts, "Unsafe frozen evidence path")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Frozen evidence file is missing")
    return path


def read_protocol(output, root):
    protocol = read_json(output / "protocol.json")
    require(protocol.get("schema") == "feedbackbench.vlajepa_closed_loop_protocol.v1", "Unexpected protocol schema")
    require(digest_json({k: v for k, v in protocol.items() if k != "fingerprint"}) == protocol["fingerprint"],
            "Frozen protocol fingerprint changed")
    sources = protocol["sources"]
    require({RUNNER, LAUNCHER} <= set(sources), "Frozen execution sources are missing")
    for name, checksum in sources.items():
        require(file_sha(safe_file(root, name)) == checksum, "Frozen source hash differs: " + name)
    require(file_sha(output / "inputs.json") == protocol["inputs_sha256"], "Frozen input snapshot changed")
    cases = protocol["spec"]["cases"]
    require(len(cases) == len({case["id"] for case in cases}) == 4, "Exactly four frozen cases are required")
    for case in cases:
        name = case["id"]
        require(isinstance(name, str) and name and name not in {".", ".."}
                and not any(character in name for character in "/\\:"), "Unsafe case identity")
    minimum_gib = protocol["spec"]["memory_guard"]["minimum_available_gib"]
    require(type(minimum_gib) in (int, float) and minimum_gib == 5.0, "Frozen 5 GiB gate changed")
    return protocol, int(minimum_gib * 1024**3)


def episode_states(output, protocol):
    """Triage only; the frozen runner owns the full seal and scientific validation."""
    cases = protocol["spec"]["cases"]
    episodes = output / "episodes"
    require(all(path.name in {case["id"] for case in cases} for path in episodes.glob("*")), "Unplanned episode entry")
    states = {}
    for case in cases:
        directory = episodes / case["id"]
        if not directory.exists():
            states[case["id"]] = "missing"
            continue
        require(directory.is_dir() and not directory.is_symlink(), "Invalid episode directory")
        if (directory / "error.json").exists():
            states[case["id"]] = "error"
            continue
        if not all((directory / name).is_file() for name in ("result.json", "receipt.json")):
            states[case["id"]] = "unsealed"
            continue
        states[case["id"]] = "sealed"
    return states


def available_memory():
    import psutil

    return psutil.virtual_memory().available


def complete_stage(output=DEFAULT_OUTPUT, *, root=ROOT, wait_seconds=0, poll_seconds=30,
                   memory=None, monotonic=None, sleep=None, subprocess_run=None):
    """Return a truthful outcome; only a fully verified four-case stage has exit 0."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "Keep stage evidence inside this project")
    require(type(wait_seconds) in (int, float) and math.isfinite(wait_seconds) and wait_seconds >= 0,
            "wait-seconds must be finite and nonnegative")
    require(type(poll_seconds) in (int, float) and math.isfinite(poll_seconds) and 0 < poll_seconds <= 60,
            "poll-seconds must be greater than zero and at most 60")
    memory = available_memory if memory is None else memory
    monotonic = time.monotonic if monotonic is None else monotonic
    sleep = time.sleep if sleep is None else sleep
    subprocess_run = subprocess.run if subprocess_run is None else subprocess_run
    directory = root / ".runtime/closed_loop_completion" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema": "feedbackbench.closed_loop_completion.v1", "status": "checking",
              "started_at": utc_now(), "output": output.relative_to(root).as_posix(),
              "operator_record": (directory / "status.json").relative_to(root).as_posix(),
              "wait_seconds": wait_seconds, "poll_seconds": poll_seconds, "children": [], "memory_samples": []}

    def save():
        temporary = directory / "status.json.tmp"
        try:
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            replace_with_retry(temporary, directory / "status.json")
        except OSError as error:
            raise StatusWriteError(f"{type(error).__name__}: {error}") from error

    def finish(status, code):
        record.update(status=status, exit_code=code, finished_at=utc_now())
        save()
        return record

    def status_failure(error):
        # Never retry the same failing status path from the exception handler.
        # A new exclusive file preserves the old status and pending .tmp bytes.
        failure_path = directory / ("terminal_failure_" + uuid.uuid4().hex + ".json")
        record.update(failed_phase=record["status"], status="status_record_write_failed", exit_code=1,
                      finished_at=utc_now(), status_write_error=str(error), primary_status_is_stale=True,
                      previous_operator_record=record["operator_record"], terminal_record_persisted=True)
        record["operator_record"] = failure_path.relative_to(root).as_posix()
        try:
            with failure_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        except OSError as fallback_error:
            # Even a fully unavailable filesystem still yields a nonzero JSON
            # outcome on stdout; do not claim the fallback file was saved.
            record.update(terminal_record_persisted=False, fallback_write_error_type=type(fallback_error).__name__)
            record["operator_record"] = None
        return record

    def child(name, flags):
        command = [sys.executable, str(root / LAUNCHER), str(root / RUNNER), "--output", str(output), *flags]
        stdout_path, stderr_path = directory / (name + ".stdout.log"), directory / (name + ".stderr.log")
        child_record = {"name": name, "started_at": utc_now(), "flags": flags,
                        "stdout": stdout_path.relative_to(root).as_posix(), "stderr": stderr_path.relative_to(root).as_posix()}
        record["children"].append(child_record)
        save()
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess_run(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, check=False)
        code = result.returncode
        require(type(code) is int, "Invalid child exit code")
        child_record.update(returncode=code, finished_at=utc_now(),
                            stdout_sha256=file_sha(stdout_path), stderr_sha256=file_sha(stderr_path))
        save()
        return code, stdout_path

    try:
        save()
        protocol, minimum_bytes = read_protocol(output, root)
        record.update(protocol_fingerprint=protocol["fingerprint"], protocol_sha256=file_sha(output / "protocol.json"),
                      minimum_available_bytes=minimum_bytes, planned=4)
        states = episode_states(output, protocol)
        record["initial_episode_states"] = states
        if (output / "active.lock").exists():
            return finish("active_or_stale_lock_requires_review", 4)
        if any(state in {"error", "unsealed"} for state in states.values()):
            return finish("partial_or_error_requires_review", 4)
        run_code = 0
        if any(state == "missing" for state in states.values()):
            start = monotonic()
            while True:
                elapsed = monotonic() - start
                available = memory()
                require(type(available) is int and available >= 0, "Invalid available-memory reading")
                record["memory_samples"].append({"utc": utc_now(), "elapsed_seconds": elapsed, "available_bytes": available})
                save()
                if available >= minimum_bytes:
                    break
                if elapsed >= wait_seconds:
                    return finish("blocked_memory_wait_expired", 3)
                record["status"] = "waiting_for_memory"
                save()
                sleep(min(poll_seconds, wait_seconds - elapsed))
            current, _ = read_protocol(output, root)
            require(current["fingerprint"] == protocol["fingerprint"], "Protocol changed while waiting")
            require(episode_states(output, protocol) == states and not (output / "active.lock").exists(),
                    "Episode state or lock changed while waiting; no competing runner was launched")
            record["status"] = "running_frozen_stage"
            save()
            run_code, _ = child("run", [])

        # Run both consumers even after a nonzero runner exit; partial evidence
        # remains visible, but neither a report nor exit 0 alone proves completion.
        record["status"] = "checking_final_evidence"
        save()
        report_code, _ = child("report", ["--report"])
        validate_code, validation_path = child("validate", ["--validate-only"])
        record.update(runner_returncode=run_code, report_returncode=report_code, validation_returncode=validate_code)
        for name, code in (("runner", run_code), ("report", report_code), ("validation", validate_code)):
            if code != 0:
                return finish(name + "_failed_or_incomplete", code if code > 0 else 1)
        current, _ = read_protocol(output, root)
        require(current["fingerprint"] == protocol["fingerprint"], "Protocol changed during execution")
        states = episode_states(output, protocol)
        require(all(value == "sealed" for value in states.values()), "All four episodes must be sealed")
        require(not (output / "active.lock").exists(), "Runner did not release its lock")
        validation = read_json(validation_path)
        report = read_json(output / "report.json")
        for outcome in (validation, report):
            require(outcome.get("complete") is True and outcome.get("planned") == 4 and outcome.get("completed") == 4,
                    "Consumer did not confirm all four complete episodes")
            require(outcome.get("protocol_fingerprint") == protocol["fingerprint"], "Consumer protocol identity differs")
            require(outcome.get("statuses") == {case["id"]: "complete" for case in protocol["spec"]["cases"]},
                    "Consumer case statuses differ")
        record.update(completed=4, final_episode_states=states, report_sha256=file_sha(output / "report.json"),
                      successes=validation["successes"], task_failures=validation["task_failures"])
        return finish("complete_verified", 0)
    except Exception as error:
        # Detailed child output stays in this invocation's private log directory.
        record.update(error_type=type(error).__name__, error=str(error))
        if isinstance(error, StatusWriteError):
            return status_failure(error)
        try:
            return finish("error_not_complete", 1)
        except StatusWriteError as status_error:
            return status_failure(status_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    result = complete_stage(args.output, wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds)
    keys = ("status", "exit_code", "operator_record", "primary_status_is_stale", "terminal_record_persisted")
    print(json.dumps({key: result[key] for key in keys if key in result}, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
