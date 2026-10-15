"""One guarded stop attempt between delay and SmolVLA; never starts a new process.

The caller must supply the target's previously observed creation timestamp and
retain the successful JSON receipt. A saved resume_audit.json proves the stop
gate passed, not that termination succeeded. This helper never rewrites or
deletes existing files. Failed attempts retain any partial checkpoint for audit.
Only the selected process is suspended/terminated; no process-tree operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STAGES = (("coverage", "10_coverage", 40), ("development", "20_development", 32),
          ("methods", "30_methods", 128), ("delay", "40_delay", 32))
FROZEN_SOURCES = (
    "scripts/run_campaign.py", "scripts/make_plans.py",
    "src/embodied_research/selection.py", "src/embodied_research/experiment.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
    "src/embodied_research/simulator.py", "src/embodied_research/scheduling.py",
    "src/embodied_research/statistics.py",
)
CHECKPOINT = "outputs/checkpoints/before_smol"
AUDIT = "outputs/campaign/resume_audit.json"
SNAPSHOT_README = (
    "# Before-SmolVLA checkpoint\n\n"
    "These files preserve the original bytes of campaign metadata and reports before a process restart. "
    "The HTML files are historical snapshots, not standalone report entry points: their relative links "
    "retain the meaning they had in the original campaign directories. Open outputs/campaign/report.html "
    "at the project root to view the current complete report.\n\n"
    "runtime_audit.json keeps its original filename so packaging can redact local paths only in diagnostic "
    "archive copies. Original local snapshots remain unchanged; the bundle manifest records both hashes.\n"
).encode()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external checkpoint input: {relative}")
    return path


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in checkpoint input")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON value in checkpoint input")

    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def validate_identity(process, pid, create_time, root):
    """Require the actual campaign interpreter, never its launcher or shell."""
    if (process.pid != pid or not process.is_running()
            or abs(process.create_time() - create_time) > 1e-6):
        raise ValueError("Target PID/creation time differs from the selected process")
    if Path(process.cwd()).resolve() != root:
        raise ValueError("Target working directory is not this project")
    command = tuple(process.cmdline())
    if len(command) < 2 or Path(command[0]).name.lower() not in {"python", "python3", "python.exe", "python3.exe"}:
        raise ValueError("Target is not a Python campaign interpreter")
    arguments = list(command[1:])
    while arguments and arguments[0] in {"-u", "-B"}:
        arguments.pop(0)
    if not arguments or (root / arguments.pop(0)).resolve() != root / "scripts/run_campaign.py":
        raise ValueError("Target entrypoint is not this project's run_campaign.py")
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--output", default="outputs/campaign")
    parser.add_argument("--through", default="flow_steps")
    options, unknown = parser.parse_known_args(arguments)
    if unknown or options.through != "flow_steps" or (root / options.output).resolve() != root / "outputs/campaign":
        raise ValueError("Target campaign arguments are not the expected complete campaign")
    return command  # Retained in memory for equality checks; never written to audit.


def inspect_checkpoint(root):
    """Read-only gate and hashes, called only while the target is suspended."""
    campaign = root / "outputs/campaign"
    if any(campaign.rglob("*.tmp")):
        raise ValueError("Campaign has an unfinished temporary write")
    if (campaign / "60_flow_steps").exists():
        raise ValueError("Flow stage directory already exists")
    for stage in ("50_smolvla", "60_flow_steps"):
        directory = campaign / stage
        if directory.exists():
            # A completed protocol may precede the first progress write. Every
            # other entry is conservatively treated as a started attempt.
            if any(path.name != "protocol.json" or not path.is_file() for path in directory.iterdir()):
                raise ValueError(f"Later stage has attempt evidence: {stage}")
    status = _read(_file(root, "outputs/campaign/campaign_status.json"))
    if status.get("status") != "running" or status.get("through") != "flow_steps" or status.get("error"):
        raise ValueError("Campaign is not the expected running full campaign")
    if "flow_steps" in status.get("stages", {}):
        raise ValueError("Flow stage has already started")
    later_status = status.get("stages", {}).get("smolvla")
    if later_status is not None and later_status != {"status": "running"}:
        raise ValueError("SmolVLA status is inconsistent with an unstarted attempt")
    freeze = _read(_file(root, "outputs/campaign/campaign_freeze.json"))
    source_hashes = freeze.get("source_sha256", {})
    if set(source_hashes) != set(FROZEN_SOURCES):
        raise ValueError("Campaign source freeze does not contain the expected nine files")
    for name, expected in source_hashes.items():
        if _hash(_file(root, name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    raw_hashes, stage_inputs, protocols = {}, {}, {}
    snapshot_paths = set()
    for name in ("campaign_status.json", "campaign_freeze.json", "development_selection.json",
                 "report.html", "delay_comparison.json", "delay_comparison.html"):
        snapshot_paths.add(_file(root, f"outputs/campaign/{name}"))
    if (campaign / "50_smolvla/protocol.json").is_file():
        snapshot_paths.add(_file(root, "outputs/campaign/50_smolvla/protocol.json"))
    for directory in (campaign / "frozen_plans", campaign / "figures"):
        snapshot_paths.update(path for path in directory.rglob("*") if path.is_file())
    for name, stem, count in STAGES:
        directory = campaign / stem
        stage_status = status.get("stages", {}).get(name, {})
        if (stage_status.get("status") != "complete" or stage_status.get("completed") != count
                or stage_status.get("total") != count):
            raise ValueError(f"Stage status is not complete: {name}")
        progress = _read(_file(root, f"outputs/campaign/{stem}/progress.json"))
        if progress.get("finished") is not True or progress.get("completed") != count or progress.get("total") != count:
            raise ValueError(f"Stage progress is not complete: {name}")
        protocol_path = _file(root, f"outputs/campaign/{stem}/protocol.json")
        protocol = _read(protocol_path)
        fingerprint = _digest({key: protocol[key] for key in ("plan", "sources", "runtime_versions", "artifacts")})
        if protocol.get("fingerprint") != fingerprint:
            raise ValueError(f"Stage protocol fingerprint is invalid: {name}")
        if protocol["plan"] != _read(_file(root, f"outputs/campaign/frozen_plans/{stem}.json")):
            raise ValueError(f"Stage plan differs from freeze: {name}")
        for filename, checksum in protocol["sources"].items():
            if source_hashes.get(f"src/embodied_research/{filename}") != checksum:
                raise ValueError(f"Stage source differs from freeze: {name}")
        cases = protocol["cases"]
        identifiers = [case["id"] for case in cases]
        if (len(cases) != count or len(set(identifiers)) != count
                or any(not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) for identifier in identifiers)):
            raise ValueError(f"Stage cases do not match the expected denominator: {name}")
        if {path.name for path in (directory / "episodes").iterdir()} != set(identifiers):
            raise ValueError(f"Unexpected or missing episode directories: {name}")
        inputs = {"protocol.json": _hash(protocol_path)}
        for case in cases:
            relative = f"outputs/campaign/{stem}/episodes/{case['id']}"
            result_path = _file(root, f"{relative}/result.json")
            result = _read(result_path)
            if (result.get("status") != "complete" or result.get("protocol_fingerprint") != fingerprint
                    or any(result.get(key) != value for key, value in case.items())):
                raise ValueError(f"Episode is incomplete or differs from its protocol: {case['id']}")
            required = ["result.json", "trajectory.npz", "initial.png", "initial_wrist.png", "final.png"]
            if case.get("record_video"):
                if result.get("video") != "rollout.gif":
                    raise ValueError("Recorded video is missing from episode result")
                required.append("rollout.gif")
            for filename in required:
                if _file(root, f"{relative}/{filename}").stat().st_size < 1:
                    raise ValueError("Empty raw episode evidence")
            for path in result_path.parent.rglob("*"):
                if path.is_file():
                    relative_file = path.relative_to(root).as_posix()
                    raw_hashes[relative_file] = _hash(_file(root, relative_file))
            inputs[f"episodes/{case['id']}/result.json"] = raw_hashes[f"{relative}/result.json"]
        summary = _read(_file(root, f"outputs/campaign/{stem}/summary.json"))
        if summary.get("partial") is not False or summary.get("input_sha256") != inputs:
            raise ValueError(f"Stage report is partial or stale: {name}")
        for filename in ("runtime_audit.json", "report.html"):
            _file(root, f"outputs/campaign/{stem}/{filename}")
        snapshot_paths.update(path for path in directory.rglob("*")
                              if path.is_file() and "episodes" not in path.relative_to(directory).parts)
        stage_inputs[stem] = inputs
        protocols[stem] = fingerprint
    comparison = _read(campaign / "delay_comparison.json")
    if (comparison.get("baseline_input_sha256") != stage_inputs["30_methods"]
            or comparison.get("delayed_input_sha256") != stage_inputs["40_delay"]):
        raise ValueError("Delay comparison is stale")
    selection = _read(campaign / "development_selection.json")
    if (selection.get("development_protocol_fingerprint") != protocols["20_development"]
            or selection.get("development_inputs_sha256") != _digest(stage_inputs["20_development"])):
        raise ValueError("Development selection differs from completed evidence")
    snapshot_hashes = {}
    for path in snapshot_paths:
        relative = path.relative_to(root).as_posix()
        snapshot_hashes[relative] = _hash(_file(root, relative))
    return {"episode_counts": {stem: count for _, stem, count in STAGES},
            "raw_episode_sha256": dict(sorted(raw_hashes.items())),
            "snapshot_source_sha256": dict(sorted(snapshot_hashes.items())),
            "frozen_source_sha256": source_hashes, "protocol_fingerprints": protocols}


def _write_new(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise OSError("Checkpoint write verification failed")


def _resume_original(process, pid, create_time):
    # psutil also protects against PID reuse; never intentionally resume a new
    # process after the selected one has exited.
    if process.is_running() and process.pid == pid and abs(process.create_time() - create_time) <= 1e-6:
        process.resume()


def checkpoint_and_stop(root, pid, create_time, *, process_factory=None, wait_seconds=30):
    root = Path(root).resolve()
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not math.isfinite(create_time):
        raise ValueError("A valid PID and observed creation timestamp are required")
    if (root / AUDIT).exists() or (root / CHECKPOINT).exists():
        raise FileExistsError("Existing audit/checkpoint must be preserved; refusing another attempt")
    if any(not (root / name).resolve().is_relative_to(root) for name in (AUDIT, CHECKPOINT)):
        raise ValueError("Audit/checkpoint destinations must remain inside this project")
    if process_factory is None:
        if os.name != "nt":
            raise RuntimeError("Live process control is restricted to this Windows workflow")
        import psutil

        process_factory = psutil.Process
    process = process_factory(pid)
    original_command = validate_identity(process, pid, create_time, root)
    if process.status() in {"stopped", "tracing-stop", "zombie", "dead"}:
        raise ValueError("Target is not an actively running campaign")
    suspension_attempted = False
    try:
        suspension_attempted = True
        process.suspend()
        if validate_identity(process, pid, create_time, root) != original_command:
            raise ValueError("Process identity changed while suspending")
        evidence = inspect_checkpoint(root)
        checkpoint = root / CHECKPOINT
        checkpoint.mkdir(parents=True, exist_ok=False)
        for relative, checksum in evidence["snapshot_source_sha256"].items():
            raw = _file(root, relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError("Source changed while copying checkpoint")
            destination = checkpoint / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, raw)
        _write_new(checkpoint / "README.md", SNAPSHOT_README)
        audit = {
            "schema_version": 1, "kind": "verified pre-SmolVLA process checkpoint",
            "verified_at": _utc(), "reason": "Release long-lived process memory before the first SmolVLA attempt",
            "process": {"pid": pid, "create_time": create_time, "entrypoint": "scripts/run_campaign.py"},
            "gate": "Target suspended; all four stages complete; no SmolVLA or flow attempt files",
            "scope": "Operational restart only; no episode is removed, retried, or added to the denominator",
            "absence_assumption": "Frozen runner never deletes progress; external historical deletion cannot be disproved by a single check",
            "checkpoint_directory": CHECKPOINT,
            "checkpoint_readme_sha256": hashlib.sha256(SNAPSHOT_README).hexdigest(),
            "snapshot_path_rule": "Append each snapshot_source_sha256 key to checkpoint_directory",
            "stop_confirmation": "This audit authorizes a safe stop; retain the helper's successful stdout receipt to confirm process exit",
            "restart_commands": ["python scripts/run.py scripts/run_campaign.py --output outputs/campaign"],
            "helper_sha256": _hash(Path(__file__)), **evidence,
        }
        audit_bytes = (json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        _write_new(root / AUDIT, audit_bytes)
        if inspect_checkpoint(root) != evidence:
            raise ValueError("Evidence changed while target was suspended; refuse to stop")
        if validate_identity(process, pid, create_time, root) != original_command:
            raise ValueError("Process identity changed before stopping")
        # No further writes follow termination: a disk failure must never occur
        # after stopping and be misrepresented as a safely resumed process.
        process.terminate()
        exit_code = process.wait(timeout=wait_seconds)
        if process.is_running():
            raise RuntimeError("Target did not exit after termination")
        return {"status": "stopped", "exited_at": _utc(), "pid": pid, "create_time": create_time,
                "exit_code": exit_code, "audit": AUDIT, "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "checkpoint_directory": CHECKPOINT}
    except BaseException:
        if suspension_attempted:
            _resume_original(process, pid, create_time)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--create-time", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(checkpoint_and_stop(ROOT, args.pid, args.create_time), indent=2))


if __name__ == "__main__":
    main()
