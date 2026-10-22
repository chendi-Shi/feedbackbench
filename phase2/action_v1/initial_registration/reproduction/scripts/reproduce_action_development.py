"""Public, explicit local reconstruction of the excluded action-development gates.

Registration writes only new owned directories. Training and evaluation are
separate explicit commands, use one identity-audited CPU worker, and retain all
failed attempts. This version cannot extend the original 2026-10-30 deadline.
Public evidence projections are not accepted as raw training/checkpoint gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

from action_campaign_common import ROOT, DEADLINE, deadline_reached, digest, now, read, resolve, verify_protocol, verify_seal, write_json
import evaluate_action_development_public as evaluation


AREA = ROOT / "outputs/action_adaptation"
PUBLIC_SOURCE_VERSION = "action_development_public_v1"
ALIGNMENT_SOURCES = {"scripts/validate_action_demo_alignment.py", "scripts/validate_action_demo_alignment_r2.py",
                     "scripts/validate_action_demo_alignment_v2.py", "scripts/combine_action_demo_alignment.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def before_deadline():
    require(not deadline_reached(), "This reproduction version expired at 2026-10-30T06:08:00Z. "
            "Register and review a separate new protocol/version; this tool never extends or rewrites the original deadline.")


def campaign_path(name):
    require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name), "Use a new short lowercase campaign identifier")
    root, area = ROOT.resolve(), AREA.resolve()
    require(area != root and area.is_relative_to(root), "Campaign area escapes the project tree")
    campaign = resolve(name, area)
    require(campaign.is_relative_to(root) and campaign.is_relative_to(area), "Campaign path escapes its owned area")
    return campaign


def record(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Inputs must be local to this project tree")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def model_inventory():
    result = {}
    for name in ("smolvla", "smolvlm_assets"):
        path = ROOT / "models" / name / "download_manifest.json"
        result[path.relative_to(ROOT).as_posix()] = digest(path)
        for item in read(path)["files"]:
            target = resolve(item["file"], path.parent)
            require(digest(target) == item["sha256"], "Pinned local model/preprocessor bytes differ")
            result[target.relative_to(ROOT).as_posix()] = item["sha256"]
    return result


def freeze(campaign, protocol, source_names):
    """No source/result overwrite: each invocation needs a new campaign name."""
    before_deadline()
    require(campaign.is_dir() and not (campaign / "protocol.json").exists(), "Protocol already exists")
    snapshot = campaign / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    hashes = {name: digest(resolve(name)) for name in sorted(source_names)}
    index = {}
    for number, name in enumerate(hashes):
        target = f"{number:03d}_{Path(name).name}"
        shutil.copyfile(resolve(name), snapshot / target)
        require(digest(snapshot / target) == hashes[name], "Source changed while copying")
        index[name] = target
    write_json(snapshot / "index.json", index)
    protocol.update(schema="feedbackbench.action_adaptation_train.v1", frozen=True, frozen_at=now(),
                    deadline_utc=DEADLINE, development_only=True, excluded_from_formal_study=True, study_result=False,
                    public_source_version=PUBLIC_SOURCE_VERSION, source_files=hashes,
                    source_snapshot_index=record(snapshot / "index.json"))
    before_deadline()
    write_json(campaign / "protocol.json", protocol)
    fingerprint = digest(campaign / "protocol.json")
    with (campaign / "protocol.sha256").open("x", encoding="utf-8") as stream:
        stream.write(fingerprint + "\n")
    verify_protocol(campaign / "protocol.json", fingerprint)
    return fingerprint


def inspect_alignment(manifest_path, alignment_path):
    """Require newly local raw r3 evidence and both preceding diagnostic receipts."""
    manifest_path, alignment_path = Path(manifest_path).resolve(), Path(alignment_path).resolve()
    manifest, alignment = read(manifest_path), read(alignment_path)
    require(alignment.get("status") == "passed" and alignment.get("diagnostic_revision") == 3
            and alignment.get("scope") == "observation_action_alignment"
            and alignment.get("dynamics_replay_verified") is False
            and alignment.get("source_manifest_sha256") == digest(manifest_path), "Complete raw r3 alignment is required")
    expected_script = digest(ROOT / "scripts/validate_action_demo_alignment_v2.py")
    require(alignment.get("script_sha256", alignment.get("validator_script_sha256")) == expected_script,
            "Alignment must bind the audited local r3 validator")
    sources = [item for item in manifest["sources"] if item["suite"] == "libero_spatial"]
    records = [item for item in alignment["records"] if item["suite"] == "libero_spatial"]
    require(len(sources) == len(records) == 1, "Exactly one spatial source is required")
    source, evidence = sources[0], records[0]
    require(source["task_id"] == 0 and source["task_uid"] == "libero_spatial:0"
            and source["language"] == evaluation.LANGUAGE, "Wrong development task")
    require(all(evidence.get(key) == source.get(key) for key in ("path", "bytes", "sha256", "task_id", "task_uid"))
            and evidence["status"] == evidence["replay"]["status"] == "passed"
            and len(evidence["demonstrations"]) == 50 and len(evidence["replay"]["samples"]) == 5,
            "Raw alignment evidence is incomplete or targets another source")
    require(type(alignment["action_offset"]) is int and alignment["action_offset"] in (0, 1)
            and evidence["action_offset"] == alignment["action_offset"], "Unverified action offset")
    history = alignment
    if "single_source_receipts" in alignment:
        refs = [item for item in alignment["single_source_receipts"] if item["suite"] == "libero_spatial"]
        require(len(refs) == 1, "Raw single-source receipt is required")
        path = resolve(refs[0]["path"])
        require(digest(path) == refs[0]["sha256"], "Raw single-source receipt changed")
        history = read(path)
    require(type(history.get("owner_pid")) is int and history["owner_pid"] > 0
            and isinstance(history.get("owner_create_time"), (int, float)),
            "Original local r3 process evidence is required; public projections are not raw receipts")
    prior = {}
    for name in ("v1_receipt", "prior_observation_receipt"):
        reference = history[name]
        path = resolve(reference["path"])
        require(digest(path) == reference["sha256"], "Raw prior diagnostic receipt changed")
        receipt = read(path)
        script = "scripts/validate_action_demo_alignment.py" if name == "v1_receipt" else "scripts/validate_action_demo_alignment_r2.py"
        prior_records = [item for item in receipt.get("records", []) if item.get("suite") == "libero_spatial"]
        require(receipt.get("source_manifest_sha256") == digest(manifest_path)
                and receipt.get("status") in ("passed", "needs_attention")
                and receipt.get("script_sha256") == digest(ROOT / script)
                and type(receipt.get("owner_pid")) is int and receipt["owner_pid"] > 0
                and isinstance(receipt.get("owner_create_time"), (int, float))
                and len(prior_records) == 1 and len(prior_records[0].get("demonstrations", [])) == 50
                and len(prior_records[0].get("replay", {}).get("samples", [])) == 5,
                "Prior raw alignment binding is missing; public projections cannot authorize training")
        if name == "prior_observation_receipt":
            require(receipt["v1_receipt"]["sha256"] == history["v1_receipt"]["sha256"]
                    and resolve(receipt["v1_receipt"]["path"]) == resolve(history["v1_receipt"]["path"]),
                    "The r2 diagnostic does not reference this same r1 evidence")
        prior[name] = record(path)
    source_path = resolve(source["path"])
    require(source_path.stat().st_size == source["bytes"] and digest(source_path) == source["sha256"], "Expert source bytes differ")
    return source, alignment["action_offset"], prior


def register_training(name, manifest_path, alignment_path):
    before_deadline()
    campaign = campaign_path(name)
    require(not campaign.exists(), "Use a new campaign directory; existing evidence cannot be overwritten")
    source, offset, history = inspect_alignment(manifest_path, alignment_path)
    models = model_inventory()
    from train_action_adaptation import REQUIRED_SOURCES
    import h5py
    demos = []
    with h5py.File(resolve(source["path"]), "r") as dataset:
        for index, split in ((0, "train"), (1, "validation")):
            count = len(dataset[f"data/demo_{index}/actions"])
            require(count > 10 + offset, "Development demo is too short")
            demos.append({"id": f"spatial_demo_{index}", "split": split, "task": source["task_uid"],
                          "language": source["language"], "path": source["path"], "sha256": source["sha256"],
                          "episode_key": f"demo_{index}", "frame_count": count})
    before_deadline()
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(alignment_path, campaign / "alignment.json")
    write_json(campaign / "data_manifest.json", {"demos": demos, "development_only": True,
               "note": "demo_1 is integration validation only, never prospective validation/test"})
    write_json(campaign / "schedule.json", {"train": [{"demo_id": "spatial_demo_0", "observation_index": i} for i in (0, 10)],
               "validation": [{"demo_id": "spatial_demo_1", "observation_index": 0, "noise_seed": 1000, "time": .5}]})
    protocol = {"purpose_id": "action_training_public_v1", "purpose": "Two real native updates per family; integration only",
                "model": {"models_dir": "models", "files": models, "num_steps": 10},
                "data": {"manifest": record(campaign / "data_manifest.json"), "alignment_evidence": record(campaign / "alignment.json"),
                         "action_offset": offset, "image_transform": "rot180_once"},
                "raw_alignment_history": history, "source_manifest": record(manifest_path),
                "flow_time_distribution": "beta_1.5_1_scaled_0.999_plus_0.001",
                "optimizer": {"type": "AdamW", "lr": .0001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
                "resources": {"torch_threads": 2, "interop_threads": 1, "max_seconds": 600, "evaluation_max_seconds": 7200},
                "checkpoint_every": 1, "validation_every": 2,
                "runs": {f"{mode}_s17": {"mode": mode, "initialization_seed": 17, "flow_seed": 17,
                          "steps": 2, "schedule": record(campaign / "schedule.json")} for mode in ("lora", "output_head")},
                "evaluation": {"methods": [], "cases": []}}
    sources = REQUIRED_SOURCES | evaluation.REQUIRED_SOURCES | ALIGNMENT_SOURCES | {"scripts/report_action_campaign.py"}
    fingerprint = freeze(campaign, protocol, sources)
    return {"campaign": str(campaign), "protocol_sha256": fingerprint, "development_only": True, "model_started": False}


def register_evaluation(name, training_name):
    before_deadline()
    campaign, training_dir = campaign_path(name), campaign_path(training_name)
    require(not campaign.exists(), "Use a new evaluation campaign directory")
    path = training_dir / "protocol.json"
    fingerprint = (training_dir / "protocol.sha256").read_text().strip()
    require(digest(path) == fingerprint, "Local training protocol fingerprint differs")
    training = read(path)
    require(training.get("development_only") is True and training.get("excluded_from_formal_study") is True
            and training.get("public_source_version") == PUBLIC_SOURCE_VERSION, "Only locally reconstructed public development checkpoints are accepted")
    evaluation.verify_snapshot(training_dir / "source_snapshot/index.json", training["source_files"])
    manifest_path = training_dir / "trained_manifest.json"
    evaluation.worker.verify_trained_manifest(manifest_path, digest(manifest_path), training, fingerprint)
    before_deadline()
    campaign.mkdir(parents=True, exist_ok=False)
    protocol = {"purpose_id": "action_development_public_v1", "model": deepcopy(training["model"]),
                "data": deepcopy(training["data"]), "runs": deepcopy(training["runs"]),
                "resources": deepcopy(training["resources"]),
                "historical_training": {"protocol": record(path), "source_snapshot_index": record(training_dir / "source_snapshot/index.json"),
                                        "trained_manifest": record(manifest_path)},
                "evaluation": {"num_steps": 10, "replan_steps": 10, "image_size": 360, "arena_bytes": 268435456,
                               "official_horizon": 280, "methods": [deepcopy(evaluation.METHOD)], "cases": [deepcopy(evaluation.CASE)]},
                "attempt_policy": "one complete development case; preserve any failure and never retry automatically"}
    checksum = freeze(campaign, protocol, evaluation.REQUIRED_SOURCES)
    _, _, _, _, audit = evaluation.verify_plan(campaign / "protocol.json", checksum)
    write_json(campaign / "preflight_audit.json", dict(audit, checked_at=now(), model_loaded=False, simulator_started=False))
    return {"campaign": str(campaign), "protocol_sha256": checksum, "model_started": False}


def archive_own_completed_lock(campaign):
    """Only the current successful supervisor may archive its own live lock."""
    from run_action_campaign import identity, matching, identities, status
    lock_path = AREA / "worker.lock"
    lock = read(lock_path)
    owner = identity(os.getpid())
    require(lock["owner"] == owner and resolve(lock["campaign"]) == campaign, "Cannot archive a different owner's lock")
    require(status(campaign)["state"] == "complete", "Incomplete campaign lock must remain for diagnosis")
    require(not any(item != owner and matching(item) is not None for item in identities(campaign)), "Owned child still alive")
    destination = campaign / ("completed_lock_" + uuid4().hex[:12] + ".json")
    require(not destination.exists(), "Completed lock archive collision")
    lock_path.rename(destination)


def run_training(name, resume=False):
    before_deadline()
    campaign = campaign_path(name)
    protocol = read(campaign / "protocol.json")
    require(protocol.get("purpose_id") == "action_training_public_v1" and protocol.get("public_source_version") == PUBLIC_SOURCE_VERSION,
            "Training command requires its own new public development protocol")
    from run_action_campaign import main as supervise
    result = supervise(["supervise", "--campaign", str(campaign)] + (["--resume"] if resume else []))
    if result == 0 and (AREA / "worker.lock").exists():
        archive_own_completed_lock(campaign)
    return result


def run_evaluation(name):
    before_deadline()
    from run_action_campaign import Supervisor, identity, status
    campaign = campaign_path(name)
    fingerprint = (campaign / "protocol.sha256").read_text().strip()
    protocol, case, _, _, audit = evaluation.verify_plan(campaign / "protocol.json", fingerprint)
    require(not (campaign / "evaluation").exists(), "This development episode was already attempted; never rerun it")
    require(not status(campaign)["live_owned_processes"], "An owned process is still alive")
    AREA.mkdir(parents=True, exist_ok=True)
    with (AREA / "worker.lock").open("x", encoding="utf-8") as stream:
        json.dump({"owner": identity(os.getpid()), "campaign": str(campaign.relative_to(ROOT)),
                   "public_source_version": PUBLIC_SOURCE_VERSION}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    supervisor = None
    try:
        supervisor = Supervisor(campaign, fingerprint, protocol)
        if not supervisor.ready():
            return 3
        output = campaign / "evaluation" / case["case_id"] / "attempt_001"
        code = supervisor.child([sys.executable, "-B", str(ROOT / "scripts/evaluate_action_development_public.py"),
                                 "--protocol", str(campaign / "protocol.json"), "--protocol-sha256", fingerprint,
                                 "--case-id", case["case_id"], "--output", str(output)], output, 7200)
        require(code == 0, "Development episode needs diagnosis; failed attempt is preserved")
        expected = {"evaluation_protocol_sha256": fingerprint, "training_protocol_sha256": audit["training_protocol_sha256"],
                    "trained_manifest_sha256": audit["trained_manifest_sha256"], "case_id": case["case_id"],
                    "excluded_from_formal_study": True}
        verify_seal(output, expected)
        report = read(output / "report.json")
        require(report["status"] == "complete" and all(report[key] == expected[key] for key in
                ("evaluation_protocol_sha256", "training_protocol_sha256", "trained_manifest_sha256", "excluded_from_formal_study")),
                "Development report binding differs")
        supervisor.save(state="complete", evaluation_completed=1, current=None, ended_at=now(), excluded_from_formal_study=True)
        archive_own_completed_lock(campaign)
        return 0
    except BaseException as error:
        try:
            if supervisor is not None:
                supervisor.save(state="needs_attention", error=repr(error), ended_at=now())
            else:
                write_json(campaign / "launch_failure.json", {"state": "needs_attention", "error": repr(error)})
        except BaseException as persistence_error:
            print(json.dumps({"error": repr(error), "persistence_error": repr(persistence_error)}), file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Read-only description; no download, model, simulator, or directory creation")
    training = commands.add_parser("register-training")
    training.add_argument("--campaign-id", required=True)
    training.add_argument("--source-manifest", required=True, type=Path)
    training.add_argument("--alignment", required=True, type=Path)
    train = commands.add_parser("train")
    train.add_argument("--campaign-id", required=True)
    train.add_argument("--resume", action="store_true")
    registration = commands.add_parser("register-evaluation")
    registration.add_argument("--campaign-id", required=True)
    registration.add_argument("--training-campaign-id", required=True)
    execute = commands.add_parser("evaluate")
    execute.add_argument("--campaign-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps({"source_version": PUBLIC_SOURCE_VERSION, "deadline_utc": DEADLINE,
                          "before_original_deadline": not deadline_reached(), "commands_start_compute": ["train", "evaluate"],
                          "requires": "pinned local models/assets/expert HDF5 and locally generated raw r1/r2/r3 alignment receipts",
                          "public_projections_are_raw_gates": False, "creates_files": False}))
        return 0
    if args.command == "register-training":
        result = register_training(args.campaign_id, args.source_manifest, args.alignment)
    elif args.command == "register-evaluation":
        result = register_evaluation(args.campaign_id, args.training_campaign_id)
    elif args.command == "train":
        return run_training(args.campaign_id, args.resume)
    else:
        return run_evaluation(args.campaign_id)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from embodied_research.runtime import configure_platform
    configure_platform()
    raise SystemExit(main())
