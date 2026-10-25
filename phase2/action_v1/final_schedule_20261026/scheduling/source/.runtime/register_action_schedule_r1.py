"""One-time registration after explicit acceptance; never launches a worker."""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from action_campaign_common import DEADLINE, deadline_reached, digest, now, read, verify_protocol, write_json
from run_action_campaign import identities, completed_training
from run_action_adaptation_probe import matching

PARENT_SHA = "02f888259f1733177a58dfd29ad9660936e5c6e30f80ea6e26f2c37831426f03"
TRAINED_SHA = "bf36458b883eb062a9e16a02a0d631f301e895a8a1a3025b01f7923678c4dd37"
PROPOSAL_SHA = "ca7b3a723ddc05f2ff05bb9494c28076cba5c75c4aa5e5ed59a2fd966cabe4ef"


def main():
    if deadline_reached():
        raise RuntimeError("Deadline passed; registration cannot authorize evaluation")
    campaign = ROOT / "outputs/action_adaptation/study_v1"
    dest = campaign / "scheduling_r1"
    if dest.exists():
        raise FileExistsError("Registration already exists; never overwrite")
    protocol = verify_protocol(campaign / "protocol.json", PARENT_SHA, inputs=True)
    if digest(campaign / "trained_manifest.json") != TRAINED_SHA:
        raise ValueError("Frozen trained manifest changed")
    completed = completed_training(campaign, protocol, PARENT_SHA)
    if len(completed) != 6 or read(campaign / "trained_manifest.json")["runs"] != completed:
        raise ValueError("Expected all six original final checkpoints")
    proposal = ROOT / ".runtime/action_incidents/resource_exhaustion_20261023/proposed_execution_revision_v2.json"
    if digest(proposal) != PROPOSAL_SHA:
        raise ValueError("Accepted proposal changed")
    owned = identities(campaign)
    lock = ROOT / "outputs/action_adaptation/worker.lock"
    if lock.exists():
        owned.append(read(lock)["owner"])
    launch_records = []
    for launch in ROOT.glob(".runtime/action-study-v1-*.launch.json"):
        record = json.loads(launch.read_text(encoding="utf-8-sig"))
        extra = []
        if "launcher_pid" in record and "launcher_created_utc" in record:
            extra.append({"pid": record["launcher_pid"], "created": datetime.fromisoformat(record["launcher_created_utc"].replace("Z", "+00:00")).timestamp()})
            owned.extend(extra)
        launch_records.append({"path": launch.relative_to(ROOT).as_posix(), "sha256": digest(launch), "identities": extra})
        def scan(value):
            if isinstance(value, dict):
                if isinstance(value.get("pid"), int) and "created" in value:
                    owned.append({"pid": value["pid"], "created": value["created"]})
                for child in value.values():
                    scan(child)
            elif isinstance(value, list):
                for child in value:
                    scan(child)
        scan(record)
    live = [item for item in owned if matching(item) is not None]
    if live:
        raise RuntimeError(f"Parent identities are still alive: {live}")
    files = set((campaign / "evaluation").rglob("*"))
    files = {p for p in files if p.is_file()}
    for name in ("protocol.json", "protocol.sha256", "status.json", "report.json", "process_ownership.jsonl", "trained_manifest.json"):
        files.add(campaign / name)
    files.update(ROOT / path for path in protocol["source_files"])
    files.add(proposal)
    files.update(ROOT / item["path"] for item in launch_records)
    original_files = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    sources = ["scripts/run_action_schedule_r1.py", "scripts/evaluate_action_schedule_r1.py",
               "scripts/report_action_schedule_r1.py", "tests/test_action_schedule_r1.py"]
    source_files = {path: digest(ROOT / path) for path in sources}
    supporting = [".runtime/register_action_schedule_r1.py", "docs/ACTION_SCHEDULE_R1_RUNBOOK.md"]
    supporting_files = {path: digest(ROOT / path) for path in supporting}
    dest.mkdir()
    archive_files = {}
    for relative, sha in original_files.items():
        copy = dest / "original" / sha
        copy.parent.mkdir(parents=True, exist_ok=True)
        if not copy.exists():
            shutil.copy2(ROOT / relative, copy)
        if digest(copy) != sha:
            raise ValueError("Archive copy mismatch")
        archive_files[relative] = copy.relative_to(ROOT).as_posix()
    inventory = {"schema": "feedbackbench.action_schedule_prior_inventory.v1", "recorded_at": now(),
                 "files": original_files, "archive_files": archive_files,
                 "parent_identities_all_exited": True, "identities_checked": owned}
    inventory_path = dest / "prior_inventory.json"
    write_json(inventory_path, inventory)
    authorization_path = dest / "authorization.json"
    write_json(authorization_path, {"schema": "feedbackbench.action_schedule_authorization.v1",
        "recorded_at": now(), "user_message": "那就继续测评", "accepted_proposal_sha256": PROPOSAL_SHA,
        "scope": "Continue remaining 278 cases under proposed_execution_revision_v2; preserve unknown case and all 280 denominator; 6 GiB continuous 60 seconds; original deadline; no retraining or other-project intervention."})
    for relative in sources + supporting:
        copy = dest / "source_snapshot" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, copy)
    revision = {"schema": "feedbackbench.action_schedule_revision.v1", "frozen": True,
        "revision_id": "study_v1_schedule_r1", "registered_at": now(), "deadline_utc": DEADLINE,
        "parent": {"campaign": campaign.relative_to(ROOT).as_posix(), "protocol_sha256": PARENT_SHA,
                   "trained_manifest_sha256": TRAINED_SHA}, "source_files": source_files,
        "supporting_files": supporting_files,
        "resources": {"start_available_gib": 6.0, "stable_seconds": 60, "sample_interval_seconds": 5,
                      "runtime_available_gib": .75, "free_disk_gib": 4,
                      "evaluation_max_seconds": protocol["resources"]["evaluation_max_seconds"]},
        "attempt_policy": {"max_attempts_per_case": 3, "resource_exhaustion": "missing_and_continue",
                           "non_resource_failure": "needs_attention"},
        "case_ids": [case["case_id"] for case in protocol["evaluation"]["cases"]],
        "prior_inventory": {"path": inventory_path.relative_to(ROOT).as_posix(), "sha256": digest(inventory_path)},
        "authorization": {"path": authorization_path.relative_to(ROOT).as_posix(), "sha256": digest(authorization_path)},
        "launcher_records": launch_records,
        "preserved_missing_case_ids": ["libero_spatial_t0_i5_clean_lora_s17"],
        "evaluation_planned": 280, "maximum_sealed_total": 279,
        "completion_claim": "partial original study even after remaining schedule finishes",
        "unchanged": {"training": "all six original final checkpoints", "scientific_protocol": PARENT_SHA,
                      "model_workers": 1, "compute": "local CPU only; zero cloud fees; no intervention in other projects"}}
    revision_path = dest / "protocol.json"
    write_json(revision_path, revision)
    sha = digest(revision_path)
    (dest / "protocol.sha256").write_text(sha + "\n", encoding="utf-8")
    from run_action_schedule_r1 import verify_revision
    verify_revision(revision_path, sha, inputs=True)
    print(json.dumps({"registered": True, "revision": str(revision_path), "sha256": sha,
                      "source_files": len(source_files), "preserved_files": len(original_files), "launched": False}))


if __name__ == "__main__":
    main()
