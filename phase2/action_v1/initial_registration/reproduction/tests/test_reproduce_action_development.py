"""Small-file public reproduction checks; no Torch, model, simulator, network, or worker launch."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("public_reproduction_under_test", SCRIPTS / "reproduce_action_development.py")
public = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(public)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def localize(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "ROOT", tmp_path)
    monkeypatch.setattr(public, "AREA", tmp_path / "outputs/action_adaptation")
    original = public.resolve
    monkeypatch.setattr(public, "resolve", lambda value, base=None: original(value, tmp_path if base is None else base))
    monkeypatch.setattr(public, "deadline_reached", lambda: False)


def raw_alignment(tmp_path, monkeypatch):
    localize(tmp_path, monkeypatch)
    from train_action_adaptation import REQUIRED_SOURCES
    files = REQUIRED_SOURCES | public.evaluation.REQUIRED_SOURCES | public.ALIGNMENT_SOURCES | {"scripts/report_action_campaign.py"}
    for number, name in enumerate(sorted(files)):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture source {number}\n")
    import h5py
    source_path = tmp_path / "data/expert.hdf5"
    source_path.parent.mkdir()
    with h5py.File(source_path, "w") as data:
        for index in (0, 1):
            data.create_dataset(f"data/demo_{index}/actions", shape=(12, 7), dtype="f4")
    source = {"suite": "libero_spatial", "task_id": 0, "task_uid": "libero_spatial:0",
              "language": public.evaluation.LANGUAGE, "path": "data/expert.hdf5",
              "bytes": source_path.stat().st_size, "sha256": public.digest(source_path)}
    manifest_path = tmp_path / "data/source_manifest.json"
    write(manifest_path, {"sources": [source]})
    evidence = {**source, "status": "passed", "action_offset": 1, "demonstrations": [{}] * 50,
                "replay": {"status": "passed", "samples": [{}] * 5}}
    history = {}
    for key, script, filename in (("v1_receipt", "validate_action_demo_alignment.py", "r1.json"),
                                   ("prior_observation_receipt", "validate_action_demo_alignment_r2.py", "r2.json")):
        path = tmp_path / "alignment" / filename
        prior = {"status": "needs_attention", "source_manifest_sha256": public.digest(manifest_path),
                 "script_sha256": public.digest(tmp_path / "scripts" / script), "owner_pid": 12,
                 "owner_create_time": 3.5, "records": [evidence]}
        if key == "prior_observation_receipt":
            prior["v1_receipt"] = history["v1_receipt"]
        write(path, prior)
        history[key] = public.record(path)
    alignment_path = tmp_path / "alignment/r3.json"
    write(alignment_path, {"status": "passed", "diagnostic_revision": 3, "scope": "observation_action_alignment",
                           "dynamics_replay_verified": False, "source_manifest_sha256": public.digest(manifest_path),
                           "script_sha256": public.digest(tmp_path / "scripts/validate_action_demo_alignment_v2.py"),
                           "owner_pid": 12, "owner_create_time": 3.5, "action_offset": 1, "records": [evidence], **history})
    return manifest_path, alignment_path, source_path


def test_plan_is_read_only_and_does_not_start_compute(tmp_path, monkeypatch, capsys):
    localize(tmp_path, monkeypatch)
    assert public.main(["plan"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["creates_files"] is False
    assert result["public_projections_are_raw_gates"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command", ["register-training", "register-evaluation", "train", "evaluate"])
def test_original_deadline_stops_every_mutating_or_compute_entrypoint(tmp_path, monkeypatch, command):
    localize(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "deadline_reached", lambda: True)
    args = [command, "--campaign-id", "new_test"]
    if command == "register-training":
        args += ["--source-manifest", "missing", "--alignment", "missing"]
    elif command == "register-evaluation":
        args += ["--training-campaign-id", "missing"]
    with pytest.raises(ValueError, match="separate new protocol"):
        public.main(args)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "a/b", "UPPER", "x" * 33])
def test_campaign_ids_cannot_escape_or_create_long_paths(tmp_path, monkeypatch, name):
    localize(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identifier"):
        public.campaign_path(name)
    assert not list(tmp_path.iterdir())


def test_resolved_campaign_cannot_escape_project_via_a_redirected_area(tmp_path, monkeypatch):
    localize(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "AREA", tmp_path.parent / "outside_owned_project")
    with pytest.raises(ValueError, match="escapes"):
        public.campaign_path("valid_name")


def test_direct_public_worker_refuses_expired_protocol_before_creating_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(public.evaluation, "deadline_reached", lambda: True)
    with pytest.raises(ValueError, match="expired"):
        public.evaluation.main(["--protocol", str(tmp_path / "missing.json"), "--protocol-sha256", "x" * 64,
                                "--output", str(tmp_path / "attempt_001")])
    assert not list(tmp_path.iterdir())


def test_public_worker_cannot_bypass_global_lock_by_starting_without_launch_directory(tmp_path):
    campaign = tmp_path / "fresh_eval"
    campaign.mkdir()
    output = campaign / "evaluation" / public.evaluation.CASE_ID / "attempt_001"
    with pytest.raises(ValueError, match="exclusive supervisor"):
        public.evaluation.require_new_attempt(campaign, output, "x" * 64)
    assert not output.exists() and not (campaign / "evaluation").exists()


def test_public_worker_accepts_only_its_bound_live_supervisor_launch(tmp_path, monkeypatch):
    evaluator = public.evaluation
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    original = evaluator.resolve
    monkeypatch.setattr(evaluator, "resolve", lambda value, base=None: original(value, tmp_path if base is None else base))
    campaign = tmp_path / "outputs/action_adaptation/fresh_eval"
    output = campaign / "evaluation" / evaluator.CASE_ID / "attempt_001"
    launch = output.with_name("attempt_001.launch")
    launch.mkdir(parents=True)
    (launch / "stdout.log").write_text("")
    (launch / "stderr.log").write_text("")
    owner = {"pid": 123, "created": 1.5}
    write(campaign / "status.json", {"state": "launching", "protocol_sha256": "p" * 64,
                                    "current": output.relative_to(campaign).as_posix(), "owner": owner})
    write(tmp_path / "outputs/action_adaptation/worker.lock", {"owner": owner,
          "campaign": str(campaign.relative_to(tmp_path))})
    monkeypatch.setattr(evaluator, "matching", lambda record: object() if record == owner else None)
    evaluator.require_new_attempt(campaign, output, "p" * 64)
    assert not output.exists()
    monkeypatch.setattr(evaluator, "matching", lambda record: None)
    with pytest.raises(ValueError, match="live exclusive supervisor"):
        evaluator.require_new_attempt(campaign, output, "p" * 64)


def test_registration_creates_new_raw_protocol_and_snapshot_without_private_paths(tmp_path, monkeypatch):
    manifest, alignment, _ = raw_alignment(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "model_inventory", lambda: {"models/fixture": "f" * 64})
    checked = []
    monkeypatch.setattr(public, "verify_protocol", lambda path, fingerprint: checked.append((path, fingerprint)))
    result = public.register_training("fresh_train", manifest, alignment)
    campaign = public.campaign_path("fresh_train")
    protocol = public.read(campaign / "protocol.json")
    assert protocol["data"]["action_offset"] == 1
    assert set(protocol["runs"]) == {"lora_s17", "output_head_s17"}
    assert all(run["steps"] == 2 for run in protocol["runs"].values())
    assert protocol["deadline_utc"] == public.DEADLINE and protocol["excluded_from_formal_study"] is True
    assert not any(path.startswith(".runtime/") for path in protocol["source_files"])
    public.evaluation.verify_snapshot(campaign / "source_snapshot/index.json", protocol["source_files"])
    assert public.digest(campaign / "protocol.json") == result["protocol_sha256"]
    assert len(checked) == 1 and result["model_started"] is False
    original = (campaign / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="new campaign"):
        public.register_training("fresh_train", manifest, alignment)
    assert (campaign / "protocol.json").read_bytes() == original


def test_registration_rechecks_deadline_after_long_input_verification(tmp_path, monkeypatch):
    manifest, alignment, _ = raw_alignment(tmp_path, monkeypatch)
    expired = [False]
    monkeypatch.setattr(public, "deadline_reached", lambda: expired[0])

    def model_inventory():
        expired[0] = True
        return {}

    monkeypatch.setattr(public, "model_inventory", model_inventory)
    with pytest.raises(ValueError, match="expired"):
        public.register_training("too_late", manifest, alignment)
    assert not public.AREA.exists()


@pytest.mark.parametrize("change", ["projection", "prior_projection", "prior_changed", "source_changed", "offset", "wrong_script"])
def test_public_summaries_and_corrupted_raw_alignment_cannot_authorize_training(tmp_path, monkeypatch, change):
    manifest, alignment, source = raw_alignment(tmp_path, monkeypatch)
    value = public.read(alignment)
    if change == "projection":
        value.pop("records")
    elif change == "prior_projection":
        prior_path = public.resolve(value["v1_receipt"]["path"])
        prior = public.read(prior_path)
        prior.pop("owner_pid")
        write(prior_path, prior)
        value["v1_receipt"] = public.record(prior_path)
    elif change == "prior_changed":
        public.resolve(value["prior_observation_receipt"]["path"]).write_text("changed")
    elif change == "source_changed":
        source.write_bytes(b"wrong bytes")
    elif change == "offset":
        value["action_offset"] = 0
    else:
        value["script_sha256"] = "wrong"
    write(alignment, value)
    with pytest.raises((ValueError, KeyError)):
        public.inspect_alignment(manifest, alignment)
    assert not public.AREA.exists()


def test_public_evaluator_has_no_private_helper_source_dependency():
    assert not any(path.startswith(".runtime/") for path in public.evaluation.REQUIRED_SOURCES)
    assert "scripts/reproduce_action_development.py" in public.evaluation.REQUIRED_SOURCES
    assert "scripts/evaluate_action_development_public.py" in public.evaluation.REQUIRED_SOURCES


def test_historical_r2_is_exact_preserved_source():
    assert public.digest(SCRIPTS / "validate_action_demo_alignment_r2.py") == "be836f84507b66a0b7ea72787c8980e9ffefe8600d95bd589fabd71db7398edf"


@pytest.mark.parametrize("code,bad_receipt", [(75, False), (0, True), (0, False)])
def test_explicit_eval_supervision_retains_failed_lock_and_archives_only_verified_success(tmp_path, monkeypatch, code, bad_receipt):
    localize(tmp_path, monkeypatch)
    import run_action_campaign as supervisor_module
    campaign = public.campaign_path("fresh_eval")
    campaign.mkdir(parents=True)
    (campaign / "protocol.sha256").write_text("e" * 64)
    audit = {"training_protocol_sha256": "t" * 64, "trained_manifest_sha256": "m" * 64}
    case = public.evaluation.CASE
    monkeypatch.setattr(public.evaluation, "verify_plan", lambda *args: ({}, case, {}, {}, audit))
    monkeypatch.setattr(supervisor_module, "status", lambda *args: {"live_owned_processes": []})
    monkeypatch.setattr(supervisor_module, "identity", lambda pid: {"pid": pid, "created": 1.5})
    archived = []
    monkeypatch.setattr(public, "archive_own_completed_lock", lambda path: archived.append(path))

    class Supervisor:
        def __init__(self, *args):
            pass

        def ready(self):
            return True

        def child(self, command, output, limit):
            assert limit == 7200 and command[2].endswith("evaluate_action_development_public.py")
            assert not any(".runtime/" in item for item in command)
            if code == 0:
                binding = {"evaluation_protocol_sha256": "wrong" if bad_receipt else "e" * 64,
                           **audit, "case_id": case["case_id"], "excluded_from_formal_study": True}
                write(output / "report.json", {"status": "complete", **binding})
                public.evaluation.seal(output, binding)
            return code

        def save(self, **details):
            write(campaign / "status.json", details)

    monkeypatch.setattr(supervisor_module, "Supervisor", Supervisor)
    if code == 0 and not bad_receipt:
        assert public.run_evaluation("fresh_eval") == 0
        assert archived == [campaign]
    else:
        with pytest.raises(ValueError):
            public.run_evaluation("fresh_eval")
        assert not archived and (public.AREA / "worker.lock").exists()
        assert public.read(campaign / "status.json")["state"] == "needs_attention"


def test_lock_archive_rejects_another_owner_or_a_live_child(tmp_path, monkeypatch):
    localize(tmp_path, monkeypatch)
    import run_action_campaign as supervisor_module
    campaign = public.campaign_path("done")
    campaign.mkdir(parents=True)
    owner = {"pid": 111, "created": 1.5}
    lock = public.AREA / "worker.lock"
    write(lock, {"owner": owner, "campaign": str(campaign.relative_to(tmp_path))})
    monkeypatch.setattr(supervisor_module, "identity", lambda pid: owner)
    monkeypatch.setattr(supervisor_module, "status", lambda path: {"state": "complete"})
    child = {"pid": 222, "created": 2.5}
    monkeypatch.setattr(supervisor_module, "identities", lambda path: [owner, child])
    monkeypatch.setattr(supervisor_module, "matching", lambda value: object() if value == child else None)
    with pytest.raises(ValueError, match="child still alive"):
        public.archive_own_completed_lock(campaign)
    assert lock.exists()
    monkeypatch.setattr(supervisor_module, "identities", lambda path: [owner])
    write(lock, {"owner": child, "campaign": str(campaign.relative_to(tmp_path))})
    with pytest.raises(ValueError, match="different owner's lock"):
        public.archive_own_completed_lock(campaign)
    assert lock.exists()
    write(lock, {"owner": owner, "campaign": str(campaign.relative_to(tmp_path))})
    public.archive_own_completed_lock(campaign)
    assert not lock.exists() and len(list(campaign.glob("completed_lock_*.json"))) == 1
