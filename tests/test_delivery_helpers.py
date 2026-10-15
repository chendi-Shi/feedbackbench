import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_script("replay_episode")
bundle = load_script("package_results")


@pytest.fixture
def episode(tmp_path, monkeypatch):
    campaign = tmp_path / "outputs/campaign"
    original = campaign / "stage/episodes/example"
    original.mkdir(parents=True)
    (campaign / "campaign_freeze.json").write_text("{}")
    protocol = original.parent.parent / "protocol.json"
    protocol.write_text("{}")
    states = np.arange(4, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32) / 100
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = {
        "id": "example", "status": "complete", "suite": "libero_goal", "task_id": 3,
        "seed": 1000, "init_state_id": 0, "steps": 3, "success": False,
        "terminated": False, "truncated": True, "protocol_fingerprint": "test",
        "environment": {"image_size": 2}, "initial_robot_state": states[0].tolist(),
        "final_robot_state": states[-1].tolist(), "initial_image_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    }
    (original / "result.json").write_text(json.dumps(result))
    np.savez_compressed(original / "trajectory.npz", actions=np.zeros((3, 7), dtype=np.float32),
                        states=states[:-1], final_state=states[-1])
    for name, pixels in (("initial.png", frame), ("initial_wrist.png", frame + 99), ("final.png", frame + 3)):
        Image.fromarray(pixels).save(original / name)
    monkeypatch.setattr(replay, "validate_provenance", lambda *args: (protocol, protocol.read_bytes()))

    class Simulator:
        wrong_flags = False
        wrong_wrist = False

        def __init__(self, *args, **kwargs):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def metadata(self):
            return result["environment"]

        def reset(self, **kwargs):
            return {"observation.state": states[0].copy()}

        def frame(self, camera="image"):
            return frame + ((98 if self.wrong_wrist else 99) if camera == "image2" else self.index)

        def step(self, action):
            self.index += 1
            return {"observation.state": states[self.index].copy()}, 0, False, (
                self.index == 3 and not self.wrong_flags), {"is_success": False}

    return original, result, Simulator


def test_replay_rejects_new_outputs_anywhere_inside_formal_campaign(episode):
    original, _, _ = episode
    with pytest.raises(ValueError, match="outside"):
        replay.validate_output(original, original.parents[2] / "another_stage/diagnostic")


def test_replay_checks_both_cameras_states_flags_and_preserves_sources(episode, tmp_path):
    original, _, simulator = episode
    before = replay._source_hashes(original)
    output = tmp_path / "diagnostics/replay"
    audit = replay.replay_episode(original, output, simulator_factory=simulator)
    assert replay._source_hashes(original) == before
    assert audit["initial_wrist_image_equal"] and audit["terminal_flags_equal"]
    assert audit["max_state_absolute_difference"] == 0
    assert audit["source_files_sha256"]["trajectory.npz"] == before["trajectory.npz"]
    assert (output / "replay.gif").is_file()


@pytest.mark.parametrize("fault", ["wrong_flags", "wrong_wrist"])
def test_replay_rejects_parity_failures_before_writing_output(episode, tmp_path, fault):
    original, _, simulator = episode
    setattr(simulator, fault, True)
    output = tmp_path / "diagnostic"
    with pytest.raises((ValueError, AssertionError)):
        replay.replay_episode(original, output, simulator_factory=simulator)
    assert not output.exists()


def test_replay_and_bundle_reject_trajectory_inconsistent_with_json(episode, tmp_path):
    original, result, simulator = episode
    with np.load(original / "trajectory.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["final_state"][0] += 1
    np.savez_compressed(original / "trajectory.npz", **arrays)
    with pytest.raises(AssertionError):
        replay.replay_episode(original, tmp_path / "diagnostic", simulator_factory=simulator)
    with pytest.raises(AssertionError):
        bundle.validate_episode_files(original, result)


def test_bundle_requires_trajectory_and_images_even_with_complete_result(episode):
    original, result, _ = episode
    bundle.validate_episode_files(original, result)
    (original / "initial_wrist.png").unlink()
    with pytest.raises(ValueError, match="Missing"):
        bundle.validate_episode_files(original, result)


def test_bundle_requires_declared_recorded_gif(episode):
    original, result, _ = episode
    result.update(record_video=True, video="rollout.gif")
    with pytest.raises(ValueError, match="Missing"):
        bundle.validate_episode_files(original, result)
    Image.new("RGB", (2, 2)).save(original / "rollout.gif")
    bundle.validate_episode_files(original, result)
    result["video"] = None
    with pytest.raises(ValueError, match="not declared"):
        bundle.validate_episode_files(original, result)


def test_bundle_requires_relative_report_resources_inside_project(tmp_path):
    document = tmp_path / "report.html"
    document.write_text('<img src="missing.svg"><a href="https://example.org">external</a>')
    with pytest.raises(ValueError, match="report resource"):
        bundle.validate_html_resources(document, tmp_path)
    (tmp_path / "missing.svg").write_text("<svg/>")
    bundle.validate_html_resources(document, tmp_path)
    document.write_text('<a href="#results">section</a><a href="../outside.txt">outside</a>')
    with pytest.raises(ValueError, match="out-of-project"):
        bundle.validate_html_resources(document, tmp_path)


def test_bundle_redacts_only_diagnostic_copies_and_hashes_archive_bytes(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    personal = "C:" + "/" + "/".join(["Users", "private-user", "Company Drive", "project", "trace.log"])
    original = json.dumps({"source_path": personal, "checksum": "a" * 64}).encode()
    files = {
        "outputs/campaign/protocol.json": b'{"fingerprint":"unchanged"}',
        "outputs/campaign/runtime_audit.json": original,
        "artifacts/policy_smoke_observation.npz": b"small recorded input",
        "artifacts/unit-tests.xml": ('<testsuite ' + 'hostname' + '="private-host" tests="4"/>').encode(),
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    monkeypatch.setattr(bundle, "REQUIRED_FILES", tuple(name for name in files if name.startswith("artifacts/")))
    monkeypatch.setattr(bundle, "validate_campaign", lambda root: ({"fake_stage": 1}, {"fake_stage": "test"}))
    destination = root / "deliverables/test.zip"
    bundle.package(root, destination)
    assert all((root / name).read_bytes() == value for name, value in files.items())
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("feedbackbench/BUNDLE_MANIFEST.json"))
        for name, entry in manifest["files"].items():
            archived = archive.read("feedbackbench/" + name)
            assert hashlib.sha256(archived).hexdigest() == entry["sha256"]
            assert hashlib.sha256(files[name]).hexdigest() == entry["source_sha256"]
            assert b"private-user" not in archived and b"private-host" not in archived
        assert archive.read("feedbackbench/outputs/campaign/protocol.json") == files["outputs/campaign/protocol.json"]
        assert archive.getinfo("feedbackbench/artifacts/policy_smoke_observation.npz").compress_type == zipfile.ZIP_STORED
        assert any("--output outputs/reproduction" in command for command in manifest["reproduce"])
        assert all("&&" not in command for command in manifest["reproduce"])
        assert len(manifest["privacy"]["redacted_files"]) == 2


def test_bundle_refuses_to_redact_formal_evidence_or_write_inside_it(tmp_path):
    personal = "C:" + "/" + "/".join(["Users", "private-user", "project"])
    with pytest.raises(ValueError, match="outside redactable"):
        bundle.archive_content("outputs/campaign/protocol.json", json.dumps({"path": personal}).encode())
    with pytest.raises(ValueError, match="outside source and evidence"):
        bundle.package(tmp_path, tmp_path / "outputs/campaign/bundle.zip")
    assert "artifacts/policy_smoke_observation.npz" in bundle.REQUIRED_FILES


def test_bundle_refuses_incomplete_campaign(tmp_path):
    campaign = tmp_path / "outputs/campaign"
    campaign.mkdir(parents=True)
    (campaign / "campaign_status.json").write_text(json.dumps({"status": "running", "through": "flow_steps"}))
    with pytest.raises(ValueError, match="every declared"):
        bundle.validate_campaign(tmp_path)


def test_bundle_recomputes_protocol_fingerprint_before_accepting_results(tmp_path, monkeypatch):
    campaign = tmp_path / "outputs/campaign"
    campaign.mkdir(parents=True)
    (campaign / "campaign_status.json").write_text(json.dumps({
        "status": "complete_through_requested_stage", "through": "flow_steps"}))
    (campaign / "campaign_freeze.json").write_text(json.dumps({"source_sha256": {}}))
    protocol = {"plan": {}, "sources": {}, "runtime_versions": {}, "artifacts": {}, "fingerprint": "forged"}
    monkeypatch.setattr(bundle, "load_experiment", lambda path: SimpleNamespace(protocol=protocol))
    with pytest.raises(ValueError, match="fingerprint"):
        bundle.validate_campaign(tmp_path)


def test_frontier_bundle_keeps_original_unfinished_scope_explicit(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for name in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<testsuite tests="1"/>')
    observation = root / "outputs/frontier/profile.npz"
    observation.parent.mkdir(parents=True)
    observation.write_bytes(b"recorded fixture")
    monkeypatch.setattr(bundle, "REQUIRED_FILES", ())
    monkeypatch.setattr(bundle, "validate_frontier", lambda root: ({"frontier_heldout": 24}, {"pilot": "hash"}))
    destination = root / "deliverables/frontier.zip"
    bundle.package(root, destination, frontier=True)
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("feedbackbench/BUNDLE_MANIFEST.json"))
        assert manifest["unfinished_original_plan"]["50_smolvla"] == 12
        assert manifest["unfinished_original_plan"]["60_flow_steps"] == 4
        assert "not completed" in manifest["unfinished_original_plan"]["status"]
        assert "outputs/frontier/profile.npz" in manifest["files"]
        assert "reproduction-cache" in manifest["reproduce"][1]


def test_frontier_bundle_cannot_write_into_its_own_evidence_tree(tmp_path):
    with pytest.raises(ValueError, match="outside source and evidence"):
        bundle.package(tmp_path, tmp_path / "outputs/frontier/bundle.zip", frontier=True)


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["outputs/frontier/profile.json", "outputs/frontier/profile.progress.json"])
def test_frontier_profile_is_redacted_only_in_archive_with_original_hash(tmp_path, monkeypatch, name):
    personal = "C:" + "/" + "/".join(["Users", "private-user", "project"])
    value = {"runtime": {"executable": personal, "hostname": "private-host"},
             "bound_original_sha256": "a" * 64}
    profile = write_json(tmp_path, name, value)
    raw = profile.read_bytes()
    for artifact in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml"):
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('<testsuite tests="1"/>')
    monkeypatch.setattr(bundle, "REQUIRED_FILES", ())
    monkeypatch.setattr(bundle, "validate_frontier", lambda root: ({}, {}))
    destination = tmp_path / "delivery.zip"
    bundle.package(tmp_path, destination, frontier=True)
    assert profile.read_bytes() == raw
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("feedbackbench/BUNDLE_MANIFEST.json"))
        exported = archive.read("feedbackbench/" + name)
        assert b"private-user" not in exported and b"private-host" not in exported
        assert json.loads(exported)["bound_original_sha256"] == value["bound_original_sha256"]
        assert manifest["files"][name]["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert manifest["files"][name]["sha256"] == hashlib.sha256(exported).hexdigest()
        assert name in manifest["privacy"]["redacted_files"]
        assert "original source" in manifest["privacy"]["diagnostic_hash_fields"]


@pytest.mark.parametrize("suffix", ["csv", "CSV", "tsv"])
def test_public_tables_with_private_paths_are_rejected(suffix):
    personal = "C:" + "\\" + "\\".join(["Users", "private-user", "project"])
    with pytest.raises(ValueError, match="outside redactable"):
        bundle.archive_content(f"outputs/frontier/cache_study/groups.{suffix}", f"key,value\npath,{personal}\n".encode())


@pytest.mark.parametrize("key", ["hostname", "HostName", "COMPUTERNAME"])
def test_nested_json_host_keys_are_rejected_outside_explicit_diagnostics(key):
    raw = json.dumps({"items": [{"nested": {key: "private-host"}}]}).encode()
    with pytest.raises(ValueError, match="outside redactable"):
        bundle.archive_content("outputs/frontier/unapproved_probe.json", raw)
    exported, changed = bundle.archive_content("outputs/frontier/profile.json", raw)
    assert changed and b"private-host" not in exported
    # Ordinary narrative text mentioning a key is not an identifying field.
    harmless = json.dumps({"note": f"The {key} field is intentionally omitted."}).encode()
    assert bundle.archive_content("outputs/frontier/public.json", harmless) == (harmless, False)


@pytest.fixture
def probe_records(tmp_path):
    pending = write_json(tmp_path, "models/vlajepa/manifest.pending.json", {"status": "pending", "files": []})
    final = write_json(tmp_path, "models/vlajepa/manifest.json", {"status": "verified", "files": [
        {"path": "models/vlajepa/policy/model.safetensors", "sha256": "e" * 64}]})
    weight = tmp_path / "models/vlajepa/policy/model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"excluded synthetic weight payload")
    folder = tmp_path / "artifacts/frontier"
    folder.mkdir(parents=True)

    def ref(path):
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": bundle._hash(path)}

    schema = write_json(tmp_path, "artifacts/frontier/historical.schema.json", {"active": []})
    historical = {"status": "passed_schema_only", "inference_executed": False,
                  "script_sha256": "a" * 64, "manifest": ref(pending), "schema": ref(schema)}
    old = write_json(tmp_path, "artifacts/frontier/historical.json", historical)
    source = folder / "real.source.py"
    source.write_text("# retained probe source\n", encoding="utf-8")
    inputs = folder / "real.inputs.json"
    inputs.write_bytes(final.read_bytes())
    schema = write_json(tmp_path, "artifacts/frontier/real.schema.json", {"active": []})
    loading = write_json(tmp_path, "artifacts/frontier/real.loading.json", {"loaded": True})
    actions = folder / "real.actions.npz"
    actions.write_bytes(b"synthetic action bytes")
    observation = tmp_path / "artifacts/policy_smoke_observation.npz"
    observation.write_bytes(b"synthetic observation bytes")
    real = {"status": "passed_real_forward", "inference_executed": True,
            "script_sha256": bundle._hash(source), "script_snapshot": source.relative_to(tmp_path).as_posix(),
            "manifest": {**ref(final), "snapshot_path": inputs.relative_to(tmp_path).as_posix()},
            "schema": ref(schema), "loading": ref(loading), "actions": ref(actions), "observation": ref(observation)}
    current = write_json(tmp_path, "artifacts/frontier/real.json", real)
    return old, current


def test_probe_export_keeps_both_manifests_and_bound_snapshots_without_weights(probe_records, tmp_path, monkeypatch):
    for name in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml"):
        (tmp_path / name).write_text('<testsuite tests="1"/>')
    monkeypatch.setattr(bundle, "REQUIRED_FILES", ())
    monkeypatch.setattr(bundle, "validate_frontier", lambda root: ({}, {}))
    destination = tmp_path / "delivery.zip"
    bundle.package(tmp_path, destination, frontier=True)
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("feedbackbench/BUNDLE_MANIFEST.json"))
        names = set(manifest["files"])
        assert bundle.PROBE_MANIFESTS <= names
        assert {"artifacts/frontier/real.source.py", "artifacts/frontier/real.inputs.json",
                "artifacts/frontier/real.actions.npz", "artifacts/policy_smoke_observation.npz"} <= names
        assert not any(name.endswith(".safetensors") for name in names)
        historical, current = [entry for entry in manifest["probe_evidence"]
                               if entry.get("kind") != "download_metadata"]
        assert historical["status"] == "passed_schema_only" and not historical["inference_executed"]
        assert "current probe script" in historical["limitations"][0]
        assert current["inference_executed"] and not current["limitations"]


def test_probe_inputs_snapshot_remains_bound_when_live_manifest_advances(probe_records, tmp_path):
    write_json(tmp_path, "models/vlajepa/manifest.json", {"status": "different future download"})
    files, audit = bundle.frontier_probe_files(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in files}
    assert "models/vlajepa/manifest.json" in names
    assert "models/vlajepa/manifest.pending.json" in names
    assert audit[1]["manifest_archive_path"] == "artifacts/frontier/real.inputs.json"
    assert audit[-1]["kind"] == "download_metadata"
    assert audit[-1]["download_status"] == "different future download"
    assert audit[-1]["inference_executed"] is False


def test_completed_download_and_exact_console_log_export_without_probe_or_weights(tmp_path):
    name = "models/vlajepa/manifest.json"
    write_json(tmp_path, name, {"status": "verified", "files": [
        {"path": "models/vlajepa/model.safetensors", "sha256": "a" * 64}]})
    (tmp_path / "models/vlajepa/model.safetensors").write_bytes(b"excluded weight fixture")
    console_name = "artifacts/vlajepa-real-probe-console.log"
    console = tmp_path / console_name
    console.parent.mkdir(parents=True)
    console.write_bytes(b"")
    (console.parent / "unapproved-console.log").write_bytes(b"not a public probe log")
    files, audits = bundle.frontier_probe_files(tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in files} == {name, console_name}
    assert audits[0]["kind"] == "download_metadata"
    assert audits[0]["download_status"] == "verified" and not audits[0]["inference_executed"]
    assert "does not establish a real forward pass" in audits[0]["limitations"][0]
    assert bundle.archive_content(console_name, console.read_bytes()) == (b"", False)
    personal = "C:" + "/" + "/".join(["Users", "private-user", "console"])
    with pytest.raises(ValueError, match="outside redactable"):
        bundle.archive_content(console_name, personal.encode())


@pytest.mark.parametrize("filename", ["real.source.py", "real.inputs.json", "real.actions.npz", "real.schema.json"])
def test_probe_rejects_changed_referenced_artifacts(probe_records, tmp_path, filename):
    path = tmp_path / "artifacts/frontier" / filename
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        value["changed"] = True
        path.write_text(json.dumps(value))
    else:
        with path.open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("reference", ["models/vlajepa/policy/model.safetensors",
                                       "models/vlajepa/../manifest.json", "outside/manifest.json"])
def test_probe_rejects_unapproved_manifest_paths(probe_records, tmp_path, reference):
    old, _ = probe_records
    record = json.loads(old.read_text())
    record["manifest"]["path"] = reference
    old.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unsafe probe manifest"):
        bundle.frontier_probe_files(tmp_path)


def test_probe_rejects_source_escape_and_missing_real_action_evidence(probe_records, tmp_path):
    _, current = probe_records
    record = json.loads(current.read_text())
    original = dict(record)
    record["script_snapshot"] = "scripts/probe_vlajepa.py"
    current.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Unsafe or unbound"):
        bundle.frontier_probe_files(tmp_path)
    original.pop("actions")
    current.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="lacks bound evidence"):
        bundle.frontier_probe_files(tmp_path)


@pytest.fixture
def frontier_identity(tmp_path):
    assets = {"revision": "asset-revision", "files": []}
    write_json(tmp_path, "assets/libero/asset_manifest.json", assets)
    artifacts = {"assets_revision": assets["revision"], "assets_manifest_sha256": bundle.digest_json(assets), "models": {}}
    for name in ("smolvla", "smolvlm_assets"):
        manifest = {"revision": name + "-revision", "files": [{"file": "config.json", "sha256": "a" * 64}]}
        write_json(tmp_path, f"models/{name}/download_manifest.json", manifest)
        artifacts["models"][name] = {"revision": manifest["revision"], "files": {"config.json": "a" * 64}}
    return artifacts


@pytest.mark.parametrize("fault", ["smolvla", "smolvlm_assets", "assets", "asset_revision", "missing_model"])
def test_frontier_manifest_identity_rejects_stale_or_incomplete_exports(frontier_identity, tmp_path, fault):
    bundle.validate_artifact_identity(tmp_path, frontier_identity)
    if fault in {"smolvla", "smolvlm_assets"}:
        path = tmp_path / f"models/{fault}/download_manifest.json"
        value = json.loads(path.read_text())
        value["files"][0]["sha256"] = "b" * 64
        path.write_text(json.dumps(value))
    elif fault == "assets":
        write_json(tmp_path, "assets/libero/asset_manifest.json", {"revision": "replaced", "files": []})
    elif fault == "asset_revision":
        frontier_identity["assets_revision"] = "incorrect"
    else:
        del frontier_identity["models"]["smolvlm_assets"]
    with pytest.raises(ValueError, match="identity"):
        bundle.validate_artifact_identity(tmp_path, frontier_identity)


def test_frontier_validation_checks_new_manifest_identity_before_cache_and_report(tmp_path, monkeypatch):
    # Bypass completed baseline I/O only; exercise the actual frontier validation
    # entry point and stop at its newly required exported-manifest identity gate.
    data, documents = {}, {}
    assets = {"revision": "assets", "files": []}
    documents["assets/libero/asset_manifest.json"] = assets
    documents["outputs/campaign/campaign_freeze.json"] = {"source_sha256": {}}
    documents["outputs/campaign/resume_audit.json"] = {"raw_episode_sha256": {}}
    documents["outputs/checkpoints/before_smol/stop_receipt.json"] = {"status": "stopped", "audit_sha256": "audit"}
    for stage, count in zip(bundle.STAGES[:4], (40, 32, 128, 32), strict=True):
        protocol = {"plan": {}, "sources": {}, "runtime_versions": {},
                    "artifacts": {"assets_manifest_sha256": bundle.digest_json(assets), "models": {}}}
        protocol["fingerprint"] = bundle.digest_json(protocol)
        protocol["cases"] = []
        data[stage] = SimpleNamespace(protocol=protocol, partial=False, errors=[],
                                      records=[{"id": "fixture"}] * count, input_sha256={})
        documents[f"outputs/campaign/{stage}/summary.json"] = {"partial": False, "input_sha256": {}}
        documents[f"outputs/campaign/frozen_plans/{stage}.json"] = {}
    documents["outputs/campaign/development_selection.json"] = {
        "development_protocol_fingerprint": data["20_development"].protocol["fingerprint"],
        "development_inputs_sha256": bundle.digest_json({})}
    documents["outputs/campaign/delay_comparison.json"] = {"baseline_input_sha256": {}, "delayed_input_sha256": {}}
    new_artifacts = {"models": {"smolvla": "new-evidence", "smolvlm_assets": "new-evidence"}}
    protocol = {"spec": {"cases": []}, "sources": {}, "runtime_versions": {}, "artifacts": new_artifacts}
    protocol["fingerprint"] = bundle.digest_json(protocol)
    documents["outputs/frontier/cache_study/protocol.json"] = protocol
    fake_runner = SimpleNamespace(SOURCES=(), study_spec=lambda: protocol["spec"],
                                  verify_cache_binding=None, verify_complete=None)
    monkeypatch.setitem(sys.modules, "run_cache_study", fake_runner)
    monkeypatch.setitem(sys.modules, "summarize_cache_study", SimpleNamespace(validate_report=None))
    monkeypatch.setattr(bundle, "_read", lambda path: documents[path.relative_to(tmp_path).as_posix()])
    monkeypatch.setattr(bundle, "_hash", lambda path: "audit")
    monkeypatch.setattr(bundle, "load_experiment", lambda path: data[path.name])
    monkeypatch.setattr(bundle, "expand_plan", lambda plan: [])
    monkeypatch.setattr(bundle, "validate_episode_files", lambda *args: None)

    def reject_changed_identity(root, artifacts):
        assert root == tmp_path and artifacts is new_artifacts
        raise ValueError("frontier manifest gate reached")

    monkeypatch.setattr(bundle, "validate_artifact_identity", reject_changed_identity)
    with pytest.raises(ValueError, match="frontier manifest gate reached"):
        bundle.validate_frontier(tmp_path)


@pytest.mark.parametrize("only_suffix", ["json", "csv"])
def test_frontier_retrieval_diagnostics_require_both_exports(tmp_path, monkeypatch, only_suffix):
    output = tmp_path / "outputs/frontier/cache_study"
    output.mkdir(parents=True)
    (output / f"retrieval_analysis.{only_suffix}").write_text("fixture")
    calls = []
    monkeypatch.setitem(sys.modules, "analyze_cache_retrieval", SimpleNamespace(
        validate_analysis=lambda *args, **kwargs: calls.append((args, kwargs))))
    with pytest.raises(ValueError, match="Missing"):
        bundle.validate_optional_retrieval_analysis(tmp_path, output)
    assert calls == []


@pytest.mark.parametrize("rejection", [None, "partial diagnostic", "stale diagnostic"])
def test_frontier_retrieval_diagnostics_enforce_complete_readonly_validation(tmp_path, monkeypatch, rejection):
    output = tmp_path / "outputs/frontier/cache_study"
    output.mkdir(parents=True)
    paths = [output / f"retrieval_analysis.{suffix}" for suffix in ("json", "csv")]
    for path in paths:
        path.write_bytes(b"unchanged diagnostic fixture")
    calls = []

    def validate(directory, *, analysis_dir=None, partial=False, root=None):
        calls.append((directory, analysis_dir, partial, root))
        if rejection:
            raise ValueError(rejection)

    monkeypatch.setitem(sys.modules, "analyze_cache_retrieval", SimpleNamespace(validate_analysis=validate))
    if rejection:
        with pytest.raises(ValueError, match=rejection):
            bundle.validate_optional_retrieval_analysis(tmp_path, output)
    else:
        bundle.validate_optional_retrieval_analysis(tmp_path, output)
    assert calls == [(output, None, False, tmp_path)]
    assert all(path.read_bytes() == b"unchanged diagnostic fixture" for path in paths)
