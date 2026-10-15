"""Continuation scope and INT8 evidence packaging using tiny synthetic artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/package_results.py"
SPEC = importlib.util.spec_from_file_location("continuation_packager_test", PATH)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": bundle._hash(path)}


@pytest.fixture
def int8_record(tmp_path):
    folder = tmp_path / "artifacts/frontier"
    folder.mkdir(parents=True)
    own = folder / "quantized.source.py"
    own.write_text("# synthetic retained variant source\n")
    base = folder / "quantized.base_source.py"
    base.write_text("# synthetic retained original probe source\n")
    record = {"status": "blocked_preflight", "script_sha256": bundle._hash(own),
              "script_snapshot": own.relative_to(tmp_path).as_posix(), "variant": bundle.INT8_VARIANT,
              "base_script": {"path": "scripts/probe_vlajepa.py", "sha256": bundle._hash(base),
                              "snapshot_path": base.relative_to(tmp_path).as_posix()},
              "numeric_changes": ["synthetic fixture: dynamic INT8 Qwen Linear; BF16 token table; FP32 action head"],
              "inference_executed": False, "forward_started": False, "forward_calls": 0}
    path = write_json(tmp_path, "artifacts/frontier/quantized.json", record)
    return path, record


def make_passed(root, path, record):
    manifest = write_json(root, "models/vlajepa/manifest.json", {"status": "verified",
        **bundle.INT8_INPUT_REVISIONS, "files": [dict(bundle.VLAJEPA_CHECKPOINT)]})
    inputs = root / "artifacts/frontier/quantized.inputs.json"
    inputs.write_bytes(manifest.read_bytes())
    schema = write_json(root, "artifacts/frontier/quantized.schema.json", {"synthetic": "bound schema fixture"})
    loading_value = {"all_required_inference_keys_consumed": True, "all_source_tensors_finite": True,
                     "native_tied_word_embeddings": True, "linear_modules": 300,
                     "linear_elements": 1811939328, "theoretical_parameter_payload_bytes": 3072709660,
                     "tensors": [{"name": "synthetic only"}]}
    loading = write_json(root, "artifacts/frontier/quantized.loading.json", loading_value)
    actions = root / "artifacts/frontier/quantized.actions.npz"
    np.savez(actions, model_actions=np.zeros((1, 7, 7), dtype=np.float32),
             environment_actions=np.zeros((1, 7, 7), dtype=np.float32))
    observation = root / "artifacts/policy_smoke_observation.npz"
    np.savez(observation, fixture=np.zeros((8,), dtype=np.float32))
    record.update(status="passed_quantized_forward", inference_executed=True, forward_started=True,
                  forward_calls=1, flow_steps=4, batch_size=1,
                  lerobot_revision=bundle.INT8_INPUT_REVISIONS["lerobot_revision"],
                  policy_revision=bundle.INT8_INPUT_REVISIONS["policy_revision"],
                  manifest={**ref(root, manifest), "snapshot_path": inputs.relative_to(root).as_posix(),
                            "checkpoint_complete": True},
                  schema=ref(root, schema),
                  loading={**ref(root, loading), **{k: v for k, v in loading_value.items() if k != "tensors"}},
                  actions={**ref(root, actions), "shape": [1, 7, 7], "first_environment_action": [0.] * 7},
                  observation=ref(root, observation),
                  quantized_forward_audit={"module_calls": {f"synthetic-module-{i}": 1 for i in range(300)},
                                           "distinct_modules_executed": 300, "all_observed_inputs": "CPU float32"})
    path.write_text(json.dumps(record))


def audit_for(root):
    files, audits = bundle.frontier_probe_files(root)
    return files, next(a for a in audits if a["record"] == "artifacts/frontier/quantized.json")


def test_preflight_int8_preserves_both_source_snapshots_and_no_pass_claim(int8_record, tmp_path):
    path, _ = int8_record
    original = path.read_bytes()
    files, audit = audit_for(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in files}
    assert {"artifacts/frontier/quantized.source.py", "artifacts/frontier/quantized.base_source.py"} <= names
    assert audit["status"] == "blocked_preflight" and audit["forward_started"] is False
    assert audit["successful_quantized_forward"] is False
    assert audit["parity_established"] is audit["task_quality_evaluated"] is False
    assert audit["numeric_changes"] and audit["variant"] == bundle.INT8_VARIANT
    assert path.read_bytes() == original


def test_successful_int8_requires_bound_finite_actions_and_discloses_changed_numeric_scope(int8_record, tmp_path):
    make_passed(tmp_path, *int8_record)
    files, audit = audit_for(tmp_path)
    assert audit["successful_quantized_forward"] is True and audit["inference_executed"] is True
    assert audit["parity_established"] is False and audit["task_quality_evaluated"] is False
    assert any("neither original-model parity" in limit for limit in audit["limitations"])
    names = {p.relative_to(tmp_path).as_posix() for p in files}
    assert {"artifacts/frontier/quantized.inputs.json", "artifacts/frontier/quantized.schema.json",
            "artifacts/frontier/quantized.loading.json", "artifacts/frontier/quantized.actions.npz",
            "artifacts/policy_smoke_observation.npz"} <= names
    assert not any(name.endswith(".safetensors") for name in names)


@pytest.mark.parametrize("suffix", ["source.py", "base_source.py", "inputs.json", "loading.json", "actions.npz"])
def test_int8_reference_tampering_rejected(int8_record, tmp_path, suffix):
    make_passed(tmp_path, *int8_record)
    path = tmp_path / "artifacts/frontier" / f"quantized.{suffix}"
    if suffix.endswith("json"):
        value = json.loads(path.read_text())
        value["altered"] = True
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("kind", ["variant", "numeric_changes", "base_script", "script_snapshot", "base_escape"])
def test_even_blocked_int8_requires_exact_changed_source_bindings(int8_record, tmp_path, kind):
    path, record = int8_record
    if kind == "base_escape":
        record["base_script"]["snapshot_path"] = "scripts/probe_vlajepa.py"
    else:
        record.pop(kind)
    path.write_text(json.dumps(record))
    if kind == "variant":
        # A bare blocked historical record has no variant claim. Changing to
        # the explicit quantized-success label must expose the missing identity.
        record["status"] = "passed_quantized_forward"
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("field", ["actions", "loading", "schema", "manifest", "observation", "quantized_forward_audit"])
def test_passed_int8_requires_every_success_reference(int8_record, tmp_path, field):
    path, record = int8_record
    make_passed(tmp_path, path, record)
    record.pop(field)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks bound evidence"):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("fault", ["nonfinite", "shape", "dtype", "first_action"])
def test_self_consistent_hashes_do_not_hide_invalid_int8_actions(int8_record, tmp_path, fault):
    path, record = int8_record
    make_passed(tmp_path, path, record)
    actions = tmp_path / record["actions"]["path"]
    data = np.zeros((1, 7, 7), dtype=np.float32)
    if fault == "nonfinite":
        data[0, 0, 0] = np.nan
    elif fault == "shape":
        data = data[:, :6]
    elif fault == "dtype":
        data = data.astype(np.float64)
    else:
        record["actions"]["first_environment_action"][0] = 1.
    np.savez(actions, model_actions=data, environment_actions=data)
    record["actions"]["sha256"] = bundle._hash(actions)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Quantized (actions|first-action)"):
        bundle.frontier_probe_files(tmp_path)


def test_returned_but_postprocess_failed_forward_is_exportable_without_actions(int8_record, tmp_path):
    path, record = int8_record
    record.update(status="failed", forward_started=True, inference_executed=True, forward_calls=1,
                  error="synthetic postprocessing failure")
    path.write_text(json.dumps(record))
    _, audit = audit_for(tmp_path)
    assert audit["status"] == "failed" and audit["inference_executed"] is True
    assert audit["forward_calls"] == 1 and audit["successful_quantized_forward"] is False


@pytest.fixture
def legacy_native_record(tmp_path):
    current = tmp_path / "scripts/probe_vlajepa.py"
    current.parent.mkdir()
    current.write_text("# synthetic original native source\n")
    snapshot = tmp_path / "artifacts/frontier/native.source.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(current.read_bytes())
    record = {"status": "failed", "script_sha256": bundle._hash(snapshot),
              "script_snapshot": snapshot.relative_to(tmp_path).as_posix(),
              "timings": {"forward_seconds": 1.25}, "error": "synthetic postprocessing failure"}
    path = write_json(tmp_path, "artifacts/frontier/native.json", record)
    return path, record


@pytest.mark.parametrize("fault", [None, "missing_snapshot", "different_current_source", "missing_timing"])
def test_legacy_native_returned_stage_requires_current_source_binding(legacy_native_record, tmp_path, fault):
    path, record = legacy_native_record
    if fault == "missing_snapshot":
        record.pop("script_snapshot")
    elif fault == "different_current_source":
        (tmp_path / "scripts/probe_vlajepa.py").write_text("# different source\n")
    elif fault == "missing_timing":
        record["timings"] = {}
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    files, audits = bundle.frontier_probe_files(tmp_path)
    audit, = audits
    assert audit["status"] == "failed" and path.read_bytes() == before
    assert audit["inference_executed"] is (True if fault is None else None)
    assert audit.get("forward_returned_from_stage_evidence") is (True if fault is None else None)
    assert not any(item.name.endswith(".actions.npz") for item in files)


@pytest.mark.parametrize("status, expected", [("blocked_preflight", False), ("passed_schema_only", False),
                                              ("failed", None), ("running_real_forward", None)])
def test_legacy_native_missing_flag_is_not_assumed_false(legacy_native_record, tmp_path, status, expected):
    path, record = legacy_native_record
    record.update(status=status, timings={})
    path.write_text(json.dumps(record))
    _, audits = bundle.frontier_probe_files(tmp_path)
    assert audits[0]["inference_executed"] is expected


def test_explicit_native_returned_failure_does_not_require_success_artifacts(legacy_native_record, tmp_path):
    path, record = legacy_native_record
    record.update(inference_executed=True)
    path.write_text(json.dumps(record))
    _, audits = bundle.frontier_probe_files(tmp_path)
    assert audits[0]["status"] == "failed" and audits[0]["inference_executed"] is True
    record.update(status="passed_real_forward")
    record.pop("inference_executed")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks bound evidence"):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("fault", ["pending", "revision", "checkpoint_hash", "checkpoint_size", "missing_checkpoint", "incomplete_flag"])
def test_passed_int8_rejects_inconsistent_bound_download_identity(int8_record, tmp_path, fault):
    path, record = int8_record
    make_passed(tmp_path, path, record)
    inputs = tmp_path / record["manifest"]["snapshot_path"]
    manifest = json.loads(inputs.read_text())
    if fault == "pending":
        manifest["status"] = "pending"
    elif fault == "revision":
        manifest["policy_revision"] = "different"
    elif fault == "checkpoint_hash":
        manifest["files"][0]["sha256"] = "0" * 64
    elif fault == "checkpoint_size":
        manifest["files"][0]["bytes"] -= 1
    elif fault == "missing_checkpoint":
        manifest["files"] = []
    else:
        record["manifest"]["checkpoint_complete"] = False
    inputs.write_text(json.dumps(manifest))
    record["manifest"]["sha256"] = bundle._hash(inputs)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete pinned checkpoint"):
        bundle.frontier_probe_files(tmp_path)


@pytest.mark.parametrize("returned", [False, True])
def test_memory_guard_is_terminal_without_rewriting_last_stage(int8_record, tmp_path, returned):
    path, record = int8_record
    record.update(status="running_real_forward", forward_started=True,
                  inference_executed=returned, forward_calls=int(returned))
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    write_json(tmp_path, "artifacts/frontier/quantized.memory_stop.json", {"status": "stopped_memory_guard"})
    _, audit = audit_for(tmp_path)
    assert audit["status"] == "running_real_forward"
    assert audit["effective_status"] == "stopped_memory_guard"
    assert audit["memory_stop"]["sha256"] == bundle._hash(tmp_path / audit["memory_stop"]["path"])
    assert audit["successful_quantized_forward"] is False and path.read_bytes() == original


def test_memory_guard_and_pass_claim_are_inconsistent(int8_record, tmp_path):
    make_passed(tmp_path, *int8_record)
    write_json(tmp_path, "artifacts/frontier/quantized.memory_stop.json", {"status": "stopped_memory_guard"})
    with pytest.raises(ValueError, match="contradicts status"):
        bundle.frontier_probe_files(tmp_path)


@pytest.fixture
def continuation_validation(tmp_path, monkeypatch):
    counts = dict(zip(bundle.STAGES[:4], (40, 32, 128, 32), strict=True))
    counts.update(frontier_development=4, frontier_heldout=24)
    fingerprint = "a" * 64
    cases = [{"id": f"case-{i}", "init_state_id": 41 if i % 4 < 2 else 43,
              "method": "full10" if i % 2 == 0 else "noise2"} for i in range(16)]
    protocol = {"spec": {"cases": cases}, "fingerprint": fingerprint,
                "origin": {"v1_report": {"path": "old/report.json", "sha256": "b" * 64}}}
    study = {"complete": True, "protocol": protocol,
             "records": {c["id"]: {"case": c, "record": {}} for c in cases}}
    report = {"status": "complete", "counts": {"planned": 16, "complete": 16, "missing": 0, "unsealed": 0, "error": 0},
              "comparison": {"completed_pairs": 8, "observed_task_clusters": 4}}
    observed = []
    def verified(output, *, partial=False, root=None):
        assert output == tmp_path / "outputs/frontier/flow_confirmation" and partial is False and root == tmp_path
        observed.append("raw")
        return study
    def reported(output, *, partial=False, root=None):
        assert partial is False and root == tmp_path
        observed.append("report")
        return report
    monkeypatch.setitem(sys.modules, "run_flow_confirmation", SimpleNamespace(verify_study=verified))
    monkeypatch.setitem(sys.modules, "summarize_flow_confirmation", SimpleNamespace(validate_report=reported))
    monkeypatch.setattr(bundle, "validate_frontier", lambda root: (copy.deepcopy(counts), {"frontier_cache_study": "old"}))
    monkeypatch.setattr(bundle, "validate_episode_files", lambda *args: observed.append("trajectory"))
    doc = tmp_path / "docs/FLOW_CONFIRMATION_PROTOCOL.md"
    doc.parent.mkdir()
    doc.write_text("synthetic protocol")
    return study, report, counts, observed


def test_continuation_validates_both_complete_raw_and_recomputed_report(continuation_validation, tmp_path):
    _, _, old_counts, observed = continuation_validation
    counts, fingerprints, scope = bundle.validate_continuation(tmp_path)
    assert counts == {**old_counts, "flow_confirmation": 16}
    assert observed[:2] == ["raw", "report"] and observed.count("trajectory") == 16
    assert scope["episodes"] == 16 and scope["paired_initial_states"] == 8 and scope["task_clusters"] == 4
    assert scope["initial_state_ids"] == [41, 43]
    assert scope["protocol_fingerprint"] == fingerprints["flow_confirmation"]
    assert scope["hypothesis_origin"] and "not pooled" in scope["earlier_evidence_role"]


@pytest.mark.parametrize("fault", ["partial_raw", "partial_report", "missing_case", "pairs", "clusters", "old_initial", "wrong_method"])
def test_continuation_rejects_incomplete_or_wrong_scope(continuation_validation, tmp_path, fault):
    study, report, _, _ = continuation_validation
    if fault == "partial_raw":
        study["complete"] = False
    elif fault == "partial_report":
        report["status"] = "PARTIAL"
    elif fault == "missing_case":
        study["records"].pop("case-0")
    elif fault == "pairs":
        report["comparison"]["completed_pairs"] = 7
    elif fault == "clusters":
        report["comparison"]["observed_task_clusters"] = 8
    elif fault == "old_initial":
        study["protocol"]["spec"]["cases"][0]["init_state_id"] = 11
    else:
        study["protocol"]["spec"]["cases"][0]["method"] = "cache2"
    with pytest.raises(ValueError, match="complete 16-episode"):
        bundle.validate_continuation(tmp_path)


def test_continuation_bundle_preserves_old_zip_and_separate_scopes(int8_record, tmp_path, monkeypatch):
    for name in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml",
                 "artifacts/continuation-unit-tests.xml", "artifacts/continuation-targeted-tests.xml",
                 "artifacts/flow-replay-tests.xml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<testsuite tests="1"/>')
    new_report = tmp_path / "outputs/frontier/flow_confirmation/report.md"
    new_report.parent.mkdir(parents=True)
    new_report.write_text("synthetic report")
    doc = tmp_path / "docs/FLOW_CONFIRMATION_PROTOCOL.md"
    doc.parent.mkdir()
    doc.write_text("synthetic protocol")
    old_zip = tmp_path / "deliverables/feedbackbench-frontier.zip"
    old_zip.parent.mkdir()
    old_zip.write_bytes(b"existing v1 archive fixture; preserve exactly")
    old_hash = bundle._hash(old_zip)
    scope = {"episodes": 16, "paired_initial_states": 8, "task_clusters": 4,
             "initial_state_ids": [41, 43], "protocol_fingerprint": "frozen",
             "hypothesis_origin": {"v1_report_sha256": "a" * 64}}
    counts = {"baseline": 232, "frontier_development": 4, "frontier_heldout": 24, "flow_confirmation": 16}
    monkeypatch.setattr(bundle, "REQUIRED_FILES", ())
    monkeypatch.setattr(bundle, "validate_continuation", lambda root: (counts, {"flow_confirmation": "frozen"}, scope))
    replay_name = "outputs/replays/flow-spatial43-noise2/flow_replay_audit.json"
    write_json(tmp_path, replay_name, {"status": "failed", "error": "synthetic diagnostic failure"})
    destination = tmp_path / "deliverables/feedbackbench-continuation.zip"
    bundle.package(tmp_path, destination, continuation=True)
    assert bundle._hash(old_zip) == old_hash
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("feedbackbench/BUNDLE_MANIFEST.json"))
        assert manifest["schema_version"] == 4 and manifest["episode_counts_by_stage"] == counts
        assert manifest["continuation_scope"] == scope
        assert manifest["separate_scopes"] == {"minerva_baseline": 232, "action_cache_v1": 28, "flow_confirmation": 16,
                                               "pooled_success_rate": "not defined across studies or models"}
        assert manifest["unfinished_original_plan"]["50_smolvla"] == 12
        assert "artifacts/continuation-unit-tests.xml" in manifest["files"]
        assert "artifacts/continuation-targeted-tests.xml" in manifest["files"]
        assert "artifacts/flow-replay-tests.xml" in manifest["files"]
        assert "docs/FLOW_CONFIRMATION_PROTOCOL.md" in manifest["files"]
        assert "outputs/frontier/flow_confirmation/report.md" in manifest["files"]
        assert manifest["probe_evidence"][0]["status"] == "blocked_preflight"
        assert replay_name in manifest["files"]
        diagnostic, = manifest["continuation_diagnostics"]
        assert diagnostic["recorded_status"] == "failed" and diagnostic["exact_replay_verified"] is False
        assert diagnostic["new_scored_episodes"] == 0
        for name, entry in manifest["files"].items():
            data = archive.read("feedbackbench/" + name)
            assert len(data) == entry["bytes"]
            assert bundle.hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert archive.testzip() is None
    with pytest.raises(FileExistsError):
        bundle.package(tmp_path, destination, continuation=True)


def test_new_compatibility_redacts_only_archive_copy():
    personal = "C:" + "/" + "/".join(["Users", "private", "project"])
    raw = json.dumps({"runtime": {"executable": personal, "hostname": "private-host"}}).encode()
    content, changed = bundle.archive_content("outputs/frontier/flow_confirmation/compatibility.json", raw)
    assert changed and b"private-host" not in content and b"private/" not in content
    assert json.loads(raw)["runtime"]["hostname"] == "private-host"


def test_old_frontier_scope_does_not_silently_include_unvalidated_continuation(tmp_path, monkeypatch):
    for name in ("artifacts/cache-sampler-real-parity.xml", "artifacts/cache-study-unit-tests.xml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<testsuite tests="1"/>')
    extra = tmp_path / "outputs/frontier/flow_confirmation/result.json"
    extra.parent.mkdir(parents=True)
    extra.write_text('{"status":"unvalidated"}')
    replay = tmp_path / "outputs/replays/flow-spatial43-noise2/attempt.json"
    replay.parent.mkdir(parents=True)
    replay.write_text('{"status":"running"}')
    old_replay = tmp_path / "outputs/replays/FA001/replay_audit.json"
    old_replay.parent.mkdir(parents=True)
    old_replay.write_text('{"status":"preserved fixture"}')
    monkeypatch.setattr(bundle, "REQUIRED_FILES", ())
    monkeypatch.setattr(bundle, "validate_frontier", lambda root: ({}, {}))
    destination = tmp_path / "only-v1.zip"
    bundle.package(tmp_path, destination, frontier=True)
    with zipfile.ZipFile(destination) as archive:
        assert not any("flow_confirmation" in name for name in archive.namelist())
        assert not any("flow-spatial43-noise2" in name for name in archive.namelist())
        assert "feedbackbench/outputs/replays/FA001/replay_audit.json" in archive.namelist()


def test_optional_flow_replay_absent(tmp_path):
    assert bundle.validate_optional_flow_replay(tmp_path) == []


@pytest.mark.parametrize("status", ["failed", "missing_audit"])
def test_optional_flow_replay_preserves_unverified_failure(tmp_path, monkeypatch, status):
    def forbidden(*args, **kwargs):
        pytest.fail("A failed/incomplete replay must not be claimed as an exact replay")

    monkeypatch.setitem(sys.modules, "replay_flow_episode", SimpleNamespace(validate_replay=forbidden))
    relative = "outputs/replays/flow-spatial43-noise2"
    write_json(tmp_path, f"{relative}/attempt.json", {"kind": "synthetic diagnostic attempt"})
    if status == "failed":
        write_json(tmp_path, f"{relative}/flow_replay_audit.json", {"status": "failed", "error": "fixture"})
    entry, = bundle.validate_optional_flow_replay(tmp_path)
    assert entry["recorded_status"] == status and entry["exact_replay_verified"] is False
    assert entry["excluded_from_benchmark_counts"] is True and entry["new_scored_episodes"] == 0
    assert "new_policy_inference_calls" not in entry
    assert any("no trajectory" in limitation for limitation in entry["limitations"])
    assert ("audit_sha256" in entry) == (status == "failed")


def test_optional_flow_replay_pass_requires_readonly_validator(tmp_path, monkeypatch):
    output = tmp_path / "outputs/replays/flow-spatial43-noise2"
    audit = {"status": "passed_exact_replay", "original": {"episode_id": "synthetic"},
             "sources_sha256": {"scripts/replay_flow_episode.py": "a" * 64},
             "new_policy_inference_calls": 0}
    write_json(tmp_path, "outputs/replays/flow-spatial43-noise2/flow_replay_audit.json", audit)
    calls = []

    def validate(path, *, root):
        calls.append((path, root))
        return audit

    monkeypatch.setitem(sys.modules, "replay_flow_episode", SimpleNamespace(validate_replay=validate))
    entry, = bundle.validate_optional_flow_replay(tmp_path)
    assert calls == [(output, tmp_path)]
    assert entry["exact_replay_verified"] is True and entry["original"] == audit["original"]
    assert entry["new_policy_inference_calls"] == 0 and entry["new_scored_episodes"] == 0

    def tampered(*args, **kwargs):
        raise ValueError("Replay evidence differs")

    monkeypatch.setitem(sys.modules, "replay_flow_episode", SimpleNamespace(validate_replay=tampered))
    with pytest.raises(ValueError, match="Replay evidence differs"):
        bundle.validate_optional_flow_replay(tmp_path)
