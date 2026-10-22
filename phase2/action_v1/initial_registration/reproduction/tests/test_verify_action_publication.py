"""Small synthetic publication audits: no model, simulator, or private checkout."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_action_publication.py"
SPEC = importlib.util.spec_from_file_location("standalone_publication_test", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def synthetic_snapshot(*, complete=False):
    """An internally bound miniature protocol; synthetic numbers are never study evidence."""
    contents, records = {}, {}

    def add(name, value, *, category="fixture", source_sha=None, projection=False):
        raw = value if isinstance(value, bytes) else verifier.json_bytes(value)
        contents[name] = raw
        records[name] = {"sha256": verifier.sha(raw), "bytes": len(raw), "category": category,
                         "origin": None, "source_sha256": source_sha or verifier.sha(raw), "projection": projection}
        return {"path": name, "sha256": verifier.sha(raw)}

    def project(prefix, source_name, included):
        source = json.loads(contents[source_name])
        members = {name: next((member for member, record in source["files"].items()
                               if (record["sha256"] if isinstance(record, dict) else record) == records[name]["source_sha256"]), None)
                   for name in included}
        represented = {member for member in members.values() if member is not None}
        add(prefix + "/PUBLIC_PROJECTION.json", {"source_receipt": source_name,
            "source_receipt_sha256": verifier.sha(contents[source_name]), "source_binding": source["binding"],
            "original_full_seal_verifiable_from_public_subset": False,
            "included": {name: records[name] for name in included}, "source_receipt_members": members,
            "omitted_source_files": sorted(set(source["files"]) - represented)})

    code = add("source/scripts/model.py", b"# synthetic source\n")
    add("source_index.json", {"scripts/model.py": code["path"]})
    sources = add("data_sources.json", {"sources": []})
    data = add("data_manifest.json", {"demos": [], "source_manifest": sources})
    schedule = add("schedules/seed17.json", {"train": [], "validation": []})
    probe = {"path": "original/probe.json", "sha256": "1" * 64}
    alignment = {"path": "original/alignment.json", "sha256": "2" * 64}
    add("prerequisites/probe.json", {"status": "passed"}, source_sha=probe["sha256"], projection=True)
    add("prerequisites/alignment.json", {"status": "passed"}, source_sha=alignment["sha256"], projection=True)
    failed = add("prerequisites/alignment_history/r00.json", {"status": "needs_attention"}, source_sha="3" * 64, projection=True)
    add("prerequisites/alignment_history/index.json", {"records": {"3" * 64: {
        "public_path": failed["path"], "public_sha256": failed["sha256"], "status": "needs_attention",
        "source_code": code["path"], "source_code_sha256": code["sha256"]}}})
    old_protocol = {"development_only": True, "source_files": {"scripts/model.py": code["sha256"]}}
    pipeline = {key: add(f"prerequisites/pipeline/{key}.json", value) for key, value in (
        ("protocol", old_protocol), ("report", {"status": "complete"}), ("trained_manifest", {}))}
    pipeline["excluded_from_formal_results"] = True
    add("prerequisites/pipeline/source_index.json", {"scripts/model.py": "source/000_model.py"})
    add("prerequisites/pipeline/source/000_model.py", b"# synthetic source\n")
    development_protocol = add("prerequisites/development/protocol.json", {"development_only": True})
    development_report = add("prerequisites/development/report.json", {"status": "complete", "excluded_from_formal_study": True})
    development_receipt = add("prerequisites/development/source_receipt.json", {
        "binding": {"evaluation_protocol_sha256": development_protocol["sha256"]},
        "files": {"report.json": {"sha256": development_report["sha256"]}, "private.log": {"sha256": "4" * 64}}})
    project("prerequisites/development", development_receipt["path"], [development_report["path"]])
    development = {"protocol": development_protocol, "report": development_report, "receipt": development_receipt,
                   "excluded_from_formal_results": True, "success_was_not_a_gate": True}
    case = {"case_id": "synthetic_case", "suite": "libero_spatial", "task_uid": "libero_spatial:0",
            "initial_state_id": 5, "method_id": "frozen", "condition": "clean"}
    run_id = "output_head_s17"
    run = {"mode": "output_head", "steps": 2, "initialization_seed": 17, "schedule": schedule}
    protocol = {"frozen": True, "study_id": "action-adaptation-v1", "source_files": {"scripts/model.py": code["sha256"]},
        "data": {"manifest": data, "alignment_evidence": alignment}, "probe_evidence": probe,
        "pipeline_evidence": pipeline, "development_evaluation_evidence": development, "runs": {run_id: run},
        "evaluation": {"methods": [{"method_id": "frozen"}], "cases": [case], "horizons": {"libero_spatial": 280}}}
    protocol_record = add("protocol.json", protocol)
    protocol_sha = protocol_record["sha256"]
    add("protocol.sha256", (protocol_sha + "\n").encode())
    index = {"training": {}, "evaluation": {}}
    receipt_refs, trained_sha = [], None
    if complete:
        prefix = "training/" + run_id
        binding = {"protocol_sha256": protocol_sha, "run_id": run_id, "schedule_sha256": schedule["sha256"],
                   "data_manifest_sha256": data["sha256"], "alignment_sha256": alignment["sha256"]}
        binding_sha = verifier.canonical_hash(binding)
        tensors = {"weight": np.zeros((32, 480), dtype=np.float32), "bias": np.zeros(32, dtype=np.float32)}
        adapter = {"schema": "feedbackbench.action_output_head.v1", "base_fingerprint": "5" * 64, "seed": 17,
            "trainable_parameters": 15392, "tensor_sha256": {name: verifier.sha(array.tobytes()) for name, array in tensors.items()},
            "metadata": {"protocol_sha256": protocol_sha, "run_id": run_id, "binding_sha256": binding_sha, "step": 2}}
        stream = io.BytesIO()
        np.savez_compressed(stream, **tensors, manifest_utf8=np.frombuffer(verifier.json_bytes(adapter), dtype=np.uint8))
        adapter_record = add(prefix + "/adapter.npz", stream.getvalue(), category="final_adapter")
        checkpoint = {"binding": binding, "binding_sha256": binding_sha, "completed_steps": 2,
                      "files": {"adapter.npz": adapter_record["sha256"], "optimizer.npz": "6" * 64,
                                "rng.npz": "7" * 64, "state.json": "8" * 64, "history.json": "9" * 64}}
        checkpoint["receipt_sha256"] = verifier.canonical_hash(checkpoint)
        checkpoint_record = add(prefix + "/source_receipt.json", checkpoint)
        summary_record = add(prefix + "/summary.json", {"base_fingerprint": "5" * 64, "completed_steps": 2,
                             "frozen_parameters_unchanged": True, "closed_loop_success_claim": False}, source_sha="a" * 64, projection=True)
        index["training"][run_id] = {"adapter": adapter_record["path"], "source_receipt": checkpoint_record["path"], "summary": summary_record["path"]}
        project(prefix, checkpoint_record["path"], [adapter_record["path"], summary_record["path"]])
        trained = {"protocol_sha256": protocol_sha, "runs": {run_id: {"base_fingerprint": "5" * 64,
            "receipt": {"sha256": checkpoint_record["sha256"]}, "adapter": {"sha256": adapter_record["sha256"]},
            "report": {"sha256": "a" * 64}}}}
        trained_sha = add("source_trained_manifest.json", trained)["sha256"]
        result = {"status": "complete", "case": case, "official_success": True, "steps": 2, "horizon": 280,
                  "terminated": True, "truncated": False, "decision_count": 1, "NFE": 10, "elapsed_seconds": .5,
                  "inference_seconds": .25, "protocol_sha256": protocol_sha, "trained_manifest_sha256": trained_sha}
        result_ref = add("evaluation/c000/report.json", result)
        trace, calls = simulation_bytes(result)
        trace_ref = add("evaluation/c000/actions.npz", trace, category="sealed_simulation_trace")
        calls_ref = add("evaluation/c000/calls.jsonl", calls)
        seal = {"binding": {"protocol_sha256": protocol_sha, "trained_manifest_sha256": trained_sha, "case_id": case["case_id"]},
                "files": {name: {"sha256": record["sha256"]} for name, record in (
                    ("report.json", result_ref), ("actions.npz", trace_ref), ("calls.jsonl", calls_ref))}}
        seal["files"]["private.log"] = {"sha256": "b" * 64}
        seal_ref = add("evaluation/c000/source_receipt.json", seal)
        project("evaluation/c000", seal_ref["path"], [result_ref["path"], trace_ref["path"], calls_ref["path"]])
        index["evaluation"][case["case_id"]] = {"report": result_ref["path"], "actions": trace_ref["path"],
                                                "calls": calls_ref["path"], "source_receipt": seal_ref["path"]}
        receipt_refs = [{"path": "private/case/receipt.json", "sha256": seal_ref["sha256"]}]
    count = int(complete)
    rates = [{"method_id": "frozen", "condition": condition, "completed": count if condition == "clean" else 0,
              "planned": int(condition == "clean"), "successes": count if condition == "clean" else 0, "failures": 0,
              "observed_success_fraction": 1.0 if complete and condition == "clean" else None,
              "coverage_complete": complete or condition != "clean", "elapsed_seconds": .5 if complete and condition == "clean" else 0,
              "inference_seconds": .25 if complete and condition == "clean" else 0} for condition in ("clean", "brightness075")]
    report = {"protocol_sha256": protocol_sha, "trained_manifest_sha256": trained_sha,
              "training_completed": count, "training_planned": 1, "evaluation_completed": count, "evaluation_planned": 1,
              "status": "complete" if complete else "partial", "missing_cases": [] if complete else [case["case_id"]],
              "sealed_receipts": receipt_refs, "rates": rates,
              "contrasts": {f"lora_minus_{right}_{condition}": {"status": "pending_complete_pairs", "estimate": None}
                            for right in ("frozen", "output_head") for condition in ("clean", "brightness075")}}
    report_record = add("report.json", report)
    add("evidence_index.json", index)
    manifest = {"schema": verifier.SCHEMA, "snapshot_id": "synthetic", "namespace": "irrelevant/original/location",
                "protocol_sha256": protocol_sha, "report_sha256": report_record["sha256"], "status": report["status"],
                "files": copy.deepcopy(records)}
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    return contents


def simulation_bytes(result):
    stream = io.BytesIO()
    np.savez_compressed(stream, actions=np.zeros((2, 7), dtype=np.float32), states=np.zeros((2, 8), dtype=np.float32),
                        next_states=np.zeros((2, 8), dtype=np.float32), rewards=np.array([0, 1], dtype=np.float32),
                        terminated=np.array([False, True]), truncated=np.array([False, False]))
    case = result["case"]
    token = f"action-eval-v1:{case['task_uid']}:{case['initial_state_id']}:0"
    call = {"decision": 0, "start_step": 0, "executed_steps": 2, "actual_nfe": 10,
            "noise_seed": int.from_bytes(verifier.hashlib.sha256(token.encode()).digest()[:4], "little"),
            "inference_seconds": .25, "preprocess_seconds": .01, "postprocess_seconds": .01, "decision_wall_seconds": .5}
    return stream.getvalue(), (json.dumps(call) + "\n").encode()


def write_snapshot(directory, contents):
    directory.mkdir()
    for name, content in contents.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return verifier.sha(contents["PUBLICATION_MANIFEST.json"])


def reseal(contents, name, value):
    contents[name] = verifier.json_bytes(value)
    manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
    manifest["files"][name].update(sha256=verifier.sha(contents[name]), bytes=len(contents[name]))
    if name == "report.json":
        manifest["report_sha256"] = verifier.sha(contents[name])
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)


@pytest.mark.parametrize("complete", [False, True])
def test_relocated_snapshot_verifies_without_checkout(tmp_path, complete):
    contents = synthetic_snapshot(complete=complete)
    path = tmp_path / "arbitrary-directory"
    expected = write_snapshot(path, contents)
    before = {p.relative_to(path): (p.stat().st_mtime_ns, p.read_bytes()) for p in path.rglob("*") if p.is_file()}
    result = verifier.verify_snapshot(path, expected)
    assert result["study_status"] == ("complete" if complete else "partial")
    assert result["evaluation_completed"] == int(complete)
    assert result["writes"] == 0 and not result["original_full_private_seals_verified"]
    assert not result["historical_git_checkout_verified"]
    assert before == {p.relative_to(path): (p.stat().st_mtime_ns, p.read_bytes()) for p in path.rglob("*") if p.is_file()}


def test_cli_is_standalone_and_partial_is_not_complete(tmp_path):
    path = tmp_path / "snapshot"
    expected = write_snapshot(path, synthetic_snapshot())
    command = [sys.executable, "-B", str(SCRIPT), "--snapshot", str(path), "--manifest-sha256", expected]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["model_loaded"] is False
    refused = subprocess.run(command + ["--require-complete"], cwd=tmp_path, capture_output=True, text=True)
    assert refused.returncode == 2 and "partial" in refused.stderr


@pytest.mark.parametrize("mutation", ["missing", "extra", "bytes", "wrong_manifest"])
def test_inventory_or_external_hash_change_is_rejected(tmp_path, mutation):
    path = tmp_path / "snapshot"
    expected = write_snapshot(path, synthetic_snapshot())
    if mutation == "missing":
        (path / "source/scripts/model.py").unlink()
    elif mutation == "extra":
        (path / "unexpected.txt").write_text("extra")
    elif mutation == "bytes":
        (path / "source/scripts/model.py").write_text("changed")
    else:
        expected = "0" * 64
    with pytest.raises((ValueError, OSError)):
        verifier.verify_snapshot(path, expected)


def test_path_traversal_is_rejected_before_outside_read(tmp_path):
    contents = synthetic_snapshot()
    manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
    manifest["files"]["../outside.json"] = {"sha256": "0" * 64, "bytes": 2}
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    path = tmp_path / "snapshot"
    expected = write_snapshot(path, contents)
    with pytest.raises(ValueError, match="Unsafe"):
        verifier.verify_snapshot(path, expected)


@pytest.mark.parametrize("mutation", ["claim_full_seal", "binding", "omitted", "missing_projection"])
def test_projection_claims_cannot_be_forged(mutation):
    contents = synthetic_snapshot()
    name = "prerequisites/development/PUBLIC_PROJECTION.json"
    value = json.loads(contents[name])
    if mutation == "claim_full_seal":
        value["original_full_seal_verifiable_from_public_subset"] = True
    elif mutation == "binding":
        value["source_binding"] = {}
    elif mutation == "omitted":
        value["omitted_source_files"] = []
    else:
        del contents[name]
        manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])
        del manifest["files"][name]
        contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
        with pytest.raises(ValueError, match="projection"):
            verifier.validate_payload(contents)
        return
    reseal(contents, name, value)
    with pytest.raises(ValueError):
        verifier.validate_payload(contents)


@pytest.mark.parametrize("field,value", [("status", "complete"), ("evaluation_completed", 1), ("missing_cases", [])])
def test_partial_evidence_cannot_claim_completion(field, value):
    contents = synthetic_snapshot()
    report = json.loads(contents["report.json"])
    report[field] = value
    reseal(contents, "report.json", report)
    with pytest.raises(ValueError):
        verifier.validate_payload(contents)


def test_reported_success_rate_cannot_override_numeric_evidence():
    contents = synthetic_snapshot(complete=True)
    report = json.loads(contents["report.json"])
    report["rates"][0]["successes"] = 0
    reseal(contents, "report.json", report)
    with pytest.raises(ValueError, match="outcomes"):
        verifier.validate_payload(contents)


@pytest.mark.parametrize("field,value", [("noise_seed", 42), ("actual_nfe", 9), ("executed_steps", 1)])
def test_numeric_trace_and_call_accounting(field, value):
    result = {"case": {"task_uid": "libero_spatial:0", "initial_state_id": 5}, "steps": 2,
              "terminated": True, "truncated": False, "decision_count": 1, "NFE": 10, "inference_seconds": .25}
    trace, calls = simulation_bytes(result)
    call = json.loads(calls)
    call[field] = value
    with pytest.raises(ValueError):
        verifier.inspect_simulation(trace, json.dumps(call).encode(), result)


def test_task_bootstrap_matches_declared_four_task_estimator():
    results = []
    methods = ["frozen"] + [f"{family}_s{seed}" for seed in (17, 23, 47) for family in ("lora", "output_head")]
    for suite in verifier.SUITES:
        for initial in (5, 15, 25, 35, 45):
            for condition in ("clean", "brightness075"):
                for method in methods:
                    results.append({"case": {"suite": suite, "initial_state_id": initial, "condition": condition, "method_id": method},
                                    "official_success": method.startswith("lora")})
    contrasts = {}
    for right in ("frozen", "output_head"):
        for condition in ("clean", "brightness075"):
            contrasts[f"lora_minus_{right}_{condition}"] = {"status": "complete_pairs", "estimate": 1.0,
                "descriptive_task_bootstrap_95": [1.0, 1.0], "task_groups": 4, "initial_states_per_task": 5,
                "training_seeds": 3, "per_task": {suite: 1.0 for suite in verifier.SUITES},
                "paired_differences": [{"suite": suite, "initial_state_id": initial, "seed": seed, "difference": 1}
                                       for suite in verifier.SUITES for initial in (5, 15, 25, 35, 45) for seed in (17, 23, 47)]}
    verifier.verify_contrasts(results, contrasts)
    contrasts["lora_minus_frozen_clean"]["task_groups"] = 60
    with pytest.raises(ValueError, match="task-bootstrap"):
        verifier.verify_contrasts(results, contrasts)


def test_later_source_release_is_separate_and_bound():
    contents = synthetic_snapshot()
    manifest = json.loads(contents["PUBLICATION_MANIFEST.json"])

    def add(name, value, category):
        raw = value if isinstance(value, bytes) else verifier.json_bytes(value)
        contents[name] = raw
        manifest["files"][name] = {"sha256": verifier.sha(raw), "bytes": len(raw), "category": category}
        return verifier.sha(raw)

    original = "scripts/new_reproduction.py"
    target = "reproduction/" + original
    code_sha = add(target, b"# added after the experiment\n", "supplemental_source")
    release_path = "reproduction/docs/release.json"
    release_sha = add(release_path, {"schema": "feedbackbench.reproduction_source_release.v1",
        "postdates_formal_protocol_freeze": True, "executed_in_archived_development_run": False,
        "modifies_existing_frozen_protocols": False, "files": {original: code_sha},
        "byte_sizes": {original: len(contents[target])}}, "supplemental_release_manifest")
    source = {"schema": "feedbackbench.action_publication_supplements.v1", "outside_formal_source_inventory": True,
        "executed_in_archived_training_or_development": False,
        "groups": {"public_reproduction": {"release_manifest": release_path, "release_manifest_sha256": release_sha,
                   "files": {original: {"target": target, "sha256": code_sha}}}, "public_verifier": {"files": {}}}}
    manifest["supplemental_sources_sha256"] = add("supplemental_sources.json", source, "supplemental_release_index")
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    assert verifier.validate_payload(contents)["supplemental_sources_sha256"]
    source["executed_in_archived_training_or_development"] = True
    manifest["supplemental_sources_sha256"] = add("supplemental_sources.json", source, "supplemental_release_index")
    contents["PUBLICATION_MANIFEST.json"] = verifier.json_bytes(manifest)
    with pytest.raises(ValueError, match="original execution"):
        verifier.validate_payload(contents)
