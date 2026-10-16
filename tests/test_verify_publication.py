"""Portable publication evidence checks using tiny, private-free fixtures."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publication_verifier_tested", ROOT / "scripts/verify_publication.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)

STUDY = "outputs/frontier/vlajepa_isolated_closed_loop"
CASE = "vlajepa_isolated__libero_spatial__1__41__clean__native4_chunk7"
PROTOCOL = f"{STUDY}/protocol.json"
WORKER = f"{STUDY}/workers/worker_0001.json"
RUN = f"{STUDY}/runs/run_0001.json"
EPISODE = f"{STUDY}/episodes/{CASE}"
RECEIPT = f"{EPISODE}/receipt.json"
RESULT = f"{EPISODE}/result.json"
EVIDENCE = f"{EPISODE}/trajectory.npz"
SOURCE = "scripts/run_vlajepa_isolated.py"
REDACTION = {"kind": "compatibility_patch_local_path", "fields": ["compatibility.patches[0].changes[0].after"]}
FINGERPRINT = "f" * 64


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Publication:
    def __init__(self, root):
        self.root = root
        self.manifest = {"schema": "feedbackbench.github_publication.v1", "files": {},
                         "frozen_source_checks": {PROTOCOL: 1}}

    def put(self, path, value, *, original=None, redaction=None):
        value = value if isinstance(value, bytes) else encode(value)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        original = value if original is None else original
        metadata = {"sha256": digest(value), "bytes": len(value), "source_sha256": digest(original),
                    "source_bytes": len(original), "redacted": original != value}
        if redaction is not None:
            metadata["redaction"] = deepcopy(redaction)
        self.manifest["files"][path] = metadata
        return metadata

    def read(self, path):
        return json.loads((self.root / path).read_text(encoding="utf-8"))

    def reseal(self):
        self.put(RECEIPT, {"files": {Path(path).name: self.manifest["files"][path]["sha256"]
                                     for path in (RESULT, EVIDENCE)}, "sealed_at": "test"})

    def save(self):
        (self.root / "PUBLICATION_MANIFEST.json").write_bytes(encode(self.manifest))

    def verify(self):
        self.save()
        return entry.verify_publication(self.root)


@pytest.fixture
def publication(tmp_path):
    pub = Publication(tmp_path)
    source = pub.put(SOURCE, b"# Frozen minimal source fixture.\n")
    case = {"id": CASE, "model": "vlajepa_isolated", "suite": "libero_spatial", "task_id": 1,
            "init_state_id": 41, "seed": 1000, "split": "isolated_closed_loop_feasibility"}
    pub.put(PROTOCOL, {"schema": "feedbackbench.vlajepa_closed_loop_protocol.v1", "fingerprint": FINGERPRINT,
                       "sources": {SOURCE: source["sha256"]}, "spec": {"cases": [case]}})
    run = {"status": "complete", "protocol_fingerprint": FINGERPRINT,
           "compatibility": {"patches": [{"changes": [{"after": "<REDACTED_LOCAL_PATH>"}]}]}}
    original = deepcopy(run)
    original["compatibility"]["patches"][0]["changes"][0]["after"] = "C:/fixture-only/compatibility.py"
    run_meta = pub.put(RUN, run, original=encode(original), redaction=REDACTION)
    pub.put(RESULT, {**case, "status": "complete", "success": True,
                    "protocol_fingerprint": FINGERPRINT, "model_info": {"run_record": RUN}})
    pub.put(EVIDENCE, b"tiny fake trajectory bytes: no simulator required\n")
    pub.reseal()
    pub.put(WORKER, {"schema": "feedbackbench.vlajepa_isolated_worker.v1", "status": "complete",
                     "pid": 1234, "process_create_time": 1000.0, "started_at": "test-start",
                     "finished_at": "test-finish", "protocol_fingerprint": FINGERPRINT,
                     "script_sha256": source["sha256"], "case_id": CASE,
                     "completed_before": 0, "completed_after": 1,
                     "statuses_before": {CASE: "missing"}, "statuses_after": {CASE: "complete"},
                     "newly_completed": [CASE],
                     "run_records": [{"path": RUN, "sha256": run_meta["source_sha256"]}]})
    return pub


def test_import_requires_only_standard_library():
    code = ("import runpy,sys;runpy.run_path(sys.argv[1]);"
            "assert not ({'torch','numpy','mujoco','robosuite','libero'} & sys.modules.keys())")
    completed = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/verify_publication.py")],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("study", ["vlajepa_isolated_closed_loop", "vlajepa_long_horizon", "vlajepa_recovery"])
def test_sanitizer_changes_only_declared_patch_path_and_keeps_input_bytes(study):
    private = "/".join(("C:", "Users", "Fixture", "local", "config.py"))
    diagnostic = {"status": "complete", "memory": {"peak_gib": 4.75},
                  "compatibility": {"patches": [{"changes": [
                      {"before": "configuration = default", "after": f"configuration = '{private}'"},
                      {"before": "count = 1", "after": "count = 2"}]}]}}
    raw = encode(diagnostic)
    unchanged_raw = bytes(raw)
    cleaned, redacted, descriptor = entry.redact_run_diagnostic(f"outputs/frontier/{study}/runs/run_0001.json", raw)
    expected = deepcopy(diagnostic)
    expected["compatibility"]["patches"][0]["changes"][0]["after"] = "configuration = '<REDACTED_LOCAL_PATH>'"
    assert json.loads(cleaned) == expected
    assert raw == unchanged_raw
    assert redacted is True
    assert descriptor == REDACTION


def test_sanitizer_returns_exact_original_bytes_when_no_path_changed():
    raw = b'{ "status" : "complete", "compatibility" : {"patches": []} }\n\n'
    assert entry.redact_run_diagnostic(RUN, raw) == (raw, False, None)


@pytest.mark.parametrize("field", ["other_diagnostic", "compatibility.patches[0].changes[0].before"])
def test_sanitizer_rejects_private_paths_outside_allowed_after_field(field):
    private = "/".join(("C:", "Users", "Fixture", "local", "config.py"))
    if field == "other_diagnostic":
        diagnostic = {field: private}
    else:
        diagnostic = {"compatibility": {"patches": [{"changes": [{"before": private, "after": "clean"}]}]}}
    with pytest.raises(ValueError):
        entry.redact_run_diagnostic(RUN, encode(diagnostic))


def test_sanitizer_rejects_run_files_outside_allowlist():
    with pytest.raises(ValueError):
        entry.redact_run_diagnostic("outputs/frontier/vlajepa_compact_closed_loop/runs/run_0001.json", b"{}")


def test_valid_redacted_diagnostic_preserves_original_worker_binding(publication):
    report = publication.verify()
    assert report["status"] == "passed"
    assert report["files_verified"] == len(publication.manifest["files"])
    assert report["frozen_source_checks"] == {PROTOCOL: 1}
    assert report["receipt_files_verified"] == 2
    binding = report["isolated_worker_run_bindings"][0]
    assert binding["path"] == RUN
    assert binding["mode"] == "source_binding_via_redaction_manifest"
    assert binding["expected_sha256"] == publication.manifest["files"][RUN]["source_sha256"]
    assert binding["exported_sha256"] == publication.manifest["files"][RUN]["sha256"]
    assert report["redacted_diagnostics"][0]["path"] == RUN


def test_recovery_run_diagnostic_redaction_is_validated_and_reported(publication):
    recovery_run = "outputs/frontier/vlajepa_recovery/runs/run_0001.json"
    diagnostic = publication.read(RUN)
    original = deepcopy(diagnostic)
    original["compatibility"]["patches"][0]["changes"][0]["after"] = "C:/fixture-only/compatibility.py"
    publication.put(recovery_run, diagnostic, original=encode(original), redaction=REDACTION)
    report = publication.verify()
    assert report["status"] == "passed"
    assert {row["path"] for row in report["redacted_diagnostics"]} == {RUN, recovery_run}


def test_unredacted_worker_binding_is_verified_by_exported_bytes(publication):
    publication.put(RUN, {"status": "complete", "protocol_fingerprint": FINGERPRINT})
    worker = publication.read(WORKER)
    worker["run_records"][0]["sha256"] = publication.manifest["files"][RUN]["sha256"]
    publication.put(WORKER, worker)
    report = publication.verify()
    assert report["isolated_worker_run_bindings"][0]["mode"] == "byte_verified"
    assert report["redacted_diagnostics"] == []


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 1)])
def test_manifest_exported_hash_and_size_are_verified(publication, field, value):
    publication.manifest["files"][EVIDENCE][field] = value
    with pytest.raises(ValueError):
        publication.verify()


def test_worker_original_source_binding_is_checked_beyond_manifest(publication):
    worker = publication.read(WORKER)
    worker["run_records"][0]["sha256"] = "0" * 64
    publication.put(WORKER, worker)  # Top-level hashes remain internally valid.
    with pytest.raises(ValueError):
        publication.verify()


@pytest.mark.parametrize("change", ["missing_descriptor", "missing_fields", "wrong_kind", "other_field"])
def test_redaction_requires_exact_explicit_descriptor(publication, change):
    meta = publication.manifest["files"][RUN]
    if change == "missing_descriptor":
        meta.pop("redaction")
    elif change == "missing_fields":
        meta["redaction"].pop("fields")
    elif change == "wrong_kind":
        meta["redaction"]["kind"] = "arbitrary_json"
    else:
        meta["redaction"]["fields"] = ["status"]
    with pytest.raises(ValueError):
        publication.verify()


def test_redaction_source_binding_is_forbidden_outside_run_whitelist(publication):
    old_path = "outputs/frontier/vlajepa_compact_closed_loop/runs/run_0001.json"
    old_meta = publication.manifest["files"].pop(RUN)
    old_target = publication.root / old_path
    old_target.parent.mkdir(parents=True)
    (publication.root / RUN).rename(old_target)
    publication.manifest["files"][old_path] = old_meta
    worker = publication.read(WORKER)
    worker["run_records"][0]["path"] = old_path
    publication.put(WORKER, worker)
    result = publication.read(RESULT)
    result["model_info"]["run_record"] = old_path
    publication.put(RESULT, result)
    publication.reseal()
    with pytest.raises(ValueError):
        publication.verify()


@pytest.mark.parametrize("change", ["missing_marker", "hidden_second_marker"])
def test_redacted_fields_match_exported_content_exactly(publication, change):
    run = publication.read(RUN)
    if change == "missing_marker":
        run["compatibility"]["patches"][0]["changes"][0]["after"] = "unchanged"
    else:
        run["other_diagnostic"] = "<REDACTED_LOCAL_PATH>"
    metadata = publication.manifest["files"][RUN]
    content = encode(run)
    (publication.root / RUN).write_bytes(content)
    metadata.update(sha256=digest(content), bytes=len(content))
    with pytest.raises(ValueError):
        publication.verify()


def test_frozen_source_is_checked_even_when_export_manifest_is_rehashed(publication):
    publication.put(SOURCE, b"# Changed after protocol freeze.\n")
    with pytest.raises(ValueError):
        publication.verify()


def test_frozen_source_check_count_cannot_be_forged(publication):
    publication.manifest["frozen_source_checks"][PROTOCOL] = 2
    with pytest.raises(ValueError):
        publication.verify()


def test_legacy_source_sha256_protocol_mapping_is_supported(publication):
    protocol = publication.read(PROTOCOL)
    protocol["source_sha256"] = protocol.pop("sources")
    publication.put(PROTOCOL, protocol)
    assert publication.verify()["frozen_source_checks"] == {PROTOCOL: 1}


def test_receipt_hash_is_checked_even_when_export_manifest_is_rehashed(publication):
    publication.put(EVIDENCE, b"Changed sealed evidence.\n")
    with pytest.raises(ValueError):
        publication.verify()


@pytest.mark.parametrize("extra", ["unexpected.txt", "nested/unexpected.txt"])
def test_receipt_directory_must_not_contain_extra_files(publication, extra):
    publication.put(f"{EPISODE}/{extra}", b"Unsealed extra evidence.\n")
    with pytest.raises(ValueError):
        publication.verify()


@pytest.mark.parametrize("unsafe", ["../outside.bin", "/outside.bin", "C:/outside.bin", "nested/../trajectory.npz"])
def test_receipt_paths_cannot_escape_or_alias_inventory(publication, unsafe):
    receipt = publication.read(RECEIPT)
    receipt["files"][unsafe] = receipt["files"].pop("trajectory.npz")
    publication.put(RECEIPT, receipt)
    with pytest.raises(ValueError):
        publication.verify()


@pytest.mark.parametrize("unsafe", ["../outside.py", "/outside.py", "C:/outside.py", "scripts/../outside.py"])
def test_manifest_paths_cannot_escape_or_alias_root(publication, unsafe):
    publication.manifest["files"][unsafe] = deepcopy(publication.manifest["files"][SOURCE])
    with pytest.raises(ValueError):
        publication.verify()


def test_worker_result_case_identity_must_match(publication):
    result = publication.read(RESULT)
    result["id"] = "unrelated-case"
    publication.put(RESULT, result)
    publication.reseal()
    with pytest.raises(ValueError):
        publication.verify()


def test_worker_result_run_record_must_match(publication):
    result = publication.read(RESULT)
    result["model_info"]["run_record"] = f"{STUDY}/runs/run_9999.json"
    publication.put(RESULT, result)
    publication.reseal()
    with pytest.raises(ValueError):
        publication.verify()


def test_complete_worker_must_complete_exactly_its_one_case(publication):
    worker = publication.read(WORKER)
    worker["newly_completed"] = [CASE, "another-case"]
    worker["completed_after"] = 2
    publication.put(WORKER, worker)
    with pytest.raises(ValueError):
        publication.verify()


def test_worker_protocol_identity_must_match(publication):
    worker = publication.read(WORKER)
    worker["protocol_fingerprint"] = "0" * 64
    publication.put(WORKER, worker)
    with pytest.raises(ValueError):
        publication.verify()
