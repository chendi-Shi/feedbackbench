import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("phase2_data", Path(__file__).resolve().parents[1] / "scripts/prepare_action_study_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Reply(io.BytesIO):
    def __init__(self, payload, start, total, *, status=206):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{total - 1}/{total}",
                        "Content-Length": str(total - start)}

    def geturl(self):
        return "https://cdn.example/fixed-object"


def source(payload):
    return {"suite": "libero_object", "task_name": "fixed_task", "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "path": "study/libero_object.hdf5"}


def test_resume_only_requests_absent_suffix_and_keeps_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    pieces = tmp_path / "study/libero_object_ranges"
    pieces.mkdir(parents=True)
    (pieces / "0000.part").write_bytes(b"abc")
    def opener(request, **kwargs):
        assert request.headers["Range"] == "bytes=3-8"
        return Reply(b"defghi", 3, 9)
    monkeypatch.setattr(module.urllib.request, "urlopen", opener)
    record = {}
    module.download_source(source(b"abcdefghi"), record, lambda: None, resume=True, workers=2)
    assert (tmp_path / "study/libero_object.hdf5").read_bytes() == b"abcdefghi"
    assert (pieces / "0000.bin").read_bytes() == b"abcdefghi"
    assert record["ranges"][0]["received_this_attempt"] == 6


def test_http200_never_read_and_partial_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    class Refused(Reply):
        def read1(self, amount):
            raise AssertionError("Must reject headers before body")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Refused(b"abc", 0, 3, status=200))
    record = {}
    with pytest.raises(RuntimeError, match="no request retried"):
        module.download_source(source(b"abc"), record, lambda: None, resume=False, workers=1)
    assert record["ranges"][0]["received_this_attempt"] == 0
    assert not (tmp_path / "study/libero_object.hdf5").exists()


def test_bad_full_hash_refuses_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Reply(b"bad", 0, 3))
    with pytest.raises(RuntimeError, match="SHA256"):
        module.download_source(source(b"abc"), {}, lambda: None, resume=False, workers=1)
    assert not (tmp_path / "study/libero_object.hdf5").exists()


def test_manifest_tasks_and_budgets_are_fixed():
    manifest = module.source_manifest()
    assert len(manifest["sources"]) == 4
    assert {s["task_id"] for s in manifest["sources"]} == {0}
    assert manifest["original_unique_file_bytes"] == 3458688561
    assert 2 * manifest["original_unique_file_bytes"] < module.DISK_CAP
    assert manifest["original_unique_file_bytes"] < module.NETWORK_CAP
    assert manifest["sources"][-1]["language"] == "put both the alphabet soup and the tomato sauce in the basket"
    assert module.ORIGINAL_PRIVATE_SCRIPT_SHA256 == "6301819a43c0a116b0fc3aa8fffbee3350407803492cc6712f20257ee65fc38c"


def test_wrong_content_range_is_rejected_before_body(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    class WrongRange(Reply):
        def read1(self, amount):
            raise AssertionError("Mismatching body must not be consumed")
    reply = WrongRange(b"abc", 1, 4)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: reply)
    with pytest.raises(RuntimeError, match="no request retried"):
        module.download_source(source(b"abc"), {}, lambda: None, resume=False, workers=1)


def test_truncated_response_preserves_prefix_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        return Reply(b"abc", 0, 9)
    monkeypatch.setattr(module.urllib.request, "urlopen", opener)
    with pytest.raises(RuntimeError, match="Truncated range"):
        module.download_source(source(b"abcdefghi"), {}, lambda: None, resume=False, workers=1)
    assert len(calls) == 1
    assert (tmp_path / "study/libero_object_ranges/0000.part").read_bytes() == b"abc"
    assert not (tmp_path / "study/libero_object.hdf5").exists()


@pytest.mark.parametrize("kind", ["without_resume", "oversized"])
def test_existing_pieces_require_explicit_safe_resume(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    pieces = tmp_path / "study/libero_object_ranges"
    pieces.mkdir(parents=True)
    (pieces / "0000.part").write_bytes(b"a" if kind == "without_resume" else b"abcd")
    def forbidden(*args, **kwargs):
        raise AssertionError("No request allowed before resume validation")
    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden)
    with pytest.raises(RuntimeError, match="resume|exceeds"):
        module.download_source(source(b"abc"), {}, lambda: None,
                               resume=kind == "oversized", workers=1)


def test_official_metadata_must_match_fixed_sha_and_revision(monkeypatch):
    selected = module.SOURCES[1]
    entry = {"path": selected["suite"] + "/" + selected["task_name"] + "_demo.hdf5",
             "size": selected["bytes"], "lfs": {"oid": "wrong", "size": selected["bytes"]}}
    def opener(url, **kwargs):
        assert module.REVISION in url
        assert module.REPO in url
        return io.BytesIO(json.dumps([entry]).encode())
    monkeypatch.setattr(module.urllib.request, "urlopen", opener)
    with pytest.raises(ValueError, match="metadata changed"):
        module.inspect_metadata(selected)


@pytest.mark.parametrize("existing_lock", [False, True])
def test_main_fresh_runtime_directory_and_exclusive_owner_lock(tmp_path, monkeypatch, existing_lock):
    import psutil

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare_action_study_sources.py", "--download"])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024 ** 3))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024 ** 3))
    calls = []
    monkeypatch.setattr(module, "inspect_metadata", lambda source: calls.append(source["suite"]) or {})
    monkeypatch.setattr(module, "download_source", lambda source, state, persist, **kwargs:
                        state.update(status="mock_verified_without_network"))
    lock = tmp_path / ".runtime/action_phase2_download.lock"
    if existing_lock:
        lock.parent.mkdir()
        lock.write_text("owned by another invocation")
        with pytest.raises(FileExistsError):
            module.main()
        assert lock.read_text() == "owned by another invocation"
        assert calls == []
    else:
        module.main()
        assert len(calls) == 4
        assert not lock.exists()
        audit = next((tmp_path / ".runtime/action_phase2_download").glob("*.json"))
        assert json.loads(audit.read_text())["status"] == "complete"
