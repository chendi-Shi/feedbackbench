"""Cheap refusal-path tests; no Torch imports, downloads, or real process actions."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_vlajepa.py"
spec = importlib.util.spec_from_file_location("vlajepa_probe_tested", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_schema_allows_only_unused_world_model_prefixes():
    header = {"model.qwen.weight": {"shape": [2, 3], "dtype": "BF16"},
              "model.video_encoder.weight": {"shape": [5], "dtype": "BF16"}}
    result = probe.validate_schema(header, {"model.qwen.weight": (2, 3)})
    assert result["active_elements"] == 6
    assert result["skipped_elements"] == 5
    header["model.unexpected.weight"] = {"shape": [1]}
    with pytest.raises(ValueError, match="unexpected"):
        probe.validate_schema(header, {"model.qwen.weight": (2, 3)})


@pytest.mark.parametrize("header,shapes", [
    ({}, {"model.qwen.weight": (2, 3)}),
    ({"model.qwen.weight": {"shape": [3, 2]}}, {"model.qwen.weight": (2, 3)}),
])
def test_schema_refuses_missing_or_changed_active_weights(header, shapes):
    with pytest.raises(ValueError):
        probe.validate_schema(header, shapes)


def test_header_rejects_unbounded_allocation(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 2**63))
    with pytest.raises(ValueError, match="header size"):
        probe.read_safetensors_header(path)


def test_alias_payload_check_requires_actual_identical_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"first": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 4]},
              "second": {"shape": [2], "dtype": "BF16", "data_offsets": [4, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"abcdabcd")
    probe.assert_equal_tensor_payloads(path, header["first"], header["second"])
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"abcdbadc")
    with pytest.raises(ValueError, match="copies differ"):
        probe.assert_equal_tensor_payloads(path, header["first"], header["second"])


def test_download_never_overwrites_existing_corrupt_object(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    path = tmp_path / "weight"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        probe.download("unused", path, 5, hashlib.sha256(b"right").hexdigest())
    assert path.read_bytes() == b"wrong"


def test_download_refuses_server_ignoring_resume_without_changing_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    destination = tmp_path / "weight"
    partial = tmp_path / "weight.part"
    partial.write_bytes(b"123")

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="resume range"):
        probe.download("https://example.test/weight", destination, 10)
    assert partial.read_bytes() == b"123"
    assert not destination.exists()


def test_parallel_ranges_preserve_prefix_and_cover_suffix_once(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    destination = tmp_path / "weight"
    content = b"0123456789abcdefg"
    destination.with_name("weight.part").write_bytes(content[:3])
    requested = []

    class Response(BytesIO):
        status = 206

        def __init__(self, start, end):
            super().__init__(content[start:end + 1])
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}

    def open_range(request, **kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requested.append((start, end))
        return Response(start, end)

    monkeypatch.setattr(probe.urllib.request, "urlopen", open_range)
    result = probe.download_remaining_ranges("https://example.test/weight", destination,
                                              len(content), hashlib.sha256(content).hexdigest(), 4)
    assert destination.read_bytes() == content
    assert sorted(index for start, end in requested for index in range(start, end + 1)) == list(range(3, len(content)))
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert not list(tmp_path.glob("*.range-*"))


def test_parallel_range_retries_only_unwritten_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    destination = tmp_path / "weight"
    content = b"0123456789abcdefg"
    destination.with_name("weight.part").write_bytes(content[:3])
    requested = []

    class Response(BytesIO):
        status = 206

        def __init__(self, start, end):
            super().__init__(content[start:end + 1])
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
            self.interrupt = start == 3

        def read(self, size):
            if self.interrupt and self.tell() == 2:
                raise TimeoutError("synthetic network timeout")
            return super().read(2 if self.interrupt else size)

    def open_range(request, **kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requested.append((start, end))
        return Response(start, end)

    monkeypatch.setattr(probe.urllib.request, "urlopen", open_range)
    probe.download_remaining_ranges("https://example.test/weight", destination,
                                    len(content), hashlib.sha256(content).hexdigest(), 4)
    assert destination.read_bytes() == content
    assert sorted(requested) == [(3, 6), (5, 6), (7, 10), (11, 14), (15, 16)]
    assert not list(tmp_path.glob("*.range-*"))


def test_parallel_range_rejects_wrong_content_range_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: pytest.fail("Invalid range must not retry"))
    destination = tmp_path / "weight"
    destination.with_name("weight.part").write_bytes(b"012")

    class Response(BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-16/17"}

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"wrong"))
    with pytest.raises(ValueError, match="exact pinned"):
        probe.download_remaining_ranges("https://example.test/weight", destination, 17, "unused", 4)
    assert destination.with_name("weight.part").read_bytes() == b"012"
    assert not list(tmp_path.glob("*.range-*"))


def fake_guard(monkeypatch, tmp_path, identity_changed=False):
    class Process:
        pid = 123

        def __init__(self, *args):
            self.calls = 0

        def create_time(self):
            self.calls += 1
            return 100.0 if self.calls == 1 or not identity_changed else 200.0

        def memory_info(self):
            return SimpleNamespace(rss=20 * 1024**3)

    process = Process()
    fake = SimpleNamespace(Process=lambda *args: process,
                           virtual_memory=lambda: SimpleNamespace(available=4 * 1024**3))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(probe.os, "getpid", lambda: 123)
    guard = probe.MemoryGuard(tmp_path / "probe.json", max_rss_gib=7.5, min_available_gib=0.5)
    guard.stop_event = SimpleNamespace(wait=lambda *args: False)
    return guard


def test_memory_guard_checks_own_identity_and_preserves_reason(tmp_path, monkeypatch):
    guard = fake_guard(monkeypatch, tmp_path)
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(probe.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        guard._watch()
    assert exits == [72]
    record = json.loads((tmp_path / "probe.memory_stop.json").read_text())
    assert record["status"] == "stopped_memory_guard"
    assert record["pid"] == 123
    assert record["action"] == "exit own process only"


def test_memory_guard_identity_change_never_exits(tmp_path, monkeypatch):
    guard = fake_guard(monkeypatch, tmp_path, identity_changed=True)
    monkeypatch.setattr(probe.os, "_exit", lambda code: pytest.fail("Must not exit after identity changed"))
    guard._watch()
    assert not (tmp_path / "probe.memory_stop.json").exists()
