"""Fault injection for atomic evidence writes; no model or simulator needed."""
import errno
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import budget_campaign_common_v5 as common


def windows_error(code):
    error = PermissionError(errno.EACCES, "injected Windows file contention")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_replace_preserves_complete_old_then_new(tmp_path, monkeypatch, code):
    target = tmp_path / "status.json"
    target.write_text('{"old": true}', encoding="utf-8")
    real_replace, calls, delays = os.replace, [], []

    def replace(source, destination):
        assert json.loads(target.read_text()) == {"old": True}
        assert json.loads(Path(source).read_text()) == {"new": [1, 2, 3]}
        calls.append((source, destination))
        if len(calls) < 3:
            raise windows_error(code)
        return real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    common.write_json(target, {"new": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"new": [1, 2, 3]}
    assert delays == [.05, .1]
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_denial_preserves_target_and_pending_bytes(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    original = b'{"old": true}\n'
    target.write_bytes(original)
    calls, delays = [], []

    def replace(source, destination):
        calls.append(source)
        raise windows_error(5)

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        common.write_json(target, {"new": True})
    assert target.read_bytes() == original
    assert json.loads(Path(calls[-1]).read_text()) == {"new": True}
    assert len(calls) == 6
    assert tuple(delays) == common.REPLACE_RETRY_DELAYS


@pytest.mark.parametrize("error", [OSError(errno.ENOSPC, "disk full"), OSError(errno.EIO, "disk error"), windows_error(87)])
def test_unrelated_errors_are_not_retried(tmp_path, monkeypatch, error):
    calls, delays = [], []

    def replace(source, destination):
        calls.append(source)
        raise error

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(OSError):
        common.write_json(tmp_path / "status.json", {"new": True})
    assert len(calls) == 1 and delays == []


def test_invalid_json_never_replaces_existing_destination(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    target.write_text('{"old": true}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(common.os, "replace", lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        common.write_json(target, {"bad": float("nan")})
    assert json.loads(target.read_text()) == {"old": True}
    assert not calls
