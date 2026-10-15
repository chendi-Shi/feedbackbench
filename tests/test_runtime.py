import ast
import json
from pathlib import Path
import platform
import sys
from types import ModuleType, SimpleNamespace

import pytest

from embodied_research.runtime import (
    ASSET_REVISION, configure_libero_paths, configure_platform,
    patch_robosuite_logging_source, patch_robosuite_source,
)


SOURCE = '''import ctypes, os, mujoco
_SYSTEM = "Windows"
if _SYSTEM == "Windows":
    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))
if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:
    os.environ["MUJOCO_GL"] = "egl"
def physics_step(action):
    return original_physics(action)
'''


def test_windows_patch_only_changes_two_platform_statements():
    patched, audit = patch_robosuite_source(SOURCE)
    assert len(audit["changes"]) == 2
    before = ast.parse(SOURCE)
    after = ast.parse(patched)
    assert ast.dump(before.body[-1]) == ast.dump(after.body[-1])
    assert "os.path.dirname(mujoco.__file__)" in patched
    assert '_SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING' in patched
    assert audit["source_sha256"] != audit["loaded_source_sha256"]


def test_windows_patch_rejects_source_drift_and_double_patch():
    with pytest.raises(RuntimeError, match="unaudited"):
        patch_robosuite_source(SOURCE.replace("ctypes.WinDLL", "other_function"))
    patched, _ = patch_robosuite_source(SOURCE)
    with pytest.raises(RuntimeError, match="unaudited"):
        patch_robosuite_source(patched)


def test_windows_logging_patch_preserves_behavior_and_quotes_paths():
    source = 'fh = logging.FileHandler("/tmp/robosuite.log")\nfh.setLevel(logging.DEBUG)\n'
    destination = Path("C:/workspace/owner's project/.runtime/robosuite.log")
    patched, audit = patch_robosuite_logging_source(source, destination)
    before = ast.parse(source)
    after = ast.parse(patched)
    assert after.body[0].value.args[0].value == str(destination)
    assert ast.dump(before.body[1]) == ast.dump(after.body[1])
    assert len(audit["changes"]) == 1
    with pytest.raises(RuntimeError, match="unaudited"):
        patch_robosuite_logging_source(patched, destination)


def test_platform_fallback_needs_no_assets_or_simulator(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "_wmi_query", lambda *args: "original", raising=False)
    monkeypatch.delenv("EMBODIED_RESEARCH_ALLOW_WMI", raising=False)
    assert configure_platform() is True
    with pytest.raises(OSError, match="native Windows"):
        platform._wmi_query("Win32_OperatingSystem", "Version")


def test_platform_fallback_honors_diagnostic_opt_out(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "_wmi_query", lambda *args: "original", raising=False)
    monkeypatch.setenv("EMBODIED_RESEARCH_ALLOW_WMI", "1")
    assert configure_platform() is False
    assert platform._wmi_query() == "original"


def test_libero_config_preserves_short_package_path_without_global_config(monkeypatch, tmp_path):
    root = tmp_path / "project"
    package = tmp_path / "short_alias" / "libero" / "libero"
    for name in ("bddl_files", "init_files"):
        (package / name).mkdir(parents=True)
    assets = root / "assets" / "libero"
    for name in ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects", "scenes"):
        (assets / name).mkdir(parents=True)
    (assets / "asset_manifest.json").write_text(json.dumps({
        "repository": "lerobot/libero-assets", "revision": ASSET_REVISION,
        "file_count": 586, "total_bytes": 422320936,
    }), encoding="utf-8")
    original_resolve = Path.resolve

    def resolve_without_expanding_package(self, *args, **kwargs):
        if self == package:
            raise AssertionError("Expanding the package junction recreates invalid long paths")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_without_expanding_package)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: SimpleNamespace(locate_file=lambda path: package))
    parent = ModuleType("libero")
    inner = ModuleType("libero.libero")
    inner.config_file = str(root / ".runtime" / "libero" / "config.yaml")
    parent.libero = inner
    monkeypatch.setitem(sys.modules, "libero", parent)
    monkeypatch.setitem(sys.modules, "libero.libero", inner)
    configure_libero_paths(root)
    config = json.loads(Path(inner.config_file).read_text(encoding="utf-8"))
    assert config["benchmark_root"] == str(package)
    assert inner._assets_path_cache == str(assets)
