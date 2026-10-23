"""Audited, process-local compatibility for the pinned upstream simulator.

The research environment is intentionally separate from ContactLab.  None of
these changes alter MuJoCo dynamics, controllers, assets, observations, or a
checkpoint.  The Windows patch only fixes robosuite 1.4.0's DLL lookup,
unsupported EGL selection, and Unix-only log path. Installed source stays intact.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


MUJOCO_VERSION = "3.3.2"
ROBOSUITE_VERSION = "1.4.0"
# UTF-8 source with universal newlines from the upstream v1.4.0 tag.
ROBOSUITE_BINDING_SHA256 = "f282a5a48dd215363ce3f981ce4126d91639c6e00fa530f0615cd5ab598372a8"
ROBOSUITE_LOGGING_SHA256 = "11fc4063dd2b558f0d7d8e268d2b7a0e9fd869867baa5ecfc43d4a0564e551e9"
ASSET_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
_BINDING_MODULE = "robosuite.utils.binding_utils"
_LOGGING_MODULE = "robosuite.utils.log_utils"
_RECORDS: list[dict[str, Any]] = []
_HOOK: _RobosuiteFinder | None = None

_OLD_DLL = '    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))'
_NEW_DLL = '    ctypes.WinDLL(os.path.join(os.path.dirname(mujoco.__file__), "mujoco.dll"))'
_OLD_BACKEND = 'if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:'
_NEW_BACKEND = 'if _SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:'


def patch_robosuite_source(source: str) -> tuple[str, dict[str, Any]]:
    """Change exactly two known platform statements; fail on source drift."""
    for fragment in (_OLD_DLL, _OLD_BACKEND):
        if source.count(fragment) != 1:
            raise RuntimeError("Unsupported robosuite source; refusing an unaudited Windows patch")
    patched = source.replace(_OLD_DLL, _NEW_DLL).replace(_OLD_BACKEND, _NEW_BACKEND)
    return patched, {
        "module": _BINDING_MODULE,
        "upstream_version": ROBOSUITE_VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "loaded_source_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "changes": [
            {"before": _OLD_DLL, "after": _NEW_DLL},
            {"before": _OLD_BACKEND, "after": _NEW_BACKEND},
        ],
        "scope": "Windows DLL lookup and OpenGL context selection only; source unchanged on disk",
    }


def patch_robosuite_logging_source(source: str, log_path: Path) -> tuple[str, dict[str, Any]]:
    """Redirect the Unix-only logfile without changing logging behavior."""
    before = 'logging.FileHandler("/tmp/robosuite.log")'
    after = f"logging.FileHandler({str(log_path)!r})"
    if source.count(before) != 1:
        raise RuntimeError("Unsupported robosuite logging source; refusing an unaudited Windows patch")
    patched = source.replace(before, after)
    return patched, {
        "module": _LOGGING_MODULE, "upstream_version": ROBOSUITE_VERSION,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "loaded_source_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "changes": [{"before": before, "after": after}],
        "scope": "Windows logfile location only; source unchanged on disk",
    }


class _RobosuiteLoader(importlib.abc.Loader):
    def __init__(self, origin: str, module_name: str, log_path: Path):
        self.origin = origin
        self.module_name = module_name
        self.log_path = log_path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        version = importlib.metadata.version("robosuite")
        if version != ROBOSUITE_VERSION:
            raise RuntimeError(f"Expected robosuite {ROBOSUITE_VERSION}, found {version}")
        source = Path(self.origin).read_text(encoding="utf-8")
        expected = ROBOSUITE_BINDING_SHA256 if self.module_name == _BINDING_MODULE else ROBOSUITE_LOGGING_SHA256
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"{self.module_name} source differs from the audited v1.4.0 file")
        if self.module_name == _BINDING_MODULE:
            patched, record = patch_robosuite_source(source)
        else:
            patched, record = patch_robosuite_logging_source(source, self.log_path)
        record["source_path"] = self.origin
        exec(compile(patched, self.origin, "exec"), module.__dict__)
        _RECORDS.append(record)


class _RobosuiteFinder(importlib.abc.MetaPathFinder):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def find_spec(self, fullname, path, target=None):
        if fullname not in (_BINDING_MODULE, _LOGGING_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            return None
        spec.loader = _RobosuiteLoader(spec.origin, fullname, self.log_path)
        return spec


def _unavailable_wmi(*args, **kwargs):
    raise OSError("Use CPython's native Windows platform fallback for this process")


def configure_platform() -> bool:
    """Select CPython's native Windows metadata fallback without dependencies.

    Call before NumPy.testing, Torch or graphical imports.  No assets, simulator,
    vendored repository, registry or system settings are accessed.  Setting
    EMBODIED_RESEARCH_ALLOW_WMI=1 retains stock behavior for diagnostic comparisons.
    """
    if (
        sys.platform == "win32"
        and hasattr(platform, "_wmi_query")
        and os.environ.get("EMBODIED_RESEARCH_ALLOW_WMI") != "1"
    ):
        platform._wmi_query = _unavailable_wmi
        return True
    return False


def configure_runtime(project_root: str | Path | None = None) -> dict[str, Any]:
    """Expose the pinned research fork and select Windows GLFW before imports."""
    global _HOOK
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    vendor_src = root / "vendor" / "MINERVA" / "src"
    if not vendor_src.is_dir():
        raise FileNotFoundError(f"Missing pinned MINERVA source: {vendor_src}")
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))
    wmi_fallback = configure_platform()
    if sys.platform == "win32":
        if _HOOK is None:
            if _BINDING_MODULE in sys.modules:
                raise RuntimeError("Call configure_runtime before importing robosuite")
            log_path = root / ".runtime" / "robosuite.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _HOOK = _RobosuiteFinder(log_path)
            sys.meta_path.insert(0, _HOOK)
        if os.environ.get("MUJOCO_GL", "glfw") not in ("glfw", "wgl", ""):
            raise RuntimeError("The native Windows reproduction requires MUJOCO_GL=glfw")
        os.environ["MUJOCO_GL"] = "glfw"
    return {
        "platform": sys.platform,
        "mujoco_required": MUJOCO_VERSION,
        "robosuite_required": ROBOSUITE_VERSION,
        "backend": os.environ.get("MUJOCO_GL", "default"),
        "wmi_native_fallback": wmi_fallback,
        "windows_import_hook": _HOOK is not None,
        "vendor_source": str(vendor_src),
    }


def configure_libero_paths(project_root: str | Path, assets_path: str | Path | None = None) -> Path:
    """Use project-local config and pinned local assets, avoiding import prompts.

    Asset fetching belongs to the explicit download command.  Evaluation never
    silently downloads an unpinned latest snapshot.
    """
    root = Path(project_root).resolve()
    assets = Path(os.path.abspath(assets_path)) if assets_path else root / "assets" / "libero"
    required = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects", "scenes")
    if not all((assets / item).is_dir() for item in required):
        raise FileNotFoundError(f"Incomplete LIBERO assets at {assets}; download the pinned asset snapshot first")
    manifest_path = assets / "asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Asset download has not completed: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("repository") != "lerobot/libero-assets"
        or manifest.get("revision") != ASSET_REVISION
        or manifest.get("file_count") != 586
        or manifest.get("total_bytes") != 422320936
    ):
        raise RuntimeError("LIBERO asset manifest differs from the pinned reproduction snapshot")
    distribution = importlib.metadata.distribution("hf-libero")
    # Keep a short Windows junction spelling.  Resolving it back to the long
    # OneDrive path defeats the package-install workaround for long BDDL names.
    benchmark_root = Path(os.path.abspath(distribution.locate_file("libero/libero")))
    for name in ("bddl_files", "init_files"):
        if not (benchmark_root / name).is_dir():
            raise FileNotFoundError(f"Missing hf-libero benchmark files: {benchmark_root / name}")
    config_dir = root / ".runtime" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    previous = sys.modules.get("libero.libero")
    if previous is not None and Path(previous.config_file).resolve() != config_dir / "config.yaml":
        raise RuntimeError("LIBERO was already imported with another configuration")
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(root / "datasets"),
        "assets": str(assets),
    }
    # JSON is valid YAML and ASCII escaping remains readable under Windows' locale.
    config = json.dumps(paths, indent=2, ensure_ascii=True) + "\n"
    config_path = config_dir / "config.yaml"
    if not config_path.exists() or config_path.read_text(encoding="utf-8") != config:
        config_path.write_text(config, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    import libero.libero as libero

    # Upstream get_assets_path ignores config['assets']; its documented local cache
    # is supplied explicitly to prevent fallback to snapshot_download(main).
    libero._assets_path_cache = str(assets)
    return assets


def compatibility_audit() -> dict[str, Any]:
    return {"patches": list(_RECORDS), "physics_or_controller_patches": False}
