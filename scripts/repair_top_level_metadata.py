"""Supply missing package declarations without repeatedly statting wheel RECORDs.

The six-package allowlist comes from the Windows cold-import audit.  Declarations
are inferred by this interpreter's importlib.metadata implementation, then checked
against importable files/namespaces and an existing file from each wheel RECORD.
The original RECORD and package sources remain intact.  Dry-run is the default;
only --apply creates missing top_level.txt files.  Both modes append an audit.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import tempfile
from types import SimpleNamespace


TARGET_PACKAGES = frozenset({"cmake", "scipy", "numpy", "matplotlib", "wandb", "pygments"})
DEFAULT_AUDIT = Path(__file__).resolve().parents[1] / "artifacts" / "runtime_metadata_audit.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _infer_names(files) -> list[str]:
    infer = getattr(importlib.metadata, "_top_level_inferred", None)
    if not callable(infer):
        raise RuntimeError("This interpreter lacks the audited importlib.metadata inference API")
    return sorted(set(infer(SimpleNamespace(files=files))))


def _within_site(path: Path, site: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(site):
        raise RuntimeError(f"Candidate resolves outside site-packages: {path}")
    return resolved


def _validate_names(names: list[str], files: list, site: Path) -> list[dict]:
    if not names:
        raise RuntimeError("No importable names inferred; refusing an empty declaration")
    validated = []
    for name in names:
        spec = importlib.machinery.PathFinder.find_spec(name, [str(site)])
        if spec is None:
            raise RuntimeError(f"Inferred name does not resolve through PathFinder: {name}")
        if spec.origin is not None:
            origin = _within_site(Path(spec.origin), site)
            if not origin.is_file():
                raise RuntimeError(f"Inferred module file does not exist: {origin}")
            kind = "module_or_package"
        else:
            locations = list(spec.submodule_search_locations or [])
            if not locations or not all(_within_site(Path(item), site).is_dir() for item in locations):
                raise RuntimeError(f"Inferred namespace has no existing search location: {name}")
            kind = "namespace"
        evidence = None
        for entry in files:
            if name not in _infer_names([entry]):
                continue
            candidate = _within_site(site.joinpath(*entry.parts), site)
            if candidate.is_file():
                evidence = candidate
                break
        if evidence is None:
            raise RuntimeError(f"No existing RECORD file supports inferred name: {name}")
        validated.append({
            "name": name, "kind": kind, "origin": spec.origin,
            "namespace_locations": list(spec.submodule_search_locations or []),
            "record_file_evidence": str(evidence),
        })
    return validated


def plan_distribution(dist_info: Path, site: Path) -> dict:
    """Inspect one allowlisted distribution; never execute package code."""
    site = site.resolve()
    dist_info = _within_site(dist_info, site)
    if dist_info.parent != site or not dist_info.name.endswith(".dist-info"):
        raise RuntimeError(f"Expected a direct dist-info child of site-packages: {dist_info}")
    metadata = importlib.metadata.PathDistribution(dist_info).metadata
    name = metadata["Name"].lower()
    if name not in TARGET_PACKAGES:
        raise RuntimeError(f"Distribution is outside the six-package allowlist: {name}")
    record = (dist_info / "RECORD").read_bytes()
    declaration = dist_info / "top_level.txt"
    entry = {
        "distribution": name, "version": metadata["Version"],
        "dist_info": str(dist_info), "record_sha256": _sha256(record),
        "old_absence": not declaration.exists(), "new_content": None,
    }
    if declaration.exists():
        entry.update(status="preserved_existing", existing_content=declaration.read_text(encoding="utf-8"))
        return entry
    files = [
        importlib.metadata.PackagePath(row[0])
        for row in csv.reader(io.StringIO(record.decode("utf-8")))
        if row and row[0]
    ]
    names = _infer_names(files)
    entry.update(
        status="proposed", record_entry_count=len(files),
        new_content="\n".join(names) + "\n",
        validation=_validate_names(names, files, site),
    )
    return entry


def build_plan(site: Path) -> list[dict]:
    """Validate the complete allowlist before any environment write."""
    site = site.resolve()
    found = {}
    for dist_info in site.glob("*.dist-info"):
        directory_name = dist_info.name.split("-", 1)[0].lower()
        if directory_name not in TARGET_PACKAGES:
            continue
        entry = plan_distribution(dist_info, site)
        name = entry["distribution"]
        if name in found:
            raise RuntimeError(f"Ambiguous installed distributions for {name}")
        found[name] = entry
    missing = TARGET_PACKAGES - found.keys()
    if missing:
        raise RuntimeError(f"Missing allowlisted distributions: {', '.join(sorted(missing))}")
    return [found[name] for name in sorted(found)]


def _save_audit(path: Path, audit: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(audit, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def repair(site: Path, audit_path: Path, *, apply: bool = False) -> dict:
    entries = build_plan(site)
    audit = {"schema_version": 1, "runs": []}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("schema_version") != 1 or not isinstance(audit.get("runs"), list):
            raise RuntimeError("Existing metadata audit has an unsupported format")
    run = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run", "status": "validated",
        "interpreter": sys.executable, "python_version": platform.python_version(),
        "site_packages": str(site.resolve()),
        "inference_source_sha256": _sha256(inspect.getsource(importlib.metadata._top_level_inferred).encode()),
        "distributions": entries,
    }
    audit["runs"].append(run)
    _save_audit(audit_path, audit)  # Check audit writability before environment edits.
    try:
        if apply:
            for entry in entries:
                if entry["status"] != "proposed":
                    continue
                dist_info = Path(entry["dist_info"])
                if _sha256((dist_info / "RECORD").read_bytes()) != entry["record_sha256"]:
                    raise RuntimeError(f"RECORD changed after validation: {entry['distribution']}")
                # Exclusive creation prevents replacing a declaration from another process.
                with (dist_info / "top_level.txt").open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(entry["new_content"])
                entry["status"] = "applied"
                entry["new_content_sha256"] = _sha256(entry["new_content"].encode())
                _save_audit(audit_path, audit)
        run["status"] = "complete"
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _save_audit(audit_path, audit)
    return run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--apply", action="store_true", help="Create absent declarations; default only plans and audits")
    result.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_path("purelib")))
    result.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return result


def main() -> None:
    args = parser().parse_args()
    run = repair(args.site_packages, args.audit, apply=args.apply)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
