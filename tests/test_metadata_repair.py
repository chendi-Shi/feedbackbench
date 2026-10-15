import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_top_level_metadata.py"
SPEC = importlib.util.spec_from_file_location("metadata_repair", SCRIPT)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    for name in repair.TARGET_PACKAGES:
        package = site / name
        package.mkdir()
        (package / "__init__.py").write_text("raise AssertionError('must not import packages')\n")
        dist = site / f"{name}-1.0.dist-info"
        dist.mkdir()
        record = f"{name}/__init__.py,,\n"
        if name == "matplotlib":
            namespace = site / "mpl_toolkits" / "example"
            namespace.mkdir(parents=True)
            (namespace / "__init__.py").write_text("")
            (site / "pylab.py").write_text("")
            record += "mpl_toolkits/example/__init__.py,,\npylab.py,,\n"
        (dist / "RECORD").write_text(record)
    monkeypatch.setattr(repair.importlib.metadata, "PathDistribution", lambda path: SimpleNamespace(
        metadata={"Name": path.name.split("-", 1)[0], "Version": "1.0"},
    ))
    return site, tmp_path / "audit.json"


def test_dry_run_is_default_and_uses_exact_inference_without_imports(environment):
    site, audit = environment
    assert repair.parser().parse_args([]).apply is False
    run = repair.repair(site, audit)
    assert run["mode"] == "dry_run"
    assert not list(site.glob("*.dist-info/top_level.txt"))
    entries = {entry["distribution"]: entry for entry in run["distributions"]}
    assert entries["matplotlib"]["new_content"] == "matplotlib\nmpl_toolkits\npylab\n"
    assert all(entry["old_absence"] for entry in entries.values())
    numpy_record = (site / "numpy-1.0.dist-info" / "RECORD").read_bytes()
    assert entries["numpy"]["record_sha256"] == hashlib.sha256(numpy_record).hexdigest()
    assert json.loads(audit.read_text())["runs"][0]["status"] == "complete"


def test_apply_is_explicit_preserves_record_and_existing_declarations(environment):
    site, audit = environment
    records = {path: path.read_bytes() for path in site.glob("*.dist-info/RECORD")}
    existing = site / "numpy-1.0.dist-info" / "top_level.txt"
    existing.write_text("existing declaration\n")
    run = repair.repair(site, audit, apply=True)
    assert existing.read_text() == "existing declaration\n"
    assert sum(entry["status"] == "applied" for entry in run["distributions"]) == 5
    assert all(path.read_bytes() == original for path, original in records.items())
    repair.repair(site, audit)
    assert len(json.loads(audit.read_text())["runs"]) == 2


def test_invalid_inferred_name_prevents_all_writes(environment):
    site, audit = environment
    with (site / "wandb-1.0.dist-info" / "RECORD").open("a") as handle:
        handle.write("nonexistent_package/__init__.py,,\n")
    with pytest.raises(RuntimeError, match="does not resolve"):
        repair.repair(site, audit, apply=True)
    assert not list(site.glob("*.dist-info/top_level.txt"))


def test_importable_name_needs_an_existing_record_file(environment):
    site, audit = environment
    (site / "numpy-1.0.dist-info" / "RECORD").write_text("numpy/missing_file.py,,\n")
    with pytest.raises(RuntimeError, match="No existing RECORD file"):
        repair.repair(site, audit, apply=True)
    assert not list(site.glob("*.dist-info/top_level.txt"))
