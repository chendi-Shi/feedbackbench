import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("combine_alignment", Path(__file__).resolve().parents[1]
                                            / "scripts/combine_action_demo_alignment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures(tmp_path):
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    sources = [{"suite": s, "task_id": 0, "task_uid": s + ":0", "sha256": str(i) * 64,
                "bytes": 10, "path": s + ".hdf5"} for i, s in enumerate(suites)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": sources}))
    prior = tmp_path / "prior_diagnostic.json"
    prior.write_text('{"status": "needs_attention"}')
    prior_reference = {"path": str(prior), "sha256": module.digest(prior)}
    paths = []
    for source in sources:
        record = {**source, "status": "passed", "action_offset": 1, "demonstrations": [{}] * 50,
                  "replay": {"status": "passed", "samples": [{}] * 5}}
        receipt = {"status": "passed", "scope": "observation_action_alignment", "dynamics_replay_verified": False,
                   "diagnostic_revision": 3, "source_manifest_sha256": module.digest(manifest),
                   "records": [record], "script_sha256": "validator", "action_offset": 1,
                   "v1_receipt": prior_reference, "prior_observation_receipt": prior_reference}
        path = tmp_path / (source["suite"] + ".json")
        path.write_text(json.dumps(receipt))
        paths.append(path)
    return manifest, paths


def test_combines_exactly_four_bound_sources(tmp_path):
    manifest, paths = fixtures(tmp_path)
    report = module.combine(manifest, paths)
    assert report["status"] == "passed"
    assert report["action_offset"] == 1
    assert report["dynamics_replay_verified"] is False
    assert len(report["records"]) == 4


@pytest.mark.parametrize("change", ["offset", "hash", "failed", "manifest", "duplicate"])
def test_refuses_mixed_or_unverified_receipts(tmp_path, change):
    manifest, paths = fixtures(tmp_path)
    receipt = json.loads(paths[0].read_text())
    if change == "offset":
        receipt["action_offset"] = 0
    elif change == "hash":
        receipt["records"][0]["sha256"] = "changed"
    elif change == "failed":
        receipt["status"] = "needs_attention"
    elif change == "manifest":
        receipt["source_manifest_sha256"] = "different"
    else:
        paths[0] = paths[1]
    if change != "duplicate":
        paths[0].write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        module.combine(manifest, paths)


@pytest.mark.parametrize("change", ["missing", "same_size_changed"])
def test_refuses_changed_prior_diagnostic_evidence(tmp_path, change):
    manifest, paths = fixtures(tmp_path)
    prior = tmp_path / "prior_diagnostic.json"
    if change == "missing":
        prior.unlink()
    else:
        original = prior.read_text()
        prior.write_text(original.replace("needs_attention", "wrong_attention"))
        assert prior.stat().st_size == len(original.encode())
    with pytest.raises(ValueError, match="prior diagnostic receipt"):
        module.combine(manifest, paths)
