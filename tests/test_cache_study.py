"""Integrity and split tests, with no model or simulator imports."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_cache_study.py"
SPEC = importlib.util.spec_from_file_location("cache_study_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_declared_splits_and_pairing_are_disjoint_and_complete():
    cases = runner.cases_for_study()
    assert len(cases) == len({case["id"] for case in cases}) == 28
    development = [c for c in cases if c["split"] == "development"]
    heldout = [c for c in cases if c["split"] == "heldout"]
    assert len(development) == 4 and {c["init_state_id"] for c in development} == {1}
    assert len(heldout) == 24 and {c["init_state_id"] for c in heldout} == {11, 29}
    for suite in runner.SUITES:
        for initial in (11, 29):
            methods = {c["method"] for c in heldout if c["suite"] == suite and c["init_state_id"] == initial}
            assert methods == ({"full10", "cache2", "cache0", "noise2"} if initial == 11 else {"full10", "cache2"})
    assert all(c["scheduler"] == {"mode": "fixed", "fixed_interval": 10, "temporal_ensemble": False} for c in cases)


def completed_fixture(tmp_path):
    case = runner.cases_for_study()[0]
    directory = tmp_path / case["id"]
    directory.mkdir()
    runner.write_json(directory / "result.json", {**case, "status": "complete", "protocol_fingerprint": "abc"})
    (directory / "trajectory.npz").write_bytes(b"raw fixture")
    runner.seal_episode(directory)
    return case, directory


def test_completed_evidence_resumes_only_without_change(tmp_path):
    case, directory = completed_fixture(tmp_path)
    assert runner.verify_complete(directory, case, "abc")["status"] == "complete"
    with pytest.raises(ValueError, match="incompatible"):
        runner.verify_complete(directory, case, "different")
    with pytest.raises(ValueError, match="identity"):
        runner.verify_complete(directory, {**case, "seed": 999}, "abc")


@pytest.mark.parametrize("kind", ["changed", "extra", "missing", "partial", "error"])
def test_attempts_or_tampered_evidence_cannot_silently_retry(tmp_path, kind):
    case, directory = completed_fixture(tmp_path)
    if kind == "changed":
        (directory / "trajectory.npz").write_bytes(b"tampered")
    elif kind == "extra":
        (directory / "extra.txt").write_text("not sealed")
    elif kind == "missing":
        (directory / "trajectory.npz").unlink()
    elif kind == "partial":
        (directory / "receipt.json").unlink()
    else:
        record = json.loads((directory / "result.json").read_text())
        record["status"] = "error"
        runner.write_json(directory / "result.json", record)
    with pytest.raises((ValueError, RuntimeError)):
        runner.verify_complete(directory, case, "abc")


def test_missing_attempt_is_distinct_from_partial(tmp_path):
    case = runner.cases_for_study()[0]
    assert runner.verify_complete(tmp_path / "absent", case, "abc") is None


def test_freeze_requires_every_development_case(tmp_path):
    with pytest.raises(RuntimeError, match="All declared development"):
        runner.freeze_cache(tmp_path, {"spec": runner.study_spec(), "fingerprint": "abc"})


def development_fixture(tmp_path, successful):
    from embodied_research.cache_sampler import latent_time_contract
    protocol = {"spec": runner.study_spec(), "fingerprint": "abc", "artifacts": {"models": {
        "smolvla": {"revision": "pinned", "files": {"model.safetensors": "a" * 64}}}}}
    for index, case in enumerate(protocol["spec"]["cases"][:4]):
        directory = tmp_path / "episodes" / case["id"]
        directory.mkdir(parents=True)
        runner.write_json(directory / "result.json", {**case, "status": "complete", "protocol_fingerprint": "abc",
                                                      "success": index in successful, "inference_calls": 2})
        runner.write_json(directory / "cache_samples.json", {
            "count": 2, "feature_contract": {"input_dim": 8, "source": "fixture"},
            "latent_contract": latent_time_contract()})
        np.savez_compressed(directory / "cache_samples.npz", key_features=np.tile(np.arange(8), (2, 1)),
                            intermediate_latents=np.full((2, 50, 32), 2, dtype=np.float32),
                            final_normalized=np.full((2, 50, 7), 3, dtype=np.float32))
        runner.seal_episode(directory)
    return protocol


@pytest.mark.parametrize("successful", [{0, 2}, set()])
def test_success_only_development_admission_and_empty_cache_are_auditable(tmp_path, successful):
    from embodied_research.action_cache import FrozenActionCache
    protocol = development_fixture(tmp_path, successful)
    binding = runner.freeze_cache(tmp_path, protocol)
    assert set(binding["admitted_successful_case_ids"]) == {
        protocol["spec"]["cases"][i]["id"] for i in successful}
    for name, shape, time in (("cache2", (50, 32), .2), ("cache0", (50, 7), 0.)):
        cache = FrozenActionCache.load(tmp_path / "frozen_cache" / name)
        assert len(cache) == 2 * len(successful)
        assert cache.latent_shape == shape and cache.provenance.flow_time == time
        assert cache.is_frozen
    assert runner.freeze_cache(tmp_path, protocol) == binding


def test_resume_rejects_results_from_another_cache_binding(tmp_path):
    protocol = {"spec": runner.study_spec(), "fingerprint": "abc"}
    case = protocol["spec"]["cases"][4]
    directory = tmp_path / "episodes" / case["id"]
    directory.mkdir(parents=True)
    binding = {"cache": "original"}
    runner.write_json(directory / "result.json", {**case, "status": "complete", "protocol_fingerprint": "abc",
                                                  "cache_binding_sha256": runner.digest_json(binding)})
    runner.seal_episode(directory)
    runner.verify_cache_binding(tmp_path, protocol, binding)
    with pytest.raises(ValueError, match="different frozen cache"):
        runner.verify_cache_binding(tmp_path, protocol, {"cache": "changed"})
