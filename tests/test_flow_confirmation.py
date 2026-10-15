"""Prospective design and evidence tests; no model or environment imports."""
import hashlib
import importlib.util
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_flow_confirmation.py"
SPEC = importlib.util.spec_from_file_location("flow_confirmation_runner_test", PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def project_fixture(tmp_path):
    root = tmp_path / "project"
    for name in runner.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source: " + name, encoding="utf-8")
    old_sources = {name: runner.file_sha(root / name) for name in runner.SOURCES[3:]}
    artifacts = {"models": {"smolvla": {"revision": "fixture-revision", "files": {"model.safetensors": "a" * 64}}}}
    v1 = {"spec": {"fixture": True}, "sources": old_sources, "runtime_versions": {}, "artifacts": artifacts}
    v1["fingerprint"] = runner.digest_json(v1)
    runner.write_json(root / runner.ORIGIN_PATHS["campaign_freeze"], {"source_sha256": old_sources})
    runner.write_json(root / runner.ORIGIN_PATHS["v1_protocol"], v1)
    runner.write_json(root / runner.ORIGIN_PATHS["v1_report"], {"status": "complete",
        "counts": {"complete": 28, "planned": 28}, "protocol_fingerprint": v1["fingerprint"]})
    runner.write_json(root / runner.ORIGIN_PATHS["v1_exports"], {"schema": "feedbackbench.cache_study_exports.v1",
        "files": {"report.json": runner.file_sha(root / runner.ORIGIN_PATHS["v1_report"])}})
    origin, _ = runner.verify_origins(root)
    states = {"tasks": [{"suite": suite, "task_id": 1, "count": 50, "shape": [50, 8],
                          "sha256": "b" * 64} for suite in runner.SUITES]}
    body = {"schema": "feedbackbench.flow_confirmation_protocol.v1", "spec": runner.study_spec(),
            "sources": {name: runner.file_sha(root / name) for name in runner.SOURCES},
            "runtime_versions": {}, "artifacts": artifacts, "origin": origin, "initial_states": states}
    protocol = {**body, "fingerprint": runner.digest_json(body)}
    return root, protocol


def study_fixture(tmp_path):
    root, protocol = project_fixture(tmp_path)
    output = root / "outputs/frontier/flow_confirmation"
    runner.write_json(output / "protocol.json", protocol)
    return root, output, protocol


def episode_payload(directory, case, protocol, *, noise="c" * 64, initial_value=0):
    directory.mkdir(parents=True, exist_ok=True)
    pixels = np.full((2, 2, 3), initial_value, dtype=np.uint8)
    image = Image.fromarray(pixels)
    for name in ("initial.png", "initial_wrist.png", "final.png"):
        image.save(directory / name)
    image.save(directory / "rollout.gif")
    np.savez_compressed(directory / "trajectory.npz", actions=np.zeros((10, 7), dtype=np.float32),
                        states=np.zeros((10, 8), dtype=np.float32), replan_steps=np.array([0]))
    calls = [{"call_index": 0, "observation_count": 1, "mode": case["method"],
              "nfe": case["num_steps"], "hit": None, "noise_sha256": noise,
              **{key: 1.0 for key in runner.TIMINGS}}]
    record = {**case, "status": "complete", "protocol_fingerprint": protocol["fingerprint"],
              "hypothesis_origin_sha256": runner.digest_json(protocol["origin"]),
              "success": True, "terminated": True, "truncated": False, "steps": 10,
              "horizon": runner.HORIZONS[case["suite"]], "inference_calls": 1, "wall_seconds": 2.0,
              "policy_inference_seconds": 1.0, "initial_robot_state": [0.] * 8,
              "scheduler_config": runner.asdict(runner.SchedulerConfig(**case["scheduler"], horizon=runner.HORIZONS[case["suite"]])),
              "environment": {"suite": case["suite"], "task_id": 1, "horizon": runner.HORIZONS[case["suite"]],
                              "control_hz": 20, "control_mode": "relative", "hard_reset": True,
                              "settle_steps": 10, "image_size": 360, "images_flipped_hw_once": True,
                              "object_state_exposed_to_policy": False},
              "initial_image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
              "model_info": {"revision": "fixture-revision", "model_sha256": "a" * 64,
                             "cache_policy": {"mode": case["method"], "cache": None,
                                              "call_log": calls, "total_nfe": case["num_steps"]}}}
    return record, calls


def complete_case(output, case, protocol, **kwargs):
    directory = output / "episodes" / case["id"]
    record, calls = episode_payload(directory, case, protocol, **kwargs)
    runner.write_json(directory / "attempt.json", {"case": case, "protocol_fingerprint": protocol["fingerprint"]})
    runner.write_json(directory / "calls.json", calls)
    runner.write_json(directory / "result.json", record)
    runner.seal_episode(directory)
    return directory


def reseal(directory):
    (directory / "receipt.json").unlink()
    runner.seal_episode(directory)


def test_design_has_fresh_counterbalanced_eight_pairs_four_tasks():
    cases = runner.cases_for_study()
    assert len(cases) == len({case["id"] for case in cases}) == 16
    first_methods = []
    for index in range(0, len(cases), 2):
        pair = cases[index:index + 2]
        assert {case["method"] for case in pair} == {"full10", "noise2"}
        assert len({(case["suite"], case["task_id"], case["init_state_id"]) for case in pair}) == 1
        assert all(case["init_state_id"] in (41, 43) for case in pair)
        first_methods.append(pair[0]["method"])
    assert first_methods.count("full10") == first_methods.count("noise2") == 4
    assert len({case["suite"] for case in cases}) == 4
    assert not set(runner.INITIAL_STATES) & {0, 1, 2, 11, 29}


@pytest.mark.parametrize("shape", [(50, 92), (50, 110), (50, 79), (50, 123)])
def test_initial_state_shape_is_read_without_pickle_execution(tmp_path, monkeypatch, shape):
    path = tmp_path / "states.pruned_init"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(np.zeros(shape, dtype="<f8"), protocol=2))
        archive.writestr("archive/version", b"3\n")
    monkeypatch.setattr(pickle, "loads", lambda *args, **kwargs: pytest.fail("must not unpickle"))
    assert runner.initial_archive_shape(path) == shape


@pytest.mark.parametrize("kind", ["object", "float32", "nonfinite", "extra", "truncated"])
def test_unrecognized_initial_state_archives_fail_closed(tmp_path, kind):
    path = tmp_path / "states.pruned_init"
    value = np.zeros((50, 8), dtype="<f8")
    if kind == "object":
        value = {"unsafe": "not an ndarray"}
    elif kind == "float32":
        value = value.astype(np.float32)
    elif kind == "nonfinite":
        value[0, 0] = np.nan
    data = pickle.dumps(value, protocol=2)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", data[:-1] if kind == "truncated" else data)
        archive.writestr("archive/version", b"3\n")
        if kind == "extra":
            archive.writestr("other", b"x")
    with pytest.raises(ValueError):
        runner.initial_archive_shape(path)


def test_protocol_binds_design_sources_and_completed_hypothesis_origin(tmp_path):
    root, protocol = project_fixture(tmp_path)
    runner.verify_protocol(protocol, root)
    (root / runner.SOURCES[0]).write_text("changed implementation")
    with pytest.raises(ValueError, match="Frozen confirmation source"):
        runner.verify_protocol(protocol, root)


def test_predecessor_source_changes_rejected_before_freeze(tmp_path):
    root, _ = project_fixture(tmp_path)
    (root / runner.SOURCES[3]).write_text("changed predecessor")
    with pytest.raises(ValueError, match="Predecessor frozen"):
        runner.verify_origins(root)


def test_origin_report_tampering_rejected(tmp_path):
    root, protocol = project_fixture(tmp_path)
    path = root / runner.ORIGIN_PATHS["v1_report"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="v1 export changed"):
        runner.verify_protocol(protocol, root)


def test_freeze_only_and_exact_resume_never_call_runtime(tmp_path, monkeypatch):
    root, protocol = project_fixture(tmp_path)
    output = root / "new-study"
    monkeypatch.setattr(runner, "make_protocol", lambda root: protocol)
    def forbidden(root):
        pytest.fail("freeze must not import a model/runtime")
    assert runner.run(output, freeze_only=True, root=root, runtime_factory=forbidden)["total"] == 16
    saved = (output / "protocol.json").read_bytes()
    runner.run(output, freeze_only=True, root=root, runtime_factory=forbidden)
    assert (output / "protocol.json").read_bytes() == saved
    protocol["runtime_versions"] = {"changed": "yes"}
    protocol["fingerprint"] = runner.digest_json({key: protocol[key] for key in runner.PROTOCOL_FIELDS})
    with pytest.raises(ValueError, match="refusing resume"):
        runner.run(output, freeze_only=True, root=root, runtime_factory=forbidden)


@pytest.mark.parametrize("kind", ["bytes", "extra", "missing", "unsealed", "error", "case", "attempt"])
def test_changed_or_partial_episodes_cannot_be_silently_reused(tmp_path, kind):
    root, output, protocol = study_fixture(tmp_path)
    case = runner.cases_for_study()[0]
    directory = complete_case(output, case, protocol)
    if kind == "bytes":
        (directory / "final.png").write_bytes(b"changed")
    elif kind == "extra":
        (directory / "extra.json").write_text("{}")
    elif kind == "missing":
        (directory / "rollout.gif").unlink()
    elif kind == "unsealed":
        (directory / "receipt.json").unlink()
    elif kind in ("error", "case"):
        record = runner.read_json(directory / "result.json")
        record["status" if kind == "error" else "seed"] = "error" if kind == "error" else 999
        runner.write_json(directory / "result.json", record)
        reseal(directory)
    else:
        attempt = runner.read_json(directory / "attempt.json")
        attempt["case"]["seed"] = 999
        runner.write_json(directory / "attempt.json", attempt)
        reseal(directory)
    with pytest.raises((ValueError, RuntimeError)):
        runner.verify_complete(directory, case, protocol["fingerprint"])


def test_complete_study_checks_pairing_and_separate_denominator(tmp_path):
    root, output, protocol = study_fixture(tmp_path)
    for case in runner.cases_for_study():
        complete_case(output, case, protocol)
    study = runner.verify_study(output, root=root)
    assert study["complete"] and len(study["records"]) == 16
    assert study["pairing_audit"]["paired_initial_blocks"] == 8
    assert study["pairing_audit"]["same_ordinal_noise_pairs_checked"] == 8


@pytest.mark.parametrize("kind", ["noise", "pixels", "wrist", "early_failure", "nfe", "trajectory", "scheduler", "environment"])
def test_valid_receipts_do_not_hide_scientifically_inconsistent_records(tmp_path, kind):
    root, output, protocol = study_fixture(tmp_path)
    first, second = runner.cases_for_study()[:2]
    complete_case(output, first, protocol)
    directory = complete_case(output, second, protocol, noise="d" * 64 if kind == "noise" else "c" * 64,
                              initial_value=1 if kind == "pixels" else 0)
    if kind in ("early_failure", "nfe", "scheduler", "environment"):
        record = runner.read_json(directory / "result.json")
        if kind == "early_failure":
            record["success"] = False
        elif kind == "nfe":
            record["model_info"]["cache_policy"]["call_log"][0]["nfe"] = 3
            runner.write_json(directory / "calls.json", record["model_info"]["cache_policy"]["call_log"])
        elif kind == "scheduler":
            record["scheduler_config"]["fixed_interval"] = 8
        else:
            record["environment"]["control_hz"] = 30
        runner.write_json(directory / "result.json", record)
    elif kind == "trajectory":
        np.savez_compressed(directory / "trajectory.npz", actions=np.zeros((10, 7)), states=np.zeros((10, 8)),
                            replan_steps=np.array([1]))
    elif kind == "wrist":
        Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8)).save(directory / "initial_wrist.png")
    reseal(directory)
    with pytest.raises(ValueError):
        runner.verify_study(output, partial=True, root=root)


def test_partial_study_exposes_missing_and_failed_but_execution_refuses_retry(tmp_path, monkeypatch):
    root, output, protocol = study_fixture(tmp_path)
    case = runner.cases_for_study()[0]
    directory = output / "episodes" / case["id"]
    runner.write_json(directory / "result.json", {"status": "error"})
    study = runner.verify_study(output, partial=True, root=root)
    assert study["statuses"][case["id"]] == "error"
    assert list(study["statuses"].values()).count("missing") == 15
    monkeypatch.setattr(runner, "make_protocol", lambda root: protocol)
    with pytest.raises(RuntimeError, match="Partial attempt"):
        runner.run(output, root=root, runtime_factory=lambda root: pytest.fail("must fail before runtime"))


def test_fake_serial_run_limit_and_resume_warm_current_case_only(tmp_path, monkeypatch):
    root, protocol = project_fixture(tmp_path)
    output = root / "serial-study"
    monkeypatch.setattr(runner, "make_protocol", lambda root: protocol)
    resets, warm_calls = [], []

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed):
            pass

        def observe(self, observation):
            pass

        def predict_chunk(self):
            warm_calls.append(1)

    class Simulator:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed, init_state_id):
            resets.append(init_state_id)
            return {}

        def close(self):
            pass

    class Policy:
        def __init__(self, *args):
            self.call_log = []

        def reset(self, seed):
            pass

    def execute(simulator, policy, case, directory):
        simulator.reset(seed=case["seed"], init_state_id=case["init_state_id"])
        policy.reset(seed=case["seed"])
        record, policy.call_log = episode_payload(directory, case, protocol)
        return record

    kwargs = {"root": root, "runtime_factory": lambda root: (Adapter, Policy, Simulator, lambda: {}),
              "episode_runner": execute, "limit": 1}
    assert runner.run(output, **kwargs) == {"completed": 1, "total": 16}
    first_path = output / "episodes" / runner.cases_for_study()[0]["id"] / "receipt.json"
    first_receipt = first_path.read_bytes()
    assert runner.run(output, **kwargs) == {"completed": 2, "total": 16}
    assert runner.run(output, **kwargs) == {"completed": 3, "total": 16}
    assert first_path.read_bytes() == first_receipt
    assert resets == [41, 41, 41, 41, 43, 43] and len(warm_calls) == 6
    assert not (output / "active.lock").exists()


def test_failed_fake_run_preserves_attempt_and_error_and_releases_lock(tmp_path, monkeypatch):
    root, protocol = project_fixture(tmp_path)
    output = root / "failed-study"
    monkeypatch.setattr(runner, "make_protocol", lambda root: protocol)

    def broken_adapter(*args, **kwargs):
        raise RuntimeError("fixture loading error")

    with pytest.raises(RuntimeError, match="fixture loading error"):
        runner.run(output, root=root, runtime_factory=lambda root: (broken_adapter, None, None, lambda: {}))
    case = runner.cases_for_study()[0]
    directory = output / "episodes" / case["id"]
    assert (directory / "attempt.json").exists()
    assert runner.read_json(directory / "result.json")["status"] == "error"
    assert not (directory / "receipt.json").exists() and not (output / "active.lock").exists()


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for value in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            runner.read_json(path)
