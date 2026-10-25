"""Model/simulator-free evaluation contract and sealing regressions."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("action_evaluation_under_test", SCRIPTS / "evaluate_action_adaptation.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def protocol_fixture():
    return {"runs": {"lora17": {"mode": "lora", "initialization_seed": 17, "steps": 2000,
                                "schedule": {"sha256": "s"}}},
            "model": {"models_dir": "models"},
            "data": {"manifest": {"sha256": "d"}, "alignment_evidence": {"sha256": "a"}},
            "resources": {"evaluation_max_seconds": 7200},
            "evaluation": {"num_steps": 10, "replan_steps": 10, "image_size": 360, "arena_bytes": 268435456,
                           "methods": [{"method_id": "frozen", "family": "frozen", "training_run_id": None}],
                           "cases": [{"case_id": "c0", "method_id": "frozen", "suite": "libero_spatial",
                                      "task_id": 0, "task_uid": "libero_spatial:0", "language": "pick the bowl",
                                      "initial_state_id": 5, "condition": "clean"}]}}


def make_trained_fixture(tmp_path, monkeypatch):
    original_resolve = evaluation.resolve
    monkeypatch.setattr(evaluation, "resolve", lambda relative, base=None:
                        original_resolve(relative, tmp_path if base is None else base))
    protocol = protocol_fixture()
    checkpoint = tmp_path / "trained/lora17/step_002000"
    checkpoint.mkdir(parents=True)
    binding = {"protocol_sha256": "p" * 64, "run_id": "lora17", "schedule_sha256": "s",
               "data_manifest_sha256": "d", "alignment_sha256": "a", "input_files": {}}
    binding_sha = evaluation.canonical_sha(binding)
    saved = {"schema": "feedbackbench.action_expert_lora.v1", "base_fingerprint": "b" * 64, "seed": 17,
             "metadata": {"protocol_sha256": "p" * 64, "run_id": "lora17", "binding_sha256": binding_sha,
                          "step": 2000}}
    np.savez(checkpoint / "adapter.npz", manifest_utf8=np.frombuffer(json.dumps(saved).encode(), dtype=np.uint8))
    for filename in ("optimizer.npz", "rng.npz", "state.json", "history.json"):
        (checkpoint / filename).write_bytes(b"fixture checkpoint evidence")
    receipt = {"schema": "feedbackbench.action_adaptation_checkpoint.v1", "completed_steps": 2000,
               "binding": binding, "binding_sha256": binding_sha,
               "files": {path.name: evaluation.digest(path) for path in checkpoint.iterdir()}}
    receipt["receipt_sha256"] = evaluation.canonical_sha(receipt)
    (checkpoint / "receipt.json").write_text(json.dumps(receipt))
    manifest = {"protocol_sha256": "p" * 64, "runs": {"lora17": {
        "mode": "lora", "seed": 17, "base_fingerprint": "b" * 64,
        "adapter": {"path": (checkpoint / "adapter.npz").relative_to(tmp_path).as_posix(),
                    "sha256": evaluation.digest(checkpoint / "adapter.npz")},
        "receipt": {"path": (checkpoint / "receipt.json").relative_to(tmp_path).as_posix(),
                    "sha256": evaluation.digest(checkpoint / "receipt.json")}}}}
    path = tmp_path / "trained_manifest.json"
    path.write_text(json.dumps(manifest))
    return protocol, path, checkpoint


def test_noise_is_paired_across_methods_and_brightness():
    case = protocol_fixture()["evaluation"]["cases"][0]
    paired = {**case, "case_id": "other", "method_id": "head47", "condition": "brightness075"}
    assert evaluation.noise_seed(case, 2) == evaluation.noise_seed(paired, 2)
    assert evaluation.noise_seed(case, 2) != evaluation.noise_seed(case, 3)
    assert evaluation.noise_seed(case, 2) != evaluation.noise_seed({**case, "initial_state_id": 15}, 2)


@pytest.mark.parametrize("development", [None, False, "true", 1])
def test_production_and_nonboolean_development_flag_reject_init_zero(development):
    protocol = protocol_fixture()
    if development is not None:
        protocol["development_only"] = development
    protocol["evaluation"]["cases"][0]["initial_state_id"] = 0
    with pytest.raises(ValueError, match="task/condition/init"):
        evaluation.select_case(protocol, "c0")


def test_explicit_development_protocol_only_allows_init_zero():
    protocol = protocol_fixture()
    protocol["development_only"] = True
    case = protocol["evaluation"]["cases"][0]
    case["initial_state_id"] = 0
    assert evaluation.select_case(protocol, "c0")[0] is case
    for initial_state in (5, 15, 25, 35, 45, False, 0.0):
        case["initial_state_id"] = initial_state
        with pytest.raises(ValueError, match="task/condition/init"):
            evaluation.select_case(protocol, "c0")
    case["initial_state_id"] = 0
    case["task_id"] = 1
    with pytest.raises(ValueError, match="task/condition/init"):
        evaluation.select_case(protocol, "c0")


def test_brightness_only_modifies_policy_images_without_leaking_object_state():
    observation = {"task": "pick", "observation.state": np.arange(8, dtype=np.float32),
                   "observation.images.image": np.ones((3, 8, 8), dtype=np.float32),
                   "observation.images.image2": np.full((3, 8, 8), .4, dtype=np.float32),
                   "object_pose": np.ones(12), "is_success": True}
    changed = evaluation.conditioned_observation(observation, "brightness075")
    np.testing.assert_array_equal(changed["observation.state"], observation["observation.state"])
    np.testing.assert_array_equal(changed["observation.images.image"], np.full((3, 8, 8), .75, dtype=np.float32))
    np.testing.assert_array_equal(observation["observation.images.image"], np.ones((3, 8, 8), dtype=np.float32))
    assert set(changed) == {"task", "observation.state", "observation.images.image", "observation.images.image2"}


def test_all_training_receipts_verified_even_for_frozen_case(tmp_path, monkeypatch):
    protocol, path, checkpoint = make_trained_fixture(tmp_path, monkeypatch)
    manifest = evaluation.verify_trained_manifest(path, evaluation.digest(path), protocol, "p" * 64)
    assert set(manifest["runs"]) == {"lora17"}
    (checkpoint / "optimizer.npz").write_bytes(b"changed nonselected optimizer")
    with pytest.raises(ValueError, match="Sealed checkpoint file changed"):
        evaluation.verify_trained_manifest(path, evaluation.digest(path), protocol, "p" * 64)


@pytest.mark.parametrize("change", ["protocol", "steps", "seed", "run_set"])
def test_manifest_protocol_final_step_seed_and_run_set_are_bound(tmp_path, monkeypatch, change):
    protocol, path, _ = make_trained_fixture(tmp_path, monkeypatch)
    expected = "p" * 64
    if change == "protocol":
        expected = "z" * 64
    elif change == "steps":
        protocol["runs"]["lora17"]["steps"] = 1999
    elif change == "seed":
        protocol["runs"]["lora17"]["initialization_seed"] = 23
    else:
        protocol["runs"]["missing"] = dict(protocol["runs"]["lora17"])
    with pytest.raises(ValueError):
        evaluation.verify_trained_manifest(path, evaluation.digest(path), protocol, expected)


def test_native_inference_counts_actual_ten_projections_without_pretrained_model():
    import torch

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_out_proj = torch.nn.Linear(32, 7)
            self.steps = 10

        def predict_action_chunk(self, batch, noise):
            for _ in range(self.steps):
                output = self.model.action_out_proj(noise)
            return output

    adapter = SimpleNamespace(policy=Policy(), _batch={}, last_timing={"preprocess_seconds": 0, "postprocess_seconds": 0},
                              observe=lambda observation: None,
                              _actions_to_numpy=lambda value: value.detach().numpy().copy())
    actions, timing = evaluation.native_chunk(adapter, {}, 17, torch)
    assert actions.shape == (10, 7) and timing["actual_nfe"] == 10
    assert not adapter.policy.model.action_out_proj._forward_hooks
    adapter.policy.steps = 9
    with pytest.raises(ValueError, match="instead of ten"):
        evaluation.native_chunk(adapter, {}, 17, torch)
    assert not adapter.policy.model.action_out_proj._forward_hooks


def setup_lifecycle(monkeypatch, tmp_path, *, error=None):
    protocol = protocol_fixture()
    campaign = tmp_path / "outputs/action_adaptation/train_v1"
    campaign.mkdir(parents=True)
    path = campaign / "protocol.json"
    path.write_text("fixture")
    output = campaign / "evaluation/case0/attempt_001"
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "verify_protocol", lambda *args: protocol)
    monkeypatch.setattr(evaluation, "verify_trained_manifest", lambda *args: {})

    class Session:
        def __init__(self, output, limit):
            self.output = output
            self.started = time.monotonic()
            output.mkdir(parents=True, exist_ok=False)
            (output / "resources.jsonl").write_text("{}\n")

        def begin(self):
            pass

        def checkpoint(self, *args, **kwargs):
            pass

        def finish(self, status, **details):
            (self.output / "status.json").write_text(json.dumps({"status": status, **details}))

    def execute(args, protocol, case, method, trained, session):
        if error:
            (session.output / "partial.txt").write_text("retained")
            raise error
        return {"status": "complete", "official_success": False, "steps": 280, "case": case}

    monkeypatch.setattr(evaluation, "EvaluationSession", Session)
    monkeypatch.setattr(evaluation, "execute_episode", execute)
    arguments = ["--protocol", str(path), "--protocol-sha256", "p" * 64,
                 "--trained-manifest", str(campaign / "trained_manifest.json"),
                 "--trained-manifest-sha256", "m" * 64, "--case-id", "c0", "--output", str(output)]
    return output, arguments


def test_complete_task_failure_is_sealed_final_and_cannot_be_overwritten(monkeypatch, tmp_path):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path)
    assert evaluation.main(arguments) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["binding"]["case_id"] == "c0"
    assert json.loads((output / "report.json").read_text())["official_success"] is False
    original_receipt = (output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.main(arguments)
    assert (output / "receipt.json").read_bytes() == original_receipt


@pytest.mark.parametrize("error,code,status", [(RuntimeError("simulator failure"), 1, "failed"),
                                               (evaluation.ResourceStop("memory"), 75, "resource_stopped")])
def test_partial_attempts_preserved_without_sealing(monkeypatch, tmp_path, error, code, status):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path, error=error)
    assert evaluation.main(arguments) == code
    assert not (output / "receipt.json").exists()
    assert (output / "partial.txt").read_text() == "retained"
    assert json.loads((output / "status.json").read_text())["status"] == status


def test_late_error_never_rewrites_already_sealed_evidence(monkeypatch, tmp_path):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path)
    original_seal = evaluation.seal

    def seal_then_fail(*args):
        original_seal(*args)
        raise OSError("failure after receipt publication")

    monkeypatch.setattr(evaluation, "seal", seal_then_fail)
    assert evaluation.main(arguments) == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert evaluation.digest(output / "status.json") == receipt["files"]["status.json"]["sha256"]
    assert json.loads((output / "status.json").read_text())["status"] == "complete"


def test_finished_episode_never_passes_another_retryable_resource_guard(monkeypatch, tmp_path):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path)
    stages = []

    def checkpoint(self, stage, **details):
        stages.append(stage)
        if stage == "episode_finished":
            raise evaluation.ResourceStop("memory fell only after official terminal")

    monkeypatch.setattr(evaluation.EvaluationSession, "checkpoint", checkpoint)
    assert evaluation.main(arguments) == 0
    assert "episode_finished" not in stages
    assert (output / "receipt.json").exists()


def test_terminal_marker_prevents_resource_retry_before_final_report(monkeypatch, tmp_path):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path)

    def terminal_then_fail(*args):
        (output / "terminal_observed.json").write_text(json.dumps({"official_success": False}))
        raise evaluation.ResourceStop("late resource failure")

    monkeypatch.setattr(evaluation, "execute_episode", terminal_then_fail)
    assert evaluation.main(arguments) == 1
    assert not (output / "receipt.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "needs_attention"


def test_complete_report_persisted_before_status_or_sealing_failure(monkeypatch, tmp_path):
    output, arguments = setup_lifecycle(monkeypatch, tmp_path)
    original_finish = evaluation.EvaluationSession.finish

    def finish(self, status, **details):
        if status == "complete":
            assert json.loads((output / "report.json").read_text())["status"] == "complete"
            raise evaluation.ResourceStop("finish failed after official terminal")
        return original_finish(self, status, **details)

    monkeypatch.setattr(evaluation.EvaluationSession, "finish", finish)
    assert evaluation.main(arguments) == 1
    assert json.loads((output / "report.json").read_text())["official_success"] is False
    assert json.loads((output / "status.json").read_text())["status"] == "needs_attention"
    assert not (output / "receipt.json").exists()


def test_trajectory_uses_short_independent_temporary_basename(monkeypatch, tmp_path):
    original_replace = evaluation.os.replace
    seen = []

    def replace(source, destination):
        seen.append(source.name)
        original_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", replace)
    path = tmp_path / "partial_actions.npz"
    evaluation.save_trajectory(path, [], [], [], [], [], [])
    assert len(seen) == 1 and len(seen[0]) == 17 and seen[0].startswith(".")
    assert "partial_actions" not in seen[0]
    with np.load(path, allow_pickle=False) as archive:
        assert archive["actions"].shape == (0, 7)


def test_complete_rollout_executes_native_chunk_to_full_horizon_without_real_simulator(monkeypatch, tmp_path):
    import torch
    import embodied_research.runtime as runtime
    import embodied_research.policies as policies
    import embodied_research.simulator as simulator_module
    import embodied_research.action_adaptation as learning
    import budget_startup_memory

    monkeypatch.setattr(runtime, "configure_runtime", lambda root: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(learning, "frozen_base_checksum", lambda policy: "b" * 64)
    monkeypatch.setattr(budget_startup_memory, "reclaim_own_startup_pages",
                        lambda: {"supported": True, "success": True})

    class Adapter:
        def __init__(self, *args, **kwargs):
            assert kwargs["models_dir"] == evaluation.resolve("models")
            self.config = SimpleNamespace(rtc_config=None)
            self.policy = SimpleNamespace(requires_grad_=lambda value: None)
            self.metadata = {"fixture": True}

        def reset(self, seed):
            pass

    class Simulator:
        def __init__(self, *args, **kwargs):
            self.horizon = 3
            self.task = "pick the bowl"
            self.steps = 0

        def observation(self):
            return {"task": self.task, "observation.state": np.full(8, self.steps, dtype=np.float32),
                    "observation.images.image": np.ones((3, 8, 8), dtype=np.float32),
                    "observation.images.image2": np.ones((3, 8, 8), dtype=np.float32)}

        def reset(self, seed, init_state_id):
            assert seed == 1000 and init_state_id == 5
            return self.observation()

        def step(self, action):
            self.steps += 1
            return self.observation(), 0., False, self.steps == self.horizon, {"is_success": False}

        def metadata(self):
            return {"horizon": self.horizon, "fixture": True}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    @contextmanager
    def arena(*args, **kwargs):
        assert kwargs["memory_bytes"] == 268435456
        yield {"native_warnings": []}

    monkeypatch.setattr(policies, "PolicyAdapter", Adapter)
    monkeypatch.setattr(simulator_module, "LiberoSimulator", Simulator)
    monkeypatch.setattr(simulator_module, "SUITE_HORIZONS", {"libero_spatial": 3})
    monkeypatch.setitem(sys.modules, "libero_arena_budget", SimpleNamespace(arena_budget=arena))
    monkeypatch.setitem(sys.modules, "robosuite.utils.binding_utils", SimpleNamespace(MjSim=object))
    monkeypatch.setattr(evaluation, "save_observation_images", lambda *args: None)
    seen = []

    def chunk(adapter, observation, seed, torch):
        seen.append(observation)
        return np.ones((10, 7), dtype=np.float32), {"actual_nfe": 10, "inference_seconds": .1}

    monkeypatch.setattr(evaluation, "native_chunk", chunk)
    protocol = protocol_fixture()
    protocol["resources"].update(torch_threads=2, interop_threads=1)
    case = {**protocol["evaluation"]["cases"][0], "condition": "brightness075"}
    method = protocol["evaluation"]["methods"][0]
    args = SimpleNamespace(protocol_sha256="p" * 64, trained_manifest_sha256="m" * 64)
    stages = []

    def checkpoint(stage, **details):
        stages.append(stage)
        if stage == "decision_complete":
            raise evaluation.ResourceStop("terminal resource guard must not run")

    session = SimpleNamespace(output=tmp_path, started=time.monotonic(), checkpoint=checkpoint)
    report = evaluation.execute_episode(args, protocol, case, method,
                                        {"runs": {"r": {"base_fingerprint": "b" * 64}}}, session)
    assert report["status"] == "complete" and report["official_success"] is False
    assert report["steps"] == 3 and report["decision_count"] == 1 and report["NFE"] == 10
    assert report["completed_episode_may_be_retried"] is False
    assert "decision_complete" not in stages
    marker = json.loads((tmp_path / "terminal_observed.json").read_text())
    assert marker["steps"] == 3 and marker["official_success"] is False
    assert marker["completed_episode_may_be_retried"] is False
    np.testing.assert_array_equal(seen[0]["observation.images.image"], np.full((3, 8, 8), .75, dtype=np.float32))
    with np.load(tmp_path / "actions.npz") as saved:
        assert saved["actions"].shape == (3, 7)
        assert saved["states"].shape == saved["next_states"].shape == (3, 8)
        np.testing.assert_array_equal(saved["states"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(saved["next_states"][:, 0], [1, 2, 3])
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls[0]["executed_steps"] == 3
