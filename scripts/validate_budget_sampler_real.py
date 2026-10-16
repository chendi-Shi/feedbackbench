"""Opt-in real SmolVLA budget parity on one training-task reset observation.

Creates a fresh private evidence directory. This script never fits a selector,
executes a policy action in the simulator, or reads a held-out task. The one
LIBERO environment is closed before loading the policy to avoid overlapping
renderer/model allocations. Timings are diagnostics, not speed comparisons.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/validate_budget_sampler_real.py",
    "src/embodied_research/budget_sampler.py", "src/embodied_research/cache_sampler.py",
    "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
    "src/embodied_research/simulator.py", "scripts/libero_arena_budget.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
    "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
    "vendor/MINERVA/src/lerobot/policies/common/flow_matching.py",
    "models/smolvla/config.json", "models/smolvla/download_manifest.json",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_array(value):
    return value.detach().cpu().numpy().copy()


def tensor_digest(value):
    return hashlib.sha256(tensor_array(value).tobytes(order="C")).hexdigest()


def kv_length(context):
    getter = getattr(context.past_key_values, "get_seq_length", None)
    if not callable(getter):
        raise ValueError("Real parity requires an inspectable prefix KV length")
    length = int(getter())
    if length != context.prefix_pad_masks.shape[1]:
        raise ValueError("Prefix KV length has not been restored")
    return length


@contextmanager
def capture_denoise(model, context):
    """Count real calls and copy the last input/velocity without changing output.

    The frozen reference exposes only seven output channels. Its final full
    latent is exactly last_input + (-1/K) * last_velocity. Both tensors are
    copied, with the same dtype, before the reference performs its final update.
    The original method descriptor/instance override is restored on every exit.
    """
    original = model.denoise_step
    existed = "denoise_step" in vars(model)
    old_attribute = vars(model).get("denoise_step")
    audit = {"phase": None, "calls": [], "last_input": None, "last_velocity": None}

    @wraps(original)
    def measured(*args, **kwargs):
        if args or not {"x_t", "timestep", "past_key_values", "prefix_pad_masks"}.issubset(kwargs):
            raise ValueError("Unexpected denoise API in real parity capture")
        if (kwargs["past_key_values"] is not context.past_key_values
                or kwargs["prefix_pad_masks"] is not context.prefix_pad_masks):
            raise ValueError("Denoise used a different prefix context")
        row = {"phase": audit["phase"], "native_time": float(kwargs["timestep"][0]),
               "kv_length_before": kv_length(context), "input_sha256": tensor_digest(kwargs["x_t"])}
        audit["last_input"] = kwargs["x_t"].detach().clone()
        started = time.perf_counter()
        velocity = original(**kwargs)
        row["seconds"] = time.perf_counter() - started
        audit["last_velocity"] = velocity.detach().clone()
        row.update(kv_length_after=kv_length(context), velocity_sha256=tensor_digest(velocity))
        audit["calls"].append(row)
        return velocity

    model.denoise_step = measured
    try:
        yield audit
    finally:
        if existed:
            model.denoise_step = old_attribute
        else:
            delattr(model, "denoise_step")


def compare_tensors(left, right):
    import numpy as np

    a, b = tensor_array(left), tensor_array(right)
    same = a.shape == b.shape and a.dtype == b.dtype
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "same_shape_and_dtype": same,
            "value_equal": bool(same and np.array_equal(a, b)),
            "byte_equal": bool(same and a.tobytes(order="C") == b.tobytes(order="C")),
            "max_abs_difference": float(np.max(np.abs(a.astype(np.float64) - b))) if same else None}


def run(output):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if not output.is_relative_to((ROOT / ".runtime").resolve()):
        raise ValueError("Real parity evidence must stay in a fresh private .runtime directory")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "feedbackbench.budget_real_parity.v1", "status": "running", "phase": "initializing",
        "started_at": utc_now(), "case": {"suite": "libero_spatial", "task_id": 0,
                                            "init_state_id": 0, "seed": 1000, "split": "train"},
        "scope": "One real training observation; sampler correctness only, zero policy actions or benchmark episodes",
        "execution_order": "reset under 256MiB arena and save observation; close environment; load policy once",
        "noise_seed": 37, "budgets": [2, 4, 10], "threads": 2, "interop_threads": 1,
        "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCES},
        "comparisons": {}, "files": {},
    }
    arrays = {}
    report_path = output / "report.json"
    write_json(report_path, record)
    try:
        from embodied_research.runtime import configure_runtime
        record["runtime"] = configure_runtime(ROOT)
        import numpy as np
        import torch
        from PIL import Image
        from robosuite.utils.binding_utils import MjSim
        from libero_arena_budget import arena_budget, check_simulator_capacity
        from embodied_research.budget_sampler import BudgetSampler, extract_budget_features
        from embodied_research.cache_sampler import SmolVLACacheSampler
        from embodied_research.policies import PolicyAdapter
        from embodied_research.simulator import LiberoSimulator

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        record["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "numpy", "transformers", "mujoco", "robosuite")}
        record["phase"] = "reset_training_observation"
        write_json(report_path, record)
        print("Resetting only training task libero_spatial/task0/init0 under 256MiB arena...", flush=True)
        simulator = None
        with arena_budget(MjSim) as arena:
            record["arena"] = arena
            try:
                simulator = LiberoSimulator("libero_spatial", 0, project_root=ROOT, image_size=360)
                observation = simulator.reset(seed=1000, init_state_id=0)
                if arena["native_warnings"] or not arena["compilations"]:
                    raise RuntimeError("Arena compilation missing or native warning during reset")
                for compiled in arena["compilations"]:
                    if (compiled["status"] != "compiled" or not compiled["nconmax_preserved"]
                            or not compiled["outside_size_opening_tag_bytes_unchanged"]):
                        raise RuntimeError("Invalid arena compilation audit")
                record["arena_after_reset"] = check_simulator_capacity(simulator.env._env.sim)
                record["environment"] = simulator.metadata()
                np.savez_compressed(output / "observation.npz", **observation)
                Image.fromarray(simulator.frame()).save(output / "initial.png")
                Image.fromarray(simulator.frame("image2")).save(output / "initial_wrist.png")
                record["policy_actions_executed"] = simulator.steps
            finally:
                if simulator is not None:
                    simulator.close()
                    simulator = None
        gc.collect()
        if record["policy_actions_executed"] != 0:
            raise RuntimeError("Parity script unexpectedly executed a policy action")
        record["phase"] = "strict_model_loading"
        write_json(report_path, record)
        print("Strictly loading the pinned SmolVLA checkpoint once...", flush=True)
        adapter = PolicyAdapter("smolvla", num_steps=10, temporal_ensemble_coeff=None)
        record["policy"] = adapter.metadata
        adapter.reset(seed=1000)
        adapter.observe(observation)
        sampler = SmolVLACacheSampler(adapter.policy)
        budget = BudgetSampler(sampler)
        context = sampler.prepare(adapter._batch)
        noise = torch.randn((1, 50, 32), dtype=torch.float32,
                            generator=torch.Generator(device="cpu").manual_seed(37))
        arrays["noise"] = tensor_array(noise)
        record["context"] = {"feature_contract": context.feature_contract,
                             "timings": context.timings, "kv_length": kv_length(context),
                             "kv_persistence": "KV is regenerated from saved observation and pinned model; not serialized"}
        np.savez_compressed(output / "context.npz", key_features=context.key_features,
                            normalized_state=tensor_array(context.state),
                            prefix_pad_masks=tensor_array(context.prefix_pad_masks),
                            language_tokens=tensor_array(adapter._batch["observation.language.tokens"]),
                            language_mask=tensor_array(adapter._batch["observation.language.attention_mask"]))
        before_rng = torch.get_rng_state().clone()
        noise_before = noise.clone()
        references = {}
        record["phase"] = "reference_and_shared_sampling"
        write_json(report_path, record)
        with torch.inference_mode(), capture_denoise(sampler.model, context) as captured:
            record["denoise_calls"] = captured["calls"]
            for steps in (2, 4, 10):
                captured["phase"] = f"reference_{steps}"
                print(f"Frozen reference: {steps} steps...", flush=True)
                reference = sampler.sample(context, num_steps=steps, noise=noise)
                endpoint = captured["last_input"] + (-1.0 / steps) * captured["last_velocity"]
                check = compare_tensors(endpoint[:, :, :7], reference.actions)
                if not check["byte_equal"]:
                    raise AssertionError("Captured full endpoint disagrees with frozen reference actions")
                references[steps] = (reference.actions, endpoint)
                arrays[f"reference_{steps}_actions"] = tensor_array(reference.actions)
                arrays[f"reference_{steps}_full_latent"] = tensor_array(endpoint)
            captured["phase"] = "shared_probe"
            probe = budget.probe(context, noise)
            arrays["first_velocity"] = tensor_array(probe.velocity)
            record["probe"] = {"nfe": probe.nfe, "noise_sha256": probe.noise_sha256,
                               "velocity_sha256": probe.velocity_sha256,
                               "context_metadata_sha256": probe.context_metadata_sha256,
                               "timings": probe.timings}
            features = extract_budget_features(context, observation["observation.state"], noise, probe)
            arrays["budget_features"] = features.values
            record["feature_layout"], record["feature_slices"] = features.layout, features.feature_slices
            for steps in (2, 4, 10):
                captured["phase"] = f"shared_{steps}"
                print(f"Shared probe: {steps}-step grid, {steps - 1} new evaluations...", flush=True)
                sampled = budget.sample(context, noise, steps, probe)
                arrays[f"shared_{steps}_actions"] = tensor_array(sampled.actions)
                arrays[f"shared_{steps}_full_latent"] = tensor_array(sampled.full_latent)
                checks = {"actions": compare_tensors(sampled.actions, references[steps][0]),
                          "full_latent": compare_tensors(sampled.full_latent, references[steps][1]),
                          "nfe": sampled.nfe, "new_evaluations": sampled.new_evaluations,
                          "timings": sampled.timings, "kv_length": kv_length(context)}
                record["comparisons"][str(steps)] = checks
                if not checks["actions"]["byte_equal"] or not checks["full_latent"]["byte_equal"]:
                    raise AssertionError(f"Shared-probe {steps}-step output differs from the frozen sampler")
                write_json(report_path, record)
        record["rng_unchanged_during_sampling"] = torch.equal(before_rng, torch.get_rng_state())
        record["noise_unchanged"] = compare_tensors(noise_before, noise)["byte_equal"]
        counts = {phase: sum(call["phase"] == phase for call in captured["calls"])
                  for phase in ("reference_2", "reference_4", "reference_10", "shared_probe",
                                "shared_2", "shared_4", "shared_10")}
        record["actual_evaluations"] = {"by_phase": counts,
                                        "reference_total": sum(counts[f"reference_{k}"] for k in (2, 4, 10)),
                                        "shared_total": sum(value for key, value in counts.items() if key.startswith("shared")),
                                        "total": len(captured["calls"])}
        if (counts != {"reference_2": 2, "reference_4": 4, "reference_10": 10,
                      "shared_probe": 1, "shared_2": 1, "shared_4": 3, "shared_10": 9}
                or not record["rng_unchanged_during_sampling"] or not record["noise_unchanged"]):
            raise AssertionError("NFE or RNG/noise contract differs")
        if {name: file_sha256(ROOT / name) for name in SOURCES} != record["source_sha256"]:
            raise AssertionError("Parity source changed during execution")
        record.update(status="passed", phase="complete", finished_at=utc_now())
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      traceback=traceback.format_exc(), finished_at=utc_now())
        raise
    finally:
        if arrays:
            import numpy as np
            np.savez_compressed(output / "outputs.npz", **arrays)
        record["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                           for path in sorted(output.iterdir()) if path.is_file() and path.name != "report.json"}
        write_json(report_path, record)
    print(json.dumps({"status": record["status"], "shared_actual_nfe": 14,
                      "full_latent_byte_parity": True, "report": str(report_path)}, indent=2), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/budget-real-parity"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
