"""One complete LIBERO episode; launched serially by run_budget_campaign.py.

Partial attempts never enter the training corpus or success denominator. Every
action is produced by the pinned pretrained policy; only its Euler budget varies.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import traceback

from budget_campaign_common import (ROOT, deadline_reached, digest, owned_path, read_json,
                                    seal_episode, utc_now, verify_protocol, write_json)


class ResourceWait(RuntimeError):
    pass


def noise_seed(case, decision):
    # Same task/initial/decision across all test methods; independent of budget
    # and training seed. After trajectories diverge this pairs RNG, not states.
    identity = f"budget-v1:{case['task_uid']}:{case['initial_state_id']}:{decision}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")


def deviation_targets(candidates, environment_actions):
    import numpy as np
    reference = candidates[10][0, :10, :7]
    rows = []
    for budget in (2, 4):
        candidate = candidates[budget][0, :10, :7]
        difference = candidate - reference
        rows.append([np.sqrt(np.mean(difference[:, :3] ** 2)),
                     np.sqrt(np.mean(difference[:, 3:6] ** 2)),
                     np.mean((environment_actions[budget][0, :10, 6] >= 0)
                             != (environment_actions[10][0, :10, 6] >= 0))])
    return np.asarray(rows, dtype=np.float32)


def run_episode(campaign, phase, case_id, attempt):
    # Compatibility must precede Torch, NumPy.testing and renderer imports.
    from embodied_research.runtime import configure_runtime
    configure_runtime(ROOT)
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from embodied_research.policies import PolicyAdapter
    from embodied_research.simulator import LiberoSimulator
    from embodied_research.cache_sampler import SmolVLACacheSampler
    from embodied_research.budget_sampler import BudgetSampler, extract_budget_features
    from libero_arena_budget import arena_budget
    from robosuite.utils.binding_utils import MjSim

    protocol = verify_protocol(campaign)
    cases = protocol[phase]["cases"]
    case = next(row for row in cases if row["case_id"] == case_id)
    attempt = Path(attempt).resolve()
    if not attempt.is_relative_to(owned_path(campaign, "attempts")) or not attempt.is_dir():
        raise ValueError("worker requires an existing owned attempt directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    limits = protocol["resource_limits"]
    started = time.perf_counter()
    peak_rss, minimum_free = 0, psutil.virtual_memory().available
    current_stage = "initializing"

    def checkpoint(stage, **extra):
        nonlocal peak_rss, minimum_free, current_stage
        current_stage = stage
        rss, free = process.memory_info().rss, psutil.virtual_memory().available
        peak_rss, minimum_free = max(peak_rss, rss), min(minimum_free, free)
        write_json(attempt / "progress.json", {"case_id": case_id, "stage": stage, "updated_at": utc_now(),
                   "rss_bytes": rss, "available_bytes": free, "elapsed_seconds": time.perf_counter() - started,
                   **extra})
        if deadline_reached(protocol):
            raise ResourceWait("hard two-week deadline reached")
        if owned_path(campaign, "controls", attempt.parent.parent.name,
                      attempt.parent.name, attempt.name + ".json").exists():
            raise ResourceWait("supervisor requested resource stop")
        if rss > limits["max_process_rss_bytes"] or free < limits["min_system_available_bytes"]:
            raise ResourceWait("process/system memory guard")

    checkpoint("loading_model")
    adapter = PolicyAdapter("smolvla", num_steps=10, n_action_steps=1)
    adapter.reset(seed=1000)
    base = SmolVLACacheSampler(adapter.policy)
    sampler = BudgetSampler(base)
    checkpoint("model_loaded")
    from budget_startup_memory import reclaim_own_startup_pages
    startup_memory = reclaim_own_startup_pages()
    write_json(attempt / "startup_memory.json", startup_memory)
    if startup_memory["supported"] and not startup_memory["success"]:
        raise RuntimeError("the preregistered one-time worker working-set reclamation failed")
    method = None
    selector = None
    random_probabilities = None
    trained_binding = None
    if phase == "evaluation":
        method = next(row for row in protocol["evaluation"]["methods"] if row["method_id"] == case["method_id"])
        trained = read_json(Path(campaign) / "trained_manifest.json")
        trained_binding = {"manifest_sha256": digest(Path(campaign) / "trained_manifest.json"),
                           "selector": None, "calibration_budget_probabilities": None}
        for relative, record in trained["files"].items():
            path = owned_path(campaign, relative)
            if digest(path) != record["sha256"]:
                raise ValueError("frozen trained controller changed")
        if method["family"] in ("learned", "random_calibrated"):
            from embodied_research.budget_learning import load_selector
            variant = method["feature_mode"] if method["family"] == "learned" else "full"
            selector_relative = f"trained/{variant}_seed{method['seed']}/selector.pt"
            if selector_relative not in trained["files"]:
                raise ValueError("selected controller is absent from frozen trained manifest")
            selector = load_selector(owned_path(campaign, selector_relative),
                                     expected_sha256=trained["files"][selector_relative]["sha256"])
            trained_binding["selector"] = {"path": selector_relative,
                                           "sha256": trained["files"][selector_relative]["sha256"]}
            if (selector.metadata["checkpoint_id"] != adapter.metadata["revision"]
                    or selector.metadata["binding"]["campaign_fingerprint"] != protocol["campaign_fingerprint"]):
                raise ValueError("controller was trained for a different policy or campaign")
            if method["family"] == "random_calibrated":
                random_probabilities = selector.calibration_budget_probabilities
                trained_binding["calibration_budget_probabilities"] = random_probabilities.tolist()
    features_rows, target_rows, step_indices, noise_seeds = [], [], [], []
    actions, states, rewards, calls = [], [], [], []
    terminated = truncated = success = False
    arena_audit = None
    try:
        with ExitStack() as stack:
            arena_audit = stack.enter_context(arena_budget(MjSim, memory_bytes=protocol["runtime"]["arena_bytes"]))
            simulator = stack.enter_context(LiberoSimulator(case["suite"], case["task_id"], image_size=360))
            observation = simulator.reset(seed=1000, init_state_id=case["initial_state_id"])
            import gc
            gc.collect()  # Release obsolete reset/compiler Python objects before inference.
            expected_task = next(row for row in protocol["tasks"] if row["task_uid"] == case["task_uid"])
            from budget_protocol import normalize_language
            if normalize_language(simulator.task) != expected_task["normalized_language"]:
                raise ValueError("simulator task instruction differs from frozen official catalog")
            Image.fromarray(simulator.frame()).save(attempt / "initial.png")
            metadata = {"case_id": case_id, "task_key": case["task_uid"], "official_task_key": case["task_key"],
                        "split": case["split"], "phase": phase, "case": case,
                        "checkpoint_id": adapter.metadata["revision"], "policy": adapter.metadata,
                        "campaign_fingerprint": protocol["campaign_fingerprint"], "simulator": simulator.metadata(),
                        "method": method, "trained_binding": trained_binding,
                        "startup_memory": startup_memory, "started_at": utc_now()}
            write_json(attempt / "metadata.json", metadata)
            while not (terminated or truncated):
                decision, step = len(calls), len(actions)
                checkpoint("preparing_context", decisions=decision, steps=step)
                decision_start = time.perf_counter()
                adapter.observe(observation)
                context = base.prepare(adapter._batch)
                seed = noise_seed(case, decision)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32)
                probe = sampler.probe(context, noise)
                feature_start = time.perf_counter()
                # Fixed/random controls do not pay a learned-controller feature
                # extraction cost. Collection and learned variants do.
                need_features = phase == "collection" or method["family"] == "learned"
                feature = extract_budget_features(context, observation["observation.state"], noise, probe) if need_features else None
                feature_seconds = time.perf_counter() - feature_start
                controller_seconds = 0.0
                candidates = {}
                full_latents = {}
                controller_record = None
                checkpoint("sampling", decisions=decision, steps=step)
                if phase == "collection":
                    nfe = 1
                    for budget in (2, 4, 10):
                        sample = sampler.sample(context, noise, num_steps=budget, probe=probe)
                        nfe += sample.new_evaluations
                        candidates[budget] = sample.actions.detach().numpy().copy()
                        full_latents[budget] = sample.full_latent.detach().numpy().copy()
                        checkpoint("sampling", decisions=decision, steps=step, finished_budget=budget)
                    if nfe != 14:
                        raise RuntimeError("counterfactual collection NFE accounting changed")
                    budget = case["behavior_budget"]
                    environment_actions = {key: adapter._actions_to_numpy(torch.from_numpy(value))
                                           for key, value in candidates.items()}
                    targets = deviation_targets(candidates, environment_actions)
                    features_rows.append(feature.values.copy())
                    target_rows.append(targets)
                    step_indices.append(step)
                    noise_seeds.append(seed)
                    normalized = torch.from_numpy(candidates[budget])
                    metadata.update(feature_slices=feature.feature_slices, feature_layout=feature.layout)
                else:
                    tick = time.perf_counter()
                    if method["family"] == "fixed":
                        budget = method["budget"]
                    elif method["family"] == "random_calibrated":
                        rng = np.random.default_rng(seed ^ (method["seed"] + 1) * 1000003)
                        budget = int(rng.choice([2, 4, 10], p=random_probabilities))
                    else:
                        controller_record = selector.select_budget(feature.values)
                        budget = int(controller_record["budget"])
                        controller_record = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                             for key, value in controller_record.items()}
                    controller_seconds = time.perf_counter() - tick
                    sample = sampler.sample(context, noise, num_steps=budget, probe=probe)
                    normalized = sample.actions
                    candidates[budget] = normalized.detach().numpy().copy()
                    full_latents[budget] = sample.full_latent.detach().numpy().copy()
                    nfe = 1 + sample.new_evaluations
                    if nfe != budget:
                        raise RuntimeError("online budget has hidden probe evaluations")
                output_actions = (environment_actions[budget] if phase == "collection"
                                  else adapter._actions_to_numpy(normalized))[0, :10]
                decision_seconds = time.perf_counter() - decision_start
                evidence = {"noise": noise.numpy(), "first_velocity": probe.velocity.numpy(),
                            "raw_state": observation["observation.state"],
                            "image": simulator.frame("image"), "image2": simulator.frame("image2"),
                            "executed_chunk": output_actions}
                for key, value in full_latents.items():
                    evidence[f"latent_{key}"] = value
                if phase == "collection":
                    for key, value in environment_actions.items():
                        evidence[f"environment_actions_{key}"] = value
                if feature is not None:
                    evidence["features"] = feature.values
                if phase == "collection":
                    evidence["targets"] = targets
                call_dir = attempt / "decisions"
                call_dir.mkdir(exist_ok=True)
                np.savez_compressed(call_dir / f"{decision:03d}.npz", **evidence)
                call = {"decision": decision, "start_step": step, "noise_seed": seed, "budget": budget,
                        "actual_nfe": nfe, "decision_wall_seconds": decision_seconds,
                        "feature_seconds": feature_seconds, "controller_seconds": controller_seconds,
                        "controller": controller_record, "probe_timings": probe.timings,
                        "context_timings": context.timings}
                for action in output_actions:
                    checkpoint("executing", decisions=decision, steps=len(actions))
                    states.append(observation["observation.state"].copy())
                    observation, reward, terminated, truncated, info = simulator.step(action)
                    actions.append(action.copy())
                    rewards.append(reward)
                    success = bool(info.get("is_success", False))
                    if terminated or truncated:
                        break
                call["executed_steps"] = len(actions) - step
                calls.append(call)
                write_json(call_dir / f"{decision:03d}.json", call)
                np.savez_compressed(attempt / "partial_trajectory.npz", actions=np.asarray(actions, dtype=np.float32),
                                    states=np.asarray(states, dtype=np.float32), rewards=np.asarray(rewards))
                write_json(attempt / "metadata.json", metadata)
                write_json(attempt / "arena.json", arena_audit)
                checkpoint("decision_complete", decisions=len(calls), steps=len(actions), success=success)
                del context, probe, normalized, sample, noise
            Image.fromarray(simulator.frame()).save(attempt / "final.png")
        if arena_audit["native_warnings"]:
            raise RuntimeError("native simulation warning; trajectory is not valid evidence")
        np.savez_compressed(attempt / "trajectory.npz", actions=np.asarray(actions, dtype=np.float32),
                            states=np.asarray(states, dtype=np.float32), rewards=np.asarray(rewards))
        # Empty samples for evaluation prevents those states entering training.
        np.savez_compressed(attempt / "samples.npz", features=np.asarray(features_rows, dtype=np.float32),
                            targets=np.asarray(target_rows, dtype=np.float32),
                            step_indices=np.asarray(step_indices, dtype=np.int32), noise_seeds=np.asarray(noise_seeds, dtype=np.uint32))
        write_json(attempt / "arena.json", arena_audit)
        result = {"complete": True, "success": success, "terminated": terminated, "truncated": truncated,
                  "steps": len(actions), "horizon": simulator.horizon, "decisions": len(calls),
                  "actual_nfe": sum(row["actual_nfe"] for row in calls),
                  "decision_wall_seconds": sum(row["decision_wall_seconds"] for row in calls),
                  "controller_overhead_seconds": sum(row["feature_seconds"] + row["controller_seconds"] for row in calls),
                  "episode_wall_seconds": time.perf_counter() - started,
                  "budget_frequencies": {str(k): sum(row["budget"] == k for row in calls) for k in (2, 4, 10)},
                  "peak_rss_bytes": peak_rss, "minimum_available_bytes": minimum_free, "completed_at": utc_now()}
        write_json(attempt / "result.json", result)
        seal_episode(attempt)
        return result
    except BaseException as error:
        write_json(attempt / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "stage": current_stage, "steps": len(actions),
                   "decisions": len(calls), "completed": False, "at": utc_now(),
                   "peak_rss_bytes": peak_rss, "minimum_available_bytes": minimum_free})
        if arena_audit is not None:
            write_json(attempt / "arena.json", arena_audit)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("collection", "evaluation"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run_episode(args.campaign, args.phase, args.case_id, args.attempt)))
    except ResourceWait:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
