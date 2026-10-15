"""Profile real CPU SmolVLA components, with an explicitly oracle continuation.

This is a fixed-observation timing experiment, not a retrieval experiment or a
success benchmark. Each measured condition recomputes the complete current
observation prefix. The oracle receives its own exact intermediate latent from
a warmup; its cost excludes producing that latent and excludes cache lookup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from embodied_research.runtime import configure_runtime


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, default=Path("artifacts/policy_smoke_observation.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/frontier/profile.json"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    root = Path(__file__).resolve().parents[1]
    observation_path = root / args.observation
    output = root / args.output
    arrays_path = output.with_suffix(".npz")
    progress_path = output.with_suffix(".progress.json")
    for path in (output, arrays_path, progress_path):
        if path.exists():
            parser.error(f"refusing to overwrite existing evidence: {path}")
    if not observation_path.is_file():
        parser.error(f"saved real observation missing: {observation_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = configure_runtime(root)
    print("Importing CPU runtime for one serial real-model profile...", flush=True)
    import numpy as np
    import psutil
    import torch

    from embodied_research.cache_sampler import SmolVLACacheSampler
    from embodied_research.policies import PolicyAdapter

    torch.set_num_threads(args.threads)
    process = psutil.Process()

    def memory() -> dict:
        virtual = psutil.virtual_memory()
        return {"process_rss_bytes": process.memory_info().rss,
                "system_available_bytes": virtual.available, "system_total_bytes": virtual.total}

    source_paths = [
        "scripts/profile_cache_sampler.py", "src/embodied_research/cache_sampler.py",
        "src/embodied_research/policies.py", "src/embodied_research/runtime.py",
        "vendor/MINERVA/src/lerobot/policies/smolvla/modeling_smolvla.py",
        "vendor/MINERVA/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
        "vendor/MINERVA/src/lerobot/policies/common/flow_matching.py",
        "models/smolvla/download_manifest.json", "models/smolvla/config.json",
    ]
    record = {
        "schema_version": 1, "status": "running", "started_utc": utc_now(),
        "scope": "One saved real LIBERO observation; repeated CPU timing, not independent task samples.",
        "conditions": {
            "full10": "Ten Euler updates from the fixed noise; fresh current-observation prefix.",
            "noise2": "Two Euler updates on the coarse full interval from the identical fixed noise.",
            "oracle2": "Own same-observation/same-noise warmup latent at native t=.2, then two original-grid updates."
        },
        "limitations": [
            "oracle2 is an ideal continuation upper bound; it does not retrieve any cached entry.",
            "No measured condition includes projection/search, cache construction or simulator costs.",
            "oracle2 excludes producing the supplied latent; this is not an achievable cold-query cost.",
            "No success/quality-retaining speedup or real cache benefit is established.",
            "Memory values are before/after snapshots, not sampled or system-wide peak memory.",
            "Three timing repeats share one observation and noise; they are not three task replications.",
        ],
        "timing_scope": {
            "total_seconds": "Fresh adapter.observe + sampler.prepare + sample/refine + action postprocessing.",
            "outside_total": "Precreated fixed noise, memory snapshots, array exports, equality checks and file I/O.",
            "nested_components": "Context/result timings overlap prepare/sample wall timers; do not sum all fields.",
            "full10_capture": False,
        },
        "runtime": {**runtime, "python": sys.version, "platform": platform.platform(),
                    "machine": platform.machine(), "executable": sys.executable,
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "pid": process.pid, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                    "versions": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "torch", "transformers", "psutil")}},
        "source_sha256": {name: sha256(root / name) for name in source_paths},
        "observation": {"path": str(observation_path.relative_to(root)),
                        "sha256": sha256(observation_path)},
        "noise_seed": 37, "noise_shared_across_all_conditions_and_repeats": True,
        "warmups": [], "samples": [], "memory_before_model_load": memory(),
    }

    def progress() -> None:
        progress_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    progress()
    print("Strictly loading pinned SmolVLA checkpoint once...", flush=True)
    started = time.perf_counter()
    adapter = PolicyAdapter("smolvla", num_steps=10, temporal_ensemble_coeff=None)
    record.update(total_adapter_construction_seconds=time.perf_counter() - started,
                  policy=adapter.metadata, memory_after_model_load=memory())
    progress()
    with np.load(observation_path, allow_pickle=False) as snapshot:
        observation = {key: snapshot[key].copy() for key in snapshot.files}
    observation["task"] = str(observation["task"].item())
    record["observation"].update(task=observation["task"], shapes={
        key: list(value.shape) for key, value in observation.items() if key != "task"})
    adapter.reset(seed=123)
    sampler = SmolVLACacheSampler(adapter.policy)
    noise = torch.randn((1, adapter.config.chunk_size, adapter.config.max_action_dim),
                        generator=torch.Generator().manual_seed(37), dtype=torch.float32)
    evidence = {"noise": noise.numpy().copy()}
    own_latent = None

    def measure(method: str, *, capture: bool = False):
        before = memory()
        started = time.perf_counter()
        tick = time.perf_counter()
        adapter.observe(observation)
        pre = time.perf_counter() - tick
        tick = time.perf_counter()
        context = sampler.prepare(adapter._batch)
        prepare = time.perf_counter() - tick
        tick = time.perf_counter()
        if method == "oracle2":
            result = sampler.refine(context, own_latent, full_num_steps=10, start_step=8)
        else:
            result = sampler.sample(context, num_steps=10 if method == "full10" else 2,
                                    noise=noise, capture_step=8 if capture else None)
        sample = time.perf_counter() - tick
        tick = time.perf_counter()
        robot_actions = adapter._actions_to_numpy(result.actions)
        post = time.perf_counter() - tick
        total = time.perf_counter() - started
        after = memory()
        normalized = result.actions.detach().cpu().numpy().copy()
        measurement = {
            "method": method, "total_seconds": total, "preprocess_seconds": pre,
            "prepare_wall_seconds": prepare, "sample_wall_seconds": sample,
            "postprocess_seconds": post, "context_timings": context.timings,
            "sampling_timings": result.timings, "nfe": result.nfe,
            "memory_before": before, "memory_after": after,
            "feature_sha256": hashlib.sha256(context.key_features.tobytes()).hexdigest(),
            "action_shape": list(normalized.shape),
        }
        return measurement, result.latent, normalized, robot_actions, context.feature_contract

    for index in range(2):
        print(f"Warmup {index + 1}/2: fresh context + full10, capturing own latent...", flush=True)
        measurement, own_latent, reference_normalized, reference_robot, contract = measure("full10", capture=True)
        if own_latent is None:
            raise AssertionError("warmup did not return the requested intermediate latent")
        record["warmups"].append({"index": index, **measurement})
        record["feature_contract"] = contract
        record["latent_contract"] = sampler.latent_contract
        progress()
        print(f"Warmup {index + 1}: {measurement['total_seconds']:.3f}s", flush=True)
    evidence.update(own_warmup_latent=own_latent, reference_normalized=reference_normalized,
                    reference_robot_actions=reference_robot)
    # Each method occupies each position once; every invocation computes a fresh context.
    orders = [("full10", "noise2", "oracle2"), ("noise2", "oracle2", "full10"),
              ("oracle2", "full10", "noise2")]
    record["measurement_order"] = orders
    for repeat, order in enumerate(orders):
        for position, method in enumerate(order):
            print(f"Repeat {repeat + 1}/3, position {position + 1}/3: {method}...", flush=True)
            measurement, _, normalized, robot_actions, _ = measure(method)
            if method in ("full10", "oracle2"):
                np.testing.assert_array_equal(normalized, reference_normalized)
                np.testing.assert_array_equal(robot_actions, reference_robot)
            measurement.update(repeat=repeat, order_position=position,
                               normalized_max_abs_difference_from_full10=float(np.max(
                                   np.abs(normalized - reference_normalized))),
                               robot_action_max_abs_difference_from_full10=float(np.max(
                                   np.abs(robot_actions - reference_robot))))
            record["samples"].append(measurement)
            evidence[f"repeat_{repeat}_{method}_normalized"] = normalized
            evidence[f"repeat_{repeat}_{method}_robot_actions"] = robot_actions
            progress()
            print(f"{method}: {measurement['total_seconds']:.3f}s total; {measurement['nfe']} NFE; "
                  f"prepare {measurement['prepare_wall_seconds']:.3f}s, "
                  f"sampling {measurement['sample_wall_seconds']:.3f}s", flush=True)
    record["summary"] = {}
    for method in orders[0]:
        samples = [sample for sample in record["samples"] if sample["method"] == method]
        record["summary"][method] = {name: {
            "median": float(np.median([sample[name] for sample in samples])),
            "min": float(min(sample[name] for sample in samples)),
            "max": float(max(sample[name] for sample in samples)),
        } for name in ("total_seconds", "preprocess_seconds", "prepare_wall_seconds",
                       "sample_wall_seconds", "postprocess_seconds")}
    for name, expected in record["source_sha256"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"source changed during the profile: {name}")
    with arrays_path.open("xb") as handle:
        np.savez_compressed(handle, **evidence)
    record.update(status="passed", finished_utc=utc_now(), memory_at_completion=memory(),
                  evidence={"path": str(arrays_path.relative_to(root)), "sha256": sha256(arrays_path),
                            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                       for key, value in evidence.items()}},
                  own_continuation_exact_equal_to_full10=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    progress()
    print(json.dumps({"status": record["status"], "output": str(output),
                      "summary": record["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
