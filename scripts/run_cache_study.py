"""Frozen 28-episode CPU ActionCache transfer pilot; no training or test admission.

Four full-horizon development episodes precede a frozen cache. Held-out execution
uses eight paired full10/cache2 cases and four paired noise2/cache0 ablations.
The common init11 subset is the only four-way comparison. Errors and partial
attempts are retained and block automatic retries. Source changes block resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from embodied_research.experiment import (
    artifact_provenance, digest_json, episode_id, run_episode, utc_now, versions, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SOURCES = (
    "scripts/run_cache_study.py", "src/embodied_research/cache_policy.py",
    "src/embodied_research/cache_sampler.py", "src/embodied_research/action_cache.py",
    "src/embodied_research/experiment.py", "src/embodied_research/policies.py",
    "src/embodied_research/runtime.py", "src/embodied_research/simulator.py",
    "src/embodied_research/scheduling.py",
)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cases_for_study():
    cases = []
    for split, initial_states in (("development", (1,)), ("heldout", (11, 29))):
        for task_index, suite in enumerate(SUITES):
            for initial_index, initial in enumerate(initial_states):
                methods = (["full10"] if split == "development" else
                           ["full10", "cache2", "noise2", "cache0"] if initial == 11 else
                           ["full10", "cache2"])
                offset = (task_index + initial_index) % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    case = {
                        "model": "smolvla", "suite": suite, "task_id": 1,
                        "init_state_id": initial, "seed": 1000, "condition": "clean",
                        "method": method, "num_steps": 2 if method == "noise2" else 10,
                        "scheduler": {"mode": "fixed", "fixed_interval": 10,
                                      "temporal_ensemble": False},
                        "record_video": initial == 11, "split": split,
                    }
                    case["id"] = episode_id(case)
                    cases.append(case)
    return cases


def study_spec():
    return {
        "name": "SmolVLA frozen offline ActionCache transfer pilot v1",
        "question": "Can successful development latents reduce flow NFE on held-out initial states without losing task success?",
        "primary": {"methods": ["full10", "cache2"], "task_clusters": 4, "pairs": 8,
                    "initial_states": [11, 29], "metrics": ["success", "nfe_per_replan", "seconds_per_replan"]},
        "ablations": {"methods": ["full10", "noise2", "cache2", "cache0"], "pairs": 4,
                      "initial_states": [11], "scope": "exploratory; do not compare unmatched denominators"},
        "development": {"initial_states": [1], "episodes": 4,
                        "admission": "all replan samples from successful full10 development episodes only"},
        "cache": {"projection_dim": 500, "density": 0.01, "projection_seed": 20261014,
                  "capacity": 3000, "cosine_threshold": 0.95, "threshold_selection": "fixed before any pilot rollout; not tuned",
                  "cache2_native_time": 0.2, "cache2_paper_tau": 0.8, "cache0_native_time": 0.0,
                  "on_miss": "full 10-step inference", "test_updates": False,
                  "retrieval": "global exact top1, no task-ID routing; immutable cache"},
        "execution": {"threads": 2, "image_size": 360, "warmup_calls": 2, "replan_interval": 10,
                      "temporal_ensemble": False, "simulation_control_hz": 20,
                      "noise_pairing": "one same-shaped draw on EVERY decision, including cache hits, after per-episode seed reset",
                      "termination": "official success or original full horizon; no wall-clock early exit"},
        "limitations": ["CPU pilot; four known pretrained LIBERO tasks, not task generalization or full benchmark",
                        "independent migration inspired by ActionCache 2607.06370, not official code reproduction",
                        "frozen offline cache differs from paper online successful-rollout LRU",
                        "no GPU, paid cloud, model training, real robot, or real-time asynchronous execution",
                        "small paired sample; failure-induced shorter rollouts are not speedups"],
        "cases": cases_for_study(),
    }


def verify_complete(directory, case, fingerprint):
    """Fail closed on partial attempts, wrong identity, or changed evidence."""
    directory = Path(directory)
    if not directory.exists():
        return None
    result_path, receipt_path = directory / "result.json", directory / "receipt.json"
    if not result_path.exists() or not receipt_path.exists():
        raise RuntimeError(f"Partial attempt retained; automatic retry forbidden: {case['id']}")
    result, receipt = read_json(result_path), read_json(receipt_path)
    if result.get("status") != "complete" or result.get("protocol_fingerprint") != fingerprint:
        raise ValueError(f"Failed or incompatible record: {case['id']}")
    if any(result.get(key) != value for key, value in case.items()):
        raise ValueError(f"Case identity/configuration changed: {case['id']}")
    actual_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != "receipt.json"}
    if actual_files != set(receipt["files"]):
        raise ValueError(f"Episode artifact inventory changed: {case['id']}")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or file_sha(directory / name) != expected:
            raise ValueError(f"Episode artifact hash mismatch: {case['id']}/{name}")
    return result


def seal_episode(directory):
    write_json(Path(directory) / "receipt.json", {
        "files": {p.name: file_sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file()},
        "sealed_at": utc_now(),
    })


def save_samples(directory, policy):
    samples = policy.collected_samples
    if not samples:
        raise RuntimeError("Development episode has no collected inference samples")
    contract = policy.feature_contract
    if any(sample.feature_contract != contract for sample in samples):
        raise RuntimeError("Feature layout changed inside development episode")
    np.savez_compressed(Path(directory) / "cache_samples.npz",
                        key_features=np.stack([s.key_features for s in samples]),
                        intermediate_latents=np.stack([s.intermediate_latent for s in samples]),
                        final_normalized=np.stack([s.final_normalized for s in samples]))
    write_json(Path(directory) / "cache_samples.json", {
        "count": len(samples), "feature_contract": contract,
        "latent_contract": samples[0].latent_contract,
        "scope": "candidate samples only; failed development episodes are never admitted",
    })


def freeze_cache(output, protocol):
    from embodied_research.action_cache import (
        CacheProvenance, FrozenActionCache, ProjectionConfig, SparseTernaryProjection,
    )
    from embodied_research.cache_sampler import latent_time_contract
    destination = output / "frozen_cache"
    binding_path = destination / "binding.json"
    development = [c for c in protocol["spec"]["cases"] if c["split"] == "development"]
    records = []
    for case in development:
        directory = output / "episodes" / case["id"]
        result = verify_complete(directory, case, protocol["fingerprint"])
        if result is None:
            raise RuntimeError("All declared development cases must finish before cache freeze")
        records.append((case, directory, result))
    evidence = {case["id"]: file_sha(directory / "receipt.json") for case, directory, _ in records}
    if binding_path.exists():
        binding = read_json(binding_path)
        if binding["protocol_fingerprint"] != protocol["fingerprint"] or binding["development_receipts"] != evidence:
            raise ValueError("Frozen cache no longer matches development/protocol")
        for name in ("cache2", "cache0"):
            cache = FrozenActionCache.load(destination / name,
                                           expected_provenance_sha256=binding[name]["provenance_sha256"])
            if cache.cache_sha256 != binding[name]["cache_sha256"]:
                raise ValueError("Frozen cache content changed")
        return binding
    if destination.exists():
        raise RuntimeError("Partial cache freeze retained; refusing to overwrite")
    contract = read_json(records[0][1] / "cache_samples.json")["feature_contract"]
    configuration = protocol["spec"]["cache"]
    projection = SparseTernaryProjection(ProjectionConfig(
        contract["input_dim"], configuration["projection_dim"], configuration["density"], configuration["projection_seed"]))
    model = protocol["artifacts"]["models"]["smolvla"]
    common = dict(model_revision=model["revision"], model_sha256=model["files"]["model.safetensors"],
                  feature_contract=contract, development_case_ids=tuple(c["id"] for c in development))
    caches = {
        "cache2": FrozenActionCache(projection, (50, 32), CacheProvenance(**common, flow_time=0.2),
                                    configuration["cosine_threshold"], configuration["capacity"]),
        "cache0": FrozenActionCache(projection, (50, 7), CacheProvenance(**common, flow_time=0.0),
                                    configuration["cosine_threshold"], configuration["capacity"]),
    }
    admitted = []
    for case, directory, result in records:
        metadata = read_json(directory / "cache_samples.json")
        if metadata["feature_contract"] != contract:
            raise ValueError("Development feature layouts differ")
        if metadata["latent_contract"] != latent_time_contract() or metadata["count"] != result["inference_calls"]:
            raise ValueError("Development sample time/count contract differs")
        if not result["success"]:
            continue
        admitted.append(case["id"])
        with np.load(directory / "cache_samples.npz", allow_pickle=False) as samples:
            if any(len(samples[name]) != result["inference_calls"] for name in
                   ("key_features", "intermediate_latents", "final_normalized")):
                raise ValueError("Development cache samples/call count disagree")
            for index, feature in enumerate(samples["key_features"]):
                caches["cache2"].add(feature, samples["intermediate_latents"][index], case_id=case["id"])
                caches["cache0"].add(feature, samples["final_normalized"][index], case_id=case["id"])
    binding = {"protocol_fingerprint": protocol["fingerprint"], "development_receipts": evidence,
               "admitted_successful_case_ids": admitted, "frozen_at": utc_now(),
               "empty_cache_is_valid": True, "test_admission": False}
    for name, cache in caches.items():
        cache.freeze().save(destination / name)
        binding[name] = {"cache_sha256": cache.cache_sha256,
                         "provenance_sha256": cache.provenance_sha256, "entries": len(cache)}
    write_json(binding_path, binding)
    return binding


def verify_cache_binding(output, protocol, binding):
    expected = digest_json(binding)
    for case in protocol["spec"]["cases"]:
        if case["split"] != "heldout":
            continue
        result = verify_complete(output / "episodes" / case["id"], case, protocol["fingerprint"])
        if result is not None and result.get("cache_binding_sha256") != expected:
            raise ValueError("Held-out result was produced with a different frozen cache binding")


def run(output, stage="all", limit=None):
    from embodied_research.runtime import compatibility_audit, configure_runtime
    configure_runtime(ROOT)
    import torch
    from embodied_research.action_cache import FrozenActionCache
    from embodied_research.cache_policy import CachePolicyAdapter
    from embodied_research.policies import PolicyAdapter
    from embodied_research.simulator import LiberoSimulator

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}))
    simulator = adapter = policy = None
    simulator_key = None
    try:
        provenance = {"spec": study_spec(), "sources": {p: file_sha(ROOT / p) for p in SOURCES},
                      "runtime_versions": versions(), "artifacts": artifact_provenance(ROOT, ["smolvla"])}
        protocol = {**provenance, "fingerprint": digest_json(provenance)}
        path = output / "protocol.json"
        if path.exists():
            if read_json(path)["fingerprint"] != protocol["fingerprint"]:
                raise ValueError("Study code/config/dependencies changed; refusing resume")
        else:
            write_json(path, {**protocol, "created_at": utc_now()})
        # Validate all completed evidence and reject prior partial attempts before loading weights.
        for case in protocol["spec"]["cases"]:
            verify_complete(output / "episodes" / case["id"], case, protocol["fingerprint"])
        caches, binding, newly_completed = {}, None, 0
        warmed = False
        for ordinal, case in enumerate(protocol["spec"]["cases"], start=1):
            if stage == "prefill" and case["split"] != "development":
                break
            if stage == "heldout" and case["split"] == "development":
                continue
            if case["split"] == "heldout" and binding is None:
                binding = freeze_cache(output, protocol)
                verify_cache_binding(output, protocol, binding)
                caches = {name: FrozenActionCache.load(output / "frozen_cache" / name,
                          expected_provenance_sha256=binding[name]["provenance_sha256"])
                          for name in ("cache2", "cache0")}
            directory = output / "episodes" / case["id"]
            if verify_complete(directory, case, protocol["fingerprint"]) is not None:
                continue
            if limit is not None and newly_completed >= limit:
                break
            directory.mkdir(parents=True)
            write_json(directory / "attempt.json", {"case": case, "protocol_fingerprint": protocol["fingerprint"],
                                                       "started_at": utc_now()})
            write_json(output / "progress.json", {"current": case, "ordinal": ordinal, "total": 28,
                                                   "updated_at": utc_now()})
            print(f"[{ordinal}/28] {case['id']} ({case['split']})", flush=True)
            try:
                if adapter is None:
                    adapter = PolicyAdapter("smolvla", num_steps=10, temporal_ensemble_coeff=None)
                key = (case["suite"], case["task_id"])
                if key != simulator_key:
                    if simulator is not None:
                        simulator.close()
                    simulator = LiberoSimulator(*key, project_root=ROOT, image_size=360)
                    simulator_key = key
                if not warmed:
                    warm = simulator.reset(seed=1000, init_state_id=1)
                    adapter.reset(seed=1000)
                    for _ in range(2):
                        adapter.observe(warm)
                        adapter.predict_chunk()
                    warmed = True
                policy = CachePolicyAdapter(adapter, case["method"], cache=caches.get(case["method"]),
                                            collect=case["split"] == "development")
                result = run_episode(simulator, policy, case, directory)
                result["protocol_fingerprint"] = protocol["fingerprint"]
                result["cache_binding_sha256"] = None if binding is None else digest_json(binding)
                write_json(directory / "calls.json", policy.call_log)
                if case["split"] == "development":
                    save_samples(directory, policy)
                write_json(directory / "result.json", result)
                seal_episode(directory)
                newly_completed += 1
                print(f"  success={result['success']} steps={result['steps']} calls={result['inference_calls']} wall={result['wall_seconds']:.1f}s", flush=True)
                policy.reset(seed=1000)  # Release candidate feature arrays before the next episode.
            except Exception as error:
                write_json(directory / "result.json", {**case, "status": "error", "error": str(error),
                           "traceback": traceback.format_exc(), "protocol_fingerprint": protocol["fingerprint"],
                           "finished_at": utc_now()})
                raise
        completed = [c["id"] for c in protocol["spec"]["cases"]
                     if verify_complete(output / "episodes" / c["id"], c, protocol["fingerprint"]) is not None]
        if len(completed) >= 4:
            verify_cache_binding(output, protocol, freeze_cache(output, protocol))
        write_json(output / "progress.json", {"completed_ids": completed, "completed": len(completed),
                   "total": 28, "finished": len(completed) == 28, "updated_at": utc_now()})
        write_json(output / "compatibility.json", compatibility_audit())
        return {"completed": len(completed), "total": 28}
    finally:
        if simulator is not None:
            simulator.close()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/frontier/cache_study")
    parser.add_argument("--stage", choices=("all", "prefill", "heldout"), default="all")
    parser.add_argument("--limit", type=int, help="Operational checkpoint after this many NEW complete episodes; denominator remains 28")
    parser.add_argument("--show-design", action="store_true")
    args = parser.parse_args()
    if args.show_design:
        print(json.dumps(study_spec(), indent=2))
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        print(json.dumps(run(args.output, args.stage, args.limit), indent=2))


if __name__ == "__main__":
    main()
