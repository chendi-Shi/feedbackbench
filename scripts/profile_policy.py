"""Measure a real pinned policy on a saved, real LIBERO observation.

This is a CPU feasibility measurement, not a closed-loop success benchmark.
The NPZ is produced by smoke_simulator.py and loaded with pickle disabled.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import time

from embodied_research.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("minerva", "smolvla"), required=True)
    parser.add_argument("--observation", type=Path, default=Path("artifacts/policy_smoke_observation.npz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--baseline-initialization", action="store_true",
                        help="Use the original random initialization before strict loading for parity comparisons")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if not args.load_only and not args.observation.is_file():
        parser.error(f"Real simulator observation is missing: {args.observation}; run smoke_simulator.py first")
    output = args.output or Path(f"artifacts/{args.policy}_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.dump_traceback_later(180, repeat=True)
    configure_runtime()
    print("Importing CPU runtime...", flush=True)
    import numpy as np
    import psutil
    import torch

    from embodied_research.policies import PolicyAdapter

    torch.set_num_threads(args.threads)
    process = psutil.Process()
    print(f"Strictly loading {args.policy}...", flush=True)
    started = time.perf_counter()
    adapter = PolicyAdapter(args.policy, num_steps=args.num_steps,
                            skip_random_initialization=not args.baseline_initialization)
    record = {
        "status": "loaded", "policy": adapter.metadata,
        "threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "total_load_seconds": time.perf_counter() - started,
        "rss_after_load_bytes": process.memory_info().rss,
        "measurement_scope": "single real LIBERO observation CPU feasibility; not rollout success",
    }

    def save():
        record["rss_bytes"] = process.memory_info().rss
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    print(json.dumps(record, indent=2), flush=True)
    if args.load_only:
        faulthandler.cancel_dump_traceback_later()
        return
    with np.load(args.observation, allow_pickle=False) as snapshot:
        observation = {key: snapshot[key].copy() for key in snapshot.files}
    observation["task"] = str(observation["task"].item())
    record["observation"] = {
        "source": str(args.observation), "task": observation["task"],
        "shapes": {key: list(value.shape) for key, value in observation.items() if key != "task"},
    }
    print("Warming up on real simulator images...", flush=True)
    adapter.reset(seed=123)
    adapter.observe(observation)
    chunk = adapter.predict_chunk()
    record["warmup_timing"] = dict(adapter.last_timing)
    record["chunk_shape"] = list(chunk.shape)
    record["chunk_first_action"] = chunk[0].tolist()
    save()
    # Same seed and same fresh temporal state must reproduce upstream behavior.
    adapter.reset(seed=123)
    upstream = adapter.select_action(observation)
    np.testing.assert_allclose(chunk[0], upstream, rtol=1e-5, atol=2e-6)
    record["first_action_upstream_parity_max_abs_error"] = float(np.max(np.abs(chunk[0] - upstream)))
    samples = []
    adapter.reset(seed=321)
    for index in range(args.repeats):
        started = time.perf_counter()
        adapter.observe(observation)
        actions = adapter.predict_chunk()
        samples.append({"wall_seconds": time.perf_counter() - started,
                        **adapter.last_timing,
                        "action_l2": float(np.linalg.norm(actions))})
        record["samples"] = samples
        save()
        print(f"Sample {index + 1}/{args.repeats}: {samples[-1]['wall_seconds']:.3f}s", flush=True)
    times = np.asarray([sample["wall_seconds"] for sample in samples])
    record.update(status="passed", wall_seconds_median=float(np.median(times)),
                  wall_seconds_p95=float(np.percentile(times, 95)))
    save()
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
