"""Exercise real rendering/dynamics and compare preprocessing with upstream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from embodied_research.runtime import compatibility_audit
from embodied_research.simulator import LiberoSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/simulator-smoke"))
    parser.add_argument("--policy-observation", type=Path, default=Path("artifacts/policy_smoke_observation.npz"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Constructing official LIBERO environment...", flush=True)
    with LiberoSimulator(args.suite, args.task_id, image_size=args.image_size) as env:
        print(f"Environment constructed in {time.perf_counter() - started:.1f}s; resetting saved state...", flush=True)
        observation = env.reset(seed=1000, init_state_id=0)
        args.policy_observation.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.policy_observation, **{key: np.asarray(value) for key, value in observation.items()})
        print(f"Real policy observation saved: {args.policy_observation}", flush=True)
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor.env_processor import LiberoProcessorStep
        from PIL import Image

        def batch(value):
            if isinstance(value, dict):
                return {key: batch(item) for key, item in value.items()}
            return np.expand_dims(value, 0)

        reference = LiberoProcessorStep()._process_observation(preprocess_observation(batch(env.raw_observation)))
        differences = {}
        for name in ("observation.images.image", "observation.images.image2", "observation.state"):
            expected = reference[name][0].cpu().numpy()
            actual = observation[name]
            differences[name] = float(np.max(np.abs(expected - actual)))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        Image.fromarray(env.frame()).save(args.output / "front.png")
        Image.fromarray(env.frame("image2")).save(args.output / "wrist.png")
        steps = []
        for _ in range(args.steps):
            _, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            steps.append({"reward": reward, "terminated": terminated, "truncated": truncated,
                          "is_success": bool(info["is_success"])})
            if terminated or truncated:
                break
        result = {"status": "passed", "simulator": env.metadata(), "preprocessing_max_abs_difference": differences,
                  "steps": steps, "elapsed_seconds": time.perf_counter() - started,
                  "compatibility": compatibility_audit()}
    (args.output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
