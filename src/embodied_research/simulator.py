"""Serial LIBERO adapter using the pinned MINERVA environment implementation.

Only two RGB views and robot proprioception reach the policy.  Success uses the
official task predicate.  There is no scripted expert, object pose, or fallback
action generator in this adapter.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import MUJOCO_VERSION, configure_libero_paths, configure_runtime


SUITE_HORIZONS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Match LiberoProcessorStep's float32 xyzw quaternion conversion."""
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected one finite xyzw quaternion with shape (4,)")
    w = np.clip(quat[3], np.float32(-1.0), np.float32(1.0))
    denominator = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - w * w))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] / denominator * (np.float32(2.0) * np.arccos(w)), dtype=np.float32)


def policy_observation(raw: dict[str, Any], task: str) -> dict[str, Any]:
    """Flatten upstream env output; the policy must not flip these images again.

    Mirrors preprocess_observation + LiberoProcessorStep: uint8 HWC -> float32
    CHW [0,1], rotate 180 degrees, concatenate eef position, axis angle and fingers.
    Object state is deliberately never read.
    """
    output: dict[str, Any] = {"task": task}
    for camera in ("image", "image2"):
        image = np.asarray(raw["pixels"][camera])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 HWC for {camera}, got {image.shape}/{image.dtype}")
        output[f"observation.images.{camera}"] = np.ascontiguousarray(
            image[::-1, ::-1].transpose(2, 0, 1), dtype=np.float32
        ) / np.float32(255.0)
    state = raw["robot_state"]
    position = np.asarray(state["eef"]["pos"], dtype=np.float32)
    gripper = np.asarray(state["gripper"]["qpos"], dtype=np.float32)
    if position.shape != (3,) or gripper.shape != (2,):
        raise ValueError("Expected eef position (3,) and gripper qpos (2,)")
    vector = np.concatenate((position, quaternion_to_axis_angle(state["eef"]["quat"]), gripper))
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite robot proprioception")
    output["observation.state"] = vector
    return output


class LiberoSimulator:
    """One explicit task/environment, one main-thread renderer, no autoreset."""

    def __init__(
        self,
        suite: str,
        task_id: int,
        *,
        project_root: str | Path | None = None,
        assets_path: str | Path | None = None,
        image_size: int = 360,
        _env_factory=None,
        _task_suite=None,
    ):
        if suite not in SUITE_HORIZONS:
            raise ValueError(f"Unsupported suite {suite!r}; choose {tuple(SUITE_HORIZONS)}")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or not 0 <= task_id < 10:
            raise ValueError("Standard LIBERO task_id must be in [0, 9]")
        if not isinstance(image_size, int) or image_size < 32:
            raise ValueError("image_size must be an integer >=32")
        root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        if _env_factory is None:
            configure_runtime(root)
            version = importlib.metadata.version("mujoco")
            if version != MUJOCO_VERSION:
                raise RuntimeError(f"Renderer mismatch: expected MuJoCo {MUJOCO_VERSION}, found {version}")
            configure_libero_paths(root, assets_path)
            from libero.libero import benchmark
            from lerobot.envs.libero import LiberoEnv

            _env_factory = LiberoEnv
            _task_suite = benchmark.get_benchmark_dict()[suite]()
        self.suite = suite
        self.task_id = task_id
        self.horizon = SUITE_HORIZONS[suite]
        self.image_size = image_size
        self.steps = 0
        self._done = True
        self._last_raw: dict[str, Any] | None = None
        self.env = _env_factory(
            task_suite=_task_suite,
            task_id=task_id,
            task_suite_name=suite,
            episode_length=self.horizon,
            obs_type="pixels_agent_pos",
            observation_width=image_size,
            observation_height=image_size,
            init_states=True,
            episode_index=0,
            n_envs=1,
            num_steps_wait=10,
            control_freq=20,
            control_mode="relative",
            hard_reset=True,
        )
        self.task = self.env.task_description

    def reset(self, *, seed: int = 1000, init_state_id: int = 0) -> dict[str, Any]:
        if not isinstance(init_state_id, int) or isinstance(init_state_id, bool) or init_state_id < 0:
            raise ValueError("init_state_id must be a nonnegative integer")
        count = len(self.env._init_states)
        if init_state_id >= count:
            raise ValueError(f"init_state_id {init_state_id} exceeds {count} published initial states")
        # Override upstream increment explicitly; paired conditions always start
        # from the requested saved state instead of silently rotating to another.
        self.env.init_state_id = init_state_id
        raw, info = self.env.reset(seed=int(seed))
        self._last_raw = raw
        self.steps = 0
        self._done = False
        self.current_seed = int(seed)
        self.current_init_state_id = init_state_id
        return policy_observation(raw, self.task)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Reset before stepping a new or completed episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected finite action with shape (7,)")
        # No extra clipping or gripper transformation: upstream checkpoint output
        # and the OSC controller retain the official evaluation action convention.
        raw, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        truncated = bool(truncated or (self.steps >= self.horizon and not terminated))
        terminated = bool(terminated)
        self._done = terminated or truncated
        self._last_raw = raw
        info = dict(info)
        info.update({"steps": self.steps, "initial_state_id": self.current_init_state_id})
        return policy_observation(raw, self.task), float(reward), terminated, truncated, info

    @property
    def raw_observation(self) -> dict[str, Any]:
        """Upstream robot/camera observation retained for preprocessing parity QA."""
        if self._last_raw is None:
            raise RuntimeError("Reset before reading an observation")
        return self._last_raw

    def frame(self, camera: str = "image") -> np.ndarray:
        if camera not in ("image", "image2"):
            raise ValueError("camera must be image or image2")
        return np.ascontiguousarray(self.raw_observation["pixels"][camera][::-1, ::-1])

    def metadata(self) -> dict[str, Any]:
        return {
            "suite": self.suite, "task_id": self.task_id, "task": self.task,
            "horizon": self.horizon, "control_hz": 20,
            "control_mode": "relative", "hard_reset": True, "settle_steps": 10,
            "image_size": self.image_size, "images_flipped_hw_once": True,
            "policy_state": "eef_xyz + eef_axis_angle + gripper_qpos (8 values)",
            "object_state_exposed_to_policy": False,
        }

    def close(self):
        self.env.close()
        self._done = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
