"""Causal, CPU-only scheduling of pretrained robot action chunks.

The adaptive rule is a candidate heuristic, not a learned policy or a claim of
novelty. Thresholds must be chosen on a separate development split and frozen
before test evaluation. It never receives reward, success, object pose, or a
future observation. All actions supplied here are already denormalized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SchedulerConfig:
    """Fixed and observation-change policies share the same execution path.

    ``ensemble_coeff`` follows ACT's ordering: the oldest still-valid
    prediction receives weight 1 and each newer prediction receives a factor
    ``exp(-ensemble_coeff)``. This is deliberately not recency weighting.
    """

    mode: str = "fixed"
    fixed_interval: int = 8
    min_interval: int = 2
    max_interval: int = 15
    vision_threshold: float = 0.04
    gripper_threshold: float = 0.02
    use_vision: bool = True
    use_gripper: bool = True
    temporal_ensemble: bool = True
    ensemble_coeff: float = 0.01
    horizon: int = 520
    thumbnail_size: int = 16
    front_image_key: str = "observation.images.image"
    state_key: str = "observation.state"

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "adaptive"}:
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        for name in ("fixed_interval", "min_interval", "max_interval", "horizon", "thumbnail_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_interval > self.max_interval:
            raise ValueError("min_interval must not exceed max_interval")
        for name in ("vision_threshold", "gripper_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if (isinstance(self.ensemble_coeff, bool) or not isinstance(self.ensemble_coeff, Real)
                or not np.isfinite(self.ensemble_coeff) or self.ensemble_coeff < 0):
            raise ValueError("ensemble_coeff must be finite and nonnegative")
        object.__setattr__(self, "ensemble_coeff", float(self.ensemble_coeff))
        for name in ("use_vision", "use_gripper", "temporal_ensemble"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("front_image_key", "state_key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")


@dataclass
class _Chunk:
    start_step: int
    actions: np.ndarray

    @property
    def end_step(self) -> int:
        return self.start_step + len(self.actions)


class ChunkScheduler:
    """Execute ``observe -> optional inference -> action`` once per simulator tick.

    Example::

        policy.observe(observation)  # update the pretrained policy's own history
        if scheduler.should_replan(observation):
            scheduler.submit_chunk(policy.predict_chunk())
        action = scheduler.next_action()

    ``next_action`` advances only the scheduler clock. Call ``reset`` along
    with the model and environment at every episode boundary. A failed model
    inference leaves a pending request; it cannot silently reuse an old action.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        if config is not None and not isinstance(config, SchedulerConfig):
            raise ValueError("config must be a SchedulerConfig")
        self.config = config or SchedulerConfig()
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._chunks: list[_Chunk] = []
        self._last_plan_step: int | None = None
        self._last_chunk_end: int | None = None
        self._reference_image: np.ndarray | None = None
        self._reference_gripper: np.ndarray | None = None
        self._prepared = False
        self._pending = False
        self._current_image: np.ndarray | None = None
        self._current_gripper: np.ndarray | None = None
        self._current_event: dict[str, Any] | None = None
        self._events: list[dict[str, Any]] = []
        self._requests = 0
        self._received = 0
        self._plan_steps: list[int] = []
        self._triggers: Counter[str] = Counter()

    @property
    def step(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._step >= self.config.horizon

    @property
    def needs_chunk(self) -> bool:
        return self._pending

    def _observation_features(self, observation: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping of flat observation keys")
        # Ablations really remove the corresponding signal, including reads of
        # its input. Fixed scheduling does not inspect either sensor.
        if self.config.mode == "fixed":
            return None, None
        thumbnail = None
        gripper = None
        if self.config.use_vision:
            if self.config.front_image_key not in observation:
                raise ValueError(f"missing front camera: {self.config.front_image_key}")
            frame = np.asarray(observation[self.config.front_image_key])
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[1:]) < 1:
                raise ValueError("front camera must have shape (3,H,W) or (1,3,H,W)")
            if not np.issubdtype(frame.dtype, np.floating):
                raise ValueError("front camera must be floating point RGB in [0,1]")
            if not np.all(np.isfinite(frame)) or np.min(frame) < 0 or np.max(frame) > 1:
                raise ValueError("front camera must contain finite RGB values in [0,1]")
            frame = frame.astype(np.float32, copy=False)
            height, width = frame.shape[1:]
            rows = np.linspace(0, height, min(self.config.thumbnail_size, height) + 1, dtype=int)
            cols = np.linspace(0, width, min(self.config.thumbnail_size, width) + 1, dtype=int)
            pooled = np.add.reduceat(np.add.reduceat(frame, rows[:-1], axis=1), cols[:-1], axis=2)
            counts = np.diff(rows)[:, None] * np.diff(cols)[None, :]
            thumbnail = (pooled / counts[None]).astype(np.float32)
            if self._reference_image is not None and thumbnail.shape != self._reference_image.shape:
                raise ValueError("front camera thumbnail shape changed within the episode")
        if self.config.use_gripper:
            if self.config.state_key not in observation:
                raise ValueError(f"missing robot state: {self.config.state_key}")
            state = np.asarray(observation[self.config.state_key], dtype=np.float32)
            if state.shape == (1, 8):
                state = state[0]
            if state.shape != (8,) or not np.all(np.isfinite(state)):
                raise ValueError("robot state must have shape (8,) or (1,8) and finite values")
            gripper = state[-2:].copy()
        return thumbnail, gripper

    def should_replan(self, observation: Mapping[str, Any]) -> bool:
        """Prepare one tick and decide without accessing any future information.

        Image drift is mean absolute RGB difference after area downsampling;
        gripper drift is maximum absolute joint-position difference. Both are
        measured from the observation at the latest successful inference.
        Initial inference and exhaustion override the adaptive minimum interval.
        """
        if self.done:
            raise StopIteration("scheduler horizon reached")
        if self._prepared:
            raise RuntimeError("next_action must finish the previous prepared tick")
        current_image, current_gripper = self._observation_features(observation)
        image_delta = None
        gripper_delta = None
        if current_image is not None and self._reference_image is not None:
            image_delta = float(np.mean(np.abs(current_image - self._reference_image)))
        if current_gripper is not None and self._reference_gripper is not None:
            gripper_delta = float(np.max(np.abs(current_gripper - self._reference_gripper)))
        age = None if self._last_plan_step is None else self._step - self._last_plan_step
        reasons: list[str] = []
        if age is None:
            reasons.append("initial")
        elif self._last_chunk_end is None or self._step >= self._last_chunk_end:
            reasons.append("chunk_exhausted")
        elif self.config.mode == "fixed":
            if age >= self.config.fixed_interval:
                reasons.append("fixed_interval")
        elif age >= self.config.min_interval:
            if age >= self.config.max_interval:
                reasons.append("max_interval")
            if image_delta is not None and image_delta >= self.config.vision_threshold:
                reasons.append("vision_change")
            if gripper_delta is not None and gripper_delta >= self.config.gripper_threshold:
                reasons.append("gripper_change")
        self._current_image = current_image
        self._current_gripper = current_gripper
        self._pending = bool(reasons)
        self._prepared = True
        self._current_event = {
            "step": self._step,
            "requested": self._pending,
            "reasons": reasons,
            "steps_since_plan": age,
            "image_delta": image_delta,
            "gripper_delta": gripper_delta,
            "chunk_received": False,
            "action_returned": False,
        }
        self._events.append(self._current_event)
        if self._pending:
            self._requests += 1
            self._triggers.update(reasons)
        return self._pending

    def submit_chunk(self, actions: np.ndarray) -> None:
        """Attach a model result at the current tick, never a future timestamp.

        Values are copied, checked for finiteness, and kept in physical action
        units. Clipping belongs to the environment's action interface. A fixed
        interval longer than a model's output is rejected rather than padded.
        """
        if not self._prepared or not self._pending:
            raise RuntimeError("submit_chunk requires a pending should_replan request")
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 1:
            raise ValueError("action chunk must have shape (T,7), with T >= 1")
        if not np.all(np.isfinite(values)):
            raise ValueError("action chunk must contain finite values")
        if self.config.mode == "fixed" and len(values) < self.config.fixed_interval:
            raise ValueError("fixed_interval exceeds the model's available action chunk length")
        self._chunks.append(_Chunk(self._step, values.copy()))
        self._last_plan_step = self._step
        self._last_chunk_end = self._step + len(values)
        self._reference_image = None if self._current_image is None else self._current_image.copy()
        self._reference_gripper = None if self._current_gripper is None else self._current_gripper.copy()
        self._pending = False
        self._received += 1
        self._plan_steps.append(self._step)
        assert self._current_event is not None
        self._current_event["chunk_received"] = True
        self._current_event["chunk_length"] = len(values)

    def next_action(self) -> np.ndarray:
        """Return the aligned action and advance exactly one control tick."""
        if self.done:
            raise StopIteration("scheduler horizon reached")
        if not self._prepared:
            raise RuntimeError("should_replan must prepare the current tick first")
        if self._pending:
            raise RuntimeError("a requested model chunk must arrive before taking an action")
        valid = [chunk for chunk in self._chunks if chunk.start_step <= self._step < chunk.end_step]
        if not valid:
            raise RuntimeError("no valid action prediction for the current tick")
        if self.config.temporal_ensemble and len(valid) > 1:
            aligned = np.stack([chunk.actions[self._step - chunk.start_step] for chunk in valid])
            with np.errstate(over="ignore"):
                log_weights = -self.config.ensemble_coeff * np.arange(len(valid), dtype=np.float64)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            action = np.sum(aligned * weights[:, None], axis=0).astype(np.float32)
        else:
            latest = valid[-1]
            action = latest.actions[self._step - latest.start_step].copy()
        assert self._current_event is not None
        self._current_event["action_returned"] = True
        self._current_event["active_predictions"] = len(valid)
        self._current_event["used_predictions"] = len(valid) if self.config.temporal_ensemble else 1
        self._step += 1
        self._chunks = [chunk for chunk in self._chunks if chunk.end_step > self._step]
        self._prepared = False
        self._current_image = None
        self._current_gripper = None
        self._current_event = None
        return action

    def diagnostics(self, *, include_events: bool = True) -> dict[str, Any]:
        """Return a detached, JSON-serializable audit record, including failures."""
        plan_steps = self._plan_steps.copy()
        completed_spans = [b - a for a, b in zip(plan_steps, plan_steps[1:])]
        executed_spans = completed_spans.copy()
        if plan_steps and self._step > plan_steps[-1]:
            executed_spans.append(self._step - plan_steps[-1])
        result: dict[str, Any] = {
            "config": asdict(self.config),
            "steps": self._step,
            "horizon_reached": self.done,
            "inference_requests": self._requests,
            "chunks_received": self._received,
            "pending_inference": self._pending,
            "inference_calls_per_100_steps": 100.0 * self._received / self._step if self._step else None,
            "trigger_counts": dict(self._triggers),
            "plan_steps": plan_steps,
            "completed_replan_intervals": completed_spans,
            "executed_chunk_spans": executed_spans,
        }
        if include_events:
            result["events"] = [{**event, "reasons": event["reasons"].copy()} for event in self._events]
        return result
