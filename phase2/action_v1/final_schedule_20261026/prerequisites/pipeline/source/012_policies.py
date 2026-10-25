"""Pinned LeRobot policy adapters for CPU LIBERO experiments.

The simulator supplies camera frames already flipped exactly once, as float32 CHW
in [0, 1]. The checkpoint's original pre/post processors handle all remaining
normalization, text tokenization and action denormalization. No object state,
success predicates, segmentation labels, or task-specific controllers enter here.

Two deliberately separate modes are supported, with reset() required to switch:
* select_action(obs): exact upstream queue/temporal-ensemble baseline semantics.
* observe(obs) every simulator tick, then predict_chunk() only at replans. This
  preserves MINERVA's consecutive two-frame history even during open-loop actions.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISIONS = {
    "minerva": "1b4fb1743f00a7d8eb87c7059c446447907d12bf",
    "smolvla": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    "smolvlm_assets": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
}
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def verify_model_files(directory: Path, name: str) -> dict:
    """Refuse incomplete downloads and byte changes before loading a checkpoint."""
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != MODEL_REVISIONS[name] or manifest["name"] != name:
        raise ValueError(f"Unexpected checkpoint provenance in {directory}")
    records = {record["file"]: record for record in manifest["files"]}
    required = {"config.json"}
    if name != "smolvlm_assets":
        required |= {"model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(records):
        raise ValueError(f"Incomplete model manifest: {required - records.keys()}")
    for filename, record in records.items():
        if Path(filename).name != filename:
            raise ValueError("Manifest artifact must be a simple filename")
        path = directory / filename
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Model artifact size changed: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Model artifact hash changed: {path}")
    return manifest


class PolicyAdapter:
    """Batch-one pretrained policy. Each instance owns all temporal model state."""

    def __init__(
        self,
        name: str,
        models_dir: str | Path | None = None,
        *,
        num_steps: int = 10,
        n_action_steps: int = 1,
        temporal_ensemble_coeff: float | None = 0.01,
        skip_random_initialization: bool = True,
        verify: bool = True,
    ):
        if name not in ("minerva", "smolvla"):
            raise ValueError(f"Unknown policy: {name}")
        if num_steps < 1 or n_action_steps < 1:
            raise ValueError("Integration and execution step counts must be positive")
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        self.name = name
        self.models_dir = Path(models_dir or PROJECT_ROOT / "models").resolve()
        self.directory = self.models_dir / name
        self.manifest = verify_model_files(self.directory, name) if verify else None
        if name == "minerva":
            from lerobot.policies.tinyflow.modeling_tinyflow import TinyflowPolicy

            policy_class = TinyflowPolicy
        else:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            policy_class = SmolVLAPolicy
        config = PreTrainedConfig.from_pretrained(
            self.directory, local_files_only=True, cli_overrides=["--device=cpu"]
        )
        config.device = "cpu"
        config.use_amp = False
        config.n_action_steps = n_action_steps
        if name == "minerva":
            if temporal_ensemble_coeff is not None and n_action_steps != 1:
                raise ValueError("MINERVA temporal ensembling requires n_action_steps=1")
            config.temporal_ensemble_coeff = temporal_ensemble_coeff
            config._validate()
        else:
            assets = self.models_dir / "smolvlm_assets"
            if verify:
                verify_model_files(assets, "smolvlm_assets")
            config.vlm_model_name = str(assets)
            # The policy safetensors contains the complete VLM. Building its
            # architecture from the pinned config avoids downloading a second,
            # unrelated 2 GB base checkpoint. strict=True proves full coverage.
            config.load_vlm_weights = False
            config.num_steps = num_steps
            config.compile_model = False
        start = time.perf_counter()
        self.skip_random_initialization = name == "smolvla" and skip_random_initialization
        initialization = nullcontext()
        if self.skip_random_initialization:
            from transformers.initialization import no_init_weights

            # The complete checkpoint overwrites every parameter under strict
            # loading. Avoid minutes of discarded RNG work on the CPU. In the
            # pinned Transformers version, the only nonpersistent model buffers
            # are Llama rotary frequencies, computed directly in the constructor
            # with arange/powers (unaffected by this official loading context).
            initialization = no_init_weights()
        with initialization:
            self.policy = policy_class.from_pretrained(
                self.directory, config=config, local_files_only=True, strict=True
            )
        self.policy = self.policy.to(device="cpu", dtype=torch.float32).eval()
        self.policy.requires_grad_(False)
        self.loading_audit = self._audit_smolvla_loading() if name == "smolvla" else None
        overrides = {"device_processor": {"device": "cpu"},
                     "normalizer_processor": {"device": "cpu"}}
        if name == "smolvla":
            overrides["tokenizer_processor"] = {"tokenizer_name": config.vlm_model_name}
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, str(self.directory), preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"},
                                     "unnormalizer_processor": {"device": "cpu"}},
        )
        self.load_seconds = time.perf_counter() - start
        self.config = config
        self.reset()

    @property
    def chunk_length(self) -> int:
        return (self.config.horizon - self.config.n_obs_steps + 1
                if self.name == "minerva" else self.config.chunk_size)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name, "revision": MODEL_REVISIONS[self.name],
            "model_sha256": None if self.manifest is None else next(
                item["sha256"] for item in self.manifest["files"] if item["file"] == "model.safetensors"
            ),
            "device": "cpu", "dtype": "float32",
            "parameters": sum(parameter.numel() for parameter in self.policy.parameters()),
            "n_observation_steps": self.config.n_obs_steps, "chunk_length": self.chunk_length,
            "n_action_steps": self.config.n_action_steps,
            "num_steps": self.config.num_steps if self.name == "smolvla" else None,
            "temporal_ensemble_coeff": getattr(self.config, "temporal_ensemble_coeff", None),
            "temporal_ensemble_applies_to": "upstream select_action only",
            "scheduled_chunks_use_temporal_ensemble": False,
            "conditioning": "language" if self.name == "smolvla" else "closed_set_task_id",
            "load_seconds": self.load_seconds,
            "strict_checkpoint_load": True,
            "skipped_discarded_random_initialization": self.skip_random_initialization,
            "loading_audit": self.loading_audit,
            "camera_flip": "performed once by simulator, never by this adapter",
        }

    def _audit_smolvla_loading(self) -> dict[str, Any]:
        """Check full state coverage and the buffers omitted by safetensors.

        Strict loading itself checks parameter coverage. This independent header
        audit makes that coverage reviewable and verifies that skipping random
        initialization leaves every nonpersistent model buffer well defined.
        """
        import torch
        from safetensors import safe_open
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        state = self.policy.state_dict()
        with safe_open(str(self.directory / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
            if set(checkpoint.keys()) != set(state):
                raise RuntimeError("SmolVLA model/checkpoint state keys differ despite strict loading")
            for name, tensor in state.items():
                if tuple(checkpoint.get_slice(name).get_shape()) != tuple(tensor.shape):
                    raise RuntimeError(f"SmolVLA checkpoint shape mismatch: {name}")
        buffers = {}
        for module_name, module in self.policy.named_modules():
            for name in module._non_persistent_buffers_set:
                value = module._buffers[name]
                if value is None:
                    continue
                if not isinstance(module, LlamaRotaryEmbedding) or name not in {"inv_freq", "original_inv_freq"}:
                    raise RuntimeError(f"Unaudited nonpersistent SmolVLA buffer: {module_name}.{name}")
                config = module.config
                if config.rope_parameters["rope_type"] != "default":
                    raise RuntimeError("Only the pinned checkpoint's default rotary frequencies are audited")
                dimension = config.head_dim or config.hidden_size // config.num_attention_heads
                # Independent closed-form reconstruction; no initialization hook.
                expected = 1.0 / (config.rope_parameters["rope_theta"] ** (
                    torch.arange(0, dimension, 2, dtype=torch.int64).float() / dimension))
                torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
                buffers[f"{module_name}.{name}"] = {
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
                    "matches_closed_form": True,
                }
        if len(buffers) != 4:
            raise RuntimeError(f"Expected two rotary buffers for VLM and expert, found {len(buffers)}")
        return {"checkpoint_state_tensors": len(state), "exact_state_keys_and_shapes": True,
                "nonpersistent_buffers": buffers}

    def reset(self, seed: int | None = None) -> None:
        """Clear queues/ensembler/processor state between complete episodes."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        self.policy.reset()
        self.preprocess.reset()
        self.postprocess.reset()
        self._mode: str | None = None
        self._batch = None
        self.observation_count = 0
        self.inference_count = 0
        self.last_timing = {"preprocess_seconds": 0.0, "inference_seconds": 0.0,
                            "postprocess_seconds": 0.0}

    def _set_mode(self, mode: str) -> None:
        if self._mode is not None and self._mode != mode:
            raise RuntimeError("Call reset() before switching upstream and scheduled control modes")
        self._mode = mode

    def _prepare_observation(self, observation: dict) -> dict:
        import torch

        start = time.perf_counter()
        if not isinstance(observation.get("task"), str) or not observation["task"].strip():
            raise ValueError("A nonempty official LIBERO task string is required")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 8D robot proprioceptive state")
        batch = {"task": observation["task"], "observation.state": torch.from_numpy(state.copy())}
        for key in IMAGE_KEYS:
            pixels = np.asarray(observation[key])
            if pixels.ndim != 3 or pixels.shape[0] != 3 or not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError(f"{key} must be floating CHW RGB")
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise ValueError(f"{key} must contain finite pixels in [0, 1]")
            batch[key] = torch.from_numpy(np.array(pixels, dtype=np.float32, order="C", copy=True))
        result = self.preprocess(batch)
        self.last_timing["preprocess_seconds"] = time.perf_counter() - start
        return result

    def _actions_to_numpy(self, actions) -> np.ndarray:
        start = time.perf_counter()
        actions = self.postprocess(actions).detach().cpu().numpy().astype(np.float32, copy=True)
        self.last_timing["postprocess_seconds"] = time.perf_counter() - start
        if not np.isfinite(actions).all() or actions.shape[-1] != 7:
            raise RuntimeError("Policy produced invalid robot actions")
        # Do not clip here: the original simulator/controller applies its own
        # action limits. Preserve raw model outputs for parity and analysis.
        return actions

    def observe(self, observation: dict) -> None:
        """Record the current tick even if the external scheduler does not replan."""
        from lerobot.policies.utils import populate_queues

        self._set_mode("scheduled")
        batch = self._prepare_observation(observation)
        if self.name == "minerva":
            batch = self.policy._prepare_batch(batch)
            self.policy._queues = populate_queues(self.policy._queues, batch, exclude_keys=["action"])
        self._batch = batch
        self.observation_count += 1

    def predict_chunk(self) -> np.ndarray:
        """Generate future actions in simulator units without advancing history."""
        import torch

        self._set_mode("scheduled")
        if self._batch is None:
            raise RuntimeError("observe() is required before predict_chunk()")
        start = time.perf_counter()
        with torch.inference_mode():
            if self.name == "minerva":
                # Upstream _plan_chunk uses the queues populated at every tick,
                # then removes the past action (n_obs_steps - 1) from the chunk.
                actions = self.policy._plan_chunk(self._batch)
            else:
                actions = self.policy.predict_action_chunk(dict(self._batch))
        self.last_timing["inference_seconds"] = time.perf_counter() - start
        self.inference_count += 1
        return self._actions_to_numpy(actions)[0]

    def select_action(self, observation: dict) -> np.ndarray:
        """Run upstream select_action, including the original temporal ensemble."""
        import torch

        self._set_mode("upstream")
        batch = self._prepare_observation(observation)
        will_infer = not self.policy._queues["action"] or (
            self.name == "minerva" and self.policy._ensembler is not None
        )
        start = time.perf_counter()
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        self.last_timing["inference_seconds"] = time.perf_counter() - start
        self.observation_count += 1
        self.inference_count += int(will_infer)
        return self._actions_to_numpy(action)[0]


def load_policy(name: str, **kwargs) -> PolicyAdapter:
    return PolicyAdapter(name, **kwargs)
