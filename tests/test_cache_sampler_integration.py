"""OPT-IN real checkpoint parity; never run automatically with lightweight tests.

PowerShell (run sequentially, no other model process):
    $env:RUN_SMOLVLA_CACHE_INTEGRATION='1'
    python scripts/run.py -m pytest tests/test_cache_sampler_integration.py -q

Loads the full local SmolVLA checkpoint and uses the saved real camera input.
It performs no simulator rollout, downloading, fitting, or cache-index updates.
"""
import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SMOLVLA_CACHE_INTEGRATION") != "1",
    reason="explicit opt-in required: loads the 605M SmolVLA checkpoint and performs real CPU inference",
)


def test_real_smolvla_full_noise_parity_and_exact_cached_continuation():
    import torch

    from embodied_research.cache_sampler import SmolVLACacheSampler
    from embodied_research.policies import PolicyAdapter

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "artifacts/policy_smoke_observation.npz", allow_pickle=False) as snapshot:
        observation = {key: snapshot[key].copy() for key in snapshot.files}
    observation["task"] = str(observation["task"].item())
    adapter = PolicyAdapter("smolvla", num_steps=10, temporal_ensemble_coeff=None)
    adapter.reset(seed=123)
    adapter.observe(observation)
    sampler = SmolVLACacheSampler(adapter.policy)
    shape = (1, adapter.config.chunk_size, adapter.config.max_action_dim)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(37), dtype=torch.float32)
    with torch.inference_mode():
        expected = adapter.policy.predict_action_chunk(dict(adapter._batch), noise=noise.clone())
    context = sampler.prepare(adapter._batch)
    result = sampler.sample(context, noise=noise, capture_step=8)
    torch.testing.assert_close(result.actions, expected, rtol=0, atol=0)
    resumed = sampler.refine(context, result.latent)
    torch.testing.assert_close(resumed.actions, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(adapter._actions_to_numpy(result.actions), adapter._actions_to_numpy(expected))
    assert result.latent.shape == (50, 32) and resumed.nfe == 2
    assert context.key_features.ndim == 1 and np.isfinite(context.key_features).all()
