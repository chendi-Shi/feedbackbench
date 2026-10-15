"""Small tensor fixtures only: no pretrained model, Transformers, or simulator."""
import copy
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_research.cache_sampler import (
    LANGUAGE_MASK,
    LANGUAGE_TOKENS,
    SmolVLACacheSampler,
    euler_grid,
    latent_time_contract,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")

    class PrefixCache:
        def __init__(self, length, condition):
            self.length = length
            self.condition = condition

        def get_seq_length(self):
            return self.length

    class Backend:
        def __init__(self):
            self.vlm_with_expert = self
            self.prefill_calls = 0
            self.noise_calls = 0
            self.times = []
            self.extra_image_token = False

        def embed_prefix(self, images, image_masks, tokens, masks, *, state):
            pieces = [image.mean().expand(1, 1, 4) for image in images]
            if self.extra_image_token:
                pieces.append(pieces[0].clone())
            language = tokens.float()[..., None].expand(-1, -1, 4)
            embeddings = torch.cat([*pieces, language, state[:, None, :4]], dim=1)
            padding = torch.cat([torch.ones(1, len(pieces), dtype=torch.bool), masks.bool(),
                                 torch.ones(1, 1, dtype=torch.bool)], dim=1)
            attention = torch.zeros_like(padding)
            attention[:, -1] = True
            # Exercise upstream prefix_length padding after the state token.
            embeddings = torch.cat([embeddings, torch.full((1, 2, 4), 123.)], dim=1)
            padding = torch.cat([padding, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
            attention = torch.cat([attention, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
            return embeddings, padding, attention

        def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
            assert use_cache and past_key_values is None and inputs_embeds[1] is None
            self.prefill_calls += 1
            valid = attention_mask.any(dim=-1)
            embeddings = inputs_embeds[0]
            context = embeddings.masked_fill(~valid[:, :, None], 0).sum() / valid.sum()
            output = embeddings * 2 + context
            return [output, None], PrefixCache(embeddings.shape[1], context)

        def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
            self.times.append(float(timestep[0]))
            # Time and every padded action channel affect the velocity.
            return .35 * x_t + .02 * x_t.mean(dim=-1, keepdim=True) + (
                .15 * timestep[:, None, None] + .001 * past_key_values.condition
            )

        def sample_noise(self, shape, device):
            self.noise_calls += 1
            return torch.zeros(shape, dtype=torch.float32, device=device)

    config = SimpleNamespace(n_obs_steps=1, adapt_to_pi_aloha=False, use_cache=True,
                             rtc_config=None, chunk_size=3, max_action_dim=8,
                             tokenizer_max_length=6, action_feature=SimpleNamespace(shape=(7,)),
                             image_features={"front": None, "wrist": None}, num_steps=10)
    backend = Backend()
    policy = SimpleNamespace(config=config, model=backend,
                             prepare_state=lambda batch: batch["observation.state"],
                             prepare_images=lambda batch: ([batch["front"], batch["wrist"]],
                                                            [torch.ones(1, dtype=torch.bool)] * 2),
                             eval=lambda: None)
    batch = {"front": torch.ones(1, 3, 2, 2), "wrist": torch.full((1, 3, 2, 2), .5),
             "observation.state": torch.arange(8, dtype=torch.float32)[None],
             LANGUAGE_TOKENS: torch.tensor([[7, 8]]), LANGUAGE_MASK: torch.ones(1, 2, dtype=torch.bool)}
    noise = torch.linspace(-1.0, 1.0, 24).reshape(1, 3, 8)
    return SimpleNamespace(torch=torch, policy=policy, backend=backend, batch=batch, noise=noise)


def test_module_import_does_not_import_model_or_torch_stack():
    script = (
        "import embodied_research.cache_sampler; import sys; "
        "assert not any(name.split('.')[0] in {'torch','lerobot','transformers'} for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_native_time_is_reversed_paper_time_and_refinement_preserves_original_grid():
    dt, grid = euler_grid(10)
    assert dt == -.1 and len(grid) == 10
    assert euler_grid(10, start_step=8) == (dt, grid[8:])
    contract = latent_time_contract()
    assert contract["native_time"] == .2 and contract["paper_tau"] == .8
    assert contract["remaining_nfe"] == 2
    assert euler_grid(2)[1] == (1., .5)
    assert euler_grid(2)[1] != euler_grid(10, start_step=8)[1]


@pytest.mark.parametrize("steps,start", [(0, 0), (-1, 0), (True, 0), (2.0, 0), (10, -1),
                                         (10, 10), (10, True), (10, 1.5)])
def test_invalid_time_grids_rejected(steps, start):
    with pytest.raises(ValueError):
        euler_grid(steps, start_step=start)


def test_complete_sampling_and_two_step_resume_are_exact_and_do_not_touch_inputs(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    original = {key: value.clone() for key, value in tiny.batch.items()}
    config = copy.deepcopy(vars(tiny.policy.config))
    noise = tiny.noise.clone()
    context = sampler.prepare(tiny.batch)
    full = sampler.sample(context, num_steps=10, noise=noise, capture_step=8)
    assert full.nfe == 10 and full.latent.shape == (3, 8) and full.actions.shape == (1, 3, 7)
    assert full.latent.dtype == np.float32 and not full.latent.flags.writeable
    assert full.latent_contract == sampler.latent_contract
    saved = full.latent.copy()
    resumed = sampler.refine(context, full.latent)
    tiny.torch.testing.assert_close(resumed.actions, full.actions, rtol=0, atol=0)
    np.testing.assert_array_equal(full.latent, saved)
    assert resumed.nfe == 2 and resumed.latent is None and resumed.latent_contract is None
    assert tiny.backend.prefill_calls == 1 and tiny.backend.noise_calls == 0
    assert tiny.backend.times[-2:] == [float(np.float32(t)) for t in euler_grid(10)[1][-2:]]
    tiny.torch.testing.assert_close(noise, tiny.noise, rtol=0, atol=0)
    for key, value in original.items():
        tiny.torch.testing.assert_close(tiny.batch[key], value, rtol=0, atol=0)
    assert vars(tiny.policy.config) == config
    assert all(np.isfinite(value) and value >= 0 for value in context.timings.values())
    assert all(np.isfinite(value) and value >= 0 for value in full.timings.values())


@pytest.mark.parametrize("steps", [2, 5, 10])
def test_against_installed_upstream_euler_primitive_with_tiny_velocity_field(tiny, steps):
    # This upstream file imports Torch only, not any pretrained model stack.
    path = ROOT / "vendor/MINERVA/src/lerobot/policies/common/flow_matching.py"
    if not path.is_file():
        pytest.skip("pinned upstream source is not installed")
    upstream_euler = runpy.run_path(str(path))["euler_integrate"]
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    with tiny.torch.inference_mode():
        expected = upstream_euler(
            lambda x, t: tiny.backend.denoise_step(x_t=x, timestep=t,
                prefix_pad_masks=context.prefix_pad_masks, past_key_values=context.past_key_values),
            tiny.noise.clone(), steps,
        )[:, :, :7]
    result = sampler.sample(context, num_steps=steps, noise=tiny.noise)
    tiny.torch.testing.assert_close(result.actions, expected, rtol=0, atol=0)
    assert result.nfe == steps and tiny.policy.config.num_steps == 10


def test_key_layout_is_fixed_across_instruction_lengths_and_ignores_masked_tokens(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    first = sampler.prepare(tiny.batch)
    padded = dict(tiny.batch)
    padded[LANGUAGE_TOKENS] = tiny.torch.tensor([[7, 8, 999, 888]])
    padded[LANGUAGE_MASK] = tiny.torch.tensor([[True, True, False, False]])
    second = sampler.prepare(padded)
    assert first.feature_contract == second.feature_contract
    np.testing.assert_array_equal(first.key_features, second.key_features)
    assert first.key_features.shape == (36,) and first.key_features.dtype == np.float32
    assert not first.key_features.flags.writeable
    canonical = first.key_features.reshape(9, 4)
    np.testing.assert_array_equal(canonical[4:8], np.zeros((4, 4)))
    changed = dict(tiny.batch)
    changed[LANGUAGE_TOKENS] = tiny.torch.tensor([[7, 9, 11]])
    changed[LANGUAGE_MASK] = tiny.torch.ones(1, 3, dtype=tiny.torch.bool)
    third = sampler.prepare(changed)
    assert third.feature_contract == first.feature_contract
    assert not np.array_equal(third.key_features, first.key_features)
    changed = dict(tiny.batch)
    changed["observation.state"] = tiny.batch["observation.state"] + .5
    fourth = sampler.prepare(changed)
    assert not np.array_equal(fourth.key_features, first.key_features)


def test_refinement_uses_current_context_and_preserves_a_frozen_cached_latent(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    cached = sampler.sample(context, noise=tiny.noise, capture_step=8)
    changed = dict(tiny.batch)
    changed["front"] = tiny.batch["front"] + 2
    current = sampler.prepare(changed)
    refined = sampler.refine(current, cached.latent)
    assert not tiny.torch.equal(refined.actions, cached.actions)
    repeated = sampler.refine(context, cached.latent)
    tiny.torch.testing.assert_close(repeated.actions, cached.actions, rtol=0, atol=0)


def test_foreign_context_corrupted_kv_and_changed_feature_layout_rejected(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    other = SmolVLACacheSampler(tiny.policy)
    with pytest.raises(ValueError, match="this sampler"):
        other.sample(context)
    context.past_key_values.length += 1
    with pytest.raises(ValueError, match="restored"):
        sampler.sample(context)
    tiny.backend.extra_image_token = True
    with pytest.raises(ValueError, match="layout changed"):
        sampler.prepare(tiny.batch)


@pytest.mark.parametrize("field,value", [("n_obs_steps", 2), ("adapt_to_pi_aloha", True),
                                         ("use_cache", False), ("rtc_config", SimpleNamespace(enabled=True))])
def test_unsupported_configuration_rejected(tiny, field, value):
    setattr(tiny.policy.config, field, value)
    with pytest.raises(ValueError):
        SmolVLACacheSampler(tiny.policy)


@pytest.mark.parametrize("kind", ["wrong_shape", "float64", "nan", "list"])
def test_invalid_cached_latent_rejected(tiny, kind):
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    latent = np.zeros((3, 8), dtype=np.float32)
    if kind == "wrong_shape":
        latent = latent[:, :7]
    elif kind == "float64":
        latent = latent.astype(np.float64)
    elif kind == "nan":
        latent[0, 0] = np.nan
    else:
        latent = latent.tolist()
    with pytest.raises(ValueError, match="latent"):
        sampler.refine(context, latent)


def test_invalid_noise_capture_and_batch_rejected(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    for noise in (tiny.noise.double(), tiny.noise[:, :, :7],
                  tiny.torch.full_like(tiny.noise, float("nan"))):
        with pytest.raises(ValueError):
            sampler.sample(context, noise=noise)
    for step in (0, 10, -1, True):
        with pytest.raises(ValueError):
            sampler.sample(context, noise=tiny.noise, capture_step=step)
    missing = dict(tiny.batch)
    del missing["wrist"]
    with pytest.raises(ValueError, match="missing"):
        sampler.prepare(missing)
    long_batch = dict(tiny.batch)
    long_batch[LANGUAGE_TOKENS] = tiny.torch.ones(1, 7, dtype=tiny.torch.long)
    with pytest.raises(ValueError, match="tokenizer_max_length"):
        sampler.prepare(long_batch)


def test_nonfinite_velocity_cannot_escape_as_actions_or_cached_latent(tiny):
    sampler = SmolVLACacheSampler(tiny.policy)
    context = sampler.prepare(tiny.batch)
    tiny.backend.denoise_step = lambda **kwargs: tiny.torch.full_like(kwargs["x_t"], float("nan"))
    with pytest.raises(ValueError, match="finite"):
        sampler.sample(context, noise=tiny.noise, capture_step=8)
