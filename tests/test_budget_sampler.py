"""Tiny CPU tensor tests: no checkpoint, Transformers or real simulator."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_research.budget_sampler import BudgetSampler, extract_budget_features
from embodied_research.cache_sampler import (
    LANGUAGE_MASK, LANGUAGE_TOKENS, SmolVLACacheSampler, euler_grid,
)


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)

    class Cache:
        def __init__(self, length, condition):
            self.length, self.condition = length, condition

        def get_seq_length(self):
            return self.length

    class Backend:
        def __init__(self):
            self.vlm_with_expert = self
            self.prefills, self.noise_calls = 0, 0
            self.times = []

        def embed_prefix(self, images, image_masks, tokens, masks, *, state):
            embeddings = torch.cat([
                *[image.mean().expand(1, 1, 4) for image in images],
                tokens.float()[..., None].expand(-1, -1, 4), state[:, None, :4],
            ], dim=1)
            padding = torch.cat([torch.ones(1, len(images), dtype=torch.bool), masks.bool(),
                                 torch.ones(1, 1, dtype=torch.bool)], dim=1)
            attention = torch.zeros_like(padding)
            attention[:, -1] = True
            # Actual upstream may append padding after its state token.
            embeddings = torch.cat([embeddings, torch.full((1, 2, 4), 73.)], dim=1)
            padding = torch.cat([padding, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
            attention = torch.cat([attention, torch.zeros(1, 2, dtype=torch.bool)], dim=1)
            return embeddings, padding, attention

        def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
            self.prefills += 1
            assert past_key_values is None and use_cache
            embeddings = inputs_embeds[0]
            condition = embeddings.masked_fill(~attention_mask.any(dim=-1)[:, :, None], 0).sum() / 100
            return [embeddings * 2 + condition, None], Cache(embeddings.shape[1], condition)

        def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
            self.times.append(float(timestep[0]))
            # Padded channels affect the seven real channels; early slicing fails.
            return (.35 * x_t + .02 * x_t.mean(dim=-1, keepdim=True)
                    + .15 * timestep[:, None, None] + .001 * past_key_values.condition)

        def sample_noise(self, *args):
            self.noise_calls += 1
            raise AssertionError("budget code must never draw noise")

    backend = Backend()
    config = SimpleNamespace(n_obs_steps=1, adapt_to_pi_aloha=False, use_cache=True,
                             rtc_config=None, chunk_size=50, max_action_dim=32,
                             tokenizer_max_length=6, action_feature=SimpleNamespace(shape=(7,)),
                             image_features={"front": None, "wrist": None}, num_steps=10)
    policy = SimpleNamespace(config=config, model=backend,
                             prepare_state=lambda batch: batch["observation.state"],
                             prepare_images=lambda batch: ([batch["front"], batch["wrist"]], None),
                             eval=lambda: None)
    batch = {"front": torch.ones(1, 3, 2, 2), "wrist": torch.full((1, 3, 2, 2), .5),
             "observation.state": torch.arange(8, dtype=torch.float32)[None],
             LANGUAGE_TOKENS: torch.tensor([[7, 8, 99]]),
             LANGUAGE_MASK: torch.tensor([[True, True, False]])}
    base = SmolVLACacheSampler(policy)
    context = base.prepare(batch)
    noise = torch.linspace(-2, 3, 1600, dtype=torch.float32).reshape(1, 50, 32)
    return SimpleNamespace(torch=torch, backend=backend, policy=policy, batch=batch,
                           base=base, budget=BudgetSampler(base), context=context, noise=noise)


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", (
        "import embodied_research.budget_sampler; import sys; "
        "assert not {'torch', 'lerobot', 'transformers'} & set(sys.modules)"
    )], check=True, capture_output=True, text=True)


def reference_full(tiny, steps):
    dt, grid = euler_grid(steps)
    with tiny.torch.inference_mode():
        x = tiny.noise.clone()
        for native_time in grid:
            velocity = tiny.backend.denoise_step(
                x_t=x, prefix_pad_masks=tiny.context.prefix_pad_masks,
                past_key_values=tiny.context.past_key_values,
                timestep=tiny.torch.tensor(native_time, dtype=tiny.torch.float32).expand(1),
            )
            x = x + dt * velocity
    return x


@pytest.mark.parametrize("steps", [2, 4, 10])
@pytest.mark.parametrize("with_probe", [False, True])
def test_exact_all_channel_parity_nfe_and_unchanged_inputs(tiny, steps, with_probe):
    expected = reference_full(tiny, steps)
    old = tiny.base.sample(tiny.context, num_steps=steps, noise=tiny.noise)
    state = tiny.context.state.clone()
    keys = tiny.context.key_features.copy()
    padding = tiny.context.prefix_pad_masks.clone()
    noise = tiny.noise.clone()
    rng = tiny.torch.get_rng_state().clone()
    tiny.backend.times.clear()
    probe = tiny.budget.probe(tiny.context, tiny.noise) if with_probe else None
    result = tiny.budget.sample(tiny.context, tiny.noise, steps, probe)
    tiny.torch.testing.assert_close(result.full_latent, expected, rtol=0, atol=0)
    tiny.torch.testing.assert_close(result.actions, old.actions, rtol=0, atol=0)
    assert result.full_latent.shape == (1, 50, 32)
    assert result.actions.shape == (1, 50, 7)
    assert result.nfe == steps and result.new_evaluations == steps - int(with_probe)
    assert tiny.backend.times == [float(np.float32(t)) for t in euler_grid(steps)[1]]
    assert tiny.backend.noise_calls == 0 and tiny.backend.prefills == 1
    tiny.torch.testing.assert_close(noise, tiny.noise, rtol=0, atol=0)
    tiny.torch.testing.assert_close(state, tiny.context.state, rtol=0, atol=0)
    tiny.torch.testing.assert_close(padding, tiny.context.prefix_pad_masks, rtol=0, atol=0)
    tiny.torch.testing.assert_close(rng, tiny.torch.get_rng_state(), rtol=0, atol=0)
    np.testing.assert_array_equal(keys, tiny.context.key_features)
    assert tiny.context.past_key_values.get_seq_length() == padding.shape[1]
    assert all(np.isfinite(v) and v >= 0 for v in result.timings.values())
    assert result.actions.data_ptr() != result.full_latent.data_ptr()


def test_three_paired_budgets_share_only_one_first_evaluation(tiny):
    originals = {steps: reference_full(tiny, steps) for steps in (2, 4, 10)}
    tiny.backend.times.clear()
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    original_velocity = probe.velocity.clone()
    results = [tiny.budget.sample(tiny.context, tiny.noise.clone(), steps, probe) for steps in (2, 4, 10)]
    assert probe.nfe == 1 and len(tiny.backend.times) == 14
    assert tiny.backend.times.count(1.) == 1
    assert sum(r.new_evaluations for r in results) + probe.nfe == 14
    for steps, result in zip((2, 4, 10), results, strict=True):
        tiny.torch.testing.assert_close(result.full_latent, originals[steps], rtol=0, atol=0)
    tiny.torch.testing.assert_close(probe.velocity, original_velocity, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [0, 1, 3, -2, True, 2., "2", None])
def test_invalid_budget_fails_before_inference(tiny, steps):
    with pytest.raises(ValueError, match="num_steps"):
        tiny.budget.sample(tiny.context, tiny.noise, steps)
    assert not tiny.backend.times


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "list"])
def test_invalid_noise_rejected_by_both_entries(tiny, kind):
    noise = {"shape": tiny.noise[:, :, :7], "dtype": tiny.noise.double(),
             "nan": tiny.torch.full_like(tiny.noise, float("nan")), "list": [1]}[kind]
    for invoke in (tiny.budget.probe, tiny.budget.sample):
        with pytest.raises(ValueError, match="noise"):
            invoke(tiny.context, noise)
    assert not tiny.backend.times


def test_foreign_sampler_context_changed_noise_and_modified_probe_rejected(tiny):
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    other_base = SmolVLACacheSampler(tiny.policy)
    other_context = other_base.prepare(tiny.batch)
    another_context = tiny.base.prepare(tiny.batch)
    before = len(tiny.backend.times)
    with pytest.raises(ValueError, match="this sampler"):
        tiny.budget.sample(other_context, tiny.noise, 2)
    with pytest.raises(ValueError, match="exact context"):
        tiny.budget.sample(another_context, tiny.noise, 2, probe)
    with pytest.raises(ValueError, match="BudgetSampler"):
        BudgetSampler(tiny.base).sample(tiny.context, tiny.noise, 2, probe)
    altered = tiny.noise.clone()
    altered[0, 49, 31] += .125  # Even unused output channels bind the probe.
    with pytest.raises(ValueError, match="exact noise"):
        tiny.budget.sample(tiny.context, altered, 2, probe)
    replacement = probe.velocity.clone()
    replacement[0, 0, 0] += .1
    with pytest.raises(ValueError, match="velocity changed"):
        tiny.budget.sample(tiny.context, tiny.noise, 2, replace(probe, velocity=replacement))
    assert len(tiny.backend.times) == before


@pytest.mark.parametrize("where", ["before", "during"])
def test_unrestored_kv_fails_before_further_evaluation(tiny, where):
    if where == "before":
        tiny.context.past_key_values.length += 1
    else:
        original = tiny.backend.denoise_step

        def corrupt(**kwargs):
            result = original(**kwargs)
            kwargs["past_key_values"].length += 1
            return result

        tiny.backend.denoise_step = corrupt
    with pytest.raises(ValueError, match="restored"):
        tiny.budget.sample(tiny.context, tiny.noise, 10)
    assert len(tiny.backend.times) == (0 if where == "before" else 1)


@pytest.mark.parametrize("kind", ["nan", "dtype", "shape"])
@pytest.mark.parametrize("entry", ["probe", "sample"])
def test_invalid_velocity_fails_closed(tiny, kind, entry):
    def bad(**kwargs):
        value = kwargs["x_t"]
        return {"nan": tiny.torch.full_like(value, float("nan")),
                "dtype": value.double(), "shape": value[:, :, :7]}[kind]

    tiny.backend.denoise_step = bad
    with pytest.raises(ValueError, match="velocity"):
        getattr(tiny.budget, entry)(tiny.context, tiny.noise)


def test_features_use_valid_language_mask_and_modes_have_no_hidden_vlm(tiny):
    # A valid language token has all-zero output. It must still count in mean.
    values = tiny.context.key_features.copy().reshape(9, 4)
    values[2] = 0
    values[3] = np.array([2, 4, 6, 8], dtype=np.float32)
    context = replace(tiny.context, key_features=values.reshape(-1))
    probe = tiny.budget.probe(context, tiny.noise)
    raw = np.arange(8, dtype=np.float32)
    features = extract_budget_features(context, raw, tiny.noise, probe)
    assert features.values.shape == (3 * 4 + 8 + 14 + 14,)
    assert features.values.dtype == np.float32 and not features.values.flags.writeable
    np.testing.assert_array_equal(features.values[slice(*features.feature_slices["language_mean"])], [1, 2, 3, 4])
    np.testing.assert_array_equal(features.values[slice(*features.feature_slices["raw_state"])], raw)
    for mode, spec in features.layout["modes"].items():
        selected = features.for_mode(mode)
        assert len(selected) == spec["dimension"] and not selected.flags.writeable
        np.testing.assert_array_equal(selected, np.concatenate([
            features.values[slice(*features.feature_slices[name])] for name in spec["blocks"]
        ]))
    assert features.layout["modes"]["no_probe"]["blocks"][-1] == "noise_stats"
    assert features.layout["modes"]["proprio_probe"]["blocks"] == ["raw_state", "noise_stats", "velocity_stats"]
    assert features.layout["modes"]["proprio_probe"]["dimension"] == 36
    json.dumps(features.layout, allow_nan=False)
    json.dumps(features.feature_slices)
    with pytest.raises(ValueError, match="unknown"):
        features.for_mode("teacher")


def test_statistics_are_only_first_ten_actions_first_seven_channels(tiny):
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    features = extract_budget_features(tiny.context, np.zeros(8, np.float32), tiny.noise, probe)
    for name, tensor in (("noise_stats", tiny.noise), ("velocity_stats", probe.velocity)):
        source = tensor.numpy()[0, :10, :7]
        expected = np.concatenate([source.mean(axis=0), source.std(axis=0, ddof=0)])
        np.testing.assert_array_equal(features.values[slice(*features.feature_slices[name])], expected)


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "list"])
def test_raw_state_validation(tiny, kind):
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    state = {"dtype": np.zeros(8, np.float64), "shape": np.zeros(7, np.float32),
             "nan": np.full(8, np.nan, np.float32), "list": [0.] * 8}[kind]
    with pytest.raises(ValueError, match="raw_state"):
        extract_budget_features(tiny.context, state, tiny.noise, probe)


@pytest.mark.parametrize("kind", ["bad_dimension", "nonfinite", "masked_nonzero", "trailing_nonzero", "no_language"])
def test_invalid_feature_layout_rejected(tiny, kind):
    context = tiny.context
    if kind == "bad_dimension":
        context = replace(context, feature_contract={**context.feature_contract, "input_dim": 77})
    elif kind in ("nonfinite", "masked_nonzero", "trailing_nonzero"):
        values = context.key_features.copy().reshape(9, 4)
        values[{"nonfinite": 0, "masked_nonzero": 4, "trailing_nonzero": 5}[kind], 0] = (
            np.nan if kind == "nonfinite" else 2.)
        context = replace(context, key_features=values.reshape(-1))
    else:
        padding = context.prefix_pad_masks.clone()
        padding[0, 2:5] = False
        context = replace(context, prefix_pad_masks=padding)
    probe = tiny.budget.probe(context, tiny.noise)
    with pytest.raises(ValueError):
        extract_budget_features(context, np.zeros(8, np.float32), tiny.noise, probe)


def test_features_also_validate_probe_binding(tiny):
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    with pytest.raises(ValueError, match="exact noise"):
        extract_budget_features(tiny.context, np.zeros(8, np.float32), tiny.noise + .01, probe)


@pytest.mark.parametrize("field", ["state", "mask", "keys", "contract"])
def test_probe_rejects_mutated_context_metadata(tiny, field):
    probe = tiny.budget.probe(tiny.context, tiny.noise)
    with tiny.torch.inference_mode():
        if field == "state":
            tiny.context.state[0, 0] += 1
        elif field == "mask":
            tiny.context.prefix_pad_masks[0, 0] = False
        elif field == "keys":
            tiny.context.key_features.setflags(write=True)
            tiny.context.key_features[0] += 1
        else:
            tiny.context.feature_contract["source"] = "changed"
    before = len(tiny.backend.times)
    with pytest.raises(ValueError, match="metadata changed"):
        tiny.budget.sample(tiny.context, tiny.noise, 2, probe)
    assert len(tiny.backend.times) == before


def test_reject_wrong_configuration_at_construction_and_afterward(tiny):
    with pytest.raises(TypeError, match="existing"):
        BudgetSampler(object())
    tiny.policy.config.chunk_size = 51
    with pytest.raises(ValueError, match="contract"):
        BudgetSampler(tiny.base)
    with pytest.raises(ValueError, match="contract"):
        tiny.budget.sample(tiny.context, tiny.noise)


@pytest.fixture
def real_validator():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_budget_sampler_real.py"
    spec = importlib.util.spec_from_file_location("budget_real_validation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("steps", [2, 4, 10])
def test_real_validation_capture_recovers_full_endpoint_and_restores_method(tiny, real_validator, steps):
    original_method = tiny.backend.denoise_step.__func__
    expected = reference_full(tiny, steps)
    with tiny.torch.inference_mode(), real_validator.capture_denoise(tiny.backend, tiny.context) as capture:
        capture["phase"] = f"reference_{steps}"
        sampled = tiny.base.sample(tiny.context, noise=tiny.noise, num_steps=steps)
        recovered = capture["last_input"] + (-1.0 / steps) * capture["last_velocity"]
        assert real_validator.compare_tensors(recovered, expected)["byte_equal"]
        assert real_validator.compare_tensors(recovered[:, :, :7], sampled.actions)["byte_equal"]
        assert len(capture["calls"]) == steps
        assert all(call["phase"] == f"reference_{steps}" for call in capture["calls"])
        assert all(call["kv_length_before"] == call["kv_length_after"] for call in capture["calls"])
    assert "denoise_step" not in vars(tiny.backend)
    assert tiny.backend.denoise_step.__func__ is original_method


def test_real_validation_capture_restores_existing_override_after_exception(tiny, real_validator):
    previous = tiny.backend.denoise_step
    tiny.backend.denoise_step = previous
    with pytest.raises(RuntimeError, match="deliberate"):
        with real_validator.capture_denoise(tiny.backend, tiny.context):
            raise RuntimeError("deliberate")
    assert vars(tiny.backend)["denoise_step"] is previous


def test_real_validation_distinguishes_value_equality_from_byte_equality(tiny, real_validator):
    left = tiny.torch.tensor([0.], dtype=tiny.torch.float32)
    right = tiny.torch.tensor([-0.], dtype=tiny.torch.float32)
    comparison = real_validator.compare_tensors(left, right)
    assert comparison["value_equal"] and not comparison["byte_equal"]


def test_real_validation_refuses_existing_or_public_destination_before_loading(real_validator, tmp_path, monkeypatch):
    monkeypatch.setattr(real_validator, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="private"):
        real_validator.run(tmp_path / "public")
    existing = tmp_path / ".runtime/parity"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        real_validator.run(existing)


def test_real_validation_help_does_not_run_model():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_budget_sampler_real.py"
    result = subprocess.run([sys.executable, str(path), "--help"], check=True, capture_output=True, text=True)
    assert "--output" in result.stdout
