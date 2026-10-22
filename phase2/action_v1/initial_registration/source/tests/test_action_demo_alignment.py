import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("alignment", Path(__file__).resolve().parents[1] / "scripts/validate_action_demo_alignment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("offset", [0, 1])
def test_temporal_alignment_from_published_state_rows(offset):
    states = np.arange(60, dtype=float).reshape(6, 10)
    states[:, 0] = np.arange(6) / 20
    obs = states[:, 1:10].copy()
    if offset:
        obs[:-1] = states[1:, 1:10]
    assert module.infer_offset(states, obs[:, :7], obs[:, 7:])["action_offset"] == offset


def test_stationary_or_inconsistent_alignment_not_claimed_passed():
    with pytest.raises(ValueError, match="Ambiguous"):
        module.infer_offset(np.zeros((4, 10)), np.zeros((4, 7)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="Ambiguous"):
        module.infer_offset(np.zeros((4, 10)), np.ones((4, 7)), np.ones((4, 2)))


def test_mixed_source_offsets_fail_closed():
    with pytest.raises(ValueError, match="one verified"):
        module.common_offset([{"action_offset": 0}, {"action_offset": 1}])


def test_camera_orientation_comparison_distinguishes_axes():
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    scores = module.image_orientation_errors(image, image[::-1, ::-1])
    assert min(scores, key=scores.get) == "rot180"
    assert scores["rot180"] == 0


def test_asset_relocation_only_files_and_rejects_escape(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "texture.png").write_bytes(b"fixture")
    xml = '<mujoco><asset><texture name="x" file="/old/chiliocosm/assets/texture.png"/></asset><option timestep="0.002"/></mujoco>'
    rewritten, records = module.rewrite_asset_paths(xml, assets, tmp_path / "robosuite")
    assert 'timestep="0.002"' in rewritten
    assert len(records) == 1
    with pytest.raises(ValueError, match="Missing or escaped"):
        module.rewrite_asset_paths(xml.replace("texture.png", "../escape.png"), assets, tmp_path / "robosuite")


def test_post_step_cache_uses_fixed_physics_step_and_preserves_current_qpos(monkeypatch):
    second_spec = importlib.util.spec_from_file_location("alignment_v2", Path(__file__).resolve().parents[1]
                                                        / "scripts/validate_action_demo_alignment_v2.py")
    second = importlib.util.module_from_spec(second_spec)
    second_spec.loader.exec_module(second)
    model = SimpleNamespace(opt=SimpleNamespace(integrator=0, timestep=.002))
    data = SimpleNamespace(qpos=np.asarray([1., 2.]), qvel=np.asarray([.5, -.5]))
    def integrate(model, qpos, qvel, dt):
        qpos += qvel * dt
    def forward(model, data):
        data.cached_pose = data.qpos.copy()
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_integratePos=integrate, mj_forward=forward))
    sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
    second.reconstruct_post_step_cache(sim)
    np.testing.assert_array_equal(data.qpos, [1., 2.])
    np.testing.assert_allclose(data.cached_pose, [.999, 2.001])
    model.opt.integrator = 1
    with pytest.raises(ValueError, match="Euler"):
        second.reconstruct_post_step_cache(sim)
