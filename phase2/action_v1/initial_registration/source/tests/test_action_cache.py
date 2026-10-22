import hashlib
import io
import json

import numpy as np
import pytest

from embodied_research.action_cache import (
    CacheProvenance,
    FrozenActionCache,
    ProjectionConfig,
    SparseTernaryProjection,
)


def provenance(**updates):
    fields = {
        "model_revision": "frozen-revision",
        "model_sha256": "a" * 64,
        "feature_contract": {"version": 1, "parts": ["vision", "state"], "normalization": "declared"},
        "flow_time": 0.5,
        "development_case_ids": ("dev-1", "dev-2"),
    }
    return CacheProvenance(**(fields | updates))


def cache(capacity=4096):
    return FrozenActionCache(SparseTernaryProjection(ProjectionConfig(16, 8, 0.5, 17)),
                             (3, 2), provenance(), threshold=0.95, capacity=capacity)


def feature():
    return np.arange(16, dtype=np.float32)


def latent(value=1):
    return np.full((3, 2), value, dtype=np.float32)


def filled_cache():
    result = cache()
    result.add(feature(), latent(), case_id="dev-1")
    return result.freeze()


def rewrite_manifest(path, update):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    manifest.pop("manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                           allow_nan=False).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_sparse_projection_reproducible_balanced_and_matches_explicit_rows():
    config = ProjectionConfig(32, 12, 0.25, 123)
    first = SparseTernaryProjection(config)
    second = SparseTernaryProjection(config)
    np.testing.assert_array_equal(first.positive_indices, second.positive_indices)
    np.testing.assert_array_equal(first.negative_indices, second.negative_indices)
    assert first.positive_indices.shape == first.negative_indices.shape == (12, 4)
    for plus, minus in zip(first.positive_indices, first.negative_indices, strict=True):
        assert len(set(plus) | set(minus)) == 8
    values = np.arange(32, dtype=np.float64) ** 2
    explicit = values[first.positive_indices].sum(1) - values[first.negative_indices].sum(1)
    expected = explicit / np.linalg.norm(explicit)
    np.testing.assert_allclose(first.project(values), expected, atol=1e-7)
    np.testing.assert_array_equal(first.project(values), second.project(values))
    assert first.project(values).dtype == np.float32
    assert np.linalg.norm(first.project(values)) == pytest.approx(1)
    changed = SparseTernaryProjection(ProjectionConfig(32, 12, 0.25, 124))
    assert not np.array_equal(first.positive_indices, changed.positive_indices)
    assert config.nonzeros_per_row == 8


def test_high_dimensional_projection_stores_only_sparse_indices_and_handles_large_features():
    projection = SparseTernaryProjection(ProjectionConfig(1_000_000, 4, 0.00001, 2))
    assert projection.positive_indices.shape == projection.negative_indices.shape == (4, 5)
    assert projection.positive_indices.nbytes + projection.negative_indices.nbytes == 320
    small = SparseTernaryProjection(ProjectionConfig(16, 8, 0.5, 17))
    result = small.project(np.linspace(-1, 1, 16) * 1e308)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("kwargs", [
    {"input_dim": 1}, {"output_dim": 0}, {"density": 0}, {"density": float("nan")},
    {"density": 1.1}, {"seed": -1}, {"input_dim": True},
])
def test_invalid_projection_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ProjectionConfig(**({"input_dim": 16} | kwargs))


@pytest.mark.parametrize("bad", [np.ones((1, 16)), np.ones(15), np.full(16, np.nan),
                                  np.full(16, np.inf), np.ones(16, dtype=complex), ["1"] * 16])
def test_projection_rejects_nonfinite_wrong_dimension_and_nonreal_features(bad):
    with pytest.raises(ValueError):
        cache().projection.project(bad)


def test_empty_and_zero_projection_miss_and_invalid_query_is_not_silently_accepted():
    result = cache().freeze()
    assert not result.lookup(feature()).hit
    assert result.lookup(feature()).similarity is None
    with pytest.raises(ValueError):
        result.lookup(np.zeros(15))
    assert not filled_cache().lookup(np.zeros(16), threshold=-1).hit
    assert not filled_cache().lookup(np.ones(16), threshold=-1).hit  # Balanced rows annihilate constants.
    with pytest.raises(ValueError, match="zero projected"):
        cache().add(np.ones(16), latent(), case_id="dev-1")


def test_exact_cosine_threshold_ties_and_returned_latent_isolated():
    result = cache()
    original = latent(7)
    result.add(feature(), original, case_id="dev-1")
    result.add(feature(), latent(8), case_id="dev-2")
    original[:] = 999
    result.freeze()
    hit = result.lookup(feature(), threshold=1)
    assert hit.hit and hit.index == 0 and hit.case_id == "dev-1"
    assert hit.similarity == pytest.approx(1)
    np.testing.assert_array_equal(hit.latent, latent(7))
    hit.latent[:] = -1
    np.testing.assert_array_equal(result.lookup(feature()).latent, latent(7))
    miss = result.lookup(-feature())
    assert not miss.hit and miss.similarity == pytest.approx(-1) and miss.latent is None
    assert miss.index == 0  # A diagnostic nearest index is not an action/latent.
    assert result.lookup(-feature(), threshold=-1).hit
    with pytest.raises(ValueError):
        result.lookup(feature(), threshold=float("nan"))


def test_nonidentical_top1_uses_actual_cosine_instead_of_insertion_order():
    result = cache()
    result.add(-feature(), latent(1), case_id="dev-1")
    result.add(feature(), latent(2), case_id="dev-2")
    result.freeze()
    query = feature() + np.sin(feature())
    query_key = result.projection.project(query).astype(np.float64)
    stored_key = result.projection.project(feature()).astype(np.float64)
    expected = float(stored_key @ query_key / (np.linalg.norm(stored_key) * np.linalg.norm(query_key)))
    hit = result.lookup(query, threshold=expected)
    assert hit.hit and hit.index == 1 and hit.similarity == pytest.approx(expected)
    assert not result.lookup(query, threshold=min(1, expected + 1e-5)).hit


def test_exact_repeated_keys_hit_at_unit_threshold_despite_roundoff():
    projection = SparseTernaryProjection(ProjectionConfig(32, 500, 0.5, 17))
    result = FrozenActionCache(projection, (3, 2), provenance(), threshold=1)
    features = np.random.default_rng(4).normal(size=(8, 32)).astype(np.float32)
    for vector in features:
        result.add(vector, latent(), case_id="dev-1")
    result.freeze()
    for index, vector in enumerate(features):
        match = result.lookup(vector)
        assert match.hit and match.index == index and match.similarity == 1


def test_freeze_capacity_and_development_identity_are_enforced(tmp_path):
    result = cache(capacity=1)
    with pytest.raises(RuntimeError, match="frozen"):
        result.lookup(feature())
    with pytest.raises(RuntimeError, match="frozen"):
        result.save(tmp_path / "premature")
    with pytest.raises(ValueError, match="development"):
        result.add(feature(), latent(), case_id="test-case")
    assert result.add(feature(), latent(), case_id="dev-1") == 0
    with pytest.raises(OverflowError):
        result.add(feature(), latent(), case_id="dev-2")
    result.freeze()
    before = result.cache_sha256
    assert result.freeze() is result
    with pytest.raises(RuntimeError, match="frozen"):
        result.add(feature(), latent(), case_id="dev-1")
    with pytest.raises(AttributeError):
        result.threshold = 0
    with pytest.raises(TypeError):
        result.provenance.feature_contract["version"] = 5
    for _ in range(3):
        result.lookup(feature())
    assert result.cache_sha256 == before and len(result) == 1


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((3, 2), np.inf),
                                  np.full((3, 2), 1e100), np.ones((3, 2), dtype=complex)])
def test_invalid_latent_never_adds_partial_entry(bad):
    result = cache()
    with pytest.raises(ValueError):
        result.add(feature(), bad, case_id="dev-1")
    assert result.size == 0


def test_provenance_detaches_metadata_and_binds_all_critical_fields():
    contract = {"parts": ["image"]}
    meta = provenance(feature_contract=contract)
    contract["parts"].append("test oracle")
    assert meta.to_dict()["feature_contract"] == {"parts": ["image"]}
    original = cache().provenance_sha256
    for changed in [provenance(model_revision="other"), provenance(model_sha256="b" * 64),
                    provenance(feature_contract={"version": 2}), provenance(flow_time=0.4),
                    provenance(development_case_ids=("dev-1",))]:
        assert FrozenActionCache(cache().projection, (3, 2), changed).provenance_sha256 != original
    assert FrozenActionCache(SparseTernaryProjection(ProjectionConfig(16, 8, 0.5, 18)),
                             (3, 2), provenance()).provenance_sha256 != original
    with pytest.raises(ValueError):
        provenance(feature_contract={"bad": float("nan")})
    with pytest.raises(ValueError):
        provenance(model_sha256="unverified")


@pytest.mark.parametrize("empty", [False, True])
def test_roundtrip_frozen_float32_and_no_overwrite(tmp_path, empty):
    original = cache().freeze() if empty else filled_cache()
    prefix = tmp_path / "cache.v1"
    npz_path, json_path = original.save(prefix)
    snapshot = (npz_path.read_bytes(), json_path.read_bytes())
    loaded = FrozenActionCache.load(prefix, expected_provenance_sha256=original.provenance_sha256)
    assert loaded.is_frozen and loaded.size == original.size
    assert loaded.cache_sha256 == original.cache_sha256
    assert loaded.entry_case_ids == original.entry_case_ids
    if not empty:
        np.testing.assert_array_equal(loaded.lookup(feature()).latent, latent())
        assert loaded.lookup(feature()).latent.dtype == np.float32
    with pytest.raises(FileExistsError):
        original.save(prefix)
    assert (npz_path.read_bytes(), json_path.read_bytes()) == snapshot
    with pytest.raises(ValueError, match="experiment protocol"):
        FrozenActionCache.load(prefix, expected_provenance_sha256="b" * 64)


def test_partial_existing_manifest_is_not_overwritten(tmp_path):
    prefix = tmp_path / "cache"
    json_path = tmp_path / "cache.json"
    json_path.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        filled_cache().save(prefix)
    assert json_path.read_text(encoding="utf-8") == "preserve me"
    assert not (tmp_path / "cache.npz").exists()


def test_payload_and_manifest_corruption_rejected(tmp_path):
    prefix = tmp_path / "cache"
    npz_path, json_path = filled_cache().save(prefix)
    raw = npz_path.read_bytes()
    npz_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="NPZ SHA"):
        FrozenActionCache.load(prefix)
    npz_path.write_bytes(raw)
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    manifest["spec"]["provenance"]["model_revision"] = "changed"
    json_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA"):
        FrozenActionCache.load(prefix)


@pytest.mark.parametrize("corruption", ["nonfinite", "shape", "projection", "object", "content"])
def test_load_validates_arrays_even_with_rehashed_payload(tmp_path, corruption):
    prefix = tmp_path / corruption
    npz_path, json_path = filled_cache().save(prefix)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if corruption == "nonfinite":
        arrays["latents"][0, 0, 0] = np.nan
    elif corruption == "shape":
        arrays["latents"] = np.zeros((1, 2, 3), dtype=np.float32)
    elif corruption == "projection":
        arrays["positive_indices"][0, 0] ^= 1
    elif corruption == "object":
        arrays["latents"] = np.array([["pickle forbidden"]], dtype=object)
    else:
        arrays["latents"][0, 0, 0] += 1
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    npz_path.write_bytes(buffer.getvalue())
    rewrite_manifest(json_path, lambda value: value.update(npz_sha256=hashlib.sha256(buffer.getvalue()).hexdigest()))
    with pytest.raises(ValueError):
        FrozenActionCache.load(prefix)
