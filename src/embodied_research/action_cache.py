"""ActionCache-inspired offline frozen cache, not an official-code reproduction.

The caller supplies features and intermediate flow latents from development cases.
This module has no success signal, test-time update, eviction, or model dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


PROJECTION_ALGORITHM = "balanced_sparse_ternary_pcg64_floyd_v1"
FORMAT_VERSION = 1
MAX_CAPACITY = 4096


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _json_value(value: Any) -> Any:
    """Make a detached JSON value, rejecting coercions, NaNs, and non-string keys."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError("metadata must contain only finite JSON values")


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _immutable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable_json(item) for item in value)
    return value


def _array(value: Any, shape: tuple[int, ...], name: str, dtype: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite representable values")
    return result


@dataclass(frozen=True)
class ProjectionConfig:
    input_dim: int
    output_dim: int = 128
    density: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        _integer(self.input_dim, "input_dim", 2, (1 << 63) - 1)
        _integer(self.output_dim, "output_dim", 1)
        _integer(self.seed, "seed", 0, (1 << 64) - 1)
        density = _number(self.density, "density", 0.0, 1.0)
        if density == 0:
            raise ValueError("density must be positive")
        object.__setattr__(self, "density", density)

    @property
    def nonzeros_per_row(self) -> int:
        """Round to balanced sign pairs, with at least one pair in every row."""
        pairs = max(1, int(math.floor(self.input_dim * self.density / 2 + 0.5)))
        return 2 * min(pairs, self.input_dim // 2)


class SparseTernaryProjection:
    """Fixed balanced +/-1 rows; storage is O(output_dim * nonzeros_per_row).

    The requested density is rounded to an even count (at least two nonzeros).
    PCG64 raw bits, explicit rejection sampling, Floyd sampling and Fisher-Yates
    avoid a dense matrix and avoid allocating an input-dimension permutation.
    """

    def __init__(self, config: ProjectionConfig):
        if not isinstance(config, ProjectionConfig):
            raise ValueError("config must be ProjectionConfig")
        self._config = config
        rng = np.random.PCG64(config.seed)

        def randbelow(bound: int) -> int:
            limit = (1 << 64) - ((1 << 64) % bound)
            while True:
                value = int(rng.random_raw())
                if value < limit:
                    return value % bound

        nnz = config.nonzeros_per_row
        indices = np.empty((config.output_dim, nnz), dtype=np.int64)
        for row in range(config.output_dim):
            selected: set[int] = set()
            ordered = []
            for upper in range(config.input_dim - nnz, config.input_dim):
                candidate = randbelow(upper + 1)
                chosen = upper if candidate in selected else candidate
                selected.add(chosen)
                ordered.append(chosen)
            for index in range(nnz - 1, 0, -1):
                other = randbelow(index + 1)
                ordered[index], ordered[other] = ordered[other], ordered[index]
            indices[row] = ordered
        self._positive = indices[:, : nnz // 2].copy()
        self._negative = indices[:, nnz // 2 :].copy()
        self._positive.setflags(write=False)
        self._negative.setflags(write=False)

    @property
    def config(self) -> ProjectionConfig:
        return self._config

    @property
    def positive_indices(self) -> np.ndarray:
        return self._positive.copy()

    @property
    def negative_indices(self) -> np.ndarray:
        return self._negative.copy()

    def project(self, feature: Any) -> np.ndarray:
        vector = _array(feature, (self.config.input_dim,), "feature", np.float64)
        scale = np.max(np.abs(vector))
        if scale == 0:
            return np.zeros(self.config.output_dim, dtype=np.float32)
        vector /= scale  # Positive scaling preserves the key while avoiding overflow.
        projected = vector[self._positive].sum(axis=1) - vector[self._negative].sum(axis=1)
        norm = np.linalg.norm(projected)
        if norm == 0:
            return np.zeros(self.config.output_dim, dtype=np.float32)
        return (projected / norm).astype(np.float32)


@dataclass(frozen=True)
class CacheProvenance:
    model_revision: str
    model_sha256: str
    feature_contract: Mapping[str, Any]
    flow_time: float
    development_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be a nonempty string")
        _digest(self.model_sha256, "model_sha256")
        if not isinstance(self.feature_contract, Mapping) or not self.feature_contract:
            raise ValueError("feature_contract must be a nonempty JSON object")
        contract = _json_value(self.feature_contract)
        object.__setattr__(self, "feature_contract", _immutable_json(contract))
        object.__setattr__(self, "flow_time", _number(self.flow_time, "flow_time", 0.0, 1.0))
        if not isinstance(self.development_case_ids, (tuple, list)):
            raise ValueError("development_case_ids must be a sequence of case identities")
        cases = tuple(self.development_case_ids)
        if not cases or any(not isinstance(case, str) or not case.strip() for case in cases):
            raise ValueError("development_case_ids must contain nonempty strings")
        if len(cases) != len(set(cases)):
            raise ValueError("development_case_ids must be unique")
        object.__setattr__(self, "development_case_ids", cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "feature_contract": _json_value(self.feature_contract),
            "flow_time": self.flow_time,
            "development_case_ids": list(self.development_case_ids),
        }


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    similarity: float | None
    index: int | None
    latent: np.ndarray | None
    case_id: str | None = None


class FrozenActionCache:
    """Append-only development builder followed by immutable exact-cosine lookup.

    ``add`` copies inputs and accepts only declared development identities. Call
    ``freeze`` before lookup or persistence. Retrieval never updates the cache.
    Zero projected vectors have undefined cosine: additions fail, queries miss.
    """

    def __init__(
        self,
        projection: SparseTernaryProjection,
        latent_shape: tuple[int, ...],
        provenance: CacheProvenance,
        threshold: float = 0.95,
        capacity: int = MAX_CAPACITY,
    ):
        if not isinstance(projection, SparseTernaryProjection):
            raise ValueError("projection must be SparseTernaryProjection")
        if not isinstance(provenance, CacheProvenance):
            raise ValueError("provenance must be CacheProvenance")
        if not isinstance(latent_shape, (tuple, list)) or not latent_shape:
            raise ValueError("latent_shape must be a nonempty sequence of positive integers")
        self._latent_shape = tuple(_integer(value, "latent dimension", 1) for value in latent_shape)
        self._threshold = _number(threshold, "threshold", -1.0, 1.0)
        self._capacity = _integer(capacity, "capacity", 1, MAX_CAPACITY)
        self._projection = projection
        self._provenance = provenance
        self._is_frozen = False
        self._key_builder: list[np.ndarray] = []
        self._latent_builder: list[np.ndarray] = []
        self._case_ids: tuple[str, ...] = ()
        self._keys = np.empty((0, projection.config.output_dim), dtype=np.float32)
        self._latents = np.empty((0, *self.latent_shape), dtype=np.float32)

    @property
    def projection(self) -> SparseTernaryProjection:
        return self._projection

    @property
    def provenance(self) -> CacheProvenance:
        return self._provenance

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self._latent_shape

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def size(self) -> int:
        return len(self._case_ids)

    @property
    def entry_case_ids(self) -> tuple[str, ...]:
        return self._case_ids

    def __len__(self) -> int:
        return self.size

    def _spec(self) -> dict[str, Any]:
        return {
            "projection": {
                "algorithm": PROJECTION_ALGORITHM,
                "config": asdict(self.projection.config),
                "nonzeros_per_row": self.projection.config.nonzeros_per_row,
            },
            "latent_shape": list(self.latent_shape),
            "threshold": self.threshold,
            "capacity": self.capacity,
            "provenance": self.provenance.to_dict(),
        }

    @property
    def provenance_sha256(self) -> str:
        return _sha256(_canonical(self._spec()))

    @property
    def cache_sha256(self) -> str:
        self._require_frozen()
        digest = hashlib.sha256(_canonical({
            "provenance_sha256": self.provenance_sha256,
            "entry_case_ids": self.entry_case_ids,
        }))
        for name, array in (("keys", self._keys), ("latents", self._latents)):
            digest.update(_canonical({"name": name, "shape": list(array.shape), "dtype": "<f4"}))
            digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def add(self, feature: Any, latent: Any, *, case_id: str) -> int:
        if self.is_frozen:
            raise RuntimeError("frozen cache cannot be modified")
        if self.size >= self.capacity:
            raise OverflowError("cache capacity exhausted; eviction and replacement are forbidden")
        if case_id not in self.provenance.development_case_ids:
            raise ValueError("case_id must be one of the declared development cases")
        key = self.projection.project(feature)
        if not np.any(key):
            raise ValueError("zero projected features cannot be cached")
        value = _array(latent, self.latent_shape, "latent", np.float32)
        index = self.size
        self._key_builder.append(key)
        self._latent_builder.append(value)
        self._case_ids = (*self._case_ids, case_id)
        return index

    def freeze(self) -> FrozenActionCache:
        if not self.is_frozen:
            if self.size:
                self._keys = np.stack(self._key_builder)
                self._latents = np.stack(self._latent_builder)
            self._keys.setflags(write=False)
            self._latents.setflags(write=False)
            self._key_builder.clear()
            self._latent_builder.clear()
            self._is_frozen = True
        return self

    def _require_frozen(self) -> None:
        if not self.is_frozen:
            raise RuntimeError("cache must be frozen before lookup or persistence")

    def lookup(self, feature: Any, threshold: float | None = None) -> CacheLookup:
        self._require_frozen()
        cutoff = self.threshold if threshold is None else _number(threshold, "threshold", -1.0, 1.0)
        key = self.projection.project(feature).astype(np.float64)
        if not self.size or not np.any(key):
            return CacheLookup(False, None, None, None)
        keys = self._keys.astype(np.float64)
        scores = (keys @ key) / (np.linalg.norm(keys, axis=1) * np.linalg.norm(key))
        scores = np.clip(scores, -1.0, 1.0)
        # Exact repeated keys have cosine 1 even if norm arithmetic rounds below it.
        scores[np.all(keys == key, axis=1)] = 1.0
        scores[np.all(keys == -key, axis=1)] = -1.0
        index = int(np.argmax(scores))  # First insertion wins an exact tie.
        similarity = float(scores[index])
        hit = similarity >= cutoff
        return CacheLookup(hit, similarity, index,
                           self._latents[index].copy() if hit else None, self._case_ids[index])

    @staticmethod
    def _paths(path_prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(path_prefix)
        return Path(f"{prefix}.npz"), Path(f"{prefix}.json")

    def save(self, path_prefix: str | Path) -> tuple[Path, Path]:
        """Write prefix.npz and prefix.json exclusively; never overwrite either."""
        self._require_frozen()
        npz_path, json_path = self._paths(path_prefix)
        if npz_path.exists() or json_path.exists():
            raise FileExistsError("cache output already exists")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, keys=self._keys, latents=self._latents,
                            positive_indices=self.projection._positive,
                            negative_indices=self.projection._negative)
        payload = buffer.getvalue()
        manifest = {
            "format": "feedbackbench.action_cache",
            "version": FORMAT_VERSION,
            "spec": self._spec(),
            "provenance_sha256": self.provenance_sha256,
            "cache_sha256": self.cache_sha256,
            "entry_count": self.size,
            "entry_case_ids": list(self.entry_case_ids),
            "npz_sha256": _sha256(payload),
        }
        manifest["manifest_sha256"] = _sha256(_canonical(manifest))
        json_bytes = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                                allow_nan=False).encode("utf-8") + b"\n"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            for path, content in ((npz_path, payload), (json_path, json_bytes)):
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return npz_path, json_path

    @classmethod
    def load(
        cls, path_prefix: str | Path, *, expected_provenance_sha256: str | None = None,
    ) -> FrozenActionCache:
        """Load only numeric NPZ data and verify manifest, payload, spec and contents.

        Digests detect corruption, not malicious replacement of every artifact.
        Pin expected_provenance_sha256 from the experiment's separate protocol.
        """
        npz_path, json_path = cls._paths(path_prefix)

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON manifest key")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant: {value}")

        manifest = json.loads(json_path.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object, parse_constant=reject_constant)
        expected_fields = {"format", "version", "spec", "provenance_sha256", "cache_sha256",
                           "entry_count", "entry_case_ids", "npz_sha256", "manifest_sha256"}
        if not isinstance(manifest, dict) or set(manifest) != expected_fields:
            raise ValueError("invalid cache manifest fields")
        recorded_manifest_hash = _digest(manifest.pop("manifest_sha256"), "manifest_sha256")
        if _sha256(_canonical(manifest)) != recorded_manifest_hash:
            raise ValueError("manifest SHA-256 mismatch")
        if manifest["format"] != "feedbackbench.action_cache" or type(manifest["version"]) is not int:
            raise ValueError("invalid cache format")
        if manifest["version"] != FORMAT_VERSION:
            raise ValueError("unsupported cache format version")
        spec = manifest["spec"]
        if not isinstance(spec, dict) or set(spec) != {
            "projection", "latent_shape", "threshold", "capacity", "provenance",
        }:
            raise ValueError("invalid cache specification")
        projection_spec = spec["projection"]
        if not isinstance(projection_spec, dict) or set(projection_spec) != {
            "algorithm", "config", "nonzeros_per_row",
        } or projection_spec["algorithm"] != PROJECTION_ALGORITHM:
            raise ValueError("unsupported projection contract")
        try:
            projection = SparseTernaryProjection(ProjectionConfig(**projection_spec["config"]))
            provenance = CacheProvenance(**spec["provenance"])
            result = cls(projection, spec["latent_shape"], provenance, spec["threshold"], spec["capacity"])
        except (TypeError, KeyError) as error:
            raise ValueError("invalid cache specification fields") from error
        if _canonical(result._spec()) != _canonical(spec):
            raise ValueError("cache specification is not canonical")
        if result.provenance_sha256 != _digest(manifest["provenance_sha256"], "provenance_sha256"):
            raise ValueError("provenance SHA-256 mismatch")
        if expected_provenance_sha256 is not None:
            if result.provenance_sha256 != _digest(expected_provenance_sha256, "expected_provenance_sha256"):
                raise ValueError("cache provenance does not match the experiment protocol")
        count = _integer(manifest["entry_count"], "entry_count", 0, result.capacity)
        cases = manifest["entry_case_ids"]
        if not isinstance(cases, list) or len(cases) != count or any(
            not isinstance(case, str) or case not in provenance.development_case_ids for case in cases
        ):
            raise ValueError("invalid cache development case identities")
        payload = npz_path.read_bytes()
        if _sha256(payload) != _digest(manifest["npz_sha256"], "npz_sha256"):
            raise ValueError("NPZ SHA-256 mismatch")
        expected_arrays = {"keys", "latents", "positive_indices", "negative_indices"}
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if len(archive.files) != 4 or set(archive.files) != expected_arrays:
                raise ValueError("invalid NPZ array fields")
            keys = archive["keys"]
            latents = archive["latents"]
            if keys.dtype != np.float32 or latents.dtype != np.float32:
                raise ValueError("cache keys and latents must be float32")
            result._keys = _array(keys, (count, projection.config.output_dim), "keys", np.float32)
            result._latents = _array(latents, (count, *result.latent_shape), "latents", np.float32)
            for name, expected in (("positive_indices", projection._positive),
                                   ("negative_indices", projection._negative)):
                saved = archive[name]
                if saved.dtype != np.int64 or not np.array_equal(saved, expected):
                    raise ValueError("saved projection does not match its deterministic configuration")
        norms = np.linalg.norm(result._keys.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-6):
            raise ValueError("cache keys must be nonzero L2-normalized vectors")
        result._case_ids = tuple(cases)
        result._keys.setflags(write=False)
        result._latents.setflags(write=False)
        result._is_frozen = True
        if result.cache_sha256 != _digest(manifest["cache_sha256"], "cache_sha256"):
            raise ValueError("cache content SHA-256 mismatch")
        return result
