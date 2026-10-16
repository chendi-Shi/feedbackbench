"""Lightweight paging protocol regression tests; no model or Torch execution."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paged_runner_tested", ROOT / "scripts/run_vlajepa_paged.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@pytest.mark.parametrize("suffix", ["", "outputs", "outputs/frontier",
                                     "outputs/frontier/vlajepa_closed_loop",
                                     "outputs/frontier/vlajepa_closed_loop/runs/new.json"])
def test_separate_output_rejects_owned_tree_and_ancestors(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="separate"):
        entry.separate_output(tmp_path / suffix)


def test_separate_output_rejects_external_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(entry, "ROOT", project)
    with pytest.raises(ValueError, match="separate"):
        entry.separate_output(tmp_path / "elsewhere")


def test_separate_output_accepts_paged_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    target = tmp_path / "outputs/frontier/vlajepa_paged_closed_loop"
    assert entry.separate_output(target) == target.resolve()


def test_configured_runner_binds_probe_and_all_sources_without_editing_owned_sources(monkeypatch):
    original = entry.original_runner()
    original_sources = tuple(original.SOURCES)
    before = {name: (ROOT / name).read_bytes() for name in original_sources}
    proof = {"path": "artifacts/synthetic_paging_probe.json", "sha256": "a" * 64,
             "plan_sha256": "b" * 64}
    monkeypatch.setattr(entry, "verify_probe", lambda *args: deepcopy(proof))
    configured = entry.configured_runner()
    spec = configured.study_spec()
    assert spec["paired_probe"] == proof
    assert spec["variant"] == entry.VARIANT
    assert spec["memory_guard"] == {"minimum_available_gib": 3.5, "maximum_rss_gib": 5.5,
                                     "minimum_runtime_available_gib": 0.5, "sampling_seconds": 1}
    assert set(configured.SOURCES) == set(original_sources) | set(entry.EXTRA_SOURCES)
    assert len(configured.SOURCES) == len(set(configured.SOURCES))
    assert spec["embedding_storage"] == {"dtype": "BF16", "shape": [151936, 2048],
                                          "bytes": 622329856, "mapping": "copy_on_write",
                                          "lookup_output": "FP32", "lm_head_tied": True}
    cases = spec["cases"]
    assert [case["suite"] for case in cases] == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    assert len({case["id"] for case in cases}) == 4
    for case in cases:
        assert case["model"] == "vlajepa_paged"
        assert (case["task_id"], case["init_state_id"], case["seed"], case["num_steps"]) == (1, 41, 1000, 4)
        assert case["scheduler"] == {"mode": "fixed", "fixed_interval": 7, "temporal_ensemble": False}
    assert original.MIN_FREE_GIB == 5.0
    assert original.VARIANT == "cpu_qwen_dynamic_int8_bf16_embedding_fp32_action"
    assert tuple(original.SOURCES) == original_sources
    assert all((ROOT / name).read_bytes() == payload for name, payload in before.items())


def paging_audit():
    import mmap
    payload = 622329856
    start = 932910598
    aligned = start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    return {"runtime_mapping_count": 1, "embedding_read_attempts": 1,
            "fallback_tensor_reads": 844, "embedding_materialized": True,
            "expected_runtime_mapping_count_when_loaded": 1,
            "shape": [151936, 2048], "dtype": "BF16", "elements": 311164928,
            "payload_bytes": payload, "data_start": 216056,
            "checkpoint_data_offsets": [932694542, 1555024398],
            "tensor_file_range": [start, start + payload],
            "mapping_file_range": [aligned, start + payload],
            "mapping_length_bytes": start + payload - aligned, "tensor_offset_in_mapping": start - aligned,
            "allocation_granularity_bytes": mmap.ALLOCATIONGRANULARITY,
            "access": "mmap.ACCESS_COPY", "validation_mapping_count": 312,
            "validation_chunk_elements": 1000000,
            "max_validation_mapping_bytes": 2000000 + mmap.ALLOCATIONGRANULARITY - 1,
            "all_embedding_values_finite": True,
            "validation_mappings_closed_before_runtime_mapping": True,
            "storage_lifetime": "torch.frombuffer storage retains mmap owner",
            "complete_vocabulary_preserved": True, "dtype_conversion": False,
            "theoretical_owned_allocation_avoided_bytes": payload,
            "theoretical_resident_saving_upper_bound_bytes": payload,
            "measured_resident_saving_bytes": None,
            "checkpoint_mutation_policy": "immutable while mapped; private tensor writes do not update checkpoint"}


def test_complete_paging_audit_is_accepted():
    entry.validate_paging(paging_audit())


@pytest.mark.parametrize(("field", "value"), [
    ("runtime_mapping_count", True),
    ("runtime_mapping_count", 1.0),
    ("runtime_mapping_count", 0),
    ("embedding_read_attempts", 2),
    ("embedding_materialized", False),
    ("access", "owned bytearray"),
    ("shape", [151936, 1024]),
    ("dtype", "F32"),
    ("payload_bytes", 1),
    ("all_embedding_values_finite", False),
    ("validation_mappings_closed_before_runtime_mapping", False),
    ("complete_vocabulary_preserved", False),
    ("dtype_conversion", True),
    ("tensor_offset_in_mapping", 9),
    ("mapping_length_bytes", 8),
    ("validation_mapping_count", 1),
    ("validation_chunk_elements", 1000001),
    ("measured_resident_saving_bytes", 622329856),
])
def test_incomplete_or_nonpaged_audit_is_rejected(field, value):
    audit = paging_audit()
    audit[field] = value
    with pytest.raises(ValueError):
        entry.validate_paging(audit)
