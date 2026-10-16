"""Tiny real-Torch storage tests; no checkpoint download or model/simulator load."""
from __future__ import annotations

from copy import deepcopy
import gc
import importlib.util
import mmap
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/vlajepa_paged_embedding.py"
SPEC = importlib.util.spec_from_file_location("vlajepa_paged_embedding_test", PATH)
paged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paged)


@pytest.fixture(scope="module")
def torch():
    import torch as library
    library.set_num_threads(2)
    return library


@pytest.fixture
def checkpoint(tmp_path, torch):
    value = torch.tensor([[0.0, -0.0, 0.5, -1.25], [2.0, 3.5, 17.0, -100.0]], dtype=torch.bfloat16)
    payload = value.view(torch.uint16).numpy().tobytes()
    data_start = 16
    start = mmap.ALLOCATIONGRANULARITY + 6
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", 8) + b"{}      " + bytes(start) + payload + b"trailer")
    return path, {"dtype": "BF16", "shape": [2, 4], "data_offsets": [start, start + len(payload)]}, data_start, value


def test_module_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c",
                             "import runpy,sys;runpy.run_path(sys.argv[1]);assert 'torch' not in sys.modules",
                             str(PATH)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_exact_bf16_bits_full_table_lookup_and_audit(checkpoint, torch):
    path, entry, start, expected = checkpoint
    actual, audit = paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch, chunk_elements=3)
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    ids = torch.tensor([1, 0, 1])
    assert torch.equal(torch.nn.functional.embedding(ids, actual), expected[ids])
    assert actual.shape == expected.shape and not actual.requires_grad
    assert audit["runtime_mapping_count"] == 1
    assert audit["validation_mapping_count"] == 3
    assert audit["validation_mappings_closed_before_runtime_mapping"] is True
    assert audit["mapping_file_range"][0] % mmap.ALLOCATIONGRANULARITY == 0
    assert audit["mapping_file_range"][1] == start + entry["data_offsets"][1]
    assert audit["theoretical_owned_allocation_avoided_bytes"] == 16
    assert audit["measured_resident_saving_bytes"] is None
    with pytest.raises(IndexError):
        torch.nn.functional.embedding(torch.tensor([2]), actual)


def test_tied_parameter_survives_context_and_private_write_keeps_file_unchanged(checkpoint, torch):
    path, entry, start, expected = checkpoint
    original_bytes = path.read_bytes()
    original_reader = lambda *args: None
    quant = SimpleNamespace(read_tensor=original_reader, checkpoint_layout=lambda path: start)
    with paged.paged_embedding_loader(quant, path, entry) as audit:
        tensor = quant.read_tensor(path, deepcopy(entry), start, torch.bfloat16, torch)
        embedding = torch.nn.Embedding(2, 4, device="meta", dtype=torch.bfloat16)
        embedding.weight = torch.nn.Parameter(tensor, requires_grad=False)
        head = torch.nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
        head.weight = embedding.weight
        del tensor
    gc.collect()
    assert quant.read_tensor is original_reader
    assert head.weight is embedding.weight
    assert head.weight.data_ptr() == embedding.weight.data_ptr()
    assert torch.equal(head(torch.ones(1, 4, dtype=torch.bfloat16)), expected.sum(1).reshape(1, 2))
    with torch.no_grad():
        embedding.weight[1, 3] = 9
    assert head.weight[1, 3].item() == 9
    assert path.read_bytes() == original_bytes
    assert audit["runtime_mapping_count"] == 1


def test_validation_maps_close_and_runtime_owner_lives_until_last_parameter(checkpoint, torch, monkeypatch):
    path, entry, start, _ = checkpoint
    factory = mmap.mmap
    created = []

    def track(*args, **kwargs):
        mapping = factory(*args, **kwargs)
        created.append(weakref.ref(mapping))
        return mapping

    monkeypatch.setattr(paged.mmap, "mmap", track)
    tensor, audit = paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch, chunk_elements=3)
    assert len(created) == audit["validation_mapping_count"] + 1
    assert all(ref() is None or ref().closed for ref in created[:-1])
    runtime_ref = created[-1]
    parameter = torch.nn.Parameter(tensor, requires_grad=False)
    del tensor
    gc.collect()
    assert runtime_ref() is not None and not runtime_ref().closed
    assert parameter[0, 2].item() == 0.5
    del parameter
    gc.collect()
    assert runtime_ref() is None


@pytest.mark.parametrize("change", [
    {"dtype": "F32"}, {"shape": [8]}, {"shape": [True, 8]}, {"shape": [0, 4]},
    {"shape": [2.0, 4]}, {"shape": [2, 5]}, {"data_offsets": [-2, 14]},
    {"data_offsets": [1, 17]}, {"data_offsets": [0, True]}, {"data_offsets": [0]},
    {"data_offsets": [10**10, 10**10 + 16]},
])
def test_invalid_metadata_is_rejected_before_mapping(checkpoint, torch, monkeypatch, change):
    path, entry, start, _ = checkpoint
    entry.update(change)
    monkeypatch.setattr(paged.mmap, "mmap", lambda *args, **kwargs: pytest.fail("Invalid metadata reached mmap"))
    with pytest.raises(ValueError):
        paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch)


@pytest.mark.parametrize("start", [True, -8, 15, 24])
def test_data_start_alignment_and_header_consistency(checkpoint, torch, start):
    path, entry, _, _ = checkpoint
    with pytest.raises(ValueError):
        paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.0, paged.DEFAULT_CHUNK_ELEMENTS + 1])
def test_invalid_chunk_bound(checkpoint, torch, chunk):
    path, entry, start, _ = checkpoint
    with pytest.raises(ValueError):
        paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch, chunk_elements=chunk)


@pytest.mark.parametrize("bits", [0x7FC0, 0x7F80, 0xFF80])
def test_nonfinite_values_in_last_chunk_rejected_without_runtime_map(checkpoint, torch, monkeypatch, bits):
    path, entry, start, _ = checkpoint
    payload = bytearray(path.read_bytes())
    last = start + entry["data_offsets"][1] - 2
    payload[last:last + 2] = struct.pack("<H", bits)
    path.write_bytes(payload)
    factory = mmap.mmap
    created = []

    def track(*args, **kwargs):
        mapping = factory(*args, **kwargs)
        created.append(mapping)
        return mapping

    monkeypatch.setattr(paged.mmap, "mmap", track)
    with pytest.raises(ValueError, match="Nonfinite"):
        paged.map_embedding_tensor(path, entry, start, torch.bfloat16, torch, chunk_elements=3)
    assert len(created) == 3
    assert all(mapping.closed for mapping in created)


def test_other_tensors_use_original_reader_and_patch_restores_on_error(checkpoint, torch):
    path, entry, start, _ = checkpoint
    calls = []
    sentinel = object()

    def original(*args):
        calls.append(args)
        return sentinel

    quant = SimpleNamespace(read_tensor=original, checkpoint_layout=lambda path: start)
    other = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    with pytest.raises(RuntimeError, match="controlled"):
        with paged.paged_embedding_loader(quant, path, entry) as audit:
            assert quant.read_tensor(path, other, start, torch.float32, torch) is sentinel
            raise RuntimeError("controlled")
    assert quant.read_tensor is original
    assert calls == [(path, other, start, torch.float32, torch)]
    assert audit["runtime_mapping_count"] == 0 and audit["fallback_tensor_reads"] == 1


def test_second_embedding_mapping_and_dtype_conversion_are_rejected(checkpoint, torch):
    path, entry, start, _ = checkpoint
    quant = SimpleNamespace(read_tensor=lambda *args: None, checkpoint_layout=lambda path: start)
    with paged.paged_embedding_loader(quant, path, entry) as audit:
        tensor = quant.read_tensor(path, entry, start, torch.bfloat16, torch)
        with pytest.raises(ValueError, match="only once"):
            quant.read_tensor(path, entry, start, torch.bfloat16, torch)
        assert audit["runtime_mapping_count"] == 1 and tensor.numel() == 8
    with pytest.raises(ValueError, match="BF16 storage"):
        paged.map_embedding_tensor(path, entry, start, torch.float32, torch)


def test_changed_target_metadata_does_not_silently_fallback(checkpoint, torch):
    path, entry, start, _ = checkpoint
    quant = SimpleNamespace(read_tensor=lambda *args: pytest.fail("Must not fall back"),
                            checkpoint_layout=lambda path: start)
    with paged.paged_embedding_loader(quant, path, entry):
        changed = deepcopy(entry)
        changed["shape"] = [1, 8]
        with pytest.raises(ValueError, match="specified checkpoint metadata"):
            quant.read_tensor(path, changed, start, torch.bfloat16, torch)
