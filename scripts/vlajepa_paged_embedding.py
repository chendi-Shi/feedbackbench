"""Copy-on-write paging of the unchanged, complete BF16 VLA-JEPA token table.

Importing this module does not import Torch. The existing quantized loader still
assigns the Parameter, verifies checkpoint equality, and ties the language head.
This helper changes storage ownership only; it does not select vocabulary rows,
change precision, or promise a measured reduction in RSS. Use only for inference.
The checkpoint must remain immutable for the lifetime of the returned tensors.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import mmap
import os
from pathlib import Path
import stat
import struct
import sys
from unittest.mock import patch


DEFAULT_CHUNK_ELEMENTS = 1_000_000


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _layout(entry, data_start, file_size):
    """Validate metadata before exposing any bytes as a tensor."""
    if sys.byteorder != "little":
        raise ValueError("Safetensors BF16 paging requires a little-endian host")
    if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
        raise ValueError("Only the original BF16 embedding may be paged")
    shape = entry.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("Embedding shape must be the complete two-dimensional table")
    for dimension in shape:
        _positive_integer(dimension, "Embedding dimension")
    offsets = entry.get("data_offsets")
    if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2
            or any(type(value) is not int for value in offsets)):
        raise ValueError("Embedding offsets must be two integers")
    start, end = offsets
    if type(data_start) is not int or data_start < 8 or data_start % 8:
        raise ValueError("Safetensors data start must be aligned to eight bytes")
    payload_bytes = math.prod(shape) * 2
    if start < 0 or start % 2 or end % 2 or end - start != payload_bytes:
        raise ValueError("Embedding offsets, BF16 alignment, or shape disagree")
    absolute_start, absolute_end = data_start + start, data_start + end
    if absolute_end > file_size or absolute_end > sys.maxsize:
        raise ValueError("Embedding extends beyond the checkpoint or address space")
    mapping_start = absolute_start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    return {"shape": list(shape), "dtype": "BF16", "elements": math.prod(shape),
            "payload_bytes": payload_bytes, "data_start": data_start,
            "checkpoint_data_offsets": [start, end],
            "tensor_file_range": [absolute_start, absolute_end],
            "mapping_file_range": [mapping_start, absolute_end],
            "mapping_length_bytes": absolute_end - mapping_start,
            "tensor_offset_in_mapping": absolute_start - mapping_start,
            "allocation_granularity_bytes": mmap.ALLOCATIONGRANULARITY}


def _file_signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _chunk_is_finite(mapping, offset, elements, torch):
    # Returning a bool releases all temporary tensor references before close().
    tensor = torch.frombuffer(mapping, dtype=torch.bfloat16, count=elements, offset=offset)
    return bool(torch.isfinite(tensor).all())


def map_embedding_tensor(checkpoint, entry, data_start, dtype, torch, *,
                         chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    """Return (full BF16 tensor, portable audit), keeping its mmap alive in storage.

    Validation uses bounded temporary mappings, closed before the fresh runtime
    mapping is created. No mapping is forcibly closed while a returned Tensor or
    Parameter can reference it: torch.frombuffer retains the buffer owner.
    """
    _positive_integer(chunk_elements, "Validation chunk size")
    if chunk_elements > DEFAULT_CHUNK_ELEMENTS:
        raise ValueError("Validation chunk size exceeds the bounded one-million-element limit")
    if dtype != torch.bfloat16:
        raise ValueError("Paging must preserve BF16 storage without dtype conversion")
    with Path(checkpoint).open("rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Checkpoint must be a regular file")
        layout = _layout(entry, data_start, info.st_size)
        header = handle.read(8)
        if len(header) != 8 or struct.unpack("<Q", header)[0] + 8 != data_start:
            raise ValueError("Supplied data start disagrees with checkpoint header")
        validation_mappings = 0
        max_validation_mapping_bytes = 0
        absolute_start = layout["tensor_file_range"][0]
        for first in range(0, layout["elements"], chunk_elements):
            count = min(chunk_elements, layout["elements"] - first)
            chunk_start = absolute_start + first * 2
            aligned = chunk_start // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            length = chunk_start - aligned + count * 2
            validation = mmap.mmap(handle.fileno(), length, access=mmap.ACCESS_COPY, offset=aligned)
            # If Torch itself raises, ownership unwinds naturally; closing an mmap
            # still referenced by a traceback's tensor would be unsafe.
            finite = _chunk_is_finite(validation, chunk_start - aligned, count, torch)
            validation.close()
            validation_mappings += 1
            max_validation_mapping_bytes = max(max_validation_mapping_bytes, length)
            if not finite:
                raise ValueError("Nonfinite checkpoint embedding")
        if _file_signature(os.fstat(handle.fileno())) != _file_signature(info):
            raise ValueError("Checkpoint changed during embedding validation")
        runtime = mmap.mmap(handle.fileno(), layout["mapping_length_bytes"],
                            access=mmap.ACCESS_COPY, offset=layout["mapping_file_range"][0])
        tensor = torch.frombuffer(runtime, dtype=torch.bfloat16, count=layout["elements"],
                                  offset=layout["tensor_offset_in_mapping"]).reshape(layout["shape"])
        # Deliberately do not close runtime here or at the patch-context exit.
        # Tensor storage retains it, including when wrapped in nn.Parameter.
    audit = {**layout, "access": "mmap.ACCESS_COPY", "runtime_mapping_count": 1,
             "validation_mapping_count": validation_mappings,
             "validation_chunk_elements": chunk_elements,
             "max_validation_mapping_bytes": max_validation_mapping_bytes,
             "all_embedding_values_finite": True,
             "validation_mappings_closed_before_runtime_mapping": True,
             "storage_lifetime": "torch.frombuffer storage retains mmap owner",
             "complete_vocabulary_preserved": True, "dtype_conversion": False,
             "theoretical_owned_allocation_avoided_bytes": layout["payload_bytes"],
             "theoretical_resident_saving_upper_bound_bytes": layout["payload_bytes"],
             "measured_resident_saving_bytes": None,
             "resident_saving_caveat": "OS paging and accessed rows determine RSS; no saving is guaranteed",
             "checkpoint_mutation_policy": "immutable while mapped; private tensor writes do not update checkpoint"}
    return tensor, audit


@contextmanager
def paged_embedding_loader(quant, checkpoint, embedding_entry, *,
                           chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    """Scope a read_tensor patch around the unchanged quant.run or loader.

    Only the supplied embedding entry can use paging. Every other tensor follows
    the original reader. Exactly one successful mapping is permitted; a blocked
    preflight may leave count zero, which callers must not label as loaded.
    This process-local patch is intended for a single serial model load.
    """
    checkpoint = Path(checkpoint).resolve()
    expected = deepcopy(embedding_entry)
    data_start = quant.checkpoint_layout(checkpoint)
    _layout(expected, data_start, checkpoint.stat().st_size)
    _positive_integer(chunk_elements, "Validation chunk size")
    if chunk_elements > DEFAULT_CHUNK_ELEMENTS:
        raise ValueError("Validation chunk size exceeds the bounded one-million-element limit")
    original = quant.read_tensor
    audit = {"runtime_mapping_count": 0, "embedding_read_attempts": 0,
             "fallback_tensor_reads": 0, "embedding_materialized": False,
             "expected_runtime_mapping_count_when_loaded": 1}

    def read_tensor(path, entry, actual_data_start, dtype, torch):
        same_offsets = (isinstance(entry, dict)
                        and entry.get("data_offsets") == expected["data_offsets"])
        if entry == expected or (Path(path).resolve() == checkpoint and same_offsets):
            if Path(path).resolve() != checkpoint or entry != expected or actual_data_start != data_start:
                raise ValueError("Embedding read differs from the specified checkpoint metadata")
            audit["embedding_read_attempts"] += 1
            if audit["embedding_read_attempts"] != 1:
                raise ValueError("The specified embedding may be mapped only once per load")
            tensor, mapped = map_embedding_tensor(path, entry, actual_data_start, dtype, torch,
                                                   chunk_elements=chunk_elements)
            audit.update(mapped, embedding_materialized=True)
            return tensor
        audit["fallback_tensor_reads"] += 1
        return original(path, entry, actual_data_start, dtype, torch)

    with patch.object(quant, "read_tensor", read_tensor):
        yield audit
