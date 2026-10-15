"""Bounded INT8 loader tests; real tiny CPU kernel checks require explicit opt-in."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_vlajepa_int8.py"
SPEC = importlib.util.spec_from_file_location("vlajepa_int8_probe_tested", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def checkpoint(tmp_path, payload=b"abcdefgh"):
    path = tmp_path / "weights.safetensors"
    header = b"{}"
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return path


def test_owned_reader_reads_only_exact_requested_tensor(tmp_path):
    path = checkpoint(tmp_path)
    entry = {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}
    result = probe.read_tensor_bytes(path, entry, probe.checkpoint_layout(path))
    assert type(result) is bytearray
    assert result == b"cdef"
    result[0] = 1
    assert path.read_bytes().endswith(b"abcdefgh")


@pytest.mark.parametrize("entry", [
    {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]},
    {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [-2, 2]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [6, 10]},
])
def test_owned_reader_rejects_untrusted_metadata(tmp_path, entry):
    with pytest.raises(ValueError):
        probe.read_tensor_bytes(checkpoint(tmp_path), entry, 10)


@pytest.mark.parametrize("raw", [b"123", struct.pack("<Q", 2**40)])
def test_layout_rejects_truncated_or_unbounded_header(tmp_path, raw):
    path = tmp_path / "bad"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        probe.checkpoint_layout(path)


def test_autocast_override_accepts_only_explicit_cpu():
    with probe.cpu_fp32_context("cpu", "ignored-dtype"):
        pass
    with pytest.raises(ValueError, match="CPU-only"):
        probe.cpu_fp32_context("cuda")


def test_new_guard_and_inventory_are_separate_from_original():
    assert probe.MIN_FREE_GIB == 4.25
    assert probe.MAX_RSS_GIB == 5.0
    assert probe.MIN_RUNTIME_FREE_GIB == 0.5
    assert probe.EXPECTED_LINEAR_COUNT == 300
    assert probe.EXPECTED_LINEAR_ELEMENTS == 1_811_939_328
    assert probe.EXPECTED_PAYLOAD_BYTES == 3_072_709_660
    original = probe.BASE_PATH.read_text(encoding="utf-8")
    assert 'default=6.0' in original


def test_forward_return_is_persisted_before_any_validation_or_postprocessing(tmp_path):
    output = tmp_path / "probe.json"
    record = {"status": "running_single_quantized_forward", "forward_started": False,
              "inference_executed": False, "forward_calls": 0, "timings": {}}
    sentinel = object()

    def forward():
        before = json.loads(output.read_text(encoding="utf-8"))
        assert before["forward_started"] is True
        assert before["inference_executed"] is False
        return sentinel

    assert probe.invoke_recorded_forward(output, record, forward) is sentinel
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["inference_executed"] is True and saved["forward_calls"] == 1
    assert saved["status"] == "running_single_quantized_forward"
    assert saved["forward_started_at"] and saved["forward_returned_at"]
    assert saved["timings"]["forward_seconds"] >= 0


def test_exception_during_forward_preserves_attempt_without_claiming_return(tmp_path):
    output = tmp_path / "probe.json"
    record = {"status": "running_single_quantized_forward", "forward_started": False,
              "inference_executed": False, "forward_calls": 0, "timings": {}}

    def forward():
        raise RuntimeError("synthetic forward failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        probe.invoke_recorded_forward(output, record, forward)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["forward_started"] is True
    assert saved["inference_executed"] is False and saved["forward_calls"] == 0
    assert "forward_returned_at" not in saved
    assert record["timings"]["forward_attempt_seconds"] >= 0


def test_preflight_refusal_snapshots_both_sources_without_importing_torch(tmp_path, monkeypatch):
    fake = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1 * 1024**3))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(probe.base, "relative", lambda path: Path(path).name)
    monkeypatch.setattr(probe.base, "MemoryGuard", lambda *a: pytest.fail("Must refuse before starting monitor"))
    output = tmp_path / "probe.json"
    args = SimpleNamespace(output=output, threads=2, observation=tmp_path / "unused")
    before = "torch" in sys.modules
    assert probe.run(args) == 3
    assert ("torch" in sys.modules) == before
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "blocked_preflight"
    assert result["inference_executed"] is False
    assert result["preflight"]["required_available_bytes"] == 4.25 * 1024**3
    assert (tmp_path / result["script_snapshot"]).read_bytes() == SCRIPT.read_bytes()
    assert result["base_script"]["path"] == "scripts/probe_vlajepa.py"
    assert (tmp_path / result["base_script"]["snapshot_path"]).read_bytes() == probe.BASE_PATH.read_bytes()


@pytest.mark.parametrize("suffix", [".json", ".source.py", ".base_source.py", ".actions.npz", ".memory_stop.json"])
def test_refuses_overwriting_any_companion_evidence(tmp_path, monkeypatch, suffix):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace())
    existing = tmp_path / ("probe" + suffix)
    existing.write_bytes(b"retained")
    args = SimpleNamespace(output=tmp_path / "probe.json", threads=2, observation=tmp_path / "unused")
    with pytest.raises(FileExistsError):
        probe.run(args)
    assert existing.read_bytes() == b"retained"


@pytest.mark.skipif(os.environ.get("FEEDBACKBENCH_INT8_KERNEL_TEST") != "1",
                    reason="Explicit opt-in for tiny real Torch CPU kernel check")
def test_real_dynamic_linear_and_embedding_cast_contract():
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(7)
    linear = torch.nn.Linear(8, 5, device="meta")
    weight, bias = torch.randn(5, 8), torch.randn(5)
    quantized = probe.quantize_one_linear(linear, weight, bias, torch)
    table = torch.nn.Embedding(10, 8, dtype=torch.bfloat16)
    table.register_forward_hook(probe.cast_embedding_output_to_float)
    token_ids = torch.tensor([[1, 3, 2]])
    with torch.inference_mode(), probe.cpu_fp32_context("cpu"):
        values = table(token_ids)
        actual = quantized(values)
    assert values.dtype == torch.float32
    assert table.weight.dtype == torch.bfloat16
    assert actual.dtype == torch.float32 and actual.shape == (1, 3, 5)
    assert torch.isfinite(actual).all()
    assert quantized._packed_params.dtype == torch.qint8
    with pytest.raises(RuntimeError, match="Float.*BFloat16"):
        quantized(values.bfloat16())
    with pytest.raises(ValueError, match="FP32"):
        probe.quantize_one_linear(linear, weight.bfloat16(), bias, torch)


@pytest.mark.skipif(os.environ.get("FEEDBACKBENCH_INT8_KERNEL_TEST") != "1",
                    reason="Explicit opt-in for tiny owned tensor decoding")
def test_owned_tensor_decodes_bf16_and_preserves_values(tmp_path):
    import torch

    values = torch.tensor([1.0, -0.5, 2.0, 0.0], dtype=torch.bfloat16)
    payload = values.view(torch.uint8).numpy().tobytes()
    path = checkpoint(tmp_path, payload)
    entry = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}
    decoded = probe.read_tensor(path, entry, 10, torch.bfloat16, torch)
    assert torch.equal(decoded.flatten(), values)
    decoded32 = probe.read_tensor(path, entry, 10, torch.float32, torch)
    assert torch.equal(decoded32.flatten(), values.float())
