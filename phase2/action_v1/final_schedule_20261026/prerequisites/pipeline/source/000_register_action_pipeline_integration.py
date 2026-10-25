"""Freeze a two-update trainer integration check, separate from all study results."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from action_campaign_common import digest, write_json, read, now, DEADLINE
from train_action_adaptation import REQUIRED_SOURCES
import h5py

output = ROOT / "outputs/action_adaptation/pipeline_check_v1"
output.mkdir(parents=True, exist_ok=False)
alignment = ROOT / ".runtime/action_alignment_spatial_r3.json"
assert read(alignment)["status"] == "passed"
shutil.copyfile(alignment, output / "alignment.json")
source = read(ROOT / "data/action_adaptation/probe_v1/source_metadata.json")
data_path = "data/action_adaptation/probe_v1/task0.hdf5"
assert digest(ROOT / data_path) == source["sha256"]
demos = []
with h5py.File(ROOT / data_path, "r") as dataset:
    for index, split in ((0, "train"), (1, "validation")):
        demos.append({"id": f"spatial_demo_{index}", "split": split, "task": "libero_spatial:0",
                      "language": "pick up the black bowl between the plate and the ramekin and place it on the plate",
                      "path": data_path, "sha256": source["sha256"], "episode_key": f"demo_{index}",
                      "frame_count": len(dataset[f"data/demo_{index}/actions"])})
write_json(output / "data_manifest.json", {"demos": demos, "development_only": True,
    "note": "demo_1 serves only as integration validation, then belongs to production training; it is never final validation/test."})
write_json(output / "schedule.json", {"train": [{"demo_id": "spatial_demo_0", "observation_index": frame} for frame in (0, 10)],
    "validation": [{"demo_id": "spatial_demo_1", "observation_index": 0, "noise_seed": 1000, "time": .5}]})
def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
models = {}
for name in ("smolvla", "smolvlm_assets"):
    path = ROOT / "models" / name / "download_manifest.json"
    models[path.relative_to(ROOT).as_posix()] = digest(path)
    for item in read(path)["files"]:
        models[(path.parent / item["file"]).relative_to(ROOT).as_posix()] = item["sha256"]
sources = REQUIRED_SOURCES | {"scripts/action_campaign_common.py", "scripts/run_action_campaign.py",
    "scripts/report_action_campaign.py", "scripts/run_action_adaptation_probe.py",
    "scripts/budget_campaign_common.py", "scripts/budget_campaign_common_v5.py",
    "scripts/validate_action_demo_alignment.py", "scripts/validate_action_demo_alignment_v2.py",
    ".runtime/register_action_pipeline_integration.py", "tests/test_action_adaptation_train.py"}
source_hashes = {name: digest(ROOT / name) for name in sorted(sources)}
snapshot = output / "source_snapshot"
snapshot.mkdir()
for index, name in enumerate(source_hashes):
    shutil.copyfile(ROOT / name, snapshot / f"{index:03d}_{Path(name).name}")
write_json(snapshot / "index.json", {name: f"{index:03d}_{Path(name).name}" for index, name in enumerate(source_hashes)})
protocol = {"schema": "feedbackbench.action_adaptation_train.v1", "frozen": True, "frozen_at": now(),
    "deadline_utc": DEADLINE, "development_only": True, "study_result": False,
    "purpose": "Test both real native training families and complete checkpoint/optimizer/RNG/report wiring; not an efficacy result",
    "source_files": source_hashes, "model": {"models_dir": "models", "files": models, "num_steps": 10},
    "data": {"manifest": record(output / "data_manifest.json"), "alignment_evidence": record(output / "alignment.json"),
             "action_offset": 1, "image_transform": "rot180_once"},
    "flow_time_distribution": "beta_1.5_1_scaled_0.999_plus_0.001",
    "optimizer": {"type": "AdamW", "lr": .0001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
    "resources": {"torch_threads": 2, "interop_threads": 1, "max_seconds": 600, "evaluation_max_seconds": 7200},
    "checkpoint_every": 1, "validation_every": 2,
    "runs": {f"{mode}_s17": {"mode": mode, "initialization_seed": 17, "flow_seed": 17,
              "steps": 2, "schedule": record(output / "schedule.json")} for mode in ("lora", "output_head")},
    "evaluation": {"methods": [], "cases": []}}
write_json(output / "protocol.json", protocol)
(output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n")
print(json.dumps({"output": str(output), "protocol_sha256": digest(output / "protocol.json"), "development_only": True}))
