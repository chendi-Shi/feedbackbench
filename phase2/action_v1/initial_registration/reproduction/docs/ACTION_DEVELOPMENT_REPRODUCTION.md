# Public development reproduction, source version 1

This entrypoint reconstructs **new local development evidence**: two actual optimizer updates for each of LoRA and output-head adaptation, then one native ten-NFE LIBERO episode using the new LoRA checkpoint. It replaces the private registration/launch helpers. It does not reproduce the numerical outcome of the archived run, resume its optimizer, or add development episodes to the formal 280-case denominator.

The tools are separate from the already frozen formal protocol. Their release hashes are in `action_development_reproduction_sources.json`. No existing protocol or result directory is rewritten. Every registration needs a new short campaign name.

**This version stops at 2026-10-30 06:08 UTC.** After that time, registration, training and evaluation refuse to proceed. Running a later reproduction requires a separately reviewed new protocol and source version. Changing the computer clock or editing an old frozen deadline is not a supported reproduction procedure.

## Prerequisites and explicit downloads

Run commands from a complete project checkout. Follow the main README's Python 3.12/bootstrap instructions and retain the pinned vendor checkout. The native simulator needs working OpenGL. Downloads are explicit and use the existing pinned acquisition scripts:

```powershell
python scripts/bootstrap.py
python scripts/run.py scripts/download_models.py --only smolvla --only smolvlm_assets
python scripts/run.py scripts/download_assets.py
python scripts/run.py scripts/prepare_action_study_sources.py --download
```

The expert download is about 3.46 GB across four fixed files. Existing partial ranges require the acquisition tool's explicit `--resume`; do not delete its audit or lock to obtain a fresh budget. `reproduce_action_development.py` does not download anything. Its read-only plan is:

```powershell
python scripts/run.py scripts/reproduce_action_development.py plan
```

Model/preprocessor/tokenizer manifests and every model file are checked at registration/loading. This example uses the project-local `models/`, official task-0 expert sources and their unmodified seven-dimensional controller actions. It does not replace the pretrained VLA or alter simulator dynamics.

## Reconstruct the diagnostic history locally

The archived public alignment projection is not a raw gate receipt. Generate new raw reports under a new directory, retaining all three stages. The following commands perform real simulator diagnostics, so run them serially with no training worker active:

```powershell
python scripts/run.py scripts/validate_action_demo_alignment.py --source-manifest data/action_adaptation/study_v1/source_manifest.json --suite libero_spatial --output outputs/action_adaptation/repro_align_001/r1.json
python scripts/run.py scripts/validate_action_demo_alignment_r2.py --source-manifest data/action_adaptation/study_v1/source_manifest.json --suite libero_spatial --v1-receipt outputs/action_adaptation/repro_align_001/r1.json --output outputs/action_adaptation/repro_align_001/r2.json
python scripts/run.py scripts/validate_action_demo_alignment_v2.py --source-manifest data/action_adaptation/study_v1/source_manifest.json --suite libero_spatial --v1-receipt outputs/action_adaptation/repro_align_001/r1.json --prior-observation-receipt outputs/action_adaptation/repro_align_001/r2.json --output outputs/action_adaptation/repro_align_001/r3.json
```

Run each command separately. The first two diagnostic implementations historically returned exit 1 / `needs_attention`; retain their full reports and inspect their failure reasons before running the next revision. Do not interpret arbitrary execution errors or missing output as that expected diagnostic result. The new `validate_action_demo_alignment_r2.py` is an exact byte copy of the preserved r2 source, SHA256 `be836f84507b66a0b7ea72787c8980e9ffefe8600d95bd589fabd71db7398edf`. The existing filename ending `_v2.py` contains diagnostic revision 3; its report explicitly says `diagnostic_revision: 3`.

R3 verifies all 50 demonstrations' state/action indexing and five recorded-observation reconstruction samples. It accounts for the original sensor timing. It explicitly does **not** verify full dynamics replay: flattened MuJoCo state omits controller/gripper hidden state. Local source hashes, original failed diagnostic reports and their reference chain are retained. No failed report is rewritten as passed.

The two-update development registration below needs the spatial raw r3 receipt. For a separate formal study, repeat the diagnostic sequence for the other three suites and use `combine_action_demo_alignment.py` to bind all four complete raw r3 reports; the formal builder retains its existing four-source gate.

## Explicitly register and run the two tiny training pipelines

Registration hashes local inputs and saves an immutable source snapshot. It does not load the model:

```powershell
python scripts/run.py scripts/reproduce_action_development.py register-training --campaign-id repro_train_001 --source-manifest data/action_adaptation/study_v1/source_manifest.json --alignment outputs/action_adaptation/repro_align_001/r3.json
python scripts/run.py scripts/reproduce_action_development.py train --campaign-id repro_train_001
```

The second command starts real compute. It uses the original native `policy.forward`, batch size 1 and 50-by-7 action chunks; the original loss excludes padded timesteps and padded action dimensions. Both families use seed 17 and two AdamW steps at the fixed development settings. LoRA and output-head workers run serially. Validation here is a development integration check, not a held-out efficacy result. Complete adapter/optimizer/RNG/state/history checkpoints remain local, with full receipts.

The supervisor requires 4.25 GiB available RAM to start, 0.75 GiB reserve while running, and 4 GiB free disk; it uses two Torch threads and verifies PID plus creation time before terminating its own children. It never stops another project. A successful current supervisor archives only its own completed lock after verifying that its children are gone. Failures keep their lock and evidence. An explicitly requested training continuation uses `train --campaign-id repro_train_001 --resume`, retaining the original protocol, exact latest checkpoint and original runtime allowance; it is not a way to repeat completed evaluations.

## Register and run exactly one separate development episode

After both training pipelines complete:

```powershell
python scripts/run.py scripts/reproduce_action_development.py register-evaluation --training-campaign-id repro_train_001 --campaign-id repro_eval_001
python scripts/run.py scripts/reproduce_action_development.py evaluate --campaign-id repro_eval_001
```

The evaluation registration verifies the historical training protocol, its source snapshot, both complete checkpoint receipts and their base fingerprints. The new evaluation protocol separately binds the current public evaluator and supervisor. Production `evaluate_action_adaptation.py` continues to reject cross-protocol trained manifests; the explicit dual-protocol bridge exists only in the separate public development evaluator.

The case is fixed to `libero_spatial`, task 0, init 0, clean images, `lora_s17`, ten actual velocity evaluations and ten executed actions per replan. It uses the full official 280-step horizon, or earlier official termination. Success is **not** a completion gate. A complete failure is sealed and never retried. Any incomplete development attempt requires diagnosis; this entrypoint does not automatically rerun it.

The complete result is under:

```text
outputs/action_adaptation/repro_eval_001/evaluation/libero_spatial_t0_i0_clean_lora_s17/attempt_001/
```

`report.json`, `metadata.json`, `development_binding.json` and the final `receipt.json` bind both protocol hashes, the original trained manifest, and `excluded_from_formal_study: true`. The full local result also retains actions, states, calls, initial/final images and resource evidence. A public projection that omits private resource/environment data cannot pass this complete local seal, and must not be described as the original seal.

## Limits of this addition

The new CLI and its local filesystem/supervision tests use no real model, simulator or network during verification. The separately archived original development runs supply real-model integration evidence; they are not claimed as executions of this later public CLI. Exact timing, pixels, numerical replay and task success on another machine are not guaranteed.

The official 2,000-step, six-run research protocol and its 280 evaluations remain a separate workflow. Its existing raw three-step feasibility-probe, four-source alignment and development gates are unchanged. These public commands rebuild the two-update training and one-episode prerequisites; they neither bypass the remaining formal gates nor claim to restore omitted historical optimizer/RNG state from a public snapshot.
