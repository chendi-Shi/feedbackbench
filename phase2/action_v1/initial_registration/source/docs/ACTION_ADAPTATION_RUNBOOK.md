# Expert-supervised SmolVLA action adaptation

This second study is independent of the completed `learned_compute/v5` budget-controller study. Its purpose is to test whether updating the actual action expert with human demonstration targets improves closed-loop task execution. Old results, source snapshots, trained budget controllers, and publication receipts remain unchanged.

The original deadline is **2026-10-30 06:08 UTC / 14:08 China time**. Use this CPU computer only, one model worker at a time, no cloud cost, and no interference with other projects. The startup free-memory threshold is 4.25 GiB, the runtime reserve 0.75 GiB, and the free-disk reserve 4 GiB. A CPU call can be interrupted only through the supervisor's verified owned process identities. Do not lower resource limits to obtain a passing result.

## Development gates

The three-update probe in `plans/action_adaptation_probe_v1.json` is a feasibility diagnostic, not the retraining result. It checks actual backward gradients, frozen base checksums, action changes at identical noise, and exact adapter save/reload on the real checkpoint. Its one example and fixed flow noise cannot establish task improvement. Both optional cache parity and native-training results are retained; optimization must not replace the original loss without validated parity.

Before production training, validate the released demonstration observation/action timing with simulator replay. The provisional development example uses a one-row action offset because the reviewed official generation code saves post-action observations. The release's exact generation commit is unknown; published data and replay, rather than that hypothesis alone, decide the production alignment. All camera orientation, state conversion, action units, and checkpoint normalization must be bound in the new protocol.

The alignment r3 check has a narrow scope: observation indexing and reconstruction. The earlier one-step dynamics replay failed because flattened MuJoCo states do not contain the gripper controller's hidden accumulator; that failure and its unchanged thresholds are retained. Released joint/finger observations match the next stored physical state exactly. Reconstructing EEF poses accounts for `mj_step`'s documented cached derived quantities from the preceding physics step, using the XML's fixed timestep; it does not fit a time offset or change the data. A passed r3 report does **not** establish complete dynamics trajectory replay. All four source hashes and their individual alignment checks must pass before production fitting.

## Planned comparisons

The prospective study uses task 0 from each of LIBERO Spatial, Object, Goal, and Long. Tasks are chosen by official identity before new performance measurements. The pinned foundation checkpoint's original training overlap is unknown, and the previous project has already used LIBERO. This is a small, same-domain adaptation study, not a claim of unseen-task generalization, real-robot deployment, or state of the art.

Whole-demo separation prevents training/validation frame leakage. It does not prove that the prospective evaluation initialization distribution is independent of released demonstration generation. The HDF5 files retain physical initial states but no official initial-state IDs; settling and generation differences prevent equating demo indices to evaluation indices. Keep this limitation explicit and do not describe these four trained tasks as unseen tasks.

The compared methods are the frozen pretrained model, a directly trained output projection, and rank-8 q/v LoRA on all 32 action-expert layers. Each trained family uses three fixed seeds, paired expert samples and flow-noise schedules, the native flow-matching loss, and the same inference budget. LoRA trains 552,960 parameters. The output projection contains 15,392 parameters, of which the seven supervised output rows account for 3,367; it retains pretrained initialization. Neither experiment trains the visual-language backbone.

Whole demonstration trajectories are split: demo 0 is development-only; demos 1–40 are training; demos 41–49 are validation. No random frame splitting. The final protocol must verify those episodes actually exist in every source file. Validation diagnostics do not choose checkpoints or change hyperparameters: the scheduled final checkpoint is evaluated for every seed. The real native training probe measured approximately 6.6–6.9 seconds per update. The prospective budget is **2,000 updates per run**, six runs, seeds **17/23/47**, AdamW learning rate **0.0001**, no weight decay, checkpoints every 100 updates and validation every 500. This is 12,000 updates total, roughly a day of pure CPU training before resource waits and evaluation. A falling validation loss alone is not a successful robot result.

The planned prospective evaluation pairs all seven methods on each task, initial states 5/15/25/35/45, and clean versus both-camera brightness multiplied by 0.75. Each episode uses native 10-step action sampling, ten executed actions per replan, official full horizons, and official success predicates. This is 280 planned episodes. The new protocol freezes exact cases before training and evaluation. Every completed task failure is final evidence; do not replace it or retry for a higher success rate. Resource-interrupted partial attempts remain separately recorded.

## Execution and recovery

Production training must have a new frozen protocol, source copies, verified data and alignment receipts, and a passed real-checkpoint probe. Until these gates pass, the stage is **preparation**, not training complete. Do not resume old v5 supervisors or mix their denominator with this study.

The registration gate also verifies the completed two-family training pipeline check, including both sealed final adapters and the preserved historical source snapshots. Those two-update development weights are excluded from formal results and never used to initialize formal training. Each formal run starts from the same original pretrained checkpoint. A separate development episode at spatial initial state 0 validates adapter loading, native NFE accounting, simulation, and sealing. Either official success or failure satisfies that execution check; its outcome cannot tune the formal study and never enters the 280-case denominator. The four-source alignment receipt is reconstructed from its individual receipts and retained failed diagnostics before registration; a summary marked "passed" alone is insufficient.

Register once, after all preparation checks have passed:

```text
python -B scripts/prepare_action_campaign.py --sources data/action_adaptation/study_v1/source_manifest.json --alignment data/action_adaptation/study_v1/alignment_r3.json --probe-report outputs/action_adaptation/probe_v1/attempt_001/report.json --pipeline-report outputs/action_adaptation/pipeline_check_v1/report.json --development-evaluation-report outputs/action_adaptation/eval_pipeline_check_v2/evaluation/libero_spatial_t0_i0_clean_lora_s17/attempt_001/report.json --output outputs/action_adaptation/study_v1
```

The global action-study worker lock and append-only ownership records bind PID plus creation time. Before launching or recovering, inspect every recorded identity, including descendants of Windows virtual-environment launchers. A live owner or child forbids another model launch. When a stored lock's owner and all recorded children have exited, archive the lock and retain the previous status before a new supervised attempt. Unknown identity permissions fail closed.

Training checkpoints must bind source protocol, sample schedule, whole-demo data manifest, adapter arrays, optimizer state, and both flow RNG states. Resume only a verified checkpoint with the same protocol. Preserve every failed attempt and partial checkpoint directory. Real implementation errors require an explicit new revision with the old source and evidence retained. Do not silently overwrite frozen sources.

Use the existing project Python environment, with bytecode disabled:

```text
python -B scripts/run.py scripts/run_action_campaign.py status --campaign outputs/action_adaptation/study_v1
python -B scripts/run.py scripts/run_action_campaign.py verify --campaign outputs/action_adaptation/study_v1
python -B scripts/run.py scripts/run_action_campaign.py supervise --campaign outputs/action_adaptation/study_v1
python -B scripts/run.py scripts/report_action_campaign.py --campaign outputs/action_adaptation/study_v1
```

Launch detached supervisors hidden on Windows. `supervise --resume` archives a prior campaign lock only when its owner and all recorded children have exited; a probe or another campaign's lock requires separate identity review. Training resumes from the highest sealed checkpoint with unchanged protocol/RNG/optimizer binding, at most three preserved worker attempts per run. Partial checkpoint directories are never selected. The full run wall-time budget is cumulative across resumes. Evaluation may retry a genuinely partial resource-interrupted case at most three times; any terminal marker or complete report without a receipt requires audit and sealing of existing evidence, never a new simulation. A complete failed episode is counted and skipped exactly like a complete successful one.

At completion or the hard deadline, retain complete/partial reports and all planned denominators. After the deadline, only audit and package existing evidence; do not restart training or evaluation. Private logs, raw human-demonstration HDF5 files, environments, and full model weights are not publication files. Export only explicit reviewed code, protocols, small adapters, and sealed results to the existing independent private GitHub checkout, with all previous publication evidence preserved and verified.
