# Local learned-compute campaign

The authorized work window ends at **2026-10-30 06:08 UTC / 14:08 China time**. This is a maximum deadline, not a promised positive result. Use only the existing local CPU computer, one experiment worker at a time, with no cloud spending and no interruption of other projects.

The research component is a trained inference-budget controller over a frozen, pretrained SmolVLA. It predicts whether 2 or 4 Euler steps deviate from a 10-step reference, and chooses 2, 4 or 10 steps. The 10-step output is not ground-truth correctness. Only held-out closed-loop task success and measured cost can establish a useful result. The controller does not update the backbone or provide a safety guarantee.

Held-out tasks are held out from this new controller's fitting, tuning and calibration. This does not establish that the foundation policy's original pretraining never used those tasks or initial states.

The preregistered design is in [the protocol document](LEARNED_COMPUTE_PROTOCOL.md). The actual frozen machine-readable protocol is `outputs/learned_compute/v4/protocol.json`; it binds sources, official task identities, inputs, runtime limits and the real-checkpoint parity receipt. Do not edit bound files during the campaign. A necessary behavioral bug fix requires a new protocol revision and explicit accounting of previous attempts, not alteration of old evidence. Do not mix this study with earlier ActionCache, VLA-JEPA or MINERVA results.

The initial v1 integration attempt stopped at the memory guard before the first action. It produced zero training trajectories. Its protocol, complete source snapshot and interruption evidence remain preserved. Revision 2 raises the startup free-memory requirement from 4 to 5.25 GiB and monitors the actual Python descendant of the Windows venv launcher. The per-process-tree limit remains 5.5 GiB and system free-memory limit remains 0.75 GiB; task splits, model, algorithm, horizon, seeds and deadline are unchanged. Standard garbage collection after simulator reset releases obsolete Python objects; no other application is stopped or trimmed.

Revision 2 also stopped before its first action, after context preparation and the first velocity, with 0.728 GiB available. Revision 3 preserves that attempt and adds exactly one `K32EmptyWorkingSet(GetCurrentProcess())` call after simulator reset and before first inference. It touches only the model worker, evicts resident pages without changing tensor data, and is applied identically to every method. Before/after RSS and private bytes are recorded. Startup and subsequent page-fault costs remain in measured episode/inference wall time. The original resource guards remain; this is not a steady-state memory guarantee. The process monitor follows observed descendants with PID and creation-time checks; it is not Windows Job Object isolation, and its RSS sum can count shared pages more than once.

The active revision is **v4**: the same single reclamation happens after model loading and **before** simulator construction. This avoids the startup overlap that stopped the second v3 case. The one completed v3 training trial (76 actions, official success) is archived with its source snapshot; it is not added to v4's denominator. The actual research settings and deadline remain unchanged. Do not resume v1/v2/v3 supervisors.

## Stages

1. Collect all 56 full trajectories from 20 train, 4 tuning and 4 calibration tasks. Each observed decision gets the same-context/same-noise 2/4/10 predictions. Both successful and failed complete episodes are retained. Per-decision raw images, state, noise, first velocity, latent outputs and action commands are saved.
2. Train all nine small controllers: three seeds for full features, no probe features, and proprioception plus probe features. Training-set normalization only; tuning selects the epoch; calibration selects the policy under a mean-NFE constraint. Save every seed and ablation.
3. Freeze all trained checkpoint hashes, then evaluate all 15 methods on 12 held-out tasks and initial states 41/43: 360 planned full episodes. Do not train on these observations or adjust any threshold after seeing test outcomes. Random controls use probabilities fixed from calibration, not test frequencies.
4. Report task-group-weighted success, measured NFE and latency, paired contrasts and task-group bootstrap intervals. If the plan is incomplete, retain the `partial` designation and do not claim a complete benchmark gain.

The first velocity is reused by the chosen grid. Online actual NFE is exactly K, including the probe; collection of all three grids costs 14 evaluations. `no_probe` still performs the first action-head evaluation but excludes its statistics from selector features. Gripper disagreement is measured after the original checkpoint postprocessor, in the simulator's command convention.

## Start and inspect

From the project root, use the existing environment Python to invoke:

```text
python scripts/run.py scripts/run_budget_campaign.py status
python scripts/run.py scripts/run_budget_campaign.py supervise
python scripts/run.py scripts/run_budget_campaign.py verify
python scripts/run.py scripts/run_budget_campaign.py report
```

`supervise` owns an exclusive lock, checks the frozen sources, and launches a fresh child process for each episode. On Windows the child has no visible terminal. A detached supervisor must also be launched hidden. `--max-episodes 1` runs one complete episode for the initial integration check. Status records include process IDs and creation times; always verify both before considering a process alive. Never launch a second supervisor or orphan duplicate model worker.

The supervisor waits if RAM or disk capacity is insufficient. It never stops other projects. Per-episode interruptions stay under `attempts/`; completed evidence is sealed by SHA-256 receipt and moved into `collection/` or `evaluation/`. Completed official task failures are final evidence and are not retried for a better outcome. There are at most three attempts per case. A trained directory completed before a supervisor crash is revalidated and frozen; incomplete fitting is archived and deterministically rerun, with all test work still closed and at most three fitting attempts.

`PAUSE_REQUEST.json` at the campaign root requests a stop between episodes. A current episode may finish; do not kill it just to inspect progress. The hard deadline requests a cooperative stop and then terminates only the supervisor's own exact child process after the recorded grace period. Partial outputs are preserved. A dormant machine does no computation: the computer must remain on; sleep/shutdown pauses progress. The Codex app must be running for scheduled chat continuation.

## Scheduled continuation and publication

An in-chat heartbeat checks the campaign every four hours until the deadline. It reads `status.json`, `report.json`, latest run audit and this document; verifies process identity; and restarts a stopped supervisor only when the frozen protocol permits continuation. If a genuine runtime error needs code changes, diagnose it before resuming. Do not silently lower resource limits, shorten horizons, remove failed tasks, or choose seeds from test results.

When all planned work completes, or the deadline arrives, produce the final complete/partial report and pause the heartbeat. Report meaningful stage completions, actionable failures and final results. Routine unchanged resource waits do not need notifications.

Publication uses the independent private checkout `deliverables/github/feedbackbench`, not the unrelated parent repository. Export explicit source, protocol, tests and sealed evidence through the existing audited exporter; preserve all historical frozen files and validate the publication manifest before pushing. Do not export environments, full model weights, private runtime logs or unsealed attempt directories by default. Record a publication snapshot date so a stale remote report cannot be mistaken for live local status.

Related work motivating adaptive inference includes [ELASTIC](https://andrewzl.github.io/elastic/index.html), [AdaVLA](https://arxiv.org/abs/2608.29208) and [Coda](https://arxiv.org/abs/2609.21216). This is an independent small-controller study on the available CPU platform, not a reproduction of their training procedures or a claim of novelty or SOTA.
