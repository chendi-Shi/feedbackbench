# Preserved baseline campaign — 2026-10-14

This campaign was safely stopped at a completed-stage boundary at 08:09:03 UTC,
after the user prioritized current frontier research under a CPU-only, zero-cloud
constraint. The process checkpoint originally supported a restart; the subsequent
research decision is to preserve it as baseline evidence and prioritize the
independent ActionCache-to-SmolVLA study and VLA-JEPA feasibility probe.

| Stage | Completed / declared | Status |
|---|---:|---|
| MINERVA task coverage | 40 / 40 | Complete |
| Development selection | 32 / 32 | Complete |
| Main comparison and ablations | 128 / 128 | Complete |
| Controlled observation delay | 32 / 32 | Complete |
| Original SmolVLA scheduling stage | 0 / 12 | Never attempted; not resumed |
| Original flow-step stage | 0 / 4 | Never attempted; not resumed |

Thus **232 baseline episodes are complete; the original 248-episode campaign is
not complete**. The independent frontier study has its own protocol and denominator.
No global pooled success rate combines these different stages.

The original `campaign_status.json` is preserved as written before process exit;
it is not a live process monitor. Actual exit is recorded in
[`stop_receipt.json`](../checkpoints/before_smol/stop_receipt.json). The checkpoint
audit [`resume_audit.json`](resume_audit.json) binds every completed raw episode
file and frozen source. No episode was removed or retried at the transition.

Do not treat the old restart command, planned stages, or old status as evidence
that the original SmolVLA experiments ran. The new mainline is described in
[`CACHE_PROTOCOL.md`](../../docs/CACHE_PROTOCOL.md).
