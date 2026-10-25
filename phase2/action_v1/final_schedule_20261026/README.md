# Action adaptation: immutable evidence snapshot

Status: **partial**. Training 6/6; evaluation 279/280. This is a snapshot, not live status.

[Frozen protocol](protocol.json) · [Results](report.json) · [Evidence index](evidence_index.json) · [Source index](source_index.json) · [Reproduction boundaries](REPRODUCTION.md) · [Publication manifest](PUBLICATION_MANIFEST.json)

Public projection receipts identify every included file and the original private receipt hash. Original full seals require omitted private files and cannot be verified from this subset. Adapters support inference with separately obtained pinned base weights; optimizer/RNG state is not published.

Four fixed tasks, five paired initial states, and three training seeds support only limited same-domain conclusions. Pretraining overlap is unknown. Observation/action alignment is checked; full dynamics replay is not verified. A decreasing training loss or partial evaluation does not establish better task success. Sources are archived under source/ at their original relative paths; this archive is not a standalone runnable environment.

## Finished scheduling revision

279 of the original 280 evaluations are sealed; one case remains unknown after three resource interruptions. The queue is finished and the original study remains **partial**. Unknown is not counted as task failure. Clean paired contrasts remain withheld when required pairs are absent. The descriptive brightness intervals do not establish a learning benefit.

[Scheduling report](scheduling/report.json) · [Revision and every attempt](scheduling/index.json)

Verify both the original evidence and scheduling layer with Python and NumPy:

```text
python -B reproduction/scripts/verify_action_schedule_publication.py --snapshot . --manifest-sha256 <SHA256-from-the-external-publication-audit>
```

Obtain the expected manifest SHA from the external publication audit, not from this directory. Scheduling protocol, prior inventory, process outcomes and launch authorizations are privacy projections with original-source SHA declarations. They do not reconstruct or verify the full private seals. The old v1 verifier alone does not verify this scheduling layer.
