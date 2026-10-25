# Reproduction and evidence boundaries

This is an immutable audit archive, not a preconfigured training directory. Original relative paths in the protocol describe the original run. Copying this snapshot into another folder does not make those paths valid.

1. Audit the published results using the manifest, evidence index, sealed simulation reports/actions/calls, and public projection receipts. The original receipt hashes identify the private evidence; omitted optimizer/RNG/resources mean its complete seal cannot be revalidated here.
2. For adapter inference, obtain the separately licensed pinned foundation checkpoint and processors, verify every model hash in the protocol, and load the adapter using the matching archived LoRA/output-head implementation and base fingerprint. No full model weights are included.
3. For new training, use a separate fresh checkout/work directory, acquire the four pinned expert HDF5 files, and create fresh paths, source snapshots, schedules, raw prerequisite receipts, and a newly registered protocol. Preserve this study's snapshots. Changing a path, deadline, or source changes the protocol fingerprint and makes it a new study.

The registration gates require an actual real-checkpoint probe, both two-update training pipelines with complete optimizer/RNG receipts, the original observation-alignment diagnostics and r3 chain, and a separately registered development episode. Public redacted alignment summaries and projected checkpoint receipts are not acceptable replacements for those raw gate inputs. Re-run those prerequisites in the new environment; do not relabel projected JSON as an original receipt. The development episode checks execution regardless of task success and stays outside the formal denominator.

Historical integration registration and launch helpers originally lived under .runtime. A historical source copy is provenance, not a portable CLI: its fixed directories and prior-registration references must not be assumed to exist. A complete from-zero workflow needs public development-training registration, development-evaluation registration/supervision, and a new raw alignment chain. Use separately documented public replacements if supplied; this exporter does not synthesize or silently substitute them.

The archived campaign enforces the original 2026-10-30 06:08 UTC deadline. A later reproduction needs an explicitly new protocol and compatible runner revision; do not edit the historical frozen deadline. Fresh training and evaluation can produce different results. This archive has not itself been tested as a standalone from-zero installation.

The later [public development CLI](reproduction/docs/ACTION_DEVELOPMENT_REPRODUCTION.md) is supplied as a separate [source release](supplemental_sources.json). It was not used for the original archived runs and does not modify their source bindings.

A standalone read-only verifier needs Python and NumPy, but no Git checkout, model, simulator, or private runtime:

```text
python -B reproduction/scripts/verify_action_publication.py --snapshot . --manifest-sha256 <trusted-publication-manifest-sha256>
```

Run this command from the snapshot directory, or use explicit absolute paths. It verifies this public snapshot's bytes and statistics; it does not prove the original full private seals or earlier Git history.

## Finished scheduling revision

279 of the original 280 evaluations are sealed; one case remains unknown after three resource interruptions. The queue is finished and the original study remains **partial**. Unknown is not counted as task failure. Clean paired contrasts remain withheld when required pairs are absent. The descriptive brightness intervals do not establish a learning benefit.

[Scheduling report](scheduling/report.json) · [Revision and every attempt](scheduling/index.json)

Verify both the original evidence and scheduling layer with Python and NumPy:

```text
python -B reproduction/scripts/verify_action_schedule_publication.py --snapshot . --manifest-sha256 <SHA256-from-the-external-publication-audit>
```

Obtain the expected manifest SHA from the external publication audit, not from this directory. Scheduling protocol, prior inventory, process outcomes and launch authorizations are privacy projections with original-source SHA declarations. They do not reconstruct or verify the full private seals. The old v1 verifier alone does not verify this scheduling layer.
