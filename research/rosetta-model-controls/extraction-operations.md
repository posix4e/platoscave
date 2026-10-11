# Admission of the first extraction

This release admits the ten fixed model/split extractions, in protocol order,
one at a time. It admits no alignment fit. The first extraction is DINO on
`train2014`. `admit_extraction.py` is a one-shot admission for that first job;
it does not schedule the remaining jobs or retry a failed dispatch.

Before dispatch, verify the outer `release-lock.json` against its externally
recorded SHA256 and verify every entry in its repository-relative `files` map.
Verify the scientific protocol, final extraction lock, preserved baseline
evidence and published discrepancy review agree. Record the exact private input
binding file hash without publishing its machine paths. The extractor separately
checks model/source bytes, runtime, annotations and image identities.

Use the reviewed `Systemd` backend from the bound `rosetta-execution/execution.py`.
Check live resources and competing processes, require a fresh output directory
and unique service name, and retain the command ticket and dispatch record before
launch. Keep operational logs outside the output directory, which the extractor
creates. Dispatch exactly once. After a lost reply, reconcile the recorded unit
and invocation; do not dispatch the command again.

The retained service survives loss of the SSH connection. It is not a promise
of recovery after a machine reboot. Preserve any partial output or failure.
Retries require a separate review and fresh attempt; an unfavorable score is
never a reason to repeat an experiment.

Before admitting the next extraction, establish that the preceding service and
all descendants have stopped writing. Require one valid `COMPLETE.json`, no
conflicting or partial terminal marker, and verify actual tensor and metadata
hashes, model, split, row count, dimensions, lock and extractor identities.
Observe the prescribed order: DINO, MoCo, MAE, GloVe, ELMo; training then
validation for each. Do not overlap extraction with fits or output verification.

After all ten added extractions, the separate feature-lock builder must verify
all fourteen feature entries, including the four released baseline tensors.
Only a later complete execution release can admit the five reference replay
cells. Every replay must match our preserved baseline before expanded fitting.
