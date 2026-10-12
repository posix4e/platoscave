# Compact model-panel projection

`export_model_summary.py` prepares one JSON file from the accepted, complete
model-panel reader. It copies the frozen scores without recalculation or rounding:
300 method observations, 60 five-seed score series and all 103 prescribed contrast
series (25 primary, 30 interactions and 48 within-cell controls). Each series keeps
its individual seeds, means, sample standard deviations (`ddof=1`), status and,
for contrasts, signed operands. JSON formatting may change; parsed numeric values
and their integer/float types are retained.

This is a presentation tool outside the frozen scientific releases. It does not
fit models, load tensors, rescore maps, recompute metrics or interpret outcomes.
It reuses only the existing rank exporter's metadata validators. That import
requires the same reporting environment, including Torch; no new dependencies
are needed. Do not modify the frozen execution environment to run this tool.

## Inputs and use

Use externally verified SHA-256 values for all six inputs. The reader directory
must contain `summary.json`, its exact `accepted.json`, and the matching
`ticket.json`. The separate receipt must be named `model-stage-complete.json`.
The independent model audit and complete rank export manifest must refer to that
same accepted summary, protocol and selected rank artifacts. The audit must also
bind that stage receipt and ticket. This deliberately rejects an audit made from
a different, later final reader snapshot, even if its scores happen to match.

The tool supports relocated evidence: ticket and stage receipt retain their
original recorded paths privately, which must agree with each other; those paths
are not copied to the projection. A fresh output must be outside every input
file's containing directory. Existing files and symlink destinations are rejected.

```sh
python export_model_summary.py \
  --summary evidence/model-reader/summary.json --summary-sha256 <SHA256> \
  --acceptance evidence/model-reader/accepted.json --acceptance-sha256 <SHA256> \
  --stage-receipt evidence/stage/model-stage-complete.json --stage-receipt-sha256 <SHA256> \
  --protocol evidence/protocol/protocol.json --protocol-sha256 <SHA256> \
  --audit evidence/audit/model-audit.json --audit-sha256 <SHA256> \
  --rank-manifest evidence/ranks/manifest.json --rank-manifest-sha256 <SHA256> \
  --out prepared/model-summary.json
```

Replace each `<SHA256>` placeholder before using this example. The command prints
the projection's SHA-256, byte count and coverage. It writes only the requested
new JSON file and does not publish it.

## Evidence and scope

The projection requires all 60 cell/seed jobs to have a verified complete final
attempt, all 300 methods to be available, the five-reference replay gate to be
satisfied, and all expected seeds and contrast operands to be present. It binds
the accepted reader's existing eligibility, history, RNG/order and service
provenance instead of repeating those scientific checks. It checks the audit's
complete scalar coverage and exact frozen values, the selected rank hashes and
the complete 300-partition rank manifest. It does not read CSV bytes or repeat the
rank reconstruction. Manifest/audit hashes and their independently reviewed
provenance must be trusted by the caller; hashes alone are not proof of a valid
scientific run.

Attempt history retains every attempt's identity, terminal status, verification
flags, method status/eligibility, observed scalar metrics, selected attempt,
repetition counts/agreement, and allowlisted artifact hashes. A method selected
from a finalized `FAILED.json` cell remains explicitly labelled as such. Retry
category and predecessor remain visible. Unrestricted explanations, diagnostic
arrays, commands, process identities and arbitrary fields are omitted and bound
by canonical hashes/counts, as well as the full reader hash. These omissions do
not mean that failures or findings were absent. Any public prose explaining a
failure requires separate review; this helper does not safely sanitize arbitrary
prose. Bounded seed, operand and attempt lists remain; per-query data and mapping
arrays do not.

The fixed scope and hash-pinned protocol text retain metric direction, descriptive
inference, the selected exploratory panel, known prior model-selection context,
complete-pipeline confounds, and the extra counterparts given to paired controls.
Lower FOSCTTM and mean rank are better; positive signed contrasts mean worse
retrieval. Mean rank is dependent on FOSCTTM. The five seeds share data and frozen
pretrained encoders, rather than representing independent datasets or encoder
training runs. The reference replay gate compares retrieval to our own preserved
baseline; it does not reproduce author values anew, and classification was not
rerun. These 60 cell/seed jobs and 300 method observations are only the model
panel, not all 175 scheduled jobs or the separate source study.

## Preparation is not approval

The result always has `status: "projection_prepared"`,
`scientific_acceptance: "not_assessed"`, `publication_authorized: false` and
`publication_review_reference: null`. It records the independent audit hash and
its comparison counts, including any nonzero differences. No numeric tolerance,
pass/fail rule, replacement score, confidence interval or new approval procedure
is introduced. Review the bound audit for the independent values, signed/absolute
differences and reduction-order provenance; this JSON does not replace that
report. Final independent interpretation/publication review remains a separate
required step and must be linked with these artifacts when results are actually
published. The helper supplies no command that grants publication approval.

## Validation and limits

Only invented four-row fixtures have been exercised. Run the focused suite with:

```sh
python -m unittest -v test_export_model_summary.ModelSummaryTests
```

It covers all 300 observations and 163 series, exact copied scalars and operands,
retained failed-attempt evidence and private-detail omission, direction/inference,
partial or mismatched evidence rejection, changed audit differences without an
approval claim, and protected output paths. `torch.load` is guarded during every
successful export. The existing fixture creates invented tensors, but this
helper never loads them. No real model-panel output has been processed or
verified with this implementation. Review real accepted evidence and the prepared
projection independently before any public use. A complete projection is not a
successful independent scientific audit or approval to interpret the study.
