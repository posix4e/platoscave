# Compact source-overlap summary projection

`export_source_summary.py` prepares a JSON report from the accepted, complete
source-stage reader. It copies all saved scalar values without recalculation or
rounding: 115 mask/seed fits, six metrics in four score views, 92 five-seed score
series, 40 within-seed random-mask spread series, and 12 fixed-stratum contrast
series. Every individual value, mean, sample SD (`ddof=1`), minimum, maximum and
contrast operand remains present. Parsed numeric values and integer/float types
are preserved; JSON formatting may change.

The original-overall score is the fourth view. It is not an additional contrast
stratum. Contrasts remain `C−mean(Rv)`, `D−mean(Rp)`, `C−O` and `D−C` within each
of the three fixed strata. Every declared random mask contributes. Positive
rank/FOSCTTM differences and negative Recall differences mean worse retrieval.
The full 40,504-item gallery and FOSCTTM denominator 40,503 remain in every view.

This presentation adapter is outside the frozen scientific release. It never
fits models, loads tensors, scores maps, reconstructs metrics, selects attempts,
retries jobs or publishes data. It uses metadata helpers in
`export_model_summary.py` and the rank exporter's JSON/hash utilities. Their
imports require the existing reporting environment with Torch, but no tensor
loader is called. No scientific runner, reader or primitive is imported.

## Required inputs

Use externally verified SHA-256 values. Keep the accepted source-stage
`summary.json`, `accepted.json` and `ticket.json` together under those names.
Supply the exact `source-stage-complete.json` receipt, the frozen source protocol,
mask bundle, preparation compositions/intersections, and the matching independent
source numerical audit. The audit must bind this exact reader and source-stage
receipt. A preflight or later final-audit snapshot is not interchangeable.

The controller ticket is hashed as sorted, indented JSON plus a final newline;
scientific metadata hashes keep their existing compact encoding. Relocated
copies are supported: paths recorded inside the ticket and receipt must agree,
but need not exist locally. They are not copied into the report.

The mask bundle itself contains row arrays and mask identities. Group-size
histograms, known train/validation/unknown-map membership counts, and all 253
pairwise row/source-group intersection counts live in the separate frozen
`compositions.json` and `intersections.json`. This tool requires both exact pinned
files and retains those small summaries and mask hashes. It omits raw paragraph
row arrays, COCO-ID arrays and presentation/training permutations. The original
mask bundle and preparation files remain available under their original hashes;
this report is not their replacement. Source identities are metadata identities,
not proof of image-byte or perceptual deduplication.

```sh
.venv/bin/python export_source_summary.py \
  --summary evidence/source-reader/summary.json --summary-sha256 <SHA256> \
  --acceptance evidence/source-reader/accepted.json --acceptance-sha256 <SHA256> \
  --stage-receipt evidence/stage/source-stage-complete.json --stage-receipt-sha256 <SHA256> \
  --protocol evidence/protocol/protocol.json --protocol-sha256 <SHA256> \
  --masks evidence/preparation/masks.json --masks-sha256 <SHA256> \
  --compositions evidence/preparation/compositions.json --compositions-sha256 <SHA256> \
  --intersections evidence/preparation/intersections.json --intersections-sha256 <SHA256> \
  --audit evidence/audit/source-audit.json --audit-sha256 <SHA256> \
  --out prepared/source-summary.json
```

Replace every `<SHA256>` placeholder before running the example. Use a fresh
output outside all input files' containing directories. Existing files and
symlink destinations are rejected. Only the requested new output is written;
the command prints its hash, byte size and coverage. It does not publish it.

## Evidence gates and retained history

A `complete` reader status alone is insufficient. Every one of the 115 expected
identities must have one verified, eligible final `COMPLETE.json` attempt,
matching selected observation, result identity, artifact hashes, mask composition
and accepted rank checks. Contiguous attempts and any reviewed retry's predecessor
binding remain checked. Missing fits, partial/conflicting markers, duplicate
completions, extra identities or unavailable contrast operands are rejected.
The separately bound controller stage receipt supplies the completed-service
provenance; this helper does not repeat process observations or artifact reads.

Every attempt retains its safe identity, terminal/partial markers, verification
flags, observed metric scalars, allowlisted artifact hashes, and retry category
and predecessor when valid. Arbitrary failure/retry prose, commands, process
identities, extra fields and diagnostic arrays are omitted and bound by record
hashes/counts and the full reader hash. Omission does not mean that a failure or
finding was absent. Any public explanation of a failure needs separate review.

The independent audit must cover all 2,760 individual scalar fields and all
8,976 aggregate scalar fields in each of its two aggregate tracks, with exact
field identities, copied frozen values and selected rank artifacts. This tool
checks those bindings and records the audit's difference counts. It does not
recompute the audit or certify its independently calculated values. There is no
source rank-export manifest input: the matching numerical audit binds all 115
selected rank artifact hashes. The audit report and trusted provenance still
require independent review.

## Preparation is not approval

Output always has `status: "projection_prepared"`,
`scientific_acceptance: "not_assessed"`, `publication_authorized: false` and
`publication_review_reference: null`. No tolerance, replacement metric, pass/fail
scientific rule, significance test or publication approval is introduced.
Nonzero audit differences remain visible through the bound audit and its counts.
Final independent numerical and interpretation review must be linked when results
are actually published; this tool provides no command that grants approval.

The hash-pinned protocol's interpretation limits remain in the output. Masks
share sources; five algorithm seeds are not five independent corpora or encoder
training runs. Removing fitting paragraphs cannot remove pretraining exposure.
Exclusion changes semantic evidence and centering as well as known overlap, so
an effect cannot isolate memorization. Unmapped sources remain unknown. The 115
source fits do not establish completion of the 60-job model panel, the full
175-job execution or the later prospective study.

## Validation and limits

Run in the existing reporting environment:

```sh
.venv/bin/python -m unittest -v test_export_source_summary.py
```

The tests use invented four-row tensors across all 115 identities. Each fixture
produces an independent arithmetic report with the existing audit helper before
projection; successful projections forbid tensor loading and scientific module
imports. Tests cover exact copied values, all series and mask intersections,
failed history with private-detail omission, source-stage/ticket encoding,
incomplete/duplicate completions, missing operands, audit and preparation
mismatches, input hashes, duplicate JSON keys, direction, nonzero differences
without approval, and protected outputs. Production counts and pins are changed
only inside the test process; no command-line option weakens them.

No actual source-fit output has been projected or audited with this adapter.
Local read-only comparison against the frozen emitters and actual preparation
metadata is a schema check, not evidence that the source experiment has completed.
Full-data runtime and real output size remain unmeasured. Review the eventual
complete reader, numeric differences and prepared public report independently
before publication or interpretation.
