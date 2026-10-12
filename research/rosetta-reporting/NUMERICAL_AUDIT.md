# Independent saved-rank numerical comparison

`audit_rank_summaries.py` is a separate reporting implementation outside the
frozen scientific releases. It imports no scientific runner, reader or primitive.
It has been exercised only on invented four-row fixtures. It has not audited
real run outputs, established scientific correctness, or authorized publication.

The tool reads an accepted **complete** model or source summary, its exact rank
artifacts, the frozen protocol and (for source) frozen mask bundle. It independently
reconstructs metrics and every declared summary/contrast. It never changes scores,
fits a map, scores embeddings, retries a job, chooses another attempt or publishes
data. PyTorch is used only for `weights_only=True` CPU deserialization and tensor
schema checks; all metric and aggregate arithmetic is separate Python arithmetic.

## Required evidence and invocation

Use the summary named by the controller's `model-stage-complete.json`,
`source-stage-complete.json`, `final-audit-model.json` or `final-audit-source.json`.
Retain that reader's `summary.json`, `accepted.json` and `ticket.json` together
under their original names. Supply externally recorded SHA-256 values for the
summary, acceptance, study/final receipt and protocol. The reader ticket is bound
by the study receipt's canonical ticket hash. Preflight tickets/receipts are
rejected. A byte-preserving local copy is supported: recorded paths are compared
inside the receipt/ticket, without requiring the original machine's directory.
Do not rewrite those evidence files to relocate them.

The frozen reader's accepted evidence supplies eligibility, documented attempt
selection/history, RNG/identity verification, masks and feature provenance. The
controller receipt supplies the existing completed-study gate. This tool checks
their binding and type; it does not independently authenticate the externally
supplied hashes or repeat service observations. A missing receipt, incomplete
summary, missing observation, unavailable operand or mismatched artifact is
rejected. There is no partial-results mode and no successful-subset averaging.

```sh
.venv/bin/python audit_rank_summaries.py \
  --study model \
  --summary /path/to/accepted-reader/summary.json --summary-sha256 SUMMARY_SHA256 \
  --acceptance /path/to/accepted-reader/accepted.json --acceptance-sha256 ACCEPTANCE_SHA256 \
  --study-receipt /path/to/controller/model-stage-complete.json --study-receipt-sha256 STAGE_SHA256 \
  --protocol /path/to/frozen/model/protocol.json --protocol-sha256 PROTOCOL_SHA256 \
  --attempts-root /path/to/model/cells --out /path/to/new-reports/model-audit.json
```

For source, use `--study source`, the corresponding source receipt/protocol/tree,
and add `--masks /path/to/frozen/masks.json --masks-sha256 MASKS_SHA256`. The current
frozen protocol, reader, feature-lock and mask identities are pinned in the code;
a different release requires review of this adapter. No production argument
weakens the 40,504-row, five-seed or complete-identity requirements.

The output must be new and outside all input artifact/evidence directories. An
existing output is never overwritten. A partial write is not a valid report.
The JSON contains hashes, relative selected-attempt names and scalar comparisons;
it excludes absolute machine paths, process identities, photos and embeddings.
Private stderr may contain a local path when an input fails to load.

## Arithmetic and order

Model ranks already follow the validation-subset permutation. Rank `i` belongs
directly to `validation_image_ids[i]` and `validation_indices[i]`; the query
shuffle is not applied again. Each record is compared with the accepted mapping,
and canonical ID-to-source-row alignment must agree across all five seeds.

Source ranks, higher counts and tie counts follow canonical file rows. The
accepted `validation_order` identifies the original overall reduction order;
the source mask bundle supplies fixed canonical stratum indices. The three strata
remain 37,169 never-known-overlap, 3,335 originally overlapping and all 40,504
queries. All retain a 40,504-item gallery and denominator 40,503. Canonical IDs
must agree across seeds. No public-ID sorting is used for scientific reductions.

Midranks are represented as exact doubled integers. Means and FOSCTTM are exact
rational sums. Source median averages the two central ranks when the query count
is even. Recall is the exact rational mean of `clamp((k-higher)/ties, 0, 1)` for
`k=1,5,10`; a tied midrank threshold is not a substitute. The artifact must satisfy
`2*rank = 2*higher + ties + 1` with valid tie-block bounds. Model artifacts lack
tie counts, so only their prescribed FOSCTTM and mean rank are reconstructed.

The three comparison sections keep different questions separate:

1. `rank_metrics`: frozen individual metrics versus exact rational reconstruction.
2. `aggregates_from_saved_scalars`: independent aggregation of the frozen saved
   binary64 scores, converted to exact rationals. This isolates aggregation from
   differences in metric reconstruction.
3. `aggregates_from_reconstructed_metrics`: the same aggregates using the exact
   reconstructed rational metrics, without first rounding them to binary64.

All contrasts are formed within seed before aggregation. Arithmetic means and
contrasts are exact rationals. Sample variance is the exact centered sum of
squared deviations divided by `n-1`; SD uses Decimal division and square root at
80 significant digits. The SD value and its difference string are explicitly
finite-precision approximations. This implementation does not import the frozen
`statistics.mean`/`statistics.stdev` aggregation or Torch reduction helpers.

Every comparison preserves the frozen value, a binary64 independent rendering,
the exact rational or 80-digit decimal independent value, the corresponding
difference, signed/absolute float renderings and binary64 equality. Differences
use **frozen minus independent**. A rounded binary64 rendering can agree while
its underlying rational difference is nonzero; both are retained. Exact rational
sums are order independent, whereas frozen binary64 reductions preserve their
original row order. In particular, source `original_overall` and fixed `all`
can have different last-bit results despite covering the same queries.

No difference is automatically excused as rounding or labeled scientific
failure. No tolerance is introduced. The original `1e-10` reference-replay rule
is not reused as an independent-audit threshold. `status: comparison_generated`
means the requested comparisons were produced after input/coverage checks;
`scientific_acceptance: not_assessed` remains explicit regardless of differences.

## Complete reporting scope

- Model: 60 cell/seed jobs, 300 method observations, 60 five-seed score series and
  103 contrasts (25 single replacements, 30 crossing interactions and 48 within-cell
  controls), for both metrics. Eligible selected methods retain their terminal
  `COMPLETE.json` or `FAILED.json` label. A method from a failed earlier attempt
  is not itself proof that a whole cell completed; the separate controller receipt
  remains necessary.
- Source: 23 masks across five seeds, 115 fits, all six metrics and four views
  including original overall. Reconstruct 92 per-mask five-seed series, 40
  within-seed random-mask spread series, and 12 fixed-stratum contrast series:
  `C−mean(Rv)`, `D−mean(Rp)`, `C−O`, `D−C`. Every declared random mask contributes.
  Means, sample SDs, minimums, maximums, individual scores and declared operands
  are checked. Source original-overall is not an extra contrast stratum.

Exact observation/series identity sets, available/required seed and mask lists,
empty missing-operand lists, contrast signs/operands, stratum membership/counts,
rank shapes/dtypes, tie consistency, canonical mappings and artifact hashes are
input conditions. Numeric disagreements are report records, not input rejection.

The complete execution schedule is 175 jobs: 60 model jobs plus 115 source jobs.
The 300 model method results are not 300 scheduled jobs. Passing reference replay
against this project's preserved baseline is not a new agreement with author
values. This tool establishes neither reference claim by itself.

## Validation and limitations

Use the existing isolated reporting environment (Python 3.12, `torch==2.8.0`):

```sh
.venv/bin/python -m unittest -v test_audit_rank_summaries.py
```

The fixtures reduce row/stratum counts in the test process only while retaining
all 300 model and 115 source identities, all five seeds and every contrast. They
include shuffled mappings, even median, fractional tied recall, nonzero numerical
differences, missing entries/series/operands, wrong hashes/mappings/tie counts,
receipt/ticket mismatches and output protection. Invented frozen aggregates use
separate ordinary float arithmetic and standard-library statistics. No real
scientific output is loaded by the tests.

Before real use, independently review this adapter and collected evidence, then
review every real-output difference and document that review. Runtime and output
size on full data have not been measured. The source rank recorder's actual score
comparisons, saved maps, training correctness, service history and scientific
interpretation are outside this arithmetic check. It creates no significance,
equivalence, contamination, author-error or universal-model-ranking claim.
