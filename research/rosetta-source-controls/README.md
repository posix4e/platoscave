# Does known source overlap account for the result?

**Design draft. This study has not run.** It is separate from the disjoint-half
COCO reproduction and the model comparison. It uses the original released
DINOv2-B/14 mean image features and MPNet text features throughout.

The question is whether removing known shared sources hurts retrieval more than
removing the same amount of text data by a fixed random rule. Removing rows also
changes the amount and composition of the evidence available for fitting. The
comparison must expose both effects rather than treating every change as proof
of memorization.

The earlier 15 reduced fits and their protocol remain unchanged. Their scores
are already known. This new proposal uses complete fitting populations, the
authors' full algorithm settings and the entire validation gallery. It is a
descriptive follow-up; it does not inherit the earlier proposal's unaccepted
effect margin, confirmatory tests or power claims.

## Keep the encoders fixed; change the fitting sources

Every arm uses all 82,783 COCO training image rows. Only the SPC paragraph bank
changes. Original source IDs determine exclusions; absent metadata stays unknown.

| Arm | Paragraph bank | Retained rows |
| --- | --- | ---: |
| O | All original SPC paragraphs | 19,561 |
| C | Remove known COCO validation sources from O | 16,224 |
| D | Also remove known COCO training sources from C | 9,963 |
| Rv | Randomly retain whole source groups from O, matching C's row count | 16,224 |
| Rp | Randomly retain whole source groups from C, matching D's row count | 9,963 |

Repeated descriptions of a source stay together. Canonical COCO IDs take
precedence over Visual Genome IDs where the metadata supplies a mapping. This
also joins two cases in which different Visual Genome IDs refer to the same
COCO source. Unmapped sources retain their Visual Genome identity.

The proposed sampler follows the earlier audit: order whole source groups by a
seeded SHA-256 value, retain groups that fit the remaining row budget, and restore
original paragraph-row order. It is not uniform sampling of all possible row
subsets. Record this rule, each actual mask and its hash, group sizes, known
source memberships, missing mappings, and intersections among masks.

D's remaining 9,963 rows represent 9,956 sources **without COCO mappings**.
They have not been established to be different images. The historical identifier
`fully_dedup` may appear for compatibility with the old audit; it means only
removal under the recorded metadata rule. No arm is certified free of duplicates.

## Preserve the complete algorithm and its randomness

Use 30 clusters, 30 initialization attempts, a configured assignment batch cap
of 10,000 and 100 refinement iterations. Use the original five seeds:
734796314, 576165995, 2197670066, 839703249 and 2584932063. D and Rp have only
9,963 text rows. The original implementation handles the smaller actual bank;
do not duplicate rows or change the cap to conceal that fact.

For each arm, start with canonical selected rows, consume the authors' full
independent image and text permutations, and pass the same advanced randomness
object into the unchanged aligner. Reusing a seed number while omitting those
permutations would be a different execution. The old pilot launcher does omit
them, so selecting its full-settings profile alone is insufficient.

Preserve the original BF16 caption normalization and averaging before FP32
conversion. Estimate centering from each arm's own fitting rows. Retaining the
original arm's mean after removing sources would leave information from those
sources in the cleaned arm.

The proposed allocation is ten Rv masks and ten Rp masks, each crossed with
all five algorithm seeds, plus five fits each for O, C and D: **115 fits**.
Proposed selection seeds are 1000–1009 for Rv and 2000–2009 for Rp. They are
separate from fitting randomness. This allocation favors repeated deletions
while retaining the authors' five seeds. It is a practical exploratory budget,
not a power calculation. Check its resource cost against completed full-run
timings before freezing it. Any change must precede fitting and remain visible.

## Keep the exposed queries in the evaluation

Every run retrieves against the same complete gallery of 40,504 COCO validation
items. Define these strata once from original SPC membership:

- Primary: 37,169 queries with no known original SPC source match.
- Direct-exposure diagnostic: 3,335 queries with an original SPC source match.
- Overall diagnostic: all 40,504 queries.

Keep those strata fixed after cleaning. Defining exposure anew for each arm
would make the exposed group disappear when its fitting sources were removed.
Do not shrink the gallery for a stratum. A small effect averaged over all
queries can conceal a larger effect among the originally exposed queries.

Save each query's rank, strictly higher score count and tie-block size with its
original source identity. FOSCTTM divides the number ahead, with half credit for
tied alternatives, by 40,503. Lower is better. Also report mean and median rank
and Recall@1, @5 and @10. Recall assigns the expected credit from random order
within the exact tie block; testing whether an average rank is below k would
give a different measure.

Let F denote FOSCTTM. Within each original algorithm seed and each fixed
stratum, calculate:

`validation effect = F(C) − mean_masks F(Rv)`

`training-source effect = F(D) − mean_masks F(Rp)`

Also report `F(C) − F(O)` and `F(D) − F(C)`. Positive targeted-minus-random
differences mean the targeted removal hurt more than its row-count control.
Report all five contrasts, their mean and spread, and every mask's score. Do
not count fifty mask/seed combinations as fifty independent datasets. Missing
fits remain missing rather than disappearing from the average.

These are conditional, descriptive comparisons. There is no significance test,
equivalence margin or claim of adequate power in this draft. Row-count controls
do not automatically match categories, paragraph style, group sizes or mapping
coverage. The effects cannot isolate memorization from those changes.

## What must happen before execution

The original published-result reproduction and model panel remain earlier work
in the plan. This directory prepares source grouping and evaluation code while
they run. It has no final execution lock or released run schedule.

Before a real fit, verify the original four tensors and complete source metadata;
freeze all actual masks, fixed query strata, gallery IDs, source/runtime/code
identities and reporting rules; and test the new wrapper against the authors'
cross-dataset loading and randomness. Publish that complete release before
running it. Preserve every attempt, failure and numerical result.

The [draft protocol](protocol.json) records the full populations and proposed
budget. The [four-input lock](inputs-lock.json) and
[metadata record](metadata-provenance.json) identify the existing public data.
Neither enables execution while the protocol has draft status.

`source_masks.py` keeps source groups intact. `source_primitives.py` preserves
the authors' loading, randomness and full-gallery evaluation. `run_source_arm.py`
checks the release and runs one prescribed mask/seed attempt. `prepare_masks.py`
requires a final code release, independently rebuilds source identities from the
three archives and writes every declared mask and the complete schedule before
fitting. A later execution lock must bind both this bundle and the scientific
protocol. No new real masks have been constructed for this draft.

`summarize_source.py` checks every declared mask and seed, reconstructs the
metrics from canonical ranks and tie counts, and preserves incomplete attempts.
Every random mask is required for its within-seed mean; all five seeds are
required for a five-seed summary. Its checks concern the saved ranks and their
identities, rather than a second scoring of the fitted maps.

Keep every fit under `cells/{mask}/seed-{seed}/attempt-{01..99}`. A successful
attempt needs exactly one verified `COMPLETE.json`, every expected output and
matching hashes. Conflicting `COMPLETE` and `FAILED` markers, incomplete records
or multiple complete attempts remain unavailable. A reviewed execution/input
retry needs a reason and predecessor hash inside its new attempt, bound by the
final manifest; no retry follows a complete attempt or an unfavorable score.

For an already reviewed retry, pass both `--retry-review` and
`--retry-review-sha256`. The runner copies and binds that exact record before
preflight. The launcher must first verify the previous attempt's actual hashes
and establish that its process has stopped. The runner does not approve or
automatically dispatch another attempt.

This study evaluates retrieval. It does not reproduce all classification
endpoints of the authors' cross-dataset experiment. It also leaves encoder
pretraining untouched. In particular, removing SPC paragraphs cannot remove
COCO captions from MPNet's earlier training. The
[caption-exposure audit](../pretraining-audit/README.md) records that separate
issue and its evidential limits. A fresh collection with its own fitting data
and later final evaluation remains another required stage.
