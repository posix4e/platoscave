# Rosetta overlap and model-choice controls

Protocol version: 0.1.0. Recorded date: 2026-10-10 (America/New_York).

This document specifies a computational pilot and a proposed later confirmatory
study. It contains no alignment results. The reduced computational pilot checks
the pipeline; it is not a paper replication, a powered contamination study, or
evidence about universal representation convergence. Confirmatory and
model-choice phases have not started. `protocol.json` records the same design in
machine-readable form; it is a design record, not a runner configuration file.

## Questions and scope

1. With pretrained encoders fixed, does excluding known evaluation-source or
   cross-training-source overlap change retrieval beyond equally sized random
   removal?
2. After this cleaning, does the result extend to previously unused encoder
   families and harder semantic distinctions?

These questions must remain separate. Changing encoders during an overlap
comparison would confound the first question. Multiple alignment seeds measure
variation of the aligner conditional on fixed embeddings, not variability of
pretraining or generalization across model families.

Source baseline: upstream commit
`fdfad84ea488f29702a61a455be71622c4ff5d49`. The experiment includes local control
code beyond that commit. Each run must record its actual source hashes, Git
state, environment, embedding hashes, exact selected rows, and manifest hashes.
The baseline commit alone does not identify the executed experiment.

## Phase 0: source metadata audit and masks

Join COCO image IDs, Visual Genome image metadata, and SPC paragraph item IDs.
Retain the cached embedding row order. Store the source archives' SHA-256 values,
source identity mapping, and every exclusion mask. Unknown COCO mappings remain
unknown; a missing mapping does not prove that two images differ.

The current cleaning scope is **known metadata source identity**. Image bytes,
pixel hashes, perceptual near-duplicate matches, and encoder pretraining corpora
have not been audited by this protocol. The name `fully_dedup` is a short arm
identifier, not a claim that every possible duplicate has been removed.

All arms retain the same COCO training-image bank. Exclusions apply to SPC
paragraph rows through their underlying source identity. Preserve repeated
paragraph descriptions of the same source together in random controls. The
group-preserving sampler orders groups by a seeded SHA-256 value, skips groups
that exceed the remaining row budget, and fills an exact row count. This is not
a uniform sample of all possible row subsets. Record this selection mechanism;
do not describe it as uniform random row deletion.

| Arm | Manifest variant | Caption population | Purpose |
| --- | --- | --- | --- |
| O | `original` | Original SPC rows | Deliberately overlap-permitting diagnostic baseline |
| C | `validation_clean` | O excluding source IDs in COCO validation | Evaluation-source exclusion |
| D | `fully_dedup` | C additionally excluding source IDs in COCO training | Known cross-training-source exclusion |
| Rv | `random_drop_validation_size_matched` | Group-preserving random subset of O, with C's row count | Control for the sample-count change from O to C |
| Rp | `random_drop_size_matched` | Group-preserving random subset of C, with D's row count | Control for the sample-count change from C to D |

O and Rv may contain evaluation-source overlap and must remain labelled
diagnostic-only. C, D, and Rp must pass the metadata evaluation-holdout check.
D must also pass the known cross-training-source-disjointness check. The full
manifest population and the exact rows selected for a capped pilot must both be
audited; a full-population audit does not describe every property of a subset.

Future image-hash and near-duplicate analyses require separate versioned masks,
frozen thresholds, and outcome-blinded match auditing. Do not silently fold them
into the current identity-only comparison. Repeated generic caption text alone
does not establish a shared underlying scene.

## Phase 1: bounded computational pilot

Use cached `dinov2_vit-b14@224_mean` and `mpnet` embeddings. Freeze their cache
hashes and item-reduction rules. This phase does not train or select encoders.

| Setting | Pilot value |
| --- | --- |
| Cluster count | 10 |
| Initialization repetitions | 3 |
| Assignment batch size | 512 |
| Refinement iterations | 10 |
| Training row cap | 4,096 per modality per arm |
| Common primary clean-query cap | 2,048 |
| Common all-population diagnostic-query cap | 2,048 |
| Common original-SPC-exposed diagnostic-query cap | 2,048 |
| Common validation gallery | Complete 40,504-item gallery |
| Subset seed | 20261010 |
| Fit seeds | 0, 1, 2 |
| Random-control selection seed | 0, fixed for both Rv and Rp |
| Intended fits | 5 arms x 3 fit seeds = 15 |

Select only one mask for each random-control arm in this phase. Using all stored
selection-seed masks with all fit seeds would produce a different experiment.
Keep selection, subset, and fit randomness separate. Record effective row counts
and any effective batch-size adjustment. Do not adapt settings after looking at
retrieval scores. A necessary software or resource change creates a new named
pilot revision and retains failed and superseded records.

The cap can equalize arm sizes and change their composition. Therefore these
15 reduced runs establish execution feasibility, mask application, determinism,
metric correctness, artifact completeness, and runtime only. Their numerical
scores do not estimate the full-population cleanup effects defined below.
Synthetic smoke tests are software checks, not benchmark observations.

The pilot executed in an isolated Linux VM using CPU PyTorch and the original CPU aligner. Actual runtime versions are recorded with the evidence; the environment is compatible, not identical to the paper’s pinned environment.

## Evaluation: common gallery and fixed query strata

All arms in a comparison use exactly the same query IDs and gallery IDs. The
pilot uses three separately drawn, fixed query samples: up to 2,048 clean queries
for its primary metric, up to 2,048 queries from the complete validation
population for an all-population diagnostic, and up to 2,048 queries from the
original-SPC-exposed population for an exposed-query diagnostic. The exposed
sample is drawn independently from its own population; it is not restricted to
exposed members of the all-population sample. The three draws use
`subset_seed + 41`, `subset_seed + 43`, and `subset_seed + 47`, respectively.
All three query-index sets remain fixed across arms and fit seeds, and all use
the complete frozen 40,504-item COCO validation gallery. The later full study
removes query caps while retaining that same gallery. Record all three
query-index sets and the gallery-index set.

Define query strata once, using **the original SPC source population**, before
fitting any arm:

- `original_spc_overlap`: a validation query whose source appears in original SPC.
- `never_known_spc_overlap`: a validation query with no known original SPC source
  match. This wording reflects metadata coverage rather than guaranteed absence
  of near-duplicates.

The primary query population is `never_known_spc_overlap`, represented by the
manifest's `validation_clean` indices. Consequently, its cleanup contrast asks
about changes on queries without known original SPC overlap. It does not measure
the direct advantage on exposed queries. Evaluate that direct effect using the
separately sampled original-SPC-exposed diagnostic queries, alongside the
all-population diagnostic sample. Stability of the primary clean-query metric
cannot rule out direct evaluation leakage.

For each stratum, retain the same full comparison gallery. Do not remove gallery
items when computing a stratum's score. Save per-query ranks and stratum counts
so a small or absent stratum remains visible. Do not redefine strata after
cleaning: that would make the exposure group disappear by construction.

If an implementation also restricts the gallery to `validation_clean`, report
it as a separate secondary metric with its own gallery size; it is not the same
estimand as clean-query evaluation against a common full gallery.

The primary retrieval direction is image to caption. Compute FOSCTTM as
`(midrank - 1) / (gallery_size - 1)`, excluding the true partner from the
denominator and assigning half credit to tied nonpartner scores. Aggregate over
queries, with no selection of best seeds. Report Recall@1/5/10 with declared tie
handling and true-match mean/median rank as secondary summaries. The primary
paper-compatible caption-item reduction remains fixed across arms.

Center and normalize using each arm's own selected training rows only. Never
estimate those preprocessing parameters from evaluation embeddings. CKA, if
reported, is a paired-data diagnostic; it cannot select settings or successful
runs in this experiment.

## Phase 2: proposed full confirmatory overlap comparison

This phase is **not started and not yet locked as a confirmatory registration**.
Publish a versioned final analysis plan before running or unblinding its results.
The computational pilot must not choose the primary hypotheses, effect margin,
model pair, solver settings, or sample size from its observed effect estimates.
Any design amendment must be dated, justified, and identify which outcomes were
already visible. There is no retroactive preregistration.

Proposed fixed settings are the upstream paper profile: 30 clusters, 30
initialization repetitions, batch size 10,000, 100 refinement iterations, no
training cap, and no evaluation cap. Use the same cached encoder pair as phase 1.
Proposed budget is 20 paired fit seeds, 100 through 119, with three independently
recorded random-control selection seeds, 1000 through 1002. O, C, and D each run
once per fit seed; each random control runs for all three selection masks per
fit seed. This is 180 fits, not 300, and remains an unstarted proposed budget.

No achieved-power claim follows from that budget. Before the confirmatory lock,
assess its precision using external evidence or an explicitly specified
conservative variance assumption. Resource feasibility may change the proposal
before lock; reduced run counts require a prospective amendment and an honest
precision limitation. Do not stop when a desirable result appears, discard
unsuccessful seeds, or replace unavailable masks selectively.

Let `F_A` be an arm's FOSCTTM. The two proposed primary effects are:

1. `delta_validation = F_C - mean_selection_masks(F_Rv)`.
2. `delta_pairing = F_D - mean_selection_masks(F_Rp)`.

These primary contrasts use the frozen clean-query population. Prespecify and
report the same contrasts over all queries and the originally exposed stratum
as diagnostic secondary effects for direct leakage assessment.

Calculate each contrast within fit seed before aggregation. Positive values mean
targeted exclusion hurt more than its sample-count control. Also report the raw
changes `F_C - F_O` and `F_D - F_C`. These contrasts control the removed row count;
they do not automatically control changes in class mix, caption style, or group
size distribution. Any later class/length-matched removal control is an
additional preregistered sensitivity analysis, not a silent replacement.

The proposed practical effect margin is 0.02 FOSCTTM and is **provisional**. It
must be accepted or changed on scientific grounds before the final lock, without
using pilot retrieval effects. Failure to reject zero does not establish
equivalence; excluding a material deterioration requires an appropriately narrow
interval relative to the locked margin.

Report paired seed-level effects and uncertainty. Distinguish variation over
solver seeds and random-control masks from sampling uncertainty over evaluation
queries. Resample queries by source/duplicate group when reporting query
uncertainty; retain the fixed gallery and explain that these intervals are
conditional on that gallery. Do not count queries, seed replicates, or readout
variants as independent pretrained models. Use Holm correction for the two
primary confirmatory hypotheses. The final registration must specify the exact
interval/test implementation and its assumptions before results are unblinded.

## Phase 3: held-out encoder families and semantic controls

Not started. Select at least two new vision families and two new language
families before observing their alignment results. Change architecture or
training objective rather than merely size or pooling, run all four crossings,
and report failures. Freeze cleaning, hyperparameters, preprocessing, and
evaluation first. Fully paired multimodal encoders may be positive controls,
but cannot count as evidence for independently trained unimodal convergence.

Report family-level results. Two vision families crossed with two language
families are a small external-validity probe, not a representative sample of all
models. Do not use the original benchmark to select whichever new family aligns
best. Family-held-out prediction is required for any new claim that CKA predicts
success beyond the development families.

Additional proposed controls are independent category-frequency changes,
missing-category conditions, and evaluation restricted to within-category
candidate distinctions. Hard negatives should vary counts, attributes, spatial
relations, and agent/patient roles while holding broad categories fixed.

Use a category-oracle/random-within-category baseline to distinguish coarse
taxonomy from instance correspondence. Permuting evaluation target identities
calibrates chance. Shuffling training row order is not a negative control for an
algorithm whose input consists of unordered sets. CKA and top-principal-component
ablations are diagnostic and must not replace retrieval outcomes.

## Phase 4: controlled encoder training

Not started. A proposed first factorial study varies two objectives and two
disjoint training corpora with three pretraining seeds each, holding architecture
fixed: 12 trained encoders. A second architecture is a separate replication.
Match data volume and compute, report downstream competence, and preregister the
opposite-modality reference encoder. This separates training variability from
alignment variability; it still does not sample all architectures or worlds.

## Interpretation boundaries

- Deterioration beyond random-drop controls supports a contribution from the
  particular known overlap removed, beyond its row-count effect.
- Tight stability after cleaning supports robustness to this measured overlap
  in these fixed checkpoints. It does not establish pretraining decontamination.
- Success in previously unused families expands the supported population; it
  does not prove universal convergence.
- Category-level success without within-category discrimination supports coarse
  semantic transfer and leaves fine correspondence unestablished.
- A reduced pipeline pilot cannot settle any of these scientific conclusions.

## Sources

- [Paper, method and evaluation details, sections 3 and appendix B](https://arxiv.org/html/2610.09411v1)
- [Project method explanation](https://dominik-schnaus.github.io/unpaired-rosetta/#method)
- [COCO metadata archive](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
- [Visual Genome image metadata](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)
- [SPC paragraph metadata](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/paragraphs_v1.json.zip)
