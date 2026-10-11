# Does the result depend on the chosen models?

**Draft protocol. No added COCO features or model-comparison fits have been run.**
The [five-seed reproduction](../rosetta-replication/README.md) is running
separately. This directory prepares the next experiment; its draft files cannot
authorize extraction or fitting.

The question is whether the same unpaired alignment procedure succeeds across
these fixed image and language pipelines. We keep the data, splits, algorithm,
seeds and evaluation fixed. We replace one pipeline at a time, then examine their
crossings. We also measure what two fitted maps can recover when given correct
training pairs, and what happens when those pairs are shuffled.

This is a selected, exploratory panel. Its models were chosen earlier and their
[five-photo results](../model-choice-controls/README.md) are already known.
Fixing this larger experiment before running it prevents further choices from
following its scores; it does not turn the model shortlist into a random sample
or erase those earlier observations.

## The complete panel

| Image pipeline | MPNet | GloVe 6B, 300d | Original ELMo 1B |
| --- | --- | --- | --- |
| Released DINOv2-B/14 mean | Reference replay | Language replacement | Language replacement |
| Original DINOv1-B/16 CLS | Image replacement | Both replacements | Both replacements |
| Original MoCo-v2 R50 pooled backbone | Image replacement | Both replacements | Both replacements |
| Original MAE-B/16 CLS | Image replacement | Both replacements | Both replacements |

Each of the twelve cells uses all five original seeds and all five methods:
Rosetta, paired orthogonal, shuffled-pair orthogonal, paired linear and
shuffled-pair linear. This means **60 new Rosetta fits and 240 control fits**.
The reference replays are necessary because the original runner saves aggregate
scores without the fitted maps or individual retrieval ranks. These replays
retain their own identity and never replace the original reproduction.

The first five cells replay DINOv2/MPNet. Their retrieval scores must agree with
the five untouched reproduction results within the fixed absolute tolerance of
`1e-10` before any expanded cell can fit. This is an implementation check, not
a statistical threshold or an authors' tolerance. A failure remains recorded
and stops expansion while we investigate it.

## What stays fixed

Use all 82,783 COCO training identities. For each original seed, the authors'
permutation assigns 41,391 identities to the image bank and 41,392 different
identities to the language bank. Every cell uses that same split. Rosetta sees
no correct pairs. It retains 30 clusters, 30 initialization attempts, an
assignment batch cap of 10,000 and 100 refinement iterations. No reduced bank,
different seed or shorter run may stand in for this experiment.

Evaluate every method against all 40,504 COCO validation identities and the
same complete text gallery. Preserve the original independent shuffles,
4,096-query evaluation blocks and average-rank tie rule. Record each query's
rank with its public COCO identity. The primary score is FOSCTTM: the fraction
of alternatives ranked ahead of the true text, counting ties by halves. Lower
is better. Mean rank is also reported, but is an affine restatement of FOSCTTM
for this fixed gallery, not an independent endpoint.

The authors' source remains unchanged. Fits use the exact reproduction's
locked Pixi runtime. The original fit context uses four Torch threads; the
process default restored for evaluation is sixteen, with sixteen inter-op
threads. Assignment workers and threads remain four each. Run one cell at a
time. More memory or a documented scheduling change is preferable to silently
shrinking the experiment.

This panel evaluates retrieval only. The original reproduction retains its
three classification evaluations; their results do not become twelve model
comparisons without the corresponding feature banks and a separate protocol.

## What a replacement changes

The reference uses the authors' released BF16 tensors. Preserve their exact
caption reduction in BF16 before converting to FP32. The five additions use
FP32 features and frozen readouts. DINO uses its final CLS token, MoCo its
query-backbone average pool, and MAE its unmasked final CLS token. Each added
vision model receives the authors' square 224-pixel transformation, with fixed
AVX2 dispatch, batch size one and no alternative crop selected after scoring.
This shared recipe differs from some models' historical evaluation recipes.

GloVe averages known tokens using the frozen tokenizer and reports omitted
tokens. ELMo uses its final contextual layer, resets state for every caption,
and excludes boundary and padding tokens. Every original human caption is
retained in annotation order, normalized separately and stored in its original
slot with NaN padding to seven slots. The authors' reduction then averages the
available caption vectors and normalizes the item vector. The added caption
banks undergo this reduction in FP32. No generated or rewritten descriptions
are introduced.

Consequently, a cell changes a **complete encoder pipeline**: weights,
architecture, training history, readout, precision and possibly runtime. The
comparison cannot isolate any one of those causes. The current square transform
also does not prove which historical pixels produced the released tensors.
Native-preprocessing alternatives are outside this panel; any later sensitivity
study requires a separate specification and must retain these results.

## What the paired controls can tell us

The original orthogonal and linear maps receive all 82,783 correct training
pairs. Their centering means still come from Rosetta's two original disjoint
banks. Thus the controls receive missing counterparts as well as pair labels;
they have more information than Rosetta. A difference cannot be attributed to
revealing pair labels alone.

For each seed, derive one independent training permutation from the protocol
identity and seed. Reuse it across all cells and both shuffled controls. Keep
and report accidental fixed points; do not replace a permutation because of
its score. The shuffling never touches validation correspondences and never
changes the fitting banks. Shuffling the row order of Rosetta's already
unpaired banks would not provide this control.

The linear control uses the authors' unchanged default pseudoinverse. Its
default singular-value cutoff depends on matrix size and FP32 precision. Report
that cutoff, a separately computed singular spectrum and the resulting rank
diagnostic; do not describe it as an unrestricted full-rank linear map. The
diagnostic neither changes the fitted weights nor supplies a tuning rule.

If correct-pair maps retrieve much better than their shuffled controls while
Rosetta does poorly, the pipelines contain correspondence that a supplied map
can recover under this task. That narrows the question about unpaired recovery,
subject to the controls' information advantage. If both paired maps do poorly,
the experiment has not established that the representations lack corresponding
information: pooling, map capacity or the task itself may be limiting.

## Reporting before interpretation

Publish every prescribed cell, method and seed, including failures. Report
individual scores, arithmetic means and sample standard deviations (`ddof=1`).
No best seed, winning model, success threshold or significance claim is planned.
Five seeds measure variability of this procedure on fixed data; they do not
create five independent datasets or independently trained encoders.

For each seed, report the five single-replacement differences from the reference
and the six crossing interactions. An interaction is the crossed score minus
the two corresponding single-replacement scores plus the reference score.
Also report each correct-pair control minus its own shuffled control, and
Rosetta minus each correct-pair control. Positive differences in FOSCTTM mean
worse retrieval. These are descriptive contrasts, with no post-hoc selection
or claim of a pure architectural or pretraining effect. Incomplete contrasts
remain unavailable with the missing attempt identified; they are not averaged
over a convenient subset of seeds.

Each extraction and cell reserves a fresh attempt directory before work begins.
Progress and failures survive an exception. An extraction needs a verified final
`COMPLETE.json`. A cell's verified terminal manifest may preserve a complete
method even if another method failed. An aborted cell without that manifest, or
a method whose numerical diagnostic failed, retains its observed scores but
cannot contribute a complete observation. All five reference cells must finish
and agree with the untouched baseline before expanding the panel.
Partial or conflicting terminal files cannot establish completion or authorize
a retry, including files that appear while the summary is being checked.

Store every cell attempt under
`cells/{vision}--{text}/seed-{seed}/attempt-{01..99}`. A reviewed retry after an
external execution or input failure uses the next counter, with an in-directory
`retry-review.json` that binds the failed predecessor and states the reason.
Its terminal manifest must also bind this record. Never retry after a complete
cell or because of a poor score. Keep the earliest eligible method by counter;
any later eligible repetition must have identical scalar metrics and identical
per-query ranks when sorted by COCO ID. A disagreement, undocumented retry or
multiple complete cells makes the affected comparison unavailable. A numerical
change requires a separately identified amended protocol.

The summary checks every directory in the declared experiment tree. It verifies
saved ranks, metrics and the original ID/randomness order, but does not rescore
the fitted maps or independently prove the stated cause of a failed attempt.

## Release order and remaining work

The draft [protocol](protocol.json) fixes the proposed scientific choices.
Before any added corpus extraction, complete the reference audit, finish the
independent runner review, and publish the complete scientific, code, model,
runtime and input locks. No draft is executable. After extraction, verify all
fourteen model/split feature entries and bind their bytes in a separate feature
lock. This second lock records generated data; it cannot alter the earlier
scientific choices. Then run the five reference replays before the expanded
cells. Keep all published releases immutable.

The implementation has five stages:

- `acquire_coco_images.py` acquires the public image archives, checks every ZIP
  member and extracted file, and records their source and byte identities.
- `extract_features.py` requires the final extraction lock before any corpus
  inference. It preserves the fixed encoder helpers, readouts and row order.
- `build_feature_lock.py` inventories the four released tensors and ten completed
  extractions. It invokes the runner's full input checks before publishing the
  feature lock; machine paths go in a separate private file outside the repo.
- `run_cell.py` checks the complete release and runs one model-pair/seed cell.
  It saves maps, ranks and individual method outcomes. Detailed machine logs
  remain separate from public results.
- `summarize_panel.py` accounts for all 300 method slots, checks ranks and
  retry records, and reports all 103 prescribed contrast series. It keeps
  incomplete comparisons visible and requires all five seeds for an aggregate.

The synthetic tests cover original split and caption-reduction behavior,
randomness, ties, rectangular maps, correspondence shuffling, artifact identity
and retention of failures. They do not measure retrieval on COCO. In particular,
the feature-lock builder's artificial fixtures replace numeric tensor inspection
with a test stub; production has no such option and must pass the runner's real
checks on every tensor. The readiness records distinguish those checks from
actual experiment results.

The [pretraining audit](../pretraining-audit/README.md) still applies. None of
these additions is certified free of benchmark exposure. A separate COCO×SPC
source-removal study will examine known alignment-data overlap. A later fresh
collection, made after its own complete lock, must include its own fitting
population and separate final evaluation. Neither obligation is completed by
this model panel. The five private photographs and their metadata remain private
and are not inputs to this COCO experiment.
