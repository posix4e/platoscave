# What would count as a cleaner example?

Recorded 10 October 2026, before any benchmark alignment result from the frozen
DINO/GloVe experiment. No scores from the new candidates below have been seen.

The strongest next comparison is **original DINOv1 with original ELMo on the
same EPIC-KITCHENS-55 recordings**. ELMo's publisher identifies its original
training corpus as the 1 Billion Word Benchmark; that corpus's publisher traces
it to WMT 2011 news. This gives an older, bounded language source and a second
kind of text representation. Contextual representations may help, but that
must be measured rather than assumed.

| Addition | What it tests | Evidence still needed before running |
| --- | --- | --- |
| Original ELMo, 1B-Word model | Dependence on GloVe averaging and its training source | Full weight hash, fixed pooling, inference order/batch invariance |
| Original ImageNet MoCo-v2 ResNet-50 | Dependence on the DINO objective and transformer backbone | Full weight hash, checked feature extractor and competence control |
| Original ImageNet MAE ViT-B/16 | Different image objective within the same ViT-B/16 backbone family | Full weight hash, fixed representation and competence control |
| HD-EPIC, recorded in 2024 | A later, separately collected benchmark | Audited sample and human checks of the model-assisted narration transcripts |
| Dated Ego-Exo4D subset | A further collection with human-written action descriptions | Licensed metadata, actual capture dates and a fixed visibility rule |

The three image encoders crossed with GloVe and original ELMo form a useful
six-combination extension. This shortlist is chosen from provenance and method
differences, not successful alignment. An execution protocol remains to be
fixed before running that extension. Every selected combination, including
failures, should be reported. The existing DINO/GloVe experiment remains fixed.

The word **clean** needs a boundary. Under the documented training histories,
these old corpora exclude the particular later recordings. They do not exclude
shared concepts, repeated ordinary phrases, similar scenes, or ImageNet's
human choice of categories. We have not independently reconstructed the
original training runs. None of these examples alone proves universal
representations or measures contamination's causal effect in the original
Rosetta result.

For the strictest control over new test items, first lock the model bytes, then
collect fresh photographs and independent human descriptions under a fixed
sampling plan. That requires new evaluation data, not training encoders from
scratch. It still leaves the separate question of human curation in pretraining.

Evidence and rejected alternatives are recorded in the
[ELMo review](further-provenance-review.md),
[vision review](vision-expansion/README.md), and
[dataset review](alternative-datasets-review.md).

This record admits candidates for further checks. It is not a completed
alignment experiment, and availability checks are not weight-byte verification.
