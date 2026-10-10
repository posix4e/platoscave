# A pretrained pair with an older, bounded training history

Recorded 10 October 2026. This is a provenance finding and preparation for an
experiment. No alignment score has been measured for this pair.

The [continued search](further-search/README.md) adds original ELMo, ImageNet
MoCo-v2 and MAE, and later recorded datasets. It separates source evidence from
weight verification and leaves the first experiment's protocol unchanged.

**Original DINOv1 ViT-B/16, original GloVe 6B, and EPIC-KITCHENS-55 pass our
screen against direct pretraining exposure to the recorded episodes.** This
decision rests on the publishers' training and collection records. It is not a
claim that the encoders have no shared concepts or that every short description
is a new sequence of words.

## The chronological argument

| Component | Evidence | What we use |
| --- | --- | --- |
| Vision | DINO's original ImageNet pretraining recipe uses images without labels; its teacher follows its own student | Original ViT-B/16 backbone, without downstream fine-tuning |
| Language | Stanford identifies original GloVe 6B as Wikipedia 2014 plus Gigaword 5; the latter's newswire ends in 2010 | Original 300-dimensional vectors, with fixed averaging and no sentence fine-tuning |
| Evaluation source | EPIC provides a recording date for every video; all 432 dates are in 2017 | Human-recorded kitchen episodes and their subsequently collected human narrations |

The relevant dates are those of the **training corpora**. DINO was published in
2021; its documented ImageNet training set is the older 2012 release. The actual
2017 recordings could not be present in that release. The same chronology
excludes these episodes' later annotations as benchmark artifacts from GloVe's
documented text sources. A phrase such as "open drawer" could certainly occur in
older text; we do not treat that as evidence that the model saw this episode.

Primary sources: [DINO paper](https://arxiv.org/html/2104.14294),
[pinned training code](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py),
[ImageNet 2012](https://www.image-net.org/challenges/LSVRC/2012/),
[Stanford GloVe](https://nlp.stanford.edu/projects/glove/),
[Gigaword 5 catalog](https://catalog.ldc.upenn.edu/LDC2011T07), and
[EPIC collection paper, section III](https://arxiv.org/html/2005.00343).

## What was checked

We downloaded three annotation CSVs at EPIC repository revision
`63fa7d78eba41a8d2280374061a6b1bab3e7e41d`. The script verifies their SHA-256
digests, parses the mixed date formats, and rejects duplicate video records or
annotation references without recording dates. It ran successfully locally and
inside the experiment VM.

| Metadata check | Observed result |
| --- | ---: |
| Videos with parseable recording dates | 432 |
| Earliest / latest recording date | 2017-04-30 / 2017-11-21 |
| Action segments in the public training annotations | 28,472 |
| Training videos / participants | 272 / 28 |
| Missing recording-date references in either annotation table | 0 |
| Distinct exact action narrations | 8,793 |
| Action rows whose exact narration occurs more than once | 22,765 |

The paper describes collection as May–November; the metadata includes April 30.
Both place all recordings well after the selected training corpora. Dates remain
publisher-supplied records, not independently authenticated camera timestamps.

`epic55-provenance-audit.json` contains the metadata results and source hashes.
`artifact-lock.json` identifies the downloaded original DINO backbone, training
arguments, GloVe archive, and its 300d member by byte count and SHA-256. Hashing
files now makes our choice reproducible; it does not independently authenticate
their historical training. No weights, videos, or caption corpus are redistributed.

## What this example can establish

A successful alignment would show correspondence for this pair on episodes
excluded from its documented pretraining by chronology. It would weaken direct
benchmark memorization as an explanation for that result. This is a new control
on a different dataset, not an estimate of how much contamination affected the
original COCO/SPC result. Changing both models and benchmark cannot isolate that
causal effect.

Shared object categories, ordinary phrases, human curation of ImageNet, similar
scenes, and broader correlations between text and pictures remain. These are
different questions from whether these particular recordings were seen. We have
not independently reconstructed either original training run.

## Requirements before interpreting an alignment result

1. Freeze participant/kitchen-disjoint fitting, development, and evaluation
   partitions before scoring. Keep neighboring segments and recordings together.
   Keep paired identifiers and synchronized order out of the unpaired solver;
   document whether the fitting marginals contain corresponding episodes.
2. Fix preprocessing, dimensions, pooling, vocabulary treatment, solver settings,
   and seeds before held-out scores. Use the explicit ImageNet DINO backbone:
   DINO's example image-retrieval script defaults to different, Landmarks weights.
   Use original `glove.6B`, not the 2024 update or a Common Crawl release.
3. Report out-of-vocabulary tokens, empty vectors, and repeated caption/vector
   groups. Prespecify relevance with multiple valid matches. One-to-one retrieval
   would punish the model for indistinguishable descriptions of different clips.
4. Check object and action information separately. GloVe averaging loses word
   order; averaging image frames also loses temporal order. Motion-dependent
   failures cannot be read as absence of common semantic structure.
5. Include a paired alignment positive control with the same frozen features and
   separate training examples. If it fails too, a failed unpaired fit does not
   identify the unpaired solver as the cause. Report both outcomes.
6. Use a shuffled **evaluation correspondence** control with an appropriate
   grouping rule and report its purpose. Merely shuffling fitting-array order is
   not a null experiment for a method that only sees the same two marginals.
7. Disclose annotation-derived action windows or object crops. They can provide
   supervision in preprocessing even when the alignment stage never sees pairs.

These are design requirements, not a claim that a full protocol or these checks
have already run. This pair is a first control; it does not settle model-choice
robustness. A failure must remain reportable rather than trigger a search for a
pair chosen for its alignment score.

## Other candidates retained or set aside

| Candidate | Decision from this search |
| --- | --- |
| Wikipedia Sent2vec | Promising second language encoder: direct unsupervised Wikipedia training. An official March 2017 README links pretrained models, and an author recalls a 2015 dump. Exact corpus snapshot and historical byte identity remain less well documented than GloVe. |
| DOCCI main images | Do not clear the entire set from its release date. The collection includes older personal-library photographs, and the released schema lacks per-image capture dates. |
| DOCCI additional abstract/random images | Collection dates are later, but this set lacks the human descriptions needed here. |
| What'sUp controlled images | Authors made the photographs, but inspected sources do not establish capture dates. Its spatial task would also expose the limits of averaged word vectors. |

References: [Sent2vec repository](https://github.com/epfml/sent2vec),
[March 2017 README](https://github.com/epfml/sent2vec/blob/4f312cc30061fce475313c087799ff37183d1d4c/README.md),
[author's corpus clarification](https://github.com/epfml/sent2vec/issues/81#issuecomment-523423666),
[DOCCI paper](https://arxiv.org/html/2404.19753v1),
[DOCCI schema](https://google.github.io/docci/), and
[What'sUp paper](https://arxiv.org/html/2310.19785v1).

## Reproduce the preparation

Run in an isolated environment with Python 3.10 or later. Both scripts use only
the standard library. The model acquisition is about 1.21 GB compressed plus
metadata; GloVe's 300d member is hashed as a stream without extracting it.

```sh
python3 audit_epic55.py --output ./metadata
python3 fetch_models.py --directory ./models --expected-manifest ./artifact-lock.json
```

The first command fetches the pinned metadata and checks its expected hashes.
The second fetches the original model artifacts and compares their hashes with
our recorded manifest. It does not load models, generate features, or train.
