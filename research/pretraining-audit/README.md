# Pretraining provenance and caption exposure

Recorded 10 October 2026. This is a provenance screen and a released-data audit,
not an encoder-training experiment or a preregistration of completed results.

Existing pretrained models can be used if their histories meet the experiment's
requirements. Training from scratch is one way to control those histories; it is
not a requirement in itself. None of the candidates below has yet passed the
checks needed to be called an uncontaminated COCO/SPC control.

## A documented problem with the current language encoder

The original implementation's `mpnet` name resolves to
`sentence-transformers/all-mpnet-base-v2`. Its
[model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md)
and [training configuration](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/e8c3b32edf5434bc2275fc9bab85f82640a19130/data_config.json)
list COCO and Flickr caption data. The configuration names 828,395 COCO records.

We downloaded the original
[COCO caption component](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/coco_captions.jsonl.gz)
and checked its caption groups against the COCO reference metadata. Results are
in `caption-exposure-summary.json`; the complete report, including candidate
source IDs, is in `caption-exposure.json.gz`.

All 828,395 records have source identifiers corresponding to 82,783 original
COCO training images. Every caption agrees with its identified source after the
recorded normalization. No source identifier belongs to original COCO validation.
Some generic captions also match validation captions, so counting isolated text
matches as validation-image exposure would give a misleading result. This audit
does not clear the rest of the model's training mixture.
The 11 validation-source candidates that look unique under raw-string matching
become ambiguous after normalization; they do not establish validation-image
exposure.

The distinction matters: released training-component membership is not an
optimizer sampling log. The cached Rosetta embeddings also do not establish the
precise historical weight revision used to generate them. These findings do not
show that the aligner received paired image-text labels, nor measure how much
pretraining exposure affected its results.

Removing overlapping SPC rows during alignment cannot undo encoder pretraining.
The 15 completed fits remain pipeline checks, not evidence against this concern.

## Candidate decision

| Candidate | What its published history supports | What remains unresolved |
| --- | --- | --- |
| Original DINOv1 ViT-B/16 | ImageNet-only image self-supervision; its teacher follows its own student | ImageNet versus benchmark duplicates and exact checkpoint hashes |
| Original MAE ViT-B/16 | ImageNet-1K pixel reconstruction without an external teacher | The same source-overlap checks and checkpoint hashes |
| Pythia-160M base | Inspectable text pretraining and training order, without documented multimodal fine-tuning | Benchmark text in the Pile, tokenizer lineage, and sentence-representation competence |

See `candidate-audit.json` for primary references, exact candidate revision pins,
the admission requirements, and remaining uncertainties. DINO and MAE share
ImageNet; comparing them does not vary training-data provenance independently.
Pythia is a base language model, not a ready-made sentence retrieval encoder.

## What would justify starting a stronger control

One route is to compare the recorded pretraining corpora with the benchmark,
freeze exact- and near-duplicate exclusion rules, and describe negative findings
as absence under those particular checks. An incomplete audit is not a clean
bill of health.

Another route is new photographs of newly created scenes and new descriptions
after frozen checkpoints, with separate scenes for fitting and evaluation.
Chronology can exclude exposure to those exact new items. Older views of static
scenes, similar images, repeated caption text and shared training supervision
still need separate treatment. A procedural scene experiment would concern its
constructed world and would not replace the natural-image study.

Choose checkpoints and feature extraction before seeing alignment scores. Keep
unimodal competence checks and a supervised alignment positive control separate
from the unpaired experiment. Failure of a weak encoder or solver is not, on its
own, evidence that corresponding information is absent.

## Reproduce the caption audit

Use Python 3.10 or later; there are no third-party dependencies. Run inside the
experiment VM, supplying the original released caption file and
[Karpathy caption metadata archive](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip):

```sh
python3 -m unittest -v test_caption_exposure.py
python3 check_caption_exposure.py \
  --corpus coco_captions.jsonl.gz \
  --karpathy-zip caption_datasets.zip \
  --output caption-exposure.json
```

The report hashes both inputs and the executed script. Exact raw strings and
normalized strings are counted separately. Original `train2014` and `val2014`
come from source filenames, not the different Karpathy train/restval/val/test
partition. Shared generic caption text does not by itself identify an image:
whole-group candidate intersections and ambiguous cases are reported separately.
Numeric `guid` values are also cross-checked against source IDs and all captions
in the same record; their names alone are not treated as proof of provenance.
No raw image or caption corpus is redistributed here.
