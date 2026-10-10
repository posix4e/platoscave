# Change the models; keep the five photographs

The user asked to compare models on the same five photographs first. This
follow-up crosses three image encoders with two language encoders. It preserves
the original DINO/GloVe result and reports every pairing.

## Completed comparison

Every prescribed cell ran successfully. Each has a unique best assignment.
The rank is the human correspondence's position among the 120 possible
assignments, ordered by the frozen distance cost; rank 1 would be best.

| Image model | GloVe: correct; human rank | ELMo: correct; human rank |
| --- | --- | --- |
| DINO | 1/5; 94/120 | 0/5; 26/120 |
| MoCo-v2 | 1/5; 91/120 | 0/5; 40/120 |
| MAE | 1/5; 99/120 | 0/5; 24/120 |

DINO and MoCo select the same correspondence for each language model. MAE
selects different correspondences despite the same correct counts. Replacing
GloVe with ELMo changes the selected matches for all three image encoders.
Thus model choice changes the answer, but none of these six choices recovers
the five human correspondences under this rule. That is the observed result,
not evidence that all pretrained models fail or that Rosetta fails.

One correct match is the expectation for a uniformly random bijection of five
items. It is not a significance test or an assertion of equivalence to chance.
The six comparisons reuse five photographs from one outing; they are not six
independent datasets. The 120 assignments per cell are an exhaustive search,
not 120 observations. Short labels also leave meaning ambiguous: for example,
“Basketball” names a photographed scene as well as a possible object or sport.
The labels were preserved verbatim rather than rewritten after scoring.

MAE's CLS distances range from 0.0054 to 0.0397, compared with 0.4923–0.9231
for DINO and 0.3655–0.8125 for MoCo. No feature rows are exact duplicates and
all distance matrices have nonzero edge variance. These checks describe the
geometry; they do not select a different readout or scoring rule. Raw minimum
costs across model pairs are not a model-quality ranking.
Uniform positive rescaling or shifting of a graph's edge distances would not
change the ordering of complete assignments under this objective. Distance
concentration alone therefore does not explain MAE's chosen correspondence.
Four labels have one token and one has two; this is also a narrow test of a
contextual language model. Low scores establish neither clean pretraining nor
contamination. Those claims require separate evidence about training and time.

The complete [evaluation](results/evaluation.json),
[720 assignment records](results/assignments.json),
[public distance matrices](results/blind-panel.json) and
[run evidence](results/run-evidence.json) retain every result. The original
DINO/GloVe features and numerical result are unchanged.

## Design

| Image features | GloVe 6B, 300d | Original ELMo, top contextual layer |
| --- | --- | --- |
| Original DINOv1 ViT-B/16, CLS | Original reference | Language replacement |
| Original MoCo-v2 ResNet-50, pooled backbone | Image replacement | Both replacements |
| Original MAE ViT-B/16, CLS | Image replacement | Both replacements |

The photographs, labels, image crop, distance definition and matching objective
stay fixed. Each pairing examines all 120 bijections. A separate evaluation
reads the human correspondence after all six selections have been saved.
Neither encoder training nor a learned cross-modal map is involved. This tests
the model sensitivity of a small distance-graph comparison; it does not run the
Rosetta algorithm or measure accuracy on unseen examples.

The panel was shortlisted by provenance before the first result. This extension's
rules were written **after** that result. It is exploratory, and no winning model
will replace the original result. Changes between cells cannot isolate one cause:
the models differ in training objectives, corpora, architecture or representation.

## Chronology and privacy

The original DINO/GloVe files were fixed before the collector reports taking the
photographs. The other three checkpoint files were downloaded and hashed later.
Those additions therefore rely on their documented old training histories; they
do not inherit the original pair's pre-capture file-lock claim. A future collection
could follow the complete panel lock.

The five attachment files, private manifest and raw features remain private.
Public distance matrices and assignments permit replay of the matching
calculation. Public replay does not reproduce image feature extraction from the
private photographs.

## Exact encoders

MoCo uses the original ImageNet 800-epoch query encoder and its 2048-dimensional
average-pooled backbone output. Its projection head is excluded and its batch
normalization statistics remain frozen. The release identifies the exact
[checkpoint](https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/README.md).

MAE uses the original pretrained ViT-B/16 weights, the official unmasked
`models_vit` forward path and final-layer-normalized CLS. The original
[linear-probe entry point](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_linprobe.py)
fixes this readout. There is no masked-token shuffle, decoder or fine-tuned head.
The same original image preprocessing helper supplies both additional models.
For MoCo, the common bicubic resize differs from its historical linear-evaluation
script's bilinear resize; it is fixed here to keep the image input constant.

ELMo uses the original 1 Billion Word Benchmark checkpoint, not the 5.5B or a
task-trained model. Official AllenNLP inference reads the final contextual layer
directly, resets recurrent state for every label, excludes boundary and padding
positions, and averages the remaining tokens before L2 normalization. No learned
layer mixture is used. The
[original catalog](https://github.com/allenai/allennlp-website/blob/40d800e5e6365520e8ae52f9d0302549fc11f1d5/elmo.html)
distinguishes these checkpoints.

## Code and checks

- `panel.py` binds the five inputs, extracts the two additional image feature
  sets, assembles blinded matrices, selects all six assignments and then scores
  their correspondence. Its calculator is the unchanged original `solve.py`,
  verified by hash. The calculator's eight-field input schema is adapted only
  at the interface; this follow-up keeps its own protocol identity and evidence.
- `vision_features.py` strictly loads both image backbones. Official MAE source
  and the original timm wheel are pinned under `vendor/`. The compatibility alias
  for the removed `torch._six.container_abcs` changes no model arithmetic.
- `elmo_features.py` uses a separate CPU Python environment and pinned AllenNLP
  sources. It accepts JSON containing `sample_ids` and `captions`, and writes an
  NPZ plus a provenance sidecar. The original GPU environment is unchanged.
- Readiness checks use generated images and artificial sentences only. They
  check repeatability, order, batching, state reset, output shape and invalid
  input guards. These checks are not research outcomes.

The final `protocol.json` and `release-lock.json` were published in commit
[`7fcbc370`](https://github.com/posix4e/platoscave/commit/7fcbc370de6c4e7ca486a9d20cb47e7ec9568157)
before any additional features of the five study items were computed. No
checkpoint weights are stored in GitHub. Acquisition receipts identify the
official URLs and exact bytes needed to recreate the runtime.

Run each stage in order, with explicit local paths (`--help` lists them):

```sh
python panel.py captions ...
python panel.py vision --encoder moco ...
python panel.py vision --encoder mae ...
python elmo_features.py --captions ... --options ... --weights ... --out ...
python panel.py assemble ...
python panel.py solve ...
python panel.py evaluate ...
```

All outputs require unused destinations. Preserve a failed run and its evidence;
do not silently overwrite it. The frozen plan reports every prescribed cell,
including failures and ties. It adds no models, views, label variants or scoring
rules in response to the results.

## Verification and public replay

The [independent audit](results/independent-audit.json) passed 3,023 checks.
Its separate implementation recomputes the five distance matrices from saved
features and all 720 assignment costs, correlations and selections. It verifies
the original result, fixed labels, model/source receipts and provenance hashes.
[Five altered-copy checks](results/independent-audit-validation.json) were
rejected. This audit was written after the results: it does not preregister the
experiment, independently rerun the encoders, authenticate capture times or
audit the pretraining corpora.

Anyone can reproduce the matching calculation without the private photos.
From the repository root, choose an output directory that does not exist:

```sh
python3 research/model-choice-controls/panel.py solve \
  --input research/model-choice-controls/results/blind-panel.json \
  --core research/five-photo-pilot/solve.py \
  --out /tmp/model-choice-replay
```

This uses Python's standard library. Our [public replay check](results/public-replay-check.json)
reproduced the assignment file byte for byte and every selected result,
excluding its new recording timestamp. The independent feature audit also
requires NumPy and access to the private artifacts; public replay cannot
verify photo-to-feature extraction.
