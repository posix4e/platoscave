# Change the models; keep the five photographs

The user asked to compare models on the same five photographs first. This
follow-up crosses three image encoders with two language encoders. It preserves
the original DINO/GloVe result and reports every pairing.

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

The final `protocol.json` and `release-lock.json` must be published before any
additional features of the five study items are computed. No checkpoint weights
are stored in GitHub. Acquisition receipts identify the official URLs and exact
bytes needed to recreate the runtime.

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
