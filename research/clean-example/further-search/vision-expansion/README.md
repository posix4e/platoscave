# Two more image encoders with bounded training sources

Checked 10 October 2026. This is a source audit, not an alignment result.

Original **MoCo-v2 ResNet-50, 800 epochs**, and **MAE ViT-B/16** are useful
additions to the original DINO control. Their publishers describe image-only
pretraining on ImageNet-1K. Under those histories, the 2012 image collection
cannot contain the EPIC-KITCHENS recordings made in 2017.

| Candidate | Exact artifact | What changes from DINO ViT-B/16 |
| --- | --- | --- |
| MoCo-v2 | `moco_v2_800ep_pretrain.pth.tar` | Convolutional backbone and contrastive training |
| MAE | `mae_pretrain_vit_base.pth` | Same ViT-B/16 backbone family, trained with masked pixel reconstruction |

The [MoCo release at its 2020 revision](https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/README.md)
links the ImageNet pretrained checkpoint. Its training loop discards dataset
class labels; the loss targets identify augmented views of the same image.
Use the query encoder's pooled backbone features, before the projection head,
with frozen batch-normalization statistics. Do not load a downstream detector
or supervised linear classifier.

MoCo's paper also studies an Instagram training set. That is a different
training history and is not admitted by this ImageNet-only argument. The
current repository default branch has removed the implementation; the audit
therefore pins a source revision that still contains it. Both observations are
recorded in `sources.json`.

The [MAE paper, section 4](https://arxiv.org/html/2111.06377v3#S4), identifies
ImageNet-1K as its self-supervised training set. Its original
[pretraining release](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/README.md)
separates pretrained weights from supervised fine-tuned weights. Inspection of
`models_mae.py` confirms a pixel-reconstruction target without an external
learned tokenizer. Keep the original pretrained encoder; do not substitute
MAWS, a CLIP-aligned version, or a downstream fine-tuned checkpoint.

DINO learns positional embeddings; this MAE release fixes sinusoidal positional
embeddings. Their training recipes also differ. This is not a change of loss
function alone.

Before using MAE, fix whether the representation is CLS or pooled patch tokens
and test competence using the same frozen features. Choosing whichever works
best on the final test would defeat this control. A failed frozen-feature
alignment is not the same finding as failed supervised fine-tuning.

## Evidence and limits

`sources.json` records source URLs, revisions and SHA-256 digests.
`artifact-availability.json` records successful HEAD requests for both exact
weight URLs. **The weights have not been downloaded, hashed or executed in this
audit.** HTTP modification dates and ETags do not independently authenticate
historical training runs.

These candidates vary the model family or learning objective. They do not vary
the ImageNet corpus. ImageNet's selection of object categories, ordinary shared
concepts, and similar scenes remain possible explanations for agreement.
Keeping the same benchmark makes this a model-choice check, but does not
isolate the effect of architecture from every other training difference.

Pairing each vision candidate with both original GloVe 6B and original ELMo
would give six prespecified combinations including DINO. This is a proposed
extension, not an amendment to the frozen first experiment and not a claim
that any of the six align successfully. The current DINO/GloVe experiment must
still be reported regardless of its outcome.
