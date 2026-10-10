# Alternatives with evidence about recording dates

Reviewed 2026-10-10. This is a candidate review, not an amendment to the running EPIC-55 experiment. No experiment code, protocol, or live job was changed.

**The best additional dataset found is HD-EPIC.** Its authors explicitly date the recordings to 2024. **Ego-Exo4D is a strong second candidate for a date-filtered subset**, because its documented metadata identifies when each capture was recorded. Something-Something v2 and Charades remain unadmitted: commissioned recordings are promising, but the sources inspected do not establish their capture-date bounds.

The practical ranking is **EPIC-55 first for the current control, HD-EPIC first among additions, then a dated Ego-Exo4D subset**. EPIC-55's 2017 dates and human narration/annotation workflow are already audited. HD-EPIC gives a wider chronological margin but adds machine-assisted transcription. Ego-Exo4D has human-written descriptions, but its selected records still need an access-and-date audit. [Existing EPIC-55 audit](https://github.com/posix4e/platoscave/tree/main/research/clean-example).

The bounded claim is that exact recordings could not have appeared in the encoders' earlier documented corpora. It assumes those training histories are accurate. This does not exclude familiar objects, ordinary sentences, similar scenes, shared human concepts, or biases in dataset construction. It also does not establish a causal effect of contamination in the original Rosetta experiment.

| Proposed combination | Evidence for exclusion of exact recordings | Status |
| --- | --- | --- |
| Original DINOv1 ViT-B/16 + original GloVe 6B 300d, evaluated on HD-EPIC RGB and checked action narrations | ImageNet-2012 and Wikipedia-2014/older Gigaword precede the authors' stated January–July 2024 recording interval | Strong candidate for the same bounded chronology claim as EPIC-55; annotation assistance must be disclosed |
| Original MAE ViT-B/16 + original GloVe 6B 300d, evaluated on a dated Ego-Exo4D subset with human-written atomic action descriptions | MAE's original image-only ImageNet pretraining and GloVe's documented text sources precede captures whose actual `ds_recorded` falls after 2014 | Conditional candidate: audit licensed capture metadata and lock the exact checkpoint before admission |

The original DINO backbone is already pinned. MAE supplies an alternative image-only objective: masked pixel reconstruction on ImageNet. Use the publisher's **pretrained** paper checkpoint, converted from TensorFlow, without downstream fine-tuning. [MAE repository](https://github.com/facebookresearch/mae), [pretraining recipe](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md).

GloVe means original `glove.6B.300d.txt`: Wikipedia 2014 plus Gigaword 5, fixed averaging, no sentence fine-tuning. The newer 2024 releases are different artifacts. [Stanford GloVe](https://nlp.stanford.edu/projects/glove/).

HD-EPIC's appendix B.2 says: “Videos were recorded between January and July 2024.” It reports 156 videos from nine participants. Participants subsequently described their own actions. However, their spoken narrations were transcribed using Whisper, Qwen-Audio, and NeMo, then every candidate transcription was checked and corrected by human listeners. Thus the text is **human-origin, machine-assisted transcription**, not an annotation process without model assistance. The VQA recipe distractors include ChatGPT outputs; those are unnecessary for an image–narration experiment and should be excluded. Recipe text can also come from older cookbooks or websites, so recipe-source prose should not be treated as newly authored benchmark text. [HD-EPIC paper, appendices B.2, B.3, C.2.1 and D](https://arxiv.org/html/2502.04144v2).

I checked the publisher's small video index at repository commit `aa1f833febdfa14253328d11d0585c9eb1fe0b72`. Its 156 rows have 156 unique video IDs and nine participant prefixes. The date-shaped components of the filenames run from `20240202` to `20240625`, consistent with the paper's broader interval. **Filename parsing is corroboration, not independent proof of capture dates.** This check did not load the narration pickle, inspect all videos, or establish individual caption histories. [Pinned video index](https://github.com/hd-epic/hd-epic-annotations/blob/aa1f833febdfa14253328d11d0585c9eb1fe0b72/youtube-links/HD_EPIC_YouTube_URLs.csv), [annotation schema](https://github.com/hd-epic/hd-epic-annotations/blob/aa1f833febdfa14253328d11d0585c9eb1fe0b72/README.md).

```json
{
  "file": "youtube-links/HD_EPIC_YouTube_URLs.csv",
  "repository_commit": "aa1f833febdfa14253328d11d0585c9eb1fe0b72",
  "sha256": "5e316cbc6f780d5aa9cc6c69020df3d1a1ec3be06c55c471768ad8400ec3fe16",
  "bytes": 9336,
  "rows": 156,
  "unique_video_ids": 156,
  "participant_prefixes": 9,
  "minimum_filename_date": "2024-02-02",
  "maximum_filename_date": "2024-06-25"
}
```

Ego-Exo4D defines capture-level `ds_recorded` as the recording datestamp; its example `1689793200` is `2023-07-19T19:00:00Z`. These are **documentation examples**, not a completed dataset audit. Join selected takes to their captures and reject missing or insufficient dates. [Ego-Exo4D metadata](https://docs.ego-exo4d-data.org/data/metadata/).

Use its human-written atomic action descriptions. Their visibility annotations can prevent testing an ego view on an event visible only to an exo camera; any such filtering must be frozen and disclosed as annotation supervision. [Ego-Exo4D paper, section 12.C](https://arxiv.org/html/2311.18259v3). Access requires a dataset license and credentials; the full date audit remains pending. [Download instructions](https://docs.ego-exo4d-data.org/download/).

The rejected or weaker routes are:

- **Something-Something v2:** commissioned recordings, but no lower capture-date bound found. Release dates are insufficient. Order-sensitive labels also poorly suit averaged frames/words. [Paper](https://arxiv.org/html/1706.04261v2), [creators' announcement](https://medium.com/twentybn/something-something-v2-release-9107b4a8ce99).
- **Charades:** commissioned home recordings, but no capture-date bounds in the sources inspected. Requires dated records or author confirmation. [Paper](https://arxiv.org/html/1604.01753v1), [official project](https://prior.allenai.org/projects/charades).
- **Ego4D:** its FAQ promises recording dates, but the published schema lacks a clearly identified per-video capture-date field. Top-level `date` means metadata generation, not recording. [FAQ](https://ego4d-data.org/), [schema](https://ego4d-data.org/docs/data/metadata/).

Neither candidate solves the representation-competence problem: static image features and averaged GloVe discard temporal and linguistic order. A paired control and object/action-specific reporting remain necessary. These are additional checks, not grounds for abandoning or replacing the already frozen EPIC-55 experiment after observing its results.
