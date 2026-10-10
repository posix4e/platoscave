# Fresh objects, fixed models

This is a prospective **48-photograph competence pilot**. No human photographs
have been collected for it and no scientific scores have been produced.
The earlier EPIC experiment remains paused. This is a separate protocol.

First publish the model hashes, protocol and code. Then make new photographs and
write human captions. If that order is honestly followed, the photographed events
cannot have trained those already fixed model bytes. Familiar objects and phrases
remain familiar; this does not remove shared concepts or model-choice dependence.
Hashes, EXIF and local timestamps do not independently prove capture time.

## What the pilot asks

Can original DINOv1 ViT-B/16 and original GloVe 6B/300d support broad object-category
correspondence when the correct fitting pairs are supplied? Separate noun probes,
a paired ridge map and three within-round shuffled-pair maps will be reported.
All use the same fixed fit/development split. There is no unpaired Rosetta fit,
parameter search, final test, significance test or pass threshold here.

The eight classes are **mug, bowl, plate, bottle, spoon, fork, book and shoe**.
Gather two different physical examples of each: sixteen objects. Use one set in
four fitting rounds, the other in two development rounds. Photograph each object
once per round, giving 32 fitting and 16 development photographs. The latter are
two views of eight held-out objects, not sixteen independent object instances.

If these objects are unavailable, stop before capture and request a new protocol
version. Do not substitute a category silently. Success would justify designing
a larger trial; failure could concern the features, captions, readout or change
to new objects. Neither outcome settles unpaired alignment or model robustness.

## Collect the photographs

Read [protocol.json](protocol.json) and the [collection guide](COLLECTION.md).
Use the version published to GitHub before capture. The release lock lists SHA256
hashes for the protocol and the three executable scripts. Keep that entire
directory unchanged with the resulting packet.

The generator uses the Python standard library. It does not take photographs or
write captions for you. From this directory:

```sh
python3 intake.py prepare --out my-capture --collector your-alias
```

Keep the generated CAPTURE-SHEETS.txt, session plan, protocol, rounds.csv and samples.csv. The plan
contains the round markers, randomized capture orders and a shuffled caption
order. Fill the CSV records using ordinary UTC timestamps, such as
`2026-10-11T14:05:00Z`. Retain original camera files beneath the packet directory.
The collection guide explains framing, evidence photographs and replacements.

No photograph is uploaded by this kit. The website is a public guide, not an
upload service. Keep the filled packet private until you deliberately share it.
Do not commit a filled packet to this public repository.

## Seal and run

Validation requires Pillow. Feature extraction requires NumPy, Pillow, PyTorch
and the pinned DINO source. Analysis requires NumPy. The intended execution
environment is the existing research VM, where these dependencies and model files
already exist; no GPU work is required on the collector's computer.

After a person has checked every marker, object identity and caption:

```sh
python3 intake.py validate --packet my-capture --out sealed.json --attest-human-capture
```

The flag is a human attestation, not an automated authenticity test. Validation
checks the complete balanced packet, paths, hashes, image decoding, timestamps,
capture order and split rules. A seal records which bytes were used; it cannot
establish that the contents of a caption or marker are truthful.

On the research VM, supply the existing model and source paths:

```sh
python3 extract.py --sealed sealed.json --packet my-capture \
  --dino-checkpoint /path/to/dino_vitbase16_pretrain.pth \
  --dino-source /path/to/pinned-dino-source \
  --glove-zip /path/to/glove.6B.zip --device cuda --out features
python3 analyze.py --features features/features.npz --sealed sealed.json --out results
```

All outputs must be new paths. Models, source, protocol and input bytes are checked
before use. Invalid or incomplete data stops the pilot; it does not silently
change the sample. Preserve failed attempts and explain any correction before a
rerun. Retain stdout and stderr for each command: failed analysis writes a structured
JSON error record to stderr and publishes no partial results directory. Tests made from artificial arrays and images check software only; they are
never evidence about the research question.

The score is noun-class hit@1 against all 16 development captions. Either caption
with the right class counts; exact ties receive their expected credit. Balanced
random choice gives 12.5%. Report every prediction, each class, both development
rounds, duplicate captions and vectors, and all six fits. Three shuffle references
and two development rounds cannot support reliable significance or uncertainty
claims. There is no numerical go/no-go threshold to discover after looking.

## Next decision

Publish this pilot's whole result, including failures. Before an unpaired trial,
write a new collection and analysis protocol with sufficient independent objects,
sessions and unseen evaluation data. Prespecify alternative model families to
address the original model-choice criticism. This first pair cannot answer it.

Model histories: [original DINO](https://github.com/facebookresearch/dino) and
[original GloVe](https://nlp.stanford.edu/projects/glove/). The original artifact
record is in [the provenance audit](../clean-example/artifact-lock.json).
