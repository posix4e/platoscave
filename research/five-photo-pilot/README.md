# Five photographs, five names

This is a small exploratory test of correspondence between two distance graphs.
The user supplied five photographs and these exact labels: **Slide, Trashcan,
Tree, Basketball, Water fountain**. The collector confirmed that all five were
taken after the previously published DINO/GloVe model files were fixed. Capture
time is attested by the collector, not independently authenticated.

The design was written **after seeing the photographs and labels, before
extracting their features or computing scores**. It is not a preregistered sample
or a held-out test. The photographs and private feature artifacts are not public.

## The question

If two images are close to each other in DINO, are their names also close in
GloVe? We compute the ten distances between image pairs, and the ten distances
between label pairs. The matcher sees only those two distance matrices and
opaque identifiers. It examines every one of the 120 possible one-to-one
assignments and chooses the assignment whose corresponding distances differ
least in mean squared error. It never fits a map to the known image-label pairs.

After selection has been written, a separate evaluation step reads the human
correspondence. It reports the correct-match count, the human assignment's cost
and rank, and every tied optimum. Correctness never breaks a tie. All 120 costs
remain available; these are alternative assignments, not independent trials.

A random bijection gives one correct match out of five on average. That is a
descriptive reference, not a significance test. Five items, one outing and one
model pair cannot support a general accuracy or model-robustness claim.

## Frozen choices

[protocol.json](protocol.json) specifies the whole calculation. The original
DINOv1 ViT-B/16 checkpoint, original GloVe 6B/300d archive and text member are
checked by hash. The image transform is the existing fixed center crop; the
words are exactly the user's supplied labels, passed through the existing
tokenizer and mean-vector rule. No prompt expansion, alternative crops, learned
metric, feature search or paired training is used.

The three-file release lock covers the protocol, extraction wrapper and solver.
Extraction additionally checks the exact frozen helper from the earlier
48-photograph plan. The earlier plan is not changed or resumed by this work.

## Running the test

Use the research VM's GPU Python environment. The extraction CLI takes a private
manifest and image root, the locked helper, the existing model files and a new
output directory; run `python extract_five.py --help` for paths. It writes private
features and a separate private ground-truth mapping, plus a blind distance file.

```sh
python solve.py solve --input features/blind-distances.json --out selection
python solve.py evaluate --selection selection \
  --ground-truth features/private-ground-truth.json --out evaluation.json
```

Run selection to completion before evaluation. Retain logs, file hashes and all
outputs. Existing outputs must not be overwritten. Failures stay in the record;
any revised test must be specified and named separately before it is run.

## What a result would mean

A correct assignment shows that this one objective recovered the human matches
for these five items. A wrong assignment shows that it did not. Neither outcome
establishes or refutes Unpaired Rosetta: this is an exhaustive distance-graph
comparison, not that paper's trained alignment algorithm.

All candidate labels are known to be present, and all five photographs are used
in selection. There is no unseen evaluation set. Shared backgrounds, word senses,
mixed object/activity labels, multiword averaging and the fixed center crop may
dominate the geometry. Preserve that outcome rather than adjusting words or
views after observing it. A later, larger study must make independent tests of
generalization and model choice.
