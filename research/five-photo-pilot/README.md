# Five photographs, five names

This is a small exploratory test of correspondence between two distance graphs.
The user supplied five photographs and these exact labels: **Slide, Trashcan,
Tree, Basketball, Water fountain**. The collector confirmed that all five were
taken after the previously published DINO/GloVe model files were fixed. Capture
time is attested by the collector, not independently authenticated.

The design was written **after seeing the photographs and labels, before
extracting their features or computing scores**. It is not a preregistered sample
or a held-out test. The photographs and private feature artifacts are not public.

## Result, 10 October 2026

The unique minimum recovered **one of the five names**. The human assignment
ranked **94th of 120** under the fixed distance objective.

| Photograph's human label | Chosen label | Correct |
| --- | --- | --- |
| Slide | Basketball | No |
| Trashcan | Tree | No |
| Tree | Trashcan | No |
| Basketball | Slide | No |
| Water fountain | Water fountain | Yes |

The chosen assignment's cost was 0.03781860020010573; the human assignment's cost
was 0.07347346776688789. The gap to the next distinct cost was
0.00028064526353802077. The chosen assignment had distance correlation
0.8782134342420858 despite four wrong names. It was selected to make these
distances agree, so that correlation is not independent evidence of semantic
correctness.

One correct match equals the expectation for a uniform random bijection. This
single result does not establish statistical equivalence to chance. It shows
that the specified distance objective preferred a mostly incorrect assignment
for this sample. No labels, crops, layers or distance rules were revised after
seeing that outcome.

The [source commit](https://github.com/posix4e/platoscave/commit/feead1da4d77bb0dc674fcf0f5f38979fac9a90c)
was pushed before feature extraction. The saved selection preceded evaluation.
See [the evaluation](results/evaluation.json), [all 120 assignments](results/assignments.json)
and [run evidence](results/run-evidence.json). Every supplied label token was
present in GloVe; there were no missing-word fallbacks.

An [independent numerical replay](results/independent-audit.json) rebuilt the
distance matrices from the saved features, recreated both shuffled orders and
checked all 120 assignments without importing the primary extractor or solver.
It agreed on every selected match, the rank and the cost gap; differences in
floating-point costs were at most 5.56 × 10⁻¹⁷. The
[audit script](independent_audit.py) checks numerical and recorded-provenance
consistency. It does not rerun the encoders or independently authenticate the
photographs or capture time.

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

The published distance matrices allow anyone to replay the matching calculation:

```sh
python research/five-photo-pilot/solve.py solve \
  --input research/five-photo-pilot/results/blind-distances.json \
  --out /tmp/five-photo-selection-replay
```

Choose a new output directory. The private photographs and raw features are not
included, so public replay starts at the distance matrices, not the camera input.
The [public replay check](results/public-replay-check.json) reproduced the
original selection artifacts byte for byte on a second machine.

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
