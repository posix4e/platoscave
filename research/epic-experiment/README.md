# DINO and GloVe on later recordings

This experiment tests original DINOv1 ViT-B/16 and original GloVe 6B on
EPIC-KITCHENS-55. The [provenance audit](../clean-example/README.md) explains why
the recorded episodes postdate both documented training corpora and what that
chronology does not establish.

The protocol below was fixed before computing benchmark alignment scores.
Frame acquisition started on 10 October 2026. No benchmark result is reported
in this initial protocol record.

## Frozen sample

| Partition | Kitchens | Clips | Use |
| --- | ---: | ---: | --- |
| Fit | 16 | 6,639 | Fit mappings and competence controls |
| Development | 5 | 1,127 | Diagnostic scores; no initial parameter selection |
| Test | 7 | 1,626 | Held-out endpoint |

We selected 9,392 clips and four frames per clip: 37,568 frames. Participant and
clip selection use fixed hashes of identifiers, without consulting labels,
images or model scores. These are our partitions of the publicly labeled
training data, not the official EPIC challenge test sets.

The primary unpaired fit uses 3,187 image clips and 3,452 text clips. Whole
videos belong to one side only, preventing overlapping action windows from
sharing frames across those two fitting populations. All kitchens in the final
test are absent from fitting and development.

The primary endpoint is noun-class hit@1 from image to text, giving each test
kitchen equal weight. Multiple gallery entries with the same noun class count
as relevant. Action, caption and reverse-direction retrieval are secondary.
Exact ties are averaged over their possible orders.

Three solver seeds are fixed in advance. The controls include paired
Procrustes and ridge mappings, shuffled pairs, random mappings, a fit with
shared episodes but hidden correspondence, and separate image/text probes.
Paired and shared-episode controls use all fitting clips; they have a larger
information budget and serve as competence checks. Solver seeds do not vary
encoder pretraining. Frame and word averaging cannot establish temporal or
syntactic understanding.

## Records

- `protocol.json`: sample rules, parameters, endpoints, controls and limits.
- `selected-identities.csv`: every selected identity and its assigned role,
  without redistributing the narration corpus.
- `selection_summary.json` and `selection_validation.json`: counts, hashes,
  disjointness checks, and invariance to input order or semantic-label changes.
- `select_rows.py` and `verify_selection.py`: reproduce and check the full
  acquisition manifest from the pinned public annotation CSV.

The full manifest SHA-256 is
`d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be`.
The protocol SHA-256 is
`656a4fad35609bb753fb5045268e3448e4c71a36bce70a4b20050fd16a547370`.

Use the pinned `EPIC_train_action_labels.csv` downloaded by
`../clean-example/audit_epic55.py`:

```sh
python3 select_rows.py --annotations /path/to/EPIC_train_action_labels.csv \
  --protocol protocol.json --output selected_rows.json
```

The annotation source is the
[EPIC-KITCHENS-55 repository](https://github.com/epic-kitchens/epic-kitchens-55-annotations/tree/63fa7d78eba41a8d2280374061a6b1bab3e7e41d).
The [collection paper](https://arxiv.org/html/2005.00343) describes the recording
and human annotation procedure. Annotated action windows provide preprocessing
supervision even though the primary alignment stage receives no pairs.
