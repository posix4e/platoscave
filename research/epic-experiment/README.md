# DINO and GloVe on later recordings

This experiment tests original DINOv1 ViT-B/16 and original GloVe 6B on
EPIC-KITCHENS-55. The [provenance audit](../clean-example/README.md) explains why
the recorded episodes postdate both documented training corpora and what that
chronology does not establish.

The protocol below was fixed before computing benchmark alignment scores.
Frame acquisition started on 10 October 2026. No benchmark result is reported
in this initial protocol record.

**Work in progress:** the acquisition, feature extraction, alignment and
independent verification code is now published. The VM is still acquiring the
fixed sample; no full benchmark alignment has run. `wip-status.json` records
the checks completed and the work remaining. A recorded decoding failure is
retained for investigation, not used to remove a clip from the sample.

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

## Implementation and checks

The stages are `acquire_frames.py`, `verify_acquisition.py`,
`extract_features.py`, `run_experiment.py`, and `verify_results.py`. The last
script recomputes scores and metrics without importing the runner's metric
functions. It requires all 17 fits and the complete frozen sample. Partial
feature files cannot qualify for the benchmark runner.

`continue_experiment.py` supervises those stages. Its default mode prints a
plan; `--run` waits for the existing acquisition service, checks frozen source
hashes and requires every selected frame before extraction. After a terminal
acquisition failure it permits one retry using the same code and sample,
archiving prior reports first. It stops if that retry or any later verification
fails. A lock prevents concurrent supervisors. Its state-transition tests have
passed; the full pipeline has not completed.

The [P03_05 diagnosis](p03-05-failure-review.md) records one decoding failure
and an isolated recovery of the target frame. Both transport modes recovered
identical bytes. This supports a retry, but does not prove that the full video
has recovered or identify the original failure's exact cause. Its retained
diagnostic trace has HTTP cookie values redacted.

`preprocess-validation.json` records exact agreement with torchvision on 31
preprocessing cases. `feature-repeat-validation.json` records a repeat of one
action's cached features. `synthetic_check.json` tests recovery of a known
synthetic correspondence at smaller settings. These establish limited
implementation checks; they are not evidence of successful EPIC alignment.

`text_preflight.py` runs the frozen text extractor over all 9,392 selected
captions without reading images or fitting an alignment. The recorded
`text-preflight-summary.json` shows that every text vector is finite and
nonzero; GloVe covers 28,844 of 28,902 token occurrences (99.8%). There are
4,277 distinct text vectors, with 6,332 clips belonging to groups that share a
vector. This supports the protocol's treatment of repeated captions and exact
ties; it does not establish alignment or absence of pretraining contamination.
The summary omits full extractor metadata and per-group UID listings and
records the complete report's hash. Rerun the script to reproduce those details.

`benchmark_http_seek.py` and `http-seek-benchmark.json` compare transport
settings on eight frame requests. Connection reuse preserved their timestamps
and bytes and reduced elapsed time in that bounded check. The acquisition
script records its version and frame hashes; a changed version cannot silently
reuse old completed-video reports.

The `*-observed-freeze.txt` files record the actual two Python environments,
not portable installation lockfiles. The GPU environment uses CUDA 12.8
PyTorch; the alignment environment uses CPU PyTorch. The solver's editable
checkout includes our earlier research scripts at revision `263ad1d`; its
`unpaired_rosetta` package matches public upstream revision
`fdfad84ea488f29702a61a455be71622c4ff5d49`. Use that public revision when
reconstructing the solver. `wip-status.json` records source hashes. Runtime
configuration will also hash every executed upstream package file.

After recreating the full selected manifest, stages must run in order. Each
script's `--help` documents its explicit paths. Keep the original model weights
from the adjacent provenance audit and the pinned DINO source; do not replace
them with model-library defaults. Full videos, extracted frames and large
feature arrays are not stored in Git.

`publish_results.py` prepares the eventual report and CSV tables from a complete
result set. It requires the frozen runner and verifier, all 17 fits, and a
successful verification record, then reruns independent verification and
checks that input and result files stayed unchanged. It writes a new local
report directory with file hashes. It does not upload or deploy anything.
This script is preparation for reporting; no full benchmark result has yet
passed its publication checks.
