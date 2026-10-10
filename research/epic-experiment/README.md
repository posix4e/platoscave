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

The reproduction commands below download the pinned annotation CSV and recreate
the full manifest locally. Narrations are not distributed in this repository.

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

`publish_results.py` prepares the eventual report and CSV tables from a complete
result set. It requires the frozen runner and verifier, all 17 fits, and a
successful verification record, then reruns independent verification and
checks that input and result files stayed unchanged. It writes a new local
report directory with file hashes. It does not upload or deploy anything.
This script is preparation for reporting; no full benchmark result has yet
passed its publication checks.

## Reproduce the fixed experiment

These commands describe the complete run; they do not claim it has finished.
Use a Linux machine or VM with an NVIDIA GPU, network access to the original
artifact and EPIC video hosts, and room for the models, 37,568 JPEG frames,
feature caches and score arrays. Videos are read by HTTP seeking rather than
downloaded in full. Full videos, frames, narrations and large arrays stay
outside the Git checkout.

Prepare two separate Python 3.12 environments first. The GPU environment needs
CUDA-enabled PyTorch, torchvision, NumPy, Pillow and PyAV. The CPU environment
needs CPU PyTorch, NumPy, SciPy, scikit-learn and pylibmgm plus the solver's
dependencies. The observed versions are in
[`gpu-observed-freeze.txt`](gpu-observed-freeze.txt),
[`gpu-environment.json`](gpu-environment.json), and
[`solver-observed-freeze.txt`](solver-observed-freeze.txt). These records are
not a tested portable installation recipe; in particular, the editable solver
entry refers to our VM checkout. Use the public solver revision below instead.
Source and artifact hashes do not guarantee identical floating-point results
across different hardware or library versions.

Run the following blocks in the same Bash session, replacing the four absolute
paths. `EPIC_RUN` should be a new working directory. Keep Python assertions
enabled; do not use `-O` or `PYTHONOPTIMIZE`.

```bash
set -euo pipefail
EPIC_REPO=/absolute/path/to/platoscave
EPIC_RUN=/absolute/path/to/epic55-run
EPIC_GPU_PY=/absolute/path/to/gpu-environment/bin/python
EPIC_CPU_PY=/absolute/path/to/cpu-environment/bin/python
EPIC_SCRIPTS="$EPIC_REPO/research/epic-experiment"
EPIC_AUDIT="$EPIC_REPO/research/clean-example"
EPIC_DINO="$EPIC_RUN/dino-source"
EPIC_SOLVER="$EPIC_RUN/unpaired-rosetta"
mkdir -p "$EPIC_RUN"

"$EPIC_GPU_PY" -c 'import av, numpy, torch, torchvision; from PIL import Image; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$EPIC_CPU_PY" -c 'import numpy, scipy, sklearn, torch, importlib.metadata; assert torch.version.cuda is None; print(torch.__version__, importlib.metadata.version("pylibmgm"))'
```

### Original artifacts and source

The required sources are DINO revision
`7c446df5b9f45747937fb0d72314eb9f7b66930a`, Unpaired Rosetta revision
`fdfad84ea488f29702a61a455be71622c4ff5d49`, and EPIC annotations revision
`63fa7d78eba41a8d2280374061a6b1bab3e7e41d`. The annotation downloader pins that
last revision and all three CSV hashes internally.

`fetch_models.py` obtains the original `dino_vitbase16_pretrain.pth`, its training
arguments, and `glove.6B.zip`; it checks every artifact and the 300-dimensional
GloVe member against [`artifact-lock.json`](../clean-example/artifact-lock.json).
The extractor accepts the ZIP directly. Its own checks pin the checkpoint,
GloVe bytes, and DINO's `vision_transformer.py` and `utils.py`. Do not substitute
DINOv2, Landmarks fine-tuned weights, GloVe 2024, or another GloVe corpus.

```bash
"$EPIC_CPU_PY" "$EPIC_AUDIT/audit_epic55.py" --output "$EPIC_RUN/metadata"
"$EPIC_CPU_PY" "$EPIC_AUDIT/fetch_models.py" \
  --directory "$EPIC_RUN/models" \
  --expected-manifest "$EPIC_AUDIT/artifact-lock.json"

git clone https://github.com/facebookresearch/dino.git "$EPIC_DINO"
git -C "$EPIC_DINO" checkout --detach 7c446df5b9f45747937fb0d72314eb9f7b66930a
git clone https://github.com/dominik-schnaus/unpaired-rosetta.git "$EPIC_SOLVER"
git -C "$EPIC_SOLVER" checkout --detach fdfad84ea488f29702a61a455be71622c4ff5d49

export PYTHONPATH="$EPIC_SOLVER"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$EPIC_CPU_PY" -c 'import unpaired_rosetta; print(unpaired_rosetta.__file__)'
```

`PYTHONPATH` points to the checkout root so the runner imports this exact
`unpaired_rosetta` package. The verification flag `--upstream-source` below
instead points to the package directory inside that checkout. Cloning source
does not install its dependencies.

### Frozen selection and source checks

```bash
"$EPIC_CPU_PY" "$EPIC_SCRIPTS/select_rows.py" \
  --annotations "$EPIC_RUN/metadata/EPIC_train_action_labels.csv" \
  --protocol "$EPIC_SCRIPTS/protocol.json" \
  --output "$EPIC_RUN/selected_rows.json"
"$EPIC_CPU_PY" "$EPIC_SCRIPTS/verify_selection.py" \
  --annotations "$EPIC_RUN/metadata/EPIC_train_action_labels.csv" \
  --manifest "$EPIC_RUN/selected_rows.json" \
  --output "$EPIC_RUN/selection_validation.json"

"$EPIC_CPU_PY" - "$EPIC_SCRIPTS" "$EPIC_RUN" "$EPIC_SOLVER/unpaired_rosetta" <<'PY'
import hashlib, json, sys
from pathlib import Path
scripts, run, upstream = map(Path, sys.argv[1:])
sys.path.insert(0, str(scripts))
from continue_experiment import FROZEN, UPSTREAM_SHA256
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in FROZEN.items():
    path = (run if name == "selected_rows.json" else scripts) / name
    if digest(path) != expected:
        raise SystemExit(f"Frozen file differs: {path}")
files = {str(p.relative_to(upstream)): digest(p) for p in sorted(upstream.rglob("*.py"))}
encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
if hashlib.sha256(encoded).hexdigest() != UPSTREAM_SHA256:
    raise SystemExit("Upstream solver package differs from the frozen source")
print("Frozen manifest, protocol, experiment scripts and solver package match")
PY
```

The selected manifest must reproduce the SHA-256 recorded above: all 9,392
clips, their partitions and four requested frame times. This check also binds
the acquisition, extraction, runner and independent verifier to their frozen
source bytes.

### Acquisition and feature extraction

```bash
"$EPIC_GPU_PY" -u "$EPIC_SCRIPTS/acquire_frames.py" \
  --manifest "$EPIC_RUN/selected_rows.json" \
  --output "$EPIC_RUN/data" --workers 4
"$EPIC_GPU_PY" -u "$EPIC_SCRIPTS/verify_acquisition.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --data "$EPIC_RUN/data" \
  --acquirer "$EPIC_SCRIPTS/acquire_frames.py" \
  --output "$EPIC_RUN/acquisition-verification.json"

"$EPIC_GPU_PY" "$EPIC_SCRIPTS/check_preprocess.py" \
  --extractor "$EPIC_SCRIPTS/extract_features.py" --require-torchvision \
  --output "$EPIC_RUN/preprocess-validation.json"
"$EPIC_CPU_PY" "$EPIC_SCRIPTS/text_preflight.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --glove "$EPIC_RUN/models/glove.6B.zip" \
  --extractor "$EPIC_SCRIPTS/extract_features.py" \
  --output "$EPIC_RUN/text-preflight.json"
"$EPIC_GPU_PY" -u "$EPIC_SCRIPTS/extract_features.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --frame-root "$EPIC_RUN/data" \
  --dino-source "$EPIC_DINO" \
  --dino-checkpoint "$EPIC_RUN/models/dino_vitbase16_pretrain.pth" \
  --glove "$EPIC_RUN/models/glove.6B.zip" \
  --cache-dir "$EPIC_RUN/frame-cache" --batch-size 64 --device cuda \
  --output "$EPIC_RUN/features.npz"
```

Acquisition must finish all 268 selected videos and pass timestamp, frame-byte
and source checks before extraction. If a video fails, retain its failure
report and investigate; do not remove or replace clips. Completed videos can
be reused only when their recorded hashes still match. No `--videos` subset or
`--allow-partial` option belongs in this full run. The resulting `features.npz`
and adjacent `features.json` must contain all 9,392 rows with finite, nonzero
vectors. Keep both files together; the runner checks their provenance and
refuses partial or ineligible inputs.

### All fits, independent replay and report preparation

```bash
"$EPIC_CPU_PY" "$EPIC_SCRIPTS/run_experiment.py" --self-test
"$EPIC_CPU_PY" "$EPIC_SCRIPTS/verify_results.py" --self-test
"$EPIC_CPU_PY" -m unittest discover -s "$EPIC_SCRIPTS" -p test_publish_results.py

"$EPIC_CPU_PY" -u "$EPIC_SCRIPTS/run_experiment.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --features "$EPIC_RUN/features.npz" \
  --output "$EPIC_RUN/results" --seeds 0 1 2 \
  --methods rosetta_disjoint rosetta_shared paired_orthogonal paired_ridge \
    shuffled_orthogonal shuffled_ridge random_map \
  --clusters 30 --restarts 30 --iterations 100 --batch-size 2048 \
  --workers 4 --threads 4 --ridge-alpha 1 --permutations 999
"$EPIC_CPU_PY" -u "$EPIC_SCRIPTS/verify_results.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --protocol "$EPIC_SCRIPTS/protocol.json" \
  --features "$EPIC_RUN/features.npz" --results "$EPIC_RUN/results" \
  --runner "$EPIC_SCRIPTS/run_experiment.py" \
  --upstream-source "$EPIC_SOLVER/unpaired_rosetta" \
  --output "$EPIC_RUN/verification.json"
"$EPIC_CPU_PY" -u "$EPIC_SCRIPTS/publish_results.py" \
  --manifest "$EPIC_RUN/selected_rows.json" --protocol "$EPIC_SCRIPTS/protocol.json" \
  --features "$EPIC_RUN/features.npz" --results "$EPIC_RUN/results" \
  --verification "$EPIC_RUN/verification.json" \
  --runner "$EPIC_SCRIPTS/run_experiment.py" \
  --verifier "$EPIC_SCRIPTS/verify_results.py" \
  --upstream-source "$EPIC_SOLVER/unpaired_rosetta" --python "$EPIC_CPU_PY" \
  --output "$EPIC_RUN/publication"
```

The two deterministic paired controls each run once; the other five methods
each use three seeds, giving 17 fits and 34 development/test score matrices.
The verifier replays every saved score matrix, duplicate-aware retrieval
metric, diagnostic null, participant bootstrap and probe prediction. It does
not rerun encoder inference or independently retrain the six Rosetta fits.

The final command requires a new `publication` directory, reruns independent
verification, and checks that the input and result bytes remain unchanged.
Only then does it write `RESULTS.md`, JSON summaries, participant/seed scores,
control differences and intervals, diagnostic null tables and a hash receipt.
This is local report preparation; deployment is a separate action. A failure
at any stage is an incomplete experiment, not a negative alignment result.
