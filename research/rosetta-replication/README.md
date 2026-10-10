# A direct Rosetta reproduction

The five-photo trial did not run Rosetta. This experiment runs the authors'
original alignment procedure for one complete published configuration: DINOv2
B/14 mean features with MPNet on COCO, with no known pairs. It includes all five
original seeds, the full retrieval set and all three classification evaluations.

The [protocol](protocol.json) fixes the configuration and comparison rules before
fitting. The [reference](reference.json) records the authors' published values,
source URLs, retrieval date, immutable website revision and source hashes.
The reference is chosen before our results, not selected for agreement afterward.
Each reference source's `file` field names the archived download used for the
audit; the corresponding URL and SHA-256 identify the public source. The paper's
figures are not bundled in this repository.

| Fixed part | Value |
| --- | --- |
| Authors' source | `fdfad84ea488f29702a61a455be71622c4ff5d49` |
| Image encoder | Released `dinov2_vit-b14@224_mean` |
| Text encoder | Released `mpnet` |
| Training | Two disjoint halves of 82,783 COCO training identities |
| Items per modality | 41,391 images; 41,392 caption items |
| Known pairs | Zero |
| Initialization | 30 clusters; 30 restarts |
| Batch size | 10,000 |
| Refinement | 100 iterations |
| Retrieval evaluation | All 40,504 COCO validation items |
| Classification | CIFAR-10, CIFAR-100 and ImageNet-100 |
| Seeds | 734796314, 576165995, 2197670066, 839703249, 2584932063 |

The official runner supplies these configurations and computes their results.
Our operational scripts check files, configuration and completion; they do not
replace the authors' fitting or evaluation code. There are no smaller training
sets, fewer restarts or altered metrics in this reproduction.

## What agreement would mean

The published mean FOSCTTM is 0.042989911199083145, with sample standard deviation
0.0480892962510061. The paper prints these as 0.04 ± 0.05. Smaller is better;
chance expectation is 0.5. This is a rank-based measure, not percentage accuracy.
The authors' website also identifies seed 734796314 with FOSCTTM
0.016328599083198735. We retain all five seeds instead of reporting only this one.

Report the mean and sample standard deviation, every run, signed/absolute errors
against available references, and agreement at the paper's displayed precision.
A separate absolute tolerance of 1e-10 marks strict numerical agreement with
available full-precision values. This is our declared numerical check, not an
author-supplied tolerance or a statistical significance test.

The classification references are rounded means and sample standard deviations:
CIFAR-10 top-1 70% ± 10 percentage points; CIFAR-100 top-5 46% ± 14 points;
ImageNet-100 top-5 45% ± 11 points. Preserve all original top-1 and top-5 outputs,
including metrics for which we have no published comparison value.

Four individual seed references and unrounded classification aggregates were not
found. Matching the available aggregates cannot establish exact agreement of
those missing values. Any mismatch must remain visible and prompt a check of
software, input precision, seeds and conditions before a scientific conclusion.

## Inputs and environment

[inputs-lock.json](inputs-lock.json) pins all 13 released input files to one
Hugging Face dataset revision and the publisher's LFS SHA-256 hashes. Existing
cached files are reused only after hash checks; missing files are downloaded and
verified. No private photographs are inputs to this reproduction.

[source-lock.json](source-lock.json) identifies all 228 original source files,
the original Pixi lock and the verified Pixi binary. The authors' frozen default
environment supplies Python 3.14.7, PyTorch 2.14.0, NumPy 2.5.3, SciPy 1.18.0,
scikit-learn 1.9.1 and pylibmgm 1.1.2. The [environment record](environment.json)
lists installed distributions and the prescribed AVX2 settings.

The original caption loader averages captions in stored bfloat16 precision
before converting the item vectors to float32. This detail is retained. The
experiment runs inside the dedicated VM; earlier experiments remain untouched.

The executable harness requires an explicit VM root and a copy of this release:

```sh
pixi run --frozen -e default python /path/to/release/preflight.py \
  --root /path/to/replication --release /path/to/release
python3 /path/to/release/run_locked.py \
  --root /path/to/replication --release /path/to/release
```

Run preflight from the original source directory. It performs file, runtime,
split and configuration checks without fitting to these embeddings. The runner
requires fresh outputs and preserves errors. Since the authors' runner catches
individual exceptions, a zero exit code alone cannot establish completion:
all five complete result files are required.

After the run completes, the frozen comparison script reads the five original
JSON files and computes their means and sample standard deviations independently:

```sh
python3 /path/to/release/compare.py --release /path/to/release \
  --results /path/to/replication/storage/results/unpaired \
  --output /path/to/new-comparison.json
```

It reports each numerical and rounded comparison separately, preserves every
seed and refuses missing or invalid results. A comparison file does not replace
the run's source, input and completion checks.

## What remains after the baseline

This reproduces one published configuration, not the whole paper. Reusing
released embeddings tests alignment; it does not independently establish their
training history or repeat feature extraction. A successful baseline is a
prerequisite to controlled model replacements and a separate source-overlap
study. A prospective collection must follow a complete model and protocol lock.
Those later phases remain part of the approved plan; this baseline does not
complete them.
