# Per-query rank presentation

For the separate reconstruction of metrics, seed summaries and contrasts, see
[Independent saved-rank numerical comparison](NUMERICAL_AUDIT.md).

For the compact table of all frozen scores and contrasts, see
[Compact model-panel projection](MODEL_SUMMARY.md). It requires the complete
accepted model stage, independent comparison and rank manifest. It has only
synthetic validation; preparing a projection does not approve its interpretation
or publication.

For the source study, see [Compact source-overlap summary](SOURCE_SUMMARY.md).
It retains all 115 fits, every declared score and contrast, and the mask
composition and intersection counts. It requires the complete accepted source
stage and matching independent comparison. It has only synthetic validation;
real numerical differences and interpretation still require review.

`export_rank_tables.py` is a local, presentation-only helper outside the frozen
scientific releases. It has not been used on real run outputs. It requires an
accepted **complete** model-reader summary, its controller `accepted.json`
receipt, independently supplied SHA-256 values for both, and the matching
scientific attempt tree. Use the exact summary referenced by the accepted reader
receipt; do not choose a newer directory or recompute a summary for this export.

```sh
python export_rank_tables.py \
  --summary /path/to/accepted/summary.json --summary-sha256 SUMMARY_SHA256 \
  --acceptance /path/to/accepted/accepted.json --acceptance-sha256 RECEIPT_SHA256 \
  --attempts-root /path/to/model/cells --out /path/to/new/rank-tables
```

The existing frozen runner stores `ranks[i]` alongside
`validation_image_ids[i]` and `validation_indices[i]`. Query presentation order
has already been undone by the runner. The exporter applies only the reader's
verified ascending-COCO-ID permutation, to all three vectors together. It also
retains the original validation-subset position. `midrank` uses round-trip-safe
float64 text; no averages, FOSCTTM values or other scientific metrics are
recomputed after sorting.

Output consists of **300 gzip CSVs**, one for each model pairing / seed / method,
each with **40,504 rows**, plus `manifest.json`. A partition is a few MiB of
uncompressed text; gzip keeps download units small without dropping precision.
Actual compressed byte counts and hashes are recorded per partition. There is
no monolithic 12,151,200-row asset. Every row retains model, text, seed, method,
selected relative attempt, cell terminal status, original position, COCO ID,
source-row index and raw midrank. The manifest binds the accepted summary,
reader receipt, exporter, source rank artifacts and output hashes. Absolute
machine paths are not written to the output.

The exporter rejects partial/unaccepted summaries, unavailable methods, invalid
retry histories, wrong frozen identities, mismatched artifact hashes or mapping
vectors, invalid ranks, symlinked artifact paths, and existing output directories.
The destination must be outside the input artifact and receipt directories. The
manifest is published last; a directory without it is incomplete and must not be
used. Export only stopped, accepted artifacts with trustworthy externally
recorded summary/receipt hashes. The helper does not authenticate those external
hashes or independently repeat the scientific reader, service checks, RNG,
fitting, similarity scoring, or metric reductions.

Selection follows the accepted reader exactly. A verified complete method can
come from a finalized `FAILED.json` cell; that terminal status is retained in
the CSV and manifest. Exporting 300 available methods does not certify 60
complete cells. Export-manifest `status: complete` means the 300 presentation
partitions passed these export checks; controller stage acceptance and the
protocol's independent comparison before interpretation remain separate.
Publishing a completed model stage still requires all 60 accepted complete
cell/seed jobs and the controller's accepted `model-stage-complete.json`
receipt; this exporter does not supply those gates.
Reference ranks belong to the new DINOv2/MPNet replay attempts,
not the original baseline runs. The protocol, feature lock and reader/reporting
hashes are pinned for the current frozen panel; a future scientific release
requires explicit review of this presentation adapter.

Local test dependency: Python 3.12 and `torch==2.8.0` (CPU loading only). The small
synthetic fixture temporarily sets the row count to four in the test process,
uses shuffled IDs and query/key orders, and exercises all 300 output identities.
There is no production option to weaken the 40,504-row contract.

```sh
.venv/bin/python -m unittest -v test_export_rank_tables.py
```

These tests establish presentation behavior on invented data, not validation of
any real experiment. No data is published by this helper.
