# Beyond the Cave

A static research page about source overlap and model choice in Unpaired Rosetta.
The [model comparison](research/model-choice-controls/) is complete. Three image
models and two language models were given the same five photographs, human
labels, crop and distance-matching rule. Every pairing examined all 120
assignments. Each cell below gives correct matches out of five and the human
assignment’s rank among those 120 possibilities.

| Image model | GloVe | ELMo |
| --- | --- | --- |
| DINOv1 | 1/5; rank 94/120 | 0/5; rank 26/120 |
| MoCo-v2 | 1/5; rank 91/120 | 0/5; rank 40/120 |
| MAE | 1/5; rank 99/120 | 0/5; rank 24/120 |

All six optima were unique. DINO and MoCo chose the same assignments within each
language model. MAE changed the assignments despite matching their correct
counts. None recovered the five human matches. These are six comparisons on
five shared observations, not independent trials. A random assignment gets one
correct on average; these results do not establish a general level of accuracy.

The image outputs are DINOv1 ViT-B/16 CLS, MoCo-v2 ResNet-50 pooled backbone,
and MAE ViT-B/16 CLS. GloVe uses its 6B, 300-dimensional vectors averaged over
label tokens. Original ELMo uses the top contextual layer of its 1 Billion Word
Benchmark model, averaged over real tokens with fresh recurrent state per label.
All vectors are normalized to unit length. MAE’s cosine distances span 0.0054–0.0397,
versus 0.4923–0.9231 for DINO and 0.3655–0.8125 for MoCo on these photographs. Raw matching
costs therefore must not rank the models. One- or two-word labels do not test
ELMo’s behavior on richer descriptions.

This follow-up was designed after the first result. Its
[code and rules](https://github.com/posix4e/platoscave/commit/7fcbc37) were published
before additional feature extraction. No correct pairs were given to the
matcher and no cross-modal map was trained. All five photographs and the complete
candidate label set take part in selection. There is no unseen evaluation set,
significance test or winner selection, and this is not a Rosetta run. Differences
between cells cannot isolate the effects of training data, objective or architecture.

The original DINO/GloVe files were fixed before the collector reports taking
the photographs. The added MoCo, MAE and ELMo files were acquired and hashed
afterward. Their documented training histories are older, but they do not inherit
the original pair’s file lock before capture. The capture account is a human
attestation, not independent proof of time. Neither chronology excludes familiar
objects, words or places from training. Photographs and raw features remain
private; the page shows generic human labels and numerical results.

The public [distances](dist/downloads/model-choice-distances.json),
[all assignments](dist/downloads/model-choice-assignments.json),
[evaluation](dist/downloads/model-choice-evaluation.json),
[protocol](dist/downloads/model-choice-protocol.json),
[run record](dist/downloads/model-choice-run-evidence.json),
[public replay](dist/downloads/model-choice-public-replay.json) and
[numerical check](dist/downloads/model-choice-audit.json) provide the comparison’s
record without publishing the photographs or raw features. Independent arithmetic
replay agreed with all 720 assignments, using the five saved feature matrices.
It did not rerun the encoders or authenticate capture time or training history.

The [original five-photo experiment](research/five-photo-pilot/) remains unchanged.
Its DINO/GloVe features and result are reused in the table above: one correct
match out of five, with the human assignment ranked 94th. Water fountain was
matched correctly; Slide and Basketball were exchanged, as were Trashcan and
Tree. The first test was chosen after seeing the photographs, and
[its code and rules](https://github.com/posix4e/platoscave/commit/feead1d) were
published before its feature inference. Its
[distances](dist/downloads/five-photo-distances.json),
[120 assignments](dist/downloads/five-photo-assignments.json),
[evaluation](dist/downloads/five-photo-evaluation.json) and
[run record](dist/downloads/five-photo-run-evidence.json) remain available.

The earlier prospective 48-photograph paired competence study has not run. Its
[collection kit](research/fresh-pilot/README.md), protocol and analysis code are
preserved as a separate proposal; the five photographs do not complete it.

The older EPIC-KITCHENS experiment remains paused with its protocol preserved.
The site also retains the source-overlap audit and all 15 earlier reduced pilot
runs. Those older records are separate from the completed six-pair comparison.

Source: [posix4e/platoscave](https://github.com/posix4e/platoscave).
Hosted on [Cloudflare Pages](https://platoscave.pages.dev/).
Custom domain: [platoscave.si](https://platoscave.si/).

## Preview

```sh
python3 -m http.server 4381 --directory dist --bind 127.0.0.1
```

No build step or application dependencies are required. Scientific records are in
`dist/downloads/`. The page contains no experiment-machine access details.

## Deploy to Cloudflare Pages

Authenticate Wrangler with a Cloudflare account that can manage Pages, then:

```sh
npx wrangler@4 pages deploy dist --project-name platoscave --branch main --commit-hash "$(git rev-parse HEAD)"
```

The `platoscave` Pages project already exists. Deploy the committed source and
record the Git revision alongside the Cloudflare deployment ID.
Source is stored in GitHub; hosting is not GitHub Pages. Deployment is direct
upload, not an automatic Git integration.

`platoscave.si` is attached to the Pages project, with a proxied CNAME pointing
to `platoscave.pages.dev`. Namecheap has the Cloudflare nameservers saved:
`dayana.ns.cloudflare.com` and `thomas.ns.cloudflare.com`.
Delegation and certificate validation completed on 10 October 2026. Deployment
records should still include a current HTTPS check for the custom domain.

## Full reproduction in progress

The [full reproduction protocol](research/rosetta-replication/) was published at
[`e064d49`](https://github.com/posix4e/platoscave/commit/e064d49158ae244bf5c2d4ab38e38140798e24c9)
before fitting. It runs the original DINOv2 B/14 mean × MPNet COCO configuration:
all five original seeds, disjoint training halves, full settings, 40,504 retrieval
items and the three classification evaluations. The unchanged authors' software
passed all 42 method/evaluation tests; all 13 input hashes and 228 source files
passed preflight.

The first full run has started. The [status snapshot](dist/downloads/rosetta-replication-status.json)
records its observation time; it is not a live feed or a result. All five outputs
and the frozen comparison are required before a reproduction verdict. Report
paper-rounded agreement separately from available full-precision agreement.
Released embeddings test alignment, not encoder pretraining or feature extraction.

The [next model-panel draft](research/rosetta-model-controls/README.md) specifies
four image pipelines crossed with three language pipelines, five original seeds
and five methods: Rosetta plus correct-pair and shuffled-pair orthogonal and
linear controls. All 12 cells remain in the plan. The complete public COCO data
is acquired and the added extractors have passed artificial-input checks; no
added corpus features or model-panel fits have run. The draft cannot authorize
execution while the baseline and final release remain incomplete.

The paired controls need extra training counterparts because the
unpaired banks are disjoint; their advantage cannot be attributed solely to
revealing pair identities. Study source overlap separately with unchanged
encoders and size-matched random removal. A later fresh collection requires a
complete model/protocol lock and separate fitting and final evaluation scenes.
These later phases are still being prepared; they are not completed by the
baseline. Earlier frozen protocols and all six five-photo results are preserved.

## Scientific scope

The protocol and model matrix in `dist/downloads/` distinguish completed work
from proposed studies. The completed comparison concerns one distance rule and
six model pairings on five supplied photographs. It begins to test sensitivity
to model choice in this small example; it does not refute Rosetta or settle the
original model-choice criticism. The full confirmatory study has not run. The
cave text is an original inquiry written for this study, not a quotation from
Plato. This work is independent of the original paper's authors.

## Pretraining provenance

The [pretraining audit](research/pretraining-audit/README.md) records why none of
the screened checkpoints has yet been admitted as an uncontaminated COCO/SPC
control. It includes a reproducible audit of the released COCO caption component
listed in MPNet's fine-tuning configuration. Research files outside `dist/` are
stored in GitHub and are not automatically published to the website.

The [search for a cleaner example](research/clean-example/README.md) identifies
original DINOv1 and GloVe on EPIC-KITCHENS-55 as a control against direct exposure
to its 2017 recordings. It includes reproducible metadata checks, original model
artifact hashes, and the limitations that must guide the next experiment.

The [EPIC experiment](research/epic-experiment/README.md) records its frozen
protocol and 9,392 selected clips. Acquisition and its supervisor are paused
for scope review; the record contains no benchmark alignment results. Any
smaller pilot or protocol amendment must be specified separately before running.
