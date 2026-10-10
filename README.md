# Beyond the Cave

A static research page about source overlap and model choice in Unpaired Rosetta.
The current [five-photo experiment](research/five-photo-pilot/) is complete:
exhaustive distance matching selected one correct image–label match out of five.
The human assignment ranked 94th among the 120 possible assignments. The unique
optimum matched Water fountain correctly and exchanged Slide with Basketball
and Trashcan with Tree. The distance objective failed to recover four human
matches in this set.

Original DINOv1 and GloVe supplied the features. No correct pairs were given to
the matcher and no cross-modal map was trained. The test was chosen after seeing
the photographs; [its code and rules](https://github.com/posix4e/platoscave/commit/feead1d)
were published before feature inference. All five photographs and the complete
candidate label set took part in selection. There is no unseen evaluation set,
significance test or general accuracy claim, and this is not a Rosetta run.
One correct match is also the average under a random assignment; that numerical
agreement alone does not establish chance-level performance.

The collector attests that all five photographs were taken after the model files
were fixed. This supports a conditional argument against training on those
particular captures, not independent proof of capture time or freedom from
familiar concepts. Photographs and raw features remain private. The page shows
generic human labels and numerical results.

The public [distances](dist/downloads/five-photo-distances.json),
[all 120 assignments](dist/downloads/five-photo-assignments.json),
[evaluation](dist/downloads/five-photo-evaluation.json) and
[run record](dist/downloads/five-photo-run-evidence.json) allow the numerical
comparison to be replayed without publishing the photographs or raw features.

The earlier prospective 48-photograph paired competence study has not run. Its
[collection kit](research/fresh-pilot/README.md), protocol and analysis code are
preserved as a separate proposal; the five photographs do not complete it.

The older EPIC-KITCHENS experiment remains paused with its protocol preserved.
The site also retains the source-overlap audit and all 15 earlier reduced pilot
runs. None of this is presented as a completed model-choice result.

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

## Scientific scope

The protocol and model matrix in `dist/downloads/` distinguish completed work
from proposed studies. The five-photo result concerns one distance rule and one
model pair on five supplied photographs. It does not refute Rosetta or answer
the model-choice criticism. The full confirmatory study and model-family
experiments have not run. The cave text is an original inquiry written for this
study, not a quotation from Plato. This work is independent of the original
paper's authors.

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
