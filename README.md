# Beyond the Cave

A static research page about source overlap and model choice in Unpaired Rosetta.
The page presents a metadata audit, five experimental conditions, and all 15
completed reduced pilot runs. These runs check the procedure; they do not settle
the model-choice question or estimate the planned full-data exclusion effects.

Source: GitHub. Hosting: Cloudflare Pages. Intended custom domain: `platoscave.si`.

## Preview

```sh
python3 -m http.server 4381 --directory dist --bind 127.0.0.1
```

No build step or application dependencies are required. Scientific records are in
`dist/downloads/`. The page contains no experiment-machine access details.

## Deploy to Cloudflare Pages

Authenticate Wrangler with a Cloudflare account that can manage Pages, then:

```sh
npx wrangler@4 pages project create platoscave --production-branch main
npx wrangler@4 pages deploy dist --project-name platoscave --branch main
```

Create the project only once. If it already exists, reuse it. Deploy the committed
source and record the Git revision alongside the Cloudflare deployment ID.
Source is stored in GitHub; hosting is not GitHub Pages. Deployment is direct
upload, not an automatic Git integration.

Attach `platoscave.si` in the Pages project's custom-domain settings. An apex
domain requires its DNS zone in the same Cloudflare account and the domain's
nameservers set to that zone's assigned Cloudflare nameservers. Record the actual
deployment and DNS state before claiming the domain is live.

## Scientific scope

The protocol and model matrix in `dist/downloads/` distinguish completed work
from proposed studies. The full confirmatory study and model-family experiments
have not run. The cave text is an original inquiry written for this study, not a
quotation from Plato. This work is independent of the original paper's authors.
