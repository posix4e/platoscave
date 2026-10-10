# Beyond the Cave

A static research page about source overlap and model choice in Unpaired Rosetta.
The page presents a metadata audit, five experimental conditions, and all 15
completed reduced pilot runs. These runs check the procedure; they do not settle
the model-choice question or estimate the planned full-data exclusion effects.

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
At setup on 10 October 2026, delegation and certificate validation were pending;
the Pages address was serving the site. Check current DNS and HTTPS status before
claiming the custom domain is live.

## Scientific scope

The protocol and model matrix in `dist/downloads/` distinguish completed work
from proposed studies. The full confirmatory study and model-family experiments
have not run. The cave text is an original inquiry written for this study, not a
quotation from Plato. This work is independent of the original paper's authors.
