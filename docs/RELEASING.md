# Distribution release checklist

## Current release status

The first production release, ScGeo 0.1.0, was published on
[PyPI](https://pypi.org/project/scgeo/0.1.0/) on 15 October 2026. Native Bioconda
publication is tracked separately in [PR #70318](https://github.com/bioconda/bioconda-recipes/pull/70318).
The current release source is 0.2.0; do not advertise it as a public-index release
until its production verification passes. User commands are in [Installation](INSTALLATION.md).

The historical 0.1.0 production promotion remains documented in
[the publication record](PYPI_PUBLICATION.md). Do not modify the original tag,
published artifacts, checksums, or manuscript checkpoint.

## Review the release candidate

- Keep package, runtime, local conda recipe, citation metadata and generated API
  manifests aligned at the new version; inspect the changelog and guides.
- Require the release PR's documentation, packaging, library, optional-extra,
  native conda and offline promotion guard checks to pass.
- Retain explicit optional-integration skips and keep release scope separate from
  manuscript reproduction. No new scientific validation is implied.
- Preserve the manuscript checkpoint `9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`.

## Protected publishing configuration

The two existing Trusted Publishers are separate:

| Field | TestPyPI | Production PyPI |
| --- | --- | --- |
| Project | scgeo | scgeo |
| Owner | liuifrec | liuifrec |
| Repository | scgeo | scgeo |
| Workflow | publish.yml | publish-pypi.yml |
| Environment | testpypi | pypi |
| Environment allowed ref | Tag: v* | Branch: main |

Retain required reviewer `liuifrec`, permit self-review for the solo maintainer,
and disable administrator bypass. Verify these settings on GitHub and the
respective package indexes. Do not share tokens or weaken the approval gates.
The workflow filename and environment identity for the established production
Trusted Publisher are unchanged.

## Stage on TestPyPI

Merge the reviewed release PR, then create a new immutable tag at the exact
approved main commit. For 0.2.0:

```bash
git switch main
git pull --ff-only
git status --short
git log -1 --oneline
# Check the approved commit and package version before creating a tag.
git tag -a v0.2.0 -m "ScGeo 0.2.0"
git push origin v0.2.0
```

The old `v0.2.0-geometry-core` checkpoint is not this release; do not move it.
The tag push runs [Release checks and TestPyPI](https://github.com/liuifrec/scgeo/actions/workflows/publish.yml).
It builds a wheel and sdist, checks their contents, executes library and optional
installation tests, and builds native conda from the inspected sdist.
Only after those checks pass, review and approve the **testpypi** deployment.
The final staging job checks live index hashes and tests a clean installation
of the verified wheel. Require all staging jobs to complete successfully.

Save the original successful tag-push run ID (attempt 1), its distributions and
release-audit artifacts and file hashes. They expire after 30 days. The new
production workflow fails closed on expired, missing, replaced or ambiguous
artifacts and on changed tags. If staging fails, investigate before retrying;
the production guard intentionally rejects rebuilt attempts. Do not silently
rebuild a version already staged or published.

## Promote the same bytes to production PyPI

Open [Promote verified ScGeo release to PyPI](https://github.com/liuifrec/scgeo/actions/workflows/publish-pypi.yml),
choose **Run workflow** on **main**, and enter:

- Version: `0.2.0`
- Staging run ID: the original successful tag-push run above.
- Publish: leave unchecked for a read-only dry run; check it only when ready
  for production publication and owner approval.

The production workflow verifies the staging repository, tag, original successful
run and every required job, unexpired artifact IDs and digests, main ancestry,
the release manifest, and original distribution bytes. Both downloads from
TestPyPI must independently match the original manifest; a smoke test installs
the verified wheel outside the checkout. Production upload refuses an existing
version rather than skipping files.

After the preparation jobs pass, review the source commit and exact SHA256
hashes in the run summary. Choose **Review deployments**, select **pypi**, and
approve only if the evidence is correct. The upload job has no source checkout
and does not execute package code or rebuild the distributions. It verifies
the manifest and both hashes immediately before Trusted Publishing.

Require **Verify REAL PyPI hashes and fresh pip installation** to pass. Only
then advertise `python -m pip install scgeo==0.2.0` as an available production
command, record the successful run, and update the GitHub release notes.
A partial upload or post-upload failure must be investigated; do not delete
files or move tags. If only verification fails, rerun failed jobs, not the
successful publishing job. The old 0.1.0 files remain unchanged.

## Bioconda after the production release

The repository recipe still uses `source.path: ..` for local developer builds.
For `bioconda/bioconda-recipes`, use the exact production PyPI sdist URL and its
verified SHA256 instead. Retain core dependencies, `noarch: python`, the license,
and maintainer `liuifrec`; do not add optional Census packages to the base recipe.
The release sdist includes `scripts/smoke_installation.py` for recipe testing.
Adapt the recipe to current Bioconda lint/testing requirements and submit it
under `recipes/scgeo/`. A tested local recipe is not a channel publication.

Wait for Bioconda review, build, upload, and a fresh channel installation before
advertising a Bioconda install command. Bioconda is for Linux/macOS; native
Windows users retain the pip or conda-plus-pip route.

## Official guidance

- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments
- https://packaging.python.org/en/latest/guides/using-testpypi/
- https://bioconda.github.io/contributor/index.html
- https://bioconda.github.io/contributor/guidelines.html
