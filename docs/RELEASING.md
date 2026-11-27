# Distribution release checklist

## Current release status

The 0.2.1 visualization candidate is under review. Do not tag or promote it until CI and real-data visual review are complete. The production 0.2.0 release record below remains unchanged.

The first production release, ScGeo 0.1.0, was published on
[PyPI](https://pypi.org/project/scgeo/0.1.0/) on 15 October 2026. Native Bioconda
publication is tracked separately in [PR #70318](https://github.com/bioconda/bioconda-recipes/pull/70318).
ScGeo 0.2.0 was published on [production PyPI](https://pypi.org/project/scgeo/0.2.0/) on 26 November 2026. The verified public installation is `python -m pip install scgeo==0.2.0`. User commands are in [Installation](INSTALLATION.md).

The historical 0.1.0 production promotion remains documented in
[the publication record](PYPI_PUBLICATION.md). Do not modify the original tag,
published artifacts, checksums, or manuscript checkpoint.

## Verified 0.2.0 publication record

- Tag: [`v0.2.0`](https://github.com/liuifrec/scgeo/tree/v0.2.0)
- Source commit: `46ba9017ceef17255292edd0140602ed13ac5305`
- Original TestPyPI tag-push run: [45764454861](https://github.com/liuifrec/scgeo/actions/runs/45764454861), attempt 1 (all nine jobs successful).
- Production promotion: [45765641121](https://github.com/liuifrec/scgeo/actions/runs/45765641121), all four jobs successful; completed 26 November 2026 at 06:15 UTC.
- Original staging distribution artifact ID: `15091765301`; original release-audit artifact ID: `15091282735`.

| Distribution | SHA256 |
| --- | --- |
| `scgeo-0.2.0-py3-none-any.whl` | `105fe2e702ff995a1b46a9f09a5346a784175f9e75c03edf978bc7c728507a37` |
| `scgeo-0.2.0.tar.gz` | `2bdcf4720b068bff7c03d282ec6dcab4a4d494376de8edcf096dc1732d1b9546` |

The TestPyPI verification job recorded these hashes, and production verified
both files against the same release manifest. The production workflow installed
0.2.0 from PyPI in a fresh Python 3.12 environment, passed `pip check` and
installed-package smoke checks. The tagged library suite passed 785 tests with
one optional integration skip on each of Python 3.9 and 3.12. This is software
validation, not additional biological validation.

The procedure below applies to future releases. The 0.2.0 tag and artifacts
already exist; do not repeat its tag creation or publication.

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
approved main commit. The completed 0.2.0 release used the following commands
(shown for provenance; substitute a new, reviewed version for future releases):

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

- Version: the new approved release version (the completed example was `0.2.0`).
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
then advertise the new pinned production installation command, record the
successful run, and update the GitHub release notes.
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
