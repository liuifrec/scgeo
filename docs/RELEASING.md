# Distribution release checklist

## Current scope

Version 0.1.0 is prepared for release review. The existing `packaging.yml`
workflow checks installation. The new `publish.yml` performs additional release
checks and can upload to **TestPyPI only**, after the owner creates a version tag
and approves the `testpypi` environment deployment. It has no production PyPI
upload job. Merging a pull request does not publish a package.

The registered pending publisher must be exactly:

| Field | Value |
| --- | --- |
| Project | `scgeo` |
| GitHub owner | `liuifrec` |
| Repository | `scgeo` |
| Workflow filename | `publish.yml` |
| Environment | `testpypi` |

Use GitHub Settings > Environments > testpypi. Save `liuifrec` as a required
reviewer and leave Prevent self-review unchecked for a solo maintainer. Under
Selected branches and tags, add **Ref type: Tag**, **Name pattern: `v*`**.
A branch rule named `v*` is not equivalent. Zero matching tags is normal before
the first release. No password or API token belongs in repository secrets.
Pending publishers do not reserve the package name; the first upload must
succeed under the authorized owner's account.

## Before creating a tag

Review the release PR and require both the packaging checks and release checks
to pass. `publish.yml` builds one source distribution and wheel, verifies their
versions, package-source bytes, dependencies, license and required files, runs
the complete default library suite on Linux/Python 3.9 and 3.12, checks selected
optional-extra installations on Linux/Python 3.12, and builds/tests a native
conda package from that exact sdist. Existing opt-in/network test skips remain
visible in the test report; they are not silently treated as passed.

Do not weaken tests to force a release. Inspect any failures and record any
compatibility bounds supported by that evidence. Preserve the manuscript's
frozen package checkpoint `9a0ed16cbaa57f935f9c9bc87d1643a25b51012c` and numerical
outputs. This version is not a replacement manuscript-analysis checkpoint.

Keep `pyproject.toml`, `scgeo/__init__.py`, and `conda-recipe/meta.yaml` versions
consistent. The tag must exactly equal `v` plus that version. The tagged commit
must already be reachable from `main`. Do not move a published tag or overwrite
immutable release files.

## Stage on TestPyPI

After review and merge, the owner can create the first tag from the exact
approved commit, for example in a clean local checkout:

```bash
git switch main
git pull --ff-only
git status --short
# Confirm this is the approved release commit before tagging:
git log -1 --oneline
git tag -a v0.1.0 -m "ScGeo 0.1.0"
git push origin v0.1.0
```

The tag push runs **Release checks and TestPyPI**. Ordinary main-branch pushes
and pull requests run its checks but cannot upload. A manual Actions run also
requires selecting the release **tag**, not main, to reach the upload job.

After the checks pass, open the run in Actions, choose Review deployments,
select `testpypi`, and approve. Do not bypass a failed gate. The publishing job
only downloads the tested distributions and uses Trusted Publishing; it does
not check out or build source code with an upload identity.

The following job downloads the uploaded wheel and sdist, checks both against
the saved SHA256 hashes, installs core dependencies from normal PyPI, installs
the verified ScGeo wheel with `--no-deps`, and runs the installed-package smoke
test. A successful upload alone is not the end of staging: require this final
verification job to pass as well.

Save the run's `scgeo-distributions` and `scgeo-release-audit` artifacts; these
expire after 30 days. The audit includes the source commit, tag ref, filenames,
SHA256 checksums, dependency metadata and build environment. Test XML and
resolved test/optional environments are separate artifacts. These records are
not a software DOI and do not lock dependencies for later installations.

## Promote to production PyPI: a separate owner-approved step

Create and secure the real PyPI account, verify project-name ownership, and
register its Trusted Publisher. TestPyPI accounts/publishers do not authorize
production PyPI. Create a protected `pypi` environment before enabling production
publishing. Do not add upload credentials to chat or source control.

Production publishing is deliberately not enabled in this first staging PR.
A separately reviewed promotion step must retrieve the exact files from the
successful TestPyPI run (or saved artifacts), verify their manifest and TestPyPI
hashes, and upload those **same bytes** to PyPI after explicit approval. Do not
silently rebuild or replace version 0.1.0 between staging and production.

Only after a clean installation of the real PyPI package succeeds should the
README advertise `python -m pip install scgeo`.

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
