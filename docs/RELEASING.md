# Distribution release checklist

The packaging workflow only builds and tests. It has read-only repository
permissions and no publishing job. Merging installation changes does not create a
PyPI release, a conda-channel release, a tag, or a new software DOI.

## Before a first public release

1. Require successful clean wheel and conda installation checks. Run the full
   library test suite and test any optional extras to be advertised. Investigate
   failures rather than weakening the tests to force a release.
2. Verify ownership and availability of the project name on PyPI and the target
   conda channel. A missing search result is not a guarantee of name availability.
3. Select and review a release version. Keep `pyproject.toml`,
   `scgeo/__init__.py`, and the normalized version in `conda-recipe/meta.yaml`
   consistent. The current `0.1.0-dev` version is not a stable release.
4. Record the release commit, supported platforms and Python versions, changes,
   dependency policy, and known limitations. Do not relabel a documentation or
   packaging commit as the frozen manuscript analysis checkpoint.

## PyPI

Build and inspect the distributions from the reviewed release commit:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Configure a PyPI pending/trusted publisher for owner `liuifrec`, repository
`scgeo`, a separately reviewed publishing workflow, and a protected GitHub
environment. The PyPI account owner must authorize the publisher. Do not place
API tokens or passwords in this repository or in a chat.

Test on TestPyPI first. TestPyPI is not a complete dependency mirror: test the
uploaded ScGeo distribution against dependencies installed from the normal
package index, rather than assuming every dependency exists on TestPyPI.

After explicit approval, publish the tested distributions to PyPI. Verify a
fresh `python -m pip install scgeo` outside the checkout before advertising that
command. Update the installation docs only after the public package exists and
its authorship, repository, and metadata have been checked.

Official guide:
https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/

## Native conda channel

The local recipe deliberately uses `source.path: ..` to test the checkout. For
public distribution, replace that local source with the exact released sdist URL
and its verified SHA256. Keep the dependency list synchronized with the wheel
metadata; do not add optional Census dependencies to the core recipe.

Choose one initial channel, such as Bioconda for bioinformatics software or
conda-forge, and follow its recipe review process. A recipe in this repository
does not make `conda install scgeo` available on that channel. Only advertise the
channel-specific install command after the recipe has been accepted, built,
uploaded, and tested in a fresh environment.

Official guides:
- https://bioconda.github.io/contributor/index.html
- https://bioconda.github.io/contributor/guidelines.html
- https://conda-forge.org/docs/maintainer/adding_pkgs/

Do not overwrite an existing immutable release or move an existing version tag.
Preserve the published paper's frozen checkpoints and numerical outputs.
