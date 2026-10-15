# Bioconda submission for ScGeo 0.1.0

Status: recipe preparation only. A recipe in this repository is not a package
published on the Bioconda channel. Do not advertise Bioconda availability until
the upstream recipe is reviewed, merged, built, uploaded and verified.

## Exact release source

This candidate builds the published PyPI source distribution, not GitHub main:

- Version: `0.1.0`
- Source: `https://pypi.org/packages/source/s/scgeo/scgeo-0.1.0.tar.gz`
- Source SHA256: `87928f91cd5ecbe06347e5ea9f290602030e0af2a7f7a3fd831daaf488cea175`
- Package source tag: `v0.1.0`
- Package source commit: `6ab8a8ea4144f66613812b7b79976b1a67ad29d1`
- Successful production publishing run: `38692949218`

The source checksum is the same one approved on TestPyPI and verified during
production publication. Bioconda builds its own conda artifact from those source
bytes; this is not another PyPI upload or a change to the published package.

## Files to submit

Copy exactly these two files, preserving the filenames:

| This repository | In a fork of bioconda/bioconda-recipes |
| --- | --- |
| `packaging/bioconda/recipes/scgeo/meta.yaml` | `recipes/scgeo/meta.yaml` |
| `packaging/bioconda/recipes/scgeo/run_test.sh` | `recipes/scgeo/run_test.sh` |

Do not copy the GitHub workflow, this guide, data or compiled packages into the
upstream recipe submission. Leave the local-development recipe in
`conda-recipe/meta.yaml` unchanged.

The channel recipe adds a minor-version run export for the 0.x release and
uses `matplotlib-base`. Census, Scanpy, scVelo and notebooks remain optional;
they are not made core conda dependencies. Test-only pip and packaging checks
are in `run_test.sh`, because Bioconda's runtime-only container tests do not
include dependencies listed in `test.requires`.

## Submit through GitHub

1. Fork `bioconda/bioconda-recipes` into the `liuifrec` account. No separate
   Bioconda account, upload token or Trusted Publisher is needed for this route.
2. Create a branch such as `add-scgeo-0.1.0` from the up-to-date upstream default
   branch. Confirm that no existing recipe or channel package already covers it.
3. Add the two files above, then open an upstream pull request. Proposed title:
   `Add scgeo 0.1.0`.
4. Let Bioconda's lint, builds and isolated tests run. They are separate from the
   checks in this repository and may require maintainer approval for a first-time
   contributor. Address errors without skipping tests merely to obtain a pass.
5. Once checks pass, request maintainer review using the process in the current
   Bioconda contribution guide. Maintainers merge approved recipes; the upstream
   infrastructure builds and uploads to its channel.

The candidate's GitHub workflow builds from the real released sdist on Linux and
macOS with Python 3.12. Consult the actual Actions results for the checked commit.
These are standard conda-build checks, not a claim of official Bioconda lint or
isolated-container approval. They do not rerun manuscript analyses.

## Installation after channel publication

Only after a successful upstream upload and fresh-environment verification:

```bash
conda create -n scgeo --override-channels -c conda-forge -c bioconda --strict-channel-priority python=3.12 scgeo=0.1.0
conda activate scgeo
python -c "import scgeo; print(scgeo.__version__)"
```

Bioconda supports Linux and macOS, not native Windows. Native Windows users can
continue using the published pip package or the conda-plus-pip environment.
Do not alter the existing PyPI publisher, TestPyPI publisher, or release tag.

## Official guidance

- https://bioconda.github.io/contributor/setup.html
- https://bioconda.github.io/contributor/workflow.html
- https://bioconda.github.io/contributor/guidelines.html
- https://bioconda.github.io/contributor/linting.html
