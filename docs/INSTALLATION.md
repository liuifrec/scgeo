# Installation

Use a new environment rather than changing an environment used for a frozen
analysis. Python 3.12 is the recommended starting point. Package metadata retains
Python >=3.9; installation CI targets 3.9 and 3.12 on Linux, and 3.12 on Windows
and macOS. Check CI results before treating a platform as validated.

## pip: install directly from GitHub

Inside a virtual environment or an activated conda environment:

```bash
python -m pip install "scgeo @ https://github.com/liuifrec/scgeo/archive/refs/heads/main.zip"
python -c "import scgeo; print(scgeo.__version__)"
```

This is a non-editable development installation. It needs internet access but
not a manual clone or a Git executable. For a new Python virtual environment,
first run `python -m venv .venv`; activate it with `source .venv/bin/activate` on
Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

The source URL explicitly identifies this project. A bare `pip install scgeo`
should only be advertised after a release under this project's verified PyPI
account has been published. A README change does not publish a package.

## Default dependencies and optional extras

The default installation includes NumPy, pandas, AnnData, SciPy, scikit-learn,
Matplotlib, and Plotly for local representation analysis and plotting. It does
not install CELLxGENE Census, TileDB-SOMA, or Arrow as required ScGeo dependencies.
Census access uses the existing optional-dependency helpers.

| Extra | Use |
| --- | --- |
| `scanpy` | Scanpy-based preprocessing and helper workflows |
| `scvelo` | The existing scVelo, igraph, leidenalg, and scikit-misc extra |
| `census` | CELLxGENE Census access, TileDB-SOMA, and Arrow |
| `notebook` | JupyterLab and an IPython kernel |
| `dev` | pytest and Ruff for development |

The existing `plot` and `sklearn` extras remain accepted for compatibility,
although Matplotlib and scikit-learn are now included by default.

For Scanpy and notebooks:

```bash
python -m pip install "scgeo[scanpy,notebook] @ https://github.com/liuifrec/scgeo/archive/refs/heads/main.zip"
```

For Census access:

```bash
python -m pip install "scgeo[census] @ https://github.com/liuifrec/scgeo/archive/refs/heads/main.zip"
```

Choose extras only when required. Their upstream Python/platform requirements
may be narrower than ScGeo's core metadata; the installation CI does not certify
all optional extras. ScGeo consumes supplied representations, so scVI, CellRank,
and other upstream analysis tools are not silently added to the core install.

## conda: create a user environment

```bash
conda env create -f https://raw.githubusercontent.com/liuifrec/scgeo/main/environment-user.yml
conda activate scgeo
python -c "import scgeo; print(scgeo.__version__)"
```

The environment installs the scientific stack, Scanpy, and an IPython kernel
from conda-forge, then installs ScGeo with pip. This is **conda plus pip**, not a
claim that ScGeo has already been distributed on a conda channel. With an older
conda that cannot read a remote environment file, download `environment-user.yml`
and pass its local path to `conda env create -f`.

An existing environment named `scgeo` is not overwritten. Use `-n scgeo-new`
when creating a separate one, and activate that name. Install conda dependencies
before pip dependencies; recreate an environment rather than changing a frozen
analysis environment in place. `environment.yml` is retained unchanged for the
existing developer workflow.

Both the environment file and archive above follow `main`. They are convenient
for current development, but are not a lockfile or a frozen manuscript environment.

## Developer installation

```bash
git clone https://github.com/liuifrec/scgeo.git
cd scgeo
python -m pip install -e ".[dev,scanpy]"
python -m pytest -q
```

Use a dedicated environment. Editing the package requires a checkout; ordinary
users do not need an editable install.

## Local native conda build

A local-only `noarch: python` recipe is included in `conda-recipe/meta.yaml`.
It uses the current checkout, rather than a fabricated release URL or checksum:

```bash
conda create -n scgeo-build --override-channels -c conda-forge python=3.12 conda-build
conda activate scgeo-build
# Run from the repository root:
conda-build conda-recipe --override-channels -c conda-forge --python 3.12 --no-anaconda-upload
conda create -n scgeo-local --use-local --override-channels -c conda-forge python=3.12 scgeo=0.1.0.dev0
```

The recipe builds the core package. Its version is the normalized form of the
existing `0.1.0-dev` version, not a new stable release. It must be converted to
an immutable release source and checksum before submission to Bioconda or
conda-forge. See [Release checklist](RELEASING.md).

## Platform scope

Core analysis and plotting installation checks cover Linux, Windows, and macOS.
The separate `scgeo.bench` simulation suite currently uses Python's Unix-only
`resource` module for profiling. Run that suite on Linux (including WSL) or
macOS. ScGeo loads it only when requested, so it does not prevent ordinary
Windows users from importing ScGeo or using its analysis functions. This change
does not port or alter the frozen benchmark calculations.

## What installation checks cover

The packaging workflow builds an sdist and a wheel, checks distribution metadata,
installs the wheel using declared core dependencies, runs `pip check`, and checks
imports and a tiny sample-aware displacement calculation outside the source tree.
A separate conda job tests the user environment against the PR checkout and builds
the local native conda package. Nothing is uploaded to PyPI or a conda channel.

These checks do not reproduce the manuscript benchmarks or validate new scientific
results. Full library tests and optional workflow tests remain separate checks.

## Reproducibility

For a fixed GitHub install, replace `refs/heads/main.zip` with the full commit SHA
followed by `.zip`. Preserve the environment and input checksums as well; a fixed
source commit does not lock third-party dependencies.

The manuscript's frozen package checkpoint remains
`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`. Use the recorded workflow environments
and companion repository for manuscript reproduction. Installation convenience
changes do not replace that checkpoint or its numerical evidence.
