# Install ScGeo

[Quick start](QUICKSTART.md) · [API reference](api_reference.md) · [Paper reproduction](REPRODUCIBILITY.md)

**ScGeo 0.1.0 is published on [PyPI](https://pypi.org/project/scgeo/0.1.0/).**
Python 3.12 is the recommended starting point. Use a fresh environment for a new
project and preserve environments that support frozen analyses.

## Choose an installation

| Workflow | Command |
| --- | --- |
| Core geometry and plotting | `python -m pip install scgeo==0.1.0` |
| Standard Scanpy/AnnData analysis | `python -m pip install "scgeo[scanpy]==0.1.0"` |
| Scanpy with JupyterLab | `python -m pip install "scgeo[scanpy,notebook]==0.1.0"` |
| scVelo workflow dependencies | `python -m pip install "scgeo[scanpy,scvelo]==0.1.0"` |
| CELLxGENE Census access | `python -m pip install "scgeo[census]==0.1.0"` |

The core package includes NumPy, pandas, AnnData, SciPy, scikit-learn, Matplotlib,
and Plotly. ScGeo consumes existing representations; construct scVI, CellRank,
or other upstream model outputs with the appropriate tools for your workflow.

## Create a Python environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and verify:

```bash
python -m pip install "scgeo[scanpy]==0.1.0"
python -m pip check
python -c "import scgeo; print(scgeo.__version__)"
```

The last command should print `0.1.0`. Continue with the
[download-free synthetic example](QUICKSTART.md#try-the-complete-example).

## Use conda to manage the environment

```bash
conda create -n scgeo -c conda-forge python=3.12 pip
conda activate scgeo
python -m pip install "scgeo[scanpy]==0.1.0"
python -m pip check
```

Conda supplies the environment and pip installs the released ScGeo package.
As of **17 November 2026**, native Bioconda publication is pending in
[PR #70318](https://github.com/bioconda/bioconda-recipes/pull/70318).
See [Bioconda status](BIOCONDA.md) for the submission and publication checks.

The repository's [`environment-user.yml`](../environment-user.yml) installs a
conda-forge scientific stack followed by **GitHub main.zip**. Use it for current
development, rather than as a pinned release or manuscript environment:

```bash
conda env create -f https://raw.githubusercontent.com/liuifrec/scgeo/main/environment-user.yml
conda activate scgeo
```

Use another environment name if `scgeo` already exists. Install conda
dependencies before pip dependencies, and preserve the resolved environment
with your analysis records.

## Feature-specific dependencies

Some APIs require packages beyond the geometry core:

| Feature | Requirement | How to install |
| --- | --- | --- |
| `tl.alignment_driver_genes` | Scanpy | `python -m pip install "scgeo[scanpy]==0.1.0"` |
| `pp.build_reference_pool` and reference-pool search with the default backend | PyNNDescent | `python -m pip install pynndescent` |
| `tl.paga_composition_stats(method="gee")` | statsmodels | `python -m pip install statsmodels` |
| Census access | Census, TileDB-SOMA, Arrow | `python -m pip install "scgeo[census]==0.1.0"` |
| Census reference-pool construction | Census stack and PyNNDescent | `python -m pip install "scgeo[census]==0.1.0" pynndescent` |

The tested Scanpy extra also supplies PyNNDescent and statsmodels transitively.
They are listed here explicitly because ScGeo 0.1.0 does not declare separate
reference/statistics extras. Select the intended statistical method explicitly;
bootstrap and GEE analyses are different procedures.

Other extras: `notebook` installs JupyterLab/ipykernel; `scvelo` installs scVelo,
igraph, leidenalg, and scikit-misc; `dev` installs pytest and Ruff. The legacy
`plot` and `sklearn` extras remain accepted, although those packages are now
included in the core. Optional tools can have narrower Python/platform support
than ScGeo itself.

### Python 3.9 and PyNNDescent

For Python 3.9 workflows that use PyNNDescent, including default reference-pool
indexing, constrain the optional dependency to a compatible release:

```bash
python -m pip install "scgeo[scanpy]==0.1.0" "pynndescent<0.7; python_version < '3.10'"
python -m pip check
```

As of 17 November 2026, PyNNDescent 0.7.0 fails during index compilation with
Numba 0.60.0, the newest Numba available for Python 3.9. The problem and the
compatible PyNNDescent 0.6.0 combination are documented in
[upstream issue #288](https://github.com/lmcinnes/pynndescent/issues/288).
Apply the same constraint when adding PyNNDescent to a Census environment.
The Python 3.9 library CI uses this compatibility constraint and retains the
reference-pool tests. Python 3.12 remains the recommended starting point for
new analyses.

## Platform and verification scope

Package metadata declares Python >=3.9. Release CI has checked core wheel
installation on Linux (3.9 and 3.12), Windows (3.12), and macOS (3.12).
The 17 November 2026 Linux/Python 3.12 audit also passed a fresh PyPI install,
dependency checks, the installed-package smoke test, and all **191 non-network
library tests with the Scanpy extra**. One dataset-download integration test was
excluded; optional live services and full manuscript workflows are separate checks.

The `scgeo.bench` simulation profiler uses the Unix-only `resource` module.
Run benchmarks on Linux/WSL or macOS. Core analysis and plotting remain
available on native Windows. See the
[installation workflow](../.github/workflows/packaging.yml) and
[release workflow](../.github/workflows/publish.yml) for validation details.

## Development installation

For editable source:

```bash
git clone https://github.com/liuifrec/scgeo.git
cd scgeo
python -m pip install -e ".[dev,scanpy]"
python -m pytest -q -m "not integration"
```

For a non-editable development snapshot without cloning:

```bash
python -m pip install "scgeo @ https://github.com/liuifrec/scgeo/archive/refs/heads/main.zip"
```

For a fixed source snapshot, use the full commit SHA in the archive URL instead
of `refs/heads/main`. A source pin does not lock third-party dependencies.

## Local conda builds and manuscript reproduction

[`conda-recipe/meta.yaml`](../conda-recipe/meta.yaml) builds the local checkout;
[`packaging/bioconda/recipes/scgeo`](../packaging/bioconda/recipes/scgeo) contains
the submitted recipe using the immutable PyPI sdist and checksum. Building a
local artifact does not publish it to a channel. Maintainers should follow
[Bioconda status](BIOCONDA.md) and the [release process](RELEASING.md).

To reproduce the paper, use the recorded workflow environments and package
checkpoint `9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`, together with the companion
repository and input checksums. See [Reproducibility](REPRODUCIBILITY.md).
