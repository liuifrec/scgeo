# Changelog

## Unreleased documentation updates

- Lead with the published PyPI installation and clarify the pending native
  Bioconda route, optional feature dependencies, and platform scope.
- Add a runnable synthetic example, an output preview, and clearer paths for
  package users, paper readers, and researchers reproducing the frozen workflows.
- Add BibTeX/RIS citation downloads and a Methods/provenance reporting guide.
- Refresh API references and manifests from current exported signatures and
  validate their version, coverage, and documentation consistency.

These updates leave package version 0.1.0 and scientific runtime code unchanged.
They do not change published distribution files or the manuscript's numerical
evidence. Corrections to the PyPI long description will accompany a future
package release.

## 0.1.0 — 15 October 2026

First production release, available on [PyPI](https://pypi.org/project/scgeo/0.1.0/).

- Includes ScGeo analysis functions and the published-paper citation.
- Installs geometry and plotting dependencies by default; Census is optional.
  Scanpy-dependent differential expression and Unix-only benchmark profiling
  load only when needed.
- Adds distribution/source integrity checks, library and optional-extra tests,
  cross-platform wheel installation checks, and native conda build checks.
- Published through verified TestPyPI staging and production promotion of the
  same wheel/sdist bytes. Native Bioconda publication is tracked separately in
  [PR #70318](https://github.com/bioconda/bioconda-recipes/pull/70318).

Python 3.12 is the recommended starting point; metadata declares Python >=3.9.
See [Installation](docs/INSTALLATION.md) for the tested configurations and
optional dependencies. Benchmark profiling requires Linux/WSL or macOS.

The manuscript's frozen package checkpoint remains
`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`. Version 0.1.0 is a distribution
release, not a replacement checkpoint for reproducing that analysis.
