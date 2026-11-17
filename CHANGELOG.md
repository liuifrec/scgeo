# Changelog

## Unreleased plotting upgrades

These changes are available in the development checkout and will accompany a
future release. The published PyPI 0.1.0 files are unchanged. See the
[plotting guide](docs/PLOTTING.md) and [visual gallery](docs/PLOTTING_GALLERY.md).

- Preserve exact stored interval endpoints, validate intervals before drawing,
  and keep raw and normalized uncertainty in their corresponding units.
- Add effect selection, sample/inference annotations, evidence availability,
  explicit ordering/pagination, and supplied axes to the state-evidence panel.
- Give representation metrics appropriate color scales and explicit input,
  mean, or hierarchical ordering. Legacy `cluster_*` flags retain mean sorting
  with a deprecation warning.
- Separate pass rates among evaluated settings from evaluation coverage.
  `missing_policy='fail'` retains historical missing-as-failure behavior;
  the new default is `missing_policy='exclude'`.
- Add opt-in `PlotResult`, scoped paper/presentation styles, stable palettes,
  and configurable PDF/SVG/PNG export with source CSVs, schema, and checksums.
- Add table-based `evidence_comparison` and `replicate_effect`; retain exact
  biological sample centers in new sample-aware `robust_shift` outputs and
  expose them through `get.sample_centers`, without changing existing estimates.
- Add stored quantitative alignment beside a separately labeled display
  overlay; retain the legacy display-space calculation as its default mode.
- Make composition panels embeddable and replace the gallery's composition
  placeholder with a real plot. Normalize embedding aliases and add
  deterministic display-only sampling with retained groups/highlights.
- Extend report exports with configurable formats/dimensions, required-panel
  validation, and strict mode. Add executable examples and plotting CI.

Compatibility notes: legacy return values remain the default. Default state
sorting now follows the displayed effect metric; pass-rate denominators now
exclude missing settings unless `missing_policy='fail'` is chosen. These are
documented display-policy corrections. No scientific thresholds or resampling
procedures are changed, and frozen manuscript evidence is not regenerated.

## Unreleased documentation updates

- Lead with the published PyPI installation and clarify the pending native
  Bioconda route, optional feature dependencies, and platform scope.
- Add a runnable synthetic example, an output preview, and clearer paths for
  package users, paper readers, and researchers reproducing the frozen workflows.
- Add BibTeX/RIS citation downloads and a Methods/provenance reporting guide.
- Refresh API references and manifests from current exported signatures and
  validate their version, coverage, and documentation consistency.

These documentation changes do not modify published distribution files or the
manuscript's numerical evidence. Runtime additions in the current checkout are
listed separately above. Corrections to the PyPI long description will accompany
a future package release.

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
