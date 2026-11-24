# Changelog

## Unreleased shift explanations and response curves

- Add `tl.explain_shift` for checked affine gene attribution to stored mean or
  geometric-median shifts, preserving sample hierarchy and the source estimator's
  numerical precision. Export mapping/reconstruction residuals and input signatures.
- Add signed program contributions, optional local contrasts and per-sample
  composition, paired/unpaired sample bootstrap stability, and sourced signed
  regulon target responses. No causal or independent gene-association test is inferred.
- Add `pl.shift_curves` with condition curves fitted within samples, equal sample
  weighting, support gaps, observed sample ranges, and coordinated expression/
  contribution maps; add `pl.shift_embedding` for schematic curved center connectors.
- Retain legacy APIs and estimators. Add scientific edge-case and plot contracts,
  a reproducible synthetic gallery, export tables, and generated API documentation.

## Unreleased deployment readiness

- Add read-only `get.input_contract` for explicit AnnData metadata, coverage,
  representations, projected velocity, PAGA provenance and stored evidence.
- Add exact-source environment preflight and an isolated JupyterHub installation
  guide that preserves existing kernels and records dependency versions.
- Validate a four-sample synthetic workflow, six 2-vs-2 label assignments, stored
  state reports and figure/source exports. Existing statistical estimands and
  plotting signatures are unchanged; no biological validation is claimed.

## Unreleased plotting composition and source exports

- Reconcile PR #21's successful CI and public-data checks; correct unsupported
  facets' exported grids when using independent panel limits.
- Accept caller-owned Figure/SubFigure axes in embedding facets and the atlas,
  with explicit colorbar placement and parent-canvas export scope.
- Upgrade alignment, ambiguity and consensus score panels to shared rendering,
  named axes and opt-in `PlotResult`, retaining default tuple returns. Highlights
  now match the full score normalization and use a single colorbar. Histograms
  retain all finite scores independently of background drawing limits.
- Add full selection/display source tables to `highlight_topk_cells`, explicit
  colorbar/legend controls and Scanpy palette reuse to `score_embedding`, and
  a synthetic composition gallery with checksummed source exports.
- Preserve missing effects and p/q values in composition/volcano plots, validate
  probability ranges and add source exports/named composition axes. Accept
  stored odds ratios on their original scale with a null reference at one.
- Audit all 50 plotting exports in [the API status matrix](docs/PLOTTING_API_STATUS.md).
  All Matplotlib renderers now expose the opt-in result contract; Plotly and
  bundle exports retain their native contracts. No statistical estimator or
  published release changes in this continuation.
- Extend ranking, overlap, distribution and report views with axes/source
  exports; fix density fallback axis forwarding and report legend colors.
  Global distribution tests can use separate metric axes while preserving the
  historical default return shape.

- Add axes/results to mapping and OOD displays, retaining novel predictions
  outside reference palettes and distinguishing genuine missing labels. Export
  exact thresholds, display selections and supported contour bins; correct the
  contour coordinate transpose without changing the binned-mean estimator.

- Complete consensus structure, standalone legend and recovery-compass exports.
  Retain full source data when legends are truncated; preserve parent figures.
  Reuse PAGA graph validation and label compass values as descriptive display-
  coordinate summaries, with missing/insufficient alignment kept unavailable.

- Harden categorical scores against colliding string labels and reserved artist
  names, retain separate missing-label identity, and reject invalid category
  scales before figure creation. Add an automated audit of all public plotting
  exports and their axes/result contracts.

## Unreleased flexible embedding and velocity faceting

- Add `pl.embedding_facets` for one- or two-way faceting by any `adata.obs`
  categories, including ordered levels, literal-versus-missing identity,
  explicit empty combinations, and deterministic drawing-only subsampling.
- Reuse existing UMAP/embedding coordinates and matching projected velocity.
  Condition, batch, sample or cell-type facets share display limits, a
  pooled Gaussian support radius, common local velocity grid positions and
  optionally one grid-arrow multiplier. No kinetics or trajectory is fitted.
- Preserve Scanpy palettes and one global numeric color norm; export complete
  cell membership, supported/unsupported velocity grids, actual arrows or
  support-audited streamline vertices, and figure provenance.
- Add a synthetic five-view gallery, strict size/SHA-256 pinned public PBMC3k
  and CellRank pancreas checks, documentation and Python 3.9/3.12 regression
  coverage. Public examples never invent contrasts absent from their source.

See [Embedding faceting](docs/EMBEDDING_FACETS.md). The features require this
unreleased development checkout; they are not in PyPI `scgeo==0.1.0`.


## Unreleased unified geometry–dynamics atlas

- Add `pl.geometry_dynamics_atlas` as a composable, AnnData-native entry point for
  matching cell-state UMAP, projected velocity, PAGA, and explicitly stored
  evidence panels; a simple embedding still yields a valid one-panel atlas.
- `panels='auto'` records omitted optional panels; explicit requests fail
  clearly when their inputs are absent. Shared embedding limits, stable node
  palettes, highlighting, reproducible display sampling and named axes make
  figure assembly more consistent.
- Preserve scientific roles: UMAP arrows are display projections, PAGA edges
  undirected, and numeric effects come only from a compatible stored report.
  State and condition provenance is verified; raw CIs never migrate to
  normalized metrics. `sample_key` is optional and descriptive only.
- Add an executable synthetic gallery, opt-in checksum-pinned public PBMC and
  pancreas examples, complete `PlotResult` source-table exports, regression
  tests, API documentation and CI checks.

See [Unified geometry–dynamics atlas](docs/GEOMETRY_DYNAMICS_ATLAS.md).
Development checkout only; not in the published `scgeo==0.1.0` wheel.


## Unreleased geometry–velocity interpretation upgrade

- Add `pl.geometry_velocity_concordance` to relate stored state displacement magnitude to signed stored alignment while retaining all ineligible states and missing-data reasons.
- Add `pl.velocity_condition_compare` with shared projected-velocity grid coordinates, support thresholds, radius, color scale, and arrow multiplier; directional agreement is displayed only where both conditions are supported.
- Add display-only `paga_shift_map(..., focus_states=..., edge_focus='incident'|'within')` to reduce graph clutter without altering the PAGA matrix or quantitative analysis.
- Document private, explicit-input review workflows and preservation of per-mouse source data. The mouse smoke archive provides frozen state tables but not full velocity projections; no new mouse kinetics or significance tests are claimed.

See [Geometry–velocity interpretation](docs/GEOMETRY_VELOCITY_INTERPRETATION.md). Development checkout only; not in the published `scgeo==0.1.0` artifacts.

## Unreleased state-association and density upgrades

Add a two-column `pl.state_flow_chord` and improve the alluvial, Sankey, and
embedding-density displays. See the [state-association guide](docs/STATE_FLOW_AND_DENSITY.md)
and [real-data plotting examples](docs/REAL_DATA_PLOTTING.md).

- Preserve full category totals when `min_count` hides a ribbon. Alluvial and
  chord plots reserve the omitted link's space and export its original count.
- Separate genuinely missing annotations from literal strings such as `NA`;
  use unique stage-specific identities and explicit category orders.
- Add readable category labels, consistent palettes, count/percentage labels,
  caller-axis support, and complete `PlotResult` sources for the static views.
- Correct the density contour pipeline to smooth and transform once, and derive
  contour levels from the values actually rendered. Per-group normalization
  now uses a truthful relative 0–1 colorbar; global normalization shares the
  same absolute scale across panels.
- Add common density grids, finite-coordinate exclusions, visible empty groups,
  supplied axes/colorbars, and source tables that reconcile to the raw counts.
- Add reproducible public PBMC3k and pancreas plotting checks with input hashes,
  supplied annotations/coordinates, and explicit upstream velocity provenance.

State links are same-cell categorical associations, not observed longitudinal
transitions. Embedding-density panels describe the selected display coordinates,
not a biological-sample enrichment test. These additions require the development
checkout and are not present in the published `scgeo==0.1.0` distribution.

## Unreleased local velocity-field upgrades

PAGA maps now support local projected-velocity grids and streamlines. The new
`pl.velocity_field` also provides standalone and matched-condition views. See
the [local velocity guide](docs/VELOCITY_FIELDS.md) and executable gallery.

- Add opt-in `velocity_mode='grid'` and `'stream'` to `paga_shift_map` and its
  `paga_scgeo` wrapper. The default `velocity_mode='state'` preserves the
  existing reference-anchored state arrows.
- Interpolate supplied projected vectors with finite-radius Gaussian weights;
  retain support counts, local direction coherence, and explicit omission
  reasons. Unsupported gaps and opposing-vector regions remain identifiable.
- Keep cell-background sampling separate from field computation. Explicit
  `grid_extent`, radius, color normalization, and numeric arrow scales support
  matched condition panels; visual axis limits do not redefine the field.
- Export complete field inputs, grid diagnostics, actual arrow endpoints, and
  rendered streamline paths through `PlotResult`.
- Add an offline synthetic gallery, API documentation, regression tests, and
  installation checks for both field rendering modes.

These are descriptive interpolations in the supplied display coordinates.
Directional coherence is not statistical confidence, streamlines are not
observed cell trajectories, and vector scales do not establish elapsed time.
The plotting functions require a matching projected velocity array and do not
fit or project an RNA-velocity model. These features require the development
checkout; the published `scgeo==0.1.0` distribution does not include them.

## Unreleased PAGA shift-map upgrades

The development `paga_shift_map` now brings state connectivity, condition
centres, and supplied projected velocity into a consistent display. See the
[PAGA guide and gallery](docs/PAGA_SHIFT_MAP.md).

- Use equal coordinate scaling, optional shared limits, distinct reference and
  comparison markers, readable labels with leader lines, and graphical legends.
- Draw arrow heads and composition pies in display units by default. Existing
  explicit `arrow_width` values retain the legacy data-unit arrow style;
  the new default is `arrow_width=None`.
- Add `alignment_source='stored'` with explicit representation selection and
  contrast validation. Quantitative node colours use stored analysis results;
  positions and condition arrows remain descriptive display-space summaries.
- Restrict projected velocity and composition summaries to the selected two
  conditions, and require velocity coordinates compatible with the display.
- Validate connectivity matrices and their node identities before drawing.
  `graph_node_order` declares matrix row labels for supplied or reordered graphs.
- Add deterministic display sampling, shared colour scales, `PlotResult`, and
  export tables for cells, centres, edges, vectors, and label placement.
- Add an executable synthetic gallery and installation/CI coverage. The
  `paga_scgeo` wrapper accepts the same new options.

Default figure/axis and `return_data` contracts remain supported. The new
condition scope deliberately excludes unrelated conditions from velocity and
pie summaries; the background can still show all cells. Package version,
published distribution files, and frozen manuscript analyses are unchanged.

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
