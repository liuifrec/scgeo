# Figures that keep the evidence visible

Build a state-evidence figure, inspect biological sample support, compare
contrasts, and export the exact plotted tables with your figure.

[Gallery](PLOTTING_GALLERY.md) · [PAGA shift maps](PAGA_SHIFT_MAP.md) · [Velocity fields](VELOCITY_FIELDS.md) · [Plotting API](api/pl.md) ·
[Interpretation](RESULT_INTERPRETATION.md) · [Citation guide](CITING.md)

[State associations and density](STATE_FLOW_AND_DENSITY.md) · [Real-data examples](REAL_DATA_PLOTTING.md) · [Geometry–velocity interpretation](GEOMETRY_VELOCITY_INTERPRETATION.md) · [Unified atlas](GEOMETRY_DYNAMICS_ATLAS.md) · [Embedding facets](EMBEDDING_FACETS.md) · [Shift explanations](EXPLAIN_SHIFT.md) · [Sparse states](CONDITION_SPARSE_STATES.md) · [Response compass](COMPASS_EVIDENCE.md)

## Availability and installation

These APIs are included in the 0.2.0 release source; see [Installation](INSTALLATION.md) for the verified index release or a pinned source installation.
The earlier 0.1.0 wheel does not contain these additions. To reproduce the example gallery from the matching checkout:

```bash
python -m pip install -e .
python -m pip check
git rev-parse HEAD
python examples/plotting_gallery.py --output-dir results/plotting_gallery
```

Record the commit printed by Git together with your analysis settings. The
synthetic examples use core dependencies and require no dataset download.
The separate [real-data gallery](REAL_DATA_PLOTTING.md) accepts verified local
public inputs or an explicit download request. The manuscript's frozen checkpoints are listed in
[Reproducibility](REPRODUCIBILITY.md).

## Compose a complete figure from standard AnnData

```python
result = sg.pl.geometry_dynamics_atlas(adata, color="cell_type", panels="auto", show=False)
result.save("figures/atlas", formats=("png", "svg", "pdf"))
```

The [unified geometry–dynamics atlas](GEOMETRY_DYNAMICS_ATLAS.md) combines
matched UMAP, projected-velocity, PAGA, and stored evidence panels when their
inputs exist. A plain AnnData embedding requires no optional velocity or graph
and creates a valid state-only panel. The atlas preserves all input analysis
results and explicitly records unavailable optional panels.

## Compare any metadata group with matched UMAP and projected velocity

```python
result = sg.pl.embedding_facets(
    adata, facet="condition", color="cell_type", velocity="grid",
    return_result=True, show=False,
)
result.save("figures/faceted_velocity", formats=("png", "svg", "pdf"))
```

The [faceting guide](EMBEDDING_FACETS.md) explains one- and two-way layouts,
missing labels, shared color scales and velocity support masks. The function
plots only existing coordinates and projected vectors; it has no organism- or
experiment-specific assumptions. Omit `velocity` to facet any existing UMAP.

## One evidence figure, several distinct questions

After a sample-aware `tl.robust_shift` analysis, build a report and select the
effect metric explicitly:

```python
import scgeo as sg

report = sg.get.state_report(adata, node_key="cell_type")
with sg.pl.style_context("paper"):
    result = sg.pl.state_evidence_panel(
        report,
        effect_metric="delta_norm",
        representation="X_pca",
        show_sample_counts=True,
        show_inference=True,
        evidence_mode="full",
        max_states=None,
        return_result=True,
        show=False,
    )
    result.save("figures/state_evidence", formats=("pdf", "svg", "png"), dpi=300)
```

The full mode shows evidence availability separately from effect magnitude and
representation consensus. Sample counts identify the number of biological
units in each condition; inference labels identify the stored resampling unit.
These annotations do not manufacture missing evidence or change classifications.

Raw magnitude intervals use `magnitude_ci95_low/high`. A normalized effect can
show an interval only when separately defined
`normalized_magnitude_ci95_low/high` columns are supplied. Every interval is
drawn at its stored endpoints, including when the point estimate is outside
that interval. A missing or partly missing interval stays unavailable. A
reversed finite interval raises an error. Magnitude intervals are not automatic
hypothesis tests.

The default order follows the displayed metric. Supply `state_order` to align
panels, or `sort_by` to choose another ordering deliberately. The state panel
puts requested states first and retains the others. `max_states=25, page=0`
shows the first page and annotates how many states are displayed; subsequent
pages use `page=1`, `page=2`, and so on.

## Use an inspectable plotting result

Supported upgraded plots accept `return_result=True`:

| Attribute | Contents |
|---|---|
| `result.fig` | Root Matplotlib Figure |
| `result.axes` | Named axes for further annotation |
| `result.artists` | Relevant plotted artists |
| `result.data` | Table or mapping of tables used by the plot |
| `result.metadata` | Metric, ordering, display choices, and available provenance |
| `result.warnings` | Interpretation or availability notes |

Existing default return values remain supported. All Matplotlib rendering
functions in `scgeo.pl` now accept `return_result=True`, including compatibility
aliases `score_umap` and `paga_scgeo`. `state_flow_sankey` retains its native
Plotly figure and `layout.meta` provenance; `perturbation_report` retains its
multi-figure bundle exporter. Utilities do not need a result flag. The
[API audit](PLOTTING_API_STATUS.md) records all 50 exported names and their
axes/result contracts.

The stored composition renderers `paga_composition_volcano`,
`paga_composition_bar`, and `paga_composition_panel`, plus
`distribution_test_volcano`, also accept `return_result=True`. Their exports
retain the original source table, transformed plot coordinates, missingness,
selected rows and stored analysis parameters where available.

Missing effects are never drawn as zero; missing p/q values are never filled
with one. Composition bars reserve labelled rows for unavailable effects,
while volcano plots omit incomplete effect/probability pairs and report the
omission count. When no p/q column exists at all, the composition volcano
uses an explicitly labelled effect-only strip. Its constant height carries
no probability meaning. Values outside the probability range raise; values
below `1e-300` use that floor only for display, with unchanged source values.

Stored odds ratios (`OR`) are supported on their original scale, with a null
reference of one; signed effects retain a reference of zero. No logarithm of
the effect or interval transformation is inferred. Existing intervals remain
in the source table; these plots do not manufacture or reinterpret intervals.
The composition panel accepts named `volcano`/`bar` axes, and retains its
historical `(fig, axes_array)` return by default.

## Compose panels in your own figure

The legacy summary family also supports `PlotResult`: `delta_rank`,
`density_overlap`, `density_overlap_grid`, `distribution_test`, and
`distribution_report`. Default return shapes are retained. Density summaries
accept named metric axes (`bc`, `hellinger`) or `summary` for global mode;
distribution summaries use `summary`. The report accepts `embedding`,
`density`, `test`, and `score` axes, with a categorical legend matching its
actual points. Source exports retain original and displayed rows, exact summary
text, and histogram counts.

For a global distribution-test figure, use `separate_metrics=True` to place
the statistic and probability on separate `statistic`/`probability` axes.
They have different units; the historical single-axis view remains the default
for compatibility. Density-grid exports record pixel centers, missing pixels,
the actual extent and whether any bounds came from the embedding. Its per-cell
fallback uses `score_embedding` and honors `ax`, `cax` and `colorbar=False`.
The [legacy API gallery](../examples/legacy_plotting_gallery.py) runs unchanged
ScGeo summary methods on the existing synthetic generator and exports their views.

Pass `ax=` for an upgraded single-axis plot or named `axes=` for multi-panel
functions. Supplied figures are not automatically resized or laid out.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
sg.pl.score_embedding(adata, score_key="score", basis="X_umap", ax=axes[0], show=False)
sg.pl.composition_drift(
    adata, node_key="cell_type", condition_key="condition",
    group0="control", group1="treated", panels=("change",),
    axes={"change": axes[1]}, show=False,
)
fig.tight_layout()
sg.pl.export_figure(fig, "figures/composed", formats=("pdf", "png"))
```

The score must already exist in `adata.obs['score']`. The composition panel
summarizes observed cell proportions; it does not replace sample-level
composition inference.

Common named axes are:

| Plot | Axis names |
|---|---|
| State evidence | `effect`, `samples`, `stability`, `local_preservation`, `local_distortion`, `dynamics`, `coverage` or `availability`, as applicable |
| Representation heatmap | `heatmap`, optional `consensus`, `diagnostics`, `colorbar` |
| Composition | `embedding`, `change`, `proportions`, selected by `panels` |
| Stored velocity alignment | `overlay`, `alignment` |
| PAGA shift map | `map`, optional `colorbar` |
| Local velocity field | `field`, optional `colorbar` |
| State associations | `alluvial` or `chord` |
| Embedding density | `density:<group>`, optional `colorbar` |
| Contrast comparison | Contrast labels |
| Biological replicates | State labels |
| Robustness | `heatmap`, optional `summary` |

When axes belong to a SubFigure, the returned result references the root Figure;
export saves the full parent canvas and records that scope.
The accompanying source tables cover the named result axes; sibling panels or
annotations added separately by the caller are not reconstructed automatically.

## Compose stored score panels

`alignment_panel`, `ambiguity_panel`, and `consensus_subspace_panel` reuse the
existing score and highlight renderers. Their top-k layer uses the same color
normalization as the full score layer, with one colorbar. Stored values are
never recalculated. Supply `axes={"embedding": ax0, "histogram": ax1,
"colorbar": cax}` or disable the scale with `colorbar=False`.

```python
result = sg.pl.alignment_panel(
    adata, score_key="stored_alignment", topk=30, bins=40,
    max_points=2000, return_result=True, show=False,
)
result.save("figures/alignment", formats=("png", "svg", "pdf"))
```

The histogram uses all finite input scores, including observations that cannot
be drawn because their embedding is nonfinite. Background sampling preserves
all drawable top-k cells or raises if the budget is too small. Exports include
every cell's selection/display flags and the exact histogram bin edges and
counts. Missing scores stay missing. `topk=0` or `None` disables highlighting.
The consensus panel also exports supplied singular values if available.

For a standalone overlay, `highlight_topk_cells(..., background=False,
norm=existing_norm, add_colorbar=False, return_result=True)` adds the selected
cells without repainting the background or changing its scale. A standalone
highlight plot without explicit limits retains its selected-score scale.
`score_embedding`/`score_umap` now allow `colorbar=False`, `cax=`, and
`legend=False`; valid Scanpy categorical palettes are reused automatically.

The [composition gallery](../examples/plotting_composition_gallery.py) exports
all three score panels from explicitly synthetic scores, as well as a composed
atlas/facet figure. Those examples exercise rendering; the scores are not
biological alignment or consensus evidence.

## Compare representations on meaningful scales

```python
result = sg.pl.representation_stability_heatmap(
    adata, metric="alignment_cosine",
    state_order=state_order, rep_order=["X_pca", "X_latent"],
    return_result=True, show=False,
)
```

Cosines use a diverging scale centered at zero with default limits `[-1, 1]`.
Coverage uses `[0, 1]`; alignment classes use discrete categories. Missing
entries remain visually distinct from zero. Supply `norm` or `vmin`/`vmax` to
coordinate compatible panels. The returned metadata records the scale.

Use `order_by='input'`, `'mean'`, or `'cluster'` deliberately. The deprecated
`cluster_states` and `cluster_reps` flags retain their historical mean sorting
and issue a warning; they do not silently become hierarchical clustering.
Keep related PCA or rotated views labeled as sensitivity analyses. The plot
does not redefine representation roles or recompute consensus.

## Compare effects across contrasts

`evidence_comparison` takes a tidy table, so archived source CSVs can be plotted
without reloading the expression object. Required fields are `state`,
`contrast`, `metric`, and `estimate`. Optional intervals require all three of
`interval_low`, `interval_high`, and `interval_type`.

```python
result = sg.pl.evidence_comparison(
    effects, metric="delta_norm",
    state_order=state_order, contrast_order=contrast_order,
    return_result=True, show=False,
)
```

Use explicit interval labels such as `sample_bootstrap_ci95` or
`representation_iqr`. Different interval types receive different line styles
and legend entries. Optional `evidence_status` or `status` labels remain
visible. Missing combinations are retained; duplicate state/contrast rows
must be resolved in the source table.

Automatic shared axes require the same nonempty `representation` and `unit`,
or a common nonempty `scale_id` that explicitly declares comparable coordinates.
Unrelated latent spaces must not be assigned the same scale identifier merely
to make a figure align. Comparison orders must include all observed labels;
filter the input explicitly to select a subset.

## Show biological samples, one point per unit

New `robust_shift` runs with `sample_key` retain their exact sample centers.
Retrieve those values and choose a coordinate explicitly:

```python
samples = sg.get.sample_centers(adata)
result = sg.pl.replicate_effect(
    samples, coordinate=0,
    condition_order=["control", "treated"],
    return_result=True, show=False,
)
```

Coordinates are zero-based. The plot gives each sample one point and equal
weight in its descriptive mean or median. That coordinate-wise summary is
not necessarily the multivariate robust condition center. Cell counts remain
descriptive. No confidence interval or hypothesis test is fitted here.

Pair lines require a real pairing identifier supplied through `pair_key`.
Independent mice, donors, or cross-sectional samples remain unconnected.
Old stored results without sample centers raise an actionable error; they are
never reconstructed from aggregate estimates. Rerunning an old analysis should
use its original data, parameters, and recorded environment in a separate output.

## Keep quantitative alignment independent of the display

```python
result = sg.pl.velocity_shift_alignment(
    adata, node_key="cell_type", condition_key="condition",
    group0="control", group1="treated",
    mode="stored", analysis_rep="X_pca", basis="umap",
    return_result=True, show=False,
)
```

Stored mode reads the selected representation's alignment table and displays
those fixed values beside an embedding overlay. Changing UMAP coordinates
does not change the quantitative panel. Display arrows require compatible
projected velocity coordinates; unavailable vectors are explained in the plot.
Legacy `mode='display'` retains its calculation in the first two display
coordinates and is labeled accordingly.

For state connectivity and condition centres on the same display, use the
[PAGA shift-map guide](PAGA_SHIFT_MAP.md). Its `alignment_source='stored'`
option uses the declared representation's quantitative alignment for node
colours, while the node positions and arrows remain in display coordinates.

For within-state directional patterns, set `show_velocity=True` together with
`velocity_mode='grid'` or `'stream'` on the PAGA map, or use `pl.velocity_field`
for a standalone display. The
[local velocity guide](VELOCITY_FIELDS.md) explains finite-radius support,
directional coherence, comparable condition panels, and field-source exports.
Coherence describes agreement among supplied projected vectors; it is not a
confidence interval or a probability that a lineage relationship is correct.

## Treat robustness and coverage separately

`robustness_matrix(..., summary='pass_rate', missing_policy='exclude')` divides
the number of passing settings by the number evaluated. Evaluation coverage is
shown separately. A completely unavailable row has an unavailable pass rate.
Select `missing_policy='fail'` for the historical policy that counts missing
settings as failures. Both policies are recorded in the returned metadata.

## Mapping and descriptive OOD displays

`mapping_confidence_umap` accepts any stored embedding, including an `X_`
prefix, despite its historical name. Low-score selection uses finite scores
from all cells; selected cells without finite coordinates remain in exported
sources with `highlighted=False`. Ties follow input order.

`mapping_qc_panel` composes on axes named `prediction`, `confidence`, `ood`,
plus `confidence_colorbar` and `ood_colorbar` when colorbars are enabled. Use
`colorbar=False` to omit scales. Reference Scanpy colors are reused where
available; novel labels receive deterministic colors and stay visible. Query
filtering applies only to predictions; scalar panels use all observations.
The exported legend counts include cells without drawable coordinates.

```python
result = sg.pl.mapping_qc_panel(
    adata, basis="umap", palette_from="reference_labels",
    condition_key="dataset", query_value="query",
    return_result=True, show=False,
)
result.save("figures/mapping_qc", formats=("png", "svg", "pdf"))
```

`ood_cells` never infers a threshold: without one, exported flags are unknown.
With a threshold, missing scores remain unknown; the historical monochrome
`show_only_flagged=True` view is retained. `ood_landscape` retains its historical
95th-percentile default (over finite input scores), with that choice recorded.
Its optional contours represent means within display-coordinate bins supported
by at least three cells; they are descriptive, not inferential boundaries.
Source exports contain every bin's location, count, mean, support status and
edges, plus exact contour levels and reasons for absent contours. Group
fractions retain all input group cells in their denominator, including cells
with unavailable scores (which are not flagged). No statistical test is fitted.

For caller-owned OOD summaries, pass `axes={"landscape": ax, "summary": summary_ax,
"colorbar": cax}`. All four mapping/OOD renderers support `return_result=True`
and retain their historical default returns. The executable
[legacy gallery](../examples/legacy_plotting_gallery.py) uses only synthetic
scores and explicitly labels them as illustrative; it fits no mapping model.

## Consensus selection and the state response compass

`consensus_structure(..., return_result=True)` exports all input cells, exact
finite-score selection, visible highlights, complete group summaries and
the filtered legend. Global top-k remains the default; use
`selection="threshold", threshold=...` for a declared common threshold.
The group and optional sample summaries expose all-cell and finite-score
denominators, observed zeros, and score coverage. Face and outline opacity are
independent. Selection statistics include cells without finite display
coordinates; the table makes that distinction explicit. `max_points` changes
only the display and retains all highlights and group/sample strata. Use `cax` or
`colorbar=False` to compose the view without changing neighbouring axes.
Without the result flag, the historical `(fig, ax, stats, legend_data)` remains.

`legend_from_data(legend_data, ax=legend_ax, return_result=True)` accepts a
caller-owned area. Its source table retains all entries when `max_items`
truncates the visible legend, and the result records that omission. Default
`(fig, ax)` returns and input dictionaries remain supported.

`recovery_compass` retains its map-first default and legacy return values.
Its name does **not** imply evidence of biological recovery. In `mode="display"`,
means, shifts and cosine are descriptive summaries in the first two display
coordinates. The corrected default `velocity_scope="contrast"` uses only the
selected conditions; request `"all"` explicitly for the older scope. Incomplete
vectors and undefined zero-shift alignment remain unavailable. OOD threshold,
scope and score coverage are exported; no finite scores means unavailable OOD.

Use `mode="stored"`, `shift_df`, optional `alignment_df`, and `analysis_rep`
to display existing analysis evidence without refitting it. Known provenance
conflicts raise; missing metadata stays unverified. Source counts determine
comparison support independently of the display subset. Pooled nodes retain
sparse states, while `layout="summary"` adds stored effect/alignment, observed
sample abundance and coverage axes. See the full
[compass guide and executable gallery](COMPASS_EVIDENCE.md).

The compass now validates PAGA dimensions, symmetry, grouping and matrix order
through the existing PAGA renderer. Supply `graph_node_order` when needed;
otherwise the chosen source of matrix order is recorded. Its opt-in result
exports cells, node summaries, all nonzero graph edges (including suppressed
ones), matrix labels, and actual glyph sizes/arrow endpoints. Stored mode adds
unchanged source tables and separate evidence/coverage availability. Shared
normalization, node-size references, point-sized arrowheads and label offsets
support comparable figures. Unknown alignment and insufficient condition
support stay unavailable. Place numeric scales with
`caxes={"fill": fill_ax, "arrow": arrow_ax}` alongside `ax`, or use `legend=False`.

## Style, display size, and export

Use `style_context('paper')` or `style_context('presentation')` for scoped font,
line, and export defaults. Settings are restored when the context exits.
`category_palette(labels)` gives stable colors across subsets; pass a curated
mapping when specific publication colors matter. Categorical score artists are
available through `result.artists["categories"][label]`, including labels such
as `"legend"` or `"missing"` that would collide with auxiliary artist names.
True missing entries receive a distinct display label, retained in source data;
distinct categories that stringify to the same text raise before rendering.

For supported embedding maps, `max_points`, `random_state`, and
`rasterized='auto'` control display cost. The sampler preserves each state and
explicit required highlights or raises when the budget cannot do so. Source
summaries and color normalization use the full data; metadata records the
displayed cell positions and seed.

```python
result.save(
    "figures/evidence", formats=("pdf", "svg", "png"), dpi=300,
    width_mm=180, include_data=True, include_metadata=True,
)
```

Canvas dimensions are exact with the default `bbox_inches=None`; choose
`bbox_inches='tight'` for cropping, which changes the exported dimensions.
Resizing an existing figure is not automatic reflow: inspect labels at the
intended size. Export restores the original canvas and temporary panel labels.
SVG text remains editable when exported within the paper/presentation style;
the viewing computer needs the selected font.

Every exported source table has a schema and SHA-256 digest in the JSON manifest.
Figure hashes, software versions, plotting settings, available analysis metadata,
and warnings accompany it. Add missing study identifiers or input checksums via
`metadata=`; the exporter cannot infer provenance that was never supplied.

`perturbation_report` also accepts `formats`, `dpi`, physical dimensions, and
`panel_kwargs`. Set `strict=True` or specify `required_panels` when an incomplete
bundle should fail instead of exporting available panels with warnings.
