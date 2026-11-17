# Figures that keep the evidence visible

Build a state-evidence figure, inspect biological sample support, compare
contrasts, and export the exact plotted tables with your figure.

[Gallery](PLOTTING_GALLERY.md) · [PAGA shift maps](PAGA_SHIFT_MAP.md) · [Velocity fields](VELOCITY_FIELDS.md) · [Plotting API](api/pl.md) ·
[Interpretation](RESULT_INTERPRETATION.md) · [Citation guide](CITING.md)

[State associations and density](STATE_FLOW_AND_DENSITY.md) · [Real-data examples](REAL_DATA_PLOTTING.md)

## Availability and installation

The plotting additions on this page are **unreleased development features**.
They require the source checkout containing this guide. The published PyPI
package `scgeo==0.1.0` does not contain these additions. From that checkout:

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

Existing default return values remain supported. This opt-in result is
available for `state_evidence_panel`, `representation_stability_heatmap`,
`consensus_state_map`, `local_distortion_map`, `composition_drift`,
`gallery_overview`, `robustness_matrix`, `velocity_shift_alignment`,
`score_embedding`/`score_umap`, `evidence_comparison`, `replicate_effect`,
`paga_shift_map`/`paga_scgeo`, `velocity_field`, `state_flow_alluvial`,
`state_flow_chord`, and `embedding_density`.
Other plotting functions retain their documented interfaces.

## Compose panels in your own figure

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

## Style, display size, and export

Use `style_context('paper')` or `style_context('presentation')` for scoped font,
line, and export defaults. Settings are restored when the context exits.
`category_palette(labels)` gives stable colors across subsets; pass a curated
mapping when specific publication colors matter.

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
