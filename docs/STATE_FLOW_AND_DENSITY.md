# State associations and embedding density

Use alluvial and chord plots to inspect how the same cells are assigned across
categorical observation columns. Use density panels to inspect where annotated
groups occur in a shared display embedding. These are descriptive views of the
supplied cells and annotations.

[Plotting guide](PLOTTING.md) · [Real-data examples](REAL_DATA_PLOTTING.md) ·
[Plotting API](api/pl.md) · [Interpretation](RESULT_INTERPRETATION.md)

The upgrades described here are **unreleased development features**. Use the
source checkout containing this guide; the published `scgeo==0.1.0` package
does not contain them.

## Choose the view for the question

| View | Use it for | Main quantity |
|---|---|---|
| `state_flow_alluvial` | Comparing two or more ordered annotation columns | Cell counts shared by adjacent columns |
| `state_flow_chord` | A compact circular comparison of two annotation columns | Cell counts connecting the two sets of categories |
| `state_flow_sankey` | Interactive inspection with hover information | Counts on visible links, with complete and omitted counts reported |
| `embedding_density` | Comparing the spatial distribution of groups on one embedding | Counts per common display bin, with an explicit display normalization |

An observation contributes once to each adjacent pair of columns. A ribbon
between an annotation and a mapping label is a same-cell cross-tabulation.
A plot of `timepoint`, `alignment_group`, and `macrostate` is also a
cross-tabulation of those columns: it does not track the same cell through time.
The column names and upstream analysis determine what the association means.

## Readable alluvial and chord plots

```python
import scgeo as sg

alluvial = sg.pl.state_flow_alluvial(
    query,
    columns=["dataset_annotation", "mapped_annotation"],
    min_count=2,
    color_by="source",
    label_format="both",
    return_result=True,
    show=False,
)

chord = sg.pl.state_flow_chord(
    query,
    columns=["dataset_annotation", "mapped_annotation"],
    min_count=2,
    color_by="source",
    label_format="count",
    return_result=True,
    show=False,
)
```

Both calls use categorical columns that already exist in `query.obs`. The
chord view accepts exactly two columns. Its two sides remain separate even
when they contain identical label names. It shows category associations;
source and target refer to the two displayed columns, not inferred biological
direction or causation.

### Filtering must preserve the denominator

`min_count` controls which ribbons are drawn. All positive associations are
counted and receive layout space before this filter is applied. The full
category totals remain visible, and hidden links leave unfilled space. Retained
links are never enlarged to fill the omitted mass.

For example, if a 24-cell cross-tab has link counts 12, 6, 4, and 2, then
`min_count=5` draws 18 cells' associations and omits 6. The displayed links
still represent 18/24, not a renormalized 18/18. The source tables retain all
four links, with `displayed` and `reason` fields.

The alluvial layout and its legacy `value` field use fractions of the full
included population. `normalize=True` selects percentage labels by default;
`normalize=False` selects count labels. Neither option changes the source
counts or the denominator after filtering. `label_format` provides an explicit
choice among labels, counts, percentages, and combined labels.

Sankey geometry is supplied by Plotly and follows the visible links. Its hover
information and metadata distinguish complete category counts from visible
and omitted counts. Use the Matplotlib alluvial or chord view when reserved
space for omitted links is needed in a static figure.

### Ordering, colors, and missing annotations

The default input order respects declared categorical order and otherwise
uses observed label order. Use `order="count"` to order by frequency or
`order="barycentric"` for an ordering based on connected categories. Ordering
changes the arrangement, not the association counts.

`category_order` can specify a complete observed label order for each stage:

```python
category_order = {
    "dataset_annotation": ["T cells", "B cells", "Monocytes"],
    "mapped_annotation": ["T cells", "B cells", "Monocytes"],
}
```

These labels are illustrative; the order must cover the actual observed
categories. A shared label-to-color `palette` keeps equivalent categories
consistent across plots. A nested stage-to-palette mapping can give the same
text different colors when it represents different concepts in two stages.

True missing values have their own explicit category. Literal strings such
as `"NA"`, `"nan"`, or `"(missing)"` remain ordinary observed labels.
`drop_na=True` excludes rows with genuinely missing values in any requested
column, before computing the common denominator. The full cell-source table
still records which rows were excluded.

## Density panels with honest color scales

```python
density = sg.pl.embedding_density(
    adata,
    "cell_type",
    basis="umap",
    groups=["T cells", "B cells", "Monocytes"],
    gridsize=80,
    contour=True,
    smooth_k=3,
    log1p=True,
    normalize="global",
    return_result=True,
    show=False,
)
```

All groups use common bin edges. The renderer starts with the raw 2D cell
histogram. In contour mode it optionally smooths once and applies `log1p`
once. Contour levels are computed from the values actually drawn. Heatmaps
retain the raw-count behavior of the existing interface.

| Normalization | Shared colorbar means | Appropriate comparison |
|---|---|---|
| `"global"` | The same absolute count or transformed-count value in every panel | Display-bin occupancy on a common scale |
| `"per_group"` | Relative intensity from 0 to 1, with each group's peak set to 1 | Spatial shape within each group |

Per-group peak normalization cannot support a comparison of group abundance.
A rare population can have the same peak color as a large population. Global
counts also mix group size and spatial concentration; they are not a
biological-sample test of enrichment or depletion.

UMAP coordinates are a display representation. Their local area and distance
do not establish density in gene-expression space. Interpret these panels as
diagnostics of the chosen embedding and selected cells.

### Comparable panels and visible omissions

Set `grid_extent=(xmin, xmax, ymin, ymax)` to fix the computational region and
use the same `gridsize` across separate calls. For a common absolute heatmap
color scale across calls, supply a `Normalize` with explicit limits or the same
`vmin`/`vmax`. For comparable filled contours, request the groups together in
one call: this shares both the color normalization and the contour bands.
Automatic contour boundaries depend on the selected groups, so a shared
normalizer alone does not make separately generated contour bands identical.
`xlim` and `ylim` only crop the view; they do not recompute counts or color limits.

Cells with nonfinite display coordinates do not enter the histogram. Cells
outside an explicit grid extent are recorded separately. Declared empty
categories retain annotated blank panels so their absence is visible. No
observations are created by smoothing or by increasing the grid resolution.

Use `axes=` to place several density panels into an existing figure, or `ax=`
for one group. `cax=` supplies a shared colorbar axis; `show_colorbar=False`
lets a parent figure manage the colorbar. Supplied figure sizes and fixed
axis limits are preserved. Equal aspect is the default.

## Export what was actually displayed

Alluvial and chord plots support `return_result=True` with named `alluvial`
and `chord` axes. Their source tables include:

- `cells`: all original cell identities, stage labels, and inclusion flags.
- `nodes`: stage-specific categories, raw counts, fractions, and visible or
  omitted incoming/outgoing counts.
- `links`: every positive adjacent-stage association, including filtered ones.
- `node_layout`, `link_layout`, and `labels`: the geometry and labels used by
  the static figure.

Density results name their panels `density:<group>` and include an optional
`colorbar` axis. Their `cells`, `grid`, `groups`, and `contour_levels` tables
preserve raw bin counts, smoothing and transformation values, plotted values,
normalization, and exclusions.

```python
chord.save("results/state_flow/chord", formats=("png", "svg", "pdf"))
density.save("results/state_flow/density", formats=("png", "svg", "pdf"))
```

Existing return forms remain available. Alluvial returns `(fig, ax)` for an
owned figure, `ax` when supplied, and `(fig, ax, stage_boxes, alloc_df)` with
`return_data=True`. Chord follows the same four-part data return convention
with node and link layouts. Density keeps `(fig, axes)` and adds a source-data
dictionary with `return_data=True`. The Plotly Sankey retains its
existing figure and three-part data return.

The [real-data gallery](REAL_DATA_PLOTTING.md) records public input hashes and
preserves the resulting figures and source tables. See the
[citation guide](CITING.md) for software and checkout attribution.
