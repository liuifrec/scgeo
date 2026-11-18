# scgeo.pl

ScGeo 0.1.0 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
**Figures.** Visualize stored results and assemble evidence panels, reports and figure-source exports.

- [`PlotResult`](#scgeoplplotresult)
- [`alignment_panel`](#scgeoplalignment_panel)
- [`ambiguity_panel`](#scgeoplambiguity_panel)
- [`category_palette`](#scgeoplcategory_palette)
- [`composition_drift`](#scgeoplcomposition_drift)
- [`consensus_state_map`](#scgeoplconsensus_state_map)
- [`consensus_structure`](#scgeoplconsensus_structure)
- [`consensus_subspace_panel`](#scgeoplconsensus_subspace_panel)
- [`delta_rank`](#scgeopldelta_rank)
- [`density_overlap`](#scgeopldensity_overlap)
- [`density_overlap_grid`](#scgeopldensity_overlap_grid)
- [`distribution_report`](#scgeopldistribution_report)
- [`distribution_test`](#scgeopldistribution_test)
- [`distribution_test_volcano`](#scgeopldistribution_test_volcano)
- [`embedding_density`](#scgeoplembedding_density)
- [`embedding_facets`](#scgeoplembedding_facets)
- [`evidence_comparison`](#scgeoplevidence_comparison)
- [`export_figure`](#scgeoplexport_figure)
- [`gallery_overview`](#scgeoplgallery_overview)
- [`geometry_dynamics_atlas`](#scgeoplgeometry_dynamics_atlas)
- [`geometry_velocity_concordance`](#scgeoplgeometry_velocity_concordance)
- [`highlight_topk_cells`](#scgeoplhighlight_topk_cells)
- [`legend_from_data`](#scgeopllegend_from_data)
- [`local_distortion_map`](#scgeopllocal_distortion_map)
- [`mapping_confidence_umap`](#scgeoplmapping_confidence_umap)
- [`mapping_qc_panel`](#scgeoplmapping_qc_panel)
- [`ood_cells`](#scgeoplood_cells)
- [`ood_landscape`](#scgeoplood_landscape)
- [`paga_composition_bar`](#scgeoplpaga_composition_bar)
- [`paga_composition_panel`](#scgeoplpaga_composition_panel)
- [`paga_composition_volcano`](#scgeoplpaga_composition_volcano)
- [`paga_scgeo`](#scgeoplpaga_scgeo)
- [`paga_shift_map`](#scgeoplpaga_shift_map)
- [`perturbation_report`](#scgeoplperturbation_report)
- [`recovery_compass`](#scgeoplrecovery_compass)
- [`replicate_effect`](#scgeoplreplicate_effect)
- [`representation_stability_heatmap`](#scgeoplrepresentation_stability_heatmap)
- [`robustness_matrix`](#scgeoplrobustness_matrix)
- [`score_embedding`](#scgeoplscore_embedding)
- [`score_umap`](#scgeoplscore_umap)
- [`state_evidence_panel`](#scgeoplstate_evidence_panel)
- [`state_flow_alluvial`](#scgeoplstate_flow_alluvial)
- [`state_flow_chord`](#scgeoplstate_flow_chord)
- [`state_flow_sankey`](#scgeoplstate_flow_sankey)
- [`style_context`](#scgeoplstyle_context)
- [`velocity_condition_compare`](#scgeoplvelocity_condition_compare)
- [`velocity_field`](#scgeoplvelocity_field)
- [`velocity_shift_alignment`](#scgeoplvelocity_shift_alignment)

### `scgeo.pl.PlotResult`

A figure, named axes, plotted data, and interpretation metadata.

```text
scgeo.pl.PlotResult(
    fig: 'Any',
    axes: 'dict',
    data: 'Any' = None,
    metadata: 'dict' = <factory>,
    warnings: 'list' = <factory>,
    artists: 'dict' = <factory>,
) -> None
```

[View implementation](../../scgeo/pl/_result.py)

- Opt-in return type for upgraded plotting functions. Existing default returns remain supported. Holds fig, named axes, artists, source data, metadata, and warnings; save() delegates to export_figure.
- SubFigure results reference and export the root parent canvas; source tables cover the named result axes, not independently added sibling panels.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
A figure, named axes, plotted data, and interpretation metadata.

Obtain this opt-in result with ``return_result=True`` on supported plots.
Existing default return values remain available. ``data`` holds the values
used by the plot; metadata records its metric, ordering, and display choices.
Use :meth:`save` to export figures, source tables, and a JSON manifest.
The result is editable: changing an artist does not update its source data.
```

</details>

### `scgeo.pl.alignment_panel`

1×2 panel:   (A) embedding colored by alignment score   (B) histogram of alignment scores Optional: highlight top-k aligned cells.

```text
scgeo.pl.alignment_panel(
    adata,
    score_key: 'str',
    *,
    basis: 'str' = 'umap',
    topk: 'Optional[int]' = 200,
    cmap: 'str' = 'viridis',
    figsize=(11, 5),
    title: 'Optional[str]' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_alignment_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
1×2 panel:
  (A) embedding colored by alignment score
  (B) histogram of alignment scores
Optional: highlight top-k aligned cells.
```

</details>

### `scgeo.pl.ambiguity_panel`

1×2 panel:   (A) embedding colored by ambiguity/disagreement score   (B) histogram of ambiguity scores Optional: highlight top-k ambiguous cells.

```text
scgeo.pl.ambiguity_panel(
    adata,
    score_key: 'str',
    *,
    basis: 'str' = 'umap',
    topk: 'Optional[int]' = 200,
    cmap: 'str' = 'inferno',
    figsize=(11, 5),
    title: 'Optional[str]' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_ambiguity_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
1×2 panel:
  (A) embedding colored by ambiguity/disagreement score
  (B) histogram of ambiguity scores
Optional: highlight top-k ambiguous cells.
```

</details>

### `scgeo.pl.category_palette`

Return a stable label-to-color mapping, independent of subset ordering.

```text
scgeo.pl.category_palette(labels, palette=None)
```

[View implementation](../../scgeo/pl/_style.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Return a stable label-to-color mapping, independent of subset ordering.

Default colors use a deterministic label hash. For publication, pass a
curated mapping; it must cover all requested labels. Keys in the returned
mapping are strings. Colors alone cannot identify arbitrarily many groups,
so retain labels and consider separate panels for large category sets.
```

</details>

### `scgeo.pl.composition_drift`

Plot descriptive cell-composition changes between two conditions.

```text
scgeo.pl.composition_drift(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    agg: 'str' = 'mean',
    bg_size: 'float' = 6.0,
    bg_alpha: 'float' = 0.1,
    centroid_size: 'float' = 320.0,
    centroid_scale_by_n: 'bool' = True,
    centroid_edgecolor: 'str' = 'white',
    centroid_lw: 'float' = 1.0,
    drift_cmap: 'str' = 'coolwarm',
    drift_vmax: 'Optional[float]' = None,
    bar_alpha: 'float' = 0.9,
    top_n: 'Optional[int]' = None,
    sort_by: 'str' = 'abs_delta_frac',
    palette: 'Optional[dict[str, Any]]' = None,
    title: 'Optional[str]' = None,
    figsize: 'tuple[float, float]' = (14.0, 5.2),
    panels: 'Sequence[str]' = ('embedding', 'change', 'proportions'),
    axes: 'Optional[Mapping[str, Any]]' = None,
    subfigure=None,
    state_order: 'Optional[Sequence[str]]' = None,
    condition_palette: 'Optional[Mapping[str, Any]]' = None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_composition_drift.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot descriptive cell-composition changes between two conditions.

The default three panels and legacy ``(fig, axes[, data])`` returns are
preserved. ``panels`` selects any ordered subset of ``embedding``,
``change``, and ``proportions``. Supply a matching named ``axes`` mapping
or a Matplotlib ``subfigure`` to compose a larger figure. No global layout
adjustment is made when the caller supplies the drawing surface.

Fractions use all observed cells in each condition, not biological-sample
replicates, and are descriptive. ``state_order`` controls the top-to-bottom
order of the bar panels. ``condition_palette`` maps condition labels to
colors. ``return_result=True`` returns a :class:`PlotResult`.
```

</details>

### `scgeo.pl.consensus_state_map`

Display stored consensus labels; never recompute them in the embedding.

```text
scgeo.pl.consensus_state_map(
    adata,
    *,
    node_key,
    basis: 'str' = 'umap',
    store_key: 'str' = 'representation_stability',
    label_states: 'bool' = True,
    palette=None,
    max_points=None,
    random_state=0,
    required_indices=None,
    rasterized='auto',
    ax=None,
    figsize=(5.8, 4.8),
    title=None,
    return_data=False,
    return_result=False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Display stored consensus labels; never recompute them in the embedding.

``max_points`` optionally samples display points reproducibly while retaining
each state and every ``required_indices`` position. State-label positions
use the full data. The returned source table records which cells are shown.
``rasterized='auto'`` rasterizes more than 5,000 displayed points.
```

</details>

### `scgeo.pl.consensus_structure`

Consensus structure (UMAP only) + legend returned separately.

```text
scgeo.pl.consensus_structure(
    adata,
    *,
    score_key: 'str' = 'cs_score',
    basis: 'str' = 'umap',
    groupby: 'str' = 'louvain',
    topk: 'int' = 300,
    bg_size: 'float' = 6.0,
    bg_alpha: 'float' = 0.25,
    hi_size: 'float' = 30.0,
    hi_alpha: 'float' = 0.95,
    cmap: 'str' = 'viridis',
    outline_lw: 'float' = 1.6,
    outline_alpha: 'float' = 0.95,
    use_scanpy_colors: 'bool' = True,
    min_frac_in_legend: 'float' = 0.0,
    title: 'Optional[str]' = None,
    figsize=(6.5, 5.5),
    ax=None,
    show: 'bool' = True,
    print_legend: 'bool' = True,
) -> 'tuple[Any, Any, ConsensusStructureStats, List[Dict[str, Any]]]'
```

[View implementation](../../scgeo/pl/_consensus_structure.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Consensus structure (UMAP only) + legend returned separately.

Figure:
  - Background: all cells colored by `groupby`
  - Top-k: larger points colored by `score_key`
  - Top-k edges: outlined by `groupby` palette (Scanpy palette if available)

Outputs:
  - Returns (fig, ax, stats, legend_data)
  - legend_data is a list of dicts:
      {"group": str, "fraction": float, "count": int, "color": <mpl color>}

Note:
  - This function NEVER draws a legend inside the plot.
  - If you want a legend figure, build it from legend_data.
```

</details>

### `scgeo.pl.consensus_subspace_panel`

Panel for consensus subspace:   (A) embedding colored by cs_score   (B) histogram of cs_score + optional text for singular values

```text
scgeo.pl.consensus_subspace_panel(
    adata,
    *,
    score_key: 'str' = 'cs_score',
    basis: 'str' = 'umap',
    store_key: 'str' = 'consensus_subspace',
    topk: 'Optional[int]' = 200,
    figsize=(11, 5),
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_consensus_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Panel for consensus subspace:
  (A) embedding colored by cs_score
  (B) histogram of cs_score + optional text for singular values
```

</details>

### `scgeo.pl.delta_rank`

Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.

```text
scgeo.pl.delta_rank(
    adata,
    store_key: 'str' = 'scgeo',
    kind: 'str' = 'shift',
    level: 'str' = 'by',
    *,
    top_n: 'int | None' = None,
    rotate_xticks: 'int' = 60,
)
```

[View implementation](../../scgeo/pl/_delta_rank.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.

Notes
-----
Uses sg.get.table(...). Expected to contain:
  - delta_norm column
  - some label column identifying the group (often 'label', or sometimes `level`)
```

</details>

### `scgeo.pl.density_overlap`

Plot density overlap summaries from ``sg.tl.density_overlap``.

```text
scgeo.pl.density_overlap(
    adata,
    *,
    store_key: 'str' = 'density_overlap',
    level: 'str' = 'by',
    metrics: 'Tuple[str, ...]' = ('bc', 'hellinger'),
    top_k: 'Optional[int]' = 10,
    highlight: 'str' = 'worst',
    sort_by: 'Optional[str]' = None,
    style: "Literal['bar', 'lollipop']" = 'lollipop',
    annotate: 'bool' = True,
    grid: 'bool' = True,
    figsize=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_density_overlap.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot density overlap summaries from ``sg.tl.density_overlap``.

Global:
  - Shows one summary value per metric.

By-group:
  - Shows per-group values for each metric.
  - ``highlight`` + ``top_k`` helps focus on worst/best groups.

Notes:
  - For Bhattacharyya coefficient (bc): **lower = more different**.
  - For Hellinger distance: **higher = more different**.
```

</details>

### `scgeo.pl.density_overlap_grid`

Visualize density overlap results from `tl.density_overlap`.

```text
scgeo.pl.density_overlap_grid(
    adata,
    *,
    key: 'str' = 'density_overlap',
    basis: 'str' = 'umap',
    panel: 'str' = 'overlap',
    ax=None,
    figsize=(5.5, 5.0),
    cmap: 'str' = 'magma',
    alpha_points: 'float' = 0.05,
    show_points: 'bool' = True,
    title: 'Optional[str]' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_overlap.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Visualize density overlap results from `tl.density_overlap`.

Supports:
- grid-based fields stored in `adata.uns["scgeo"][key]`
- or per-cell scores in `adata.obs[key]` (fallback)
```

</details>

### `scgeo.pl.distribution_report`

2×2 report panel for cross-condition embedding comparison.

```text
scgeo.pl.distribution_report(
    adata,
    *,
    basis: 'str' = 'umap',
    condition_key: 'str' = 'condition',
    group0: 'Optional[str]' = None,
    group1: 'Optional[str]' = None,
    density_key: 'str' = 'density_overlap',
    test_key: 'str' = 'distribution_test',
    score_key: 'Optional[str]' = None,
    top_k: 'int' = 8,
    figsize=(11, 8),
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
2×2 report panel for cross-condition embedding comparison.

Panels:
  (A) embedding colored by condition
  (B) density overlap summary (global + worst groups)
  (C) distribution test summary (global + strongest groups)
  (D) optional score histogram (or empty text if score_key None)

Requires tl outputs stored in adata.uns['scgeo'] under density_key/test_key.
```

</details>

### `scgeo.pl.distribution_test`

Plot distribution test summaries from `sg.tl.distribution_test`.

```text
scgeo.pl.distribution_test(
    adata,
    *,
    store_key: 'str' = 'distribution_test',
    level: 'str' = 'by',
    value: 'str' = 'p_perm',
    stat_key: 'str' = 'stat',
    top_k: 'Optional[int]' = 10,
    highlight: 'str' = 'strongest',
    figsize=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_distribution_test.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot distribution test summaries from `sg.tl.distribution_test`.

Global: prints and plots stat + p_perm.
By: bar chart by group; highlight top-k strongest (smallest p) groups by default.

Assumes dict schema:
  global: {"stat":..., "p_perm":..., "n0":..., "n1":..., "n_perm":...}
  by: {group: {"stat":..., "p_perm":..., ...}, ...}
```

</details>

### `scgeo.pl.distribution_test_volcano`

Volcano-like plot for distribution/comparison results.

```text
scgeo.pl.distribution_test_volcano(
    adata,
    *,
    key: 'str' = 'distribution_test',
    effect_col: 'str' = 'effect',
    p_col: 'str' = 'p_adj',
    label_col: 'str' = 'group',
    top_n_labels: 'int' = 10,
    ax=None,
    figsize=(6.0, 5.0),
    title: 'str | None' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_volcano.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Volcano-like plot for distribution/comparison results.

x = effect (distance / signed effect)
y = -log10(p_adj)

Requires a table with at least: effect_col, p_col.
Optionally label the top hits by p-value.
```

</details>

### `scgeo.pl.embedding_density`

Compare group occupancy on a common embedding grid with explicit normalization, source counts, and composable panels.

```text
scgeo.pl.embedding_density(
    adata,
    groupby: 'str',
    *,
    basis: 'str' = 'umap',
    groups: 'Iterable[str] | None' = None,
    gridsize: 'int' = 160,
    normalize: 'str' = 'per_group',
    cmap: 'str' = 'magma',
    contour: 'bool' = False,
    contour_levels: 'int' = 12,
    imshow_alpha: 'float' = 0.85,
    transparent_background: 'bool' = True,
    mask_zeros: 'bool' = True,
    background: 'str | None' = None,
    figsize=None,
    show: 'bool' = True,
    smooth_k: 'int' = 5,
    log1p: 'bool' = True,
    contour_lines: 'bool' = True,
    contour_linewidth: 'float' = 0.7,
    contour_alpha: 'float' = 0.95,
    contour_level_mode: 'str' = 'quantile',
    ax=None,
    axes=None,
    cax=None,
    show_colorbar: 'bool' = True,
    grid_extent=None,
    xlim=None,
    ylim=None,
    aspect: 'str' = 'equal',
    norm=None,
    vmin=None,
    vmax=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../../scgeo/pl/_density.py)

- Common bins preserve raw count totals. Contour mode optionally smooths once and applies log1p once; contour levels use the actual rendered values.
- normalize=per_group uses a shared relative 0–1 scale with each group peak set to one; normalize=global uses one absolute count or transformed-count scale across panels.
- Per-group peak normalization cannot establish relative abundance. Embedding density describes display coordinates and does not constitute sample-level enrichment inference.
- grid_extent controls computational binning. xlim/ylim only crop the view. Full cell sources retain nonfinite, missing-group, and outside-grid exclusions.
- For matched filled-contour bands, render comparison groups in one call. Automatic level boundaries depend on selected groups; a shared norm alone does not make separate calls use identical bands. Shared explicit color limits provide direct heatmap comparability.
- Supplied axes/cax and shared Normalize objects support composition. Default (fig, axes) is retained; return_data=True adds a source dictionary, and return_result=True supplies density:<group> and colorbar axes.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot comparable per-group histograms on the first two embedding axes.

All groups use the same grid. ``normalize='per_group'`` divides each
group's transformed grid by its own peak, giving a shared relative scale
from zero to one. This compares spatial shape, not group abundance.
``normalize='global'`` retains absolute bin values and uses one color scale
across panels. Counts describe the supplied cells and depend on sampling,
bin width, and embedding geometry; they are not a probability density or
an estimate of population abundance.

Heatmaps show raw cell counts before normalization. Only ``contour=True``
applies one optional ``smooth_k`` box blur followed by one optional
``log1p`` transform. ``smooth_k=0`` or ``1`` disables smoothing; other values
must be positive odd integers. Replicated-edge smoothing is descriptive
and does not guarantee count conservation; exported ``raw_count`` does.
Contour levels are shared across panels and computed from the actual
plotted values, using quantile or linear spacing over their full range.
When zeros are masked, the lowest level is half the smallest positive
plotted value; this preserves sparse peaks and lower-count groups on a
common scale. Repeated levels are removed. Zeros are omitted when ``mask_zeros=True``;
contour boundaries interpolate toward zero-valued neighbors so sparse
peaks remain visible.

Parameters
----------
groups
    Ordered known group labels. By default, categorical declaration order
    is retained, including unused levels; other labels use observed order.
    Known groups with no finite cells remain explicit empty panels.
    Missing labels and nonfinite coordinate pairs are excluded jointly and
    retained in the source table. Unknown requested groups raise an error.
gridsize
    Number of bins per coordinate, an integer of at least two.
grid_extent
    Optional ``(xmin, xmax, ymin, ymax)`` computational bounds. The default
    uses all finite labeled cells, including unselected groups, so selecting
    panels does not silently alter their bins. Constant coordinates receive
    a small explicit padding. Cells outside the grid remain in source data
    but do not contribute to histograms; the final bin includes its upper edge.
xlim, ylim
    View limits only: these do not change histograms or color normalization.
    Supplied axes retain already fixed limits unless overridden explicitly.
norm, vmin, vmax
    Optional shared Matplotlib color normalization. Supply ``norm`` or
    numeric limits. Defaults are zero to one for relative panels and zero
    to the largest transformed count for global panels. Normalization maps
    colors only; contour geometry uses the transformed count values.
    A shared norm directly aligns heatmap color scales. For matching
    automatic filled-contour bands, render the comparison groups in one
    call: their shared levels depend on all selected plotted values.
ax, axes, cax
    Compose into an existing figure. ``ax`` is for one group; ``axes`` is a
    flat sequence or mapping keyed by group (or ``'density:<group>'``).
    Use ``cax`` for a reserved shared colorbar, or disable it with
    ``show_colorbar=False``. Supplied figure size and sibling panels are
    preserved. ``aspect='equal'`` keeps embedding units equally scaled.
return_data, return_result
    The default remains ``(fig, flat_axes_array)``. ``return_data=True``
    returns ``(fig, flat_axes_array, data)``. ``return_result=True`` returns
    a :class:`PlotResult` with named ``density:<group>`` axes, optional
    ``colorbar``, and four source tables: full ``cells`` with inclusion
    flags, every ``grid`` bin with raw and transformed values, ``groups``
    with counts and normalization factors, and exact ``contour_levels``.
    The grid's ``displayed`` flag records unmasked values at the contour
    threshold, independently of view cropping; contours interpolate those
    grid values. A SubFigure result exports its full parent canvas.
```

</details>

### `scgeo.pl.embedding_facets`

Facet existing AnnData embeddings with optional independently supported local projected velocity.

```text
scgeo.pl.embedding_facets(
    adata,
    *,
    basis='umap',
    facet=None,
    facet_row=None,
    facet_col=None,
    facet_order=None,
    row_order=None,
    col_order=None,
    col_wrap=4,
    color=None,
    palette=None,
    cmap='viridis',
    velocity=None,
    velocity_basis=None,
    velocity_options=None,
    shared_limits=True,
    shared_velocity_scale=True,
    context='none',
    max_points=None,
    random_state=0,
    point_size=7.0,
    point_alpha=0.82,
    highlight=None,
    highlight_key=None,
    legend=True,
    show_colorbar=True,
    style='paper',
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_embedding_facets.py)

- Uses the supplied embedding and matching projected velocity; no kinetics, transition probabilities or embedding are inferred.
- Default shared axes, Gaussian radius and grid-arrow multiplier allow honest across-facet comparisons. Support masks are explicit.
- One-way and two-way faceting accept any observation labels and preserve missing-value identity. Source tables retain all cells and grid reasons.
- A valid Scanpy category palette can be reused. Separate numeric colors and velocity diagnostics have shared scales.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Facet an existing embedding, optionally overlaying projected velocity.

``facet`` is an alias for ``facet_col``. ``facet_row`` and ``facet_col``
can be combined for a two-way grid. Each observation belongs to precisely
one panel, including a separate level for missing facet labels. Unobserved
levels supplied in ``row_order``/``col_order`` produce explicit empty panels.

All facets use one global categorical palette or numeric color scale.
``shared_limits=True`` fixes both axes identically; with velocity, the
support grid and Gaussian radius are also shared. The optional
``shared_velocity_scale`` fixes the *grid-arrow multiplier* using supported
local field speeds across all facets, not a separate auto scale per panel.
The same velocity numeric colormap normalization is always shared.

Velocity requires the *matching* ``velocity_<basis>`` array in ``adata.obsm``.
Smoothing is calculated independently for each facet using the existing
ScGeo local-field estimator. Unsupported positions remain blank. Streamlines
are descriptive interpolations, not trajectories or elapsed time. No UMAP,
kinetics, or quantitative ScGeo evidence is recomputed here.

``max_points`` limits *drawing per facet* only; local velocity uses all
eligible source cells. ``context='all'`` optionally draws the full embedding
in muted grey behind each facet. Metadata and PlotResult exports record all
source cells, facet membership, supported velocity grid, and vector/stream
geometry. A supplied optional Scanpy categorical palette is honored.

With ``return_result=True`` returns a :class:`PlotResult`. By default
returns ``(fig, axes_dict)``; ``return_data=True`` adds source tables.
```

</details>

### `scgeo.pl.evidence_comparison`

Draw aligned state-by-contrast effect panels from a tidy results table.

```text
scgeo.pl.evidence_comparison(
    data,
    *,
    metric=None,
    state_order=None,
    contrast_order=None,
    palette=None,
    sharex='auto',
    axes=None,
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_evidence_comparison.py)

- Plots tidy stored effects without recomputing inference. Intervals require explicit interval_type; sample-bootstrap uncertainty and across-representation sensitivity remain distinct.
- Shared axes require declared comparable representation/units or a common scale_id. Missing combinations remain visible and duplicate state/contrast records are rejected.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw aligned state-by-contrast effect panels from a tidy results table.

Required columns are ``state``, ``contrast``, ``metric``, and ``estimate``.
Supply ``interval_low``, ``interval_high``, and ``interval_type`` together
for intervals; their endpoints are drawn independently of the estimate.
Labels such as ``sample_bootstrap_ci95`` and ``representation_iqr`` remain
distinct in the legend. No intervals, tests, or composite scores are fitted.

Optional ``representation`` and ``unit`` columns describe the effect scale.
``sharex='auto'`` shares limits only when every finite estimate has the same
declared representation and unit. Alternatively, a nonempty common
``scale_id`` explicitly declares comparability across representations.
``sharex=True`` rejects incompatible or unspecified scales. Missing state
and contrast combinations stay visible. Duplicate state/contrast rows for
the selected metric are rejected rather than silently aggregated.
Optional ``evidence_status`` (preferred) or ``status`` labels are preserved
and displayed with distinct marker shapes; a finite estimate never promotes
supplied insufficient or unavailable evidence to ``'assessed'``.

``axes`` is an optional mapping from contrast labels to distinct Axes in
one Figure. Orders must retain observed labels; filter the input explicitly
to select a subset. Extra ordered labels create visible missing rows/panels.
Returns a Figure by default, ``(Figure, plotted_table)`` with
``return_data=True``, or a :class:`PlotResult` with ``return_result=True``.
```

</details>

### `scgeo.pl.export_figure`

Export a Matplotlib figure or PlotResult with source-data provenance.

```text
scgeo.pl.export_figure(
    figure_or_result,
    filename,
    *,
    formats=('png', 'svg'),
    dpi=300,
    width_mm=None,
    height_mm=None,
    panel_labels=None,
    include_data=True,
    include_metadata=True,
    transparent=False,
    metadata=None,
    bbox_inches=None,
)
```

[View implementation](../../scgeo/pl/_result.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Export a Matplotlib figure or PlotResult with source-data provenance.

Parameters
----------
figure_or_result
    A Matplotlib Figure or ``PlotResult``. Source CSVs require a PlotResult.
    A result constructed from a SubFigure exports the full parent canvas.
filename
    Output stem, optionally ending in .png, .svg or .pdf. ``formats`` selects
    the outputs; a supplied suffix is removed before constructing names.
formats
    One or more of ``'png'``, ``'svg'``, ``'pdf'``.
dpi
    Resolution of PNG and rasterized artists in vector figures.
width_mm, height_mm
    Optional canvas dimensions in millimetres. If only one is supplied,
    preserve the original aspect ratio. Original figure size is restored.
    ``bbox_inches=None`` preserves that canvas size; ``'tight'`` crops it.
panel_labels
    Mapping from PlotResult axis names to labels, or a label sequence for
    the result's axes. Labels are temporary and removed after exporting.
include_data, include_metadata
    Export source DataFrames/Series/arrays as CSVs and a strict JSON manifest.
    Named indexes are retained. The manifest includes table schemas, hashes,
    figure hashes, plotting metadata, warnings, and software versions.
metadata
    Additional JSON-compatible provenance, merged over result metadata.

Returns
-------
dict
    ``figures`` and ``tables`` mappings and the ``metadata`` JSON path.

Notes
-----
The exporter records input metadata; it does not reconstruct analysis
settings that were never supplied or infer statistical significance.
```

</details>

### `scgeo.pl.gallery_overview`

Render four coordinated panels, including actual composition-change bars.

```text
scgeo.pl.gallery_overview(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    paga_key: 'str' = 'paga',
    ood_key: 'str' = 'scgeo_ood',
    velocity_basis: 'Optional[str]' = None,
    figsize: 'tuple[float, float]' = (16.0, 12.0),
    title: 'Optional[str]' = None,
    axes: 'Optional[Mapping[str, Any]]' = None,
    state_order=None,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_gallery.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Render four coordinated panels, including actual composition-change bars.

``axes`` may supply the named panels ``shift``, ``ood``, ``alignment``, and
``composition``. The alignment panel is explicitly a display-space overlay;
use ``velocity_shift_alignment(mode='stored')`` for stored quantitative
alignment. The legacy return is ``(fig, (ax1, ax2, ax3, ax4))``.
```

</details>

### `scgeo.pl.geometry_dynamics_atlas`

Build coordinated states, projected-velocity, PAGA, and evidence panels.

```text
scgeo.pl.geometry_dynamics_atlas(
    adata,
    *,
    basis='umap',
    color=None,
    node_key=None,
    condition_key=None,
    group0=None,
    group1=None,
    sample_key=None,
    panels='auto',
    velocity_mode='stream',
    velocity_options=None,
    paga_options=None,
    report=None,
    effect_metric='auto',
    report_state_key=None,
    max_states=12,
    highlight_states=None,
    palette=None,
    max_points=6000,
    random_state=0,
    label_states=True,
    style='paper',
    figsize=None,
    title=None,
    return_result=True,
    show=True,
)
```

[View implementation](../../scgeo/pl/_geometry_dynamics_atlas.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build coordinated states, projected-velocity, PAGA, and evidence panels.

A standard AnnData embedding is enough for a one-panel atlas. With
``panels='auto'``, optional panels are included *only* when inputs exist:
a matching ``velocity_<basis>`` projection, a PAGA graph plus a specified
two-condition contrast, or an explicitly supplied stored report. Missing
optional panels are recorded; an explicitly requested unavailable panel
raises instead. Allowed panels are ``'states'``, ``'velocity'``, ``'paga'``
(alias ``'geometry'``), and ``'evidence'``.

The states panel reads ``adata.obs[color]`` (categorical or numeric).
``node_key`` identifies states used for PAGA and highlighting; it defaults
to ``color`` when that is categorical. ``condition_key`` can infer the
contrast only when precisely two nonmissing groups are observed. When
samples are available, ``sample_key`` adds descriptive counts to metadata
without inventing statistical tests. ``max_points`` affects drawing only.

The velocity and PAGA panels delegate to existing ScGeo plotting functions
with common embedding limits and compatible node palettes. Projected
velocity describes the *display* embedding. PAGA edges are undirected, and
condition arrows are display-coordinate mean/median contrasts. The evidence
panel displays stored ``delta_norm`` or ``normalized_delta_norm`` from
``report`` without inferring effects or confidence intervals. Raw CIs are
never reused on normalized effects. No results are written to AnnData.

``velocity_options`` and ``paga_options`` pass through supported renderer
settings, but cannot replace shared basis, axes, condition scope, palette,
and output controls. Use ``return_result=True`` (default) to export PNG,
SVG, PDF and each panel's source tables via ``PlotResult.save``.
```

</details>

### `scgeo.pl.geometry_velocity_concordance`

Show descriptive concordance between stored displacement and alignment.

```text
scgeo.pl.geometry_velocity_concordance(
    shift_df,
    alignment_df,
    *,
    state_key='state',
    alignment_state_key=None,
    effect_metric='normalized_delta_norm',
    alignment_metric='alignment_cosine',
    status_key='coverage_status',
    eligible_statuses=('eligible_exploratory', 'eligible'),
    usable_key='usable',
    label_key=None,
    analysis_rep=None,
    positive_threshold=0.5,
    opposed_threshold=-0.25,
    sample_count_keys=('n_mice_control', 'n_mice_irradiated'),
    show_ci=True,
    highlight_states=None,
    max_labels=16,
    axes=None,
    figsize=(12.0, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_geometry_velocity_concordance.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Show descriptive concordance between stored displacement and alignment.

Provide one row per state in ``shift_df`` and ``alignment_df``; the latter
may use a different state column. Only rows with usable alignment, finite
values and an eligible coverage status are plotted as estimated results.
Ineligible states are counted and exported rather than silently dropped.
``positive_threshold`` and ``opposed_threshold`` are visual descriptions,
never significance thresholds or evidence of trajectories. A shared
``analysis_rep`` is declared by the caller, not inferred from UMAP.

Raw ``delta_norm`` may use stored ``magnitude_ci95_low/high`` intervals.
Normalized magnitudes require their own explicitly named intervals; raw
intervals are never applied to normalized estimates. No confidence
intervals, mouse-level replication or missing velocities are fabricated.

Default return is ``(fig, axes)``; ``return_result=True`` provides a
:class:`scgeo.pl.PlotResult` with included and excluded state tables.
```

</details>

### `scgeo.pl.highlight_topk_cells`

Highlight top-k cells by a score on an embedding.

```text
scgeo.pl.highlight_topk_cells(
    adata,
    score_key: 'str',
    basis: 'str' = 'umap',
    *,
    topk: 'int' = 300,
    ax=None,
    figsize=(5.8, 5.2),
    bg_size: 'float' = 6.0,
    bg_alpha: 'float' = 0.2,
    hi_size: 'float' = 30.0,
    hi_alpha: 'float' = 0.95,
    cmap: 'str' = 'viridis',
    title: 'str | None' = None,
    show: 'bool' = True,
    groupby: 'str | None' = None,
    use_scanpy_colors: 'bool' = True,
    outline_topk: 'bool' = False,
    outline_lw: 'float' = 1.6,
    outline_alpha: 'float' = 0.95,
    add_colorbar: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_highlight.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Highlight top-k cells by a score on an embedding.

Default look:
  - background: all cells in light grey
  - top-k: larger points colored by score_key

If groupby is provided and outline_topk=True:
  - top-k points get edgecolors from group palette (Scanpy palette if available)
  - gives Scanpy-like cluster structure without hull polygons
```

</details>

### `scgeo.pl.legend_from_data`

Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

```text
scgeo.pl.legend_from_data(
    legend_data: 'Iterable[Dict[str, Any]]',
    *,
    max_items: 'int' = 20,
    ncol: 'int' = 1,
    fontsize: 'int' = 8,
    markersize: 'int' = 6,
    title: 'str | None' = None,
    figsize: 'Tuple[float, float] | None' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_legend.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

legend_data items:
  {"group": str, "fraction": float, "count": int, "color": <matplotlib color>}
```

</details>

### `scgeo.pl.local_distortion_map`

Display stored distortion values with a shared scale and optional sampling.

```text
scgeo.pl.local_distortion_map(
    adata,
    *,
    basis: 'str' = 'umap',
    store_key: 'str' = 'local_geometry_stability',
    rep_a=None,
    rep_b=None,
    metric: 'str' = 'local_shape_distortion',
    aggregation: 'str' = 'median',
    node_key=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap='magma',
    na_color='#BDBDBD',
    max_points=None,
    random_state=0,
    required_indices=None,
    rasterized='auto',
    ax=None,
    colorbar=True,
    cax=None,
    figsize=(5.8, 4.8),
    title=None,
    return_data=False,
    return_result=False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Display stored distortion values with a shared scale and optional sampling.

Sampling affects displayed cells only, with states preserved when node_key
is supplied. Aggregation always uses the complete stored results. Missing
values use ``na_color``. ``norm`` cannot be combined with ``vmin/vmax``.
```

</details>

### `scgeo.pl.mapping_confidence_umap`

Display reference-mapping confidence on an embedding.

```text
scgeo.pl.mapping_confidence_umap(
    adata,
    conf_key: 'str' = 'map_confidence',
    *,
    basis: 'str' = 'umap',
    highlight_low_k: 'Optional[int]' = 200,
    title: 'Optional[str]' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_mapping_qc.py)

### `scgeo.pl.mapping_qc_panel`

Display reference-label predictions, confidence and reference-deviation scores together.

```text
scgeo.pl.mapping_qc_panel(
    adata,
    *,
    pred_key: 'str' = 'map_pred',
    conf_key: 'str' = 'map_confidence',
    ood_key: 'str' = 'map_ood_score',
    basis: 'str' = 'umap',
    show: 'bool' = True,
    palette_from: 'str | None' = None,
    condition_key: 'str | None' = None,
    query_value: 'str | None' = None,
    show_ref_as_grey: 'bool' = True,
    return_legend_data: 'bool' = False,
)
```

[View implementation](../../scgeo/pl/_mapping_qc.py)

### `scgeo.pl.ood_cells`

Display cells flagged by a stored reference-deviation score.

```text
scgeo.pl.ood_cells(
    adata,
    ood_key: 'str' = 'map_ood_score',
    *,
    basis: 'str' = 'umap',
    threshold: 'Optional[float]' = None,
    show_only_flagged: 'bool' = False,
    title: 'Optional[str]' = None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_mapping_qc.py)

### `scgeo.pl.ood_landscape`

Plot a continuous OOD landscape on an embedding, with optional contour and optional right-side group summary.

```text
scgeo.pl.ood_landscape(
    adata,
    *,
    ood_key: 'str' = 'scgeo_ood',
    basis: 'str' = 'umap',
    threshold: 'Optional[float]' = None,
    show_only_flagged: 'bool' = False,
    flagged_outline: 'bool' = True,
    flagged_size: 'float' = 28.0,
    flagged_lw: 'float' = 0.8,
    bg_size: 'float' = 6.0,
    bg_alpha: 'float' = 0.12,
    score_size: 'float' = 10.0,
    score_alpha: 'float' = 0.85,
    cmap: 'str' = 'magma',
    contour: 'bool' = True,
    contour_quantile: 'float' = 0.95,
    contour_levels: 'int' = 1,
    contour_color: 'str' = 'cyan',
    contour_lw: 'float' = 1.6,
    contour_alpha: 'float' = 0.95,
    contour_gridsize: 'int' = 150,
    groupby: 'Optional[str]' = None,
    top_n_groups: 'int' = 10,
    summary_kind: 'str' = 'flagged_frac',
    figsize: 'tuple[float, float]' = (10.5, 5.2),
    title: 'Optional[str]' = None,
    ax=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_ood_landscape.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a continuous OOD landscape on an embedding, with optional contour and
optional right-side group summary.

Parameters
----------
adata
    AnnData with embedding in ``adata.obsm`` and numeric OOD scores in ``adata.obs``.
ood_key
    Observation column containing numeric OOD scores.
basis
    Embedding basis, e.g. ``'umap'`` or ``'X_umap'``.
threshold
    OOD threshold for flagged cells. If None, uses the 95th percentile.
show_only_flagged
    If True, only plot flagged cells in the score layer.
groupby
    Optional obs column for a right-side group summary.
summary_kind
    One of ``{'flagged_frac', 'flagged_n', 'mean_score', 'median_score'}``.
return_data
    If True, return computed summary artifacts.
```

</details>

### `scgeo.pl.paga_composition_bar`

Bar plot of top_k nodes by evidence + effect.

```text
scgeo.pl.paga_composition_bar(
    adata,
    *,
    store_key: 'str' = 'scgeo',
    kind: 'str' = 'paga_composition_stats',
    effect: 'Optional[str]' = None,
    p_col: 'Optional[str]' = None,
    top_k: 'int' = 15,
    sort_by: 'Optional[str]' = None,
    ax=None,
    figsize=(6.2, 4.2),
    title: 'str' = 'ScGeo: Top composition shifts',
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Bar plot of top_k nodes by evidence + effect.
```

</details>

### `scgeo.pl.paga_composition_panel`

1×2 panel: volcano + top-k bars.

```text
scgeo.pl.paga_composition_panel(
    adata,
    *,
    store_key: 'str' = 'scgeo',
    kind: 'str' = 'paga_composition_stats',
    effect: 'Optional[str]' = None,
    p_col: 'Optional[str]' = None,
    top_k: 'int' = 10,
    figsize=(12, 4.2),
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
1×2 panel: volcano + top-k bars.
```

</details>

### `scgeo.pl.paga_composition_volcano`

Volcano plot: x = signed effect (logOR/beta/effect), y = -log10(q or p). Highlights top_k by smallest p/q then largest |effect|.

```text
scgeo.pl.paga_composition_volcano(
    adata,
    *,
    store_key: 'str' = 'scgeo',
    kind: 'str' = 'paga_composition_stats',
    effect: 'Optional[str]' = None,
    p_col: 'Optional[str]' = None,
    top_k: 'int' = 10,
    label: 'bool' = True,
    ax=None,
    figsize=(5.2, 4.2),
    title: 'str' = 'ScGeo: PAGA composition volcano',
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Volcano plot: x = signed effect (logOR/beta/effect), y = -log10(q or p).
Highlights top_k by smallest p/q then largest |effect|.
```

</details>

### `scgeo.pl.paga_scgeo`

Convenience wrapper for the PAGA shift map with composition pies and optional projected velocity.

```text
scgeo.pl.paga_scgeo(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    paga_key: 'str' = 'paga',
    pie_key: 'Optional[str]' = 'timepoint',
    velocity_basis: 'Optional[str]' = None,
    show_velocity: 'bool' = True,
    node_color_mode: 'str' = 'delta',
    highlight_nodes: 'Optional[list[str]]' = None,
    **kwargs,
)
```

[View implementation](../../scgeo/pl/_paga_scgeo.py)

- Additional keyword arguments are forwarded to paga_shift_map, including stored alignment, shared limits, label placement, and return_result.
- Composition pies show the selected conditions. Numeric node colour is retained in the pie rim; the composition sectors and quantitative rim have separate legends.
- velocity_mode and velocity_options are forwarded to the PAGA map for local projected-velocity grids and streamlines.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw the PAGA shift map with composition and projected velocity enabled.

Pies pool the two selected conditions; scalar colors remain visible as
reference rims and comparison fills. Dark arrows show display-coordinate
condition contrasts and teal arrows show mean projected velocity, with
separate display multipliers recorded in the legend. These arrows describe
geometry, not inferred trajectories. ``velocity_basis=None`` uses the same
basis as the display and requires its matching projected velocity array.

``velocity_mode='grid'`` or ``'stream'`` replaces the state velocity arrows
with a supported local projected field. ``velocity_options`` controls its
neighborhood, support thresholds, and display style. Local directional
agreement is descriptive and does not measure statistical confidence.

Other options, including axes, label placement, stored alignment coloring,
and ``return_result``, are forwarded to :func:`paga_shift_map`.
```

</details>

### `scgeo.pl.paga_shift_map`

Display stored state connectivity, condition centres, and optional projected velocity on an embedding, with explicit quantitative evidence and source exports.

```text
scgeo.pl.paga_shift_map(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    paga_key: 'str' = 'paga',
    min_cells: 'int' = 15,
    connectivity_threshold: 'float' = 0.05,
    agg: 'str' = 'mean',
    background_size: 'float' = 6.0,
    background_alpha: 'float' = 0.12,
    node_size: 'float' = 220.0,
    node_scale_by_n: 'bool' = True,
    edge_lw: 'float' = 2.0,
    edge_alpha: 'float' = 0.32,
    arrow_width: 'Optional[float]' = None,
    arrow_alpha: 'float' = 0.95,
    arrow_scale: 'float' = 1.0,
    label: 'bool' = True,
    label_top_n: 'Optional[int]' = None,
    label_fontsize: 'int' = 8,
    palette: 'Optional[dict[str, Any]]' = None,
    pie_key: 'Optional[str]' = None,
    pie_categories: 'Optional[list[str]]' = None,
    pie_palette: 'Optional[dict[str, Any]]' = None,
    pie_size_scale: 'float' = 1.0,
    velocity_basis: 'Optional[str]' = None,
    show_velocity: 'bool' = False,
    velocity_color: 'str' = '#008C95',
    velocity_scale: 'float' = 50.0,
    velocity_alpha: 'float' = 0.95,
    node_color_mode: 'str' = 'palette',
    alignment_df: 'Optional[pd.DataFrame]' = None,
    alignment_key: 'str' = 'alignment_cosine',
    delta_key: 'str' = 'delta_frac',
    constant_node_color: 'str' = 'gold',
    highlight_nodes: 'Optional[list[str]]' = None,
    focus_states: 'Optional[list[str]]' = None,
    edge_focus: 'str' = 'all',
    highlight_edgecolor: 'str' = 'black',
    highlight_lw: 'float' = 2.0,
    ax=None,
    figsize: 'tuple[float, float]' = (9.0, 6.4),
    title: 'Optional[str]' = None,
    return_data: 'bool' = False,
    show: 'bool' = True,
    aspect='equal',
    xlim=None,
    ylim=None,
    show_endpoints: 'bool' = True,
    endpoint_size: 'float' = 46.0,
    arrow_linewidth: 'float' = 1.6,
    arrow_head_size: 'float' = 9.0,
    label_mode: 'str' = 'auto',
    label_offsets=None,
    legend: 'bool' = True,
    legend_loc: 'str' = 'outside',
    graph_node_order=None,
    background: 'str' = 'all',
    max_points=None,
    random_state: 'int' = 0,
    required_indices=None,
    alignment_source: 'str' = 'provided',
    store_key: 'str' = 'representation_stability',
    analysis_rep: 'Optional[str]' = None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    show_colorbar: 'bool' = True,
    cax=None,
    return_result: 'bool' = False,
    velocity_mode: 'str' = 'state',
    velocity_options=None,
)
```

[View implementation](../../scgeo/pl/_paga_shift_map.py)

- These upgrades are unreleased development features; see the [PAGA shift-map guide](https://github.com/liuifrec/scgeo/blob/main/docs/PAGA_SHIFT_MAP.md) and install the matching checkout.
- Node positions and condition arrows are descriptive mean/median summaries in the chosen display. Use alignment_source="stored" and analysis_rep to colour nodes from stored quantitative alignment; display coordinates do not redefine those values.
- PAGA connectivity is undirected. Supply graph_node_order when matrix labels differ from the current category order; graph shape, weights, symmetry, and group identity are validated.
- Projected velocity must match the display basis. Velocity and composition summaries use only group0/group1; background="all" can retain other cells as context.
- Equal aspect, distinct condition endpoints, point-sized glyphs, automatic labels, and graphical legends are the new defaults. Explicit arrow_width retains the legacy data-unit style.
- return_result=True returns PlotResult with named map axis and full cell/edge/vector sources. Legacy owned (fig, ax), supplied ax, and (fig, ax, data) returns remain supported.
- velocity_mode="state" preserves the original aggregate arrows. Opt-in "grid"/"stream" use selected-condition local fields; velocity_options controls finite-radius support and rendering, with full velocity_* source tables.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw undirected PAGA topology and condition contrasts on an embedding.

Positions, mean/median cell centers, and arrows describe the first two
display coordinates. They are not inferred cell trajectories. Reference
anchors are open markers; comparison endpoints are filled and remain at
their actual centers even when ``arrow_scale`` changes the displayed vector.
States present in only one condition or below ``min_cells`` remain visible.
Graph edges require usable reference anchors at both ends.

Velocity and pie summaries use only ``group0`` and ``group1``. Projected
velocity must use the matching ``velocity_<basis>`` key and match the
embedding shape. Selected, labeled cells with nonfinite display coordinates
are rejected before drawing; nonfinite background rows are omitted. Missing
velocity rows are excluded jointly and their support counts are exported.

``velocity_mode='state'`` preserves the reference-anchored state arrows.
``'grid'`` and ``'stream'`` interpolate the selected cells' supplied projected
velocities locally, using finite-radius support and directional-coherence
masks. These are descriptive display fields, not inferred trajectories or
statistical confidence. ``velocity_options`` controls grid_size, grid_extent,
n_neighbors, radius, min_neighbors, min_coherence, scale, arrow_linewidth,
arrow_head_size, stream_density, and stream_linewidth. Local means are
Gaussian weighted regardless of the state-center ``agg`` option.
``velocity_scale`` applies to state arrows; local grid arrows use the
options' scale (default ``'auto'``), and streamlines carry no time scale.
Local field color follows ``velocity_color`` so stored node colors retain
their independent meaning. Use :func:`velocity_field` for numeric field
colors or separate condition panels.

Explicit ``pie_categories`` must include every observed selected category;
missing values use a separate ``(missing)`` category (escaped if that literal
label already exists). Retained slices are never silently renormalized.

``graph_node_order`` labels matrix rows/columns explicitly, including graph
states absent from a subset. Without it, a labeled matrix, stored node_order,
or current categorical order is used. The plot cannot recover historical
category order that was not stored. Connectivity must be finite, nonnegative,
symmetric, and the exact declared size. No edges or centers are synthesized.

Alignment coloring accepts an ``alignment_df`` or reads stored analysis with
``alignment_source='stored'`` and ``analysis_rep``. Quantitative values are
never recalculated from the display embedding. Known source metadata must
match the requested contrast; unknown provenance is labeled explicitly.
``alignment_source`` is inactive for other color modes. Scalar colorbars
describe numeric reference rims and comparison fills, including with pies.

Default arrows and pies use point-sized graphics. An explicit ``arrow_width``
retains legacy data-unit arrow widths. Labels use deterministic point-space
placement, with ``label_offsets={state: (dx, dy)}`` overrides or
``label_mode='on_data'``. Label placement never changes centers or values.
``aspect='equal'`` preserves display-coordinate angles. Supplied axes own
their layout and figure size; ``xlim``/``ylim`` support matched canvases.

Background sampling affects drawing only, preserving state-by-condition
strata and ``required_indices``. Source tables retain all input cells with
display flags, all positive graph edges with omission reasons, scaled vector
endpoints, pie counts, source alignment, and label offsets. Cell fractions
and support counts are descriptive; they do not substitute for sample-level
inference. ``return_result=True`` returns a :class:`PlotResult`; historical
tuple/axis returns and ``return_data`` keys remain available.
```

</details>

### `scgeo.pl.perturbation_report`

Create figures, source tables, and provenance for a perturbation report.

```text
scgeo.pl.perturbation_report(
    adata,
    *,
    node_key,
    basis: 'str' = 'umap',
    report: 'Optional[pd.DataFrame]' = None,
    save_dir=None,
    prefix: 'str' = 'scgeo',
    comparison_label: 'Optional[str]' = None,
    local_k=None,
    pair_aggregation: 'str' = 'median',
    include_worst_case: 'bool' = True,
    formats=('png', 'svg'),
    dpi=300,
    width_mm=None,
    height_mm=None,
    transparent=False,
    bbox_inches=None,
    strict=False,
    required_panels=None,
    panel_kwargs=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)

- Export bundle: state report CSV, representation diagnostics CSV, metadata JSON, warnings text, alt-text text, figure-source CSVs and configurable PNG/SVG/PDF figures.
- Use strict=True or required_panels to reject incomplete requested figures before export. panel_kwargs configures individual panels.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Create figures, source tables, and provenance for a perturbation report.

Partial bundles remain the default when optional evidence is absent.
``strict=True`` requires all three standard panels unless a narrower
``required_panels`` list is supplied. An explicitly required panel always
raises if unavailable, before any files are written. Panel names are
``consensus_state_map``, ``state_evidence_panel``, ``representation_heatmap``;
``panel_kwargs`` maps these names to their plotting options.

``formats`` accepts PNG/SVG/PDF; ``dpi`` also controls rasterized artists.
``width_mm``/``height_mm`` specify export canvas dimensions. The default
``bbox_inches=None`` preserves those dimensions. The bundle retains legacy
Figure entries and adds ``plot_results`` with source data and metadata.
```

</details>

### `scgeo.pl.recovery_compass`

Signature ScGeo synthesis plot combining: - PAGA topology - centroid displacement - velocity-shift alignment - OOD burden - abundance

```text
scgeo.pl.recovery_compass(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    paga_key: 'str' = 'paga',
    velocity_basis: 'Optional[str]' = None,
    ood_key: 'Optional[str]' = None,
    min_cells: 'int' = 15,
    connectivity_threshold: 'float' = 0.05,
    node_size_mode: 'str' = 'group1_n',
    node_size_scale: 'float' = 380.0,
    fill_color_mode: 'str' = 'alignment',
    fill_cmap: 'str' = 'coolwarm',
    fill_vmin: 'float' = -1.0,
    fill_vmax: 'float' = 1.0,
    ring_mode: 'str' = 'ood_frac',
    ring_color: 'str' = 'gold',
    ring_max_lw: 'float' = 4.0,
    arrow_color_mode: 'str' = 'shift',
    arrow_color: 'str' = 'black',
    arrow_cmap: 'str' = 'magma',
    arrow_scale: 'float' = 1.0,
    arrow_width: 'float' = 0.008,
    edge_alpha: 'float' = 0.45,
    edge_lw: 'float' = 2.0,
    bg_size: 'float' = 5.0,
    bg_alpha: 'float' = 0.08,
    label: 'bool' = True,
    label_top_n: 'Optional[int]' = 12,
    label_fontsize: 'int' = 8,
    legend: 'bool' = True,
    title: 'Optional[str]' = None,
    figsize: 'tuple[float, float]' = (9.5, 8.0),
    ax=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_recovery_compass.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Signature ScGeo synthesis plot combining:
- PAGA topology
- centroid displacement
- velocity-shift alignment
- OOD burden
- abundance

Returns
-------
If return_data is False:
    (fig, ax) or ax
If return_data is True:
    (fig, ax, node_df, edges)
```

</details>

### `scgeo.pl.replicate_effect`

Plot one biological sample center per point in an explicit coordinate.

```text
scgeo.pl.replicate_effect(
    data_or_adata,
    *,
    coordinate,
    store_key='robust_shift',
    scope='auto',
    state_order=None,
    condition_order=None,
    pair_key=None,
    summary='mean',
    palette=None,
    jitter=0.08,
    seed=0,
    axes=None,
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_replicate_effect.py)

- One point per biological sample in an explicitly selected coordinate. Descriptive mean/median summaries give equal weight to samples; no CI or test is fitted.
- Pair lines require an explicit real pairing key. Independent cross-sectional samples remain unconnected.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot one biological sample center per point in an explicit coordinate.

``data_or_adata`` is an AnnData with sample-aware robust-shift results, the
result dictionary, or a tidy table from ``sg.get.sample_centers``. Required
table columns are ``state``, ``condition``, ``sample``, ``coordinate``,
``value``, and ``representation``. ``coordinate`` is explicit and zero-based
for tables produced by the getter. Select one representation before plotting.

``summary='mean'`` or ``'median'`` gives an equally weighted, descriptive
summary of the plotted sample coordinates; it is not necessarily the
multivariate robust condition center. ``summary=None`` hides it. No interval,
hypothesis test, or cell-weighted estimate is computed. Pair lines are drawn
only when ``pair_key`` explicitly identifies matching biological samples.
Missing pairs stay unconnected; ambiguous pairs are rejected.

``axes`` is an optional mapping from state labels to Axes. Returns a Figure,
``(Figure, data_dict)`` with ``return_data=True``, or a :class:`PlotResult`.
```

</details>

### `scgeo.pl.representation_stability_heatmap`

Plot stored state-by-representation values with metric-aware scales.

```text
scgeo.pl.representation_stability_heatmap(
    adata,
    *,
    store_key: 'str' = 'representation_stability',
    metric: 'str' = 'normalized_delta_norm',
    annotate_consensus: 'bool' = True,
    cluster_states: 'bool' = False,
    cluster_reps: 'bool' = False,
    order_by='input',
    state_order=None,
    rep_order=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap=None,
    class_palette=None,
    na_color='#BDBDBD',
    show_colorbar=True,
    axes=None,
    figsize=None,
    title=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)

- Supported metrics: normalized displacement, displacement rank, alignment cosine, alignment class, and supplied coverage metrics. Cosine/coverage values are validated against their domains; categories use a discrete legend.
- Choose explicit state/representation orders or order_by='input', 'mean', or 'cluster'. Legacy cluster_* flags retain mean sorting with a deprecation warning.
- When local-geometry representation diagnostics are available, a diagnostic strip shows neighborhood, distortion, state-graph, and coverage outlier flags; missing values are visually distinct from numerical zero.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored state-by-representation values with metric-aware scales.

Input may be AnnData, a tidy ``per_rep_state`` DataFrame (``node``, ``rep``
and metric columns), or a result mapping containing that table plus optional
``consensus_state`` and ``representation_diagnostics`` tables. Table input
makes archived figure data reusable without loading AnnData.

Alignment cosine defaults to a zero-centered [-1, 1] diverging scale;
coverage to [0, 1]; alignment classes use a discrete legend. Other numeric
metrics use a sequential scale. ``norm`` cannot be combined with bounds.
Explicit orders move listed labels first and retain remaining labels.
``order_by`` accepts 'input', 'mean', 'cluster', or separate ``states`` and
``representations`` modes. Clustering uses average linkage and RMS distance
over jointly finite entries; all-missing rows are appended. It never fills
missing plotted values. Legacy ``cluster_*`` flags retain mean sorting and
emit a deprecation warning.

Supplied axes use ``heatmap`` plus ``consensus`` and ``diagnostics`` when
applicable; optional ``colorbar`` supplies a dedicated colorbar axis.
```

</details>

### `scgeo.pl.robustness_matrix`

Plot a robustness heatmap with optional row-summary side bar.

```text
scgeo.pl.robustness_matrix(
    data: 'pd.DataFrame',
    *,
    row_key: 'str' = 'feature',
    col_key: 'str' = 'setting',
    value_key: 'str' = 'value',
    annot_key: 'Optional[str]' = None,
    row_order: 'Optional[Sequence[str]]' = None,
    col_order: 'Optional[Sequence[str]]' = None,
    sort_rows_by: 'Optional[str]' = None,
    ascending: 'bool' = False,
    summary: 'Optional[str]' = 'mean',
    pass_threshold: 'Optional[float]' = None,
    missing_policy: 'str' = 'exclude',
    cmap: 'str' = 'viridis',
    vmin: 'Optional[float]' = None,
    vmax: 'Optional[float]' = None,
    center: 'Optional[float]' = None,
    show_values: 'bool' = True,
    value_fmt: 'str' = '.2f',
    annot_fontsize: 'int' = 8,
    na_color: 'str' = '#d9d9d9',
    grid_lw: 'float' = 0.8,
    grid_color: 'str' = 'white',
    cbar_label: 'Optional[str]' = None,
    summary_label: 'Optional[str]' = None,
    figsize: 'tuple[float, float]' = (10.0, 6.0),
    title: 'Optional[str]' = None,
    axes: 'Optional[Mapping[str, Any]]' = None,
    show_coverage: 'bool' = True,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_robustness_matrix.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a robustness heatmap with optional row-summary side bar.

Parameters
----------
data
    Either a tidy DataFrame with [row_key, col_key, value_key] columns,
    or a wide DataFrame with rows as findings and columns as settings.
summary
    Optional row summary shown in a right-side bar plot. One of
    {'mean', 'median', 'min', 'max', 'pass_rate'} or None.
missing_policy
    For pass rates, exclude unavailable settings from the denominator
    (default) or count them as failures with ``"fail"`` (historical behavior).
    Evaluation coverage is always returned separately. Rows without any
    evaluated setting have NaN pass rate under ``"exclude"``.
axes
    Optional named mapping with ``heatmap`` and, when enabled, ``summary``.
center
    Optional visual midpoint for diverging data. If provided, color scaling
    is made symmetric around this value when vmin/vmax are not supplied.
```

</details>

### `scgeo.pl.score_embedding`

Plot an observation score or category on the first two embedding axes.

```text
scgeo.pl.score_embedding(
    adata,
    score_key: 'str',
    basis: 'str' = 'umap',
    *,
    layer: 'Optional[str]' = None,
    ax: 'Optional[plt.Axes]' = None,
    title: 'Optional[str]' = None,
    size: 'float' = 6.0,
    alpha: 'float' = 0.8,
    cmap: 'str' = 'viridis',
    vmin=None,
    vmax=None,
    norm=None,
    palette: 'Optional[Mapping[str, str]]' = None,
    na_color: 'str' = 'lightgrey',
    figsize=None,
    max_points: 'Optional[int]' = None,
    stratify_by: 'Optional[str]' = None,
    random_state: 'int' = 0,
    required_indices=None,
    rasterized='auto',
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_score.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot an observation score or category on the first two embedding axes.

Both ``umap`` and ``X_umap`` resolve to the same coordinates. Continuous
color limits use all finite input scores, even when ``max_points`` selects
a reproducible display subset. ``stratify_by`` preserves at least one point
per observed state; a budget too small for those states raises an error.
``required_indices`` preserves selected original cell positions when sampling;
these cells must have finite display coordinates. Categorical scores use a
stable label-to-color mapping and no colorbar.

The legacy return is ``(fig, ax)``. ``return_data=True`` adds the full source
table with a ``displayed`` indicator; ``return_result=True`` supplies a
:class:`PlotResult` with the original displayed indices and random seed.
The retained ``layer`` argument has no effect: scores come from ``obs``.
```

</details>

### `scgeo.pl.score_umap`

Display an observation score using the score_embedding plotting interface.

```text
scgeo.pl.score_umap(adata, score_key: 'str', **kwargs)
```

[View implementation](../../scgeo/pl/_score.py)

### `scgeo.pl.state_evidence_panel`

Plot effect estimates and independently assessed evidence by state.

```text
scgeo.pl.state_evidence_panel(
    report_or_adata,
    *,
    node_key=None,
    effect_metric: 'str' = 'auto',
    representation=None,
    sort_by=None,
    state_order=None,
    max_states: 'Optional[int]' = 25,
    page: 'int' = 0,
    show_ci: 'bool' = True,
    show_coverage: 'bool' = True,
    show_sample_counts: 'bool' = False,
    show_inference: 'bool' = False,
    evidence_mode: 'str' = 'compact',
    axes=None,
    label_wrap: 'Optional[int]' = 32,
    figsize=None,
    title=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)

- Raw and normalized effect intervals remain separate. Stored endpoints are drawn independently of the point estimate; reversed finite intervals fail before drawing.
- Full evidence mode shows availability separately from scientific consensus. Optional sample counts and inference labels reflect stored analysis, without fitting new statistics.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot effect estimates and independently assessed evidence by state.

``effect_metric='auto'`` prefers raw ``delta_norm`` when finite values
exist; otherwise it uses ``normalized_delta_norm``. Sorting defaults to
the displayed metric. Raw ``magnitude_ci95_low/high`` intervals are drawn
only with the raw metric. Normalized intervals require separately supplied
``normalized_magnitude_ci95_low/high`` columns in the same units.

``evidence_mode='full'`` adds an availability matrix and textual consensus
labels. Sample support is enabled with ``show_sample_counts``; inference
labels with ``show_inference``. Neither option performs a new analysis.
``state_order`` moves specified states first and retains remaining states.
``max_states=None`` displays all states; ``page`` is zero-based.

Supplied ``axes`` use lowercase keys: ``effect``, ``samples``,
``stability``, ``local_preservation``, ``local_distortion``, ``dynamics``,
``coverage``, ``availability`` or ``summary``, as applicable. Supplied
figures are not resized or laid out. Legacy returns are preserved;
``return_result=True`` returns a :class:`scgeo.pl.PlotResult`.
```

</details>

### `scgeo.pl.state_flow_alluvial`

Compare categorical annotations with readable ribbons, conserved counts, and explicit omitted associations.

```text
scgeo.pl.state_flow_alluvial(
    adata,
    *,
    columns: 'Sequence[str]',
    min_count: 'int' = 1,
    drop_na: 'bool' = False,
    normalize: 'bool' = False,
    sort_categories: 'bool' = False,
    color_by: 'str' = 'target',
    alpha: 'float' = 0.7,
    column_gap: 'float' = 1.8,
    category_gap: 'float' = 0.02,
    ribbon_curve: 'float' = 0.35,
    figsize: 'tuple[float, float]' = (11, 6),
    title: 'Optional[str]' = None,
    palette=None,
    ax=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
    category_order=None,
    order='input',
    label_format=None,
    label_fontsize=9,
    bar_width=0.08,
    show_omissions=True,
    return_result=False,
)
```

[View implementation](../../scgeo/pl/_state_flow_alluvial.py)

- Unreleased development upgrade; see the state-association and density guide and real-data plotting gallery.
- Each ribbon counts the same observations across adjacent columns. It does not establish tracked longitudinal transitions, lineage, or causality.
- All positive associations receive layout slots before min_count filtering. Category totals and fractions retain the full included-cell denominator; hidden links leave gaps.
- The legacy value field and geometry are fractions of included cells. normalize selects the default count or percentage labels and never renormalizes after filtering.
- True missing values are distinct from literal NA/nan labels. drop_na excludes genuinely missing rows across all requested columns and records their source flags.
- return_result=True adds cells, nodes, links, layout, and label sources with an alluvial axis. Existing owned, supplied-axis, and four-part return_data conventions are retained.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw count-conserving associations between categorical cell annotations.

Each row contributes one count to each adjacent pair in ``columns``. These
links connect annotations on the same cells; they do not establish lineage,
transitions between different cells, or elapsed time.

All positive counts and their layout slots are retained before ``min_count``
hides ribbons. Filtered mass remains blank, and category totals stay complete.
Both legacy ``normalize`` settings retain fraction-based geometry and
``value=count/N``, where N is the number of included cells before filtering.
``normalize=True`` changes the default labels to percentages; False shows
counts. ``label_format`` can explicitly choose label/count/percent/both.
Raw counts and full-denominator fractions are always exported separately.

``drop_na`` removes rows with genuine missing selected annotations; literal
labels such as 'NA', 'nan', and '(missing)' remain distinct categories.
Otherwise missing observations receive their own explicitly marked category.
Input ordering respects declared categorical levels, then first appearance
for other columns. ``category_order`` maps stage names to complete observed
orders (use None for missing); ``order='count'`` or ``'barycentric'`` provides
deterministic alternative layouts. Explicit orders remain fixed.

``palette`` may map labels to colors or stage names to complete label-color
mappings. Labels sit outside the bars with point-space spacing and leaders.
Supplied axes retain figure size and already fixed limits. ``return_result``
returns a PlotResult with the named ``alluvial`` axis and complete cells,
nodes, links, layouts, label positions, and provenance. Existing returns are
preserved: owned (fig,ax), supplied ax, or return_data=(fig,ax,stage_boxes,
displayed_allocations). The result object's layout includes hidden links.
```

</details>

### `scgeo.pl.state_flow_chord`

Display a two-column category cross-tab as circular ribbons with complete source counts and omission-aware geometry.

```text
scgeo.pl.state_flow_chord(
    adata,
    *,
    columns,
    min_count=1,
    drop_na=False,
    normalize=False,
    category_order=None,
    order='input',
    sort_categories=False,
    color_by='target',
    palette=None,
    alpha=0.65,
    category_gap=2.0,
    stage_gap=18.0,
    node_width=0.09,
    label_format=None,
    label_fontsize=9,
    show_omissions=True,
    ax=None,
    figsize=(8, 7),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_state_flow_chord.py)

- Requires exactly two obs columns; each stage keeps distinct node identities even when category text is identical.
- Ribbons show category associations among the supplied cells. Source/target denotes the input-column order, not inferred biological direction or elapsed time.
- min_count hides links without reallocating their slots. Raw counts, full-denominator fractions, retained/omitted flags, angular layouts, and labels remain exportable.
- Supports explicit category orders, stable palettes, supplied axes/SubFigures, and return_result=True with a chord axis. Legacy-style owned (fig, ax), supplied ax, and four-part return_data are supported.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw a two-stage bipartite chord of same-cell category associations.

``columns`` must contain exactly two categorical annotations. The left half
is the source annotation and the right half is its target annotation on the
same cells. Repeated category names remain distinct stage-specific nodes.
Ribbons encode counts of co-assigned labels, not lineage, a transition rate,
or elapsed time. There are no inferred connections between different cells.

Each half has a common angular scale for the full included N cells. Full
category totals and every positive link receive angular slots before
``min_count`` hides small links, leaving honest gaps instead of renormalizing
retained mass. Category fractions and ribbon fractions use N, not 2N, even
though each cell contributes to one arc on each half. ``category_gap`` and
``stage_gap`` are degrees; ``node_width`` is a fraction of the unit radius.

``category_order`` maps stage names to complete observed orders; None denotes
genuine missing labels. Input order respects categorical levels, otherwise
first appearance. Count and deterministic barycentric ordering are optional.
``drop_na`` drops genuine missing selected annotations only; literal 'NA'
remains a real category. A flat palette shares label colors across stages;
nested stage palettes can give distinct colors to equal labels.

Geometry uses full-denominator fractions in both ``normalize`` modes.
Normalize selects percentage labels by default; False selects counts.
``label_format`` explicitly accepts label/count/percent/both. Supplied axes
retain figure size and fixed limits. Default returns are (fig,ax) for owned
figures and ax for supplied axes. ``return_data`` returns
(fig,ax,node_layout,link_layout), including hidden links. ``return_result``
returns a PlotResult with the named ``chord`` axis, all source cells and
counts, angular geometry, label positions, and interpretation metadata.
```

</details>

### `scgeo.pl.state_flow_sankey`

Inspect categorical associations interactively with complete and visible counts in hover information.

```text
scgeo.pl.state_flow_sankey(
    adata,
    *,
    columns: 'Sequence[str]',
    min_count: 'int' = 1,
    drop_na: 'bool' = False,
    title: 'Optional[str]' = None,
    pad: 'int' = 18,
    thickness: 'int' = 18,
    width: 'int' = 1000,
    height: 'int' = 550,
    arrangement: 'str' = 'snap',
    node_color: 'str' = 'rgba(120,120,120,0.85)',
    link_color: 'str' = 'rgba(120,120,120,0.28)',
    return_data: 'bool' = False,
    show: 'bool' = True,
    category_order=None,
    order='input',
    sort_categories=False,
    palette=None,
    color_by='target',
)
```

[View implementation](../../scgeo/pl/_state_flow_sankey.py)

- Uses the same missing-value and stage-identity policy as the static state-association views. Links are same-cell cross-tabulations, not tracked transitions.
- Plotly sizes its nodes from visible links after filtering. Hover and metadata report complete and omitted counts; use alluvial/chord when reserved omitted geometry is required.
- Existing Plotly figure and (fig, nodes, links) data returns are retained.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot interactive categorical associations between annotations on cells.

Shared counts match ``state_flow_alluvial`` and ``state_flow_chord``.
``columns`` names ordered annotations on the same rows; links do not imply
cell lineage, longitudinal transitions, or elapsed time. Missing values are
detected before string conversion, so ``drop_na`` leaves literal 'NA' intact.
``category_order`` maps stages to complete observed orders (None denotes
genuine missingness); input order respects categorical levels. Optional
count or barycentric ordering and complete flat/nested palettes are shared
with the Matplotlib views.

``min_count`` hides links without changing their raw counts or denominator.
Native Plotly node geometry follows visible links after filtering, so its
displayed node areas need not represent complete category marginals. Hover
reports full counts alongside visible and omitted incoming/outgoing counts.
For reserved gaps and full-marginal geometry, use alluvial or chord.

The default return remains a Plotly Figure. ``return_data=True`` preserves
(fig,nodes_df,displayed_links_df); link ``value`` is the raw cell count.
Complete node/link counts and filtering provenance are retained in
``fig.layout.meta['scgeo']``. PlotResult/export_figure are Matplotlib APIs;
Plotly's native HTML/JSON exports remain available for this interactive view.
```

</details>

### `scgeo.pl.style_context`

Temporarily apply a paper or presentation style without global changes.

```text
scgeo.pl.style_context(style='paper', *, width_mm=None, height_mm=None, overrides=None)
```

[View implementation](../../scgeo/pl/_style.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Temporarily apply a paper or presentation style without global changes.

``width_mm`` and ``height_mm`` set the default canvas for plots that do not
supply their own figsize. Use ``export_figure`` to size an existing figure.
``overrides`` accepts any Matplotlib rcParams mapping. SVG text remains text
and PDF uses TrueType fonts; viewers need the selected font for SVG editing.
```

</details>

### `scgeo.pl.velocity_condition_compare`

Compare two observed condition velocity fields at identical grid points.

```text
scgeo.pl.velocity_condition_compare(
    adata,
    *,
    condition_key,
    group0,
    group1,
    basis='umap',
    mode='grid',
    grid_size=25,
    n_neighbors=30,
    radius=None,
    min_neighbors=5,
    min_coherence=0.2,
    scale='auto',
    color_by='coherence',
    cmap='viridis',
    max_points=3500,
    random_state=0,
    stream_density=1.0,
    axes=None,
    figsize=(15.5, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_velocity_condition_compare.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compare two observed condition velocity fields at identical grid points.

Each field uses only the cells in its selected condition. Grid coordinates,
Gaussian radius, neighbor cap, support thresholds, color norm, view limits,
and (in grid mode) the arrow multiplier are shared. The third panel shows
cosine agreement of projected local means *only where both are supported*;
singly supported positions are indicated but never assigned a direction.
It is descriptive, not a causal or longitudinal cell-trajectory estimate.

``adata.obsm['X_<basis>']`` and matching ``velocity_<basis>`` are required.
Reusing an upstream projected velocity does not refit its kinetics.
``return_result=True`` enables source-grid, support-mask and parameter
exports via :class:`scgeo.pl.PlotResult`.
```

</details>

### `scgeo.pl.velocity_field`

Display locally averaged projected velocity as a support-masked arrow grid or streamlines, with condition selection, comparable scales, and complete source exports.

```text
scgeo.pl.velocity_field(
    adata,
    *,
    basis='umap',
    velocity_basis=None,
    mode='grid',
    condition_key=None,
    groups=None,
    background_key=None,
    palette=None,
    background='all',
    max_points=None,
    random_state=0,
    required_indices=None,
    grid_size=25,
    grid_extent=None,
    n_neighbors=30,
    radius=None,
    min_neighbors=5,
    min_coherence=0.2,
    scale='auto',
    color='#008C95',
    color_by=None,
    cmap='viridis',
    norm=None,
    vmin=None,
    vmax=None,
    arrow_linewidth=1.1,
    arrow_head_size=5.5,
    stream_density=1.0,
    stream_linewidth=1.2,
    background_size=5.0,
    background_alpha=0.18,
    alpha=0.9,
    aspect='equal',
    xlim=None,
    ylim=None,
    show_colorbar=True,
    cax=None,
    legend=True,
    legend_loc='outside',
    ax=None,
    figsize=(7.0, 6.0),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../../scgeo/pl/_velocity_field.py)

- This is an unreleased development feature; see the [local velocity guide](https://github.com/liuifrec/scgeo/blob/main/docs/VELOCITY_FIELDS.md) and executable gallery.
- Requires a velocity array projected into the matching display basis. The function does not fit RNA kinetics, project high-dimensional vectors, infer cell trajectories, or estimate elapsed time.
- Finite-radius Gaussian weighting summarizes eligible selected cells. Local coherence is norm(weighted mean velocity) divided by weighted mean velocity norm; it is a direction-agreement diagnostic, not statistical confidence.
- groups selects field inputs independently of the background. max_points changes only background drawing; full eligible inputs define the field.
- Use common grid_extent, radius, neighborhood/support settings, numeric arrow scale, and color normalization for matched condition panels. Axis limits crop the view without redefining the grid.
- Streamlines are masked and their actual segments are checked against supported interpolation cells. Source exports include rejected grid positions, filtering reasons, scaled arrows, and rendered path vertices.
- return_result=True returns PlotResult with field and optional colorbar axes. Default owned (fig, ax) and supplied ax conventions are supported; return_data=True returns (fig, ax, data).


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot local summaries of already projected velocity on an embedding.

The input must contain a matching ``velocity_<basis>`` array with exactly
the display embedding shape. Only the first two display coordinates and
velocity components are used. This function does not project vectors from
another representation or estimate RNA kinetics, trajectories, or time.

``condition_key`` and ``groups`` select the field's source cells; ``groups``
omitted means all cells. Coordinates and velocity are filtered jointly for
finite rows. Background scope is independent: ``'all'`` or ``'selected'``.
``background_key`` colors categorical cell labels with a stable palette.
``max_points`` affects background drawing only. The field always uses every
eligible selected cell, with deterministic background sampling by category
and condition and optional required observation positions.

A regular ``grid_size`` (integer or ``(nx, ny)``) summarizes at most
``n_neighbors`` nearest cells inside a finite ``radius``. Gaussian weights
use sigma=radius/2. The default radius is 1.5 times the median source-cell
kth-neighbor distance. ``grid_extent=(xmin,xmax,ymin,ymax)`` fixes evaluation
coordinates across panels; otherwise full selected-source bounds are used.
``xlim`` and ``ylim`` change only the view. A grid position is supported when
its count reaches ``min_neighbors``, its mean vector is nonzero, and its
coherence reaches ``min_coherence``. Coherence is the norm of the weighted
mean vector divided by the weighted mean individual norm: a directional
agreement diagnostic, not a confidence level or significance test.

``mode='grid'`` draws arrows with line widths and heads in points. ``scale``
is a positive multiplier on the raw mean vectors or ``'auto'``; their lengths
remain projected vectors times that scale. The resolved multiplier is
explicit in the legend and metadata. ``mode='stream'`` traces geometric direction paths,
masks unsupported regions, and audits the rendered segments for support.
Scale does not apply to streamlines and never represents elapsed time.
``color_by='speed'`` uses the norm of the local mean; ``'coherence'`` uses its
directional agreement. ``norm`` or ``vmin``/``vmax`` support shared scales.

Supplied axes retain their figure size and already fixed limits. Use an
explicit ``cax`` for a shared colorbar (orientation follows its shape), or
``show_colorbar=False``. Owned figures return ``(fig, ax)``, supplied axes
return ``ax``, and ``return_data=True`` returns ``(fig, ax, data)``. Opt-in
``return_result=True`` returns a :class:`PlotResult` with a named ``field``
axis, all source cells and grid rows, display flags, scaled vector endpoints,
actual streamline path vertices, support rules, and plotting provenance.
```

</details>

### `scgeo.pl.velocity_shift_alignment`

Plot display-space vectors or stored quantitative alignment with an overlay.

```text
scgeo.pl.velocity_shift_alignment(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    velocity_basis: 'Optional[str]' = None,
    min_cells: 'int' = 15,
    agg: 'str' = 'mean',
    bg_size: 'float' = 6.0,
    bg_alpha: 'float' = 0.1,
    node_size: 'float' = 180.0,
    shift_scale: 'float' = 1.0,
    velocity_scale: 'float' = 50.0,
    shift_color: 'str' = 'black',
    velocity_color: 'str' = 'cyan',
    shift_alpha: 'float' = 0.95,
    velocity_alpha: 'float' = 0.95,
    arrow_width: 'float' = 0.006,
    show_shift_arrow: 'bool' = True,
    show_velocity_arrow: 'bool' = True,
    color_by_alignment: 'bool' = True,
    alignment_cmap: 'str' = 'coolwarm',
    alignment_pos_thr: 'float' = 0.3,
    alignment_neg_thr: 'float' = -0.3,
    palette: 'Optional[dict[str, Any]]' = None,
    label: 'bool' = True,
    label_top_n: 'Optional[int]' = None,
    label_mode: 'str' = 'shift',
    label_fontsize: 'int' = 8,
    title: 'Optional[str]' = None,
    ax=None,
    figsize: 'tuple[float, float]' = (8.2, 7.0),
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    mode: 'str' = 'display',
    store_key: 'str' = 'representation_stability',
    analysis_rep: 'Optional[str]' = None,
    alignment_table: 'Optional[pd.DataFrame]' = None,
    axes: 'Optional[Mapping[str, Any]]' = None,
    state_order=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_velocity_shift_alignment.py)

- mode='display' retains the legacy calculation in display coordinates. mode='stored' renders selected quantitative alignment separately from the display overlay.
- Stored-mode projected velocities pool only the two requested conditions. Changing display coordinates does not change quantitative alignment.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot display-space vectors or stored quantitative alignment with an overlay.

``mode="display"`` preserves the legacy calculation in the first two display
coordinates and pools projected velocity over all observations in each state.
``mode="stored"`` renders stored alignment values on a separate
axis and colors a display-only overlay by those fixed values. It reads
``adata.uns['scgeo'][store_key]['per_rep_state']`` by default, or an explicit
``alignment_table``. A representation must be selected with ``analysis_rep``
when multiple representations exist or the table has no representation
metadata. Stored contrast metadata is checked against the requested groups.

Stored mode accepts an ``axes`` mapping with ``overlay`` and ``alignment``
and returns ``(fig, (overlay_ax, alignment_ax)[, data])``. ``return_result``
is an opt-in consistent result for either mode. High-dimensional analysis
vectors are never used as display arrows; missing or incompatible projected
velocity leaves a condition-contrast overlay with an explanatory note. In
stored mode, projected velocity is aggregated only over the selected two
conditions, matching the contrast scope.

Node fill encodes cosine alignment between shift and velocity:
  +1  aligned
   0  unrelated
  -1  discordant

Notes
-----
`velocity_scale` is a display-only scaling factor. Velocity vectors are often
much smaller than condition-shift vectors in embedding coordinates, so a larger
default is used to keep the velocity arrows visible.

Returns
-------
If return_data is False:
    (fig, ax) or ax
If return_data is True:
    (fig, ax, align_df)
```

</details>
