# scgeo.pl

ScGeo 0.1.0 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
**Figures.** Visualize stored results and assemble evidence panels, reports and figure-source exports.

- [`alignment_panel`](#scgeoplalignment_panel)
- [`ambiguity_panel`](#scgeoplambiguity_panel)
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
- [`gallery_overview`](#scgeoplgallery_overview)
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
- [`representation_stability_heatmap`](#scgeoplrepresentation_stability_heatmap)
- [`robustness_matrix`](#scgeoplrobustness_matrix)
- [`score_embedding`](#scgeoplscore_embedding)
- [`score_umap`](#scgeoplscore_umap)
- [`state_evidence_panel`](#scgeoplstate_evidence_panel)
- [`state_flow_alluvial`](#scgeoplstate_flow_alluvial)
- [`state_flow_sankey`](#scgeoplstate_flow_sankey)
- [`velocity_shift_alignment`](#scgeoplvelocity_shift_alignment)

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

### `scgeo.pl.composition_drift`

Plot a 3-panel composition drift report: 1) embedding centroids sized by abundance and colored by signed delta fraction 2) horizontal barplot of signed delta fraction 3) side-by-side proportion bars for group0 and group1

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
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_composition_drift.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a 3-panel composition drift report:
1) embedding centroids sized by abundance and colored by signed delta fraction
2) horizontal barplot of signed delta fraction
3) side-by-side proportion bars for group0 and group1
```

</details>

### `scgeo.pl.consensus_state_map`

Plot state-level consensus labels on a display embedding.

```text
scgeo.pl.consensus_state_map(
    adata,
    *,
    node_key,
    basis: 'str' = 'umap',
    store_key: 'str' = 'representation_stability',
    label_states: 'bool' = True,
    title=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot state-level consensus labels on a display embedding.

The display embedding is used only for visualization; consensus labels are
read from ``representation_stability`` and are not recalculated here.
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

Plot embedding density per group (small multiples).

```text
scgeo.pl.embedding_density(
    adata,
    groupby: 'str',
    *,
    basis: 'str' = 'umap',
    groups: 'Optional[Iterable[str]]' = None,
    gridsize: 'int' = 160,
    normalize: 'str' = 'per_group',
    cmap: 'str' = 'magma',
    contour: 'bool' = False,
    contour_levels: 'int' = 12,
    imshow_alpha: 'float' = 0.85,
    transparent_background: 'bool' = True,
    mask_zeros: 'bool' = True,
    background: 'Optional[str]' = None,
    figsize=None,
    show: 'bool' = True,
    smooth_k: 'int' = 5,
    log1p: 'bool' = True,
    contour_lines: 'bool' = True,
    contour_linewidth: 'float' = 0.7,
    contour_alpha: 'float' = 0.95,
    contour_level_mode: 'str' = 'quantile',
)
```

[View implementation](../../scgeo/pl/_density.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot embedding density per group (small multiples).

Primary goal: quick, stable, and notebook-friendly density diagnostics.

Implementation:
  - Computes a 2D histogram on the embedding.
  - Renders either an ``imshow`` heatmap (default) or a ``contourf`` plot
    (set ``contour=True``) to avoid the "blocky / too dark" look.

Parameters
----------
contour
    If True, draw a filled contour plot (``contourf``) instead of imshow.
transparent_background
    If True, set figure + axes patches to transparent so the density layers
    don't sit on a dark rectangle in some notebook themes.
```

</details>

### `scgeo.pl.gallery_overview`

Render a 2x2 overview gallery of core ScGeo plots: - paga_shift_map - ood_landscape - velocity_shift_alignment - composition_drift

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
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_gallery.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Render a 2x2 overview gallery of core ScGeo plots:
- paga_shift_map
- ood_landscape
- velocity_shift_alignment
- composition_drift
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

Plot stored per-cell local distortion values on a display embedding.

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
    title=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored per-cell local distortion values on a display embedding.
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

ScGeo-style PAGA summary: - graph edges from PAGA - node pies from pie_key (e.g. timepoint) - black arrows = geometry shift - optional cyan arrows = mean velocity - node color by delta/alignment/palette/constant

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
    velocity_basis: 'Optional[str]' = 'umap',
    show_velocity: 'bool' = True,
    node_color_mode: 'str' = 'delta',
    highlight_nodes: 'Optional[list[str]]' = None,
    **kwargs,
)
```

[View implementation](../../scgeo/pl/_paga_scgeo.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
ScGeo-style PAGA summary:
- graph edges from PAGA
- node pies from pie_key (e.g. timepoint)
- black arrows = geometry shift
- optional cyan arrows = mean velocity
- node color by delta/alignment/palette/constant
```

</details>

### `scgeo.pl.paga_shift_map`

Overlay a PAGA graph anchored on group0 centroids in embedding space, with arrows pointing from group0 -> group1 centroids for each node.

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
    background_alpha: 'float' = 0.15,
    node_size: 'float' = 220.0,
    node_scale_by_n: 'bool' = True,
    edge_lw: 'float' = 2.0,
    edge_alpha: 'float' = 0.55,
    arrow_width: 'float' = 0.008,
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
    velocity_color: 'str' = 'cyan',
    velocity_scale: 'float' = 50.0,
    velocity_alpha: 'float' = 0.95,
    node_color_mode: 'str' = 'palette',
    alignment_df: 'Optional[pd.DataFrame]' = None,
    alignment_key: 'str' = 'alignment_cosine',
    delta_key: 'str' = 'delta_frac',
    constant_node_color: 'str' = 'gold',
    highlight_nodes: 'Optional[list[str]]' = None,
    highlight_edgecolor: 'str' = 'black',
    highlight_lw: 'float' = 2.0,
    ax=None,
    figsize: 'tuple[float, float]' = (8.0, 7.0),
    title: 'Optional[str]' = None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_paga_shift_map.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Overlay a PAGA graph anchored on group0 centroids in embedding space,
with arrows pointing from group0 -> group1 centroids for each node.

Optional upgrades:
- pie_key: draw node pies for composition (e.g. timepoint)
- show_velocity: overlay node-level mean velocity arrows
- node_color_mode: color nodes by palette / alignment / delta / constant
- highlight_nodes: emphasize selected nodes
```

</details>

### `scgeo.pl.perturbation_report`

Create the standard ScGeo perturbation report bundle.

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
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)

- Export bundle: state report CSV, representation diagnostics CSV, metadata JSON, warnings text, alt-text text, and PNG/SVG figures.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Create the standard ScGeo perturbation report bundle.
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

### `scgeo.pl.representation_stability_heatmap`

Plot a state-by-representation heatmap from stored representation stability.

```text
scgeo.pl.representation_stability_heatmap(
    adata,
    *,
    store_key: 'str' = 'representation_stability',
    metric: 'str' = 'normalized_delta_norm',
    annotate_consensus: 'bool' = True,
    cluster_states: 'bool' = False,
    cluster_reps: 'bool' = False,
    figsize=None,
    title=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)

- Supported metrics: normalized displacement, displacement rank, alignment cosine, and alignment class.
- When local-geometry representation diagnostics are available, a diagnostic strip shows neighborhood, distortion, state-graph, and coverage outlier flags; missing values are visually distinct from numerical zero.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a state-by-representation heatmap from stored representation stability.
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
    return_data: 'bool' = False,
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
center
    Optional visual midpoint for diverging data. If provided, color scaling
    is made symmetric around this value when vmin/vmax are not supplied.
```

</details>

### `scgeo.pl.score_embedding`

Plot an obs score on an embedding (UMAP/PCA/etc) with minimal dependencies. Returns (fig, ax).

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
    na_color: 'str' = 'lightgrey',
    figsize=None,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_score.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot an obs score on an embedding (UMAP/PCA/etc) with minimal dependencies.
Returns (fig, ax).
```

</details>

### `scgeo.pl.score_umap`

Display an observation score using the score_embedding plotting interface.

```text
scgeo.pl.score_umap(adata, score_key: 'str', **kwargs)
```

[View implementation](../../scgeo/pl/_score.py)

### `scgeo.pl.state_evidence_panel`

Plot a progressive-disclosure state evidence panel from a ScGeo report.

```text
scgeo.pl.state_evidence_panel(
    report_or_adata,
    *,
    node_key=None,
    sort_by: 'str' = 'normalized_delta_norm',
    max_states: 'int' = 25,
    show_ci: 'bool' = True,
    show_coverage: 'bool' = True,
    figsize=None,
    title=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_perturbation_report.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a progressive-disclosure state evidence panel from a ScGeo report.

Tracks are omitted when their source columns are unavailable. Effect,
stability, local-neighborhood preservation, local distortion, dynamics,
and coverage/warning indicators are drawn on separate axes.
```

</details>

### `scgeo.pl.state_flow_alluvial`

Draw an alluvial / ribbon plot for ordered categorical columns in `adata.obs`.

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
    palette: 'Optional[dict[str, tuple[float, float, float, float]]]' = None,
    ax=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_state_flow_alluvial.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw an alluvial / ribbon plot for ordered categorical columns in `adata.obs`.

Parameters
----------
adata
    AnnData object.
columns
    Ordered categorical columns, e.g. ["timepoint", "alignment_group"].
min_count
    Minimum flow count to retain.
drop_na
    If True, drop rows containing 'NA' in any selected column.
normalize
    If True, each stage totals to 1.0 instead of raw counts.
sort_categories
    If True, sort labels alphabetically within each stage.
color_by
    One of {"source", "target"}; determines ribbon color source.
alpha
    Ribbon transparency.
return_data
    If True, return (fig, ax, stage_boxes, link_df).
```

</details>

### `scgeo.pl.state_flow_sankey`

Plot a categorical state-flow Sankey diagram from columns in `adata.obs`.

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
)
```

[View implementation](../../scgeo/pl/_state_flow_sankey.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a categorical state-flow Sankey diagram from columns in `adata.obs`.

Parameters
----------
adata
    AnnData object.
columns
    Ordered categorical columns from `adata.obs`, e.g.
    ["timepoint", "alignment_group", "macrostates_fwd"].
min_count
    Minimum flow count to retain.
drop_na
    If True, rows containing 'NA' in any selected column are removed.
return_data
    If True, return (fig, nodes_df, links_df).

Notes
-----
This is a categorical flow summary, useful for showing how cells distribute
across timepoints, alignment classes, clusters, macrostates, annotations, or OOD groups.
```

</details>

### `scgeo.pl.velocity_shift_alignment`

Plot node-wise observed shift vectors and mean velocity vectors on the same embedding.

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
    show: 'bool' = True,
)
```

[View implementation](../../scgeo/pl/_velocity_shift_alignment.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot node-wise observed shift vectors and mean velocity vectors on the same embedding.

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
