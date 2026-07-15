# scgeo.pl

## `scgeo.pl.alignment_panel`

**Signature**  
`(adata, score_key: 'str', *, basis: 'str' = 'umap', topk: 'Optional[int]' = 200, cmap: 'str' = 'viridis', figsize=(11, 5), title: 'Optional[str]' = None, show: 'bool' = True)`

**Docstring**  
1×2 panel:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.ambiguity_panel`

**Signature**  
`(adata, score_key: 'str', *, basis: 'str' = 'umap', topk: 'Optional[int]' = 200, cmap: 'str' = 'inferno', figsize=(11, 5), title: 'Optional[str]' = None, show: 'bool' = True)`

**Docstring**  
1×2 panel:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.composition_drift`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', agg: 'str' = 'mean', bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.1, centroid_size: 'float' = 320.0, centroid_scale_by_n: 'bool' = True, centroid_edgecolor: 'str' = 'white', centroid_lw: 'float' = 1.0, drift_cmap: 'str' = 'coolwarm', drift_vmax: 'Optional[float]' = None, bar_alpha: 'float' = 0.9, top_n: 'Optional[int]' = None, sort_by: 'str' = 'abs_delta_frac', palette: 'Optional[dict[str, Any]]' = None, title: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (14.0, 5.2), return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a 3-panel composition drift report:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.consensus_state_map`

**Signature**  
`(adata, *, node_key, basis: 'str' = 'umap', store_key: 'str' = 'representation_stability', label_states: 'bool' = True, title=None, show: 'bool' = True)`

**Docstring**  
Plot state-level consensus labels on a display embedding.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.consensus_structure`

**Signature**  
`(adata, *, score_key: 'str' = 'cs_score', basis: 'str' = 'umap', groupby: 'str' = 'louvain', topk: 'int' = 300, bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.25, hi_size: 'float' = 30.0, hi_alpha: 'float' = 0.95, cmap: 'str' = 'viridis', outline_lw: 'float' = 1.6, outline_alpha: 'float' = 0.95, use_scanpy_colors: 'bool' = True, min_frac_in_legend: 'float' = 0.0, title: 'Optional[str]' = None, figsize=(6.5, 5.5), ax=None, show: 'bool' = True, print_legend: 'bool' = True) -> 'tuple[Any, Any, ConsensusStructureStats, List[Dict[str, Any]]]'`

**Docstring**  
Consensus structure (UMAP only) + legend returned separately.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.consensus_subspace_panel`

**Signature**  
`(adata, *, score_key: 'str' = 'cs_score', basis: 'str' = 'umap', store_key: 'str' = 'consensus_subspace', topk: 'Optional[int]' = 200, figsize=(11, 5), show: 'bool' = True)`

**Docstring**  
Panel for consensus subspace:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.delta_rank`

**Signature**  
`(adata, store_key: 'str' = 'scgeo', kind: 'str' = 'shift', level: 'str' = 'by', *, top_n: 'int | None' = None, rotate_xticks: 'int' = 60)`

**Docstring**  
Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.density_overlap`

**Signature**  
`(adata, *, store_key: 'str' = 'density_overlap', level: 'str' = 'by', metrics: 'Tuple[str, ...]' = ('bc', 'hellinger'), top_k: 'Optional[int]' = 10, highlight: 'str' = 'worst', sort_by: 'Optional[str]' = None, style: "Literal['bar', 'lollipop']" = 'lollipop', annotate: 'bool' = True, grid: 'bool' = True, figsize=None, show: 'bool' = True)`

**Docstring**  
Plot density overlap summaries from ``sg.tl.density_overlap``.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.density_overlap_grid`

**Signature**  
`(adata, *, key: 'str' = 'density_overlap', basis: 'str' = 'umap', panel: 'str' = 'overlap', ax=None, figsize=(5.5, 5.0), cmap: 'str' = 'magma', alpha_points: 'float' = 0.05, show_points: 'bool' = True, title: 'Optional[str]' = None, show: 'bool' = True)`

**Docstring**  
Visualize density overlap results from `tl.density_overlap`.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.distribution_report`

**Signature**  
`(adata, *, basis: 'str' = 'umap', condition_key: 'str' = 'condition', group0: 'Optional[str]' = None, group1: 'Optional[str]' = None, density_key: 'str' = 'density_overlap', test_key: 'str' = 'distribution_test', score_key: 'Optional[str]' = None, top_k: 'int' = 8, figsize=(11, 8), show: 'bool' = True)`

**Docstring**  
2×2 report panel for cross-condition embedding comparison.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.distribution_test`

**Signature**  
`(adata, *, store_key: 'str' = 'distribution_test', level: 'str' = 'by', value: 'str' = 'p_perm', stat_key: 'str' = 'stat', top_k: 'Optional[int]' = 10, highlight: 'str' = 'strongest', figsize=None, show: 'bool' = True)`

**Docstring**  
Plot distribution test summaries from `sg.tl.distribution_test`.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.distribution_test_volcano`

**Signature**  
`(adata, *, key: 'str' = 'distribution_test', effect_col: 'str' = 'effect', p_col: 'str' = 'p_adj', label_col: 'str' = 'group', top_n_labels: 'int' = 10, ax=None, figsize=(6.0, 5.0), title: 'str | None' = None, show: 'bool' = True)`

**Docstring**  
Volcano-like plot for distribution/comparison results.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.embedding_density`

**Signature**  
`(adata, groupby: 'str', *, basis: 'str' = 'umap', groups: 'Optional[Iterable[str]]' = None, gridsize: 'int' = 160, normalize: 'str' = 'per_group', cmap: 'str' = 'magma', contour: 'bool' = False, contour_levels: 'int' = 12, imshow_alpha: 'float' = 0.85, transparent_background: 'bool' = True, mask_zeros: 'bool' = True, background: 'Optional[str]' = None, figsize=None, show: 'bool' = True, smooth_k: 'int' = 5, log1p: 'bool' = True, contour_lines: 'bool' = True, contour_linewidth: 'float' = 0.7, contour_alpha: 'float' = 0.95, contour_level_mode: 'str' = 'quantile')`

**Docstring**  
Plot embedding density per group (small multiples).

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.gallery_overview`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', ood_key: 'str' = 'scgeo_ood', velocity_basis: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (16.0, 12.0), title: 'Optional[str]' = None, show: 'bool' = True)`

**Docstring**  
Render a 2x2 overview gallery of core ScGeo plots:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.highlight_topk_cells`

**Signature**  
`(adata, score_key: 'str', basis: 'str' = 'umap', *, topk: 'int' = 300, ax=None, figsize=(5.8, 5.2), bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.2, hi_size: 'float' = 30.0, hi_alpha: 'float' = 0.95, cmap: 'str' = 'viridis', title: 'str | None' = None, show: 'bool' = True, groupby: 'str | None' = None, use_scanpy_colors: 'bool' = True, outline_topk: 'bool' = False, outline_lw: 'float' = 1.6, outline_alpha: 'float' = 0.95, add_colorbar: 'bool' = True)`

**Docstring**  
Highlight top-k cells by a score on an embedding.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.legend_from_data`

**Signature**  
`(legend_data: 'Iterable[Dict[str, Any]]', *, max_items: 'int' = 20, ncol: 'int' = 1, fontsize: 'int' = 8, markersize: 'int' = 6, title: 'str | None' = None, figsize: 'Tuple[float, float] | None' = None, show: 'bool' = True)`

**Docstring**  
Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.local_distortion_map`

**Signature**  
`(adata, *, basis: 'str' = 'umap', store_key: 'str' = 'local_geometry_stability', rep_a=None, rep_b=None, metric: 'str' = 'local_shape_distortion', aggregation: 'str' = 'median', node_key=None, title=None, show: 'bool' = True)`

**Docstring**  
Plot stored per-cell local distortion values on a display embedding.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.mapping_confidence_umap`

**Signature**  
`(adata, conf_key: 'str' = 'map_confidence', *, basis: 'str' = 'umap', highlight_low_k: 'Optional[int]' = 200, title: 'Optional[str]' = None, show: 'bool' = True)`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.mapping_qc_panel`

**Signature**  
`(adata, *, pred_key: 'str' = 'map_pred', conf_key: 'str' = 'map_confidence', ood_key: 'str' = 'map_ood_score', basis: 'str' = 'umap', show: 'bool' = True, palette_from: 'str | None' = None, condition_key: 'str | None' = None, query_value: 'str | None' = None, show_ref_as_grey: 'bool' = True, return_legend_data: 'bool' = False)`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.ood_cells`

**Signature**  
`(adata, ood_key: 'str' = 'map_ood_score', *, basis: 'str' = 'umap', threshold: 'Optional[float]' = None, show_only_flagged: 'bool' = False, title: 'Optional[str]' = None, show: 'bool' = True)`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.ood_landscape`

**Signature**  
`(adata, *, ood_key: 'str' = 'scgeo_ood', basis: 'str' = 'umap', threshold: 'Optional[float]' = None, show_only_flagged: 'bool' = False, flagged_outline: 'bool' = True, flagged_size: 'float' = 28.0, flagged_lw: 'float' = 0.8, bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.12, score_size: 'float' = 10.0, score_alpha: 'float' = 0.85, cmap: 'str' = 'magma', contour: 'bool' = True, contour_quantile: 'float' = 0.95, contour_levels: 'int' = 1, contour_color: 'str' = 'cyan', contour_lw: 'float' = 1.6, contour_alpha: 'float' = 0.95, contour_gridsize: 'int' = 150, groupby: 'Optional[str]' = None, top_n_groups: 'int' = 10, summary_kind: 'str' = 'flagged_frac', figsize: 'tuple[float, float]' = (10.5, 5.2), title: 'Optional[str]' = None, ax=None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a continuous OOD landscape on an embedding, with optional contour and

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.paga_composition_bar`

**Signature**  
`(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 15, sort_by: 'Optional[str]' = None, ax=None, figsize=(6.2, 4.2), title: 'str' = 'ScGeo: Top composition shifts', show: 'bool' = True)`

**Docstring**  
Bar plot of top_k nodes by evidence + effect.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.paga_composition_panel`

**Signature**  
`(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 10, figsize=(12, 4.2), show: 'bool' = True)`

**Docstring**  
1×2 panel: volcano + top-k bars.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.paga_composition_volcano`

**Signature**  
`(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 10, label: 'bool' = True, ax=None, figsize=(5.2, 4.2), title: 'str' = 'ScGeo: PAGA composition volcano', show: 'bool' = True)`

**Docstring**  
Volcano plot: x = signed effect (logOR/beta/effect), y = -log10(q or p).

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.paga_scgeo`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', pie_key: 'Optional[str]' = 'timepoint', velocity_basis: 'Optional[str]' = 'umap', show_velocity: 'bool' = True, node_color_mode: 'str' = 'delta', highlight_nodes: 'Optional[list[str]]' = None, **kwargs)`

**Docstring**  
ScGeo-style PAGA summary:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.paga_shift_map`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', min_cells: 'int' = 15, connectivity_threshold: 'float' = 0.05, agg: 'str' = 'mean', background_size: 'float' = 6.0, background_alpha: 'float' = 0.15, node_size: 'float' = 220.0, node_scale_by_n: 'bool' = True, edge_lw: 'float' = 2.0, edge_alpha: 'float' = 0.55, arrow_width: 'float' = 0.008, arrow_alpha: 'float' = 0.95, arrow_scale: 'float' = 1.0, label: 'bool' = True, label_top_n: 'Optional[int]' = None, label_fontsize: 'int' = 8, palette: 'Optional[dict[str, Any]]' = None, pie_key: 'Optional[str]' = None, pie_categories: 'Optional[list[str]]' = None, pie_palette: 'Optional[dict[str, Any]]' = None, pie_size_scale: 'float' = 1.0, velocity_basis: 'Optional[str]' = None, show_velocity: 'bool' = False, velocity_color: 'str' = 'cyan', velocity_scale: 'float' = 50.0, velocity_alpha: 'float' = 0.95, node_color_mode: 'str' = 'palette', alignment_df: 'Optional[pd.DataFrame]' = None, alignment_key: 'str' = 'alignment_cosine', delta_key: 'str' = 'delta_frac', constant_node_color: 'str' = 'gold', highlight_nodes: 'Optional[list[str]]' = None, highlight_edgecolor: 'str' = 'black', highlight_lw: 'float' = 2.0, ax=None, figsize: 'tuple[float, float]' = (8.0, 7.0), title: 'Optional[str]' = None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Overlay a PAGA graph anchored on group0 centroids in embedding space,

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.perturbation_report`

**Signature**  
`(adata, *, node_key, basis: 'str' = 'umap', report: 'Optional[pd.DataFrame]' = None, save_dir=None, prefix: 'str' = 'scgeo', comparison_label: 'Optional[str]' = None, local_k=None, pair_aggregation: 'str' = 'median', include_worst_case: 'bool' = True, show: 'bool' = True)`

**Docstring**  
Create the standard ScGeo perturbation report bundle.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.recovery_compass`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', velocity_basis: 'Optional[str]' = None, ood_key: 'Optional[str]' = None, min_cells: 'int' = 15, connectivity_threshold: 'float' = 0.05, node_size_mode: 'str' = 'group1_n', node_size_scale: 'float' = 380.0, fill_color_mode: 'str' = 'alignment', fill_cmap: 'str' = 'coolwarm', fill_vmin: 'float' = -1.0, fill_vmax: 'float' = 1.0, ring_mode: 'str' = 'ood_frac', ring_color: 'str' = 'gold', ring_max_lw: 'float' = 4.0, arrow_color_mode: 'str' = 'shift', arrow_color: 'str' = 'black', arrow_cmap: 'str' = 'magma', arrow_scale: 'float' = 1.0, arrow_width: 'float' = 0.008, edge_alpha: 'float' = 0.45, edge_lw: 'float' = 2.0, bg_size: 'float' = 5.0, bg_alpha: 'float' = 0.08, label: 'bool' = True, label_top_n: 'Optional[int]' = 12, label_fontsize: 'int' = 8, legend: 'bool' = True, title: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (9.5, 8.0), ax=None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Signature ScGeo synthesis plot combining:

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.representation_stability_heatmap`

**Signature**  
`(adata, *, store_key: 'str' = 'representation_stability', metric: 'str' = 'normalized_delta_norm', annotate_consensus: 'bool' = True, cluster_states: 'bool' = False, cluster_reps: 'bool' = False, figsize=None, title=None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a state-by-representation heatmap from stored representation stability.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.robustness_matrix`

**Signature**  
`(data: 'pd.DataFrame', *, row_key: 'str' = 'feature', col_key: 'str' = 'setting', value_key: 'str' = 'value', annot_key: 'Optional[str]' = None, row_order: 'Optional[Sequence[str]]' = None, col_order: 'Optional[Sequence[str]]' = None, sort_rows_by: 'Optional[str]' = None, ascending: 'bool' = False, summary: 'Optional[str]' = 'mean', pass_threshold: 'Optional[float]' = None, cmap: 'str' = 'viridis', vmin: 'Optional[float]' = None, vmax: 'Optional[float]' = None, center: 'Optional[float]' = None, show_values: 'bool' = True, value_fmt: 'str' = '.2f', annot_fontsize: 'int' = 8, na_color: 'str' = '#d9d9d9', grid_lw: 'float' = 0.8, grid_color: 'str' = 'white', cbar_label: 'Optional[str]' = None, summary_label: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (10.0, 6.0), title: 'Optional[str]' = None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a robustness heatmap with optional row-summary side bar.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.score_embedding`

**Signature**  
`(adata, score_key: 'str', basis: 'str' = 'umap', *, layer: 'Optional[str]' = None, ax: 'Optional[plt.Axes]' = None, title: 'Optional[str]' = None, size: 'float' = 6.0, alpha: 'float' = 0.8, cmap: 'str' = 'viridis', vmin=None, vmax=None, na_color: 'str' = 'lightgrey', figsize=None, show: 'bool' = True)`

**Docstring**  
Plot an obs score on an embedding (UMAP/PCA/etc) with minimal dependencies.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.score_umap`

**Signature**  
`(adata, score_key: 'str', **kwargs)`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.state_evidence_panel`

**Signature**  
`(report_or_adata, *, node_key=None, sort_by: 'str' = 'normalized_delta_norm', max_states: 'int' = 25, show_ci: 'bool' = True, show_coverage: 'bool' = True, figsize=None, title=None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a progressive-disclosure state evidence panel from a ScGeo report.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.state_flow_alluvial`

**Signature**  
`(adata, *, columns: 'Sequence[str]', min_count: 'int' = 1, drop_na: 'bool' = False, normalize: 'bool' = False, sort_categories: 'bool' = False, color_by: 'str' = 'target', alpha: 'float' = 0.7, column_gap: 'float' = 1.8, category_gap: 'float' = 0.02, ribbon_curve: 'float' = 0.35, figsize: 'tuple[float, float]' = (11, 6), title: 'Optional[str]' = None, palette: 'Optional[dict[str, tuple[float, float, float, float]]]' = None, ax=None, return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Draw an alluvial / ribbon plot for ordered categorical columns in `adata.obs`.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.state_flow_sankey`

**Signature**  
`(adata, *, columns: 'Sequence[str]', min_count: 'int' = 1, drop_na: 'bool' = False, title: 'Optional[str]' = None, pad: 'int' = 18, thickness: 'int' = 18, width: 'int' = 1000, height: 'int' = 550, arrangement: 'str' = 'snap', node_color: 'str' = 'rgba(120,120,120,0.85)', link_color: 'str' = 'rgba(120,120,120,0.28)', return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot a categorical state-flow Sankey diagram from columns in `adata.obs`.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl.velocity_shift_alignment`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, min_cells: 'int' = 15, agg: 'str' = 'mean', bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.1, node_size: 'float' = 180.0, shift_scale: 'float' = 1.0, velocity_scale: 'float' = 50.0, shift_color: 'str' = 'black', velocity_color: 'str' = 'cyan', shift_alpha: 'float' = 0.95, velocity_alpha: 'float' = 0.95, arrow_width: 'float' = 0.006, show_shift_arrow: 'bool' = True, show_velocity_arrow: 'bool' = True, color_by_alignment: 'bool' = True, alignment_cmap: 'str' = 'coolwarm', alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, palette: 'Optional[dict[str, Any]]' = None, label: 'bool' = True, label_top_n: 'Optional[int]' = None, label_mode: 'str' = 'shift', label_fontsize: 'int' = 8, title: 'Optional[str]' = None, ax=None, figsize: 'tuple[float, float]' = (8.2, 7.0), return_data: 'bool' = False, show: 'bool' = True)`

**Docstring**  
Plot node-wise observed shift vectors and mean velocity vectors on the same embedding.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---
