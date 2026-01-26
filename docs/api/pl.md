# scgeo.pl

## `scgeo.pl._fallback_palette`

**Signature**  
`(labels: 'np.ndarray')`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.pl._get_scanpy_palette`

**Signature**  
`(adata, groupby: 'str')`

**Docstring**  
Return Scanpy-style palette mapping {category -> color} if available.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

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
