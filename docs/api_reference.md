# API Reference

This document is generated from `api_manifest.json` and `scgeo_io_manifest.json` to reflect the current public API only.

It is the canonical API documentation source in this repository.

Manual contract supplements are no longer maintained; uncertain entries are tracked directly in this document.

## Uncertain entries

Entries marked **Uncertain** are listed in `scgeo_io_manifest.json` under `skipped` (normalized I/O could not be fully observed).

## scgeo.tl

### `scgeo.tl.align_vectors`

- Full name: `scgeo.tl.align_vectors`
- Signature: `(adata, vec_key: 'str', ref_vec_key: 'Optional[str]' = None, *, ref_from_shift: 'bool' = False, shift_store_key: 'str' = 'scgeo', shift_kind: 'str' = 'shift', shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_align', store_key: 'str' = 'scgeo') -> 'None'`
- Description: Cosine alignment between vectors.
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.alignment_driver_genes`

- Full name: `scgeo.tl.alignment_driver_genes`
- Signature: `(adata, *, alignment_key: 'str', group1: 'str' = 'discordant', group2: 'str' = 'aligned', subset_key: 'Optional[str]' = None, subset_values: 'Optional[Sequence[str]]' = None, layer: 'Optional[str]' = None, method: 'str' = 'wilcoxon', pts: 'bool' = True, min_cells: 'int' = 20, key_added: 'str' = 'alignment_driver_genes') -> 'pd.DataFrame'`
- Description: Identify genes associated with alignment or discordance between geometric shift
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.analyze_shift`

- Full name: `scgeo.tl.analyze_shift`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, ood_key: 'Optional[str]' = None, ood_groupby: 'Optional[str]' = None, robustness: 'Optional[pd.DataFrame]' = None, min_cells: 'int' = 10, store_key: 'str' = 'shift', overwrite: 'bool' = True)`
- Description: High-level ScGeo analysis orchestrator.
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.consensus_subspace`

- Full name: `scgeo.tl.consensus_subspace`
- Signature: `(adata: 'AnnData', rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'str | None' = None, group1: 'str | None' = None, sample_key: 'str | None' = None, n_components: 'int' = 2, obs_key_prefix: 'str' = 'cs', store_key: 'str' = 'consensus_subspace', min_cells: 'int' = 20, center: 'bool' = False) -> 'None'`
- Description: Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.density_overlap`

- Full name: `scgeo.tl.density_overlap`
- Signature: `(adata, rep: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, k: 'int' = 30, eval_on: 'str' = 'union', store_key: 'str' = 'scgeo') -> 'None'`
- Description: Compute density overlap between two conditions on an embedding using kNN density estimates.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`dens_test`], touched [none], removed [none]

### `scgeo.tl.distribution_test`

- Full name: `scgeo.tl.distribution_test`
- Signature: `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, sample_key: 'Optional[str]' = None, by: 'Optional[str]' = None, method: 'str' = 'energy', n_perm: 'int' = 500, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`
- Description: Distribution difference test with sample-aware permutation.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`dist_test`], touched [none], removed [none]

### `scgeo.tl.map_knn`

- Full name: `scgeo.tl.map_knn`
- Signature: `(adata_ref, adata_q, label_key: 'str' = 'cell_type', rep: 'str' = 'X_pca', k: 'int' = 25, ood_quantile: 'float' = 0.99, out_label_key: 'str' = 'scgeo_label', out_conf_key: 'str' = 'scgeo_confidence', out_ent_key: 'str' = 'scgeo_entropy', out_ood_key: 'str' = 'scgeo_ood', store_key: 'str' = 'scgeo') -> 'None'`
- Description: kNN mapping reference->query with QC:
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.map_query_to_ref`

- Full name: `scgeo.tl.map_query_to_ref`
- Signature: `(adata, *, ref_key: 'str', ref_value: 'str', label_key: 'str', query_key: 'Optional[str]' = None, query_value: 'Optional[str]' = None, graph_key: 'str' = 'connectivities', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'connectivity_mass', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, rep: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
- Description: Graph-native reference mapping + QC that is agnostic to how the graph was built
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.map_query_to_ref_pool`

- Full name: `scgeo.tl.map_query_to_ref_pool`
- Signature: `(adata, pool: 'ReferencePool', *, rep: 'str', k: 'int' = 30, weight_method: 'str' = 'inv', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
- Description: Embedding-only mapping using a ReferencePool (ANN index).
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.map_query_to_ref_pool_census`

- Full name: `scgeo.tl.map_query_to_ref_pool_census`
- Signature: `(adata_q, *, pool=None, census=None, rep: 'str' = 'X_emb', embedding_name: 'Optional[str]' = None, organism: 'str' = 'homo_sapiens', label_key: 'str' = 'cell_type', obs_columns: 'Optional[Sequence[str]]' = None, k: 'int' = 50, max_refs: 'int' = 200000, dedup: 'bool' = True, index_metric: 'str' = 'euclidean', index_seed: 'int' = 0, census_obs_filter: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
- Description: Phase B canonical spell:
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.mixscore`

- Full name: `scgeo.tl.mixscore`
- Signature: `(adata, label_key: 'str' = 'batch', rep: 'str' = 'X_pca', k: 'int' = 50, use_connectivities: 'bool' = True, connectivities_key: 'str' = 'connectivities', obs_key: 'str' = 'scgeo_mixscore', store_key: 'str' = 'scgeo') -> 'None'`
- Description: kNN label mixing score in [0,1]:
- Uncertain: no
- Normalized I/O:
  - obs cols: added [`scgeo_mixscore`], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`mix_test`], touched [none], removed [none]

### `scgeo.tl.paga_composition_stats`

- Full name: `scgeo.tl.paga_composition_stats`
- Signature: `(adata, group_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', *, sample_key: 'Optional[str]' = None, method: 'str' = 'gee', n_boot: 'int' = 1000, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`
- Description: For each cluster/node, test enrichment of condition1 vs condition0.
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.projection_disagreement`

- Full name: `scgeo.tl.projection_disagreement`
- Signature: `(adata, sources: 'Sequence[Dict[str, Any]]', obs_key: 'str' = 'scgeo_disagree', store_key: 'str' = 'scgeo') -> 'None'`
- Description: Compute per-cell projection disagreement among multiple vector sources.
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.representation_stability`

- Full name: `scgeo.tl.representation_stability`
- Signature: `(adata, *, reps, node_key, condition_key, group0, group1, sample_key=None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, velocity_keys=None, alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, min_cells: 'int' = 20, seed: 'int' = 0, store_key: 'str' = 'representation_stability')`
- Description: Assess whether state-level perturbation geometry is stable across representations.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`scgeo`], touched [none], removed [none]

### `scgeo.tl.robust_shift`

- Full name: `scgeo.tl.robust_shift`
- Signature: `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'Any' = None, group1: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, bootstrap_unit: 'str' = 'auto', normalize_by: 'Optional[str]' = 'pooled_robust_scale', seed: 'int' = 0, store_key: 'str' = 'robust_shift') -> 'Dict[str, Any]'`
- Description: Robust condition displacement with sample-aware bootstrap uncertainty.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`scgeo`], touched [none], removed [none]

### `scgeo.tl.shift`

- Full name: `scgeo.tl.shift`
- Signature: `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, store_key: 'str' = 'scgeo') -> 'None'`
- Description: Compute mean shift vector Δ = μ1 - μ0 in representation space.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`shift_test`], touched [none], removed [none]

### `scgeo.tl.velocity_delta_alignment`

- Full name: `scgeo.tl.velocity_delta_alignment`
- Signature: `(adata, *, velocity_key: 'Optional[str]' = None, rep_for_shift: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_vel_delta_align', store_key: 'str' = 'scgeo') -> 'None'`
- Description: Convenience wrapper for scVelo/CellRank workflows:
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.velocity_shift_alignment`

- Full name: `scgeo.tl.velocity_shift_alignment`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, min_cells: 'int' = 15, agg: 'str' = 'mean', alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, key_added: 'Optional[str]' = 'velocity_shift_alignment', propagate_to_obs: 'bool' = False) -> 'pd.DataFrame'`
- Description: Compute node-wise alignment between observed geometric shift and mean velocity.
- Uncertain: **yes**
- Normalized I/O: **uncertain** — entry is listed under `skipped` in `scgeo_io_manifest.json` (needs domain-specific inputs).

### `scgeo.tl.wasserstein`

- Full name: `scgeo.tl.wasserstein`
- Signature: `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, n_proj: 'int' = 128, p: 'int' = 2, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`
- Description: Compute (sliced) Wasserstein distance between two conditions in embedding space.
- Uncertain: no
- Normalized I/O:
  - obs cols: added [none], touched [`batch`, `cluster`, `condition`], removed [none]
  - obsm keys: added [none], touched [`X_pca`, `X_umap`], removed [none]
  - layers keys: added [none], touched [none], removed [none]
  - uns keys: added [`dist_test`], touched [none], removed [none]

## scgeo.pl

### `scgeo.pl.alignment_panel`

- Full name: `scgeo.pl.alignment_panel`
- Signature: `(adata, score_key: 'str', *, basis: 'str' = 'umap', topk: 'Optional[int]' = 200, cmap: 'str' = 'viridis', figsize=(11, 5), title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: 1×2 panel:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.ambiguity_panel`

- Full name: `scgeo.pl.ambiguity_panel`
- Signature: `(adata, score_key: 'str', *, basis: 'str' = 'umap', topk: 'Optional[int]' = 200, cmap: 'str' = 'inferno', figsize=(11, 5), title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: 1×2 panel:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.composition_drift`

- Full name: `scgeo.pl.composition_drift`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', agg: 'str' = 'mean', bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.1, centroid_size: 'float' = 320.0, centroid_scale_by_n: 'bool' = True, centroid_edgecolor: 'str' = 'white', centroid_lw: 'float' = 1.0, drift_cmap: 'str' = 'coolwarm', drift_vmax: 'Optional[float]' = None, bar_alpha: 'float' = 0.9, top_n: 'Optional[int]' = None, sort_by: 'str' = 'abs_delta_frac', palette: 'Optional[dict[str, Any]]' = None, title: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (14.0, 5.2), return_data: 'bool' = False, show: 'bool' = True)`
- Description: Plot a 3-panel composition drift report:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.consensus_structure`

- Full name: `scgeo.pl.consensus_structure`
- Signature: `(adata, *, score_key: 'str' = 'cs_score', basis: 'str' = 'umap', groupby: 'str' = 'louvain', topk: 'int' = 300, bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.25, hi_size: 'float' = 30.0, hi_alpha: 'float' = 0.95, cmap: 'str' = 'viridis', outline_lw: 'float' = 1.6, outline_alpha: 'float' = 0.95, use_scanpy_colors: 'bool' = True, min_frac_in_legend: 'float' = 0.0, title: 'Optional[str]' = None, figsize=(6.5, 5.5), ax=None, show: 'bool' = True, print_legend: 'bool' = True) -> 'tuple[Any, Any, ConsensusStructureStats, List[Dict[str, Any]]]'`
- Description: Consensus structure (UMAP only) + legend returned separately.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.consensus_subspace_panel`

- Full name: `scgeo.pl.consensus_subspace_panel`
- Signature: `(adata, *, score_key: 'str' = 'cs_score', basis: 'str' = 'umap', store_key: 'str' = 'consensus_subspace', topk: 'Optional[int]' = 200, figsize=(11, 5), show: 'bool' = True)`
- Description: Panel for consensus subspace:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.delta_rank`

- Full name: `scgeo.pl.delta_rank`
- Signature: `(adata, store_key: 'str' = 'scgeo', kind: 'str' = 'shift', level: 'str' = 'by', *, top_n: 'int | None' = None, rotate_xticks: 'int' = 60)`
- Description: Rank groups by delta magnitude (||Δ||) and plot with readable x tick labels.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.density_overlap`

- Full name: `scgeo.pl.density_overlap`
- Signature: `(adata, *, store_key: 'str' = 'density_overlap', level: 'str' = 'by', metrics: 'Tuple[str, ...]' = ('bc', 'hellinger'), top_k: 'Optional[int]' = 10, highlight: 'str' = 'worst', sort_by: 'Optional[str]' = None, style: "Literal['bar', 'lollipop']" = 'lollipop', annotate: 'bool' = True, grid: 'bool' = True, figsize=None, show: 'bool' = True)`
- Description: Plot density overlap summaries from ``sg.tl.density_overlap``.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.density_overlap_grid`

- Full name: `scgeo.pl.density_overlap_grid`
- Signature: `(adata, *, key: 'str' = 'density_overlap', basis: 'str' = 'umap', panel: 'str' = 'overlap', ax=None, figsize=(5.5, 5.0), cmap: 'str' = 'magma', alpha_points: 'float' = 0.05, show_points: 'bool' = True, title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: Visualize density overlap results from `tl.density_overlap`.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.distribution_report`

- Full name: `scgeo.pl.distribution_report`
- Signature: `(adata, *, basis: 'str' = 'umap', condition_key: 'str' = 'condition', group0: 'Optional[str]' = None, group1: 'Optional[str]' = None, density_key: 'str' = 'density_overlap', test_key: 'str' = 'distribution_test', score_key: 'Optional[str]' = None, top_k: 'int' = 8, figsize=(11, 8), show: 'bool' = True)`
- Description: 2×2 report panel for cross-condition embedding comparison.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.distribution_test`

- Full name: `scgeo.pl.distribution_test`
- Signature: `(adata, *, store_key: 'str' = 'distribution_test', level: 'str' = 'by', value: 'str' = 'p_perm', stat_key: 'str' = 'stat', top_k: 'Optional[int]' = 10, highlight: 'str' = 'strongest', figsize=None, show: 'bool' = True)`
- Description: Plot distribution test summaries from `sg.tl.distribution_test`.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.distribution_test_volcano`

- Full name: `scgeo.pl.distribution_test_volcano`
- Signature: `(adata, *, key: 'str' = 'distribution_test', effect_col: 'str' = 'effect', p_col: 'str' = 'p_adj', label_col: 'str' = 'group', top_n_labels: 'int' = 10, ax=None, figsize=(6.0, 5.0), title: 'str | None' = None, show: 'bool' = True)`
- Description: Volcano-like plot for distribution/comparison results.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.embedding_density`

- Full name: `scgeo.pl.embedding_density`
- Signature: `(adata, groupby: 'str', *, basis: 'str' = 'umap', groups: 'Optional[Iterable[str]]' = None, gridsize: 'int' = 160, normalize: 'str' = 'per_group', cmap: 'str' = 'magma', contour: 'bool' = False, contour_levels: 'int' = 12, imshow_alpha: 'float' = 0.85, transparent_background: 'bool' = True, mask_zeros: 'bool' = True, background: 'Optional[str]' = None, figsize=None, show: 'bool' = True, smooth_k: 'int' = 5, log1p: 'bool' = True, contour_lines: 'bool' = True, contour_linewidth: 'float' = 0.7, contour_alpha: 'float' = 0.95, contour_level_mode: 'str' = 'quantile')`
- Description: Plot embedding density per group (small multiples).
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.gallery_overview`

- Full name: `scgeo.pl.gallery_overview`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', ood_key: 'str' = 'scgeo_ood', velocity_basis: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (16.0, 12.0), title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: Render a 2x2 overview gallery of core ScGeo plots:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.highlight_topk_cells`

- Full name: `scgeo.pl.highlight_topk_cells`
- Signature: `(adata, score_key: 'str', basis: 'str' = 'umap', *, topk: 'int' = 300, ax=None, figsize=(5.8, 5.2), bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.2, hi_size: 'float' = 30.0, hi_alpha: 'float' = 0.95, cmap: 'str' = 'viridis', title: 'str | None' = None, show: 'bool' = True, groupby: 'str | None' = None, use_scanpy_colors: 'bool' = True, outline_topk: 'bool' = False, outline_lw: 'float' = 1.6, outline_alpha: 'float' = 0.95, add_colorbar: 'bool' = True)`
- Description: Highlight top-k cells by a score on an embedding.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.legend_from_data`

- Full name: `scgeo.pl.legend_from_data`
- Signature: `(legend_data: 'Iterable[Dict[str, Any]]', *, max_items: 'int' = 20, ncol: 'int' = 1, fontsize: 'int' = 8, markersize: 'int' = 6, title: 'str | None' = None, figsize: 'Tuple[float, float] | None' = None, show: 'bool' = True)`
- Description: Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.mapping_confidence_umap`

- Full name: `scgeo.pl.mapping_confidence_umap`
- Signature: `(adata, conf_key: 'str' = 'map_confidence', *, basis: 'str' = 'umap', highlight_low_k: 'Optional[int]' = 200, title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: _No docstring available._
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.mapping_qc_panel`

- Full name: `scgeo.pl.mapping_qc_panel`
- Signature: `(adata, *, pred_key: 'str' = 'map_pred', conf_key: 'str' = 'map_confidence', ood_key: 'str' = 'map_ood_score', basis: 'str' = 'umap', show: 'bool' = True, palette_from: 'str | None' = None, condition_key: 'str | None' = None, query_value: 'str | None' = None, show_ref_as_grey: 'bool' = True, return_legend_data: 'bool' = False)`
- Description: _No docstring available._
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.ood_cells`

- Full name: `scgeo.pl.ood_cells`
- Signature: `(adata, ood_key: 'str' = 'map_ood_score', *, basis: 'str' = 'umap', threshold: 'Optional[float]' = None, show_only_flagged: 'bool' = False, title: 'Optional[str]' = None, show: 'bool' = True)`
- Description: _No docstring available._
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.ood_landscape`

- Full name: `scgeo.pl.ood_landscape`
- Signature: `(adata, *, ood_key: 'str' = 'scgeo_ood', basis: 'str' = 'umap', threshold: 'Optional[float]' = None, show_only_flagged: 'bool' = False, flagged_outline: 'bool' = True, flagged_size: 'float' = 28.0, flagged_lw: 'float' = 0.8, bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.12, score_size: 'float' = 10.0, score_alpha: 'float' = 0.85, cmap: 'str' = 'magma', contour: 'bool' = True, contour_quantile: 'float' = 0.95, contour_levels: 'int' = 1, contour_color: 'str' = 'cyan', contour_lw: 'float' = 1.6, contour_alpha: 'float' = 0.95, contour_gridsize: 'int' = 150, groupby: 'Optional[str]' = None, top_n_groups: 'int' = 10, summary_kind: 'str' = 'flagged_frac', figsize: 'tuple[float, float]' = (10.5, 5.2), title: 'Optional[str]' = None, ax=None, return_data: 'bool' = False, show: 'bool' = True)`
- Description: Plot a continuous OOD landscape on an embedding, with optional contour and
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.paga_composition_bar`

- Full name: `scgeo.pl.paga_composition_bar`
- Signature: `(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 15, sort_by: 'Optional[str]' = None, ax=None, figsize=(6.2, 4.2), title: 'str' = 'ScGeo: Top composition shifts', show: 'bool' = True)`
- Description: Bar plot of top_k nodes by evidence + effect.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.paga_composition_panel`

- Full name: `scgeo.pl.paga_composition_panel`
- Signature: `(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 10, figsize=(12, 4.2), show: 'bool' = True)`
- Description: 1×2 panel: volcano + top-k bars.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.paga_composition_volcano`

- Full name: `scgeo.pl.paga_composition_volcano`
- Signature: `(adata, *, store_key: 'str' = 'scgeo', kind: 'str' = 'paga_composition_stats', effect: 'Optional[str]' = None, p_col: 'Optional[str]' = None, top_k: 'int' = 10, label: 'bool' = True, ax=None, figsize=(5.2, 4.2), title: 'str' = 'ScGeo: PAGA composition volcano', show: 'bool' = True)`
- Description: Volcano plot: x = signed effect (logOR/beta/effect), y = -log10(q or p).
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.paga_scgeo`

- Full name: `scgeo.pl.paga_scgeo`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', pie_key: 'Optional[str]' = 'timepoint', velocity_basis: 'Optional[str]' = 'umap', show_velocity: 'bool' = True, node_color_mode: 'str' = 'delta', highlight_nodes: 'Optional[list[str]]' = None, **kwargs)`
- Description: ScGeo-style PAGA summary:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.paga_shift_map`

- Full name: `scgeo.pl.paga_shift_map`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', min_cells: 'int' = 15, connectivity_threshold: 'float' = 0.05, agg: 'str' = 'mean', background_size: 'float' = 6.0, background_alpha: 'float' = 0.15, node_size: 'float' = 220.0, node_scale_by_n: 'bool' = True, edge_lw: 'float' = 2.0, edge_alpha: 'float' = 0.55, arrow_width: 'float' = 0.008, arrow_alpha: 'float' = 0.95, arrow_scale: 'float' = 1.0, label: 'bool' = True, label_top_n: 'Optional[int]' = None, label_fontsize: 'int' = 8, palette: 'Optional[dict[str, Any]]' = None, pie_key: 'Optional[str]' = None, pie_categories: 'Optional[list[str]]' = None, pie_palette: 'Optional[dict[str, Any]]' = None, pie_size_scale: 'float' = 1.0, velocity_basis: 'Optional[str]' = None, show_velocity: 'bool' = False, velocity_color: 'str' = 'cyan', velocity_scale: 'float' = 50.0, velocity_alpha: 'float' = 0.95, node_color_mode: 'str' = 'palette', alignment_df: 'Optional[pd.DataFrame]' = None, alignment_key: 'str' = 'alignment_cosine', delta_key: 'str' = 'delta_frac', constant_node_color: 'str' = 'gold', highlight_nodes: 'Optional[list[str]]' = None, highlight_edgecolor: 'str' = 'black', highlight_lw: 'float' = 2.0, ax=None, figsize: 'tuple[float, float]' = (8.0, 7.0), title: 'Optional[str]' = None, return_data: 'bool' = False, show: 'bool' = True)`
- Description: Overlay a PAGA graph anchored on group0 centroids in embedding space,
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.recovery_compass`

- Full name: `scgeo.pl.recovery_compass`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', paga_key: 'str' = 'paga', velocity_basis: 'Optional[str]' = None, ood_key: 'Optional[str]' = None, min_cells: 'int' = 15, connectivity_threshold: 'float' = 0.05, node_size_mode: 'str' = 'group1_n', node_size_scale: 'float' = 380.0, fill_color_mode: 'str' = 'alignment', fill_cmap: 'str' = 'coolwarm', fill_vmin: 'float' = -1.0, fill_vmax: 'float' = 1.0, ring_mode: 'str' = 'ood_frac', ring_color: 'str' = 'gold', ring_max_lw: 'float' = 4.0, arrow_color_mode: 'str' = 'shift', arrow_color: 'str' = 'black', arrow_cmap: 'str' = 'magma', arrow_scale: 'float' = 1.0, arrow_width: 'float' = 0.008, edge_alpha: 'float' = 0.45, edge_lw: 'float' = 2.0, bg_size: 'float' = 5.0, bg_alpha: 'float' = 0.08, label: 'bool' = True, label_top_n: 'Optional[int]' = 12, label_fontsize: 'int' = 8, legend: 'bool' = True, title: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (9.5, 8.0), ax=None, return_data: 'bool' = False, show: 'bool' = True)`
- Description: Signature ScGeo synthesis plot combining:
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.robustness_matrix`

- Full name: `scgeo.pl.robustness_matrix`
- Signature: `(data: 'pd.DataFrame', *, row_key: 'str' = 'feature', col_key: 'str' = 'setting', value_key: 'str' = 'value', annot_key: 'Optional[str]' = None, row_order: 'Optional[Sequence[str]]' = None, col_order: 'Optional[Sequence[str]]' = None, sort_rows_by: 'Optional[str]' = None, ascending: 'bool' = False, summary: 'Optional[str]' = 'mean', pass_threshold: 'Optional[float]' = None, cmap: 'str' = 'viridis', vmin: 'Optional[float]' = None, vmax: 'Optional[float]' = None, center: 'Optional[float]' = None, show_values: 'bool' = True, value_fmt: 'str' = '.2f', annot_fontsize: 'int' = 8, na_color: 'str' = '#d9d9d9', grid_lw: 'float' = 0.8, grid_color: 'str' = 'white', cbar_label: 'Optional[str]' = None, summary_label: 'Optional[str]' = None, figsize: 'tuple[float, float]' = (10.0, 6.0), title: 'Optional[str]' = None, return_data: 'bool' = False, show: 'bool' = True)`
- Description: Plot a robustness heatmap with optional row-summary side bar.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.score_embedding`

- Full name: `scgeo.pl.score_embedding`
- Signature: `(adata, score_key: 'str', basis: 'str' = 'umap', *, layer: 'Optional[str]' = None, ax: 'Optional[plt.Axes]' = None, title: 'Optional[str]' = None, size: 'float' = 6.0, alpha: 'float' = 0.8, cmap: 'str' = 'viridis', vmin=None, vmax=None, na_color: 'str' = 'lightgrey', figsize=None, show: 'bool' = True)`
- Description: Plot an obs score on an embedding (UMAP/PCA/etc) with minimal dependencies.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.score_umap`

- Full name: `scgeo.pl.score_umap`
- Signature: `(adata, score_key: 'str', **kwargs)`
- Description: _No docstring available._
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.state_flow_alluvial`

- Full name: `scgeo.pl.state_flow_alluvial`
- Signature: `(adata, *, columns: 'Sequence[str]', min_count: 'int' = 1, drop_na: 'bool' = False, normalize: 'bool' = False, sort_categories: 'bool' = False, color_by: 'str' = 'target', alpha: 'float' = 0.7, column_gap: 'float' = 1.8, category_gap: 'float' = 0.02, ribbon_curve: 'float' = 0.35, figsize: 'tuple[float, float]' = (11, 6), title: 'Optional[str]' = None, palette: 'Optional[dict[str, tuple[float, float, float, float]]]' = None, ax=None, return_data: 'bool' = False, show: 'bool' = True)`
- Description: Draw an alluvial / ribbon plot for ordered categorical columns in `adata.obs`.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.state_flow_sankey`

- Full name: `scgeo.pl.state_flow_sankey`
- Signature: `(adata, *, columns: 'Sequence[str]', min_count: 'int' = 1, drop_na: 'bool' = False, title: 'Optional[str]' = None, pad: 'int' = 18, thickness: 'int' = 18, width: 'int' = 1000, height: 'int' = 550, arrangement: 'str' = 'snap', node_color: 'str' = 'rgba(120,120,120,0.85)', link_color: 'str' = 'rgba(120,120,120,0.28)', return_data: 'bool' = False, show: 'bool' = True)`
- Description: Plot a categorical state-flow Sankey diagram from columns in `adata.obs`.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).

### `scgeo.pl.velocity_shift_alignment`

- Full name: `scgeo.pl.velocity_shift_alignment`
- Signature: `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, min_cells: 'int' = 15, agg: 'str' = 'mean', bg_size: 'float' = 6.0, bg_alpha: 'float' = 0.1, node_size: 'float' = 180.0, shift_scale: 'float' = 1.0, velocity_scale: 'float' = 50.0, shift_color: 'str' = 'black', velocity_color: 'str' = 'cyan', shift_alpha: 'float' = 0.95, velocity_alpha: 'float' = 0.95, arrow_width: 'float' = 0.006, show_shift_arrow: 'bool' = True, show_velocity_arrow: 'bool' = True, color_by_alignment: 'bool' = True, alignment_cmap: 'str' = 'coolwarm', alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, palette: 'Optional[dict[str, Any]]' = None, label: 'bool' = True, label_top_n: 'Optional[int]' = None, label_mode: 'str' = 'shift', label_fontsize: 'int' = 8, title: 'Optional[str]' = None, ax=None, figsize: 'tuple[float, float]' = (8.2, 7.0), return_data: 'bool' = False, show: 'bool' = True)`
- Description: Plot node-wise observed shift vectors and mean velocity vectors on the same embedding.
- Uncertain: no
- Normalized I/O: not available in `scgeo_io_manifest.json` (this manifest currently covers `scgeo.tl` only).
