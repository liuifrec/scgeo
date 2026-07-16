# scgeo.tl

## `scgeo.tl.align_vectors`

**Signature**  
`(adata, vec_key: 'str', ref_vec_key: 'Optional[str]' = None, *, ref_from_shift: 'bool' = False, shift_store_key: 'str' = 'scgeo', shift_kind: 'str' = 'shift', shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_align', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Cosine alignment between vectors.

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, vec_key: 'str', ref_vec_key: 'Optional[str]' = None, *, ref_from_shift: 'bool' = False, shift_store_key: 'str' = 'scgeo', shift_kind: 'str' = 'shift', shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_align', store_key: 'str' = 'scgeo') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.alignment_driver_genes`

**Signature**  
`(adata, *, alignment_key: 'str', group1: 'str' = 'discordant', group2: 'str' = 'aligned', subset_key: 'Optional[str]' = None, subset_values: 'Optional[Sequence[str]]' = None, layer: 'Optional[str]' = None, method: 'str' = 'wilcoxon', pts: 'bool' = True, min_cells: 'int' = 20, key_added: 'str' = 'alignment_driver_genes') -> 'pd.DataFrame'`

**Docstring**  
Identify genes associated with alignment or discordance between geometric shift

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, *, alignment_key: 'str', group1: 'str' = 'discordant', group2: 'str' = 'aligned', subset_key: 'Optional[str]' = None, subset_values: 'Optional[Sequence[str]]' = None, layer: 'Optional[str]' = None, method: 'str' = 'wilcoxon', pts: 'bool' = True, min_cells: 'int' = 20, key_added: 'str' = 'alignment_driver_genes') -> 'pd.DataFrame'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.analyze_shift`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, ood_key: 'Optional[str]' = None, ood_groupby: 'Optional[str]' = None, robustness: 'Optional[pd.DataFrame]' = None, min_cells: 'int' = 10, store_key: 'str' = 'shift', overwrite: 'bool' = True)`

**Docstring**  
High-level ScGeo analysis orchestrator.

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, ood_key: 'Optional[str]' = None, ood_groupby: 'Optional[str]' = None, robustness: 'Optional[pd.DataFrame]' = None, min_cells: 'int' = 10, store_key: 'str' = 'shift', overwrite: 'bool' = True)`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.consensus_subspace`

**Signature**  
`(adata: 'AnnData', rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'str | None' = None, group1: 'str | None' = None, sample_key: 'str | None' = None, n_components: 'int' = 2, obs_key_prefix: 'str' = 'cs', store_key: 'str' = 'consensus_subspace', min_cells: 'int' = 20, center: 'bool' = False) -> 'None'`

**Docstring**  
Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata: 'AnnData', rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'str | None' = None, group1: 'str | None' = None, sample_key: 'str | None' = None, n_components: 'int' = 2, obs_key_prefix: 'str' = 'cs', store_key: 'str' = 'consensus_subspace', min_cells: 'int' = 20, center: 'bool' = False) -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.density_overlap`

**Signature**  
`(adata, rep: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, k: 'int' = 30, eval_on: 'str' = 'union', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute density overlap between two conditions on an embedding using kNN density estimates.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, rep: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, k: 'int' = 30, eval_on: 'str' = 'union', store_key: 'str' = 'scgeo') -> 'None'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `dens_test`
  - touched: —
  - removed: —

---

## `scgeo.tl.distribution_test`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, sample_key: 'Optional[str]' = None, by: 'Optional[str]' = None, method: 'str' = 'energy', n_perm: 'int' = 500, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Distribution difference test with sample-aware permutation.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, sample_key: 'Optional[str]' = None, by: 'Optional[str]' = None, method: 'str' = 'energy', n_perm: 'int' = 500, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `dist_test`
  - touched: —
  - removed: —

---

## `scgeo.tl.local_geometry_stability`

**Signature**  
`(adata, *, reps, node_key=None, sample_key=None, k_values=(15, 30, 50), metric='euclidean', pair_mode='all', reference_rep=None, n_boot=500, max_exact_cells=3000, stratify_key=None, store_per_cell=False, seed=0, store_key='local_geometry_stability')`

**Docstring**  
Quantify local-geometry preservation across latent representations.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, *, reps, node_key=None, sample_key=None, k_values=(15, 30, 50), metric='euclidean', pair_mode='all', reference_rep=None, n_boot=500, max_exact_cells=3000, stratify_key=None, store_per_cell=False, seed=0, store_key='local_geometry_stability')`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `scgeo`
  - touched: —
  - removed: —

---

## `scgeo.tl.map_knn`

**Signature**  
`(adata_ref, adata_q, label_key: 'str' = 'cell_type', rep: 'str' = 'X_pca', k: 'int' = 25, ood_quantile: 'float' = 0.99, out_label_key: 'str' = 'scgeo_label', out_conf_key: 'str' = 'scgeo_confidence', out_ent_key: 'str' = 'scgeo_entropy', out_ood_key: 'str' = 'scgeo_ood', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
kNN mapping reference->query with QC:

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata_ref, adata_q, label_key: 'str' = 'cell_type', rep: 'str' = 'X_pca', k: 'int' = 25, ood_quantile: 'float' = 0.99, out_label_key: 'str' = 'scgeo_label', out_conf_key: 'str' = 'scgeo_confidence', out_ent_key: 'str' = 'scgeo_entropy', out_ood_key: 'str' = 'scgeo_ood', store_key: 'str' = 'scgeo') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.map_query_to_ref`

**Signature**  
`(adata, *, ref_key: 'str', ref_value: 'str', label_key: 'str', query_key: 'Optional[str]' = None, query_value: 'Optional[str]' = None, graph_key: 'str' = 'connectivities', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'connectivity_mass', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, rep: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Graph-native reference mapping + QC that is agnostic to how the graph was built

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, *, ref_key: 'str', ref_value: 'str', label_key: 'str', query_key: 'Optional[str]' = None, query_value: 'Optional[str]' = None, graph_key: 'str' = 'connectivities', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'connectivity_mass', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, rep: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.map_query_to_ref_pool`

**Signature**  
`(adata, pool: 'ReferencePool', *, rep: 'str', k: 'int' = 30, weight_method: 'str' = 'inv', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Embedding-only mapping using a ReferencePool (ANN index).

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, pool: 'ReferencePool', *, rep: 'str', k: 'int' = 30, weight_method: 'str' = 'inv', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.map_query_to_ref_pool_census`

**Signature**  
`(adata_q, *, pool=None, census=None, rep: 'str' = 'X_emb', embedding_name: 'Optional[str]' = None, organism: 'str' = 'homo_sapiens', label_key: 'str' = 'cell_type', obs_columns: 'Optional[Sequence[str]]' = None, k: 'int' = 50, max_refs: 'int' = 200000, dedup: 'bool' = True, index_metric: 'str' = 'euclidean', index_seed: 'int' = 0, census_obs_filter: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Phase B canonical spell:

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata_q, *, pool=None, census=None, rep: 'str' = 'X_emb', embedding_name: 'Optional[str]' = None, organism: 'str' = 'homo_sapiens', label_key: 'str' = 'cell_type', obs_columns: 'Optional[Sequence[str]]' = None, k: 'int' = 50, max_refs: 'int' = 200000, dedup: 'bool' = True, index_metric: 'str' = 'euclidean', index_seed: 'int' = 0, census_obs_filter: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.mixscore`

**Signature**  
`(adata, label_key: 'str' = 'batch', rep: 'str' = 'X_pca', k: 'int' = 50, use_connectivities: 'bool' = True, connectivities_key: 'str' = 'connectivities', obs_key: 'str' = 'scgeo_mixscore', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
kNN label mixing score in [0,1]:

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, label_key: 'str' = 'batch', rep: 'str' = 'X_pca', k: 'int' = 50, use_connectivities: 'bool' = True, connectivities_key: 'str' = 'connectivities', obs_key: 'str' = 'scgeo_mixscore', store_key: 'str' = 'scgeo') -> 'None'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: `scgeo_mixscore`
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `mix_test`
  - touched: —
  - removed: —

---

## `scgeo.tl.paga_composition_stats`

**Signature**  
`(adata, group_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', *, sample_key: 'Optional[str]' = None, method: 'str' = 'gee', n_boot: 'int' = 1000, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
For each cluster/node, test enrichment of condition1 vs condition0.

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, group_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', *, sample_key: 'Optional[str]' = None, method: 'str' = 'gee', n_boot: 'int' = 1000, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.projection_disagreement`

**Signature**  
`(adata, sources: 'Sequence[Dict[str, Any]]', obs_key: 'str' = 'scgeo_disagree', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute per-cell projection disagreement among multiple vector sources.

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, sources: 'Sequence[Dict[str, Any]]', obs_key: 'str' = 'scgeo_disagree', store_key: 'str' = 'scgeo') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.representation_stability`

**Signature**  
`(adata, *, reps, node_key, condition_key, group0, group1, sample_key=None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, velocity_keys=None, alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, min_cells: 'int' = 20, consensus_rules=None, seed: 'int' = 0, store_key: 'str' = 'representation_stability')`

**Docstring**  
Assess whether state-level perturbation geometry is stable across representations.

Consensus labels distinguish `stable_neutral`, magnitude-only `stable_effect`,
velocity-supported `stable_aligned`/`stable_discordant`,
`representation_unstable`, and `insufficient_coverage`.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, *, reps, node_key, condition_key, group0, group1, sample_key=None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, velocity_keys=None, alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, min_cells: 'int' = 20, consensus_rules=None, seed: 'int' = 0, store_key: 'str' = 'representation_stability')`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `scgeo`
  - touched: —
  - removed: —

---

## `scgeo.tl.robust_shift`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'Any' = None, group1: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, bootstrap_unit: 'str' = 'auto', normalize_by: 'Optional[str]' = 'pooled_robust_scale', seed: 'int' = 0, store_key: 'str' = 'robust_shift') -> 'Dict[str, Any]'`

**Docstring**  
Robust condition displacement with sample-aware bootstrap uncertainty.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'Any' = None, group1: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, center: 'str' = 'geometric_median', trim_fraction: 'float' = 0.1, n_boot: 'int' = 500, bootstrap_unit: 'str' = 'auto', normalize_by: 'Optional[str]' = 'pooled_robust_scale', seed: 'int' = 0, store_key: 'str' = 'robust_shift') -> 'Dict[str, Any]'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `scgeo`
  - touched: —
  - removed: —

---

## `scgeo.tl.shift`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute mean shift vector Δ = μ1 - μ0 in representation space.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, store_key: 'str' = 'scgeo') -> 'None'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `shift_test`
  - touched: —
  - removed: —

---

## `scgeo.tl.velocity_delta_alignment`

**Signature**  
`(adata, *, velocity_key: 'Optional[str]' = None, rep_for_shift: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_vel_delta_align', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Convenience wrapper for scVelo/CellRank workflows:

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, *, velocity_key: 'Optional[str]' = None, rep_for_shift: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_vel_delta_align', store_key: 'str' = 'scgeo') -> 'None'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.velocity_shift_alignment`

**Signature**  
`(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, min_cells: 'int' = 15, agg: 'str' = 'mean', alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, key_added: 'Optional[str]' = 'velocity_shift_alignment', propagate_to_obs: 'bool' = False) -> 'pd.DataFrame'`

**Docstring**  
Compute node-wise alignment between observed geometric shift and mean velocity.

### I/O contract

**Manifest status:** `skipped`
**Probed signature:** `(adata, *, node_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', basis: 'str' = 'umap', velocity_basis: 'Optional[str]' = None, min_cells: 'int' = 15, agg: 'str' = 'mean', alignment_pos_thr: 'float' = 0.3, alignment_neg_thr: 'float' = -0.3, key_added: 'Optional[str]' = 'velocity_shift_alignment', propagate_to_obs: 'bool' = False) -> 'pd.DataFrame'`
**Note:** needs domain-specific inputs

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: —
  - touched: —
  - removed: —

---

## `scgeo.tl.wasserstein`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, n_proj: 'int' = 128, p: 'int' = 2, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute (sliced) Wasserstein distance between two conditions in embedding space.

### I/O contract

**Manifest status:** `ok`
**Probed signature:** `(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, n_proj: 'int' = 128, p: 'int' = 2, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Writes / touches (key-level)**
- `obs_cols`:
  - added: —
  - touched: `batch`, `cluster`, `condition`
  - removed: —
- `obsm_keys`:
  - added: —
  - touched: `X_pca`, `X_umap`
  - removed: —
- `layers_keys`:
  - added: —
  - touched: —
  - removed: —
- `uns_keys`:
  - added: `dist_test`
  - touched: —
  - removed: —

---
