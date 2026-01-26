# scgeo.tl

> Auto-generated from `api_manifest.json` + `scgeo_io_manifest.json`.
> If anything here is wrong, fix the manifests or the generator—not the generated files.

## `scgeo.tl.align_vectors`

**Signature**  
`(adata, vec_key: 'str', ref_vec_key: 'Optional[str]' = None, *, ref_from_shift: 'bool' = False, shift_store_key: 'str' = 'scgeo', shift_kind: 'str' = 'shift', shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_align', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Cosine alignment between vectors.

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.consensus_subspace`

**Signature**  
`(adata: 'AnnData', rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group0: 'str | None' = None, group1: 'str | None' = None, sample_key: 'str | None' = None, n_components: 'int' = 2, obs_key_prefix: 'str' = 'cs', store_key: 'str' = 'consensus_subspace', min_cells: 'int' = 20, center: 'bool' = False) -> 'None'`

**Docstring**  
Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).

### I/O contract
**Writes**
- `obs_cols.cs_score` (unknown) — auto-detected
- `obsm_keys.X_cs` (unknown) — auto-detected
- `uns_keys.scgeo` (unknown) — auto-detected

---

## `scgeo.tl.density_overlap`

**Signature**  
`(adata, rep: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, k: 'int' = 30, eval_on: 'str' = 'union', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute density overlap between two conditions on an embedding using kNN density estimates.

### I/O contract
**Writes**
- `uns_keys.scgeo` (unknown) — auto-detected

---

## `scgeo.tl.distribution_test`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, sample_key: 'Optional[str]' = None, by: 'Optional[str]' = None, method: 'str' = 'energy', n_perm: 'int' = 500, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Distribution difference test with sample-aware permutation.

### I/O contract
**Writes**
- `uns_keys.scgeo` (unknown) — auto-detected

---

## `scgeo.tl.map_knn`

**Signature**  
`(adata_ref, adata_q, label_key: 'str' = 'cell_type', rep: 'str' = 'X_pca', k: 'int' = 25, ood_quantile: 'float' = 0.99, out_label_key: 'str' = 'scgeo_label', out_conf_key: 'str' = 'scgeo_confidence', out_ent_key: 'str' = 'scgeo_entropy', out_ood_key: 'str' = 'scgeo_ood', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
kNN mapping reference->query with QC:

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.map_query_to_ref`

**Signature**  
`(adata, *, ref_key: 'str', ref_value: 'str', label_key: 'str', query_key: 'Optional[str]' = None, query_value: 'Optional[str]' = None, graph_key: 'str' = 'connectivities', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'connectivity_mass', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, rep: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Graph-native reference mapping + QC that is agnostic to how the graph was built

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.map_query_to_ref_pool`

**Signature**  
`(adata, pool: 'ReferencePool', *, rep: 'str', k: 'int' = 30, weight_method: 'str' = 'inv', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Embedding-only mapping using a ReferencePool (ANN index).

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.map_query_to_ref_pool_census`

**Signature**  
`(adata_q, *, pool=None, census=None, rep: 'str' = 'X_emb', embedding_name: 'Optional[str]' = None, organism: 'str' = 'homo_sapiens', label_key: 'str' = 'cell_type', obs_columns: 'Optional[Sequence[str]]' = None, k: 'int' = 50, max_refs: 'int' = 200000, dedup: 'bool' = True, index_metric: 'str' = 'euclidean', index_seed: 'int' = 0, census_obs_filter: 'Optional[str]' = None, store_key: 'str' = 'map_query_to_ref', pred_key: 'str' = 'scgeo_pred', conf_key: 'str' = 'scgeo_conf', conf_entropy_key: 'str' = 'scgeo_conf_entropy', conf_margin_key: 'str' = 'scgeo_conf_margin', ood_key: 'str' = 'scgeo_ood', reject_key: 'str' = 'scgeo_reject', conf_method: 'str' = 'entropy_margin', ood_method: 'str' = 'distance', reject_conf: 'Optional[float]' = None, reject_ood: 'Optional[float]' = None, return_probs: 'bool' = False, probs_key: 'str' = 'X_map_probs', label_order_key: 'str' = 'map_label_order') -> 'None'`

**Docstring**  
Phase B canonical spell:

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.mixscore`

**Signature**  
`(adata, label_key: 'str' = 'batch', rep: 'str' = 'X_pca', k: 'int' = 50, use_connectivities: 'bool' = True, connectivities_key: 'str' = 'connectivities', obs_key: 'str' = 'scgeo_mixscore', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
kNN label mixing score in [0,1]:

### I/O contract
**Writes**
- `obs_cols.scgeo_mixscore` (unknown) — auto-detected
- `uns_keys.scgeo` (unknown) — auto-detected

---

## `scgeo.tl.paga_composition_stats`

**Signature**  
`(adata, group_key: 'str', condition_key: 'str', group0: 'Any', group1: 'Any', *, sample_key: 'Optional[str]' = None, method: 'str' = 'gee', n_boot: 'int' = 1000, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
For each cluster/node, test enrichment of condition1 vs condition0.

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.projection_disagreement`

**Signature**  
`(adata, sources: 'Sequence[Dict[str, Any]]', obs_key: 'str' = 'scgeo_disagree', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute per-cell projection disagreement among multiple vector sources.

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.shift`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute mean shift vector Δ = μ1 - μ0 in representation space.

### I/O contract
**Writes**
- `uns_keys.scgeo` (unknown) — auto-detected

---

## `scgeo.tl.velocity_delta_alignment`

**Signature**  
`(adata, *, velocity_key: 'Optional[str]' = None, rep_for_shift: 'str' = 'X_umap', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, by: 'Optional[str]' = None, sample_key: 'Optional[str]' = None, shift_level: 'str' = 'global', shift_index_key: 'Optional[str]' = None, obs_key: 'str' = 'scgeo_vel_delta_align', store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Convenience wrapper for scVelo/CellRank workflows:

### I/O contract
_No I/O entries found in `scgeo_io_manifest.json`._

---

## `scgeo.tl.wasserstein`

**Signature**  
`(adata, rep: 'str' = 'X_pca', condition_key: 'str' = 'condition', group1: 'Any' = None, group0: 'Any' = None, *, by: 'Optional[str]' = None, n_proj: 'int' = 128, p: 'int' = 2, seed: 'int' = 0, store_key: 'str' = 'scgeo') -> 'None'`

**Docstring**  
Compute (sliced) Wasserstein distance between two conditions in embedding space.

### I/O contract
**Writes**
- `uns_keys.scgeo` (unknown) — auto-detected

---
