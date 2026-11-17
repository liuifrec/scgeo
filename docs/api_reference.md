# API reference

Public API for **ScGeo 0.1.0** · **87 exported callables**.

[Quick start](QUICKSTART.md) · [Installation](INSTALLATION.md) · [Interpretation](RESULT_INTERPRETATION.md)

Start with `tl.robust_shift`, `get.state_report` and `pl.state_evidence_panel`. Add `tl.representation_stability` and `tl.local_geometry_stability` when the required representations are available.

| Namespace | Purpose | Browse separately |
|---|---|---|
| [`scgeo.tl`](#scgeotl) | Analysis | [20 callables](api/tl.md) |
| [`scgeo.pl`](#scgeopl) | Figures | [36 callables](api/pl.md) |
| [`scgeo.get`](#scgeoget) | Reports and tables | [8 callables](api/get.md) |
| [`scgeo.pp`](#scgeopp) | Reference preparation | [6 callables](api/pp.md) |
| [`scgeo.data`](#scgeodata) | Census access | [12 callables](api/data.md) |
| [`scgeo.bench`](#scgeobench) | Synthetic benchmarks | [5 callables](api/bench.md) |

## Reading this reference

Signatures, type annotations and defaults are generated from the package's public exports. Expand each function's usage notes for its docstring. Use `help(scgeo.tl.robust_shift)` (or another public symbol) for the documentation of your installed version.

The I/O tables describe key presence observed on a small synthetic input, not every supported workflow. A skipped probe means that domain-specific inputs are needed; it does not mean the API is unavailable. The legacy manifest field `touched` means a key exists before and after the call, not that its values were changed. Nested values and read-only access are not inferred from this probe.

For regeneration and validation, see [API and I/O provenance](contracts/README.md). These development manifests do not replace the paper's frozen analysis records.

## scgeo.tl

**Analysis.** Measure displacement, representation stability, local geometry, distributional change and dynamics agreement.

- [`align_vectors`](#scgeotlalign_vectors)
- [`alignment_driver_genes`](#scgeotlalignment_driver_genes)
- [`analyze_shift`](#scgeotlanalyze_shift)
- [`consensus_subspace`](#scgeotlconsensus_subspace)
- [`density_overlap`](#scgeotldensity_overlap)
- [`distribution_test`](#scgeotldistribution_test)
- [`local_geometry_stability`](#scgeotllocal_geometry_stability)
- [`map_knn`](#scgeotlmap_knn)
- [`map_query_to_ref`](#scgeotlmap_query_to_ref)
- [`map_query_to_ref_pool`](#scgeotlmap_query_to_ref_pool)
- [`map_query_to_ref_pool_census`](#scgeotlmap_query_to_ref_pool_census)
- [`mixscore`](#scgeotlmixscore)
- [`paga_composition_stats`](#scgeotlpaga_composition_stats)
- [`projection_disagreement`](#scgeotlprojection_disagreement)
- [`representation_stability`](#scgeotlrepresentation_stability)
- [`robust_shift`](#scgeotlrobust_shift)
- [`shift`](#scgeotlshift)
- [`velocity_delta_alignment`](#scgeotlvelocity_delta_alignment)
- [`velocity_shift_alignment`](#scgeotlvelocity_shift_alignment)
- [`wasserstein`](#scgeotlwasserstein)

### `scgeo.tl.align_vectors`

Cosine alignment between vectors.

```text
scgeo.tl.align_vectors(
    adata,
    vec_key: 'str',
    ref_vec_key: 'Optional[str]' = None,
    *,
    ref_from_shift: 'bool' = False,
    shift_store_key: 'str' = 'scgeo',
    shift_kind: 'str' = 'shift',
    shift_level: 'str' = 'global',
    shift_index_key: 'Optional[str]' = None,
    obs_key: 'str' = 'scgeo_align',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_align_vectors.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Cosine alignment between vectors.

Primary vectors:
  - adata.obsm[vec_key]  (n_cells x d)

Reference:
  - if ref_vec_key provided: adata.obsm[ref_vec_key]  (n_cells x d)
  - if ref_from_shift=True:
      shift_level="global": uses single delta vector (broadcast to all cells)
      shift_level="by": uses adata.uns[...]["by"][level]["delta"], level from obs[shift_index_key]
      shift_level="samples": uses adata.uns[...]["samples"][sample]["delta"], sample from obs[shift_index_key]

Writes:
  - adata.obs[obs_key] : per-cell cosine
  - adata.uns[store_key]["align_vectors"] : params + summary
```

</details>

### `scgeo.tl.alignment_driver_genes`

Identify genes associated with alignment or discordance between geometric shift and velocity direction.

```text
scgeo.tl.alignment_driver_genes(
    adata,
    *,
    alignment_key: 'str',
    group1: 'str' = 'discordant',
    group2: 'str' = 'aligned',
    subset_key: 'Optional[str]' = None,
    subset_values: 'Optional[Sequence[str]]' = None,
    layer: 'Optional[str]' = None,
    method: 'str' = 'wilcoxon',
    pts: 'bool' = True,
    min_cells: 'int' = 20,
    key_added: 'str' = 'alignment_driver_genes',
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/tl/_alignment_driver_genes.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Identify genes associated with alignment or discordance between geometric shift
and velocity direction.

Parameters
----------
adata
    AnnData object.
alignment_key
    obs column containing alignment labels, e.g. "alignment_group".
group1, group2
    The two groups to compare, usually "discordant" vs "aligned".
subset_key
    Optional obs column for stratified analysis, e.g. "cluster_label_manual".
subset_values
    Optional subset values to analyze. If None and subset_key is provided,
    all categories/unique values are used.
layer
    Optional layer to use as expression matrix. If provided, it is copied into
    a minimal temporary AnnData as X before DE.
method
    Differential expression method passed to scanpy.tl.rank_genes_groups.
pts
    Whether to compute fraction of expressing cells.
min_cells
    Minimum number of cells required in each group for a valid comparison.
key_added
    Key used to store the concatenated result table in adata.uns.

Returns
-------
pd.DataFrame
    Tidy dataframe of DE results with columns such as:
    ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', ...]
    plus ['subset', 'group1', 'group2'].
```

</details>

### `scgeo.tl.analyze_shift`

High-level ScGeo analysis orchestrator.

```text
scgeo.tl.analyze_shift(
    adata,
    *,
    node_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    basis: 'str' = 'umap',
    velocity_basis: 'Optional[str]' = None,
    ood_key: 'Optional[str]' = None,
    ood_groupby: 'Optional[str]' = None,
    robustness: 'Optional[pd.DataFrame]' = None,
    min_cells: 'int' = 10,
    store_key: 'str' = 'shift',
    overwrite: 'bool' = True,
)
```

[View implementation](../scgeo/tl/_analyze_shift.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
High-level ScGeo analysis orchestrator.

Stores a compact analysis bundle at:
    adata.uns["scgeo"][store_key]

Stored keys
-----------
params
shift_summary
velocity_alignment   (if velocity embedding exists / requested)
composition
ood_summary          (if ood_key is provided)
robustness           (if robustness DataFrame is provided)
```

</details>

### `scgeo.tl.consensus_subspace`

Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).

```text
scgeo.tl.consensus_subspace(
    adata: 'AnnData',
    rep: 'str' = 'X_pca',
    condition_key: 'str' = 'condition',
    group0: 'str | None' = None,
    group1: 'str | None' = None,
    sample_key: 'str | None' = None,
    n_components: 'int' = 2,
    obs_key_prefix: 'str' = 'cs',
    store_key: 'str' = 'consensus_subspace',
    min_cells: 'int' = 20,
    center: 'bool' = False,
) -> 'None'
```

[View implementation](../scgeo/tl/_consensus_subspace.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute consensus subspace directions from multiple delta vectors (sample-aware if sample_key is given).

Deltas are computed as mean(rep)[group1] - mean(rep)[group0] within each sample (or globally if sample_key None).
Consensus directions are the top right singular vectors of the delta matrix.

Stores:
  adata.uns["scgeo"][store_key] with components, singular_values, deltas, metadata
  adata.obsm[f"X_{obs_key_prefix}"] as cell projections onto consensus directions
  adata.obs[f"{obs_key_prefix}_score"] as per-cell magnitude in that subspace
```

</details>

### `scgeo.tl.density_overlap`

Compute density overlap between two conditions on an embedding using kNN density estimates. Outputs Bhattacharyya coefficient (BC) and Hellinger distance (H).

```text
scgeo.tl.density_overlap(
    adata,
    rep: 'str' = 'X_umap',
    condition_key: 'str' = 'condition',
    group1: 'Any' = None,
    group0: 'Any' = None,
    *,
    by: 'Optional[str]' = None,
    k: 'int' = 30,
    eval_on: 'str' = 'union',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_density_overlap.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `dens_test` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute density overlap between two conditions on an embedding using kNN density estimates.
Outputs Bhattacharyya coefficient (BC) and Hellinger distance (H).

BC = ∫ sqrt(p q) dx   (in [0,1], higher = more overlap)
H  = sqrt(1 - BC)     (in [0,1], higher = more separated)
```

</details>

### `scgeo.tl.distribution_test`

Distribution difference test with sample-aware permutation.

```text
scgeo.tl.distribution_test(
    adata,
    rep: 'str' = 'X_pca',
    condition_key: 'str' = 'condition',
    group1: 'Any' = None,
    group0: 'Any' = None,
    *,
    sample_key: 'Optional[str]' = None,
    by: 'Optional[str]' = None,
    method: 'str' = 'energy',
    n_perm: 'int' = 500,
    seed: 'int' = 0,
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_distribution_test.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `dist_test` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Distribution difference test with sample-aware permutation.

If sample_key provided:
  - compute observed statistic on all cells
  - permute condition labels at the SAMPLE level (swap sample assignments), recompute statistic
  - p-value = (1 + #{perm >= obs})/(1+n_perm)

Stores:
  adata.uns[store_key]["distribution_test"]
```

</details>

### `scgeo.tl.local_geometry_stability`

Quantify local-geometry preservation across latent representations.

```text
scgeo.tl.local_geometry_stability(
    adata,
    *,
    reps,
    node_key=None,
    sample_key=None,
    k_values=(15, 30, 50),
    metric='euclidean',
    pair_mode='all',
    reference_rep=None,
    n_boot=500,
    max_exact_cells=3000,
    stratify_key=None,
    store_per_cell=False,
    seed=0,
    store_key='local_geometry_stability',
)
```

[View implementation](../scgeo/tl/_local_geometry_stability.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `scgeo` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Quantify local-geometry preservation across latent representations.

Definitions
-----------
For cell i and representation pair (a, b), neighborhood overlap is
``|N_a(i) intersect N_b(i)| / k`` and neighborhood Jaccard is
``|N_a(i) intersect N_b(i)| / |N_a(i) union N_b(i)|``. Directed graph
edge retention is the Jaccard similarity of directed kNN edge sets.

Distance distortion uses the union of directed kNN edges. For edge
i -> j, global-scale distortion is
``abs(log((d_b / med_kNN_b + eps) / (d_a / med_kNN_a + eps)))``.
Local-shape distortion replaces the global median with the source cell's
local median kNN distance. These quantities are invariant to translation,
orthogonal rotation, and isotropic rescaling, but not to anisotropic or
nonlinear distortion.

Trustworthiness and continuity are exact rank metrics from scikit-learn.
When ``n_obs > max_exact_cells``, they are evaluated on a reproducible
selected subset, optionally stratified by ``stratify_key`` or ``node_key``.
```

</details>

### `scgeo.tl.map_knn`

kNN mapping reference->query with QC:   label = majority vote   confidence = vote fraction   entropy = vote entropy   OOD flag = mean kNN dist > quantile threshold

```text
scgeo.tl.map_knn(
    adata_ref,
    adata_q,
    label_key: 'str' = 'cell_type',
    rep: 'str' = 'X_pca',
    k: 'int' = 25,
    ood_quantile: 'float' = 0.99,
    out_label_key: 'str' = 'scgeo_label',
    out_conf_key: 'str' = 'scgeo_confidence',
    out_ent_key: 'str' = 'scgeo_entropy',
    out_ood_key: 'str' = 'scgeo_ood',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_map_knn.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
kNN mapping reference->query with QC:
  label = majority vote
  confidence = vote fraction
  entropy = vote entropy
  OOD flag = mean kNN dist > quantile threshold

Writes into adata_q.obs and stores metadata in adata_q.uns[store_key]["map_knn"].
```

</details>

### `scgeo.tl.map_query_to_ref`

Graph-native reference mapping + QC that is agnostic to how the graph was built (BBKNN/Harmony/scVI/scanpy neighbors/...).

```text
scgeo.tl.map_query_to_ref(
    adata,
    *,
    ref_key: 'str',
    ref_value: 'str',
    label_key: 'str',
    query_key: 'Optional[str]' = None,
    query_value: 'Optional[str]' = None,
    graph_key: 'str' = 'connectivities',
    conf_method: 'str' = 'entropy_margin',
    ood_method: 'str' = 'connectivity_mass',
    reject_conf: 'Optional[float]' = None,
    reject_ood: 'Optional[float]' = None,
    rep: 'Optional[str]' = None,
    store_key: 'str' = 'map_query_to_ref',
    pred_key: 'str' = 'scgeo_pred',
    conf_key: 'str' = 'scgeo_conf',
    conf_entropy_key: 'str' = 'scgeo_conf_entropy',
    conf_margin_key: 'str' = 'scgeo_conf_margin',
    ood_key: 'str' = 'scgeo_ood',
    reject_key: 'str' = 'scgeo_reject',
    return_probs: 'bool' = False,
    probs_key: 'str' = 'X_map_probs',
    label_order_key: 'str' = 'map_label_order',
) -> 'None'
```

[View implementation](../scgeo/tl/_map_query_to_ref.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Graph-native reference mapping + QC that is agnostic to how the graph was built
(BBKNN/Harmony/scVI/scanpy neighbors/...).

Requirements:
  - adata.obs[ref_key] exists and defines reference cells.
  - adata.obs[label_key] exists for reference cells.
  - adata.obsp[graph_key] exists as a CSR/CSC sparse affinity matrix.

Writes:
  adata.obs[pred_key]                predicted label for query cells (NaN for ref)
  adata.obs[conf_entropy_key]        1 - normalized entropy of label probs (query)
  adata.obs[conf_margin_key]         top1 - top2 probability margin (query)
  adata.obs[conf_key]                combined confidence (query)
  adata.obs[ood_key]                 OOD score (higher = more OOD-like) (query)
  adata.obs[reject_key]              reject flag (query) if thresholds set

  adata.uns["scgeo"][store_key]      params + per_label summary (+ optional label_deltas)

Notes:
  - OOD is computed as 1 - (sum of graph weights to reference neighbors).
  - Confidence is derived from graph-weighted label voting among reference neighbors.
```

</details>

### `scgeo.tl.map_query_to_ref_pool`

Embedding-only mapping using a ReferencePool (ANN index).

```text
scgeo.tl.map_query_to_ref_pool(
    adata,
    pool: 'ReferencePool',
    *,
    rep: 'str',
    k: 'int' = 30,
    weight_method: 'str' = 'inv',
    conf_method: 'str' = 'entropy_margin',
    ood_method: 'str' = 'distance',
    reject_conf: 'Optional[float]' = None,
    reject_ood: 'Optional[float]' = None,
    store_key: 'str' = 'map_query_to_ref',
    pred_key: 'str' = 'scgeo_pred',
    conf_key: 'str' = 'scgeo_conf',
    conf_entropy_key: 'str' = 'scgeo_conf_entropy',
    conf_margin_key: 'str' = 'scgeo_conf_margin',
    ood_key: 'str' = 'scgeo_ood',
    reject_key: 'str' = 'scgeo_reject',
    return_probs: 'bool' = False,
    probs_key: 'str' = 'X_map_probs',
    label_order_key: 'str' = 'map_label_order',
) -> 'None'
```

[View implementation](../scgeo/tl/_map_query_to_ref_pool.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Embedding-only mapping using a ReferencePool (ANN index).

Writes same obs/uns keys as graph-native map_query_to_ref so plotting + get.table work.
```

</details>

### `scgeo.tl.map_query_to_ref_pool_census`

Map query cells to a reference pool built from CELLxGENE Census embeddings.

```text
scgeo.tl.map_query_to_ref_pool_census(
    adata_q,
    *,
    pool=None,
    census=None,
    rep: 'str' = 'X_emb',
    embedding_name: 'str' = 'scvi',
    organism: 'str' = 'Homo sapiens',
    label_key: 'str' = 'cell_type',
    obs_value_filter: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
    max_refs: 'int' = 200000,
    index_metric: 'str' = 'euclidean',
    index_seed: 'int' = 0,
    n_neighbors: 'int' = 30,
    k: 'Optional[int]' = None,
    dedup: 'Optional[bool]' = None,
    census_obs_filter: 'Optional[str]' = None,
    store_key: 'str' = 'map_query_to_ref',
    pred_key: 'str' = 'scgeo_pred',
    conf_key: 'str' = 'scgeo_conf',
    conf_entropy_key: 'str' = 'scgeo_conf_entropy',
    conf_margin_key: 'str' = 'scgeo_conf_margin',
    ood_key: 'str' = 'scgeo_ood',
    reject_key: 'str' = 'scgeo_reject',
    conf_method: 'str' = 'entropy_margin',
    ood_method: 'str' = 'distance',
    reject_conf: 'Optional[float]' = None,
    reject_ood: 'Optional[float]' = None,
    return_probs: 'bool' = False,
    probs_key: 'str' = 'X_map_probs',
    label_order_key: 'str' = 'map_label_order',
) -> 'None'
```

[View implementation](../scgeo/tl/_map_query_to_ref_pool_census.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

### `scgeo.tl.mixscore`

kNN label mixing score in [0,1]:   per cell = fraction of neighbors with a different label.

```text
scgeo.tl.mixscore(
    adata,
    label_key: 'str' = 'batch',
    rep: 'str' = 'X_pca',
    k: 'int' = 50,
    use_connectivities: 'bool' = True,
    connectivities_key: 'str' = 'connectivities',
    obs_key: 'str' = 'scgeo_mixscore',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_mixscore.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | `scgeo_mixscore` | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `mix_test` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
kNN label mixing score in [0,1]:
  per cell = fraction of neighbors with a different label.

Preference order:
  1) adata.obsp[connectivities_key] if present and use_connectivities=True (scanpy/bbknn native)
  2) kNN on adata.obsm[rep] (requires scikit-learn)

Writes:
  adata.obs[obs_key]
  adata.uns[store_key]["mixscore"]
```

</details>

### `scgeo.tl.paga_composition_stats`

For each cluster/node, test enrichment of condition1 vs condition0.

```text
scgeo.tl.paga_composition_stats(
    adata,
    group_key: 'str',
    condition_key: 'str',
    group0: 'Any',
    group1: 'Any',
    *,
    sample_key: 'Optional[str]' = None,
    method: 'str' = 'gee',
    n_boot: 'int' = 1000,
    seed: 'int' = 0,
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_paga_composition_stats.py)

- The default method="gee" requires statsmodels. Choose the analysis method explicitly; method="bootstrap" is a different statistical procedure.

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
For each cluster/node, test enrichment of condition1 vs condition0.

If method="gee" and sample_key provided:
  - Fit GEE logistic regression: y = 1(cell in node), x = condition (0/1), cluster = sample
  - Returns OR and robust SE CI

Fallback method="bootstrap":
  - Bootstrap over samples: compute per-sample proportions in node, compare groups, CI + effect
```

</details>

### `scgeo.tl.projection_disagreement`

Compute per-cell projection disagreement among multiple vector sources.

```text
scgeo.tl.projection_disagreement(
    adata,
    sources: 'Sequence[Dict[str, Any]]',
    obs_key: 'str' = 'scgeo_disagree',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_projection_disagreement.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute per-cell projection disagreement among multiple vector sources.

sources: list of dicts describing vector sources, e.g.
  {"type": "obsm", "key": "velocity_umap", "name": "velocity"}
  {"type": "shift", "level": "global", "name": "delta_global"}
  {"type": "shift", "level": "by", "index_key": "cell_type", "name": "delta_by_ct"}

Output:
  adata.obs[obs_key] = 1 - mean_pairwise_cosine (nan-safe)
  adata.uns[store_key]["projection_disagreement"] contains params + summary
```

</details>

### `scgeo.tl.representation_stability`

Assess whether state-level perturbation geometry is stable across representations.

```text
scgeo.tl.representation_stability(
    adata,
    *,
    reps,
    node_key,
    condition_key,
    group0,
    group1,
    sample_key=None,
    center: 'str' = 'geometric_median',
    trim_fraction: 'float' = 0.1,
    n_boot: 'int' = 500,
    velocity_keys=None,
    alignment_pos_thr: 'float' = 0.3,
    alignment_neg_thr: 'float' = -0.3,
    min_cells: 'int' = 20,
    consensus_rules=None,
    seed: 'int' = 0,
    store_key: 'str' = 'representation_stability',
)
```

[View implementation](../scgeo/tl/_representation_stability.py)

- Consensus labels include `stable_neutral` for consistently negligible effects, `stable_effect` for non-neutral magnitude agreement without velocity evidence, `stable_aligned`/`stable_discordant` when velocity classes agree, `representation_unstable` for substantive representation disagreement, and `insufficient_coverage`.

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `scgeo` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Assess whether state-level perturbation geometry is stable across representations.

Consensus labels use explicit rules stored in ``params['consensus_label_rules']``.
The function never averages raw displacement vectors across representations; it
combines only coordinate-safe scalars: normalized magnitudes, ranks, velocity
cosines computed within each representation, and class agreement. Non-neutral
magnitude agreement without velocity evidence is labeled ``stable_effect``;
rank and direction instability are reserved for substantive representation
disagreement after coverage and stable-neutral checks.
```

</details>

### `scgeo.tl.robust_shift`

Robust condition displacement with sample-aware bootstrap uncertainty.

```text
scgeo.tl.robust_shift(
    adata,
    rep: 'str' = 'X_pca',
    condition_key: 'str' = 'condition',
    group0: 'Any' = None,
    group1: 'Any' = None,
    by: 'Optional[str]' = None,
    sample_key: 'Optional[str]' = None,
    center: 'str' = 'geometric_median',
    trim_fraction: 'float' = 0.1,
    n_boot: 'int' = 500,
    bootstrap_unit: 'str' = 'auto',
    normalize_by: 'Optional[str]' = 'pooled_robust_scale',
    seed: 'int' = 0,
    store_key: 'str' = 'robust_shift',
) -> 'Dict[str, Any]'
```

[View implementation](../scgeo/tl/_robust_shift.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `scgeo` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Robust condition displacement with sample-aware bootstrap uncertainty.

When ``sample_key`` is provided, per-condition centers are estimated from
sample-level centers and bootstrap resampling uses samples by default.

Minimal example
---------------
>>> out = sg.tl.robust_shift(
...     adata,
...     rep="X_pca",
...     condition_key="condition",
...     group0="control",
...     group1="treated",
...     sample_key="donor",
... )
>>> out["global"]["delta_norm"]

Stores results at ``adata.uns["scgeo"][store_key]`` and returns the same
dictionary.
```

</details>

### `scgeo.tl.shift`

Compute mean shift vector Δ = μ1 - μ0 in representation space.

```text
scgeo.tl.shift(
    adata,
    rep: 'str' = 'X_pca',
    condition_key: 'str' = 'condition',
    group1: 'Any' = None,
    group0: 'Any' = None,
    by: 'Optional[str]' = None,
    sample_key: 'Optional[str]' = None,
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_shift.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `shift_test` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute mean shift vector Δ = μ1 - μ0 in representation space.

Stores results in:
  adata.uns[store_key]["shift"]["global"]
  adata.uns[store_key]["shift"]["by"][<level>] (if by is provided)
  adata.uns[store_key]["shift"]["samples"] (if sample_key is provided)
```

</details>

### `scgeo.tl.velocity_delta_alignment`

Convenience wrapper for scVelo/CellRank workflows:   1) compute Δ via sg.tl.shift on rep_for_shift   2) align velocity vectors to Δ using sg.tl.align_vectors

```text
scgeo.tl.velocity_delta_alignment(
    adata,
    *,
    velocity_key: 'Optional[str]' = None,
    rep_for_shift: 'str' = 'X_umap',
    condition_key: 'str' = 'condition',
    group1: 'Any' = None,
    group0: 'Any' = None,
    by: 'Optional[str]' = None,
    sample_key: 'Optional[str]' = None,
    shift_level: 'str' = 'global',
    shift_index_key: 'Optional[str]' = None,
    obs_key: 'str' = 'scgeo_vel_delta_align',
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_velocity_delta_alignment.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Convenience wrapper for scVelo/CellRank workflows:
  1) compute Δ via sg.tl.shift on rep_for_shift
  2) align velocity vectors to Δ using sg.tl.align_vectors

velocity_key auto-detection order:
  - "velocity_umap"
  - "velocity_pca"
```

</details>

### `scgeo.tl.velocity_shift_alignment`

Compute node-wise alignment between observed geometric shift and mean velocity.

```text
scgeo.tl.velocity_shift_alignment(
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
    alignment_pos_thr: 'float' = 0.3,
    alignment_neg_thr: 'float' = -0.3,
    key_added: 'Optional[str]' = 'velocity_shift_alignment',
    propagate_to_obs: 'bool' = False,
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/tl/_velocity_shift_alignment.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute node-wise alignment between observed geometric shift and mean velocity.

Parameters
----------
adata
    AnnData object.
node_key
    obs column defining nodes/groups (e.g. cluster labels).
condition_key
    obs column defining the condition or timepoint.
group0, group1
    Two values in `condition_key` to compare. Shift is computed as group1 - group0.
basis
    Embedding basis in `obsm`, with or without `X_` prefix.
velocity_basis
    Velocity embedding basis in `obsm`, with or without `velocity_` prefix.
    If None, auto-detect from `basis`.
min_cells
    Minimum cells required per node per condition.
agg
    Aggregation method for centroids and mean velocity: {'mean', 'median'}.
alignment_pos_thr, alignment_neg_thr
    Thresholds for classifying alignment cosine.
key_added
    If not None, store the node-level dataframe in `adata.uns[key_added]`.
propagate_to_obs
    If True, propagate node-level cosine/class back to cells in `adata.obs`.

Returns
-------
pd.DataFrame
    Node-level dataframe with shift, velocity, cosine alignment, and classes.
```

</details>

### `scgeo.tl.wasserstein`

Compute (sliced) Wasserstein distance between two conditions in embedding space.

```text
scgeo.tl.wasserstein(
    adata,
    rep: 'str' = 'X_pca',
    condition_key: 'str' = 'condition',
    group1: 'Any' = None,
    group0: 'Any' = None,
    *,
    by: 'Optional[str]' = None,
    n_proj: 'int' = 128,
    p: 'int' = 2,
    seed: 'int' = 0,
    store_key: 'str' = 'scgeo',
) -> 'None'
```

[View implementation](../scgeo/tl/_wasserstein.py)

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `batch`, `cluster`, `condition` | — |
| `obsm_keys` | — | `X_pca`, `X_umap` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | `dist_test` | — | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compute (sliced) Wasserstein distance between two conditions in embedding space.

Stores:
  adata.uns[store_key]["wasserstein"]["global"]
  adata.uns[store_key]["wasserstein"]["by"][level] (if by provided)

Notes:
  - Uses sliced Wasserstein by default (no extra deps).
  - Equal-weight empirical distributions; uses min(n0, n1) truncation per projection.
```

</details>

## scgeo.pl

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

[View implementation](../scgeo/pl/_alignment_panel.py)


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

[View implementation](../scgeo/pl/_ambiguity_panel.py)


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

[View implementation](../scgeo/pl/_composition_drift.py)


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

[View implementation](../scgeo/pl/_perturbation_report.py)


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

[View implementation](../scgeo/pl/_consensus_structure.py)


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

[View implementation](../scgeo/pl/_consensus_panel.py)


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

[View implementation](../scgeo/pl/_delta_rank.py)


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

[View implementation](../scgeo/pl/_density_overlap.py)


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

[View implementation](../scgeo/pl/_overlap.py)


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

[View implementation](../scgeo/pl/_report.py)


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

[View implementation](../scgeo/pl/_distribution_test.py)


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

[View implementation](../scgeo/pl/_volcano.py)


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

[View implementation](../scgeo/pl/_density.py)


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

[View implementation](../scgeo/pl/_gallery.py)


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

[View implementation](../scgeo/pl/_highlight.py)


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

[View implementation](../scgeo/pl/_legend.py)


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

[View implementation](../scgeo/pl/_perturbation_report.py)


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

[View implementation](../scgeo/pl/_mapping_qc.py)

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

[View implementation](../scgeo/pl/_mapping_qc.py)

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

[View implementation](../scgeo/pl/_mapping_qc.py)

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

[View implementation](../scgeo/pl/_ood_landscape.py)


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

[View implementation](../scgeo/pl/_paga_composition.py)


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

[View implementation](../scgeo/pl/_paga_composition.py)


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

[View implementation](../scgeo/pl/_paga_composition.py)


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

[View implementation](../scgeo/pl/_paga_scgeo.py)


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

[View implementation](../scgeo/pl/_paga_shift_map.py)


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

[View implementation](../scgeo/pl/_perturbation_report.py)

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

[View implementation](../scgeo/pl/_recovery_compass.py)


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

[View implementation](../scgeo/pl/_perturbation_report.py)

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

[View implementation](../scgeo/pl/_robustness_matrix.py)


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

[View implementation](../scgeo/pl/_score.py)


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

[View implementation](../scgeo/pl/_score.py)

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

[View implementation](../scgeo/pl/_perturbation_report.py)


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

[View implementation](../scgeo/pl/_state_flow_alluvial.py)


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

[View implementation](../scgeo/pl/_state_flow_sankey.py)


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

[View implementation](../scgeo/pl/_velocity_shift_alignment.py)


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

## scgeo.get

**Reports and tables.** Retrieve analysis tables and combine available evidence into a state-level report.

- [`get_available_tables`](#scgeogetget_available_tables)
- [`get_composition_table`](#scgeogetget_composition_table)
- [`get_ood_summary`](#scgeogetget_ood_summary)
- [`get_robustness_table`](#scgeogetget_robustness_table)
- [`get_shift_summary`](#scgeogetget_shift_summary)
- [`get_velocity_alignment_summary`](#scgeogetget_velocity_alignment_summary)
- [`state_report`](#scgeogetstate_report)
- [`table`](#scgeogettable)

### `scgeo.get.get_available_tables`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_available_tables(adata, *, store_key: 'str' = 'shift') -> 'list[str]'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.get_composition_table`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_composition_table(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.get_ood_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_ood_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.get_robustness_table`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_robustness_table(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.get_shift_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_shift_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.get_velocity_alignment_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_velocity_alignment_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_figure_tables.py)

### `scgeo.get.state_report`

Build a canonical state-level ScGeo report from stored analysis results.

```text
scgeo.get.state_report(
    adata,
    *,
    node_key=None,
    robust_shift_key: 'str' = 'robust_shift',
    representation_key: 'str' = 'representation_stability',
    local_geometry_key: 'str' = 'local_geometry_stability',
    local_k=None,
    pair_aggregation: 'str' = 'median',
    include_worst_case: 'bool' = True,
    comparison_label: 'Optional[str]' = None,
    condition_key: 'Optional[str]' = None,
    group0: 'Any' = None,
    group1: 'Any' = None,
    strict: 'bool' = False,
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_state_report.py)

- Read-only: consumes existing adata.uns["scgeo"] results and returns a pandas DataFrame.
- The report combines displacement and rank summaries with `effect_status`, `stability_status`, `local_geometry_status`, `dynamics_status`, `coverage_status`, rule-based `reason_codes`, and a quantitative plain-language `summary`.
- Local geometry reporting records k values used, valid representation-pair counts, median summaries, worst-case summaries, and across-k variability when available.
- Metadata in `report.attrs` includes package version, source store keys, comparison identity, representations, k values, sample key, center estimator, bootstrap unit, consensus rules, global diagnostics, representation diagnostics, warnings, and creation timestamp.
- State-graph summaries are kept as global diagnostics unless a stored analysis provides a genuinely state-specific transition-profile metric.
- Missing-data behavior: `strict=False` returns available columns with warnings; `strict=True` raises if requested storage objects are absent.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build a canonical state-level ScGeo report from stored analysis results.

The report reuses existing ``adata.uns['scgeo']`` outputs from robust shift,
representation stability, and local-geometry stability. It does not
recompute displacement, bootstrap, rank, neighborhood, or velocity metrics.

With ``strict=False``, missing modules add warnings and leave unavailable
columns as missing values. With ``strict=True``, each requested storage
object must exist.
```

</details>

### `scgeo.get.table`

Return tidy table from adata.uns[store_key][kind].

```text
scgeo.get.table(
    adata,
    store_key: 'str' = 'scgeo',
    kind: 'str' = 'shift',
    level: 'str' = 'global',
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/get/_table.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Return tidy table from adata.uns[store_key][kind].

Conventions:
  - shift-like objects: expect keys: ["global"] and optionally ["by"], ["samples"].
  - paga_composition_stats: expects obj["table"] as a DataFrame-like.
  - map_query_to_ref: expects obj["global"] and either obj["per_label_dict"] or obj["per_label"].
```

</details>

## scgeo.pp

**Reference preparation.** Build reference pools and query-to-reference neighbor graphs. Approximate reference-pool construction requires pynndescent.

- [`ReferencePool`](#scgeoppreferencepool)
- [`build_block_connectivities_from_q2r`](#scgeoppbuild_block_connectivities_from_q2r)
- [`build_query_to_ref_knn_edges`](#scgeoppbuild_query_to_ref_knn_edges)
- [`build_query_to_ref_knn_edges_from_census`](#scgeoppbuild_query_to_ref_knn_edges_from_census)
- [`build_reference_pool`](#scgeoppbuild_reference_pool)
- [`build_reference_pool_from_census`](#scgeoppbuild_reference_pool_from_census)

### `scgeo.pp.ReferencePool`

Store reference coordinates, observation labels, an approximate-neighbor index and provenance.

```text
scgeo.pp.ReferencePool(
    X: 'np.ndarray',
    obs: 'Dict[str, np.ndarray]',
    label_key: 'str',
    index: 'Any',
    meta: 'Dict[str, Any]',
    joinids: 'Optional[np.ndarray]' = None,
) -> None
```

[View implementation](../scgeo/pp/_reference_pool.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
ReferencePool(X: 'np.ndarray', obs: 'Dict[str, np.ndarray]', label_key: 'str', index: 'Any', meta: 'Dict[str, Any]', joinids: 'Optional[np.ndarray]' = None)
```

</details>

### `scgeo.pp.build_block_connectivities_from_q2r`

Build a square CSR connectivities matrix for (ref + qry) nodes where edges go from query -> ref.

```text
scgeo.pp.build_block_connectivities_from_q2r(
    n_ref: 'int',
    n_qry: 'int',
    rows_q: 'np.ndarray',
    cols_r: 'np.ndarray',
    data: 'np.ndarray',
)
```

[View implementation](../scgeo/pp/_knn_graph.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build a square CSR connectivities matrix for (ref + qry) nodes where
edges go from query -> ref.

Node indexing:
  ref: [0..n_ref)
  qry: [n_ref..n_ref+n_qry)
```

</details>

### `scgeo.pp.build_query_to_ref_knn_edges`

Return (rows, cols, data) edges for a bipartite graph from query -> ref.

```text
scgeo.pp.build_query_to_ref_knn_edges(
    X_ref: 'np.ndarray',
    X_qry: 'np.ndarray',
    *,
    k: 'int' = 30,
    metric: "Literal['euclidean', 'cosine']" = 'cosine',
    mode: "Literal['distance', 'similarity']" = 'similarity',
    sigma: 'Optional[float]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]'
```

[View implementation](../scgeo/pp/_knn_graph.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Return (rows, cols, data) edges for a bipartite graph from query -> ref.

rows: query row indices in [0, n_qry)
cols: ref col indices in [0, n_ref)
data: weights (float32)
```

</details>

### `scgeo.pp.build_query_to_ref_knn_edges_from_census`

Build query->ref edges from Census embedding search outputs.

```text
scgeo.pp.build_query_to_ref_knn_edges_from_census(
    *,
    n_qry: 'int',
    ref_pool,
    nn_joinids: 'Sequence[Sequence[object]]',
    nn_dists: 'Sequence[Sequence[float]]',
    weight_mode: 'str' = 'inv',
    drop_missing: 'bool' = True,
    return_diagnostics: 'bool' = False,
    return_csr: 'bool' = False,
    sum_duplicates: 'bool' = True,
)
```

[View implementation](../scgeo/pp/_census_knn.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build query->ref edges from Census embedding search outputs.

If return_csr=True, returns (C_qr, diag) where:
  C_qr is CSR of shape (n_qry, n_ref)

Else returns (rows, cols, data, diag) in COO triplet form.
```

</details>

### `scgeo.pp.build_reference_pool`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.pp.build_reference_pool(
    X_ref,
    obs: 'Dict[str, Any]',
    *,
    label_key: 'str',
    joinids: 'Optional[np.ndarray]' = None,
    index_backend: 'str' = 'pynndescent',
    n_neighbors: 'int' = 30,
    metric: 'str' = 'euclidean',
    random_state: 'int' = 0,
    meta: 'Optional[Dict[str, Any]]' = None,
) -> 'ReferencePool'
```

[View implementation](../scgeo/pp/_reference_pool.py)

- The default pynndescent backend requires the optional pynndescent package. See the installation guide for feature-specific dependencies.

### `scgeo.pp.build_reference_pool_from_census`

Build a ReferencePool from cellxgene Census.

```text
scgeo.pp.build_reference_pool_from_census(
    *,
    census,
    organism: 'str' = 'Homo sapiens',
    embedding_name: 'str' = 'scvi',
    label_key: 'str' = 'cell_type',
    obs_value_filter: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
    var_value_filter: 'str' = "feature_name in ['CD34']",
    max_refs: 'Optional[int]' = 200000,
    index_metric: 'str' = 'euclidean',
    index_seed: 'int' = 0,
    n_neighbors: 'int' = 30,
    adata_q=None,
    rep: 'Optional[str]' = None,
    k: 'Optional[int]' = None,
    dedup: 'Optional[bool]' = None,
    census_obs_filter: 'Optional[str]' = None,
) -> 'ReferencePool'
```

[View implementation](../scgeo/pp/_reference_pool.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build a ReferencePool from cellxgene Census.

Two modes are supported:

1) Stable production mode:
   Uses cellxgene_census.get_anndata(..., obs_embeddings=[embedding_name])
   and requires obs_value_filter.

2) Legacy compatibility / test mode:
   If obs_value_filter is not provided but adata_q is provided, use the older
   query-driven embedding fetch path via scgeo.data helper functions.
   This keeps existing unit tests and older call sites working.
```

</details>

## scgeo.data

**Census access.** Access CELLxGENE Census data and embeddings with the optional Census dependencies. Data requests require network access; importing ScGeo does not download data.

- [`CensusQuery`](#scgeodatacensusquery)
- [`census_axis_query_tables`](#scgeodatacensus_axis_query_tables)
- [`census_find_nearest_obs`](#scgeodatacensus_find_nearest_obs)
- [`census_get_anndata`](#scgeodatacensus_get_anndata)
- [`census_get_embedding`](#scgeodatacensus_get_embedding)
- [`census_get_embedding_metadata_by_name`](#scgeodatacensus_get_embedding_metadata_by_name)
- [`census_predict_obs_metadata`](#scgeodatacensus_predict_obs_metadata)
- [`census_query_obs_dataframe`](#scgeodatacensus_query_obs_dataframe)
- [`fetch_obs_by_joinids`](#scgeodatafetch_obs_by_joinids)
- [`find_nearest_obs`](#scgeodatafind_nearest_obs)
- [`get_embedding_metadata_by_name`](#scgeodataget_embedding_metadata_by_name)
- [`open_census`](#scgeodataopen_census)

### `scgeo.data.CensusQuery`

Describe a CELLxGENE Census query using explicit organism, version and filtering options.

```text
scgeo.data.CensusQuery(
    organism: 'str' = 'Homo sapiens',
    census_version: 'Optional[str]' = None,
    measurement_name: 'str' = 'RNA',
    x_name: 'str' = 'raw',
    obs_value_filter: 'Optional[str]' = None,
    var_value_filter: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
    var_columns: 'Optional[Sequence[str]]' = None,
) -> None
```

[View implementation](../scgeo/data/_census.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
CensusQuery(organism: 'str' = 'Homo sapiens', census_version: 'Optional[str]' = None, measurement_name: 'str' = 'RNA', x_name: 'str' = 'raw', obs_value_filter: 'Optional[str]' = None, var_value_filter: 'Optional[str]' = None, obs_columns: 'Optional[Sequence[str]]' = None, var_columns: 'Optional[Sequence[str]]' = None)
```

</details>

### `scgeo.data.census_axis_query_tables`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_axis_query_tables(
    *,
    organism: 'str' = 'Homo sapiens',
    census_version: 'Optional[str]' = None,
    measurement_name: 'str' = 'RNA',
    x_name: 'str' = 'raw',
    obs_value_filter: 'Optional[str]' = None,
    var_value_filter: 'Optional[str]' = None,
) -> 'Iterator[Dict[str, object]]'
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.census_find_nearest_obs`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_find_nearest_obs(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.census_get_anndata`

Convenience wrapper around cellxgene_census.open_soma() + get_anndata().

```text
scgeo.data.census_get_anndata(
    *,
    organism: 'str' = 'Homo sapiens',
    census_version: 'Optional[str]' = None,
    measurement_name: 'str' = 'RNA',
    x_name: 'str' = 'raw',
    obs_value_filter: 'Optional[str]' = None,
    var_value_filter: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
    var_columns: 'Optional[Sequence[str]]' = None,
    obs_embeddings: 'Optional[Sequence[str]]' = None,
)
```

[View implementation](../scgeo/data/_census.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Convenience wrapper around cellxgene_census.open_soma() + get_anndata().

Supports stable embedding retrieval via obs_embeddings=[...].
```

</details>

### `scgeo.data.census_get_embedding`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_get_embedding(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.census_get_embedding_metadata_by_name`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_get_embedding_metadata_by_name(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.census_predict_obs_metadata`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_predict_obs_metadata(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.census_query_obs_dataframe`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_query_obs_dataframe(
    *,
    organism: 'str' = 'Homo sapiens',
    census_version: 'Optional[str]' = None,
    obs_value_filter: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
)
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.fetch_obs_by_joinids`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.fetch_obs_by_joinids(
    joinids: 'Iterable[int]',
    *,
    organism: 'str' = 'homo_sapiens',
    census_version: 'Optional[str]' = None,
    obs_columns: 'Optional[Sequence[str]]' = None,
    chunk_size: 'int' = 1000,
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/data/_census.py)

### `scgeo.data.find_nearest_obs`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.find_nearest_obs(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

- Backward-compatible alias; prefer the corresponding census_-prefixed function in new workflows.

### `scgeo.data.get_embedding_metadata_by_name`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.get_embedding_metadata_by_name(*args, **kwargs)
```

[View implementation](../scgeo/data/_census.py)

- Backward-compatible alias; prefer the corresponding census_-prefixed function in new workflows.

### `scgeo.data.open_census`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.open_census(*, census_version: 'Optional[str]' = 'stable')
```

[View implementation](../scgeo/data/_census.py)

## scgeo.bench

**Synthetic benchmarks.** Simulate controlled perturbations and evaluate stored outputs against known truth. The benchmark profiler requires Linux, WSL or macOS.

- [`evaluate_ground_truth`](#scgeobenchevaluate_ground_truth)
- [`framework_ablation`](#scgeobenchframework_ablation)
- [`plot_framework_ablation`](#scgeobenchplot_framework_ablation)
- [`run_simulation_suite`](#scgeobenchrun_simulation_suite)
- [`simulate_perturbation_geometry`](#scgeobenchsimulate_perturbation_geometry)

### `scgeo.bench.evaluate_ground_truth`

Evaluate stored ScGeo outputs against synthetic simulation truth.

```text
scgeo.bench.evaluate_ground_truth(
    adata,
    *,
    robust_shift_key: 'str' = 'robust_shift',
    representation_key: 'str' = 'representation_stability',
    local_geometry_key: 'str' = 'local_geometry_stability',
) -> 'dict[str, pd.DataFrame]'
```

[View implementation](../scgeo/bench/_simulation.py)

- Output: dictionary of tidy DataFrames for state magnitude errors, rank agreement, shifted-state precision/recall/F1, null false classification, bootstrap uncertainty interval coverage for nonzero effects, representation consensus labels, abundance truth, condition distribution-shape truth, non-identifiability truth, dynamics class accuracy, representation quality outlier detection, representation distortion detection, explicit corruption detection, pair-state representation-local-distortion detection, neighborhood preservation discrimination, replicate/sample diagnostics, coverage, runtime, and threshold sensitivity. Zero-effect magnitude coverage is marked `not_applicable`; null behavior is evaluated through shifted-state false-call rates and prespecified null diagnostics. Non-identifiable batch-confounded states are excluded from ordinary biological shifted-state classification rates. No composite benchmark score is created.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Evaluate stored ScGeo outputs against synthetic simulation truth.

The function returns tidy DataFrames and does not create a composite score.
Baseline magnitude comparisons are run on a temporary copy of ``adata``.
Bootstrap magnitude intervals are treated as uncertainty intervals; states
whose true magnitude is at or below ``_MAGNITUDE_ZERO_EPSILON`` are marked
``not_applicable`` for interval coverage rather than counted as failures.
```

</details>

### `scgeo.bench.framework_ablation`

Build a tidy framework-ablation table from synthetic benchmark outputs.

```text
scgeo.bench.framework_ablation(
    tables: 'Mapping[str, pd.DataFrame]',
    *,
    final_split: 'str' = 'evaluation',
) -> 'pd.DataFrame'
```

[View implementation](../scgeo/bench/_simulation.py)

- Variants: A original mean shift on one representation; B robust shift on one representation; C robust shift plus representation consensus; D C plus local geometry diagnostics; E D plus dynamics agreement.
- Output: one row per scenario, seed/job, variant, failure mode, and evaluated unit with explicit `truth`, `call`, `status`, and `outcome`; held-out evaluation rows are marked by `final_evaluation`. `condition_distribution_shape_change` is separate from pair-state `representation_local_distortion`; representation-level targets are split into `representation_quality_outlier`, `representation_distortion_detection`, and `explicit_corruption_detection`. Zero-effect bootstrap magnitude coverage rows use `not_applicable` rather than `misses`. No composite score is created.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Build a tidy framework-ablation table from synthetic benchmark outputs.

The ablation compares five explicitly named framework variants and records
per-unit outcomes without creating a composite score. Calibration and
held-out evaluation rows are preserved; ``final_split`` only marks which
rows should be used for final reporting.
```

</details>

### `scgeo.bench.plot_framework_ablation`

Plot held-out framework-ablation summaries without mixing unavailable evidence with performance outcomes.

```text
scgeo.bench.plot_framework_ablation(
    ablation_table: 'pd.DataFrame',
    *,
    split: 'str' = 'evaluation',
    normalize: 'bool' = True,
    figsize='auto',
    row_height: 'float' = 0.3,
    min_height: 'float' = 3.2,
    wrap_width: 'int' = 18,
    show_values: 'bool | str' = 'auto',
    title: 'Optional[str]' = None,
    save_path=None,
    show: 'bool' = True,
)
```

[View implementation](../scgeo/bench/_simulation.py)

- Output: compact matplotlib Figure with a fixed 5 x 5 capability table, a curated applicable-performance heatmap, and a separate support-status heatmap. `figsize='auto'` sizes height from row counts and row height, and panel widths from wrapped label lengths within fixed caps. Unsupported cells are blank; `not_computed`, `not_applicable`, and scientifically irrelevant scenario/failure-mode rows are not plotted on the performance color scale. Plotting metadata is attached to the returned Figure.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot a manuscript-ready framework-ablation summary.

``figsize='auto'`` sizes the figure from performance/support row counts,
row height, the documented minimum height, and wrapped label lengths.
The figure separates capability coverage, applicable performance outcomes,
and support statuses so unavailable or not-computed evidence is not plotted
on the same color scale as performance.
```

</details>

### `scgeo.bench.run_simulation_suite`

Run a reproducible synthetic ScGeo benchmark suite.

```text
scgeo.bench.run_simulation_suite(
    *,
    profile: 'str' = 'smoke',
    scenarios=None,
    seeds=None,
    output_dir=None,
    resume: 'bool' = True,
    n_jobs: 'int' = 1,
) -> 'dict[str, Any]'
```

[View implementation](../scgeo/bench/_simulation.py)

- Profiles: `smoke` is approximately 1,000-2,000 cells/job with 25 bootstrap iterations and `k=(15,)`; `quick` is approximately 3,000-5,000 cells/job with 100 bootstrap iterations and `k=(15, 30)`; `manuscript` is approximately 5,000-10,000 cells/job with 300 bootstrap iterations and `k=(15, 30, 50)`.
- Calibration and held-out evaluation seeds are tracked separately. The suite exports threshold-sensitivity tables, the complete long-form framework-ablation table, seed/job-level ablation summaries, and final held-out summaries that report mean, median, spread, and confidence intervals across held-out jobs when the number of jobs permits. The suite does not modify consensus thresholds automatically.
- Synthetic benchmarks are methodological checks, not experimental validation.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Run a reproducible synthetic ScGeo benchmark suite.

Calibration and evaluation seeds are kept separate in all result tables.
The suite does not alter consensus thresholds automatically; threshold
sensitivity tables are exported for downstream calibration analyses. Final
ablation summaries aggregate outcomes within simulation seed/job before
summarizing across held-out jobs.
```

</details>

### `scgeo.bench.simulate_perturbation_geometry`

Generate a synthetic perturbation-geometry benchmark AnnData object.

```text
scgeo.bench.simulate_perturbation_geometry(
    *,
    scenario: 'str' = 'centroid_shift',
    n_states: 'int' = 5,
    n_samples_per_condition: 'int' = 4,
    cells_per_sample: 'int' = 400,
    latent_dim: 'int' = 8,
    effect_size: 'float' = 1.0,
    affected_states=None,
    outlier_fraction: 'float' = 0.0,
    outlier_scale: 'float' = 10.0,
    sample_heterogeneity: 'float' = 0.15,
    abundance_effect: 'float' = 0.0,
    covariance_effect: 'float' = 0.0,
    warp_strength: 'float' = 0.0,
    velocity_mode=None,
    seed: 'int' = 0,
) -> 'ad.AnnData'
```

[View implementation](../scgeo/bench/_simulation.py)

- Scenarios: `null_effect`, `centroid_shift`, `abundance_only`, `covariance_only`, `local_warp`, `outlier_contamination`, `unequal_cell_counts`, `balanced_replicate_heterogeneity`, `batch_condition_confounding`, `representation_corruption`, `aligned_dynamics`, `discordant_dynamics`. The legacy input aliases `null` and `replicate_heterogeneity` are accepted but exported as `null_effect` and `balanced_replicate_heterogeneity`, respectively.
- Replicate designs: `balanced_replicate_heterogeneity` draws random sample offsets and exactly zero-centers them within each condition, so the expected outcome is wider uncertainty with a low neutral-state false-call rate. `batch_condition_confounding` applies a systematic sample/batch offset correlated with condition and is marked non-identifiable without additional design information; observed condition deltas are not counted as ordinary biological shifted-state detection successes or false positives.
- Output: synthetic AnnData with `obs['state']`, `obs['condition']`, `obs['sample']`, representation ensemble, optional velocity embeddings, and `adata.uns['simulation_truth']`.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Generate a synthetic perturbation-geometry benchmark AnnData object.

The simulation is synthetic only. It stores known truth under
``adata.uns['simulation_truth']`` and creates multiple latent
representations without requiring Scanpy, scVelo, or GPU libraries.
```

</details>
