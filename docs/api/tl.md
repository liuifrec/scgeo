# scgeo.tl

ScGeo 0.2.1 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
**Analysis.** Measure displacement, representation stability, local geometry, distributional change and dynamics agreement.

- [`align_vectors`](#scgeotlalign_vectors)
- [`alignment_driver_genes`](#scgeotlalignment_driver_genes)
- [`analyze_shift`](#scgeotlanalyze_shift)
- [`consensus_subspace`](#scgeotlconsensus_subspace)
- [`density_overlap`](#scgeotldensity_overlap)
- [`distribution_test`](#scgeotldistribution_test)
- [`explain_shift`](#scgeotlexplain_shift)
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

[View implementation](../../scgeo/tl/_align_vectors.py)

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

[View implementation](../../scgeo/tl/_alignment_driver_genes.py)

**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.

Probe note: needs domain-specific inputs

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Identify genes associated with alignment or discordance between geometric shift
and velocity direction.

This legacy helper performs cell-level DE of supplied labels; it does not
account for biological samples or explain a displacement. Labels propagated
from nodes may identify node markers. Use ``explain_shift`` for descriptive
molecular attribution to a supported affine state shift.

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

[View implementation](../../scgeo/tl/_analyze_shift.py)

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

[View implementation](../../scgeo/tl/_consensus_subspace.py)

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

[View implementation](../../scgeo/tl/_density_overlap.py)

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

[View implementation](../../scgeo/tl/_distribution_test.py)

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

### `scgeo.tl.explain_shift`

Account for a stored sample-aware displacement using signed gene and program contributions in a verified affine expression representation.

```text
scgeo.tl.explain_shift(
    adata,
    *,
    shift_key='robust_shift',
    layer=None,
    loadings_key='PCs',
    loadings=None,
    feature_scale=None,
    states=None,
    modules=None,
    local_key=None,
    representation_key=None,
    regulons=None,
    regulator_source=None,
    min_cells=10,
    min_samples=2,
    n_boot=200,
    paired=False,
    seed=0,
    mapping_rtol=1e-05,
    mapping_atol=1e-06,
    zero_tol=1e-08,
    store_key='explain_shift',
)
```

[View implementation](../../scgeo/tl/_explain_shift.py)

- Unreleased development API. Requires a stored robust_shift result with a mean or geometric-median center, the exact expression matrix, and a matching affine loading/scale map. Ordinary PCA loadings cannot explain integrated or nonlinear coordinates unless that exact mapping is verified.
- Contributions use the original numerical center weights and sum to the measured displacement norm within the declared numerical tolerance. Positive and negative contributions support or oppose the chosen direction; they do not establish upstream causal drivers.
- Bootstrap intervals resample biological samples conditional on the frozen representation and shift direction. Insufficient samples, insufficient cells, zero shifts, missing comparisons, and unassessed representation evidence remain explicit.
- The output includes state-wide and optional local gene attribution, program summaries, sample centers, continuous cell scores, composition tables, supplied-regulon evidence and source-integrity signatures. Local parent-axis contributions and abundance changes have distinct meanings.
- Writes the complete result only to adata.uns['scgeo'][store_key]. The synthetic I/O probe prepares a valid sample-aware robust_shift before its snapshot, so the existing top-level scgeo key is present before and after; the probe does not enumerate nested writes.

**Observed AnnData keys in the synthetic I/O probe**

| Field | Added | Present before and after | Removed |
|---|---|---|---|
| `obs_cols` | — | `cell_type`, `condition`, `sample` | — |
| `obsm_keys` | — | `X_pca` | — |
| `layers_keys` | — | `None` | — |
| `uns_keys` | — | `scgeo` | — |

<details>
<summary>Parameters, returns and usage notes</summary>

```text
Explain a stored robust shift with signed gene and program contributions.

Parameters
----------
adata
    AnnData containing a result from :func:`scgeo.tl.robust_shift`.
shift_key
    Source under ``adata.uns['scgeo']``. Its state, contrast, representation,
    sample key and center estimator are reused and checked against the data.
    Supported estimators are ``mean`` and ``geometric_median``.
layer
    Exact expression input for the affine representation; None uses X.
    Sparse inputs stay sparse. Ordinary PCA loadings cannot explain a
    nonlinear or batch-corrected embedding unless its map is verified.
loadings_key, loadings
    Gene-by-coordinate coefficients, from varm[loadings_key] by default.
    An explicit array overrides the key. Extra trailing PCA columns are
    ignored; all coordinates of the source representation are explained.
feature_scale
    Optional positive per-gene divisors applied to loading rows. A common
    affine intercept is inferred and verified, so centering is supported.
states
    Optional state subset. With no source ``by``, the sole state is 'all'.
modules
    Mapping of program names to unique member genes. Program expression is
    their mean; program contribution is their sum. Overlapping programs
    do not partition the displacement and are not percentages.
local_key
    Optional obs column of independently justified local identity groups.
    Local contrasts use their own direction; projection onto the parent
    direction is also reported. Per-sample proportions are retained,
    including zero-count local groups. No cell-to-cell transitions or
    exact additive composition decomposition are inferred.
representation_key
    Optional compatible representation_stability result. Its evidence
    labels are carried forward, never inferred from one representation.
regulons, regulator_source
    Optional regulator -> {target: signed weight} mapping and required
    evidence provenance. Returns signed target-response summaries, not
    inferred networks, simulated interventions or causal-driver scores.
min_cells, min_samples
    Coverage thresholds per condition. min_samples must be at least two.
    Missing biological replication retains descriptive attribution with
    status 'insufficient_samples' and no bootstrap interval.
n_boot, paired, seed
    Biological-sample bootstrap draws, optionally matched by sample ID.
    Shared IDs require paired=True; paired analyses require equal ID sets.
    Intervals condition on the observed representation and shift direction;
    they are descriptive stability intervals, not selection-adjusted tests.
mapping_rtol, mapping_atol, zero_tol
    Mapping/reconstruction tolerances and absolute zero-shift threshold.
    Mapping relative tolerance uses centered coordinate variation.
store_key
    New result key under adata.uns['scgeo']; cannot overwrite a source key.

Returns
-------
dict
    ``summary``, ``genes``, ``programs``, ``cells``, ``samples``,
    ``local_summary``, ``local_genes``, ``composition`` and ``regulators``
    tables, memberships, mapping diagnostics and numerical center weights.
    ``state_results`` uses safe internal keys and stores each state label.

Notes
-----
For matched robust expression contrast d, affine loading L and observed
unit direction u, C_g = d_g (L u)_g and sum(C_g) = ||delta|| up to the
exported numerical residual. This is descriptive molecular accounting.
Genes help define the axis, so this is not independent association evidence.
Zero shifts have no inferred direction; nonzero local contrasts can still
be reported. Existing robust-shift estimates and input matrices are unchanged.
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

[View implementation](../../scgeo/tl/_local_geometry_stability.py)

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

[View implementation](../../scgeo/tl/_map_knn.py)

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

[View implementation](../../scgeo/tl/_map_query_to_ref.py)

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

[View implementation](../../scgeo/tl/_map_query_to_ref_pool.py)

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

[View implementation](../../scgeo/tl/_map_query_to_ref_pool_census.py)

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

[View implementation](../../scgeo/tl/_mixscore.py)

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

[View implementation](../../scgeo/tl/_paga_composition_stats.py)

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

[View implementation](../../scgeo/tl/_projection_disagreement.py)

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

[View implementation](../../scgeo/tl/_representation_stability.py)

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

[View implementation](../../scgeo/tl/_robust_shift.py)

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
The exact sample centers are retained for ``sg.get.sample_centers`` and
descriptive replicate plots. This additive output does not change the
estimator or bootstrap sequence.

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

[View implementation](../../scgeo/tl/_shift.py)

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

[View implementation](../../scgeo/tl/_velocity_delta_alignment.py)

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

[View implementation](../../scgeo/tl/_velocity_shift_alignment.py)

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

[View implementation](../../scgeo/tl/_wasserstein.py)

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
