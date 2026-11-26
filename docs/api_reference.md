# API reference

Public API for **ScGeo 0.2.0** · **104 exported callables**.

This reference describes the current checkout. Plotting additions marked [unreleased in the changelog](../CHANGELOG.md) require that development checkout; they are not included in the published 0.1.0 distribution.

[Quick start](QUICKSTART.md) · [Installation](INSTALLATION.md) · [Interpretation](RESULT_INTERPRETATION.md)

Start with `tl.robust_shift`, `get.state_report` and `pl.state_evidence_panel`. Add `tl.representation_stability` and `tl.local_geometry_stability` when the required representations are available.

| Namespace | Purpose | Browse separately |
|---|---|---|
| [`scgeo.tl`](#scgeotl) | Analysis | [21 callables](api/tl.md) |
| [`scgeo.pl`](#scgeopl) | Figures | [50 callables](api/pl.md) |
| [`scgeo.get`](#scgeoget) | Reports and tables | [10 callables](api/get.md) |
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

[View implementation](../scgeo/tl/_explain_shift.py)

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
- [`shift_curves`](#scgeoplshift_curves)
- [`shift_embedding`](#scgeoplshift_embedding)
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

[View implementation](../scgeo/pl/_result.py)

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

Plot stored alignment scores and their full-data histogram.

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
    axes=None,
    bins=40,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: 'bool' = True,
    max_points=None,
    random_state=0,
    rasterized='auto',
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_alignment_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored alignment scores and their full-data histogram.

Highlights share the embedding color normalization. ``axes`` accepts named
``embedding`` and ``histogram`` axes, plus ``colorbar`` unless
``colorbar=False``. Supplied figures retain their size and layout.
``max_points`` affects background drawing only and must accommodate every
drawable top-k cell. The histogram always uses all finite input scores.
No analysis or biological interpretation is recomputed.

The legacy return is ``(fig, axes_array)``. ``return_data=True`` adds cell
and histogram tables. ``return_result=True`` supplies a PlotResult with
exact bin edges/counts, cell identities, selections, and displayed flags.
```

</details>

### `scgeo.pl.ambiguity_panel`

Plot stored ambiguity scores and their full-data histogram.

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
    axes=None,
    bins=40,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: 'bool' = True,
    max_points=None,
    random_state=0,
    rasterized='auto',
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_ambiguity_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored ambiguity scores and their full-data histogram.

Highlights share the embedding color normalization. ``axes`` accepts named
``embedding`` and ``histogram`` axes, plus ``colorbar`` unless
``colorbar=False``. Supplied figures retain their size and layout.
``max_points`` affects background drawing only and must accommodate every
drawable top-k cell. The histogram always uses all finite input scores.
No analysis or biological interpretation is recomputed.

The legacy return is ``(fig, axes_array)``. ``return_data=True`` adds cell
and histogram tables. ``return_result=True`` supplies a PlotResult with
exact bin edges/counts, cell identities, selections, and displayed flags.
```

</details>

### `scgeo.pl.category_palette`

Return a stable label-to-color mapping, independent of subset ordering.

```text
scgeo.pl.category_palette(labels, palette=None)
```

[View implementation](../scgeo/pl/_style.py)


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

[View implementation](../scgeo/pl/_composition_drift.py)


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

[View implementation](../scgeo/pl/_perturbation_report.py)


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

Highlight global top-k or threshold-selected scores with complete group and sample denominators.

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
    colorbar: 'bool' = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_result: 'bool' = False,
    selection: 'str' = 'topk',
    threshold=None,
    sample_key=None,
    max_points=None,
    random_state=0,
    rasterized='auto',
)
```

[View implementation](../scgeo/pl/_consensus_structure.py)

- selection='topk' preserves the default rule; selection='threshold' requires a finite threshold and allows an empty selection. Missing/nonfinite scores are not selected; finite-coordinate filtering changes display only.
- Face and outline opacity use separate artists and share the declared numerical color scale. Group and optional sample summaries distinguish composition among selected cells from selection prevalence among all or scored cells, retaining observed zeros and missing-score coverage.
- max_points is a total display budget that preserves all drawable selected cells and group/sample strata; an insufficient budget raises with retention details. Source selection and denominators remain unchanged. The historical four-item return remains the default; return_result=True exports cells, summaries and samples.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Consensus score structure on a stored embedding, with a separate legend.

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
  - ``return_result=True`` exports all cells, selection/display flags and
    full group summaries (including groups excluded from the legend).
  - ``colorbar=False`` or ``cax`` preserves supplied panel bounds. An
    automatic colorbar reserves space from the embedding axis. Supplied
    figures are never laid out or resized. The four-item return is unchanged.

``selection='threshold'`` selects finite scores >= the explicit finite
``threshold`` and ignores ``topk``. An empty threshold selection is valid.
The default top-k rule and deterministic tie order are unchanged.
``summary.fraction`` describes the selected set's group composition;
``selected_fraction_all_cells`` and ``selected_fraction_scored_cells`` use
each group's full input or finite-score denominator. ``score_coverage`` is
the finite-score fraction. Optional ``samples`` exports the full group×sample
roster, including absent combinations with zero counts and unavailable
within-group rates. Its ``fraction`` denominator is selected cells in the
same sample. Missing sample IDs are explicitly marked, not identified as
biological samples. No statistical test or uncertainty is fitted.

``max_points`` limits visible cells only, retaining every selected cell
with finite coordinates and one cell per group (group×sample when supplied).
A budget too small for these requirements raises. Source counts, selection,
and color normalization precede display filtering. ``rasterized='auto'``
rasterizes cell layers above 2,000 displayed cells. ``hi_alpha`` controls
score faces independently of ``outline_alpha``; the score colorbar remains
opaque and uses the same normalization.
```

</details>

### `scgeo.pl.consensus_subspace_panel`

Plot stored consensus scores, their histogram, and existing singular values.

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
    axes=None,
    bins=40,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: 'bool' = True,
    max_points=None,
    random_state=0,
    rasterized='auto',
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_consensus_panel.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored consensus scores, their histogram, and existing singular values.

Uses the same composition and source-export contract as ``alignment_panel``:
named ``embedding``/``histogram`` axes, optional ``colorbar`` axis,
a shared color scale for highlights, and draw-only sampling. The histogram
uses all finite scores; singular values are read from the selected stored
consensus result and never recomputed.

The default return remains ``(fig, axes_array)``; ``return_data=True`` adds
source tables, and ``return_result=True`` returns an exportable PlotResult.
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
    ax=None,
    figsize=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = False,
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

Supplied axes retain their layout. ``return_result=True`` exports original
and ranked rows; ``return_data=True`` returns ``(ax, ranked_table)``.
The default return remains an Axes and does not call ``plt.show``.
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
    axes=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
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

``axes`` maps ``summary`` (global) or metric names (by-group) to existing
axes. ``return_result=True`` exports all stored and selected rows with
shared ordering; ``return_data=True`` adds them to the legacy tuple.
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
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: 'bool' = True,
    cax=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
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

The per-cell fallback forwards the supplied axis and color controls to
``score_embedding``. Grid exports include every pixel's value/center and
all observation coordinates; metadata records whether extent was supplied
or inferred. Missing pixels remain missing. ``return_result=True`` opts
into PlotResult, and ``return_data=True`` adds tables to the legacy tuple.
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
    axes=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
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
Named axes are ``embedding``, ``density``, ``test`` and ``score``.
``return_result=True`` exports source cells, stored summaries, exact text
and histogram bins without recalculating tests. Default returns remain
``(fig, axes_2d_array)``; ``return_data=True`` adds these source tables.
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
    axes=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    separate_metrics: 'bool' = False,
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

Compose with ``axes={'summary': ax}``. ``return_result=True`` exports
original and displayed rows without rerunning inference. The global legacy
panel displays two different quantities; its common axis does not imply
comparable units. ``return_data=True`` adds source tables to the tuple.
Use ``separate_metrics=True`` for separate ``statistic`` and ``probability``
axes in global mode; the default preserves the historical single-axis view.
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
    return_result: 'bool' = False,
    p_threshold: 'float | None' = 0.05,
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

Missing values remain unavailable. Finite probabilities must lie in [0, 1];
only values below 1e-300 are clipped for the log display. ``p_threshold``
controls the descriptive reference line (None disables it), not a new test.
The legacy return is ``(fig, ax, source_table)``. ``return_result=True``
provides original rows, exact plotted coordinates, labels and omissions.
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

[View implementation](../scgeo/pl/_density.py)

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
    axes=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../scgeo/pl/_embedding_facets.py)

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

``axes`` accepts a mapping named ``panel_00``, ``panel_01``, ... in facet
order, all in one Figure or SubFigure. A supplied canvas is neither resized
nor laid out and its title is preserved. Optional ``legend``, ``colorbar``
(horizontal observation scale), and ``velocity_colorbar`` (vertical) axes
place those artists explicitly. Enabled numeric scales require their
colorbar axes when composing; alternatively set ``show_colorbar=False``.
Without a supplied legend axis, the first facet contains the legend.
``result.axes`` retains only the facet panels; auxiliary artists are in
``result.artists``. ``figsize`` and ``title`` apply to owned canvases.
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

[View implementation](../scgeo/pl/_evidence_comparison.py)

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

[View implementation](../scgeo/pl/_result.py)


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

[View implementation](../scgeo/pl/_gallery.py)


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

Compose geometry and dynamics views with comparable color scales, controlled labels and explicit coverage.

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
    cmap='viridis',
    norm=None,
    vmin=None,
    vmax=None,
    max_points=6000,
    random_state=0,
    label_states=True,
    max_labels=14,
    label_mode='auto',
    label_offsets=None,
    legend='auto',
    legend_max_items=12,
    legend_loc='outside',
    layout='grid',
    show_sample_coverage=True,
    style='paper',
    figsize=None,
    title=None,
    axes=None,
    show_colorbar=True,
    return_result=True,
    show=True,
)
```

[View implementation](../scgeo/pl/_geometry_dynamics_atlas.py)

- Supports caller-owned Figure/SubFigure axes and opt-in adaptive layout. Shared cmap/norm/vmin/vmax use full input values independently of background display sampling.
- State labels retain explicit highlights before filling max_labels. Label offsets and omission metadata remain auditable. Long legends can be truncated with a visible notice and complete exported category table; a supplied legend axis can hold the full legend.
- Optional sample coverage is preserved for supplied axes and uses a reserved footer for an owned figure. Labels, legend entries, sample coverage and panel source tables are exported without refitting stored scientific evidence.


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
samples are available, ``sample_key`` exports descriptive counts and
``show_sample_coverage=True`` displays them even on supplied axes, without
inventing statistical tests. ``max_points`` affects drawing only.

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

Compose with ``axes={panel_name: ax, ...}`` for the selected panels, all
in one Figure or SubFigure. Supplied figures retain their size, layout,
and title. For a numeric states panel, supply ``axes['colorbar']`` or use
``show_colorbar=False``. The optional colorbar artist is available through
``result.artists['states_colorbar']``. ``figsize``/``title`` apply only to
a canvas created by this function.

Numeric states use ``cmap`` and either ``norm`` or ``vmin``/``vmax`` for
shared figure scales. Automatic normalization uses all finite input scores,
before drawing subsampling. ``max_labels`` limits ordinary state labels;
explicit highlighted states are always retained when their coordinates are
finite, even if they exceed that budget. Label anchors are unchanged state
medians; ``label_offsets`` supplies manual point offsets and ``label_mode``
selects deterministic collision reduction (``'auto'``) or ``'on_data'``.

``legend='auto'`` shows at most ``legend_max_items`` categories with a
visible omitted-count notice and a full exported legend table. ``True``
shows all categories; ``False`` hides the legend. A supplied ``axes['legend']``
can host it separately; otherwise ``legend_loc='outside'`` places it below
the states panel. ``layout='adaptive'`` opts into a compact evidence strip
under three or more panels. The default grid and supplied canvas geometry
are preserved.
```

</details>

### `scgeo.pl.geometry_velocity_concordance`

Compare compatible stored displacement and alignment while retaining missing evidence and source support.

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
    condition_key=None,
    group0=None,
    group1=None,
    node_key=None,
    min_cells=None,
    min_samples=None,
    positive_threshold=0.5,
    opposed_threshold=-0.25,
    sample_count_keys=('n_mice_control', 'n_mice_irradiated'),
    show_ci=True,
    highlight_states=None,
    max_labels=16,
    label_mode='auto',
    label_offsets=None,
    axes=None,
    figsize=(12.0, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
)
```

[View implementation](../scgeo/pl/_geometry_velocity_concordance.py)

- Validates available representation, ordered contrast and estimator metadata across both tables without modifying them. Missing metadata remains explicitly unverified; optional caller declarations do not verify source provenance.
- Effect and alignment eligibility are separate. Explicit or recorded min_cells/min_samples apply to the corresponding source tables. Unavailable rows remain visible; zero shift or zero velocity has no defined alignment. Known descriptive-only results have no inferential interval and normalized effects never borrow raw-magnitude intervals.
- label_key supplies display text while state IDs remain join keys. max_labels is a deterministic soft budget that retains all requested highlights; omissions and overflow are exported. Distinct caller-owned axes preserve the parent canvas.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Show stored displacement and alignment with separate availability.

Recorded provenance is checked within and between both tables. Optional
representation/contrast declarations must agree with recorded metadata;
missing metadata remains unverified. No embedding coordinates are used.

The left panel shows states with both effects and alignment available. The
right panel retains every state, including valid displacement estimates
with missing velocity and named unavailable results. ``effect_status``
takes precedence over a combined ``coverage_status`` when present.
Optional ``min_cells``/``min_samples`` and recorded source support floors
are enforced against recorded counts; unknown counts do not satisfy a
requested floor. No count threshold is invented when none is supplied.

Raw and normalized effects use only their own named stored intervals.
Known descriptive-only results retain their estimates but not confidence
intervals. These displays do not create hypotheses, bootstrap replicates,
uncertainty or recovery claims. The thresholds classify cosine values
descriptively, not statistically.

``label_key`` supplies display text while stable state IDs remain the join
keys. ``max_labels`` is a soft budget: explicit highlights are retained,
then remaining states are ordered by effect and stable ID. ``label_offsets``
uses state IDs and point units. Caller-owned axes retain their layout.
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
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    background: 'bool' = True,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
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

``norm`` or ``vmin``/``vmax`` can match an existing score layer. Without
these, the scale uses the selected finite scores. ``background=False``
adds only the highlighted layer. ``cax`` places an optional colorbar.
``topk=0`` selects no cells. Ranking uses finite scores from all input
observations; selected cells with nonfinite coordinates remain in the
source table but cannot be drawn. Ties follow input order deterministically.
The default return is ``(fig, ax)``; ``return_data=True`` adds the full cell
table, and ``return_result=True`` enables checksummed PlotResult exports.
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
    ax=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_legend.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Render a tiny legend-only figure from legend_data produced by sg.pl.consensus_structure(...).

legend_data items:
  {"group": str, "fraction": float, "count": int, "color": <matplotlib color>}

``ax`` accepts a caller-owned legend area. Source exports retain every
supplied entry, with a displayed flag when ``max_items`` truncates it.
The default return remains ``(fig, ax)``.
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

[View implementation](../scgeo/pl/_perturbation_report.py)


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
    ax=None,
    colorbar: 'bool' = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_mapping_qc.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Show stored confidence and the lowest finite scores on any embedding.

Lowest-score selection uses all finite input scores, with input order for
ties. Selected cells lacking finite coordinates remain in the source table.
The legacy return remains ``(fig, array([ax]))``; ``return_data`` appends
the cell table and ``return_result`` enables figure/source exports.
```

</details>

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
    axes=None,
    colorbar: 'bool' = True,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_mapping_qc.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compose predictions, confidence and OOD scores from stored obs columns.

``axes`` names are ``prediction``, ``confidence``, ``ood``; when colorbars
are enabled on supplied axes, also provide ``confidence_colorbar`` and
``ood_colorbar``. Reference colors are reused for matching predictions;
novel labels receive deterministic colors and are never omitted. True
missing values remain separate from literal labels such as ``'nan'``.
Query filtering applies only to predictions; both scalar panels use all
input observations. No mapping model or threshold is computed here.

Legacy ``(fig, axes)`` and ``return_legend_data`` returns are preserved.
``return_data`` appends all source tables; ``return_result`` takes priority.
```

</details>

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
    ax=None,
    colorbar: 'bool' = True,
    cax=None,
    cmap: 'str' = 'viridis',
    norm=None,
    vmin=None,
    vmax=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_mapping_qc.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Display stored OOD scores or cells above an explicitly supplied threshold.

No threshold is inferred: ``threshold=None`` draws all cells, as before,
and exports nullable flags. With ``show_only_flagged=True`` and a threshold,
the legacy monochrome flagged-cell view is retained (no numeric colorbar).
Default returns ``(fig, array([ax]))``; data/results are opt-in.
```

</details>

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
    axes=None,
    colorbar: 'bool' = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_result: 'bool' = False,
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
axes
    Optional mapping containing ``landscape`` and, with groupby, ``summary``.
    Add ``colorbar`` to place the scale explicitly. Do not combine with ax.
return_result
    Export source cells, full group summaries, actual contour bin means,
    counts and edges. Contours/thresholds are descriptive, not a test.
```

</details>

### `scgeo.pl.paga_composition_bar`

Display stored composition effects with explicit unavailable entries.

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
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Display stored composition effects with explicit unavailable entries.

Rank by p/q then distance from the null, or ``sort_by='abs_effect'``.
Missing effects occupy labelled rows without a zero-length bar; missing
p-values are labelled unavailable. Original-scale OR effects use one as
their null and bar baseline. No statistics or intervals are recomputed.
``return_result=True`` exports the untouched source, full node table with
selection flags, displayed row order and stored analysis parameters.
Legacy returns remain ``(fig, axes_array)``; ``return_data=True`` adds tables.
```

</details>

### `scgeo.pl.paga_composition_panel`

Compose the existing composition volcano and effect bars.

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
    axes=None,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Compose the existing composition volcano and effect bars.

Supply named ``volcano`` and ``bar`` axes from one Figure or SubFigure.
Caller-owned size/layout are preserved. The default return remains
``(fig, axes_array)``; opt into source tables with ``return_data=True`` or
an exportable PlotResult with ``return_result=True``. Missing evidence is
retained and labelled by each child renderer, never refitted or filled.
```

</details>

### `scgeo.pl.paga_composition_volcano`

Plot stored composition effects against their supplied p/q values.

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
    return_data: 'bool' = False,
    return_result: 'bool' = False,
)
```

[View implementation](../scgeo/pl/_paga_composition.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot stored composition effects against their supplied p/q values.

Missing effects or p-values are omitted and counted, never filled with zero
or one. If no p/q column exists, show an explicitly labelled effect-only
strip whose constant ordinate has no inferential meaning. Zero p-values
use a display floor of 1e-300; original values remain in exported sources.
Probabilities outside [0, 1] and infinite effects raise before drawing.

Auto effect selection accepts logOR, beta, effect, coef, then OR. Stored
odds ratios remain on their original scale with a null reference at one;
other effects use zero. No p-values, FDR or confidence intervals are fitted.
``return_result=True`` provides original source rows, transformed points,
selection flags and stored analysis parameters. ``return_data=True`` adds
these tables to the legacy ``(fig, axes_array)`` return.
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

[View implementation](../scgeo/pl/_paga_scgeo.py)

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
    node_position: 'str' = 'reference',
    edge_lw: 'float' = 2.0,
    edge_alpha: 'float' = 0.32,
    arrow_width: 'Optional[float]' = None,
    arrow_alpha: 'float' = 0.95,
    arrow_scale: 'float' = 1.0,
    show_shift: 'bool' = True,
    label: 'bool' = True,
    label_top_n: 'Optional[int]' = None,
    label_fontsize: 'int' = 8,
    palette: 'Optional[dict[str, Any]]' = None,
    pie_key: 'Optional[str]' = None,
    pie_categories: 'Optional[list[str]]' = None,
    pie_palette: 'Optional[dict[str, Any]]' = None,
    pie_size_scale: 'float' = 1.0,
    pie_mode: 'str' = 'cell_fraction',
    pie_sample_key: 'Optional[str]' = None,
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

[View implementation](../scgeo/pl/_paga_shift_map.py)

- These upgrades are unreleased development features; see the [PAGA shift-map guide](https://github.com/liuifrec/scgeo/blob/main/docs/PAGA_SHIFT_MAP.md) and install the matching checkout.
- Node positions and condition arrows are descriptive mean/median summaries in the chosen display. Use alignment_source="stored" and analysis_rep to colour nodes from stored quantitative alignment; display coordinates do not redefine those values.
- PAGA connectivity is undirected. Supply graph_node_order when matrix labels differ from the current category order; graph shape, weights, symmetry, and group identity are validated.
- Projected velocity must match the display basis. Velocity and composition summaries use only group0/group1; background="all" can retain other cells as context.
- Equal aspect, distinct condition endpoints, point-sized glyphs, automatic labels, and graphical legends are the new defaults. Explicit arrow_width retains the legacy data-unit style.
- return_result=True returns PlotResult with named map axis and full cell/edge/vector sources. Legacy owned (fig, ax), supplied ax, and (fig, ax, data) returns remain supported.
- velocity_mode="state" preserves the original aggregate arrows. Opt-in "grid"/"stream" use selected-condition local fields; velocity_options controls finite-radius support and rendering, with full velocity_* source tables.
- node_position="reference" preserves the legacy reference-anchored layout. node_position="pooled" places topology, node glyphs and pies at observed pooled selected-condition centers even with sparse or absent reference cells. Actual condition centers and displacement-arrow origins remain unchanged; show_shift=False supports composition-only views.
- pie_key selects categorical sectors. pie_mode="cell_fraction" retains raw captured-cell proportions. pie_mode="sample_abundance", with pie_key equal to condition_key and pie_sample_key supplied, averages state frequencies equally across valid samples within each condition and normalizes condition means for plotting. It exports raw counts, sample denominators, means and plotted values; these sectors are not raw cell fractions or inferential evidence.
- Zero-state samples with positive denominator contribute frequency zero; zero denominator is undefined. The supplied AnnData defines the denominator population and observed sample roster. Apply focus_states for display rather than subsetting away denominator cells or absent-state samples.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Draw undirected PAGA topology and condition contrasts on an embedding.

Positions, mean/median cell centers, and arrows describe the first two
display coordinates. They are not inferred cell trajectories. Reference
anchors are open markers; comparison endpoints are filled and remain at
their actual centers even when ``arrow_scale`` changes the displayed vector.
States present in only one condition or below ``min_cells`` remain visible.
By default graph edges require usable reference anchors at both ends.
``node_position='pooled'`` instead places topology nodes and pies at the
pooled selected-condition cell centers. Every observed pooled node remains
drawable, including a state absent from one condition; low pooled coverage
is flagged in ``node_anchors``. Actual x0/x1 centers and condition vectors
are never re-anchored. ``show_shift=False`` omits condition-shift arrows;
combine with ``show_endpoints=False`` for a composition-only topology view.

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
``pie_mode='cell_fraction'`` retains the captured-cell composition.
``pie_mode='sample_abundance'`` requires ``pie_key=condition_key`` and
``pie_sample_key``. It first computes state fractions within each
sample×condition over annotated cells, averages those fractions equally
within each condition, then normalizes the condition means into pie slices.
This is a relative sample-abundance profile, not within-node cell fractions.
Zero-state samples contribute zeros; sample units with no annotated cells
retain unavailable denominators. All selected sample units are retained
before focus or display filtering. ``pie_values`` and ``pie_sample_counts``
expose the actual slice values and their denominators; ``composition_counts``
always retains the raw cell counts/fractions. These summaries are descriptive.

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

[View implementation](../scgeo/pl/_perturbation_report.py)

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

Combine a state map with descriptive display geometry or stored evidence, explicit source support, and individual-sample summaries.

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
    fill_vmin: 'Optional[float]' = None,
    fill_vmax: 'Optional[float]' = None,
    ring_mode: 'str' = 'ood_frac',
    ring_color: 'str' = 'gold',
    ring_max_lw: 'float' = 4.0,
    arrow_color_mode: 'str' = 'shift',
    arrow_color: 'str' = 'black',
    arrow_cmap: 'str' = 'magma',
    arrow_scale: 'float' = 1.0,
    arrow_width: 'Optional[float]' = None,
    edge_alpha: 'float' = 0.45,
    edge_lw: 'float' = 2.0,
    bg_size: 'float' = 5.0,
    bg_alpha: 'float' = 0.08,
    label: 'bool' = True,
    label_top_n: 'Optional[int]' = 12,
    label_fontsize: 'int' = 8,
    legend: 'bool' = True,
    title: 'Optional[str]' = None,
    figsize=None,
    ax=None,
    return_data: 'bool' = False,
    show: 'bool' = True,
    caxes=None,
    graph_node_order=None,
    return_result: 'bool' = False,
    mode: 'str' = 'display',
    shift_df=None,
    alignment_df=None,
    analysis_rep=None,
    effect_metric: 'str' = 'auto',
    report_state_key=None,
    alignment_state_key=None,
    velocity_scope: 'str' = 'contrast',
    node_position: 'str' = 'reference',
    sample_key=None,
    ood_threshold=None,
    ood_scope: 'str' = 'contrast',
    fill_norm=None,
    arrow_norm=None,
    arrow_vmin=None,
    arrow_vmax=None,
    node_size_reference=None,
    arrow_linewidth: 'float' = 1.6,
    arrow_head_size: 'float' = 11.0,
    show_endpoints: 'bool' = True,
    label_mode: 'str' = 'auto',
    label_offsets=None,
    label_names=None,
    max_points=None,
    random_state: 'int' = 0,
    layout: 'str' = 'map',
    axes=None,
    show_colorbar=None,
    max_states: 'int' = 12,
)
```

[View implementation](../scgeo/pl/_recovery_compass.py)

- Legacy tuple returns and map-first layout remain available. mode='stored' requires shift_df and an explicit analysis_rep; alignment_df is optional. Stored values are joined by state identity and source inputs are never modified or refitted.
- Recorded representation, ordered contrast, center and other available provenance must agree. Caller declarations fill missing metadata as unverified declarations, not verified analysis provenance. Shift and velocity weighting retain their distinct meanings.
- node_position='pooled' retains observed sparse states without inventing a reference center. Source counts determine stored comparison support; current display coverage does not replace them. Missing velocity does not suppress a supported effect; known descriptive-only intervals are withheld. Raw and normalized intervals must match their effect metric.
- Display mode defaults to velocity_scope='contrast'; request 'all' explicitly for legacy pooling. Velocity must match the display basis and complete finite rows; zero shifts and zero-scale arrows have no invented direction. OOD threshold, scope and missing-score coverage are explicit, and missing OOD remains unavailable.
- layout='summary' adds effect, alignment, observed sample-frequency and current-coverage axes. Full observed sample rosters preserve state zeros with positive denominators. Shared color norms, node-size references, point-sized arrows, label controls and max_points affect rendering only.
- return_result=True exports original source tables, evidence/support flags, current coverage, sample counts/frequencies, all graph edges and display glyphs. The historical function name does not establish recovery or lineage.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot condition geometry with explicit evidence, coverage and missingness.

``mode='display'`` (default) calculates descriptive means and cosine in the
first two display coordinates. ``velocity_scope='contrast'`` now limits
velocity to the selected conditions; ``'all'`` restores the legacy scope,
and ``'group0'``/``'group1'`` select one condition. Projected velocities must
match ``basis`` and only jointly finite vector rows count toward support.
Zero displacement and ``arrow_scale=0`` produce no arrow.

``mode='stored'`` requires ``shift_df`` and ``analysis_rep``. Optional
``alignment_df`` supplies independently eligible alignment. The stored
quantities and provenance are validated, never estimated from the map.
``report_state_key``/``alignment_state_key`` resolve nonstandard state IDs.
``effect_metric='auto'`` prefers a finite normalized magnitude, then raw
magnitude. Intervals must have matching units; raw intervals never label
normalized effects. Source counts/status gate evidence independently of
current display coverage. Known low-sample or descriptive intervals are
withheld, while original estimates/endpoints remain in exported sources.

``node_position='pooled'`` retains observed condition-enriched states as
topology nodes. Actual condition centers and vector origins stay separate;
no missing reference or shift is synthesized. ``ood_scope`` defaults to
the contrast. OOD fractions divide flagged by finitely scored cells;
unmeasured states remain unavailable. ``ood_threshold`` freezes a supplied
cutoff; otherwise the selected scope's finite-score 95th percentile is
used and recorded as a descriptive threshold.

``layout='summary'`` adds effect, alignment, per-sample abundance and current
coverage panels. ``sample_key`` uses the complete selected-condition sample
roster: zero-state samples have observed zero frequency, and samples with
no annotated denominator stay missing. Without a sample key no samples or
intervals are fabricated. Supplied ``axes`` map map/effect/alignment/
abundance/coverage to distinct axes. ``max_states`` limits side panels only.

``fill_norm``/``arrow_norm`` or explicit limits support matched panels;
signed abundance defaults to a zero-centered scale. ``node_size_reference``
shares the sizing denominator. Default arrows/labels use point units;
explicit ``arrow_width`` retains legacy data units. ``max_points`` affects
background drawing only. ``show_colorbar`` is independent of ``legend``;
None preserves the legacy coupling. Caller figures retain their layout.

Default map returns are ``(fig, ax)`` or the supplied ax; ``return_data``
adds the observed-contrast node table and displayed edges as before.
Summary returns ``(fig, axes)``. ``return_result=True`` takes priority and
preserves cells/nodes/edges/graph_nodes/glyphs plus evidence and coverage
tables. No AnnData or supplied result table is modified. This plot does not
establish recovery, causality, cell trajectories or directed PAGA edges.
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

[View implementation](../scgeo/pl/_replicate_effect.py)

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

[View implementation](../scgeo/pl/_perturbation_report.py)

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
    colorbar: 'bool' = True,
    cax=None,
    legend: 'bool' = True,
    return_data: 'bool' = False,
    return_result: 'bool' = False,
    show: 'bool' = True,
)
```

[View implementation](../scgeo/pl/_score.py)


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
``colorbar=False`` suppresses the numeric scale; ``cax`` places it on an
explicit axis without changing the embedding panel's allocated space.
``legend=False`` suppresses categorical legends. Valid Scanpy category
colors are reused unless ``palette`` is explicitly supplied. Category
artists are always accessible through ``artists['categories'][label]``;
this avoids collisions with auxiliary artist names such as ``'legend'``.
True missing values receive a distinct label and never merge with a literal
category. Invalid category/norm combinations fail before creating a figure.
```

</details>

### `scgeo.pl.score_umap`

Display an observation score using the score_embedding plotting interface.

```text
scgeo.pl.score_umap(adata, score_key: 'str', **kwargs)
```

[View implementation](../scgeo/pl/_score.py)

### `scgeo.pl.shift_curves`

Plot sample curves, observed expression distributions and expression/contribution maps while retaining states with sparse condition coverage.

```text
scgeo.pl.shift_curves(
    adata,
    var_names=None,
    *,
    program_names=None,
    state=None,
    key='explain_shift',
    basis='umap',
    panels=('curve', 'expression', 'contribution'),
    coordinate_key=None,
    color=None,
    palette=None,
    cmap='viridis',
    contribution_cmap='coolwarm',
    bandwidth=0.18,
    grid_size=80,
    min_cells=5,
    uncertainty='sample_range',
    show_samples=True,
    max_points=3000,
    random_state=0,
    size=6,
    alpha=0.3,
    style='paper',
    figsize=None,
    title=None,
    axes=None,
    ax=None,
    show_colorbar=True,
    return_result=False,
    show=True,
    save=None,
)
```

[View implementation](../scgeo/pl/_shift_explanation.py)

- Unreleased development API. The default curve coordinate is a frozen, expression-derived shift score. coordinate_key can explicitly name a numeric observation coordinate without changing stored shifts or contribution coefficients; neither choice automatically supplies time or trajectory evidence.
- Compact-kernel smoothing occurs within each sample and condition, with equal weighting of supported samples. Unsupported regions remain gaps and curves do not extrapolate beyond sample support. The optional ribbon is the observed between-sample range, not a confidence interval; fewer than two supporting samples produce no range.
- Program curves show mean member expression while contribution maps sum member-gene contributions. A difference between condition curves at the same expression-derived coordinate is not a matched causal treatment effect.
- Reads stored explanations without changing AnnData; source signatures and labels reject stale inputs before drawing. Scatter subsampling changes display only. Supplied axes support composition, and return_result=True exposes complete source tables and figure exports.
- Expression and distribution panels do not require a shift direction. Unsupported requested curve/contribution panels receive availability messages. Explicit genes/programs are required when no valid contribution ranking exists, including zero-shift states.
- Distribution panels show empirical sample medians and interquartile ranges, with observed counts. The full supplied AnnData roster retains zero-cell state/sample slots as missing expression; a singleton remains one observed point. Coverage, per-sample expression and panel availability are exported. Distribution-only calls do not require an embedding.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Plot expression along a stored shift axis and coordinated embedding maps.

Each requested gene or program gets a row with ``curve``, ``expression``
and ``contribution`` panels. Use ``panels=('curve',)`` without an embedding.
``var_names=None`` selects up to three genes by absolute stored contribution.
Select one ``state`` when the explanation contains several states.

States without a supported shift can still use explicit ``var_names`` or
``program_names`` with ``panels=('distribution', 'expression')``. The
distribution panel shows each biological sample's observed expression
median and cell interquartile range, with its cell count. A singleton is
one observed point without an interval. Zero-cell samples from the full
supplied AnnData roster stay blank; missing expression is never zero.
Both declared conditions remain visible. Without ``sample_key``, these
are explicitly pooled-cell distributions. A distribution-only plot needs
no embedding. Unsupported curve/contribution panels are labelled as such.
An unsupported state has no automatic contribution-based gene selection.

``coordinate_key`` optionally names a numeric ``adata.obs`` column for
descriptive curves. It must be explicitly supplied; the plot does not
manufacture a direction or extrapolate an absent condition. The original
stored ``shift_score`` is preserved. Exports use ``curve_coordinate`` and
``coordinate_source`` to distinguish a supplied coordinate from the shift.

Curves are tricube-kernel means estimated separately in each biological
sample and condition, then averaged with equal sample weights at every
supported grid point. ``bandwidth`` is a fraction of the state's observed
score range; ``min_cells`` is the required local support in each sample.
Curves stop outside each sample's observed score range. Missing support
stays missing. Thin lines show individual samples. The default ribbon is
the observed between-sample range where at least two samples support a
point. It is *not* a confidence interval. ``uncertainty='none'`` hides it.
With no biological sample key, only a pooled descriptive curve is drawn.

The default x axis is an expression-derived geometric coordinate, not time
or an inferred trajectory. A supplied coordinate is used as provided and
has no inferred temporal or causal meaning here. Curves and ribbons are
conditional on the coordinate and smoothing choice, and cannot establish
causal drivers.
Program expression uses stored membership weights (a mean); program
contribution sums the member genes' signed axis contributions.

``max_points`` affects scatter drawing only. ``color`` optionally selects
an observation annotation for curve points; condition curves retain their
own colors. ``axes`` maps names such as ``'gene:Gata1:curve'`` to distinct
axes in one Figure. For supplied axes and ``show_colorbar=True``, also
provide ``'gene:Gata1:expression_colorbar'`` and the analogous contribution
colorbar axis. ``ax`` is a convenience for one feature and one panel.
Supplied figures retain their size, layout and title. ``return_result=True``
exposes full cell, curve, sample-curve, feature, coverage, sample-expression
and panel-status tables via ``PlotResult``.
No AnnData data are modified. Files are written only when ``save`` is set.
```

</details>

### `scgeo.pl.shift_embedding`

Display schematic curved condition contrasts using the original effective cell weights projected into the selected display embedding.

```text
scgeo.pl.shift_embedding(
    adata,
    *,
    states=None,
    key='explain_shift',
    basis='umap',
    color=None,
    palette=None,
    curvature=0.2,
    max_points=5000,
    random_state=0,
    size=8,
    alpha=0.35,
    style='paper',
    figsize=(6.0, 5.0),
    title=None,
    ax=None,
    return_result=False,
    show=True,
    save=None,
)
```

[View implementation](../scgeo/pl/_shift_explanation.py)

- Unreleased development API. Curvature is a graphic choice; arrows connect condition centers and do not estimate paths, fate probabilities, or an RNA-velocity field.
- Endpoints apply frozen analysis weights to display coordinates. A nonlinear display does not invert the analysis displacement. Zero-shift and insufficient-cell states are not assigned unsupported arrows.
- Reads stored explanations without changing AnnData, validates source identity, and keeps display subsampling separate from center calculations. return_result=True exposes center weights, endpoints, full cell sources and interpretation metadata.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Display condition-center contrasts as explicitly schematic curved arrows.

Endpoints use the exact cell weights stored by ``explain_shift`` applied
to the requested display coordinates. A nonlinear display embedding is
not an inverse mapping of the analysis displacement. Arcs are graphic
connectors between condition centers, not estimated cell trajectories or
a velocity field. Use ``scgeo.pl.velocity_field`` for independently supplied
RNA-velocity vectors. ``curvature=0`` draws straight arrows.

``color`` is a categorical observation key (default: stored condition).
``return_result=True`` exposes endpoint weights, all displayed-source cell
coordinates and the exact center table. ``max_points`` only subsamples
displayed cells; all cells retain their original center weights.
```

</details>

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

[View implementation](../scgeo/pl/_perturbation_report.py)

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

[View implementation](../scgeo/pl/_state_flow_alluvial.py)

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

[View implementation](../scgeo/pl/_state_flow_chord.py)

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

[View implementation](../scgeo/pl/_state_flow_sankey.py)

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

[View implementation](../scgeo/pl/_style.py)


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

[View implementation](../scgeo/pl/_velocity_condition_compare.py)


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

[View implementation](../scgeo/pl/_velocity_field.py)

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

[View implementation](../scgeo/pl/_velocity_shift_alignment.py)

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

## scgeo.get

**Reports and tables.** Retrieve analysis tables and combine available evidence into a state-level report.

- [`get_available_tables`](#scgeogetget_available_tables)
- [`get_composition_table`](#scgeogetget_composition_table)
- [`get_ood_summary`](#scgeogetget_ood_summary)
- [`get_robustness_table`](#scgeogetget_robustness_table)
- [`get_shift_summary`](#scgeogetget_shift_summary)
- [`get_velocity_alignment_summary`](#scgeogetget_velocity_alignment_summary)
- [`input_contract`](#scgeogetinput_contract)
- [`sample_centers`](#scgeogetsample_centers)
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

### `scgeo.get.input_contract`

Inspect explicitly configured AnnData inputs without modifying or fitting them.

```text
scgeo.get.input_contract(adata, *, config)
```

[View implementation](../scgeo/get/_input_contract.py)


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Inspect explicitly configured AnnData inputs without modifying or fitting them.

Parameters
----------
adata : AnnData
    Only obs, obsm and uns are inspected; X, layers and raw are not read.
config : dict
    JSON-compatible configuration with purpose (descriptive/sample_inference),
    state_key, condition_key, group0, group1, sample_key (or None), reps (exact
    obsm keys), methods, min_cells and min_samples_per_condition. Optional
    design='independent', bootstrap_unit, k_values, embedding_key,
    representation_provenance, projected_velocity, paga and evidence are
    described in docs/JUPYTERHUB_PREFLIGHT.md. No keys or groups are inferred.

Returns
-------
dict
    JSON-safe checks, inventory and per-state coverage. ok means the requested
    input contract passed; excluded_states remain ineligible for sample-level
    interpretation. Eligibility is a coverage screen, not significance or a
    guarantee of biological independence. Provenance assertions are checked
    for consistency, not authenticated. No results are stored in adata.
```

</details>

### `scgeo.get.sample_centers`

Return a tidy copy of the stored biological sample centers.

```text
scgeo.get.sample_centers(adata_or_result, *, store_key='robust_shift', scope='auto')
```

[View implementation](../scgeo/get/_sample_centers.py)

- Reads exact biological sample centers retained by updated sample-aware robust_shift runs. Coordinates are zero-based; cell counts are descriptive, not replicate weights.
- Old aggregate-only results raise an actionable error rather than reconstructing sample-level values.


<details>
<summary>Parameters, returns and usage notes</summary>

```text
Return a tidy copy of the stored biological sample centers.

Parameters
----------
adata_or_result
    AnnData with a stored robust-shift result, or that result dictionary.
    Old results without sample-center matrices are not recomputed silently.
store_key
    Key inside ``adata.uns['scgeo']``; ignored for a result dictionary.
scope
    ``'state'`` selects stratified results, ``'global'`` selects the global
    result, and ``'auto'`` selects states when available, otherwise global.

Returns
-------
pandas.DataFrame
    Columns ``scope``, ``state``, ``condition``, ``sample``, ``coordinate``,
    ``value``, ``cell_count``, ``representation``, and ``center``. Coordinates
    are zero-based. Each sample has one row per representation coordinate;
    ``cell_count`` is descriptive and must not be used as a replicate weight.
    The global state label is ``'__global__'``. Provenance is in ``attrs``.
```

</details>

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
- Metadata in `report.attrs` includes package version, source store keys, comparison identity, the effect representation from robust_shift, the representation ensemble, k values, sample key, center estimator, bootstrap unit, consensus rules, global diagnostics, representation diagnostics, warnings, and creation timestamp.
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
