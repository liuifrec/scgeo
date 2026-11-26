# scgeo.pp

ScGeo 0.2.0 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
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

[View implementation](../../scgeo/pp/_reference_pool.py)


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

[View implementation](../../scgeo/pp/_knn_graph.py)


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

[View implementation](../../scgeo/pp/_knn_graph.py)


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

[View implementation](../../scgeo/pp/_census_knn.py)


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

[View implementation](../../scgeo/pp/_reference_pool.py)

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

[View implementation](../../scgeo/pp/_reference_pool.py)


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
