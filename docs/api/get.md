# scgeo.get

ScGeo 0.2.1 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
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

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.get_composition_table`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_composition_table(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.get_ood_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_ood_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.get_robustness_table`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_robustness_table(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.get_shift_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_shift_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.get_velocity_alignment_summary`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.get.get_velocity_alignment_summary(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'
```

[View implementation](../../scgeo/get/_figure_tables.py)

### `scgeo.get.input_contract`

Inspect explicitly configured AnnData inputs without modifying or fitting them.

```text
scgeo.get.input_contract(adata, *, config)
```

[View implementation](../../scgeo/get/_input_contract.py)


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

[View implementation](../../scgeo/get/_sample_centers.py)

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

[View implementation](../../scgeo/get/_state_report.py)

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

[View implementation](../../scgeo/get/_table.py)


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
