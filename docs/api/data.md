# scgeo.data

ScGeo 0.2.0 · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)

Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).

Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).
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

[View implementation](../../scgeo/data/_census.py)


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

[View implementation](../../scgeo/data/_census.py)

### `scgeo.data.census_find_nearest_obs`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_find_nearest_obs(*args, **kwargs)
```

[View implementation](../../scgeo/data/_census.py)

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

[View implementation](../../scgeo/data/_census.py)


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

[View implementation](../../scgeo/data/_census.py)

### `scgeo.data.census_get_embedding_metadata_by_name`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_get_embedding_metadata_by_name(*args, **kwargs)
```

[View implementation](../../scgeo/data/_census.py)

### `scgeo.data.census_predict_obs_metadata`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.census_predict_obs_metadata(*args, **kwargs)
```

[View implementation](../../scgeo/data/_census.py)

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

[View implementation](../../scgeo/data/_census.py)

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

[View implementation](../../scgeo/data/_census.py)

### `scgeo.data.find_nearest_obs`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.find_nearest_obs(*args, **kwargs)
```

[View implementation](../../scgeo/data/_census.py)

- Backward-compatible alias; prefer the corresponding census_-prefixed function in new workflows.

### `scgeo.data.get_embedding_metadata_by_name`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.get_embedding_metadata_by_name(*args, **kwargs)
```

[View implementation](../../scgeo/data/_census.py)

- Backward-compatible alias; prefer the corresponding census_-prefixed function in new workflows.

### `scgeo.data.open_census`

Public ScGeo helper; inspect the signature and source for supported arguments.

```text
scgeo.data.open_census(*, census_version: 'Optional[str]' = 'stable')
```

[View implementation](../../scgeo/data/_census.py)
