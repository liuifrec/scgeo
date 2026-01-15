# scgeo/data/__init__.py

from ._census import (
    CensusQuery,
    census_get_anndata,
    census_query_obs_dataframe,
    census_axis_query_tables,
    # Phase B:
    open_census,
    get_embedding_metadata_by_name,
    find_nearest_obs,
    fetch_obs_by_joinids,
)


__all__ = [
    "CensusQuery",
    "census_get_anndata",
    "census_query_obs_dataframe",
    "census_axis_query_tables",
    "open_census",
    "get_embedding_metadata_by_name",
    "find_nearest_obs",
    "fetch_obs_by_joinids",
]
