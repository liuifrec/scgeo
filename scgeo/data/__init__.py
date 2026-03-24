# scgeo/data/__init__.py
from ._census import (
    CensusQuery,
    census_get_anndata,
    census_query_obs_dataframe,
    census_axis_query_tables,
    open_census,
    fetch_obs_by_joinids,
    census_get_embedding_metadata_by_name,
    census_get_embedding,
    census_find_nearest_obs,
    census_predict_obs_metadata,
    # temporary backward-compatible aliases
    get_embedding_metadata_by_name,
    find_nearest_obs,
)
__all__ = [
    "CensusQuery",
    "census_get_anndata",
    "census_query_obs_dataframe",
    "census_axis_query_tables",
    "open_census",
    "fetch_obs_by_joinids",
    "census_get_embedding_metadata_by_name",
    "census_get_embedding",
    "census_find_nearest_obs",
    "census_predict_obs_metadata",
    "get_embedding_metadata_by_name",
    "find_nearest_obs",
]