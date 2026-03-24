from __future__ import annotations

import scgeo as sg


def test_data_exports_option2_census_api():
    expected = [
        "open_census",
        "fetch_obs_by_joinids",
        "census_get_anndata",
        "census_query_obs_dataframe",
        "census_axis_query_tables",
        "census_get_embedding_metadata_by_name",
        "census_get_embedding",
        "census_find_nearest_obs",
        "census_predict_obs_metadata",
    ]

    for name in expected:
        assert hasattr(sg.data, name), f"scgeo.data missing export: {name}"