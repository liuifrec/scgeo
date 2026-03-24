# scgeo/data/_census.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


def _require_census_core():
    try:
        import cellxgene_census  # type: ignore
        return cellxgene_census
    except Exception as e:
        raise ImportError(
            "cellxgene-census is required for scgeo.data.census_*.\n"
            "Install with: pip install cellxgene-census tiledbsoma pyarrow"
        ) from e


def _require_census():
    """
    Backward-compatible helper for functions that only require the core
    cellxgene_census package.
    """
    return _require_census_core()


@dataclass
class CensusEmbeddingResult:
    joinids: np.ndarray
    distances: np.ndarray
    obs: Optional[pd.DataFrame] = None


def open_census(*, census_version: Optional[str] = "stable"):
    cellxgene_census = _require_census_core()
    if census_version is None:
        return cellxgene_census.open_soma()
    return cellxgene_census.open_soma(census_version=census_version)


def _maybe_import_tiledbsoma():
    try:
        import tiledbsoma  # type: ignore
        return tiledbsoma
    except Exception as e:
        raise ImportError(
            "tiledbsoma is required for axis_query / chunked Census access.\n"
            "Install with: pip install tiledbsoma"
        ) from e


@dataclass(frozen=True)
class CensusQuery:
    organism: str = "Homo sapiens"
    census_version: Optional[str] = None
    measurement_name: str = "RNA"
    x_name: str = "raw"
    obs_value_filter: Optional[str] = None
    var_value_filter: Optional[str] = None
    obs_columns: Optional[Sequence[str]] = None
    var_columns: Optional[Sequence[str]] = None


def fetch_obs_by_joinids(
    joinids: Iterable[int],
    *,
    organism: str = "homo_sapiens",
    census_version: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    cellxgene_census = _require_census_core()

    joinids = np.asarray(list(joinids), dtype=np.int64)
    if joinids.size == 0:
        return pd.DataFrame()

    cols = list(obs_columns) if obs_columns is not None else None
    if cols is not None and "soma_joinid" not in cols:
        cols = ["soma_joinid", *cols]

    parts = []
    with open_census(census_version=census_version) as census:
        for i in range(0, len(joinids), int(chunk_size)):
            chunk = joinids[i : i + int(chunk_size)]
            ids = ", ".join(str(int(x)) for x in chunk)
            vf = f"soma_joinid in [{ids}]"
            df = cellxgene_census.get_obs(
                census,
                organism,
                value_filter=vf,
                column_names=cols,
            )
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=cols if cols is not None else None)

    return pd.concat(parts, ignore_index=True)


def census_get_anndata(
    *,
    organism: str = "Homo sapiens",
    census_version: Optional[str] = None,
    measurement_name: str = "RNA",
    x_name: str = "raw",
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
    var_columns: Optional[Sequence[str]] = None,
    obs_embeddings: Optional[Sequence[str]] = None,
):
    """
    Convenience wrapper around cellxgene_census.open_soma() + get_anndata().

    Supports stable embedding retrieval via obs_embeddings=[...].
    """
    cellxgene_census = _require_census()

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            measurement_name=measurement_name,
            X_name=x_name,
            obs_value_filter=obs_value_filter,
            var_value_filter=var_value_filter,
            obs_column_names=list(obs_columns) if obs_columns is not None else None,
            var_column_names=list(var_columns) if var_columns is not None else None,
            obs_embeddings=list(obs_embeddings) if obs_embeddings is not None else None,
        )
    return adata


def census_query_obs_dataframe(
    *,
    organism: str = "Homo sapiens",
    census_version: Optional[str] = None,
    obs_value_filter: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
):
    cellxgene_census = _require_census()
    import pandas as pd

    with cellxgene_census.open_soma(census_version=census_version) as census:
        soma = census["census_data"][organism.lower().replace(" ", "_")]
        obs = soma.obs.read(
            value_filter=obs_value_filter,
            column_names=list(obs_columns) if obs_columns is not None else None,
        ).concat()
    return obs.to_pandas() if hasattr(obs, "to_pandas") else pd.DataFrame(obs)


def census_axis_query_tables(
    *,
    organism: str = "Homo sapiens",
    census_version: Optional[str] = None,
    measurement_name: str = "RNA",
    x_name: str = "raw",
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
) -> Iterator[Dict[str, object]]:
    cellxgene_census = _require_census()
    tiledbsoma = _maybe_import_tiledbsoma()

    with cellxgene_census.open_soma(census_version=census_version) as census:
        org_key = organism.lower().replace(" ", "_")
        org = census["census_data"][org_key]

        query = org.axis_query(
            measurement_name=measurement_name,
            obs_query=tiledbsoma.AxisQuery(value_filter=obs_value_filter) if obs_value_filter else None,
            var_query=tiledbsoma.AxisQuery(value_filter=var_value_filter) if var_value_filter else None,
        )
        try:
            x_iter = query.X(x_name).tables()
            obs_tbl = query.obs().read().concat()
            var_tbl = query.var().read().concat()

            yield {
                "obs": obs_tbl,
                "var": var_tbl,
                "x_tables": x_iter,
            }
        finally:
            query.close()


def census_embedding_api_available() -> bool:
    """
    Return True if stable embedding retrieval through get_anndata(obs_embeddings=[...])
    is available in the installed cellxgene-census package.
    """
    cx = _require_census_core()
    return hasattr(cx, "get_anndata")


# Backward-compatible placeholders for deprecated direct embedding helper APIs.
def get_embedding_metadata_by_name(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census embedding metadata helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Use census_get_anndata(..., obs_embeddings=[...]) "
        "via build_reference_pool_from_census()."
    )


def find_nearest_obs(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census nearest-neighbor embedding helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Use census_get_anndata(..., obs_embeddings=[...]) "
        "to build a ReferencePool, then query it locally."
    )


def census_get_embedding_metadata_by_name(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census embedding metadata helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Use census_get_anndata(..., obs_embeddings=[...])."
    )


def census_get_embedding(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census embedding fetch helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Use census_get_anndata(..., obs_embeddings=[...])."
    )


def census_find_nearest_obs(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census nearest-neighbor embedding helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Build a ReferencePool from Census embeddings and query locally."
    )


def census_predict_obs_metadata(*args, **kwargs):
    raise NotImplementedError(
        "Direct Census prediction helpers are deprecated in ScGeo for stable "
        "cellxgene-census versions. Build a ReferencePool from Census embeddings and use "
        "map_query_to_ref_pool()."
    )