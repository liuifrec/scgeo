# scgeo/data/_census.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# Optional deps (DO NOT bind experimental at import-time)
def _require_census_core():
    try:
        import cellxgene_census  # type: ignore
        return cellxgene_census
    except Exception as e:
        raise ImportError(
            "cellxgene-census is required for scgeo.data.census_*.\n"
            "Install with: pip install cellxgene-census tiledbsoma pyarrow"
        ) from e


def _require_census_exp():
    cellxgene_census = _require_census_core()
    try:
        from cellxgene_census import experimental as census_exp  # type: ignore
        return cellxgene_census, census_exp
    except Exception as e:
        raise ImportError(
            "cellxgene-census experimental module is required for embedding search.\n"
            "Install/upgrade: pip install -U cellxgene-census"
        ) from e


@dataclass
class CensusEmbeddingResult:
    # For each query row i, we return k neighbors (joinids + distances)
    joinids: np.ndarray      # shape (n_query, k) int64
    distances: np.ndarray    # shape (n_query, k) float32
    obs: Optional[pd.DataFrame] = None  # fetched metadata for unique joinids (optional)


def _require_census():
    if cellxgene_census is None or census_exp is None:
        raise ImportError(
            "cellxgene-census is required. Install: pip install cellxgene-census"
        )

def open_census(*, census_version: Optional[str] = None):
    cellxgene_census = _require_census_core()
    if census_version is None:
        return cellxgene_census.open_soma()
    return cellxgene_census.open_soma(census_version=census_version)


def get_embedding_metadata_by_name(
    embedding_name: str,
    *,
    census_version: Optional[str] = None,
) -> Dict[str, Any]:
    _, census_exp = _require_census_exp()
    return census_exp.get_embedding_metadata_by_name(
        embedding_name, census_version=census_version
    )

def find_nearest_obs(
    query_X: np.ndarray,
    *,
    embedding_name: str,
    k: int = 50,
    census_version: Optional[str] = None,
) -> CensusEmbeddingResult:
    _, census_exp = _require_census_exp()

    query_X = np.asarray(query_X, dtype=np.float32)
    if query_X.ndim != 2:
        raise ValueError(f"query_X must be 2D, got shape={query_X.shape}")

    nn = census_exp.find_nearest_obs(
        query_X=query_X,
        embedding_name=embedding_name,
        k=k,
        census_version=census_version,
    )

    joinids = np.asarray(nn["soma_joinid"], dtype=np.int64)
    distances = np.asarray(nn["distance"], dtype=np.float32)
    return CensusEmbeddingResult(joinids=joinids, distances=distances)

def fetch_obs_by_joinids(
    joinids: Iterable[int],
    *,
    organism: str = "homo_sapiens",
    census_version: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cellxgene_census = _require_census_core()

    joinids = np.asarray(list(joinids), dtype=np.int64)
    if joinids.size == 0:
        return pd.DataFrame()

    with open_census(census_version=census_version) as census:
        obs_df = cellxgene_census.get_obs(
            census=census,
            organism=organism,
            obs_coords=joinids,
            columns=list(obs_columns) if obs_columns is not None else None,
        )
    return obs_df



def _maybe_import_tiledbsoma():
    try:
        import tiledbsoma  # type: ignore
        return tiledbsoma
    except Exception as e:
        raise ImportError(
            "tiledbsoma is required for axis_query / chunked Census access.\n"
            "Install with: pip install tiledbsoma"
        ) from e
def _org_key(organism: str) -> str:
    o = organism.strip().lower()
    if o in {"homo sapiens", "human", "h. sapiens"}:
        return "homo_sapiens"
    if o in {"mus musculus", "mouse", "m. musculus"}:
        return "mus_musculus"
    # fallback: best-effort
    return o.replace(" ", "_")


@dataclass(frozen=True)
class CensusQuery:
    organism: str = "Homo sapiens"
    census_version: Optional[str] = None  # pin for reproducibility
    measurement_name: str = "RNA"
    x_name: str = "raw"  # 'raw' is common; see Census docs/examples
    obs_value_filter: Optional[str] = None
    var_value_filter: Optional[str] = None
    obs_columns: Optional[Sequence[str]] = None
    var_columns: Optional[Sequence[str]] = None


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
):
    """
    Convenience wrapper around cellxgene_census.open_soma() + get_anndata().

    This is the "demo notebook" path: simplest way to fetch a slice as AnnData.

    Census quick-start uses:
      with open_soma() as census:
          adata = get_anndata(census=..., organism=..., obs_value_filter=..., var_value_filter=..., column_names=...)

    Ref: official docs quick-start. :contentReference[oaicite:1]{index=1}
    """
    cellxgene_census = _require_census()

    column_names: Dict[str, List[str]] = {}
    if obs_columns is not None:
        column_names["obs"] = list(obs_columns)
    if var_columns is not None:
        column_names["var"] = list(var_columns)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            measurement_name=measurement_name,
            X_name=x_name,
            obs_value_filter=obs_value_filter,
            var_value_filter=var_value_filter,
            column_names=(column_names if len(column_names) else None),
        )
    return adata


def census_query_obs_dataframe(
    *,
    organism: str = "Homo sapiens",
    census_version: Optional[str] = None,
    obs_value_filter: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
):
    """
    Return obs metadata slice as a pandas DataFrame.

    Mirrors the "Querying a slice of cell metadata" quick-start pattern. :contentReference[oaicite:2]{index=2}
    """
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
    """
    Memory-efficient lane: yields chunked tables using TileDB-SOMA axis_query.

    You can iterate the returned dicts and decide how to build sparse matrices / AnnData
    without ever materializing the full slice in RAM.

    Pattern is straight from the quick-start "Memory-efficient queries". :contentReference[oaicite:3]{index=3}
    """
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
            # X("raw").tables() yields Arrow tables in chunks (example in docs). :contentReference[oaicite:4]{index=4}
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


def _require_cellxgene_census():
    """
    Lazy import so scgeo can import without cellxgene_census installed.
    Raises ImportError only when census functionality is actually called.
    """
    try:
        import cellxgene_census  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "cellxgene_census is required for Census features. "
            "Install with: pip install cellxgene-census"
        ) from e
    return cellxgene_census


def _require_census_embeddings():
    cx = _require_cellxgene_census()
    try:
        exp = cx.experimental  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("cellxgene_census.experimental not available in this version.") from e
    return cx, exp


def census_get_embedding_metadata_by_name(
    census,
    *,
    embedding_name: str,
    organism: str = "homo_sapiens",
) -> Dict[str, Any]:
    """
    Wrapper around cellxgene_census.experimental.get_embedding_metadata_by_name.
    """
    _, exp = _require_census_embeddings()
    return exp.get_embedding_metadata_by_name(
        census=census,
        embedding_name=embedding_name,
        organism=organism,
    )


def census_get_embedding(
    census,
    *,
    embedding_name: str,
    obs_joinids: Sequence[int],
    organism: str = "homo_sapiens",
) -> np.ndarray:
    """
    Wrapper around cellxgene_census.experimental.get_embedding.
    Returns dense ndarray (n_obs, d).
    """
    _, exp = _require_census_embeddings()
    obs_joinids = np.asarray(obs_joinids, dtype=np.int64)
    return exp.get_embedding(
        census=census,
        embedding_name=embedding_name,
        organism=organism,
        obs_joinids=obs_joinids,
    )


def census_find_nearest_obs(
    census,
    *,
    embedding_name: str,
    query_embedding: np.ndarray,
    k: int = 50,
    organism: str = "homo_sapiens",
    # pass-through extras (kept flexible across census versions)
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper around cellxgene_census.experimental.find_nearest_obs.
    Returns a DataFrame typically containing:
      - query_index (or similar)
      - soma_joinid (neighbor id)
      - distance
      - (sometimes) dataset_id, etc.
    """
    _, exp = _require_census_embeddings()

    q = np.asarray(query_embedding)
    if q.ndim != 2:
        raise ValueError(f"query_embedding must be 2D, got {q.shape}")

    df = exp.find_nearest_obs(
        census=census,
        embedding_name=embedding_name,
        organism=organism,
        query_embedding=q,
        k=int(k),
        **kwargs,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def census_predict_obs_metadata(
    census,
    *,
    embedding_name: str,
    query_embedding: np.ndarray,
    obs_columns: Sequence[str],
    k: int = 50,
    organism: str = "homo_sapiens",
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper around cellxgene_census.experimental.predict_obs_metadata.
    Useful for 'majority vote' / weighted vote labels directly from census.
    """
    _, exp = _require_census_embeddings()

    q = np.asarray(query_embedding)
    if q.ndim != 2:
        raise ValueError(f"query_embedding must be 2D, got {q.shape}")

    df = exp.predict_obs_metadata(
        census=census,
        embedding_name=embedding_name,
        organism=organism,
        query_embedding=q,
        obs_columns=list(obs_columns),
        k=int(k),
        **kwargs,
    )
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df