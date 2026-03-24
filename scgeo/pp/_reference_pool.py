#scgeo/pp/_reference_pool.py 
from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Dict, Optional, Tuple, Sequence

import pandas as pd   
import numpy as np



def _as_float32_2d(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X.astype(np.float32, copy=False)

def _require_pynndescent():
    try:
        from pynndescent import NNDescent  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "ReferencePool requires 'pynndescent'. Install with: pip install pynndescent"
        ) from e
    return NNDescent


@dataclass
class ReferencePool:
    X: np.ndarray
    obs: Dict[str, np.ndarray]
    label_key: str
    index: Any
    meta: Dict[str, Any]
    joinids: Optional[np.ndarray] = None

    # cache (NOT part of dataclass __init__)
    _joinid_to_col: Optional[Dict[Any, int]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Validate shapes
        n_ref = int(self.X.shape[0])
        if self.joinids is not None:
            self.joinids = np.asarray(self.joinids)
            if self.joinids.shape[0] != n_ref:
                raise ValueError(f"joinids has len {self.joinids.shape[0]} but expected {n_ref}")

            # Build cache once
            self._joinid_to_col = {self.joinids[i]: int(i) for i in range(n_ref)}

    @property
    def n_ref(self) -> int:
        return int(self.X.shape[0])

    @property
    def joinid_to_col(self) -> Dict[Any, int]:
        """
        Mapping joinid -> column index (0..n_ref-1).
        If joinids is None, default to identity mapping on row index.
        """
        if self._joinid_to_col is not None:
            return self._joinid_to_col

        # no joinids provided -> identity mapping
        self._joinid_to_col = {int(i): int(i) for i in range(self.n_ref)}
        return self._joinid_to_col

    def search(self, Xq, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        Xq = _as_float32_2d(Xq)
        if k <= 0:
            raise ValueError("k must be positive")
        k_eff = min(k, self.n_ref)
        idx, dist = self.index.query(Xq, k=k_eff)
        return np.asarray(idx), np.asarray(dist)

def build_reference_pool(
    X_ref,
    obs: Dict[str, Any],
    *,
    label_key: str,
    joinids: Optional[np.ndarray] = None,
    index_backend: str = "pynndescent",
    n_neighbors: int = 30,
    metric: str = "euclidean",
    random_state: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> ReferencePool:
    X = _as_float32_2d(X_ref)
    n_ref = X.shape[0]
    if n_ref < 2:
        raise ValueError("Need at least 2 reference points")

    if label_key not in obs:
        raise KeyError(f"label_key '{label_key}' not found in obs")

    obs2: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        if arr.shape[0] != n_ref:
            raise ValueError(f"obs['{k}'] has len {arr.shape[0]} but expected {n_ref}")
        obs2[k] = arr

    if joinids is not None:
        joinids = np.asarray(joinids)
        if joinids.shape[0] != n_ref:
            raise ValueError(f"joinids has len {joinids.shape[0]} but expected {n_ref}")

    if index_backend != "pynndescent":
        raise ValueError("index_backend currently supports only 'pynndescent'")

    NNDescent = _require_pynndescent()
    try:
        index = NNDescent(
            X,
            n_neighbors=min(int(n_neighbors), max(2, n_ref - 1)),
            metric=metric,
            random_state=int(random_state),
        )
    except TypeError:
        # backward-compat / test doubles that do not accept n_neighbors
        index = NNDescent(
            X,
            metric=metric,
            random_state=int(random_state),
        )

    return ReferencePool(
        X=X,
        obs=obs2,
        label_key=label_key,
        index=index,
        meta=dict(meta or {}),
        joinids=joinids,
    )



def _build_nndescent_index(Xref: np.ndarray, *, metric: str = "euclidean", seed: int = 0):
    """
    Lazy import to avoid forcing dependency at import time.
    """
    try:
        from pynndescent import NNDescent  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pynndescent is required to build ReferencePool index") from e

    return NNDescent(Xref, metric=metric, random_state=seed)
def build_reference_pool_from_census(
    *,
    census,
    organism: str = "Homo sapiens",
    embedding_name: str = "scvi",
    label_key: str = "cell_type",
    obs_value_filter: Optional[str] = None,
    obs_columns: Optional[Sequence[str]] = None,
    var_value_filter: str = "feature_name in ['CD34']",
    max_refs: Optional[int] = 200_000,
    index_metric: str = "euclidean",
    index_seed: int = 0,
    n_neighbors: int = 30,
    # --- backward-compat legacy args ---
    adata_q=None,
    rep: Optional[str] = None,
    k: Optional[int] = None,
    dedup: Optional[bool] = None,
    census_obs_filter: Optional[str] = None,
) -> ReferencePool:
    """
    Build a ReferencePool from cellxgene Census.

    Two modes are supported:

    1) Stable production mode:
       Uses cellxgene_census.get_anndata(..., obs_embeddings=[embedding_name])
       and requires obs_value_filter.

    2) Legacy compatibility / test mode:
       If obs_value_filter is not provided but adata_q is provided, use the older
       query-driven embedding fetch path via scgeo.data helper functions.
       This keeps existing unit tests and older call sites working.
    """
    # backward-compat alias
    if obs_value_filter is None and census_obs_filter is not None:
        obs_value_filter = census_obs_filter

    # ------------------------------------------------------------------
    # Mode A: stable Census route via get_anndata(..., obs_embeddings=...)
    # ------------------------------------------------------------------
    if obs_value_filter is not None:
        import cellxgene_census

        obs_cols = list(dict.fromkeys(["soma_joinid", label_key, *(obs_columns or [])]))

        ad_ref = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            obs_value_filter=obs_value_filter,
            obs_column_names=obs_cols,
            var_column_names=["feature_name"],
            var_value_filter=var_value_filter,
            obs_embeddings=[embedding_name],
        )

        if embedding_name not in ad_ref.obsm:
            raise KeyError(f"Embedding '{embedding_name}' not found in ad_ref.obsm")

        if max_refs is not None and ad_ref.n_obs > int(max_refs):
            ad_ref = ad_ref[: int(max_refs)].copy()

        X_ref = np.asarray(ad_ref.obsm[embedding_name], dtype=np.float32)

        if label_key not in ad_ref.obs.columns:
            raise KeyError(f"label_key '{label_key}' not found in Census obs")

        obs = {c: ad_ref.obs[c].to_numpy() for c in obs_cols if c in ad_ref.obs.columns}
        joinids = ad_ref.obs["soma_joinid"].to_numpy()

        return build_reference_pool(
            X_ref,
            obs=obs,
            label_key=label_key,
            joinids=joinids,
            n_neighbors=n_neighbors,
            metric=index_metric,
            random_state=index_seed,
            meta={
                "source": "cellxgene-census",
                "organism": organism,
                "embedding_name": embedding_name,
                "obs_value_filter": obs_value_filter,
                "var_value_filter": var_value_filter,
            },
        )

    # ------------------------------------------------------------------
    # Mode B: legacy/mock route for backward compatibility and unit tests
    # ------------------------------------------------------------------
    if adata_q is None:
        raise ValueError(
            "Either obs_value_filter (stable Census mode) or adata_q (legacy/test mode) "
            "must be provided."
        )

    from ..data import census_find_nearest_obs, census_get_embedding, fetch_obs_by_joinids

    if obs_columns is None:
        obs_columns = []
    obs_cols = list(dict.fromkeys([label_key, *list(obs_columns)]))

    rep_key = rep or "X_emb"
    Xq = adata_q.X if rep_key == "X" else adata_q.obsm[rep_key]
    Xq = _as_float32_2d(Xq)

    nn = census_find_nearest_obs(
        census,
        embedding_name=embedding_name,
        organism=organism,
        query_embedding=Xq,
        k=int(k) if k is not None else 50,
    )

    if isinstance(nn, pd.DataFrame):
        joinids = nn["soma_joinid"].to_numpy(dtype=np.int64)
    elif hasattr(nn, "joinids"):
        joinids = np.asarray(nn.joinids, dtype=np.int64).ravel()
    else:
        raise TypeError("Unsupported return type from census_find_nearest_obs")

    if dedup is None or bool(dedup):
        joinids = np.unique(joinids)

    if max_refs is not None and len(joinids) > int(max_refs):
        joinids = np.sort(joinids)[: int(max_refs)]

    X_ref = census_get_embedding(
        census,
        embedding_name=embedding_name,
        organism=organism,
        obs_joinids=joinids,
    )
    X_ref = _as_float32_2d(X_ref)

    obs_df = fetch_obs_by_joinids(
        joinids,
        organism=organism,
        obs_columns=["soma_joinid", *obs_cols],
    )

    if "soma_joinid" not in obs_df.columns:
        raise KeyError("fetch_obs_by_joinids did not return soma_joinid")

    obs_df = (
        obs_df.drop_duplicates("soma_joinid", keep="first")
        .set_index("soma_joinid")
        .reindex(joinids)
    )

    if label_key not in obs_df.columns:
        raise KeyError(f"label_key '{label_key}' not found in fetched obs columns")

    obs: Dict[str, np.ndarray] = {}
    for c in obs_cols:
        if c in obs_df.columns:
            obs[c] = obs_df[c].to_numpy()
        else:
            obs[c] = np.array([np.nan] * len(joinids), dtype=object)

    return build_reference_pool(
        X_ref,
        obs=obs,
        label_key=label_key,
        joinids=np.asarray(joinids, dtype=np.int64),
        n_neighbors=n_neighbors,
        metric=index_metric,
        random_state=index_seed,
        meta={
            "source": "cellxgene-census",
            "organism": organism,
            "embedding_name": embedding_name,
            "legacy_mode": True,
            "rep_query": rep_key,
            "legacy_k": int(k) if k is not None else None,
            "legacy_dedup": bool(dedup) if dedup is not None else None,
        },
    )