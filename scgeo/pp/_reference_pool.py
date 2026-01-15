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
    index = NNDescent(
        X,
        n_neighbors=min(int(n_neighbors), max(2, n_ref - 1)),
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
    adata_q,
    rep: str,
    embedding_name: str,
    label_key: str,
    obs_columns: Optional[Sequence[str]] = None,
    k: int = 50,
    organism: str = "homo_sapiens",
    max_refs: Optional[int] = 200_000,
    dedup: bool = True,
    index_metric: str = "euclidean",
    index_seed: int = 0,
    census_obs_filter: Optional[str] = None,  # keep for future use; your fetch handles joinids anyway
) -> ReferencePool:
    """
    Build an embedding-only ReferencePool from cellxgene-census WITHOUT concatenation.

    Uses scgeo.data:
      - find_nearest_obs
      - fetch_obs_by_joinids
      - census_get_embedding
    """
    from ..data import find_nearest_obs, fetch_obs_by_joinids, census_get_embedding

    if obs_columns is None:
        obs_columns = []
    obs_cols = list(dict.fromkeys([label_key, *list(obs_columns)]))  # unique

    # --- query embedding
    Xq = adata_q.X if rep == "X" else adata_q.obsm[rep]
    Xq = _as_float32_2d(Xq)

    # --- nearest neighbor search in Census embedding space
    # Your find_nearest_obs should return a DataFrame-like or object containing joinids/dists.
    nn = find_nearest_obs(
        census,
        embedding_name=embedding_name,
        organism=organism,
        query_embedding=Xq,
        k=int(k),
    )

    # Normalize outputs
    if isinstance(nn, pd.DataFrame):
        joinids = nn["soma_joinid"].to_numpy(dtype=np.int64)
    elif hasattr(nn, "joinids"):
        joinids = np.asarray(nn.joinids, dtype=np.int64).ravel()
    else:
        raise TypeError("Unsupported return type from find_nearest_obs")

    if dedup:
        joinids = np.unique(joinids)

    if max_refs is not None and len(joinids) > int(max_refs):
        joinids = np.sort(joinids)[: int(max_refs)]

    # --- fetch ref embeddings for those joinids
    Xref = census_get_embedding(
        census,
        embedding_name=embedding_name,
        organism=organism,
        obs_joinids=joinids,
    )
    Xref = _as_float32_2d(Xref)

    # --- fetch obs metadata for those joinids (label + extras)
    # Your fetch_obs_by_joinids already exists: use it.
    obs_df = fetch_obs_by_joinids(
        census,
        organism=organism,
        joinids=joinids,
        columns=["soma_joinid", *obs_cols],
    )

    # align obs order to joinids
    if "soma_joinid" not in obs_df.columns:
        raise KeyError("fetch_obs_by_joinids did not return soma_joinid")
    obs_df = obs_df.drop_duplicates("soma_joinid", keep="first").set_index("soma_joinid").reindex(joinids)

    obs: Dict[str, np.ndarray] = {}
    for c in obs_cols:
        obs[c] = obs_df[c].to_numpy() if c in obs_df.columns else np.array([np.nan] * len(joinids), dtype=object)

    if label_key not in obs:
        raise KeyError(f"label_key '{label_key}' not found in fetched obs columns")

    # --- build ANN index over ref embeddings
    NNDescent = _require_pynndescent()
    index = NNDescent(Xref, metric=index_metric, random_state=int(index_seed))

    meta: Dict[str, Any] = dict(
        source="cellxgene-census",
        organism=organism,
        embedding_name=embedding_name,
        k=int(k),
        rep_query=rep,
        obs_columns=list(obs_cols),
        dedup=bool(dedup),
        max_refs=int(max_refs) if max_refs is not None else None,
        census_obs_filter=census_obs_filter,
    )
    joinids = np.asarray(joinids, dtype=np.int64)
    joinid_to_col = {int(j): int(i) for i, j in enumerate(joinids)}

    return ReferencePool(
        X=Xref,
        obs=obs,
        label_key=label_key,
        index=index,
        meta=meta,
        joinids=joinids,
    )