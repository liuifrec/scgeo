# scgeo/pp/_census_knn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any

import numpy as np

from .._utils import coo_to_csr


def _as_2d_object_array(x) -> np.ndarray:
    """
    Normalize neighbor joinids into shape (n_qry, k) object array.
    Accepts:
      - list[list[...]]
      - np.ndarray
    """
    arr = np.asarray(x, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array-like (n_qry, k), got shape {arr.shape}")
    return arr


def _as_2d_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array-like (n_qry, k), got shape {arr.shape}")
    return arr


def _dist_to_weight(d: np.ndarray, mode: str = "inv", eps: float = 1e-8) -> np.ndarray:
    """
    Convert distances to weights.
    - mode="inv": 1/(d+eps)
    - mode="exp": exp(-d)
    - mode="one": all ones
    """
    if mode == "inv":
        return 1.0 / (d + eps)
    if mode == "exp":
        return np.exp(-d)
    if mode == "one":
        return np.ones_like(d, dtype=np.float32)
    raise ValueError("mode must be one of: inv, exp, one")

def build_query_to_ref_knn_edges_from_census(
    *,
    n_qry: int,
    ref_pool,
    nn_joinids: Sequence[Sequence[object]],
    nn_dists: Sequence[Sequence[float]],
    weight_mode: str = "inv",
    drop_missing: bool = True,
    return_diagnostics: bool = False,
    # NEW:
    return_csr: bool = False,
    sum_duplicates: bool = True,
):
    """
    Build query->ref edges from Census embedding search outputs.

    If return_csr=True, returns (C_qr, diag) where:
      C_qr is CSR of shape (n_qry, n_ref)

    Else returns (rows, cols, data, diag) in COO triplet form.
    """
    J = _as_2d_object_array(nn_joinids)
    D = _as_2d_float_array(nn_dists)

    if J.shape != D.shape:
        raise ValueError(f"nn_joinids shape {J.shape} != nn_dists shape {D.shape}")
    if J.shape[0] != n_qry:
        raise ValueError(f"n_qry={n_qry} but nn_joinids has {J.shape[0]} rows")

    joinid_to_col = getattr(ref_pool, "joinid_to_col", None)
    n_ref = getattr(ref_pool, "n_ref", None)
    if joinid_to_col is None or n_ref is None:
        raise TypeError("ref_pool must have attributes: joinid_to_col and n_ref")

    W = _dist_to_weight(D, mode=weight_mode)

    rows_out: list[int] = []
    cols_out: list[int] = []
    data_out: list[float] = []

    missing = 0
    used = 0

    for i in range(n_qry):
        for j in range(J.shape[1]):
            joinid = J[i, j]
            if joinid is None or joinid != joinid:  # None or NaN
                continue
            col = joinid_to_col.get(joinid, None)
            if col is None:
                if drop_missing:
                    missing += 1
                    continue
                raise KeyError(f"joinid {joinid!r} not found in ref_pool.joinid_to_col")
            rows_out.append(int(i))
            cols_out.append(int(col))
            data_out.append(float(W[i, j]))
            used += 1

    rows = np.asarray(rows_out, dtype=np.int64)
    cols = np.asarray(cols_out, dtype=np.int64)
    data = np.asarray(data_out, dtype=np.float32)

    diag: Dict[str, Any] | None = None
    if return_diagnostics:
        diag = dict(
            n_qry=int(n_qry),
            n_ref=int(n_ref),
            k=int(J.shape[1]),
            edges=int(used),
            missing=int(missing),
            missing_rate=float(missing / max(1, n_qry * J.shape[1])),
            weight_mode=weight_mode,
        )
    
    if not return_csr:
        return rows, cols, data, diag

    C_qr = coo_to_csr(
        rows=rows,
        cols=cols,
        data=data,
        shape=(int(n_qry), int(n_ref)),
        dtype=np.float32,
        sum_duplicates=bool(sum_duplicates),
    )
    return C_qr, diag