from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _as_2d_array(X) -> np.ndarray:
    """Convert input (np array / sparse / array-like) to dense float32 2D array."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X.astype(np.float32, copy=False)


def _mask_from_obs(adata, key: str, value) -> np.ndarray:
    if key not in adata.obs:
        raise KeyError(f"obs key '{key}' not found")
    return (adata.obs[key].values == value)


def _unique_nonnull(values: Sequence) -> list:
    out = []
    for v in values:
        if v is None:
            continue
        if v != v:  # NaN
            continue
        if v not in out:
            out.append(v)
    return out


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

# ---- sparse helpers (optional dependency: scipy) ----
# Keep these imports local-ish so scgeo can still import in ultra-minimal envs.
try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover
    sparse = None


def _is_sparse(x) -> bool:
    return (sparse is not None) and sparse.issparse(x)

def _to_csr(M, *, copy: bool = False, allow_dense: bool = False, dtype=np.float32):
    """Ensure CSR matrix. If allow_dense=True, convert dense -> CSR."""
    if sparse is None:
        raise ImportError("scipy is required for sparse matrix operations")
    if sparse.issparse(M):
        X = M.tocsr()
        return X.copy() if copy else X
    if allow_dense:
        X = np.asarray(M)
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        return sparse.csr_matrix(X.astype(dtype, copy=False))
    raise TypeError("Expected a scipy sparse matrix")



def coo_to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
    *,
    dtype=np.float32,
    sum_duplicates: bool = True,
):
    """
    Build CSR from edge lists safely.
    rows/cols/data are 1D arrays of equal length.
    """
    if sparse is None:
        raise ImportError("scipy is required for sparse matrix operations")

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.asarray(data, dtype=dtype)

    M = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    if sum_duplicates:
        M.sum_duplicates()
    return M.tocsr()


def row_sums(M) -> np.ndarray:
    """Return row sums as dense float64 vector."""
    M = _to_csr(M, copy=False)
    return np.asarray(M.sum(axis=1)).ravel()


def row_normalize_csr(M, *, eps: float = 1e-12, inplace: bool = False):
    """
    Row-normalize CSR so each row sums to 1 when row sum > eps.
    This makes OOD/conf scores stable across different graph builders.
    """
    if sparse is None:
        raise ImportError("scipy is required for sparse matrix operations")

    X = _to_csr(M, copy=not inplace)
    rs = row_sums(X)
    inv = np.zeros_like(rs, dtype=np.float64)
    mask = rs > eps
    inv[mask] = 1.0 / rs[mask]

    Dinv = sparse.diags(inv.astype(np.float32), format="csr")
    return (Dinv @ X).tocsr()


def subset_rows_csr(M, idx: np.ndarray):
    """Fast row subset; preserves CSR."""
    M = _to_csr(M, copy=False)
    idx = np.asarray(idx, dtype=np.int64)
    return M[idx, :].tocsr()

def ref_mass_ratio_from_row(
    row,  # csr row (1×n)
    mask_on_indices: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    Given a CSR row, compute:
      total_mass = sum(all weights)
      ref_mass   = sum(weights where mask_on_indices is True)
      ref_ratio  = ref_mass / total_mass

    mask_on_indices MUST be boolean array of same length as row.indices.
    Example:
      cols = row.indices
      m = ref_mask[cols]          # ref_mask is length n_obs
      total, ref_mass, ratio = ref_mass_ratio_from_row(row, m)
    """
    cols = row.indices
    vals = row.data
    total = float(vals.sum())
    if total <= eps or vals.size == 0:
        return 0.0, 0.0, 0.0

    m = np.asarray(mask_on_indices, dtype=bool)
    if m.shape[0] != cols.shape[0]:
        raise ValueError("mask_on_indices must match len(row.indices)")

    ref_mass = float(vals[m].sum())
    ratio = float(ref_mass / total) if total > eps else 0.0
    return total, ref_mass, ratio
