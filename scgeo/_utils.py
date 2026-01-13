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
