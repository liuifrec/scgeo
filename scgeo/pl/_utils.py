from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _get_embedding_xy(adata, basis: str = "umap"):
    basis = basis[2:] if basis.startswith("X_") else basis
    key = f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"{key} not found in adata.obsm")
    X = np.asarray(adata.obsm[key])
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"{key} must be 2D with >=2 columns; got shape {X.shape}")
    return X[:, 0], X[:, 1]


def _get_ax(ax=None, figsize=(5.0, 4.5)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _topk_mask(values: np.ndarray, k: int):
    values = np.asarray(values)
    finite = np.isfinite(values)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    k = int(min(k, idx.size))
    top_idx = idx[np.argsort(values[idx])[-k:]]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[top_idx] = True
    return mask
