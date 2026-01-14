from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_ax  # <-- add this


def score_embedding(
    adata,
    score_key: str,
    basis: str = "umap",
    *,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    size: float = 6.0,
    alpha: float = 0.8,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    na_color: str = "lightgrey",
    figsize=None,                 # <-- add this
    show: bool = True,            # <-- keep this
):
    """
    Plot an obs score on an embedding (UMAP/PCA/etc) with minimal dependencies.
    Returns (fig, ax).
    """
    fig, ax = _get_ax(ax=ax, figsize=figsize)

    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(f"{emb_key} not found in adata.obsm. Run embedding first.")
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    X = np.asarray(adata.obsm[emb_key])
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"{emb_key} must be 2D with >=2 columns; got shape {X.shape}")

    scores = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)
    is_finite = np.isfinite(scores)

    # background NaNs
    if (~is_finite).any():
        ax.scatter(
            X[~is_finite, 0], X[~is_finite, 1],
            s=size, alpha=alpha, c=na_color, linewidths=0
        )

    sc = ax.scatter(
        X[is_finite, 0],
        X[is_finite, 1],
        s=size,
        alpha=alpha,
        c=scores[is_finite],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(score_key)

    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    ax.set_title(title or score_key)

    if show:
        plt.show()

    return fig, ax


def score_umap(adata, score_key: str, **kwargs):
    return score_embedding(adata, score_key=score_key, basis="umap", **kwargs)
