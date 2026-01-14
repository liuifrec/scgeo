from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_embedding_xy, _get_ax, _topk_mask


def highlight_topk_cells(
    adata,
    score_key: str,
    basis: str = "umap",
    *,
    topk: int = 300,
    ax=None,
    figsize=(5.5, 5.0),
    bg_size: float = 4.0,
    hi_size: float = 18.0,
    bg_alpha: float = 0.15,
    hi_alpha: float = 0.95,
    cmap: str = "viridis",
    title: str | None = None,
    show: bool = True,
):
    """
    Highlight top-k cells by a score on an embedding.

    - background: all cells in light grey
    - highlight: top-k cells colored by score_key

    Returns (fig, ax).
    """
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    x, y = _get_embedding_xy(adata, basis=basis)
    score = np.asarray(adata.obs[score_key].to_numpy())
    fig, ax = _get_ax(ax=ax, figsize=figsize)

    # background
    ax.scatter(x, y, s=bg_size, c="lightgrey", alpha=bg_alpha, linewidths=0)

    mask = _topk_mask(score, topk)
    if mask.any():
        sc = ax.scatter(
            x[mask],
            y[mask],
            s=hi_size,
            c=score[mask],
            cmap=cmap,
            alpha=hi_alpha,
            linewidths=0,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(score_key)

    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    ax.set_title(title or f"Top-{topk} cells by {score_key}")
    ax.set_aspect("equal", adjustable="datalim")

    if show and ax is None:
        plt.show()

    return fig, ax
