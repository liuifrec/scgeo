from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_embedding_xy, _get_ax


def _get_scanpy_palette(adata, groupby: str):
    """
    Return Scanpy-style palette mapping {category -> color} if available.
    Uses adata.uns[f"{groupby}_colors"] aligned to adata.obs[groupby].cat.categories.
    """
    key = f"{groupby}_colors"
    if key not in adata.uns:
        return None
    s = adata.obs[groupby]
    if not hasattr(s, "cat"):
        return None
    cats = s.cat.categories.astype(str).tolist()
    cols = list(adata.uns[key])
    if len(cats) != len(cols):
        return None
    return dict(zip(cats, cols))


def _fallback_palette(labels: np.ndarray):
    uniq = np.unique(labels.astype(str))
    cmap = plt.get_cmap("tab10")
    return {k: cmap(i % 10) for i, k in enumerate(uniq)}


def _topk_mask_robust(score: np.ndarray, topk: int) -> np.ndarray:
    """
    Robust top-k selection: highest scores among finite values.
    Always returns a boolean mask of same length as score.
    """
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    n = int(finite.sum())
    if n == 0:
        return np.zeros(score.shape[0], dtype=bool)

    k = int(min(max(topk, 1), n))
    idx_f = np.flatnonzero(finite)
    # argsort on finite subset
    ord_f = np.argsort(score[finite])
    top_idx = idx_f[ord_f[-k:]]
    mask = np.zeros(score.shape[0], dtype=bool)
    mask[top_idx] = True
    return mask


def highlight_topk_cells(
    adata,
    score_key: str,
    basis: str = "umap",
    *,
    topk: int = 300,
    ax=None,
    figsize=(5.8, 5.2),

    # background
    bg_size: float = 6.0,
    bg_alpha: float = 0.20,

    # highlight (top-k)
    hi_size: float = 30.0,
    hi_alpha: float = 0.95,
    cmap: str = "viridis",

    title: str | None = None,
    show: bool = True,

    # Scanpy-outline-ish structure on top-k
    groupby: str | None = None,
    use_scanpy_colors: bool = True,
    outline_topk: bool = False,     # IMPORTANT: default False so other panels don't break
    outline_lw: float = 1.6,
    outline_alpha: float = 0.95,
    add_colorbar: bool = True,
):
    """
    Highlight top-k cells by a score on an embedding.

    Default look:
      - background: all cells in light grey
      - top-k: larger points colored by score_key

    If groupby is provided and outline_topk=True:
      - top-k points get edgecolors from group palette (Scanpy palette if available)
      - gives Scanpy-like cluster structure without hull polygons
    """
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    x, y = _get_embedding_xy(adata, basis=basis)
    score = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)

    fig, ax0 = _get_ax(ax=ax, figsize=figsize)

    # background (light)
    ax0.scatter(x, y, s=bg_size, c="lightgrey", alpha=bg_alpha, linewidths=0, zorder=1)

    # robust top-k
    mask = _topk_mask_robust(score, int(topk))
    if not mask.any():
        # nothing finite or selectable; don't crash
        ax0.set_xlabel(f"{basis.upper()}1")
        ax0.set_ylabel(f"{basis.upper()}2")
        ax0.set_title(title or f"Top-{topk} cells by {score_key} (no finite values)")
        ax0.set_aspect("equal", adjustable="datalim")
        if show and ax is None:
            plt.show()
        return fig, ax0

    # edgecolors for top-k (optional)
    edgecolors = "none"
    if outline_topk and groupby is not None and groupby in adata.obs:
        g = adata.obs[groupby].astype(str).to_numpy()
        pal = _get_scanpy_palette(adata, groupby) if use_scanpy_colors else None
        if pal is None:
            pal = _fallback_palette(g)
        edgecolors = [pal.get(k, "k") for k in g[mask]]
    else:
        outline_topk = False  # silently disable if not available
    
    sc = ax0.scatter(
        x[mask],
        y[mask],
        s=hi_size,
        c=score[mask],
        cmap=cmap,
        alpha=hi_alpha,
        linewidths=float(outline_lw) if outline_topk else 0.0,
        edgecolors=edgecolors,
        zorder=3,
    )

    # (optional) tweak alpha on the PathCollection itself (matplotlib sometimes treats this separately)
    if outline_topk:
        sc.set_alpha(float(outline_alpha))

    if add_colorbar:
        cb = fig.colorbar(sc, ax=ax0)
        cb.set_label(score_key)

    ax0.set_xlabel(f"{basis.upper()}1")
    ax0.set_ylabel(f"{basis.upper()}2")
    ax0.set_title(title or f"Top-{int(mask.sum())} cells by {score_key}")
    ax0.set_aspect("equal", adjustable="datalim")

    if show and ax is None:
        plt.show()

    return fig, ax0
