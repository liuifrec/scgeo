from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._utils import _get_embedding_xy, _get_ax, resolve_embedding
from ._result import PlotResult


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

    k = int(min(max(topk, 0), n))
    if k == 0:
        return np.zeros(score.shape[0], dtype=bool)
    idx_f = np.flatnonzero(finite)
    # argsort on finite subset
    ord_f = np.argsort(score[finite], kind="stable")
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
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    background: bool = True,
    return_data: bool = False,
    return_result: bool = False,
):
    """
    Highlight top-k cells by a score on an embedding.

    Default look:
      - background: all cells in light grey
      - top-k: larger points colored by score_key

    If groupby is provided and outline_topk=True:
      - top-k points get edgecolors from group palette (Scanpy palette if available)
      - gives Scanpy-like cluster structure without hull polygons

    ``norm`` or ``vmin``/``vmax`` can match an existing score layer. Without
    these, the scale uses the selected finite scores. ``background=False``
    adds only the highlighted layer. ``cax`` places an optional colorbar.
    ``topk=0`` selects no cells. Ranking uses finite scores from all input
    observations; selected cells with nonfinite coordinates remain in the
    source table but cannot be drawn. Ties follow input order deterministically.
    The default return is ``(fig, ax)``; ``return_data=True`` adds the full cell
    table, and ``return_result=True`` enables checksummed PlotResult exports.
    """
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    if isinstance(topk, bool) or not isinstance(topk, (int, np.integer)) or topk < 0:
        raise ValueError("topk must be a nonnegative integer.")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if cax is not None:
        if ax is None or not add_colorbar:
            raise ValueError("cax requires ax and add_colorbar=True.")
        from ._table_plot_utils import panel_axes
        panel_axes(["embedding", "colorbar"], {"embedding": ax, "colorbar": cax}, None)
    key, xy = resolve_embedding(adata, basis)
    x, y = xy[:, 0], xy[:, 1]
    score = pd.to_numeric(adata.obs[score_key], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    finite_xy = np.isfinite(xy[:, :2]).all(axis=1)
    selected = _topk_mask_robust(score, int(topk))
    mask = selected & finite_xy
    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    norm.autoscale_None(score[selected])

    fig, ax0 = _get_ax(ax=ax, figsize=figsize)
    artists = {}

    # background (light)
    if background:
        artists["background"] = ax0.scatter(
            x[finite_xy], y[finite_xy], s=bg_size, c="lightgrey",
            alpha=bg_alpha, linewidths=0, zorder=1)

    # robust top-k
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
        norm=norm,
        alpha=hi_alpha,
        linewidths=float(outline_lw) if outline_topk else 0.0,
        edgecolors=edgecolors,
        zorder=3,
    )
    artists["highlights"] = sc

    # (optional) tweak alpha on the PathCollection itself (matplotlib sometimes treats this separately)
    if outline_topk:
        sc.set_alpha(float(outline_alpha))

    if add_colorbar and mask.any():
        cb = fig.colorbar(sc, ax=ax0, cax=cax)
        cb.set_label(score_key)
        artists["colorbar"] = cb

    basis_name = key.removeprefix("X_").upper()
    ax0.set_xlabel(f"{basis_name}1")
    ax0.set_ylabel(f"{basis_name}2")
    ax0.set_title(title or f"Top-{int(mask.sum())} cells by {score_key}")
    ax0.set_aspect("equal", adjustable="datalim")

    if show and ax is None:
        plt.show()

    cells = pd.DataFrame({
        "obs_position": np.arange(adata.n_obs), "cell": adata.obs_names.astype(str),
        "x": x, "y": y, "value": score, "finite_embedding": finite_xy,
        "selected": selected, "highlighted": mask,
        "displayed": finite_xy if background else mask,
    })
    if groupby is not None and groupby in adata.obs:
        cells["group"] = adata.obs[groupby].to_numpy()
    if return_result:
        notes = []
        if (selected & ~finite_xy).any():
            notes.append("Selected cells with nonfinite embedding coordinates were not drawn; selection was not recomputed.")
        return PlotResult(fig, {"embedding": ax0}, data=cells, artists=artists, warnings=notes,
                          metadata={"plot": "highlight_topk_cells", "score_key": score_key,
                                    "basis": key, "topk": int(topk), "n_selected": int(selected.sum()),
                                    "n_highlighted": int(mask.sum()), "background": background,
                                    "groupby": groupby, "outline_topk": outline_topk,
                                    "norm": {"type": type(norm).__name__, "vmin": norm.vmin, "vmax": norm.vmax}})
    if return_data:
        return fig, ax0, cells
    return fig, ax0
