from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_embedding_xy, _get_ax


def density_overlap_grid(
    adata,
    *,
    key: str = "density_overlap",
    basis: str = "umap",
    panel: str = "overlap",  # "overlap" | "diff" | "p0" | "p1"
    ax=None,
    figsize=(5.5, 5.0),
    cmap: str = "magma",
    alpha_points: float = 0.05,
    show_points: bool = True,
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize density overlap results from `tl.density_overlap`.

    Supports:
    - grid-based fields stored in `adata.uns["scgeo"][key]`
    - or per-cell scores in `adata.obs[key]` (fallback)
    """
    if "scgeo" not in adata.uns:
        raise KeyError("adata.uns['scgeo'] not found")
    obj = adata.uns["scgeo"].get(key)

    # Fallback: per-cell score
    if obj is None:
        if key in adata.obs:
            # plot as per-cell score on embedding
            import scgeo as sg
            return sg.pl.score_embedding(
                adata,
                key,
                basis=basis,
                title=title or f"{key} (cell score)",
                show=show,
            )
        raise KeyError(f"No adata.uns['scgeo']['{key}'] and no adata.obs['{key}']")

    if not isinstance(obj, dict):
        raise TypeError(f"adata.uns['scgeo']['{key}'] must be a dict; got {type(obj)}")

    if panel not in obj:
        # try common nested structure
        if "fields" in obj and isinstance(obj["fields"], dict) and panel in obj["fields"]:
            field = np.asarray(obj["fields"][panel])
        else:
            raise KeyError(f"panel '{panel}' not found in adata.uns['scgeo']['{key}'] keys={list(obj.keys())}")
    else:
        field = np.asarray(obj[panel])

    # grid metadata
    grid = obj.get("grid", {})
    xmin = grid.get("xmin")
    xmax = grid.get("xmax")
    ymin = grid.get("ymin")
    ymax = grid.get("ymax")

    # If extent missing, infer from embedding ranges (safe)
    x, y = _get_embedding_xy(adata, basis=basis)
    if xmin is None: xmin = float(np.min(x))
    if xmax is None: xmax = float(np.max(x))
    if ymin is None: ymin = float(np.min(y))
    if ymax is None: ymax = float(np.max(y))

    fig, ax = _get_ax(ax=ax, figsize=figsize)

    im = ax.imshow(
        field,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=panel)

    if show_points:
        ax.scatter(x, y, s=3, c="white", alpha=alpha_points, linewidths=0)

    g0 = obj.get("group0")
    g1 = obj.get("group1")
    default_title = f"{key}: {panel}"
    if g0 is not None and g1 is not None:
        default_title += f" ({g0} vs {g1})"
    ax.set_title(title or default_title)
    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()

    return fig, ax
