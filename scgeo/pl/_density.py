from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ._utils import _get_embedding_xy, _get_ax


def embedding_density(
    adata,
    groupby: str,
    *,
    basis: str = "umap",
    groups: Optional[Iterable[str]] = None,
    gridsize: int = 160,
    normalize: str = "per_group",  # "per_group" | "global"
    cmap: str = "magma",
    figsize=None,
    show: bool = True,
):
    """
    Plot embedding density per group (small multiples).

    Uses a 2D histogram on the embedding (fast and stable).
    """
    if groupby not in adata.obs:
        raise KeyError(f"groupby '{groupby}' not found in adata.obs")

    x, y = _get_embedding_xy(adata, basis=basis)
    s = adata.obs[groupby]

    # drop NaNs safely
    valid = s.notna().to_numpy()
    if not valid.any():
        raise ValueError(f"All values in obs['{groupby}'] are NaN")

    g = s[valid].astype(str).to_numpy()
    x = x[valid]
    y = y[valid]

    if groups is None:
        groups = list(pd.unique(g))
    else:
        groups = [str(z) for z in groups]
    n = len(groups)
    if n == 0:
        raise ValueError("No groups to plot")

    # layout
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.3 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    # global range
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # precompute global vmax if requested
    global_max = 0.0
    if normalize == "global":
        for grp in groups:
            m = g == grp
            if m.sum() == 0:
                continue
            H, _, _ = np.histogram2d(
                x[m], y[m],
                bins=gridsize,
                range=[[xmin, xmax], [ymin, ymax]],
            )
            global_max = max(global_max, float(H.max()))
        if global_max <= 0:
            global_max = 1.0

    for i, grp in enumerate(groups):
        ax = axes[i]
        m = g == grp
        ax.set_title(str(grp))

        if m.sum() == 0:
            ax.axis("off")
            continue

        H, xe, ye = np.histogram2d(
            x[m], y[m],
            bins=gridsize,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        H = H.T  # for imshow

        vmax = None
        if normalize == "per_group":
            vmax = H.max() if H.max() > 0 else 1.0
        elif normalize == "global":
            vmax = global_max

        im = ax.imshow(
            H,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="equal",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # one colorbar for the whole figure
    fig.colorbar(im, ax=axes[:n].tolist(), fraction=0.025, pad=0.02, label="cell density (hist2d)")

    fig.suptitle(f"Embedding density by {groupby} ({basis})", y=1.02)

    if show:
        plt.show()

    return fig, axes[:n]
