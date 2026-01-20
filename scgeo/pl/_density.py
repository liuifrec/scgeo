from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ._utils import _get_embedding_xy

def _smooth2d(H: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple box blur smoothing (NumPy-only).
    k must be odd. k=3 or 5 is usually enough for contour plots.
    """
    if k <= 1:
        return H
    if k % 2 == 0:
        raise ValueError("smooth_k must be odd")
    pad = k // 2
    Hp = np.pad(H, pad_width=pad, mode="edge")
    out = np.zeros_like(H, dtype=float)
    # box filter via summed area would be faster, but this is small and robust
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            out[i, j] = Hp[i:i+k, j:j+k].mean()
    return out

def embedding_density(
    adata,
    groupby: str,
    *,
    basis: str = "umap",
    groups: Optional[Iterable[str]] = None,
    gridsize: int = 160,
    normalize: str = "per_group",  # "per_group" | "global"
    cmap: str = "magma",
    contour: bool = False,
    contour_levels: int = 12,
    imshow_alpha: float = 0.85,
    transparent_background: bool = True,
    mask_zeros: bool = True,
    background: Optional[str] = None,
    figsize=None,
    show: bool = True,
    smooth_k: int = 5,               # smoothing kernel for contour mode
    log1p: bool = True,              # makes sparse densities visible
    contour_lines: bool = True,       # draw topographic lines
    contour_linewidth: float = 0.7,
    contour_alpha: float = 0.95,
    contour_level_mode: str = "quantile"  # "linear" | "quantile"


):
    """Plot embedding density per group (small multiples).

    Primary goal: quick, stable, and notebook-friendly density diagnostics.

    Implementation:
      - Computes a 2D histogram on the embedding.
      - Renders either an ``imshow`` heatmap (default) or a ``contourf`` plot
        (set ``contour=True``) to avoid the "blocky / too dark" look.

    Parameters
    ----------
    contour
        If True, draw a filled contour plot (``contourf``) instead of imshow.
    transparent_background
        If True, set figure + axes patches to transparent so the density layers
        don't sit on a dark rectangle in some notebook themes.
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

    # background handling
    if background is not None:
        fig.patch.set_facecolor(background)
        fig.patch.set_alpha(1.0)
        for ax in axes:
            ax.set_facecolor(background)
            ax.patch.set_alpha(1.0)
    elif transparent_background:
        # fully transparent fig + axes background
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.patch.set_alpha(0.0)

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
                x[m],
                y[m],
                bins=gridsize,
                range=[[xmin, xmax], [ymin, ymax]],
            )
            global_max = max(global_max, float(H.max()))
        if global_max <= 0:
            global_max = 1.0

    last_mappable = None


    for i, grp in enumerate(groups):
        ax = axes[i]
        ax.set_title(str(grp))

        m = g == grp
        if m.sum() == 0:
            ax.axis("off")
            continue

        H, xe, ye = np.histogram2d(
            x[m],
            y[m],
            bins=gridsize,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        H = H.T  # for imshow/contour orientation
        H = H.astype(float)

        # mask zeros so we don't paint the entire panel dark
        # mask zeros so we don't paint the entire panel dark
        if mask_zeros:
            H_plot = np.ma.masked_where(H <= 0, H)
        else:
            H_plot = H


        vmax = None
        if normalize == "per_group":
            vmax = float(H.max()) if H.max() > 0 else 1.0
        elif normalize == "global":
            vmax = float(global_max)
        # Optional transform to make sparse structure visible


        if contour:
            ax.set_facecolor("none")
            ax.patch.set_alpha(0.0)
            # Smooth first (contours hate sparse spike maps)
            if smooth_k and smooth_k > 1:
                H_plot = _smooth2d(H_plot, k=int(smooth_k))

            # log transform helps reveal structure
            if log1p:
                H_plot = np.log1p(H_plot)

            # mask zeros so we don't paint the entire panel dark
            if mask_zeros: 
                Hm = np.ma.masked_less_equal(H_plot, 0.0) 
            else: 
                Hm = H_plot


            # Bin centers
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            Xc, Yc = np.meshgrid(xc, yc)

            vmax_eff = float(np.nanmax(H_plot)) if np.size(H_plot) else 1.0
            if vmax_eff <= 0:
                vmax_eff = 1.0

            # IMPORTANT: do NOT include 0 as a level for sparse maps
            Z = H_plot.copy()

            if smooth_k and smooth_k > 1:
                Z = _smooth2d(Z, k=int(smooth_k))
            if log1p:
                Z = np.log1p(Z)

            if mask_zeros:
                Zm = np.ma.masked_less_equal(Z, 0.0)
            else:
                Zm = Z

            vals = np.asarray(Zm.compressed() if np.ma.is_masked(Zm) else Zm.ravel(), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.axis("off")
                continue

            if contour_level_mode == "quantile":
                qs = np.linspace(0.60, 0.995, int(contour_levels))  # start high -> avoids huge flat background
                levels = np.unique(np.quantile(vals, qs))
            else:
                lo = float(np.quantile(vals, 0.60))
                hi = float(np.max(vals))
                levels = np.unique(np.linspace(lo, hi, int(contour_levels)))

            # safety
            if levels.size < 3:
                lo = float(np.min(vals))
                hi = float(np.max(vals))
                levels = np.linspace(lo, hi, 6)

            mappable = ax.contourf(
                Xc, Yc, Hm,
                levels=levels,
                cmap=cmap,
                alpha=float(contour_alpha),
            )

            if contour_lines:
                ax.contour(
                    Xc, Yc, Hm,
                    levels=levels,
                    colors="k",
                    linewidths=float(contour_linewidth),
                    alpha=0.35,
                )

            # Ensure white background for the axes
            ax.set_facecolor("white")


        else:
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad((1, 1, 1, 0))  # masked -> transparent

            mappable = ax.imshow(
                H_plot,
                origin="lower",
                extent=[xmin, xmax, ymin, ymax],
                aspect="equal",
                cmap=cm,
                vmin=0,
                vmax=vmax,
                alpha=float(imshow_alpha),
            )

        last_mappable = mappable
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if last_mappable is not None:
        # one colorbar for the whole figure
        fig.colorbar(
            last_mappable,
            ax=axes[:n].tolist(),
            fraction=0.025,
            pad=0.02,
            label="cell density (hist2d)",
        )

    fig.suptitle(f"Embedding density by {groupby} ({basis})", y=1.02)

    if show:
        plt.show()

    return fig, axes[:n]
