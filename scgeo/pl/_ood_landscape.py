from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    xy = np.asarray(adata.obsm[key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (n_cells, >=2).")
    return xy[:, :2]


def _to_numeric_obs(adata, key: str) -> np.ndarray:
    if key not in adata.obs:
        raise KeyError(f"'{key}' not found in adata.obs.")
    values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
    return values


def _flag_from_threshold(values: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    mask = np.isfinite(values)
    if threshold is None:
        if not np.any(mask):
            return np.zeros_like(values, dtype=bool)
        threshold = float(np.nanquantile(values[mask], 0.95))
    return mask & (values >= float(threshold))


def _group_ood_summary(
    adata,
    *,
    groupby: str,
    flagged: np.ndarray,
    value: np.ndarray,
) -> pd.DataFrame:
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not found in adata.obs.")

    df = pd.DataFrame(
        {
            "group": adata.obs[groupby].astype(str).values,
            "flagged": flagged.astype(bool),
            "value": value,
        },
        index=adata.obs_names,
    )
    out = (
        df.groupby("group", observed=False)
        .agg(
            n=("group", "size"),
            flagged_n=("flagged", "sum"),
            flagged_frac=("flagged", "mean"),
            mean_score=("value", "mean"),
            median_score=("value", "median"),
        )
        .reset_index()
        .sort_values(["flagged_frac", "mean_score"], ascending=[False, False])
    )
    return out


def _compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    gridsize: int = 150,
    min_points_per_bin: int = 3,
):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[mask]
    y = y[mask]
    values = values[mask]

    if x.size == 0:
        raise ValueError("No finite points available for contour computation.")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    if np.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if np.isclose(ymin, ymax):
        ymax = ymin + 1.0

    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    sums, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=values)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = sums / counts

    mean_grid[counts < min_points_per_bin] = np.nan

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    return Xg, Yg, mean_grid.T


def ood_landscape(
    adata,
    *,
    ood_key: str = "scgeo_ood",
    basis: str = "umap",
    threshold: Optional[float] = None,
    show_only_flagged: bool = False,
    flagged_outline: bool = True,
    flagged_size: float = 28.0,
    flagged_lw: float = 0.8,
    bg_size: float = 6.0,
    bg_alpha: float = 0.12,
    score_size: float = 10.0,
    score_alpha: float = 0.85,
    cmap: str = "magma",
    contour: bool = True,
    contour_quantile: float = 0.95,
    contour_levels: int = 1,
    contour_color: str = "cyan",
    contour_lw: float = 1.6,
    contour_alpha: float = 0.95,
    contour_gridsize: int = 150,
    groupby: Optional[str] = None,
    top_n_groups: int = 10,
    summary_kind: str = "flagged_frac",
    figsize: tuple[float, float] = (10.5, 5.2),
    title: Optional[str] = None,
    ax=None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Plot a continuous OOD landscape on an embedding, with optional contour and
    optional right-side group summary.

    Parameters
    ----------
    adata
        AnnData with embedding in ``adata.obsm`` and numeric OOD scores in ``adata.obs``.
    ood_key
        Observation column containing numeric OOD scores.
    basis
        Embedding basis, e.g. ``'umap'`` or ``'X_umap'``.
    threshold
        OOD threshold for flagged cells. If None, uses the 95th percentile.
    show_only_flagged
        If True, only plot flagged cells in the score layer.
    groupby
        Optional obs column for a right-side group summary.
    summary_kind
        One of ``{'flagged_frac', 'flagged_n', 'mean_score', 'median_score'}``.
    return_data
        If True, return computed summary artifacts.
    """
    xy = _get_basis_xy(adata, basis)
    scores = _to_numeric_obs(adata, ood_key)
    flagged = _flag_from_threshold(scores, threshold=threshold)

    mask_score = np.isfinite(scores)
    if show_only_flagged:
        mask_score &= flagged

    summary_df = None
    if groupby is not None:
        summary_df = _group_ood_summary(
            adata,
            groupby=groupby,
            flagged=flagged,
            value=scores,
        )
        if summary_kind not in {"flagged_frac", "flagged_n", "mean_score", "median_score"}:
            raise ValueError(
                "summary_kind must be one of "
                "{'flagged_frac', 'flagged_n', 'mean_score', 'median_score'}."
            )

    made_fig = False
    if ax is not None and groupby is not None:
        raise ValueError("Pass ax only when groupby is None. For group summary, let the function create axes.")

    if ax is None:
        if groupby is None:
            fig, ax_main = plt.subplots(figsize=figsize)
            ax_bar = None
        else:
            fig, (ax_main, ax_bar) = plt.subplots(
                1,
                2,
                figsize=figsize,
                gridspec_kw={"width_ratios": [3.2, 1.15]},
            )
        made_fig = True
    else:
        fig = ax.figure
        ax_main = ax
        ax_bar = None

    # background cells
    ax_main.scatter(
        xy[:, 0],
        xy[:, 1],
        s=bg_size,
        c="lightgrey",
        alpha=bg_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    # continuous score layer
    sc = ax_main.scatter(
        xy[mask_score, 0],
        xy[mask_score, 1],
        c=scores[mask_score],
        s=score_size,
        cmap=cmap,
        alpha=score_alpha,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # flagged outlines
    if flagged_outline and np.any(flagged):
        ax_main.scatter(
            xy[flagged, 0],
            xy[flagged, 1],
            s=flagged_size,
            facecolors="none",
            edgecolors="white",
            linewidths=flagged_lw,
            alpha=0.95,
            zorder=3,
        )

    # contour on high-OOD field
    contour_level_value = None
    if contour:
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            contour_level_value = float(np.nanquantile(scores[finite_mask], contour_quantile))
            try:
                Xg, Yg, Zg = _compute_density_grid(
                    xy[:, 0],
                    xy[:, 1],
                    scores,
                    gridsize=contour_gridsize,
                )
                if np.isfinite(Zg).sum() > 0:
                    levels = [contour_level_value]
                    if contour_levels > 1:
                        hi = float(np.nanmax(scores[finite_mask]))
                        if hi > contour_level_value:
                            levels = np.linspace(contour_level_value, hi, contour_levels)
                    ax_main.contour(
                        Xg,
                        Yg,
                        Zg,
                        levels=levels,
                        colors=contour_color,
                        linewidths=contour_lw,
                        alpha=contour_alpha,
                        zorder=4,
                    )
            except ValueError:
                pass

    basis_name = basis[2:] if basis.startswith("X_") else basis
    ax_main.set_xlabel(f"{basis_name.upper()}1")
    ax_main.set_ylabel(f"{basis_name.upper()}2")
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f"ScGeo: OOD landscape ({ood_key})"
    ax_main.set_title(title)

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label(ood_key)

    if ax_bar is not None and summary_df is not None and not summary_df.empty:
        top = summary_df.head(int(top_n_groups)).iloc[::-1]
        ax_bar.barh(top["group"], top[summary_kind], color="dimgray", alpha=0.9)
        ax_bar.set_title("OOD summary")
        ax_bar.set_xlabel(summary_kind)
        ax_bar.grid(axis="x", alpha=0.25)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)

    if show:
        plt.show()

    if return_data:
        out = {
            "flagged": flagged,
            "threshold_used": float(np.nanquantile(scores[np.isfinite(scores)], 0.95))
            if threshold is None and np.any(np.isfinite(scores))
            else threshold,
            "summary": summary_df,
            "contour_level_value": contour_level_value,
        }
        if groupby is None:
            return fig, ax_main, out
        return fig, (ax_main, ax_bar), out

    if made_fig:
        if groupby is None:
            return fig, ax_main
        return fig, (ax_main, ax_bar)
    return ax_main