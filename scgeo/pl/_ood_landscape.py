from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._result import PlotResult
from ._table_plot_utils import panel_axes
from ._utils import observation_labels, resolve_embedding


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    try:
        _, xy = resolve_embedding(adata, basis)
    except KeyError as exc:
        raise KeyError(f"Embedding {basis!r} not found in adata.obsm.") from exc
    return xy[:, :2]


def _to_numeric_obs(adata, key: str) -> np.ndarray:
    if key not in adata.obs:
        raise KeyError(f"'{key}' not found in adata.obs.")
    values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
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
            "group": observation_labels(adata.obs[groupby], groupby)[0].to_numpy(),
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
    return_data: bool = False,
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
    # Rows of the transposed histogram are y bins, columns are x bins.
    Xg, Yg = np.meshgrid(xc, yc, indexing="xy")
    if return_data:
        return Xg, Yg, mean_grid.T, counts.T, xedges, yedges
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
    axes=None,
    colorbar: bool = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_result: bool = False,
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
    axes
        Optional mapping containing ``landscape`` and, with groupby, ``summary``.
        Add ``colorbar`` to place the scale explicitly. Do not combine with ax.
    return_result
        Export source cells, full group summaries, actual contour bin means,
        counts and edges. Contours/thresholds are descriptive, not a test.
    """
    xy = _get_basis_xy(adata, basis)
    scores = _to_numeric_obs(adata, ood_key)
    if threshold is not None and (isinstance(threshold, bool) or not np.isfinite(threshold)):
        raise ValueError("threshold must be finite or None.")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if contour:
        if not np.isfinite(contour_quantile) or not 0 <= contour_quantile <= 1:
            raise ValueError("contour_quantile must be between zero and one.")
        for name, value, minimum in (("contour_gridsize", contour_gridsize, 2), ("contour_levels", contour_levels, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(top_n_groups, bool) or not isinstance(top_n_groups, (int, np.integer)) or top_n_groups < 1:
        raise ValueError("top_n_groups must be a positive integer.")
    flagged = _flag_from_threshold(scores, threshold=threshold)
    finite_xy = np.isfinite(xy).all(axis=1)

    mask_score = np.isfinite(scores) & finite_xy
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

    if axes is not None and ax is not None:
        raise ValueError("Supply ax or axes, not both.")
    if ax is not None and groupby is not None:
        raise ValueError("Pass ax only when groupby is None; use axes for a group summary.")
    names = ["landscape"] + (["summary"] if groupby is not None else [])
    supplied = axes if axes is not None else (None if ax is None else {"landscape": ax})
    if axes is not None and "colorbar" in axes:
        if cax is not None:
            raise ValueError("Supply cax or axes['colorbar'], not both.")
        cax = axes["colorbar"]
    if cax is not None:
        if supplied is None or not colorbar:
            raise ValueError("cax requires supplied axes and colorbar=True.")
        panel_axes(names + ["colorbar"], {**supplied, "colorbar": cax}, None)
    if supplied is None and groupby is not None:
        fig, (ax_main, ax_bar) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True,
            gridspec_kw={"width_ratios": [3.2, 1.15]},
        )
        panels = {"landscape": ax_main, "summary": ax_bar}
        made_fig = True
    else:
        fig, panels, made_fig = panel_axes(names, supplied, figsize)
        ax_main, ax_bar = panels["landscape"], panels.get("summary")
    finite_scores = scores[np.isfinite(scores)]
    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    norm.autoscale_None(finite_scores if len(finite_scores) else np.array([0., 1.]))
    artists = {}

    # background cells
    artists["background"] = ax_main.scatter(
        xy[finite_xy, 0],
        xy[finite_xy, 1],
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
        norm=norm,
        alpha=score_alpha,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    artists["scores"] = sc

    # flagged outlines
    if flagged_outline and np.any(flagged):
        artists["flagged"] = ax_main.scatter(
            xy[flagged & finite_xy, 0],
            xy[flagged & finite_xy, 1],
            s=flagged_size,
            facecolors="none",
            edgecolors="white",
            linewidths=flagged_lw,
            alpha=0.95,
            zorder=3,
        )

    # Preserve the existing binned-mean estimator; export the exact plotted grid.
    contour_level_value = None
    contour_grid = pd.DataFrame(columns=["x", "y", "mean_score", "count", "supported"])
    contour_bins = pd.DataFrame(columns=["dimension", "bin", "left", "right"])
    levels = []
    contour_status = "disabled"
    warnings = []
    if contour:
        contour_status = "no_finite_scores"
        if len(finite_scores):
            contour_level_value = float(np.nanquantile(finite_scores, contour_quantile))
            try:
                Xg, Yg, Zg, counts, xedges, yedges = _compute_density_grid(
                    xy[:, 0], xy[:, 1], scores, gridsize=contour_gridsize, return_data=True,
                )
                contour_grid = pd.DataFrame({"x": Xg.ravel(), "y": Yg.ravel(),
                    "mean_score": Zg.ravel(), "count": counts.ravel().astype(int),
                    "supported": np.isfinite(Zg).ravel()})
                contour_bins = pd.concat([pd.DataFrame({"dimension": dim,
                    "bin": np.arange(len(edges)-1), "left": edges[:-1], "right": edges[1:]})
                    for dim, edges in (("x", xedges), ("y", yedges))], ignore_index=True)
                contour_status = "no_supported_bins"
                if np.isfinite(Zg).any():
                    levels = [contour_level_value]
                    hi = float(np.nanmax(finite_scores))
                    if contour_levels > 1 and hi > contour_level_value:
                        levels = np.linspace(contour_level_value, hi, contour_levels).tolist()
                    artists["contour"] = ax_main.contour(
                        Xg, Yg, Zg, levels=levels, colors=contour_color,
                        linewidths=contour_lw, alpha=contour_alpha, zorder=4,
                    )
                    contour_status = "drawn" if any(
                        len(segment) for level in artists["contour"].allsegs for segment in level
                    ) else "no_level_crossing"
            except ValueError as exc:
                contour_status = "unavailable"
                warnings.append(f"Contour unavailable: {exc}")
        if contour_status not in {"drawn", "unavailable"}:
            warnings.append(f"Contour omitted: {contour_status}.")

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

    if colorbar:
        artists["colorbar"] = fig.colorbar(sc, ax=ax_main, cax=cax, fraction=0.046, pad=0.04)
        artists["colorbar"].set_label(ood_key)

    if ax_bar is not None and summary_df is not None and not summary_df.empty:
        top = summary_df.head(int(top_n_groups)).iloc[::-1]
        artists["summary"] = ax_bar.barh(top["group"], top[summary_kind], color="dimgray", alpha=0.9)
        ax_bar.set_title("OOD summary")
        ax_bar.set_xlabel(summary_kind)
        ax_bar.grid(axis="x", alpha=0.25)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)

    if show:
        plt.show()

    threshold_used = float(np.nanquantile(finite_scores, 0.95)) if threshold is None and len(finite_scores) else threshold
    if return_result:
        cells = pd.DataFrame({"cell": adata.obs_names.astype(str), "obs_position": np.arange(adata.n_obs),
            "x": xy[:, 0], "y": xy[:, 1], "score": scores, "score_available": np.isfinite(scores),
            "flagged": flagged, "score_displayed": mask_score, "background_displayed": finite_xy,
            "outline_displayed": flagged & finite_xy & flagged_outline})
        data = {"cells": cells, "contour_grid": contour_grid, "contour_bins": contour_bins}
        if summary_df is not None:
            data["summary"] = summary_df.copy()
            data["summary"]["displayed"] = np.arange(len(summary_df)) < top_n_groups
            cells["group"] = adata.obs[groupby].to_numpy()
            cells["group_label"] = observation_labels(adata.obs[groupby], groupby)[0].to_numpy()
        metadata = {"plot": "ood_landscape", "basis": resolve_embedding(adata, basis)[0],
            "ood_key": ood_key, "threshold": threshold_used,
            "threshold_source": "finite_score_quantile_0.95" if threshold is None else "user_supplied",
            "show_only_flagged": show_only_flagged, "groupby": groupby, "summary_kind": summary_kind,
            "summary_denominator": "all_input_group_cells; unavailable scores are not flagged",
            "contour_status": contour_status, "contour_levels": levels,
            "contour_quantile": contour_quantile, "contour_grid_size": contour_gridsize,
            "contour_min_points_per_bin": 3, "contour_level_value": contour_level_value,
            "norm": {"type": type(norm).__name__, "vmin": norm.vmin, "vmax": norm.vmax},
            "color_scale_uses_full_data": True, "owns_figure": made_fig,
            "export_scope": "created_figure" if made_fig else "parent_figure",
            "interpretation": "Descriptive threshold and local score means; no inferential OOD test."}
        return PlotResult(fig, panels, data, metadata, warnings, artists)
    if return_data:
        out = {"flagged": flagged, "threshold_used": threshold_used,
               "summary": summary_df, "contour_level_value": contour_level_value}
        return fig, ax_main if groupby is None else (ax_main, ax_bar), out
    if made_fig or axes is not None:
        return fig, ax_main if groupby is None else (ax_main, ax_bar)
    return ax_main
