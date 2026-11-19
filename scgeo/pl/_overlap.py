from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ._utils import _get_embedding_xy, _get_ax, resolve_embedding
from ._result import PlotResult


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
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: bool = True,
    cax=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """
    Visualize density overlap results from `tl.density_overlap`.

    Supports:
    - grid-based fields stored in `adata.uns["scgeo"][key]`
    - or per-cell scores in `adata.obs[key]` (fallback)

    The per-cell fallback forwards the supplied axis and color controls to
    ``score_embedding``. Grid exports include every pixel's value/center and
    all observation coordinates; metadata records whether extent was supplied
    or inferred. Missing pixels remain missing. ``return_result=True`` opts
    into PlotResult, and ``return_data=True`` adds tables to the legacy tuple.
    """
    obj = adata.uns.get("scgeo", {}).get(key)

    # Fallback: per-cell score
    if obj is None:
        if key in adata.obs:
            # plot as per-cell score on embedding
            import scgeo as sg
            result = sg.pl.score_embedding(
                adata,
                key,
                basis=basis,
                ax=ax, figsize=figsize, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                colorbar=colorbar, cax=cax, return_result=True,
                title=title or f"{key} (cell score)",
                show=show,
            )
            result.metadata.update(plot="density_overlap_grid", source_mode="observation_score")
            if return_result:
                return result
            if return_data:
                return result.fig, result.axes["embedding"], result.data
            return result.fig, result.axes["embedding"]
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

    if field.ndim != 2 or not field.size:
        raise ValueError("Stored density field must be a nonempty two-dimensional array.")
    field = np.asarray(field, dtype=float)
    if np.isinf(field).any():
        raise ValueError("Stored density field must contain finite values or missing pixels.")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if cax is not None:
        if ax is None or not colorbar:
            raise ValueError("cax requires ax and colorbar=True.")
        from ._table_plot_utils import panel_axes
        panel_axes(["density", "colorbar"], {"density": ax, "colorbar": cax}, None)
    # grid metadata
    grid = obj.get("grid", {})
    xmin = grid.get("xmin")
    xmax = grid.get("xmax")
    ymin = grid.get("ymin")
    ymax = grid.get("ymax")

    # If extent missing, infer from embedding ranges (safe)
    emb_key, coords = resolve_embedding(adata, basis)
    x, y = coords[:, 0], coords[:, 1]
    finite = np.isfinite(coords[:, :2]).all(axis=1)
    inferred = [name for name, value in zip(("xmin", "xmax", "ymin", "ymax"),
                                             (xmin, xmax, ymin, ymax)) if value is None]
    if inferred and not finite.any():
        raise ValueError("Grid extent is missing and no finite embedding coordinates exist.")
    if xmin is None: xmin = float(np.min(x[finite]))
    if xmax is None: xmax = float(np.max(x[finite]))
    if ymin is None: ymin = float(np.min(y[finite]))
    if ymax is None: ymax = float(np.max(y[finite]))
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("Grid extent must be finite and increasing along both dimensions.")

    fig, ax = _get_ax(ax=ax, figsize=figsize)

    im = ax.imshow(
        field,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        cmap=cmap,
        norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    artists = {"image": im}
    if colorbar:
        artists["colorbar"] = fig.colorbar(im, ax=ax, cax=cax, fraction=0.03, pad=0.02, label=panel)

    if show_points:
        artists["points"] = ax.scatter(x[finite], y[finite], s=3, c="white", alpha=alpha_points, linewidths=0)

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

    ny, nx = field.shape
    gx = xmin + (np.arange(nx) + .5) * (xmax - xmin) / nx
    gy = ymin + (np.arange(ny) + .5) * (ymax - ymin) / ny
    xx, yy = np.meshgrid(gx, gy)
    tables = {"grid": pd.DataFrame({"ix": np.tile(np.arange(nx), ny), "iy": np.repeat(np.arange(ny), nx),
                                    "x": xx.ravel(), "y": yy.ravel(), "value": field.ravel()}),
              "cells": pd.DataFrame({"cell": adata.obs_names.astype(str), "obs_position": np.arange(adata.n_obs),
                                     "x": x, "y": y, "displayed": finite & show_points})}
    if return_result:
        return PlotResult(fig, {"density": ax}, tables,
                          {"plot": "density_overlap_grid", "source_mode": "stored_grid", "key": key,
                           "panel": panel, "basis": emb_key, "extent": [xmin, xmax, ymin, ymax],
                           "inferred_extent_bounds": inferred, "field_shape": [ny, nx],
                           "norm": {"type": type(im.norm).__name__, "vmin": im.norm.vmin, "vmax": im.norm.vmax},
                           "group0": g0, "group1": g1, "analysis_recomputed": False}, artists=artists)
    if return_data:
        return fig, ax, tables
    return fig, ax
