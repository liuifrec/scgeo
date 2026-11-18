"""Matched, condition-separated projected-velocity fields on a common UMAP grid."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from ._result import PlotResult
from ._velocity_field import velocity_field
from ._velocity_grid import _prepare_velocity_field


def velocity_condition_compare(
    adata,
    *,
    condition_key,
    group0,
    group1,
    basis="umap",
    mode="grid",
    grid_size=25,
    n_neighbors=30,
    radius=None,
    min_neighbors=5,
    min_coherence=0.2,
    scale="auto",
    color_by="coherence",
    cmap="viridis",
    max_points=3500,
    random_state=0,
    stream_density=1.0,
    axes=None,
    figsize=(15.5, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Compare two observed condition velocity fields at identical grid points.

    Each field uses only the cells in its selected condition. Grid coordinates,
    Gaussian radius, neighbor cap, support thresholds, color norm, view limits,
    and (in grid mode) the arrow multiplier are shared. The third panel shows
    cosine agreement of projected local means *only where both are supported*;
    singly supported positions are indicated but never assigned a direction.
    It is descriptive, not a causal or longitudinal cell-trajectory estimate.

    ``adata.obsm['X_<basis>']`` and matching ``velocity_<basis>`` are required.
    Reusing an upstream projected velocity does not refit its kinetics.
    ``return_result=True`` enables source-grid, support-mask and parameter
    exports via :class:`scgeo.pl.PlotResult`.
    """
    if condition_key not in adata.obs:
        raise KeyError(f"{condition_key!r} is missing from adata.obs.")
    if pd.isna(group0) or pd.isna(group1) or str(group0) == str(group1):
        raise ValueError("group0 and group1 must be distinct nonmissing labels.")
    g0, g1 = str(group0), str(group1)
    if mode not in {"grid", "stream"}:
        raise ValueError("mode must be 'grid' or 'stream'.")
    if color_by not in {"coherence", "speed", None}:
        raise ValueError("color_by must be 'coherence', 'speed', or None.")
    if axes is not None:
        if not isinstance(axes, Mapping) or set(axes) != {"reference", "comparison", "agreement"}:
            raise ValueError("axes must map 'reference', 'comparison', and 'agreement'.")
        if len({id(ax.figure) for ax in axes.values()}) != 1:
            raise ValueError("All supplied axes must belong to one figure.")
    # Evaluate the union first. The union determines the shared grid *and*
    # neighborhood radius, not a radius separately optimized for either group.
    pooled = _prepare_velocity_field(
        adata, basis=basis, condition_key=condition_key, groups=[g0, g1],
        grid_size=grid_size, n_neighbors=n_neighbors, radius=radius,
        min_neighbors=min_neighbors, min_coherence=min_coherence,
    )
    extent = pooled["metadata"]["grid_extent"]
    common_radius = pooled["metadata"]["radius"]
    configured = {
        "basis": basis, "condition_key": condition_key,
        "grid_size": grid_size, "grid_extent": extent,
        "radius": common_radius, "n_neighbors": n_neighbors,
        "min_neighbors": min_neighbors, "min_coherence": min_coherence,
    }
    p0 = _prepare_velocity_field(adata, groups=[g0], **configured)
    p1 = _prepare_velocity_field(adata, groups=[g1], **configured)
    frame0, frame1 = p0["grid"], p1["grid"]
    if not (
        np.array_equal(frame0[["grid_id", "x", "y"]].to_numpy(),
                       frame1[["grid_id", "x", "y"]].to_numpy())
    ):
        raise AssertionError("The two projected-velocity fields have mismatched grid coordinates.")
    if scale == "auto" and mode == "grid":
        speeds = np.concatenate([
            g.loc[g.supported, "speed"].to_numpy(dtype=float)
            for g in (frame0, frame1)
        ])
        nx, ny = pooled["x"], pooled["y"]
        spacing = min(float(np.diff(nx)[0]), float(np.diff(ny)[0]))
        shared_scale = 0.65 * spacing / float(np.quantile(speeds, 0.9)) if len(speeds) else 1.0
    elif scale == "auto":
        shared_scale = "auto"
    else:
        if isinstance(scale, bool) or not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be 'auto' or a positive finite number.")
        shared_scale = float(scale)
    if color_by == "coherence":
        norm = Normalize(vmin=0.0, vmax=1.0)
    elif color_by == "speed":
        speeds = np.concatenate([
            g.loc[g.supported, "speed"].to_numpy(dtype=float)
            for g in (frame0, frame1)
        ])
        norm = Normalize(vmin=0.0, vmax=max(float(speeds.max()) if len(speeds) else 0.0, 1e-12))
    else:
        norm = None
    owns_figure = axes is None
    if owns_figure:
        fig, triplet = plt.subplots(1, 3, figsize=figsize)
        axes = dict(zip(("reference", "comparison", "agreement"), triplet))
    else:
        fig = axes["reference"].figure
    xmin, xmax, ymin, ymax = extent
    padding = max(xmax - xmin, ymax - ymin) * 0.05
    xlim, ylim = (xmin - padding, xmax + padding), (ymin - padding, ymax + padding)
    common = {
        **configured,
        "mode": mode,
        "color_by": color_by,
        "cmap": cmap,
        "norm": norm,
        "scale": shared_scale,
        "max_points": max_points,
        "random_state": random_state,
        "stream_density": stream_density,
        "background": "selected",
        "legend": False,
        "show_colorbar": False,
        "xlim": xlim,
        "ylim": ylim,
        "show": False,
        "return_result": True,
    }
    reference = velocity_field(adata, groups=[g0], ax=axes["reference"], title=g0, **common)
    comparison = velocity_field(adata, groups=[g1], ax=axes["comparison"], title=g1, **common)
    grid0 = reference.data["grid"].reset_index(drop=True)
    grid1 = comparison.data["grid"].reset_index(drop=True)
    if not np.array_equal(grid0[["grid_id", "x", "y"]].to_numpy(),
                          grid1[["grid_id", "x", "y"]].to_numpy()):
        raise AssertionError("Rendered grid coordinates differ despite shared settings.")
    paired = grid0[["grid_id", "x", "y", "vx", "vy", "speed", "coherence",
                    "n_neighbors", "supported", "reason"]].copy()
    paired.rename(columns={c: f"{c}_reference" for c in
                           ("vx", "vy", "speed", "coherence", "n_neighbors", "supported", "reason")}, inplace=True)
    for key in ("vx", "vy", "speed", "coherence", "n_neighbors", "supported", "reason"):
        paired[f"{key}_comparison"] = grid1[key].to_numpy()
    s0 = paired.supported_reference.to_numpy(dtype=bool)
    s1 = paired.supported_comparison.to_numpy(dtype=bool)
    paired["support_status"] = np.select(
        [s0 & s1, s0 & ~s1, ~s0 & s1],
        ["both_supported", "reference_only", "comparison_only"],
        default="both_unsupported",
    )
    v0 = paired[["vx_reference", "vy_reference"]].to_numpy(dtype=float)
    v1 = paired[["vx_comparison", "vy_comparison"]].to_numpy(dtype=float)
    norms = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    mutual = s0 & s1 & (norms > 0)
    cosine = np.full(len(paired), np.nan)
    cosine[mutual] = np.sum(v0[mutual] * v1[mutual], axis=1) / norms[mutual]
    paired["direction_cosine"] = np.clip(cosine, -1, 1)
    ax = axes["agreement"]
    x, y = paired.x.to_numpy(), paired.y.to_numpy()
    artists = {"direction": ax.scatter(x[mutual], y[mutual], c=cosine[mutual],
                                        cmap="coolwarm_r", vmin=-1, vmax=1,
                                        marker="s", s=37, linewidths=0, zorder=2)}
    for status, mark, color in (
        ("reference_only", "x", "#177E89"),
        ("comparison_only", "+", "#C45A55"),
    ):
        rows = paired.support_status.to_numpy() == status
        if rows.any():
            ax.scatter(x[rows], y[rows], marker=mark, color=color, s=22,
                       alpha=0.8, linewidths=0.8, zorder=3)
    ax.set(xlim=xlim, ylim=ylim, title="Local direction agreement",
           xlabel=f"{basis.upper()} 1", ylabel=f"{basis.upper()} 2", xticks=[], yticks=[])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    artists["agreement_colorbar"] = fig.colorbar(
        artists["direction"], ax=ax, fraction=0.05, pad=0.02,
        label="Projected-velocity cosine (both supported)",
    )
    if owns_figure:
        fig.suptitle(title or f"Condition-specific projected velocity: {g0} vs {g1}", y=1.01)
        fig.tight_layout()
    metadata = {
        "plot": "velocity_condition_compare",
        "condition_key": condition_key,
        "groups": [g0, g1],
        "display_basis": basis,
        "mode": mode,
        "grid_size": pooled["metadata"]["grid_size"],
        "grid_extent": extent,
        "radius": common_radius,
        "radius_source": "pooled_selected_conditions" if radius is None else "provided",
        "shared_scale": shared_scale if mode == "grid" else None,
        "color_by": color_by,
        "shared_color_limits": None if norm is None else [norm.vmin, norm.vmax],
        "n_both_supported": int(mutual.sum()),
        "n_reference_only": int((s0 & ~s1).sum()),
        "n_comparison_only": int((~s0 & s1).sum()),
        "n_both_unsupported": int((~s0 & ~s1).sum()),
        "quantitative_interpretation": "display-space projected velocity only; not latent-space shift alignment",
    }
    result = PlotResult(
        fig=fig, axes=axes,
        data={"paired_grid": paired, "reference_grid": grid0,
              "comparison_grid": grid1, "reference_cells": reference.data["cells"],
              "comparison_cells": comparison.data["cells"],
              "reference_vectors": reference.data["vectors"],
              "comparison_vectors": comparison.data["vectors"]},
        metadata=metadata,
        warnings=["Directional agreement is descriptive, conditional on grid support and smoothing.",
                  "These are projected display vectors; no RNA kinetics, trajectories or time were inferred."],
        artists=artists,
    )
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return fig, axes, result.data
    return fig, axes
