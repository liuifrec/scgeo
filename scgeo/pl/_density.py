"""Comparable embedding histograms with explicit source counts and omissions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from scipy.ndimage import convolve1d

from ._result import PlotResult
from ._utils import resolve_embedding


def _smooth2d(H: np.ndarray, k: int = 3) -> np.ndarray:
    """Apply one box blur, retaining the historical replicated-edge boundary."""
    if k <= 1:
        return H
    if k % 2 == 0:
        raise ValueError("smooth_k must be odd")
    # Separable convolution performs the same single 2D box blur without the
    # tiny residual values that a running-sum filter can leave in empty bins.
    weights = np.full(k, 1.0 / k)
    return convolve1d(
        convolve1d(np.asarray(H, dtype=float), weights, axis=0, mode="nearest"),
        weights,
        axis=1,
        mode="nearest",
    )


def _density_labels(series, groups):
    declared = (
        list(series.cat.categories)
        if isinstance(series.dtype, pd.CategoricalDtype)
        else list(pd.unique(series.dropna()))
    )
    labels = [str(value) for value in declared]
    if len(labels) != len(set(labels)):
        raise ValueError("Group labels must remain distinct when converted to strings.")
    values = series.astype(object).to_numpy(copy=True)
    missing = series.isna().to_numpy()
    canonical = np.empty(len(series), dtype=object)
    canonical[:] = None
    canonical[~missing] = [str(value) for value in values[~missing]]
    if groups is None:
        order = labels
    elif isinstance(groups, str):
        order = [groups]
    else:
        order = [str(value) for value in groups]
    if not order:
        raise ValueError("No groups to plot; supply a known nonmissing group.")
    if len(order) != len(set(order)):
        raise ValueError("groups must contain unique labels.")
    unknown = [label for label in order if label not in labels]
    if unknown:
        raise ValueError(f"Unknown groups for {series.name!r}: {unknown}.")
    return canonical, values, missing, order


def _density_bounds(value, name, length=2):
    try:
        bounds = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite increasing bounds.") from exc
    if bounds.shape != (length,) or not np.isfinite(bounds).all():
        raise ValueError(f"{name} must contain {length} finite bounds.")
    if np.any(bounds[::2] >= bounds[1::2]):
        raise ValueError(f"{name} must contain increasing lower/upper bounds.")
    with np.errstate(over="ignore"):
        if not np.isfinite(bounds[1::2] - bounds[::2]).all():
            raise ValueError(f"{name} has an unrepresentable coordinate range.")
    return bounds


def _density_axes(order, ax, axes, cax):
    if ax is not None and axes is not None:
        raise ValueError("Supply ax or axes, not both.")
    if ax is not None:
        if len(order) != 1:
            raise ValueError(
                "ax is a single-group convenience; use axes for several groups."
            )
        supplied = [ax]
    elif isinstance(axes, Mapping):
        supplied = []
        for label in order:
            if label in axes:
                supplied.append(axes[label])
            elif f"density:{label}" in axes:
                supplied.append(axes[f"density:{label}"])
            else:
                raise ValueError(f"axes is missing an axis for group {label!r}.")
    elif axes is None:
        supplied = None
    elif isinstance(axes, Axes):
        supplied = [axes]
    else:
        supplied = list(np.asarray(axes, dtype=object).ravel())
    if supplied is None:
        if cax is not None:
            raise ValueError("cax requires supplied density axes in the same figure.")
        return None
    if len(supplied) != len(order) or not all(
        isinstance(item, Axes) for item in supplied
    ):
        raise ValueError("Supply exactly one Matplotlib axis per requested group.")
    if len({id(item) for item in supplied}) != len(supplied):
        raise ValueError("Density panels require distinct axes.")
    container = supplied[0].figure
    if any(item.figure is not container for item in supplied):
        raise ValueError(
            "All density axes must belong to the same Figure or SubFigure."
        )
    if cax is not None and (
        not isinstance(cax, Axes)
        or cax.figure is not container
        or any(cax is item for item in supplied)
    ):
        raise ValueError("cax must be a distinct axis in the same Figure or SubFigure.")
    return np.asarray(supplied, dtype=object)


def _density_norm(norm, vmin, vmax, upper, values):
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if norm is not None and not isinstance(norm, Normalize):
        raise TypeError("norm must be a Matplotlib Normalize instance.")
    if norm is None:
        color_norm = Normalize(
            vmin=0.0 if vmin is None else vmin, vmax=upper if vmax is None else vmax
        )
    elif norm.vmin is None or norm.vmax is None:
        # Fill unresolved bounds without changing a caller's shared normalizer.
        color_norm = copy(norm)
        if color_norm.vmin is None:
            color_norm.vmin = 0.0
        if color_norm.vmax is None:
            color_norm.vmax = upper
    else:
        color_norm = norm
    _density_bounds((color_norm.vmin, color_norm.vmax), "color limits")
    try:
        endpoints = np.ma.asarray(color_norm([color_norm.vmin, color_norm.vmax]))
        if np.ma.getmaskarray(endpoints).any() or not np.isfinite(endpoints.data).all():
            raise ValueError("The supplied norm cannot map its color limits.")
        mapped = np.ma.asarray(color_norm(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid density color normalization: {exc}") from exc
    return color_norm, ~np.ma.getmaskarray(mapped) & np.isfinite(mapped.data)


def _density_contour_levels(values, count, mode):
    """Use the final plotted values, including their upper endpoint, once."""
    if not len(values):
        return np.empty(0, dtype=float)
    low, high = float(values.min()), float(values.max())
    low = low / 2.0 if low > 0 else 0.0
    if high <= low:
        high = 1.0
    if mode == "quantile":
        levels = np.quantile(values, np.linspace(0.0, 1.0, count))
        levels[0], levels[-1] = low, high
        levels = np.unique(levels)
    else:
        levels = np.linspace(low, high, count)
    return levels


def embedding_density(
    adata,
    groupby: str,
    *,
    basis: str = "umap",
    groups: Iterable[str] | None = None,
    gridsize: int = 160,
    normalize: str = "per_group",
    cmap: str = "magma",
    contour: bool = False,
    contour_levels: int = 12,
    imshow_alpha: float = 0.85,
    transparent_background: bool = True,
    mask_zeros: bool = True,
    background: str | None = None,
    figsize=None,
    show: bool = True,
    smooth_k: int = 5,
    log1p: bool = True,
    contour_lines: bool = True,
    contour_linewidth: float = 0.7,
    contour_alpha: float = 0.95,
    contour_level_mode: str = "quantile",
    ax=None,
    axes=None,
    cax=None,
    show_colorbar: bool = True,
    grid_extent=None,
    xlim=None,
    ylim=None,
    aspect: str = "equal",
    norm=None,
    vmin=None,
    vmax=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """Plot comparable per-group histograms on the first two embedding axes.

    All groups use the same grid. ``normalize='per_group'`` divides each
    group's transformed grid by its own peak, giving a shared relative scale
    from zero to one. This compares spatial shape, not group abundance.
    ``normalize='global'`` retains absolute bin values and uses one color scale
    across panels. Counts describe the supplied cells and depend on sampling,
    bin width, and embedding geometry; they are not a probability density or
    an estimate of population abundance.

    Heatmaps show raw cell counts before normalization. Only ``contour=True``
    applies one optional ``smooth_k`` box blur followed by one optional
    ``log1p`` transform. ``smooth_k=0`` or ``1`` disables smoothing; other values
    must be positive odd integers. Replicated-edge smoothing is descriptive
    and does not guarantee count conservation; exported ``raw_count`` does.
    Contour levels are shared across panels and computed from the actual
    plotted values, using quantile or linear spacing over their full range.
    When zeros are masked, the lowest level is half the smallest positive
    plotted value; this preserves sparse peaks and lower-count groups on a
    common scale. Repeated levels are removed. Zeros are omitted when ``mask_zeros=True``;
    contour boundaries interpolate toward zero-valued neighbors so sparse
    peaks remain visible.

    Parameters
    ----------
    groups
        Ordered known group labels. By default, categorical declaration order
        is retained, including unused levels; other labels use observed order.
        Known groups with no finite cells remain explicit empty panels.
        Missing labels and nonfinite coordinate pairs are excluded jointly and
        retained in the source table. Unknown requested groups raise an error.
    gridsize
        Number of bins per coordinate, an integer of at least two.
    grid_extent
        Optional ``(xmin, xmax, ymin, ymax)`` computational bounds. The default
        uses all finite labeled cells, including unselected groups, so selecting
        panels does not silently alter their bins. Constant coordinates receive
        a small explicit padding. Cells outside the grid remain in source data
        but do not contribute to histograms; the final bin includes its upper edge.
    xlim, ylim
        View limits only: these do not change histograms or color normalization.
        Supplied axes retain already fixed limits unless overridden explicitly.
    norm, vmin, vmax
        Optional shared Matplotlib color normalization. Supply ``norm`` or
        numeric limits. Defaults are zero to one for relative panels and zero
        to the largest transformed count for global panels. Normalization maps
        colors only; contour geometry uses the transformed count values.
        A shared norm directly aligns heatmap color scales. For matching
        automatic filled-contour bands, render the comparison groups in one
        call: their shared levels depend on all selected plotted values.
    ax, axes, cax
        Compose into an existing figure. ``ax`` is for one group; ``axes`` is a
        flat sequence or mapping keyed by group (or ``'density:<group>'``).
        Use ``cax`` for a reserved shared colorbar, or disable it with
        ``show_colorbar=False``. Supplied figure size and sibling panels are
        preserved. ``aspect='equal'`` keeps embedding units equally scaled.
    return_data, return_result
        The default remains ``(fig, flat_axes_array)``. ``return_data=True``
        returns ``(fig, flat_axes_array, data)``. ``return_result=True`` returns
        a :class:`PlotResult` with named ``density:<group>`` axes, optional
        ``colorbar``, and four source tables: full ``cells`` with inclusion
        flags, every ``grid`` bin with raw and transformed values, ``groups``
        with counts and normalization factors, and exact ``contour_levels``.
        The grid's ``displayed`` flag records unmasked values at the contour
        threshold, independently of view cropping; contours interpolate those
        grid values. A SubFigure result exports its full parent canvas.
    """
    if groupby not in adata.obs:
        raise KeyError(f"groupby {groupby!r} not found in adata.obs")
    if normalize not in {"per_group", "global"}:
        raise ValueError("normalize must be 'per_group' or 'global'.")
    if (
        isinstance(gridsize, bool)
        or not isinstance(gridsize, (int, np.integer))
        or gridsize < 2
    ):
        raise ValueError("gridsize must be an integer of at least two.")
    gridsize = int(gridsize)
    if contour_level_mode not in {"quantile", "linear"}:
        raise ValueError("contour_level_mode must be 'quantile' or 'linear'.")
    if (
        isinstance(contour_levels, bool)
        or not isinstance(contour_levels, (int, np.integer))
        or contour_levels < 2
    ):
        raise ValueError("contour_levels must be an integer of at least two.")
    contour_levels = int(contour_levels)
    if contour and (
        isinstance(smooth_k, bool)
        or not isinstance(smooth_k, (int, np.integer))
        or smooth_k < 0
        or (smooth_k > 1 and smooth_k % 2 == 0)
    ):
        raise ValueError("smooth_k must be zero, one, or a positive odd integer.")
    effective_smooth = int(smooth_k) if contour and smooth_k > 1 else 1
    if aspect not in {"equal", "auto"}:
        raise ValueError("aspect must be 'equal' or 'auto'.")
    for name, value in (
        ("imshow_alpha", imshow_alpha),
        ("contour_alpha", contour_alpha),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between zero and one.")
    if (
        not np.isscalar(contour_linewidth)
        or not np.isfinite(contour_linewidth)
        or contour_linewidth < 0
    ):
        raise ValueError("contour_linewidth must be a finite nonnegative number.")
    if figsize is not None:
        size = np.asarray(figsize, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("figsize must contain two positive finite values.")
    view_x = None if xlim is None else _density_bounds(xlim, "xlim")
    view_y = None if ylim is None else _density_bounds(ylim, "ylim")
    if background is not None:
        to_rgba(background)
    colormap = plt.get_cmap(cmap).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    embedding_key, coordinates = resolve_embedding(adata, basis)
    try:
        xy = np.array(coordinates[:, :2], dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding coordinates must be numeric.") from exc
    labels, original_labels, missing, order = _density_labels(
        adata.obs[groupby], groups
    )
    supplied_axes = _density_axes(order, ax, axes, cax)
    finite = np.isfinite(xy).all(axis=1)
    selected = np.isin(labels, order) & ~missing
    padded_dimensions = []
    if grid_extent is None:
        extent_rows = finite & ~missing
        if not extent_rows.any():
            raise ValueError(
                "No finite labeled embedding coordinates define the grid; supply grid_extent explicitly."
            )
        extent = np.array(
            [
                xy[extent_rows, 0].min(),
                xy[extent_rows, 0].max(),
                xy[extent_rows, 1].min(),
                xy[extent_rows, 1].max(),
            ]
        )
        for dimension, start in (("x", 0), ("y", 2)):
            if extent[start] == extent[start + 1]:
                padding = max(0.5, abs(float(extent[start])) * 1e-6)
                extent[start] -= padding
                extent[start + 1] += padding
                padded_dimensions.append(dimension)
        extent = _density_bounds(extent, "inferred grid_extent", 4)
    else:
        extent = _density_bounds(grid_extent, "grid_extent", 4)
    in_grid = (
        finite
        & (xy[:, 0] >= extent[0])
        & (xy[:, 0] <= extent[1])
        & (xy[:, 1] >= extent[2])
        & (xy[:, 1] <= extent[3])
    )
    included = selected & in_grid
    cells = pd.DataFrame(
        {
            "obs_position": np.arange(adata.n_obs),
            "cell": adata.obs_names.astype(str),
            "group": labels,
            "group_value": original_labels,
            "group_missing": missing,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "selected": selected,
            "finite_coordinates": finite,
            "in_grid": in_grid,
            "included": included,
            "reason": np.select(
                [missing, ~selected, ~finite, ~in_grid],
                [
                    "missing_group",
                    "group_not_selected",
                    "nonfinite_coordinates",
                    "outside_grid",
                ],
                default="included",
            ),
        }
    )
    x_edges = np.linspace(extent[0], extent[1], gridsize + 1)
    y_edges = np.linspace(extent[2], extent[3], gridsize + 1)
    if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0):
        raise ValueError(
            "gridsize is too fine for the coordinate precision of grid_extent."
        )
    ix, iy = np.meshgrid(np.arange(gridsize), np.arange(gridsize))
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2
    y_centers = y_edges[:-1] + np.diff(y_edges) / 2
    frames, summaries = [], []
    for group in order:
        rows = labels == group
        used = rows & included
        raw = np.histogram2d(xy[used, 0], xy[used, 1], bins=(x_edges, y_edges))[0].T
        smoothed = _smooth2d(raw, effective_smooth) if contour else raw.copy()
        # Numerical cancellation in a separable box filter can leave tiny
        # negative values in empty bins. A count intensity is nonnegative.
        smoothed = np.maximum(smoothed, 0.0)
        transformed = np.log1p(smoothed) if contour and log1p else smoothed.copy()
        peak = float(transformed.max())
        factor = peak if normalize == "per_group" and peak > 0 else 1.0
        plotted = transformed / factor
        frames.append(
            pd.DataFrame(
                {
                    "group": group,
                    "ix": ix.ravel(),
                    "iy": iy.ravel(),
                    "x": x_centers[ix.ravel()],
                    "y": y_centers[iy.ravel()],
                    "x_left": x_edges[ix.ravel()],
                    "x_right": x_edges[ix.ravel() + 1],
                    "y_bottom": y_edges[iy.ravel()],
                    "y_top": y_edges[iy.ravel() + 1],
                    "raw_count": raw.astype(np.int64).ravel(),
                    "smoothed_count": smoothed.ravel(),
                    "transformed_value": transformed.ravel(),
                    "plotted_value": plotted.ravel(),
                    "masked_zero": (plotted <= 0).ravel() & bool(mask_zeros),
                }
            )
        )
        n_input, n_finite, n_in_grid = (
            int(rows.sum()),
            int((rows & finite).sum()),
            int(used.sum()),
        )
        status = (
            "available"
            if n_in_grid
            else "no_observations"
            if not n_input
            else "no_finite_coordinates"
            if not n_finite
            else "outside_grid"
        )
        summaries.append(
            {
                "group": group,
                "n_input": n_input,
                "n_finite": n_finite,
                "n_in_grid": n_in_grid,
                "n_nonfinite": n_input - n_finite,
                "n_outside_grid": n_finite - n_in_grid,
                "raw_count_sum": int(raw.sum()),
                "transformed_peak": peak,
                "normalization_factor": factor,
                "status": status,
            }
        )
    grid = pd.concat(frames, ignore_index=True)
    group_summary = pd.DataFrame(summaries)
    maximum = float(grid.plotted_value.max())
    color_norm, finite_color = _density_norm(
        norm,
        vmin,
        vmax,
        1.0 if normalize == "per_group" or maximum <= 0 else maximum,
        grid.plotted_value.to_numpy(),
    )
    grid["masked_norm"] = ~finite_color
    active = grid.group.map(group_summary.set_index("group").n_in_grid).to_numpy() > 0
    eligible_bins = active & ~grid.masked_zero.to_numpy() & finite_color
    levels = (
        _density_contour_levels(
            grid.loc[eligible_bins, "plotted_value"].to_numpy(),
            contour_levels,
            contour_level_mode,
        )
        if contour
        else np.empty(0, dtype=float)
    )
    grid["displayed"] = eligible_bins
    if contour:
        grid["displayed"] &= grid.plotted_value.ge(levels[0]) if len(levels) else False
    group_summary["color_vmin"] = float(color_norm.vmin)
    group_summary["color_vmax"] = float(color_norm.vmax)
    group_summary["n_displayed_bins"] = [
        int(grid.loc[grid.group == group, "displayed"].sum()) for group in order
    ]
    level_table = pd.DataFrame(
        {
            "group": pd.Series(np.repeat(order, len(levels)), dtype=object),
            "level_index": pd.Series(
                np.tile(np.arange(len(levels)), len(order)), dtype=np.int64
            ),
            "value": pd.Series(np.tile(levels, len(order)), dtype=float),
        }
    )
    messages = []
    for mask, explanation in (
        (missing, "missing group labels"),
        (selected & ~finite, "nonfinite coordinate pairs"),
        (selected & finite & ~in_grid, "coordinates outside grid_extent"),
    ):
        count = int(mask.sum())
        if count:
            messages.append(
                f"Excluded {count} cells with {explanation}; their original rows remain in cells."
            )
    if normalize == "per_group":
        color_label = "Relative density (group peak = 1)"
    else:
        color_label = (
            "Smoothed cell count per bin"
            if effective_smooth > 1
            else "Cell count per bin"
        )
        if contour and log1p:
            color_label = f"log1p({color_label.lower()})"

    # All data, style, grouping, normalizer and axes checks above precede canvas
    # creation or any modification to the caller's axes.
    owns_axes = supplied_axes is None
    if owns_axes:
        columns = min(4, len(order))
        rows = int(np.ceil(len(order) / columns))
        fig, all_axes = plt.subplots(
            rows,
            columns,
            squeeze=False,
            figsize=figsize if figsize is not None else (3.3 * columns, 3.0 * rows),
        )
        all_axes = all_axes.ravel()
        panel_axes = all_axes[: len(order)]
        for unused in all_axes[len(order) :]:
            unused.axis("off")
        if background is not None:
            fig.patch.set_facecolor(background)
            fig.patch.set_alpha(1.0)
        elif transparent_background:
            fig.patch.set_alpha(0.0)
    else:
        panel_axes = supplied_axes
        fig = panel_axes[0].figure
    fixed_limits = [
        (
            item.get_xlim() if not item.get_autoscalex_on() else None,
            item.get_ylim() if not item.get_autoscaley_on() else None,
        )
        for item in panel_axes
    ]
    artists = {"density": {}, "contour_lines": {}, "empty_labels": {}}
    for position, (group, panel) in enumerate(zip(order, panel_axes)):
        subset = grid.loc[grid.group == group]
        summary = group_summary.iloc[position]
        if background is not None:
            panel.set_facecolor(background)
            panel.patch.set_alpha(1.0)
        elif transparent_background:
            panel.patch.set_alpha(0.0)
        panel.set_title(f"{group} (n={summary.n_in_grid:,})")
        if summary.n_in_grid and subset.displayed.any():
            values = subset.plotted_value.to_numpy().reshape(gridsize, gridsize)
            if contour:
                # Let contour interpolation approach a zero-valued neighboring
                # node; masking zeros themselves erases isolated positive bins.
                # Even a color norm undefined at zero (such as LogNorm) must
                # not remove zero-valued neighbors from geometric interpolation.
                geometry_mask = subset.masked_norm.to_numpy().reshape(
                    gridsize, gridsize
                ) & (values != 0)
                field = np.ma.masked_where(geometry_mask, values)
                mappable = panel.contourf(
                    x_centers,
                    y_centers,
                    field,
                    levels=levels,
                    cmap=colormap,
                    alpha=float(contour_alpha),
                )
                # Passing LogNorm into contourf itself masks nonpositive nodes,
                # which can erase isolated positive peaks. Set the color scale
                # after geometry has been computed in the actual field units.
                mappable.set_norm(color_norm)
                if contour_lines:
                    finite_values = values[~np.ma.getmaskarray(field)]
                    line_levels = levels[
                        (levels > finite_values.min()) & (levels < finite_values.max())
                    ]
                    if len(line_levels):
                        artists["contour_lines"][group] = panel.contour(
                            x_centers,
                            y_centers,
                            field,
                            levels=line_levels,
                            colors="k",
                            linewidths=float(contour_linewidth),
                            alpha=0.35,
                        )
            else:
                mask = (~subset.displayed.to_numpy()).reshape(gridsize, gridsize)
                mappable = panel.imshow(
                    np.ma.masked_where(mask, values),
                    origin="lower",
                    extent=extent,
                    cmap=colormap,
                    norm=color_norm,
                    alpha=float(imshow_alpha),
                    interpolation="nearest",
                )
            artists["density"][group] = mappable
        else:
            reason = {
                "no_observations": "No observed cells",
                "no_finite_coordinates": "No finite coordinates",
                "outside_grid": "No cells inside grid",
            }.get(summary.status, "No values at displayed levels")
            artists["empty_labels"][group] = panel.text(
                0.5,
                0.5,
                reason,
                transform=panel.transAxes,
                ha="center",
                va="center",
                color="#66717E",
            )
        old_x, old_y = fixed_limits[position]
        panel.set_xlim(
            view_x if view_x is not None else old_x if old_x is not None else extent[:2]
        )
        panel.set_ylim(
            view_y if view_y is not None else old_y if old_y is not None else extent[2:]
        )
        panel.set_aspect(aspect, adjustable="box")
        panel.set_xticks([])
        panel.set_yticks([])
    if owns_axes:
        # Reserve a title band inside the fixed canvas before adding a shared
        # colorbar. Notebook tight bounding boxes previously hid this clipping.
        height_points = 72.0 * fig.get_figheight()
        heading = fig.suptitle(
            f"Embedding density by {groupby} ({basis})",
            y=1.0 - 6.0 / height_points,
        )
        title_band = (
            heading.get_fontsize()
            + max(panel.title.get_fontsize() for panel in panel_axes)
            + float(plt.rcParams["axes.titlepad"])
            + 10.0
        )
        fig.subplots_adjust(
            top=max(
                fig.subplotpars.bottom + 0.05,
                min(fig.subplotpars.top, 1.0 - title_band / height_points),
            )
        )
    named_axes = {f"density:{group}": panel for group, panel in zip(order, panel_axes)}
    if show_colorbar and artists["density"]:
        mappable = ScalarMappable(norm=color_norm, cmap=colormap)
        if cax is None:
            colorbar = fig.colorbar(
                mappable,
                ax=panel_axes.tolist(),
                fraction=0.025,
                pad=0.02,
                label=color_label,
            )
        else:
            bbox = cax.get_window_extent()
            orientation = "horizontal" if bbox.width > bbox.height else "vertical"
            colorbar = fig.colorbar(
                mappable, cax=cax, orientation=orientation, label=color_label
            )
        named_axes["colorbar"] = colorbar.ax
        artists["colorbar"] = colorbar
    metadata = {
        "plot": "embedding_density",
        "basis": embedding_key,
        "dimensions": [0, 1],
        "groupby": groupby,
        "groups": order,
        "normalize": normalize,
        "normalization_rule": "divide_by_group_peak"
        if normalize == "per_group"
        else "absolute_transformed_count",
        "color_label": color_label,
        "color_limits": [float(color_norm.vmin), float(color_norm.vmax)],
        "norm": type(color_norm).__name__,
        "cmap": colormap.name,
        "contour": bool(contour),
        "smooth_k": effective_smooth,
        "smoothing": "one_box_blur_with_replicated_edges"
        if effective_smooth > 1
        else "none",
        "log1p": bool(contour and log1p),
        "mask_zeros": bool(mask_zeros),
        "contour_level_mode": contour_level_mode if contour else None,
        "contour_levels": levels.tolist(),
        "contour_lines": bool(contour and contour_lines),
        "gridsize": gridsize,
        "grid_extent": extent.tolist(),
        "grid_extent_source": "explicit"
        if grid_extent is not None
        else "all_finite_labeled_cells",
        "padded_constant_dimensions": padded_dimensions,
        "histogram_source_scope": "selected_finite_cells_inside_grid",
        "finite_coordinates_policy": "both_coordinates_finite",
        "grid_upper_edge_included": True,
        "n_input": int(adata.n_obs),
        "n_selected": int(selected.sum()),
        "n_included": int(included.sum()),
        "n_missing_groups": int(missing.sum()),
        "n_nonfinite_selected": int((selected & ~finite).sum()),
        "n_outside_grid_selected": int((selected & finite & ~in_grid).sum()),
        "obs_index_name": adata.obs_names.name,
        "aspect": aspect,
        "view_limits": {
            group: {"xlim": list(panel.get_xlim()), "ylim": list(panel.get_ylim())}
            for group, panel in zip(order, panel_axes)
        },
        "view_limits_change_computation": False,
        "colorbar_shown": "colorbar" in named_axes,
        "transparent_background": bool(transparent_background),
        "background": background,
        "alpha": float(contour_alpha if contour else imshow_alpha),
        "interpretation": "Descriptive cell counts in embedding bins; relative panels do not compare abundance.",
    }
    data = {
        "cells": cells,
        "grid": grid,
        "groups": group_summary,
        "contour_levels": level_table,
    }
    if show:
        plt.show()
    if return_result:
        return PlotResult(fig, named_axes, data, metadata, messages, artists)
    if return_data:
        return fig, panel_axes, data
    return fig, panel_axes
