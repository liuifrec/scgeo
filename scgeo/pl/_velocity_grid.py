"""Transparent local summaries of supplied, projected velocity vectors.

These helpers do not infer kinetics or cell trajectories. Preparation has no
plotting side effects; rendering adds only field artists and returns their data.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import cKDTree

from ._utils import resolve_embedding


def _labels(series, name):
    raw = pd.unique(series.dropna().astype(object))
    if len({str(value) for value in raw}) != len(raw):
        raise ValueError(
            f"{name} contains distinct values with identical string labels."
        )
    return pd.Series(
        [None if pd.isna(value) else str(value) for value in series], dtype=object
    )


def _positive_integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _positive_number(value, name):
    if isinstance(value, bool) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _grid_shape(grid_size):
    if isinstance(grid_size, (int, np.integer)) and not isinstance(grid_size, bool):
        grid_size = (grid_size, grid_size)
    if not isinstance(grid_size, (tuple, list, np.ndarray)) or len(grid_size) != 2:
        raise ValueError("grid_size must be an integer or a pair (nx, ny).")
    return tuple(
        _positive_integer(value, "grid_size", minimum=2) for value in grid_size
    )


def _extent(grid_extent):
    if grid_extent is None:
        return None
    bounds = np.asarray(grid_extent, dtype=float)
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError(
            "grid_extent must be (xmin, xmax, ymin, ymax) with finite increasing bounds."
        )
    return tuple(float(value) for value in bounds)


def _query(tree, points, k, radius=np.inf):
    distances, indices = tree.query(points, k=k, distance_upper_bound=radius)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    return distances, indices


def _infer_radius(tree, points, n_neighbors):
    if len(points) < 2:
        raise ValueError(
            "radius is required when fewer than two distinct eligible display positions are available."
        )
    k = min(n_neighbors, len(points) - 1)
    chunk_size = max(1, min(2048, 1_000_000 // (k + 1)))
    distances = np.empty(len(points))
    for start in range(0, len(points), chunk_size):
        values, _ = _query(tree, points[start : start + chunk_size], k + 1)
        distances[start : start + len(values)] = values[:, -1]
    distance = float(np.median(distances))
    fallback = False
    if distance <= 0:
        positive = distances[distances > 0]
        if not len(positive):
            raise ValueError(
                "Cannot infer a positive radius from coincident eligible positions; supply radius explicitly."
            )
        distance, fallback = float(np.median(positive)), True
    return 1.5 * distance, k, fallback


def _prepare_velocity_field(
    adata,
    *,
    basis="umap",
    velocity_basis=None,
    condition_key=None,
    groups=None,
    grid_size=25,
    grid_extent=None,
    n_neighbors=30,
    radius=None,
    min_neighbors=5,
    min_coherence=0.2,
):
    """Return full source cells and a masked local-mean grid without any axes.

    At each grid point, up to ``n_neighbors`` nearest source cells inside
    ``radius`` receive Gaussian weights with sigma=radius/2. The neighbor count
    and direction coherence are descriptive support filters, not uncertainty.
    """
    nx, ny = _grid_shape(grid_size)
    requested_extent = _extent(grid_extent)
    n_neighbors = _positive_integer(n_neighbors, "n_neighbors")
    min_neighbors = _positive_integer(min_neighbors, "min_neighbors")
    if min_neighbors > n_neighbors:
        raise ValueError("min_neighbors cannot exceed n_neighbors.")
    if (
        not np.isscalar(min_coherence)
        or not np.isfinite(min_coherence)
        or not 0 <= min_coherence <= 1
    ):
        raise ValueError("min_coherence must be between zero and one.")
    if radius is not None:
        radius = _positive_number(radius, "radius")
    embedding_key, coordinates = resolve_embedding(adata, basis)
    coordinates = np.asarray(coordinates, dtype=float)
    expected = f"velocity_{embedding_key.removeprefix('X_')}"
    velocity_key = expected if velocity_basis is None else str(velocity_basis)
    if not velocity_key.startswith("velocity_"):
        velocity_key = f"velocity_{velocity_key}"
    if velocity_key != expected:
        raise ValueError(
            f"Projected velocity must match the display basis: expected {expected!r}, got {velocity_key!r}."
        )
    if velocity_key not in adata.obsm:
        raise KeyError(
            f"Matching projected velocity {velocity_key!r} is missing from adata.obsm."
        )
    velocities = np.asarray(adata.obsm[velocity_key], dtype=float)
    if velocities.shape != coordinates.shape:
        raise ValueError(
            "Projected velocity shape must exactly match the display embedding shape."
        )
    if condition_key is not None and condition_key not in adata.obs:
        raise KeyError(f"{condition_key!r} is missing from adata.obs.")
    if groups is not None and condition_key is None:
        raise ValueError("groups requires condition_key.")
    conditions = (
        _labels(adata.obs[condition_key], condition_key)
        if condition_key is not None
        else pd.Series([None] * adata.n_obs, dtype=object)
    )
    selected_groups = None
    selected = np.ones(adata.n_obs, dtype=bool)
    if groups is not None:
        raw_groups = [groups] if isinstance(groups, str) else list(groups)
        if not raw_groups or any(pd.isna(value) for value in raw_groups):
            raise ValueError("groups must contain nonmissing condition labels.")
        selected_groups = [str(value) for value in raw_groups]
        if len(set(selected_groups)) != len(selected_groups):
            raise ValueError("groups must contain unique condition labels.")
        unknown = set(selected_groups) - set(conditions.dropna())
        if unknown:
            raise ValueError(
                f"Requested groups are absent from condition_key: {sorted(unknown)}."
            )
        selected = conditions.isin(selected_groups).to_numpy()
    cells = pd.DataFrame(
        {
            "obs_position": np.arange(adata.n_obs),
            "cell": adata.obs_names.astype(str),
            "condition": conditions.to_numpy(),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "vx": velocities[:, 0],
            "vy": velocities[:, 1],
            "selected": selected,
            "finite_coordinates": np.isfinite(coordinates[:, :2]).all(axis=1),
            "finite_velocity": np.isfinite(velocities[:, :2]).all(axis=1),
            "displayed": False,
        }
    )
    cells["eligible"] = (
        cells.selected & cells.finite_coordinates & cells.finite_velocity
    )
    points = cells.loc[cells.eligible, ["x", "y"]].to_numpy(dtype=float)
    vectors = cells.loc[cells.eligible, ["vx", "vy"]].to_numpy(dtype=float)
    if not len(points):
        raise ValueError(
            "No eligible selected cells have jointly finite display coordinates and projected velocity."
        )
    notes = []
    excluded = int((cells.selected & ~cells.eligible).sum())
    if excluded:
        notes.append(
            f"{excluded} selected cells with nonfinite coordinates or velocity were excluded jointly from field estimation."
        )
    observed_groups = conditions[selected].dropna().unique().tolist()
    if condition_key is not None and len(observed_groups) > 1:
        notes.append(
            "The local field pools the selected conditions; opposing directions may cancel. Compare separate conditions when their fields differ."
        )
    if condition_key is None:
        notes.append(
            "No condition key was supplied; the field pools all eligible cells and does not distinguish condition or state boundaries."
        )
    tree = cKDTree(points)
    radius_source, radius_k = "provided", None
    if radius is None:
        radius, radius_k, fallback = _infer_radius(tree, points, n_neighbors)
        radius_source = "1.5 * median source kth-neighbor distance"
        if fallback:
            radius_source = "1.5 * median positive source kth-neighbor distance"
            notes.append(
                "Repeated positions made the median neighbor distance zero; the automatic radius uses the median positive kth-neighbor distance."
            )
    bandwidth = radius / 2.0
    if requested_extent is None:
        minima, maxima = points.min(axis=0), points.max(axis=0)
        spans = maxima - minima
        padding = np.where(spans > 0, spans * 0.025, radius)
        bounds = (
            minima[0] - padding[0],
            maxima[0] + padding[0],
            minima[1] - padding[1],
            maxima[1] + padding[1],
        )
    else:
        bounds = requested_extent
    gx, gy = (
        np.linspace(bounds[0], bounds[1], nx),
        np.linspace(bounds[2], bounds[3], ny),
    )
    xx, yy = np.meshgrid(gx, gy)
    queries = np.column_stack([xx.ravel(), yy.ravel()])
    k = min(n_neighbors, len(points))
    chunk_size = max(1, min(2048, 1_000_000 // k))
    means = np.full((len(queries), 2), np.nan)
    mean_speeds, weight_sum = np.full(len(queries), np.nan), np.zeros(len(queries))
    counts = np.zeros(len(queries), dtype=int)
    cell_speeds = np.hypot(vectors[:, 0], vectors[:, 1])
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        distances, indices = _query(tree, query, k, radius=np.nextafter(radius, np.inf))
        valid = np.isfinite(distances) & (indices < len(points)) & (distances <= radius)
        indices = np.minimum(indices, len(points) - 1)
        weights = np.exp(-0.5 * np.square(distances / bandwidth))
        weights[~valid] = 0.0
        denominator = weights.sum(axis=1)
        rows = slice(start, start + len(query))
        counts[rows], weight_sum[rows] = valid.sum(axis=1), denominator
        averages = np.sum(weights[:, :, None] * vectors[indices], axis=1)
        average_speeds = np.sum(weights * cell_speeds[indices], axis=1)
        np.divide(
            averages,
            denominator[:, None],
            out=means[rows],
            where=denominator[:, None] > 0,
        )
        np.divide(
            average_speeds, denominator, out=mean_speeds[rows], where=denominator > 0
        )
    speed = np.hypot(means[:, 0], means[:, 1])
    coherence = np.full(len(queries), np.nan)
    np.divide(speed, mean_speeds, out=coherence, where=mean_speeds > 0)
    coherence = np.clip(coherence, 0, 1)
    count_supported = counts >= min_neighbors
    nonzero = np.isfinite(speed) & (
        speed > np.finfo(float).eps * 16 * np.maximum(mean_speeds, np.finfo(float).tiny)
    )
    coherence_supported = np.isfinite(coherence) & (coherence >= min_coherence)
    supported = count_supported & nonzero & coherence_supported
    reasons = []
    for enough, moving, directional, value in zip(
        count_supported, nonzero, coherence_supported, coherence
    ):
        failed = []
        if not enough:
            failed.append("insufficient_neighbors")
        if not moving:
            failed.append(
                "zero_vector" if np.isfinite(value) or enough else "no_vector"
            )
        if not directional:
            failed.append(
                "low_coherence" if np.isfinite(value) else "undefined_coherence"
            )
        reasons.append("supported" if not failed else ";".join(failed))
    grid = pd.DataFrame(
        {
            "grid_id": np.arange(len(queries)),
            "ix": np.tile(np.arange(nx), ny),
            "iy": np.repeat(np.arange(ny), nx),
            "x": queries[:, 0],
            "y": queries[:, 1],
            "vx": means[:, 0],
            "vy": means[:, 1],
            "speed": speed,
            "mean_speed": mean_speeds,
            "coherence": coherence,
            "n_neighbors": counts,
            "radius": float(radius),
            "bandwidth": float(bandwidth),
            "weight_sum": weight_sum,
            "count_supported": count_supported,
            "coherence_supported": coherence_supported,
            "nonzero": nonzero,
            "supported": supported,
            "reason": reasons,
        }
    )
    if not supported.any():
        notes.append(
            "No grid positions pass the count, nonzero-vector, and coherence filters; no field arrows or streamlines are drawn."
        )
    metadata = {
        "display_basis": embedding_key,
        "velocity_key": velocity_key,
        "display_dimensions": [0, 1],
        "condition_key": condition_key,
        "groups": selected_groups,
        "observed_selected_groups": observed_groups,
        "field_source_scope": "selected eligible cells",
        "n_cells_input": adata.n_obs,
        "n_cells_selected": int(selected.sum()),
        "n_cells_eligible": len(points),
        "n_cells_excluded": excluded,
        "grid_size": [nx, ny],
        "grid_extent": [float(v) for v in bounds],
        "grid_extent_source": "provided"
        if requested_extent is not None
        else "full eligible source bounds",
        "n_neighbors": n_neighbors,
        "effective_neighbor_cap": k,
        "min_neighbors": min_neighbors,
        "radius": float(radius),
        "radius_source": radius_source,
        "radius_neighbor_k": radius_k,
        "bandwidth": float(bandwidth),
        "kernel": "exp(-0.5*(distance/(radius/2))**2), truncated at radius",
        "min_coherence": float(min_coherence),
        "coherence_definition": "norm(weighted mean velocity) / weighted mean velocity norm",
        "coherence_is_confidence": False,
        "grid_positions": len(grid),
        "grid_supported": int(supported.sum()),
        "coordinate_units": "display embedding",
        "velocity_units": "supplied projected velocity",
        "direction_interpretation": "descriptive local projected velocity; not inferred cell trajectories or elapsed time",
        "query_chunk_size": chunk_size,
    }
    return {
        "cells": cells,
        "grid": grid,
        "x": gx,
        "y": gy,
        "metadata": metadata,
        "warnings": notes,
    }


def _field_style(
    prepared,
    *,
    mode="grid",
    scale="auto",
    color="#008C95",
    color_by=None,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    arrow_linewidth=1.1,
    arrow_head_size=5.5,
    stream_density=1.0,
    stream_linewidth=1.2,
    alpha=0.9,
):
    """Validate renderer options and resolve numeric color/arrow scales."""
    if mode not in {"grid", "stream"}:
        raise ValueError("mode must be 'grid' or 'stream'.")
    if scale != "auto":
        scale = _positive_number(scale, "scale")
    if not is_color_like(color):
        raise ValueError(f"Invalid velocity color {color!r}.")
    if color_by not in {None, "speed", "coherence"}:
        raise ValueError("color_by must be None, 'speed', or 'coherence'.")
    arrow_linewidth = _positive_number(arrow_linewidth, "arrow_linewidth")
    arrow_head_size = _positive_number(arrow_head_size, "arrow_head_size")
    stream_linewidth = _positive_number(stream_linewidth, "stream_linewidth")
    density = np.asarray(stream_density, dtype=float)
    if (
        density.ndim > 1
        or density.size not in {1, 2}
        or not np.isfinite(density).all()
        or (density <= 0).any()
    ):
        raise ValueError(
            "stream_density must be positive or a pair of positive values."
        )
    if (np.floor(30 * density) < 2).any():
        raise ValueError(
            "stream_density must allocate at least two streamline seed bins per "
            "dimension: floor(30 * density) >= 2 (each density >= 2/30)."
        )
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one.")
    grid = prepared["grid"]
    resolved_scale = None
    if mode == "grid":
        resolved_scale = scale
        if scale == "auto":
            speeds = grid.loc[grid.supported, "speed"].to_numpy()
            spacing = min(np.diff(prepared["x"])[0], np.diff(prepared["y"])[0])
            resolved_scale = (
                0.65 * spacing / np.quantile(speeds, 0.9) if len(speeds) else 1.0
            )
        resolved_scale = float(resolved_scale)
    color_map, mappable = None, None
    if color_by is None:
        if norm is not None or vmin is not None or vmax is not None:
            raise ValueError("norm/vmin/vmax require numeric color_by.")
    else:
        values = grid.loc[grid.supported, color_by].to_numpy()
        upper = (
            1.0
            if color_by == "coherence" or not len(values)
            else max(float(np.max(values)), np.finfo(float).tiny)
        )
        if norm is not None and (vmin is not None or vmax is not None):
            raise ValueError("Supply norm or vmin/vmax, not both.")
        if norm is None:
            norm = Normalize(
                0.0 if vmin is None else vmin, upper if vmax is None else vmax
            )
        elif not isinstance(norm, Normalize):
            raise TypeError("norm must be a Matplotlib Normalize instance.")
        else:
            if norm.vmin is None:
                norm.vmin = 0.0
            if norm.vmax is None:
                norm.vmax = upper
        if not np.isfinite([norm.vmin, norm.vmax]).all() or norm.vmin >= norm.vmax:
            raise ValueError("Numeric color limits must be finite with vmin < vmax.")
        color_map = plt.get_cmap(cmap)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    return {
        "mode": mode,
        "scale_requested": scale,
        "scale": resolved_scale,
        "color": color,
        "color_by": color_by,
        "norm": norm,
        "cmap": color_map,
        "mappable": mappable,
        "arrow_linewidth": arrow_linewidth,
        "arrow_head_size": arrow_head_size,
        "stream_density": float(density.ravel()[0])
        if density.size == 1
        else tuple(density),
        "stream_linewidth": stream_linewidth,
        "alpha": float(alpha),
    }


def _inside_supported(points, x, y, supported):
    """Require all bilinear-interpolation corners to be supported."""
    gx = (points[:, 0] - x[0]) / (x[1] - x[0])
    gy = (points[:, 1] - y[0]) / (y[1] - y[0])
    inside = (
        (gx >= -1e-9)
        & (gy >= -1e-9)
        & (gx <= len(x) - 1 + 1e-9)
        & (gy <= len(y) - 1 + 1e-9)
    )
    ix = np.clip(np.floor(gx).astype(int), 0, len(x) - 2)
    iy = np.clip(np.floor(gy).astype(int), 0, len(y) - 2)
    return (
        inside
        & supported[iy, ix]
        & supported[iy, ix + 1]
        & supported[iy + 1, ix]
        & supported[iy + 1, ix + 1]
    )


def _safe_paths(raw_paths, x, y, supported):
    paths, current = [], []
    dropped = 0
    step = min(x[1] - x[0], y[1] - y[0]) / 4
    for raw in raw_paths:
        raw = np.asarray(raw, dtype=float)
        if len(raw) < 2:
            continue
        if current and not np.allclose(current[-1], raw[0], rtol=0, atol=step * 1e-7):
            if len(current) >= 2:
                paths.append(np.asarray(current))
            current = []
        for start, end in zip(raw[:-1], raw[1:]):
            subdivisions = max(2, int(np.ceil(np.linalg.norm(end - start) / step)) + 1)
            samples = np.linspace(start, end, subdivisions)
            safe = (
                np.isfinite(samples).all()
                and _inside_supported(samples, x, y, supported).all()
            )
            if safe:
                if not current:
                    current.append(start)
                current.append(end)
            else:
                dropped += 1
                if len(current) >= 2:
                    paths.append(np.asarray(current))
                current = []
    if len(current) >= 2:
        paths.append(np.asarray(current))
    return paths, dropped


def _interpolate(values, point, x, y):
    gx, gy = (point[0] - x[0]) / (x[1] - x[0]), (point[1] - y[0]) / (y[1] - y[0])
    ix, iy = (
        int(np.clip(np.floor(gx), 0, len(x) - 2)),
        int(np.clip(np.floor(gy), 0, len(y) - 2)),
    )
    dx, dy = np.clip(gx - ix, 0, 1), np.clip(gy - iy, 0, 1)
    return float(
        (1 - dx) * (1 - dy) * values[iy, ix]
        + dx * (1 - dy) * values[iy, ix + 1]
        + (1 - dx) * dy * values[iy + 1, ix]
        + dx * dy * values[iy + 1, ix + 1]
    )


def _render_velocity_field(
    ax,
    prepared,
    *,
    mode="grid",
    scale="auto",
    color="#008C95",
    color_by=None,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    arrow_linewidth=1.1,
    arrow_head_size=5.5,
    stream_density=1.0,
    stream_linewidth=1.2,
    alpha=0.9,
):
    """Add only local-field artists; preserve axis limits and autoscale settings."""
    style = _field_style(
        prepared,
        mode=mode,
        scale=scale,
        color=color,
        color_by=color_by,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        arrow_linewidth=arrow_linewidth,
        arrow_head_size=arrow_head_size,
        stream_density=stream_density,
        stream_linewidth=stream_linewidth,
        alpha=alpha,
    )
    scale = style["scale_requested"]
    arrow_linewidth = style["arrow_linewidth"]
    arrow_head_size = style["arrow_head_size"]
    stream_linewidth = style["stream_linewidth"]
    alpha = style["alpha"]
    grid = prepared["grid"].copy(deep=True)
    x, y = prepared["x"], prepared["y"]
    shape = (len(y), len(x))
    supported = grid.supported.to_numpy().reshape(shape)
    artists = {"grid_arrows": [], "stream_lines": None, "stream_arrows": []}
    notes, paths_records, vector_records, extent_points = [], [], [], []
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    autoscale_x, autoscale_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()
    # streamplot calls autoscale_view internally; the embedding owner controls it.
    ax.set_autoscalex_on(False)
    ax.set_autoscaley_on(False)
    segments_checked, segments_dropped = 0, 0
    try:
        if mode == "grid":
            resolved_scale = style["scale"]
            for row in grid.itertuples(index=False):
                endpoint = (
                    row.x + row.vx * resolved_scale,
                    row.y + row.vy * resolved_scale,
                )
                record = {
                    "grid_id": row.grid_id,
                    "kind": "grid",
                    "x": row.x,
                    "y": row.y,
                    "source_x": row.x,
                    "source_y": row.y,
                    "target_x": endpoint[0],
                    "target_y": endpoint[1],
                    "vx": row.vx,
                    "vy": row.vy,
                    "scale": resolved_scale,
                    "displayed": bool(row.supported),
                    "reason": "displayed" if row.supported else row.reason,
                }
                vector_records.append(record)
                if not row.supported:
                    continue
                arrow_color = (
                    color
                    if color_by is None
                    else style["cmap"](style["norm"](getattr(row, color_by)))
                )
                patch = FancyArrowPatch(
                    (row.x, row.y),
                    endpoint,
                    arrowstyle="-|>",
                    mutation_scale=arrow_head_size,
                    linewidth=arrow_linewidth,
                    color=arrow_color,
                    alpha=alpha,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=2.5,
                )
                ax.add_patch(patch)
                artists["grid_arrows"].append(patch)
                extent_points.extend([(row.x, row.y), endpoint])
            legend_label = f"Local projected velocity ×{resolved_scale:.3g}"
        else:
            if scale != "auto":
                notes.append(
                    "scale is ignored in stream mode; streamline paths show geometric direction, not elapsed time."
                )
            if supported.any():
                before_patches = set(ax.patches)
                streams = ax.streamplot(
                    x,
                    y,
                    np.ma.array(grid.vx.to_numpy().reshape(shape), mask=~supported),
                    np.ma.array(grid.vy.to_numpy().reshape(shape), mask=~supported),
                    density=style["stream_density"],
                    color=color,
                    linewidth=stream_linewidth,
                    arrowsize=1.0,
                    zorder=2.5,
                    minlength=0.08,
                    maxlength=4.0,
                    integration_direction="both",
                    broken_streamlines=True,
                )
                # Rebuild direction arrows from verified paths so no arrow can
                # remain on a line segment rejected by the support audit.
                for patch in list(ax.patches):
                    if patch not in before_patches:
                        patch.remove()
                raw_segments = streams.lines.get_segments()
                segments_checked = sum(
                    max(0, len(segment) - 1) for segment in raw_segments
                )
                safe_paths, segments_dropped = _safe_paths(
                    raw_segments, x, y, supported
                )
                line_segments, color_values = [], []
                color_grid = (
                    grid[color_by].to_numpy().reshape(shape) if color_by else None
                )
                for path_id, path in enumerate(safe_paths):
                    pieces = (
                        [path]
                        if color_by is None
                        else [np.stack([a, b]) for a, b in zip(path[:-1], path[1:])]
                    )
                    for piece in pieces:
                        segment_id = len(line_segments)
                        line_segments.append(piece)
                        scalar_value = (
                            np.nan
                            if color_by is None
                            else _interpolate(color_grid, np.mean(piece, axis=0), x, y)
                        )
                        color_values.append(scalar_value)
                        paths_records.extend(
                            {
                                "path_id": path_id,
                                "segment_id": segment_id,
                                "vertex": vertex,
                                "x": float(point[0]),
                                "y": float(point[1]),
                                "color_value": scalar_value,
                            }
                            for vertex, point in enumerate(piece)
                        )
                    extent_points.extend(path.tolist())
                    distance = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
                    if distance.sum() <= 0:
                        continue
                    middle = min(
                        int(np.searchsorted(np.cumsum(distance), distance.sum() / 2)),
                        len(path) - 2,
                    )
                    start, end = path[middle], (path[middle] + path[middle + 1]) / 2
                    arrow_color = (
                        color
                        if color_by is None
                        else style["cmap"](
                            style["norm"](
                                _interpolate(color_grid, (start + end) / 2, x, y)
                            )
                        )
                    )
                    arrow = FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-|>",
                        mutation_scale=arrow_head_size + 1,
                        linewidth=stream_linewidth,
                        color=arrow_color,
                        alpha=alpha,
                        shrinkA=0,
                        shrinkB=0,
                        zorder=2.6,
                    )
                    ax.add_patch(arrow)
                    artists["stream_arrows"].append(arrow)
                    vector_records.append(
                        {
                            "grid_id": None,
                            "kind": "stream_direction",
                            "path_id": path_id,
                            "x": start[0],
                            "y": start[1],
                            "source_x": start[0],
                            "source_y": start[1],
                            "target_x": end[0],
                            "target_y": end[1],
                            "vx": end[0] - start[0],
                            "vy": end[1] - start[1],
                            "scale": np.nan,
                            "displayed": True,
                            "reason": "displayed",
                        }
                    )
                streams.lines.set_segments(line_segments)
                streams.lines.set_alpha(alpha)
                if color_by is not None:
                    # A fixed streamplot color otherwise overrides scalar
                    # mapping even after set_array; clear it explicitly.
                    streams.lines.set_color(None)
                    streams.lines.set_array(np.asarray(color_values))
                    streams.lines.set_cmap(style["cmap"])
                    streams.lines.set_norm(style["norm"])
                artists["stream_lines"] = streams.lines
                if segments_dropped:
                    notes.append(
                        f"{segments_dropped} stream segments touching unsupported interpolation cells were removed."
                    )
            legend_label = "Projected velocity streamlines (direction only)"
    finally:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_autoscalex_on(autoscale_x)
        ax.set_autoscaley_on(autoscale_y)
    vectors_table = pd.DataFrame(
        vector_records,
        columns=[
            "grid_id",
            "kind",
            "path_id",
            "x",
            "y",
            "source_x",
            "source_y",
            "target_x",
            "target_y",
            "vx",
            "vy",
            "scale",
            "displayed",
            "reason",
        ],
    )
    paths_table = pd.DataFrame(
        paths_records,
        columns=["path_id", "segment_id", "vertex", "x", "y", "color_value"],
    )
    for key in ("grid_id", "path_id"):
        vectors_table[key] = vectors_table[key].astype("Int64")
    vectors_table["displayed"] = vectors_table.displayed.astype(bool)
    for key in (
        "x",
        "y",
        "source_x",
        "source_y",
        "target_x",
        "target_y",
        "vx",
        "vy",
        "scale",
    ):
        vectors_table[key] = vectors_table[key].astype(float)
    for key in ("path_id", "segment_id", "vertex"):
        paths_table[key] = paths_table[key].astype(np.int64)
    for key in ("x", "y", "color_value"):
        paths_table[key] = paths_table[key].astype(float)
    grid["displayed"] = grid.supported if mode == "grid" else False
    legend_color = color if color_by is None else style["cmap"](0.65)
    rendered_field = (
        bool(artists["grid_arrows"]) if mode == "grid" else not paths_table.empty
    )
    if rendered_field:
        legend_handles = [
            Line2D(
                [],
                [],
                color=legend_color,
                lw=arrow_linewidth if mode == "grid" else stream_linewidth,
                marker=">",
                markersize=4.5,
                label=legend_label,
            )
        ]
    else:
        legend_handles = [
            Line2D(
                [],
                [],
                color="#85939D",
                marker="x",
                linestyle="none",
                markersize=5,
                label="No supported local field",
            )
        ]
        if grid.supported.any():
            notes.append(
                "Supported grid positions did not yield a supported continuous streamline path; no local field geometry was rendered."
            )
    metadata = copy.deepcopy(prepared["metadata"])
    metadata.update(
        {
            "mode": mode,
            "scale_requested": scale,
            "scale_resolved": style["scale"],
            "scale_applied": mode == "grid",
            "auto_scale_rule": "0.65 * minimum grid spacing / 90th percentile supported speed"
            if mode == "grid" and scale == "auto"
            else None,
            "color": color,
            "color_by": color_by,
            "alpha": float(alpha),
            "arrow_linewidth_points": float(arrow_linewidth),
            "arrow_head_size_points": float(arrow_head_size),
            "stream_density": style["stream_density"],
            "stream_linewidth_points": float(stream_linewidth),
            "stream_paths_are_time": False,
            "stream_mask_policy": "all bilinear interpolation corners supported; sampled segment audit",
            "stream_segments_checked": segments_checked,
            "stream_segments_removed": segments_dropped,
            "stream_paths_rendered": int(paths_table.path_id.nunique())
            if len(paths_table)
            else 0,
            "grid_arrows_rendered": len(artists["grid_arrows"]),
            "rendered_field_available": rendered_field,
            "color_norm": None
            if color_by is None
            else {
                "class": type(style["norm"]).__name__,
                "vmin": style["norm"].vmin,
                "vmax": style["norm"].vmax,
            },
            "colormap": None if color_by is None else style["cmap"].name,
        }
    )
    return {
        "artists": artists,
        "data": {"grid": grid, "vectors": vectors_table, "paths": paths_table},
        "metadata": metadata,
        "legend_handles": legend_handles,
        "mappable": style["mappable"],
        "warnings": notes,
        "extent_points": np.asarray(extent_points, dtype=float).reshape(-1, 2),
    }
