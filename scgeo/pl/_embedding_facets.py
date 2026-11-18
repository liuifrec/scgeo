"""Matched embedding and projected-velocity facets for standard AnnData workflows.

Facets are display-only slices through an existing embedding. They never fit an
embedding, estimate velocity, or reinterpret the plots as tracked trajectories.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, is_color_like
from matplotlib.lines import Line2D

from ._result import PlotResult
from ._style import category_palette, style_context
from ._utils import display_indices, resolve_embedding
from ._velocity_grid import (
    _field_style,
    _grid_shape,
    _positive_integer,
    _positive_number,
    _prepare_velocity_field,
    _render_velocity_field,
)

_FIELD_ALLOWED = {
    "grid_size", "n_neighbors", "radius", "min_neighbors", "min_coherence",
    "scale", "color", "color_by", "cmap", "arrow_linewidth",
    "arrow_head_size", "stream_density", "stream_linewidth", "alpha",
}


def _labels(series, name):
    """Keep true missing entries apart from literal labels such as 'NA'."""
    raw = pd.Series(series).reset_index(drop=True)
    present = pd.unique(raw.dropna().astype(object))
    names = [str(x) for x in present]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} has distinct values with colliding text labels.")
    missing = "(missing)"
    while missing in names:
        missing = f"({missing})"
    mapped = raw.map(lambda x: missing if pd.isna(x) else str(x))
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [str(x) for x in series.cat.categories if str(x) in names]
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"{name} has colliding categorical levels.")
    else:
        ordered = names
    if raw.isna().any():
        ordered.append(missing)
    return mapped, ordered, missing if raw.isna().any() else None


def _order(observed, requested, name):
    if requested is None:
        return list(observed)
    if isinstance(requested, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of labels, not a string.")
    levels = [str(x) for x in requested]
    if not levels or len(levels) != len(set(levels)):
        raise ValueError(f"{name} must contain unique labels.")
    omitted = set(observed) - set(levels)
    if omitted:
        raise ValueError(f"{name} omits observed labels: {sorted(omitted)}.")
    return levels


def _bounds(xy):
    """Compute an honest, nondegenerate bounding box in display coordinates."""
    if not len(xy):
        raise ValueError("No finite embedding coordinates available for panel limits.")
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    span = upper - lower
    padding = np.where(span > 0, 0.06 * span, 0.5)
    return ((float(lower[0] - padding[0]), float(upper[0] + padding[0])),
            (float(lower[1] - padding[1]), float(upper[1] + padding[1])))


def _resolve_colors(adata, color, palette):
    if color is None:
        if palette is not None:
            raise ValueError("palette requires a categorical color key.")
        return None, None, {}, None, None, None
    if color not in adata.obs:
        raise KeyError(f"color={color!r} not found in adata.obs.")
    series = adata.obs[color]
    if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
        if palette is not None:
            raise ValueError("palette is categorical; numeric color uses cmap and one shared norm.")
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Numeric color contains infinite values.")
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
            if low == high:
                low -= 0.5
                high += 0.5
            norm = Normalize(low, high)
        else:
            norm = None
        return "numeric", pd.Series(values), {}, norm, None, None
    mapped, categories, missing = _labels(series, f"color={color}")
    origin = "provided" if palette is not None else "deterministic_hash"
    if palette is None and isinstance(series.dtype, pd.CategoricalDtype):
        scanpy = adata.uns.get(f"{color}_colors")
        if scanpy is not None and len(scanpy) == len(series.cat.categories):
            if all(is_color_like(item) for item in scanpy):
                palette = dict(zip(map(str, series.cat.categories), scanpy))
                if missing is not None:
                    palette[missing] = "#B6BFC7"
                origin = "scanpy_uns"
    colors = category_palette(categories, palette)
    if missing is not None and origin == "deterministic_hash":
        colors[missing] = "#B6BFC7"
    return "categorical", mapped, colors, None, categories, origin


def _field_options(options):
    if options is None:
        return {"grid_size": 25, "n_neighbors": 30, "min_neighbors": 5,
                "min_coherence": 0.2, "scale": "auto", "color": "#157F8B",
                "color_by": None, "cmap": "viridis", "stream_density": 1.0}
    if not isinstance(options, Mapping):
        raise TypeError("velocity_options must be a mapping.")
    extra = set(options) - _FIELD_ALLOWED
    if extra:
        raise ValueError(f"Unsupported velocity_options keys: {sorted(extra)}.")
    return {**_field_options(None), **options}


def embedding_facets(
    adata,
    *,
    basis="umap",
    facet=None,
    facet_row=None,
    facet_col=None,
    facet_order=None,
    row_order=None,
    col_order=None,
    col_wrap=4,
    color=None,
    palette=None,
    cmap="viridis",
    velocity=None,
    velocity_basis=None,
    velocity_options=None,
    shared_limits=True,
    shared_velocity_scale=True,
    context="none",
    max_points=None,
    random_state=0,
    point_size=7.0,
    point_alpha=0.82,
    highlight=None,
    highlight_key=None,
    legend=True,
    show_colorbar=True,
    style="paper",
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Facet an existing embedding, optionally overlaying projected velocity.

    ``facet`` is an alias for ``facet_col``. ``facet_row`` and ``facet_col``
    can be combined for a two-way grid. Each observation belongs to precisely
    one panel, including a separate level for missing facet labels. Unobserved
    levels supplied in ``row_order``/``col_order`` produce explicit empty panels.
    
    All facets use one global categorical palette or numeric color scale.
    ``shared_limits=True`` fixes both axes identically; with velocity, the
    support grid and Gaussian radius are also shared. The optional
    ``shared_velocity_scale`` fixes the *grid-arrow multiplier* using supported
    local field speeds across all facets, not a separate auto scale per panel.
    The same velocity numeric colormap normalization is always shared.

    Velocity requires the *matching* ``velocity_<basis>`` array in ``adata.obsm``.
    Smoothing is calculated independently for each facet using the existing
    ScGeo local-field estimator. Unsupported positions remain blank. Streamlines
    are descriptive interpolations, not trajectories or elapsed time. No UMAP,
    kinetics, or quantitative ScGeo evidence is recomputed here.

    ``max_points`` limits *drawing per facet* only; local velocity uses all
    eligible source cells. ``context='all'`` optionally draws the full embedding
    in muted grey behind each facet. Metadata and PlotResult exports record all
    source cells, facet membership, supported velocity grid, and vector/stream
    geometry. A supplied optional Scanpy categorical palette is honored.

    With ``return_result=True`` returns a :class:`PlotResult`. By default
    returns ``(fig, axes_dict)``; ``return_data=True`` adds source tables.
    """
    if facet is not None:
        if facet_col is not None or col_order is not None:
            raise ValueError("facet cannot be combined with facet_col or col_order; use facet_order.")
        facet_col = facet
        col_order = facet_order
    elif facet_order is not None:
        raise ValueError("facet_order requires facet=.")
    if row_order is not None and facet_row is None:
        raise ValueError("row_order requires facet_row.")
    if col_order is not None and facet_col is None:
        raise ValueError("col_order requires facet_col or facet.")
    if facet_row is not None and facet_col == facet_row:
        raise ValueError("facet_row and facet_col must be different observation keys.")
    for name, key in (("facet_row", facet_row), ("facet_col", facet_col)):
        if key is not None and key not in adata.obs:
            raise KeyError(f"{name}={key!r} is not in adata.obs.")
    if velocity not in {None, "grid", "stream"}:
        raise ValueError("velocity must be None, 'grid', or 'stream'.")
    if velocity is None and (velocity_options is not None or velocity_basis is not None):
        raise ValueError("velocity_options and velocity_basis require velocity='grid' or 'stream'.")
    if context not in {"none", "all"}:
        raise ValueError("context must be 'none' or 'all'.")
    for name, setting in (("shared_limits", shared_limits), ("shared_velocity_scale", shared_velocity_scale),
                          ("legend", legend), ("show_colorbar", show_colorbar)):
        if not isinstance(setting, bool):
            raise TypeError(f"{name} must be a boolean.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer.")
    if random_state < 0:
        raise ValueError("random_state must be nonnegative.")
    col_wrap = _positive_integer(col_wrap, "col_wrap")
    point_size = _positive_number(point_size, "point_size")
    if not np.isscalar(point_alpha) or not np.isfinite(point_alpha) or not 0 <= point_alpha <= 1:
        raise ValueError("point_alpha must lie between zero and one.")
    if max_points is not None:
        _positive_integer(max_points, "max_points")
    if highlight is not None and highlight_key is None:
        raise ValueError("highlight requires highlight_key (an adata.obs column).")
    if highlight_key is not None and highlight_key not in adata.obs:
        raise KeyError(f"highlight_key={highlight_key!r} is not in adata.obs.")
    if highlight is not None:
        highlight = [highlight] if isinstance(highlight, str) else list(highlight)
        if not highlight or any(pd.isna(x) for x in highlight):
            raise ValueError("highlight must contain nonmissing labels.")
        highlight = [str(x) for x in highlight]
        if len(highlight) != len(set(highlight)):
            raise ValueError("highlight labels must be unique.")
    embedding_key, coordinates = resolve_embedding(adata, basis)
    xy = np.asarray(coordinates[:, :2], dtype=float)
    finite_xy = np.isfinite(xy).all(axis=1)
    if not finite_xy.any():
        raise ValueError("The embedding contains no finite display coordinates.")
    color_type, color_values, colors, color_norm, categories, palette_origin = _resolve_colors(
        adata, color, palette
    )
    row_values, row_levels, row_missing = (
        _labels(adata.obs[facet_row], facet_row) if facet_row is not None
        else (pd.Series(["All"] * adata.n_obs), ["All"], None)
    )
    col_values, col_levels, col_missing = (
        _labels(adata.obs[facet_col], facet_col) if facet_col is not None
        else (pd.Series(["All"] * adata.n_obs), ["All"], None)
    )
    rows = _order(row_levels, row_order, "row_order")
    cols = _order(col_levels, col_order, "col_order")
    if facet_row is not None and facet_col is not None:
        items = [(row, col) for row in rows for col in cols]
        nrows, ncols = len(rows), len(cols)
    elif facet_row is not None:
        items = [(row, "All") for row in rows]
        nrows, ncols = len(rows), 1
    else:
        items = [("All", col) for col in cols]
        ncols = min(col_wrap, len(items))
        nrows = (len(items) + ncols - 1) // ncols
    if len(items) > 36:
        raise ValueError("More than 36 panels requested; facet fewer groups or prepare multiple figures.")
    global_limits = _bounds(xy[finite_xy])
    panel_masks = []
    frame_keys = []
    for i, (row, col) in enumerate(items):
        mask = (row_values.to_numpy() == row) & (col_values.to_numpy() == col)
        panel_masks.append(mask)
        frame_keys.append(f"panel_{i:02d}")
    include = np.zeros(adata.n_obs, dtype=bool)
    panel_id = np.empty(adata.n_obs, dtype=object)
    panel_id[:] = None
    for key, mask in zip(frame_keys, panel_masks):
        include |= mask
        panel_id[mask] = key
    if not include.all():
        raise AssertionError("Every observation must be assigned exactly one facet.")
    if color_values is None:
        mapped_colors = np.full(adata.n_obs, "#A9B5BE", dtype=object)
    elif color_type == "categorical":
        mapped_colors = color_values.map(colors).to_numpy()
    else:
        mapped_colors = np.full(adata.n_obs, "#C0C7CD", dtype=object)
    marked = np.zeros(adata.n_obs, dtype=bool)
    if highlight is not None:
        highlight_labels, _, _ = _labels(adata.obs[highlight_key], highlight_key)
        unknown = set(highlight) - set(highlight_labels.unique())
        if unknown:
            raise ValueError(f"Unknown highlight labels in {highlight_key!r}: {sorted(unknown)}")
        marked = highlight_labels.isin(highlight).to_numpy()
    fields = {}
    radius_source = None
    common_radius = None
    global_grid_extent = None
    vv = None
    field_config = _field_options(velocity_options) if velocity else None
    if velocity is not None:
        expected = f"velocity_{embedding_key.removeprefix('X_')}"
        actual = expected if velocity_basis is None else str(velocity_basis)
        if not actual.startswith("velocity_"):
            actual = f"velocity_{actual}"
        if actual != expected or actual not in adata.obsm:
            raise ValueError(f"Expected matching projected velocity in adata.obsm[{expected!r}].")
        vv = np.asarray(adata.obsm[actual], dtype=float)
        if vv.shape != np.asarray(coordinates).shape:
            raise ValueError("Projected velocity must match the complete display embedding shape.")
        eligible = finite_xy & np.isfinite(vv[:, :2]).all(axis=1)
        if not eligible.any():
            raise ValueError("No jointly finite embedding and projected velocity rows exist.")
        # Derive the radius once, using the complete source pool, not a separate
        # radius optimized for each facet. Numeric facets with no eligible input
        # are retained as blank fields; they do not receive invented vectors.
        pooled = _prepare_velocity_field(
            adata, basis=basis, velocity_basis=actual,
            grid_size=field_config["grid_size"],
            n_neighbors=field_config["n_neighbors"],
            min_neighbors=field_config["min_neighbors"],
            min_coherence=field_config["min_coherence"],
            radius=field_config.get("radius"),
            grid_extent=(global_limits[0][0], global_limits[0][1],
                         global_limits[1][0], global_limits[1][1]),
        )
        common_radius = float(pooled["metadata"]["radius"])
        radius_source = "provided" if field_config.get("radius") is not None else "pooled_all_facets"
        for key, mask in zip(frame_keys, panel_masks):
            sel = np.flatnonzero(mask)
            if not (mask & eligible).any():
                fields[key] = None
                continue
            if shared_limits:
                box = global_limits
            else:
                local = xy[mask & finite_xy]
                box = _bounds(local) if len(local) else global_limits
            # A lightweight observation-only view avoids materializing a large
            # expression matrix when faceting an AnnData object.
            subset = SimpleNamespace(
                obs=adata.obs.iloc[sel].reset_index(drop=True),
                obs_names=pd.Index(adata.obs_names).take(sel),
                n_obs=len(sel),
                obsm={embedding_key: np.asarray(coordinates)[sel], actual: vv[sel]},
            )
            prep = _prepare_velocity_field(
                subset, basis=basis, velocity_basis=actual,
                grid_size=field_config["grid_size"], n_neighbors=field_config["n_neighbors"],
                min_neighbors=field_config["min_neighbors"], min_coherence=field_config["min_coherence"],
                radius=common_radius,
                grid_extent=(box[0][0], box[0][1], box[1][0], box[1][1]),
            )
            fields[key] = prep
            if shared_limits:
                this_grid = prep["grid"][['grid_id', 'x', 'y']].to_numpy()
                reference_grid = pooled["grid"][['grid_id', 'x', 'y']].to_numpy()
                if not np.array_equal(this_grid, reference_grid):
                    raise AssertionError("Facet velocity grids differ despite shared limits.")
        if shared_limits:
            global_grid_extent = list(pooled["metadata"]["grid_extent"])
    scale = None
    color_norm_field = None
    if velocity is not None:
        all_supported = [f["grid"].loc[f["grid"].supported, "speed"].to_numpy(dtype=float)
                         for f in fields.values() if f is not None]
        supported_speed = np.concatenate(all_supported) if all_supported else np.array([])
        if velocity == "grid" and shared_velocity_scale:
            if field_config["scale"] == "auto":
                # Shared limits guarantee identical grid spacing. If not, an
                # explicit common multiplier still avoids facet-wise scaling.
                grid_dx = min(float(np.diff(pooled["x"])[0]), float(np.diff(pooled["y"])[0]))
                scale = (0.65 * grid_dx / np.quantile(supported_speed, 0.9)
                         if len(supported_speed) else 1.0)
            else:
                scale = _positive_number(field_config["scale"], "scale")
        else:
            scale = field_config["scale"]
        if field_config["color_by"] == "coherence":
            color_norm_field = Normalize(0, 1)
        elif field_config["color_by"] == "speed":
            limit = max(float(supported_speed.max()) if len(supported_speed) else 0, 1e-12)
            color_norm_field = Normalize(0, limit)
        # Validate style before a figure exists, including stream density.
        for prepared in fields.values():
            if prepared is None:
                continue
            _field_style(prepared, mode=velocity, scale=scale, color=field_config["color"],
                         color_by=field_config["color_by"], cmap=field_config["cmap"],
                         norm=color_norm_field,
                         arrow_linewidth=field_config.get("arrow_linewidth", 1.1),
                         arrow_head_size=field_config.get("arrow_head_size", 5.5),
                         stream_density=field_config["stream_density"],
                         stream_linewidth=field_config.get("stream_linewidth", 1.2),
                         alpha=field_config.get("alpha", 0.9))
    default_size = (min(18.0, 4.35 * ncols + (1.4 if legend else .2)),
                    min(19.0, 4.0 * nrows + (0.8 if title else 0)))
    if figsize is not None:
        dims = np.asarray(figsize, dtype=float)
        if dims.shape != (2,) or not np.isfinite(dims).all() or (dims <= 0).any():
            raise ValueError("figsize must be a pair of positive finite values.")
    else:
        dims = default_size
    data_cells = pd.DataFrame({
        "obs_position": np.arange(adata.n_obs),
        "cell_id": pd.Index(adata.obs_names).astype(str),
        "facet_id": panel_id, "row_label": row_values.to_numpy(),
        "col_label": col_values.to_numpy(),
        "x": xy[:, 0], "y": xy[:, 1],
        "finite_embedding": finite_xy, "highlighted": marked,
        "displayed": np.zeros(adata.n_obs, dtype=bool),
    })
    if color is not None:
        data_cells["color_value"] = color_values.to_numpy()
    if vv is not None:
        data_cells["vx"] = vv[:, 0]
        data_cells["vy"] = vv[:, 1]
        data_cells["finite_velocity"] = np.isfinite(vv[:, :2]).all(axis=1)
    summaries = []
    grids, vectors, paths = [], [], []
    warning = []
    artists = {}
    axes = {}
    with style_context(style):
        # Figure legends are not included in constrained-layout calculations;
        # reserve an actual GridSpec row rather than floating a legend over
        # bottom-row facet labels or relying on bbox_inches='tight'.
        use_legend_row = bool(color_type == "categorical" and legend and colors
                              and len(categories) <= 30)
        if use_legend_row:
            legend_columns = min(6, len(categories))
            legend_rows = (len(categories) + legend_columns - 1) // legend_columns
            fig = plt.figure(figsize=dims, constrained_layout=True)
            gs = fig.add_gridspec(nrows + 1, ncols,
                                  height_ratios=[1.0] * nrows + [0.19 * legend_rows])
            axs = np.asarray([[fig.add_subplot(gs[r, c]) for c in range(ncols)]
                              for r in range(nrows)], dtype=object)
            legend_ax = fig.add_subplot(gs[-1, :])
            legend_ax.axis("off")
        else:
            fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=dims,
                                   constrained_layout=True)
            legend_ax = None
        for ax in axs.ravel()[len(items):]:
            ax.set_visible(False)
        for index, ((row, col), mask, key) in enumerate(zip(items, panel_masks, frame_keys)):
            ax = axs.ravel()[index]
            axes[key] = ax
            available = np.flatnonzero(mask & finite_xy)
            if len(available):
                if color_type == "categorical":
                    strata = color_values.iloc[available].to_numpy()
                elif highlight is not None:
                    strata = marked[available]
                else:
                    strata = np.full(len(available), "all", dtype=object)
                shown = available[display_indices(strata, max_points=max_points, random_state=int(random_state))]
            else:
                shown = np.empty(0, dtype=int)
            data_cells.loc[shown, "displayed"] = True
            if len(available):
                limits = global_limits if shared_limits else _bounds(xy[available])
            else:
                limits = global_limits
            if context == "all":
                ax.scatter(xy[finite_xy, 0], xy[finite_xy, 1], c="#B8C0C8", s=3.5,
                           alpha=0.11, linewidths=0, rasterized=True, zorder=0)
            if color_type == "numeric":
                vals = color_values.iloc[shown].to_numpy(dtype=float)
                good = np.isfinite(vals)
                if good.any():
                    artists[f"{key}_points"] = ax.scatter(
                        xy[shown[good], 0], xy[shown[good], 1], c=vals[good], cmap=cmap,
                        norm=color_norm, s=point_size, alpha=point_alpha, linewidths=0,
                        rasterized=len(shown) > 2000, zorder=1)
                if (~good).any():
                    ax.scatter(xy[shown[~good], 0], xy[shown[~good], 1], c="#B7BEC5",
                               s=point_size, alpha=point_alpha, linewidths=0, zorder=1)
            else:
                artists[f"{key}_points"] = ax.scatter(
                    xy[shown, 0], xy[shown, 1],
                    c=mapped_colors[shown].tolist(), s=point_size, alpha=point_alpha,
                    linewidths=0, rasterized=len(shown) > 2000, zorder=1)
            if marked[shown].any():
                high = shown[marked[shown]]
                ax.scatter(xy[high, 0], xy[high, 1], facecolors="none", edgecolors="#253E54",
                           linewidths=0.75, s=point_size * 2.1, zorder=3)
            ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{embedding_key.removeprefix('X_').upper()} 1", fontsize=8)
            ax.set_ylabel(f"{embedding_key.removeprefix('X_').upper()} 2", fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            title_parts = []
            if facet_row is not None:
                title_parts.append(f"{facet_row}: {row}")
            if facet_col is not None:
                title_parts.append(f"{facet_col}: {col}")
            ax.set_title("  |  ".join(title_parts) if title_parts else "All cells", fontsize=10)
            ax.text(0.02, 0.02, f"n = {int(mask.sum()):,}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, color="#536779")
            supported = 0
            if velocity is not None and fields[key] is not None:
                prepared = fields[key]
                rendered = _render_velocity_field(
                    ax, prepared, mode=velocity, scale=scale,
                    color=field_config["color"], color_by=field_config["color_by"],
                    cmap=field_config["cmap"], norm=color_norm_field,
                    arrow_linewidth=field_config.get("arrow_linewidth", 1.1),
                    arrow_head_size=field_config.get("arrow_head_size", 5.5),
                    stream_density=field_config["stream_density"],
                    stream_linewidth=field_config.get("stream_linewidth", 1.2),
                    alpha=field_config.get("alpha", 0.9))
                supported = int(prepared["grid"].supported.sum())
                artists[f"{key}_velocity"] = rendered["artists"]
                for source, collection in (("grid", grids), ("vectors", vectors), ("paths", paths)):
                    table = prepared["grid"] if source == "grid" else rendered["data"][source]
                    table = table.copy(deep=True)
                    table.insert(0, "facet_id", key)
                    table.insert(1, "row_label", row)
                    table.insert(2, "col_label", col)
                    collection.append(table)
                warning.extend(f"{key}: {note}" for note in prepared["warnings"] + rendered["warnings"])
                if supported == 0:
                    ax.text(.5, .9, "No supported local field", transform=ax.transAxes,
                            ha="center", va="top", fontsize=8, color="#8B4F4F")
            elif velocity is not None:
                # Preserve the attempted grid even when a facet has zero valid
                # projected vectors; omissions must remain inspectable.
                empty_grid = pooled["grid"].copy(deep=True)
                # An independently framed facet still attempted its own grid.
                # Reusing pooled coordinates here would misreport the support
                # locations whenever that facet has no eligible vectors.
                nx, ny = _grid_shape(field_config["grid_size"])
                gx, gy = np.meshgrid(np.linspace(*limits[0], nx),
                                     np.linspace(*limits[1], ny))
                empty_grid["x"] = gx.ravel()
                empty_grid["y"] = gy.ravel()
                empty_grid["vx"] = np.nan
                empty_grid["vy"] = np.nan
                empty_grid["speed"] = np.nan
                empty_grid["mean_speed"] = np.nan
                empty_grid["coherence"] = np.nan
                empty_grid["n_neighbors"] = 0
                empty_grid["weight_sum"] = 0.0
                for flag in ("supported", "count_supported", "coherence_supported", "nonzero", "displayed"):
                    empty_grid[flag] = False
                empty_grid["reason"] = "no_eligible_vectors"
                empty_grid.insert(0, "facet_id", key)
                empty_grid.insert(1, "row_label", row)
                empty_grid.insert(2, "col_label", col)
                grids.append(empty_grid)
                ax.text(.5, .9, "No eligible projected vectors", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8, color="#8B4F4F")
                warning.append(f"{key}: no jointly finite projected velocity rows")
            summaries.append({
                "facet_id": key, "row_label": row, "col_label": col,
                "n_cells": int(mask.sum()), "n_finite_embedding": len(available),
                "n_displayed": len(shown),
                "n_velocity_eligible": int((mask & finite_xy & np.isfinite(vv[:, :2]).all(axis=1)).sum()) if vv is not None else None,
                "n_grid_supported": supported if velocity is not None else None,
                "n_grid_total": len(fields[key]["grid"]) if velocity is not None and fields[key] is not None else len(pooled["grid"]) if velocity else None,
                "x_min": limits[0][0], "x_max": limits[0][1],
                "y_min": limits[1][0], "y_max": limits[1][1],
            })
        if color_type == "categorical" and legend and colors:
            handles = [Line2D([], [], color=colors[c], linestyle="none", marker="o",
                              markersize=6, label=c) for c in categories]
            if len(handles) <= 30:
                artists["legend"] = legend_ax.legend(
                    handles=handles, loc="center", frameon=False,
                    ncol=min(6, len(handles)), fontsize=8,
                    columnspacing=1.0, handletextpad=0.35,
                )
            else:
                warning.append("More than 30 categories; the figure legend was omitted. All palette entries remain in metadata.")
        if color_type == "numeric" and show_colorbar and color_norm is not None:
            scalar = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
            fig.colorbar(scalar, ax=list(axes.values()), orientation="horizontal",
                         fraction=0.028, pad=0.065, label=color)
        if velocity is not None and field_config["color_by"] is not None and show_colorbar:
            scalar = plt.cm.ScalarMappable(norm=color_norm_field, cmap=field_config["cmap"])
            fig.colorbar(scalar, ax=list(axes.values()), orientation="vertical",
                         fraction=0.02, pad=0.03,
                         label=f"Velocity {field_config['color_by']} (projected)")
        if title is not None:
            fig.suptitle(title, fontsize=13)
    result = PlotResult(
        fig=fig, axes=axes,
        data={
            "cells": data_cells,
            "facets": pd.DataFrame(summaries),
            "velocity_grid": pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
            "velocity_vectors": pd.concat(vectors, ignore_index=True) if vectors else pd.DataFrame(),
            "velocity_paths": pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(),
        },
        metadata={
            "plot": "embedding_facets", "display_basis": embedding_key,
            "facets": {"row": facet_row, "col": facet_col, "row_order": rows,
                       "col_order": cols, "missing_labels": {"row": row_missing, "col": col_missing}},
            "color_key": color, "color_type": color_type, "palette": colors,
            "palette_origin": palette_origin,
            "numeric_color_limits": [color_norm.vmin, color_norm.vmax] if color_norm is not None else None,
            "velocity": velocity, "velocity_key": f"velocity_{embedding_key.removeprefix('X_')}" if velocity is not None else None,
            "velocity_options": field_config, "velocity_radius": common_radius,
            "velocity_radius_source": radius_source, "shared_velocity_scale": shared_velocity_scale,
            "velocity_scale_resolved": float(scale) if velocity == "grid" and scale != "auto" else None,
            "velocity_shared_color_limits": [color_norm_field.vmin, color_norm_field.vmax] if color_norm_field is not None else None,
            "velocity_grid_extent": global_grid_extent,
            "velocity_support_definition": "neighbor count, nonzero local vector, and direction coherence; not statistical confidence" if velocity else None,
            "shared_limits": shared_limits, "context": context,
            "max_points_per_facet": max_points, "random_state": int(random_state),
            "n_cells_input": int(adata.n_obs), "n_panels": len(items),
            "interprets_time_or_trajectory": False,
        },
        warnings=warning,
        artists=artists,
    )
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return fig, axes, result.data
    return fig, axes
