"""PAGA topology on a display embedding with auditable condition contrasts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy import sparse

from ._paga_layout import pie_glyph, place_labels
from ._result import PlotResult
from ._style import category_palette
from ._utils import display_indices, resolve_embedding
from ._velocity_shift_alignment import _stored_alignment_table


def _string_labels(series, name):
    """Preserve missing labels separately from literal strings such as 'NA'."""
    unique = pd.unique(series.dropna().astype(object))
    labels = [str(value) for value in unique]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"{name} contains distinct values with identical string labels."
        )
    return pd.Series(
        [None if pd.isna(value) else str(value) for value in series],
        index=series.index,
        dtype=object,
    )


def _ordered_labels(values, name):
    raw = list(values)
    if any(pd.isna(value) for value in raw):
        raise ValueError(f"{name} cannot contain missing labels.")
    labels = [str(value) for value in raw]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique, nonempty labels.")
    return labels


def _get_palette(adata, key, names, supplied=None):
    if supplied is not None:
        return category_palette(names, supplied)
    series = adata.obs[key]
    stored = adata.uns.get(f"{key}_colors")
    if isinstance(series.dtype, pd.CategoricalDtype) and stored is not None:
        levels = series.cat.categories.astype(str).tolist()
        if len(stored) == len(levels):
            known = dict(zip(levels, stored))
            default = category_palette(names)
            default.update({name: known[name] for name in names if name in known})
            return category_palette(names, default)
    return category_palette(names)


def _graph(adata, node_key, paga_key, explicit_order, observed):
    if paga_key not in adata.uns or "connectivities" not in adata.uns[paga_key]:
        raise KeyError(f"adata.uns['{paga_key}']['connectivities'] not found.")
    payload = adata.uns[paga_key]
    groups = payload.get("groups")
    if groups is not None and (not isinstance(groups, str) or groups != node_key):
        raise ValueError(
            f"PAGA groups metadata {groups!r} does not match node_key={node_key!r}."
        )
    conn = payload["connectivities"]
    table_order = None
    if isinstance(conn, pd.DataFrame):
        table_order = _ordered_labels(conn.index, "connectivity row labels")
        columns = _ordered_labels(conn.columns, "connectivity column labels")
        if set(columns) != set(table_order):
            raise ValueError("Connectivity DataFrame row and column labels must match.")
        frame = conn.copy()
        frame.index, frame.columns = table_order, columns
        conn = frame.loc[table_order, table_order].to_numpy(dtype=float)
    if explicit_order is not None:
        names = _ordered_labels(explicit_order, "graph_node_order")
        source = "explicit"
        if table_order is not None and names != table_order:
            raise ValueError(
                "graph_node_order must match the labeled connectivity matrix row order."
            )
    elif table_order is not None:
        names, source = table_order, "connectivity_dataframe"
    elif payload.get("node_order") is not None:
        names = _ordered_labels(payload["node_order"], "stored graph node_order")
        source = "stored_node_order"
    elif isinstance(adata.obs[node_key].dtype, pd.CategoricalDtype):
        names = _ordered_labels(adata.obs[node_key].cat.categories, "state categories")
        source = "current_categorical_order"
    else:
        names = sorted(set(observed))
        source = "sorted_observed_labels"
    if not set(observed) <= set(names):
        raise ValueError(
            "graph_node_order must include every observed state; unknown/omitted labels were found."
        )
    if sparse.issparse(conn):
        matrix = conn.astype(float).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if matrix.shape != (len(names), len(names)):
            raise ValueError(
                "Connectivity shape must exactly match graph_node_order (or categorical graph order)."
            )
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Connectivity values must be finite and nonnegative.")
        difference = matrix - matrix.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-10 + 1e-7 * max(
            float(matrix.max()), 1.0
        ):
            raise ValueError(
                "PAGA connectivities must be symmetric; directed evidence needs a separate plot."
            )
        upper = sparse.triu(matrix, k=0).tocoo()
        entries = sorted(zip(upper.row, upper.col, upper.data))
    else:
        matrix = np.asarray(conn, dtype=float)
        if matrix.ndim != 2 or matrix.shape != (len(names), len(names)):
            raise ValueError(
                "Connectivity shape must exactly match graph_node_order (or categorical graph order)."
            )
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("Connectivity values must be finite and nonnegative.")
        if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
            raise ValueError(
                "PAGA connectivities must be symmetric; directed evidence needs a separate plot."
            )
        rows, columns = np.where(np.triu(matrix) > 0)
        entries = [(i, j, matrix[i, j]) for i, j in zip(rows, columns)]
    edges = pd.DataFrame(
        [
            {
                "source": names[i],
                "target": names[j],
                "weight": float(weight),
                "self_loop": bool(i == j),
            }
            for i, j, weight in entries
            if weight > 0
        ],
        columns=["source", "target", "weight", "self_loop"],
    )
    return names, edges, source


def _centers(cells, names, node_key, group0, group1, min_cells, agg):
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be one of {'mean', 'median'}.")
    aggregate = np.mean if agg == "mean" else np.median
    totals = {group: int(cells.condition.eq(group).sum()) for group in (group0, group1)}
    records = []
    for state in names:
        row = {node_key: state}
        if node_key != "state":
            row["state"] = state
        for suffix, group in enumerate((group0, group1)):
            selected = cells.state.eq(state) & cells.condition.eq(group)
            points = cells.loc[selected, ["x", "y"]].to_numpy(dtype=float)
            n = len(points)
            center = aggregate(points, axis=0) if n else [np.nan, np.nan]
            row.update(
                {
                    f"n{suffix}": n,
                    f"x{suffix}": float(center[0]),
                    f"y{suffix}": float(center[1]),
                    f"present{suffix}": n >= min_cells,
                    f"frac{suffix}": n / totals[group],
                }
            )
        row["dx"], row["dy"] = row["x1"] - row["x0"], row["y1"] - row["y0"]
        row["shift_display"] = float(np.hypot(row["dx"], row["dy"]))
        row["shift_umap"] = row["shift_display"]  # historical public source-table alias
        row["delta_frac"] = row["frac1"] - row["frac0"]
        if row["n0"] == 0 and row["n1"] == 0:
            status = "absent_from_contrast"
        elif row["n0"] == 0:
            status = "comparison_only"
        elif row["n1"] == 0:
            status = "reference_only"
        elif not (row["present0"] and row["present1"]):
            status = "insufficient_cells"
        else:
            status = "usable"
        row["coverage_status"] = status
        records.append(row)
    return pd.DataFrame(records)


def _velocity(
    adata,
    cells,
    names,
    node_key,
    embedding_key,
    coordinates,
    velocity_basis,
    min_cells,
    agg,
):
    expected = f"velocity_{embedding_key.removeprefix('X_')}"
    key = expected if velocity_basis is None else str(velocity_basis)
    if not key.startswith("velocity_"):
        key = f"velocity_{key}"
    if key != expected:
        raise ValueError(
            f"Projected velocity basis must match the display embedding: expected {expected!r}, got {key!r}."
        )
    if key not in adata.obsm:
        raise KeyError(f"Velocity embedding {key!r} not found in adata.obsm.")
    velocity = np.asarray(adata.obsm[key], dtype=float)
    if velocity.shape != coordinates.shape:
        raise ValueError(
            "Projected velocity shape must match the display embedding shape exactly."
        )
    finite = np.isfinite(velocity[:, :2]).all(axis=1)
    cells["velocity_x"], cells["velocity_y"] = velocity[:, 0], velocity[:, 1]
    cells["finite_velocity"] = finite
    eligible = cells.in_contrast & cells.state.notna() & finite
    aggregate = np.mean if agg == "mean" else np.median
    records = []
    for state in names:
        selected = eligible & cells.state.eq(state)
        values = velocity[selected.to_numpy(), :2]
        mean = aggregate(values, axis=0) if len(values) else [np.nan, np.nan]
        records.append(
            {
                node_key: state,
                "state": state,
                "vx": float(mean[0]),
                "vy": float(mean[1]),
                "n": len(values),
                "n_velocity": len(values),
                "present": len(values) >= min_cells,
                "vel_norm": float(np.hypot(*mean)),
            }
        )
    return (
        pd.DataFrame(records),
        key,
        int((cells.in_contrast & cells.state.notna() & ~finite).sum()),
    )


def _composition(adata, cells, names, pie_key, categories, palette):
    if pie_key not in adata.obs:
        raise KeyError(f"'{pie_key}' not found in adata.obs.")
    # All source tables use observation positions, independently of obs_names.
    values = _string_labels(adata.obs[pie_key], pie_key).reset_index(drop=True)
    selected = cells.in_contrast & cells.state.notna()
    observed = list(pd.unique(values[selected & values.notna()]))
    missing_key = None
    if values[selected].isna().any():
        missing_key = "(missing)"
        while missing_key in observed:
            missing_key = f"({missing_key})"
        observed.append(missing_key)
    if categories is None:
        categories = observed
    else:
        categories = _ordered_labels(categories, "pie_categories")
        if not set(observed) <= set(categories):
            raise ValueError(
                "pie_categories must include all observed categories in the selected conditions; omitted mass cannot be renormalized."
            )
    if not categories:
        raise ValueError("No pie categories are available in the selected conditions.")
    labels = values.copy()
    if missing_key is not None:
        labels = labels.fillna(missing_key)
    colors = _get_palette(adata, pie_key, categories, palette)
    if missing_key is not None and palette is None:
        colors[missing_key] = "#B8BEC4"
    records, long_records = [], []
    for state in names:
        subset = labels[selected & cells.state.eq(state)]
        total = len(subset)
        record = {"node": state}
        for category in categories:
            count = int(subset.eq(category).sum())
            fraction = count / total if total else 0.0
            record[category] = fraction
            long_records.append(
                {
                    "state": state,
                    "category": None if category == missing_key else category,
                    "category_label": category,
                    "is_missing": category == missing_key,
                    "count": count,
                    "denominator": total,
                    "fraction": fraction,
                }
            )
        records.append(record)
    return (
        pd.DataFrame(records),
        pd.DataFrame(long_records),
        categories,
        colors,
        missing_key,
    )


def _numeric_colors(
    adata,
    centroids,
    node_key,
    condition_key,
    group0,
    group1,
    mode,
    alignment_df,
    alignment_key,
    alignment_source,
    store_key,
    analysis_rep,
    delta_key,
    cmap,
    norm,
    vmin,
    vmax,
):
    table, params, representation = None, {}, None
    notes = []
    if mode == "alignment":
        if alignment_source not in {"provided", "stored"}:
            raise ValueError("alignment_source must be 'provided' or 'stored'.")
        if alignment_source == "provided":
            if not isinstance(alignment_df, pd.DataFrame):
                raise ValueError(
                    "node_color_mode='alignment' requires alignment_df or alignment_source='stored'."
                )
            table = alignment_df.copy(deep=True)
            if alignment_key not in table:
                raise ValueError(f"alignment_df has no {alignment_key!r} column.")
            if alignment_key != "alignment_cosine":
                table = table.rename(columns={alignment_key: "alignment_cosine"})
            params = dict(table.attrs.get("params", {}))
            chosen_rep = analysis_rep
            if "rep" not in table and chosen_rep is None:
                chosen_rep = params.get("rep") or params.get("basis") or "unspecified"
            table, representation, params = _stored_alignment_table(
                adata,
                store_key=store_key,
                alignment_table=table,
                analysis_rep=chosen_rep,
                node_key=node_key,
                condition_key=condition_key,
                group0=group0,
                group1=group1,
                state_order=None,
            )
            if representation == "unspecified":
                notes.append(
                    "Provided alignment representation is unspecified; values are rendered without inferring their analysis space."
                )
        else:
            table, representation, params = _stored_alignment_table(
                adata,
                store_key=store_key,
                alignment_table=None,
                analysis_rep=analysis_rep,
                node_key=node_key,
                condition_key=condition_key,
                group0=group0,
                group1=group1,
                state_order=None,
            )
        if not all(
            key in params and params[key] is not None
            for key in ("node_key", "condition_key", "group0", "group1")
        ):
            source_label = "Stored" if alignment_source == "stored" else "Provided"
            notes.append(
                f"{source_label} alignment contrast metadata is incomplete; the requested contrast could not be fully verified."
            )
        lookup = table.set_index(node_key)
        values = centroids[node_key].map(lookup.alignment_cosine).to_numpy(dtype=float)
        usable = (
            centroids[node_key].map(lookup.usable).fillna(False).to_numpy(dtype=bool)
        )
        color_values = np.where(usable, values, np.nan)
        title = f"{'Stored' if alignment_source == 'stored' else 'Provided'} alignment cosine\n{representation}"
        defaults = (-1.0, 1.0)
    else:
        if delta_key not in centroids:
            raise ValueError(
                f"delta_key {delta_key!r} is not in the descriptive centroid table."
            )
        values = pd.to_numeric(centroids[delta_key], errors="raise").to_numpy(
            dtype=float
        )
        color_values = values.copy()
        bound = max(
            float(np.max(np.abs(values[np.isfinite(values)])))
            if np.isfinite(values).any()
            else 0.0,
            0.01,
        )
        defaults = (-bound, bound)
        title = (
            f"Δ cell fraction ({group1} − {group0})"
            if delta_key == "delta_frac"
            else f"Display metric: {delta_key}"
        )
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if norm is None:
        norm = Normalize(
            defaults[0] if vmin is None else vmin, defaults[1] if vmax is None else vmax
        )
    elif not isinstance(norm, Normalize):
        raise TypeError("norm must be a Matplotlib Normalize instance.")
    else:
        norm.autoscale_None(color_values[np.isfinite(color_values)])
    if (
        norm.vmin is None
        or norm.vmax is None
        or not np.isfinite([norm.vmin, norm.vmax]).all()
        or norm.vmin >= norm.vmax
    ):
        raise ValueError("Numeric color limits must be finite with vmin < vmax.")
    color_map = plt.get_cmap(cmap or "coolwarm").with_extremes(bad="#A6ADB4")
    colors = {
        state: color_map(norm(value)) if np.isfinite(value) else "#A6ADB4"
        for state, value in zip(centroids[node_key], color_values)
    }
    return (
        colors,
        values,
        color_values,
        norm,
        color_map,
        title,
        table,
        representation,
        params,
        notes,
    )


def _size_for_count(count, maximum, node_size, scaled):
    return (
        float(node_size * (0.55 + 0.45 * np.sqrt(max(count, 0.0) / maximum)))
        if scaled
        else float(node_size)
    )


def _edge_width(weight, maximum, scale):
    return float(scale * (0.25 + 0.75 * weight / maximum))


def _draw_vector(
    ax, start, end, *, color, alpha, width, linewidth, head_size, velocity=False
):
    if width is not None:
        factor = 0.55 if velocity else 1.0
        dx, dy = np.asarray(end) - np.asarray(start)
        return ax.arrow(
            *start,
            dx,
            dy,
            width=width * factor,
            head_width=width * (5.5 if velocity else 8.5),
            head_length=max(
                float(np.hypot(dx, dy)) * 0.12, width * (5.5 if velocity else 8.5)
            ),
            length_includes_head=True,
            color=color,
            alpha=alpha,
            zorder=4,
        )
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=head_size,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def paga_shift_map(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    min_cells: int = 15,
    connectivity_threshold: float = 0.05,
    agg: str = "mean",
    background_size: float = 6.0,
    background_alpha: float = 0.12,
    node_size: float = 220.0,
    node_scale_by_n: bool = True,
    edge_lw: float = 2.0,
    edge_alpha: float = 0.32,
    arrow_width: Optional[float] = None,
    arrow_alpha: float = 0.95,
    arrow_scale: float = 1.0,
    label: bool = True,
    label_top_n: Optional[int] = None,
    label_fontsize: int = 8,
    palette: Optional[dict[str, Any]] = None,
    pie_key: Optional[str] = None,
    pie_categories: Optional[list[str]] = None,
    pie_palette: Optional[dict[str, Any]] = None,
    pie_size_scale: float = 1.0,
    velocity_basis: Optional[str] = None,
    show_velocity: bool = False,
    velocity_color: str = "#008C95",
    velocity_scale: float = 50.0,
    velocity_alpha: float = 0.95,
    node_color_mode: str = "palette",
    alignment_df: Optional[pd.DataFrame] = None,
    alignment_key: str = "alignment_cosine",
    delta_key: str = "delta_frac",
    constant_node_color: str = "gold",
    highlight_nodes: Optional[list[str]] = None,
    highlight_edgecolor: str = "black",
    highlight_lw: float = 2.0,
    ax=None,
    figsize: tuple[float, float] = (9.0, 6.4),
    title: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
    aspect="equal",
    xlim=None,
    ylim=None,
    show_endpoints: bool = True,
    endpoint_size: float = 46.0,
    arrow_linewidth: float = 1.6,
    arrow_head_size: float = 9.0,
    label_mode: str = "auto",
    label_offsets=None,
    legend: bool = True,
    legend_loc: str = "outside",
    graph_node_order=None,
    background: str = "all",
    max_points=None,
    random_state: int = 0,
    required_indices=None,
    alignment_source: str = "provided",
    store_key: str = "representation_stability",
    analysis_rep: Optional[str] = None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    show_colorbar: bool = True,
    cax=None,
    return_result: bool = False,
):
    """Draw undirected PAGA topology and condition contrasts on an embedding.

    Positions, mean/median cell centers, and arrows describe the first two
    display coordinates. They are not inferred cell trajectories. Reference
    anchors are open markers; comparison endpoints are filled and remain at
    their actual centers even when ``arrow_scale`` changes the displayed vector.
    States present in only one condition or below ``min_cells`` remain visible.
    Graph edges require usable reference anchors at both ends.

    Velocity and pie summaries use only ``group0`` and ``group1``. Projected
    velocity must use the matching ``velocity_<basis>`` key and match the
    embedding shape. Selected, labeled cells with nonfinite display coordinates
    are rejected before drawing; nonfinite background rows are omitted. Missing
    velocity rows are excluded jointly and their support counts are exported.
    Explicit ``pie_categories`` must include every observed selected category;
    missing values use a separate ``(missing)`` category (escaped if that literal
    label already exists). Retained slices are never silently renormalized.

    ``graph_node_order`` labels matrix rows/columns explicitly, including graph
    states absent from a subset. Without it, a labeled matrix, stored node_order,
    or current categorical order is used. The plot cannot recover historical
    category order that was not stored. Connectivity must be finite, nonnegative,
    symmetric, and the exact declared size. No edges or centers are synthesized.

    Alignment coloring accepts an ``alignment_df`` or reads stored analysis with
    ``alignment_source='stored'`` and ``analysis_rep``. Quantitative values are
    never recalculated from the display embedding. Known source metadata must
    match the requested contrast; unknown provenance is labeled explicitly.
    ``alignment_source`` is inactive for other color modes. Scalar colorbars
    describe numeric reference rims and comparison fills, including with pies.

    Default arrows and pies use point-sized graphics. An explicit ``arrow_width``
    retains legacy data-unit arrow widths. Labels use deterministic point-space
    placement, with ``label_offsets={state: (dx, dy)}`` overrides or
    ``label_mode='on_data'``. Label placement never changes centers or values.
    ``aspect='equal'`` preserves display-coordinate angles. Supplied axes own
    their layout and figure size; ``xlim``/``ylim`` support matched canvases.

    Background sampling affects drawing only, preserving state-by-condition
    strata and ``required_indices``. Source tables retain all input cells with
    display flags, all positive graph edges with omission reasons, scaled vector
    endpoints, pie counts, source alignment, and label offsets. Cell fractions
    and support counts are descriptive; they do not substitute for sample-level
    inference. ``return_result=True`` returns a :class:`PlotResult`; historical
    tuple/axis returns and ``return_data`` keys remain available.
    """
    for key in (node_key, condition_key):
        if key not in adata.obs:
            raise KeyError(f"'{key}' not found in adata.obs.")
    try:
        embedding_key, coordinates = resolve_embedding(adata, basis)
    except KeyError as exc:
        raise KeyError(f"Embedding {basis!r} not found in adata.obsm.") from exc
    coordinates = np.asarray(coordinates, dtype=float)
    xy = coordinates[:, :2]
    if pd.isna(group0) or pd.isna(group1) or str(group0) == str(group1):
        raise ValueError(
            "group0 and group1 must be distinct, nonmissing condition labels."
        )
    g0, g1 = str(group0), str(group1)
    if (
        isinstance(min_cells, bool)
        or not isinstance(min_cells, (int, np.integer))
        or min_cells < 1
    ):
        raise ValueError("min_cells must be a positive integer.")
    if not np.isfinite(connectivity_threshold) or connectivity_threshold < 0:
        raise ValueError("connectivity_threshold must be finite and nonnegative.")
    if node_color_mode not in {"palette", "alignment", "delta", "constant"}:
        raise ValueError(
            "node_color_mode must be one of {'palette', 'alignment', 'delta', 'constant'}."
        )
    if background not in {"all", "contrast"}:
        raise ValueError("background must be 'all' or 'contrast'.")
    if aspect not in {"equal", "auto"}:
        raise ValueError("aspect must be 'equal' or 'auto'.")
    if label_mode not in {"auto", "on_data"}:
        raise ValueError("label_mode must be 'auto' or 'on_data'.")
    if label_top_n is not None and (
        isinstance(label_top_n, bool)
        or not isinstance(label_top_n, (int, np.integer))
        or label_top_n < 0
    ):
        raise ValueError("label_top_n must be a nonnegative integer or None.")
    for key, value in {
        "node_size": node_size,
        "endpoint_size": endpoint_size,
        "edge_lw": edge_lw,
        "arrow_linewidth": arrow_linewidth,
        "arrow_head_size": arrow_head_size,
        "pie_size_scale": pie_size_scale,
        "background_size": background_size,
        "highlight_lw": highlight_lw,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key, value in {
        "arrow_scale": arrow_scale,
        "velocity_scale": velocity_scale,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    for key, value in {
        "background_alpha": background_alpha,
        "edge_alpha": edge_alpha,
        "arrow_alpha": arrow_alpha,
        "velocity_alpha": velocity_alpha,
    }.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between zero and one.")
    if arrow_width is not None and (not np.isfinite(arrow_width) or arrow_width <= 0):
        raise ValueError(
            "arrow_width must be positive data units or None for point-sized arrows."
        )
    for key, limits in (("xlim", xlim), ("ylim", ylim)):
        if limits is not None and (
            len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]
        ):
            raise ValueError(f"{key} must contain two finite increasing bounds.")
    for color in (velocity_color, constant_node_color, highlight_edgecolor):
        if not is_color_like(color):
            raise ValueError(f"Invalid color: {color!r}")
    states = _string_labels(adata.obs[node_key], node_key)
    conditions = _string_labels(adata.obs[condition_key], condition_key)
    cells = pd.DataFrame(
        {
            "obs_position": np.arange(adata.n_obs),
            "cell": adata.obs_names.astype(str),
            "state": states.to_numpy(),
            "condition": conditions.to_numpy(),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "finite_coordinates": np.isfinite(xy).all(axis=1),
        }
    )
    cells["in_contrast"] = cells.condition.isin([g0, g1])
    if any(not cells.condition.eq(group).any() for group in (g0, g1)):
        raise ValueError(
            "Both requested conditions must be present globally; a condition is absent."
        )
    analysis = cells.in_contrast & cells.state.notna()
    if (analysis & ~cells.finite_coordinates).any():
        raise ValueError(
            "Nonfinite display coordinates in selected, labeled cells; remove or repair whole rows before plotting."
        )
    observed = cells.state.dropna().unique().tolist()
    names, edge_table, order_source = _graph(
        adata, node_key, paga_key, graph_node_order, observed
    )
    centers = _centers(cells, names, node_key, g0, g1, min_cells, agg)
    center_index = centers.set_index(node_key, drop=False)
    highlights = {str(value) for value in (highlight_nodes or [])}
    if not highlights <= set(names):
        raise ValueError("highlight_nodes contains unknown graph states.")
    offsets = {}
    if label_offsets is not None:
        if not isinstance(label_offsets, Mapping):
            raise TypeError(
                "label_offsets must map state labels to (dx, dy) offsets in points."
            )
        for state, offset in label_offsets.items():
            if (
                str(state) not in names
                or len(offset) != 2
                or not np.isfinite(offset).all()
            ):
                raise ValueError(
                    "label_offsets must contain known states and two finite point offsets."
                )
            offsets[str(state)] = tuple(float(value) for value in offset)
    notes = []
    if order_source in {"current_categorical_order", "sorted_observed_labels"}:
        notes.append(
            "Graph rows use the current state order; supply graph_node_order if categories were reordered or removed after graph construction."
        )
    if cells.state.isna().any():
        notes.append(
            "Cells with missing state labels remain background-only and are excluded from state summaries."
        )
    if (~cells.finite_coordinates).any():
        notes.append(
            "Nonfinite context-background coordinate rows were omitted from drawing."
        )
    velocity, velocity_key = None, None
    if show_velocity:
        velocity, velocity_key, excluded_velocity = _velocity(
            adata,
            cells,
            names,
            node_key,
            embedding_key,
            coordinates,
            velocity_basis,
            min_cells,
            agg,
        )
        if excluded_velocity:
            notes.append(
                f"{excluded_velocity} selected cells with nonfinite projected velocity were excluded jointly from velocity summaries."
            )
    composition, composition_counts, pie_colors, missing_pie = None, None, {}, None
    if pie_key is not None:
        composition, composition_counts, pie_categories, pie_colors, missing_pie = (
            _composition(
                adata,
                cells,
                names,
                pie_key,
                pie_categories,
                pie_palette,
            )
        )
    node_colors = _get_palette(adata, node_key, names, palette)
    alignment, alignment_rep, alignment_params = None, None, {}
    numeric = node_color_mode in {"alignment", "delta"}
    if numeric:
        (
            node_colors,
            numeric_values,
            color_values,
            plot_norm,
            plot_cmap,
            scalar_title,
            alignment,
            alignment_rep,
            alignment_params,
            color_notes,
        ) = _numeric_colors(
            adata,
            centers,
            node_key,
            condition_key,
            group0,
            group1,
            node_color_mode,
            alignment_df,
            alignment_key,
            alignment_source,
            store_key,
            analysis_rep,
            delta_key,
            cmap,
            norm,
            vmin,
            vmax,
        )
        notes.extend(color_notes)
        centers["color_value"] = numeric_values
        centers["color_usable"] = np.isfinite(color_values)
    elif node_color_mode == "constant":
        node_colors = {state: constant_node_color for state in names}
    elif norm is not None or vmin is not None or vmax is not None:
        raise ValueError(
            "Numeric color limits apply only to delta or alignment coloring."
        )

    # All graph edges are retained, including those without usable reference anchors.
    reasons, drawn = [], []
    for edge in edge_table.itertuples(index=False):
        failures = []
        if edge.self_loop:
            failures.append("self_loop")
        if edge.weight < connectivity_threshold:
            failures.append("below_threshold")
        for endpoint in (edge.source, edge.target):
            entry = center_index.loc[endpoint]
            if entry.n0 == 0:
                failures.append("missing_reference_anchor")
            elif not entry.present0:
                failures.append("insufficient_reference_cells")
        reasons.append(";".join(dict.fromkeys(failures)) if failures else "displayed")
        drawn.append(not failures)
    edge_table["displayed"] = pd.Series(drawn, index=edge_table.index, dtype=bool)
    edge_table["reason"] = pd.Series(reasons, index=edge_table.index, dtype=object)
    for endpoint, suffix in (("source", "0"), ("target", "0")):
        edge_table[f"{endpoint}_x"] = edge_table[endpoint].map(
            center_index[f"x{suffix}"]
        )
        edge_table[f"{endpoint}_y"] = edge_table[endpoint].map(
            center_index[f"y{suffix}"]
        )
    edge_max = max(float(edge_table.weight.max()) if len(edge_table) else 0.0, 1e-12)
    edge_table["linewidth_points"] = [
        _edge_width(weight, edge_max, edge_lw) for weight in edge_table.weight
    ]
    edges = [
        (row.source, row.target, row.weight)
        for row in edge_table.itertuples(index=False)
        if row.displayed
    ]

    # Draw-only sampling is stratified by state and condition, without string sentinels.
    background_scope = cells.finite_coordinates & (
        cells.in_contrast if background == "contrast" else True
    )
    eligible = np.flatnonzero(background_scope.to_numpy())
    if required_indices is None:
        required = np.empty(0, dtype=int)
    else:
        raw = np.asarray(required_indices)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError(
                "required_indices must contain integer observation positions."
            )
        required = np.unique(raw.astype(int))
        if (
            np.any(required < 0)
            or np.any(required >= adata.n_obs)
            or not np.isin(required, eligible).all()
        ):
            raise ValueError(
                "required_indices must identify finite cells within the chosen background scope."
            )
    strata = np.empty(len(eligible), dtype=object)
    strata[:] = list(zip(cells.state.iloc[eligible], cells.condition.iloc[eligible]))
    shown = eligible[
        display_indices(
            strata, max_points, random_state, np.searchsorted(eligible, required)
        )
    ]
    cells["displayed"] = False
    cells.loc[shown, "displayed"] = True
    cells["background_scope"] = background_scope

    nmax = max(float(centers.n0.max()), 1.0)
    centers["reference_size_points2"] = [
        _size_for_count(n, nmax, node_size, node_scale_by_n) for n in centers.n0
    ]
    centers["comparison_size_points2"] = float(endpoint_size)
    centers["pie_radius_points"] = (
        np.sqrt(centers.reference_size_points2) / 2 * pie_size_scale
    )
    center_index = centers.set_index(node_key, drop=False)
    velocity_index = None if velocity is None else velocity.set_index(node_key)
    vector_rows = []
    for row in centers.itertuples(index=False):
        state = row.state
        start = (row.x0, row.y0)
        valid_shift = bool(row.present0 and row.present1)
        reason = "displayed" if valid_shift else row.coverage_status
        length = float(np.hypot(row.dx, row.dy))
        visible = valid_shift and arrow_scale > 0 and length > 0
        if valid_shift and not visible:
            reason = "zero_vector" if length == 0 else "zero_display_scale"
        vector_rows.append(
            {
                "state": state,
                "kind": "shift",
                "source_x": start[0],
                "source_y": start[1],
                "target_x": start[0] + row.dx * arrow_scale,
                "target_y": start[1] + row.dy * arrow_scale,
                "dx": row.dx,
                "dy": row.dy,
                "scale": float(arrow_scale),
                "displayed": visible,
                "reason": reason,
                "comparison_x": row.x1,
                "comparison_y": row.y1,
            }
        )
        if velocity_index is not None:
            vr = velocity_index.loc[state]
            valid_velocity = bool(row.present0 and vr.present)
            visible = (
                valid_velocity
                and velocity_scale > 0
                and np.isfinite(vr.vel_norm)
                and vr.vel_norm > 0
            )
            reason = (
                "displayed"
                if visible
                else (
                    "insufficient_velocity_cells"
                    if not vr.present
                    else "missing_reference_anchor"
                    if not row.present0
                    else "zero_vector"
                    if vr.vel_norm == 0
                    else "zero_display_scale"
                )
            )
            vector_rows.append(
                {
                    "state": state,
                    "kind": "velocity",
                    "source_x": start[0],
                    "source_y": start[1],
                    "target_x": start[0] + vr.vx * velocity_scale,
                    "target_y": start[1] + vr.vy * velocity_scale,
                    "dx": vr.vx,
                    "dy": vr.vy,
                    "scale": float(velocity_scale),
                    "displayed": visible,
                    "reason": reason,
                    "comparison_x": row.x1,
                    "comparison_y": row.y1,
                }
            )
    vectors = pd.DataFrame(vector_rows)

    owns_figure = ax is None
    if (
        cax is not None
        and numeric
        and show_colorbar
        and (ax is None or cax.figure is not ax.figure)
    ):
        raise ValueError("cax requires a supplied map ax in the same figure.")
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # Explicitly locked caller limits are part of a matched-panel layout.
    if not owns_figure:
        if xlim is None and not ax.get_autoscalex_on():
            xlim = ax.get_xlim()
        if ylim is None and not ax.get_autoscaley_on():
            ylim = ax.get_ylim()
    axes = {"map": ax}
    artists = {
        "reference_nodes": {},
        "comparison_nodes": {},
        "insufficient_nodes": [],
        "pies": {},
        "metric_rings": {},
        "highlight_rings": {},
        "vectors": {"shift": [], "velocity": []},
    }
    artists["background"] = ax.scatter(
        cells.x.iloc[shown],
        cells.y.iloc[shown],
        s=background_size,
        c="#C6CCD1",
        alpha=background_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    displayed_edges = edge_table.loc[edge_table.displayed]
    segments = [
        ((row.source_x, row.source_y), (row.target_x, row.target_y))
        for row in displayed_edges.itertuples(index=False)
    ]
    artists["topology"] = LineCollection(
        segments,
        colors="#82919F",
        linewidths=displayed_edges.linewidth_points,
        alpha=edge_alpha,
        zorder=1,
    )
    ax.add_collection(artists["topology"])
    composition_index = None if composition is None else composition.set_index("node")
    obstacles, label_anchors = [], {}
    for _, row in centers.iterrows():
        state, color = row[node_key], node_colors[row[node_key]]
        if row.n0:
            anchor = (float(row.x0), float(row.y0))
            label_anchors[state] = anchor
            reference_size = float(row.reference_size_points2)
            if row.present0:
                artists["reference_nodes"][state] = ax.scatter(
                    *anchor,
                    s=reference_size,
                    marker="o",
                    facecolors="white",
                    edgecolors=[color],
                    linewidths=1.7,
                    zorder=5,
                )
                radius = np.sqrt(reference_size) / 2
                if composition_index is not None:
                    radius = float(row.pie_radius_points)
                    box, ring, highlight = pie_glyph(
                        ax,
                        anchor,
                        [
                            composition_index.loc[state, category]
                            for category in pie_categories
                        ],
                        [pie_colors[category] for category in pie_categories],
                        radius,
                        color,
                        state in highlights,
                        highlight_edgecolor,
                        highlight_lw,
                    )
                    artists["pies"][state], artists["metric_rings"][state] = box, ring
                    if highlight is not None:
                        artists["highlight_rings"][state] = highlight
                elif state in highlights:
                    artists["highlight_rings"][state] = ax.scatter(
                        *anchor,
                        s=(np.sqrt(reference_size) + 6) ** 2,
                        facecolors="none",
                        edgecolors=highlight_edgecolor,
                        linewidths=highlight_lw,
                        zorder=6,
                    )
                obstacles.append((*anchor, radius + 3))
            else:
                marker = ax.scatter(
                    *anchor,
                    s=endpoint_size,
                    marker="x",
                    c=[color],
                    linewidths=1.5,
                    zorder=5,
                )
                artists["insufficient_nodes"].append(marker)
                obstacles.append((*anchor, np.sqrt(endpoint_size) / 2 + 2))
        if row.n1 and (show_endpoints or not row.n0):
            point = (float(row.x1), float(row.y1))
            if state not in label_anchors:
                label_anchors[state] = point
            marker_shape = (
                "o" if row.present1 and row.n0 else "^" if row.present1 else "x"
            )
            kwargs = (
                {"edgecolors": "white", "linewidths": 0.8}
                if marker_shape != "x"
                else {"linewidths": 1.5}
            )
            marker = ax.scatter(
                *point,
                s=endpoint_size,
                marker=marker_shape,
                c=[color],
                zorder=6,
                **kwargs,
            )
            artists["comparison_nodes"][state] = marker
            if not row.present1:
                artists["insufficient_nodes"].append(marker)
            obstacles.append((*point, np.sqrt(endpoint_size) / 2 + 2))
        if (
            state in highlights
            and state not in artists["highlight_rings"]
            and state in label_anchors
        ):
            # A requested highlight also applies to one-condition states
            # and states with an observed but insufficient reference group.
            artists["highlight_rings"][state] = ax.scatter(
                *label_anchors[state],
                s=(np.sqrt(endpoint_size) + 6) ** 2,
                facecolors="none",
                edgecolors=highlight_edgecolor,
                linewidths=highlight_lw,
                zorder=7,
            )
    shift_color = "#263746"
    for row in vectors.itertuples(index=False):
        if row.displayed:
            velocity_arrow = row.kind == "velocity"
            artist = _draw_vector(
                ax,
                (row.source_x, row.source_y),
                (row.target_x, row.target_y),
                color=velocity_color if velocity_arrow else shift_color,
                alpha=velocity_alpha if velocity_arrow else arrow_alpha,
                width=arrow_width,
                linewidth=arrow_linewidth * 0.8 if velocity_arrow else arrow_linewidth,
                head_size=arrow_head_size * 0.9 if velocity_arrow else arrow_head_size,
                velocity=velocity_arrow,
            )
            artists["vectors"][row.kind].append(artist)

    limits_data = cells.loc[background_scope, ["x", "y"]].to_numpy()
    anchors_data = np.vstack(
        [centers[["x0", "y0"]].to_numpy(), centers[["x1", "y1"]].to_numpy()]
    )
    drawn_vectors = vectors.loc[vectors.displayed, ["target_x", "target_y"]].to_numpy()
    limits_data = np.vstack([limits_data, anchors_data, drawn_vectors])
    limits_data = limits_data[np.isfinite(limits_data).all(axis=1)]
    if not len(limits_data):
        raise ValueError(
            "No finite coordinates or state anchors are available to display."
        )
    for axis, supplied, dimension in (("x", xlim, 0), ("y", ylim, 1)):
        low, high = np.min(limits_data[:, dimension]), np.max(limits_data[:, dimension])
        padding = max(float(high - low) * 0.08, 0.12)
        bounds = (low - padding, high + padding) if supplied is None else supplied
        getattr(ax, f"set_{axis}lim")(bounds)
    ax.set_aspect(aspect, adjustable="box")
    basis_name = embedding_key.removeprefix("X_")
    ax.set(
        xlabel=f"{basis_name.upper()} 1",
        ylabel=f"{basis_name.upper()} 2",
        xticks=[],
        yticks=[],
        title=title or f"PAGA topology and condition contrast · {g0} → {g1}",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    if numeric and show_colorbar:
        scalar = plt.cm.ScalarMappable(norm=plot_norm, cmap=plot_cmap)
        horizontal_colorbar = cax is None and legend and legend_loc == "outside"
        colorbar = fig.colorbar(
            scalar,
            ax=ax if cax is None else None,
            cax=cax,
            orientation="horizontal" if horizontal_colorbar else "vertical",
            fraction=0.035,
            pad=0.075 if horizontal_colorbar else 0.025,
        )
        colorbar.set_label(scalar_title, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        axes["colorbar"] = colorbar.ax
        artists["colorbar"] = colorbar
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#52616C",
            markersize=8,
            label=f"Reference: {g0}",
        ),
        Line2D(
            [],
            [],
            color=shift_color,
            marker=">",
            markersize=5,
            lw=arrow_linewidth,
            label=f"Condition contrast ×{arrow_scale:g}",
        ),
    ]
    if show_endpoints:
        legend_handles.insert(
            1,
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="#52616C",
                markeredgecolor="white",
                markersize=6,
                label=f"Comparison center: {g1}",
            ),
        )
    if show_velocity:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=velocity_color,
                marker=">",
                markersize=5,
                lw=arrow_linewidth * 0.8,
                label=f"Projected velocity ×{velocity_scale:g}",
            )
        )
    if (centers.coverage_status == "comparison_only").any():
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="^",
                linestyle="none",
                color="#687782",
                markersize=6,
                label="Comparison only; no reference",
            )
        )
    if (centers.coverage_status == "reference_only").any():
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#687782",
                markersize=7,
                label="Reference only; no comparison",
            )
        )
    if len(artists["insufficient_nodes"]):
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color="#687782",
                markersize=6,
                label=f"Insufficient cells (<{min_cells})",
            )
        )
    if numeric and (~centers.color_usable).any():
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#A6ADB4",
                markersize=6,
                label="Numeric evidence unavailable",
            )
        )
    if pie_key is not None:
        legend_handles.extend(
            Patch(
                facecolor=pie_colors[category],
                edgecolor="white",
                label=f"{pie_key}: {category}",
            )
            for category in pie_categories
        )
    if node_scale_by_n and centers.present0.any():
        counts = np.unique(
            np.round(
                np.quantile(centers.loc[centers.present0, "n0"], [0, 0.5, 1])
            ).astype(int)
        )
        for count in counts[:3]:
            size = _size_for_count(count, nmax, node_size, True)
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor="#83919A",
                    markersize=np.sqrt(size)
                    * (pie_size_scale if pie_key is not None else 1.0),
                    label=f"{g0}: {count} cells",
                )
            )
    if len(displayed_edges):
        weights = np.unique(np.quantile(displayed_edges.weight, [0, 1]))
        for weight in weights:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="#82919F",
                    alpha=edge_alpha,
                    lw=_edge_width(weight, edge_max, edge_lw),
                    label=f"Connectivity {weight:.2g}",
                )
            )
    if legend:
        largest_marker = max(
            (
                handle.get_markersize()
                for handle in legend_handles
                if isinstance(handle, Line2D)
            ),
            default=0.0,
        )
        # Matplotlib reserves fontsize * (0.65 * handleheight + 0.245)
        # points for a handle. Reserve its full diameter before row spacing.
        legend_handleheight = max(0.7, (largest_marker / 8.0 - 0.245) / 0.65)
        kwargs = (
            {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}
            if legend_loc == "outside"
            else {"loc": legend_loc}
        )
        artists["legend"] = ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8,
            handlelength=2.2,
            handleheight=legend_handleheight,
            labelspacing=0.65,
            borderaxespad=0.2,
            **kwargs,
        )
    if owns_figure:
        fig.tight_layout()
    ranking = centers.sort_values(
        "shift_display", ascending=False, na_position="last", kind="stable"
    )
    chosen = (
        list(ranking[node_key])
        if label_top_n is None
        else list(ranking[node_key].head(label_top_n))
    )
    chosen = [state for state in names if state in highlights] + chosen
    if label_top_n is None:
        chosen += list(label_anchors)
    chosen = list(dict.fromkeys(state for state in chosen if state in label_anchors))
    label_rows = [(state, *label_anchors[state]) for state in chosen] if label else []
    artists["labels"], labels_table = place_labels(
        ax,
        label_rows,
        fontsize=label_fontsize,
        mode=label_mode,
        offsets=offsets,
        obstacles=obstacles,
    )
    data = {
        "centroids": centers,
        "edges": edges,
        "edge_table": edge_table,
        "display_cells": cells,
        "vectors": vectors,
        "labels": labels_table,
    }
    if composition is not None:
        data.update(
            {"composition": composition, "composition_counts": composition_counts}
        )
    if velocity is not None:
        data["velocity"] = velocity
    if alignment is not None:
        data["alignment"] = alignment
    metadata = {
        "plot": "paga_shift_map",
        "display_only_geometry": True,
        "display_basis": embedding_key,
        "display_dimensions": [0, 1],
        "node_key": node_key,
        "condition_key": condition_key,
        "group0": g0,
        "group1": g1,
        "center_method": agg,
        "center_unit": "cells",
        "min_cells": int(min_cells),
        "node_color_mode": node_color_mode,
        "graph_key": paga_key,
        "graph_node_order": names,
        "graph_order_source": order_source,
        "graph_scope": "provided connectivity; not recomputed",
        "graph_directed": False,
        "connectivity_threshold": float(connectivity_threshold),
        "edge_weight_max": edge_max,
        "edge_width_formula": "edge_lw * (0.25 + 0.75 * weight / max_positive_graph_weight)",
        "graph_edges_total": len(edge_table),
        "graph_edges_displayed": len(edges),
        "edge_omission_counts": edge_table.loc[~edge_table.displayed, "reason"]
        .value_counts()
        .to_dict(),
        "background_scope": background,
        "background_size": background_size,
        "background_alpha": background_alpha,
        "display_sampling_strata": "state_by_condition",
        "max_points": max_points,
        "random_state": int(random_state),
        "required_indices": required.tolist(),
        "displayed_indices": shown.tolist(),
        "n_cells_input": adata.n_obs,
        "n_cells_displayed": len(shown),
        "velocity_key": velocity_key,
        "velocity_scope": "selected_conditions" if show_velocity else None,
        "velocity_conditions": [g0, g1] if show_velocity else None,
        "composition_scope": "selected_conditions" if pie_key else None,
        "pie_key": pie_key,
        "pie_categories": pie_categories if pie_key else None,
        "pie_palette": pie_colors,
        "pie_missing_category": missing_pie,
        "pie_size_units": "points",
        "pie_radius_formula": "sqrt(reference_size_points2)/2 * pie_size_scale",
        "node_size_units": "points_squared",
        "node_size_scale_by_n": node_scale_by_n,
        "reference_count_max": nmax,
        "reference_size_formula": "node_size * (0.55 + 0.45 * sqrt(n0 / max_n0))"
        if node_scale_by_n
        else "constant node_size",
        "fraction_denominator": "all cells in each selected condition, including missing state labels",
        "node_palette": node_colors,
        "show_endpoints": show_endpoints,
        "endpoint_size": endpoint_size,
        "comparison_markers": "actual unscaled comparison centers",
        "arrow_scale": arrow_scale,
        "velocity_scale": velocity_scale,
        "arrow_width": arrow_width,
        "arrow_width_units": "data (legacy)" if arrow_width is not None else "points",
        "arrow_linewidth": arrow_linewidth,
        "arrow_head_size": arrow_head_size,
        "arrow_shrink_points": 2.0 if arrow_width is None else 0.0,
        "vector_coordinate_units": "display embedding",
        "shift_color": shift_color,
        "velocity_color": velocity_color,
        "aspect": aspect,
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
        "label_mode": label_mode,
        "label_fontsize": label_fontsize,
        "label_offsets": {key: list(value) for key, value in offsets.items()},
        "highlight_nodes": sorted(highlights),
        "highlight_edgecolor": highlight_edgecolor,
        "legend": legend,
        "legend_loc": legend_loc,
        "legend_labels": [item.get_label() for item in legend_handles],
    }
    if numeric:
        metadata.update(
            {
                "scalar_source": alignment_source
                if node_color_mode == "alignment"
                else "display_cell_composition",
                "scalar_representation": alignment_rep,
                "alignment_params": alignment_params,
                "scalar_metric": alignment_key
                if node_color_mode == "alignment"
                else delta_key,
                "color_norm": {
                    "class": type(plot_norm).__name__,
                    "vmin": plot_norm.vmin,
                    "vmax": plot_norm.vmax,
                },
                "colormap": plot_cmap.name,
                "colorbar_label": scalar_title,
                "show_colorbar": show_colorbar,
                "quantitative_values_recomputed": False,
            }
        )
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes=axes,
            data=data,
            metadata=metadata,
            warnings=notes,
            artists=artists,
        )
    if return_data:
        return fig, ax, data
    return (fig, ax) if owns_figure else ax
