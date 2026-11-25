"""Compass maps with separate display context and stored scientific evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm, is_color_like
from matplotlib.lines import Line2D

from ._compass_data import (
    context_nodes,
    ood_nodes,
    sample_context,
    stored_evidence,
    velocity_nodes,
)
from ._paga_layout import place_labels
from ._paga_shift_map import _draw_vector, _graph, _node_anchors, _string_labels
from ._result import PlotResult
from ._style import category_palette
from ._table_plot_utils import panel_axes
from ._utils import display_indices, resolve_embedding


def _positive(value, name, *, zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}."
        )


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _normalization(values, supplied, vmin, vmax, *, kind):
    if supplied is not None:
        if not isinstance(supplied, Normalize):
            raise TypeError(f"{kind}_norm must be a matplotlib Normalize.")
        if vmin is not None or vmax is not None:
            raise ValueError(f"Supply {kind}_norm or its vmin/vmax, not both.")
        norm = supplied
    else:
        finite = np.asarray(values)[np.isfinite(values)]
        if kind == "alignment":
            low, high = -1.0, 1.0
        elif kind == "delta":
            extent = max(float(np.max(np.abs(finite))) if len(finite) else 0.0, 1e-6)
            low, high = -extent, extent
        else:
            low, high = (
                (float(finite.min()), float(finite.max()))
                if len(finite)
                else (0.0, 1.0)
            )
            if np.isclose(low, high):
                low, high = 0.0, max(high, 1.0)
        low = low if vmin is None else vmin
        high = high if vmax is None else vmax
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Color limits must be finite and strictly increasing.")
        norm = (
            TwoSlopeNorm(0.0, vmin=low, vmax=high)
            if kind == "delta" and low < 0 < high
            else Normalize(low, high)
        )
    finite = np.asarray(values)[np.isfinite(values)]
    norm.autoscale_None(
        finite
        if len(finite)
        else np.array([-1.0, 1.0])
        if kind == "alignment"
        else np.array([0.0, 1.0])
    )
    if not np.isfinite([norm.vmin, norm.vmax]).all() or norm.vmin >= norm.vmax:
        raise ValueError("Color normalization needs finite increasing bounds.")
    return norm


def _canvas(layout, ax, axes, figsize):
    names = (
        ["map"]
        if layout == "map"
        else ["map", "effect", "alignment", "abundance", "coverage"]
    )
    if ax is not None:
        if axes is not None or layout != "map":
            raise ValueError(
                "ax is available only for layout='map'; use axes for summary composition."
            )
        axes = {"map": ax}
    if axes is not None:
        return panel_axes(names, axes, None)
    if layout == "map":
        return panel_axes(names, None, figsize or (9.5, 8.0))
    fig = plt.figure(figsize=figsize or (14.0, 8.5), constrained_layout=True)
    if hasattr(fig, "get_layout_engine"):
        fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.955))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.8, 1, 1))
    return (
        fig,
        {
            "map": fig.add_subplot(grid[:, 0]),
            "effect": fig.add_subplot(grid[0, 1]),
            "alignment": fig.add_subplot(grid[0, 2]),
            "abundance": fig.add_subplot(grid[1, 1]),
            "coverage": fig.add_subplot(grid[1, 2]),
        },
        True,
    )


def _unavailable(ax, message):
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=".4",
        wrap=True,
    )


def _summary_panels(
    axes,
    evidence,
    sample_counts,
    coverage,
    states,
    groups,
    colors,
    mode,
    metric,
    sample_key,
):
    positions = {state: i for i, state in enumerate(states)}
    artists = {"effect": [], "alignment": [], "abundance": [], "coverage": []}
    for name in ("effect", "alignment", "abundance", "coverage"):
        axis = axes[name]
        axis.set_yticks(np.arange(len(states)))
        axis.set_yticklabels(states, fontsize=8)
        axis.set_ylim(len(states) - 0.5, -0.5)
        axis.grid(axis="x", alpha=0.15)
    for row in evidence.loc[evidence.state.isin(states)].itertuples(index=False):
        y = positions[row.state]
        if row.effect_available:
            artists["effect"].append(
                axes["effect"].scatter(
                    row.effect,
                    y,
                    s=34,
                    facecolor="white" if row.effect_descriptive else "#315F78",
                    edgecolor="#315F78",
                    zorder=3,
                )
            )
            if row.interval_available:
                axes["effect"].hlines(
                    y, row.ci_low, row.ci_high, color="#315F78", lw=1.5
                )
            if row.effect_descriptive:
                axes["effect"].annotate(
                    "unverified support"
                    if row.effect_support_status == "source_support_not_recorded"
                    else "descriptive",
                    (row.effect, y),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=6,
                    color=".4",
                )
        else:
            axes["effect"].text(
                0.02,
                y,
                row.effect_support_status.replace("source_", "").replace("_", " "),
                transform=axes["effect"].get_yaxis_transform(),
                fontsize=7,
                color=".45",
                va="center",
            )
        if row.alignment_available:
            artists["alignment"].append(
                axes["alignment"].scatter(row.alignment, y, s=34, color="#467E81")
            )
        else:
            axes["alignment"].text(
                0.02,
                y,
                "unavailable",
                transform=axes["alignment"].get_yaxis_transform(),
                fontsize=7,
                color=".45",
                va="center",
            )
    axes["effect"].set(
        title="Stored displacement" if mode == "stored" else "Display displacement",
        xlabel=metric.replace("_", " "),
    )
    axes["effect"].set_xlim(left=0)
    if (
        evidence.loc[evidence.state.isin(states), "interval_reason"]
        .eq("stored_interval_unverified_support")
        .any()
    ):
        axes["effect"].text(
            0,
            -0.17,
            "Supplied intervals; source support unverified",
            transform=axes["effect"].transAxes,
            fontsize=7,
            color=".4",
        )
    axes["alignment"].set(
        title="Stored alignment" if mode == "stored" else "Display alignment",
        xlabel="Cosine",
        xlim=(-1.08, 1.08),
    )
    axes["alignment"].axvline(0, color=".7", lw=0.8, ls=":")
    if sample_key is None:
        _unavailable(
            axes["abundance"],
            "Per-sample abundance unavailable\nNo sample_key supplied",
        )
    else:
        sample_ids = sorted(sample_counts["sample"].unique())
        symbols = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "h"]
        sample_symbols = {
            sample: symbols[i] if i < len(symbols) else f"${i + 1}$"
            for i, sample in enumerate(sample_ids)
        }
        for i, group in enumerate(groups):
            frame = sample_counts.loc[
                sample_counts.condition.eq(group) & sample_counts.state.isin(states)
            ]
            for state, rows in frame.groupby("state", sort=False):
                offset = (i - 0.5) * 0.28
                valid = rows.loc[rows.included]
                jitter = (
                    np.linspace(-0.055, 0.055, len(valid))
                    if len(valid) > 1
                    else np.zeros(len(valid))
                )
                for j, sample_row in enumerate(valid.itertuples(index=False)):
                    artists["abundance"].append(
                        axes["abundance"].scatter(
                            sample_row.state_fraction * 100,
                            positions[state] + offset + jitter[j],
                            s=25,
                            marker=sample_symbols[sample_row.sample],
                            color=colors[group],
                            alpha=0.8,
                        )
                    )
                if (~rows.included).any():
                    axes["abundance"].text(
                        0.98,
                        positions[state] + offset,
                        "NA denominator",
                        ha="right",
                        va="center",
                        transform=axes["abundance"].get_yaxis_transform(),
                        fontsize=6,
                        color=colors[group],
                    )
        axes["abundance"].set_xlim(left=-1)
        artists["sample_legend"] = axes["abundance"].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker=sample_symbols[sample],
                    color=".4",
                    linestyle="none",
                    markersize=4,
                    label=sample,
                )
                for sample in sample_ids
            ],
            fontsize=6,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.17),
            ncol=min(6, max(len(sample_ids), 1)),
        )
    axes["abundance"].set(
        title="Observed sample abundance", xlabel="State / annotated sample cells (%)"
    )
    for i, group in enumerate(groups):
        frame = coverage.loc[coverage.condition.eq(group) & coverage.state.isin(states)]
        for row in frame.itertuples(index=False):
            y = positions[row.state] + (i - 0.5) * 0.3
            artists["coverage"].append(
                axes["coverage"].scatter(row.n_cells, y, s=22, color=colors[group])
            )
            text = (
                f"{row.n_cells} cells"
                if sample_key is None
                else f"{row.n_cells} cells / {int(row.n_samples)}/{int(row.n_samples_total)} samples"
            )
            axes["coverage"].annotate(
                text,
                (row.n_cells, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=colors[group],
            )
    max_count = (
        float(coverage.loc[coverage.state.isin(states), "n_cells"].max())
        if len(coverage)
        else 1
    )
    axes["coverage"].set(
        xlim=(-max(1.0, max_count * 0.04), max(4.0, max_count * 2.4)),
        title="Current data coverage",
        xlabel="Observed state cells",
    )
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", color=colors[g], markersize=4, label=g
        )
        for g in groups
    ]
    axes["coverage"].legend(
        handles=handles, fontsize=7, frameon=False, loc="lower right"
    )
    return artists


def recovery_compass(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    velocity_basis: Optional[str] = None,
    ood_key: Optional[str] = None,
    min_cells: int = 15,
    connectivity_threshold: float = 0.05,
    node_size_mode: str = "group1_n",
    node_size_scale: float = 380.0,
    fill_color_mode: str = "alignment",
    fill_cmap: str = "coolwarm",
    fill_vmin: Optional[float] = None,
    fill_vmax: Optional[float] = None,
    ring_mode: str = "ood_frac",
    ring_color: str = "gold",
    ring_max_lw: float = 4.0,
    arrow_color_mode: str = "shift",
    arrow_color: str = "black",
    arrow_cmap: str = "magma",
    arrow_scale: float = 1.0,
    arrow_width: Optional[float] = None,
    edge_alpha: float = 0.45,
    edge_lw: float = 2.0,
    bg_size: float = 5.0,
    bg_alpha: float = 0.08,
    label: bool = True,
    label_top_n: Optional[int] = 12,
    label_fontsize: int = 8,
    legend: bool = True,
    title: Optional[str] = None,
    figsize=None,
    ax=None,
    return_data: bool = False,
    show: bool = True,
    caxes=None,
    graph_node_order=None,
    return_result: bool = False,
    mode: str = "display",
    shift_df=None,
    alignment_df=None,
    analysis_rep=None,
    effect_metric: str = "auto",
    report_state_key=None,
    alignment_state_key=None,
    velocity_scope: str = "contrast",
    node_position: str = "reference",
    sample_key=None,
    ood_threshold=None,
    ood_scope: str = "contrast",
    fill_norm=None,
    arrow_norm=None,
    arrow_vmin=None,
    arrow_vmax=None,
    node_size_reference=None,
    arrow_linewidth: float = 1.6,
    arrow_head_size: float = 11.0,
    show_endpoints: bool = True,
    label_mode: str = "auto",
    label_offsets=None,
    label_names=None,
    max_points=None,
    random_state: int = 0,
    layout: str = "map",
    axes=None,
    show_colorbar=None,
    max_states: int = 12,
):
    """Plot condition geometry with explicit evidence, coverage and missingness.

    ``mode='display'`` (default) calculates descriptive means and cosine in the
    first two display coordinates. ``velocity_scope='contrast'`` now limits
    velocity to the selected conditions; ``'all'`` restores the legacy scope,
    and ``'group0'``/``'group1'`` select one condition. Projected velocities must
    match ``basis`` and only jointly finite vector rows count toward support.
    Zero displacement and ``arrow_scale=0`` produce no arrow.

    ``mode='stored'`` requires ``shift_df`` and ``analysis_rep``. Optional
    ``alignment_df`` supplies independently eligible alignment. The stored
    quantities and provenance are validated, never estimated from the map.
    ``report_state_key``/``alignment_state_key`` resolve nonstandard state IDs.
    ``effect_metric='auto'`` prefers a finite normalized magnitude, then raw
    magnitude. Intervals must have matching units; raw intervals never label
    normalized effects. Source counts/status gate evidence independently of
    current display coverage. Known low-sample or descriptive intervals are
    withheld, while original estimates/endpoints remain in exported sources.

    ``node_position='pooled'`` retains observed condition-enriched states as
    topology nodes. Actual condition centers and vector origins stay separate;
    no missing reference or shift is synthesized. ``ood_scope`` defaults to
    the contrast. OOD fractions divide flagged by finitely scored cells;
    unmeasured states remain unavailable. ``ood_threshold`` freezes a supplied
    cutoff; otherwise the selected scope's finite-score 95th percentile is
    used and recorded as a descriptive threshold.

    ``layout='summary'`` adds effect, alignment, per-sample abundance and current
    coverage panels. ``sample_key`` uses the complete selected-condition sample
    roster: zero-state samples have observed zero frequency, and samples with
    no annotated denominator stay missing. Without a sample key no samples or
    intervals are fabricated. Supplied ``axes`` map map/effect/alignment/
    abundance/coverage to distinct axes. ``max_states`` limits side panels only.

    ``fill_norm``/``arrow_norm`` or explicit limits support matched panels;
    signed abundance defaults to a zero-centered scale. ``node_size_reference``
    shares the sizing denominator. Default arrows/labels use point units;
    explicit ``arrow_width`` retains legacy data units. ``max_points`` affects
    background drawing only. ``show_colorbar`` is independent of ``legend``;
    None preserves the legacy coupling. Caller figures retain their layout.

    Default map returns are ``(fig, ax)`` or the supplied ax; ``return_data``
    adds the observed-contrast node table and displayed edges as before.
    Summary returns ``(fig, axes)``. ``return_result=True`` takes priority and
    preserves cells/nodes/edges/graph_nodes/glyphs plus evidence and coverage
    tables. No AnnData or supplied result table is modified. This plot does not
    establish recovery, causality, cell trajectories or directed PAGA edges.
    """
    for name, value, allowed in (
        ("mode", mode, {"display", "stored"}),
        ("layout", layout, {"map", "summary"}),
        ("node_position", node_position, {"reference", "pooled"}),
        (
            "node_size_mode",
            node_size_mode,
            {"group1_n", "group0_n", "delta_abs", "constant"},
        ),
        ("fill_color_mode", fill_color_mode, {"alignment", "delta_frac", "constant"}),
        ("arrow_color_mode", arrow_color_mode, {"shift", "constant"}),
        ("ring_mode", ring_mode, {"ood_frac", "none"}),
        ("velocity_scope", velocity_scope, {"contrast", "all", "group0", "group1"}),
        ("ood_scope", ood_scope, {"contrast", "all", "group0", "group1"}),
        ("label_mode", label_mode, {"auto", "on_data"}),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}.")
    if mode == "display" and any(
        v is not None
        for v in (
            shift_df,
            alignment_df,
            analysis_rep,
            report_state_key,
            alignment_state_key,
        )
    ):
        raise ValueError("Stored evidence inputs require mode='stored'.")
    if pd.isna(group0) or pd.isna(group1) or str(group0) == str(group1):
        raise ValueError("group0 and group1 must be distinct nonmissing labels.")
    groups = (str(group0), str(group1))
    for key in (node_key, condition_key):
        if key not in adata.obs:
            raise KeyError(f"obs[{key!r}] not found.")
    for name, value in (("min_cells", min_cells), ("max_states", max_states)):
        _integer(value, name, 1)
    _integer(random_state, "random_state")
    if label_top_n is not None:
        _integer(label_top_n, "label_top_n")
    if max_points is not None:
        _integer(max_points, "max_points", 1)
    for name, value in (
        ("connectivity_threshold", connectivity_threshold),
        ("arrow_scale", arrow_scale),
        ("ring_max_lw", ring_max_lw),
    ):
        _positive(value, name, zero=True)
    for name, value in (
        ("node_size_scale", node_size_scale),
        ("edge_lw", edge_lw),
        ("bg_size", bg_size),
        ("arrow_linewidth", arrow_linewidth),
        ("arrow_head_size", arrow_head_size),
        ("label_fontsize", label_fontsize),
    ):
        _positive(value, name)
    if arrow_width is not None:
        _positive(arrow_width, "arrow_width")
    if node_size_reference is not None:
        _positive(node_size_reference, "node_size_reference")
    for name, value in (("edge_alpha", edge_alpha), ("bg_alpha", bg_alpha)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
    for value in (ring_color, arrow_color):
        if not is_color_like(value):
            raise ValueError(f"Invalid color {value!r}.")
    for name, value in (
        ("label", label),
        ("legend", legend),
        ("show_endpoints", show_endpoints),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
    if show_colorbar is None:
        show_colorbar = legend
    elif not isinstance(show_colorbar, (bool, np.bool_)):
        raise TypeError("show_colorbar must be boolean or None.")
    if figsize is not None and (
        len(figsize) != 2 or not np.isfinite(figsize).all() or min(figsize) <= 0
    ):
        raise ValueError("figsize must contain two positive finite values.")
    try:
        embedding_key, coordinates = resolve_embedding(adata, basis)
    except KeyError as exc:
        raise KeyError(f"Embedding {basis!r} not found in adata.obsm.") from exc
    xy = np.asarray(coordinates[:, :2], dtype=float)
    labels = _string_labels(adata.obs[node_key], node_key)
    conditions = _string_labels(adata.obs[condition_key], condition_key)
    cells = pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "obs_position": np.arange(adata.n_obs),
            "node": adata.obs[node_key].to_numpy(),
            "node_label": labels.to_numpy(),
            "state": labels.to_numpy(),
            "condition": conditions.to_numpy(),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "finite_coordinates": np.isfinite(xy).all(axis=1),
        }
    )
    cells["in_contrast"] = cells.condition.isin(groups)
    cells["contrast_included"] = cells.in_contrast & cells.state.notna()
    if not cells.contrast_included.any():
        raise ValueError("No annotated cells are available in the selected contrast.")
    if (cells.contrast_included & ~cells.finite_coordinates).any():
        raise ValueError(
            "Selected annotated cells require jointly finite display coordinates."
        )
    names, edge_table, order_source = _graph(
        adata, node_key, paga_key, graph_node_order, labels.dropna().unique().tolist()
    )
    nodes = context_nodes(cells, names, node_key, groups, min_cells)
    velocities, velocity_key, velocity_available = velocity_nodes(
        adata,
        cells,
        names,
        embedding_key,
        velocity_basis,
        groups,
        velocity_scope,
        min_cells,
    )
    nodes = nodes.merge(velocities, on="state", how="left", validate="one_to_one")
    dot = nodes.dx * nodes.vx + nodes.dy * nodes.vy
    denominator = nodes.shift_norm * nodes.vel_norm
    nodes["display_alignment_cosine"] = np.divide(
        dot,
        denominator,
        out=np.full(len(nodes), np.nan),
        where=np.isfinite(denominator) & denominator.gt(0),
    )
    nodes["alignment_cosine"] = nodes.display_alignment_cosine
    nodes["usable_velocity"] = (
        nodes.usable_shift
        & nodes.velocity_present
        & np.isfinite(nodes.display_alignment_cosine)
    )
    ood, resolved_threshold = ood_nodes(
        adata, cells, names, ood_key, ood_threshold, groups, ood_scope
    )
    nodes = nodes.merge(ood, on="state", how="left", validate="one_to_one")
    notes = []
    if not velocity_available:
        notes.append(
            f"Projected velocity {velocity_key!r} unavailable; display alignment remains missing."
        )
    if velocity_scope == "all":
        notes.append("Velocity pools all input conditions (explicit legacy scope).")
    if ood_key is None and ring_mode == "ood_frac":
        notes.append("OOD score not supplied; node borders do not encode OOD burden.")
    elif ring_mode == "ood_frac" and nodes.ood_n_scored.eq(0).any():
        notes.append(
            "States without finite OOD scores have unavailable burden, not zero burden."
        )
    if ood_key is not None and ood_threshold is None:
        notes.append(
            "OOD threshold is the selected scope's descriptive 95th percentile, not a fitted calibration."
        )
    if order_source in {"current_categorical_order", "sorted_observed_labels"}:
        notes.append(
            "PAGA matrix order is inferred from current labels; supply graph_node_order when known."
        )
    source_shift = source_alignment = None
    provenance = {"status": "display_only"}
    if mode == "stored":
        evidence, source_shift, source_alignment, metric, provenance, messages = (
            stored_evidence(
                shift_df,
                alignment_df,
                node_key=node_key,
                condition_key=condition_key,
                groups=groups,
                analysis_rep=analysis_rep,
                report_state_key=report_state_key,
                alignment_state_key=alignment_state_key,
                effect_metric=effect_metric,
                min_cells=min_cells,
            )
        )
        notes.extend(messages)
        lookup = evidence.set_index("state")
        nodes["alignment_cosine"] = nodes.state.map(lookup.alignment)
        nodes["stored_alignment_available"] = (
            nodes.state.map(lookup.alignment_available).fillna(False).astype(bool)
        )
    else:
        metric = "display_shift_norm"
        evidence = pd.DataFrame(
            {
                "state": nodes.state,
                "effect": nodes.shift_norm,
                "effect_source_eligible": nodes.usable_shift,
                "effect_available": nodes.usable_shift & np.isfinite(nodes.shift_norm),
                "effect_source_status": nodes.coverage_status,
                "effect_metric": metric,
                "effect_descriptive": True,
                "effect_supported": False,
                "effect_support_status": np.where(
                    nodes.usable_shift, "display_descriptive", nodes.coverage_status
                ),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "interval_available": False,
                "interval_reason": "not_estimated",
                "alignment": nodes.display_alignment_cosine,
                "alignment_source_eligible": nodes.usable_velocity,
                "alignment_available": nodes.usable_velocity,
                "alignment_source_status": "display_descriptive",
                "alignment_reason": np.where(
                    nodes.usable_velocity, "available", "display_unavailable"
                ),
            }
        )
    context_states = list(dict.fromkeys([*names, *evidence.state]))
    sample_counts, coverage = sample_context(
        adata, cells, context_states, groups, sample_key
    )
    anchors = _node_anchors(cells, nodes, min_cells, "mean", node_position)
    anchor_index = anchors.set_index("state")
    reasons = []
    for row in edge_table.itertuples(index=False):
        reason = (
            "self_loop"
            if row.self_loop
            else "below_threshold"
            if row.weight < connectivity_threshold
            else "displayed"
        )
        if reason == "displayed":
            for state in (row.source, row.target):
                anchor = anchor_index.loc[state]
                if not anchor.observed:
                    reason = f"missing_{node_position}_anchor"
                    break
                if not anchor.usable:
                    reason = "insufficient_reference_cells"
                    break
        reasons.append(reason)
    edge_table["reason"] = reasons
    edge_table["displayed"] = edge_table.reason.eq("displayed")
    for endpoint in ("source", "target"):
        for dim in ("x", "y"):
            edge_table[f"{endpoint}_{dim}"] = edge_table[endpoint].map(
                anchor_index[dim]
            )
    edges = list(
        edge_table.loc[edge_table.displayed, ["source", "target", "weight"]].itertuples(
            index=False, name=None
        )
    )
    size_values = (
        (
            nodes.n1
            if node_size_mode == "group1_n"
            else nodes.n0
            if node_size_mode == "group0_n"
            else nodes.delta_abs
            if node_size_mode == "delta_abs"
            else pd.Series(1.0, index=nodes.index)
        )
        .fillna(0)
        .to_numpy(float)
    )
    size_max = (
        float(node_size_reference)
        if node_size_reference is not None
        else max(
            float(size_values.max()), 1.0 if node_size_mode != "delta_abs" else 1e-12
        )
    )
    sizes = node_size_scale * (
        0.45 + 0.55 * np.sqrt(np.maximum(size_values, 0) / size_max)
    )
    if fill_color_mode == "alignment":
        fill_values = nodes.alignment_cosine.to_numpy(float).copy()
        available = (
            nodes.stored_alignment_available
            if mode == "stored"
            else nodes.usable_velocity
        )
        fill_values[~available.to_numpy()] = np.nan
        fill_kind = "alignment"
    elif fill_color_mode == "delta_frac":
        fill_values, fill_kind = nodes.delta_frac.to_numpy(float), "delta"
    else:
        fill_values, fill_kind = np.full(len(nodes), 0.5), "constant"
    fnorm = _normalization(fill_values, fill_norm, fill_vmin, fill_vmax, kind=fill_kind)
    arrow_values = nodes.shift_norm.where(
        nodes.usable_shift & nodes.shift_norm.gt(1e-12) & (arrow_scale > 0)
    ).to_numpy(float)
    anorm = _normalization(
        arrow_values,
        arrow_norm,
        arrow_vmin,
        arrow_vmax,
        kind="arrow",
    )
    fcmap, acmap = plt.get_cmap(fill_cmap), plt.get_cmap(arrow_cmap)
    offsets = {} if label_offsets is None else label_offsets
    if not isinstance(offsets, Mapping) or not set(map(str, offsets)) <= set(names):
        raise ValueError(
            "label_offsets must map known state IDs to two finite point offsets."
        )
    offsets = {str(k): tuple(v) for k, v in offsets.items()}
    if any(len(v) != 2 or not np.isfinite(v).all() for v in offsets.values()):
        raise ValueError("label_offsets require two finite point offsets.")
    if label_names is not None and (
        not isinstance(label_names, Mapping)
        or not set(map(str, label_names)) <= set(names)
    ):
        raise ValueError("label_names must map known state IDs to display labels.")
    label_names = (
        None
        if label_names is None
        else {str(k): str(v) for k, v in label_names.items()}
    )
    eligible = np.flatnonzero(cells.finite_coordinates)
    strata = (
        cells.loc[eligible, ["state", "condition"]]
        .fillna("(missing)")
        .astype(str)
        .agg(" / ".join, axis=1)
        .to_numpy()
    )
    drawn = eligible[display_indices(strata, max_points, random_state)]
    cells["background_displayed"] = False
    cells.loc[drawn, "background_displayed"] = True
    if caxes is not None:
        if not isinstance(caxes, Mapping):
            raise TypeError("caxes must map fill/arrow to colorbar axes.")
        active = ({"fill"} if fill_color_mode != "constant" else set()) | (
            {"arrow"} if arrow_color_mode == "shift" else set()
        )
        map_ax = (
            ax
            if ax is not None
            else axes.get("map")
            if isinstance(axes, Mapping)
            else None
        )
        if map_ax is None or not show_colorbar or not set(caxes) <= active:
            raise ValueError(
                "caxes requires supplied map axes and active color encodings with show_colorbar=True."
            )
        panel_axes(["map", *caxes], {"map": map_ax, **caxes}, None)
    fig, panels, owns = _canvas(layout, ax, axes, figsize)
    ax = panels["map"]
    if owns and layout == "map":
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("constrained")
        else:
            fig.set_constrained_layout(True)
    artists = {
        "edges": [],
        "nodes": {},
        "arrows": {},
        "labels": {},
        "reference_nodes": {},
        "comparison_nodes": {},
    }
    artists["background"] = ax.scatter(
        xy[drawn, 0],
        xy[drawn, 1],
        s=bg_size,
        c="lightgrey",
        alpha=bg_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    for row in edge_table.loc[edge_table.displayed].itertuples(index=False):
        artists["edges"].extend(
            ax.plot(
                [row.source_x, row.target_x],
                [row.source_y, row.target_y],
                color=".5",
                lw=edge_lw * max(row.weight, 0.15),
                alpha=edge_alpha,
                zorder=1,
            )
        )
    glyphs, obstacles, label_anchors = [], [], {}
    for i, row in nodes.iterrows():
        state = str(row.state)
        anchor = anchor_index.loc[state]
        ring_available = ring_mode == "ood_frac" and np.isfinite(row.ood_frac)
        ring_lw = 0.8 + ring_max_lw * row.ood_frac if ring_available else 0.8
        face = (
            fcmap(fnorm(fill_values[i])) if np.isfinite(fill_values[i]) else "lightgrey"
        )
        drawable = bool(anchor.observed)
        arrow_valid = bool(
            row.usable_shift
            and np.isfinite(row.shift_norm)
            and row.shift_norm > 1e-12
            and arrow_scale > 0
        )
        arrow_reason = (
            "displayed"
            if arrow_valid
            else "zero_scale"
            if arrow_scale == 0
            else "zero_shift"
            if row.usable_shift and row.shift_norm <= 1e-12
            else row.coverage_status
        )
        if drawable:
            marker = "o" if anchor.usable else "x"
            kwargs = (
                {
                    "facecolor": face,
                    "edgecolor": ring_color if ring_available else ".55",
                }
                if marker == "o"
                else {"color": ".5"}
            )
            artists["nodes"][state] = ax.scatter(
                anchor.x,
                anchor.y,
                s=sizes[i],
                marker=marker,
                linewidths=ring_lw,
                zorder=4,
                **kwargs,
            )
            label_anchors[state] = (anchor.x, anchor.y)
            obstacles.append((anchor.x, anchor.y, np.sqrt(sizes[i]) / 2 + 3))
        if show_endpoints:
            if node_position == "pooled" and row.n0:
                artists["reference_nodes"][state] = ax.scatter(
                    row.x0,
                    row.y0,
                    s=28,
                    facecolor="white",
                    edgecolor=".3",
                    linewidths=1,
                    zorder=5,
                )
            if row.n1:
                artists["comparison_nodes"][state] = ax.scatter(
                    row.x1,
                    row.y1,
                    s=24,
                    marker="o" if row.present1 else "x",
                    color=".3",
                    zorder=5,
                )
                if not drawable:
                    label_anchors[state] = (row.x1, row.y1)
        if arrow_valid:
            color = (
                acmap(anorm(row.shift_norm))
                if arrow_color_mode == "shift"
                else arrow_color
            )
            artists["arrows"][state] = _draw_vector(
                ax,
                (row.x0, row.y0),
                (row.x0 + row.dx * arrow_scale, row.y0 + row.dy * arrow_scale),
                color=color,
                alpha=0.95,
                width=arrow_width,
                linewidth=arrow_linewidth,
                head_size=arrow_head_size,
            )
        glyphs.append(
            {
                "node": state,
                "x": anchor.x,
                "y": anchor.y,
                "displayed": drawable,
                "position_source": node_position,
                "size": sizes[i],
                "size_value": size_values[i],
                "fill_value": fill_values[i],
                "fill_available": np.isfinite(fill_values[i]),
                "ring_lw": ring_lw,
                "ring_available": ring_available,
                "arrow_displayed": arrow_valid,
                "arrow_reason": arrow_reason,
                "arrow_source_x": row.x0,
                "arrow_source_y": row.y0,
                "arrow_end_x": row.x0 + row.dx * arrow_scale if arrow_valid else np.nan,
                "arrow_end_y": row.y0 + row.dy * arrow_scale if arrow_valid else np.nan,
                "comparison_x": row.x1,
                "comparison_y": row.y1,
                "label_displayed": False,
            }
        )
    glyphs = pd.DataFrame(glyphs)
    if show_colorbar:
        if fill_color_mode != "constant":
            bar = fig.colorbar(
                plt.cm.ScalarMappable(norm=fnorm, cmap=fcmap),
                ax=ax,
                cax=None if caxes is None else caxes.get("fill"),
                fraction=0.035,
                pad=0.025,
            )
            bar.set_label(
                (
                    "Stored alignment cosine"
                    if mode == "stored"
                    else "Display-space cosine"
                )
                if fill_color_mode == "alignment"
                else "Captured-cell fraction difference",
                fontsize=8,
            )
            artists["fill_colorbar"] = bar
        if arrow_color_mode == "shift":
            bar = fig.colorbar(
                plt.cm.ScalarMappable(norm=anorm, cmap=acmap),
                ax=ax,
                cax=None if caxes is None else caxes.get("arrow"),
                fraction=0.035,
                pad=0.07,
            )
            bar.set_label("Display-space shift magnitude", fontsize=8)
            artists["arrow_colorbar"] = bar
    if legend:
        handles = [
            Line2D(
                [],
                [],
                color=arrow_color,
                marker=">",
                lw=arrow_linewidth,
                label=f"Display contrast ×{arrow_scale:g}",
            )
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="lightgrey",
                markeredgecolor=".5",
                label="Evidence unavailable",
            )
        )
        if node_size_mode != "constant":
            for value in np.unique(np.quantile(size_values, [0, 0.5, 1])):
                diameter = np.sqrt(
                    node_size_scale * (0.45 + 0.55 * np.sqrt(max(value, 0) / size_max))
                )
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markerfacecolor="none",
                        markeredgecolor=".4",
                        markersize=diameter,
                        label=f"{node_size_mode}: {value:.3g}",
                    )
                )
        if ring_mode == "ood_frac" and ood_key is not None:
            for fraction in (0.0, 0.5, 1.0):
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markerfacecolor="white",
                        markeredgecolor=ring_color,
                        markeredgewidth=0.8 + ring_max_lw * fraction,
                        markersize=8,
                        label=f"OOD among scored: {fraction:.0%}",
                    )
                )
        largest_marker = max(handle.get_markersize() for handle in handles)
        artists["legend"] = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0, -0.18),
            frameon=False,
            fontsize=7,
            ncol=2,
            handleheight=max(0.7, (largest_marker / 7 - 0.245) / 0.65),
            labelspacing=0.65,
        )
    finite_xy = xy[np.isfinite(xy).all(axis=1)]
    endpoints = glyphs.loc[
        glyphs.arrow_displayed, ["arrow_end_x", "arrow_end_y"]
    ].to_numpy(float)
    bounds = np.vstack([finite_xy, endpoints])
    if len(bounds):
        for dimension, method in enumerate((ax.set_xlim, ax.set_ylim)):
            if not owns and not (
                ax.get_autoscalex_on() if dimension == 0 else ax.get_autoscaley_on()
            ):
                continue
            low, high = bounds[:, dimension].min(), bounds[:, dimension].max()
            pad = max((high - low) * 0.08, 0.12)
            method(low - pad, high + pad)
    ax.set(
        xlabel=f"{embedding_key.removeprefix('X_').upper()}1",
        ylabel=f"{embedding_key.removeprefix('X_').upper()}2",
        xticks=[],
        yticks=[],
        title=title
        or f"ScGeo compass · {groups[0]} → {groups[1]}\n{'Stored evidence; display-only arrows' if mode == 'stored' else 'Descriptive display-space contrast'}",
    )
    ax.set_aspect("equal", adjustable="box" if owns else "datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    summary_states = (
        evidence.sort_values(
            ["effect_available", "effect"],
            ascending=[False, False],
            kind="stable",
            na_position="last",
        )
        .state.head(max_states)
        .tolist()
    )
    evidence["summary_displayed"] = evidence.state.isin(summary_states) & (
        layout == "summary"
    )
    if layout == "summary":
        artists["summary"] = _summary_panels(
            panels,
            evidence,
            sample_counts,
            coverage,
            summary_states,
            groups,
            category_palette(groups),
            mode,
            metric,
            sample_key,
        )
    ranking = nodes.sort_values(
        "shift_norm", ascending=False, kind="stable", na_position="last"
    )
    chosen = [state for state in ranking.state if state in label_anchors]
    if label_top_n is not None:
        chosen = chosen[:label_top_n]
    rows = [(state, *label_anchors[state]) for state in chosen] if label else []
    artists["labels"], label_table = place_labels(
        ax,
        rows,
        fontsize=label_fontsize,
        mode=label_mode,
        offsets=offsets,
        obstacles=obstacles,
        labels=label_names,
    )
    glyphs["label_displayed"] = glyphs.node.isin(artists["labels"])
    nodes["displayed"] = (
        nodes.state.map(anchor_index.observed).fillna(False).astype(bool)
    )
    data = {
        "cells": cells,
        "nodes": nodes,
        "edges": edge_table,
        "graph_nodes": pd.DataFrame(
            {"node": names, "matrix_position": np.arange(len(names))}
        ),
        "glyphs": glyphs,
        "node_anchors": anchors,
        "coverage": coverage,
        "sample_abundance": sample_counts,
        "evidence": evidence,
        "labels": label_table,
    }
    if source_shift is not None:
        data["shift_source"] = source_shift.drop(columns="_state_id")
    if source_alignment is not None:
        data["alignment_source"] = source_alignment.drop(columns="_state_id")
    metadata = {
        "plot": "recovery_compass",
        "mode": mode,
        "layout": layout,
        "basis": embedding_key,
        "node_key": node_key,
        "condition_key": condition_key,
        "group0": groups[0],
        "group1": groups[1],
        "alignment_source": "stored" if mode == "stored" else "display_coordinates",
        "analysis_rep": analysis_rep,
        "effect_metric": metric,
        "stored_provenance": provenance,
        "centroid_aggregation": "mean",
        "centroid_unit": "cells",
        "node_position": node_position,
        "velocity_key": velocity_key,
        "velocity_scope": "all_input_conditions"
        if velocity_scope == "all"
        else velocity_scope,
        "velocity_aggregation": "jointly finite cell mean",
        "ood_key": ood_key,
        "ood_threshold": resolved_threshold,
        "ood_threshold_source": "explicit"
        if ood_threshold is not None
        else "scope_quantile_0.95",
        "ood_scope": ood_scope,
        "ood_threshold_scope": "finite_scores_all_input_cells"
        if ood_scope == "all"
        else f"finite_scores_{ood_scope}",
        "ood_fraction_denominator": "finite scored node cells in the selected OOD scope",
        "sample_key": sample_key,
        "abundance_denominator": "annotated cells per sample-condition unit",
        "sample_roster_scope": "all supplied selected-condition observations before state or display filtering",
        "paga_key": paga_key,
        "graph_order_source": order_source,
        "graph_node_order": names,
        "connectivity_threshold": connectivity_threshold,
        "graph_directed": False,
        "min_cells": min_cells,
        "node_size_mode": node_size_mode,
        "node_size_scale": node_size_scale,
        "node_size_reference": size_max,
        "fill_color_mode": fill_color_mode,
        "fill_norm": {
            "type": type(fnorm).__name__,
            "vmin": fnorm.vmin,
            "vmax": fnorm.vmax,
        },
        "fill_cmap": fill_cmap,
        "ring_mode": ring_mode,
        "arrow_color_mode": arrow_color_mode,
        "arrow_scale": arrow_scale,
        "arrow_cmap": arrow_cmap,
        "arrow_width_units": "points" if arrow_width is None else "data",
        "arrow_norm": {
            "type": type(anorm).__name__,
            "vmin": anorm.vmin,
            "vmax": anorm.vmax,
        },
        "arrow_source": "actual reference condition center; independent of node position",
        "label_mode": label_mode,
        "label_names": label_names,
        "max_points": max_points,
        "random_state": random_state,
        "n_background_displayed": len(drawn),
        "summary_states": summary_states if layout == "summary" else [],
        "show_colorbar": bool(show_colorbar),
        "owns_figure": owns,
        "export_scope": "created_figure" if owns else "parent_figure",
        "interpretation": "Stored evidence and observed display context remain separate; no recovery, causality or trajectories inferred.",
    }
    result_axes = {"compass": ax, **panels}
    result = PlotResult(fig, result_axes, data, metadata, notes, artists)
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return (
            fig,
            ax if layout == "map" else panels,
            nodes.loc[nodes.n0.add(nodes.n1).gt(0)].copy(),
            edges,
        )
    if layout == "summary":
        return fig, panels
    return (fig, ax) if owns else ax
