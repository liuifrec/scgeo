"""Count-conserving alluvial views of categorical cell associations."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from ._result import PlotResult
from ._state_flow import (
    _figure_axis,
    _flow_palette,
    _label_text,
    _labels_on_axes,
    _number,
    _omission_text,
    _prepare_state_flow,
    _validate_flow_style,
)


def _make_ribbon(x0, x1, y0_low, y0_high, y1_low, y1_high, curve=0.35):
    dx = x1 - x0
    c0, c1 = x0 + dx * curve, x1 - dx * curve
    return Path(
        [
            (x0, y0_high),
            (c0, y0_high),
            (c1, y1_high),
            (x1, y1_high),
            (x1, y1_low),
            (c1, y1_low),
            (c0, y0_low),
            (x0, y0_low),
            (x0, y0_high),
        ],
        [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ],
    )


def _alluvial_layout(prepared, column_gap, category_gap, bar_width):
    records = []
    for stage, nodes in prepared["nodes"].groupby("stage", sort=False):
        height = 1.0 + category_gap * max(0, len(nodes) - 1)
        top = 0.5 + height / 2
        for _, row in nodes.iterrows():
            item = row.to_dict()
            item.update(
                {
                    "x": row.stage_index * column_gap,
                    "y0": top - row.fraction,
                    "y1": top,
                    "value": row.fraction,
                    "displayed": True,
                    "reason": "displayed",
                }
            )
            records.append(item)
            top -= row.fraction + category_gap
    boxes = pd.DataFrame(records)
    lookup = boxes.set_index("node_id")
    links = prepared["links"].copy(deep=True)
    source_slots, target_slots = {}, {}
    rank = prepared["nodes"].set_index("node_id").category_index.to_dict()
    for _, group in links.groupby("pair_index", sort=False):
        for node_id, edges in group.groupby("source_id", sort=False):
            top = float(lookup.loc[node_id, "y1"])
            edges = edges.assign(order=edges.target_id.map(rank)).sort_values(
                "order", kind="stable"
            )
            for row in edges.itertuples(index=False):
                source_slots[row.link_id] = (top - row.fraction, top)
                top -= row.fraction
        for node_id, edges in group.groupby("target_id", sort=False):
            top = float(lookup.loc[node_id, "y1"])
            edges = edges.assign(order=edges.source_id.map(rank)).sort_values(
                "order", kind="stable"
            )
            for row in edges.itertuples(index=False):
                target_slots[row.link_id] = (top - row.fraction, top)
                top -= row.fraction
    links["value"] = links.fraction
    links["x0"] = links.source_id.map(lookup.x) + bar_width / 2
    links["x1"] = links.target_id.map(lookup.x) - bar_width / 2
    links["y0_low"] = [source_slots[index][0] for index in links.link_id]
    links["y0_high"] = [source_slots[index][1] for index in links.link_id]
    links["y1_low"] = [target_slots[index][0] for index in links.link_id]
    links["y1_high"] = [target_slots[index][1] for index in links.link_id]
    links["left_stage"], links["right_stage"] = links.source_stage, links.target_stage
    links["left_label"], links["right_label"] = (
        links.source_display_label,
        links.target_display_label,
    )
    return boxes, links


def state_flow_alluvial(
    adata,
    *,
    columns: Sequence[str],
    min_count: int = 1,
    drop_na: bool = False,
    normalize: bool = False,
    sort_categories: bool = False,
    color_by: str = "target",
    alpha: float = 0.7,
    column_gap: float = 1.8,
    category_gap: float = 0.02,
    ribbon_curve: float = 0.35,
    figsize: tuple[float, float] = (11, 6),
    title: Optional[str] = None,
    palette=None,
    ax=None,
    return_data: bool = False,
    show: bool = True,
    category_order=None,
    order="input",
    label_format=None,
    label_fontsize=9,
    bar_width=0.08,
    show_omissions=True,
    return_result=False,
):
    """Draw count-conserving associations between categorical cell annotations.

    Each row contributes one count to each adjacent pair in ``columns``. These
    links connect annotations on the same cells; they do not establish lineage,
    transitions between different cells, or elapsed time.

    All positive counts and their layout slots are retained before ``min_count``
    hides ribbons. Filtered mass remains blank, and category totals stay complete.
    Both legacy ``normalize`` settings retain fraction-based geometry and
    ``value=count/N``, where N is the number of included cells before filtering.
    ``normalize=True`` changes the default labels to percentages; False shows
    counts. ``label_format`` can explicitly choose label/count/percent/both.
    Raw counts and full-denominator fractions are always exported separately.

    ``drop_na`` removes rows with genuine missing selected annotations; literal
    labels such as 'NA', 'nan', and '(missing)' remain distinct categories.
    Otherwise missing observations receive their own explicitly marked category.
    Input ordering respects declared categorical levels, then first appearance
    for other columns. ``category_order`` maps stage names to complete observed
    orders (use None for missing); ``order='count'`` or ``'barycentric'`` provides
    deterministic alternative layouts. Explicit orders remain fixed.

    ``palette`` may map labels to colors or stage names to complete label-color
    mappings. Labels sit outside the bars with point-space spacing and leaders.
    Supplied axes retain figure size and already fixed limits. ``return_result``
    returns a PlotResult with the named ``alluvial`` axis and complete cells,
    nodes, links, layouts, label positions, and provenance. Existing returns are
    preserved: owned (fig,ax), supplied ax, or return_data=(fig,ax,stage_boxes,
    displayed_allocations). The result object's layout includes hidden links.
    """
    alpha, label_fontsize = _validate_flow_style(
        color_by=color_by,
        alpha=alpha,
        label_format=label_format,
        label_fontsize=label_fontsize,
        normalize=normalize,
    )
    column_gap = _number(column_gap, "column_gap", strict=True)
    category_gap = _number(category_gap, "category_gap")
    ribbon_curve = _number(ribbon_curve, "ribbon_curve", maximum=1)
    bar_width = _number(bar_width, "bar_width", strict=True)
    if bar_width >= column_gap:
        raise ValueError("bar_width must be smaller than column_gap.")
    prepared = _prepare_state_flow(
        adata,
        columns=columns,
        min_count=min_count,
        drop_na=drop_na,
        category_order=category_order,
        order=order,
        sort_categories=sort_categories,
    )
    colors = _flow_palette(prepared["nodes"], palette)
    boxes, allocations = _alluvial_layout(prepared, column_gap, category_gap, bar_width)
    fig, ax, owns_figure, fixed_x, fixed_y = _figure_axis(ax, figsize)
    if owns_figure:
        fig.subplots_adjust(left=0.17, right=0.83, top=0.84, bottom=0.12)
    artists = {"ribbons": {}, "nodes": {}, "stage_labels": {}}
    for row in allocations.itertuples(index=False):
        if not row.displayed:
            continue
        color = colors[row.source_id if color_by == "source" else row.target_id]
        patch = PathPatch(
            _make_ribbon(
                row.x0,
                row.x1,
                row.y0_low,
                row.y0_high,
                row.y1_low,
                row.y1_high,
                ribbon_curve,
            ),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)
        artists["ribbons"][row.link_id] = patch
    specifications = []
    last_stage = len(prepared["metadata"]["columns"]) - 1
    for _, row in boxes.iterrows():
        patch = Rectangle(
            (row.x - bar_width / 2, row.y0),
            bar_width,
            row.y1 - row.y0,
            facecolor=colors[row.node_id],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.add_patch(patch)
        artists["nodes"][row.node_id] = patch
        left = row.stage_index == 0
        specifications.append(
            {
                "node_id": row.node_id,
                "stage": row.stage,
                "label": row.label,
                "text": _label_text(row, label_format, normalize, width=24),
                "anchor_x": row.x - bar_width / 2 if left else row.x + bar_width / 2,
                "anchor_y": (row.y0 + row.y1) / 2,
                "offset_x_points": -8 if left else 8,
                "ha": "right" if left else "left",
            }
        )
    bottom, top = float(boxes.y0.min()), float(boxes.y1.max())
    height = top - bottom
    for stage_index, column in enumerate(prepared["metadata"]["columns"]):
        artists["stage_labels"][column] = ax.text(
            stage_index * column_gap,
            top + height * 0.065,
            textwrap.fill(column, 22),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#273B49",
        )
    ax.set_xlim(
        fixed_x
        if fixed_x is not None
        else (-column_gap * 0.08, last_stage * column_gap + column_gap * 0.08)
    )
    ax.set_ylim(
        fixed_y
        if fixed_y is not None
        else (bottom - height * 0.04, top + height * 0.13)
    )
    ax.axis("off")
    ax.set_title(title or "Categorical cell associations", fontsize=13, pad=25)
    artists["labels"], labels, label_notes = _labels_on_axes(
        ax, specifications, label_fontsize
    )
    omission = _omission_text(prepared)
    if show_omissions and omission:
        artists["omissions"] = ax.text(
            0.5,
            -0.075,
            omission,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#617784",
        )
    metadata = {
        **prepared["metadata"],
        "plot": "state_flow_alluvial",
        "normalize": bool(normalize),
        "normalize_semantics": "fraction geometry in both modes; chooses default labels only",
        "geometry_units": "fraction of all included cells",
        "category_gap": category_gap,
        "column_gap": column_gap,
        "bar_width": bar_width,
        "ribbon_curve": ribbon_curve,
        "color_by": color_by,
        "alpha": alpha,
        "node_palette": colors,
        "label_format": label_format or ("percent" if normalize else "count"),
        "label_fontsize": label_fontsize,
        "show_omissions": show_omissions,
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    data = {
        "cells": prepared["cells"],
        "nodes": prepared["nodes"],
        "links": prepared["links"],
        "node_layout": boxes,
        "link_layout": allocations,
        "labels": labels,
    }
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes={"alluvial": ax},
            data=data,
            metadata=metadata,
            warnings=prepared["warnings"] + label_notes,
            artists=artists,
        )
    if return_data:
        return fig, ax, boxes, allocations.loc[allocations.displayed].copy()
    return (fig, ax) if owns_figure else ax
