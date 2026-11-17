"""Bipartite chord views of same-cell categorical associations."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Wedge
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


def _chord_layout(prepared, category_gap, stage_gap, node_width):
    maximum_categories = max(len(ids) for ids in prepared["stage_ids"].values())
    mass_angle = 180.0 - stage_gap - category_gap * (maximum_categories - 1)
    if mass_angle <= 0:
        raise ValueError(
            "category_gap and stage_gap leave no angular space for categories; reduce the gaps."
        )
    records = []
    for stage_index, (stage, ids) in enumerate(prepared["stage_ids"].items()):
        direction = 1.0 if stage_index == 0 else -1.0
        unused = category_gap * (maximum_categories - len(ids))
        angle = 90 + direction * (stage_gap / 2 + unused / 2)
        lookup = prepared["nodes"].set_index("node_id")
        for node_id in ids:
            row = lookup.loc[node_id]
            end = angle + direction * row.fraction * mass_angle
            item = row.to_dict()
            item.update(
                {
                    "node_id": node_id,
                    "theta_start": angle,
                    "theta_end": end,
                    "theta_low": min(angle, end),
                    "theta_high": max(angle, end),
                    "angle_width": abs(end - angle),
                    "radius": 1.0,
                    "inner_radius": 1.0 - node_width,
                    "value": row.fraction,
                    "displayed": True,
                    "reason": "displayed",
                }
            )
            records.append(item)
            angle = end + direction * category_gap
    nodes = pd.DataFrame(records)
    lookup = nodes.set_index("node_id")
    links = prepared["links"].copy(deep=True)
    source_slots, target_slots = {}, {}
    ranks = lookup.category_index.to_dict()
    for node_id, edges in links.groupby("source_id", sort=False):
        angle = lookup.loc[node_id, "theta_start"]
        edges = edges.assign(order=edges.target_id.map(ranks)).sort_values(
            "order", kind="stable"
        )
        for row in edges.itertuples(index=False):
            end = angle + row.fraction * mass_angle
            source_slots[row.link_id] = (angle, end)
            angle = end
    for node_id, edges in links.groupby("target_id", sort=False):
        angle = lookup.loc[node_id, "theta_start"]
        edges = edges.assign(order=edges.source_id.map(ranks)).sort_values(
            "order", kind="stable"
        )
        for row in edges.itertuples(index=False):
            end = angle - row.fraction * mass_angle
            target_slots[row.link_id] = (angle, end)
            angle = end
    links["source_theta_start"] = [source_slots[index][0] for index in links.link_id]
    links["source_theta_end"] = [source_slots[index][1] for index in links.link_id]
    links["target_theta_start"] = [target_slots[index][0] for index in links.link_id]
    links["target_theta_end"] = [target_slots[index][1] for index in links.link_id]
    links["angle_width"] = links.fraction * mass_angle
    links["inner_radius"] = 1.0 - node_width
    links["value"] = links.fraction
    return nodes, links, mass_angle


def _point(angle, radius):
    theta = np.deg2rad(angle)
    return radius * np.array([np.cos(theta), np.sin(theta)])


def _chord_path(source_start, source_end, target_start, target_end, radius):
    source_arc = Path.arc(source_start, source_end)
    target_arc = Path.arc(target_end, target_start)
    vertices = (source_arc.vertices * radius).tolist()
    codes = source_arc.codes.tolist()
    source_low, target_low = _point(source_end, radius), _point(target_end, radius)
    vertices.extend([source_low * 0.18, target_low * 0.18, target_low])
    codes.extend([Path.CURVE4] * 3)
    vertices.extend((target_arc.vertices[1:] * radius).tolist())
    codes.extend(target_arc.codes[1:].tolist())
    target_high, source_high = (
        _point(target_start, radius),
        _point(source_start, radius),
    )
    vertices.extend([target_high * 0.18, source_high * 0.18, source_high, source_high])
    codes.extend([Path.CURVE4] * 3 + [Path.CLOSEPOLY])
    return Path(np.asarray(vertices), codes)


def state_flow_chord(
    adata,
    *,
    columns,
    min_count=1,
    drop_na=False,
    normalize=False,
    category_order=None,
    order="input",
    sort_categories=False,
    color_by="target",
    palette=None,
    alpha=0.65,
    category_gap=2.0,
    stage_gap=18.0,
    node_width=0.09,
    label_format=None,
    label_fontsize=9,
    show_omissions=True,
    ax=None,
    figsize=(8, 7),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Draw a two-stage bipartite chord of same-cell category associations.

    ``columns`` must contain exactly two categorical annotations. The left half
    is the source annotation and the right half is its target annotation on the
    same cells. Repeated category names remain distinct stage-specific nodes.
    Ribbons encode counts of co-assigned labels, not lineage, a transition rate,
    or elapsed time. There are no inferred connections between different cells.

    Each half has a common angular scale for the full included N cells. Full
    category totals and every positive link receive angular slots before
    ``min_count`` hides small links, leaving honest gaps instead of renormalizing
    retained mass. Category fractions and ribbon fractions use N, not 2N, even
    though each cell contributes to one arc on each half. ``category_gap`` and
    ``stage_gap`` are degrees; ``node_width`` is a fraction of the unit radius.

    ``category_order`` maps stage names to complete observed orders; None denotes
    genuine missing labels. Input order respects categorical levels, otherwise
    first appearance. Count and deterministic barycentric ordering are optional.
    ``drop_na`` drops genuine missing selected annotations only; literal 'NA'
    remains a real category. A flat palette shares label colors across stages;
    nested stage palettes can give distinct colors to equal labels.

    Geometry uses full-denominator fractions in both ``normalize`` modes.
    Normalize selects percentage labels by default; False selects counts.
    ``label_format`` explicitly accepts label/count/percent/both. Supplied axes
    retain figure size and fixed limits. Default returns are (fig,ax) for owned
    figures and ax for supplied axes. ``return_data`` returns
    (fig,ax,node_layout,link_layout), including hidden links. ``return_result``
    returns a PlotResult with the named ``chord`` axis, all source cells and
    counts, angular geometry, label positions, and interpretation metadata.
    """
    if isinstance(columns, str):
        raise TypeError("columns must contain exactly two distinct annotations.")
    columns = list(columns)
    if len(columns) != 2:
        raise ValueError("columns must contain exactly two distinct annotations.")
    alpha, label_fontsize = _validate_flow_style(
        color_by=color_by,
        alpha=alpha,
        label_format=label_format,
        label_fontsize=label_fontsize,
        normalize=normalize,
    )
    category_gap = _number(category_gap, "category_gap")
    stage_gap = _number(stage_gap, "stage_gap", maximum=179)
    node_width = _number(node_width, "node_width", strict=True, maximum=0.95)
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
    nodes, links, mass_angle = _chord_layout(
        prepared, category_gap, stage_gap, node_width
    )
    fig, ax, owns_figure, fixed_x, fixed_y = _figure_axis(ax, figsize)
    if owns_figure:
        fig.subplots_adjust(left=0.07, right=0.93, top=0.83, bottom=0.13)
    artists = {"ribbons": {}, "nodes": {}, "stage_labels": {}}
    for row in links.itertuples(index=False):
        if not row.displayed:
            continue
        color = colors[row.source_id if color_by == "source" else row.target_id]
        patch = PathPatch(
            _chord_path(
                row.source_theta_start,
                row.source_theta_end,
                row.target_theta_start,
                row.target_theta_end,
                row.inner_radius,
            ),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)
        artists["ribbons"][row.link_id] = patch
    specifications = []
    for _, row in nodes.iterrows():
        patch = Wedge(
            (0, 0),
            row.radius,
            row.theta_low,
            row.theta_high,
            width=node_width,
            facecolor=colors[row.node_id],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.add_patch(patch)
        artists["nodes"][row.node_id] = patch
        mid = (row.theta_start + row.theta_end) / 2
        anchor = _point(mid, 1.015)
        left = row.stage_index == 0
        specifications.append(
            {
                "node_id": row.node_id,
                "stage": row.stage,
                "label": row.label,
                "text": _label_text(row, label_format, normalize, width=20),
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "text_x": -1.18 if left else 1.18,
                "ha": "right" if left else "left",
            }
        )
    ax.set_xlim(fixed_x if fixed_x is not None else (-1.95, 1.95))
    ax.set_ylim(fixed_y if fixed_y is not None else (-1.17, 1.17))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for index, stage in enumerate(columns):
        artists["stage_labels"][stage] = ax.text(
            0.25 if index == 0 else 0.75,
            1.035,
            textwrap.fill(stage, 24),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#273B49",
        )
    ax.set_title(title or "Same-cell category associations", fontsize=13, pad=45)
    artists["labels"], labels, label_notes = _labels_on_axes(
        ax, specifications, label_fontsize
    )
    annotation = f"{prepared['metadata']['n_cells_included']:,} cells · source annotation (left) → target annotation (right)"
    omission = _omission_text(prepared)
    if show_omissions and omission:
        annotation += "\n" + omission
    artists["annotation"] = ax.text(
        0.5,
        -0.065,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#617784",
    )
    data = {
        "cells": prepared["cells"],
        "nodes": prepared["nodes"],
        "links": prepared["links"],
        "node_layout": nodes,
        "link_layout": links,
        "labels": labels,
    }
    metadata = {
        **prepared["metadata"],
        "plot": "state_flow_chord",
        "normalize": bool(normalize),
        "normalize_semantics": "fraction geometry in both modes; chooses default labels only",
        "layout": "bipartite; separate source/target semicircles",
        "geometry_units": "degrees",
        "mass_angle_per_stage": mass_angle,
        "angle_per_cell": mass_angle / prepared["metadata"]["n_cells_included"],
        "category_gap": category_gap,
        "stage_gap": stage_gap,
        "node_width": node_width,
        "ribbon_control_radius_factor": 0.18,
        "radius": 1.0,
        "color_by": color_by,
        "alpha": alpha,
        "node_palette": colors,
        "label_format": label_format or ("percent" if normalize else "count"),
        "label_fontsize": label_fontsize,
        "show_omissions": show_omissions,
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
    }
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes={"chord": ax},
            data=data,
            metadata=metadata,
            warnings=prepared["warnings"] + label_notes,
            artists=artists,
        )
    if return_data:
        return fig, ax, nodes, links
    return (fig, ax) if owns_figure else ax
