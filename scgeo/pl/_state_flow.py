"""Shared categorical association counts, ordering, and display helpers.

A row is one cell with several annotations. Adjacent-stage links count jointly
observed labels on those same cells; they are not longitudinal trajectories.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.font_manager import FontProperties

from ._style import category_palette


def _missing(value):
    result = pd.isna(value)
    return bool(result) if np.ndim(result) == 0 else False


def _token(value):
    return None if _missing(value) else str(value)


def _number(value, name, *, minimum=0.0, strict=False, maximum=None):
    if isinstance(value, bool) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if (
        not np.isfinite(result)
        or (result <= minimum if strict else result < minimum)
        or (maximum is not None and result > maximum)
    ):
        sign = ">" if strict else ">="
        raise ValueError(
            f"{name} must be finite and {sign} {minimum}"
            + (f" and <= {maximum}." if maximum is not None else ".")
        )
    return result


def _prepare_state_flow(
    adata,
    *,
    columns,
    min_count=1,
    drop_na=False,
    category_order=None,
    order="input",
    sort_categories=False,
):
    if isinstance(columns, str):
        raise TypeError(
            "columns must contain at least 2 distinct obs columns, not one string."
        )
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("columns must contain at least 2 obs columns.")
    if any(not isinstance(column, str) for column in columns) or len(
        set(columns)
    ) != len(columns):
        raise ValueError("columns must contain distinct string column names.")
    for column in columns:
        if column not in adata.obs:
            raise KeyError(f"'{column}' not found in adata.obs.")
    if (
        isinstance(min_count, bool)
        or not isinstance(min_count, (int, np.integer))
        or min_count < 1
    ):
        raise ValueError("min_count must be a positive integer.")
    if not isinstance(drop_na, (bool, np.bool_)):
        raise TypeError("drop_na must be boolean.")
    if order not in {"input", "count", "barycentric"}:
        raise ValueError("order must be 'input', 'count', or 'barycentric'.")
    if not isinstance(sort_categories, (bool, np.bool_)):
        raise TypeError("sort_categories must be boolean.")
    if sort_categories and order != "input":
        raise ValueError("sort_categories cannot be combined with a non-input order.")
    if category_order is not None and not isinstance(category_order, Mapping):
        raise TypeError(
            "category_order must map stage names to complete category orders."
        )
    category_order = dict(category_order or {})
    if not set(category_order) <= set(columns):
        raise ValueError("category_order contains unknown stage names.")
    values = {}
    for column in columns:
        raw = pd.unique(adata.obs[column].dropna().astype(object))
        if len({str(value) for value in raw}) != len(raw):
            raise ValueError(
                f"{column!r} has distinct values with identical string labels."
            )
        values[column] = [_token(value) for value in adata.obs[column]]
    missing = np.column_stack(
        [[value is None for value in values[column]] for column in columns]
    )
    included = ~missing.any(axis=1) if drop_na else np.ones(adata.n_obs, dtype=bool)
    total = int(included.sum())
    if total == 0:
        raise ValueError("No cells remain after the missing-data policy.")
    cells = pd.DataFrame(
        {
            "obs_position": np.arange(adata.n_obs),
            "cell": adata.obs_names.astype(str),
            "included": included,
        }
    )
    cells["drop_reason"] = [
        "included"
        if keep
        else "missing: "
        + ", ".join(column for column, absent in zip(columns, mask) if absent)
        for keep, mask in zip(included, missing)
    ]
    nodes, stage_ids, token_ids, source_fields = [], {}, {}, {}
    next_id = 0
    for stage_index, column in enumerate(columns):
        counts_by_label = Counter(
            value for value, keep in zip(values[column], included) if keep
        )
        observed = list(counts_by_label)
        series = adata.obs[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            base = [
                _token(value)
                for value in series.cat.categories
                if _token(value) in observed
            ]
            base += [value for value in observed if value not in base]
        else:
            base = observed
        missing_label = "(missing)"
        while missing_label in {value for value in observed if value is not None}:
            missing_label = f"({missing_label})"
        ids = {}
        for label in base:
            count = counts_by_label[label]
            ids[label] = next_id
            nodes.append(
                {
                    "node_id": next_id,
                    "node_key": (stage_index, label),
                    "stage_index": stage_index,
                    "stage": column,
                    "label": label,
                    "display_label": missing_label if label is None else label,
                    "is_missing": label is None,
                    "count": int(count),
                    "fraction": count / total,
                }
            )
            next_id += 1
        token_ids[column] = ids
        stage_ids[column] = [ids[label] for label in base]
        fields = {
            "label": f"stage_{stage_index}_label",
            "missing": f"stage_{stage_index}_missing",
            "node_id": f"stage_{stage_index}_node_id",
        }
        source_fields[column] = fields
        cells[fields["label"]] = values[column]
        cells[fields["missing"]] = missing[:, stage_index]
        cells[fields["node_id"]] = pd.array(
            [ids.get(value, pd.NA) for value in values[column]], dtype="Int64"
        )
    nodes = pd.DataFrame(nodes)
    links = []
    for pair_index, (source_stage, target_stage) in enumerate(
        zip(columns[:-1], columns[1:])
    ):
        source = cells.loc[included, source_fields[source_stage]["node_id"]].to_numpy(
            dtype=int
        )
        target = cells.loc[included, source_fields[target_stage]["node_id"]].to_numpy(
            dtype=int
        )
        pairs, counts = np.unique(
            np.column_stack([source, target]), axis=0, return_counts=True
        )
        for (source_id, target_id), count in zip(pairs, counts):
            left, right = nodes.iloc[source_id], nodes.iloc[target_id]
            links.append(
                {
                    "link_id": len(links),
                    "pair_index": pair_index,
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "source_label": left.label,
                    "target_label": right.label,
                    "source_display_label": left.display_label,
                    "target_display_label": right.display_label,
                    "source_missing": bool(left.is_missing),
                    "target_missing": bool(right.is_missing),
                    "count": int(count),
                    "fraction": count / total,
                    "displayed": bool(count >= min_count),
                    "reason": "displayed" if count >= min_count else "below_min_count",
                }
            )
    links = pd.DataFrame(links)
    original_rank = {
        node: rank for ids in stage_ids.values() for rank, node in enumerate(ids)
    }
    for column in columns:
        if sort_categories:
            stage_ids[column].sort(
                key=lambda node: (
                    bool(nodes.iloc[node].is_missing),
                    nodes.iloc[node].display_label,
                )
            )
        elif order == "count":
            stage_ids[column].sort(
                key=lambda node: (-nodes.iloc[node]["count"], original_rank[node])
            )
        if column in category_order:
            requested = category_order[column]
            if isinstance(requested, str):
                raise ValueError(
                    "Each category_order entry must be a sequence of labels."
                )
            requested = [_token(value) for value in requested]
            if len(set(requested)) != len(requested) or set(requested) != set(
                token_ids[column]
            ):
                raise ValueError(
                    f"category_order for {column!r} must include every observed category exactly once (use None for missing)."
                )
            stage_ids[column] = [token_ids[column][label] for label in requested]
    if order == "barycentric":
        for _ in range(4):
            for indices in (range(1, len(columns)), range(len(columns) - 2, -1, -1)):
                for index in indices:
                    column = columns[index]
                    if column in category_order:
                        continue
                    from_left = isinstance(indices, range) and indices.step > 0
                    neighbor = columns[index - 1] if from_left else columns[index + 1]
                    ranks = {
                        node: rank for rank, node in enumerate(stage_ids[neighbor])
                    }
                    selected = (
                        links.loc[
                            links.target_stage.eq(column)
                            & links.source_stage.eq(neighbor)
                        ]
                        if from_left
                        else links.loc[
                            links.source_stage.eq(column)
                            & links.target_stage.eq(neighbor)
                        ]
                    )
                    node_column, adjacent_column = (
                        ("target_id", "source_id")
                        if from_left
                        else ("source_id", "target_id")
                    )
                    scores = {}
                    for node in stage_ids[column]:
                        edges = selected.loc[selected[node_column].eq(node)]
                        scores[node] = float(
                            np.average(
                                edges[adjacent_column].map(ranks),
                                weights=edges["count"],
                            )
                        )
                    previous = {
                        node: rank for rank, node in enumerate(stage_ids[column])
                    }
                    stage_ids[column].sort(
                        key=lambda node: (scores[node], previous[node])
                    )
    ranks = {node: rank for ids in stage_ids.values() for rank, node in enumerate(ids)}
    nodes["category_index"] = nodes.node_id.map(ranks)
    for direction, endpoint in (("in", "target_id"), ("out", "source_id")):
        visible_counts = links.loc[links.displayed].groupby(endpoint)["count"].sum()
        hidden_counts = links.loc[~links.displayed].groupby(endpoint)["count"].sum()
        nodes[f"visible_{direction}_count"] = (
            nodes.node_id.map(visible_counts).fillna(0).astype(int)
        )
        nodes[f"omitted_{direction}_count"] = (
            nodes.node_id.map(hidden_counts).fillna(0).astype(int)
        )
    nodes = nodes.sort_values(
        ["stage_index", "category_index"], kind="stable"
    ).reset_index(drop=True)
    summaries = []
    for pair_index, (left, right) in enumerate(zip(columns[:-1], columns[1:])):
        pair_links = links.loc[links.pair_index.eq(pair_index)]
        visible_count = int(pair_links.loc[pair_links.displayed, "count"].sum())
        summaries.append(
            {
                "pair_index": pair_index,
                "source_stage": left,
                "target_stage": right,
                "total_count": total,
                "displayed_count": visible_count,
                "omitted_count": total - visible_count,
                "displayed_fraction": visible_count / total,
            }
        )
    notes = [
        "Links show same-cell categorical associations between adjacent annotations; they do not establish lineage, transitions, or elapsed time."
    ]
    if not links.displayed.any():
        notes.append(
            "All links are below min_count; category totals remain visible and no ribbons are drawn."
        )
    elif (~links.displayed).any():
        notes.append(
            "Filtered links retain their capacity in the layout as gaps; displayed mass is not renormalized."
        )
    metadata = {
        "columns": columns,
        "count_unit": "cells",
        "association_semantics": "same-cell adjacent categorical annotations",
        "directed_stage_order": True,
        "lineage_or_time_inferred": False,
        "n_cells_input": adata.n_obs,
        "n_cells_included": total,
        "n_cells_dropped": int((~included).sum()),
        "drop_na": bool(drop_na),
        "missing_policy": "drop whole rows with genuine missing selected annotations"
        if drop_na
        else "separate missing category",
        "min_count": int(min_count),
        "denominator": total,
        "fraction_denominator": "all included cells before min_count filtering",
        "order": "alphabetical" if sort_categories else order,
        "category_order": {
            column: [nodes.set_index("node_id").loc[node, "label"] for node in ids]
            for column, ids in stage_ids.items()
        },
        "source_columns": source_fields,
        "node_identity": "(stage_index, original string label or None for genuine missing)",
        "n_links_total": len(links),
        "n_links_displayed": int(links.displayed.sum()),
        "pair_coverage": summaries,
        "filtered_mass_renormalized": False,
    }
    return {
        "cells": cells,
        "nodes": nodes,
        "links": links,
        "stage_ids": stage_ids,
        "metadata": metadata,
        "warnings": notes,
    }


def _flow_palette(nodes, palette=None):
    if palette is not None and not isinstance(palette, Mapping):
        raise TypeError(
            "palette must be a category mapping or a mapping of stage names to category mappings."
        )
    nested = palette is not None and any(
        isinstance(value, Mapping) for value in palette.values()
    )
    if nested and not all(isinstance(value, Mapping) for value in palette.values()):
        raise ValueError(
            "Use either a flat category palette or a complete nested stage palette."
        )
    default = category_palette(nodes.display_label.tolist()) if palette is None else {}
    colors = {}
    for row in nodes.itertuples(index=False):
        if palette is None:
            value = "#AEB6BD" if row.is_missing else default[row.display_label]
        else:
            choices = palette.get(row.stage) if nested else palette
            if choices is None:
                raise ValueError(f"palette is missing the stage {row.stage!r}.")
            label_key = None if row.is_missing else row.label
            key = label_key if label_key in choices else row.display_label
            if key not in choices:
                raise ValueError(
                    f"palette is missing category {row.display_label!r} in stage {row.stage!r}."
                )
            value = choices[key]
        if not is_color_like(value):
            raise ValueError(
                f"Invalid palette color for {row.display_label!r}: {value!r}."
            )
        colors[row.node_id] = value
    return colors


def _label_text(row, label_format, normalize=False, width=24):
    if label_format is None:
        label_format = "percent" if normalize else "count"
    label = textwrap.fill(
        str(row["display_label"]),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if label_format == "label":
        return label
    if label_format == "count":
        return f"{label} · {row['count']:,}"
    if label_format == "percent":
        return f"{label} · {row['fraction']:.1%}"
    if label_format == "both":
        return f"{label}\n{row['count']:,} cells · {row['fraction']:.1%}"
    raise ValueError(
        "label_format must be None, 'label', 'count', 'percent', or 'both'."
    )


def _validate_flow_style(*, color_by, alpha, label_format, label_fontsize, normalize):
    if color_by not in {"source", "target"}:
        raise ValueError("color_by must be one of {'source', 'target'}.")
    if label_format not in {None, "label", "count", "percent", "both"}:
        raise ValueError(
            "label_format must be None, 'label', 'count', 'percent', or 'both'."
        )
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean.")
    return _number(alpha, "alpha", maximum=1), _number(
        label_fontsize, "label_fontsize", strict=True
    )


def _labels_on_axes(ax, specifications, fontsize):
    """Use point-space label spacing without moving category or link anchors."""
    fig = ax.figure
    canvas = fig.canvas if hasattr(fig, "canvas") else fig.figure.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    px_per_point = fig.dpi / 72
    font = FontProperties(size=fontsize)
    limits = ax.get_window_extent(renderer)
    artists, records, notes = {}, [], []
    groups = {}
    for item in specifications:
        groups.setdefault(item["stage"], []).append(item)
    for stage, items in groups.items():
        items = sorted(items, key=lambda row: -row["anchor_y"])
        positions, heights = [], []
        for item in items:
            point = ax.transData.transform((item["anchor_x"], item["anchor_y"]))
            item = dict(item)
            positions.append(point[1])
            # An empty string is a valid category; reserve one line of space
            # without replacing its label or changing its source identity.
            lines = item["text"].splitlines() or [""]
            height = max(
                renderer.get_text_width_height_descent(line, font, ismath=False)[1]
                for line in lines
            )
            heights.append(
                max(height, fontsize * px_per_point) * len(lines) + 4 * px_per_point
            )
        positions, heights = np.asarray(positions), np.asarray(heights)
        gap = 3 * px_per_point
        if heights.sum() + gap * max(0, len(items) - 1) > limits.height:
            notes.append(
                f"Labels in {stage!r} exceed the available height; increase figure height or reduce label_fontsize."
            )
        if len(items):
            positions[0] = min(positions[0], limits.y1 - heights[0] / 2)
            for index in range(1, len(items)):
                positions[index] = min(
                    positions[index],
                    positions[index - 1]
                    - (heights[index - 1] + heights[index]) / 2
                    - gap,
                )
            positions[-1] = max(positions[-1], limits.y0 + heights[-1] / 2)
            for index in range(len(items) - 2, -1, -1):
                positions[index] = max(
                    positions[index],
                    positions[index + 1]
                    + (heights[index] + heights[index + 1]) / 2
                    + gap,
                )
        for item, y in zip(items, positions):
            anchor = ax.transData.transform((item["anchor_x"], item["anchor_y"]))
            dx = item.get("offset_x_points")
            if dx is None:
                text_pixel = ax.transData.transform((item["text_x"], item["anchor_y"]))[
                    0
                ]
                dx = (text_pixel - anchor[0]) / px_per_point
            dy = (y - anchor[1]) / px_per_point
            annotation = ax.annotate(
                item["text"],
                xy=(item["anchor_x"], item["anchor_y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=item["ha"],
                va="center",
                fontsize=fontsize,
                color="#273B49",
                bbox={
                    "boxstyle": "round,pad=.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.88,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#8C989F",
                    "lw": 0.65,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                annotation_clip=False,
                zorder=5,
            )
            artists[item["node_id"]] = annotation
            records.append(
                {
                    "node_id": item["node_id"],
                    "stage": stage,
                    "label": item["label"],
                    "text": item["text"],
                    "anchor_x": item["anchor_x"],
                    "anchor_y": item["anchor_y"],
                    "offset_x_points": float(dx),
                    "offset_y_points": float(dy),
                    "ha": item["ha"],
                    "fontsize": fontsize,
                }
            )
    return artists, pd.DataFrame(records), notes


def _omission_text(prepared):
    hidden = int((~prepared["links"].displayed).sum())
    return (
        f"{hidden} links below {prepared['metadata']['min_count']} cells omitted; full category totals retained."
        if hidden
        else None
    )


def _figure_axis(ax, figsize):
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
        fixed_x, fixed_y = None, None
    else:
        fig = ax.figure
        fixed_x = ax.get_xlim() if not ax.get_autoscalex_on() else None
        fixed_y = ax.get_ylim() if not ax.get_autoscaley_on() else None
    return fig, ax, owns_figure, fixed_x, fixed_y
