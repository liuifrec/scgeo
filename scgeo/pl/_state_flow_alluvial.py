from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def _coerce_str_categories(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            raise KeyError(f"'{c}' not found in adata.obs.")
        out[c] = out[c].astype(str).fillna("NA")
    return out


def _default_palette(labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    uniq = list(dict.fromkeys(labels))
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def _make_ribbon(x0, x1, y0_low, y0_high, y1_low, y1_high, curve=0.35):
    dx = x1 - x0
    c0 = x0 + dx * curve
    c1 = x1 - dx * curve

    verts = [
        (x0, y0_high),
        (c0, y0_high),
        (c1, y1_high),
        (x1, y1_high),

        (x1, y1_low),
        (c1, y1_low),
        (c0, y0_low),
        (x0, y0_low),

        (x0, y0_high),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,

        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


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
    palette: Optional[dict[str, tuple[float, float, float, float]]] = None,
    ax=None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Draw an alluvial / ribbon plot for ordered categorical columns in `adata.obs`.

    Parameters
    ----------
    adata
        AnnData object.
    columns
        Ordered categorical columns, e.g. ["timepoint", "alignment_group"].
    min_count
        Minimum flow count to retain.
    drop_na
        If True, drop rows containing 'NA' in any selected column.
    normalize
        If True, each stage totals to 1.0 instead of raw counts.
    sort_categories
        If True, sort labels alphabetically within each stage.
    color_by
        One of {"source", "target"}; determines ribbon color source.
    alpha
        Ribbon transparency.
    return_data
        If True, return (fig, ax, stage_boxes, link_df).
    """
    if len(columns) < 2:
        raise ValueError("columns must contain at least 2 obs columns.")

    obs = _coerce_str_categories(adata.obs, columns)
    obs = obs[list(columns)].copy()

    if drop_na:
        for c in columns:
            obs = obs[obs[c] != "NA"].copy()

    # stage totals
    stage_order = list(columns)
    stage_labels: dict[str, list[str]] = {}
    stage_counts: dict[str, pd.Series] = {}

    for c in stage_order:
        counts = obs[c].value_counts(dropna=False)
        if sort_categories:
            counts = counts.sort_index()
        stage_counts[c] = counts
        stage_labels[c] = counts.index.tolist()

    # global palette
    all_labels = []
    for c in stage_order:
        all_labels.extend(stage_labels[c])
    if palette is None:
        palette = _default_palette(all_labels)

    # scale each stage to height
    total_n = len(obs)
    stage_boxes = []  # each row: stage, label, y0, y1, x, value

    for i, c in enumerate(stage_order):
        counts = stage_counts[c].copy()

        if normalize:
            vals = counts / counts.sum()
        else:
            vals = counts / total_n

        current_y = 1.0
        for lab, v in vals.items():
            y1 = current_y
            y0 = y1 - float(v)
            stage_boxes.append(
                {
                    "stage": c,
                    "label": str(lab),
                    "x": i * column_gap,
                    "y0": y0,
                    "y1": y1,
                    "value": float(v),
                }
            )
            current_y = y0 - category_gap

    stage_boxes = pd.DataFrame(stage_boxes)

    # link tables between adjacent stages
    link_frames = []
    for left, right in zip(stage_order[:-1], stage_order[1:]):
        tmp = (
            obs.groupby([left, right], observed=False)
            .size()
            .rename("count")
            .reset_index()
        )
        tmp = tmp[tmp["count"] >= int(min_count)].copy()
        if tmp.empty:
            continue

        if normalize:
            tmp["value"] = tmp["count"] / tmp["count"].sum()
        else:
            tmp["value"] = tmp["count"] / total_n

        tmp["left_stage"] = left
        tmp["right_stage"] = right
        tmp = tmp.rename(columns={left: "left_label", right: "right_label"})
        link_frames.append(tmp)

    if not link_frames:
        raise ValueError("No links to plot. Check columns, filters, or min_count.")

    link_df = pd.concat(link_frames, ignore_index=True)

    # allocate flow segments inside each stage box
    box_lookup = {
        (row["stage"], row["label"]): row
        for _, row in stage_boxes.iterrows()
    }

    left_offsets = {(row["stage"], row["label"]): row["y0"] for _, row in stage_boxes.iterrows()}
    right_offsets = {(row["stage"], row["label"]): row["y0"] for _, row in stage_boxes.iterrows()}

    alloc_rows = []
    for left, right in zip(stage_order[:-1], stage_order[1:]):
        sub = link_df[(link_df["left_stage"] == left) & (link_df["right_stage"] == right)].copy()

        if sort_categories:
            sub = sub.sort_values(["left_label", "right_label"])
        else:
            sub = sub.sort_values(["left_label", "right_label"])

        for _, r in sub.iterrows():
            lk = (left, str(r["left_label"]))
            rk = (right, str(r["right_label"]))
            v = float(r["value"])

            y0_low = left_offsets[lk]
            y0_high = y0_low + v
            left_offsets[lk] = y0_high

            y1_low = right_offsets[rk]
            y1_high = y1_low + v
            right_offsets[rk] = y1_high

            alloc_rows.append(
                {
                    "left_stage": left,
                    "right_stage": right,
                    "left_label": str(r["left_label"]),
                    "right_label": str(r["right_label"]),
                    "value": v,
                    "x0": box_lookup[lk]["x"],
                    "x1": box_lookup[rk]["x"],
                    "y0_low": y0_low,
                    "y0_high": y0_high,
                    "y1_low": y1_low,
                    "y1_high": y1_high,
                }
            )

    alloc_df = pd.DataFrame(alloc_rows)

    made_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        made_fig = True
    else:
        fig = ax.figure

    # draw ribbons first
    for _, r in alloc_df.iterrows():
        if color_by == "source":
            col = palette.get(r["left_label"], (0.6, 0.6, 0.6, 1.0))
        elif color_by == "target":
            col = palette.get(r["right_label"], (0.6, 0.6, 0.6, 1.0))
        else:
            raise ValueError("color_by must be one of {'source', 'target'}.")

        path = _make_ribbon(
            r["x0"],
            r["x1"],
            r["y0_low"],
            r["y0_high"],
            r["y1_low"],
            r["y1_high"],
            curve=ribbon_curve,
        )
        patch = PathPatch(path, facecolor=col, edgecolor="none", alpha=alpha, zorder=1)
        ax.add_patch(patch)

    # draw stage category bars and labels
    bar_width = 0.08
    for _, r in stage_boxes.iterrows():
        x = r["x"]
        y0 = r["y0"]
        y1 = r["y1"]
        h = y1 - y0
        rect = plt.Rectangle(
            (x - bar_width / 2, y0),
            bar_width,
            h,
            facecolor=palette.get(r["label"], (0.5, 0.5, 0.5, 1.0)),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x,
            (y0 + y1) / 2,
            str(r["label"]),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )

    # stage titles
    for i, c in enumerate(stage_order):
        ax.text(
            i * column_gap,
            1.08,
            c,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlim(-0.6, (len(stage_order) - 1) * column_gap + 0.6)
    ax.set_ylim(-0.02, 1.14)
    ax.axis("off")

    if title is None:
        title = "ScGeo state flow alluvial"
    ax.set_title(title, fontsize=14)

    if show:
        plt.show()

    if return_data:
        return fig, ax, stage_boxes, alloc_df
    if made_fig:
        return fig, ax
    return ax