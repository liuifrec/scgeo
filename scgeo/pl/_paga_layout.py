"""Display-only glyphs and deterministic label placement for PAGA maps."""

from __future__ import annotations

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Wedge
from matplotlib.transforms import Bbox


def pie_glyph(
    ax,
    xy,
    fractions,
    colors,
    radius_points,
    rim_color,
    highlighted,
    highlight_color,
    highlight_lw,
):
    """Draw a point-sized composition pie with independent metric/highlight rims."""
    padding = 5.0
    center = radius_points + padding
    area = DrawingArea(2 * center, 2 * center, 0, 0, clip=False)
    angle = 0.0
    for fraction, color in zip(fractions, colors):
        if fraction <= 0:
            continue
        end = angle + 360 * float(fraction)
        area.add_artist(
            Wedge(
                (center, center),
                radius_points,
                angle,
                end,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
            )
        )
        angle = end
    ring = Circle(
        (center, center),
        radius_points + 1,
        fill=False,
        edgecolor=rim_color,
        linewidth=1.7,
    )
    area.add_artist(ring)
    highlight = None
    if highlighted:
        highlight = Circle(
            (center, center),
            radius_points + 3,
            fill=False,
            edgecolor=highlight_color,
            linewidth=highlight_lw,
        )
        area.add_artist(highlight)
    box = AnnotationBbox(
        area,
        xy,
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        annotation_clip=False,
    )
    box.set_zorder(5)
    ax.add_artist(box)
    return box, ring, highlight


def _intersection_area(left, right):
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height


def place_labels(ax, rows, *, fontsize, mode, offsets, obstacles, labels=None):
    """Place labels in screen space, retaining unchanged data-coordinate anchors.

    ``rows`` is an ordered list of (state, x, y). Candidate selection minimizes
    overlap with glyphs/previous labels and clipping. It is deterministic and
    requires no external label-adjustment dependency. Offsets are in points.
    ``labels`` optionally maps stable state IDs to display text; artist and
    source-table keys remain the original state IDs, including repeated text.
    """
    import pandas as pd

    fig = ax.figure
    canvas = fig.canvas if hasattr(fig, "canvas") else fig.figure.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    font = FontProperties(size=fontsize)
    occupied = []
    for x, y, radius in obstacles:
        px, py = ax.transData.transform((x, y))
        radius_px = (radius + 2) * px_per_pt
        occupied.append(
            Bbox.from_extents(
                px - radius_px, py - radius_px, px + radius_px, py + radius_px
            )
        )
    bounds = ax.get_window_extent(renderer)
    candidates = [
        (0, 16),
        (0, -16),
        (20, 12),
        (-20, 12),
        (20, -12),
        (-20, -12),
        (30, 0),
        (-30, 0),
        (0, 28),
        (0, -28),
        (32, 22),
        (-32, 22),
        (32, -22),
        (-32, -22),
        (48, 0),
        (-48, 0),
        (0, 42),
        (0, -42),
    ]
    artists, records = {}, []
    for state, x, y in rows:
        text = str(labels.get(state, state)) if labels is not None else str(state)
        anchor = ax.transData.transform((x, y))
        width, height, _ = renderer.get_text_width_height_descent(
            text, font, ismath=False
        )
        width += 6 * px_per_pt
        height = max(height, fontsize * px_per_pt) + 4 * px_per_pt
        choices = (
            [offsets[state]]
            if state in offsets
            else ([(0, 0)] if mode == "on_data" else candidates)
        )
        ranked = []
        for ordinal, (dx, dy) in enumerate(choices):
            px, py = anchor + np.array([dx, dy]) * px_per_pt
            bbox = Bbox.from_bounds(px - width / 2, py - height / 2, width, height)
            overlap = sum(_intersection_area(bbox, other) for other in occupied)
            outside = width * height - _intersection_area(bbox, bounds)
            ranked.append(
                (
                    overlap * 10 + outside * 20 + 0.025 * (dx * dx + dy * dy),
                    ordinal,
                    dx,
                    dy,
                    bbox,
                )
            )
        _, _, dx, dy, bbox = min(ranked, key=lambda item: (item[0], item[1]))
        occupied.append(bbox)
        leader = (
            None
            if dx == 0 and dy == 0
            else {
                "arrowstyle": "-",
                "color": "#7B8791",
                "lw": 0.65,
                "shrinkA": 3,
                "shrinkB": 7,
                "connectionstyle": "arc3,rad=0",
            }
        )
        annotation = ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#243746",
            arrowprops=leader,
            bbox={
                "boxstyle": "round,pad=.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.88,
            },
            zorder=8,
            annotation_clip=False,
        )
        artists[state] = annotation
        records.append(
            {
                "state": state,
                "anchor_x": x,
                "anchor_y": y,
                "offset_x_points": float(dx),
                "offset_y_points": float(dy),
                "placement": "manual" if state in offsets else mode,
            }
        )
    return artists, pd.DataFrame(
        records,
        columns=[
            "state",
            "anchor_x",
            "anchor_y",
            "offset_x_points",
            "offset_y_points",
            "placement",
        ],
    )
