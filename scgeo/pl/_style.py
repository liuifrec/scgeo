"""Scoped plotting styles and reusable category palettes."""
from __future__ import annotations

import colorsys
import hashlib
from collections.abc import Mapping

import numpy as np


def category_palette(labels, palette=None):
    """Return a stable label-to-color mapping, independent of subset ordering.

    Default colors use a deterministic label hash. For publication, pass a
    curated mapping; it must cover all requested labels. Keys in the returned
    mapping are strings. Colors alone cannot identify arbitrarily many groups,
    so retain labels and consider separate panels for large category sets.
    """
    from matplotlib.colors import is_color_like, to_hex

    if palette is not None and not isinstance(palette, Mapping):
        raise TypeError("palette must be a label-to-color mapping.")
    out = {}
    for label in labels:
        key = str(label)
        if key in out:
            continue
        if palette is not None:
            if label in palette:
                color = palette[label]
            elif key in palette:
                color = palette[key]
            else:
                raise ValueError(f"palette does not contain a color for {key!r}.")
            if not is_color_like(color):
                raise ValueError(f"Invalid color for {key!r}: {color!r}.")
            out[key] = color
        else:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            hue = int.from_bytes(digest[:4], "big") / 2**32
            saturation = 0.53 + digest[4] / 255 * 0.16
            value = 0.65 + digest[5] / 255 * 0.14
            out[key] = to_hex(colorsys.hsv_to_rgb(hue, saturation, value))
    return out


def style_context(style="paper", *, width_mm=None, height_mm=None, overrides=None):
    """Temporarily apply a paper or presentation style without global changes.

    ``width_mm`` and ``height_mm`` set the default canvas for plots that do not
    supply their own figsize. Use ``export_figure`` to size an existing figure.
    ``overrides`` accepts any Matplotlib rcParams mapping. SVG text remains text
    and PDF uses TrueType fonts; viewers need the selected font for SVG editing.
    """
    import matplotlib as mpl

    if style not in {"paper", "presentation"}:
        raise ValueError("style must be 'paper' or 'presentation'.")
    scale = 1.0 if style == "paper" else 1.45
    params = {
        "font.family": "DejaVu Sans", "font.size": 10 * scale,
        "axes.titlesize": 11 * scale, "axes.labelsize": 10 * scale,
        "xtick.labelsize": 9 * scale, "ytick.labelsize": 9 * scale,
        "legend.fontsize": 9 * scale, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    size = list(mpl.rcParams["figure.figsize"])
    for i, (name, value) in enumerate((("width_mm", width_mm), ("height_mm", height_mm))):
        if value is not None:
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
            size[i] = value / 25.4
    if width_mm is not None or height_mm is not None:
        params["figure.figsize"] = size
    if overrides is not None:
        params.update(overrides)
    return mpl.rc_context(params)
