"""Inspectable plotting results and reproducible figure exports."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_ready(value):
    """Convert metadata to strict JSON without emitting NaN or object reprs."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict(orient="split"))
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    raise TypeError(f"Unsupported export metadata type: {type(value).__name__}")


def _source_tables(data, prefix="data"):
    if isinstance(data, pd.DataFrame):
        yield prefix, data
    elif isinstance(data, pd.Series):
        yield prefix, data.to_frame()
    elif isinstance(data, np.ndarray) and data.ndim in (1, 2):
        yield prefix, pd.DataFrame(data)
    elif isinstance(data, Mapping):
        for key, value in data.items():
            yield from _source_tables(value, str(key) if prefix == "data" else f"{prefix}_{key}")


@dataclass
class PlotResult:
    """A figure, named axes, plotted data, and interpretation metadata.

    Obtain this opt-in result with ``return_result=True`` on supported plots.
    Existing default return values remain available. ``data`` holds the values
    used by the plot; metadata records its metric, ordering, and display choices.
    Use :meth:`save` to export figures, source tables, and a JSON manifest.
    The result is editable: changing an artist does not update its source data.
    """

    fig: Any
    axes: dict
    data: Any = None
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    artists: dict = field(default_factory=dict)

    def __post_init__(self):
        # A SubFigure owns axes but cannot save its canvas independently.
        # Expose the root Figure consistently and make export scope explicit.
        original = self.fig
        seen = set()
        while not callable(getattr(self.fig, "savefig", None)):
            if id(self.fig) in seen or getattr(self.fig, "figure", None) is None:
                raise TypeError("fig must be a Matplotlib Figure or SubFigure.")
            seen.add(id(self.fig))
            self.fig = self.fig.figure
        if original is not self.fig:
            self.metadata = dict(self.metadata)
            self.metadata.setdefault("export_scope", "parent_figure")
            self.metadata.setdefault("source_scope", "Data for named result axes; sibling panels are not reconstructed.")

    def save(self, filename, **kwargs):
        """Export this result; see :func:`scgeo.pl.export_figure` for options."""
        return export_figure(self, filename, **kwargs)


def export_figure(
    figure_or_result,
    filename,
    *,
    formats=("png", "svg"),
    dpi=300,
    width_mm=None,
    height_mm=None,
    panel_labels=None,
    include_data=True,
    include_metadata=True,
    transparent=False,
    metadata=None,
    bbox_inches=None,
):
    """Export a Matplotlib figure or PlotResult with source-data provenance.

    Parameters
    ----------
    figure_or_result
        A Matplotlib Figure or ``PlotResult``. Source CSVs require a PlotResult.
        A result constructed from a SubFigure exports the full parent canvas.
    filename
        Output stem, optionally ending in .png, .svg or .pdf. ``formats`` selects
        the outputs; a supplied suffix is removed before constructing names.
    formats
        One or more of ``'png'``, ``'svg'``, ``'pdf'``.
    dpi
        Resolution of PNG and rasterized artists in vector figures.
    width_mm, height_mm
        Optional canvas dimensions in millimetres. If only one is supplied,
        preserve the original aspect ratio. Original figure size is restored.
        ``bbox_inches=None`` preserves that canvas size; ``'tight'`` crops it.
    panel_labels
        Mapping from PlotResult axis names to labels, or a label sequence for
        the result's axes. Labels are temporary and removed after exporting.
    include_data, include_metadata
        Export source DataFrames/Series/arrays as CSVs and a strict JSON manifest.
        Named indexes are retained. The manifest includes table schemas, hashes,
        figure hashes, plotting metadata, warnings, and software versions.
    metadata
        Additional JSON-compatible provenance, merged over result metadata.

    Returns
    -------
    dict
        ``figures`` and ``tables`` mappings and the ``metadata`` JSON path.

    Notes
    -----
    The exporter records input metadata; it does not reconstruct analysis
    settings that were never supplied or infer statistical significance.
    """
    import matplotlib

    from .. import __version__

    result = figure_or_result if isinstance(figure_or_result, PlotResult) else None
    fig = result.fig if result is not None else figure_or_result
    if not callable(getattr(fig, "savefig", None)):
        raise TypeError("figure_or_result must be a Matplotlib Figure or PlotResult.")
    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(str(fmt).lower().lstrip(".") for fmt in formats))
    if not formats or any(fmt not in {"png", "svg", "pdf"} for fmt in formats):
        raise ValueError("formats must contain png, svg, and/or pdf.")
    if isinstance(dpi, bool) or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a positive finite number.")
    for name, value in (("width_mm", width_mm), ("height_mm", height_mm)):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number.")
    stem = Path(filename)
    if stem.suffix.lower() in {".png", ".svg", ".pdf"}:
        stem = stem.with_suffix("")
    if not stem.name:
        raise ValueError("filename must contain an output stem.")

    provenance = dict(result.metadata) if result is not None else {}
    if metadata is not None:
        provenance.update(metadata)
    # Fail on unsupported metadata before producing a partial export.
    provenance = _json_ready(provenance)
    result_warnings = _json_ready(result.warnings) if result is not None else []
    tables = list(_source_tables(result.data)) if include_data and result is not None else []
    prepared_tables, used_names = [], set()
    for name, frame in tables:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._") or "data"
        if safe_name in used_names:
            raise ValueError(f"Source table names collide after filename sanitization: {name!r}.")
        used_names.add(safe_name)
        default_index = (isinstance(frame.index, pd.RangeIndex)
                         and frame.index.start == 0 and frame.index.stop == len(frame)
                         and frame.index.step == 1 and frame.index.name is None)
        index = not default_index
        content = frame.to_csv(index=index, lineterminator="\n").encode("utf-8")
        path = stem.parent / f"{stem.name}_{safe_name}.csv"
        schema = {
            "rows": len(frame), "columns": [str(c) for c in frame.columns],
            "dtypes": [str(t) for t in frame.dtypes], "index_included": index,
            "index_names": _json_ready(list(frame.index.names)),
            "attributes": _json_ready(frame.attrs),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        prepared_tables.append((str(name), path, content, schema))

    axes = result.axes if result is not None else {str(i): ax for i, ax in enumerate(fig.axes)}
    labels = []
    label_config = {}
    if panel_labels is not None:
        if isinstance(panel_labels, Mapping):
            missing = set(panel_labels).difference(axes)
            if missing:
                raise ValueError(f"Unknown panel axes: {sorted(missing)}")
            labels = [(axes[name], str(label)) for name, label in panel_labels.items()]
            label_config = {str(name): str(label) for name, label in panel_labels.items()}
        else:
            if isinstance(panel_labels, str):
                panel_labels = list(panel_labels)
            else:
                panel_labels = list(panel_labels)
            if len(panel_labels) != len(axes):
                raise ValueError("Supply exactly one panel label per named axis.")
            labels = list(zip(axes.values(), map(str, panel_labels)))
            label_config = dict(zip(map(str, axes.keys()), map(str, panel_labels)))

    stem.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "scgeo.plot_export.v1",
        "software": {"scgeo": __version__, "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "plot": provenance, "warnings": result_warnings,
        "export": {"formats": list(formats), "dpi": float(dpi), "transparent": bool(transparent),
                   "bbox_inches": bbox_inches, "panel_labels": label_config},
        "style": {key: _json_ready(matplotlib.rcParams[key]) for key in (
            "font.family", "font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "pdf.fonttype", "svg.fonttype")},
        "figures": {}, "tables": {},
    }
    output = {"figures": {}, "tables": {}, "metadata": None}
    old_size = np.array(fig.get_size_inches(), copy=True)
    size = old_size.copy()
    if width_mm is not None:
        size[0] = width_mm / 25.4
    if height_mm is not None:
        size[1] = height_mm / 25.4
    if width_mm is not None and height_mm is None:
        size[1] = old_size[1] * size[0] / old_size[0]
    if height_mm is not None and width_mm is None:
        size[0] = old_size[0] * size[1] / old_size[1]
    manifest["export"]["canvas_mm"] = (size * 25.4).tolist()
    label_artists = []
    try:
        fig.set_size_inches(size)
        for ax, label in labels:
            label_artists.append(ax.text(-0.03, 1.03, label, transform=ax.transAxes,
                                         weight="bold", va="bottom", ha="right"))
        # Stable SVG IDs/date fields make source-identical re-exports reviewable.
        with matplotlib.rc_context({"svg.hashsalt": "scgeo.plot_export.v1"}):
            for fmt in formats:
                path = stem.parent / f"{stem.name}.{fmt}"
                creator = f"ScGeo {__version__}"
                if fmt == "png":
                    image_metadata = {"Software": creator}
                elif fmt == "pdf":
                    image_metadata = {"Creator": creator, "CreationDate": None, "ModDate": None}
                else:
                    image_metadata = {"Creator": creator, "Date": None}
                fig.savefig(path, format=fmt, dpi=dpi, transparent=transparent,
                            bbox_inches=bbox_inches, metadata=image_metadata)
                output["figures"][fmt] = str(path)
                manifest["figures"][fmt] = {
                    "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    finally:
        for artist in label_artists:
            artist.remove()
        fig.set_size_inches(old_size)

    for name, path, content, schema in prepared_tables:
        path.write_bytes(content)
        output["tables"][name] = str(path)
        manifest["tables"][name] = {"file": path.name, **schema}
    if include_metadata:
        path = stem.parent / f"{stem.name}.json"
        path.write_text(json.dumps(_json_ready(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output["metadata"] = str(path)
    return output
