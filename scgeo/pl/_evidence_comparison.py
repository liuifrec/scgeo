"""Compare stored effects across contrasts without recomputing evidence."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ._result import PlotResult
from ._style import category_palette
from ._table_plot_utils import ordered_labels, panel_axes, required_table

_INTERVAL_LABELS = {
    "sample_bootstrap_ci95": "Sample bootstrap 95% CI",
    "bootstrap_ci95": "Bootstrap 95% CI",
    "representation_iqr": "Across-representation IQR",
    "representation_range": "Across-representation range",
}
_METRIC_LABELS = {
    "delta_norm": "Raw displacement magnitude",
    "normalized_delta_norm": "Normalized displacement magnitude",
    "alignment_cosine": "Alignment cosine",
}


def evidence_comparison(
    data,
    *,
    metric=None,
    state_order=None,
    contrast_order=None,
    palette=None,
    sharex="auto",
    axes=None,
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Draw aligned state-by-contrast effect panels from a tidy results table.

    Required columns are ``state``, ``contrast``, ``metric``, and ``estimate``.
    Supply ``interval_low``, ``interval_high``, and ``interval_type`` together
    for intervals; their endpoints are drawn independently of the estimate.
    Labels such as ``sample_bootstrap_ci95`` and ``representation_iqr`` remain
    distinct in the legend. No intervals, tests, or composite scores are fitted.

    Optional ``representation`` and ``unit`` columns describe the effect scale.
    ``sharex='auto'`` shares limits only when every finite estimate has the same
    declared representation and unit. Alternatively, a nonempty common
    ``scale_id`` explicitly declares comparability across representations.
    ``sharex=True`` rejects incompatible or unspecified scales. Missing state
    and contrast combinations stay visible. Duplicate state/contrast rows for
    the selected metric are rejected rather than silently aggregated.
    Optional ``evidence_status`` (preferred) or ``status`` labels are preserved
    and displayed with distinct marker shapes; a finite estimate never promotes
    supplied insufficient or unavailable evidence to ``'assessed'``.

    ``axes`` is an optional mapping from contrast labels to distinct Axes in
    one Figure. Orders must retain observed labels; filter the input explicitly
    to select a subset. Extra ordered labels create visible missing rows/panels.
    Returns a Figure by default, ``(Figure, plotted_table)`` with
    ``return_data=True``, or a :class:`PlotResult` with ``return_result=True``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    work = required_table(data, ["state", "contrast", "metric", "estimate"])
    for key in ("state", "contrast", "metric"):
        if work[key].isna().any():
            raise ValueError(f"{key} labels must not be missing.")
        work[key] = work[key].astype(str)
    metrics = list(work["metric"].unique())
    if metric is None:
        if len(metrics) != 1:
            raise ValueError("Select metric= explicitly when the table contains multiple metrics.")
        metric = metrics[0]
    metric = str(metric)
    work = work.loc[work["metric"] == metric].copy()
    if work.empty:
        raise ValueError(f"No rows for metric={metric!r}.")
    if work.duplicated(["state", "contrast"]).any():
        raise ValueError("Duplicate state/contrast rows; select one representation or summarize explicitly.")
    work["estimate"] = pd.to_numeric(work["estimate"], errors="raise").astype(float)
    if np.isinf(work["estimate"]).any():
        raise ValueError("estimate values must be finite or missing.")
    bounds = ["interval_low", "interval_high"]
    if any(key in work for key in bounds) and not all(key in work for key in bounds):
        raise ValueError("Supply both interval_low and interval_high.")
    if not all(key in work for key in bounds):
        work["interval_low"] = np.nan
        work["interval_high"] = np.nan
    for key in bounds:
        work[key] = pd.to_numeric(work[key], errors="raise").astype(float)
        if np.isinf(work[key]).any():
            raise ValueError("Interval endpoints must be finite or missing.")
    has_low, has_high = work.interval_low.notna(), work.interval_high.notna()
    if (has_low != has_high).any():
        raise ValueError("Each interval needs two endpoints, or two missing endpoints.")
    if (work.loc[has_low, "interval_low"] > work.loc[has_low, "interval_high"]).any():
        raise ValueError("interval_low must not exceed interval_high.")
    if work.loc[has_low, "estimate"].isna().any():
        raise ValueError("A supplied interval needs a finite estimate.")
    if "interval_type" not in work:
        work["interval_type"] = pd.NA
    valid_type = work.interval_type.notna() & work.interval_type.astype(str).str.strip().ne("")
    if (has_low & ~valid_type).any():
        raise ValueError("Every supplied interval needs an explicit interval_type.")
    states = ordered_labels(work.state, state_order, "state_order")
    contrasts = ordered_labels(work.contrast, contrast_order, "contrast_order")
    colors = category_palette(contrasts, palette)
    finite = work.loc[work.estimate.notna()]
    shared_scale = False
    scale_description = None
    if not finite.empty:
        scale_present = (
            finite.scale_id.notna() & finite.scale_id.astype(str).str.strip().ne("")
            if "scale_id" in finite else pd.Series(False, index=finite.index)
        )
        if scale_present.any():
            shared_scale = bool(scale_present.all() and finite.scale_id.astype(str).nunique() == 1)
            if shared_scale:
                scale_description = str(finite.scale_id.iloc[0])
        elif "representation" in finite and "unit" in finite:
            representations = finite.representation
            known = representations.notna().all() and representations.astype(str).str.strip().ne("").all()
            units = finite["unit"]
            known_units = units.notna().all() and units.astype(str).str.strip().ne("").all()
            shared_scale = bool(known and known_units and representations.astype(str).nunique() == 1 and units.nunique() == 1)
            if shared_scale:
                scale_description = str(representations.iloc[0])
    if sharex not in {"auto", True, False}:
        raise ValueError("sharex must be 'auto', True, or False.")
    if sharex is True and len(contrasts) > 1 and not shared_scale:
        raise ValueError("Shared limits require the same declared representation/unit or a common scale_id.")
    share_limits = shared_scale if sharex == "auto" else bool(sharex)
    # Even inside a single panel, raw distances from unrelated scales cannot be
    # presented as directly comparable state effects without an explicit scale.
    for contrast, rows in finite.groupby("contrast", sort=False):
        if "scale_id" in rows:
            scale_present = rows.scale_id.notna() & rows.scale_id.astype(str).str.strip().ne("")
            if scale_present.any():
                if scale_present.all() and rows.scale_id.astype(str).nunique() == 1:
                    continue
                raise ValueError(f"Contrast {contrast!r} has incomplete or mixed scale_id declarations.")
        if "representation" in rows and rows.representation.dropna().nunique() > 1:
            raise ValueError(f"Contrast {contrast!r} mixes representations; select a common scale explicitly.")
        if "unit" in rows and rows.unit.dropna().nunique() > 1:
            raise ValueError(f"Contrast {contrast!r} mixes units.")
    types = list(dict.fromkeys(work.loc[has_low, "interval_type"].astype(str)))
    styles = ["-", "--", ":", "-."]
    if len(types) > len(styles):
        raise ValueError("At most four interval types may be compared in one figure; split the table.")
    type_styles = dict(zip(types, styles))
    figure, axis_map, created = panel_axes(
        contrasts, axes, figsize or (max(4, 3.5 * len(contrasts)), max(2.8, .38 * len(states) + 1.6)),
        sharex=share_limits, sharey=True,
    )
    table_index = pd.MultiIndex.from_product([contrasts, states], names=["contrast", "state"])
    plotted = work.set_index(["contrast", "state"]).reindex(table_index).reset_index()
    plotted["metric"] = metric
    plotted["estimate_available"] = plotted.estimate.notna()
    display_status = pd.Series(pd.NA, index=plotted.index, dtype=object)
    status_columns = [key for key in ("evidence_status", "status", "availability") if key in plotted]
    for key in status_columns:
        supplied = plotted[key].where(plotted[key].notna() & plotted[key].astype(str).str.strip().ne(""))
        display_status = display_status.where(display_status.notna(), supplied)
    plotted["display_status"] = display_status.fillna(pd.Series(
        np.where(plotted.estimate.notna(), "assessed", "not_computed"), index=plotted.index,
    )).astype(str)
    if "availability" not in plotted:
        plotted["availability"] = plotted["display_status"]
    plotted.attrs = deepcopy(data.attrs)
    status_markers = {"assessed": "o", "stable_effect": "o", "stable_neutral": "s",
                      "representation_unstable": "^", "unstable": "^",
                      "insufficient_coverage": "X", "insufficient": "X",
                      "not_computed": "x", "unavailable": "P", "numerical_degeneracy": "D"}
    def status_marker(value):
        return status_markers.get(str(value).lower().replace(" ", "_").replace("-", "_"), "d")
    artists = {}
    for panel_index, contrast in enumerate(contrasts):
        ax = axis_map[contrast]
        rows = plotted.loc[plotted.contrast == contrast].set_index("state")
        points, intervals, missing_marks = [], [], []
        for y, state in enumerate(states):
            row = rows.loc[state]
            if np.isfinite(row.estimate):
                if np.isfinite(row.interval_low):
                    intervals.append(ax.hlines(y, row.interval_low, row.interval_high,
                                               color=colors[contrast], linewidth=1.8,
                                               linestyles=type_styles[str(row.interval_type)]))
                points.append(ax.scatter(row.estimate, y, color=colors[contrast],
                                         marker=status_marker(row.display_status), s=34, zorder=3))
            else:
                missing_marks.append(ax.text(.025, y, row.display_status.replace("_", " "), transform=ax.get_yaxis_transform(),
                                             va="center", color="0.48", fontsize=8))
        ax.set_yticks(np.arange(len(states)))
        ax.set_yticklabels(states)
        ax.set_ylim(len(states) - .5, -.5)
        ax.tick_params(axis="y", labelleft=panel_index == 0, length=0)
        reps = list(work.loc[work.contrast == contrast, "representation"].dropna().astype(str).unique()) if "representation" in work else []
        label = _METRIC_LABELS.get(metric, metric.replace("_", " "))
        if len(reps) == 1:
            label += f"\n{reps[0]}"
        units = list(work.loc[work.contrast == contrast, "unit"].dropna().astype(str).unique()) if "unit" in work else []
        if len(units) == 1:
            label += f" · {units[0]}" if len(reps) == 1 else f"\n{units[0]}"
        ax.set_xlabel(label)
        ax.set_title(contrast)
        ax.grid(axis="x", color="0.9", linewidth=.6)
        ax.axvline(0, color="0.65", linewidth=.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        artists[contrast] = {"estimates": points, "intervals": intervals, "missing": missing_marks}
        panel_types = list(dict.fromkeys(rows.interval_type.dropna().astype(str)))
        handles = [Line2D([0], [0], color="0.35", linestyle=type_styles[k], label=_INTERVAL_LABELS.get(k, k.replace("_", " ")))
                   for k in panel_types if k in type_styles]
        if status_columns:
            handles.extend(Line2D([0], [0], color="0.35", marker=status_marker(k), linestyle="none",
                                  label=k.replace("_", " "))
                           for k in rows.loc[rows.estimate.notna(), "display_status"].unique())
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)
    if share_limits:
        low = min(ax.get_xlim()[0] for ax in axis_map.values())
        high = max(ax.get_xlim()[1] for ax in axis_map.values())
        for ax in axis_map.values():
            ax.set_xlim(low, high)
    if title:
        if created:
            figure.suptitle(title)
        else:
            axis_map[contrasts[0]].set_title(f"{title}\n{contrasts[0]}")
    if created:
        figure.tight_layout()
    plot_warnings = []
    if not share_limits and len(contrasts) > 1:
        plot_warnings.append("Panels use independent x limits; compare only scientifically compatible scales.")
    if len(types) > 1:
        plot_warnings.append("Interval types differ; refer to their explicit legend labels and source provenance.")
    metadata = {
        "plot": "evidence_comparison", "metric": metric, "state_order": states,
        "contrast_order": contrasts, "shared_limits": share_limits,
        "scale_description": scale_description, "interval_types": types,
        "status_columns": status_columns,
        "interpretation": "Stored effects and declared intervals; no inference or aggregation was computed.",
        "source_attrs": deepcopy(data.attrs),
    }
    result = PlotResult(fig=figure, axes=axis_map, data=plotted, metadata=metadata,
                        warnings=plot_warnings, artists=artists)
    if show:
        plt.show()
    if return_result:
        return result
    return (figure, plotted) if return_data else figure
