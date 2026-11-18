"""Interpret stored geometry displacement beside stored velocity alignment.

No analysis is recomputed: the quantitative axes must originate from the same
specified analysis representation, not from projected UMAP arrows.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ._result import PlotResult


_PALETTE = {
    "positive": "#177E89",
    "opposed": "#C45A55",
    "intermediate": "#8B8F9B",
}


def _identifiers(values, title):
    if values.isna().any():
        raise ValueError(f"{title} cannot contain missing state identifiers.")
    result = values.map(str)
    if result.duplicated().any():
        raise ValueError(f"{title} contains duplicate or string-colliding states.")
    return result


def geometry_velocity_concordance(
    shift_df,
    alignment_df,
    *,
    state_key="state",
    alignment_state_key=None,
    effect_metric="normalized_delta_norm",
    alignment_metric="alignment_cosine",
    status_key="coverage_status",
    eligible_statuses=("eligible_exploratory", "eligible"),
    usable_key="usable",
    label_key=None,
    analysis_rep=None,
    positive_threshold=0.5,
    opposed_threshold=-0.25,
    sample_count_keys=("n_mice_control", "n_mice_irradiated"),
    show_ci=True,
    highlight_states=None,
    max_labels=16,
    axes=None,
    figsize=(12.0, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Show descriptive concordance between stored displacement and alignment.

    Provide one row per state in ``shift_df`` and ``alignment_df``; the latter
    may use a different state column. Only rows with usable alignment, finite
    values and an eligible coverage status are plotted as estimated results.
    Ineligible states are counted and exported rather than silently dropped.
    ``positive_threshold`` and ``opposed_threshold`` are visual descriptions,
    never significance thresholds or evidence of trajectories. A shared
    ``analysis_rep`` is declared by the caller, not inferred from UMAP.

    Raw ``delta_norm`` may use stored ``magnitude_ci95_low/high`` intervals.
    Normalized magnitudes require their own explicitly named intervals; raw
    intervals are never applied to normalized estimates. No confidence
    intervals, mouse-level replication or missing velocities are fabricated.

    Default return is ``(fig, axes)``; ``return_result=True`` provides a
    :class:`scgeo.pl.PlotResult` with included and excluded state tables.
    """
    if not isinstance(shift_df, pd.DataFrame) or not isinstance(alignment_df, pd.DataFrame):
        raise TypeError("shift_df and alignment_df must be pandas DataFrames.")
    akey = alignment_state_key or (state_key if state_key in alignment_df else "leiden")
    for frame, columns, name in (
        (shift_df, (state_key, effect_metric), "shift_df"),
        (alignment_df, (akey, alignment_metric), "alignment_df"),
    ):
        absent = set(columns) - set(frame)
        if absent:
            raise KeyError(f"{name} lacks required column(s): {sorted(absent)}")
    if not (-1 <= opposed_threshold < positive_threshold <= 1):
        raise ValueError("Require -1 <= opposed_threshold < positive_threshold <= 1.")
    if isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels < 0:
        raise ValueError("max_labels must be a nonnegative integer.")
    if axes is not None and (not isinstance(axes, Mapping) or set(axes) != {"concordance", "effect"}):
        raise ValueError("axes must map 'concordance' and 'effect' to two axes.")
    if axes is not None and axes["concordance"].figure is not axes["effect"].figure:
        raise ValueError("Supplied axes must belong to one figure.")
    shifts = shift_df.copy(deep=True)
    alignments = alignment_df.copy(deep=True)
    # Filter scope/center/reference before plotting: do not silently aggregate
    # independent analysis choices into one state estimate.
    for source_name, frame in (("shift_df", shifts), ("alignment_df", alignments)):
        for column in ("scope", "center", "reference", "graph_mode"):
            if column in frame and frame[column].dropna().astype(str).nunique() > 1:
                raise ValueError(
                    f"{source_name} mixes {column!r} settings; select a single analysis before plotting."
                )
    if analysis_rep is not None and "rep" in alignments:
        reps = set(alignments["rep"].dropna().astype(str))
        if reps and reps != {str(analysis_rep)}:
            raise ValueError(f"alignment_df uses representations {sorted(reps)}, not {analysis_rep!r}.")
    shifts["_state_id"] = _identifiers(shifts[state_key], "shift_df")
    alignments["_state_id"] = _identifiers(alignments[akey], "alignment_df")
    # Merge by identity, never by row position or sorted numeric cluster index.
    selected_align = alignments.set_index("_state_id")
    shifts["alignment"] = pd.to_numeric(
        shifts["_state_id"].map(selected_align[alignment_metric]), errors="coerce"
    )
    shifts["effect"] = pd.to_numeric(shifts[effect_metric], errors="coerce")
    if usable_key in alignments:
        usable = selected_align[usable_key].eq(True)
        shifts["alignment_usable"] = shifts["_state_id"].map(usable).fillna(False).astype(bool)
    else:
        shifts["alignment_usable"] = True
    if status_key in shifts:
        shifts["coverage_eligible"] = shifts[status_key].isin(tuple(eligible_statuses))
    else:
        shifts["coverage_eligible"] = True
    # A value outside the cosine domain indicates malformed inputs, not a new class.
    out_of_range = shifts["alignment"].notna() & ~shifts["alignment"].between(-1, 1)
    if out_of_range.any():
        raise ValueError("alignment cosine must be between -1 and 1.")
    negative_effect = shifts["effect"].notna() & (shifts["effect"] < 0)
    if negative_effect.any():
        raise ValueError("Displacement magnitude cannot be negative.")
    shifts["eligible"] = (
        shifts["coverage_eligible"]
        & shifts["alignment_usable"]
        & np.isfinite(shifts["alignment"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(shifts["effect"].to_numpy(dtype=float, na_value=np.nan))
    )
    shifts["reason"] = np.select(
        [
            ~shifts["coverage_eligible"],
            ~shifts["alignment_usable"],
            ~np.isfinite(shifts["effect"].to_numpy(dtype=float, na_value=np.nan)),
            ~np.isfinite(shifts["alignment"].to_numpy(dtype=float, na_value=np.nan)),
        ],
        ["coverage_ineligible", "alignment_unusable", "effect_missing", "alignment_missing"],
        default="eligible",
    )
    shifts["alignment_class"] = "not_assessed"
    eligible = shifts.loc[shifts.eligible].copy()
    if len(eligible):
        classes = np.select(
            [eligible.alignment >= positive_threshold, eligible.alignment <= opposed_threshold],
            ["positive", "opposed"], default="intermediate",
        )
        eligible["alignment_class"] = classes
        shifts.loc[eligible.index, "alignment_class"] = classes
    eligible.sort_values("effect", ascending=False, kind="stable", inplace=True)
    excluded = shifts.loc[~shifts.eligible].copy()
    if label_key is not None and label_key not in shifts:
        raise KeyError(f"shift_df lacks label_key {label_key!r}.")
    if highlight_states is not None:
        highlighted = {str(x) for x in highlight_states}
        unknown = highlighted - set(shifts._state_id)
        if unknown:
            raise ValueError(f"highlight_states contains unknown states: {sorted(unknown)}")
    else:
        highlighted = set()
    interval_columns = {
        "delta_norm": ("magnitude_ci95_low", "magnitude_ci95_high"),
        "normalized_delta_norm": (
            "normalized_magnitude_ci95_low", "normalized_magnitude_ci95_high"
        ),
    }.get(effect_metric)
    ci_available = bool(
        show_ci and interval_columns and all(c in eligible for c in interval_columns)
    )
    if ci_available:
        low, high = (pd.to_numeric(eligible[c], errors="coerce") for c in interval_columns)
        reversed_interval = np.isfinite(low) & np.isfinite(high) & (low > high)
        if reversed_interval.any():
            raise ValueError("Finite stored interval endpoints must be ordered.")
        eligible["ci_low"], eligible["ci_high"] = low, high
    else:
        eligible["ci_low"], eligible["ci_high"] = np.nan, np.nan
    # Validate before modifying caller-supplied axes.
    owns_figure = axes is None
    if owns_figure:
        fig, pair = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.15, 1]})
        axes = dict(zip(("concordance", "effect"), pair))
    else:
        fig = axes["concordance"].figure
    left, right = axes["concordance"], axes["effect"]
    artists = {"points": []}
    left.axhline(0, color="#9CA3AF", lw=1, ls="--", zorder=0)
    left.axhline(positive_threshold, color=_PALETTE["positive"], lw=0.8, ls=":", alpha=0.7)
    left.axhline(opposed_threshold, color=_PALETTE["opposed"], lw=0.8, ls=":", alpha=0.7)
    y = np.arange(len(eligible))
    for idx, (_, row) in enumerate(eligible.iterrows()):
        color = _PALETTE[row["alignment_class"]]
        marker = "D" if row["_state_id"] in highlighted else "o"
        artists["points"].append(
            left.scatter(row["effect"], row["alignment"], s=75, color=color, marker=marker,
                         edgecolor="white", linewidth=0.9, zorder=3)
        )
        right.hlines(idx, 0, row["effect"], color=color, alpha=0.65, lw=2)
        right.scatter(row["effect"], idx, s=55, color=color, edgecolor="white", lw=0.6, zorder=3)
        if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
            right.hlines(idx, row["ci_low"], row["ci_high"], color=color, lw=1.4, zorder=4)
            right.plot([row["ci_low"], row["ci_high"]], [idx, idx], "|", color=color, ms=7)
    for _, row in eligible.iterrows():
        if len(eligible) <= max_labels or row["_state_id"] in highlighted:
            left.annotate(str(row["_state_id"]), (row["effect"], row["alignment"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=9)
    left.set(xlabel=effect_metric.replace("_", " "), ylabel="Stored alignment cosine",
             ylim=(-1.08, 1.08), title="Magnitude versus direction")
    right.set(xlabel=effect_metric.replace("_", " "), title="Eligible states (ranked)")
    right.set_yticks(y)
    right.set_yticklabels([str(s) for s in eligible._state_id], fontsize=9)
    right.invert_yaxis()
    right.set_xlim(left=0)
    if not len(eligible):
        left.text(0.5, 0.5, "No eligible paired estimates", transform=left.transAxes,
                  ha="center", va="center")
        right.text(0.5, 0.5, "No eligible states", transform=right.transAxes,
                   ha="center", va="center")
    left.grid(alpha=0.12)
    right.grid(axis="x", alpha=0.15)
    handles = [Line2D([], [], marker="o", ls="", color=color, label=caption)
               for caption, color in (
                   (f"Positive ≥ {positive_threshold:g}", _PALETTE["positive"]),
                   (f"Opposed ≤ {opposed_threshold:g}", _PALETTE["opposed"]),
                   ("Intermediate", _PALETTE["intermediate"])
               )]
    left.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    summary = f"{len(eligible)} eligible / {len(shifts)} states; {len(excluded)} not assessed"
    if sample_count_keys and all(c in shifts for c in sample_count_keys):
        count_values = shifts.loc[shifts.eligible, list(sample_count_keys)]
        if len(count_values):
            minima = [int(pd.to_numeric(count_values[c], errors="coerce").min())
                      for c in sample_count_keys]
            summary += "; minimum biological samples per condition: " + "/".join(map(str, minima))
    notes = [
        "Stored magnitude and alignment are descriptive; thresholds are visual classifications, not hypothesis tests.",
        "Embedding arrows are not used to compute these coordinates or alignment values.",
    ]
    if show_ci and not ci_available:
        notes.append("No metric-matched stored interval: intervals are not shown.")
    if len(excluded):
        notes.append(f"{len(excluded)} states lack paired eligible evidence; they remain in the exported table.")
    if owns_figure:
        fig.suptitle(title or "Geometry–velocity concordance", fontsize=14, y=0.99)
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color="#52616C")
        fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    metadata = {
        "plot": "geometry_velocity_concordance",
        "analysis_rep": analysis_rep,
        "state_key": state_key,
        "alignment_state_key": akey,
        "effect_metric": effect_metric,
        "alignment_metric": alignment_metric,
        "status_key": status_key if status_key in shifts else None,
        "usable_key": usable_key if usable_key in alignments else None,
        "positive_threshold": float(positive_threshold),
        "opposed_threshold": float(opposed_threshold),
        "n_states": len(shifts),
        "n_eligible": len(eligible),
        "n_not_assessed": len(excluded),
        "interval_columns": list(interval_columns) if ci_available else None,
        "sample_count_keys": [c for c in sample_count_keys if c in shifts],
        "quantitative_basis": "supplied stored analysis, not display embedding",
        "classification": "descriptive cosine intervals only",
    }
    result = PlotResult(
        fig=fig, axes=axes,
        data={"states": shifts, "eligible": eligible, "excluded": excluded},
        metadata=metadata, warnings=notes, artists=artists,
    )
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return fig, axes, result.data
    return fig, axes
