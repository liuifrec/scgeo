"""Interpret stored displacement and dynamics without recomputing analysis."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ._paga_layout import place_labels
from ._result import PlotResult
from ._stored_evidence import validate_stored_evidence
from ._table_plot_utils import panel_axes

_PALETTE = {
    "positive": "#177E89",
    "opposed": "#C45A55",
    "intermediate": "#8B8F9B",
    "not_assessed": "#7A8491",
}
_AVAILABLE = {
    "assessed",
    "available",
    "ok",
    "usable",
    "eligible",
    "eligible_exploratory",
    "zero_shift",
}
_DESCRIPTIVE = {"sample_descriptive", "cell_descriptive", "descriptive", "not_computed"}


def _identifiers(values, title):
    if values.isna().any():
        raise ValueError(f"{title} cannot contain missing state identifiers.")
    result = values.map(str)
    if result.duplicated().any():
        raise ValueError(f"{title} contains duplicate or string-colliding states.")
    return result


def _numeric(values, name):
    result = pd.to_numeric(values, errors="raise").astype(float)
    if np.isinf(result).any():
        raise ValueError(f"{name} must be finite or missing.")
    return result


def _minimum(frame, field, requested):
    """Retain a recorded support floor; a display request cannot relax it."""
    aliases = (
        (field, "min_samples_per_condition") if field == "min_samples" else (field,)
    )
    values = []
    for alias in aliases:
        if alias in frame:
            values.extend((value, f"column {alias}") for value in frame[alias].dropna())
    seen = set()

    def visit(mapping, path):
        if id(mapping) in seen:
            return
        seen.add(id(mapping))
        for alias in aliases:
            value = mapping.get(alias)
            if value is not None and not pd.isna(value):
                values.append((value, f"{path}.{alias}"))
        for key in ("params", "provenance", "source_params", "metadata", "analysis"):
            if isinstance(mapping.get(key), Mapping):
                visit(mapping[key], f"{path}.{key}")

    visit(frame.attrs, "attrs")

    def integer(value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{field} must be a positive integer or None.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer or None.") from exc
        if not np.isfinite(number) or number < 1 or number != int(number):
            raise ValueError(f"{field} must be a positive integer or None.")
        return int(number)

    recorded = [(integer(value), source) for value, source in values]
    unique = {value for value, _ in recorded}
    if len(unique) > 1:
        raise ValueError(f"Conflicting recorded {field} thresholds: {sorted(unique)}.")
    source_value = next(iter(unique), None)
    requested = integer(requested) if requested is not None else None
    candidates = [value for value in (source_value, requested) if value is not None]
    return {
        "recorded": source_value,
        "requested": requested,
        "effective": max(candidates) if candidates else None,
        "sources": sorted({source for _, source in recorded}),
    }


def _count_pair(frame, candidates):
    selected = [
        next((name for name in names if name in frame), None) for names in candidates
    ]
    values = [
        _numeric(frame[name], name) if name else pd.Series(np.nan, index=frame.index)
        for name in selected
    ]
    for value in values:
        known = value.notna()
        if (value[known] < 0).any() or (value[known] % 1 != 0).any():
            raise ValueError(
                "Recorded cell/sample counts must be nonnegative integers or missing."
            )
    return selected, values


def _cell_bootstrap_metadata(metadata, seen=None):
    """Recognize explicit cell resampling in the shift source's metadata."""
    seen = set() if seen is None else seen
    if not isinstance(metadata, Mapping) or id(metadata) in seen:
        return False
    seen.add(id(metadata))
    for key in ("bootstrap_unit", "resolved_bootstrap_unit"):
        value = metadata.get(key)
        if isinstance(value, str) and value == "cell":
            return True
    return any(
        _cell_bootstrap_metadata(metadata.get(key), seen)
        for key in (
            "params",
            "provenance",
            "source_params",
            "metadata",
            "analysis",
            "analysis_metadata",
            "comparison",
            "comparison_identity",
        )
    )


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
    condition_key=None,
    group0=None,
    group1=None,
    node_key=None,
    min_cells=None,
    min_samples=None,
    positive_threshold=0.5,
    opposed_threshold=-0.25,
    sample_count_keys=("n_mice_control", "n_mice_irradiated"),
    show_ci=True,
    highlight_states=None,
    max_labels=16,
    label_mode="auto",
    label_offsets=None,
    axes=None,
    figsize=(12.0, 5.6),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Show stored displacement and alignment with separate availability.

    Recorded provenance is checked within and between both tables. Optional
    representation/contrast declarations must agree with recorded metadata;
    missing metadata remains unverified. No embedding coordinates are used.

    The left panel shows states with both effects and alignment available. The
    right panel retains every state, including valid displacement estimates
    with missing velocity and named unavailable results. ``effect_status``
    takes precedence over a combined ``coverage_status`` when present.
    Optional ``min_cells``/``min_samples`` and recorded source support floors
    are enforced against recorded counts; unknown counts do not satisfy a
    requested floor. No count threshold is invented when none is supplied.

    Raw and normalized effects use only their own named stored intervals.
    Known descriptive-only results retain their estimates but not confidence
    intervals. These displays do not create hypotheses, bootstrap replicates,
    uncertainty or recovery claims. The thresholds classify cosine values
    descriptively, not statistically.

    ``label_key`` supplies display text while stable state IDs remain the join
    keys. ``max_labels`` is a soft budget: explicit highlights are retained,
    then remaining states are ordered by effect and stable ID. ``label_offsets``
    uses state IDs and point units. Caller-owned axes retain their layout.
    """
    if not isinstance(shift_df, pd.DataFrame) or not isinstance(
        alignment_df, pd.DataFrame
    ):
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
    if (
        isinstance(max_labels, bool)
        or not isinstance(max_labels, (int, np.integer))
        or max_labels < 0
    ):
        raise ValueError("max_labels must be a nonnegative integer.")
    if label_mode not in {"auto", "on_data"}:
        raise ValueError("label_mode must be 'auto' or 'on_data'.")
    if axes is not None and (
        not isinstance(axes, Mapping) or set(axes) != {"concordance", "effect"}
    ):
        raise ValueError("axes must map 'concordance' and 'effect' to two axes.")
    provenance, provenance_warnings = validate_stored_evidence(
        shift_df,
        alignment_df,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        analysis_rep=analysis_rep,
    )
    thresholds = {
        field: _minimum(shift_df, field, requested)
        for field, requested in (("min_cells", min_cells), ("min_samples", min_samples))
    }
    alignment_thresholds = {
        field: _minimum(alignment_df, field, requested)
        for field, requested in (("min_cells", min_cells), ("min_samples", min_samples))
    }
    shifts, alignments = shift_df.copy(deep=True), alignment_df.copy(deep=True)
    shifts["_state_id"] = _identifiers(shifts[state_key], "shift_df")
    alignments["_state_id"] = _identifiers(alignments[akey], "alignment_df")
    if label_key is not None and label_key not in shifts:
        raise KeyError(f"shift_df lacks label_key {label_key!r}.")
    extra = sorted(set(alignments._state_id) - set(shifts._state_id))
    if extra:
        shifts = pd.concat(
            [shifts, pd.DataFrame({state_key: extra, "_state_id": extra})],
            ignore_index=True,
        )
    else:
        shifts = shifts.reset_index(drop=True)
    selected_align = alignments.set_index("_state_id")
    shifts["alignment"] = _numeric(
        shifts._state_id.map(selected_align[alignment_metric]), alignment_metric
    )
    shifts["effect"] = _numeric(shifts[effect_metric], effect_metric)
    if (shifts.alignment.notna() & ~shifts.alignment.between(-1, 1)).any():
        raise ValueError("alignment cosine must be between -1 and 1.")
    if (shifts.effect.dropna() < 0).any():
        raise ValueError("Displacement magnitude cannot be negative.")
    shifts["display_label"] = shifts._state_id
    if label_key is not None:
        present = shifts[label_key].notna()
        shifts.loc[present, "display_label"] = shifts.loc[present, label_key].astype(
            str
        )
    if isinstance(highlight_states, (str, bytes)):
        raise TypeError("highlight_states must be a sequence of state IDs.")
    highlighted = (
        {str(value) for value in highlight_states}
        if highlight_states is not None
        else set()
    )
    unknown = highlighted - set(shifts._state_id)
    if unknown:
        raise ValueError(f"highlight_states contains unknown states: {sorted(unknown)}")
    if label_offsets is not None and not isinstance(label_offsets, Mapping):
        raise TypeError("label_offsets must map state IDs to finite point offsets.")
    offsets = {}
    for key, value in (label_offsets or {}).items():
        key = str(key)
        if key not in set(shifts._state_id):
            raise ValueError(f"label_offsets contains unknown state {key!r}.")
        value = np.asarray(value, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(
                "label_offsets must contain finite two-dimensional offsets."
            )
        offsets[key] = tuple(value)

    shifts["coverage_eligible"] = (
        shifts[status_key].isin(tuple(eligible_statuses))
        if status_key in shifts
        else True
    )
    if "effect_status" in shifts:
        effect_supported = shifts.effect_status.isin(
            _AVAILABLE | {"insufficient_samples"}
        )
        effect_status_source = "effect_status"
    elif "status" in shifts and status_key not in shifts:
        effect_supported = shifts.status.isin(_AVAILABLE | {"insufficient_samples"})
        effect_status_source = "status"
    else:
        effect_supported = shifts.coverage_eligible.copy()
        effect_status_source = status_key if status_key in shifts else None
    if usable_key in shifts:
        effect_supported &= shifts[usable_key].eq(True)
    cell_columns, cell_counts = _count_pair(
        shifts, (("n_cells0", "n0"), ("n_cells1", "n1"))
    )
    sample_keys = tuple(sample_count_keys or ())
    sample_columns, sample_counts = _count_pair(
        shifts,
        (
            ("n_samples0", "n_samples_group0", *sample_keys[:1]),
            ("n_samples1", "n_samples_group1", *sample_keys[1:2]),
        ),
    )
    shifts["effect_reason"] = np.where(
        effect_supported, "available", "effect_status_ineligible"
    )
    for kind, counts in (("cells", cell_counts), ("samples", sample_counts)):
        known = counts[0].notna() & counts[1].notna()
        shifts[f"{kind}_count_status"] = np.where(known, "recorded", "unverified")
        floor = thresholds[f"min_{kind}"]["effective"]
        if floor is not None:
            adequate = known & counts[0].ge(floor) & counts[1].ge(floor)
            missing_support = effect_supported & ~adequate
            shifts.loc[missing_support, "effect_reason"] = np.where(
                known[missing_support],
                f"insufficient_{kind}",
                f"{kind}_counts_unverified",
            )
            effect_supported &= adequate
    absent_cells = cell_counts[0].eq(0) | cell_counts[1].eq(0)
    shifts.loc[effect_supported & absent_cells, "effect_reason"] = (
        "absent_condition_cells"
    )
    effect_supported &= ~absent_cells
    shifts.loc[~np.isfinite(shifts.effect), "effect_reason"] = "effect_missing"
    shifts["effect_eligible"] = effect_supported & np.isfinite(shifts.effect)
    shifts["alignment_usable"] = (
        shifts._state_id.map(selected_align[usable_key].eq(True))
        .fillna(False)
        .astype(bool)
        if usable_key in alignments
        else shifts._state_id.isin(selected_align.index)
    )
    if "alignment_status" in alignments:
        shifts["alignment_usable"] &= shifts._state_id.map(
            selected_align.alignment_status
        ).isin(_AVAILABLE)
    elif "status" in alignments:
        shifts["alignment_usable"] &= shifts._state_id.map(selected_align.status).isin(
            _AVAILABLE
        )
    alignment_rows = selected_align.reindex(shifts._state_id.to_numpy()).reset_index(
        drop=True
    )
    alignment_cell_columns, alignment_cell_counts = _count_pair(
        alignment_rows, (("n_cells0", "n0"), ("n_cells1", "n1"))
    )
    alignment_sample_columns, alignment_sample_counts = _count_pair(
        alignment_rows,
        (
            ("n_samples0", "n_samples_group0", *sample_keys[:1]),
            ("n_samples1", "n_samples_group1", *sample_keys[1:2]),
        ),
    )
    alignment_supported = shifts.alignment_usable.copy()
    alignment_support_reason = pd.Series("available", index=shifts.index)
    for kind, counts in (
        ("cells", alignment_cell_counts),
        ("samples", alignment_sample_counts),
    ):
        known = counts[0].notna() & counts[1].notna()
        shifts[f"alignment_{kind}_count_status"] = np.where(
            known, "recorded", "unverified"
        )
        floor = alignment_thresholds[f"min_{kind}"]["effective"]
        if floor is not None:
            adequate = known & counts[0].ge(floor) & counts[1].ge(floor)
            unsupported = alignment_supported & ~adequate
            alignment_support_reason.loc[unsupported] = np.where(
                known[unsupported],
                f"alignment_insufficient_{kind}",
                f"alignment_{kind}_counts_unverified",
            )
            alignment_supported &= adequate
    absent_alignment_cells = alignment_cell_counts[0].eq(0) | alignment_cell_counts[
        1
    ].eq(0)
    alignment_support_reason.loc[alignment_supported & absent_alignment_cells] = (
        "alignment_absent_condition_cells"
    )
    alignment_supported &= ~absent_alignment_cells
    shifts["alignment_support_eligible"] = alignment_supported
    zero_shift = shifts.effect.eq(0)
    for column in ("delta_norm", "normalized_delta_norm"):
        for frame, values in (
            (shifts, shifts.get(column)),
            (
                alignments,
                shifts._state_id.map(selected_align[column])
                if column in alignments
                else None,
            ),
        ):
            if column in frame:
                magnitudes = _numeric(values, column)
                if (magnitudes.dropna() < 0).any():
                    raise ValueError("Displacement magnitude cannot be negative.")
                zero_shift |= magnitudes.eq(0)
    for column in ("effect_status", "status"):
        if column in shifts:
            zero_shift |= shifts[column].eq("zero_shift")
        if column in alignments:
            zero_shift |= shifts._state_id.map(selected_align[column]).eq("zero_shift")
    zero_velocity = pd.Series(False, index=shifts.index)
    velocity_norm_key = next(
        (key for key in ("velocity_norm", "vel_norm") if key in alignments), None
    )
    if velocity_norm_key is not None:
        velocity_norm = _numeric(
            shifts._state_id.map(selected_align[velocity_norm_key]), velocity_norm_key
        )
        if (velocity_norm.dropna() < 0).any():
            raise ValueError("Velocity magnitude cannot be negative.")
        zero_velocity = velocity_norm.eq(0)
    shifts["alignment_eligible"] = (
        alignment_supported
        & np.isfinite(shifts.alignment)
        & ~zero_shift
        & ~zero_velocity
    )
    shifts["alignment_reason"] = np.select(
        [
            zero_shift,
            zero_velocity,
            ~np.isfinite(shifts.alignment),
            ~shifts.alignment_usable,
            ~alignment_supported,
        ],
        [
            "zero_shift",
            "zero_velocity",
            "alignment_missing",
            "alignment_unusable",
            alignment_support_reason,
        ],
        default="available",
    )
    shifts["eligible"] = shifts.effect_eligible & shifts.alignment_eligible
    shifts["reason"] = np.where(
        ~shifts.effect_eligible,
        shifts.effect_reason,
        np.where(~shifts.alignment_eligible, shifts.alignment_reason, "eligible"),
    )
    shifts["alignment_class"] = "not_assessed"
    shifts.loc[shifts.eligible, "alignment_class"] = np.select(
        [
            shifts.loc[shifts.eligible, "alignment"] >= positive_threshold,
            shifts.loc[shifts.eligible, "alignment"] <= opposed_threshold,
        ],
        ["positive", "opposed"],
        default="intermediate",
    )
    descriptive = pd.Series(False, index=shifts.index)
    if "descriptive_only" in shifts:
        descriptive |= shifts.descriptive_only.eq(True)
    if "inference_level" in shifts:
        descriptive |= shifts.inference_level.isin(_DESCRIPTIVE)
    if "effect_status" in shifts:
        descriptive |= shifts.effect_status.eq("insufficient_samples")
    if "status" in shifts:
        descriptive |= shifts.status.eq("insufficient_samples")
    descriptive |= sample_counts[0].lt(2) | sample_counts[1].lt(2)
    cell_bootstrap = pd.Series(
        _cell_bootstrap_metadata(shift_df.attrs), index=shifts.index
    )
    for column in ("bootstrap_unit", "resolved_bootstrap_unit"):
        if column in shifts:
            cell_bootstrap |= shifts[column].eq("cell").fillna(False)
    shifts["cell_bootstrap_known"] = cell_bootstrap
    descriptive |= cell_bootstrap
    shifts["descriptive_only_known"] = descriptive

    interval_columns = {
        "delta_norm": ("magnitude_ci95_low", "magnitude_ci95_high"),
        "normalized_delta_norm": (
            "normalized_magnitude_ci95_low",
            "normalized_magnitude_ci95_high",
        ),
    }.get(effect_metric)
    has_intervals = bool(
        interval_columns and all(column in shifts for column in interval_columns)
    )
    shifts["ci_low"], shifts["ci_high"] = np.nan, np.nan
    shifts["interval_reason"] = "metric_matched_interval_missing"
    if has_intervals:
        low, high = (_numeric(shifts[column], column) for column in interval_columns)
        finite = low.notna() & high.notna()
        if (finite & (low > high)).any():
            raise ValueError("Finite stored interval endpoints must be ordered.")
        available = finite & shifts.effect_eligible & ~descriptive & bool(show_ci)
        shifts.loc[available, "ci_low"] = low[available]
        shifts.loc[available, "ci_high"] = high[available]
        shifts["interval_reason"] = np.select(
            [
                ~shifts.effect_eligible,
                descriptive,
                ~finite,
                pd.Series(not show_ci, index=shifts.index),
            ],
            [
                "effect_unavailable",
                "descriptive_only",
                "stored_interval_missing",
                "display_disabled",
            ],
            default="available",
        )
    shifts["interval_displayed"] = shifts.ci_low.notna() & shifts.ci_high.notna()
    order = shifts.sort_values(
        ["effect_eligible", "effect", "_state_id"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    )
    eligible = order.loc[order.eligible].copy()
    excluded = order.loc[~order.eligible].copy()
    highlighted_eligible = eligible.loc[eligible._state_id.isin(highlighted)]
    ordinary = eligible.loc[~eligible._state_id.isin(highlighted)]
    label_ids = list(highlighted_eligible._state_id) + list(
        ordinary._state_id.head(max(0, max_labels - len(highlighted_eligible)))
    )
    omitted_labels = [
        state for state in eligible._state_id if state not in set(label_ids)
    ]
    shifts["label_displayed"] = shifts._state_id.isin(label_ids)

    fig, axes, owns_figure = panel_axes(["concordance", "effect"], axes, figsize)
    left, right = axes["concordance"], axes["effect"]
    artists = {
        "points": [],
        "effects": [],
        "intervals": [],
        "unavailable": [],
        "labels": {},
    }
    left.axhline(0, color="#9CA3AF", lw=1, ls="--", zorder=0)
    for value, color in (
        (positive_threshold, _PALETTE["positive"]),
        (opposed_threshold, _PALETTE["opposed"]),
    ):
        left.axhline(value, color=color, lw=0.8, ls=":", alpha=0.7)
    for _, row in eligible.iterrows():
        artists["points"].append(
            left.scatter(
                row.effect,
                row.alignment,
                s=75,
                color=_PALETTE[row.alignment_class],
                marker="D" if row._state_id in highlighted else "o",
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        )
    for position, (_, row) in enumerate(order.iterrows()):
        color = _PALETTE[row.alignment_class]
        if row.effect_eligible:
            right.hlines(position, 0, row.effect, color=color, alpha=0.65, lw=2)
            artists["effects"].append(
                right.scatter(
                    row.effect,
                    position,
                    s=55,
                    color=color,
                    marker="D" if row._state_id in highlighted else "o",
                    edgecolor="white",
                    lw=0.6,
                    zorder=3,
                )
            )
            if row.interval_displayed:
                artists["intervals"].append(
                    right.hlines(
                        position, row.ci_low, row.ci_high, color=color, lw=1.4, zorder=4
                    )
                )
                right.plot(
                    [row.ci_low, row.ci_high],
                    [position, position],
                    "|",
                    color=color,
                    ms=7,
                )
        else:
            artists["unavailable"].append(
                right.text(
                    0.025,
                    position,
                    str(row.effect_reason).replace("_", " "),
                    transform=right.get_yaxis_transform(),
                    va="center",
                    fontsize=8,
                    color="#66717D",
                )
            )
    labels = []
    for _, row in order.iterrows():
        caption = row.display_label
        if row.effect_eligible and not row.alignment_eligible:
            caption += " · alignment unavailable"
        if row.descriptive_only_known and row.effect_eligible:
            caption += " · descriptive"
        labels.append(caption)
    left.set(
        xlabel=effect_metric.replace("_", " "),
        ylabel="Stored alignment cosine",
        ylim=(-1.08, 1.08),
        title="Magnitude versus alignment",
    )
    right.set(
        xlabel=effect_metric.replace("_", " "), title="Displacement and availability"
    )
    right.set_yticks(np.arange(len(order)))
    right.set_yticklabels(labels, fontsize=9)
    right.set_ylim(max(len(order) - 0.5, 0.5), -0.5)
    right.set_xlim(left=0)
    if not len(eligible):
        left.text(
            0.5,
            0.5,
            "No states with both estimates available",
            transform=left.transAxes,
            ha="center",
            va="center",
        )
    if not len(order):
        right.text(
            0.5,
            0.5,
            "No stored states",
            transform=right.transAxes,
            ha="center",
            va="center",
        )
    left.grid(alpha=0.12)
    right.grid(axis="x", alpha=0.15)
    handles = [
        Line2D([], [], marker="o", ls="", color=_PALETTE[key], label=caption)
        for key, caption in (
            ("positive", f"Positive ≥ {positive_threshold:g}"),
            ("opposed", f"Opposed ≤ {opposed_threshold:g}"),
            ("intermediate", "Intermediate"),
            ("not_assessed", "Alignment unavailable"),
        )
    ]
    left.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    summary = f"{int(shifts.effect_eligible.sum())} displacement estimates; {len(eligible)} with alignment; {len(excluded)} not jointly assessed"
    if owns_figure:
        fig.suptitle(title or "Geometry–velocity concordance", fontsize=14, y=0.99)
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color="#52616C")
        fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    chosen = eligible.set_index("_state_id").reindex(label_ids)
    label_artists, placement = place_labels(
        left,
        [
            (state, float(row.effect), float(row.alignment))
            for state, row in chosen.iterrows()
        ],
        fontsize=9,
        mode=label_mode,
        offsets=offsets,
        obstacles=[
            (float(row.effect), float(row.alignment), 5)
            for _, row in eligible.iterrows()
        ],
        labels=dict(zip(shifts._state_id, shifts.display_label)),
    )
    artists["labels"] = label_artists
    placement["display_label"] = placement.state.map(
        dict(zip(shifts._state_id, shifts.display_label))
    )
    notes = [
        *provenance_warnings,
        "Stored magnitude and alignment are descriptive; cosine thresholds are not hypothesis tests.",
        "Embedding arrows are not used to compute these coordinates or alignment values.",
    ]
    if show_ci and not has_intervals:
        notes.append("No metric-matched stored interval: intervals are not shown.")
    if descriptive.any():
        notes.append(
            "Known descriptive-only results retain their effect estimates; confidence intervals are withheld."
        )
    if len(excluded):
        notes.append(
            f"{len(excluded)} states lack jointly eligible evidence and remain visible in the displacement panel."
        )
    if (shifts.samples_count_status == "unverified").any():
        notes.append(
            "Biological sample counts are missing for some states; replication is unverified."
        )
    metadata = {
        "plot": "geometry_velocity_concordance",
        "analysis_rep": provenance["analysis_rep"],
        "provenance": provenance,
        "state_key": state_key,
        "alignment_state_key": akey,
        "effect_metric": effect_metric,
        "alignment_metric": alignment_metric,
        "status_key": status_key if status_key in shifts else None,
        "effect_status_source": effect_status_source,
        "usable_key": usable_key if usable_key in alignments else None,
        "positive_threshold": float(positive_threshold),
        "opposed_threshold": float(opposed_threshold),
        "n_states": len(shifts),
        "n_eligible": len(eligible),
        "n_not_assessed": len(excluded),
        "n_effect_available": int(shifts.effect_eligible.sum()),
        "n_alignment_available": int(shifts.alignment_eligible.sum()),
        "interval_columns": list(interval_columns)
        if show_ci and has_intervals
        else None,
        "interval_source_columns": list(interval_columns) if has_intervals else None,
        "n_intervals_displayed": int(shifts.interval_displayed.sum()),
        "sample_count_keys": [name for name in sample_columns if name],
        "cell_count_keys": [name for name in cell_columns if name],
        "support_thresholds": thresholds,
        "alignment_support_thresholds": alignment_thresholds,
        "alignment_sample_count_keys": [
            name for name in alignment_sample_columns if name
        ],
        "alignment_cell_count_keys": [name for name in alignment_cell_columns if name],
        "label_key": label_key,
        "label_mode": label_mode,
        "max_labels": int(max_labels),
        "labelled_states": label_ids,
        "omitted_labels": omitted_labels,
        "highlight_label_overflow": max(0, len(highlighted_eligible) - max_labels),
        "highlighted_unavailable": sorted(highlighted - set(eligible._state_id)),
        "quantitative_basis": "supplied stored analysis, not display embedding",
        "classification": "descriptive cosine intervals only",
        "owns_figure": owns_figure,
    }
    ranked = shifts.loc[order.index]
    result = PlotResult(
        fig=fig,
        axes=axes,
        data={
            "states": shifts,
            "eligible": ranked.loc[ranked.eligible].copy(),
            "excluded": ranked.loc[~ranked.eligible].copy(),
            "effect_available": ranked.loc[ranked.effect_eligible].copy(),
            "display_order": order[["_state_id", "display_label"]].reset_index(
                drop=True
            ),
            "label_placement": placement,
            "alignment_source": alignments,
        },
        metadata=metadata,
        warnings=notes,
        artists=artists,
    )
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return fig, axes, result.data
    return fig, axes
