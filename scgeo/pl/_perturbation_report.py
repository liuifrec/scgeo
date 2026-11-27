from __future__ import annotations

import json
import re
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..get import state_report

_CONSENSUS_PALETTE = {
    "stable_aligned": "#0072B2",
    "stable_discordant": "#D55E00",
    "stable_neutral": "#009E73",
    "stable_effect": "#25769A",
    "representation_unstable": "#CC79A7",
    "insufficient_coverage": "#7F7F7F",
    "missing": "#BDBDBD",
}

_CLASS_TO_VALUE = {
    "discordant": -1.0,
    "neutral": 0.0,
    "aligned": 1.0,
    "missing": np.nan,
}


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "matplotlib is required for ScGeo plotting functions. "
            "Install it with `pip install matplotlib` or `pip install scgeo[plot]`."
        ) from exc
    return plt, ListedColormap


def _embedding_key(adata, basis: str) -> str:
    key = str(basis)
    if key in adata.obsm:
        return key
    x_key = f"X_{key}"
    if x_key in adata.obsm:
        return x_key
    raise KeyError(
        f"Embedding {basis!r} not found as adata.obsm[{key!r}] or adata.obsm[{x_key!r}]."
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_column(df: pd.DataFrame, column: str) -> bool:
    return column in df and np.isfinite(_numeric(df[column])).any()


def _alignment_thresholds(report: pd.DataFrame) -> tuple[float, float]:
    rules = (
        report.attrs.get("scgeo_report_rules", {}) if hasattr(report, "attrs") else {}
    )
    thresholds = (
        rules.get("alignment_thresholds", {}) if isinstance(rules, dict) else {}
    )
    return (
        float(thresholds.get("alignment_pos_thr", 0.3)),
        float(thresholds.get("alignment_neg_thr", -0.3)),
    )


def _ordered_labels(existing, requested, *, name: str):
    values = list(existing)
    if requested is None:
        return values
    requested = list(requested)
    if len(set(requested)) != len(requested):
        raise ValueError(f"{name} must not contain duplicate labels.")
    unknown = [value for value in requested if value not in values]
    if unknown:
        raise ValueError(f"{name} contains unknown labels: {unknown!r}.")
    return requested + [value for value in values if value not in requested]


def _sorted_report(report, sort_by, max_states, *, state_order=None, page=0):
    out = report.copy()
    if "state" not in out:
        out["state"] = out.index.astype(str)
    if out["state"].isna().any() or out["state"].duplicated().any():
        raise ValueError("The report must contain one non-missing row per state.")
    if sort_by in out:
        values = _numeric(out[sort_by])
        out = out.assign(_sort_value=values).sort_values(
            ["_sort_value", "state"],
            ascending=[False, True],
            na_position="last",
        )
        out = out.drop(columns=["_sort_value"])
    if state_order is not None:
        order = _ordered_labels(out["state"], state_order, name="state_order")
        out = out.set_index("state", drop=False).loc[order]
    if isinstance(page, bool) or not isinstance(page, (int, np.integer)) or page < 0:
        raise ValueError("page must be a non-negative integer.")
    if max_states is not None:
        if (
            isinstance(max_states, bool)
            or not isinstance(max_states, (int, np.integer))
            or max_states < 1
        ):
            raise ValueError("max_states must be a positive integer or None.")
        out = out.iloc[page * max_states : (page + 1) * max_states]
    elif page:
        raise ValueError("page requires max_states to be specified.")
    if out.empty:
        raise ValueError("The requested page contains no states.")
    return out.reset_index(drop=True)


def _supplied_axes(axes, names):
    if not isinstance(axes, Mapping):
        raise TypeError("axes must be a mapping from panel names to Matplotlib axes.")
    missing = [name for name in names if name not in axes]
    if missing:
        raise ValueError(f"axes is missing required panels: {missing!r}.")
    selected = {name: axes[name] for name in names}
    figures = {id(ax.figure) for ax in selected.values()}
    if len(figures) != 1:
        raise ValueError("All supplied axes must belong to the same figure.")
    if len({id(ax) for ax in selected.values()}) != len(selected):
        raise ValueError("Each panel requires a distinct axis.")
    return next(iter(selected.values())).figure, selected


def _availability_value(rec, key):
    value = rec.get(key, "not_supplied")
    if pd.isna(value):
        return "not_supplied"
    return str(value)


def _effect_intervals(data, metric, show_ci):
    """Validate endpoints before allocating or decorating any figure."""
    if not show_ci or metric is None:
        return None, []
    keys = (
        ("magnitude_ci95_low", "magnitude_ci95_high")
        if metric == "delta_norm"
        else ("normalized_magnitude_ci95_low", "normalized_magnitude_ci95_high")
    )
    if all(key in data for key in keys):
        low, high = [
            _numeric(data[key]).to_numpy(dtype=float, na_value=np.nan) for key in keys
        ]
        finite = np.isfinite(low) & np.isfinite(high)
        invalid = finite & (low > high)
        if invalid.any():
            raise ValueError(
                "Interval lower endpoint exceeds upper endpoint for states: "
                f"{data.loc[invalid, 'state'].tolist()!r}."
            )
        incomplete = (
            (np.isfinite(low) != np.isfinite(high)) | np.isinf(low) | np.isinf(high)
        )
        messages = (
            [
                "Incomplete or non-finite interval endpoints were omitted; estimates are retained."
            ]
            if incomplete.any()
            else []
        )
        return (low, high, finite), messages
    if any(key in data for key in keys):
        missing = [key for key in keys if key not in data]
        return None, [
            f"Interval unavailable because endpoint columns are missing: {missing!r}; estimates are retained."
        ]
    if metric == "normalized_delta_norm" and any(
        _finite_column(data, key)
        for key in ("magnitude_ci95_low", "magnitude_ci95_high")
    ):
        return None, [
            "Raw magnitude intervals were omitted because the displayed metric is normalized_delta_norm."
        ]
    return None, []


def _show_or_close(plt, show: bool) -> None:
    if show:
        plt.show()


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return value or "comparison"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    if isinstance(value, tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if not isinstance(value, (str, bytes, list, dict, tuple)):
        try:
            if value is pd.NA or pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _records_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return pd.DataFrame(payload["records"], columns=payload.get("columns"))
    return pd.DataFrame()


def _comparison_from_report(report: pd.DataFrame) -> str:
    provenance = report.attrs.get("provenance", {}) if hasattr(report, "attrs") else {}
    if isinstance(provenance, dict):
        return str(provenance.get("comparison_label", "comparison"))
    return "comparison"


def _alt_text_state_evidence(
    report: pd.DataFrame, tracks: Sequence[str], comparison: str
) -> str:
    n_states = int(report.shape[0])
    stable = (
        report.loc[
            report.get("representation_consensus_label", pd.Series(dtype=object))
            .astype(str)
            .str.startswith("stable_"),
            "state",
        ]
        .astype(str)
        .tolist()
        if "representation_consensus_label" in report and "state" in report
        else []
    )
    unstable = (
        report.loc[
            report.get("representation_consensus_label", pd.Series(dtype=object))
            .astype(str)
            .isin(["representation_unstable", "insufficient_coverage"]),
            "state",
        ]
        .astype(str)
        .tolist()
        if "representation_consensus_label" in report and "state" in report
        else []
    )
    possible = [
        "Effect",
        "Stability",
        "Local preservation",
        "Local distortion",
        "Dynamics",
        "Coverage",
    ]
    omitted = [track for track in possible if track not in tracks]
    return (
        f"State evidence panel for {comparison} with {n_states} states. "
        f"Stable states: {', '.join(stable[:5]) if stable else 'none'}. "
        f"Unstable or insufficient states: {', '.join(unstable[:5]) if unstable else 'none'}. "
        f"Omitted evidence tracks: {', '.join(omitted) if omitted else 'none'}."
    )


def _alt_text_heatmap(
    matrix: pd.DataFrame, annotations: pd.DataFrame, comparison: str
) -> str:
    unstable = []
    if "consensus_label" in annotations:
        unstable = (
            annotations.index[
                annotations["consensus_label"]
                .astype(str)
                .isin(["representation_unstable", "insufficient_coverage"])
            ]
            .astype(str)
            .tolist()
        )
    return (
        f"Representation stability heatmap for {comparison} with {matrix.shape[0]} states "
        f"and {matrix.shape[1]} representations. "
        f"Unstable or insufficient states: {', '.join(unstable[:5]) if unstable else 'none'}. "
        "Missing values are shown separately from numerical zero."
    )


def _alt_text_consensus_map(report: pd.DataFrame, comparison: str, basis: str) -> str:
    n_states = int(report.shape[0])
    unstable = (
        report.loc[
            report.get("representation_consensus_label", pd.Series(dtype=object))
            .astype(str)
            .isin(["representation_unstable", "insufficient_coverage"]),
            "state",
        ]
        .astype(str)
        .tolist()
        if "representation_consensus_label" in report and "state" in report
        else []
    )
    return (
        f"Consensus state map for {comparison} using {basis} as a display embedding only; "
        "consensus was calculated across representations. "
        f"The report contains {n_states} states. "
        f"Unstable or insufficient states: {', '.join(unstable[:5]) if unstable else 'none'}."
    )


def state_evidence_panel(
    report_or_adata,
    *,
    node_key=None,
    effect_metric: str = "auto",
    representation=None,
    sort_by=None,
    state_order=None,
    max_states: Optional[int] = 25,
    page: int = 0,
    show_ci: bool = True,
    show_coverage: bool = True,
    show_sample_counts: bool = False,
    show_inference: bool = False,
    evidence_mode: str = "compact",
    axes=None,
    label_wrap: Optional[int] = 32,
    figsize=None,
    title=None,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """Plot effect estimates and independently assessed evidence by state.

    ``effect_metric='auto'`` prefers raw ``delta_norm`` when finite values
    exist; otherwise it uses ``normalized_delta_norm``. Sorting defaults to
    the displayed metric. Raw ``magnitude_ci95_low/high`` intervals are drawn
    only with the raw metric. Normalized intervals require separately supplied
    ``normalized_magnitude_ci95_low/high`` columns in the same units.

    ``evidence_mode='full'`` adds an availability matrix and textual consensus
    labels. Sample support is enabled with ``show_sample_counts``; inference
    labels with ``show_inference``. Neither option performs a new analysis.
    ``state_order`` moves specified states first and retains remaining states.
    ``max_states=None`` displays all states; ``page`` is zero-based.

    Supplied ``axes`` use lowercase keys: ``effect``, ``samples``,
    ``stability``, ``local_preservation``, ``local_distortion``, ``dynamics``,
    ``coverage``, ``availability`` or ``summary``, as applicable. Supplied
    figures are not resized or laid out. Legacy returns are preserved;
    ``return_result=True`` returns a :class:`scgeo.pl.PlotResult`.
    """
    plt, _ = _lazy_matplotlib()
    if evidence_mode not in {"compact", "full"}:
        raise ValueError("evidence_mode must be 'compact' or 'full'.")
    if label_wrap is not None and (
        isinstance(label_wrap, bool)
        or not isinstance(label_wrap, (int, np.integer))
        or label_wrap < 1
    ):
        raise ValueError("label_wrap must be a positive integer or None.")
    is_table = isinstance(report_or_adata, pd.DataFrame)
    report = (
        report_or_adata.copy()
        if is_table
        else state_report(report_or_adata, node_key=node_key)
    )
    if report.empty:
        raise ValueError("state_evidence_panel requires at least one report row.")
    if effect_metric not in {"auto", "delta_norm", "normalized_delta_norm"}:
        raise ValueError(
            "effect_metric must be 'auto', 'delta_norm', or 'normalized_delta_norm'."
        )
    if effect_metric == "auto":
        metric = next(
            (
                key
                for key in ("delta_norm", "normalized_delta_norm")
                if _finite_column(report, key)
            ),
            None,
        )
        metric = metric or next(
            (key for key in ("delta_norm", "normalized_delta_norm") if key in report),
            None,
        )
    else:
        metric = effect_metric
        if metric not in report:
            raise ValueError(f"The report does not contain effect metric {metric!r}.")
    provenance = dict(report.attrs.get("provenance", {}) or {})
    # Explicit uniform contrast columns may accompany exported CSV reports.
    # Never infer labels when mixed/unknown, and never override stored provenance.
    for key in ("group0", "group1"):
        if key not in provenance and key in report:
            values = report[key].dropna().unique()
            if len(values) == 1 and report[key].notna().all():
                provenance[key] = values[0]
    if representation is None:
        representation = provenance.get("effect_representation", provenance.get("rep"))
        if representation is None and not is_table:
            shift_key = provenance.get("source_store_keys", {}).get(
                "robust_shift", "robust_shift"
            )
            representation = (
                report_or_adata.uns.get("scgeo", {})
                .get(shift_key, {})
                .get("params", {})
                .get("rep")
            )
    resolved_sort = sort_by if sort_by is not None else metric
    data = _sorted_report(
        report, resolved_sort, max_states, state_order=state_order, page=page
    )
    pos_thr, neg_thr = _alignment_thresholds(report)
    tracks = []
    if metric is not None:
        tracks.append("Effect")
    if show_sample_counts or show_inference:
        tracks.append("Samples")
    has_consensus = (
        "representation_consensus_label" in data
        and data["representation_consensus_label"].notna().any()
    )
    if has_consensus or _finite_column(data, "representation_coverage_fraction"):
        tracks.append("Stability")
    if any(
        _finite_column(data, key)
        for key in ("median_neighbor_overlap", "median_neighbor_jaccard")
    ):
        tracks.append("Local preservation")
    if any(
        _finite_column(data, key)
        for key in ("median_local_shape_distortion", "median_global_scale_distortion")
    ):
        tracks.append("Local distortion")
    if _finite_column(data, "median_alignment_cosine"):
        tracks.append("Dynamics")
    if evidence_mode == "full":
        tracks.append("Availability")
    elif show_coverage and any(
        key in data for key in ("coverage_status", "warnings", "reason_codes")
    ):
        tracks.append("Coverage")
    if not tracks:
        tracks.append("Summary")
    intervals, warnings_out = _effect_intervals(data, metric, show_ci)
    n = len(data)
    names = [track.lower().replace(" ", "_") for track in tracks]
    owns_figure = axes is None
    if owns_figure:
        fig, axis_array = plt.subplots(
            1,
            len(tracks),
            sharey=True,
            squeeze=False,
            figsize=figsize
            or (
                max(7.5, 2.5 * len(tracks)),
                max(3.8, 0.60 * n + 2.3)
                if evidence_mode == "full" or show_inference
                else max(2.8, 0.42 * n + 1.6),
            ),
        )
        axis_map = dict(zip(names, axis_array[0]))
    else:
        fig, axis_map = _supplied_axes(axes, names)
    y = np.arange(n)
    labels = data["state"].astype(str).tolist()
    if label_wrap is not None:
        labels = [
            textwrap.fill(label, width=label_wrap, break_long_words=False)
            for label in labels
        ]
    artists = {}
    availability = pd.DataFrame({"state": data["state"]})
    for name, key in (
        ("effect", "effect_status"),
        ("representation", "stability_status"),
        ("local", "local_geometry_status"),
        ("dynamics", "dynamics_status"),
    ):
        availability[name] = [
            _availability_value(rec, key) for _, rec in data.iterrows()
        ]
    for index, (track, name) in enumerate(zip(tracks, names)):
        ax = axis_map[name]
        ax.set_title(
            "Representation evidence"
            if track == "Stability" and evidence_mode == "full"
            else track
        )
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(labelleft=False)
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        if index:
            ax.spines["left"].set_visible(False)
        if track in {"Samples", "Availability", "Coverage", "Summary"} or (
            track == "Stability" and evidence_mode == "full"
        ):
            ax.spines["bottom"].set_visible(False)
        for row in range(0, n, 2):
            ax.axhspan(row - 0.5, row + 0.5, color="#F1F5F7", zorder=-3)
        if track == "Effect":
            x = _numeric(data[metric]).to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(x)
            if intervals is not None:
                low, high, finite_interval = intervals
                valid = finite & finite_interval
                artists["effect_intervals"] = ax.hlines(
                    y[valid],
                    low[valid],
                    high[valid],
                    color="#25769A",
                    linewidth=1.8,
                    zorder=2,
                )
                if valid.any():
                    ax.vlines(
                        np.r_[low[valid], high[valid]],
                        np.r_[y[valid] - 0.08, y[valid] - 0.08],
                        np.r_[y[valid] + 0.08, y[valid] + 0.08],
                        color="#25769A",
                        linewidth=1,
                    )
            artists["effect_estimates"] = ax.scatter(
                x[finite],
                y[finite],
                color="#25769A",
                s=30,
                edgecolors="white",
                linewidth=0.5,
                zorder=3,
            )
            for row in np.flatnonzero(~finite):
                status = _availability_value(data.iloc[row], "effect_status").replace(
                    "_", " "
                )
                ax.text(
                    0.02,
                    y[row],
                    status,
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    color="0.45",
                    fontsize=8,
                )
            label = (
                "Raw displacement magnitude"
                if metric == "delta_norm"
                else "Normalized displacement magnitude"
            )
            if representation is not None:
                label += f"\n{representation}"
            ax.set_xlabel(label)
            ax.axvline(0, color="0.5", linewidth=0.7)
        elif track == "Samples":
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            groups = [
                str(provenance[key])
                if key in provenance and pd.notna(provenance[key])
                else default
                for key, default in (("group0", "group0"), ("group1", "group1"))
            ]
            ax.set_title("Samples\n" + " / ".join(groups))
            for row, rec in data.iterrows():
                count_values = [
                    pd.to_numeric(rec.get(key, np.nan), errors="coerce")
                    for key in ("n_samples0", "n_samples1")
                ]
                count = " / ".join(
                    f"{value:g}" if pd.notna(value) and np.isfinite(value) else "—"
                    for value in count_values
                )
                if show_sample_counts:
                    ax.text(
                        0.5,
                        row - (0.16 if show_inference else 0),
                        count,
                        ha="center",
                        va="center",
                        weight="bold",
                        fontsize=10,
                    )
                if show_inference:
                    unit = rec.get("bootstrap_unit")
                    inference = rec.get("inference_level")
                    label = (
                        f"{unit} bootstrap"
                        if pd.notna(unit) and unit
                        else "unit not supplied"
                    )
                    if pd.notna(inference) and inference:
                        label += "\n" + str(inference).replace("_", " ")
                    ax.text(
                        0.5,
                        row + (0.22 if show_sample_counts else 0),
                        label,
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="0.4",
                    )
            ax.grid(False)
        elif track == "Stability":
            x = _numeric(
                data.get(
                    "representation_coverage_fraction",
                    pd.Series(np.nan, index=data.index),
                )
            ).to_numpy(dtype=float, na_value=np.nan)
            consensus_labels = (
                data.get(
                    "representation_consensus_label",
                    pd.Series("missing", index=data.index),
                )
                .fillna("missing")
                .astype(str)
            )
            colors = [
                _CONSENSUS_PALETTE.get(label, _CONSENSUS_PALETTE["missing"])
                for label in consensus_labels
            ]
            if evidence_mode == "full":
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.grid(False)
                for row, label in enumerate(consensus_labels):
                    ax.text(
                        0.02,
                        row - 0.10,
                        label.replace("_", " "),
                        color=colors[row],
                        va="center",
                        fontsize=8,
                        weight="bold",
                    )
                    count = pd.to_numeric(
                        data.iloc[row].get("representation_coverage_count", np.nan),
                        errors="coerce",
                    )
                    usable = (
                        f"{x[row]:.0%} usable"
                        if np.isfinite(x[row])
                        else "coverage not supplied"
                    )
                    if pd.notna(count) and np.isfinite(count):
                        usable = f"{count:g} usable views · " + usable
                    ax.text(
                        0.02, row + 0.20, usable, color="0.4", fontsize=7, va="center"
                    )
            else:
                artists["representation_coverage"] = ax.scatter(
                    x, y, color=colors, s=34
                )
                ax.set_xlim(-0.03, 1.03)
                ax.set_xlabel("Usable fraction")
        elif track in {"Local preservation", "Local distortion"}:
            specs = (
                (
                    ("median_neighbor_overlap", "Overlap", "#009E73", "o"),
                    ("median_neighbor_jaccard", "Jaccard", "#56B4E9", "s"),
                )
                if track == "Local preservation"
                else (
                    ("median_local_shape_distortion", "Local", "#E69F00", "o"),
                    ("median_global_scale_distortion", "Global", "#D55E00", "s"),
                    ("worst_local_shape_distortion", "Worst local", "#000000", "x"),
                )
            )
            for column, label, color, marker in specs:
                if _finite_column(data, column):
                    artists[column] = ax.scatter(
                        _numeric(data[column]),
                        y,
                        label=label,
                        color=color,
                        marker=marker,
                        s=25,
                    )
            if track == "Local preservation":
                ax.set_xlim(-0.03, 1.03)
                ax.set_xlabel("0–1")
            else:
                ax.set_xlim(left=0)
                ax.set_xlabel("Absolute log-ratio")
            ax.legend(loc="lower right", fontsize=6, frameon=False)
        elif track == "Dynamics":
            x = _numeric(data["median_alignment_cosine"]).to_numpy(
                dtype=float, na_value=np.nan
            )
            colors = np.where(
                x >= pos_thr, "#0072B2", np.where(x <= neg_thr, "#D55E00", "#999999")
            )
            artists["dynamics"] = ax.scatter(x, y, color=colors, s=30)
            for value, style in ((0, "-"), (pos_thr, ":"), (neg_thr, ":")):
                ax.axvline(value, color="0.6", linewidth=0.7, linestyle=style)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("Alignment cosine")
        elif track == "Availability":
            ax.set_xlim(-0.5, 3.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(
                ["Effect", "Repr.", "Local", "Dynamics"],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.grid(False)
            markers = {
                "assessed": ("o", "#147D80"),
                "representation_unstable": ("D", "#CC79A7"),
                "not_supplied": ("s", "#BDBDBD"),
                "not_computed": ("_", "#647482"),
                "unavailable": ("x", "#647482"),
                "insufficient_coverage": ("^", "#B8651A"),
                "numerical_degeneracy": ("X", "#B8651A"),
            }
            for row in range(n):
                for column, key in enumerate(
                    ("effect", "representation", "local", "dynamics")
                ):
                    status = availability.iloc[row][key]
                    marker, color = markers.get(status, ("?", "#647482"))
                    if marker == "?":
                        ax.text(column, row, "?", color=color, ha="center", va="center")
                    else:
                        ax.scatter(
                            column, row, marker=marker, color=color, s=27, linewidths=1
                        )
            from matplotlib.lines import Line2D

            used = set(availability.drop(columns="state").to_numpy().ravel())
            handles = [
                Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    linestyle="",
                    label=status.replace("_", " "),
                    markersize=5,
                )
                for status, (marker, color) in markers.items()
                if status in used
            ]
            if used - markers.keys():
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="#647482",
                        marker="$?$",
                        linestyle="",
                        label="unknown status",
                        markersize=5,
                    )
                )
            if owns_figure:
                artists["availability_legend"] = fig.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    ncol=min(4, len(handles)),
                    fontsize=7,
                    frameon=False,
                )
            else:
                artists["availability_legend"] = ax.legend(
                    handles=handles,
                    loc="upper left",
                    bbox_to_anchor=(0, -0.55),
                    fontsize=6,
                    frameon=False,
                )
        else:
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.grid(False)
            for row, rec in data.iterrows():
                text = (
                    str(rec.get("coverage_status", "unknown"))
                    if track == "Coverage"
                    else str(rec.get("summary", ""))
                )
                if track == "Coverage":
                    text = {
                        "qualified_per_mouse_before_scgeo": "Eligible mice verified",
                        "insufficient_coverage": "Insufficient coverage",
                        "descriptive_only": "Descriptive only",
                        "not_supplied": "Not supplied",
                        "numerical_degeneracy": "Numerical degeneracy",
                    }.get(text, text.replace("_", " "))
                warning = rec.get("warnings")
                if track == "Coverage" and pd.notna(warning) and str(warning):
                    text += " *"
                ax.text(0, row, textwrap.fill(text, 28), va="center", fontsize=8)
    total = len(report)
    comparison = _comparison_from_report(report)
    annotation = f"{n} of {total} states shown" if n < total else None
    heading = title or (comparison if comparison != "comparison" else None)
    if owns_figure:
        if heading or annotation:
            fig.suptitle(
                " · ".join(str(value) for value in (heading, annotation) if value)
            )
        if evidence_mode == "full":
            fig.tight_layout(rect=(0, min(0.20, 0.7 / fig.get_figheight()), 1, 0.95))
        else:
            fig.tight_layout()
    elif annotation:
        axis_map[names[0]].annotate(
            annotation,
            (0, 1),
            xytext=(0, 24),
            textcoords="offset points",
            xycoords="axes fraction",
            fontsize=8,
        )
    if title is not None and not owns_figure:
        axis_map[names[0]].set_title(f"{title}\n{tracks[0]}")
    metadata = {
        "plot": "state_evidence_panel",
        "effect_metric": metric,
        "effect_representation": representation,
        "sort_by": resolved_sort,
        "state_order": data["state"].tolist(),
        "displayed_states": n,
        "total_states": total,
        "page": int(page),
        "evidence_mode": evidence_mode,
        "provenance": provenance,
        "interval_note": "Magnitude intervals summarize uncertainty; they are not automatic hypothesis tests.",
    }
    payload = {"report": data, "tracks": tracks, "availability": availability}
    for message in warnings_out:
        warnings.warn(message, UserWarning, stacklevel=2)
    _show_or_close(plt, bool(show))
    if return_result:
        from ._result import PlotResult

        return PlotResult(
            fig=fig,
            axes=axis_map,
            data=payload,
            metadata=metadata,
            warnings=warnings_out,
            artists=artists,
        )
    if return_data:
        return fig, payload
    return fig


def _representation_tables(
    adata, store_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(adata, pd.DataFrame):
        out = dict(adata.attrs)
        out["per_rep_state"] = adata
    elif isinstance(adata, Mapping):
        out = adata
    else:
        scgeo = adata.uns.get("scgeo", {})
        if store_key not in scgeo:
            raise KeyError(
                f"representation_stability result not found at adata.uns['scgeo'][{store_key!r}]."
            )
        out = scgeo[store_key]
    per = out.get("per_rep_state")
    consensus = out.get("consensus_state")
    representation_summary = out.get("representation_summary")
    if not isinstance(per, pd.DataFrame) or per.empty:
        raise ValueError(
            "representation_stability per_rep_state table is unavailable or empty."
        )
    if not isinstance(consensus, pd.DataFrame):
        consensus = pd.DataFrame()
    if not isinstance(representation_summary, pd.DataFrame):
        representation_summary = pd.DataFrame()
    return per.copy(), consensus.copy(), representation_summary.copy()


def _heatmap_metric(metric):
    aliases = {
        "normalized_displacement": "normalized_delta_norm",
        "normalized displacement": "normalized_delta_norm",
        "displacement_rank": "magnitude_rank",
        "rank": "magnitude_rank",
        "alignment class": "alignment_class",
    }
    return aliases.get(str(metric), str(metric))


def _heatmap_matrix(per: pd.DataFrame, metric: str) -> pd.DataFrame:
    column = _heatmap_metric(metric)
    allowed = {
        "delta_norm",
        "normalized_delta_norm",
        "magnitude_rank",
        "alignment_cosine",
        "alignment_class",
        "usable_fraction",
        "coverage_fraction",
    }
    if column not in allowed:
        raise ValueError(
            f"metric must name one of {sorted(allowed)!r} (or 'displacement_rank')."
        )
    missing = [key for key in ("node", "rep", column) if key not in per]
    if missing:
        raise ValueError(f"The representation table is missing columns: {missing!r}.")
    if per[["node", "rep"]].isna().any().any() or per.duplicated(["node", "rep"]).any():
        raise ValueError(
            "The representation table requires one row per non-missing (node, rep) pair."
        )
    if column == "alignment_class":
        unknown = set(per[column].dropna().astype(str)) - _CLASS_TO_VALUE.keys()
        if unknown:
            raise ValueError(f"Unknown alignment classes: {sorted(unknown)!r}.")
        values = per.assign(_value=per[column].map(_CLASS_TO_VALUE))
    else:
        values = per.assign(_value=pd.to_numeric(per[column], errors="coerce"))
    return values.pivot(index="node", columns="rep", values="_value").reindex(
        index=pd.unique(per["node"]), columns=pd.unique(per["rep"])
    )


def _cluster_order(matrix):
    """Average linkage using RMS distance over jointly finite coordinates."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    values = matrix.to_numpy(dtype=float)
    valid = np.isfinite(values).any(axis=1)
    keep = np.flatnonzero(valid)
    absent = np.flatnonzero(~valid)
    if len(keep) < 2:
        return list(matrix.index[keep]) + list(matrix.index[absent])
    distances = []
    for i, first in enumerate(keep[:-1]):
        for second in keep[i + 1 :]:
            shared = np.isfinite(values[first]) & np.isfinite(values[second])
            if not shared.any():
                raise ValueError(
                    "Clustering requires shared finite values for each pair; supply an explicit order for disconnected missing data."
                )
            distances.append(
                float(
                    np.sqrt(
                        np.mean((values[first, shared] - values[second, shared]) ** 2)
                    )
                )
            )
    order = leaves_list(linkage(distances, method="average", optimal_ordering=True))
    return list(matrix.index[keep[order]]) + list(matrix.index[absent])


def _sort_matrix(
    matrix,
    *,
    cluster_states=False,
    cluster_reps=False,
    order_by="input",
    state_order=None,
    rep_order=None,
):
    modes = (
        {"states": order_by, "representations": order_by}
        if isinstance(order_by, str)
        else dict(order_by)
    )
    unknown = set(modes) - {"states", "representations"}
    if unknown or any(
        value not in {"input", "mean", "cluster"} for value in modes.values()
    ):
        raise ValueError(
            "order_by must be 'input', 'mean', 'cluster', or a mapping with 'states'/'representations' keys."
        )
    if cluster_states or cluster_reps:
        warnings.warn(
            "cluster_states/cluster_reps preserve legacy mean sorting; use order_by='cluster' for hierarchical clustering or 'mean' for mean sorting.",
            FutureWarning,
            stacklevel=3,
        )
    out = matrix.copy()
    for name, legacy in (("states", cluster_states), ("representations", cluster_reps)):
        mode = "mean" if legacy else modes.get(name, "input")
        work = out if name == "states" else out.T
        if mode == "mean":
            order = list(
                work.mean(axis=1, skipna=True)
                .sort_values(na_position="last", kind="stable")
                .index
            )
        elif mode == "cluster":
            order = _cluster_order(work)
        else:
            order = list(work.index)
        if name == "states":
            out = out.loc[order]
        else:
            out = out.loc[:, order]
    out = out.loc[
        _ordered_labels(out.index, state_order, name="state_order"),
        _ordered_labels(out.columns, rep_order, name="rep_order"),
    ]
    return out


def _local_representation_diagnostics(adata, reps: Sequence[str]) -> pd.DataFrame:
    if isinstance(adata, (Mapping, pd.DataFrame)):
        container = adata.attrs if isinstance(adata, pd.DataFrame) else adata
        rep_summary = container.get(
            "representation_diagnostics", container.get("representation_summary")
        )
        rep_summary = _records_to_frame(rep_summary)
    else:
        scgeo = adata.uns.get("scgeo", {})
        local = (
            scgeo.get("local_geometry_stability") if isinstance(scgeo, dict) else None
        )
        rep_summary = (
            local.get("representation_summary") if isinstance(local, dict) else None
        )
    if not isinstance(rep_summary, pd.DataFrame) or "rep" not in rep_summary:
        return pd.DataFrame(index=pd.Index(reps, name="rep"))
    return rep_summary.set_index("rep").reindex(list(reps)).copy()


def representation_stability_heatmap(
    adata,
    *,
    store_key: str = "representation_stability",
    metric: str = "normalized_delta_norm",
    annotate_consensus: bool = True,
    cluster_states: bool = False,
    cluster_reps: bool = False,
    order_by="input",
    state_order=None,
    rep_order=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap=None,
    class_palette=None,
    na_color="#BDBDBD",
    show_colorbar=True,
    axes=None,
    figsize=None,
    title=None,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """Plot stored state-by-representation values with metric-aware scales.

    Input may be AnnData, a tidy ``per_rep_state`` DataFrame (``node``, ``rep``
    and metric columns), or a result mapping containing that table plus optional
    ``consensus_state`` and ``representation_diagnostics`` tables. Table input
    makes archived figure data reusable without loading AnnData.

    Alignment cosine defaults to a zero-centered [-1, 1] diverging scale;
    coverage to [0, 1]; alignment classes use a discrete legend. Other numeric
    metrics use a sequential scale. ``norm`` cannot be combined with bounds.
    Explicit orders move listed labels first and retain remaining labels.
    ``order_by`` accepts 'input', 'mean', 'cluster', or separate ``states`` and
    ``representations`` modes. Clustering uses average linkage and RMS distance
    over jointly finite entries; all-missing rows are appended. It never fills
    missing plotted values. Legacy ``cluster_*`` flags retain mean sorting and
    emit a deprecation warning.

    Supplied axes use ``heatmap`` plus ``consensus`` and ``diagnostics`` when
    applicable; optional ``colorbar`` supplies a dedicated colorbar axis.
    """
    plt, ListedColormap = _lazy_matplotlib()
    from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch

    per, consensus, _ = _representation_tables(adata, store_key)
    column = _heatmap_metric(metric)
    matrix = _sort_matrix(
        _heatmap_matrix(per, metric),
        cluster_states=bool(cluster_states),
        cluster_reps=bool(cluster_reps),
        order_by=order_by,
        state_order=state_order,
        rep_order=rep_order,
    )
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    categorical = column == "alignment_class"
    if categorical and (
        norm is not None or vmin is not None or vmax is not None or cmap is not None
    ):
        raise ValueError(
            "Alignment classes use a discrete scale; customize class_palette instead of norm/cmap/bounds."
        )
    if not categorical and class_palette is not None:
        raise ValueError("class_palette applies only to alignment_class.")
    rep_diagnostics = _local_representation_diagnostics(adata, matrix.columns)
    annotations = pd.DataFrame(index=matrix.index)
    if not consensus.empty and "node" in consensus:
        if consensus["node"].duplicated().any():
            raise ValueError("consensus_state requires one row per node.")
        ann = consensus.set_index("node")
        for key in ("consensus_label", "usable_fraction"):
            if key in ann:
                annotations[key] = ann.reindex(matrix.index)[key]
        if "consensus_label" in annotations:
            annotations["instability_warning"] = annotations["consensus_label"].isin(
                ["representation_unstable", "insufficient_coverage"]
            )
    flag_cols = [
        "neighborhood_outlier",
        "distortion_outlier",
        "state_graph_outlier",
        "insufficient_coverage",
    ]
    has_rep_diagnostics = not rep_diagnostics.empty and any(
        key in rep_diagnostics for key in flag_cols
    )
    finite_values = matrix.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    tolerance = 1e-10
    if column == "alignment_cosine" and (
        (finite_values < -1 - tolerance).any() or (finite_values > 1 + tolerance).any()
    ):
        raise ValueError("alignment_cosine values must lie in [-1, 1].")
    if column in {"usable_fraction", "coverage_fraction"} and (
        (finite_values < -tolerance).any() or (finite_values > 1 + tolerance).any()
    ):
        raise ValueError("Coverage fractions must lie in [0, 1].")
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    if categorical:
        palette = {"discordant": "#D55E00", "neutral": "#999999", "aligned": "#0072B2"}
        palette.update(class_palette or {})
        plot_cmap = ListedColormap(
            [palette[key] for key in ("discordant", "neutral", "aligned")]
        )
        plot_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    else:
        plot_cmap = plt.get_cmap(
            cmap or ("coolwarm" if column == "alignment_cosine" else "viridis")
        ).copy()
        if norm is not None:
            plot_norm = norm
        elif column == "alignment_cosine":
            low, high = -1 if vmin is None else vmin, 1 if vmax is None else vmax
            if not low < 0 < high:
                raise ValueError(
                    "Alignment cosine bounds must straddle zero, or supply a custom norm."
                )
            plot_norm = TwoSlopeNorm(vcenter=0, vmin=low, vmax=high)
        elif column in {"usable_fraction", "coverage_fraction"}:
            plot_norm = Normalize(
                vmin=0 if vmin is None else vmin, vmax=1 if vmax is None else vmax
            )
        else:
            plot_norm = Normalize(vmin=vmin, vmax=vmax)
    if (
        plot_norm.vmin is not None
        and plot_norm.vmax is not None
        and plot_norm.vmin > plot_norm.vmax
    ):
        raise ValueError("vmin must not exceed vmax.")
    plot_cmap = plot_cmap.with_extremes(bad=na_color)
    names = (
        ["heatmap"]
        + (["consensus"] if annotate_consensus else [])
        + (["diagnostics"] if has_rep_diagnostics else [])
    )
    owns_figure = axes is None
    if owns_figure:
        width = max(6.4, 0.7 * matrix.shape[1] + (4.8 if annotate_consensus else 1.0))
        height = max(4.2, 0.45 * matrix.shape[0] + 2.0)
        if has_rep_diagnostics:
            height += 1.2
        fig = plt.figure(figsize=figsize or (width, height))
        native_colorbar = not categorical and show_colorbar
        widths = [max(1.5, matrix.shape[1])]
        if annotate_consensus:
            widths.append(2.2)
        if native_colorbar:
            widths.append(0.12)
        gs = fig.add_gridspec(
            2 if has_rep_diagnostics else 1,
            len(widths),
            width_ratios=widths,
            height_ratios=[max(2, matrix.shape[0]), 1.4]
            if has_rep_diagnostics
            else [1],
            wspace=0.45,
            hspace=0.6,
        )
        ax = fig.add_subplot(gs[0, 0])
        axis_map = {"heatmap": ax}
        if annotate_consensus:
            axis_map["consensus"] = fig.add_subplot(gs[0, 1], sharey=ax)
        if has_rep_diagnostics:
            axis_map["diagnostics"] = fig.add_subplot(gs[1, 0], sharex=ax)
        if native_colorbar:
            axis_map["colorbar"] = fig.add_subplot(gs[0, -1])
    else:
        fig, axis_map = _supplied_axes(axes, names)
        ax = axis_map["heatmap"]
    im = ax.imshow(values, aspect="auto", norm=plot_norm, cmap=plot_cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_xlabel("Representation")
    ax.set_ylabel("State")
    metric_label = {
        "delta_norm": "Raw displacement magnitude",
        "normalized_delta_norm": "Normalized displacement magnitude",
        "magnitude_rank": "Displacement rank (1 = largest)",
        "alignment_cosine": "Alignment cosine",
        "alignment_class": "Alignment class",
        "usable_fraction": "Usable fraction",
        "coverage_fraction": "Coverage fraction",
    }[column]
    ax.set_title(
        metric_label
        if owns_figure or title is None
        else textwrap.fill(str(title), width=32) + "\n" + metric_label
    )
    if owns_figure and title is not None:
        fig.suptitle(
            textwrap.fill(str(title), width=max(30, int(fig.get_figwidth() * 10))),
            y=0.98,
        )
    artists = {"heatmap": im}
    if categorical:
        handles = [
            Patch(facecolor=palette[key], label=key)
            for key in ("discordant", "neutral", "aligned")
        ]
        handles.append(Patch(facecolor=na_color, label="missing"))
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.12),
            ncol=4,
            fontsize=7,
            frameon=False,
        )
    elif show_colorbar:
        cax = axes.get("colorbar") if axes is not None else axis_map.get("colorbar")
        if cax is not None and cax.figure is not fig:
            raise ValueError("The colorbar axis must belong to the same figure.")
        parent_axes = [ax, axis_map["consensus"]] if annotate_consensus else [ax]
        if has_rep_diagnostics:
            parent_axes.append(axis_map["diagnostics"])
        bar = fig.colorbar(im, ax=parent_axes, cax=cax, fraction=0.046, pad=0.06)
        bar.set_label(metric_label, fontsize=8)
        bar.ax.tick_params(labelsize=8)
        axis_map["colorbar"] = bar.ax
        if np.ma.getmaskarray(values).any():
            ax.legend(
                handles=[Patch(facecolor=na_color, label="missing")],
                loc="lower left",
                bbox_to_anchor=(0, 1.12),
                fontsize=7,
                frameon=False,
            )
    if annotate_consensus:
        ann_ax = axis_map["consensus"]
        ann_ax.set_xlim(0, 1)
        ann_ax.set_xticks([])
        ann_ax.tick_params(labelleft=False, left=False)
        ann_ax.set_title("Consensus")
        for row, state in enumerate(matrix.index):
            label = str(
                annotations.get("consensus_label", pd.Series(dtype=object)).get(
                    state, "not supplied"
                )
            )
            frac = annotations.get("usable_fraction", pd.Series(dtype=float)).get(
                state, np.nan
            )
            text = textwrap.fill(label.replace("_", " "), 23)
            if pd.notna(frac) and np.isfinite(float(frac)):
                text += f"\n{float(frac):.0%} usable"
            ann_ax.text(
                0,
                row,
                text,
                va="center",
                fontsize=8,
                color=_CONSENSUS_PALETTE.get(label, "0.4"),
            )
        ann_ax.set_ylim(ax.get_ylim())
        ann_ax.grid(False)
        for spine in ann_ax.spines.values():
            spine.set_visible(False)
    if has_rep_diagnostics:
        diag_ax = axis_map["diagnostics"]
        flag_values = np.full((len(flag_cols), matrix.shape[1]), np.nan)
        for row, key in enumerate(flag_cols):
            if key in rep_diagnostics:
                for col, value in enumerate(
                    rep_diagnostics.reindex(matrix.columns)[key]
                ):
                    if pd.notna(value):
                        flag_values[row, col] = float(bool(value))
        diag_cmap = ListedColormap(["#FFFFFF", "#D55E00"])
        diag_cmap = diag_cmap.with_extremes(bad=na_color)
        artists["diagnostics"] = diag_ax.imshow(
            np.ma.masked_invalid(flag_values),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap=diag_cmap,
        )
        diag_ax.set_yticks(range(len(flag_cols)))
        diag_ax.set_yticklabels(
            ["neighborhood", "distortion", "state graph", "coverage"], fontsize=7
        )
        diag_ax.set_xticks(range(matrix.shape[1]))
        diag_ax.set_xticklabels(
            matrix.columns.astype(str), rotation=45, ha="right", fontsize=8
        )
        diag_ax.set_title("Representation diagnostics", fontsize=9)
        diag_ax.set_xlabel("Representation")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
        artists["diagnostic_legend"] = diag_ax.legend(
            handles=[
                Patch(facecolor="#FFFFFF", edgecolor="0.7", label="not flagged"),
                Patch(facecolor="#D55E00", label="flagged"),
                Patch(facecolor=na_color, label="unavailable"),
            ],
            loc="upper left",
            bbox_to_anchor=(1.25, 1.05),
            fontsize=7,
            frameon=False,
        )
    if owns_figure:
        fig.subplots_adjust(
            bottom=0.22,
            right=0.88 if not categorical and show_colorbar else 0.95,
            top=0.75
            if title is not None
            else 0.80
            if categorical or np.ma.getmaskarray(values).any()
            else 0.9,
        )
    payload = {
        "matrix": matrix,
        "annotations": annotations,
        "representation_diagnostics": rep_diagnostics,
    }
    metadata = {
        "plot": "representation_stability_heatmap",
        "metric": column,
        "state_order": list(matrix.index),
        "rep_order": list(matrix.columns),
        "order_by": order_by,
        "legacy_mean_sort": {
            "states": bool(cluster_states),
            "representations": bool(cluster_reps),
        },
        "normalization": {
            "class": type(plot_norm).__name__,
            "vmin": plot_norm.vmin,
            "vmax": plot_norm.vmax,
        },
        "missing_color": na_color,
        "clustering": "average linkage; RMS distance over shared finite entries; all-missing last",
    }
    _show_or_close(plt, bool(show))
    if return_result:
        from ._result import PlotResult

        return PlotResult(
            fig=fig, axes=axis_map, data=payload, metadata=metadata, artists=artists
        )
    if return_data:
        return fig, payload
    return fig


def consensus_state_map(
    adata,
    *,
    node_key,
    basis: str = "umap",
    store_key: str = "representation_stability",
    label_states: bool = True,
    palette=None,
    max_points=None,
    random_state=0,
    required_indices=None,
    rasterized="auto",
    ax=None,
    figsize=(5.8, 4.8),
    title=None,
    return_data=False,
    return_result=False,
    show: bool = True,
):
    """Display stored consensus labels; never recompute them in the embedding.

    ``max_points`` optionally samples display points reproducibly while retaining
    each state and every ``required_indices`` position. State-label positions
    use the full data. The returned source table records which cells are shown.
    ``rasterized='auto'`` rasterizes more than 5,000 displayed points.
    """
    plt, _ = _lazy_matplotlib()
    from ._utils import display_indices, resolve_embedding

    if node_key not in adata.obs:
        raise KeyError(f"obs key {node_key!r} not found")
    if rasterized not in {True, False, "auto"}:
        raise ValueError("rasterized must be True, False, or 'auto'.")
    key, coords = resolve_embedding(adata, basis)
    _, consensus, _ = _representation_tables(adata, store_key)
    if consensus.empty or "node" not in consensus or "consensus_label" not in consensus:
        raise ValueError(
            "representation_stability consensus_state table is unavailable."
        )
    if consensus["node"].astype(str).duplicated().any():
        raise ValueError("consensus_state requires one distinct node label per row.")
    label_map = dict(zip(consensus["node"].astype(str), consensus["consensus_label"]))
    states = adata.obs[node_key].astype(str)
    labels = states.map(label_map).fillna("missing").astype(str)
    colors = dict(_CONSENSUS_PALETTE)
    colors.update(palette or {})
    color = labels.map(lambda label: colors.get(label, colors["missing"]))
    selected = display_indices(
        states.to_numpy(), max_points, random_state, required_indices
    )
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    points = ax.scatter(
        coords[selected, 0],
        coords[selected, 1],
        c=color.iloc[selected],
        s=12,
        linewidths=0,
        alpha=0.85,
        rasterized=len(selected) > 5000 if rasterized == "auto" else rasterized,
    )
    display_basis = key.removeprefix("X_")
    ax.set_xlabel(f"{display_basis} 1")
    ax.set_ylabel(f"{display_basis} 2")
    ax.set_title(
        title or "Display embedding only; consensus calculated across representations"
    )
    if label_states:
        for state in pd.unique(states):
            locations = coords[(states == state).to_numpy(), :2]
            locations = locations[np.isfinite(locations).all(axis=1)]
            if len(locations):
                center = np.median(locations, axis=0)
                ax.text(
                    center[0],
                    center[1],
                    str(state),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            color=colors.get(label, colors["missing"]),
            label=label.replace("_", " "),
            markersize=5,
        )
        for label in dict.fromkeys(list(colors) + list(pd.unique(labels)))
        if label in set(labels)
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)
    if owns_figure:
        fig.tight_layout()
    shown = np.zeros(adata.n_obs, dtype=bool)
    shown[selected] = True
    data = pd.DataFrame(
        {
            "cell_index": np.arange(adata.n_obs),
            "cell_id": adata.obs_names.astype(str),
            "state": states.to_numpy(),
            "consensus_label": labels.to_numpy(),
            "display_x": coords[:, 0],
            "display_y": coords[:, 1],
            "displayed": shown,
        }
    )
    metadata = {
        "plot": "consensus_state_map",
        "basis": key,
        "display_only": True,
        "n_cells": adata.n_obs,
        "displayed_cells": len(selected),
        "max_points": max_points,
        "random_state": random_state,
        "palette": colors,
        "store_key": store_key,
    }
    _show_or_close(plt, bool(show))
    if return_result:
        from ._result import PlotResult

        return PlotResult(
            fig=fig,
            axes={"map": ax},
            data=data,
            metadata=metadata,
            artists={"points": points},
        )
    if return_data:
        return fig, data
    return fig


def perturbation_report(
    adata,
    *,
    node_key,
    basis: str = "umap",
    report: Optional[pd.DataFrame] = None,
    save_dir=None,
    prefix: str = "scgeo",
    comparison_label: Optional[str] = None,
    local_k=None,
    pair_aggregation: str = "median",
    include_worst_case: bool = True,
    formats=("png", "svg"),
    dpi=300,
    width_mm=None,
    height_mm=None,
    transparent=False,
    bbox_inches=None,
    strict=False,
    required_panels=None,
    panel_kwargs=None,
    show: bool = True,
):
    """Create figures, source tables, and provenance for a perturbation report.

    Partial bundles remain the default when optional evidence is absent.
    ``strict=True`` requires all three standard panels unless a narrower
    ``required_panels`` list is supplied. An explicitly required panel always
    raises if unavailable, before any files are written. Panel names are
    ``consensus_state_map``, ``state_evidence_panel``, ``representation_heatmap``;
    ``panel_kwargs`` maps these names to their plotting options.

    ``formats`` accepts PNG/SVG/PDF; ``dpi`` also controls rasterized artists.
    ``width_mm``/``height_mm`` specify export canvas dimensions. The default
    ``bbox_inches=None`` preserves those dimensions. The bundle retains legacy
    Figure entries and adds ``plot_results`` with source data and metadata.
    """
    plt, _ = _lazy_matplotlib()
    from ._result import export_figure

    names = ("consensus_state_map", "state_evidence_panel", "representation_heatmap")
    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(str(value).lower().lstrip(".") for value in formats))
    if not formats or any(value not in {"png", "svg", "pdf"} for value in formats):
        raise ValueError("formats must contain png, svg, and/or pdf.")
    for key, value in (("dpi", dpi), ("width_mm", width_mm), ("height_mm", height_mm)):
        if value is not None and (
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{key} must be a positive finite number.")
    if dpi is None:
        raise ValueError("dpi must be a positive finite number.")
    if required_panels is None:
        required = set(names) if strict else set()
    else:
        required = (
            {required_panels}
            if isinstance(required_panels, str)
            else set(required_panels)
        )
    if required - set(names):
        raise ValueError(f"Unknown required panels: {sorted(required - set(names))!r}.")
    options = {} if panel_kwargs is None else dict(panel_kwargs)
    if set(options) - set(names):
        raise ValueError(
            f"Unknown panel_kwargs keys: {sorted(set(options) - set(names))!r}."
        )
    for key, kwargs in options.items():
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"panel_kwargs[{key!r}] must be a mapping.")
        if set(kwargs) & {"show", "return_data", "return_result"}:
            raise ValueError(
                "panel_kwargs must not override show/return_data/return_result."
            )
    report_df = (
        report.copy()
        if isinstance(report, pd.DataFrame)
        else state_report(
            adata,
            node_key=node_key,
            local_k=local_k,
            pair_aggregation=pair_aggregation,
            include_worst_case=include_worst_case,
            comparison_label=comparison_label,
        )
    )
    comparison = comparison_label or _comparison_from_report(report_df)
    safe_comparison = _safe_filename(comparison)
    safe_prefix = _safe_filename(prefix)
    warnings_out = [
        value
        for value in report_df.get("warnings", pd.Series(dtype=object)).astype(str)
        if value and value != "nan"
    ]
    warnings_out.extend(
        str(value) for value in report_df.attrs.get("warnings", []) if str(value)
    )
    rep_diag = _records_to_frame(report_df.attrs.get("representation_diagnostics"))
    bundle = {
        "state_report": report_df,
        "representation_diagnostics": rep_diag,
        "metadata": report_df.attrs.get("provenance", {}),
        "global_diagnostics": report_df.attrs.get("global_diagnostics", {}),
        "rules": report_df.attrs.get("scgeo_report_rules", {}),
        "warnings": sorted(set(warnings_out)),
        "saved_paths": {},
        "alt_text": {},
        "plot_results": {},
    }
    bundle.update({name: None for name in names})
    existing_figures = set(plt.get_fignums())
    failures = {}
    for name in names:
        figures_before_panel = set(plt.get_fignums())
        try:
            kwargs = dict(options.get(name, {}))
            if name == "consensus_state_map":
                kwargs.setdefault("node_key", node_key)
                kwargs.setdefault("basis", basis)
                result = consensus_state_map(
                    adata, **kwargs, show=False, return_result=True
                )
                alt = _alt_text_consensus_map(
                    report_df, str(comparison), str(kwargs["basis"])
                )
            elif name == "state_evidence_panel":
                result = state_evidence_panel(
                    report_df, **kwargs, show=False, return_result=True
                )
                alt = _alt_text_state_evidence(
                    result.data["report"], result.data["tracks"], str(comparison)
                )
            else:
                result = representation_stability_heatmap(
                    adata, **kwargs, show=False, return_result=True
                )
                alt = _alt_text_heatmap(
                    result.data["matrix"], result.data["annotations"], str(comparison)
                )
            bundle[name] = result.fig
            bundle["plot_results"][name] = result
            bundle["alt_text"][name] = alt
            bundle["warnings"].extend(result.warnings)
        except Exception as exc:  # noqa: BLE001 - partial reports record panel failures; required panels raise below.
            for number in set(plt.get_fignums()) - figures_before_panel:
                plt.close(number)
            failures[name] = exc
            bundle["warnings"].append(
                f"{name} unavailable: {type(exc).__name__}: {exc}"
            )
    missing_required = required & failures.keys()
    if missing_required:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
        details = "; ".join(
            f"{name}: {failures[name]}" for name in sorted(missing_required)
        )
        raise ValueError(
            f"Required report panels are unavailable: {details}"
        ) from failures[min(missing_required)]
    bundle["warnings"] = sorted(set(bundle["warnings"]))
    export_config = {
        "formats": list(formats),
        "dpi": dpi,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "transparent": bool(transparent),
        "bbox_inches": bbox_inches,
        "required_panels": sorted(required),
    }
    bundle["export_config"] = export_config
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{safe_prefix}_{safe_comparison}"
        for name, table in (
            ("state_report", report_df),
            ("representation_diagnostics", rep_diag),
        ):
            path = out_dir / f"{stem}_{name}.csv"
            table.to_csv(path, index=False)
            bundle["saved_paths"][f"{name}_csv"] = str(path)
        for name, result in bundle["plot_results"].items():
            exported = export_figure(
                result,
                out_dir / f"{stem}_{name}",
                formats=formats,
                dpi=dpi,
                width_mm=width_mm,
                height_mm=height_mm,
                transparent=transparent,
                bbox_inches=bbox_inches,
                metadata={
                    "comparison_label": comparison,
                    "alt_text": bundle["alt_text"][name],
                },
            )
            for ext, path in exported["figures"].items():
                bundle["saved_paths"][f"{name}_{ext}"] = path
            for key, path in exported["tables"].items():
                bundle["saved_paths"][f"{name}_{key}_csv"] = path
            if exported["metadata"]:
                bundle["saved_paths"][f"{name}_metadata_json"] = exported["metadata"]
        metadata = {
            "provenance": report_df.attrs.get("provenance", {}),
            "rules": report_df.attrs.get("scgeo_report_rules", {}),
            "global_diagnostics": report_df.attrs.get("global_diagnostics", {}),
            "alt_text": bundle["alt_text"],
            "export_config": export_config,
            "panels": {
                name: result.metadata for name, result in bundle["plot_results"].items()
            },
            "warnings": bundle["warnings"],
            "saved_paths": dict(bundle["saved_paths"]),
        }
        metadata_path = out_dir / f"{stem}_metadata.json"
        metadata_path.write_text(
            json.dumps(_json_safe(metadata), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        bundle["saved_paths"]["metadata_json"] = str(metadata_path)
        warnings_path = out_dir / f"{stem}_warnings.txt"
        warnings_path.write_text(
            "\n".join(bundle["warnings"]) + ("\n" if bundle["warnings"] else ""),
            encoding="utf-8",
        )
        bundle["saved_paths"]["warnings_txt"] = str(warnings_path)
        alt_path = out_dir / f"{stem}_alt_text.txt"
        alt_path.write_text(
            "\n\n".join(
                f"{key}: {value}" for key, value in sorted(bundle["alt_text"].items())
            ),
            encoding="utf-8",
        )
        bundle["saved_paths"]["alt_text_txt"] = str(alt_path)
    if show:
        for name in names:
            if bundle[name] is not None:
                bundle[name].show()
        plt.show()
    return bundle


def _per_cell_distortion(
    local_out: dict[str, Any],
    *,
    rep_a: Optional[str],
    rep_b: Optional[str],
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    per_cell = local_out.get("per_cell")
    if not isinstance(per_cell, pd.DataFrame) or per_cell.empty:
        warnings.warn(
            "Per-cell local geometry values were not stored. "
            "Rerun scgeo.tl.local_geometry_stability(..., store_per_cell=True).",
            RuntimeWarning,
            stacklevel=2,
        )
        raise ValueError("per-cell local geometry values are unavailable")

    aliases = {
        "local_shape_distortion": "local_distortion_median",
        "global_scale_distortion": "global_distortion_median",
    }
    column = aliases.get(str(metric), str(metric))
    if column not in per_cell:
        raise ValueError(
            f"metric {metric!r} is not available in stored per-cell local geometry."
        )

    df = per_cell.copy()
    if (rep_a is None) ^ (rep_b is None):
        raise ValueError(
            "rep_a and rep_b must be provided together, or both left as None."
        )
    if rep_a is not None and rep_b is not None:
        forward = (df["rep_a"].astype(str) == str(rep_a)) & (
            df["rep_b"].astype(str) == str(rep_b)
        )
        reverse = (df["rep_a"].astype(str) == str(rep_b)) & (
            df["rep_b"].astype(str) == str(rep_a)
        )
        df = df[forward | reverse]
        if df.empty:
            raise ValueError(
                f"No per-cell values found for representation pair {rep_a!r}, {rep_b!r}."
            )
        values = df.groupby("cell_index", sort=False)[column].median()
    else:
        if aggregation in {"median", "per_cell"}:
            values = df.groupby("cell_index", sort=False)[column].median()
        elif aggregation in {"worst", "worst_case", "worst-case", "max"}:
            values = df.groupby("cell_index", sort=False)[column].max()
        else:
            raise ValueError("aggregation must be 'median' or 'worst-case'.")
    return pd.DataFrame(
        {"cell_index": values.index.astype(int), "value": values.to_numpy(dtype=float)}
    )


def _state_distortion_values(
    adata,
    local_out: dict[str, Any],
    *,
    node_key: Optional[str],
    rep_a: Optional[str],
    rep_b: Optional[str],
    metric: str,
) -> pd.DataFrame:
    params = local_out.get("params", {}) if isinstance(local_out, dict) else {}
    resolved_node_key = node_key or (
        params.get("node_key") if isinstance(params, dict) else None
    )
    if resolved_node_key is None:
        raise ValueError(
            "aggregation='state' requires node_key or a stored local_geometry_stability node_key."
        )
    if resolved_node_key not in adata.obs:
        raise KeyError(f"obs key {resolved_node_key!r} not found")
    state_pair = local_out.get("state_pair_summary")
    if not isinstance(state_pair, pd.DataFrame) or state_pair.empty:
        raise ValueError(
            "aggregation='state' requires stored state_pair_summary values."
        )
    aliases = {
        "local_shape_distortion": "local_distortion_median",
        "global_scale_distortion": "global_distortion_median",
    }
    metric_name = aliases.get(str(metric), str(metric))
    df = state_pair[state_pair["metric"] == metric_name].copy()
    if rep_a is not None and rep_b is not None:
        forward = (df["rep_a"].astype(str) == str(rep_a)) & (
            df["rep_b"].astype(str) == str(rep_b)
        )
        reverse = (df["rep_a"].astype(str) == str(rep_b)) & (
            df["rep_b"].astype(str) == str(rep_a)
        )
        df = df[forward | reverse]
    if df.empty:
        raise ValueError(
            f"No state-level local geometry values found for metric {metric!r}."
        )
    df["median"] = pd.to_numeric(df["median"], errors="coerce")
    state_values = df.groupby("state", sort=False)["median"].median()
    states = adata.obs[resolved_node_key].astype(str).to_numpy()
    values = np.asarray(
        [state_values.get(state, np.nan) for state in states], dtype=float
    )
    return pd.DataFrame(
        {"cell_index": np.arange(adata.n_obs, dtype=int), "value": values}
    )


def local_distortion_map(
    adata,
    *,
    basis: str = "umap",
    store_key: str = "local_geometry_stability",
    rep_a=None,
    rep_b=None,
    metric: str = "local_shape_distortion",
    aggregation: str = "median",
    node_key=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap="magma",
    na_color="#BDBDBD",
    max_points=None,
    random_state=0,
    required_indices=None,
    rasterized="auto",
    ax=None,
    colorbar=True,
    cax=None,
    figsize=(5.8, 4.8),
    title=None,
    return_data=False,
    return_result=False,
    show: bool = True,
):
    """Display stored distortion values with a shared scale and optional sampling.

    Sampling affects displayed cells only, with states preserved when node_key
    is supplied. Aggregation always uses the complete stored results. Missing
    values use ``na_color``. ``norm`` cannot be combined with ``vmin/vmax``.
    """
    plt, _ = _lazy_matplotlib()
    from ._utils import display_indices, resolve_embedding

    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if (rep_a is None) ^ (rep_b is None):
        raise ValueError("rep_a and rep_b must be supplied together.")
    if rasterized not in {True, False, "auto"}:
        raise ValueError("rasterized must be True, False, or 'auto'.")
    if node_key is not None and node_key not in adata.obs:
        raise KeyError(f"obs key {node_key!r} not found")
    scgeo = adata.uns.get("scgeo", {})
    if store_key not in scgeo:
        raise KeyError(
            f"local_geometry_stability result not found at adata.uns['scgeo'][{store_key!r}]."
        )
    aggregation = str(aggregation)
    state_level = aggregation == "state"
    if state_level:
        values = _state_distortion_values(
            adata,
            scgeo[store_key],
            node_key=node_key,
            rep_a=rep_a,
            rep_b=rep_b,
            metric=metric,
        )
    else:
        values = _per_cell_distortion(
            scgeo[store_key],
            rep_a=rep_a,
            rep_b=rep_b,
            metric=metric,
            aggregation=aggregation,
        )
    key, coords = resolve_embedding(adata, basis)
    color = np.full(adata.n_obs, np.nan)
    valid_idx = values["cell_index"].to_numpy(dtype=int)
    if np.any(valid_idx < 0) or np.any(valid_idx >= adata.n_obs):
        raise ValueError(
            "Stored local geometry cell indices are outside the current AnnData object."
        )
    color[valid_idx] = values["value"].to_numpy(dtype=float)
    group_key = node_key or scgeo[store_key].get("params", {}).get("node_key")
    groups = (
        adata.obs[group_key].to_numpy()
        if group_key in adata.obs
        else np.zeros(adata.n_obs)
    )
    selected = display_indices(groups, max_points, random_state, required_indices)
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if cax is not None and cax.figure is not fig:
        raise ValueError("The colorbar axis must belong to the same figure.")
    plot_cmap = plt.get_cmap(cmap).copy()
    plot_cmap = plot_cmap.with_extremes(bad=na_color)
    from matplotlib.colors import Normalize

    plot_norm = norm if norm is not None else Normalize(vmin=vmin, vmax=vmax)
    plot_norm.autoscale_None(np.ma.masked_invalid(color))
    sc = ax.scatter(
        coords[selected, 0],
        coords[selected, 1],
        c=np.ma.masked_invalid(color[selected]),
        s=12,
        cmap=plot_cmap,
        norm=plot_norm,
        linewidths=0,
        plotnonfinite=True,
        rasterized=len(selected) > 5000 if rasterized == "auto" else rasterized,
    )
    axis_map = {"map": ax}
    if colorbar:
        bar = fig.colorbar(sc, ax=ax, cax=cax, label=str(metric).replace("_", " "))
        axis_map["colorbar"] = bar.ax
    if (~np.isfinite(color[selected])).any():
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(facecolor=na_color, label="unavailable")],
            frameon=False,
            fontsize=8,
        )
    display_basis = key.removeprefix("X_")
    ax.set_xlabel(f"{display_basis} 1")
    ax.set_ylabel(f"{display_basis} 2")
    if title is None:
        if state_level:
            title = f"State-level display of {metric}; no per-cell statistic recomputed"
        elif rep_a is not None:
            title = f"{metric}: {rep_a} vs {rep_b}"
        else:
            title = f"{metric}: {aggregation} across representation pairs"
    ax.set_title(title)
    if owns_figure:
        fig.tight_layout()
    shown = np.zeros(adata.n_obs, dtype=bool)
    shown[selected] = True
    data = pd.DataFrame(
        {
            "cell_index": np.arange(adata.n_obs),
            "cell_id": adata.obs_names.astype(str),
            "value": color,
            "display_x": coords[:, 0],
            "display_y": coords[:, 1],
            "displayed": shown,
        }
    )
    if group_key in adata.obs:
        data["state"] = adata.obs[group_key].to_numpy()
    metadata = {
        "plot": "local_distortion_map",
        "basis": key,
        "metric": metric,
        "aggregation": aggregation,
        "rep_a": rep_a,
        "rep_b": rep_b,
        "display_only": True,
        "normalization": {
            "class": type(sc.norm).__name__,
            "vmin": sc.norm.vmin,
            "vmax": sc.norm.vmax,
        },
        "n_cells": adata.n_obs,
        "displayed_cells": len(selected),
        "max_points": max_points,
        "random_state": random_state,
        "store_key": store_key,
        "missing_color": na_color,
    }
    _show_or_close(plt, bool(show))
    if return_result:
        from ._result import PlotResult

        return PlotResult(
            fig=fig, axes=axis_map, data=data, metadata=metadata, artists={"points": sc}
        )
    if return_data:
        return fig, data
    return fig
