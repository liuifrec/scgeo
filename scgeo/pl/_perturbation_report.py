from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional
import json
import re
import warnings

import numpy as np
import pandas as pd

from ..get import state_report


_CONSENSUS_PALETTE = {
    "stable_aligned": "#0072B2",
    "stable_discordant": "#D55E00",
    "stable_neutral": "#009E73",
    "stable_effect": "#4D4D4D",
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
    raise KeyError(f"Embedding {basis!r} not found as adata.obsm[{key!r}] or adata.obsm[{x_key!r}].")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_column(df: pd.DataFrame, column: str) -> bool:
    return column in df and np.isfinite(_numeric(df[column])).any()


def _alignment_thresholds(report: pd.DataFrame) -> tuple[float, float]:
    rules = report.attrs.get("scgeo_report_rules", {}) if hasattr(report, "attrs") else {}
    thresholds = rules.get("alignment_thresholds", {}) if isinstance(rules, dict) else {}
    return (
        float(thresholds.get("alignment_pos_thr", 0.3)),
        float(thresholds.get("alignment_neg_thr", -0.3)),
    )


def _sorted_report(report: pd.DataFrame, sort_by: str, max_states: int) -> pd.DataFrame:
    out = report.copy()
    if sort_by in out:
        values = _numeric(out[sort_by])
        out = out.assign(_sort_value=values).sort_values(
            ["_sort_value", "state"],
            ascending=[False, True],
            na_position="last",
        )
        out = out.drop(columns=["_sort_value"])
    elif "state" in out:
        out = out.sort_values("state")
    return out.head(int(max_states)).reset_index(drop=True)


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


def _alt_text_state_evidence(report: pd.DataFrame, tracks: Sequence[str], comparison: str) -> str:
    n_states = int(report.shape[0])
    stable = report.loc[
        report.get("representation_consensus_label", pd.Series(dtype=object)).astype(str).str.startswith("stable_"),
        "state",
    ].astype(str).tolist() if "representation_consensus_label" in report and "state" in report else []
    unstable = report.loc[
        report.get("representation_consensus_label", pd.Series(dtype=object)).astype(str).isin(
            ["representation_unstable", "insufficient_coverage"]
        ),
        "state",
    ].astype(str).tolist() if "representation_consensus_label" in report and "state" in report else []
    possible = ["Effect", "Stability", "Local preservation", "Local distortion", "Dynamics", "Coverage"]
    omitted = [track for track in possible if track not in tracks]
    return (
        f"State evidence panel for {comparison} with {n_states} states. "
        f"Stable states: {', '.join(stable[:5]) if stable else 'none'}. "
        f"Unstable or insufficient states: {', '.join(unstable[:5]) if unstable else 'none'}. "
        f"Omitted evidence tracks: {', '.join(omitted) if omitted else 'none'}."
    )


def _alt_text_heatmap(matrix: pd.DataFrame, annotations: pd.DataFrame, comparison: str) -> str:
    unstable = []
    if "consensus_label" in annotations:
        unstable = annotations.index[
            annotations["consensus_label"].astype(str).isin(["representation_unstable", "insufficient_coverage"])
        ].astype(str).tolist()
    return (
        f"Representation stability heatmap for {comparison} with {matrix.shape[0]} states "
        f"and {matrix.shape[1]} representations. "
        f"Unstable or insufficient states: {', '.join(unstable[:5]) if unstable else 'none'}. "
        "Missing values are shown separately from numerical zero."
    )


def _alt_text_consensus_map(report: pd.DataFrame, comparison: str, basis: str) -> str:
    n_states = int(report.shape[0])
    unstable = report.loc[
        report.get("representation_consensus_label", pd.Series(dtype=object)).astype(str).isin(
            ["representation_unstable", "insufficient_coverage"]
        ),
        "state",
    ].astype(str).tolist() if "representation_consensus_label" in report and "state" in report else []
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
    sort_by: str = "normalized_delta_norm",
    max_states: int = 25,
    show_ci: bool = True,
    show_coverage: bool = True,
    figsize=None,
    title=None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Plot a progressive-disclosure state evidence panel from a ScGeo report.

    Tracks are omitted when their source columns are unavailable. Effect,
    stability, local-neighborhood preservation, local distortion, dynamics,
    and coverage/warning indicators are drawn on separate axes.
    """
    plt, _ = _lazy_matplotlib()
    if isinstance(report_or_adata, pd.DataFrame):
        report = report_or_adata.copy()
    else:
        report = state_report(report_or_adata, node_key=node_key)
    if report.empty:
        raise ValueError("state_evidence_panel requires at least one report row.")

    pos_thr, neg_thr = _alignment_thresholds(report)
    data = _sorted_report(report, sort_by, max_states)
    tracks: list[str] = []
    if _finite_column(data, "delta_norm") or _finite_column(data, "normalized_delta_norm"):
        tracks.append("Effect")
    has_consensus = (
        "representation_consensus_label" in data
        and data["representation_consensus_label"].notna().any()
    )
    if has_consensus or _finite_column(data, "representation_coverage_fraction"):
        tracks.append("Stability")
    if _finite_column(data, "median_neighbor_overlap") or _finite_column(
        data,
        "median_neighbor_jaccard",
    ):
        tracks.append("Local preservation")
    if _finite_column(data, "median_local_shape_distortion") or _finite_column(
        data,
        "median_global_scale_distortion",
    ):
        tracks.append("Local distortion")
    if _finite_column(data, "median_alignment_cosine"):
        tracks.append("Dynamics")
    if show_coverage and (
        "coverage_status" in data
        or "warnings" in data
        or "reason_codes" in data
    ):
        tracks.append("Coverage")
    if not tracks:
        tracks.append("Summary")

    n = data.shape[0]
    height = max(2.8, 0.34 * n + 1.2)
    width = max(7.5, 2.35 * len(tracks))
    fig, axes = plt.subplots(
        1,
        len(tracks),
        sharey=True,
        figsize=figsize or (width, height),
        squeeze=False,
    )
    axes_list = list(axes[0])
    y = np.arange(n)
    labels = data["state"].astype(str).tolist() if "state" in data else [str(i) for i in range(n)]

    for ax, track in zip(axes_list, tracks):
        ax.set_title(track)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y)
        if ax is axes_list[0]:
            ax.set_yticklabels(labels)
        else:
            ax.tick_params(labelleft=False)

        if track == "Effect":
            if _finite_column(data, "delta_norm"):
                x = _numeric(data["delta_norm"]).to_numpy(dtype=float)
                ax.set_xlabel("Magnitude")
                if (
                    show_ci
                    and _finite_column(data, "magnitude_ci95_low")
                    and _finite_column(data, "magnitude_ci95_high")
                ):
                    low = _numeric(data["magnitude_ci95_low"]).to_numpy(dtype=float)
                    high = _numeric(data["magnitude_ci95_high"]).to_numpy(dtype=float)
                    left = np.clip(x - low, 0, None)
                    right = np.clip(high - x, 0, None)
                    valid = np.isfinite(x)
                    ax.errorbar(
                        x[valid],
                        y[valid],
                        xerr=np.vstack([left[valid], right[valid]]),
                        fmt="o",
                        color="#4C78A8",
                        ecolor="#4C78A8",
                        alpha=0.9,
                    )
                else:
                    ax.scatter(x, y, color="#4C78A8", s=26)
            else:
                x = _numeric(data["normalized_delta_norm"]).to_numpy(dtype=float)
                ax.scatter(x, y, color="#4C78A8", s=26)
                ax.set_xlabel("Normalized")
            ax.axvline(0.0, color="0.2", linewidth=0.8)

        elif track == "Stability":
            x = (
                _numeric(data["representation_coverage_fraction"]).to_numpy(dtype=float)
                if "representation_coverage_fraction" in data
                else np.full(n, np.nan)
            )
            labels_consensus = (
                data["representation_consensus_label"].fillna("missing").astype(str).tolist()
                if "representation_consensus_label" in data
                else ["missing"] * n
            )
            colors = [_CONSENSUS_PALETTE.get(label, _CONSENSUS_PALETTE["missing"]) for label in labels_consensus]
            ax.scatter(x, y, color=colors, s=34)
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel("Usable fraction")

        elif track == "Local preservation":
            if _finite_column(data, "median_neighbor_overlap"):
                ax.scatter(
                    _numeric(data["median_neighbor_overlap"]),
                    y,
                    label="Overlap",
                    color="#009E73",
                    s=28,
                )
            if _finite_column(data, "median_neighbor_jaccard"):
                ax.scatter(
                    _numeric(data["median_neighbor_jaccard"]),
                    y,
                    label="Jaccard",
                    color="#56B4E9",
                    marker="s",
                    s=24,
                )
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel("0-1")
            ax.legend(loc="lower right", fontsize=7, frameon=False)

        elif track == "Local distortion":
            if _finite_column(data, "median_local_shape_distortion"):
                ax.scatter(
                    _numeric(data["median_local_shape_distortion"]),
                    y,
                    label="Local",
                    color="#E69F00",
                    s=28,
                )
            if _finite_column(data, "median_global_scale_distortion"):
                ax.scatter(
                    _numeric(data["median_global_scale_distortion"]),
                    y,
                    label="Global",
                    color="#D55E00",
                    marker="s",
                    s=24,
                )
            if _finite_column(data, "worst_local_shape_distortion"):
                ax.scatter(
                    _numeric(data["worst_local_shape_distortion"]),
                    y,
                    label="Worst local",
                    color="#000000",
                    marker="x",
                    s=24,
                )
            ax.set_xlabel("abs log-ratio")
            ax.set_xlim(left=0)
            ax.legend(loc="lower right", fontsize=7, frameon=False)

        elif track == "Dynamics":
            x = _numeric(data["median_alignment_cosine"]).to_numpy(dtype=float)
            colors = np.where(x >= pos_thr, "#0072B2", np.where(x <= neg_thr, "#D55E00", "#999999"))
            ax.scatter(x, y, color=colors, s=30)
            ax.axvline(0.0, color="0.2", linewidth=0.8)
            ax.axvline(pos_thr, color="0.7", linewidth=0.8, linestyle=":")
            ax.axvline(neg_thr, color="0.7", linewidth=0.8, linestyle=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("Cosine")

        elif track == "Coverage":
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            for i, rec in data.iterrows():
                status = str(rec.get("coverage_status", "unknown"))
                warning = str(rec.get("warnings", ""))
                text = status
                if warning and warning != "nan":
                    text += " *"
                ax.text(0.0, i, text, va="center", fontsize=8)
            ax.grid(False)

        else:
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            for i, rec in data.iterrows():
                ax.text(0.0, i, str(rec.get("summary", "")), va="center", fontsize=8)
            ax.grid(False)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    _show_or_close(plt, bool(show))
    if return_data:
        return fig, {"report": data, "tracks": tracks}
    return fig


def _representation_tables(adata, store_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scgeo = adata.uns.get("scgeo", {})
    if store_key not in scgeo:
        raise KeyError(f"representation_stability result not found at adata.uns['scgeo'][{store_key!r}].")
    out = scgeo[store_key]
    per = out.get("per_rep_state")
    consensus = out.get("consensus_state")
    representation_summary = out.get("representation_summary")
    if not isinstance(per, pd.DataFrame) or per.empty:
        raise ValueError("representation_stability per_rep_state table is unavailable or empty.")
    if not isinstance(consensus, pd.DataFrame):
        consensus = pd.DataFrame()
    if not isinstance(representation_summary, pd.DataFrame):
        representation_summary = pd.DataFrame()
    return per.copy(), consensus.copy(), representation_summary.copy()


def _heatmap_matrix(per: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric = str(metric)
    aliases = {
        "normalized_displacement": "normalized_delta_norm",
        "normalized displacement": "normalized_delta_norm",
        "displacement_rank": "magnitude_rank",
        "rank": "magnitude_rank",
        "alignment class": "alignment_class",
    }
    column = aliases.get(metric, metric)
    if column == "alignment_class":
        values = per.assign(_value=per["alignment_class"].map(_CLASS_TO_VALUE))
        value_col = "_value"
    elif column in {"normalized_delta_norm", "magnitude_rank", "alignment_cosine"}:
        values = per.assign(_value=pd.to_numeric(per[column], errors="coerce"))
        value_col = "_value"
    else:
        raise ValueError(
            "metric must be one of 'normalized_delta_norm', 'displacement_rank', "
            "'alignment_cosine', or 'alignment_class'."
        )
    return values.pivot(index="node", columns="rep", values=value_col)


def _sort_matrix(matrix: pd.DataFrame, *, cluster_states: bool, cluster_reps: bool) -> pd.DataFrame:
    out = matrix.copy()
    if cluster_states and out.shape[0] > 1:
        order = out.mean(axis=1, skipna=True).sort_values(na_position="last").index
        out = out.loc[order]
    if cluster_reps and out.shape[1] > 1:
        order = out.mean(axis=0, skipna=True).sort_values(na_position="last").index
        out = out.loc[:, order]
    return out


def _local_representation_diagnostics(adata, reps: Sequence[str]) -> pd.DataFrame:
    scgeo = adata.uns.get("scgeo", {})
    local = scgeo.get("local_geometry_stability") if isinstance(scgeo, dict) else None
    if not isinstance(local, dict):
        return pd.DataFrame(index=pd.Index(reps, name="rep"))
    rep_summary = local.get("representation_summary")
    if not isinstance(rep_summary, pd.DataFrame) or "rep" not in rep_summary:
        return pd.DataFrame(index=pd.Index(reps, name="rep"))
    return rep_summary.set_index("rep").reindex(list(reps))


def representation_stability_heatmap(
    adata,
    *,
    store_key: str = "representation_stability",
    metric: str = "normalized_delta_norm",
    annotate_consensus: bool = True,
    cluster_states: bool = False,
    cluster_reps: bool = False,
    figsize=None,
    title=None,
    return_data: bool = False,
    show: bool = True,
):
    """Plot a state-by-representation heatmap from stored representation stability."""
    plt, ListedColormap = _lazy_matplotlib()
    per, consensus, _ = _representation_tables(adata, store_key)
    matrix = _sort_matrix(
        _heatmap_matrix(per, metric),
        cluster_states=bool(cluster_states),
        cluster_reps=bool(cluster_reps),
    )
    rep_diagnostics = _local_representation_diagnostics(adata, matrix.columns)
    annotations = pd.DataFrame(index=matrix.index)
    if not consensus.empty and "node" in consensus:
        ann = consensus.set_index("node")
        annotations["consensus_label"] = ann.reindex(matrix.index)["consensus_label"]
        annotations["usable_fraction"] = ann.reindex(matrix.index)["usable_fraction"]
        annotations["instability_warning"] = annotations["consensus_label"].isin(
            ["representation_unstable", "insufficient_coverage"]
        )

    width = max(5.5, 0.7 * matrix.shape[1] + (2.7 if annotate_consensus else 1.0))
    height = max(3.0, 0.35 * matrix.shape[0] + 1.2)
    fig = plt.figure(figsize=figsize or (width, height))
    has_rep_diagnostics = not rep_diagnostics.empty and any(
        col in rep_diagnostics for col in (
            "neighborhood_outlier",
            "distortion_outlier",
            "state_graph_outlier",
            "insufficient_coverage",
        )
    )
    if annotate_consensus:
        gs = fig.add_gridspec(
            2 if has_rep_diagnostics else 1,
            2,
            width_ratios=[max(1, matrix.shape[1]), 1.6],
            height_ratios=[max(1, matrix.shape[0]), 0.9] if has_rep_diagnostics else None,
            wspace=0.05,
            hspace=0.18,
        )
        ax = fig.add_subplot(gs[0, 0])
        ann_ax = fig.add_subplot(gs[0, 1], sharey=ax)
        diag_ax = fig.add_subplot(gs[1, 0], sharex=ax) if has_rep_diagnostics else None
    else:
        if has_rep_diagnostics:
            gs = fig.add_gridspec(2, 1, height_ratios=[max(1, matrix.shape[0]), 0.9], hspace=0.18)
            ax = fig.add_subplot(gs[0, 0])
            diag_ax = fig.add_subplot(gs[1, 0], sharex=ax)
        else:
            ax = fig.add_subplot(111)
            diag_ax = None
        ann_ax = None

    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    if str(metric) in {"alignment_class", "alignment class"}:
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#BDBDBD")
        im = ax.imshow(values, aspect="auto", vmin=-1, vmax=1, cmap=cmap)
    else:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#BDBDBD")
        im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_xlabel("Representation")
    ax.set_ylabel("State")
    ax.set_title(title or str(metric).replace("_", " "))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if ann_ax is not None:
        ann_ax.set_xlim(0, 1)
        ann_ax.set_xticks([])
        ann_ax.tick_params(labelleft=False, left=False)
        ann_ax.set_title("Consensus")
        for i, state in enumerate(matrix.index):
            label = str(annotations.get("consensus_label", pd.Series()).get(state, "missing"))
            frac = annotations.get("usable_fraction", pd.Series(dtype=float)).get(state, np.nan)
            warn = bool(annotations.get("instability_warning", pd.Series(dtype=bool)).get(state, False))
            text = label
            frac_value = float(frac) if not pd.isna(frac) else np.nan
            if np.isfinite(frac_value):
                text += f"\n{frac_value:.0%} usable"
            if warn:
                text += "\nwarning"
            ann_ax.text(
                0.0,
                i,
                text,
                va="center",
                fontsize=8,
                color=_CONSENSUS_PALETTE.get(label, "0.2"),
            )
        ann_ax.set_ylim(ax.get_ylim())
        ann_ax.grid(False)

    if has_rep_diagnostics and diag_ax is not None:
        flag_cols = [
            "neighborhood_outlier",
            "distortion_outlier",
            "state_graph_outlier",
            "insufficient_coverage",
        ]
        flag_labels = ["neighborhood", "distortion", "state graph", "coverage"]
        flag_values = np.full((len(flag_cols), matrix.shape[1]), np.nan, dtype=float)
        for i, col in enumerate(flag_cols):
            if col not in rep_diagnostics:
                continue
            vals = rep_diagnostics.reindex(matrix.columns)[col]
            for j, value in enumerate(vals):
                if pd.isna(value):
                    flag_values[i, j] = np.nan
                else:
                    flag_values[i, j] = 1.0 if bool(value) else 0.0
        cmap = ListedColormap(["#FFFFFF", "#D55E00"])
        cmap.set_bad("#BDBDBD")
        diag_ax.imshow(flag_values, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        diag_ax.set_yticks(np.arange(len(flag_cols)))
        diag_ax.set_yticklabels(flag_labels, fontsize=7)
        diag_ax.set_xticks(np.arange(matrix.shape[1]))
        diag_ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right")
        diag_ax.set_title("Representation diagnostics", fontsize=9)
        diag_ax.tick_params(axis="x", labelsize=8)

    fig.subplots_adjust(bottom=0.22, right=0.96)
    _show_or_close(plt, bool(show))
    if return_data:
        return fig, {
            "matrix": matrix,
            "annotations": annotations,
            "representation_diagnostics": rep_diagnostics,
        }
    return fig


def consensus_state_map(
    adata,
    *,
    node_key,
    basis: str = "umap",
    store_key: str = "representation_stability",
    label_states: bool = True,
    title=None,
    show: bool = True,
):
    """
    Plot state-level consensus labels on a display embedding.

    The display embedding is used only for visualization; consensus labels are
    read from ``representation_stability`` and are not recalculated here.
    """
    plt, _ = _lazy_matplotlib()
    if node_key not in adata.obs:
        raise KeyError(f"obs key {node_key!r} not found")
    key = _embedding_key(adata, basis)
    coords = np.asarray(adata.obsm[key])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(f"Embedding {key!r} must have shape (n_obs, >=2).")
    _, consensus, _ = _representation_tables(adata, store_key)
    if consensus.empty or "node" not in consensus or "consensus_label" not in consensus:
        raise ValueError("representation_stability consensus_state table is unavailable.")
    label_map = consensus.set_index("node")["consensus_label"].astype(str).to_dict()
    states = adata.obs[node_key].astype(str)
    labels = states.map(label_map).fillna("insufficient_coverage")
    colors = labels.map(lambda value: _CONSENSUS_PALETTE.get(value, _CONSENSUS_PALETTE["missing"]))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, linewidths=0, alpha=0.85)
    ax.set_xlabel(f"{basis} 1")
    ax.set_ylabel(f"{basis} 2")
    ax.set_title(title or "Display embedding only; consensus calculated across representations")
    if label_states:
        for state in pd.unique(states):
            mask = states == state
            if int(mask.sum()) == 0:
                continue
            center = np.median(coords[mask.to_numpy(), :2], axis=0)
            ax.text(center[0], center[1], str(state), ha="center", va="center", fontsize=8)
    handles = []
    for label, color in _CONSENSUS_PALETTE.items():
        if label == "missing":
            continue
        handles.append(
            ax.scatter([], [], c=color, s=24, label=label.replace("_", " "))
        )
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    _show_or_close(plt, bool(show))
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
    show: bool = True,
):
    """Create the standard ScGeo perturbation report bundle."""
    plt, _ = _lazy_matplotlib()
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
    warnings_out = sorted(
        {
            warning
            for warning in report_df.get("warnings", pd.Series(dtype=object)).astype(str)
            if warning and warning != "nan"
        }
    )
    warnings_out.extend(str(w) for w in report_df.attrs.get("warnings", []) if str(w))
    warnings_out = sorted(set(warnings_out))
    rep_diag = _records_to_frame(report_df.attrs.get("representation_diagnostics"))
    bundle: dict[str, Any] = {
        "state_report": report_df,
        "representation_diagnostics": rep_diag,
        "metadata": report_df.attrs.get("provenance", {}),
        "global_diagnostics": report_df.attrs.get("global_diagnostics", {}),
        "rules": report_df.attrs.get("scgeo_report_rules", {}),
        "consensus_state_map": None,
        "state_evidence_panel": None,
        "representation_heatmap": None,
        "warnings": warnings_out,
        "saved_paths": {},
        "alt_text": {},
    }

    try:
        bundle["consensus_state_map"] = consensus_state_map(
            adata,
            node_key=node_key,
            basis=basis,
            show=False,
        )
    except Exception as exc:
        bundle["warnings"].append(f"consensus_state_map unavailable: {type(exc).__name__}: {exc}")
    try:
        fig, panel_data = state_evidence_panel(report_df, return_data=True, show=False)
        bundle["state_evidence_panel"] = fig
        bundle["alt_text"]["state_evidence_panel"] = _alt_text_state_evidence(
            panel_data["report"],
            panel_data["tracks"],
            str(comparison),
        )
    except Exception as exc:
        bundle["warnings"].append(f"state_evidence_panel unavailable: {type(exc).__name__}: {exc}")
    try:
        fig, heatmap_data = representation_stability_heatmap(adata, return_data=True, show=False)
        bundle["representation_heatmap"] = fig
        bundle["alt_text"]["representation_heatmap"] = _alt_text_heatmap(
            heatmap_data["matrix"],
            heatmap_data["annotations"],
            str(comparison),
        )
    except Exception as exc:
        bundle["warnings"].append(f"representation_heatmap unavailable: {type(exc).__name__}: {exc}")
    if bundle["consensus_state_map"] is not None:
        bundle["alt_text"]["consensus_state_map"] = _alt_text_consensus_map(
            report_df,
            str(comparison),
            str(basis),
        )

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{prefix}_{safe_comparison}_state_report.csv"
        report_df.to_csv(report_path, index=False)
        bundle["saved_paths"]["state_report_csv"] = str(report_path)

        rep_diag_path = out_dir / f"{prefix}_{safe_comparison}_representation_diagnostics.csv"
        rep_diag.to_csv(rep_diag_path, index=False)
        bundle["saved_paths"]["representation_diagnostics_csv"] = str(rep_diag_path)

        metadata = {
            "provenance": report_df.attrs.get("provenance", {}),
            "rules": report_df.attrs.get("scgeo_report_rules", {}),
            "global_diagnostics": report_df.attrs.get("global_diagnostics", {}),
            "alt_text": bundle["alt_text"],
        }
        metadata_path = out_dir / f"{prefix}_{safe_comparison}_metadata.json"
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
        bundle["saved_paths"]["metadata_json"] = str(metadata_path)

        warnings_path = out_dir / f"{prefix}_{safe_comparison}_warnings.txt"
        warnings_path.write_text("\n".join(bundle["warnings"]) + ("\n" if bundle["warnings"] else ""), encoding="utf-8")
        bundle["saved_paths"]["warnings_txt"] = str(warnings_path)

        alt_text_path = out_dir / f"{prefix}_{safe_comparison}_alt_text.txt"
        alt_text_path.write_text(
            "\n\n".join(f"{key}: {value}" for key, value in sorted(bundle["alt_text"].items())),
            encoding="utf-8",
        )
        bundle["saved_paths"]["alt_text_txt"] = str(alt_text_path)

        for key, fig in [
            ("consensus_state_map", bundle["consensus_state_map"]),
            ("state_evidence_panel", bundle["state_evidence_panel"]),
            ("representation_heatmap", bundle["representation_heatmap"]),
        ]:
            if fig is None:
                continue
            for ext in ("png", "svg"):
                path = out_dir / f"{prefix}_{safe_comparison}_{key}.{ext}"
                kwargs = {"dpi": 150} if ext == "png" else {}
                fig.savefig(path, bbox_inches="tight", **kwargs)
                bundle["saved_paths"][f"{key}_{ext}"] = str(path)

    if show:
        for fig in [
            bundle["consensus_state_map"],
            bundle["state_evidence_panel"],
            bundle["representation_heatmap"],
        ]:
            if fig is not None:
                fig.show()
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
        raise ValueError(f"metric {metric!r} is not available in stored per-cell local geometry.")

    df = per_cell.copy()
    if (rep_a is None) ^ (rep_b is None):
        raise ValueError("rep_a and rep_b must be provided together, or both left as None.")
    if rep_a is not None and rep_b is not None:
        forward = (df["rep_a"].astype(str) == str(rep_a)) & (df["rep_b"].astype(str) == str(rep_b))
        reverse = (df["rep_a"].astype(str) == str(rep_b)) & (df["rep_b"].astype(str) == str(rep_a))
        df = df[forward | reverse]
        if df.empty:
            raise ValueError(f"No per-cell values found for representation pair {rep_a!r}, {rep_b!r}.")
        values = df.groupby("cell_index", sort=False)[column].median()
    else:
        if aggregation in {"median", "per_cell"}:
            values = df.groupby("cell_index", sort=False)[column].median()
        elif aggregation in {"worst", "worst_case", "worst-case", "max"}:
            values = df.groupby("cell_index", sort=False)[column].max()
        else:
            raise ValueError("aggregation must be 'median' or 'worst-case'.")
    return pd.DataFrame({"cell_index": values.index.astype(int), "value": values.to_numpy(dtype=float)})


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
    resolved_node_key = node_key or (params.get("node_key") if isinstance(params, dict) else None)
    if resolved_node_key is None:
        raise ValueError("aggregation='state' requires node_key or a stored local_geometry_stability node_key.")
    if resolved_node_key not in adata.obs:
        raise KeyError(f"obs key {resolved_node_key!r} not found")
    state_pair = local_out.get("state_pair_summary")
    if not isinstance(state_pair, pd.DataFrame) or state_pair.empty:
        raise ValueError("aggregation='state' requires stored state_pair_summary values.")
    aliases = {
        "local_shape_distortion": "local_distortion_median",
        "global_scale_distortion": "global_distortion_median",
    }
    metric_name = aliases.get(str(metric), str(metric))
    df = state_pair[state_pair["metric"] == metric_name].copy()
    if rep_a is not None and rep_b is not None:
        forward = (df["rep_a"].astype(str) == str(rep_a)) & (df["rep_b"].astype(str) == str(rep_b))
        reverse = (df["rep_a"].astype(str) == str(rep_b)) & (df["rep_b"].astype(str) == str(rep_a))
        df = df[forward | reverse]
    if df.empty:
        raise ValueError(f"No state-level local geometry values found for metric {metric!r}.")
    df["median"] = pd.to_numeric(df["median"], errors="coerce")
    state_values = df.groupby("state", sort=False)["median"].median()
    states = adata.obs[resolved_node_key].astype(str).to_numpy()
    values = np.asarray([state_values.get(state, np.nan) for state in states], dtype=float)
    return pd.DataFrame({"cell_index": np.arange(adata.n_obs, dtype=int), "value": values})


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
    title=None,
    show: bool = True,
):
    """Plot stored per-cell local distortion values on a display embedding."""
    plt, _ = _lazy_matplotlib()
    scgeo = adata.uns.get("scgeo", {})
    if store_key not in scgeo:
        raise KeyError(f"local_geometry_stability result not found at adata.uns['scgeo'][{store_key!r}].")
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
    key = _embedding_key(adata, basis)
    coords = np.asarray(adata.obsm[key])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(f"Embedding {key!r} must have shape (n_obs, >=2).")

    color = np.full(adata.n_obs, np.nan, dtype=float)
    valid_idx = values["cell_index"].to_numpy(dtype=int)
    color[valid_idx] = values["value"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, s=12, cmap="magma", linewidths=0)
    fig.colorbar(sc, ax=ax, label=str(metric).replace("_", " "))
    ax.set_xlabel(f"{basis} 1")
    ax.set_ylabel(f"{basis} 2")
    if title is None:
        if state_level:
            title = f"State-level display of {metric}; no per-cell statistic recomputed"
        elif rep_a is not None and rep_b is not None:
            title = f"{metric}: {rep_a} vs {rep_b}"
        else:
            title = f"{metric}: {aggregation} across representation pairs"
    ax.set_title(title)
    fig.tight_layout()
    _show_or_close(plt, bool(show))
    return fig
