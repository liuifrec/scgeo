from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import __version__


_STATUS_VALUES = {
    "assessed",
    "not_computed",
    "unavailable",
    "insufficient_coverage",
    "representation_unstable",
    "numerical_degeneracy",
}


def _scgeo_store(adata) -> dict[str, Any]:
    store = adata.uns.get("scgeo", {})
    return store if isinstance(store, dict) else {}


def _load_result(
    adata,
    key: Optional[str],
    *,
    name: str,
    strict: bool,
    warnings: list[str],
) -> Optional[dict[str, Any]]:
    if key is None:
        return None
    store = _scgeo_store(adata)
    if key not in store:
        msg = f"{name} result not found at adata.uns['scgeo'][{key!r}]."
        if strict:
            raise KeyError(msg)
        warnings.append(msg)
        return None
    result = store[key]
    if not isinstance(result, dict):
        msg = f"{name} result at adata.uns['scgeo'][{key!r}] is not a dictionary."
        if strict:
            raise TypeError(msg)
        warnings.append(msg)
        return None
    return result


def _unique_extend(out: list[str], values) -> None:
    seen = set(out)
    for value in values:
        if pd.isna(value):
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)


def _resolve_local_k(local_k, available: Sequence[int] | None) -> list[int]:
    if available is None:
        available_values: list[int] = []
    else:
        available_values = sorted({int(k) for k in available})
    if local_k is None:
        return available_values
    if isinstance(local_k, (str, bytes)) or not isinstance(local_k, Sequence):
        requested = [int(local_k)]
    else:
        requested = [int(k) for k in local_k]
    if not requested:
        raise ValueError("local_k must be None, an integer, or a non-empty sequence of integers")
    missing = sorted(set(requested) - set(available_values))
    if missing and available_values:
        raise ValueError(f"Requested local_k values are not available: {missing}; available={available_values}")
    return requested


def _format_values(values: Sequence[Any]) -> str:
    out = []
    for value in values:
        try:
            if np.isfinite(float(value)):
                out.append(str(int(value)) if float(value).is_integer() else f"{float(value):.3g}")
        except (TypeError, ValueError):
            continue
    return ",".join(out)


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value))


def _states_from_inputs(
    adata,
    *,
    node_key: Optional[str],
    robust: Optional[dict[str, Any]],
    representation: Optional[dict[str, Any]],
    local_geometry: Optional[dict[str, Any]],
) -> list[str]:
    states: list[str] = []

    if representation is not None:
        consensus = representation.get("consensus_state")
        if isinstance(consensus, pd.DataFrame) and "node" in consensus:
            _unique_extend(states, consensus["node"].tolist())

    if robust is not None and isinstance(robust.get("by"), dict):
        _unique_extend(states, robust["by"].keys())

    if local_geometry is not None:
        coverage = local_geometry.get("coverage_summary", {})
        order = coverage.get("state_order") if isinstance(coverage, dict) else None
        if order is not None:
            _unique_extend(states, order)
        state_pair = local_geometry.get("state_pair_summary")
        if isinstance(state_pair, pd.DataFrame) and "state" in state_pair:
            _unique_extend(states, state_pair["state"].dropna().tolist())

    if node_key is not None and node_key in adata.obs:
        _unique_extend(states, adata.obs[node_key].astype(str).tolist())

    if not states and robust is not None and "global" in robust:
        states.append("global")

    return states


def _empty_rows(states: list[str]) -> dict[str, dict[str, Any]]:
    return {
        state: {
            "state": state,
            "n_cells0": np.nan,
            "n_cells1": np.nan,
            "n_cells_total": np.nan,
            "n_samples0": np.nan,
            "n_samples1": np.nan,
            "delta_norm": np.nan,
            "normalized_delta_norm": np.nan,
            "magnitude_ci95_low": np.nan,
            "magnitude_ci95_high": np.nan,
            "directional_stability": np.nan,
            "outlier_delta_difference_norm": np.nan,
            "outlier_relative_norm_change": np.nan,
            "outlier_cosine_to_mean": np.nan,
            "bootstrap_unit": pd.NA,
            "inference_level": "not_computed",
            "n_samples_group0": np.nan,
            "n_samples_group1": np.nan,
            "descriptive_only": True,
            "representation_coverage_count": np.nan,
            "representation_coverage_fraction": np.nan,
            "representation_consensus_label": pd.NA,
            "magnitude_rank_median": np.nan,
            "magnitude_rank_mean": np.nan,
            "magnitude_rank_std": np.nan,
            "leave_one_rep_sensitivity": np.nan,
            "median_neighbor_overlap": np.nan,
            "median_neighbor_jaccard": np.nan,
            "median_local_shape_distortion": np.nan,
            "median_global_scale_distortion": np.nan,
            "worst_neighbor_overlap": np.nan,
            "worst_neighbor_jaccard": np.nan,
            "worst_local_shape_distortion": np.nan,
            "worst_global_scale_distortion": np.nan,
            "neighbor_overlap_across_k_std": np.nan,
            "neighbor_jaccard_across_k_std": np.nan,
            "local_shape_distortion_across_k_std": np.nan,
            "global_scale_distortion_across_k_std": np.nan,
            "local_k_values": "",
            "local_n_k_values": 0,
            "local_n_valid_pairs": 0,
            "state_graph_agreement": np.nan,
            "median_alignment_cosine": np.nan,
            "aligned_fraction": np.nan,
            "discordant_fraction": np.nan,
            "neutral_fraction": np.nan,
            "effect_status": "not_computed",
            "stability_status": "not_computed",
            "local_geometry_status": "not_computed",
            "dynamics_status": "not_computed",
            "coverage_status": "not_computed",
            "warnings": [],
            "reason_codes": [],
            "summary": "",
        }
        for state in states
    }


def _coerce_count(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _copy_shift_row(row: dict[str, Any], shift: dict[str, Any]) -> None:
    row["n_cells0"] = _coerce_count(shift.get("n_cells0", shift.get("n0", np.nan)))
    row["n_cells1"] = _coerce_count(shift.get("n_cells1", shift.get("n1", np.nan)))
    if np.isfinite(row["n_cells0"]) and np.isfinite(row["n_cells1"]):
        row["n_cells_total"] = row["n_cells0"] + row["n_cells1"]
    row["n_samples0"] = _coerce_count(shift.get("n_samples0", np.nan))
    row["n_samples1"] = _coerce_count(shift.get("n_samples1", np.nan))
    row["delta_norm"] = float(shift.get("delta_norm", np.nan))
    row["normalized_delta_norm"] = float(shift.get("normalized_delta_norm", np.nan))
    if (row["n_cells0"] == 0) or (row["n_cells1"] == 0):
        row["effect_status"] = "insufficient_coverage"
    elif np.isfinite(row["delta_norm"]) or np.isfinite(row["normalized_delta_norm"]):
        row["effect_status"] = "assessed"
    else:
        row["effect_status"] = "numerical_degeneracy"
    ci = shift.get("bootstrap_magnitude_ci95", [np.nan, np.nan])
    if ci is None or len(ci) != 2:
        ci = [np.nan, np.nan]
    row["magnitude_ci95_low"] = float(ci[0])
    row["magnitude_ci95_high"] = float(ci[1])
    row["directional_stability"] = float(
        shift.get("bootstrap_directional_resultant_length", np.nan)
    )
    sensitivity = shift.get("outlier_sensitivity") or {}
    row["outlier_delta_difference_norm"] = float(
        sensitivity.get("delta_difference_norm", np.nan)
    )
    row["outlier_relative_norm_change"] = float(
        sensitivity.get("relative_norm_change", np.nan)
    )
    row["outlier_cosine_to_mean"] = float(sensitivity.get("cosine_to_mean", np.nan))
    row["n_samples_group0"] = row["n_samples0"]
    row["n_samples_group1"] = row["n_samples1"]


def _merge_robust_shift(
    rows: dict[str, dict[str, Any]],
    robust: Optional[dict[str, Any]],
    *,
    node_key: Optional[str],
) -> None:
    if robust is None:
        return

    by_results = robust.get("by")
    if isinstance(by_results, dict):
        for state, shift in by_results.items():
            key = str(state)
            if key in rows and isinstance(shift, dict):
                _copy_shift_row(rows[key], shift)
        params = robust.get("params", {})
        by_key = params.get("by") if isinstance(params, dict) else None
        if node_key is not None and by_key is not None and by_key != node_key:
            for row in rows.values():
                row["warnings"].append(
                    f"robust_shift was stratified by {by_key!r}, not node_key={node_key!r}."
                )
        return

    if list(rows.keys()) == ["global"] and isinstance(robust.get("global"), dict):
        _copy_shift_row(rows["global"], robust["global"])
    elif isinstance(robust.get("global"), dict):
        for row in rows.values():
            row["warnings"].append("robust_shift is global and was not stratified by state.")


def _apply_inference_metadata(
    rows: dict[str, dict[str, Any]],
    robust: Optional[dict[str, Any]],
    representation: Optional[dict[str, Any]],
) -> None:
    params: dict[str, Any] = {}
    if isinstance(robust, dict) and isinstance(robust.get("params"), dict):
        params.update(robust["params"])
    elif isinstance(representation, dict) and isinstance(representation.get("params"), dict):
        params.update(representation["params"])

    bootstrap_unit = params.get("resolved_bootstrap_unit", params.get("bootstrap_unit", pd.NA))
    sample_key = params.get("sample_key")
    for row in rows.values():
        if not _is_missing(bootstrap_unit) and bootstrap_unit == "sample" and sample_key is not None:
            n0 = row.get("n_samples0", np.nan)
            n1 = row.get("n_samples1", np.nan)
            if np.isfinite(n0) and np.isfinite(n1) and n0 >= 2 and n1 >= 2:
                inference_level = "biological_sample"
                descriptive_only = False
            else:
                inference_level = "sample_descriptive"
                descriptive_only = True
                row["warnings"].append(
                    "sample-level bootstrap has fewer than two biological samples in at least one group."
                )
        elif not _is_missing(bootstrap_unit) and bootstrap_unit == "cell":
            inference_level = "cell_descriptive"
            descriptive_only = True
        elif _is_missing(bootstrap_unit):
            inference_level = "not_computed"
            descriptive_only = True
        else:
            inference_level = "descriptive"
            descriptive_only = True
        row["bootstrap_unit"] = bootstrap_unit
        row["inference_level"] = inference_level
        row["descriptive_only"] = bool(descriptive_only)


def _fill_counts_from_representation(
    rows: dict[str, dict[str, Any]],
    representation: Optional[dict[str, Any]],
) -> None:
    if representation is None:
        return
    per = representation.get("per_rep_state")
    if not isinstance(per, pd.DataFrame) or per.empty or "node" not in per:
        return
    for state, df_state in per.groupby("node", sort=False):
        key = str(state)
        if key not in rows:
            continue
        row = rows[key]
        for col in ("n_cells0", "n_cells1", "n_samples0", "n_samples1"):
            if col not in df_state or np.isfinite(row.get(col, np.nan)):
                continue
            vals = pd.to_numeric(df_state[col], errors="coerce").dropna()
            if not vals.empty:
                row[col] = float(vals.max())
        if np.isfinite(row.get("n_cells0", np.nan)) and np.isfinite(row.get("n_cells1", np.nan)):
            row["n_cells_total"] = row["n_cells0"] + row["n_cells1"]


def _merge_representation(
    rows: dict[str, dict[str, Any]],
    representation: Optional[dict[str, Any]],
) -> None:
    if representation is None:
        return
    consensus = representation.get("consensus_state")
    if not isinstance(consensus, pd.DataFrame) or consensus.empty or "node" not in consensus:
        return

    params = representation.get("params", {})
    pos_thr = float(params.get("alignment_pos_thr", 0.3)) if isinstance(params, dict) else 0.3
    neg_thr = float(params.get("alignment_neg_thr", -0.3)) if isinstance(params, dict) else -0.3
    for rec in consensus.to_dict(orient="records"):
        key = str(rec.get("node"))
        if key not in rows:
            continue
        row = rows[key]
        row["representation_coverage_count"] = float(
            rec.get("n_usable_representations", np.nan)
        )
        row["representation_coverage_fraction"] = float(rec.get("usable_fraction", np.nan))
        row["representation_consensus_label"] = rec.get("consensus_label", pd.NA)
        row["magnitude_rank_mean"] = float(rec.get("magnitude_rank_mean", np.nan))
        row["magnitude_rank_std"] = float(rec.get("magnitude_rank_std", np.nan))
        row["leave_one_rep_sensitivity"] = float(
            rec.get("loo_rep_magnitude_max_relative_deviation", np.nan)
        )
        row["median_alignment_cosine"] = float(rec.get("alignment_cosine_median", np.nan))
        row["aligned_fraction"] = float(rec.get("aligned_fraction", np.nan))
        row["discordant_fraction"] = float(rec.get("discordant_fraction", np.nan))
        row["neutral_fraction"] = float(rec.get("neutral_fraction", np.nan))
        row["alignment_pos_thr"] = pos_thr
        row["alignment_neg_thr"] = neg_thr
        label = rec.get("consensus_label")
        if label == "insufficient_coverage":
            row["stability_status"] = "insufficient_coverage"
        elif label == "representation_unstable":
            row["stability_status"] = "representation_unstable"
        elif pd.notna(label):
            row["stability_status"] = "assessed"

        if np.isfinite(row["median_alignment_cosine"]):
            row["dynamics_status"] = "assessed"
        elif np.isfinite(row["aligned_fraction"]) or np.isfinite(row["discordant_fraction"]) or np.isfinite(row["neutral_fraction"]):
            row["dynamics_status"] = "unavailable"
        else:
            row["dynamics_status"] = "unavailable"

    _fill_counts_from_representation(rows, representation)
    per = representation.get("per_rep_state")
    if isinstance(per, pd.DataFrame) and {"node", "magnitude_rank"} <= set(per.columns):
        for state, df_state in per.groupby("node", sort=False):
            key = str(state)
            if key not in rows:
                continue
            vals = pd.to_numeric(df_state["magnitude_rank"], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if not vals.empty:
                rows[key]["magnitude_rank_median"] = float(vals.median())


def _median_metric(df: pd.DataFrame, metric: str) -> float:
    if df.empty or "metric" not in df or "median" not in df:
        return np.nan
    values = pd.to_numeric(df.loc[df["metric"] == metric, "median"], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if not values.empty else np.nan


def _pair_aggregate(values: pd.Series, method: str) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return np.nan
    if method == "median":
        return float(finite.median())
    if method == "mean":
        return float(finite.mean())
    raise ValueError("pair_aggregation must be one of {'median', 'mean'}")


def _metric_across_k(
    df: pd.DataFrame,
    metric: str,
    *,
    pair_aggregation: str,
    worst_kind: str,
) -> tuple[float, float, float]:
    if df.empty or "metric" not in df or "median" not in df or "k" not in df:
        return np.nan, np.nan, np.nan
    metric_df = df[df["metric"] == metric].copy()
    if metric_df.empty:
        return np.nan, np.nan, np.nan
    metric_df["median"] = pd.to_numeric(metric_df["median"], errors="coerce")
    metric_df = metric_df[np.isfinite(metric_df["median"])]
    if metric_df.empty:
        return np.nan, np.nan, np.nan

    by_k = metric_df.groupby("k", sort=True)["median"].apply(
        lambda values: _pair_aggregate(values, pair_aggregation)
    )
    by_k = by_k[np.isfinite(by_k)]
    summary = float(by_k.median()) if not by_k.empty else np.nan
    variability = float(by_k.std(ddof=0)) if by_k.shape[0] > 1 else np.nan

    if worst_kind == "min":
        worst = float(metric_df["median"].min())
    elif worst_kind == "max":
        worst = float(metric_df["median"].max())
    else:
        worst = np.nan
    return summary, worst, variability


def _valid_pair_count(df: pd.DataFrame) -> int:
    if df.empty or not {"rep_a", "rep_b", "median"} <= set(df.columns):
        return 0
    work = df.copy()
    work["median"] = pd.to_numeric(work["median"], errors="coerce")
    work = work[np.isfinite(work["median"])]
    if "status" in work:
        work = work[work["status"].astype(str) == "ok"]
    if work.empty:
        return 0
    return int(work[["rep_a", "rep_b"]].drop_duplicates().shape[0])


def _merge_local_geometry(
    rows: dict[str, dict[str, Any]],
    local_geometry: Optional[dict[str, Any]],
    *,
    local_k_values: Sequence[int],
    pair_aggregation: str,
    include_worst_case: bool,
) -> None:
    if local_geometry is None:
        return

    state_pair = local_geometry.get("state_pair_summary")
    if isinstance(state_pair, pd.DataFrame) and not state_pair.empty:
        if local_k_values and "k" in state_pair:
            state_pair = state_pair[state_pair["k"].astype(int).isin(local_k_values)]
        for state, df_state in state_pair.groupby("state", sort=False):
            key = str(state)
            if key not in rows:
                continue
            row = rows[key]
            overlap, overlap_worst, overlap_std = _metric_across_k(
                df_state,
                "neighbor_overlap",
                pair_aggregation=pair_aggregation,
                worst_kind="min",
            )
            jaccard, jaccard_worst, jaccard_std = _metric_across_k(
                df_state,
                "neighbor_jaccard",
                pair_aggregation=pair_aggregation,
                worst_kind="min",
            )
            local_dist, local_worst, local_std = _metric_across_k(
                df_state,
                "local_distortion_median",
                pair_aggregation=pair_aggregation,
                worst_kind="max",
            )
            global_dist, global_worst, global_std = _metric_across_k(
                df_state,
                "global_distortion_median",
                pair_aggregation=pair_aggregation,
                worst_kind="max",
            )
            row["median_neighbor_overlap"] = overlap
            row["median_neighbor_jaccard"] = jaccard
            row["median_local_shape_distortion"] = local_dist
            row["median_global_scale_distortion"] = global_dist
            if include_worst_case:
                row["worst_neighbor_overlap"] = overlap_worst
                row["worst_neighbor_jaccard"] = jaccard_worst
                row["worst_local_shape_distortion"] = local_worst
                row["worst_global_scale_distortion"] = global_worst
            row["neighbor_overlap_across_k_std"] = overlap_std
            row["neighbor_jaccard_across_k_std"] = jaccard_std
            row["local_shape_distortion_across_k_std"] = local_std
            row["global_scale_distortion_across_k_std"] = global_std
            used_k = sorted({int(k) for k in df_state["k"].dropna().unique()}) if "k" in df_state else []
            row["local_k_values"] = _format_values(used_k)
            row["local_n_k_values"] = len(used_k)
            row["local_n_valid_pairs"] = _valid_pair_count(df_state)
            statuses = sorted(
                {
                    str(value)
                    for value in df_state.get("status", pd.Series(dtype=object)).dropna()
                    if str(value) != "ok"
                }
            )
            if statuses:
                row["warnings"].append(
                    "local_geometry_stability state status: " + ", ".join(statuses)
                )
                if any(status.startswith("underpowered") for status in statuses):
                    row["local_geometry_status"] = "insufficient_coverage"
            elif any(
                np.isfinite(row[col])
                for col in (
                    "median_neighbor_overlap",
                    "median_neighbor_jaccard",
                    "median_local_shape_distortion",
                    "median_global_scale_distortion",
                )
            ):
                row["local_geometry_status"] = "assessed"
            else:
                row["local_geometry_status"] = "numerical_degeneracy"
    elif list(rows.keys()) == ["global"]:
        pair_summary = local_geometry.get("pair_summary")
        if isinstance(pair_summary, pd.DataFrame) and not pair_summary.empty:
            df_global = pair_summary[pair_summary["scope"] == "global"]
            if local_k_values and "k" in df_global:
                df_global = df_global[df_global["k"].astype(int).isin(local_k_values)]
            row = rows["global"]
            row["median_neighbor_overlap"], row["worst_neighbor_overlap"], row["neighbor_overlap_across_k_std"] = _metric_across_k(
                df_global,
                "neighbor_overlap",
                pair_aggregation=pair_aggregation,
                worst_kind="min",
            )
            row["median_neighbor_jaccard"], row["worst_neighbor_jaccard"], row["neighbor_jaccard_across_k_std"] = _metric_across_k(
                df_global,
                "neighbor_jaccard",
                pair_aggregation=pair_aggregation,
                worst_kind="min",
            )
            row["median_local_shape_distortion"], row["worst_local_shape_distortion"], row["local_shape_distortion_across_k_std"] = _metric_across_k(
                df_global,
                "local_distortion_median",
                pair_aggregation=pair_aggregation,
                worst_kind="max",
            )
            row["median_global_scale_distortion"], row["worst_global_scale_distortion"], row["global_scale_distortion_across_k_std"] = _metric_across_k(
                df_global,
                "global_distortion_median",
                pair_aggregation=pair_aggregation,
                worst_kind="max",
            )
            used_k = sorted({int(k) for k in df_global["k"].dropna().unique()}) if "k" in df_global else []
            row["local_k_values"] = _format_values(used_k)
            row["local_n_k_values"] = len(used_k)
            row["local_n_valid_pairs"] = _valid_pair_count(df_global)
            row["local_geometry_status"] = "assessed"


def _append_reason_codes(row: dict[str, Any]) -> None:
    codes: list[str] = []

    label_value = row.get("representation_consensus_label")
    label = None if pd.isna(label_value) else str(label_value)
    if label in {"stable_aligned", "stable_discordant", "stable_neutral"}:
        codes.append("stable_across_representations")
    if label == "representation_unstable":
        codes.append("representation_sensitive")
    if label == "insufficient_coverage" or row.get("coverage_status") == "insufficient_coverage":
        codes.append("insufficient_coverage")

    cosine = float(row.get("median_alignment_cosine", np.nan))
    pos_thr = float(row.get("alignment_pos_thr", 0.3))
    neg_thr = float(row.get("alignment_neg_thr", -0.3))
    if np.isfinite(cosine):
        if cosine >= pos_thr:
            codes.append("dynamics_aligned")
        elif cosine <= neg_thr:
            codes.append("dynamics_discordant")
        else:
            codes.append("dynamics_unavailable")
    elif not any(np.isfinite(float(row.get(col, np.nan))) for col in ("aligned_fraction", "discordant_fraction", "neutral_fraction")):
        codes.append("dynamics_unavailable")

    row["reason_codes"] = codes


def _format_count(value: Any) -> str:
    try:
        if np.isfinite(float(value)):
            return str(int(round(float(value))))
    except (TypeError, ValueError):
        pass
    return "?"


def _summary(row: dict[str, Any]) -> str:
    codes = set(row.get("reason_codes", []))
    parts: list[str] = []

    norm = float(row.get("normalized_delta_norm", np.nan))
    raw = float(row.get("delta_norm", np.nan))
    if np.isfinite(norm):
        parts.append(f"Normalized displacement {norm:.3g}")
    elif np.isfinite(raw):
        parts.append(f"Displacement magnitude {raw:.3g}")
    else:
        parts.append("Effect unavailable")

    label_value = row.get("representation_consensus_label")
    label = None if pd.isna(label_value) else str(label_value)
    if "stable_across_representations" in codes:
        used = _format_count(row.get("representation_coverage_count"))
        total_fraction = row.get("representation_coverage_fraction", np.nan)
        if np.isfinite(float(total_fraction)):
            parts.append(f"stable across {used} representations ({float(total_fraction):.0%} usable)")
        else:
            parts.append("stable across representations")
    elif label == "representation_unstable":
        parts.append("representation-sensitive")
    elif label == "insufficient_coverage":
        parts.append("insufficient representation coverage")

    overlap = float(row.get("median_neighbor_overlap", np.nan))
    local_dist = float(row.get("median_local_shape_distortion", np.nan))
    local_bits = []
    if np.isfinite(overlap):
        local_bits.append(f"median neighbor overlap {overlap:.3g}")
    if np.isfinite(local_dist):
        text = f"median local-shape distortion {local_dist:.3g}"
        worst = float(row.get("worst_local_shape_distortion", np.nan))
        if np.isfinite(worst):
            text += f" (worst {worst:.3g})"
        local_bits.append(text)
    if local_bits:
        k_values = row.get("local_k_values", "")
        suffix = f" across k={k_values}" if k_values else ""
        parts.append("; ".join(local_bits) + suffix)

    if "dynamics_aligned" in codes:
        parts.append(f"dynamics aligned (cosine {float(row.get('median_alignment_cosine')):.3g})")
    elif "dynamics_discordant" in codes:
        parts.append(f"dynamics discordant (cosine {float(row.get('median_alignment_cosine')):.3g})")
    elif "dynamics_unavailable" in codes:
        parts.append("dynamics unavailable")

    if "insufficient_coverage" in codes and "insufficient representation coverage" not in parts:
        parts.append("insufficient coverage")

    return "; ".join(parts) + "."


def _combined_coverage_status(row: dict[str, Any]) -> str:
    statuses = [
        str(row.get("effect_status", "not_computed")),
        str(row.get("stability_status", "not_computed")),
        str(row.get("local_geometry_status", "not_computed")),
        str(row.get("dynamics_status", "not_computed")),
    ]
    if "insufficient_coverage" in statuses:
        return "insufficient_coverage"
    if "representation_unstable" in statuses:
        return "representation_unstable"
    if "numerical_degeneracy" in statuses:
        return "numerical_degeneracy"
    if "assessed" in statuses:
        return "assessed"
    if "unavailable" in statuses:
        return "unavailable"
    return "not_computed"


def _first_params(*results: Optional[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for result in results:
        params = result.get("params") if isinstance(result, dict) else None
        if isinstance(params, dict):
            for key, value in params.items():
                out.setdefault(key, value)
    return out


def _comparison_label(condition_key: Any, group0: Any, group1: Any, comparison_label: Optional[str]) -> str:
    if comparison_label is not None:
        return str(comparison_label)
    if group0 is not None and group1 is not None:
        return f"{group1}_vs_{group0}"
    if condition_key is not None:
        return str(condition_key)
    return "comparison"


def _metadata(
    *,
    robust: Optional[dict[str, Any]],
    representation: Optional[dict[str, Any]],
    local_geometry: Optional[dict[str, Any]],
    robust_shift_key: str,
    representation_key: str,
    local_geometry_key: str,
    comparison_label: Optional[str],
    condition_key: Optional[str],
    group0: Any,
    group1: Any,
    local_k_values: Sequence[int],
    pair_aggregation: str,
    include_worst_case: bool,
) -> dict[str, Any]:
    params = _first_params(robust, representation, local_geometry)
    condition = condition_key if condition_key is not None else params.get("condition_key")
    g0 = group0 if group0 is not None else params.get("group0")
    g1 = group1 if group1 is not None else params.get("group1")
    reps = []
    for result in (representation, local_geometry):
        p = result.get("params") if isinstance(result, dict) else None
        if isinstance(p, dict) and p.get("reps") is not None:
            reps = list(p["reps"])
            break
    bootstrap_unit = params.get("resolved_bootstrap_unit", params.get("bootstrap_unit"))
    sample_key = params.get("sample_key")
    consensus_rules = None
    rep_params = representation.get("params") if isinstance(representation, dict) else None
    if isinstance(rep_params, dict):
        consensus_rules = rep_params.get("consensus_label_rules")
    return {
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_store_keys": {
            "robust_shift": robust_shift_key,
            "representation_stability": representation_key,
            "local_geometry_stability": local_geometry_key,
        },
        "comparison_label": _comparison_label(condition, g0, g1, comparison_label),
        "condition_key": condition,
        "group0": g0,
        "group1": g1,
        "representations": reps,
        "k_values": list(local_k_values),
        "sample_key": sample_key,
        "center": params.get("center"),
        "bootstrap_unit": bootstrap_unit,
        "consensus_rules": consensus_rules,
        "local_pair_aggregation": pair_aggregation,
        "include_worst_case": bool(include_worst_case),
    }


def _global_diagnostics(
    local_geometry: Optional[dict[str, Any]],
    *,
    local_k_values: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(local_geometry, dict):
        return {}
    out: dict[str, Any] = {}
    graph = local_geometry.get("state_graph_summary")
    if isinstance(graph, pd.DataFrame):
        work = graph.copy()
        if local_k_values and "k" in work:
            work = work[work["k"].astype(int).isin(local_k_values)]
        out["state_graph_summary"] = {
            "columns": list(work.columns),
            "records": work.where(pd.notna(work), None).to_dict(orient="records"),
        }
    pair = local_geometry.get("pair_summary")
    if isinstance(pair, pd.DataFrame):
        work = pair[pair["scope"] == "global"].copy() if "scope" in pair else pair.copy()
        if local_k_values and "k" in work:
            work = work[work["k"].astype(int).isin(local_k_values)]
        out["local_pair_summary_global"] = {
            "columns": list(work.columns),
            "records": work.where(pd.notna(work), None).to_dict(orient="records"),
        }
    return out


def _representation_diagnostics(local_geometry: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if isinstance(local_geometry, dict) and isinstance(local_geometry.get("representation_summary"), pd.DataFrame):
        df = local_geometry["representation_summary"].copy()
        return {
            "columns": list(df.columns),
            "records": df.where(pd.notna(df), None).to_dict(orient="records"),
        }
    return None


def _finalize(
    rows: dict[str, dict[str, Any]],
    module_warnings: list[str],
    *,
    metadata: dict[str, Any],
    global_diagnostics: dict[str, Any],
    representation_diagnostics: Optional[dict[str, Any]],
    rules: dict[str, Any],
) -> pd.DataFrame:
    records = []
    for row in rows.values():
        for warning in module_warnings:
            row["warnings"].append(warning)
        if row["stability_status"] == "not_computed" and "representation_consensus_label" in row and pd.notna(row["representation_consensus_label"]):
            row["stability_status"] = "assessed"
        row["coverage_status"] = _combined_coverage_status(row)
        _append_reason_codes(row)
        row["summary"] = _summary(row)
        row["warnings"] = "; ".join(dict.fromkeys(row["warnings"]))
        records.append(row)

    out = pd.DataFrame.from_records(records)
    out.attrs["scgeo_report_rules"] = rules
    out.attrs["provenance"] = metadata
    out.attrs["global_diagnostics"] = global_diagnostics
    out.attrs["representation_diagnostics"] = representation_diagnostics
    out.attrs["warnings"] = list(dict.fromkeys(module_warnings))
    return out


def state_report(
    adata,
    *,
    node_key=None,
    robust_shift_key: str = "robust_shift",
    representation_key: str = "representation_stability",
    local_geometry_key: str = "local_geometry_stability",
    local_k=None,
    pair_aggregation: str = "median",
    include_worst_case: bool = True,
    comparison_label: Optional[str] = None,
    condition_key: Optional[str] = None,
    group0: Any = None,
    group1: Any = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Build a canonical state-level ScGeo report from stored analysis results.

    The report reuses existing ``adata.uns['scgeo']`` outputs from robust shift,
    representation stability, and local-geometry stability. It does not
    recompute displacement, bootstrap, rank, neighborhood, or velocity metrics.

    With ``strict=False``, missing modules add warnings and leave unavailable
    columns as missing values. With ``strict=True``, each requested storage
    object must exist.
    """
    pair_aggregation = str(pair_aggregation)
    if pair_aggregation not in {"median", "mean"}:
        raise ValueError("pair_aggregation must be one of {'median', 'mean'}")

    module_warnings: list[str] = []
    robust = _load_result(
        adata,
        robust_shift_key,
        name="robust_shift",
        strict=bool(strict),
        warnings=module_warnings,
    )
    representation = _load_result(
        adata,
        representation_key,
        name="representation_stability",
        strict=bool(strict),
        warnings=module_warnings,
    )
    local_geometry = _load_result(
        adata,
        local_geometry_key,
        name="local_geometry_stability",
        strict=bool(strict),
        warnings=module_warnings,
    )

    local_params = local_geometry.get("params", {}) if isinstance(local_geometry, dict) else {}
    available_k = local_params.get("k_values") if isinstance(local_params, dict) else []
    local_k_values = _resolve_local_k(local_k, available_k)

    states = _states_from_inputs(
        adata,
        node_key=node_key,
        robust=robust,
        representation=representation,
        local_geometry=local_geometry,
    )
    rows = _empty_rows(states)
    _merge_robust_shift(rows, robust, node_key=node_key)
    _merge_representation(rows, representation)
    _apply_inference_metadata(rows, robust, representation)
    _merge_local_geometry(
        rows,
        local_geometry,
        local_k_values=local_k_values,
        pair_aggregation=pair_aggregation,
        include_worst_case=bool(include_worst_case),
    )

    rep_params = representation.get("params") if isinstance(representation, dict) else {}
    rules = {
        "qualitative_labels": (
            "Report summaries do not apply report-local thresholds for large, preserved, or distorted. "
            "Qualitative stability and dynamics labels reuse stored analysis-module rules."
        ),
        "consensus_label_rules": rep_params.get("consensus_label_rules") if isinstance(rep_params, dict) else None,
        "alignment_thresholds": {
            "alignment_pos_thr": rep_params.get("alignment_pos_thr", 0.3) if isinstance(rep_params, dict) else 0.3,
            "alignment_neg_thr": rep_params.get("alignment_neg_thr", -0.3) if isinstance(rep_params, dict) else -0.3,
        },
        "local_pair_aggregation": pair_aggregation,
        "include_worst_case": bool(include_worst_case),
    }
    metadata = _metadata(
        robust=robust,
        representation=representation,
        local_geometry=local_geometry,
        robust_shift_key=robust_shift_key,
        representation_key=representation_key,
        local_geometry_key=local_geometry_key,
        comparison_label=comparison_label,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        local_k_values=local_k_values,
        pair_aggregation=pair_aggregation,
        include_worst_case=bool(include_worst_case),
    )
    return _finalize(
        rows,
        module_warnings,
        metadata=metadata,
        global_diagnostics=_global_diagnostics(local_geometry, local_k_values=local_k_values),
        representation_diagnostics=_representation_diagnostics(local_geometry),
        rules=rules,
    )
