from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._utils import _as_2d_array, _mask_from_obs
from ._robust_shift import _calculate_shift, _cosine, _validate_params


_ALIGNMENT_CLASSES = ("aligned", "discordant", "neutral")
_CONSENSUS_LABEL_RULES: Dict[str, Any] = {
    "min_usable_representations": 2,
    "min_usable_fraction": 0.5,
    "min_samples_per_condition": 2,
    "min_pairwise_spearman": 0.5,
    "max_rank_std": 1.25,
    "max_leave_one_rep_magnitude_relative_deviation": 0.5,
    "max_leave_one_rep_class_switch_fraction": 0.34,
    "class_fraction_threshold": 0.6,
    "neutral_normalized_magnitude_max": 0.25,
    "neutral_delta_norm_median_max": 0.5,
    "label_order": [
        "insufficient_coverage",
        "representation_unstable",
        "stable_neutral",
        "stable_effect",
        "stable_aligned",
        "stable_discordant",
    ],
    "stable_aligned": (
        "coverage passes, coordinate-safe magnitude/rank diagnostics pass, "
        "and aligned velocity classes meet class_fraction_threshold"
    ),
    "stable_discordant": (
        "coverage passes, coordinate-safe magnitude/rank diagnostics pass, "
        "and discordant velocity classes meet class_fraction_threshold"
    ),
    "stable_neutral": (
        "coverage passes and normalized magnitudes are consistently below "
        "neutral_normalized_magnitude_max, or raw median magnitude is below "
        "neutral_delta_norm_median_max with leave-one-representation agreement"
    ),
    "stable_effect": (
        "coverage passes, coordinate-safe magnitude/rank diagnostics pass, "
        "and a non-neutral effect is stable across representations without "
        "velocity evidence"
    ),
    "representation_unstable": (
        "coverage passes for a non-neutral effect, but rank agreement, rank spread, "
        "leave-one-representation-out magnitude deviation, or velocity-class agreement "
        "fails an explicit threshold"
    ),
    "insufficient_coverage": (
        "too few usable representations, too low usable representation fraction, "
        "or no velocity evidence for a non-neutral shifted state when velocity_keys is provided"
    ),
}
_CONSENSUS_NUMERIC_RULE_RANGES: Dict[str, tuple[float | None, float | None, bool]] = {
    "min_usable_representations": (1, None, True),
    "min_usable_fraction": (0.0, 1.0, False),
    "min_samples_per_condition": (1, None, True),
    "min_pairwise_spearman": (-1.0, 1.0, False),
    "max_rank_std": (0.0, None, False),
    "max_leave_one_rep_magnitude_relative_deviation": (0.0, None, False),
    "max_leave_one_rep_class_switch_fraction": (0.0, 1.0, False),
    "class_fraction_threshold": (0.0, 1.0, False),
    "neutral_normalized_magnitude_max": (0.0, None, False),
    "neutral_delta_norm_median_max": (0.0, None, False),
}


def _as_sequence_of_reps(reps: Sequence[str]) -> list[str]:
    if isinstance(reps, str) or not isinstance(reps, Sequence):
        raise TypeError("reps must be a non-empty sequence of adata.obsm keys")
    out = [str(rep) for rep in reps]
    if len(out) == 0:
        raise ValueError("reps must contain at least one representation key")
    if len(set(out)) != len(out):
        raise ValueError("reps contains duplicate representation keys")
    return out


def _resolve_consensus_rules(consensus_rules: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    resolved = dict(_CONSENSUS_LABEL_RULES)
    if consensus_rules is None:
        return resolved
    if not isinstance(consensus_rules, Mapping):
        raise TypeError("consensus_rules must be a mapping or None")

    for key, value in consensus_rules.items():
        if key not in _CONSENSUS_NUMERIC_RULE_RANGES:
            allowed = ", ".join(sorted(_CONSENSUS_NUMERIC_RULE_RANGES))
            raise ValueError(f"Unknown consensus rule {key!r}. Configurable rules are: {allowed}")

        lo, hi, integer = _CONSENSUS_NUMERIC_RULE_RANGES[key]
        if isinstance(value, bool) or not np.isscalar(value):
            raise ValueError(f"Consensus rule {key!r} must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"Consensus rule {key!r} must be finite")
        if lo is not None and numeric < lo:
            raise ValueError(f"Consensus rule {key!r} must be >= {lo}")
        if hi is not None and numeric > hi:
            raise ValueError(f"Consensus rule {key!r} must be <= {hi}")
        if integer:
            if not float(numeric).is_integer():
                raise ValueError(f"Consensus rule {key!r} must be an integer")
            resolved[key] = int(numeric)
        else:
            resolved[key] = numeric
    return resolved


def _as_str_values(values: pd.Series) -> np.ndarray:
    return values.astype(str).to_numpy()


def _unique_in_order(values: np.ndarray) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _count_samples(sample_values: Optional[np.ndarray], mask: np.ndarray) -> Optional[int]:
    if sample_values is None:
        return None
    return len(_unique_in_order(sample_values[mask]))


def _quantile_iqr(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    q25, q75 = np.nanpercentile(values, [25.0, 75.0])
    return float(q25), float(q75), float(q75 - q25)


def _alignment_class_from_cosine(
    value: float,
    *,
    pos_thr: float,
    neg_thr: float,
) -> str:
    if not np.isfinite(value):
        return "missing"
    if value >= pos_thr:
        return "aligned"
    if value <= neg_thr:
        return "discordant"
    return "neutral"


def _class_entropy(classes: Sequence[str]) -> float:
    valid = [cls for cls in classes if cls in _ALIGNMENT_CLASSES]
    if not valid:
        return np.nan
    counts = np.asarray([valid.count(cls) for cls in _ALIGNMENT_CLASSES], dtype=float)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _majority_class(classes: Sequence[str]) -> str:
    valid = [cls for cls in classes if cls in _ALIGNMENT_CLASSES]
    if not valid:
        return "missing"
    counts = {cls: valid.count(cls) for cls in _ALIGNMENT_CLASSES}
    max_count = max(counts.values())
    winners = [cls for cls, count in counts.items() if count == max_count]
    if len(winners) != 1:
        return "mixed"
    return winners[0]


def _velocity_matrix(
    adata,
    *,
    rep: str,
    reps_dim: int,
    velocity_keys: Optional[Mapping[str, Optional[str]]],
    warnings: list[str],
) -> tuple[Optional[np.ndarray], str, Optional[str]]:
    if velocity_keys is None:
        return None, "not_provided", None
    vel_key = velocity_keys.get(rep)
    if vel_key is None:
        return None, "not_requested", None
    vel_key = str(vel_key)
    if vel_key not in adata.obsm:
        warnings.append(f"Velocity key {vel_key!r} for representation {rep!r} was not found.")
        return None, "missing_key", vel_key
    V = _as_2d_array(adata.obsm[vel_key]).astype(np.float64, copy=False)
    if V.shape[0] != adata.n_obs or V.shape[1] != reps_dim:
        warnings.append(
            f"Velocity key {vel_key!r} for representation {rep!r} has shape {V.shape}; "
            f"expected ({adata.n_obs}, {reps_dim})."
        )
        return None, "dimension_mismatch", vel_key
    return V, "available", vel_key


def _status_from_counts(
    *,
    n0: int,
    n1: int,
    n_samples0: Optional[int],
    n_samples1: Optional[int],
    min_cells: int,
    sample_key: Optional[str],
    min_samples_per_condition: int,
) -> str:
    if n0 == 0 or n1 == 0:
        return "missing_condition"
    if n0 < min_cells or n1 < min_cells:
        return "underpowered_cells"
    if sample_key is not None:
        if (n_samples0 or 0) < min_samples_per_condition or (n_samples1 or 0) < min_samples_per_condition:
            return "underpowered_samples"
    return "usable"


def _add_rep_ranks(per_rep_state: pd.DataFrame) -> pd.DataFrame:
    out = per_rep_state.copy()
    out["magnitude_rank"] = np.nan
    for rep, idx in out.groupby("rep", sort=False).groups.items():
        rep_mask = out.index.isin(idx) & out["usable"] & np.isfinite(out["normalized_delta_norm"])
        if not np.any(rep_mask):
            continue
        out.loc[rep_mask, "magnitude_rank"] = out.loc[rep_mask, "normalized_delta_norm"].rank(
            ascending=False,
            method="average",
        )
    return out


def _add_leave_one_rep_diagnostics(per_rep_state: pd.DataFrame) -> pd.DataFrame:
    out = per_rep_state.copy()
    out["loo_magnitude_median_without_rep"] = np.nan
    out["loo_magnitude_abs_deviation"] = np.nan
    out["loo_class_consensus_without_rep"] = "missing"
    out["loo_class_changes_consensus"] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)

    for _, df_node in out.groupby("node", sort=False):
        usable = df_node[df_node["usable"]]
        for idx, row in usable.iterrows():
            others = usable[usable["rep"] != row["rep"]]
            other_mags = others["normalized_delta_norm"].to_numpy(dtype=float)
            other_mags = other_mags[np.isfinite(other_mags)]
            if other_mags.size:
                med = float(np.nanmedian(other_mags))
                out.at[idx, "loo_magnitude_median_without_rep"] = med
                mag = float(row["normalized_delta_norm"])
                out.at[idx, "loo_magnitude_abs_deviation"] = abs(mag - med) if np.isfinite(mag) else np.nan

            other_class = _majority_class(others["alignment_class"].tolist())
            out.at[idx, "loo_class_consensus_without_rep"] = other_class
            cls = row["alignment_class"]
            if cls in _ALIGNMENT_CLASSES and other_class in _ALIGNMENT_CLASSES:
                out.at[idx, "loo_class_changes_consensus"] = bool(cls != other_class)
    return out


def _rank_correlation(per_rep_state: pd.DataFrame, reps: list[str]) -> pd.DataFrame:
    usable = per_rep_state[per_rep_state["usable"]].copy()
    pivot = usable.pivot(index="node", columns="rep", values="normalized_delta_norm")
    rows = []
    for rep_a, rep_b in combinations(reps, 2):
        if rep_a not in pivot or rep_b not in pivot:
            n_states = 0
            rho = np.nan
        else:
            pair = pivot[[rep_a, rep_b]].dropna()
            n_states = int(pair.shape[0])
            rho = float(pair[rep_a].corr(pair[rep_b], method="spearman")) if n_states >= 2 else np.nan
        rows.append({"rep_a": rep_a, "rep_b": rep_b, "spearman_r": rho, "n_states": n_states})
    return pd.DataFrame(rows)


def _class_agreement(per_rep_state: pd.DataFrame, reps: list[str]) -> pd.DataFrame:
    rows = []
    for node, df_node in per_rep_state.groupby("node", sort=False):
        by_rep = df_node.set_index("rep")
        for rep_a, rep_b in combinations(reps, 2):
            cls_a = by_rep.at[rep_a, "alignment_class"] if rep_a in by_rep.index else "missing"
            cls_b = by_rep.at[rep_b, "alignment_class"] if rep_b in by_rep.index else "missing"
            available = cls_a in _ALIGNMENT_CLASSES and cls_b in _ALIGNMENT_CLASSES
            rows.append(
                {
                    "node": node,
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "class_a": cls_a,
                    "class_b": cls_b,
                    "both_available": bool(available),
                    "agreement": bool(cls_a == cls_b) if available else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _label_consensus(
    *,
    n_usable: int,
    usable_fraction: float,
    n_velocity_available: int,
    normalized_magnitude_median: float,
    normalized_magnitude_max: float,
    delta_norm_median: float,
    magnitude_rank_std: float,
    pairwise_spearman_median: float,
    loo_rep_magnitude_max_relative_deviation: float,
    loo_class_switch_fraction: float,
    aligned_fraction: float,
    discordant_fraction: float,
    neutral_fraction: float,
    velocity_requested: bool,
    rules: Mapping[str, Any],
) -> str:
    if n_usable < rules["min_usable_representations"] or usable_fraction < rules["min_usable_fraction"]:
        return "insufficient_coverage"

    neutral_by_normalized_magnitude = (
        np.isfinite(normalized_magnitude_median)
        and normalized_magnitude_median <= rules["neutral_normalized_magnitude_max"]
        and np.isfinite(normalized_magnitude_max)
        and normalized_magnitude_max <= rules["neutral_normalized_magnitude_max"]
    )
    neutral_by_raw_magnitude = (
        np.isfinite(delta_norm_median)
        and delta_norm_median <= rules["neutral_delta_norm_median_max"]
        and (
            not np.isfinite(loo_rep_magnitude_max_relative_deviation)
            or loo_rep_magnitude_max_relative_deviation
            <= rules["max_leave_one_rep_magnitude_relative_deviation"]
        )
    )
    neutral_by_magnitude = neutral_by_normalized_magnitude or neutral_by_raw_magnitude

    if neutral_by_magnitude:
        return "stable_neutral"

    if (
        np.isfinite(magnitude_rank_std)
        and magnitude_rank_std > rules["max_rank_std"]
    ):
        return "representation_unstable"
    if (
        np.isfinite(pairwise_spearman_median)
        and pairwise_spearman_median < rules["min_pairwise_spearman"]
    ):
        return "representation_unstable"
    if (
        np.isfinite(loo_rep_magnitude_max_relative_deviation)
        and loo_rep_magnitude_max_relative_deviation
        > rules["max_leave_one_rep_magnitude_relative_deviation"]
    ):
        return "representation_unstable"
    if (
        np.isfinite(loo_class_switch_fraction)
        and loo_class_switch_fraction > rules["max_leave_one_rep_class_switch_fraction"]
    ):
        return "representation_unstable"

    if velocity_requested and n_velocity_available == 0:
        return "insufficient_coverage"

    if not velocity_requested:
        return "stable_effect"

    threshold = rules["class_fraction_threshold"]
    if np.isfinite(aligned_fraction) and aligned_fraction >= threshold:
        return "stable_aligned"
    if np.isfinite(discordant_fraction) and discordant_fraction >= threshold:
        return "stable_discordant"
    if np.isfinite(neutral_fraction) and neutral_fraction >= threshold:
        return "stable_neutral"
    return "representation_unstable"


def _consensus_state(
    per_rep_state: pd.DataFrame,
    rank_correlation: pd.DataFrame,
    reps: list[str],
    nodes: list[str],
    *,
    velocity_requested: bool,
    rules: Mapping[str, Any],
) -> pd.DataFrame:
    rank_pairs = {
        frozenset((row.rep_a, row.rep_b)): row.spearman_r
        for row in rank_correlation.itertuples(index=False)
    }
    rows = []
    for node in nodes:
        df_node = per_rep_state[per_rep_state["node"] == node]
        usable = df_node[df_node["usable"]].copy()
        n_usable = int(usable.shape[0])
        usable_fraction = float(n_usable / len(reps)) if reps else np.nan

        mags = usable["normalized_delta_norm"].to_numpy(dtype=float)
        mags = mags[np.isfinite(mags)]
        mag_median = float(np.nanmedian(mags)) if mags.size else np.nan
        mag_max = float(np.nanmax(mags)) if mags.size else np.nan
        mag_q25, mag_q75, mag_iqr = _quantile_iqr(mags)
        delta_norms = usable["delta_norm"].to_numpy(dtype=float)
        delta_norms = delta_norms[np.isfinite(delta_norms)]
        delta_norm_median = float(np.nanmedian(delta_norms)) if delta_norms.size else np.nan

        ranks = usable["magnitude_rank"].to_numpy(dtype=float)
        ranks = ranks[np.isfinite(ranks)]
        rank_mean = float(np.nanmean(ranks)) if ranks.size else np.nan
        rank_std = float(np.nanstd(ranks)) if ranks.size else np.nan

        pair_values = []
        usable_reps = usable["rep"].tolist()
        for rep_a, rep_b in combinations(usable_reps, 2):
            value = rank_pairs.get(frozenset((rep_a, rep_b)), np.nan)
            if np.isfinite(value):
                pair_values.append(float(value))
        pairwise_spearman_median = float(np.nanmedian(pair_values)) if pair_values else np.nan

        classes = [cls for cls in usable["alignment_class"].tolist() if cls in _ALIGNMENT_CLASSES]
        n_velocity_available = len(classes)
        if n_velocity_available:
            aligned_fraction = classes.count("aligned") / n_velocity_available
            discordant_fraction = classes.count("discordant") / n_velocity_available
            neutral_fraction = classes.count("neutral") / n_velocity_available
        else:
            aligned_fraction = np.nan
            discordant_fraction = np.nan
            neutral_fraction = np.nan

        cosines = usable["alignment_cosine"].to_numpy(dtype=float)
        cosines = cosines[np.isfinite(cosines)]
        cosine_median = float(np.nanmedian(cosines)) if cosines.size else np.nan
        cosine_q25, cosine_q75, cosine_iqr = _quantile_iqr(cosines)
        entropy = _class_entropy(classes)

        loo_abs = usable["loo_magnitude_abs_deviation"].to_numpy(dtype=float)
        loo_abs = loo_abs[np.isfinite(loo_abs)]
        loo_max_abs = float(np.nanmax(loo_abs)) if loo_abs.size else np.nan
        loo_relative = (
            float(loo_max_abs / max(abs(mag_median), rules["neutral_normalized_magnitude_max"]))
            if np.isfinite(loo_max_abs) and np.isfinite(mag_median)
            else np.nan
        )

        changes = usable["loo_class_changes_consensus"].dropna()
        loo_class_switch_fraction = float(np.mean(changes.astype(bool))) if changes.size else np.nan

        label = _label_consensus(
            n_usable=n_usable,
            usable_fraction=usable_fraction,
            n_velocity_available=n_velocity_available,
            normalized_magnitude_median=mag_median,
            normalized_magnitude_max=mag_max,
            delta_norm_median=delta_norm_median,
            magnitude_rank_std=rank_std,
            pairwise_spearman_median=pairwise_spearman_median,
            loo_rep_magnitude_max_relative_deviation=loo_relative,
            loo_class_switch_fraction=loo_class_switch_fraction,
            aligned_fraction=aligned_fraction,
            discordant_fraction=discordant_fraction,
            neutral_fraction=neutral_fraction,
            velocity_requested=velocity_requested,
            rules=rules,
        )

        rows.append(
            {
                "node": node,
                "n_representations": len(reps),
                "n_usable_representations": n_usable,
                "usable_fraction": usable_fraction,
                "normalized_magnitude_median": mag_median,
                "normalized_magnitude_max": mag_max,
                "delta_norm_median": delta_norm_median,
                "normalized_magnitude_q25": mag_q25,
                "normalized_magnitude_q75": mag_q75,
                "normalized_magnitude_iqr": mag_iqr,
                "magnitude_rank_mean": rank_mean,
                "magnitude_rank_std": rank_std,
                "pairwise_spearman_median": pairwise_spearman_median,
                "aligned_fraction": aligned_fraction,
                "discordant_fraction": discordant_fraction,
                "neutral_fraction": neutral_fraction,
                "n_velocity_available": n_velocity_available,
                "alignment_cosine_median": cosine_median,
                "alignment_cosine_q25": cosine_q25,
                "alignment_cosine_q75": cosine_q75,
                "alignment_cosine_iqr": cosine_iqr,
                "alignment_class_entropy": entropy,
                "loo_rep_magnitude_max_abs_deviation": loo_max_abs,
                "loo_rep_magnitude_max_relative_deviation": loo_relative,
                "loo_class_switch_fraction": loo_class_switch_fraction,
                "consensus_label": label,
            }
        )
    return pd.DataFrame(rows)


def representation_stability(
    adata,
    *,
    reps,
    node_key,
    condition_key,
    group0,
    group1,
    sample_key=None,
    center: str = "geometric_median",
    trim_fraction: float = 0.1,
    n_boot: int = 500,
    velocity_keys=None,
    alignment_pos_thr: float = 0.3,
    alignment_neg_thr: float = -0.3,
    min_cells: int = 20,
    consensus_rules=None,
    seed: int = 0,
    store_key: str = "representation_stability",
):
    """
    Assess whether state-level perturbation geometry is stable across representations.

    Consensus labels use explicit rules stored in ``params['consensus_label_rules']``.
    The function never averages raw displacement vectors across representations; it
    combines only coordinate-safe scalars: normalized magnitudes, ranks, velocity
    cosines computed within each representation, and class agreement. Non-neutral
    magnitude agreement without velocity evidence is labeled ``stable_effect``;
    rank and direction instability are reserved for substantive representation
    disagreement after coverage and stable-neutral checks.
    """
    reps = _as_sequence_of_reps(reps)
    missing = [rep for rep in reps if rep not in adata.obsm]
    if missing:
        raise KeyError(f"Representation key(s) not found in adata.obsm: {missing}")
    if node_key not in adata.obs:
        raise KeyError(f"obs key '{node_key}' not found")
    if condition_key not in adata.obs:
        raise KeyError(f"obs key '{condition_key}' not found")
    if sample_key is not None and sample_key not in adata.obs:
        raise KeyError(f"obs key '{sample_key}' not found")
    if int(min_cells) < 1:
        raise ValueError("min_cells must be >= 1")
    if alignment_neg_thr >= alignment_pos_thr:
        raise ValueError("alignment_neg_thr must be less than alignment_pos_thr")
    if velocity_keys is not None and not isinstance(velocity_keys, Mapping):
        raise TypeError("velocity_keys must be a mapping from representation key to velocity key, or None")

    center = str(center)
    resolved_bootstrap_unit = _validate_params(
        center,
        float(trim_fraction),
        int(n_boot),
        "auto",
        "pooled_robust_scale",
        sample_key,
    )
    rules = _resolve_consensus_rules(consensus_rules)
    min_samples_per_condition = int(rules["min_samples_per_condition"])

    node_values = _as_str_values(adata.obs[node_key])
    nodes = _unique_in_order(node_values)
    m0_all = _mask_from_obs(adata, condition_key, group0)
    m1_all = _mask_from_obs(adata, condition_key, group1)
    if int(m0_all.sum()) == 0 or int(m1_all.sum()) == 0:
        raise ValueError(
            f"Groups must be non-empty: group0={group0!r} (n={int(m0_all.sum())}), "
            f"group1={group1!r} (n={int(m1_all.sum())})"
        )
    sample_values = _as_str_values(adata.obs[sample_key]) if sample_key is not None else None
    rng = np.random.RandomState(seed)
    warnings: list[str] = []
    rows: list[Dict[str, Any]] = []

    for rep in reps:
        X = _as_2d_array(adata.obsm[rep]).astype(np.float64, copy=False)
        velocity_matrix, velocity_status, velocity_key = _velocity_matrix(
            adata,
            rep=rep,
            reps_dim=X.shape[1],
            velocity_keys=velocity_keys,
            warnings=warnings,
        )
        for node in nodes:
            node_mask = node_values == node
            mask0 = node_mask & m0_all
            mask1 = node_mask & m1_all
            n0 = int(mask0.sum())
            n1 = int(mask1.sum())
            n_samples0 = _count_samples(sample_values, mask0)
            n_samples1 = _count_samples(sample_values, mask1)
            status = _status_from_counts(
                n0=n0,
                n1=n1,
                n_samples0=n_samples0,
                n_samples1=n_samples1,
                min_cells=int(min_cells),
                sample_key=sample_key,
                min_samples_per_condition=min_samples_per_condition,
            )
            shift_out: Dict[str, Any]
            error_message = ""
            try:
                shift_out = _calculate_shift(
                    X,
                    mask0,
                    mask1,
                    sample_values=sample_values,
                    center=center,
                    trim_fraction=float(trim_fraction),
                    n_boot=int(n_boot),
                    bootstrap_unit=resolved_bootstrap_unit,
                    normalize_by="pooled_robust_scale",
                    rng=rng,
                )
            except Exception as exc:
                shift_out = {}
                status = "error"
                error_message = f"{type(exc).__name__}: {exc}"

            normalized_delta_norm = float(shift_out.get("normalized_delta_norm", np.nan))
            delta_norm = float(shift_out.get("delta_norm", np.nan))
            ci = shift_out.get("bootstrap_magnitude_ci95", [np.nan, np.nan])
            if ci is None or len(ci) != 2:
                ci = [np.nan, np.nan]
            if status == "usable" and not np.isfinite(normalized_delta_norm):
                status = "nonfinite_shift"
            usable = status == "usable"

            delta = shift_out.get("delta", None)
            alignment_cosine = np.nan
            alignment_class = "missing"
            velocity_n_cells = int(node_mask.sum())
            if usable and velocity_matrix is not None and delta is not None:
                velocity_mask = node_mask & (m0_all | m1_all)
                if int(velocity_mask.sum()) > 0:
                    velocity_vector = velocity_matrix[velocity_mask].mean(axis=0)
                    alignment_cosine = _cosine(np.asarray(delta, dtype=float), velocity_vector)
                    alignment_class = _alignment_class_from_cosine(
                        alignment_cosine,
                        pos_thr=float(alignment_pos_thr),
                        neg_thr=float(alignment_neg_thr),
                    )

            sensitivity = shift_out.get("outlier_sensitivity") or {}
            sensitivity_out = {
                "delta_difference_norm": sensitivity.get("delta_difference_norm", np.nan),
                "relative_norm_change": sensitivity.get("relative_norm_change", np.nan),
                "cosine_to_mean": sensitivity.get("cosine_to_mean", np.nan),
            }
            rows.append(
                {
                    "rep": rep,
                    "node": node,
                    "status": status,
                    "usable": bool(usable),
                    "error_message": error_message,
                    "n_cells0": n0,
                    "n_cells1": n1,
                    "n_samples0": n_samples0,
                    "n_samples1": n_samples1,
                    "delta_norm": delta_norm,
                    "normalized_delta_norm": normalized_delta_norm,
                    "magnitude_ci95_low": float(ci[0]),
                    "magnitude_ci95_high": float(ci[1]),
                    "directional_resultant_length": shift_out.get(
                        "bootstrap_directional_resultant_length",
                        np.nan,
                    ),
                    "outlier_sensitivity": sensitivity_out,
                    "outlier_delta_difference_norm": sensitivity_out["delta_difference_norm"],
                    "outlier_relative_norm_change": sensitivity_out["relative_norm_change"],
                    "outlier_cosine_to_mean": sensitivity_out["cosine_to_mean"],
                    "velocity_key": velocity_key,
                    "velocity_status": velocity_status,
                    "velocity_n_cells": velocity_n_cells,
                    "alignment_cosine": alignment_cosine,
                    "alignment_class": alignment_class,
                }
            )

    per_rep_state = pd.DataFrame(rows)
    per_rep_state = _add_rep_ranks(per_rep_state)
    per_rep_state = _add_leave_one_rep_diagnostics(per_rep_state)
    rank_correlation = _rank_correlation(per_rep_state, reps)
    class_agreement = _class_agreement(per_rep_state, reps)
    consensus_state = _consensus_state(
        per_rep_state,
        rank_correlation,
        reps,
        nodes,
        velocity_requested=velocity_keys is not None,
        rules=rules,
    )

    status_counts = per_rep_state["status"].value_counts().to_dict()
    label_counts = consensus_state["consensus_label"].value_counts().to_dict()
    coverage_summary = {
        "n_representations": len(reps),
        "n_states": len(nodes),
        "n_per_rep_state_rows": int(per_rep_state.shape[0]),
        "n_usable_rows": int(per_rep_state["usable"].sum()),
        "usable_row_fraction": float(per_rep_state["usable"].mean()) if per_rep_state.shape[0] else np.nan,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "consensus_label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "n_missing_velocity_representations": int(
            per_rep_state.drop_duplicates("rep")["velocity_status"].isin(
                ["not_provided", "not_requested", "missing_key", "dimension_mismatch"]
            ).sum()
        ),
    }
    if any(per_rep_state["status"].isin(["underpowered_cells", "underpowered_samples", "missing_condition"])):
        warnings.append("Some representation-state combinations were not usable because of coverage.")

    out = {
        "params": {
            "reps": reps,
            "node_key": node_key,
            "condition_key": condition_key,
            "group0": group0,
            "group1": group1,
            "sample_key": sample_key,
            "center": center,
            "trim_fraction": float(trim_fraction),
            "n_boot": int(n_boot),
            "resolved_bootstrap_unit": resolved_bootstrap_unit,
            "normalize_by": "pooled_robust_scale",
            "velocity_keys": None if velocity_keys is None else dict(velocity_keys),
            "alignment_pos_thr": float(alignment_pos_thr),
            "alignment_neg_thr": float(alignment_neg_thr),
            "min_cells": int(min_cells),
            "consensus_rules": None if consensus_rules is None else dict(consensus_rules),
            "seed": int(seed),
            "store_key": store_key,
            "consensus_label_rules": rules,
        },
        "per_rep_state": per_rep_state,
        "consensus_state": consensus_state,
        "rank_correlation": rank_correlation,
        "class_agreement": class_agreement,
        "warnings": warnings,
        "coverage_summary": coverage_summary,
    }

    adata.uns.setdefault("scgeo", {})
    adata.uns["scgeo"][store_key] = out
    return out
