"""Validated descriptive context and stored evidence for compass plots.

No estimator is fitted here. In particular, display coverage never replaces
the eligibility or effect values recorded in a supplied scientific result.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd

from ._paga_pies import sample_abundance_pies
from ._paga_shift_map import _string_labels
from ._table_plot_utils import numeric_column


def scope_mask(cells, scope, group0, group1):
    if scope not in {"contrast", "all", "group0", "group1"}:
        raise ValueError("Scope must be contrast, all, group0, or group1.")
    if scope == "all":
        return np.ones(len(cells), dtype=bool)
    if scope == "contrast":
        return cells.condition.isin([group0, group1]).to_numpy()
    return cells.condition.eq(group0 if scope == "group0" else group1).to_numpy()


def context_nodes(cells, names, node_key, groups, min_cells):
    records = []
    totals = {
        group: int((cells.condition.eq(group) & cells.state.notna()).sum())
        for group in groups
    }
    for state in names:
        row = {node_key: state, "state": state}
        for i, group in enumerate(groups):
            frame = cells.loc[cells.state.eq(state) & cells.condition.eq(group)]
            count = len(frame)
            center = (
                frame[["x", "y"]].to_numpy().mean(axis=0) if count else [np.nan, np.nan]
            )
            row.update(
                {
                    f"n{i}": count,
                    f"x{i}": center[0],
                    f"y{i}": center[1],
                    f"present{i}": count >= min_cells,
                    f"n{i}_comp": count,
                    f"frac{i}": count / totals[group] if totals[group] else np.nan,
                }
            )
        row["dx"], row["dy"] = row["x1"] - row["x0"], row["y1"] - row["y0"]
        row["shift_norm"] = float(np.hypot(row["dx"], row["dy"]))
        row["delta_frac"] = row["frac1"] - row["frac0"]
        row["delta_abs"] = abs(row["delta_frac"])
        row["usable_shift"] = bool(row["present0"] and row["present1"])
        row["coverage_status"] = (
            "absent_from_contrast"
            if not row["n0"] and not row["n1"]
            else "comparison_only"
            if not row["n0"]
            else "reference_only"
            if not row["n1"]
            else "insufficient_cells"
            if not row["usable_shift"]
            else "usable"
        )
        records.append(row)
    return pd.DataFrame(records)


def velocity_nodes(
    adata, cells, names, embedding_key, velocity_basis, groups, scope, min_cells
):
    expected = f"velocity_{embedding_key.removeprefix('X_')}"
    key = (
        expected
        if velocity_basis is None
        else f"velocity_{str(velocity_basis).removeprefix('velocity_').removeprefix('X_')}"
    )
    if key != expected:
        raise ValueError(
            f"Projected velocity must match the display embedding: expected {expected!r}, got {key!r}."
        )
    selected = scope_mask(cells, scope, *groups) & cells.state.notna().to_numpy()
    available = key in adata.obsm
    if not available and velocity_basis is not None:
        raise KeyError(
            f"Explicit projected velocity {key!r} is absent from adata.obsm."
        )
    if available:
        velocity = np.asarray(adata.obsm[key], dtype=float)
        if velocity.shape != np.asarray(adata.obsm[embedding_key]).shape:
            raise ValueError(
                "Projected velocity shape must match the display embedding shape."
            )
        finite = np.isfinite(velocity[:, :2]).all(axis=1)
        cells["vx"], cells["vy"] = velocity[:, 0], velocity[:, 1]
    else:
        finite = np.zeros(len(cells), dtype=bool)
        cells["vx"], cells["vy"] = np.nan, np.nan
    cells["velocity_in_scope"] = selected
    cells["finite_velocity"] = finite
    cells["velocity_included"] = selected & finite
    records = []
    for state in names:
        mask = cells.state.eq(state).to_numpy() & selected
        used = mask & finite
        mean = (
            cells.loc[used, ["vx", "vy"]].to_numpy().mean(axis=0)
            if used.any()
            else [np.nan, np.nan]
        )
        records.append(
            {
                "state": state,
                "vx": mean[0],
                "vy": mean[1],
                "n_vel": int(used.sum()),
                "n_vel_observed": int(mask.sum()),
                "n_vel_missing": int((mask & ~finite).sum()),
                "velocity_present": int(used.sum()) >= min_cells,
                "vel_norm": float(np.hypot(*mean)),
            }
        )
    return pd.DataFrame(records), key, available


def ood_nodes(adata, cells, names, ood_key, threshold, groups, scope):
    selected = scope_mask(cells, scope, *groups) & cells.state.notna().to_numpy()
    if threshold is not None and (
        isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold)
    ):
        raise ValueError("ood_threshold must be a finite number or None.")
    if ood_key is None:
        if threshold is not None:
            raise ValueError("ood_threshold requires ood_key.")
        scores = np.full(len(cells), np.nan)
    else:
        if ood_key not in adata.obs:
            raise KeyError(f"OOD score {ood_key!r} is absent from adata.obs.")
        scores = pd.to_numeric(adata.obs[ood_key], errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
    finite = np.isfinite(scores)
    used = finite & selected
    resolved = (
        float(threshold)
        if threshold is not None
        else float(np.quantile(scores[used], 0.95))
        if used.any()
        else np.nan
    )
    flagged = used & (scores >= resolved)
    cells["ood_score"] = scores
    cells["ood_in_scope"] = selected
    cells["ood_scored"] = used
    cells["ood_flagged"] = pd.array(
        np.where(used, flagged.astype(float), np.nan), dtype="Float64"
    )
    records = []
    for state in names:
        mask = selected & cells.state.eq(state).to_numpy()
        measured = mask & finite
        n = int(measured.sum())
        records.append(
            {
                "state": state,
                "ood_mean": float(scores[measured].mean()) if n else np.nan,
                "ood_median": float(np.median(scores[measured])) if n else np.nan,
                "ood_flagged_n": int(flagged[mask].sum()) if n else np.nan,
                "ood_frac": float(flagged[mask].sum() / n) if n else np.nan,
                "ood_n": int(mask.sum()),
                "ood_n_scored": n,
                "ood_n_missing": int(mask.sum()) - n,
                "ood_score_coverage": n / int(mask.sum()) if mask.any() else np.nan,
                "ood_threshold_used": resolved,
            }
        )
    return pd.DataFrame(records), resolved


def sample_context(adata, cells, states, groups, sample_key):
    """Full contrast roster, including observed samples with zero state counts."""
    sample_counts = pd.DataFrame(
        columns=[
            "state",
            "condition",
            "sample",
            "count",
            "denominator",
            "n_cells",
            "n_missing_state",
            "state_fraction",
            "included",
            "status",
        ]
    )
    if sample_key is not None:
        if sample_key not in adata.obs:
            raise KeyError(f"sample_key {sample_key!r} is absent from adata.obs.")
        labels = _string_labels(adata.obs[sample_key], sample_key).reset_index(
            drop=True
        )
        cells["sample"] = labels
        raw = []
        for state in states:
            frame = cells.loc[cells.in_contrast & cells.state.eq(state)]
            for group in groups:
                count = int(frame.condition.eq(group).sum())
                raw.append(
                    {
                        "state": state,
                        "category": group,
                        "category_label": group,
                        "is_missing": False,
                        "count": count,
                        "denominator": len(frame),
                        "fraction": count / len(frame) if len(frame) else 0.0,
                    }
                )
        _, _, sample_counts = sample_abundance_pies(
            cells, labels, states, list(groups), groups, pd.DataFrame(raw)
        )
    coverage = []
    for state in states:
        for group in groups:
            mask = cells.state.eq(state) & cells.condition.eq(group)
            samples = sample_counts.loc[
                sample_counts.state.eq(state) & sample_counts.condition.eq(group)
            ]
            n = int(mask.sum())
            coverage.append(
                {
                    "state": state,
                    "condition": group,
                    "n_cells": n,
                    "n_samples": int(samples["count"].gt(0).sum())
                    if sample_key is not None
                    else np.nan,
                    "n_samples_total": len(samples)
                    if sample_key is not None
                    else np.nan,
                    "n_sample_denominators": int(samples.included.sum())
                    if sample_key is not None
                    else np.nan,
                    "n_velocity": int(cells.loc[mask, "velocity_included"].sum()),
                    "n_ood_scored": int(cells.loc[mask, "ood_scored"].sum()),
                    "status": "no_observed_cells"
                    if not n
                    else "single_observed_cell"
                    if n == 1
                    else "observed",
                }
            )
    return sample_counts, pd.DataFrame(coverage)


def _state_table(frame, explicit, node_key, title):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{title} must be a nonempty DataFrame.")
    out = frame.copy(deep=True)
    out.attrs = deepcopy(frame.attrs)
    if explicit is not None:
        candidates = [explicit] if explicit in out else []
    else:
        candidates = list(
            dict.fromkeys(c for c in ("state", "node", node_key, "leiden") if c in out)
        )
        if not candidates and out.index.name in {"state", "node", node_key, "leiden"}:
            out = out.reset_index()
            candidates = [out.columns[0]]
    if not candidates:
        raise ValueError(
            f"{title} requires an unambiguous state column; supply its state_key."
        )
    key = candidates[0]
    if any(not out[key].astype(str).equals(out[c].astype(str)) for c in candidates[1:]):
        raise ValueError(
            f"{title} has ambiguous state columns; supply its state_key explicitly."
        )
    if out[key].isna().any():
        raise ValueError(f"{title} state labels cannot be missing.")
    states = out[key].astype(str)
    if states.duplicated().any():
        raise ValueError(f"{title} contains duplicate or string-colliding states.")
    out["_state_id"] = states
    return out


def _source_eligibility(frame, status_keys, *, allow_descriptive=False):
    valid = pd.Series(True, index=frame.index)
    for key in ("usable", "eligible"):
        if key in frame:
            if not frame[key].dropna().isin([True, False, 0, 1]).all():
                raise ValueError(f"Stored {key} values must be boolean or missing.")
            valid &= frame[key].fillna(False).astype(bool)
    status_key = next((key for key in status_keys if key in frame), None)
    status = (
        frame[status_key].astype(str)
        if status_key
        else pd.Series("not_recorded", index=frame.index)
    )
    excluded = {
        "insufficient_coverage",
        "insufficient_cells",
        "insufficient_samples",
        "unavailable",
        "not_computed",
        "not_assessed",
        "numerical_degeneracy",
        "missing",
        "False",
    }
    if allow_descriptive:
        excluded.discard("insufficient_samples")
    valid &= ~status.isin(excluded)
    if status_key:
        valid &= frame[status_key].notna()
    return valid, status


def _descriptive_inference(frame):
    """Recognize recorded descriptive sampling, without inferring missing design."""
    levels = {
        "sample_descriptive",
        "cell_descriptive",
        "descriptive",
        "not_computed",
        "cell_bootstrap",
        "cell_level",
        "pooled_cells",
    }
    known = np.zeros(len(frame), dtype=bool)
    if "descriptive_only" in frame:
        known |= frame.descriptive_only.eq(True).fillna(False).to_numpy(bool)
    if "inference_level" in frame:
        known |= frame.inference_level.isin(levels).to_numpy()
    if "effect_status" in frame:
        known |= frame.effect_status.eq("insufficient_samples").to_numpy()
    for key in ("bootstrap_unit", "resolved_bootstrap_unit"):
        if key in frame:
            known |= frame[key].isin({"cell", "cells", "cell_bootstrap"}).to_numpy()
    seen = set()

    def visit(mapping):
        if id(mapping) in seen:
            return
        seen.add(id(mapping))
        inference = mapping.get("inference_level")
        descriptive = mapping.get("descriptive_only")
        if (isinstance(inference, str) and inference in levels) or descriptive is True:
            known[:] = True
        for key in ("bootstrap_unit", "resolved_bootstrap_unit"):
            value = mapping.get(key)
            if isinstance(value, str) and value in {"cell", "cells", "cell_bootstrap"}:
                known[:] = True
        for key in (
            "params",
            "metadata",
            "provenance",
            "source_params",
            "analysis",
            "analysis_metadata",
        ):
            child = mapping.get(key)
            if isinstance(child, Mapping):
                visit(child)

    visit(frame.attrs)
    return known


def stored_evidence(
    shift_df,
    alignment_df,
    *,
    node_key,
    condition_key,
    groups,
    analysis_rep,
    report_state_key,
    alignment_state_key,
    effect_metric,
    min_cells,
):
    from ._stored_evidence import validate_stored_evidence

    if not isinstance(analysis_rep, str) or not analysis_rep:
        raise ValueError("mode='stored' requires analysis_rep.")
    provenance, warnings = validate_stored_evidence(
        shift_df,
        alignment_df,
        node_key=node_key,
        condition_key=condition_key,
        group0=groups[0],
        group1=groups[1],
        analysis_rep=analysis_rep,
        strict=True,
    )
    source = _state_table(shift_df, report_state_key, node_key, "shift_df")
    metric = effect_metric
    if metric == "auto":
        available = [m for m in ("normalized_delta_norm", "delta_norm") if m in source]
        metric = next(
            (m for m in available if np.isfinite(numeric_column(source, m)).any()),
            available[0] if available else None,
        )
    if metric not in {"delta_norm", "normalized_delta_norm"}:
        raise ValueError(
            "effect_metric must be auto, delta_norm, or normalized_delta_norm."
        )
    effect = numeric_column(source, metric)
    if (effect < 0).any():
        raise ValueError("Stored displacement magnitudes cannot be negative.")
    valid, status = _source_eligibility(
        source, ("effect_status", "coverage_status", "status"), allow_descriptive=True
    )
    evidence = pd.DataFrame(
        {
            "state": source._state_id.to_numpy(),
            "effect": effect,
            "effect_source_eligible": valid.to_numpy(),
            "effect_available": valid.to_numpy() & np.isfinite(effect),
            "effect_source_status": status.to_numpy(),
            "effect_metric": metric,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "interval_available": False,
            "interval_reason": "not_supplied",
            "alignment": np.nan,
            "alignment_source_eligible": False,
            "alignment_available": False,
            "alignment_source_status": "not_supplied",
            "alignment_reason": "not_supplied",
        }
    )
    cell_limited = np.zeros(len(source), dtype=bool)
    cells_recorded = np.zeros(len(source), dtype=bool)
    for names in (("n_cells0", "n_cells1"), ("n0", "n1")):
        if all(k in source for k in names):
            counts = np.column_stack([numeric_column(source, k) for k in names])
            if (counts < 0).any() or (
                np.isfinite(counts) & (counts != np.floor(counts))
            ).any():
                raise ValueError(
                    "Source cell counts must be nonnegative integers or missing."
                )
            cell_limited |= np.any(np.isfinite(counts) & (counts < min_cells), axis=1)
            cells_recorded |= np.isfinite(counts).all(axis=1)
    evidence.loc[cell_limited, "effect_available"] = False
    descriptive = np.ones(len(source), dtype=bool)
    if "descriptive_only" in source:
        if not source.descriptive_only.dropna().isin([True, False, 0, 1]).all():
            raise ValueError("descriptive_only must be boolean or missing.")
        descriptive = (
            source.descriptive_only.fillna(True).astype(bool).to_numpy(copy=True)
        )
    elif "inference_level" in source:
        descriptive = ~source.inference_level.eq("biological_sample").to_numpy()
    low_samples = np.zeros(len(source), dtype=bool)
    samples_recorded = np.zeros(len(source), dtype=bool)
    for names in (
        ("n_samples0", "n_samples1"),
        ("n_samples_group0", "n_samples_group1"),
    ):
        if all(k in source for k in names):
            counts = np.column_stack([numeric_column(source, k) for k in names])
            if (counts < 0).any() or (
                np.isfinite(counts) & (counts != np.floor(counts))
            ).any():
                raise ValueError(
                    "Source sample counts must be nonnegative integers or missing."
                )
            low_samples |= np.any(np.isfinite(counts) & (counts < 2), axis=1)
            samples_recorded |= np.isfinite(counts).all(axis=1)
    low_samples |= status.eq("insufficient_samples").to_numpy()
    known_descriptive = _descriptive_inference(source)
    descriptive |= low_samples | known_descriptive
    evidence["effect_descriptive"] = descriptive | ~cells_recorded | ~samples_recorded
    evidence["effect_supported"] = (
        evidence.effect_available & ~evidence.effect_descriptive
    )
    evidence["effect_support_status"] = np.select(
        [
            cell_limited,
            ~valid.to_numpy(),
            ~np.isfinite(effect),
            low_samples,
            descriptive & samples_recorded,
            ~cells_recorded | ~samples_recorded,
        ],
        [
            "source_insufficient_cells",
            "source_ineligible",
            "effect_missing",
            "source_insufficient_samples",
            "source_descriptive",
            "source_support_not_recorded",
        ],
        default="source_supported",
    )
    keys = (
        ("magnitude_ci95_low", "magnitude_ci95_high")
        if metric == "delta_norm"
        else ("normalized_magnitude_ci95_low", "normalized_magnitude_ci95_high")
    )
    if any(k in source for k in keys):
        if not all(k in source for k in keys):
            raise ValueError(
                "Both endpoints of the matching effect interval are required."
            )
        low, high = [numeric_column(source, k) for k in keys]
        if (
            (np.isfinite(low) ^ np.isfinite(high)).any()
            or (low > high).any()
            or (low < 0).any()
        ):
            raise ValueError(
                "Stored effect intervals must be paired, nonnegative and ordered."
            )
        evidence["ci_low"], evidence["ci_high"] = low, high
        evidence["interval_available"] = evidence.effect_available & np.isfinite(low)
        evidence["interval_reason"] = np.where(
            evidence.interval_available, "stored_interval", "missing_or_ineligible"
        )
        limited = low_samples | known_descriptive
        if "descriptive_only" in source:
            if not source.descriptive_only.dropna().isin([True, False, 0, 1]).all():
                raise ValueError("descriptive_only must be boolean or missing.")
            limited |= source.descriptive_only.fillna(True).astype(bool).to_numpy()
        for names in (
            ("n_samples0", "n_samples1"),
            ("n_samples_group0", "n_samples_group1"),
        ):
            if all(k in source for k in names):
                counts = np.column_stack([numeric_column(source, k) for k in names])
                limited |= np.any(np.isfinite(counts) & (counts < 2), axis=1)
        evidence.loc[limited, "interval_available"] = False
        evidence.loc[limited, "interval_reason"] = (
            "source_descriptive_or_low_sample_support"
        )
        unverified_interval = evidence.interval_available & (
            ~cells_recorded | ~samples_recorded
        )
        evidence.loc[unverified_interval, "interval_reason"] = (
            "stored_interval_unverified_support"
        )
        if unverified_interval.any():
            warnings.append(
                "Displayed supplied intervals have unverified source support; they do not establish supported biological-sample inference."
            )
    alignment_source = None
    if alignment_df is not None:
        alignment_source = _state_table(
            alignment_df, alignment_state_key, node_key, "alignment_df"
        )
        align = numeric_column(alignment_source, "alignment_cosine")
        if (np.abs(align[np.isfinite(align)]) > 1 + 1e-8).any():
            raise ValueError("Stored alignment cosine must be between -1 and 1.")
        usable, align_status = _source_eligibility(
            alignment_source, ("dynamics_status", "status", "coverage_status")
        )
        cell_limited_alignment = np.zeros(len(alignment_source), dtype=bool)
        for columns in (("n_cells0", "n_cells1"), ("n0", "n1")):
            if all(k in alignment_source for k in columns):
                counts = np.column_stack(
                    [numeric_column(alignment_source, k) for k in columns]
                )
                if (counts < 0).any() or (
                    np.isfinite(counts) & (counts != np.floor(counts))
                ).any():
                    raise ValueError(
                        "Alignment source cell counts must be nonnegative integers or missing."
                    )
                cell_limited_alignment |= np.any(
                    np.isfinite(counts) & (counts < min_cells), axis=1
                )
        zero_shift = align_status.eq("zero_shift").to_numpy(copy=True)
        for frame in (alignment_source, source):
            for column in ("delta_norm", "normalized_delta_norm"):
                if column in frame:
                    norms = numeric_column(frame, column)
                    if (norms < 0).any():
                        raise ValueError(
                            "Stored displacement magnitudes cannot be negative."
                        )
                    zeros = pd.Series(
                        np.isfinite(norms) & (norms == 0), index=frame._state_id
                    )
                    zero_shift |= (
                        alignment_source._state_id.map(zeros)
                        .fillna(False)
                        .to_numpy(bool)
                    )
        zero_velocity = np.zeros(len(alignment_source), dtype=bool)
        for column in ("velocity_norm", "vel_norm"):
            if column in alignment_source:
                norms = numeric_column(alignment_source, column)
                if (norms < 0).any():
                    raise ValueError("Stored velocity magnitudes cannot be negative.")
                zero_velocity |= np.isfinite(norms) & (norms == 0)
        available_alignment = (
            usable.to_numpy()
            & np.isfinite(align)
            & ~cell_limited_alignment
            & ~zero_shift
            & ~zero_velocity
        )
        alignment_reason = np.select(
            [
                zero_shift,
                zero_velocity,
                cell_limited_alignment,
                ~usable.to_numpy(),
                ~np.isfinite(align),
            ],
            [
                "undefined_zero_shift",
                "undefined_zero_velocity",
                "source_insufficient_cells",
                "source_ineligible",
                "alignment_missing",
            ],
            default="available",
        )
        lookup = pd.DataFrame(
            {
                "state": alignment_source._state_id.to_numpy(),
                "alignment": align,
                "alignment_source_eligible": usable.to_numpy(),
                "alignment_available": available_alignment,
                "alignment_source_status": align_status.to_numpy(),
                "alignment_reason": alignment_reason,
            }
        ).set_index("state")
        for column in lookup:
            evidence[column] = evidence.state.map(lookup[column])
        for column in ("alignment_source_eligible", "alignment_available"):
            evidence[column] = evidence[column].fillna(False).astype(bool)
        evidence["alignment_source_status"] = evidence.alignment_source_status.fillna(
            "state_not_supplied"
        )
        evidence["alignment_reason"] = evidence.alignment_reason.fillna(
            "state_not_supplied"
        )
    return evidence, source, alignment_source, metric, provenance, warnings
