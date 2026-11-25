"""Descriptive composition values for PAGA pie glyphs.

These helpers never fit topology, state shifts, velocities, or statistical
tests. A sample-normalized abundance profile has a different denominator from
the pooled captured-cell composition and retains both quantities explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cell_fraction_pie_values(counts):
    """Expose the legacy pooled-cell pie fractions with their denominator."""
    values = counts.copy(deep=True)
    values["cell_fraction"] = values["fraction"]
    values["mode"] = "cell_fraction"
    values["status"] = np.where(
        values["denominator"].gt(0), "available", "absent_from_contrast"
    )
    return values


def sample_abundance_pies(cells, sample_labels, states, categories, groups, counts):
    """Average state frequencies within conditions before normalizing pie slices.

    The roster comprises every sample×condition unit in the selected contrast,
    before any state, focus, or drawing filter. Each sample denominator contains
    its annotated selected-condition cells. A state absent from an otherwise
    observed sample has frequency zero. A sample with no annotated cells has
    unavailable frequency and is retained explicitly instead of becoming zero.
    """
    selected = cells.in_contrast.to_numpy(dtype=bool)
    if sample_labels[selected].isna().any():
        raise ValueError(
            "pie_sample_key contains missing sample IDs in the selected conditions."
        )
    roster = cells.loc[selected, ["condition", "state"]].copy()
    roster["sample"] = sample_labels[selected].to_numpy()
    units = (
        roster.groupby(["condition", "sample"], sort=True, observed=True)
        .agg(n_cells=("state", "size"), n_annotated_cells=("state", "count"))
        .reset_index()
    )
    observed_counts = (
        roster.loc[roster.state.notna()]
        .groupby(["condition", "sample", "state"], sort=False, observed=True)
        .size()
        .to_dict()
    )
    records = []
    for state in states:
        for unit in units.itertuples(index=False):
            count = int(observed_counts.get((unit.condition, unit.sample, state), 0))
            denominator = int(unit.n_annotated_cells)
            records.append(
                {
                    "state": state,
                    "condition": unit.condition,
                    "sample": unit.sample,
                    "count": count,
                    "denominator": denominator,
                    "n_cells": int(unit.n_cells),
                    "n_missing_state": int(unit.n_cells) - denominator,
                    "state_fraction": count / denominator if denominator else np.nan,
                    "included": denominator > 0,
                    "status": "available" if denominator else "no_annotated_cells",
                }
            )
    sample_counts = pd.DataFrame(
        records,
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
        ],
    )
    values = counts.copy(deep=True)
    values["cell_fraction"] = values["fraction"]
    means, valid_counts, total_counts = {}, {}, {}
    for state in states:
        for group in groups:
            selected_rows = sample_counts.loc[
                sample_counts.state.eq(state) & sample_counts.condition.eq(group)
            ]
            valid = selected_rows.loc[selected_rows.included, "state_fraction"]
            means[state, group] = float(valid.mean()) if len(valid) else np.nan
            valid_counts[state, group] = len(valid)
            total_counts[state, group] = len(selected_rows)

    abundance, sample_n, roster_n, denominators, fractions, statuses = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    wide = []
    for state in states:
        condition_means = np.array([means[state, group] for group in groups])
        complete = np.isfinite(condition_means).all()
        denominator = float(condition_means.sum()) if complete else np.nan
        row = {"node": state}
        for category in categories:
            if category in groups:
                mean = means[state, category]
                n_valid = valid_counts[state, category]
                n_total = total_counts[state, category]
                if not complete:
                    fraction, status = np.nan, "missing_sample_denominator"
                elif denominator == 0:
                    fraction, status = 0.0, "absent_from_contrast"
                else:
                    fraction, status = mean / denominator, "available"
            else:
                mean, n_valid, n_total = np.nan, 0, 0
                fraction, status = 0.0, "outside_selected_conditions"
            abundance.append(mean)
            sample_n.append(n_valid)
            roster_n.append(n_total)
            denominators.append(denominator)
            fractions.append(fraction)
            statuses.append(status)
            row[category] = fraction
        wide.append(row)
    values["mean_sample_abundance"] = abundance
    values["n_samples"] = sample_n
    values["n_samples_total"] = roster_n
    values["n_samples_missing_denominator"] = values.n_samples_total - values.n_samples
    values["abundance_denominator"] = denominators
    values["fraction"] = fractions
    values["mode"] = "sample_abundance"
    values["status"] = statuses
    return pd.DataFrame(wide), values, sample_counts
