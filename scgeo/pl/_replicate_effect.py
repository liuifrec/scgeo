"""Descriptive biological-replicate views of stored representation coordinates."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ..get._sample_centers import sample_centers
from ._result import PlotResult
from ._style import category_palette
from ._table_plot_utils import ordered_labels, panel_axes, required_table


def replicate_effect(
    data_or_adata,
    *,
    coordinate,
    store_key="robust_shift",
    scope="auto",
    state_order=None,
    condition_order=None,
    pair_key=None,
    summary="mean",
    palette=None,
    jitter=.08,
    seed=0,
    axes=None,
    figsize=None,
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Plot one biological sample center per point in an explicit coordinate.

    ``data_or_adata`` is an AnnData with sample-aware robust-shift results, the
    result dictionary, or a tidy table from ``sg.get.sample_centers``. Required
    table columns are ``state``, ``condition``, ``sample``, ``coordinate``,
    ``value``, and ``representation``. ``coordinate`` is explicit and zero-based
    for tables produced by the getter. Select one representation before plotting.

    ``summary='mean'`` or ``'median'`` gives an equally weighted, descriptive
    summary of the plotted sample coordinates; it is not necessarily the
    multivariate robust condition center. ``summary=None`` hides it. No interval,
    hypothesis test, or cell-weighted estimate is computed. Pair lines are drawn
    only when ``pair_key`` explicitly identifies matching biological samples.
    Missing pairs stay unconnected; ambiguous pairs are rejected.

    ``axes`` is an optional mapping from state labels to Axes. Returns a Figure,
    ``(Figure, data_dict)`` with ``return_data=True``, or a :class:`PlotResult`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    source = data_or_adata if isinstance(data_or_adata, pd.DataFrame) else sample_centers(
        data_or_adata, store_key=store_key, scope=scope,
    )
    work = required_table(source, ["state", "condition", "sample", "coordinate", "value", "representation"])
    if isinstance(coordinate, np.generic):
        coordinate = coordinate.item()
    if coordinate is None:
        raise ValueError("Select coordinate explicitly; projections are never inferred from the observed contrast.")
    for key in ("state", "condition", "sample", "representation"):
        if work[key].isna().any():
            raise ValueError(f"{key} values must not be missing.")
        work[key] = work[key].astype(str)
    if work.representation.nunique() != 1:
        raise ValueError("Select exactly one representation before plotting sample coordinates.")
    work = work.loc[work.coordinate == coordinate].copy()
    if work.empty:
        raise ValueError(f"No sample centers for coordinate={coordinate!r}.")
    if work.duplicated(["state", "condition", "sample"]).any():
        raise ValueError("Each state/condition/sample must have exactly one selected coordinate value.")
    work["value"] = pd.to_numeric(work["value"], errors="raise").astype(float)
    if np.isinf(work.value).any():
        raise ValueError("Sample values must be finite or missing.")
    if summary not in {None, "mean", "median"}:
        raise ValueError("summary must be 'mean', 'median', or None.")
    if not np.isfinite(jitter) or not 0 <= jitter < .4:
        raise ValueError("jitter must be finite and in [0, 0.4).")
    states = ordered_labels(work.state, state_order, "state_order")
    conditions = ordered_labels(work.condition, condition_order, "condition_order")
    colors = category_palette(conditions, palette)
    if pair_key is not None:
        if pair_key not in work:
            raise ValueError(f"Explicit pair_key={pair_key!r} is absent from the table.")
        paired = work.loc[work[pair_key].notna()]
        if paired.duplicated(["state", "condition", pair_key]).any():
            raise ValueError("Pair identifiers must select at most one sample per state and condition.")
    figure, axis_map, created = panel_axes(
        states, axes, figsize or (max(4, 3.3 * len(states)), 3.8), sharey=True,
    )
    rng = np.random.RandomState(seed)
    work["display_x"] = work.condition.map({name: i for i, name in enumerate(conditions)}).astype(float)
    work["display_x"] += rng.uniform(-jitter, jitter, size=len(work))
    summaries, pair_records, artists = [], [], {}
    representation = str(work.representation.iloc[0])
    for panel_index, state in enumerate(states):
        ax = axis_map[state]
        rows = work.loc[work.state == state]
        point_artists, line_artists, summary_artists = [], [], []
        counts = []
        for x, condition in enumerate(conditions):
            group = rows.loc[rows.condition == condition]
            valid = group.loc[group.value.notna()]
            n = len(valid)
            counts.append(n)
            point_artists.append(ax.scatter(valid.display_x, valid.value, color=colors[condition],
                                             s=32, edgecolor="white", linewidth=.5, zorder=3))
            value = np.nan
            if n and summary is not None:
                value = float(valid.value.mean() if summary == "mean" else valid.value.median())
                summary_artists.append(ax.hlines(value, x-.22, x+.22, color="0.2", linewidth=2, zorder=4))
            summaries.append({"state": state, "condition": condition, "coordinate": coordinate,
                              "representation": representation, "summary": summary,
                              "value": value, "n_samples": n, "n_missing": len(group)-n})
        if pair_key is not None:
            for pair, group in rows.loc[rows[pair_key].notna() & rows.value.notna()].groupby(pair_key, sort=False):
                if len(group) < 2:
                    continue
                group = group.assign(_order=group.condition.map({v: i for i, v in enumerate(conditions)})).sort_values("_order")
                line_artists.extend(ax.plot(group.display_x, group.value, color="0.65", linewidth=.8, zorder=1))
                for left, right in zip(group.iloc[:-1].to_dict("records"), group.iloc[1:].to_dict("records")):
                    pair_records.append({"state": state, "pair_id": str(pair), "condition0": left["condition"],
                                         "condition1": right["condition"], "sample0": left["sample"],
                                         "sample1": right["sample"], "value0": left["value"], "value1": right["value"]})
        if not counts or sum(counts) == 0:
            ax.text(.5, .5, "No evaluable samples", transform=ax.transAxes, ha="center", color="0.45")
        missing_count = int(rows.value.isna().sum())
        if missing_count:
            ax.text(.02, .98, f"{missing_count} sample value(s) unavailable", transform=ax.transAxes,
                    ha="left", va="top", color="0.45", fontsize=8)
        ax.set_xticks(np.arange(len(conditions)))
        ax.set_xticklabels([f"{condition}\nn={n}" for condition, n in zip(conditions, counts)])
        ax.set_xlim(-.5, len(conditions)-.5)
        ax.set_title(state)
        ax.set_ylabel(f"{representation} · coordinate {coordinate}\nSample center (descriptive)" if panel_index == 0 else "")
        ax.tick_params(axis="y", labelleft=panel_index == 0)
        ax.grid(axis="y", color="0.9", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        if summary is not None and panel_index == 0:
            ax.legend(handles=[Line2D([0], [0], color="0.2", lw=2, label=f"{summary} of sample coordinates")],
                      loc="best", frameon=False, fontsize=8)
        artists[state] = {"samples": point_artists, "pairs": line_artists, "summaries": summary_artists}
    # Respect supplied axes while giving all state panels a comparable scale.
    low = min(ax.get_ylim()[0] for ax in axis_map.values())
    high = max(ax.get_ylim()[1] for ax in axis_map.values())
    for ax in axis_map.values():
        ax.set_ylim(low, high)
    if title:
        if created:
            figure.suptitle(title)
        else:
            axis_map[states[0]].set_title(f"{title}\n{states[0]}")
    if created:
        figure.tight_layout()
    work.attrs = deepcopy(source.attrs)
    data = {"sample_centers": work, "sample_summary": pd.DataFrame(summaries),
            "pairs": pd.DataFrame(pair_records, columns=["state", "pair_id", "condition0", "condition1",
                                                          "sample0", "sample1", "value0", "value1"])}
    metadata = {"plot": "replicate_effect", "coordinate": coordinate, "representation": representation,
                "state_order": states, "condition_order": conditions, "pair_key": pair_key,
                "summary": summary, "seed": int(seed), "jitter": float(jitter),
                "inference": "Descriptive, equally weighted biological samples; no hypothesis test or CI computed.",
                "source_attrs": deepcopy(source.attrs)}
    result = PlotResult(fig=figure, axes=axis_map, data=data, metadata=metadata, warnings=[], artists=artists)
    if show:
        plt.show()
    if return_result:
        return result
    return (figure, data) if return_data else figure
