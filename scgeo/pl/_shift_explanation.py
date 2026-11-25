"""Expression curves and display maps for a stored geometric shift explanation.

These plots describe a frozen, expression-derived coordinate. They do not fit
pseudotime, estimate RNA velocity, or reconstruct a biological trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm, is_color_like
from matplotlib.patches import FancyArrowPatch
from scipy import sparse

from ._result import PlotResult
from ._style import category_palette, style_context
from ._table_plot_utils import panel_axes
from ._utils import display_indices, observation_labels, resolve_embedding


def _positive_int(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _labels(values, name):
    if isinstance(values, (str, bytes)):
        values = [values]
    try:
        labels = [str(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a label or a sequence of labels.") from exc
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"{name} must contain unique labels and cannot be empty.")
    return labels


def _explanation(adata, key):
    bundle = adata.uns.get("scgeo", {}).get(key)
    if not isinstance(bundle, Mapping):
        raise KeyError(
            f"Run scgeo.tl.explain_shift first; adata.uns['scgeo'][{key!r}] is missing."
        )
    if not isinstance(bundle.get("params"), Mapping):
        raise TypeError("The stored explanation requires a params mapping.")
    cells = bundle.get("cells")
    required = {
        "obs_position",
        "obs_name",
        "state",
        "condition",
        "sample",
        "shift_score",
    }
    if not isinstance(cells, pd.DataFrame) or not required <= set(cells):
        raise ValueError(
            f"The explanation cells table requires columns {sorted(required)}."
        )
    cells = cells.copy(deep=True).reset_index(drop=True)
    if cells.empty or cells[["state", "condition", "obs_name"]].isna().any().any():
        raise ValueError(
            "Explanation cells must have nonmissing state, condition and observation labels."
        )
    position = pd.to_numeric(cells.obs_position, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(position).all() or np.any(position != np.floor(position)):
        raise ValueError("obs_position must contain finite integer positions.")
    position = position.astype(int)
    if np.any(position < 0) or np.any(position >= adata.n_obs):
        raise ValueError("Explanation obs_position is outside the current AnnData.")
    if len(np.unique(position)) != len(position):
        raise ValueError("Explanation cells contains duplicate observation positions.")
    if not np.array_equal(
        np.asarray(adata.obs_names[position], dtype=str),
        cells.obs_name.astype(str).to_numpy(),
    ):
        raise ValueError(
            "Explanation observation names no longer match AnnData; recompute after reordering or subsetting."
        )
    from ..tl._explain_shift import _check_input_signature

    _check_input_signature(adata, bundle)
    cells["obs_position"] = position
    for column in ("state", "condition"):
        values, _, _ = observation_labels(cells[column], column)
        cells[column] = values.to_numpy()
    score = pd.to_numeric(cells.shift_score, errors="raise").to_numpy(dtype=float)
    if np.isinf(score).any():
        raise ValueError("shift_score must be finite or missing.")
    cells["shift_score"] = score
    has_samples = bundle["params"].get("sample_key") is not None
    if has_samples and cells["sample"].isna().any():
        raise ValueError("Biological sample labels cannot be missing.")
    if has_samples:
        values, _, _ = observation_labels(cells["sample"], "sample")
        cells["sample"] = values.to_numpy()
    else:
        cells["sample"] = "pooled cells"
    for column, parameter in (
        ("condition", "condition_key"),
        ("state", "by"),
        ("sample", "sample_key"),
    ):
        annotation = bundle["params"].get(parameter)
        if annotation is not None:
            if annotation not in adata.obs:
                raise KeyError(
                    f"Stored annotation {annotation!r} is absent from AnnData."
                )
            current = adata.obs.iloc[position][annotation]
            if current.isna().any() or not np.array_equal(
                current.astype(str).to_numpy(), cells[column].to_numpy()
            ):
                raise ValueError(
                    f"Stored {column} labels no longer match AnnData; recompute explain_shift."
                )
    rep = bundle["params"].get("rep")
    if "input_signature" not in bundle and rep in adata.obsm:
        coordinates = np.asarray(adata.obsm[rep], dtype=float)
        rtol = float(bundle["params"].get("mapping_rtol", 1e-5))
        atol = float(bundle["params"].get("mapping_atol", 1e-6))
        for entry in bundle.get("state_results", {}).values():
            if not isinstance(entry, Mapping) or "state" not in entry:
                continue
            selected = cells.loc[
                cells.state.eq(str(entry["state"])) & np.isfinite(cells.shift_score)
            ]
            if selected.empty:
                continue
            direction = np.asarray(entry.get("direction"), dtype=float)
            reference = np.asarray(entry.get("center0"), dtype=float)
            if (
                coordinates.ndim != 2
                or direction.shape != (coordinates.shape[1],)
                or reference.shape != direction.shape
            ):
                raise ValueError(
                    "Stored shift direction no longer matches the analysis representation."
                )
            current = (
                coordinates[selected.obs_position.to_numpy(dtype=int)] - reference
            ) @ direction
            if not np.allclose(
                current, selected.shift_score.to_numpy(), rtol=rtol, atol=atol
            ):
                raise ValueError(
                    "Stored shift scores no longer match the analysis representation; recompute explain_shift."
                )
    return bundle, cells, has_samples


def _selected_state(cells, state):
    states = list(pd.unique(cells.state))
    if state is None:
        if len(states) != 1:
            raise ValueError(
                f"Choose one state for comparable shift coordinates; available states: {states}."
            )
        state = states[0]
    state = str(state)
    if state not in states:
        raise ValueError(f"Unknown state {state!r}; available states: {states}.")
    return state, cells.loc[cells.state.eq(state)].copy().reset_index(drop=True)


def _expression_column(adata, positions, gene, layer):
    if not adata.var_names.is_unique:
        raise ValueError("AnnData var_names must be unique.")
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} is not present in adata.var_names.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Stored expression layer {layer!r} is not present in AnnData.")
    matrix = adata.X if layer is None else adata.layers[layer]
    if matrix is None:
        raise ValueError("The selected expression matrix is absent.")
    column = matrix[positions, adata.var_names.get_loc(gene)]
    if sparse.issparse(column):
        column = column.toarray()
    values = np.asarray(column, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Expression values for {gene!r} must be finite.")
    return values


def _features(
    adata, bundle, cells, state, var_names, program_names, *, attribution_available=True
):
    genes = bundle.get("genes")
    if not isinstance(genes, pd.DataFrame) or not {"state", "gene"} <= set(genes):
        raise ValueError(
            "The stored explanation requires a genes table with state and gene columns."
        )
    genes = genes.loc[genes.state.astype(str).eq(state)].copy()
    if genes.gene.astype(str).duplicated().any():
        raise ValueError("The explanation contains duplicate genes within this state.")
    genes["gene"] = genes.gene.astype(str)
    genes = genes.set_index("gene", drop=False)
    if var_names is None and program_names is None:
        if not attribution_available:
            raise ValueError(
                "No valid contribution ranking is available for this state. "
                "Supply var_names or program_names to inspect observed expression."
            )
        if "contribution" not in genes:
            raise ValueError(
                "Supply var_names when no stored contribution ranking is available."
            )
        scores = pd.to_numeric(genes.contribution, errors="raise").abs()
        var_names = (
            scores.loc[np.isfinite(scores)]
            .sort_values(ascending=False, kind="stable")
            .head(3)
            .index.tolist()
        )
        if not var_names:
            raise ValueError(
                "No finite contribution ranking is available. Supply var_names or "
                "program_names to inspect observed expression."
            )
    requested = []
    if var_names is not None:
        requested.extend(
            ("gene", gene, [(gene, 1.0)]) for gene in _labels(var_names, "var_names")
        )
    if program_names is not None:
        membership = bundle.get("module_membership")
        if not isinstance(membership, pd.DataFrame) or not {
            "program",
            "gene",
            "weight",
        } <= set(membership):
            raise ValueError(
                "Program curves require stored module_membership (program, gene, weight)."
            )
        for program in _labels(program_names, "program_names"):
            selected = membership.loc[membership.program.astype(str).eq(program)].copy()
            if selected.empty:
                raise ValueError(f"Unknown program {program!r}.")
            if selected.gene.astype(str).duplicated().any():
                raise ValueError(f"Program {program!r} contains duplicate genes.")
            weights = pd.to_numeric(selected.weight, errors="raise").to_numpy(
                dtype=float
            )
            if (
                not np.isfinite(weights).all()
                or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1.0)
            ):
                raise ValueError(
                    f"Program {program!r} requires positive expression weights summing to one."
                )
            requested.append(
                ("program", program, list(zip(selected.gene.astype(str), weights)))
            )
    positions = cells.obs_position.to_numpy(dtype=int)
    layer = bundle["params"].get("layer")
    cache, output, feature_rows = {}, {}, []
    state_entries = [
        entry
        for entry in bundle.get("state_results", {}).values()
        if isinstance(entry, Mapping) and str(entry.get("state")) == state
    ]
    entry = state_entries[0] if len(state_entries) == 1 else None
    for kind, name, members in requested:
        feature_id = f"{kind}:{name}"
        expression, contribution = np.zeros(len(cells)), np.zeros(len(cells))
        mapped = 0
        for gene, weight in members:
            if gene not in cache:
                cache[gene] = _expression_column(adata, positions, gene, layer)
                if (
                    "input_signature" not in bundle
                    and entry is not None
                    and gene in genes.index
                ):
                    for suffix in ("0", "1"):
                        reference_name = f"expression_center{suffix}"
                        if reference_name not in genes:
                            continue
                        old = float(genes.loc[gene, reference_name])
                        group_positions = np.asarray(
                            entry.get(f"obs_positions{suffix}", []), dtype=int
                        )
                        weights = np.asarray(
                            entry.get(f"center_weights{suffix}", []), dtype=float
                        )
                        if (
                            group_positions.size
                            and weights.shape == group_positions.shape
                            and np.isfinite(old)
                        ):
                            current = weights @ _expression_column(
                                adata, group_positions, gene, layer
                            )
                            if not np.isclose(
                                current,
                                old,
                                rtol=bundle["params"].get("mapping_rtol", 1e-5),
                                atol=bundle["params"].get("mapping_atol", 1e-6),
                            ):
                                raise ValueError(
                                    f"Expression for {gene!r} no longer matches stored feature centers; recompute explain_shift."
                                )
            expression += weight * cache[gene]
            if gene in genes.index and {"axis_loading", "expression_center0"} <= set(
                genes
            ):
                row = genes.loc[gene]
                coefficient, reference = (
                    float(row.axis_loading),
                    float(row.expression_center0),
                )
                if np.isfinite(coefficient) and np.isfinite(reference):
                    contribution += (cache[gene] - reference) * coefficient
                    mapped += int(bool(row.get("in_representation", True)))
                else:
                    contribution[:] = np.nan
            else:
                contribution[:] = np.nan
        if not attribution_available:
            contribution[:] = np.nan
            mapped = 0
        output[feature_id] = (expression, contribution)
        feature_rows.append(
            {
                "feature": feature_id,
                "name": name,
                "kind": kind,
                "n_members": len(members),
                "n_attributed_members": mapped,
                "expression_summary": "selected-layer expression"
                if kind == "gene"
                else "weighted mean expression",
                "contribution_summary": "signed axis contribution"
                if kind == "gene"
                else "sum of member axis contributions",
            }
        )
    return output, pd.DataFrame(feature_rows)


def _smooth_samples(
    cells,
    values,
    *,
    grid,
    bandwidth,
    min_cells,
    has_samples,
    uncertainty,
    coordinate="shift_score",
    conditions=None,
):
    """Compact-kernel means within each sample, with explicit unsupported points."""
    sample_rows, summary_rows = [], []
    for condition in pd.unique(cells.condition) if conditions is None else conditions:
        group = cells.loc[cells.condition.eq(condition)]
        fits = []
        counts = []
        for sample, frame in group.groupby("sample", sort=False, observed=True):
            indexes = frame.index.to_numpy(dtype=int)
            x, y = frame[coordinate].to_numpy(dtype=float), values[indexes]
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            estimate, support = (
                np.full(len(grid), np.nan),
                np.zeros(len(grid), dtype=int),
            )
            if len(x):
                for j, point in enumerate(grid):
                    # Neither extrapolate beyond this sample nor bridge gaps wider than the kernel.
                    distance = np.abs(x - point) / bandwidth
                    used = distance < 1.0
                    support[j] = int(used.sum())
                    if x.min() <= point <= x.max() and support[j] >= min_cells:
                        weights = np.power(1.0 - np.power(distance[used], 3), 3)
                        if weights.sum() > 0:
                            estimate[j] = float(
                                np.dot(weights, y[used]) / weights.sum()
                            )
                    sample_rows.append(
                        {
                            "condition": condition,
                            "sample": sample,
                            "shift_score": float(point),
                            "expression": estimate[j],
                            "n_cells": support[j],
                            "supported": bool(np.isfinite(estimate[j])),
                        }
                    )
            fits.append(estimate)
            counts.append(support)
        fits = np.asarray(fits).reshape((-1, len(grid)))
        counts = np.asarray(counts).reshape((-1, len(grid)))
        for j, point in enumerate(grid):
            valid = np.isfinite(fits[:, j])
            n = int(valid.sum())
            present = fits[valid, j]
            low = high = np.nan
            if uncertainty == "sample_range" and has_samples and n >= 2:
                low, high = float(present.min()), float(present.max())
            summary_rows.append(
                {
                    "condition": condition,
                    "shift_score": float(point),
                    "expression": float(present.mean()) if n else np.nan,
                    "lower": low,
                    "upper": high,
                    "n_samples": n if has_samples else 0,
                    "n_smoothing_units": n,
                    "n_cells": int(counts[valid, j].sum()),
                    "support": (
                        "no_observed_cells"
                        if group.empty
                        else "unsupported"
                        if n == 0
                        else "pooled_cells_only"
                        if not has_samples
                        else "single_sample"
                        if n == 1
                        else "multiple_samples"
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def _state_plot_status(bundle, cells, state):
    entries = [
        entry
        for entry in bundle.get("state_results", {}).values()
        if isinstance(entry, Mapping) and str(entry.get("state")) == state
    ]
    status = str(entries[0].get("status", "unknown")) if entries else "unknown"
    finite = np.isfinite(cells.shift_score)
    supported = (
        status not in {"zero_shift", "insufficient_cells"}
        and finite.any()
        and float(np.ptp(cells.loc[finite, "shift_score"])) > 1e-12
    )
    return status, bool(supported)


def _coverage(adata, bundle, cells, has_samples):
    """Keep zero-cell samples from the supplied contrast roster as missing data."""
    params = bundle["params"]
    conditions = [str(params["group0"]), str(params["group1"])]
    if len(set(conditions)) != 2 or not set(cells.condition) <= set(conditions):
        raise ValueError(
            "Stored cells must belong to the two distinct declared conditions."
        )
    condition_key, sample_key = params.get("condition_key"), params.get("sample_key")
    if condition_key not in adata.obs:
        raise KeyError(
            "The stored condition key is required to retain the sample roster."
        )
    condition_labels = adata.obs[condition_key].astype(str)
    rows = []
    for condition in conditions:
        roster = adata.obs.loc[condition_labels.eq(condition)]
        if has_samples:
            if sample_key not in roster or roster[sample_key].isna().any():
                raise ValueError(
                    "The supplied contrast roster requires nonmissing sample labels."
                )
            _, samples, _ = observation_labels(roster[sample_key], sample_key)
            sample_names = samples or [None]
        else:
            sample_names = ["pooled cells"]
        for sample in sample_names:
            mask = cells.condition.eq(condition) & cells["sample"].eq(sample)
            n_cells = int(mask.sum())
            rows.append(
                {
                    "state": str(cells.state.iloc[0]),
                    "condition": condition,
                    "sample": sample,
                    "n_cells": n_cells,
                    "n_coordinate_cells": int(
                        np.isfinite(cells.loc[mask, "curve_coordinate"]).sum()
                    ),
                    "sample_in_roster": bool(
                        len(roster) and (not has_samples or sample is not None)
                    ),
                    "biological_sample": bool(has_samples and sample is not None),
                    "status": "no_observed_cells"
                    if n_cells == 0
                    else "single_observed_cell"
                    if n_cells == 1
                    else "observed",
                }
            )
    return conditions, pd.DataFrame(rows)


def _sample_expression(cells, expression, coverage):
    rows = []
    for row in coverage.to_dict(orient="records"):
        mask = cells.condition.eq(row["condition"]) & cells["sample"].eq(row["sample"])
        values = expression[mask.to_numpy()]
        # A singleton is an observed value, never an estimated interval.
        q25, q75 = (
            np.quantile(values, [0.25, 0.75]) if len(values) >= 2 else (np.nan, np.nan)
        )
        rows.append(
            {
                **row,
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(q25),
                "q75": float(q75),
            }
        )
    return pd.DataFrame(rows)


def _unavailable_axis(axis, label, reason):
    axis.text(
        0.5,
        0.52,
        label,
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=10,
        weight="bold",
    )
    axis.text(
        0.5,
        0.38,
        reason,
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=8,
        color="0.35",
        wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _canvas(panel_names, axes, ax, nrows, ncols, figsize, colorbar_keys, show_colorbar):
    if ax is not None:
        if axes is not None or len(panel_names) != 1:
            raise ValueError(
                "ax is available only for a single panel and cannot be combined with axes."
            )
        axes = {panel_names[0]: ax}
    if axes is not None:
        required = list(panel_names) + (list(colorbar_keys) if show_colorbar else [])
        fig, all_axes, _ = panel_axes(required, axes, figsize)
        return fig, {name: all_axes[name] for name in panel_names}, all_axes, False
    fig, array = plt.subplots(
        nrows, ncols, squeeze=False, figsize=figsize, constrained_layout=True
    )
    selected = dict(zip(panel_names, array.ravel()))
    return fig, selected, selected.copy(), True


def _finish(result, return_result, show, save, default_stem):
    if save is not None and save is not False:
        stem = default_stem if save is True else save
        suffix = Path(stem).suffix.lower().lstrip(".")
        result.save(
            stem,
            formats=(suffix,) if suffix in {"png", "svg", "pdf"} else ("png", "svg"),
        )
    if show:
        plt.show()
    if return_result:
        return result
    return next(iter(result.axes.values())) if len(result.axes) == 1 else result.axes


def _validate_style(style, size, alpha, random_state, figsize, save):
    if style not in {"paper", "presentation"}:
        raise ValueError("style must be 'paper' or 'presentation'.")
    if isinstance(size, bool) or not np.isfinite(size) or size <= 0:
        raise ValueError("size must be a positive finite number.")
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one.")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer.")
    if figsize is not None:
        array = np.asarray(figsize, dtype=float)
        if array.shape != (2,) or not np.isfinite(array).all() or np.any(array <= 0):
            raise ValueError("figsize must contain two positive finite dimensions.")
    if save is not None and not isinstance(save, (bool, str, Path)):
        raise TypeError("save must be a path, boolean, or None.")


def shift_curves(
    adata,
    var_names=None,
    *,
    program_names=None,
    state=None,
    key="explain_shift",
    basis="umap",
    panels=("curve", "expression", "contribution"),
    coordinate_key=None,
    color=None,
    palette=None,
    cmap="viridis",
    contribution_cmap="coolwarm",
    bandwidth=0.18,
    grid_size=80,
    min_cells=5,
    uncertainty="sample_range",
    show_samples=True,
    max_points=3000,
    random_state=0,
    size=6,
    alpha=0.3,
    style="paper",
    figsize=None,
    title=None,
    axes=None,
    ax=None,
    show_colorbar=True,
    return_result=False,
    show=True,
    save=None,
):
    """Plot expression along a stored shift axis and coordinated embedding maps.

    Each requested gene or program gets a row with ``curve``, ``expression``
    and ``contribution`` panels. Use ``panels=('curve',)`` without an embedding.
    ``var_names=None`` selects up to three genes by absolute stored contribution.
    Select one ``state`` when the explanation contains several states.

    States without a supported shift can still use explicit ``var_names`` or
    ``program_names`` with ``panels=('distribution', 'expression')``. The
    distribution panel shows each biological sample's observed expression
    median and cell interquartile range, with its cell count. A singleton is
    one observed point without an interval. Zero-cell samples from the full
    supplied AnnData roster stay blank; missing expression is never zero.
    Both declared conditions remain visible. Without ``sample_key``, these
    are explicitly pooled-cell distributions. A distribution-only plot needs
    no embedding. Unsupported curve/contribution panels are labelled as such.
    An unsupported state has no automatic contribution-based gene selection.

    ``coordinate_key`` optionally names a numeric ``adata.obs`` column for
    descriptive curves. It must be explicitly supplied; the plot does not
    manufacture a direction or extrapolate an absent condition. The original
    stored ``shift_score`` is preserved. Exports use ``curve_coordinate`` and
    ``coordinate_source`` to distinguish a supplied coordinate from the shift.

    Curves are tricube-kernel means estimated separately in each biological
    sample and condition, then averaged with equal sample weights at every
    supported grid point. ``bandwidth`` is a fraction of the state's observed
    score range; ``min_cells`` is the required local support in each sample.
    Curves stop outside each sample's observed score range. Missing support
    stays missing. Thin lines show individual samples. The default ribbon is
    the observed between-sample range where at least two samples support a
    point. It is *not* a confidence interval. ``uncertainty='none'`` hides it.
    With no biological sample key, only a pooled descriptive curve is drawn.

    The default x axis is an expression-derived geometric coordinate, not time
    or an inferred trajectory. A supplied coordinate is used as provided and
    has no inferred temporal or causal meaning here. Curves and ribbons are
    conditional on the coordinate and smoothing choice, and cannot establish
    causal drivers.
    Program expression uses stored membership weights (a mean); program
    contribution sums the member genes' signed axis contributions.

    ``max_points`` affects scatter drawing only. ``color`` optionally selects
    an observation annotation for curve points; condition curves retain their
    own colors. ``axes`` maps names such as ``'gene:Gata1:curve'`` to distinct
    axes in one Figure. For supplied axes and ``show_colorbar=True``, also
    provide ``'gene:Gata1:expression_colorbar'`` and the analogous contribution
    colorbar axis. ``ax`` is a convenience for one feature and one panel.
    Supplied figures retain their size, layout and title. ``return_result=True``
    exposes full cell, curve, sample-curve, feature, coverage, sample-expression
    and panel-status tables via ``PlotResult``.
    No AnnData data are modified. Files are written only when ``save`` is set.
    """
    _validate_style(style, size, alpha, random_state, figsize, save)
    grid_size, min_cells = (
        _positive_int(grid_size, "grid_size", 3),
        _positive_int(min_cells, "min_cells", 2),
    )
    if (
        isinstance(bandwidth, bool)
        or not np.isfinite(bandwidth)
        or not 0 < bandwidth <= 1
    ):
        raise ValueError("bandwidth must be a fraction in (0, 1].")
    if uncertainty not in {"sample_range", "none"}:
        raise ValueError("uncertainty must be 'sample_range' or 'none'.")
    if not isinstance(show_samples, (bool, np.bool_)) or not isinstance(
        show_colorbar, (bool, np.bool_)
    ):
        raise TypeError("show_samples and show_colorbar must be boolean.")
    panels = _labels(panels, "panels")
    if not set(panels) <= {"curve", "distribution", "expression", "contribution"}:
        raise ValueError(
            "panels can contain only curve, distribution, expression and contribution."
        )
    expression_cmap, signed_cmap = (
        mpl.colormaps.get_cmap(cmap),
        mpl.colormaps.get_cmap(contribution_cmap),
    )
    bundle, cells, has_samples = _explanation(adata, key)
    state, cells = _selected_state(cells, state)
    state_status, shift_available = _state_plot_status(bundle, cells, state)
    values, features = _features(
        adata,
        bundle,
        cells,
        state,
        var_names,
        program_names,
        attribution_available=shift_available,
    )
    if coordinate_key is not None:
        if not isinstance(coordinate_key, str):
            raise TypeError("coordinate_key must name a numeric observation column.")
        if coordinate_key not in adata.obs:
            raise KeyError(
                f"coordinate_key {coordinate_key!r} is not present in adata.obs."
            )
        raw_coordinate = adata.obs.iloc[cells.obs_position.to_numpy(dtype=int)][
            coordinate_key
        ]
        if not pd.api.types.is_numeric_dtype(
            raw_coordinate
        ) or pd.api.types.is_bool_dtype(raw_coordinate):
            raise TypeError(
                "coordinate_key must name a numeric, nonboolean observation column."
            )
        coordinate = pd.to_numeric(raw_coordinate, errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
        if np.isinf(coordinate).any():
            raise ValueError("The supplied coordinate must be finite or missing.")
        coordinate_source = f"obs:{coordinate_key}"
    else:
        coordinate = cells.shift_score.to_numpy(dtype=float).copy()
        coordinate_source = "stored_shift_score"
    cells["curve_coordinate"] = coordinate
    cells["coordinate_source"] = coordinate_source
    finite_coordinate = np.isfinite(coordinate)
    curve_available = bool(
        finite_coordinate.any()
        and np.ptp(coordinate[finite_coordinate]) > 1e-12
        and (coordinate_key is not None or shift_available)
    )
    conditions, coverage = _coverage(adata, bundle, cells, has_samples)
    colors = category_palette(conditions, palette)
    scatter_colors = cells.condition.map(colors).to_numpy()
    scatter_numeric = None
    if color is not None:
        if color in adata.obs:
            annotation = adata.obs.iloc[cells.obs_position.to_numpy(dtype=int)][color]
            if pd.api.types.is_numeric_dtype(
                annotation
            ) and not pd.api.types.is_bool_dtype(annotation):
                scatter_numeric = pd.to_numeric(annotation, errors="raise").to_numpy(
                    dtype=float
                )
                if not np.isfinite(scatter_numeric).all():
                    raise ValueError(
                        "Numeric color annotations must be finite for selected cells."
                    )
                scatter_colors = scatter_numeric
            else:
                labels, categories, _ = observation_labels(annotation, color)
                annotation_palette = category_palette(categories)
                scatter_colors = labels.map(annotation_palette).to_numpy()
        elif is_color_like(color):
            scatter_colors = color
        else:
            raise KeyError(
                f"color {color!r} is neither an observation annotation nor a Matplotlib color."
            )
    display = display_indices(cells.condition.to_numpy(), max_points, random_state)
    cells["displayed"] = False
    cells.loc[display, "displayed"] = True
    feature_attribution = {
        feature: bool(
            shift_available
            and np.isfinite(data[1]).any()
            and features.set_index("feature").loc[feature, "n_attributed_members"] > 0
        )
        for feature, data in values.items()
    }
    if "expression" in panels or (
        "contribution" in panels and any(feature_attribution.values())
    ):
        display_key, coordinates = resolve_embedding(adata, basis)
        xy = np.asarray(
            coordinates[cells.obs_position.to_numpy(dtype=int), :2], dtype=float
        )
        if not np.isfinite(xy).all():
            raise ValueError("Selected embedding coordinates must be finite.")
        cells["x"], cells["y"] = xy[:, 0], xy[:, 1]
    else:
        display_key, xy = None, None
    if curve_available:
        low, high = (
            coordinate[finite_coordinate].min(),
            coordinate[finite_coordinate].max(),
        )
        grid = np.linspace(low, high, grid_size)
        absolute_bandwidth = float(bandwidth * (high - low))
    else:
        grid, absolute_bandwidth = np.empty(0), np.nan
    curve_tables, sample_tables, cell_tables, expression_tables, panel_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for feature, (expression, contribution) in values.items():
        if curve_available:
            curves, samples = _smooth_samples(
                cells,
                expression,
                grid=grid,
                bandwidth=absolute_bandwidth,
                min_cells=min_cells,
                has_samples=has_samples,
                uncertainty=uncertainty,
                coordinate="curve_coordinate",
                conditions=conditions,
            )
            observed_pairs = set(zip(samples.condition, samples["sample"]))
            missing_samples = [
                {
                    "condition": row.condition,
                    "sample": row.sample,
                    "shift_score": np.nan,
                    "expression": np.nan,
                    "n_cells": 0,
                    "supported": False,
                }
                for row in coverage.itertuples(index=False)
                if (row.condition, row.sample) not in observed_pairs
            ]
            if missing_samples:
                samples = pd.concat(
                    [samples, pd.DataFrame(missing_samples)], ignore_index=True
                )
        else:
            curves = pd.DataFrame(
                {
                    "condition": conditions,
                    "shift_score": np.nan,
                    "expression": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "n_samples": 0,
                    "n_smoothing_units": 0,
                    "n_cells": 0,
                    "support": "coordinate_unavailable",
                }
            )
            samples = coverage[["condition", "sample", "n_cells"]].copy()
            samples["shift_score"], samples["expression"], samples["supported"] = (
                np.nan,
                np.nan,
                False,
            )
        for table in (curves, samples):
            table.insert(0, "feature", feature)
            table.insert(0, "state", state)
            table["curve_coordinate"] = table.shift_score
            table["coordinate_source"] = coordinate_source
            if coordinate_key is not None:
                table["shift_score"] = np.nan
        curve_tables.append(curves)
        sample_tables.append(samples)
        distribution = _sample_expression(cells, expression, coverage)
        distribution.insert(1, "feature", feature)
        expression_tables.append(distribution)
        for panel in panels:
            available = (
                curve_available
                if panel == "curve"
                else feature_attribution[feature]
                if panel == "contribution"
                else True
            )
            reason = (
                ""
                if available
                else "no_supported_shift_coordinate"
                if panel == "curve" and coordinate_key is None
                else "no_variable_supplied_coordinate"
                if panel == "curve"
                else "shift_attribution_unavailable"
            )
            panel_rows.append(
                {
                    "state": state,
                    "feature": feature,
                    "panel": panel,
                    "status": "available" if available else "unavailable",
                    "reason": reason,
                }
            )
        source = cells.copy(deep=True)
        source["feature"], source["expression"], source["contribution"] = (
            feature,
            expression,
            contribution,
        )
        cell_tables.append(source)
    curves = pd.concat(curve_tables, ignore_index=True)
    samples = pd.concat(sample_tables, ignore_index=True)
    sources = pd.concat(cell_tables, ignore_index=True)
    sample_expression = pd.concat(expression_tables, ignore_index=True)
    panel_status = pd.DataFrame(panel_rows)
    warnings = [
        "Geometric shift coordinates are expression-derived; association and attribution are descriptive, not causal."
    ]
    if not has_samples:
        warnings.append(
            "No biological sample key: pooled cell curves have no sample uncertainty band."
        )
    if not shift_available:
        warnings.append(
            "No supported shift direction for this state: expression remains descriptive and no shift contribution is inferred."
        )
    if not curve_available and "curve" in panels:
        warnings.append(
            "No supported variable curve coordinate; the requested curve panel is labelled unavailable."
        )
    if coverage.n_cells.eq(0).any():
        warnings.append(
            "Zero observed cells imply unavailable state expression, not zero expression or biological absence."
        )
    if coordinate_key is not None:
        warnings.append(
            "Curves use the explicitly supplied observation coordinate; its causal or temporal interpretation is not inferred."
        )
    if has_samples and any(
        cells.loc[cells.condition.eq(c), "sample"].nunique() < 2 for c in conditions
    ):
        warnings.append(
            "At least one condition has fewer than two biological samples; its uncertainty band is withheld."
        )
    if shift_available and features.n_attributed_members.lt(features.n_members).any():
        warnings.append(
            "Some requested genes are not used in the analysis representation; attribution coverage is listed in features."
        )
    panel_names = [f"{feature}:{panel}" for feature in values for panel in panels]
    colorbar_keys = [
        f"{feature}:{panel}_colorbar"
        for feature in values
        for panel in panels
        if panel == "expression"
        or (panel == "contribution" and feature_attribution[feature])
    ]
    if figsize is None:
        figsize = (4.1 * len(panels), 3.4 * len(values))
    with style_context(style):
        fig, selected, supplied_axes, created = _canvas(
            panel_names,
            axes,
            ax,
            len(values),
            len(panels),
            figsize,
            colorbar_keys,
            show_colorbar,
        )
        artists = {}
        for feature, (expression, contribution) in values.items():
            feature_row = features.loc[features.feature.eq(feature)].iloc[0]
            for panel in panels:
                name = f"{feature}:{panel}"
                axis = selected[name]
                if panel == "distribution":
                    distribution = sample_expression.loc[
                        sample_expression.feature.eq(feature)
                    ]
                    tick_labels, median_artists, interval_artists = [], [], []
                    for x, row in enumerate(distribution.itertuples(index=False)):
                        condition_color = colors[row.condition]
                        if row.n_cells >= 2:
                            interval_artists.append(
                                axis.vlines(
                                    x,
                                    row.q25,
                                    row.q75,
                                    color=condition_color,
                                    lw=3,
                                    alpha=0.75,
                                )
                            )
                        if row.n_cells:
                            median_artists.append(
                                axis.scatter(
                                    [x],
                                    [row.median],
                                    s=36,
                                    color=condition_color,
                                    edgecolors="white",
                                    linewidths=0.5,
                                    zorder=3,
                                )
                            )
                        else:
                            axis.text(
                                x,
                                0.06,
                                "0 observed\ncells",
                                transform=axis.get_xaxis_transform(),
                                fontsize=7,
                                color="0.4",
                                ha="center",
                                va="bottom",
                            )
                        sample_label = (
                            str(row.sample)
                            if row.biological_sample
                            else "pooled cells"
                            if not has_samples
                            else "no sample records"
                        )
                        tick_labels.append(
                            f"{sample_label}\n{row.condition}\nn={row.n_cells}"
                        )
                    axis.set_xticks(np.arange(len(distribution)))
                    axis.set_xticklabels(tick_labels, fontsize=7)
                    axis.set_xlim(-0.6, len(distribution) - 0.4)
                    axis.set_ylabel(
                        "Mean program expression"
                        if feature_row.kind == "program"
                        else "Expression"
                    )
                    axis.set_title(
                        f"{feature_row['name']} · {'samples' if has_samples else 'pooled cells'}"
                    )
                    axis.text(
                        0.02,
                        0.98,
                        "Point: median · bar: observed cell IQR",
                        transform=axis.transAxes,
                        fontsize=7,
                        color="0.35",
                        va="top",
                    )
                    artists[f"{name}:medians"] = median_artists
                    artists[f"{name}:iqr"] = interval_artists
                elif panel == "curve":
                    if not curve_available:
                        _unavailable_axis(
                            axis,
                            "Curve unavailable",
                            "No supported shift coordinate."
                            if coordinate_key is None
                            else f"No variable observed coordinate in {coordinate_key}.",
                        )
                        axis.set_title(f"{feature_row['name']} · {state}")
                        continue
                    point_colors = (
                        scatter_colors
                        if isinstance(scatter_colors, str)
                        else scatter_colors[display]
                    )
                    artist = axis.scatter(
                        cells.curve_coordinate.iloc[display],
                        expression[display],
                        c=point_colors,
                        cmap=expression_cmap if scatter_numeric is not None else None,
                        s=size,
                        alpha=alpha,
                        linewidths=0,
                        rasterized=True,
                    )
                    artists[f"{name}:points"] = artist
                    for condition in conditions:
                        condition_color = colors[condition]
                        subset = samples.loc[
                            samples.feature.eq(feature)
                            & samples.condition.eq(condition)
                        ]
                        if show_samples and has_samples:
                            for _, frame in subset.groupby(
                                "sample", sort=False, observed=True
                            ):
                                axis.plot(
                                    frame.curve_coordinate,
                                    frame.expression,
                                    color=condition_color,
                                    lw=0.75,
                                    alpha=0.45,
                                )
                        frame = curves.loc[
                            curves.feature.eq(feature) & curves.condition.eq(condition)
                        ]
                        n_samples = cells.loc[
                            cells.condition.eq(condition), "sample"
                        ].nunique()
                        label = (
                            f"{condition} (0 observed cells)"
                            if not cells.condition.eq(condition).any()
                            else (
                                f"{condition} (n={n_samples} samples)"
                                if has_samples
                                else f"{condition} (pooled cells)"
                            )
                        )
                        axis.plot(
                            frame.curve_coordinate,
                            frame.expression,
                            color=condition_color,
                            lw=2,
                            label=label,
                        )
                        if uncertainty == "sample_range":
                            axis.fill_between(
                                frame.curve_coordinate.to_numpy(),
                                frame.lower.to_numpy(),
                                frame.upper.to_numpy(),
                                color=condition_color,
                                alpha=0.16,
                                linewidth=0,
                            )
                    axis.set_xlabel(
                        "Geometric shift coordinate"
                        if coordinate_key is None
                        else coordinate_key
                    )
                    axis.set_ylabel(
                        "Mean program expression"
                        if feature_row.kind == "program"
                        else "Expression"
                    )
                    axis.set_title(f"{feature_row['name']} · {state}")
                    axis.legend(loc="best", fontsize=8)
                    if has_samples and uncertainty == "sample_range":
                        axis.text(
                            0.02,
                            0.02,
                            "Band: between-sample range",
                            transform=axis.transAxes,
                            fontsize=7,
                            color="0.35",
                            va="bottom",
                        )
                else:
                    data = expression if panel == "expression" else contribution
                    finite = np.isfinite(data)
                    if panel == "contribution" and not feature_attribution[feature]:
                        if xy is None:
                            _unavailable_axis(
                                axis,
                                "Contribution unavailable",
                                "No supported shift attribution for this state.",
                            )
                            axis.set_title("Signed shift contribution")
                            continue
                        artist = axis.scatter(
                            xy[display, 0],
                            xy[display, 1],
                            c="0.8",
                            s=size,
                            alpha=alpha,
                            linewidths=0,
                        )
                        axis.text(
                            0.5,
                            0.5,
                            "Attribution unavailable"
                            if not shift_available or not finite.any()
                            else "Not used in analysis representation",
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
                    else:
                        if panel == "contribution":
                            bound = float(np.max(np.abs(data[finite]))) or 1.0
                            norm, color_map = (
                                TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                                signed_cmap,
                            )
                        else:
                            lower, upper = (
                                float(data[finite].min()),
                                float(data[finite].max()),
                            )
                            norm, color_map = (
                                Normalize(
                                    lower, upper if upper > lower else lower + 1.0
                                ),
                                expression_cmap,
                            )
                        artist = axis.scatter(
                            xy[display, 0],
                            xy[display, 1],
                            c=data[display],
                            cmap=color_map,
                            norm=norm,
                            s=size,
                            alpha=max(alpha, 0.65),
                            linewidths=0,
                            rasterized=True,
                        )
                        if show_colorbar:
                            cbar = (
                                fig.colorbar(artist, ax=axis, shrink=0.8, pad=0.03)
                                if created
                                else fig.colorbar(
                                    artist, cax=supplied_axes[f"{name}_colorbar"]
                                )
                            )
                            cbar.ax.tick_params(labelsize=8)
                            artists[f"{name}:colorbar"] = cbar
                    artists[f"{name}:points"] = artist
                    axis.set_title(
                        "Expression"
                        if panel == "expression"
                        else "Signed shift contribution"
                    )
                    axis.set_xlabel(f"{display_key.removeprefix('X_')} 1")
                    axis.set_ylabel(f"{display_key.removeprefix('X_')} 2")
                    axis.set_aspect("equal", adjustable="datalim")
                    axis.set_xticks([])
                    axis.set_yticks([])
        if created and title is not None:
            fig.suptitle(title)
        result = PlotResult(
            fig,
            selected,
            data={
                "cells": sources,
                "curves": curves,
                "sample_curves": samples,
                "features": features,
                "coverage": coverage,
                "sample_expression": sample_expression,
                "panel_status": panel_status,
            },
            metadata={
                "plot": "shift_curves",
                "state": state,
                "state_status": state_status,
                "shift_direction_available": shift_available,
                "explanation_key": key,
                "analysis_rep": bundle["params"].get("rep"),
                "display_basis": display_key,
                "layer": bundle["params"].get("layer"),
                "panels": panels,
                "sample_key": bundle["params"].get("sample_key"),
                "bandwidth_fraction": float(bandwidth),
                "bandwidth": absolute_bandwidth,
                "grid_size": grid_size,
                "min_cells_per_sample": min_cells,
                "sample_weighting": "equal among locally supported samples",
                "uncertainty": uncertainty,
                "interval_is_confidence_interval": False,
                "coordinate_interpretation": "frozen geometric projection; not time or a trajectory"
                if coordinate_key is None
                else "explicitly supplied observation coordinate; no temporal or causal meaning inferred",
                "coordinate_key": coordinate_key,
                "coordinate_source": coordinate_source,
                "curve_coordinate_available": curve_available,
                "sample_roster_source": "full supplied AnnData observations in the declared contrast",
                "distribution_summary": "one observed median and within-sample cell IQR per sample; no confidence interval"
                if has_samples
                else "pooled-cell median and observed cell IQR; no biological-sample inference",
                "max_points": max_points,
                "n_drawn_cells": len(display),
                "show_samples": bool(show_samples),
                "condition_colors": colors,
                "point_color": color or bundle["params"].get("condition_key"),
                "smoothing": "compact tricube kernel mean without extrapolation",
            },
            warnings=warnings,
            artists=artists,
        )
    return _finish(result, return_result, show, save, "scgeo_shift_curves")


def shift_embedding(
    adata,
    *,
    states=None,
    key="explain_shift",
    basis="umap",
    color=None,
    palette=None,
    curvature=0.2,
    max_points=5000,
    random_state=0,
    size=8,
    alpha=0.35,
    style="paper",
    figsize=(6.0, 5.0),
    title=None,
    ax=None,
    return_result=False,
    show=True,
    save=None,
):
    """Display condition-center contrasts as explicitly schematic curved arrows.

    Endpoints use the exact cell weights stored by ``explain_shift`` applied
    to the requested display coordinates. A nonlinear display embedding is
    not an inverse mapping of the analysis displacement. Arcs are graphic
    connectors between condition centers, not estimated cell trajectories or
    a velocity field. Use ``scgeo.pl.velocity_field`` for independently supplied
    RNA-velocity vectors. ``curvature=0`` draws straight arrows.

    ``color`` is a categorical observation key (default: stored condition).
    ``return_result=True`` exposes endpoint weights, all displayed-source cell
    coordinates and the exact center table. ``max_points`` only subsamples
    displayed cells; all cells retain their original center weights.
    """
    _validate_style(style, size, alpha, random_state, figsize, save)
    if isinstance(curvature, bool) or not np.isfinite(curvature) or abs(curvature) > 1:
        raise ValueError("curvature must be a finite number between -1 and 1.")
    bundle, cells, _has_samples = _explanation(adata, key)
    observed = list(pd.unique(cells.state))
    states = observed if states is None else _labels(states, "states")
    if not set(states) <= set(observed):
        raise ValueError(f"Unknown states: {sorted(set(states) - set(observed))}.")
    cells = cells.loc[cells.state.isin(states)].copy().reset_index(drop=True)
    display_key, coordinates = resolve_embedding(adata, basis)
    coordinates = np.asarray(coordinates[:, :2], dtype=float)
    xy = coordinates[cells.obs_position.to_numpy(dtype=int)]
    if not np.isfinite(xy).all():
        raise ValueError("Selected embedding coordinates must be finite.")
    color = color or bundle["params"].get("condition_key")
    if color is None or color not in adata.obs:
        raise KeyError("color must identify a categorical observation column.")
    annotation = adata.obs.iloc[cells.obs_position.to_numpy(dtype=int)][color]
    if pd.api.types.is_numeric_dtype(annotation) and not pd.api.types.is_bool_dtype(
        annotation
    ):
        raise ValueError(
            "shift_embedding color must be categorical; use shift_curves for continuous expression maps."
        )
    labels, categories, _ = observation_labels(annotation, color)
    colors = category_palette(categories, palette)
    cells["color_label"] = labels.to_numpy()
    cells["x"], cells["y"] = xy[:, 0], xy[:, 1]
    display_groups = pd.MultiIndex.from_frame(
        cells[["state", "condition"]]
    ).factorize()[0]
    display = display_indices(display_groups, max_points, random_state)
    cells["displayed"] = False
    cells.loc[display, "displayed"] = True
    raw_states = bundle.get("state_results")
    if not isinstance(raw_states, Mapping):
        raise TypeError(
            "Schematic arrows require stored state_results with center weights."
        )
    state_map = {}
    for entry in raw_states.values():
        if not isinstance(entry, Mapping) or "state" not in entry:
            raise ValueError("Each state_results entry must record its state label.")
        state_label = str(entry["state"])
        if state_label in state_map:
            raise ValueError("state_results contains duplicate state labels.")
        state_map[state_label] = entry
    rows, weight_rows = [], []
    for state in states:
        if state not in state_map:
            raise ValueError(f"No stored center weights for state {state!r}.")
        entry = state_map[state]
        row = {
            "state": state,
            "status": str(entry.get("status", "unknown")),
            "drawn": False,
        }
        supported = True
        for suffix in ("0", "1"):
            positions = np.asarray(entry.get(f"obs_positions{suffix}", []))
            weights = np.asarray(entry.get(f"center_weights{suffix}", []), dtype=float)
            if positions.size == 0 and weights.size == 0:
                supported = False
                row[f"x{suffix}"], row[f"y{suffix}"] = np.nan, np.nan
                continue
            if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
                raise ValueError(
                    "Stored center observation positions must be one-dimensional integers."
                )
            if (
                weights.shape != positions.shape
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1)
            ):
                raise ValueError(
                    "Stored center weights must be nonnegative, finite, normalized and match observation positions."
                )
            if (
                np.any(positions < 0)
                or np.any(positions >= adata.n_obs)
                or len(np.unique(positions)) != len(positions)
            ):
                raise ValueError("Stored center positions must be valid and unique.")
            expected = cells.loc[
                cells.state.eq(state)
                & cells.condition.eq(str(bundle["params"].get(f"group{suffix}"))),
                "obs_position",
            ]
            if not set(positions).issubset(set(expected)):
                raise ValueError(
                    "Stored center positions do not match their state and condition."
                )
            if not np.isfinite(coordinates[positions]).all():
                raise ValueError(
                    "Stored center positions contain nonfinite display coordinates."
                )
            center = weights @ coordinates[positions]
            row[f"x{suffix}"], row[f"y{suffix}"] = float(center[0]), float(center[1])
            for position, weight in zip(positions, weights):
                weight_rows.append(
                    {
                        "state": state,
                        "condition": str(bundle["params"].get(f"group{suffix}")),
                        "obs_position": int(position),
                        "weight": float(weight),
                    }
                )
        row["drawn"] = bool(
            supported
            and row["status"] not in {"zero_shift", "insufficient_cells"}
            and np.hypot(row["x1"] - row["x0"], row["y1"] - row["y0"]) > 1e-12
        )
        rows.append(row)
    centers = pd.DataFrame(rows)
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a Matplotlib Axes.")
    with style_context(style):
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        artists = {}
        for category in categories:
            selected = display[cells.color_label.iloc[display].eq(category).to_numpy()]
            artists[f"cells:{category}"] = ax.scatter(
                xy[selected, 0],
                xy[selected, 1],
                color=colors[category],
                s=size,
                alpha=alpha,
                linewidths=0,
                label=category,
                rasterized=True,
            )
        for row in centers.itertuples(index=False):
            if row.drawn:
                arrow = FancyArrowPatch(
                    (row.x0, row.y0),
                    (row.x1, row.y1),
                    connectionstyle=f"arc3,rad={float(curvature)}",
                    arrowstyle="-|>",
                    color="0.15",
                    lw=1.7,
                    mutation_scale=13,
                    zorder=5,
                )
                ax.add_patch(arrow)
                artists[f"arrow:{row.state}"] = arrow
                ax.scatter(
                    [row.x0, row.x1],
                    [row.y0, row.y1],
                    s=40,
                    c=["white", "0.15"],
                    edgecolor="0.15",
                    linewidth=1.2,
                    zorder=6,
                )
                ax.annotate(
                    row.state,
                    ((row.x0 + row.x1) / 2, (row.y0 + row.y1) / 2),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=9,
                )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"{display_key.removeprefix('X_')} 1")
        ax.set_ylabel(f"{display_key.removeprefix('X_')} 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)
        ax.text(
            0.02,
            0.02,
            "Schematic condition contrasts; curves are not trajectories",
            transform=ax.transAxes,
            fontsize=8,
            color="0.3",
            va="bottom",
        )
        if created:
            ax.set_title(
                title
                or f"{bundle['params'].get('group0')} → {bundle['params'].get('group1')}"
            )
        result = PlotResult(
            fig,
            {"embedding": ax},
            data={
                "cells": cells,
                "centers": centers,
                "center_weights": pd.DataFrame(weight_rows),
            },
            metadata={
                "plot": "shift_embedding",
                "explanation_key": key,
                "states": states,
                "analysis_rep": bundle["params"].get("rep"),
                "display_basis": display_key,
                "curvature": float(curvature),
                "arrow_semantics": "schematic condition-center contrast",
                "center_method": "frozen analysis cell weights applied to display coordinates",
                "not_a_trajectory": True,
                "max_points": max_points,
                "n_drawn_cells": len(display),
                "sample_key": bundle["params"].get("sample_key"),
                "color": color,
                "category_colors": colors,
            },
            warnings=[
                "Curvature is a display choice, not an estimated path, fate, or velocity field."
            ],
            artists=artists,
        )
    return _finish(result, return_result, show, save, "scgeo_shift_embedding")
