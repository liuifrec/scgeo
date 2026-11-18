"""Compose existing ScGeo views into a publication-ready AnnData atlas.

The atlas neither estimates dynamics nor refits geometry. A display embedding is
used for coordinates; numerical evidence is accepted only as a stored report.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ._result import PlotResult
from ._style import category_palette, style_context
from ._utils import display_indices, resolve_embedding

_PANEL_ORDER = ("states", "velocity", "paga", "evidence")
_CONFLICTING_VELOCITY = {
    "adata", "ax", "figsize", "basis", "velocity_basis", "condition_key", "groups",
    "return_result", "return_data", "show", "legend", "legend_loc", "xlim",
    "ylim", "aspect", "show_colorbar", "background_key", "palette", "max_points",
    "random_state", "title",
}
_CONFLICTING_PAGA = {
    "adata", "ax", "figsize", "basis", "node_key", "condition_key", "group0",
    "group1", "return_result", "return_data", "show", "legend", "legend_loc",
    "show_colorbar", "xlim", "ylim", "aspect", "palette", "max_points",
    "random_state", "title", "highlight_nodes", "focus_states", "edge_focus",
}


def _options(name, options, forbidden):
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"{name} must be a mapping of {name.split('_')[0]} options.")
    overlap = set(options) & forbidden
    if overlap:
        raise ValueError(f"{name} cannot override atlas-managed options: {sorted(overlap)}")
    return dict(options)


def _get_categories(values):
    """Preserve actual missing values separately from a literal 'NA' label."""
    raw = pd.Series(values).reset_index(drop=True)
    existing = [str(value) for value in pd.unique(raw.dropna())]
    if len(set(existing)) != len(existing):
        raise ValueError("Distinct categorical values collide after string conversion.")
    missing = "(missing)"
    while missing in existing:
        missing = f"({missing})"
    labels = raw.map(lambda x: missing if pd.isna(x) else str(x))
    if raw.isna().any():
        existing.append(missing)
    return labels, existing, missing if raw.isna().any() else None


def _prepare_evidence(report, effect_metric, report_state_key, max_states, highlights):
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame returned by ScGeo or a compatible state table.")
    work = report.copy(deep=True)
    if work.empty:
        raise ValueError("report must contain at least one state.")
    state_col = report_state_key
    if state_col is None:
        state_col = next((name for name in ("state", "node") if name in work), None)
    if state_col is None:
        if work.index.name in {"state", "node"}:
            work = work.reset_index()
            state_col = work.columns[0]
        else:
            raise ValueError("report requires a 'state'/'node' column or report_state_key.")
    if state_col not in work:
        raise KeyError(f"report_state_key {state_col!r} is missing from report.")
    if work[state_col].isna().any():
        raise ValueError("report state labels cannot be missing.")
    work["_state_id"] = work[state_col].astype(str)
    if work._state_id.duplicated().any():
        raise ValueError("report contains duplicate or string-colliding states.")
    if effect_metric == "auto":
        metric = next((m for m in ("delta_norm", "normalized_delta_norm")
                       if m in work and pd.to_numeric(work[m], errors="coerce").notna().any()), None)
        if metric is None:
            metric = next((m for m in ("delta_norm", "normalized_delta_norm") if m in work), None)
    else:
        metric = effect_metric
    if metric not in {"delta_norm", "normalized_delta_norm"}:
        raise ValueError("effect_metric must select delta_norm or normalized_delta_norm.")
    if metric not in work:
        raise KeyError(f"report lacks selected effect_metric {metric!r}.")
    work["effect"] = pd.to_numeric(work[metric], errors="raise").astype(float)
    if np.isinf(work.effect).any() or work.effect.dropna().lt(0).any():
        raise ValueError("Report displacement magnitudes must be nonnegative or missing.")
    if "effect_status" in work:
        work["assessed"] = work.effect_status.eq("assessed") & work.effect.notna()
    else:
        work["assessed"] = work.effect.notna()
    if "coverage_status" in work:
        ineligible = work.coverage_status.isin(("insufficient_coverage", "unavailable", "not_computed"))
        work.loc[ineligible, "assessed"] = False
    work["status_label"] = np.where(work.assessed, "assessed", "limited / unavailable")
    ci_keys = {
        "delta_norm": ("magnitude_ci95_low", "magnitude_ci95_high"),
        "normalized_delta_norm": ("normalized_magnitude_ci95_low", "normalized_magnitude_ci95_high"),
    }[metric]
    work["ci_low"], work["ci_high"] = np.nan, np.nan
    if all(key in work for key in ci_keys):
        low = pd.to_numeric(work[ci_keys[0]], errors="raise").astype(float)
        high = pd.to_numeric(work[ci_keys[1]], errors="raise").astype(float)
        if (np.isinf(low) | np.isinf(high)).any():
            raise ValueError("Report CI endpoints must be finite or missing.")
        if (low.notna() ^ high.notna()).any() or (low > high).any():
            raise ValueError("Report CI endpoints must be paired and increasing.")
        work["ci_low"], work["ci_high"] = low, high
    elif any(key in work for key in ci_keys):
        raise ValueError("Both endpoints of a matching effect CI must be supplied.")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer.")
    order = work.assign(_highlight=work._state_id.isin(highlights)).sort_values(
        ["_highlight", "assessed", "effect"], ascending=[False, False, False], kind="stable",
        na_position="last",
    )
    return order, order.head(max_states).copy(), metric


def geometry_dynamics_atlas(
    adata,
    *,
    basis="umap",
    color=None,
    node_key=None,
    condition_key=None,
    group0=None,
    group1=None,
    sample_key=None,
    panels="auto",
    velocity_mode="stream",
    velocity_options=None,
    paga_options=None,
    report=None,
    effect_metric="auto",
    report_state_key=None,
    max_states=12,
    highlight_states=None,
    palette=None,
    max_points=6000,
    random_state=0,
    label_states=True,
    style="paper",
    figsize=None,
    title=None,
    return_result=True,
    show=True,
):
    """Build coordinated states, projected-velocity, PAGA, and evidence panels.

    A standard AnnData embedding is enough for a one-panel atlas. With
    ``panels='auto'``, optional panels are included *only* when inputs exist:
    a matching ``velocity_<basis>`` projection, a PAGA graph plus a specified
    two-condition contrast, or an explicitly supplied stored report. Missing
    optional panels are recorded; an explicitly requested unavailable panel
    raises instead. Allowed panels are ``'states'``, ``'velocity'``, ``'paga'``
    (alias ``'geometry'``), and ``'evidence'``.

    The states panel reads ``adata.obs[color]`` (categorical or numeric).
    ``node_key`` identifies states used for PAGA and highlighting; it defaults
    to ``color`` when that is categorical. ``condition_key`` can infer the
    contrast only when precisely two nonmissing groups are observed. When
    samples are available, ``sample_key`` adds descriptive counts to metadata
    without inventing statistical tests. ``max_points`` affects drawing only.

    The velocity and PAGA panels delegate to existing ScGeo plotting functions
    with common embedding limits and compatible node palettes. Projected
    velocity describes the *display* embedding. PAGA edges are undirected, and
    condition arrows are display-coordinate mean/median contrasts. The evidence
    panel displays stored ``delta_norm`` or ``normalized_delta_norm`` from
    ``report`` without inferring effects or confidence intervals. Raw CIs are
    never reused on normalized effects. No results are written to AnnData.

    ``velocity_options`` and ``paga_options`` pass through supported renderer
    settings, but cannot replace shared basis, axes, condition scope, palette,
    and output controls. Use ``return_result=True`` (default) to export PNG,
    SVG, PDF and each panel's source tables via ``PlotResult.save``.
    """
    # Validate all availability and semantic rules before creating the canvas.
    key, raw = resolve_embedding(adata, basis)
    xy = np.asarray(raw[:, :2], dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        raise ValueError("No finite display coordinates are available.")
    if node_key is not None and node_key not in adata.obs:
        raise KeyError(f"node_key {node_key!r} not found in adata.obs.")
    if color is not None and color not in adata.obs:
        raise KeyError(f"color {color!r} must be an adata.obs column.")
    if condition_key is not None and condition_key not in adata.obs:
        raise KeyError(f"condition_key {condition_key!r} not found in adata.obs.")
    if sample_key is not None and sample_key not in adata.obs:
        raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
    if style not in {"paper", "presentation"}:
        raise ValueError("style must be paper or presentation.")
    if not isinstance(label_states, bool):
        raise TypeError("label_states must be boolean.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be an integer.")
    if color is None and node_key is not None:
        color = node_key
    is_numeric = bool(color is not None and pd.api.types.is_numeric_dtype(adata.obs[color])
                      and not pd.api.types.is_bool_dtype(adata.obs[color]))
    if node_key is None and color is not None and not is_numeric:
        node_key = color
    if highlight_states is None:
        highlights = set()
    else:
        if node_key is None:
            raise ValueError("highlight_states requires node_key or categorical color.")
        if isinstance(highlight_states, (str, bytes)):
            highlights = {str(highlight_states)}
        else:
            highlights = {str(x) for x in highlight_states}
        actual = set(adata.obs[node_key].dropna().astype(str))
        if not highlights <= actual:
            raise ValueError(f"Unknown highlight_states: {sorted(highlights - actual)}")
    options_v = _options("velocity_options", velocity_options, _CONFLICTING_VELOCITY)
    options_p = _options("paga_options", paga_options, _CONFLICTING_PAGA)
    if velocity_mode not in {"grid", "stream"}:
        raise ValueError("velocity_mode must be grid or stream.")

    if (group0 is None) != (group1 is None):
        raise ValueError("group0 and group1 must be supplied together.")
    groups_source = "explicit" if group0 is not None else "none"
    if condition_key is not None:
        known = sorted(str(x) for x in pd.unique(adata.obs[condition_key].dropna()))
        if len(set(known)) != len(known):
            raise ValueError("Condition labels collide after string conversion.")
        if group0 is None and len(known) == 2:
            group0, group1 = known
            groups_source = "inferred_two_conditions"
        if group0 is not None:
            group0, group1 = str(group0), str(group1)
            if group0 == group1 or {group0, group1} - set(known):
                raise ValueError("group0/group1 must be distinct observed conditions.")
    elif group0 is not None:
        raise ValueError("condition_key is required with group0/group1.")
    contrast_ready = condition_key is not None and group0 is not None
    expected_velocity = f"velocity_{key.removeprefix('X_')}"
    velocity_ready = expected_velocity in adata.obsm
    if velocity_ready and np.asarray(adata.obsm[expected_velocity]).shape != raw.shape:
        raise ValueError("Matching projected-velocity and embedding shapes differ.")
    graph_ready = bool(isinstance(adata.uns.get("paga"), Mapping)
                       and "connectivities" in adata.uns["paga"] and node_key is not None
                       and contrast_ready)
    available = {"states": True, "velocity": velocity_ready, "paga": graph_ready,
                 "evidence": report is not None}
    missing_reasons = {
        "velocity": f"matching {expected_velocity} absent from adata.obsm",
        "paga": "requires PAGA connectivities, node_key, and a two-condition contrast",
        "evidence": "no stored report explicitly supplied",
    }
    if panels == "auto":
        requested = list(_PANEL_ORDER)
        chosen = [p for p in requested if available[p]]
        omitted = {p: missing_reasons[p] for p in requested if not available[p]}
    else:
        requested = [panels] if isinstance(panels, str) else list(panels)
        if not requested:
            raise ValueError("panels must not be empty.")
        requested = ["paga" if p == "geometry" else p for p in requested]
        if any(p not in _PANEL_ORDER for p in requested) or len(set(requested)) != len(requested):
            raise ValueError(f"panels must contain distinct names from {_PANEL_ORDER}.")
        missing = [p for p in requested if not available[p]]
        if missing:
            raise ValueError("Requested panel unavailable: " + "; ".join(
                f"{p}: {missing_reasons[p]}" for p in missing))
        chosen, omitted = requested, {}
    if color is not None and not is_numeric:
        obs_colors, categories, missing_label = _get_categories(adata.obs[color])
        colors = category_palette(categories, palette)
        if missing_label is not None and palette is None:
            colors[missing_label] = "#AEB5BB"
    else:
        obs_colors, categories, missing_label = None, [], None
        colors = {}
        if palette is not None:
            raise ValueError("palette is only used for categorical color.")
    report_all, report_shown, metric = None, None, None
    if "evidence" in chosen:
        if node_key is None:
            raise ValueError("An evidence panel requires node_key to link stored states to AnnData.")
        report_all, report_shown, metric = _prepare_evidence(
            report, effect_metric, report_state_key, max_states, highlights
        )
        observed_states = set(adata.obs[node_key].dropna().astype(str))
        unknown_states = set(report_all._state_id) - observed_states
        if unknown_states:
            raise ValueError(f"report states not present in adata.obs[{node_key!r}]: {sorted(unknown_states)}")
        provenance = dict(report.attrs.get("provenance", {}) or {})
        if contrast_ready:
            for group_name, actual in (("group0", group0), ("group1", group1)):
                reported = provenance.get(group_name)
                if reported is not None and str(reported) != actual:
                    raise ValueError(
                        f"report provenance {group_name}={reported!r} disagrees with atlas {actual!r}.")
        if provenance.get("node_key") is not None and str(provenance["node_key"]) != node_key:
            raise ValueError("report provenance node_key disagrees with atlas node_key.")
    # No statistical use of cell-level counts; only display sampling is performed.
    eligible = np.flatnonzero(finite)
    if node_key is not None:
        strata = np.asarray(adata.obs[node_key], dtype=object)[eligible]
    elif obs_colors is not None:
        strata = obs_colors.iloc[eligible].to_numpy(dtype=object)
    else:
        strata = np.full(len(eligible), "all", dtype=object)
    chosen_indices = eligible[display_indices(strata, max_points=max_points,
                                               random_state=int(random_state))]
    shown_mask = np.zeros(adata.n_obs, dtype=bool)
    shown_mask[chosen_indices] = True
    if node_key is None:
        node_values = pd.Series([None] * adata.n_obs, dtype=object)
    else:
        node_values = adata.obs[node_key].reset_index(drop=True)
    cell_state = node_values.map(lambda x: None if pd.isna(x) else str(x))
    highlighted = cell_state.isin(highlights).to_numpy()
    scope_conditions = pd.Series([None] * adata.n_obs, dtype=object)
    if condition_key is not None:
        scope_conditions = adata.obs[condition_key].reset_index(drop=True).map(
            lambda x: None if pd.isna(x) else str(x))
    cells = pd.DataFrame({
        "cell_id": pd.Index(adata.obs_names).astype(str),
        "state": cell_state.to_numpy(),
        "condition": scope_conditions.to_numpy(),
        "x": xy[:, 0], "y": xy[:, 1],
        "finite": finite, "displayed": shown_mask, "highlighted": highlighted,
    })
    if sample_key is not None:
        cells["sample"] = adata.obs[sample_key].reset_index(drop=True).map(
            lambda x: None if pd.isna(x) else str(x)).to_numpy()
    if color is not None:
        cells["color_value"] = (pd.to_numeric(adata.obs[color], errors="raise").to_numpy()
                                if is_numeric else obs_colors.to_numpy())
    span = xy[finite].max(axis=0) - xy[finite].min(axis=0)
    padding = np.maximum(0.05 * span, 0.08)
    limits = [(float(xy[finite, i].min() - padding[i]),
               float(xy[finite, i].max() + padding[i])) for i in (0, 1)]

    n = len(chosen)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    size = figsize or ((7.8, 6.1) if n == 1 else
                       (13.8, 5.9) if n == 2 else (14.2, 10.8))
    notes = [f"{panel} omitted: {reason}" for panel, reason in omitted.items()]
    if not finite.all():
        notes.append(f"{int((~finite).sum())} cells have nonfinite display coordinates and were excluded from drawing.")
    if not contrast_ready:
        notes.append("No two-condition contrast is declared; graph shifts are not shown.")
    axes, artists, output = {}, {}, {"cells": cells}
    panel_provenance = {}
    with style_context(style):
        fig, axs = plt.subplots(nrows, ncols, figsize=size, squeeze=False,
                                constrained_layout=True)
        for empty in axs.flat[n:]:
            empty.set_visible(False)
        for index, panel in enumerate(chosen):
            ax = axs.flat[index]
            axes[panel] = ax
            if panel == "states":
                if color is None:
                    scatter = ax.scatter(xy[chosen_indices, 0], xy[chosen_indices, 1],
                                         s=7, c="#9AA5B1", alpha=0.72, linewidths=0,
                                         rasterized=len(chosen_indices) > 2000)
                elif is_numeric:
                    values = pd.to_numeric(adata.obs[color], errors="raise").to_numpy(dtype=float)
                    use = values[chosen_indices]
                    bad = ~np.isfinite(use)
                    if not bad.all():
                        vmin, vmax = np.nanmin(values[np.isfinite(values)]), np.nanmax(values[np.isfinite(values)])
                        if vmin == vmax:
                            vmax = vmin + 1
                        scatter = ax.scatter(xy[chosen_indices[~bad], 0],
                                             xy[chosen_indices[~bad], 1], s=7, c=use[~bad],
                                             vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.85,
                                             linewidths=0, rasterized=len(chosen_indices) > 2000)
                        fig.colorbar(scatter, ax=ax, shrink=0.66, pad=0.02).set_label(color, fontsize=8)
                    else:
                        scatter = ax.scatter([], [], s=7)
                    if bad.any():
                        ax.scatter(xy[chosen_indices[bad], 0], xy[chosen_indices[bad], 1],
                                   s=7, c="#BFC4CC", alpha=0.55, linewidths=0)
                else:
                    values = obs_colors.iloc[chosen_indices]
                    scatter = ax.scatter(xy[chosen_indices, 0], xy[chosen_indices, 1],
                                         s=7, c=values.map(colors).tolist(), alpha=0.83,
                                         linewidths=0, rasterized=len(chosen_indices) > 2000)
                    if len(categories) <= 12:
                        handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                                          color=colors[c], label=c) for c in categories]
                        artists["states_legend"] = ax.legend(handles=handles, loc="upper left",
                                                              bbox_to_anchor=(0.0, -0.08),
                                                              ncol=min(4, len(handles)), fontsize=7,
                                                              frameon=False, borderaxespad=0)
                artists["states_points"] = scatter
                if highlighted.any():
                    positions = chosen_indices[highlighted[chosen_indices]]
                    artists["states_highlight"] = ax.scatter(
                        xy[positions, 0], xy[positions, 1], facecolors="none",
                        edgecolors="#152B40", linewidths=0.55, s=16, alpha=0.9)
                if label_states and node_key is not None:
                    states = list(pd.unique(cell_state[finite].dropna()))
                    if len(states) <= 14:
                        for state in states:
                            pts = xy[(cell_state == state).to_numpy() & finite]
                            if len(pts):
                                center = np.median(pts, axis=0)
                                ax.annotate(state, center, ha="center", va="center",
                                            fontsize=8, weight="bold" if state in highlights else "normal",
                                            color="#24384B", bbox={"fc": "white", "ec": "none", "alpha": .78, "pad": .8})
                ax.set_title("Cell-state landscape" if color is None else f"Cell-state landscape · {color}")
                panel_provenance[panel] = {"color_key": color, "color_type": "numeric" if is_numeric else
                                           "categorical" if color else "none", "drawn": len(chosen_indices),
                                           "palette": colors, "label_states": label_states}
            elif panel == "velocity":
                from ._velocity_field import velocity_field
                velocity_kwargs = {
                    "background_size": 4.5, "background_alpha": .12,
                    **options_v,
                }
                local = velocity_field(
                    adata, basis=basis, condition_key=condition_key if contrast_ready else None,
                    groups=[group0, group1] if contrast_ready else None,
                    background_key=color if color is not None and not is_numeric else None,
                    palette=colors if colors else None, mode=velocity_mode,
                    ax=ax, legend=False, show_colorbar=False, max_points=max_points,
                    random_state=int(random_state), xlim=limits[0], ylim=limits[1],
                    return_result=True, show=False, **velocity_kwargs,
                )
                output.update({f"velocity_{k}": v for k, v in local.data.items()})
                artists["velocity"] = local.artists
                panel_provenance[panel] = local.metadata
                notes.extend(local.warnings)
            elif panel == "paga":
                from ._paga_shift_map import paga_shift_map
                # If color != node_key, use the node-key palette rather than
                # pretending the different categorical legends are equivalent.
                node_labels = (adata.obs[node_key].cat.categories.astype(str).tolist()
                               if isinstance(adata.obs[node_key].dtype, pd.CategoricalDtype)
                               else adata.obs[node_key].dropna().astype(str).unique().tolist())
                node_palette = category_palette(node_labels)
                if color == node_key and colors:
                    node_palette.update({state: col for state, col in colors.items()
                                         if state in node_palette})
                paga_kwargs = {"label_top_n": 8 if not highlights else 0,
                               "background": "contrast", **options_p}
                local = paga_shift_map(
                    adata, node_key=node_key, condition_key=condition_key,
                    group0=group0, group1=group1, basis=basis, palette=node_palette,
                    ax=ax, xlim=limits[0], ylim=limits[1], max_points=max_points,
                    random_state=int(random_state), focus_states=sorted(highlights) if highlights else None,
                    edge_focus="incident" if highlights else "all",
                    show=False, legend=False, show_colorbar=False, return_result=True,
                    **paga_kwargs,
                )
                output.update({f"paga_{k}": v for k, v in local.data.items() if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))})
                artists["paga"] = local.artists
                panel_provenance[panel] = local.metadata
                notes.extend(local.warnings)
            else:
                displayed = report_shown.iloc[::-1]
                y = np.arange(len(displayed))
                assessed = displayed.assessed.to_numpy(dtype=bool)
                point = displayed.effect.to_numpy(dtype=float)
                ok = assessed & np.isfinite(point)
                low = displayed.ci_low.to_numpy(dtype=float)
                high = displayed.ci_high.to_numpy(dtype=float)
                for row_idx in np.flatnonzero(ok & np.isfinite(low) & np.isfinite(high)):
                    ax.plot([low[row_idx], high[row_idx]], [y[row_idx], y[row_idx]],
                            color="#5A7B93", linewidth=1.5, zorder=2)
                assessed_colors = (
                    [colors.get(k, "#247C88") for k in displayed._state_id[ok]]
                    if color == node_key and colors else "#247C88"
                )
                ax.scatter(point[ok], y[ok], s=36, c=assessed_colors, zorder=3,
                           label="Assessed estimate")
                limited = ~assessed
                if limited.any():
                    ax.scatter(point[limited & np.isfinite(point)],
                               y[limited & np.isfinite(point)], s=36,
                               facecolors="none", edgecolors="#A45D5B", marker="o",
                               label="Limited / unavailable")
                ax.set_yticks(y, displayed._state_id)
                ax.set_xscale("linear")
                ax.axvline(0, linewidth=.7, color="#B3BAC1", zorder=0)
                ax.grid(axis="x", alpha=.15)
                ax.set_xlabel("Raw displacement" if metric == "delta_norm" else "Normalized displacement")
                ax.set_title("Stored geometry evidence (not UMAP-derived)")
                if limited.any():
                    ax.legend(frameon=False, fontsize=7, loc="lower right")
                output["evidence_all"] = report_all
                output["evidence_displayed"] = report_shown
                panel_provenance[panel] = {"metric": metric, "report_state_key": report_state_key,
                                           "assessed_displayed": int(assessed.sum()),
                                           "total_report_states": len(report_all),
                                           "max_states": max_states,
                                           "intervals": "matching stored endpoints only"}
            if panel != "evidence":
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])
                ax.set_aspect("equal", adjustable="box")
                label = key.removeprefix("X_").upper()
                ax.set_xlabel(f"{label} 1")
                ax.set_ylabel(f"{label} 2")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        if title:
            fig.suptitle(title, fontsize=14 if style == "paper" else 19)
    sample_counts = {}
    if sample_key is not None:
        if condition_key is None:
            sample_counts["all"] = int(adata.obs[sample_key].nunique(dropna=True))
        else:
            source = adata.obs[[condition_key, sample_key]]
            sample_counts = {str(k): int(v) for k, v in source.groupby(condition_key, observed=True)[sample_key]
                             .nunique(dropna=True).items()}
    metadata = {
        "plot": "geometry_dynamics_atlas",
        "requested_panels": requested,
        "panels": chosen,
        "omitted_panels": omitted,
        "display_basis": key,
        "velocity_basis": expected_velocity if "velocity" in chosen else None,
        "analysis_representation": None if report is None else dict(report.attrs.get("provenance", {})).get("effect_representation", dict(report.attrs.get("provenance", {})).get("rep")),
        "condition_key": condition_key, "group0": group0, "group1": group1,
        "condition_choice": groups_source, "node_key": node_key,
        "sample_key": sample_key, "unique_samples_by_condition": sample_counts,
        "cell_count": int(adata.n_obs), "finite_coordinates": int(finite.sum()),
        "drawn_state_cells": int(shown_mask.sum()),
        "max_points": max_points, "random_state": int(random_state),
        "highlight_states": sorted(highlights), "panel_provenance": panel_provenance,
        "display_is_inference": False,
        "effect_source": "explicit stored report" if report is not None else "unavailable",
        "report_provenance": dict(report.attrs.get("provenance", {}) or {}) if report is not None else None,
        "velocity_interpretation": "descriptive already projected vectors; not trajectories",
        "paga_interpretation": "undirected connectivity; not transitions",
        "figure_style": style,
    }
    result = PlotResult(fig=fig, axes=axes, data=output, metadata=metadata,
                        warnings=notes, artists=artists)
    if show:
        plt.show()
    return result if return_result else (fig, axes)
