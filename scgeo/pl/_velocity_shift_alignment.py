from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    xy = np.asarray(adata.obsm[key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (n_cells, >=2).")
    return xy[:, :2]


def _as_str_series(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.astype(str)
    return s.astype("category").astype(str)


def _compute_node_centroids(
    adata,
    *,
    node_key: str,
    condition_key: Optional[str],
    group0: Optional[Any],
    group1: Optional[Any],
    basis: str,
    min_cells: int = 10,
    agg: str = "mean",
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")

    xy = _get_basis_xy(adata, basis)
    df = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            node_key: _as_str_series(adata.obs[node_key]),
        },
        index=adata.obs_names,
    )

    if condition_key is None:
        grouped = df.groupby(node_key, observed=False)[["x", "y"]]
        if agg == "median":
            cent = grouped.median().reset_index()
        elif agg == "mean":
            cent = grouped.mean().reset_index()
        else:
            raise ValueError("agg must be one of {'mean', 'median'}.")
        counts = df.groupby(node_key, observed=False).size().rename("n").reset_index()
        cent = cent.merge(counts, on=node_key, how="left")
        cent["present"] = cent["n"].fillna(0).astype(int) >= int(min_cells)
        return cent

    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' not found in adata.obs.")
    if group0 is None or group1 is None:
        raise ValueError(
            "group0 and group1 must be provided when condition_key is used."
        )

    g0 = str(group0)
    g1 = str(group1)
    df[condition_key] = adata.obs[condition_key].astype(str).values
    df = df[df[condition_key].isin([g0, g1])].copy()
    if df.empty:
        raise ValueError(f"No cells found for groups {g0!r} and {g1!r}.")

    grouped = df.groupby([node_key, condition_key], observed=False)[["x", "y"]]
    if agg == "median":
        cent = grouped.median().reset_index()
    elif agg == "mean":
        cent = grouped.mean().reset_index()
    else:
        raise ValueError("agg must be one of {'mean', 'median'}.")

    counts = (
        df.groupby([node_key, condition_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cent = cent.merge(counts, on=[node_key, condition_key], how="left")

    c0 = cent[cent[condition_key] == g0].rename(
        columns={"x": "x0", "y": "y0", "n_cells": "n0"}
    )[[node_key, "x0", "y0", "n0"]]
    c1 = cent[cent[condition_key] == g1].rename(
        columns={"x": "x1", "y": "y1", "n_cells": "n1"}
    )[[node_key, "x1", "y1", "n1"]]

    out = c0.merge(c1, on=node_key, how="outer")
    out["present0"] = out["n0"].fillna(0).astype(int) >= int(min_cells)
    out["present1"] = out["n1"].fillna(0).astype(int) >= int(min_cells)
    out["dx"] = out["x1"] - out["x0"]
    out["dy"] = out["y1"] - out["y0"]
    out["shift_norm"] = np.sqrt(np.square(out["dx"]) + np.square(out["dy"]))
    return out


def _compute_velocity_vectors(
    adata,
    *,
    node_key: str,
    basis: str,
    velocity_basis: Optional[str] = None,
    min_cells: int = 10,
    agg: str = "mean",
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")

    xy = _get_basis_xy(adata, basis)

    if velocity_basis is None:
        key = (
            f"velocity_{basis}"
            if not basis.startswith("X_")
            else f"velocity_{basis[2:]}"
        )
    else:
        key = (
            velocity_basis
            if velocity_basis.startswith("velocity_")
            else f"velocity_{velocity_basis}"
        )

    if key not in adata.obsm:
        raise KeyError(f"Velocity embedding '{key}' not found in adata.obsm.")

    vel = np.asarray(adata.obsm[key])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity embedding '{key}' must have shape (n_cells, >=2).")

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            "vx": vel[:, 0],
            "vy": vel[:, 1],
            "x": xy[:, 0],
            "y": xy[:, 1],
        },
        index=adata.obs_names,
    )

    grouped = df.groupby(node_key, observed=False)
    if agg == "median":
        out = grouped[["vx", "vy", "x", "y"]].median().reset_index()
    elif agg == "mean":
        out = grouped[["vx", "vy", "x", "y"]].mean().reset_index()
    else:
        raise ValueError("agg must be one of {'mean', 'median'}.")

    counts = grouped.size().rename("n").reset_index()
    out = out.merge(counts, on=node_key, how="left")
    out["present"] = out["n"].fillna(0).astype(int) >= int(min_cells)
    out["vel_norm"] = np.sqrt(np.square(out["vx"]) + np.square(out["vy"]))
    return out


def _safe_cosine(
    dx: np.ndarray, dy: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    dot = dx * vx + dy * vy
    dn = np.sqrt(dx * dx + dy * dy)
    vn = np.sqrt(vx * vx + vy * vy)
    denom = dn * vn
    out = np.full_like(dot, np.nan, dtype=float)
    mask = np.isfinite(denom) & (denom > 0)
    out[mask] = dot[mask] / denom[mask]
    return out


def _alignment_class(
    vals: pd.Series,
    *,
    pos_thr: float = 0.3,
    neg_thr: float = -0.3,
) -> pd.Series:
    out = pd.Series(index=vals.index, dtype="object")
    out[:] = "neutral"
    out[vals > pos_thr] = "aligned"
    out[vals < neg_thr] = "discordant"
    out[vals.isna()] = "missing"
    return out


def _get_palette(adata, groupby: str, categories: list[str]) -> dict[str, Any]:
    colors_key = f"{groupby}_colors"
    if colors_key in adata.uns:
        colors = list(adata.uns[colors_key])
        if isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
            levels = adata.obs[groupby].cat.categories.astype(str).tolist()
            if len(colors) >= len(levels):
                mapping = dict(zip(levels, colors))
                if all(category in mapping for category in categories):
                    return {category: mapping[category] for category in categories}
        elif len(colors) >= len(categories):
            return {cat: colors[i] for i, cat in enumerate(categories)}
    from ._style import category_palette

    return category_palette(categories)


def velocity_shift_alignment(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    velocity_basis: Optional[str] = None,
    min_cells: int = 15,
    agg: str = "mean",
    bg_size: float = 6.0,
    bg_alpha: float = 0.10,
    node_size: float = 180.0,
    shift_scale: float = 1.0,
    velocity_scale: float = 50.0,
    shift_color: str = "black",
    velocity_color: str = "cyan",
    shift_alpha: float = 0.95,
    velocity_alpha: float = 0.95,
    arrow_width: float = 0.006,
    show_shift_arrow: bool = True,
    show_velocity_arrow: bool = True,
    color_by_alignment: bool = True,
    alignment_cmap: str = "coolwarm",
    alignment_pos_thr: float = 0.3,
    alignment_neg_thr: float = -0.3,
    palette: Optional[dict[str, Any]] = None,
    label: bool = True,
    label_top_n: Optional[int] = None,
    label_mode: str = "shift",
    label_fontsize: int = 8,
    title: Optional[str] = None,
    ax=None,
    figsize: tuple[float, float] = (8.2, 7.0),
    return_data: bool = False,
    return_result: bool = False,
    mode: str = "display",
    store_key: str = "representation_stability",
    analysis_rep: Optional[str] = None,
    alignment_table: Optional[pd.DataFrame] = None,
    axes: Optional[Mapping[str, Any]] = None,
    state_order=None,
    show: bool = True,
):
    """
    Plot display-space vectors or stored quantitative alignment with an overlay.

    ``mode="display"`` preserves the legacy calculation in the first two display
    coordinates and pools projected velocity over all observations in each state.
    ``mode="stored"`` renders stored alignment values on a separate
    axis and colors a display-only overlay by those fixed values. It reads
    ``adata.uns['scgeo'][store_key]['per_rep_state']`` by default, or an explicit
    ``alignment_table``. A representation must be selected with ``analysis_rep``
    when multiple representations exist or the table has no representation
    metadata. Stored contrast metadata is checked against the requested groups.

    Stored mode accepts an ``axes`` mapping with ``overlay`` and ``alignment``
    and returns ``(fig, (overlay_ax, alignment_ax)[, data])``. ``return_result``
    is an opt-in consistent result for either mode. High-dimensional analysis
    vectors are never used as display arrows; missing or incompatible projected
    velocity leaves a condition-contrast overlay with an explanatory note. In
    stored mode, projected velocity is aggregated only over the selected two
    conditions, matching the contrast scope.

    Node fill encodes cosine alignment between shift and velocity:
      +1  aligned
       0  unrelated
      -1  discordant

    Notes
    -----
    `velocity_scale` is a display-only scaling factor. Velocity vectors are often
    much smaller than condition-shift vectors in embedding coordinates, so a larger
    default is used to keep the velocity arrows visible.

    Returns
    -------
    If return_data is False:
        (fig, ax) or ax
    If return_data is True:
        (fig, ax, align_df)
    """
    from ._result import PlotResult

    if mode not in {"display", "stored"}:
        raise ValueError("mode must be 'display' or 'stored'.")
    if mode == "stored":
        if ax is not None:
            raise ValueError(
                "Stored mode requires axes={'overlay': ..., 'alignment': ...}, not ax."
            )
        return _stored_alignment_plot(
            adata,
            node_key=node_key,
            condition_key=condition_key,
            group0=group0,
            group1=group1,
            basis=basis,
            velocity_basis=velocity_basis,
            min_cells=min_cells,
            agg=agg,
            bg_size=bg_size,
            bg_alpha=bg_alpha,
            node_size=node_size,
            shift_scale=shift_scale,
            velocity_scale=velocity_scale,
            shift_color=shift_color,
            velocity_color=velocity_color,
            shift_alpha=shift_alpha,
            velocity_alpha=velocity_alpha,
            arrow_width=arrow_width,
            show_shift_arrow=show_shift_arrow,
            show_velocity_arrow=show_velocity_arrow,
            alignment_cmap=alignment_cmap,
            color_by_alignment=color_by_alignment,
            palette=palette,
            label=label,
            label_top_n=label_top_n,
            label_mode=label_mode,
            label_fontsize=label_fontsize,
            title=title,
            figsize=figsize,
            return_data=return_data,
            return_result=return_result,
            store_key=store_key,
            analysis_rep=analysis_rep,
            alignment_table=alignment_table,
            axes=axes,
            state_order=state_order,
            show=show,
        )
    if axes is not None:
        if ax is not None or not isinstance(axes, Mapping) or "overlay" not in axes:
            raise ValueError("Display mode accepts ax or axes={'overlay': ax}.")
        ax = axes["overlay"]
    xy = _get_basis_xy(adata, basis)

    shift_df = _compute_node_centroids(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        min_cells=min_cells,
        agg=agg,
    )

    vel_df = _compute_velocity_vectors(
        adata,
        node_key=node_key,
        basis=basis,
        velocity_basis=velocity_basis,
        min_cells=min_cells,
        agg=agg,
    )

    align = shift_df.merge(
        vel_df[[node_key, "x", "y", "vx", "vy", "vel_norm", "present"]],
        on=node_key,
        how="left",
        suffixes=("", "_vel"),
    )
    align["velocity_present"] = align["present"].fillna(False).astype(bool)
    align["alignment_cosine"] = _safe_cosine(
        align["dx"].to_numpy(dtype=float),
        align["dy"].to_numpy(dtype=float),
        align["vx"].to_numpy(dtype=float),
        align["vy"].to_numpy(dtype=float),
    )
    align["abs_alignment_cosine"] = np.abs(align["alignment_cosine"])
    align["alignment_class"] = _alignment_class(
        align["alignment_cosine"],
        pos_thr=alignment_pos_thr,
        neg_thr=alignment_neg_thr,
    )

    align["usable"] = (
        align["present0"].fillna(False).astype(bool)
        & align["present1"].fillna(False).astype(bool)
        & align["velocity_present"]
    )

    if state_order is not None:
        order = [str(x) for x in state_order]
        if (
            not order
            or len(set(order)) != len(order)
            or not set(order) <= set(align[node_key].astype(str))
        ):
            raise ValueError("state_order must contain unique known state labels.")
        align = (
            align.assign(**{node_key: align[node_key].astype(str)})
            .set_index(node_key)
            .loc[order]
            .reset_index()
        )
    node_names = [str(x) for x in pd.unique(_as_str_series(adata.obs[node_key]))]
    if palette is None:
        palette = _get_palette(adata, node_key, node_names)

    made_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        made_fig = True
    else:
        fig = ax.figure

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=bg_size,
        c="lightgrey",
        alpha=bg_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    usable = align[align["usable"]].copy()

    if label_mode == "shift":
        ranking = usable.sort_values("shift_norm", ascending=False)
    elif label_mode == "discordant":
        ranking = usable.sort_values("alignment_cosine", ascending=True)
    elif label_mode == "aligned":
        ranking = usable.sort_values("alignment_cosine", ascending=False)
    else:
        raise ValueError(
            "label_mode must be one of {'shift', 'discordant', 'aligned'}."
        )

    if label_top_n is not None and not usable.empty:
        label_nodes = set(ranking[node_key].head(label_top_n))
    else:
        label_nodes = set(usable[node_key].tolist())

    if color_by_alignment and not usable.empty:
        cmap_obj = plt.get_cmap(alignment_cmap)
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    else:
        cmap_obj = None
        norm = None

    for _, row in usable.iterrows():
        n = str(row[node_key])

        x0 = float(row["x0"])
        y0 = float(row["y0"])
        dx = float(row["dx"]) * shift_scale
        dy = float(row["dy"]) * shift_scale
        vx = float(row["vx"]) * velocity_scale
        vy = float(row["vy"]) * velocity_scale

        if color_by_alignment and np.isfinite(row["alignment_cosine"]):
            base_color = cmap_obj(norm(float(row["alignment_cosine"])))
        else:
            base_color = palette.get(n, "tab:gray")

        ax.scatter(
            x0,
            y0,
            s=node_size,
            color=base_color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.98,
            zorder=3,
        )

        if show_shift_arrow:
            ax.arrow(
                x0,
                y0,
                dx,
                dy,
                width=arrow_width,
                head_width=arrow_width * 8.0,
                head_length=max(np.hypot(dx, dy) * 0.12, arrow_width * 8.0),
                length_includes_head=True,
                color=shift_color,
                alpha=shift_alpha,
                zorder=4,
            )

        if show_velocity_arrow:
            ax.arrow(
                x0,
                y0,
                vx,
                vy,
                width=arrow_width * 0.55,
                head_width=arrow_width * 5.5,
                head_length=max(np.hypot(vx, vy) * 0.12, arrow_width * 5.5),
                length_includes_head=True,
                color=velocity_color,
                alpha=velocity_alpha,
                zorder=5,
            )

        if label and n in label_nodes:
            ax.text(
                x0,
                y0,
                n,
                fontsize=label_fontsize,
                ha="center",
                va="center",
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
            )

    if color_by_alignment and not usable.empty:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Display-space cosine(contrast, velocity)")

    _vector_legend(
        ax, shift_color, velocity_color, show_shift_arrow, show_velocity_arrow
    )

    basis_name = basis.removeprefix("X_")
    ax.set_xlabel(f"{basis_name.upper()}1")
    ax.set_ylabel(f"{basis_name.upper()}2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f"ScGeo: display-space alignment ({group0} → {group1})"
    ax.set_title(title)

    if show:
        plt.show()

    if return_result:
        return PlotResult(
            fig=fig,
            axes={"overlay": ax},
            data=align.copy(),
            metadata={
                "plot": "velocity_shift_alignment",
                "mode": "display",
                "display_basis": basis,
                "velocity_basis": velocity_basis,
                "group0": str(group0),
                "group1": str(group1),
                "node_key": node_key,
                "alignment_space": "first two display coordinates",
                "projected_velocity_scope": "all_conditions",
                "aggregation": agg,
                "min_cells": min_cells,
                "shift_scale": shift_scale,
                "velocity_scale": velocity_scale,
                "shift_color": shift_color,
                "velocity_color": velocity_color,
                "show_shift_arrow": show_shift_arrow,
                "show_velocity_arrow": show_velocity_arrow,
                "alignment_cmap": plt.get_cmap(alignment_cmap).name,
                "alignment_thresholds": [alignment_neg_thr, alignment_pos_thr],
            },
        )
    if return_data:
        return fig, ax, align
    if made_fig:
        return fig, ax
    return ax


def _vector_legend(ax, shift_color, velocity_color, show_shift, show_velocity):
    from matplotlib.lines import Line2D

    handles = []
    if show_shift:
        handles.append(
            Line2D(
                [],
                [],
                color=shift_color,
                marker=">",
                markersize=6,
                label="Condition contrast (display)",
            )
        )
    if show_velocity:
        handles.append(
            Line2D(
                [],
                [],
                color=velocity_color,
                marker=">",
                markersize=6,
                label="Mean projected velocity",
            )
        )
    if handles:
        ax.legend(
            handles=handles, loc="lower left", frameon=True, framealpha=0.85, fontsize=8
        )


def _stored_alignment_table(
    adata,
    *,
    store_key,
    alignment_table,
    analysis_rep,
    node_key,
    condition_key,
    group0,
    group1,
    state_order,
):
    params = {}
    if alignment_table is not None:
        if not isinstance(alignment_table, pd.DataFrame):
            raise TypeError("alignment_table must be a pandas DataFrame.")
        table = alignment_table.copy(deep=True)
        params = dict(table.attrs.get("params", {}))
    else:
        namespace = adata.uns.get("scgeo", {})
        stored = namespace.get(store_key) if isinstance(namespace, dict) else None
        if stored is None:
            stored = adata.uns.get(store_key)
        if isinstance(stored, pd.DataFrame):
            table = stored.copy(deep=True)
            params = dict(table.attrs.get("params", {}))
        elif isinstance(stored, dict) and isinstance(
            stored.get("per_rep_state"), pd.DataFrame
        ):
            table = stored["per_rep_state"].copy(deep=True)
            params = dict(stored.get("params", {}))
        else:
            raise KeyError(
                f"No stored alignment table under {store_key!r}; run representation_stability with velocity_keys or supply alignment_table."
            )
    for key, expected in {
        "node_key": node_key,
        "condition_key": condition_key,
        "group0": group0,
        "group1": group1,
    }.items():
        if (
            key in params
            and params[key] is not None
            and str(params[key]) != str(expected)
        ):
            raise ValueError(
                f"Stored alignment {key}={params[key]!r} does not match requested {expected!r}."
            )
    if "rep" in table:
        choices = table["rep"].dropna().astype(str).unique().tolist()
        if analysis_rep is None:
            if len(choices) != 1:
                raise ValueError(
                    "Select analysis_rep explicitly for a table with multiple representations."
                )
            analysis_rep = choices[0]
        resolved = str(analysis_rep)
        if resolved not in choices and f"X_{resolved}" in choices:
            resolved = f"X_{resolved}"
        if resolved not in choices:
            raise ValueError(
                f"analysis_rep {analysis_rep!r} is not in the stored representations: {choices}"
            )
        analysis_rep = resolved
        table = table.loc[table["rep"].astype(str).eq(resolved)].copy()
    else:
        declared = params.get("rep") or params.get("basis")
        if (
            analysis_rep is not None
            and declared is not None
            and str(analysis_rep).removeprefix("X_") != str(declared).removeprefix("X_")
        ):
            raise ValueError(
                "analysis_rep does not match the stored representation metadata."
            )
        analysis_rep = analysis_rep or declared
        if not isinstance(analysis_rep, str) or not analysis_rep:
            raise ValueError(
                "analysis_rep is required for a stored table without representation metadata."
            )
    source_node = next(
        (key for key in (node_key, "node", "state") if key in table), None
    )
    if source_node is None or "alignment_cosine" not in table:
        raise ValueError(
            "The stored table needs a state/node column and alignment_cosine."
        )
    if source_node != node_key:
        table = table.rename(columns={source_node: node_key})
    if table[node_key].isna().any():
        raise ValueError("Stored alignment state labels cannot be missing.")
    table[node_key] = table[node_key].astype(str)
    if table.empty or table[node_key].duplicated().any():
        raise ValueError(
            "The selected stored alignment must contain one row per state."
        )
    table["alignment_cosine"] = pd.to_numeric(table["alignment_cosine"], errors="raise")
    values = table.alignment_cosine.to_numpy(dtype=float, na_value=np.nan)
    table["alignment_cosine"] = values
    if np.any(np.isinf(values)) or np.any(
        np.abs(values[np.isfinite(values)]) > 1 + 1e-8
    ):
        raise ValueError("Stored alignment_cosine must be in [-1, 1] or missing.")
    if "usable" in table:
        if not table["usable"].dropna().isin([True, False, 0, 1]).all():
            raise ValueError("Stored usable values must be boolean.")
        table["usable"] = table["usable"].fillna(False).astype(bool)
    else:
        table["usable"] = np.isfinite(values)
    if state_order is not None:
        order = [str(x) for x in state_order]
        if (
            not order
            or len(set(order)) != len(order)
            or not set(order) <= set(table[node_key])
        ):
            raise ValueError("state_order must contain unique known state labels.")
        table = table.set_index(node_key).loc[order].reset_index()
    return table, str(analysis_rep), params


def _stored_alignment_plot(
    adata,
    *,
    node_key,
    condition_key,
    group0,
    group1,
    basis,
    velocity_basis,
    min_cells,
    agg,
    bg_size,
    bg_alpha,
    node_size,
    shift_scale,
    velocity_scale,
    shift_color,
    velocity_color,
    shift_alpha,
    velocity_alpha,
    arrow_width,
    show_shift_arrow,
    show_velocity_arrow,
    alignment_cmap,
    color_by_alignment,
    palette,
    label,
    label_top_n,
    label_mode,
    label_fontsize,
    title,
    figsize,
    return_data,
    return_result,
    store_key,
    analysis_rep,
    alignment_table,
    axes,
    state_order,
    show,
):
    from ._result import PlotResult

    if label_mode not in {"shift", "discordant", "aligned"}:
        raise ValueError(
            "label_mode must be one of {'shift', 'discordant', 'aligned'}."
        )
    if label_top_n is not None and (
        isinstance(label_top_n, bool)
        or int(label_top_n) != label_top_n
        or label_top_n < 1
    ):
        raise ValueError("label_top_n must be a positive integer or None.")
    stored, analysis_rep, params = _stored_alignment_table(
        adata,
        store_key=store_key,
        alignment_table=alignment_table,
        analysis_rep=analysis_rep,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        state_order=state_order,
    )
    xy = _get_basis_xy(adata, basis)
    display = _compute_node_centroids(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        min_cells=min_cells,
        agg=agg,
    )
    basis_name = basis.removeprefix("X_")
    expected_velocity = f"velocity_{basis_name}"
    velocity_key = (
        expected_velocity
        if velocity_basis is None
        else (
            velocity_basis
            if velocity_basis.startswith("velocity_")
            else f"velocity_{velocity_basis}"
        )
    )
    notes = []
    if not all(
        key in params for key in ("node_key", "condition_key", "group0", "group1")
    ):
        notes.append(
            "Stored contrast metadata is incomplete; condition and state labels were supplied by the caller."
        )
    projected_velocity = (
        show_velocity_arrow
        and velocity_key == expected_velocity
        and velocity_key in adata.obsm
    )
    if projected_velocity:
        condition_mask = (
            adata.obs[condition_key]
            .astype(str)
            .isin([str(group0), str(group1)])
            .to_numpy()
        )
        velocity = _compute_velocity_vectors(
            adata[condition_mask],
            node_key=node_key,
            basis=basis,
            velocity_basis=velocity_key,
            min_cells=min_cells,
            agg=agg,
        )
        display = display.merge(
            velocity[[node_key, "vx", "vy", "present"]], on=node_key, how="left"
        )
    elif show_velocity_arrow:
        notes.append(
            f"Projected velocity unavailable in {basis_name}; velocity arrows omitted. Stored alignment is unchanged."
        )
    if palette is None:
        palette = _get_palette(adata, node_key, stored[node_key].tolist())
    else:
        from ._style import category_palette

        palette = category_palette(stored[node_key].tolist(), palette=palette)
    ranking = display.merge(
        stored[[node_key, "alignment_cosine"]], on=node_key, how="inner"
    )
    rank_column = "shift_norm" if label_mode == "shift" else "alignment_cosine"
    ranking = ranking.sort_values(
        rank_column, ascending=label_mode == "discordant", kind="stable"
    )
    label_nodes = (
        set(ranking[node_key].head(label_top_n).astype(str))
        if label_top_n is not None
        else set(stored[node_key])
    )
    owned = axes is None
    if owned:
        fig, aa = plt.subplots(
            1,
            2,
            figsize=(max(figsize[0], 10), figsize[1]),
            gridspec_kw={"width_ratios": [1.5, 1.0]},
        )
        panel_axes = dict(zip(("overlay", "alignment"), aa))
    else:
        if not isinstance(axes, Mapping) or not {"overlay", "alignment"} <= set(axes):
            raise ValueError("axes must contain overlay and alignment.")
        panel_axes = {name: axes[name] for name in ("overlay", "alignment")}
        fig = panel_axes["overlay"].figure
        if (
            panel_axes["alignment"].figure is not fig
            or panel_axes["alignment"] is panel_axes["overlay"]
        ):
            raise ValueError(
                "Stored mode requires two distinct axes in the same figure."
            )
    ax, quantitative = panel_axes["overlay"], panel_axes["alignment"]
    cmap = plt.get_cmap(alignment_cmap)
    norm = plt.Normalize(-1, 1)
    finite_background = np.isfinite(xy).all(axis=1)
    positions = np.flatnonzero(finite_background)
    display_cells = pd.DataFrame(
        {
            "obs_position": positions,
            "cell": adata.obs_names[positions].astype(str),
            "state": adata.obs[node_key].iloc[positions].astype(str).to_numpy(),
            "condition": adata.obs[condition_key]
            .iloc[positions]
            .astype(str)
            .to_numpy(),
            "x": xy[positions, 0],
            "y": xy[positions, 1],
        }
    )
    background = ax.scatter(
        display_cells["x"],
        display_cells["y"],
        s=bg_size,
        c="lightgrey",
        alpha=bg_alpha,
        linewidths=0,
        rasterized=True,
    )
    lookup = stored.set_index(node_key)
    arrow_endpoints = []
    for _, row in display.iterrows():
        name = str(row[node_key])
        if (
            name not in lookup.index
            or not bool(row.present0)
            or not np.isfinite([row.x0, row.y0]).all()
        ):
            continue
        evidence = lookup.loc[name]
        cosine = float(evidence.alignment_cosine)
        usable = bool(evidence.usable) and np.isfinite(cosine)
        color = (
            (cmap(norm(cosine)) if usable else "#999999")
            if color_by_alignment
            else palette[name]
        )
        ax.scatter(
            row.x0, row.y0, s=node_size, color=color, edgecolor="white", zorder=3
        )
        if (
            show_shift_arrow
            and bool(row.present1)
            and np.isfinite([row.dx, row.dy]).all()
        ):
            arrow_endpoints.append(
                (row.x0 + row.dx * shift_scale, row.y0 + row.dy * shift_scale)
            )
            ax.annotate(
                "",
                xy=(row.x0 + row.dx * shift_scale, row.y0 + row.dy * shift_scale),
                xytext=(row.x0, row.y0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": shift_color,
                    "alpha": shift_alpha,
                    "lw": max(0.6, arrow_width * 150),
                },
                zorder=4,
            )
        if (
            projected_velocity
            and bool(row.get("present", False))
            and np.isfinite([row.vx, row.vy]).all()
        ):
            arrow_endpoints.append(
                (row.x0 + row.vx * velocity_scale, row.y0 + row.vy * velocity_scale)
            )
            ax.annotate(
                "",
                xy=(row.x0 + row.vx * velocity_scale, row.y0 + row.vy * velocity_scale),
                xytext=(row.x0, row.y0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": velocity_color,
                    "alpha": velocity_alpha,
                    "lw": max(0.6, arrow_width * 100),
                },
                zorder=5,
            )
        if label and name in label_nodes:
            ax.text(
                row.x0,
                row.y0,
                name,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=.18",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
            )
    if arrow_endpoints:
        ax.update_datalim(np.asarray(arrow_endpoints, dtype=float))
        ax.autoscale_view()
    _vector_legend(
        ax, shift_color, velocity_color, show_shift_arrow, projected_velocity
    )
    ax.set(
        title=f"Display overlay · {basis_name}",
        xlabel=f"{basis_name.upper()}1",
        ylabel=f"{basis_name.upper()}2",
        xticks=[],
        yticks=[],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_velocity_arrow and not projected_velocity:
        ax.text(
            0.01,
            1.01,
            f"No compatible projected velocity in {basis_name}",
            transform=ax.transAxes,
            fontsize=8,
            color="0.4",
            ha="left",
            va="bottom",
        )
    for y, (_, row) in enumerate(stored.iterrows()):
        cosine = float(row.alignment_cosine)
        if np.isfinite(cosine):
            good = bool(row.usable)
            quantitative.scatter(
                cosine,
                y,
                s=45,
                color=cmap(norm(cosine)) if good else "#999999",
                marker="o" if good else "x",
                zorder=3,
            )
            quantitative.text(
                1.04,
                y,
                f"{cosine:.2f}" + ("" if good else " (unusable)"),
                transform=quantitative.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
        else:
            quantitative.text(
                0, y, "unavailable", ha="center", va="center", color="0.5", fontsize=8
            )
    quantitative.set(
        yticks=np.arange(len(stored)),
        yticklabels=stored[node_key],
        xlim=(-1.05, 1.05),
        ylim=(len(stored) - 0.5, -0.5),
        xlabel="Stored alignment cosine",
        title=f"Quantitative alignment · {analysis_rep}",
    )
    quantitative.axvline(0, color="0.6", linewidth=0.8)
    quantitative.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        quantitative.spines[spine].set_visible(False)
    if owned:
        fig.suptitle(title or f"ScGeo: stored alignment ({group0} → {group1})")
        fig.tight_layout()
    elif title is not None:
        quantitative.set_title(title)
    data = {
        "stored_alignment": stored.copy(),
        "display_vectors": display.copy(),
        "display_cells": display_cells,
    }
    metadata = {
        "plot": "velocity_shift_alignment",
        "mode": "stored",
        "analysis_rep": analysis_rep,
        "display_basis": basis,
        "display_embedding_key": basis if basis.startswith("X_") else f"X_{basis}",
        "background_scope": "all_conditions_finite_display_coordinates",
        "background_n_cells": len(display_cells),
        "background_n_input_cells": adata.n_obs,
        "background_style": {
            "size": bg_size,
            "alpha": bg_alpha,
            "color": "lightgrey",
            "rasterized": True,
        },
        "node_size": node_size,
        "label_fontsize": label_fontsize,
        "alignment_limits": [-1.0, 1.0],
        "store_key": store_key if alignment_table is None else None,
        "group0": str(group0),
        "group1": str(group1),
        "node_key": node_key,
        "condition_key": condition_key,
        "projected_velocity_key": velocity_key if projected_velocity else None,
        "projected_velocity_scope": "selected_conditions",
        "projected_velocity_conditions": [str(group0), str(group1)],
        "quantitative_values_recomputed": False,
        "source_params": params,
        "arrow_coordinates": "first two display coordinates",
        "state_order": stored[node_key].tolist(),
        "shift_scale": shift_scale,
        "velocity_scale": velocity_scale,
        "shift_color": shift_color,
        "velocity_color": velocity_color,
        "show_shift_arrow": show_shift_arrow,
        "show_velocity_arrow": show_velocity_arrow,
        "alignment_cmap": cmap.name,
        "min_cells": min_cells,
        "display_aggregation": agg,
        "color_by_alignment": color_by_alignment,
        "palette": palette if not color_by_alignment else None,
        "label_mode": label_mode,
        "label_top_n": label_top_n,
    }
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes=panel_axes,
            data=data,
            metadata=metadata,
            warnings=notes,
            artists={"background": background},
        )
    if return_data:
        return fig, (ax, quantitative), data
    return fig, (ax, quantitative)
