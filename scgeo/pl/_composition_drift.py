from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _compute_group_condition_table(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' not found in adata.obs.")

    g0 = str(group0)
    g1 = str(group1)

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            condition_key: adata.obs[condition_key].astype(str).values,
        },
        index=adata.obs_names,
    )
    df = df[df[condition_key].isin([g0, g1])].copy()
    if df.empty:
        raise ValueError(f"No cells found for groups {g0!r} and {g1!r}.")

    counts = (
        df.groupby([node_key, condition_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )

    totals = (
        counts.groupby(condition_key, observed=False)["n_cells"]
        .sum()
        .rename("total_cells")
        .reset_index()
    )
    counts = counts.merge(totals, on=condition_key, how="left")
    counts["fraction"] = counts["n_cells"] / counts["total_cells"]

    c0 = counts[counts[condition_key] == g0].rename(
        columns={"n_cells": "n0", "fraction": "frac0"}
    )[[node_key, "n0", "frac0"]]
    c1 = counts[counts[condition_key] == g1].rename(
        columns={"n_cells": "n1", "fraction": "frac1"}
    )[[node_key, "n1", "frac1"]]

    out = c0.merge(c1, on=node_key, how="outer")
    out["n0"] = out["n0"].fillna(0).astype(int)
    out["n1"] = out["n1"].fillna(0).astype(int)
    out["frac0"] = out["frac0"].fillna(0.0).astype(float)
    out["frac1"] = out["frac1"].fillna(0.0).astype(float)
    out["delta_frac"] = out["frac1"] - out["frac0"]
    out["log2_fc"] = np.log2((out["frac1"] + 1e-9) / (out["frac0"] + 1e-9))
    out["direction"] = np.where(out["delta_frac"] >= 0, "up", "down")
    return out


def _compute_node_centroids(
    adata,
    *,
    node_key: str,
    basis: str,
    agg: str = "mean",
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")

    xy = _get_basis_xy(adata, basis)
    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            "x": xy[:, 0],
            "y": xy[:, 1],
        },
        index=adata.obs_names,
    )

    grouped = df.groupby(node_key, observed=False)[["x", "y"]]
    if agg == "median":
        cent = grouped.median().reset_index()
    elif agg == "mean":
        cent = grouped.mean().reset_index()
    else:
        raise ValueError("agg must be one of {'mean', 'median'}.")

    counts = df.groupby(node_key, observed=False).size().rename("n_total").reset_index()
    cent = cent.merge(counts, on=node_key, how="left")
    return cent


def _get_palette(adata, groupby: str, categories: list[str]) -> dict[str, Any]:
    colors_key = f"{groupby}_colors"
    if colors_key in adata.uns:
        colors = list(adata.uns[colors_key])
        if len(colors) >= len(categories):
            return {cat: colors[i] for i, cat in enumerate(categories)}
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def composition_drift(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    agg: str = "mean",
    bg_size: float = 6.0,
    bg_alpha: float = 0.10,
    centroid_size: float = 320.0,
    centroid_scale_by_n: bool = True,
    centroid_edgecolor: str = "white",
    centroid_lw: float = 1.0,
    drift_cmap: str = "coolwarm",
    drift_vmax: Optional[float] = None,
    bar_alpha: float = 0.90,
    top_n: Optional[int] = None,
    sort_by: str = "abs_delta_frac",
    palette: Optional[dict[str, Any]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14.0, 5.2),
    panels: Sequence[str] = ("embedding", "change", "proportions"),
    axes: Optional[Mapping[str, Any]] = None,
    subfigure=None,
    state_order: Optional[Sequence[str]] = None,
    condition_palette: Optional[Mapping[str, Any]] = None,
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """Plot descriptive cell-composition changes between two conditions.

    The default three panels and legacy ``(fig, axes[, data])`` returns are
    preserved. ``panels`` selects any ordered subset of ``embedding``,
    ``change``, and ``proportions``. Supply a matching named ``axes`` mapping
    or a Matplotlib ``subfigure`` to compose a larger figure. No global layout
    adjustment is made when the caller supplies the drawing surface.

    Fractions use all observed cells in each condition, not biological-sample
    replicates, and are descriptive. ``state_order`` controls the top-to-bottom
    order of the bar panels. ``condition_palette`` maps condition labels to
    colors. ``return_result=True`` returns a :class:`PlotResult`.
    """
    from ._result import PlotResult
    from ._style import category_palette

    names = list(panels)
    allowed = {"embedding", "change", "proportions"}
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError(
            "panels must be a nonempty unique subset of embedding, change, proportions."
        )
    if axes is not None and subfigure is not None:
        raise ValueError("Supply axes or subfigure, not both.")
    if str(group0) == str(group1):
        raise ValueError("group0 and group1 must identify different conditions.")
    comp = _compute_group_condition_table(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
    )
    xy = _get_basis_xy(adata, basis) if "embedding" in names else None
    if xy is not None:
        cent = _compute_node_centroids(adata, node_key=node_key, basis=basis, agg=agg)
        plot_df = comp.merge(cent, on=node_key, how="left")
    else:
        plot_df = comp.copy()
    if state_order is not None:
        order = [str(x) for x in state_order]
        if len(order) != len(set(order)) or not order:
            raise ValueError("state_order must contain unique state labels.")
        indexed = plot_df.assign(**{node_key: plot_df[node_key].astype(str)}).set_index(
            node_key
        )
        missing = [x for x in order if x not in indexed.index]
        if missing:
            raise ValueError(f"state_order contains unknown states: {missing}")
        plot_df = indexed.loc[order].reset_index()
    elif sort_by == "abs_delta_frac":
        plot_df = plot_df.sort_values(
            "delta_frac", key=np.abs, ascending=False, kind="stable"
        )
    elif sort_by in {"delta_frac", "frac1", "frac0"}:
        plot_df = plot_df.sort_values(sort_by, ascending=False, kind="stable")
    else:
        raise ValueError(
            "sort_by must be one of {'abs_delta_frac', 'delta_frac', 'frac1', 'frac0'}."
        )
    if top_n is not None:
        if isinstance(top_n, bool) or int(top_n) != top_n or int(top_n) < 1:
            raise ValueError("top_n must be a positive integer or None.")
        plot_df = plot_df.head(int(top_n)).copy()
    if plot_df.empty:
        raise ValueError("No states available to plot.")
    condition_colors = category_palette(
        [str(group0), str(group1)], palette=condition_palette
    )
    if drift_vmax is not None and (not np.isfinite(drift_vmax) or drift_vmax <= 0):
        raise ValueError("drift_vmax must be finite and positive.")
    vmax = (
        float(drift_vmax)
        if drift_vmax is not None
        else max(float(plot_df.delta_frac.abs().max()), 1e-6)
    )
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = plt.get_cmap(drift_cmap)

    owned = axes is None and subfigure is None
    if axes is not None:
        if not isinstance(axes, Mapping) or any(name not in axes for name in names):
            raise ValueError(f"axes must map each requested panel name: {names}")
        panel_axes = {name: axes[name] for name in names}
        fig = panel_axes[names[0]].figure
        if any(ax.figure is not fig for ax in panel_axes.values()):
            raise ValueError(
                "All supplied axes must belong to the same figure or subfigure."
            )
        if len({id(ax) for ax in panel_axes.values()}) != len(names):
            raise ValueError("Each panel requires a distinct axis.")
    elif subfigure is not None:
        aa = subfigure.subplots(1, len(names), squeeze=False)
        fig = subfigure
        panel_axes = dict(zip(names, aa[0]))
    else:
        fig, aa = plt.subplots(
            1,
            len(names),
            figsize=figsize,
            squeeze=False,
            gridspec_kw={
                "width_ratios": [1.35 if name == "embedding" else 1.0 for name in names]
            },
        )
        panel_axes = dict(zip(names, aa[0]))
    artists = {}
    if "embedding" in panel_axes:
        ax = panel_axes["embedding"]
        artists["background"] = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=bg_size,
            c="lightgrey",
            alpha=bg_alpha,
            linewidths=0,
            rasterized=True,
            zorder=0,
        )
        max_n = max(float(plot_df.n1.max()), 1.0)
        for _, row in plot_df.iterrows():
            size = centroid_size
            if centroid_scale_by_n:
                size *= 0.55 + 0.45 * np.sqrt(max(float(row.n1), 1.0) / max_n)
            ax.scatter(
                float(row.x),
                float(row.y),
                s=size,
                color=cmap_obj(norm(float(row.delta_frac))),
                edgecolor=centroid_edgecolor,
                linewidth=centroid_lw,
                alpha=0.98,
                zorder=3,
            )
            ax.text(
                float(row.x),
                float(row.y),
                str(row[node_key]),
                ha="center",
                va="center",
                fontsize=8,
                zorder=4,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.72,
                },
            )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label(
            f"Δ cell fraction ({group1} − {group0})"
        )
        basis_name = basis.removeprefix("X_")
        ax.set(
            xlabel=f"{basis_name.upper()}1",
            ylabel=f"{basis_name.upper()}2",
            xticks=[],
            yticks=[],
            title="Composition on display embedding",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar_df = plot_df.iloc[::-1]
    if "change" in panel_axes:
        ax = panel_axes["change"]
        artists["change"] = ax.barh(
            bar_df[node_key].astype(str),
            bar_df.delta_frac,
            color=[cmap_obj(norm(float(v))) for v in bar_df.delta_frac],
            alpha=bar_alpha,
        )
        ax.axvline(0, color="0.3", linewidth=1)
        ax.set(
            xlabel=f"Δ cell fraction ({group1} − {group0})",
            title="Signed abundance change",
        )
    if "proportions" in panel_axes:
        ax = panel_axes["proportions"]
        y = np.arange(len(bar_df))
        h = 0.38
        artists["group0"] = ax.barh(
            y - h / 2,
            bar_df.frac0,
            height=h,
            color=condition_colors[str(group0)],
            alpha=bar_alpha,
            label=str(group0),
        )
        artists["group1"] = ax.barh(
            y + h / 2,
            bar_df.frac1,
            height=h,
            color=condition_colors[str(group1)],
            alpha=bar_alpha,
            label=str(group1),
        )
        ax.set(
            yticks=y,
            yticklabels=bar_df[node_key].astype(str),
            xlabel="Cell fraction",
            title="Condition-specific composition",
        )
        ax.legend(frameon=False)
    for name in ("change", "proportions"):
        if name in panel_axes:
            panel_axes[name].grid(axis="x", alpha=0.2)
            for spine in ("top", "right"):
                panel_axes[name].spines[spine].set_visible(False)
    if owned or subfigure is not None:
        fig.suptitle(title or f"ScGeo: descriptive composition ({group0} → {group1})")
    elif title is not None:
        panel_axes[names[0]].set_title(title)
    if owned:
        fig.tight_layout()
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes=panel_axes,
            data=plot_df.copy(),
            artists=artists,
            metadata={
                "plot": "composition_drift",
                "basis": basis,
                "group0": str(group0),
                "group1": str(group1),
                "condition_key": condition_key,
                "node_key": node_key,
                "inference_level": "descriptive_cell_fractions",
                "panels": names,
                "state_order": plot_df[node_key].astype(str).tolist(),
                "condition_palette": condition_colors,
                "drift_limits": [-vmax, vmax],
                "drift_cmap": cmap_obj.name,
                "aggregation": agg,
                "top_n": top_n,
                "sort_by": sort_by,
            },
        )
    legacy_axes = tuple(panel_axes[name] for name in names)
    if return_data:
        return fig, legacy_axes, plot_df
    return fig, legacy_axes
