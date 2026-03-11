from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    xy = np.asarray(adata.obsm[key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (n_cells, >=2).")
    return xy[:, :2]


def _as_str_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
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
    return_data: bool = False,
    show: bool = True,
):
    """
    Plot a 3-panel composition drift report:
    1) embedding centroids sized by abundance and colored by signed delta fraction
    2) horizontal barplot of signed delta fraction
    3) side-by-side proportion bars for group0 and group1
    """
    xy = _get_basis_xy(adata, basis)
    comp = _compute_group_condition_table(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
    )
    cent = _compute_node_centroids(
        adata,
        node_key=node_key,
        basis=basis,
        agg=agg,
    )

    plot_df = comp.merge(cent, on=node_key, how="left")
    node_names = list(plot_df[node_key].astype(str))
    if palette is None:
        palette = _get_palette(adata, node_key, node_names)

    if sort_by == "abs_delta_frac":
        plot_df = plot_df.sort_values("delta_frac", key=np.abs, ascending=False)
    elif sort_by == "delta_frac":
        plot_df = plot_df.sort_values("delta_frac", ascending=False)
    elif sort_by == "frac1":
        plot_df = plot_df.sort_values("frac1", ascending=False)
    elif sort_by == "frac0":
        plot_df = plot_df.sort_values("frac0", ascending=False)
    else:
        raise ValueError("sort_by must be one of {'abs_delta_frac', 'delta_frac', 'frac1', 'frac0'}.")

    if top_n is not None:
        plot_df = plot_df.head(int(top_n)).copy()

    fig, (ax0, ax1, ax2) = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )

    # Panel 1: embedding + centroid drift coloring
    ax0.scatter(
        xy[:, 0],
        xy[:, 1],
        s=bg_size,
        c="lightgrey",
        alpha=bg_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    vmax = drift_vmax
    if vmax is None:
        vmax = float(np.nanmax(np.abs(plot_df["delta_frac"].to_numpy(dtype=float)))) if not plot_df.empty else 1.0
        vmax = max(vmax, 1e-6)

    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap_obj = plt.get_cmap(drift_cmap)

    max_n = max(float(plot_df["n1"].max()) if not plot_df.empty else 1.0, 1.0)

    for _, row in plot_df.iterrows():
        size = centroid_size
        if centroid_scale_by_n:
            frac = np.sqrt(max(float(row["n1"]), 1.0) / max_n)
            size = centroid_size * (0.55 + 0.45 * frac)

        ax0.scatter(
            float(row["x"]),
            float(row["y"]),
            s=size,
            color=cmap_obj(norm(float(row["delta_frac"]))),
            edgecolor=centroid_edgecolor,
            linewidth=centroid_lw,
            alpha=0.98,
            zorder=3,
        )
        ax0.text(
            float(row["x"]),
            float(row["y"]),
            str(row[node_key]),
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(f"Δ fraction ({group1} - {group0})")

    basis_name = basis[2:] if basis.startswith("X_") else basis
    ax0.set_xlabel(f"{basis_name.upper()}1")
    ax0.set_ylabel(f"{basis_name.upper()}2")
    ax0.set_xticks([])
    ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_title("Embedding-level composition drift")

    # Panel 2: signed delta fractions
    bar_df = plot_df.iloc[::-1].copy()
    bar_colors = [cmap_obj(norm(float(v))) for v in bar_df["delta_frac"]]
    ax1.barh(bar_df[node_key].astype(str), bar_df["delta_frac"], color=bar_colors, alpha=bar_alpha)
    ax1.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel(f"Δ fraction ({group1} - {group0})")
    ax1.set_title("Signed abundance change")
    ax1.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # Panel 3: side-by-side proportions
    y = np.arange(len(bar_df))
    h = 0.38
    ax2.barh(
        y - h / 2,
        bar_df["frac0"],
        height=h,
        color="dimgray",
        alpha=0.85,
        label=str(group0),
    )
    ax2.barh(
        y + h / 2,
        bar_df["frac1"],
        height=h,
        color="tab:blue",
        alpha=0.85,
        label=str(group1),
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(bar_df[node_key].astype(str))
    ax2.set_xlabel("Fraction")
    ax2.set_title("Condition-specific composition")
    ax2.legend(frameon=False)
    ax2.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    if title is None:
        title = f"ScGeo: composition drift ({group0} → {group1})"
    fig.suptitle(title, y=1.02)

    fig.tight_layout()

    if show:
        plt.show()

    if return_data:
        return fig, (ax0, ax1, ax2), plot_df
    return fig, (ax0, ax1, ax2)