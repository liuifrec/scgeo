from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    xy = np.asarray(adata.obsm[key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (n_cells, >=2).")
    return xy[:, :2]


def _as_str_categorical(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        return s.astype(str)
    return s.astype("category").astype(str)


def _get_palette(adata, groupby: str, categories: list[str]) -> dict[str, Any]:
    colors_key = f"{groupby}_colors"
    if colors_key in adata.uns:
        colors = list(adata.uns[colors_key])
        if len(colors) >= len(categories):
            return {cat: colors[i] for i, cat in enumerate(categories)}
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def _compute_centroids(
    adata,
    *,
    basis: str,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    min_cells: int = 10,
    agg: str = "mean",
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' not found in adata.obs.")

    xy = _get_basis_xy(adata, basis)
    df = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            node_key: _as_str_categorical(adata.obs[node_key]),
            condition_key: adata.obs[condition_key].astype(str).values,
        },
        index=adata.obs_names,
    )

    g0 = str(group0)
    g1 = str(group1)
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
    out["shift_umap"] = np.sqrt(np.square(out["dx"]) + np.square(out["dy"]))
    return out


def _get_paga_connectivities(adata, paga_key: str = "paga"):
    if paga_key not in adata.uns:
        raise KeyError(f"adata.uns['{paga_key}'] not found.")
    paga = adata.uns[paga_key]
    if "connectivities" not in paga:
        raise KeyError(f"adata.uns['{paga_key}']['connectivities'] not found.")
    conn = paga["connectivities"]
    if sparse is not None and sparse.issparse(conn):
        return conn.tocsr()
    return np.asarray(conn)


def _node_categories(adata, node_key: str) -> list[str]:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    s = adata.obs[node_key]
    if pd.api.types.is_categorical_dtype(s):
        return [str(x) for x in s.cat.categories]
    return sorted(map(str, pd.unique(s)))


def _edge_list(connectivities, node_names: list[str], threshold: float) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    n = len(node_names)

    if sparse is not None and sparse.issparse(connectivities):
        coo = connectivities.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i < j and i < n and j < n and float(w) >= threshold:
                edges.append((node_names[i], node_names[j], float(w)))
        return edges

    arr = np.asarray(connectivities)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(arr[i, j])
            if w >= threshold:
                edges.append((node_names[i], node_names[j], w))
    return edges


def paga_shift_map(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    min_cells: int = 15,
    connectivity_threshold: float = 0.05,
    agg: str = "mean",
    background_size: float = 6.0,
    background_alpha: float = 0.15,
    node_size: float = 220.0,
    node_scale_by_n: bool = True,
    edge_lw: float = 2.0,
    edge_alpha: float = 0.55,
    arrow_width: float = 0.008,
    arrow_alpha: float = 0.95,
    arrow_scale: float = 1.0,
    label: bool = True,
    label_top_n: Optional[int] = None,
    label_fontsize: int = 8,
    palette: Optional[dict[str, Any]] = None,
    ax=None,
    figsize: tuple[float, float] = (8.0, 7.0),
    title: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Overlay a PAGA graph anchored on group0 centroids in embedding space,
    with arrows pointing from group0 -> group1 centroids for each node.

    Returns
    -------
    If return_data is False:
        (fig, ax) or ax
    If return_data is True:
        (fig, ax, centroids_df, edges)
    """
    xy = _get_basis_xy(adata, basis)
    node_names = _node_categories(adata, node_key)
    cent = _compute_centroids(
        adata,
        basis=basis,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        min_cells=min_cells,
        agg=agg,
    )
    conn = _get_paga_connectivities(adata, paga_key=paga_key)
    edges = _edge_list(conn, node_names=node_names, threshold=connectivity_threshold)

    cent_idx = cent.set_index(node_key, drop=False)
    placeable = {
        n for n in node_names
        if n in cent_idx.index and bool(cent_idx.loc[n, "present0"])
    }
    edges = [(a, b, w) for a, b, w in edges if a in placeable and b in placeable]

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
        s=background_size,
        c="lightgrey",
        alpha=background_alpha,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )

    for a, b, w in edges:
        ra = cent_idx.loc[a]
        rb = cent_idx.loc[b]
        ax.plot(
            [float(ra["x0"]), float(rb["x0"])],
            [float(ra["y0"]), float(rb["y0"])],
            color="black",
            lw=edge_lw * max(w, 0.15),
            alpha=edge_alpha,
            zorder=1,
        )

    max_n0 = max(float(cent["n0"].fillna(1).max()), 1.0)

    for n in node_names:
        if n not in cent_idx.index:
            continue
        row = cent_idx.loc[n]
        if not bool(row["present0"]):
            continue

        size = node_size
        if node_scale_by_n and pd.notna(row["n0"]):
            frac = np.sqrt(max(float(row["n0"]), 1.0) / max_n0)
            size = node_size * (0.55 + 0.45 * frac)

        ax.scatter(
            float(row["x0"]),
            float(row["y0"]),
            s=size,
            color=palette.get(n, "tab:blue"),
            edgecolor="white",
            linewidth=1.0,
            alpha=0.98,
            zorder=3,
        )

    arrow_rows = cent[(cent["present0"]) & (cent["present1"])].copy()
    if label_top_n is not None:
        label_nodes = set(
            arrow_rows.sort_values("shift_umap", ascending=False)[node_key].head(label_top_n)
        )
    else:
        label_nodes = set(arrow_rows[node_key].tolist())

    for _, row in arrow_rows.iterrows():
        n = str(row[node_key])
        dx = float(row["dx"]) * arrow_scale
        dy = float(row["dy"]) * arrow_scale
        if not np.isfinite(dx) or not np.isfinite(dy):
            continue

        ax.arrow(
            float(row["x0"]),
            float(row["y0"]),
            dx,
            dy,
            width=arrow_width,
            head_width=arrow_width * 8.5,
            head_length=max(np.hypot(dx, dy) * 0.12, arrow_width * 8.5),
            length_includes_head=True,
            color=palette.get(n, "tab:red"),
            alpha=arrow_alpha,
            zorder=4,
        )

        if label and n in label_nodes:
            ax.text(
                float(row["x0"]),
                float(row["y0"]),
                n,
                fontsize=label_fontsize,
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

    basis_name = basis[2:] if basis.startswith("X_") else basis
    ax.set_xlabel(f"{basis_name.upper()}1")
    ax.set_ylabel(f"{basis_name.upper()}2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f"ScGeo: PAGA shift map ({group0} → {group1})"
    ax.set_title(title)

    if show:
        plt.show()

    if return_data:
        return fig, ax, cent, edges
    if made_fig:
        return fig, ax
    return ax