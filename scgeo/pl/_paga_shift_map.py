from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

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

    total0 = max(float(out["n0"].fillna(0).sum()), 1.0)
    total1 = max(float(out["n1"].fillna(0).sum()), 1.0)
    out["frac0"] = out["n0"].fillna(0).astype(float) / total0
    out["frac1"] = out["n1"].fillna(0).astype(float) / total1
    out["delta_frac"] = out["frac1"] - out["frac0"]
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
        key = f"velocity_{basis}" if not basis.startswith("X_") else f"velocity_{basis[2:]}"
    else:
        key = velocity_basis if velocity_basis.startswith("velocity_") else f"velocity_{velocity_basis}"

    if key not in adata.obsm:
        raise KeyError(f"Velocity embedding '{key}' not found in adata.obsm.")

    vel = np.asarray(adata.obsm[key])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity embedding '{key}' must have shape (n_cells, >=2).")

    df = pd.DataFrame(
        {
            node_key: _as_str_categorical(adata.obs[node_key]),
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


def _compute_node_composition(
    adata,
    *,
    node_key: str,
    pie_key: str,
    categories: Optional[list[str]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if pie_key not in adata.obs:
        raise KeyError(f"'{pie_key}' not found in adata.obs.")

    df = (
        adata.obs[[node_key, pie_key]]
        .astype({node_key: "str", pie_key: "str"})
        .copy()
    )

    tab = pd.crosstab(df[node_key], df[pie_key], normalize="index" if normalize else False)

    if categories is not None:
        for c in categories:
            if c not in tab.columns:
                tab[c] = 0.0
        tab = tab[categories]

    tab = tab.reset_index().rename(columns={node_key: "node"})
    return tab


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


def _draw_pie_node(ax, x, y, values, colors, radius, edgecolor="white", lw=1.0, zorder=3):
    total = float(np.sum(values))
    if total <= 0:
        circ = Circle((x, y), radius=radius, facecolor="lightgrey", edgecolor=edgecolor, linewidth=lw, zorder=zorder)
        ax.add_patch(circ)
        return

    start = 0.0
    for v, c in zip(values, colors):
        if v <= 0:
            continue
        frac = float(v) / total
        end = start + 360.0 * frac
        wedge = Wedge(
            (x, y),
            r=radius,
            theta1=start,
            theta2=end,
            facecolor=c,
            edgecolor=edgecolor,
            linewidth=0.3,
            zorder=zorder,
        )
        ax.add_patch(wedge)
        start = end

    circ = Circle((x, y), radius=radius, facecolor="none", edgecolor=edgecolor, linewidth=lw, zorder=zorder + 0.1)
    ax.add_patch(circ)


def _node_color_from_mode(
    node_name: str,
    row: pd.Series,
    *,
    mode: str,
    palette: dict[str, Any],
    alignment_map: Optional[dict[str, float]] = None,
    delta_col: str = "delta_frac",
    constant_color: str = "gold",
):
    if mode == "palette":
        return palette.get(node_name, "tab:blue")

    if mode == "constant":
        return constant_color

    if mode == "delta":
        val = float(row[delta_col]) if delta_col in row and pd.notna(row[delta_col]) else 0.0
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        return cmap(norm(val))

    if mode == "alignment":
        if alignment_map is None:
            return palette.get(node_name, "tab:blue")
        val = alignment_map.get(node_name, np.nan)
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        return cmap(norm(val)) if np.isfinite(val) else "lightgrey"

    raise ValueError("node_color_mode must be one of {'palette', 'alignment', 'delta', 'constant'}")


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
    pie_key: Optional[str] = None,
    pie_categories: Optional[list[str]] = None,
    pie_palette: Optional[dict[str, Any]] = None,
    pie_size_scale: float = 1.0,
    velocity_basis: Optional[str] = None,
    show_velocity: bool = False,
    velocity_color: str = "cyan",
    velocity_scale: float = 50.0,
    velocity_alpha: float = 0.95,
    node_color_mode: str = "palette",  # palette | alignment | delta | constant
    alignment_df: Optional[pd.DataFrame] = None,
    alignment_key: str = "alignment_cosine",
    delta_key: str = "delta_frac",
    constant_node_color: str = "gold",
    highlight_nodes: Optional[list[str]] = None,
    highlight_edgecolor: str = "black",
    highlight_lw: float = 2.0,
    ax=None,
    figsize: tuple[float, float] = (8.0, 7.0),
    title: Optional[str] = None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Overlay a PAGA graph anchored on group0 centroids in embedding space,
    with arrows pointing from group0 -> group1 centroids for each node.

    Optional upgrades:
    - pie_key: draw node pies for composition (e.g. timepoint)
    - show_velocity: overlay node-level mean velocity arrows
    - node_color_mode: color nodes by palette / alignment / delta / constant
    - highlight_nodes: emphasize selected nodes
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

    # Optional: velocity
    vel_df = None
    vel_idx = None
    if show_velocity:
        vel_df = _compute_velocity_vectors(
            adata,
            node_key=node_key,
            basis=basis,
            velocity_basis=velocity_basis,
            min_cells=min_cells,
            agg=agg,
        )
        vel_idx = vel_df.set_index(node_key, drop=False)

    # Optional: composition pies
    pie_df = None
    pie_idx = None
    if pie_key is not None:
        pie_df = _compute_node_composition(
            adata,
            node_key=node_key,
            pie_key=pie_key,
            categories=pie_categories,
            normalize=True,
        )
        pie_idx = pie_df.set_index("node", drop=False)

        if pie_categories is None:
            pie_categories = [c for c in pie_df.columns if c != "node"]

        if pie_palette is None:
            pie_palette = _get_palette(adata, pie_key, list(pie_categories))

    # Optional: alignment map
    alignment_map = None
    if alignment_df is not None:
        if node_key in alignment_df.columns and alignment_key in alignment_df.columns:
            alignment_map = dict(
                zip(
                    alignment_df[node_key].astype(str),
                    pd.to_numeric(alignment_df[alignment_key], errors="coerce"),
                )
            )

    highlight_nodes = set(map(str, highlight_nodes)) if highlight_nodes is not None else set()

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

    # edges
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

    # nodes
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

        radius = np.sqrt(size) / 40.0 * pie_size_scale

        base_color = _node_color_from_mode(
            n,
            row,
            mode=node_color_mode,
            palette=palette,
            alignment_map=alignment_map,
            delta_col=delta_key,
            constant_color=constant_node_color,
        )

        edgecolor = highlight_edgecolor if n in highlight_nodes else "white"
        lw = highlight_lw if n in highlight_nodes else 1.0

        if pie_idx is not None and n in pie_idx.index:
            pie_row = pie_idx.loc[n]
            vals = [float(pie_row[c]) for c in pie_categories]
            cols = [pie_palette[c] for c in pie_categories]
            _draw_pie_node(
                ax,
                float(row["x0"]),
                float(row["y0"]),
                values=vals,
                colors=cols,
                radius=radius,
                edgecolor=edgecolor,
                lw=lw,
                zorder=3,
            )
        else:
            ax.scatter(
                float(row["x0"]),
                float(row["y0"]),
                s=size,
                color=base_color,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=0.98,
                zorder=3,
            )

    # labels + geometry shift arrows
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
            color="black",
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
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

    # optional velocity arrows
    if show_velocity and vel_idx is not None:
        for _, row in arrow_rows.iterrows():
            n = str(row[node_key])
            if n not in vel_idx.index:
                continue
            vr = vel_idx.loc[n]
            if not bool(vr["present"]):
                continue

            vx = float(vr["vx"]) * velocity_scale
            vy = float(vr["vy"]) * velocity_scale
            if not np.isfinite(vx) or not np.isfinite(vy):
                continue

            ax.arrow(
                float(row["x0"]),
                float(row["y0"]),
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
        out = {"centroids": cent, "edges": edges}
        if vel_df is not None:
            out["velocity"] = vel_df
        if pie_df is not None:
            out["composition"] = pie_df
        return fig, ax, out

    if made_fig:
        return fig, ax
    return ax