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


def _as_str_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        return s.astype(str)
    return s.astype("category").astype(str)


def _compute_shift_summary(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str,
    min_cells: int = 10,
    agg: str = "mean",
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' not found in adata.obs.")

    xy = _get_basis_xy(adata, basis)
    g0 = str(group0)
    g1 = str(group1)

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            condition_key: adata.obs[condition_key].astype(str).values,
            "x": xy[:, 0],
            "y": xy[:, 1],
        },
        index=adata.obs_names,
    )
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
    out["n0"] = out["n0"].fillna(0).astype(int)
    out["n1"] = out["n1"].fillna(0).astype(int)
    out["present0"] = out["n0"] >= int(min_cells)
    out["present1"] = out["n1"] >= int(min_cells)
    out["dx"] = out["x1"] - out["x0"]
    out["dy"] = out["y1"] - out["y0"]
    out["shift_norm"] = np.sqrt(np.square(out["dx"]) + np.square(out["dy"]))
    return out


def _compute_velocity_summary(
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

    if velocity_basis is None:
        vel_key = f"velocity_{basis}" if not basis.startswith("X_") else f"velocity_{basis[2:]}"
    else:
        vel_key = velocity_basis if velocity_basis.startswith("velocity_") else f"velocity_{velocity_basis}"

    if vel_key not in adata.obsm:
        raise KeyError(f"Velocity embedding '{vel_key}' not found in adata.obsm.")

    vel = np.asarray(adata.obsm[vel_key])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity embedding '{vel_key}' must have shape (n_cells, >=2).")

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            "vx": vel[:, 0],
            "vy": vel[:, 1],
        },
        index=adata.obs_names,
    )

    grouped = df.groupby(node_key, observed=False)[["vx", "vy"]]
    if agg == "median":
        out = grouped.median().reset_index()
    elif agg == "mean":
        out = grouped.mean().reset_index()
    else:
        raise ValueError("agg must be one of {'mean', 'median'}.")

    counts = df.groupby(node_key, observed=False).size().rename("n_vel").reset_index()
    out = out.merge(counts, on=node_key, how="left")
    out["velocity_present"] = out["n_vel"].fillna(0).astype(int) >= int(min_cells)
    out["vel_norm"] = np.sqrt(np.square(out["vx"]) + np.square(out["vy"]))
    return out


def _compute_composition_summary(
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
        columns={"n_cells": "n0_comp", "fraction": "frac0"}
    )[[node_key, "n0_comp", "frac0"]]
    c1 = counts[counts[condition_key] == g1].rename(
        columns={"n_cells": "n1_comp", "fraction": "frac1"}
    )[[node_key, "n1_comp", "frac1"]]

    out = c0.merge(c1, on=node_key, how="outer")
    out["n0_comp"] = out["n0_comp"].fillna(0).astype(int)
    out["n1_comp"] = out["n1_comp"].fillna(0).astype(int)
    out["frac0"] = out["frac0"].fillna(0.0).astype(float)
    out["frac1"] = out["frac1"].fillna(0.0).astype(float)
    out["delta_frac"] = out["frac1"] - out["frac0"]
    out["delta_abs"] = np.abs(out["delta_frac"])
    return out


def _compute_ood_node_summary(
    adata,
    *,
    node_key: str,
    ood_key: str,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if ood_key not in adata.obs:
        raise KeyError(f"'{ood_key}' not found in adata.obs.")

    scores = pd.to_numeric(adata.obs[ood_key], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError(f"No finite values found in '{ood_key}'.")

    thr = float(np.nanquantile(scores[finite], 0.95)) if threshold is None else float(threshold)
    flagged = finite & (scores >= thr)

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            "ood_score": scores,
            "ood_flagged": flagged,
        },
        index=adata.obs_names,
    )

    out = (
        df.groupby(node_key, observed=False)
        .agg(
            ood_mean=("ood_score", "mean"),
            ood_median=("ood_score", "median"),
            ood_flagged_n=("ood_flagged", "sum"),
            ood_frac=("ood_flagged", "mean"),
            ood_n=(node_key, "size"),
        )
        .reset_index()
    )
    out["ood_threshold_used"] = thr
    return out


def _safe_cosine(dx: np.ndarray, dy: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    dot = dx * vx + dy * vy
    dn = np.sqrt(dx * dx + dy * dy)
    vn = np.sqrt(vx * vx + vy * vy)
    denom = dn * vn
    out = np.full_like(dot, np.nan, dtype=float)
    mask = np.isfinite(denom) & (denom > 0)
    out[mask] = dot[mask] / denom[mask]
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


def _get_node_categories(adata, node_key: str) -> list[str]:
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    s = adata.obs[node_key]
    if pd.api.types.is_categorical_dtype(s):
        return [str(x) for x in s.cat.categories]
    return sorted(map(str, pd.unique(s)))


def _extract_edge_list(connectivities, node_names: list[str], threshold: float) -> list[tuple[str, str, float]]:
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


def _assemble_recovery_compass_table(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str,
    velocity_basis: Optional[str],
    ood_key: Optional[str],
    min_cells: int,
    agg: str = "mean",
) -> pd.DataFrame:
    shift_df = _compute_shift_summary(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        min_cells=min_cells,
        agg=agg,
    )
    comp_df = _compute_composition_summary(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
    )

    out = shift_df.merge(comp_df, on=node_key, how="left")

    try:
        vel_df = _compute_velocity_summary(
            adata,
            node_key=node_key,
            basis=basis,
            velocity_basis=velocity_basis,
            min_cells=min_cells,
            agg=agg,
        )
        out = out.merge(vel_df, on=node_key, how="left")
        out["alignment_cosine"] = _safe_cosine(
            out["dx"].to_numpy(dtype=float),
            out["dy"].to_numpy(dtype=float),
            out["vx"].to_numpy(dtype=float),
            out["vy"].to_numpy(dtype=float),
        )
    except KeyError:
        out["vx"] = np.nan
        out["vy"] = np.nan
        out["vel_norm"] = np.nan
        out["velocity_present"] = False
        out["alignment_cosine"] = np.nan

    if ood_key is not None:
        ood_df = _compute_ood_node_summary(
            adata,
            node_key=node_key,
            ood_key=ood_key,
        )
        out = out.merge(ood_df, on=node_key, how="left")
    else:
        out["ood_mean"] = np.nan
        out["ood_median"] = np.nan
        out["ood_flagged_n"] = np.nan
        out["ood_frac"] = np.nan
        out["ood_n"] = np.nan
        out["ood_threshold_used"] = np.nan

    out["usable_shift"] = out["present0"].fillna(False).astype(bool) & out["present1"].fillna(False).astype(bool)
    out["usable_velocity"] = out["usable_shift"] & out["velocity_present"].fillna(False).astype(bool)
    return out


def recovery_compass(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    velocity_basis: Optional[str] = None,
    ood_key: Optional[str] = None,
    min_cells: int = 15,
    connectivity_threshold: float = 0.05,
    node_size_mode: str = "group1_n",
    node_size_scale: float = 380.0,
    fill_color_mode: str = "alignment",
    fill_cmap: str = "coolwarm",
    fill_vmin: float = -1.0,
    fill_vmax: float = 1.0,
    ring_mode: str = "ood_frac",
    ring_color: str = "gold",
    ring_max_lw: float = 4.0,
    arrow_color_mode: str = "shift",
    arrow_color: str = "black",
    arrow_cmap: str = "magma",
    arrow_scale: float = 1.0,
    arrow_width: float = 0.008,
    edge_alpha: float = 0.45,
    edge_lw: float = 2.0,
    bg_size: float = 5.0,
    bg_alpha: float = 0.08,
    label: bool = True,
    label_top_n: Optional[int] = 12,
    label_fontsize: int = 8,
    legend: bool = True,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (9.5, 8.0),
    ax=None,
    return_data: bool = False,
    show: bool = True,
):
    """
    Signature ScGeo synthesis plot combining:
    - PAGA topology
    - centroid displacement
    - velocity-shift alignment
    - OOD burden
    - abundance

    Returns
    -------
    If return_data is False:
        (fig, ax) or ax
    If return_data is True:
        (fig, ax, node_df, edges)
    """
    xy = _get_basis_xy(adata, basis)
    node_df = _assemble_recovery_compass_table(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        velocity_basis=velocity_basis,
        ood_key=ood_key,
        min_cells=min_cells,
    )

    node_names = _get_node_categories(adata, node_key)
    conn = _get_paga_connectivities(adata, paga_key=paga_key)
    edges = _extract_edge_list(conn, node_names=node_names, threshold=connectivity_threshold)

    node_idx = node_df.set_index(node_key, drop=False)
    placeable = {
        n for n in node_names
        if n in node_idx.index and bool(node_idx.loc[n, "present0"])
    }
    edges = [(a, b, w) for a, b, w in edges if a in placeable and b in placeable]

    made_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        made_fig = True
    else:
        fig = ax.figure

    # Step 2: base plot
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

    for a, b, w in edges:
        ra = node_idx.loc[a]
        rb = node_idx.loc[b]
        ax.plot(
            [float(ra["x0"]), float(rb["x0"])],
            [float(ra["y0"]), float(rb["y0"])],
            color="black",
            lw=edge_lw * max(w, 0.15),
            alpha=edge_alpha,
            zorder=1,
        )

    # node size encoding
    if node_size_mode == "group1_n":
        size_values = node_df["n1"].fillna(0).to_numpy(dtype=float)
    elif node_size_mode == "group0_n":
        size_values = node_df["n0"].fillna(0).to_numpy(dtype=float)
    elif node_size_mode == "delta_abs":
        size_values = node_df["delta_abs"].fillna(0).to_numpy(dtype=float)
    elif node_size_mode == "constant":
        size_values = np.ones(node_df.shape[0], dtype=float)
    else:
        raise ValueError("node_size_mode must be one of {'group1_n', 'group0_n', 'delta_abs', 'constant'}.")

    size_max = max(float(np.nanmax(size_values)) if np.any(np.isfinite(size_values)) else 1.0, 1.0)

    # node fill encoding
    if fill_color_mode == "alignment":
        fill_values = node_df["alignment_cosine"].to_numpy(dtype=float)
    elif fill_color_mode == "delta_frac":
        fill_values = node_df["delta_frac"].to_numpy(dtype=float)
        fill_vmin = float(np.nanmin(fill_values)) if np.any(np.isfinite(fill_values)) else -1.0
        fill_vmax = float(np.nanmax(fill_values)) if np.any(np.isfinite(fill_values)) else 1.0
        if np.isclose(fill_vmin, fill_vmax):
            fill_vmin, fill_vmax = -1.0, 1.0
    elif fill_color_mode == "constant":
        fill_values = np.full(node_df.shape[0], 0.5, dtype=float)
        fill_vmin, fill_vmax = 0.0, 1.0
    else:
        raise ValueError("fill_color_mode must be one of {'alignment', 'delta_frac', 'constant'}.")

    fill_norm = plt.Normalize(vmin=fill_vmin, vmax=fill_vmax)
    fill_cmap_obj = plt.get_cmap(fill_cmap)

    # Step 3: arrows and nodes
    if label_top_n is not None:
        label_nodes = set(
            node_df.sort_values("shift_norm", ascending=False)[node_key].head(label_top_n).astype(str)
        )
    else:
        label_nodes = set(node_df[node_key].astype(str))

    if arrow_color_mode == "shift":
        arrow_vals = node_df["shift_norm"].to_numpy(dtype=float)
        arrow_vmin = float(np.nanmin(arrow_vals)) if np.any(np.isfinite(arrow_vals)) else 0.0
        arrow_vmax = float(np.nanmax(arrow_vals)) if np.any(np.isfinite(arrow_vals)) else 1.0
        if np.isclose(arrow_vmin, arrow_vmax):
            arrow_vmin, arrow_vmax = 0.0, max(arrow_vmax, 1.0)
        arrow_norm = plt.Normalize(vmin=arrow_vmin, vmax=arrow_vmax)
        arrow_cmap_obj = plt.get_cmap(arrow_cmap)
    elif arrow_color_mode == "constant":
        arrow_norm = None
        arrow_cmap_obj = None
    else:
        raise ValueError("arrow_color_mode must be one of {'shift', 'constant'}.")

    for _, row in node_df.iterrows():
        n = str(row[node_key])

        if not bool(row["present0"]):
            continue

        x0 = float(row["x0"])
        y0 = float(row["y0"])

        sval = size_values[node_df.index.get_loc(row.name)]
        frac = np.sqrt(max(float(sval), 0.0) / size_max) if size_max > 0 else 1.0
        node_size = node_size_scale * (0.45 + 0.55 * frac)

        fval = fill_values[node_df.index.get_loc(row.name)]
        if np.isfinite(fval):
            facecolor = fill_cmap_obj(fill_norm(float(fval)))
        else:
            facecolor = "lightgrey"

        # Step 3: ring encoding
        if ring_mode == "ood_frac" and "ood_frac" in row.index and pd.notna(row["ood_frac"]):
            ring_lw = 0.8 + ring_max_lw * float(row["ood_frac"])
        elif ring_mode == "none":
            ring_lw = 0.8
        else:
            ring_lw = 0.8

        ax.scatter(
            x0,
            y0,
            s=node_size,
            facecolor=facecolor,
            edgecolor=ring_color,
            linewidth=ring_lw,
            alpha=0.98,
            zorder=3,
        )

        if bool(row["usable_shift"]):
            dx = float(row["dx"]) * arrow_scale
            dy = float(row["dy"]) * arrow_scale

            if arrow_color_mode == "shift" and np.isfinite(row["shift_norm"]):
                acolor = arrow_cmap_obj(arrow_norm(float(row["shift_norm"])))
            else:
                acolor = arrow_color

            ax.arrow(
                x0,
                y0,
                dx,
                dy,
                width=arrow_width,
                head_width=arrow_width * 8.0,
                head_length=max(np.hypot(dx, dy) * 0.12, arrow_width * 8.0),
                length_includes_head=True,
                color=acolor,
                alpha=0.95,
                zorder=4,
            )

        if label and n in label_nodes:
            ax.text(
                x0,
                y0,
                n,
                fontsize=label_fontsize,
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
            )

    # Step 4: colorbars
    if legend:
        if fill_color_mode != "constant":
            sm_fill = plt.cm.ScalarMappable(norm=fill_norm, cmap=fill_cmap_obj)
            sm_fill.set_array([])
            cbar_fill = fig.colorbar(sm_fill, ax=ax, fraction=0.046, pad=0.04)
            if fill_color_mode == "alignment":
                cbar_fill.set_label("cosine(shift, velocity)")
            elif fill_color_mode == "delta_frac":
                cbar_fill.set_label("Δ fraction")

        if arrow_color_mode == "shift":
            sm_arrow = plt.cm.ScalarMappable(norm=arrow_norm, cmap=arrow_cmap_obj)
            sm_arrow.set_array([])
            cbar_arrow = fig.colorbar(sm_arrow, ax=ax, fraction=0.046, pad=0.10)
            cbar_arrow.set_label("shift magnitude")

        # small manual annotation
        annotation_lines = [
            f"node size: {node_size_mode}",
            f"ring: {ring_mode}",
            f"fill: {fill_color_mode}",
        ]
        ax.text(
            0.01,
            0.01,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            zorder=10,
        )

    basis_name = basis[2:] if basis.startswith("X_") else basis
    ax.set_xlabel(f"{basis_name.upper()}1")
    ax.set_ylabel(f"{basis_name.upper()}2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f"ScGeo recovery compass ({group0} → {group1})"
    ax.set_title(title)

    if show:
        plt.show()

    if return_data:
        return fig, ax, node_df, edges
    if made_fig:
        return fig, ax
    return ax