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
        raise ValueError("group0 and group1 must be provided when condition_key is used.")

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


def _safe_cosine(dx: np.ndarray, dy: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
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
        if len(colors) >= len(categories):
            return {cat: colors[i] for i, cat in enumerate(categories)}
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


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
    show: bool = True,
):
    """
    Plot node-wise observed shift vectors and mean velocity vectors on the same embedding.

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
        raise ValueError("label_mode must be one of {'shift', 'discordant', 'aligned'}.")

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
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

    if color_by_alignment and not usable.empty:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cosine(shift, velocity)")

    # compact manual legend
    ax.text(
        0.01,
        0.01,
        "black arrow: observed shift\nblue arrow: mean velocity\nfill: alignment cosine",
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
        title = f"ScGeo: velocity-shift alignment ({group0} → {group1})"
    ax.set_title(title)

    if show:
        plt.show()

    if return_data:
        return fig, ax, align
    if made_fig:
        return fig, ax
    return ax