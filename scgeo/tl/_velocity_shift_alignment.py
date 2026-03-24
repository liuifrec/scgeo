from __future__ import annotations

from typing import Any, Optional

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
    alignment_pos_thr: float = 0.3,
    alignment_neg_thr: float = -0.3,
    key_added: Optional[str] = "velocity_shift_alignment",
    propagate_to_obs: bool = False,
) -> pd.DataFrame:
    """
    Compute node-wise alignment between observed geometric shift and mean velocity.

    Parameters
    ----------
    adata
        AnnData object.
    node_key
        obs column defining nodes/groups (e.g. cluster labels).
    condition_key
        obs column defining the condition or timepoint.
    group0, group1
        Two values in `condition_key` to compare. Shift is computed as group1 - group0.
    basis
        Embedding basis in `obsm`, with or without `X_` prefix.
    velocity_basis
        Velocity embedding basis in `obsm`, with or without `velocity_` prefix.
        If None, auto-detect from `basis`.
    min_cells
        Minimum cells required per node per condition.
    agg
        Aggregation method for centroids and mean velocity: {'mean', 'median'}.
    alignment_pos_thr, alignment_neg_thr
        Thresholds for classifying alignment cosine.
    key_added
        If not None, store the node-level dataframe in `adata.uns[key_added]`.
    propagate_to_obs
        If True, propagate node-level cosine/class back to cells in `adata.obs`.

    Returns
    -------
    pd.DataFrame
        Node-level dataframe with shift, velocity, cosine alignment, and classes.
    """
    align = _compute_node_centroids(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        min_cells=min_cells,
        agg=agg,
    ).merge(
        _compute_velocity_vectors(
            adata,
            node_key=node_key,
            basis=basis,
            velocity_basis=velocity_basis,
            min_cells=min_cells,
            agg=agg,
        )[[node_key, "x", "y", "vx", "vy", "vel_norm", "present"]],
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

    if key_added is not None:
        adata.uns[key_added] = align.copy()

    if propagate_to_obs:
        node_series = adata.obs[node_key].astype(str)

        cosine_map = align.set_index(node_key)["alignment_cosine"].to_dict()
        class_map = align.set_index(node_key)["alignment_class"].to_dict()
        usable_map = align.set_index(node_key)["usable"].to_dict()
        shift_norm_map = align.set_index(node_key)["shift_norm"].to_dict()
        vel_norm_map = align.set_index(node_key)["vel_norm"].to_dict()

        prefix = key_added if key_added is not None else "velocity_shift_alignment"
        adata.obs[f"{prefix}_cosine"] = node_series.map(cosine_map).astype(float)
        adata.obs[f"{prefix}_class"] = node_series.map(class_map).astype("category")
        adata.obs[f"{prefix}_usable"] = node_series.map(usable_map).astype("boolean")
        adata.obs[f"{prefix}_shift_norm"] = node_series.map(shift_norm_map).astype(float)
        adata.obs[f"{prefix}_vel_norm"] = node_series.map(vel_norm_map).astype(float)

    return align