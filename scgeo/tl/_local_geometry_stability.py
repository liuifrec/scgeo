from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd

from .._utils import _as_2d_array


_EPS = 1e-12
_DIAGNOSTIC_RULES = {
    "mad_multiplier": 3.0,
    "min_pairs_for_outlier": 2,
    "neighborhood_metric": "median_pairwise_neighbor_overlap",
    "distortion_metric": "median_global_distortion",
    "state_graph_metric": "median_state_graph_spearman",
}


def _import_sklearn():
    try:
        from sklearn.manifold import trustworthiness
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            "scikit-learn is required for local_geometry_stability. "
            "Install it with `pip install scikit-learn` or `pip install scgeo[sklearn]`."
        ) from exc
    return NearestNeighbors, trustworthiness


def _as_reps(reps) -> list[str]:
    if isinstance(reps, str) or not isinstance(reps, Sequence):
        raise TypeError("reps must be a non-empty sequence of adata.obsm keys")
    out = [str(rep) for rep in reps]
    if not out:
        raise ValueError("reps must contain at least one representation key")
    if len(set(out)) != len(out):
        raise ValueError("reps contains duplicate representation keys")
    return out


def _validate_k_values(k_values, n_obs: int) -> tuple[int, ...]:
    if isinstance(k_values, (str, bytes)) or not isinstance(k_values, Sequence):
        raise TypeError("k_values must be a non-empty sequence of positive integers")
    out = tuple(int(k) for k in k_values)
    if not out:
        raise ValueError("k_values must contain at least one value")
    for k in out:
        if k <= 0:
            raise ValueError("All k_values must be positive")
        if k >= n_obs:
            raise ValueError(f"All k_values must satisfy k < n_obs; got k={k}, n_obs={n_obs}")
    return out


def _load_representations(adata, reps: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for rep in reps:
        if rep not in adata.obsm:
            raise KeyError(f"Representation key {rep!r} not found in adata.obsm")
        X = _as_2d_array(adata.obsm[rep]).astype(np.float64, copy=False)
        if X.shape[0] != adata.n_obs:
            raise ValueError(f"Representation {rep!r} has {X.shape[0]} rows; expected adata.n_obs={adata.n_obs}")
        if not np.isfinite(X).all():
            raise ValueError(f"Representation {rep!r} contains non-finite values")
        out[rep] = X
    return out


def _representation_pairs(reps: list[str], pair_mode: str, reference_rep: Optional[str]) -> list[tuple[str, str]]:
    if pair_mode not in {"all", "reference"}:
        raise ValueError("pair_mode must be one of {'all', 'reference'}")
    if pair_mode == "all":
        return list(combinations(reps, 2))
    if reference_rep is None:
        raise ValueError("reference_rep must be provided when pair_mode='reference'")
    reference_rep = str(reference_rep)
    if reference_rep not in reps:
        raise ValueError("reference_rep must be one of reps")
    return [(reference_rep, rep) for rep in reps if rep != reference_rep]


def _knn(NearestNeighbors, X: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    n = X.shape[0]
    out_idx = np.empty((n, k), dtype=np.int64)
    out_dist = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        keep = idx[i] != i
        row_idx = idx[i][keep]
        row_dist = dist[i][keep]
        if row_idx.size < k:
            row_idx = idx[i][1 : k + 1]
            row_dist = dist[i][1 : k + 1]
        out_idx[i] = row_idx[:k]
        out_dist[i] = row_dist[:k]
    return out_idx, out_dist


def _edge_codes(idx: np.ndarray, n_obs: int) -> np.ndarray:
    src = np.repeat(np.arange(idx.shape[0], dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64)
    return src * int(n_obs) + dst


def _edge_jaccard(idx_a: np.ndarray, idx_b: np.ndarray, n_obs: int) -> float:
    a = np.unique(_edge_codes(idx_a, n_obs))
    b = np.unique(_edge_codes(idx_b, n_obs))
    union = np.union1d(a, b)
    if union.size == 0:
        return np.nan
    return float(np.intersect1d(a, b).size / union.size)


def _neighbor_per_cell(idx_a: np.ndarray, idx_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k = idx_a.shape
    overlap = np.empty(n, dtype=float)
    jaccard = np.empty(n, dtype=float)
    for i in range(n):
        a = set(idx_a[i].tolist())
        b = set(idx_b[i].tolist())
        inter = len(a & b)
        union = len(a | b)
        overlap[i] = inter / k
        jaccard[i] = inter / union if union else np.nan
    return overlap, jaccard


def _global_median_distance(dist: np.ndarray) -> float:
    vals = dist[np.isfinite(dist) & (dist > _EPS)]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def _local_median_distance(dist: np.ndarray, global_median: float) -> np.ndarray:
    out = np.median(np.maximum(dist, _EPS), axis=1)
    out[~np.isfinite(out) | (out <= _EPS)] = max(global_median, _EPS)
    return out


def _union_edges(idx_a: np.ndarray, idx_b: np.ndarray, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    codes = np.union1d(_edge_codes(idx_a, n_obs), _edge_codes(idx_b, n_obs))
    return codes // n_obs, codes % n_obs


def _edge_distances(X: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[dst] - X[src], axis=1)


def _distortion_per_cell(
    X_a: np.ndarray,
    X_b: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> dict[str, np.ndarray]:
    n_obs = X_a.shape[0]
    src, dst = _union_edges(idx_a, idx_b, n_obs)
    da = _edge_distances(X_a, src, dst)
    db = _edge_distances(X_b, src, dst)
    global_a = _global_median_distance(dist_a)
    global_b = _global_median_distance(dist_b)
    local_a = _local_median_distance(dist_a, global_a)
    local_b = _local_median_distance(dist_b, global_b)

    global_edge = np.abs(np.log(((db / global_b) + _EPS) / ((da / global_a) + _EPS)))
    local_edge = np.abs(np.log(((db / local_b[src]) + _EPS) / ((da / local_a[src]) + _EPS)))

    out = {
        "global_distortion_median": np.full(n_obs, np.nan, dtype=float),
        "global_distortion_p90": np.full(n_obs, np.nan, dtype=float),
        "global_distortion_max": np.full(n_obs, np.nan, dtype=float),
        "local_distortion_median": np.full(n_obs, np.nan, dtype=float),
        "local_distortion_p90": np.full(n_obs, np.nan, dtype=float),
        "local_distortion_max": np.full(n_obs, np.nan, dtype=float),
    }
    for i in range(n_obs):
        mask = src == i
        if not np.any(mask):
            continue
        ge = global_edge[mask]
        le = local_edge[mask]
        out["global_distortion_median"][i] = float(np.median(ge))
        out["global_distortion_p90"][i] = float(np.percentile(ge, 90.0))
        out["global_distortion_max"][i] = float(np.max(ge))
        out["local_distortion_median"][i] = float(np.median(le))
        out["local_distortion_p90"][i] = float(np.percentile(le, 90.0))
        out["local_distortion_max"][i] = float(np.max(le))
    return out


def _sample_units(values: np.ndarray, sample_values: Optional[np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, int]:
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    if sample_values is None:
        return vals, 0
    rows = []
    for sample in pd.unique(sample_values[mask]):
        sm = mask & (sample_values == sample)
        sample_vals = values[sm]
        sample_vals = sample_vals[np.isfinite(sample_vals)]
        if sample_vals.size:
            rows.append(float(np.mean(sample_vals)))
    return np.asarray(rows, dtype=float), len(rows)


def _summary_row(
    values: np.ndarray,
    *,
    rng: np.random.RandomState,
    n_boot: int,
    sample_values: Optional[np.ndarray],
    mask: np.ndarray,
) -> dict[str, float | int]:
    units, n_samples = _sample_units(values, sample_values, mask)
    n_cells = int(mask.sum())
    if units.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "n_cells": n_cells,
            "n_samples": n_samples if sample_values is not None else np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    ci_low = np.nan
    ci_high = np.nan
    if n_boot > 0:
        boot = np.empty(int(n_boot), dtype=float)
        for i in range(int(n_boot)):
            boot[i] = float(np.mean(units[rng.choice(units.size, size=units.size, replace=True)]))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(np.mean(units)),
        "median": float(np.median(units)),
        "std": float(np.std(units)),
        "p05": float(np.percentile(units, 5.0)),
        "p25": float(np.percentile(units, 25.0)),
        "p75": float(np.percentile(units, 75.0)),
        "p95": float(np.percentile(units, 95.0)),
        "n_cells": n_cells,
        "n_samples": n_samples if sample_values is not None else np.nan,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _state_status(n_cells: int, n_samples: Optional[int], k: int, sample_key_provided: bool) -> str:
    if n_cells == 0:
        return "missing"
    if n_cells < max(2, k):
        return "underpowered_cells"
    if sample_key_provided and (n_samples or 0) < 2:
        return "underpowered_samples"
    return "ok"


def _add_metric_summaries(
    rows: list[dict[str, Any]],
    metric_values: Mapping[str, np.ndarray],
    *,
    rep_a: str,
    rep_b: str,
    k: int,
    rng: np.random.RandomState,
    n_boot: int,
    sample_values: Optional[np.ndarray],
    node_values: Optional[np.ndarray],
    state_order: list[str],
    graph_edge_retention: float,
) -> None:
    n_obs = next(iter(metric_values.values())).shape[0]
    scopes = [("global", None, np.ones(n_obs, dtype=bool))]
    if node_values is not None:
        for state in state_order:
            scopes.append(("state", state, node_values == state))

    for scope, state, mask in scopes:
        n_samples = None if sample_values is None else len(pd.unique(sample_values[mask]))
        status = "ok" if scope == "global" else _state_status(int(mask.sum()), n_samples, k, sample_values is not None)
        for metric, values in metric_values.items():
            summary = _summary_row(values, rng=rng, n_boot=n_boot, sample_values=sample_values, mask=mask)
            rows.append(
                {
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "k": int(k),
                    "scope": scope,
                    "state": state,
                    "metric": metric,
                    "status": status,
                    "graph_edge_retention": graph_edge_retention,
                    **summary,
                }
            )


def _choose_subset(
    n_obs: int,
    max_exact_cells: int,
    rng: np.random.RandomState,
    stratify_values: Optional[np.ndarray],
) -> tuple[np.ndarray, bool]:
    if max_exact_cells <= 0:
        raise ValueError("max_exact_cells must be positive")
    if n_obs <= max_exact_cells:
        return np.arange(n_obs, dtype=np.int64), False
    if stratify_values is None:
        return np.sort(rng.choice(n_obs, size=max_exact_cells, replace=False)).astype(np.int64), True

    selected = []
    values, counts = np.unique(stratify_values, return_counts=True)
    raw = counts / counts.sum() * max_exact_cells
    alloc = np.floor(raw).astype(int)
    alloc[counts > 0] = np.maximum(alloc[counts > 0], 1)
    while alloc.sum() > max_exact_cells:
        candidates = np.where(alloc > 1)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(alloc[candidates])]
        alloc[idx] -= 1
    while alloc.sum() < max_exact_cells:
        idx = int(np.argmax(raw - alloc))
        alloc[idx] += 1
    for value, take in zip(values, alloc):
        idx = np.flatnonzero(stratify_values == value)
        take = min(int(take), idx.size)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(selected) > max_exact_cells:
        selected = rng.choice(np.asarray(selected), size=max_exact_cells, replace=False).tolist()
    return np.sort(np.asarray(selected, dtype=np.int64)), True


def _ordered_rank_summary(
    trustworthiness,
    arrays: Mapping[str, np.ndarray],
    pairs: list[tuple[str, str]],
    k_values: tuple[int, ...],
    *,
    subset_idx: np.ndarray,
    subset_sampled: bool,
    metric: str,
) -> pd.DataFrame:
    tw = {}
    rows = []
    ordered = []
    for rep_a, rep_b in pairs:
        ordered.append((rep_a, rep_b))
        ordered.append((rep_b, rep_a))
    for k in k_values:
        for rep_a, rep_b in ordered:
            if subset_idx.size <= 2 * k:
                value = np.nan
            else:
                value = float(
                    trustworthiness(
                        arrays[rep_a][subset_idx],
                        arrays[rep_b][subset_idx],
                        n_neighbors=int(k),
                        metric=metric,
                    )
                )
            tw[(rep_a, rep_b, k)] = value
        for rep_a, rep_b in ordered:
            rows.append(
                {
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "k": int(k),
                    "trustworthiness": tw[(rep_a, rep_b, k)],
                    "continuity": tw[(rep_b, rep_a, k)],
                    "n_subset": int(subset_idx.size),
                    "subset_sampled": bool(subset_sampled),
                }
            )
    return pd.DataFrame(rows)


def _state_transition(idx: np.ndarray, node_codes: np.ndarray, n_states: int) -> tuple[np.ndarray, float, float]:
    mat = np.zeros((n_states, n_states), dtype=float)
    src_codes = np.repeat(node_codes, idx.shape[1])
    dst_codes = node_codes[idx.reshape(-1)]
    np.add.at(mat, (src_codes, dst_codes), 1.0)
    total = float(mat.sum())
    within = float(np.trace(mat) / total) if total > 0 else np.nan
    between = float(1.0 - within) if np.isfinite(within) else np.nan
    row_sums = mat.sum(axis=1, keepdims=True)
    out = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    return out, within, between


def _spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    aa = pd.Series(a.ravel())
    bb = pd.Series(b.ravel())
    value = aa.corr(bb, method="spearman")
    return float(value) if value is not None else np.nan


def _state_graph_summary(
    knn_cache: Mapping[tuple[str, int], dict[str, np.ndarray | float]],
    pairs: list[tuple[str, str]],
    reps: list[str],
    k_values: tuple[int, ...],
    node_values: Optional[np.ndarray],
    state_order: list[str],
) -> Optional[pd.DataFrame]:
    if node_values is None:
        return None
    state_to_idx = {state: i for i, state in enumerate(state_order)}
    node_codes = np.asarray([state_to_idx[x] for x in node_values], dtype=np.int64)
    rows = []
    matrices = {}
    for rep in reps:
        for k in k_values:
            mat, within, between = _state_transition(knn_cache[(rep, k)]["idx"], node_codes, len(state_order))
            matrices[(rep, k)] = (mat, within, between)
    for rep_a, rep_b in pairs:
        for k in k_values:
            mat_a, within_a, between_a = matrices[(rep_a, k)]
            mat_b, within_b, between_b = matrices[(rep_b, k)]
            diff = np.abs(mat_b - mat_a)
            rows.append(
                {
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "k": int(k),
                    "spearman_r": _spearman_flat(mat_a, mat_b),
                    "mean_abs_diff": float(np.mean(diff)),
                    "max_abs_diff": float(np.max(diff)),
                    "within_state_fraction_a": within_a,
                    "within_state_fraction_b": within_b,
                    "between_state_fraction_a": between_a,
                    "between_state_fraction_b": between_b,
                }
            )
    return pd.DataFrame(rows)


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def _low_outlier(values: pd.Series, value: float, multiplier: float) -> bool:
    arr = values.to_numpy(dtype=float)
    med = float(np.nanmedian(arr))
    mad = _mad(arr)
    if not np.isfinite(value) or not np.isfinite(med) or not np.isfinite(mad):
        return False
    return bool(value < med - multiplier * mad)


def _high_outlier(values: pd.Series, value: float, multiplier: float) -> bool:
    arr = values.to_numpy(dtype=float)
    med = float(np.nanmedian(arr))
    mad = _mad(arr)
    if not np.isfinite(value) or not np.isfinite(med) or not np.isfinite(mad):
        return False
    return bool(value > med + multiplier * mad)


def _representation_summary(
    reps: list[str],
    pair_summary: pd.DataFrame,
    ordered_rank_summary: pd.DataFrame,
    state_graph_summary: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    global_rows = pair_summary[pair_summary["scope"] == "global"]
    for rep in reps:
        involved = global_rows[(global_rows["rep_a"] == rep) | (global_rows["rep_b"] == rep)]
        rank_rows = ordered_rank_summary[ordered_rank_summary["rep_a"] == rep]
        graph_rows = (
            pd.DataFrame()
            if state_graph_summary is None
            else state_graph_summary[(state_graph_summary["rep_a"] == rep) | (state_graph_summary["rep_b"] == rep)]
        )

        def med_metric(metric: str) -> float:
            vals = involved.loc[involved["metric"] == metric, "median"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.median(vals)) if vals.size else np.nan

        graph_vals = (
            np.asarray([], dtype=float)
            if graph_rows.empty
            else graph_rows["spearman_r"].to_numpy(dtype=float)
        )
        graph_vals = graph_vals[np.isfinite(graph_vals)]

        rows.append(
            {
                "rep": rep,
                "n_pairwise_comparisons": int(len([1 for a, b in combinations(reps, 2) if rep in (a, b)])),
                "median_pairwise_neighbor_overlap": med_metric("neighbor_overlap"),
                "median_pairwise_neighbor_jaccard": med_metric("neighbor_jaccard"),
                "median_global_distortion": med_metric("global_distortion_median"),
                "median_local_shape_distortion": med_metric("local_distortion_median"),
                "median_trustworthiness_to_others": float(np.nanmedian(rank_rows["trustworthiness"])) if not rank_rows.empty else np.nan,
                "median_continuity_from_others": float(np.nanmedian(rank_rows["continuity"])) if not rank_rows.empty else np.nan,
                "median_state_graph_spearman": float(np.median(graph_vals)) if graph_vals.size else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    multiplier = float(_DIAGNOSTIC_RULES["mad_multiplier"])
    out["insufficient_coverage"] = out["n_pairwise_comparisons"] < int(_DIAGNOSTIC_RULES["min_pairs_for_outlier"])
    out["neighborhood_outlier"] = [
        False if insufficient else _low_outlier(out["median_pairwise_neighbor_overlap"], value, multiplier)
        for value, insufficient in zip(out["median_pairwise_neighbor_overlap"], out["insufficient_coverage"])
    ]
    out["distortion_outlier"] = [
        False if insufficient else _high_outlier(out["median_global_distortion"], value, multiplier)
        for value, insufficient in zip(out["median_global_distortion"], out["insufficient_coverage"])
    ]
    if "median_state_graph_spearman" in out:
        out["state_graph_outlier"] = [
            False if insufficient else _low_outlier(out["median_state_graph_spearman"], value, multiplier)
            for value, insufficient in zip(out["median_state_graph_spearman"], out["insufficient_coverage"])
        ]
    else:
        out["state_graph_outlier"] = False
    return out


def local_geometry_stability(
    adata,
    *,
    reps,
    node_key=None,
    sample_key=None,
    k_values=(15, 30, 50),
    metric="euclidean",
    pair_mode="all",
    reference_rep=None,
    n_boot=500,
    max_exact_cells=3000,
    stratify_key=None,
    store_per_cell=False,
    seed=0,
    store_key="local_geometry_stability",
):
    """
    Quantify local-geometry preservation across latent representations.

    Definitions
    -----------
    For cell i and representation pair (a, b), neighborhood overlap is
    ``|N_a(i) intersect N_b(i)| / k`` and neighborhood Jaccard is
    ``|N_a(i) intersect N_b(i)| / |N_a(i) union N_b(i)|``. Directed graph
    edge retention is the Jaccard similarity of directed kNN edge sets.

    Distance distortion uses the union of directed kNN edges. For edge
    i -> j, global-scale distortion is
    ``abs(log((d_b / med_kNN_b + eps) / (d_a / med_kNN_a + eps)))``.
    Local-shape distortion replaces the global median with the source cell's
    local median kNN distance. These quantities are invariant to translation,
    orthogonal rotation, and isotropic rescaling, but not to anisotropic or
    nonlinear distortion.

    Trustworthiness and continuity are exact rank metrics from scikit-learn.
    When ``n_obs > max_exact_cells``, they are evaluated on a reproducible
    selected subset, optionally stratified by ``stratify_key`` or ``node_key``.
    """
    NearestNeighbors, trustworthiness = _import_sklearn()
    reps = _as_reps(reps)
    arrays = _load_representations(adata, reps)
    k_values = _validate_k_values(k_values, adata.n_obs)
    pairs = _representation_pairs(reps, str(pair_mode), reference_rep)
    if int(n_boot) < 0:
        raise ValueError("n_boot must be non-negative")

    if node_key is not None and node_key not in adata.obs:
        raise KeyError(f"obs key {node_key!r} not found")
    if sample_key is not None and sample_key not in adata.obs:
        raise KeyError(f"obs key {sample_key!r} not found")
    if stratify_key is not None and stratify_key not in adata.obs:
        raise KeyError(f"obs key {stratify_key!r} not found")

    node_values = adata.obs[node_key].astype(str).to_numpy() if node_key is not None else None
    state_order = list(pd.unique(node_values)) if node_values is not None else []
    sample_values = adata.obs[sample_key].astype(str).to_numpy() if sample_key is not None else None
    if stratify_key is not None:
        stratify_values = adata.obs[stratify_key].astype(str).to_numpy()
        subset_stratify_key = stratify_key
    elif node_values is not None:
        stratify_values = node_values
        subset_stratify_key = node_key
    else:
        stratify_values = None
        subset_stratify_key = None

    rng = np.random.RandomState(seed)
    subset_idx, subset_sampled = _choose_subset(adata.n_obs, int(max_exact_cells), rng, stratify_values)
    warnings: list[str] = []
    if subset_sampled:
        warnings.append(
            "Trustworthiness and continuity were evaluated on a reproducible subset "
            f"of {subset_idx.size} cells because n_obs exceeds max_exact_cells."
        )

    knn_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for rep, X in arrays.items():
        for k in k_values:
            idx, dist = _knn(NearestNeighbors, X, k, str(metric))
            knn_cache[(rep, k)] = {
                "idx": idx,
                "dist": dist,
                "global_median": _global_median_distance(dist),
                "local_median": _local_median_distance(dist, _global_median_distance(dist)),
            }

    pair_rows: list[dict[str, Any]] = []
    per_cell_rows = []
    for rep_a, rep_b in pairs:
        for k in k_values:
            idx_a = knn_cache[(rep_a, k)]["idx"]
            idx_b = knn_cache[(rep_b, k)]["idx"]
            dist_a = knn_cache[(rep_a, k)]["dist"]
            dist_b = knn_cache[(rep_b, k)]["dist"]
            overlap, jaccard = _neighbor_per_cell(idx_a, idx_b)
            edge_retention = _edge_jaccard(idx_a, idx_b, adata.n_obs)
            distortion = _distortion_per_cell(arrays[rep_a], arrays[rep_b], idx_a, idx_b, dist_a, dist_b)
            metric_values = {
                "neighbor_overlap": overlap,
                "neighbor_jaccard": jaccard,
                "graph_edge_retention": np.full(adata.n_obs, edge_retention, dtype=float),
                **distortion,
            }
            _add_metric_summaries(
                pair_rows,
                metric_values,
                rep_a=rep_a,
                rep_b=rep_b,
                k=k,
                rng=rng,
                n_boot=int(n_boot),
                sample_values=sample_values,
                node_values=node_values,
                state_order=state_order,
                graph_edge_retention=edge_retention,
            )
            if store_per_cell:
                for i in range(adata.n_obs):
                    per_cell_rows.append(
                        {
                            "cell": adata.obs_names[i],
                            "cell_index": i,
                            "node": None if node_values is None else node_values[i],
                            "sample": None if sample_values is None else sample_values[i],
                            "rep_a": rep_a,
                            "rep_b": rep_b,
                            "k": int(k),
                            **{name: values[i] for name, values in metric_values.items()},
                        }
                    )

    pair_summary = pd.DataFrame(pair_rows)
    ordered_rank_summary = _ordered_rank_summary(
        trustworthiness,
        arrays,
        pairs,
        k_values,
        subset_idx=subset_idx,
        subset_sampled=subset_sampled,
        metric=str(metric),
    )
    state_graph_summary = _state_graph_summary(knn_cache, pairs, reps, k_values, node_values, state_order)
    state_pair_summary = pair_summary[pair_summary["scope"] == "state"].copy() if node_values is not None else None
    representation_summary = _representation_summary(reps, pair_summary, ordered_rank_summary, state_graph_summary)

    coverage_summary = {
        "n_obs": int(adata.n_obs),
        "n_representations": len(reps),
        "n_pairs": len(pairs),
        "k_values": list(k_values),
        "pair_mode": pair_mode,
        "reference_rep": reference_rep,
        "rank_subset_n": int(subset_idx.size),
        "rank_subset_sampled": bool(subset_sampled),
        "rank_subset_stratify_key": subset_stratify_key,
        "n_states": len(state_order) if node_values is not None else None,
        "state_order": state_order if node_values is not None else None,
        "sample_key": sample_key,
        "n_samples": None if sample_values is None else int(pd.unique(sample_values).size),
    }

    out = {
        "params": {
            "reps": reps,
            "node_key": node_key,
            "sample_key": sample_key,
            "k_values": list(k_values),
            "metric": metric,
            "pair_mode": pair_mode,
            "reference_rep": reference_rep,
            "n_boot": int(n_boot),
            "max_exact_cells": int(max_exact_cells),
            "stratify_key": stratify_key,
            "store_per_cell": bool(store_per_cell),
            "seed": int(seed),
            "store_key": store_key,
            "diagnostic_rules": dict(_DIAGNOSTIC_RULES),
            "state_order": state_order if node_values is not None else None,
            "distance_distortion_eps": _EPS,
        },
        "representation_summary": representation_summary,
        "pair_summary": pair_summary,
        "ordered_rank_summary": ordered_rank_summary,
        "state_pair_summary": state_pair_summary,
        "state_graph_summary": state_graph_summary,
        "coverage_summary": coverage_summary,
        "warnings": warnings,
    }
    if store_per_cell:
        out["per_cell"] = pd.DataFrame(per_cell_rows)

    adata.uns.setdefault("scgeo", {})
    adata.uns["scgeo"][store_key] = out
    return out
