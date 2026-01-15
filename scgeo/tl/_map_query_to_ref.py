from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .._utils import row_normalize_csr, ref_mass_ratio_from_row



def _ensure_scgeo_uns(adata) -> Dict:
    if "scgeo" not in adata.uns or adata.uns["scgeo"] is None:
        adata.uns["scgeo"] = {}
    return adata.uns["scgeo"]


def _get_rep_matrix(adata, rep: str):
    if rep == "X":
        X = adata.X
    else:
        if rep not in adata.obsm:
            raise KeyError(f"rep '{rep}' not found in adata.obsm")
        X = adata.obsm[rep]
    # ensure dense numpy for mean computations (small dims, OK)
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    # stable entropy, p assumed >=0 and row-sums ~1
    eps = 1e-12
    q = np.clip(p, eps, 1.0)
    return -np.sum(q * np.log(q), axis=axis)


def _row_top2(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # returns top1, top2 values per row
    if p.shape[1] == 1:
        top1 = p[:, 0]
        top2 = np.zeros_like(top1)
        return top1, top2
    part = np.partition(p, kth=(-1, -2), axis=1)
    top2 = part[:, -2]
    top1 = part[:, -1]
    return top1, top2


def map_query_to_ref(
    adata,
    *,
    ref_key: str,
    ref_value: str,
    label_key: str,
    query_key: Optional[str] = None,
    query_value: Optional[str] = None,
    graph_key: str = "connectivities",
    # confidence / ood
    conf_method: str = "entropy_margin",  # "entropy" | "margin" | "entropy_margin"
    ood_method: str = "connectivity_mass",  # currently only this
    reject_conf: Optional[float] = None,
    reject_ood: Optional[float] = None,
    # optional label-deltas
    rep: Optional[str] = None,  # e.g. "X_pca"; if None, skip label_deltas
    store_key: str = "map_query_to_ref",
    # output column names
    pred_key: str = "scgeo_pred",
    conf_key: str = "scgeo_conf",
    conf_entropy_key: str = "scgeo_conf_entropy",
    conf_margin_key: str = "scgeo_conf_margin",
    ood_key: str = "scgeo_ood",
    reject_key: str = "scgeo_reject",
    # advanced
    return_probs: bool = False,
    probs_key: str = "X_map_probs",
    label_order_key: str = "map_label_order",
) -> None:
    """
    Graph-native reference mapping + QC that is agnostic to how the graph was built
    (BBKNN/Harmony/scVI/scanpy neighbors/...).

    Requirements:
      - adata.obs[ref_key] exists and defines reference cells.
      - adata.obs[label_key] exists for reference cells.
      - adata.obsp[graph_key] exists as a CSR/CSC sparse affinity matrix.

    Writes:
      adata.obs[pred_key]                predicted label for query cells (NaN for ref)
      adata.obs[conf_entropy_key]        1 - normalized entropy of label probs (query)
      adata.obs[conf_margin_key]         top1 - top2 probability margin (query)
      adata.obs[conf_key]                combined confidence (query)
      adata.obs[ood_key]                 OOD score (higher = more OOD-like) (query)
      adata.obs[reject_key]              reject flag (query) if thresholds set

      adata.uns["scgeo"][store_key]      params + per_label summary (+ optional label_deltas)

    Notes:
      - OOD is computed as 1 - (sum of graph weights to reference neighbors).
      - Confidence is derived from graph-weighted label voting among reference neighbors.
    """
    if ref_key not in adata.obs:
        raise KeyError(f"ref_key '{ref_key}' not found in adata.obs")
    if label_key not in adata.obs:
        raise KeyError(f"label_key '{label_key}' not found in adata.obs")
    if graph_key not in adata.obsp:
        raise KeyError(f"graph_key '{graph_key}' not found in adata.obsp")

    C = adata.obsp[graph_key]
    if not sparse.issparse(C):
        raise TypeError(f"adata.obsp['{graph_key}'] must be a scipy sparse matrix")
    C = C.tocsr()

    # Recommended: normalize rows so each row sums to ~1 (stable across graph builders)
    C = row_normalize_csr(C, inplace=False)

    obs = adata.obs
    is_ref = (obs[ref_key].astype(str).values == str(ref_value))

    if query_key is None:
        is_qry = ~is_ref
    else:
        if query_key not in obs:
            raise KeyError(f"query_key '{query_key}' not found in adata.obs")
        is_qry = (obs[query_key].astype(str).values == str(query_value))
        # sanity: ensure we don't overlap
        if np.any(is_qry & is_ref):
            raise ValueError("Query and reference masks overlap. Check ref/query definitions.")

    n_ref = int(is_ref.sum())
    n_qry = int(is_qry.sum())
    if n_ref == 0 or n_qry == 0:
        raise ValueError(f"Need both ref and query cells. n_ref={n_ref}, n_qry={n_qry}")

    # labels only for ref
    ref_labels = obs[label_key].astype(str).values
    ref_labels[~is_ref] = "__NONREF__"

    # define label set from ref only
    label_cats = sorted(pd.unique(ref_labels[is_ref]))
    L = len(label_cats)
    label_to_idx = {lab: i for i, lab in enumerate(label_cats)}

    # For query cells, we compute probability over labels via:
    # p_l(i) = sum_j C[i,j] for ref neighbors j with label l, then normalize.
    qry_idx = np.where(is_qry)[0]
    ref_mask = is_ref.astype(bool)
    probs = None
    if return_probs:
        probs = np.full((adata.n_obs, L), np.nan, dtype=np.float32)

    # outputs (full-length, fill NaN on ref)
    pred = np.full(adata.n_obs, np.nan, dtype=object)
    conf_entropy = np.full(adata.n_obs, np.nan, dtype=float)
    conf_margin = np.full(adata.n_obs, np.nan, dtype=float)
    conf = np.full(adata.n_obs, np.nan, dtype=float)
    ood = np.full(adata.n_obs, np.nan, dtype=float)
    reject = np.full(adata.n_obs, False, dtype=bool)
 

    # loop is fine for PBMC-scale; you can vectorize later if needed
    for i in qry_idx:
        row = C.getrow(i)
        if row.nnz == 0:
            # no neighbors: fully OOD / no confidence
            ood[i] = 1.0
            conf_entropy[i] = 0.0
            conf_margin[i] = 0.0
            conf[i] = 0.0
            pred[i] = np.nan
            continue

        cols = row.indices
        vals = row.data

        # keep only reference neighbors
        m = ref_mask[cols]
        if not np.any(m):
            ood[i] = 1.0
            conf_entropy[i] = 0.0
            conf_margin[i] = 0.0
            conf[i] = 0.0
            pred[i] = np.nan
            continue

        cols_ref = cols[m]
        vals_ref = vals[m]

        if ood_method != "connectivity_mass":
            raise ValueError("ood_method currently supports only 'connectivity_mass'")

        # After row normalization, ref_ratio is robust across graph builders.
        # m is mask aligned to row.indices (cols), perfect for helper.
        total_mass, ref_mass, ref_ratio = ref_mass_ratio_from_row(row, m)
        if total_mass <= 0:
            ood[i] = 1.0
        else:
            ood[i] = float(np.clip(1.0 - ref_ratio, 0.0, 1.0))

        # accumulate by label
        # accumulate by label (vectorized)
        # map ref neighbor indices -> label indices (0..L-1)
        neigh_lab = ref_labels[cols_ref]
        neigh_idx = np.fromiter((label_to_idx.get(l, -1) for l in neigh_lab), count=len(neigh_lab), dtype=np.int32)
        good = neigh_idx >= 0
        w = np.zeros(L, dtype=float)
        if np.any(good):
            np.add.at(w, neigh_idx[good], vals_ref[good].astype(float))

        if w.sum() <= 0:
            conf_entropy[i] = 0.0
            conf_margin[i] = 0.0
            conf[i] = 0.0
            pred[i] = np.nan
            continue

        p = w / w.sum()
        if return_probs:
            probs[i, :] = p.astype(np.float32)


        # pred
        k1 = int(np.argmax(p))
        pred[i] = label_cats[k1]

        # confidence
        H = _entropy(p[None, :], axis=1)[0]
        Hn = float(H / np.log(max(L, 2)))  # normalized entropy in [0,1]
        conf_entropy[i] = float(1.0 - Hn)

        top1, top2 = _row_top2(p[None, :])
        conf_margin[i] = float(top1[0] - top2[0])

        if conf_method == "entropy":
            conf[i] = conf_entropy[i]
        elif conf_method == "margin":
            conf[i] = conf_margin[i]
        elif conf_method == "entropy_margin":
            conf[i] = 0.5 * conf_entropy[i] + 0.5 * conf_margin[i]
        else:
            raise ValueError("conf_method must be one of: entropy, margin, entropy_margin")

        # reject (only for query cells)
        if reject_conf is not None or reject_ood is not None:
            r = False
            if reject_conf is not None:
                r = r or (conf[i] < float(reject_conf))
            if reject_ood is not None:
                r = r or (ood[i] > float(reject_ood))
            reject[i] = bool(r)

    # write obs
    adata.obs[pred_key] = pd.Series(pred, index=adata.obs.index, dtype="object")
    adata.obs[conf_entropy_key] = pd.Series(conf_entropy, index=adata.obs.index, dtype="float64")
    adata.obs[conf_margin_key] = pd.Series(conf_margin, index=adata.obs.index, dtype="float64")
    adata.obs[conf_key] = pd.Series(conf, index=adata.obs.index, dtype="float64")
    adata.obs[ood_key] = pd.Series(ood, index=adata.obs.index, dtype="float64")
    if reject_conf is not None or reject_ood is not None:
        adata.obs[reject_key] = pd.Series(reject, index=adata.obs.index, dtype="bool")
    if return_probs and probs is not None:
        # store full-length matrix (NaN for ref rows), plus label order in uns
        adata.obsm[probs_key] = probs

    # per-label summary on query cells only
    qry_obs = adata.obs.iloc[qry_idx]
    pred_q = qry_obs[pred_key].astype("object")
    out = []
    for lab in label_cats:
        m = (pred_q == lab).to_numpy()
        if m.sum() == 0:
            continue
        out.append(
            dict(
                label=lab,
                n=int(m.sum()),
                conf_mean=float(np.nanmean(qry_obs.loc[m, conf_key])),
                conf_entropy_mean=float(np.nanmean(qry_obs.loc[m, conf_entropy_key])),
                conf_margin_mean=float(np.nanmean(qry_obs.loc[m, conf_margin_key])),
                ood_mean=float(np.nanmean(qry_obs.loc[m, ood_key])),
            )
        )
    per_label = pd.DataFrame(out).sort_values("n", ascending=False) if len(out) else pd.DataFrame(
        columns=["label", "n", "conf_mean", "conf_entropy_mean", "conf_margin_mean", "ood_mean"]
    )
    # global summary (query cells only)
    global_summary = dict(
        n_qry=n_qry,
        n_ref=n_ref,
        conf_mean=float(np.nanmean(qry_obs[conf_key])),
        conf_entropy_mean=float(np.nanmean(qry_obs[conf_entropy_key])),
        conf_margin_mean=float(np.nanmean(qry_obs[conf_margin_key])),
        ood_mean=float(np.nanmean(qry_obs[ood_key])),
    )
    if (reject_conf is not None) or (reject_ood is not None):
        global_summary["reject_rate"] = float(np.nanmean(qry_obs[reject_key].astype(float)))



    per_label_dict = {}
    if len(per_label):
        for _, r in per_label.iterrows():
            per_label_dict[str(r["label"])] = dict(
                n=int(r["n"]),
                conf_mean=float(r["conf_mean"]),
                conf_entropy_mean=float(r["conf_entropy_mean"]),
                conf_margin_mean=float(r["conf_margin_mean"]),
                ood_mean=float(r["ood_mean"]),
            )

    scgeo = _ensure_scgeo_uns(adata)
    scgeo[store_key] = {
       "method": "graph_vote",
       "params": {
           "graph_key": graph_key,
           "ref_key": ref_key,
           "ref_value": str(ref_value),
           "query_key": query_key,
           "query_value": str(query_value) if query_value is not None else None,
           "label_key": label_key,
           "conf_method": conf_method,
           "ood_method": ood_method,
           "reject_conf": float(reject_conf) if reject_conf is not None else None,
           "reject_ood": float(reject_ood) if reject_ood is not None else None,
           "pred_key": pred_key,
           "conf_key": conf_key,
           "ood_key": ood_key,
           "return_probs": bool(return_probs),
           "probs_key": probs_key if return_probs else None,
       },
       "labels": label_cats,
        "global": global_summary,
       "per_label": per_label,            # DF for debugging/notebooks
       "per_label_dict": per_label_dict,  # stable for sg.get.table
   }

    # store label order for probabilities
    if return_probs:
        scgeo[store_key][label_order_key] = list(label_cats)


    # optional: label-wise deltas in a representation
    if rep is not None:
        X = _get_rep_matrix(adata, rep=rep)
        # compute deltas for labels where both ref and query exist
        label_deltas = {}
        for lab in label_cats:
            m_ref = is_ref & (ref_labels == lab)
            m_qry = is_qry & (adata.obs[pred_key].astype(str).values == lab)
            if m_ref.sum() < 2 or m_qry.sum() < 2:
                continue
            mu_ref = X[m_ref].mean(axis=0)
            mu_qry = X[m_qry].mean(axis=0)
            label_deltas[lab] = (mu_qry - mu_ref).astype(float)
        scgeo[store_key]["rep"] = rep
        scgeo[store_key]["label_deltas"] = label_deltas
