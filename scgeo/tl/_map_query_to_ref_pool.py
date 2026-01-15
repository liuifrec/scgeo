from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .._utils import _as_2d_array  # if you want; otherwise local helper
from ..pp import ReferencePool


def _ensure_scgeo_uns(adata) -> Dict:
    if "scgeo" not in adata.uns or adata.uns["scgeo"] is None:
        adata.uns["scgeo"] = {}
    return adata.uns["scgeo"]


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    eps = 1e-12
    q = np.clip(p, eps, 1.0)
    return -np.sum(q * np.log(q), axis=axis)


def _row_top2(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if p.shape[1] == 1:
        top1 = p[:, 0]
        top2 = np.zeros_like(top1)
        return top1, top2
    part = np.partition(p, kth=(-1, -2), axis=1)
    return part[:, -1], part[:, -2]


def _weights_from_dist(dist: np.ndarray, method: str = "inv") -> np.ndarray:
    """
    dist: (n, k) distances
    """
    eps = 1e-12
    if method == "inv":
        w = 1.0 / (dist + eps)
        return w
    if method == "softmax":
        # softmax over -dist row-wise
        z = -dist
        z = z - z.max(axis=1, keepdims=True)
        w = np.exp(z)
        w = w / (w.sum(axis=1, keepdims=True) + eps)
        return w
    raise ValueError("weight_method must be one of: inv, softmax")


def map_query_to_ref_pool(
    adata,
    pool: ReferencePool,
    *,
    rep: str,
    # mapping/QC config
    k: int = 30,
    weight_method: str = "inv",  # "inv" | "softmax"
    conf_method: str = "entropy_margin",
    # ood/reject
    ood_method: str = "distance",  # "distance" baseline OOD
    reject_conf: Optional[float] = None,
    reject_ood: Optional[float] = None,
    # store + outputs
    store_key: str = "map_query_to_ref",
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
    Embedding-only mapping using a ReferencePool (ANN index).

    Writes same obs/uns keys as graph-native map_query_to_ref so plotting + get.table work.
    """

    if rep == "X":
        Xq = adata.X
    else:
        if rep not in adata.obsm:
            raise KeyError(f"rep '{rep}' not found in adata.obsm")
        Xq = adata.obsm[rep]
    Xq = np.asarray(Xq)
    if hasattr(Xq, "toarray"):
        Xq = Xq.toarray()
    Xq = Xq.astype(np.float32, copy=False)
    if Xq.ndim != 2:
        raise ValueError(f"rep '{rep}' must be 2D, got {Xq.shape}")

    # pool label setup
    labels_ref = np.asarray(pool.obs[pool.label_key]).astype(str)
    label_cats = sorted(pd.unique(labels_ref))
    L = len(label_cats)
    label_to_idx = {lab: i for i, lab in enumerate(label_cats)}

    # ANN search
    idx, dist = pool.search(Xq, k=k)  # (n, k)
    w = _weights_from_dist(dist.astype(np.float64), method=weight_method)  # (n, k)

    # vote into probs
    probs = np.zeros((adata.n_obs, L), dtype=np.float32)
    for i in range(adata.n_obs):
        neigh = idx[i]
        wi = w[i]
        labs = labels_ref[neigh]
        li = np.fromiter((label_to_idx.get(x, -1) for x in labs), count=len(labs), dtype=np.int32)
        good = li >= 0
        if np.any(good):
            np.add.at(probs[i], li[good], wi[good].astype(np.float32))
        s = probs[i].sum()
        if s > 0:
            probs[i] /= s

    # pred/conf
    pred = np.array([label_cats[int(np.argmax(p))] if p.sum() > 0 else np.nan for p in probs], dtype=object)

    H = _entropy(probs.astype(np.float64), axis=1)
    Hn = H / np.log(max(L, 2))
    conf_entropy = (1.0 - Hn).astype(np.float64)

    top1, top2 = _row_top2(probs.astype(np.float64))
    conf_margin = (top1 - top2).astype(np.float64)

    if conf_method == "entropy":
        conf = conf_entropy
    elif conf_method == "margin":
        conf = conf_margin
    elif conf_method == "entropy_margin":
        conf = 0.5 * conf_entropy + 0.5 * conf_margin
    else:
        raise ValueError("conf_method must be one of: entropy, margin, entropy_margin")

    # OOD baseline: mean distance (scaled to 0..1 by robust normalization)
    if ood_method != "distance":
        raise ValueError("ood_method currently supports only 'distance'")
    dbar = dist.mean(axis=1).astype(np.float64)
    # robust scale: map [p5..p95] -> [0..1]
    lo = np.nanpercentile(dbar, 5)
    hi = np.nanpercentile(dbar, 95)
    denom = (hi - lo) if (hi > lo) else 1.0
    ood = np.clip((dbar - lo) / denom, 0.0, 1.0)

    reject = np.zeros(adata.n_obs, dtype=bool)
    if reject_conf is not None or reject_ood is not None:
        if reject_conf is not None:
            reject |= (conf < float(reject_conf))
        if reject_ood is not None:
            reject |= (ood > float(reject_ood))

    # write obs
    adata.obs[pred_key] = pd.Series(pred, index=adata.obs.index, dtype="object")
    adata.obs[conf_entropy_key] = pd.Series(conf_entropy, index=adata.obs.index, dtype="float64")
    adata.obs[conf_margin_key] = pd.Series(conf_margin, index=adata.obs.index, dtype="float64")
    adata.obs[conf_key] = pd.Series(conf, index=adata.obs.index, dtype="float64")
    adata.obs[ood_key] = pd.Series(ood, index=adata.obs.index, dtype="float64")
    if reject_conf is not None or reject_ood is not None:
        adata.obs[reject_key] = pd.Series(reject, index=adata.obs.index, dtype="bool")

    if return_probs:
        adata.obsm[probs_key] = probs
        scgeo = _ensure_scgeo_uns(adata)
        # store label order for interpretability
        # (same convention as graph-native version)
        scgeo.setdefault(store_key, {})
        scgeo[store_key][label_order_key] = list(label_cats)

    # per-label + global summaries (same structure as your graph-native)
    out = []
    for lab in label_cats:
        m = (adata.obs[pred_key].astype(str).values == lab)
        if m.sum() == 0:
            continue
        out.append(
            dict(
                label=lab,
                n=int(m.sum()),
                conf_mean=float(np.nanmean(conf[m])),
                conf_entropy_mean=float(np.nanmean(conf_entropy[m])),
                conf_margin_mean=float(np.nanmean(conf_margin[m])),
                ood_mean=float(np.nanmean(ood[m])),
            )
        )
    per_label = pd.DataFrame(out).sort_values("n", ascending=False) if len(out) else pd.DataFrame(
        columns=["label", "n", "conf_mean", "conf_entropy_mean", "conf_margin_mean", "ood_mean"]
    )

    global_summary = dict(
        n_qry=int(adata.n_obs),
        n_ref=int(pool.X.shape[0]),
        conf_mean=float(np.nanmean(conf)),
        conf_entropy_mean=float(np.nanmean(conf_entropy)),
        conf_margin_mean=float(np.nanmean(conf_margin)),
        ood_mean=float(np.nanmean(ood)),
    )
    if reject_conf is not None or reject_ood is not None:
        global_summary["reject_rate"] = float(np.nanmean(reject.astype(float)))

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
        "method": "pool_knn_vote",
        "params": {
            "rep": rep,
            "k": int(k),
            "weight_method": weight_method,
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
        "per_label": per_label,
        "per_label_dict": per_label_dict,
    }
    if return_probs:
        scgeo[store_key][label_order_key] = list(label_cats)
