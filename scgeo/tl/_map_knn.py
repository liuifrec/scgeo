from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .._utils import _as_2d_array


def _vote_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    p = counts / (counts.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def map_knn(
    adata_ref,
    adata_q,
    label_key: str = "cell_type",
    rep: str = "X_pca",
    k: int = 25,
    ood_quantile: float = 0.99,
    out_label_key: str = "scgeo_label",
    out_conf_key: str = "scgeo_confidence",
    out_ent_key: str = "scgeo_entropy",
    out_ood_key: str = "scgeo_ood",
    store_key: str = "scgeo",
) -> None:
    """
    kNN mapping from reference to query with QC metrics.

    Writes to adata_q.obs:
      - out_label_key: predicted label
      - out_conf_key: vote fraction for predicted label
      - out_ent_key: vote entropy
      - out_ood_key: boolean, mean kNN distance > threshold

    Also stores metadata in adata_q.uns[store_key]['map_knn'].
    """
    if label_key not in adata_ref.obs:
        raise KeyError(f"ref.obs key '{label_key}' not found")
    if rep not in adata_ref.obsm:
        raise KeyError(f"ref.obsm['{rep}'] not found")
    if rep not in adata_q.obsm:
        raise KeyError(f"q.obsm['{rep}'] not found")

    Xr = _as_2d_array(adata_ref.obsm[rep])
    Xq = _as_2d_array(adata_q.obsm[rep])
    y = adata_ref.obs[label_key].astype(str).values

    if k <= 0 or k > Xr.shape[0]:
        raise ValueError(f"k must be in [1, n_ref], got k={k}, n_ref={Xr.shape[0]}")

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("scikit-learn required: pip install scgeo[sklearn]") from e

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(Xr)
    dists, idx = nn.kneighbors(Xq, return_distance=True)

    # OOD threshold from query mean distances
    mean_d = dists.mean(axis=1)
    thr = float(np.quantile(mean_d, ood_quantile))

    pred = []
    conf = np.empty(Xq.shape[0], dtype=np.float32)
    ent = np.empty(Xq.shape[0], dtype=np.float32)

    # Precompute label->int mapping for fast bincount
    uniq = np.unique(y)
    lab2i = {lab: i for i, lab in enumerate(uniq)}

    for i in range(Xq.shape[0]):
        labs = y[idx[i]]
        counts = np.zeros(len(uniq), dtype=np.int32)
        for lab in labs:
            counts[lab2i[lab]] += 1
        j = int(np.argmax(counts))
        pred_lab = uniq[j]
        pred.append(pred_lab)
        conf[i] = counts[j] / float(k)
        ent[i] = _vote_entropy(counts)

    adata_q.obs[out_label_key] = np.array(pred, dtype=object)
    adata_q.obs[out_conf_key] = conf
    adata_q.obs[out_ent_key] = ent
    adata_q.obs[out_ood_key] = (mean_d > thr)

    if store_key not in adata_q.uns:
        adata_q.uns[store_key] = {}
    adata_q.uns[store_key]["map_knn"] = {
        "params": dict(label_key=label_key, rep=rep, k=k, ood_quantile=ood_quantile),
        "ood_threshold_mean_dist": thr,
        "summary": {
            "mean_confidence": float(np.nanmean(conf)),
            "ood_rate": float(np.mean(mean_d > thr)),
        },
    }
