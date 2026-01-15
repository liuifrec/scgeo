from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

import scgeo as sg  # or import scgeo; then use scgeo.pl helpers


def mapping_confidence_umap(
    adata,
    conf_key: str = "map_confidence",
    *,
    basis: str = "umap",
    highlight_low_k: Optional[int] = 200,
    title: Optional[str] = None,
    show: bool = True,
):
    if conf_key not in adata.obs:
        raise KeyError(f"{conf_key} not found in adata.obs")

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    sg.pl.score_embedding(
        adata,
        conf_key,
        basis=basis,
        ax=ax,
        title=title or "Mapping confidence",
        show=False,
    )

    if highlight_low_k and highlight_low_k > 0:
        vals = np.asarray(adata.obs[conf_key])
        idx = np.argsort(vals)[: min(highlight_low_k, adata.n_obs)]
        xy = adata.obsm[f"X_{basis}"]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=6, linewidths=0, alpha=0.9)

    if show:
        plt.show()
    return fig, np.array([ax], dtype=object)


def ood_cells(
    adata,
    ood_key: str = "map_ood_score",
    *,
    basis: str = "umap",
    threshold: Optional[float] = None,
    show_only_flagged: bool = False,
    title: Optional[str] = None,
    show: bool = True,
):
    if ood_key not in adata.obs:
        raise KeyError(f"{ood_key} not found in adata.obs")

    vals = np.asarray(adata.obs[ood_key], dtype=float)
    xy = adata.obsm[f"X_{basis}"]

    if threshold is not None:
        flagged = vals >= threshold
    else:
        flagged = np.ones_like(vals, dtype=bool)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    if show_only_flagged and threshold is not None:
        ax.scatter(xy[flagged, 0], xy[flagged, 1], s=8, alpha=0.9)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=8, alpha=0.9)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title or "Out-of-distribution (OOD)")
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")

    if show:
        plt.show()
    return fig, np.array([ax], dtype=object)


def mapping_qc_panel(
    adata,
    *,
    pred_key: str = "map_pred",
    conf_key: str = "map_confidence",
    ood_key: str = "map_ood_score",
    basis: str = "umap",
    show: bool = True,
):
    for k in (pred_key, conf_key, ood_key):
        if k not in adata.obs:
            raise KeyError(f"{k} not found in adata.obs")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    # A) predicted label
    xy = adata.obsm[f"X_{basis}"]
    pred = adata.obs[pred_key].astype(str).to_numpy()
    # simple categorical coloring via integer codes (stable)
    _, codes = np.unique(pred, return_inverse=True)
    axes[0].scatter(xy[:, 0], xy[:, 1], c=codes, s=8, alpha=0.9)
    axes[0].set_title("Predicted label")
    axes[0].set_xlabel(f"{basis.upper()}1")
    axes[0].set_ylabel(f"{basis.upper()}2")

    # B) confidence
    sg.pl.score_embedding(adata, conf_key, basis=basis, ax=axes[1], title="Confidence", show=False)

    # C) OOD score
    sg.pl.score_embedding(adata, ood_key, basis=basis, ax=axes[2], title="OOD score", show=False)

    if show:
        plt.show()
    return fig, axes
