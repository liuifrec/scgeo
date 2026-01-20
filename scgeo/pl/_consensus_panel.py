from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ._score import score_embedding
from ._highlight import highlight_topk_cells


def consensus_subspace_panel(
    adata,
    *,
    score_key: str = "cs_score",
    basis: str = "umap",
    store_key: str = "consensus_subspace",
    topk: Optional[int] = 200,
    figsize=(11, 5),
    show: bool = True,
):
    """
    Panel for consensus subspace:
      (A) embedding colored by cs_score
      (B) histogram of cs_score + optional text for singular values
    """
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axA, axB = axes

    score_embedding(adata, score_key, basis=basis, ax=axA, title="Consensus subspace score", show=False)
    if topk is not None and topk > 0:
        highlight_topk_cells(adata, score_key, basis=basis, topk=int(topk), add_colorbar=False, ax=axA, show=False)

    s = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)
    s = s[np.isfinite(s)]
    axB.hist(s, bins=40)
    axB.set_title("cs_score distribution")
    axB.set_xlabel(score_key)

    # add a small text summary if present
    sv = None
    if "scgeo" in adata.uns and store_key in adata.uns["scgeo"]:
        obj = adata.uns["scgeo"][store_key]
        if isinstance(obj, dict):
            sv = obj.get("singular_values", None)
    if sv is not None:
        axB.text(0.98, 0.98, f"singular_values:\n{np.array(sv)[:5]}", ha="right", va="top", transform=axB.transAxes, fontsize=9)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
