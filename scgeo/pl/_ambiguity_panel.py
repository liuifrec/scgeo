from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ._score import score_embedding
from ._highlight import highlight_topk_cells


def ambiguity_panel(
    adata,
    score_key: str,
    *,
    basis: str = "umap",
    topk: Optional[int] = 200,
    cmap: str = "inferno",
    figsize=(11, 5),
    title: Optional[str] = None,
    show: bool = True,
):
    """
    1×2 panel:
      (A) embedding colored by ambiguity/disagreement score
      (B) histogram of ambiguity scores
    Optional: highlight top-k ambiguous cells.
    """
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axA, axB = axes

    score_embedding(
        adata,
        score_key,
        basis=basis,
        ax=axA,
        cmap=cmap,
        title=title or f"Ambiguity score: {score_key}",
        show=False,
    )
    if topk is not None and topk > 0:
        highlight_topk_cells(
            adata,
            score_key,
            basis=basis,
            topk=int(topk),
            ax=axA,
            show=False,
        )

    s = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)
    s = s[np.isfinite(s)]
    axB.hist(s, bins=40)
    axB.set_title("Score distribution")
    axB.set_xlabel(score_key)
    axB.set_ylabel("count")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
