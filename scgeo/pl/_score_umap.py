from __future__ import annotations

import matplotlib.pyplot as plt


def score_umap(adata, score_key: str, *, title: str | None = None, s: float = 6.0):
    if "X_umap" not in adata.obsm:
        raise KeyError("obsm['X_umap'] not found. Run sc.tl.umap first.")
    if score_key not in adata.obs:
        raise KeyError(f"obs['{score_key}'] not found")

    X = adata.obsm["X_umap"]
    v = adata.obs[score_key].values

    fig = plt.figure()
    ax = plt.gca()
    sc = ax.scatter(X[:, 0], X[:, 1], c=v, s=s)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title or f"UMAP colored by {score_key}")
    plt.colorbar(sc, ax=ax, label=score_key)
    return ax
