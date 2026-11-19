from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ._result import PlotResult
from ._table_plot_utils import panel_axes
from ._score import score_embedding
from ._density_overlap import _by_dict_to_df


def _get_embedding_xy(adata, basis: str = "umap"):
    key = f"X_{basis}" if not basis.startswith("X_") else basis
    if key not in adata.obsm:
        raise KeyError(f"{key} not found in adata.obsm")
    X = np.asarray(adata.obsm[key])
    if X.shape[1] < 2:
        raise ValueError(f"{key} must have >=2 dims, got {X.shape}")
    return X[:, 0], X[:, 1]


def distribution_report(
    adata,
    *,
    basis: str = "umap",
    condition_key: str = "condition",
    group0: Optional[str] = None,
    group1: Optional[str] = None,
    density_key: str = "density_overlap",
    test_key: str = "distribution_test",
    score_key: Optional[str] = None,         # e.g. "cs_score" or "align_velocity_to_delta"
    top_k: int = 8,
    figsize=(11, 8),
    show: bool = True,
    axes=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """
    2×2 report panel for cross-condition embedding comparison.

    Panels:
      (A) embedding colored by condition
      (B) density overlap summary (global + worst groups)
      (C) distribution test summary (global + strongest groups)
      (D) optional score histogram (or empty text if score_key None)

    Requires tl outputs stored in adata.uns['scgeo'] under density_key/test_key.
    Named axes are ``embedding``, ``density``, ``test`` and ``score``.
    ``return_result=True`` exports source cells, stored summaries, exact text
    and histogram bins without recalculating tests. Default returns remain
    ``(fig, axes_2d_array)``; ``return_data=True`` adds these source tables.
    """
    if "scgeo" not in adata.uns:
        raise KeyError("adata.uns['scgeo'] not found")

    x, y = _get_embedding_xy(adata, basis=basis)

    # condition vector
    if condition_key not in adata.obs:
        raise KeyError(f"{condition_key} not found in adata.obs")
    cond = adata.obs[condition_key].astype(str).to_numpy()

    uniq = list(dict.fromkeys(cond))
    if group0 is None:
        group0 = uniq[0] if uniq else "0"
    if group1 is None:
        group1 = uniq[-1] if uniq else "1"

    owns_figure = axes is None
    names = ["embedding", "density", "test", "score"]
    if owns_figure:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        mapped = dict(zip(names, axes.flat))
    else:
        fig, mapped, _ = panel_axes(names, axes, figsize)
        axes = np.asarray(list(mapped.values()), dtype=object).reshape(2, 2)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    # --- (A) embedding colored by condition
    # simple categorical coloring via matplotlib
    # One renderer supplies both points and their actual categorical legend.
    # Numeric condition codes are still labels, so use an observation-only view.
    from types import SimpleNamespace
    obs = adata.obs[[condition_key]].copy()
    obs[condition_key] = obs[condition_key].astype("category")
    view = SimpleNamespace(obs=obs, obsm=adata.obsm, uns=adata.uns,
                           obs_names=adata.obs_names, n_obs=adata.n_obs)
    embedding = score_embedding(view, condition_key, basis=basis, ax=axA,
                                colorbar=False, return_result=True, show=False, size=5)
    axA.set_title(f"{basis.upper()} colored by {condition_key}\n({group0} vs {group1})")
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_aspect("equal", adjustable="datalim")

    # --- (B) density overlap summary text
    dobj = adata.uns["scgeo"].get(density_key, {})
    g = dobj.get("global", {})
    bc = g.get("bc", None)
    hel = g.get("hellinger", None)
    n0 = g.get("n0", None)
    n1 = g.get("n1", None)

    lines = [f"[{density_key}] global"]
    if bc is not None:
        lines.append(f"  bc: {bc:.4g}" if isinstance(bc, (int, float)) else f"  bc: {bc}")
    if hel is not None:
        lines.append(f"  hellinger: {hel:.4g}" if isinstance(hel, (int, float)) else f"  hellinger: {hel}")
    if n0 is not None and n1 is not None:
        lines.append(f"  n0={n0}, n1={n1}")

    by = dobj.get("by", None)
    if isinstance(by, dict) and len(by) > 0:
        # rank worst by bc (low) if present, else by hellinger (high)
        if all(isinstance(v, dict) for v in by.values()) and any("bc" in v for v in by.values() if isinstance(v, dict)):
            items = [(k, v.get("bc", np.nan)) for k, v in by.items() if isinstance(v, dict)]
            items = sorted(items, key=lambda t: t[1])[:top_k]
            lines.append(f"\nworst-mixed groups (by bc low):")
            for k, val in items:
                lines.append(f"  {k}: {val:.3g}")
        elif all(isinstance(v, dict) for v in by.values()) and any("hellinger" in v for v in by.values() if isinstance(v, dict)):
            items = [(k, v.get("hellinger", np.nan)) for k, v in by.items() if isinstance(v, dict)]
            items = sorted(items, key=lambda t: -t[1])[:top_k]
            lines.append(f"\nmost-different groups (by hellinger high):")
            for k, val in items:
                lines.append(f"  {k}: {val:.3g}")

    axB.axis("off")
    axB.set_title("Density overlap summary")
    axB.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")

    # --- (C) distribution test summary text
    tobj = adata.uns["scgeo"].get(test_key, {})
    tg = tobj.get("global", {})
    stat = tg.get("stat", None)
    p = tg.get("p_perm", None)
    nperm = tg.get("n_perm", None)
    tn0 = tg.get("n0", None)
    tn1 = tg.get("n1", None)

    tlines = [f"[{test_key}] global"]
    if stat is not None:
        tlines.append(f"  stat: {stat:.4g}" if isinstance(stat, (int, float)) else f"  stat: {stat}")
    if p is not None:
        tlines.append(f"  p_perm: {p:.3g}" if isinstance(p, (int, float)) else f"  p_perm: {p}")
    if tn0 is not None and tn1 is not None:
        tlines.append(f"  n0={tn0}, n1={tn1}")
    if nperm is not None:
        tlines.append(f"  n_perm={nperm}")

    tby = tobj.get("by", None)
    if isinstance(tby, dict) and len(tby) > 0:
        items = []
        for k, v in tby.items():
            if isinstance(v, dict) and "p_perm" in v:
                items.append((k, v.get("p_perm", np.nan)))
        if items:
            items = sorted(items, key=lambda t: t[1])[:top_k]
            tlines.append(f"\nstrongest groups (p_perm low):")
            for k, val in items:
                tlines.append(f"  {k}: {val:.3g}")

    axC.axis("off")
    axC.set_title("Distribution test summary")
    axC.text(0.02, 0.98, "\n".join(tlines), va="top", family="monospace")

    # --- (D) optional score distribution
    histogram = pd.DataFrame(columns=["left", "right", "count"])
    if score_key is None or score_key not in adata.obs:
        axD.axis("off")
        axD.set_title("Score panel")
        if score_key is None:
            axD.text(0.02, 0.98, "score_key=None\n(optional: cs_score / align_* / disagree_*)", va="top")
        else:
            axD.text(0.02, 0.98, f"{score_key} not found in adata.obs", va="top")
    else:
        s = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)
        s = s[np.isfinite(s)]
        counts, edges, _ = axD.hist(s, bins=40)
        histogram = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts.astype(int)})
        axD.set_title(f"Score distribution: {score_key}")
        axD.set_xlabel(score_key)
        axD.set_ylabel("count")

    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    cells = embedding.data.copy(deep=True)
    if score_key is not None and score_key in adata.obs:
        cells["score"] = adata.obs[score_key].to_numpy()
    tables = {"cells": cells, "density_global": pd.DataFrame([g]), "test_global": pd.DataFrame([tg]),
              "density_by": _by_dict_to_df(by) if isinstance(by, dict) else pd.DataFrame(),
              "test_by": _by_dict_to_df(tby) if isinstance(tby, dict) else pd.DataFrame(),
              "histogram": histogram,
              "summary_text": pd.DataFrame({"panel": ["density", "test"], "text": ["\n".join(lines), "\n".join(tlines)]})}
    if return_result:
        return PlotResult(fig, mapped, tables, {"plot": "distribution_report", "basis": basis,
                          "condition_key": condition_key, "group0": group0, "group1": group1,
                          "density_key": density_key, "test_key": test_key, "score_key": score_key,
                          "palette": embedding.metadata.get("palette"), "analysis_recomputed": False},
                          artists={"embedding": embedding.artists})
    if return_data:
        return fig, axes, tables
    return fig, axes
