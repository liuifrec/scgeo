from __future__ import annotations

from typing import Optional

from ._score import _score_distribution_panel


def consensus_subspace_panel(
    adata,
    *,
    score_key: str = "cs_score",
    basis: str = "umap",
    store_key: str = "consensus_subspace",
    topk: Optional[int] = 200,
    figsize=(11, 5),
    show: bool = True,
    axes=None,
    bins=40,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: bool = True,
    max_points=None,
    random_state=0,
    rasterized="auto",
    return_data: bool = False,
    return_result: bool = False,
):
    """Plot stored consensus scores, their histogram, and existing singular values.

    Uses the same composition and source-export contract as ``alignment_panel``:
    named ``embedding``/``histogram`` axes, optional ``colorbar`` axis,
    a shared color scale for highlights, and draw-only sampling. The histogram
    uses all finite scores; singular values are read from the selected stored
    consensus result and never recomputed.

    The default return remains ``(fig, axes_array)``; ``return_data=True`` adds
    source tables, and ``return_result=True`` returns an exportable PlotResult.
    """
    obj = adata.uns.get("scgeo", {}).get(store_key, {})
    singular_values = obj.get("singular_values") if isinstance(obj, dict) else None
    return _score_distribution_panel(
        adata, score_key, basis=basis, topk=topk, cmap="viridis", figsize=figsize,
        title="Consensus subspace score", axes=axes, bins=bins, norm=norm,
        vmin=vmin, vmax=vmax, colorbar=colorbar, max_points=max_points,
        random_state=random_state, rasterized=rasterized,
        plot_name="consensus_subspace_panel", return_data=return_data,
        return_result=return_result, show=show, singular_values=singular_values,
        store_key=store_key)
