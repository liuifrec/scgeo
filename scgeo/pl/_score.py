from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _get_ax, display_indices, resolve_embedding


def score_embedding(
    adata,
    score_key: str,
    basis: str = "umap",
    *,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    size: float = 6.0,
    alpha: float = 0.8,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    norm=None,
    palette: Optional[Mapping[str, str]] = None,
    na_color: str = "lightgrey",
    figsize=None,
    max_points: Optional[int] = None,
    stratify_by: Optional[str] = None,
    random_state: int = 0,
    required_indices=None,
    rasterized="auto",
    return_data: bool = False,
    return_result: bool = False,
    show: bool = True,
):
    """Plot an observation score or category on the first two embedding axes.

    Both ``umap`` and ``X_umap`` resolve to the same coordinates. Continuous
    color limits use all finite input scores, even when ``max_points`` selects
    a reproducible display subset. ``stratify_by`` preserves at least one point
    per observed state; a budget too small for those states raises an error.
    ``required_indices`` preserves selected original cell positions when sampling;
    these cells must have finite display coordinates. Categorical scores use a
    stable label-to-color mapping and no colorbar.

    The legacy return is ``(fig, ax)``. ``return_data=True`` adds the full source
    table with a ``displayed`` indicator; ``return_result=True`` supplies a
    :class:`PlotResult` with the original displayed indices and random seed.
    The retained ``layer`` argument has no effect: scores come from ``obs``.
    """
    from ._result import PlotResult
    from ._style import category_palette

    emb_key, coords = resolve_embedding(adata, basis)
    if score_key not in adata.obs:
        raise KeyError(f"{score_key} not found in adata.obs")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if rasterized not in (True, False, "auto"):
        raise ValueError("rasterized must be True, False, or 'auto'.")
    scores = adata.obs[score_key]
    categorical = not pd.api.types.is_numeric_dtype(
        scores.dtype
    ) or pd.api.types.is_bool_dtype(scores.dtype)
    if stratify_by is not None and stratify_by not in adata.obs:
        raise KeyError(f"{stratify_by} not found in adata.obs")
    labels = (
        adata.obs[stratify_by]
        if stratify_by is not None
        else (scores if categorical else pd.Series("all", index=adata.obs.index))
    )
    eligible = np.flatnonzero(np.isfinite(coords[:, :2]).all(axis=1))
    if not len(eligible):
        raise ValueError("No finite embedding coordinates are available to plot.")
    required_relative = None
    if required_indices is not None:
        required = np.asarray(required_indices)
        if required.ndim != 1 or (required.size and required.dtype.kind not in "iu"):
            raise ValueError(
                "required_indices must be a one-dimensional sequence of integer cell positions."
            )
        required = required.astype(int)
        if np.any(required < 0) or np.any(required >= adata.n_obs):
            raise ValueError("required_indices contains an out-of-range cell position.")
        if not np.isin(required, eligible).all():
            raise ValueError("Required cells must have finite display coordinates.")
        required_relative = np.searchsorted(eligible, required)
    relative = display_indices(
        labels.iloc[eligible].to_numpy(),
        max_points=max_points,
        random_state=random_state,
        required_indices=required_relative,
    )
    displayed = eligible[relative]
    rasterize = len(displayed) > 10000 if rasterized == "auto" else bool(rasterized)
    fig, ax = _get_ax(ax=ax, figsize=figsize)
    source = pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "value": scores.to_numpy(),
            "displayed": False,
        }
    )
    source.loc[displayed, "displayed"] = True
    if stratify_by is not None:
        source["stratum"] = labels.to_numpy()
    artists = {}
    metadata = {
        "plot": "score_embedding",
        "score_key": score_key,
        "basis": emb_key,
        "categorical": categorical,
        "random_state": int(random_state),
        "max_points": max_points,
        "stratify_by": stratify_by,
        "rasterized": rasterize,
        "required_indices": [] if required_indices is None else required.tolist(),
        "displayed_indices": displayed.tolist(),
        "n_total": len(source),
        "n_displayed": len(displayed),
        "color_scale_uses_full_data": True,
        "marker_size": size,
        "alpha": alpha,
        "na_color": na_color,
    }
    if categorical:
        if norm is not None or vmin is not None or vmax is not None:
            raise ValueError(
                "norm and numeric color limits cannot be used with categorical scores."
            )
        categories = (
            list(scores.cat.categories.astype(str))
            if isinstance(scores.dtype, pd.CategoricalDtype)
            else list(pd.unique(scores.dropna().astype(str)))
        )
        colors = category_palette(categories, palette=palette)
        values = scores.iloc[displayed]
        missing = values.isna().to_numpy()
        if missing.any():
            artists["missing"] = ax.scatter(
                coords[displayed[missing], 0],
                coords[displayed[missing], 1],
                s=size,
                alpha=alpha,
                c=na_color,
                linewidths=0,
                rasterized=rasterize,
                label="Missing",
            )
        for category in categories:
            mask = (~missing) & values.astype(str).eq(category).to_numpy()
            if mask.any():
                artists[category] = ax.scatter(
                    coords[displayed[mask], 0],
                    coords[displayed[mask], 1],
                    s=size,
                    alpha=alpha,
                    c=[colors[category]],
                    linewidths=0,
                    rasterized=rasterize,
                    label=category,
                )
        ax.legend(frameon=False, title=score_key, markerscale=1.5)
        metadata["palette"] = colors
    else:
        values = pd.to_numeric(scores, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        finite = np.isfinite(values)
        if norm is None:
            finite_values = values[finite]
            low = float(np.min(finite_values)) if len(finite_values) else 0.0
            high = float(np.max(finite_values)) if len(finite_values) else 1.0
            norm = plt.Normalize(
                vmin=low if vmin is None else vmin, vmax=high if vmax is None else vmax
            )
        else:
            # Initialize an unset caller-supplied norm using the full score array.
            norm.autoscale_None(values[finite])
        bad = displayed[~finite[displayed]]
        good = displayed[finite[displayed]]
        if len(bad):
            artists["missing"] = ax.scatter(
                coords[bad, 0],
                coords[bad, 1],
                s=size,
                alpha=alpha,
                c=na_color,
                linewidths=0,
                rasterized=rasterize,
            )
        artists["scores"] = ax.scatter(
            coords[good, 0],
            coords[good, 1],
            s=size,
            alpha=alpha,
            c=values[good],
            cmap=cmap,
            norm=norm,
            linewidths=0,
            rasterized=rasterize,
        )
        fig.colorbar(artists["scores"], ax=ax).set_label(score_key)
        metadata["cmap"] = artists["scores"].get_cmap().name
        metadata["norm"] = {
            "type": type(norm).__name__,
            "vmin": norm.vmin,
            "vmax": norm.vmax,
        }
    basis_name = emb_key.removeprefix("X_")
    ax.set(
        xlabel=f"{basis_name.upper()}1",
        ylabel=f"{basis_name.upper()}2",
        title=title or score_key,
    )
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes={"embedding": ax},
            artists=artists,
            data=source,
            metadata=metadata,
        )
    if return_data:
        return fig, ax, source
    return fig, ax


def score_umap(adata, score_key: str, **kwargs):
    return score_embedding(adata, score_key=score_key, basis="umap", **kwargs)
