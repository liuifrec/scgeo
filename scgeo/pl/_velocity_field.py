"""Local projected-velocity grids and support-masked streamlines."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ._result import PlotResult
from ._style import category_palette
from ._utils import display_indices
from ._velocity_grid import (
    _field_style,
    _labels,
    _positive_number,
    _prepare_velocity_field,
    _render_velocity_field,
)


def velocity_field(
    adata,
    *,
    basis="umap",
    velocity_basis=None,
    mode="grid",
    condition_key=None,
    groups=None,
    background_key=None,
    palette=None,
    background="all",
    max_points=None,
    random_state=0,
    required_indices=None,
    grid_size=25,
    grid_extent=None,
    n_neighbors=30,
    radius=None,
    min_neighbors=5,
    min_coherence=0.2,
    scale="auto",
    color="#008C95",
    color_by=None,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    arrow_linewidth=1.1,
    arrow_head_size=5.5,
    stream_density=1.0,
    stream_linewidth=1.2,
    background_size=5.0,
    background_alpha=0.18,
    alpha=0.9,
    aspect="equal",
    xlim=None,
    ylim=None,
    show_colorbar=True,
    cax=None,
    legend=True,
    legend_loc="outside",
    ax=None,
    figsize=(7.0, 6.0),
    title=None,
    return_data=False,
    return_result=False,
    show=True,
):
    """Plot local summaries of already projected velocity on an embedding.

    The input must contain a matching ``velocity_<basis>`` array with exactly
    the display embedding shape. Only the first two display coordinates and
    velocity components are used. This function does not project vectors from
    another representation or estimate RNA kinetics, trajectories, or time.

    ``condition_key`` and ``groups`` select the field's source cells; ``groups``
    omitted means all cells. Coordinates and velocity are filtered jointly for
    finite rows. Background scope is independent: ``'all'`` or ``'selected'``.
    ``background_key`` colors categorical cell labels with a stable palette.
    ``max_points`` affects background drawing only. The field always uses every
    eligible selected cell, with deterministic background sampling by category
    and condition and optional required observation positions.

    A regular ``grid_size`` (integer or ``(nx, ny)``) summarizes at most
    ``n_neighbors`` nearest cells inside a finite ``radius``. Gaussian weights
    use sigma=radius/2. The default radius is 1.5 times the median source-cell
    kth-neighbor distance. ``grid_extent=(xmin,xmax,ymin,ymax)`` fixes evaluation
    coordinates across panels; otherwise full selected-source bounds are used.
    ``xlim`` and ``ylim`` change only the view. A grid position is supported when
    its count reaches ``min_neighbors``, its mean vector is nonzero, and its
    coherence reaches ``min_coherence``. Coherence is the norm of the weighted
    mean vector divided by the weighted mean individual norm: a directional
    agreement diagnostic, not a confidence level or significance test.

    ``mode='grid'`` draws arrows with line widths and heads in points. ``scale``
    is a positive multiplier on the raw mean vectors or ``'auto'``; their lengths
    remain projected vectors times that scale. The resolved multiplier is
    explicit in the legend and metadata. ``mode='stream'`` traces geometric direction paths,
    masks unsupported regions, and audits the rendered segments for support.
    Scale does not apply to streamlines and never represents elapsed time.
    ``color_by='speed'`` uses the norm of the local mean; ``'coherence'`` uses its
    directional agreement. ``norm`` or ``vmin``/``vmax`` support shared scales.

    Supplied axes retain their figure size and already fixed limits. Use an
    explicit ``cax`` for a shared colorbar (orientation follows its shape), or
    ``show_colorbar=False``. Owned figures return ``(fig, ax)``, supplied axes
    return ``ax``, and ``return_data=True`` returns ``(fig, ax, data)``. Opt-in
    ``return_result=True`` returns a :class:`PlotResult` with a named ``field``
    axis, all source cells and grid rows, display flags, scaled vector endpoints,
    actual streamline path vertices, support rules, and plotting provenance.
    """
    if background not in {"all", "selected"}:
        raise ValueError("background must be 'all' or 'selected'.")
    if aspect not in {"equal", "auto"}:
        raise ValueError("aspect must be 'equal' or 'auto'.")
    background_size = _positive_number(background_size, "background_size")
    if (
        not np.isscalar(background_alpha)
        or not np.isfinite(background_alpha)
        or not 0 <= background_alpha <= 1
    ):
        raise ValueError("background_alpha must be between zero and one.")
    for name, bounds in (("xlim", xlim), ("ylim", ylim)):
        if bounds is not None:
            values = np.asarray(bounds, dtype=float)
            if (
                values.shape != (2,)
                or not np.isfinite(values).all()
                or values[0] >= values[1]
            ):
                raise ValueError(f"{name} must contain two finite increasing bounds.")
    if (
        cax is not None
        and show_colorbar
        and color_by is not None
        and (ax is None or cax.figure is not ax.figure)
    ):
        raise ValueError("cax requires a supplied field ax in the same figure.")
    if background_key is not None and background_key not in adata.obs:
        raise KeyError(f"{background_key!r} is missing from adata.obs.")
    prepared = _prepare_velocity_field(
        adata,
        basis=basis,
        velocity_basis=velocity_basis,
        condition_key=condition_key,
        groups=groups,
        grid_size=grid_size,
        grid_extent=grid_extent,
        n_neighbors=n_neighbors,
        radius=radius,
        min_neighbors=min_neighbors,
        min_coherence=min_coherence,
    )
    render_options = {
        "mode": mode,
        "scale": scale,
        "color": color,
        "color_by": color_by,
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "arrow_linewidth": arrow_linewidth,
        "arrow_head_size": arrow_head_size,
        "stream_density": stream_density,
        "stream_linewidth": stream_linewidth,
        "alpha": alpha,
    }
    # Validate style before creating a canvas or changing a caller's axes.
    _field_style(prepared, **render_options)
    cells = prepared["cells"].copy(deep=True)
    scope = cells.finite_coordinates & (
        cells.selected if background == "selected" else True
    )
    eligible_display = np.flatnonzero(scope.to_numpy())
    if required_indices is None:
        required = np.empty(0, dtype=int)
    else:
        raw = np.asarray(required_indices)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError(
                "required_indices must contain integer observation positions."
            )
        required = np.unique(raw.astype(int))
        if (
            np.any(required < 0)
            or np.any(required >= adata.n_obs)
            or not np.isin(required, eligible_display).all()
        ):
            raise ValueError(
                "required_indices must identify finite cells within the chosen background scope."
            )
    cells["background_scope"] = scope
    missing_label, background_palette = None, {}
    if background_key is None:
        cells["background_label"] = None
        cells["background_color"] = "#B8C1C9"
    else:
        cells["background_label"] = _labels(
            adata.obs[background_key], background_key
        ).to_numpy()
        labels = cells.background_label.copy()
        categories = labels[scope & labels.notna()].unique().tolist()
        if labels[scope].isna().any():
            missing_label = "(missing)"
            while missing_label in categories:
                missing_label = f"({missing_label})"
            categories.append(missing_label)
            labels = labels.fillna(missing_label)
        background_palette = category_palette(categories, palette)
        if missing_label is not None and palette is None:
            background_palette[missing_label] = "#AEB5BB"
        cells["background_color"] = labels.map(background_palette)
    strata = np.empty(len(eligible_display), dtype=object)
    strata[:] = list(
        zip(
            cells.background_label.iloc[eligible_display],
            cells.condition.iloc[eligible_display],
        )
    )
    positions = display_indices(
        strata, max_points, random_state, np.searchsorted(eligible_display, required)
    )
    shown = eligible_display[positions]
    cells["displayed"] = False
    cells.loc[shown, "displayed"] = True
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        if xlim is None and not ax.get_autoscalex_on():
            xlim = ax.get_xlim()
        if ylim is None and not ax.get_autoscaley_on():
            ylim = ax.get_ylim()
    axes = {"field": ax}
    artists = {
        "background": ax.scatter(
            cells.x.iloc[shown],
            cells.y.iloc[shown],
            s=background_size,
            c=cells.background_color.iloc[shown].tolist(),
            alpha=background_alpha,
            linewidths=0,
            rasterized=True,
            zorder=0,
        )
    }
    rendered = _render_velocity_field(ax, prepared, **render_options)
    artists.update(rendered["artists"])
    limits = np.vstack(
        [cells.loc[scope, ["x", "y"]].to_numpy(), rendered["extent_points"]]
    )
    if not len(limits):
        limits = cells.loc[cells.eligible, ["x", "y"]].to_numpy()
    for name, supplied, dimension in (("x", xlim, 0), ("y", ylim, 1)):
        low, high = limits[:, dimension].min(), limits[:, dimension].max()
        padding = max(float(high - low) * 0.05, prepared["metadata"]["radius"] * 0.1)
        getattr(ax, f"set_{name}lim")(
            (low - padding, high + padding) if supplied is None else supplied
        )
    ax.set_aspect(aspect, adjustable="box")
    basis_name = prepared["metadata"]["display_basis"].removeprefix("X_").upper()
    ax.set(
        xlabel=f"{basis_name} 1",
        ylabel=f"{basis_name} 2",
        xticks=[],
        yticks=[],
        title=title
        or (
            "Local projected velocity"
            if mode == "grid"
            else "Projected velocity streamlines"
        ),
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_label, orientation = None, None
    if color_by is not None and show_colorbar:
        if cax is not None:
            box = cax.get_window_extent()
            orientation = "horizontal" if box.width > 2 * box.height else "vertical"
        else:
            orientation = (
                "horizontal" if legend and legend_loc == "outside" else "vertical"
            )
        colorbar = fig.colorbar(
            rendered["mappable"],
            ax=ax if cax is None else None,
            cax=cax,
            orientation=orientation,
            fraction=0.035,
            pad=0.075 if orientation == "horizontal" else 0.025,
        )
        colorbar_label = (
            "Local mean speed (projected units)"
            if color_by == "speed"
            else "Directional coherence (diagnostic)"
        )
        colorbar.set_label(colorbar_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        axes["colorbar"], artists["colorbar"] = colorbar.ax, colorbar
    legend_handles = list(rendered["legend_handles"])
    legend_handles.extend(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=value,
            markersize=5,
            label=f"{background_key}: {key}",
        )
        for key, value in background_palette.items()
    )
    if legend:
        placement = (
            {"loc": "upper left", "bbox_to_anchor": (1.02, 1)}
            if legend_loc == "outside"
            else {"loc": legend_loc}
        )
        artists["legend"] = ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=8,
            labelspacing=0.6,
            **placement,
        )
    if owns_figure:
        fig.tight_layout()
    data = {"cells": cells, **rendered["data"]}
    metadata = dict(rendered["metadata"])
    metadata.update(
        {
            "plot": "velocity_field",
            "background_scope": background,
            "background_key": background_key,
            "background_palette": background_palette,
            "background_missing_label": missing_label,
            "background_size": background_size,
            "background_alpha": float(background_alpha),
            "max_points": max_points,
            "random_state": int(random_state),
            "required_indices": required.tolist(),
            "displayed_indices": shown.tolist(),
            "n_cells_displayed": len(shown),
            "sampling_strata": "background_category_by_condition",
            "aspect": aspect,
            "xlim": list(ax.get_xlim()),
            "ylim": list(ax.get_ylim()),
            "legend": legend,
            "legend_loc": legend_loc,
            "legend_labels": [handle.get_label() for handle in legend_handles],
            "show_colorbar": show_colorbar,
            "colorbar_label": colorbar_label,
            "colorbar_orientation": orientation,
        }
    )
    notes = list(prepared["warnings"]) + list(rendered["warnings"])
    if (
        ~cells.finite_coordinates
        & (cells.selected if background == "selected" else True)
    ).any():
        notes.append(
            "Background cells with nonfinite display coordinates were omitted from drawing and retained in the source table."
        )
    result = PlotResult(
        fig=fig,
        axes=axes,
        data=data,
        metadata=metadata,
        warnings=notes,
        artists=artists,
    )
    if show:
        plt.show()
    if return_result:
        return result
    if return_data:
        return fig, ax, data
    return (fig, ax) if owns_figure else ax
