from __future__ import annotations

from typing import Any, Optional

from ._paga_shift_map import paga_shift_map


def paga_scgeo(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    pie_key: Optional[str] = "timepoint",
    velocity_basis: Optional[str] = None,
    show_velocity: bool = True,
    node_color_mode: str = "delta",
    highlight_nodes: Optional[list[str]] = None,
    **kwargs,
):
    """Draw the PAGA shift map with composition and projected velocity enabled.

    Pies pool the two selected conditions; scalar colors remain visible as
    reference rims and comparison fills. Dark arrows show display-coordinate
    condition contrasts and teal arrows show mean projected velocity, with
    separate display multipliers recorded in the legend. These arrows describe
    geometry, not inferred trajectories. ``velocity_basis=None`` uses the same
    basis as the display and requires its matching projected velocity array.

    ``velocity_mode='grid'`` or ``'stream'`` replaces the state velocity arrows
    with a supported local projected field. ``velocity_options`` controls its
    neighborhood, support thresholds, and display style. Local directional
    agreement is descriptive and does not measure statistical confidence.

    Other options, including axes, label placement, stored alignment coloring,
    and ``return_result``, are forwarded to :func:`paga_shift_map`.
    """
    return paga_shift_map(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        paga_key=paga_key,
        pie_key=pie_key,
        velocity_basis=velocity_basis,
        show_velocity=show_velocity,
        node_color_mode=node_color_mode,
        highlight_nodes=highlight_nodes,
        **kwargs,
    )
