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
    velocity_basis: Optional[str] = "umap",
    show_velocity: bool = True,
    node_color_mode: str = "delta",
    highlight_nodes: Optional[list[str]] = None,
    **kwargs,
):
    """
    ScGeo-style PAGA summary:
    - graph edges from PAGA
    - node pies from pie_key (e.g. timepoint)
    - black arrows = geometry shift
    - optional cyan arrows = mean velocity
    - node color by delta/alignment/palette/constant
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