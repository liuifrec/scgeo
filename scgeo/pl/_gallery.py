from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import matplotlib.pyplot as plt

from ._composition_drift import composition_drift
from ._ood_landscape import ood_landscape
from ._paga_shift_map import paga_shift_map
from ._velocity_shift_alignment import velocity_shift_alignment


def gallery_overview(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    paga_key: str = "paga",
    ood_key: str = "scgeo_ood",
    velocity_basis: Optional[str] = None,
    figsize: tuple[float, float] = (16.0, 12.0),
    title: Optional[str] = None,
    axes: Optional[Mapping[str, Any]] = None,
    state_order=None,
    return_result: bool = False,
    show: bool = True,
):
    """Render four coordinated panels, including actual composition-change bars.

    ``axes`` may supply the named panels ``shift``, ``ood``, ``alignment``, and
    ``composition``. The alignment panel is explicitly a display-space overlay;
    use ``velocity_shift_alignment(mode='stored')`` for stored quantitative
    alignment. The legacy return is ``(fig, (ax1, ax2, ax3, ax4))``.
    """
    from ._result import PlotResult

    names = ("shift", "ood", "alignment", "composition")
    owned = axes is None
    if owned:
        fig, aa = plt.subplots(2, 2, figsize=figsize)
        panel_axes = dict(zip(names, aa.flat))
    else:
        if not isinstance(axes, Mapping) or any(name not in axes for name in names):
            raise ValueError(f"axes must contain {names}.")
        panel_axes = {name: axes[name] for name in names}
        fig = panel_axes[names[0]].figure
        if any(ax.figure is not fig for ax in panel_axes.values()):
            raise ValueError("All supplied axes must belong to the same figure.")
        if len({id(ax) for ax in panel_axes.values()}) != 4:
            raise ValueError("Each panel requires a distinct axis.")
    ax1, ax2, ax3, ax4 = (panel_axes[name] for name in names)
    paga_shift_map(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        paga_key=paga_key,
        ax=ax1,
        show=False,
    )
    ax1.set_title("PAGA shift map")
    ood_landscape(adata, ood_key=ood_key, basis=basis, ax=ax2, show=False)
    ax2.set_title("OOD landscape")
    alignment = velocity_shift_alignment(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        velocity_basis=velocity_basis,
        ax=ax3,
        return_result=True,
        show=False,
    )
    ax3.set_title("Display-space velocity alignment")
    composition = composition_drift(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        panels=("change",),
        axes={"change": ax4},
        state_order=state_order,
        return_result=True,
        show=False,
    )
    ax4.set_title("Composition change (cell fractions)")
    if owned:
        fig.suptitle(title or f"ScGeo gallery overview: {group0} → {group1}")
        fig.tight_layout()
    elif title is not None:
        ax1.set_title(title)
    if show:
        plt.show()
    if return_result:
        return PlotResult(
            fig=fig,
            axes=panel_axes,
            data={"composition": composition.data, "display_alignment": alignment.data},
            metadata={
                "plot": "gallery_overview",
                "basis": basis,
                "group0": str(group0),
                "group1": str(group1),
                "alignment_mode": "display",
                "composition_unit": "cells",
            },
        )
    return fig, (ax1, ax2, ax3, ax4)
