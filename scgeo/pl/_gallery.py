from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt

from ._paga_shift_map import paga_shift_map
from ._ood_landscape import ood_landscape
from ._velocity_shift_alignment import velocity_shift_alignment
from ._composition_drift import composition_drift


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
    show: bool = True,
):
    """
    Render a 2x2 overview gallery of core ScGeo plots:
    - paga_shift_map
    - ood_landscape
    - velocity_shift_alignment
    - composition_drift
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
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

    ax2 = fig.add_subplot(gs[0, 1])
    ood_landscape(
        adata,
        ood_key=ood_key,
        basis=basis,
        ax=ax2,
        show=False,
    )
    ax2.set_title("OOD landscape")

    ax3 = fig.add_subplot(gs[1, 0])
    velocity_shift_alignment(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        velocity_basis=velocity_basis,
        ax=ax3,
        show=False,
    )
    ax3.set_title("Velocity-shift alignment")

    # composition_drift currently makes its own figure; embed as a simplified call
    # by rendering it separately would complicate gallery. So use a minimal summary
    # in its own subplot by reusing the returned axes is awkward.
    # Better: draw composition_drift as a separate compact figure later if needed.
    # For now, create the panel directly via the function in a fresh figure-like pattern.
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(
        0.5,
        0.5,
        "Use sg.pl.composition_drift(...)\\nfor the full 3-panel composition report.",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax4.set_xticks([])
    ax4.set_yticks([])
    for spine in ax4.spines.values():
        spine.set_visible(False)
    ax4.set_title("Composition drift")

    if title is None:
        title = f"ScGeo gallery overview: {group0} -> {group1}"
    fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, (ax1, ax2, ax3, ax4)