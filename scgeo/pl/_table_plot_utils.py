"""Small validation and axes helpers for table-first evidence figures."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import pandas as pd


def ordered_labels(values, order, name):
    observed = list(dict.fromkeys(str(value) for value in values if pd.notna(value)))
    if order is None:
        return observed
    if isinstance(order, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of labels, not a string.")
    labels = [str(value) for value in order]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"{name} must contain unique labels and must not be empty.")
    omitted = set(observed) - set(labels)
    if omitted:
        raise ValueError(f"{name} omits observed labels {sorted(omitted)}; filter the input table explicitly.")
    return labels


def panel_axes(names, axes, figsize, *, sharex=False, sharey=False):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if axes is None:
        fig, array = plt.subplots(1, len(names), figsize=figsize, squeeze=False,
                                  sharex=sharex, sharey=sharey)
        return fig, dict(zip(names, array[0])), True
    if not isinstance(axes, Mapping):
        raise TypeError("axes must be a mapping from panel labels to Matplotlib Axes.")
    missing = [name for name in names if name not in axes]
    if missing:
        raise ValueError(f"axes is missing panels: {missing}")
    selected = {name: axes[name] for name in names}
    if any(not isinstance(ax, Axes) for ax in selected.values()):
        raise TypeError("axes values must be Matplotlib Axes.")
    figures = {id(ax.figure): ax.figure for ax in selected.values()}
    if len(figures) != 1 or len({id(ax) for ax in selected.values()}) != len(names):
        raise ValueError("Each panel needs a distinct Axes in the same Figure.")
    return next(iter(figures.values())), selected, False


def required_table(data, columns):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input table must contain at least one row.")
    copied = data.copy(deep=True)
    copied.attrs = deepcopy(data.attrs)
    return copied
