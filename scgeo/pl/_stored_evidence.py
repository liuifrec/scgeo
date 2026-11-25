"""Validate recorded provenance for table-driven evidence displays.

Metadata agreement is not proof that a result is fresh or that an arbitrary
velocity array was transformed correctly. Caller declarations are retained as
declarations; they never manufacture missing source provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_ALIASES = {
    "analysis_rep": ("analysis_rep", "rep", "representation", "effect_representation"),
    "n_dimensions": ("n_dimensions", "n_dims", "n_pcs", "representation_dimensions"),
    "scale": ("scale", "representation_scale", "analysis_scale", "coordinate_scale"),
    "preprocessing": ("preprocessing", "representation_preprocessing"),
    "node_key": ("node_key", "by", "groupby"),
    "condition_key": ("condition_key",),
    "group0": ("group0", "reference_group", "reference_condition"),
    "group1": ("group1", "comparison_group", "comparison_condition"),
    "center": ("center", "center_estimator"),
    "scope": ("scope", "analysis_scope"),
    "reference": ("reference",),
    "graph_mode": ("graph_mode",),
    "sample_key": ("sample_key",),
    "sample_weighting": ("sample_weighting", "shift_weighting", "weighting"),
    "aggregation_unit": ("aggregation_unit",),
    "velocity_scope": ("velocity_scope",),
    "velocity_aggregation": ("velocity_aggregation",),
    "velocity_weighting": ("velocity_weighting",),
    "effect_normalization": ("effect_normalization", "normalize_by"),
}
_CONTAINERS = {
    "params",
    "provenance",
    "source_params",
    "analysis",
    "metadata",
    "analysis_metadata",
    "comparison",
    "comparison_identity",
}
_IDENTITY_FIELDS = ("analysis_rep", "group0", "group1")
_STRING_FIELDS = set(_ALIASES) - {"n_dimensions", "scale", "preprocessing"}


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return False
    if np.isscalar(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def _plain(value: Any) -> Any:
    """Return detached JSON-compatible metadata, without changing its meaning."""
    if _missing(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Stored provenance values must be finite or missing.")
        return value
    raise TypeError(f"Unsupported stored provenance value {type(value).__name__}.")


def _canonical(field: str, value: Any, source: str) -> Any:
    value = _plain(value)
    if field == "n_dimensions":
        if isinstance(value, bool):
            raise ValueError(f"{source} must record a positive integer dimension.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} must record a positive integer dimension."
            ) from exc
        if not np.isfinite(number) or number < 1 or number != int(number):
            raise ValueError(f"{source} must record a positive integer dimension.")
        return int(number)
    if field in _STRING_FIELDS:
        if isinstance(value, (Mapping, list)):
            raise ValueError(f"{source} must record one scalar {field!r} value.")
        return str(value)
    return value


def _table_records(frame: pd.DataFrame, name: str) -> dict[str, list[dict[str, Any]]]:
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicate column names.")
    records: dict[str, list[dict[str, Any]]] = {field: [] for field in _ALIASES}

    def append(field, value, source):
        if not _missing(value):
            normalized = _canonical(field, value, source)
            if not any(
                row["value"] == normalized and row["source"] == source
                for row in records[field]
            ):
                records[field].append({"source": source, "value": normalized})

    for field, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in frame:
                for value in frame[alias].tolist():
                    append(field, value, f"{name}.columns[{alias!r}]")

    seen = set()

    def read_mapping(mapping, path, depth=0):
        if id(mapping) in seen or depth > 8:
            return
        seen.add(id(mapping))
        for field, aliases in _ALIASES.items():
            for alias in aliases:
                if alias in mapping:
                    append(field, mapping[alias], f"{path}[{alias!r}]")
        for key in sorted(_CONTAINERS):
            child = mapping.get(key)
            if isinstance(child, Mapping):
                read_mapping(child, f"{path}[{key!r}]", depth + 1)

    read_mapping(frame.attrs, f"{name}.attrs")
    return records


def validate_stored_evidence(
    shift_df,
    alignment_df=None,
    *,
    node_key=None,
    condition_key=None,
    group0=None,
    group1=None,
    analysis_rep=None,
    strict=False,
):
    """Check compatible recorded provenance without mutating source tables.

    The function returns ``(provenance, warnings)``. It reads named metadata
    columns and nested ``attrs`` mappings such as ``provenance``, ``params``
    and ``source_params``. Contradictions within a table, between tables, or
    with the supplied expected values always raise. Unrecorded fields remain
    explicitly unverified; ``sample_key`` does not imply sample weighting.

    ``strict=True`` requires a resolved analysis representation and ordered
    ``group0``/``group1`` contrast, from recorded metadata or explicit caller
    declarations. Declarations alone receive ``declared_unverified`` status.
    ``recorded_consistent`` means only that the required identity metadata was
    recorded consistently in every supplied table; it does not prove source
    freshness, correct coordinate transforms, or biological validity.
    """
    if not isinstance(shift_df, pd.DataFrame):
        raise TypeError("shift_df must be a pandas DataFrame.")
    if alignment_df is not None and not isinstance(alignment_df, pd.DataFrame):
        raise TypeError("alignment_df must be a pandas DataFrame or None.")
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict must be a boolean.")
    frames = {"shift_df": shift_df}
    if alignment_df is not None:
        frames["alignment_df"] = alignment_df
    records = {name: _table_records(frame, name) for name, frame in frames.items()}
    declarations = {
        "node_key": node_key,
        "condition_key": condition_key,
        "group0": group0,
        "group1": group1,
        "analysis_rep": analysis_rep,
    }
    declarations = {
        field: _canonical(field, value, f"declared {field}")
        for field, value in declarations.items()
        if not _missing(value)
    }
    resolved, statuses, sources = {}, {}, {}
    table_provenance = {name: {} for name in frames}
    for field in _ALIASES:
        observed = []
        for name, table_records in records.items():
            field_records = table_records[field]
            if field_records:
                first = field_records[0]["value"]
                if any(item["value"] != first for item in field_records[1:]):
                    descriptions = ", ".join(
                        f"{r['source']}={r['value']!r}" for r in field_records
                    )
                    raise ValueError(
                        f"Conflicting stored provenance for {field!r}: {descriptions}."
                    )
                table_provenance[name][field] = first
                observed.extend(field_records)
        expected = declarations.get(field)
        if expected is not None:
            observed.append({"source": f"declared {field}", "value": expected})
        if observed:
            first = observed[0]["value"]
            if any(item["value"] != first for item in observed[1:]):
                descriptions = ", ".join(
                    f"{r['source']}={r['value']!r}" for r in observed
                )
                raise ValueError(
                    f"Conflicting stored provenance for {field!r}: {descriptions}."
                )
            resolved[field] = first
        else:
            resolved[field] = None
        sources[field] = observed
        n_recorded = sum(field in source for source in table_provenance.values())
        statuses[field] = (
            "recorded_consistent"
            if n_recorded == len(frames)
            else "partially_recorded"
            if n_recorded
            else "declared_unverified"
            if field in declarations
            else "unavailable"
        )

    if resolved["group0"] is not None and resolved["group0"] == resolved["group1"]:
        raise ValueError("Stored evidence requires distinct ordered group0 and group1.")
    missing_fields = [field for field in _IDENTITY_FIELDS if resolved[field] is None]
    if strict and missing_fields:
        raise ValueError(
            "Stored evidence requires explicit or recorded analysis_rep, group0 and group1; "
            f"missing {missing_fields}."
        )
    identity_status = [statuses[field] for field in _IDENTITY_FIELDS]
    if all(status == "recorded_consistent" for status in identity_status):
        status = "recorded_consistent"
    elif any(
        status in {"partially_recorded", "recorded_consistent"}
        for status in identity_status
    ):
        status = "partially_recorded"
    elif not missing_fields:
        status = "declared_unverified"
    else:
        status = "unverified"
    unverified_fields = [
        field for field, value in statuses.items() if value != "recorded_consistent"
    ]
    warnings = []
    if status != "recorded_consistent":
        warnings.append(
            "Source provenance is incomplete; caller declarations do not verify unrecorded "
            "representation or contrast metadata."
        )
    if statuses["sample_weighting"] != "recorded_consistent":
        warnings.append(
            "Sample weighting is not recorded consistently in every source table; no weighting is inferred."
        )
    rep = resolved["analysis_rep"]
    if rep is not None and any(
        token in rep.lower() for token in ("umap", "tsne", "t_sne", "display")
    ):
        warnings.append(
            f"Declared analysis representation {rep!r} appears to be a display embedding; "
            "stored values retain that coordinate-system interpretation."
        )
    provenance = {
        **resolved,
        "status": status,
        "field_status": statuses,
        "table_provenance": table_provenance,
        "sources": sources,
        "declarations": declarations,
        "missing_fields": missing_fields,
        "unverified_fields": unverified_fields,
        "strict": bool(strict),
        "verification_scope": "Recorded metadata consistency only; no source-freshness or coordinate-transform verification.",
    }
    return provenance, warnings
