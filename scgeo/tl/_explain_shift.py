"""Descriptive molecular accounting for a stored, sample-aware displacement.

The affine map is verified against the actual expression matrix. Attribution
uses the numerical center weights, not ordinary expression means substituted
for a robust center. No causal inference or gene-level hypothesis test is made.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import sparse

from ._robust_shift import _EPS, _get_representation


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _labels(adata, key, positions):
    if key not in adata.obs:
        raise KeyError(f"obs[{key!r}] is missing.")
    values = adata.obs[key].iloc[positions]
    if values.isna().any():
        raise ValueError(
            f"Missing values in {key!r}; missing labels cannot be samples or states."
        )
    unique = list(pd.unique(values))
    if len({str(x) for x in unique}) != len(unique):
        raise ValueError(f"Colliding string labels in {key!r}.")
    return values.astype(str).to_numpy()


def _input_signature(adata, positions, layer, rep):
    """Hash selected numerical inputs in bounded canonical sparse row chunks.

    The same dense/CSR values have the same signature. Sorting a temporary
    sparse chunk never mutates a caller matrix. The complete expression matrix
    is never densified. Signatures are guards against stale plotting inputs,
    not proof of scientific reproducibility or a replacement for provenance.
    """
    expression = adata.X if layer is None else adata.layers[layer]
    representation = adata.obsm[rep]
    expressions, coordinates = hashlib.sha256(), hashlib.sha256()
    for value in adata.var_names.astype(str):
        encoded = value.encode("utf-8")
        expressions.update(len(encoded).to_bytes(8, "little"))
        expressions.update(encoded)
    for start in range(0, len(positions), 1024):
        rows = positions[start : start + 1024]
        block = sparse.csr_matrix(expression[rows], dtype=np.float64, copy=True)
        block.sum_duplicates()
        block.eliminate_zeros()
        block.sort_indices()
        expressions.update(np.asarray(block.shape, dtype="<i8").tobytes())
        expressions.update(block.indptr.astype("<i8", copy=False).tobytes())
        expressions.update(block.indices.astype("<i8", copy=False).tobytes())
        expressions.update(block.data.astype("<f8", copy=False).tobytes())
        block_z = representation[rows]
        if sparse.issparse(block_z):
            block_z = block_z.toarray()
        coordinates.update(np.asarray(block_z, dtype="<f8").tobytes())
    return {
        "expression_sha256": expressions.hexdigest(),
        "representation_sha256": coordinates.hexdigest(),
        "n_genes": adata.n_vars,
        "n_coordinates": representation.shape[1],
    }


def _check_input_signature(adata, bundle):
    """Reject stale expression/geometry before using a stored explanation plot."""
    expected = bundle.get("input_signature")
    if expected is None:
        return
    params = bundle["params"]
    layer, rep = params.get("layer"), params["rep"]
    if rep not in adata.obsm or (layer is not None and layer not in adata.layers):
        raise ValueError(
            "Explanation source matrix is missing; recompute explain_shift."
        )
    positions = bundle["cells"]["obs_position"].to_numpy(dtype=int)
    actual = _input_signature(adata, positions, layer, rep)
    if actual != dict(expected):
        raise ValueError(
            "Explanation expression or representation has changed; recompute explain_shift before plotting."
        )


def _center_weights(points, estimator):
    """Mirror the existing finite-tolerance center and retain its last weights.

    The legacy estimator is intentionally unchanged. The epsilon-degenerate
    initial-median branch has only an approximate barycentric representation;
    its reconstruction residual is checked and exported by the caller.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if not n:
        raise ValueError("Cannot explain an empty center.")
    if estimator == "mean":
        weights = np.full(n, 1.0 / n)
        return points.mean(axis=0), weights
    if n == 1:
        return points[0].copy(), np.ones(1)
    y = np.median(points, axis=0)
    weights = np.full(n, 1.0 / n)
    for _ in range(512):
        dist = np.linalg.norm(points - y, axis=1)
        if np.all(dist <= _EPS):
            return y, weights
        raw = 1.0 / np.maximum(dist, _EPS)
        # Preserve the exact arithmetic/order of robust_shift's estimator.
        y_next = (points * raw[:, None]).sum(axis=0) / raw.sum()
        weights = raw / raw.sum()
        if np.linalg.norm(y_next - y) <= 1e-7 * max(1.0, np.linalg.norm(y)):
            return y_next, weights
        y = y_next
    return y, weights


def _weighted_expression(X, positions, weights):
    return np.asarray(X[positions].T @ weights, dtype=float).reshape(-1)


def _group_center(X, Z, positions, sample_labels, estimator, has_samples):
    dimension, n_genes = Z.shape[1], X.shape[1]
    if not len(positions):
        return {
            "center": np.full(dimension, np.nan),
            "expression": np.full(n_genes, np.nan),
            "weights": np.empty(0),
            "positions": positions,
            "sample_ids": [],
            "sample_counts": np.empty(0, dtype=int),
            "sample_centers": np.empty((0, dimension)),
            "sample_expression": np.empty((0, n_genes)),
        }
    ids = list(pd.unique(sample_labels)) if has_samples else ["__pooled__"]
    centers, expressions, within_weights, within_indices, counts = [], [], [], [], []
    for sample in ids:
        index = (
            np.flatnonzero(sample_labels == sample)
            if has_samples
            else np.arange(len(positions))
        )
        center, weights = _center_weights(Z[positions[index]], estimator)
        centers.append(center)
        expressions.append(_weighted_expression(X, positions[index], weights))
        within_weights.append(weights)
        within_indices.append(index)
        counts.append(len(index))
    centers, expressions = np.asarray(centers), np.asarray(expressions)
    if has_samples:
        center, between = _center_weights(centers, estimator)
    else:
        center, between = centers[0], np.ones(1)
    weights = np.zeros(len(positions))
    for index, within, coefficient in zip(within_indices, within_weights, between):
        weights[index] = within * coefficient
    return {
        "center": center,
        "expression": between @ expressions,
        "weights": weights,
        "positions": positions,
        "sample_ids": ids,
        "sample_counts": np.asarray(counts, dtype=int),
        "sample_centers": centers,
        "sample_expression": expressions,
    }


def _verify_source(source, groups, has_samples, rtol, atol, state):
    """Reject a cached result whose selected cells/sample centers have changed."""
    present = [group["sample_centers"] for group in groups if len(group["positions"])]
    combined = np.vstack(present) if present else np.zeros((1, 1))
    scale = float(np.max(np.abs(combined - combined.mean(axis=0))))
    absolute_tolerance = atol + rtol * scale
    for suffix, group in enumerate(groups):
        if int(source.get(f"n{suffix}", -1)) != len(group["positions"]):
            raise ValueError(
                f"Stored shift is stale for state {state!r}: cell counts changed; rerun robust_shift."
            )
        if not len(group["positions"]):
            continue
        # robust_shift deliberately omits both aggregate centers if either
        # condition is absent, but retains the available sample centers.
        old_value = source.get(f"center{suffix}")
        if all(len(item["positions"]) for item in groups) or old_value is not None:
            old = np.asarray(old_value, dtype=float)
            expected = group["center"].astype(np.float32).astype(float)
            if old.shape != expected.shape or not np.allclose(
                old, expected, rtol=0, atol=absolute_tolerance
            ):
                raise ValueError(
                    f"Stored shift is stale for state {state!r}: centers changed; rerun robust_shift."
                )
        if has_samples:
            old_ids = [str(x) for x in source.get(f"sample_ids{suffix}", [])]
            if set(old_ids) != set(group["sample_ids"]):
                raise ValueError(
                    f"Stored shift is stale for state {state!r}: sample IDs changed."
                )
            order = [group["sample_ids"].index(sample) for sample in old_ids]
            old_centers = np.asarray(
                source.get(f"sample_centers{suffix}", []), dtype=float
            )
            current = group["sample_centers"][order]
            counts = np.asarray(source.get(f"sample_cell_counts{suffix}", []))
            if (
                old_centers.shape != current.shape
                or not np.allclose(
                    old_centers, current, rtol=0, atol=absolute_tolerance
                )
                or not np.array_equal(counts, group["sample_counts"][order])
            ):
                raise ValueError(
                    f"Stored shift is stale for state {state!r}: sample centers or counts changed."
                )
    if all(len(group["positions"]) for group in groups):
        delta = groups[1]["center"] - groups[0]["center"]
        old = np.asarray(source.get("delta", []), dtype=float)
        expected_delta = delta.astype(np.float32).astype(float)
        if old.shape != delta.shape or not np.allclose(
            old, expected_delta, rtol=0, atol=absolute_tolerance
        ):
            raise ValueError(
                f"Stored shift is stale for state {state!r}: displacement changed."
            )
        if not np.isclose(
            source.get("delta_norm", np.nan),
            np.linalg.norm(delta),
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                f"Stored shift is stale for state {state!r}: magnitude changed."
            )


def _affine_mapping(X, Z, positions, loadings, scale, rtol, atol):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effective = loadings / scale[:, None]
    if not np.isfinite(effective).all():
        raise ValueError("Feature scaling produced non-finite effective loadings.")
    # Work in bounded dense coordinate chunks; never densify the count matrix.
    intercept_sum = np.zeros(Z.shape[1])
    zsum = np.zeros(Z.shape[1])
    for start in range(0, len(positions), 4096):
        rows = positions[start : start + 4096]
        values = X[rows]
        finite = values.data if sparse.issparse(values) else np.asarray(values)
        if not np.isfinite(finite).all():
            raise ValueError("Selected expression contains non-finite values.")
        projected = np.asarray(values @ effective)
        if not np.isfinite(projected).all():
            raise ValueError("Affine projection produced non-finite coordinates.")
        intercept_sum += (Z[rows] - projected).sum(axis=0)
        zsum += Z[rows].sum(axis=0)
    intercept = intercept_sum / len(positions)
    zmean = zsum / len(positions)
    if not np.isfinite(intercept).all() or not np.isfinite(zmean).all():
        raise ValueError("Affine centering produced non-finite coordinates.")
    error, centered_scale = 0.0, 0.0
    for start in range(0, len(positions), 4096):
        rows = positions[start : start + 4096]
        residual = Z[rows] - np.asarray(X[rows] @ effective) - intercept
        if not np.isfinite(residual).all():
            raise ValueError("Affine validation produced a non-finite residual.")
        error = max(error, float(np.max(np.abs(residual))))
        centered_scale = max(centered_scale, float(np.max(np.abs(Z[rows] - zmean))))
    tolerance = atol + rtol * centered_scale
    if not np.isfinite(tolerance):
        raise ValueError(
            "Mapping tolerance overflowed; choose finite, meaningful tolerances."
        )
    if error > tolerance:
        raise ValueError(
            "Expression and representation do not share the declared affine mapping "
            f"(residual {error:.3g} > tolerance {tolerance:.3g}). Supply the exact layer, "
            "loadings and feature_scale; integrated/nonlinear coordinates cannot use ordinary PCA loadings."
        )
    return effective, {
        "kind": "verified_affine",
        "intercept": intercept,
        "max_abs_residual": error,
        "tolerance": tolerance,
        "centered_representation_scale": centered_scale,
    }


def _memberships(modules, regulons, names, regulator_source):
    lookup = {gene: index for index, gene in enumerate(names)}
    module_specs, regulator_specs, members, targets = [], [], [], []
    for kind, supplied in (("modules", modules), ("regulons", regulons)):
        if supplied is not None and not isinstance(supplied, Mapping):
            raise TypeError(f"{kind} must be a mapping or None.")
        if supplied and len({str(key) for key in supplied}) != len(supplied):
            raise ValueError(f"Colliding {kind} names.")
    for label, genes in (modules or {}).items():
        if isinstance(genes, (str, Mapping)):
            raise TypeError(
                "Each module must be a non-empty sequence of unique gene names."
            )
        genes = list(genes)
        if not genes or len(set(genes)) != len(genes):
            raise ValueError(
                "Each module must contain unique genes and cannot be empty."
            )
        missing = set(genes).difference(lookup)
        if missing:
            raise ValueError(f"Unknown module genes: {sorted(missing)}")
        label = str(label)
        module_specs.append((label, np.array([lookup[g] for g in genes])))
        members.extend(
            {"program": label, "gene": g, "weight": 1.0 / len(genes)} for g in genes
        )
    if regulons and (
        not isinstance(regulator_source, str) or not regulator_source.strip()
    ):
        raise ValueError(
            "regulator_source must identify the supplied regulatory evidence."
        )
    for label, weights in (regulons or {}).items():
        if not isinstance(weights, Mapping) or not weights:
            raise TypeError(
                "Each regulon must map target genes to finite nonzero signed weights."
            )
        missing = set(weights).difference(lookup)
        if missing:
            raise ValueError(f"Unknown regulon target genes: {sorted(missing)}")
        genes = list(weights)
        values = np.asarray(list(weights.values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values == 0):
            raise ValueError("Regulon weights must be finite and nonzero.")
        scaled = values / np.max(np.abs(values))
        normalized = scaled / np.abs(scaled).sum()
        label = str(label)
        regulator_specs.append(
            (label, np.array([lookup[g] for g in genes]), normalized)
        )
        targets.extend(
            {
                "regulator": label,
                "gene": g,
                "weight": float(w),
                "source": regulator_source,
            }
            for g, w in zip(genes, normalized)
        )
    return module_specs, regulator_specs, members, targets


def _bootstrap_expression(groups, estimator, n_boot, paired, rng):
    first, second = groups
    n0, n1 = len(first["sample_ids"]), len(second["sample_ids"])
    reorder = (
        [second["sample_ids"].index(s) for s in first["sample_ids"]] if paired else None
    )
    contrasts = np.empty((n_boot, first["sample_expression"].shape[1]))
    for draw in range(n_boot):
        indices0 = rng.choice(n0, n0, replace=True)
        indices1 = (
            np.asarray(reorder)[indices0]
            if paired
            else rng.choice(n1, n1, replace=True)
        )
        _, w0 = _center_weights(first["sample_centers"][indices0], estimator)
        _, w1 = _center_weights(second["sample_centers"][indices1], estimator)
        contrasts[draw] = (
            w1 @ second["sample_expression"][indices1]
            - w0 @ first["sample_expression"][indices0]
        )
    return contrasts


def _representation_labels(adata, key, source_params):
    if key is None:
        return {}
    result = adata.uns.get("scgeo", {}).get(key)
    if not isinstance(result, Mapping):
        raise KeyError(f"Representation result {key!r} is unavailable.")
    params = result.get("params", {})
    for name in ("condition_key", "sample_key", "center", "group0", "group1"):
        if params.get(name) != source_params.get(name):
            raise ValueError(
                f"Representation result and source shift disagree on {name}."
            )
    if params.get("node_key") != source_params.get("by"):
        raise ValueError(
            "Representation result and source shift use different state definitions."
        )
    if source_params["rep"] not in params.get("reps", []):
        raise ValueError(
            "Attribution representation is absent from the supplied representation result."
        )
    table = result.get("consensus_state")
    if not isinstance(table, pd.DataFrame) or not {"node", "consensus_label"}.issubset(
        table.columns
    ):
        raise ValueError("Representation result has no valid consensus_state table.")
    if table["node"].astype(str).duplicated().any():
        raise ValueError("Representation consensus contains duplicate state labels.")
    return dict(zip(table["node"].astype(str), table["consensus_label"].astype(str)))


def explain_shift(
    adata,
    *,
    shift_key="robust_shift",
    layer=None,
    loadings_key="PCs",
    loadings=None,
    feature_scale=None,
    states=None,
    modules=None,
    local_key=None,
    representation_key=None,
    regulons=None,
    regulator_source=None,
    min_cells=10,
    min_samples=2,
    n_boot=200,
    paired=False,
    seed=0,
    mapping_rtol=1e-5,
    mapping_atol=1e-6,
    zero_tol=1e-8,
    store_key="explain_shift",
):
    """Explain a stored robust shift with signed gene and program contributions.

    Parameters
    ----------
    adata
        AnnData containing a result from :func:`scgeo.tl.robust_shift`.
    shift_key
        Source under ``adata.uns['scgeo']``. Its state, contrast, representation,
        sample key and center estimator are reused and checked against the data.
        Supported estimators are ``mean`` and ``geometric_median``.
    layer
        Exact expression input for the affine representation; None uses X.
        Sparse inputs stay sparse. Ordinary PCA loadings cannot explain a
        nonlinear or batch-corrected embedding unless its map is verified.
    loadings_key, loadings
        Gene-by-coordinate coefficients, from varm[loadings_key] by default.
        An explicit array overrides the key. Extra trailing PCA columns are
        ignored; all coordinates of the source representation are explained.
    feature_scale
        Optional positive per-gene divisors applied to loading rows. A common
        affine intercept is inferred and verified, so centering is supported.
    states
        Optional state subset. With no source ``by``, the sole state is 'all'.
    modules
        Mapping of program names to unique member genes. Program expression is
        their mean; program contribution is their sum. Overlapping programs
        do not partition the displacement and are not percentages.
    local_key
        Optional obs column of independently justified local identity groups.
        Local contrasts use their own direction; projection onto the parent
        direction is also reported. Per-sample proportions are retained,
        including zero-count local groups. No cell-to-cell transitions or
        exact additive composition decomposition are inferred.
    representation_key
        Optional compatible representation_stability result. Its evidence
        labels are carried forward, never inferred from one representation.
    regulons, regulator_source
        Optional regulator -> {target: signed weight} mapping and required
        evidence provenance. Returns signed target-response summaries, not
        inferred networks, simulated interventions or causal-driver scores.
    min_cells, min_samples
        Coverage thresholds per condition. min_samples must be at least two.
        Missing biological replication retains descriptive attribution with
        status 'insufficient_samples' and no bootstrap interval.
    n_boot, paired, seed
        Biological-sample bootstrap draws, optionally matched by sample ID.
        Shared IDs require paired=True; paired analyses require equal ID sets.
        Intervals condition on the observed representation and shift direction;
        they are descriptive stability intervals, not selection-adjusted tests.
    mapping_rtol, mapping_atol, zero_tol
        Mapping/reconstruction tolerances and absolute zero-shift threshold.
        Mapping relative tolerance uses centered coordinate variation.
    store_key
        New result key under adata.uns['scgeo']; cannot overwrite a source key.

    Returns
    -------
    dict
        ``summary``, ``genes``, ``programs``, ``cells``, ``samples``,
        ``local_summary``, ``local_genes``, ``composition`` and ``regulators``
        tables, memberships, mapping diagnostics and numerical center weights.
        ``state_results`` uses safe internal keys and stores each state label.

    Notes
    -----
    For matched robust expression contrast d, affine loading L and observed
    unit direction u, C_g = d_g (L u)_g and sum(C_g) = ||delta|| up to the
    exported numerical residual. This is descriptive molecular accounting.
    Genes help define the axis, so this is not independent association evidence.
    Zero shifts have no inferred direction; nonzero local contrasts can still
    be reported. Existing robust-shift estimates and input matrices are unchanged.
    """
    min_cells = _integer(min_cells, "min_cells", 1)
    min_samples = _integer(min_samples, "min_samples", 2)
    n_boot = _integer(n_boot, "n_boot", 0)
    if not isinstance(paired, (bool, np.bool_)):
        raise TypeError("paired must be a boolean.")
    for name, value in (
        ("mapping_rtol", mapping_rtol),
        ("mapping_atol", mapping_atol),
        ("zero_tol", zero_tol),
    ):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not isinstance(store_key, str) or not store_key or "/" in store_key:
        raise ValueError("store_key must be a non-empty key without '/'.")
    if store_key in {shift_key, representation_key}:
        raise ValueError(
            "store_key cannot overwrite the source shift or representation evidence."
        )
    store = adata.uns.get("scgeo", {})
    source = store.get(shift_key)
    if not isinstance(source, Mapping) or "params" not in source:
        raise KeyError(
            f"Run robust_shift first: adata.uns['scgeo'][{shift_key!r}] is unavailable."
        )
    params = source["params"]
    required = {
        "rep",
        "condition_key",
        "group0",
        "group1",
        "sample_key",
        "by",
        "center",
    }
    if not required.issubset(params):
        raise ValueError("Source is not a compatible robust_shift result.")
    estimator = params["center"]
    if estimator not in {"mean", "geometric_median"}:
        raise ValueError(
            "explain_shift supports mean and geometric_median centers; coordinate-wise estimators need a different attribution contract."
        )
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(
            "Unique observation and gene names are required for traceable attribution."
        )
    names = np.asarray(adata.var_names.astype(str))
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Expression layer {layer!r} is missing.")
    X = adata.X if layer is None else adata.layers[layer]
    if not (sparse.issparse(X) or isinstance(X, np.ndarray)):
        raise TypeError("Expression must be an in-memory NumPy or SciPy sparse matrix.")
    if X.ndim != 2 or X.shape != adata.shape:
        raise ValueError("Expression shape must match AnnData.")
    if params["rep"] not in adata.obsm:
        raise ValueError(
            "explain_shift requires an explicit affine representation in adata.obsm."
        )
    # Use exactly the coordinates consumed by the existing robust estimator,
    # including its float32 conversion, rather than re-fitting at another precision.
    Z = _get_representation(adata, params["rep"])
    if Z.ndim != 2 or Z.shape[0] != adata.n_obs or Z.shape[1] == 0:
        raise ValueError(
            "Representation must be a non-empty cell-by-coordinate matrix."
        )
    all_positions = np.arange(adata.n_obs)
    condition = _labels(adata, params["condition_key"], all_positions)
    groups = [str(params["group0"]), str(params["group1"])]
    if groups[0] == groups[1]:
        raise ValueError("The two source conditions must be distinct.")
    contrast_mask = np.isin(condition, groups)
    contrast_positions = all_positions[contrast_mask]
    if not len(contrast_positions) or not np.isfinite(Z[contrast_positions]).all():
        raise ValueError(
            "Selected contrast must have finite representation coordinates and nonempty cells."
        )
    has_samples = params["sample_key"] is not None
    samples = np.full(adata.n_obs, "__pooled__", dtype=object)
    if has_samples:
        samples[contrast_positions] = _labels(
            adata, params["sample_key"], contrast_positions
        )
    elif paired:
        raise ValueError(
            "paired=True requires a biological sample_key in robust_shift."
        )
    state_labels = np.full(adata.n_obs, "all", dtype=object)
    if params["by"] is not None:
        state_labels[contrast_positions] = _labels(
            adata, params["by"], contrast_positions
        )
        source_states = source.get("by")
        if not isinstance(source_states, Mapping):
            raise ValueError("Stored shift has no per-state results.")
    else:
        source_states = {"all": source["global"]}
    if states is None:
        selected_states = list(source_states)
    else:
        selected_states = [states] if isinstance(states, str) else list(states)
        selected_states = [str(value) for value in selected_states]
        if not selected_states or len(set(selected_states)) != len(selected_states):
            raise ValueError("states must contain unique state labels.")
        missing = set(selected_states).difference(source_states)
        if missing:
            raise ValueError(f"States absent from source shift: {sorted(missing)}")
    local_labels = None
    if local_key is not None:
        local_labels = np.full(adata.n_obs, "", dtype=object)
        local_positions = all_positions[
            contrast_mask & np.isin(state_labels, selected_states)
        ]
        local_labels[local_positions] = _labels(adata, local_key, local_positions)
    loadings_source = "explicit" if loadings is not None else f"varm[{loadings_key!r}]"
    if loadings is None:
        if loadings_key not in adata.varm:
            raise KeyError(
                f"varm[{loadings_key!r}] is missing; supply exact affine loadings."
            )
        loadings = adata.varm[loadings_key]
    L = np.asarray(loadings, dtype=float)
    if (
        L.ndim != 2
        or L.shape[0] != adata.n_vars
        or L.shape[1] < Z.shape[1]
        or not np.isfinite(L).all()
    ):
        raise ValueError(
            "Loadings must be a finite n_genes-by-n_coordinates matrix covering the source representation."
        )
    L = L[:, : Z.shape[1]]
    scale = (
        np.ones(adata.n_vars)
        if feature_scale is None
        else np.asarray(feature_scale, dtype=float)
    )
    if (
        scale.shape != (adata.n_vars,)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError(
            "feature_scale must contain one finite positive divisor per gene."
        )
    effective, mapping = _affine_mapping(
        X, Z, contrast_positions, L, scale, mapping_rtol, mapping_atol
    )
    mapping["analysis_coordinate_dtype"] = (
        "float32 converted to float64, matching robust_shift"
    )
    mapping["loadings_source"] = loadings_source
    module_specs, regulator_specs, members, targets = _memberships(
        modules, regulons, names, regulator_source
    )
    representation_labels = _representation_labels(adata, representation_key, params)
    if representation_key is not None and set(selected_states).difference(
        representation_labels
    ):
        raise ValueError(
            "Supplied representation evidence does not cover every selected state."
        )
    rng = np.random.RandomState(seed)
    gene_tables, local_gene_tables = [], []
    summaries, local_summaries, cell_rows, sample_rows = [], [], [], []
    program_rows, regulator_rows, composition_rows, state_results = [], [], [], {}
    warnings = [
        "Descriptive attribution: genes contribute to an axis derived from expression; no independent association or causal evidence is inferred.",
        "Bootstrap intervals condition on the observed affine representation and direction; they are not selection-adjusted confidence intervals or p-values.",
    ]
    if modules:
        warnings.append(
            "Program contributions sum member-gene contributions; overlapping programs do not partition the displacement."
        )

    def analyze(
        state, positions0, positions1, *, parent_direction=None, local_state=None
    ):
        fits = [
            _group_center(X, Z, positions, samples[positions], estimator, has_samples)
            for positions in (positions0, positions1)
        ]
        n0, n1 = len(positions0), len(positions1)
        ns0, ns1 = (len(fit["sample_ids"]) if has_samples else 0 for fit in fits)
        complete = n0 > 0 and n1 > 0
        if complete and has_samples:
            shared = set(fits[0]["sample_ids"]) & set(fits[1]["sample_ids"])
            if shared and not paired:
                raise ValueError(
                    "Sample IDs occur in both conditions; use paired=True to preserve paired biological samples."
                )
            if paired and set(fits[0]["sample_ids"]) != set(fits[1]["sample_ids"]):
                raise ValueError(
                    "paired=True requires identical sample ID sets in both conditions of each state/local group."
                )
        delta = fits[1]["center"] - fits[0]["center"]
        norm = float(np.linalg.norm(delta)) if complete else np.nan
        valid = complete and min(n0, n1) >= min_cells
        nonzero = valid and norm > zero_tol
        direction = delta / norm if nonzero else np.full(Z.shape[1], np.nan)
        d = fits[1]["expression"] - fits[0]["expression"]
        if complete and not np.isfinite(d).all():
            raise ValueError(
                "Matched expression contrast is non-finite; check the expression scale."
            )
        coefficient = effective @ direction
        contribution = d * coefficient
        if nonzero and not np.isfinite(contribution).all():
            raise ValueError(
                "Gene contributions are non-finite; check the expression and loading scales."
            )
        if valid and not nonzero:
            contribution = np.zeros(adata.n_vars)
        if not valid:
            contribution[:] = np.nan
        status = (
            "insufficient_cells"
            if not valid
            else "zero_shift"
            if not nonzero
            else "insufficient_samples"
            if min(ns0, ns1) < min_samples
            else "ok"
        )
        reconstruct = (
            float(np.linalg.norm(d @ effective - delta)) if complete else np.nan
        )
        tolerance = mapping_atol + mapping_rtol * (norm if complete else 0.0)
        # Two weighted centers may each incur the verified pointwise map error.
        allowed = max(
            tolerance, 2 * np.sqrt(Z.shape[1]) * mapping["max_abs_residual"] + 1e-10
        )
        if complete and (
            not np.isfinite(reconstruct)
            or not np.isfinite(allowed)
            or reconstruct > allowed
        ):
            raise ValueError(
                f"Center-weight reconstruction failed for {state!r}: {reconstruct:.3g}; source estimator cannot be attributed faithfully."
            )
        boot = None
        if nonzero and min(ns0, ns1) >= min_samples and n_boot:
            boot = _bootstrap_expression(fits, estimator, n_boot, paired, rng)
            draws = boot * coefficient
            if not np.isfinite(boot).all() or not np.isfinite(draws).all():
                raise ValueError(
                    "Sample-bootstrap contrasts are non-finite; check input scales."
                )
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            sign = np.mean(np.sign(draws) == np.sign(contribution), axis=0)
            sign[np.abs(contribution) <= zero_tol] = np.nan
        else:
            low = high = sign = np.full(adata.n_vars, np.nan)
        frame = pd.DataFrame(
            {
                "state": state,
                "gene": names,
                "expression_center0": fits[0]["expression"],
                "expression_center1": fits[1]["expression"],
                "expression_delta": d,
                "axis_loading": coefficient,
                "contribution": contribution,
                "ci_low": low,
                "ci_high": high,
                "sign_stability": sign,
                "in_representation": np.any(effective != 0, axis=1),
                "status": status,
            }
        )
        frame["rank"] = (
            frame["contribution"].abs().rank(method="min", ascending=False)
            if nonzero
            else np.nan
        )
        if local_state is not None:
            frame["local_state"] = str(local_state)
            frame["parent_axis_contribution"] = d * (effective @ parent_direction)
        summary = {
            "state": state,
            "status": status,
            "delta_norm": norm,
            "n_cells0": n0,
            "n_cells1": n1,
            "n_samples0": ns0,
            "n_samples1": ns1,
            "contribution_sum": float(np.sum(contribution)) if valid else np.nan,
            "reconstruction_error": reconstruct,
            "representation_status": representation_labels.get(state, "not_assessed"),
            "n_boot": n_boot if boot is not None else 0,
            "interval_scope": "conditional_sample_bootstrap"
            if boot is not None
            else "unavailable",
        }
        if local_state is not None:
            summary["local_state"] = str(local_state)
            summary["projection_on_parent_axis"] = float(delta @ parent_direction)
            summary["parent_representation_status"] = summary["representation_status"]
            summary["representation_status"] = "not_assessed"
        return fits, direction, frame, summary, boot

    for state_index, state in enumerate(selected_states):
        mask = contrast_mask & (state_labels == state)
        positions = [all_positions[mask & (condition == group)] for group in groups]
        fits, direction, frame, summary, boot = analyze(state, *positions)
        _verify_source(
            source_states[state], fits, has_samples, mapping_rtol, mapping_atol, state
        )
        summaries.append(summary)
        gene_tables.append(frame)
        entry = {
            "state": state,
            "status": summary["status"],
            "center0": fits[0]["center"],
            "center1": fits[1]["center"],
            "delta": fits[1]["center"] - fits[0]["center"],
            "direction": direction,
        }
        for suffix, fit in enumerate(fits):
            entry[f"obs_positions{suffix}"] = fit["positions"]
            entry[f"center_weights{suffix}"] = fit["weights"]
            for index, sample in enumerate(fit["sample_ids"]):
                sample_rows.append(
                    {
                        "state": state,
                        "condition": groups[suffix],
                        "sample": sample,
                        "n_cells": int(fit["sample_counts"][index]),
                        "shift_score": float(
                            (fit["sample_centers"][index] - fits[0]["center"])
                            @ direction
                        ),
                    }
                )
            for position, score in zip(
                fit["positions"], (Z[fit["positions"]] - fits[0]["center"]) @ direction
            ):
                cell_rows.append(
                    {
                        "obs_position": int(position),
                        "obs_name": str(adata.obs_names[position]),
                        "state": state,
                        "condition": groups[suffix],
                        "sample": str(samples[position]),
                        "shift_score": float(score),
                        "eligible": bool(np.isfinite(score)),
                    }
                )
        state_results[f"state_{state_index}"] = entry
        d = frame["expression_delta"].to_numpy()
        c = frame["contribution"].to_numpy()
        for label, index in module_specs:
            values = (
                boot[:, index] @ frame["axis_loading"].to_numpy()[index]
                if boot is not None
                else None
            )
            interval = (
                np.quantile(values, [0.025, 0.975])
                if values is not None
                else [np.nan, np.nan]
            )
            program_rows.append(
                {
                    "state": state,
                    "program": label,
                    "n_genes": len(index),
                    "expression_delta": float(d[index].mean()),
                    "contribution": float(c[index].sum()),
                    "ci_low": float(interval[0]),
                    "ci_high": float(interval[1]),
                    "status": summary["status"],
                }
            )
        for label, index, weights in regulator_specs:
            activity = float(d[index] @ weights)
            regulator_rows.append(
                {
                    "state": state,
                    "regulator": label,
                    "n_targets": len(index),
                    "signed_target_delta": activity,
                    "target_contribution_sum": float(c[index].sum()),
                    "source": regulator_source,
                    "evidence_type": "supplied_signed_target_response",
                    "status": summary["status"],
                }
            )
        if local_labels is not None:
            local_values = list(pd.unique(local_labels[mask]))
            for local_state in local_values:
                local_positions = [p[local_labels[p] == local_state] for p in positions]
                _, _, local_frame, local_summary, _ = analyze(
                    state,
                    *local_positions,
                    parent_direction=direction,
                    local_state=local_state,
                )
                local_summaries.append(local_summary)
                local_gene_tables.append(local_frame)
                for suffix, fit in enumerate(fits):
                    for sample, total in zip(fit["sample_ids"], fit["sample_counts"]):
                        count = (
                            int(np.sum(samples[local_positions[suffix]] == sample))
                            if has_samples
                            else len(local_positions[suffix])
                        )
                        composition_rows.append(
                            {
                                "state": state,
                                "local_state": str(local_state),
                                "condition": groups[suffix],
                                "sample": sample,
                                "n_cells": count,
                                "fraction": float(count / total),
                            }
                        )
    gene_columns = [
        "state",
        "gene",
        "expression_center0",
        "expression_center1",
        "expression_delta",
        "axis_loading",
        "contribution",
        "ci_low",
        "ci_high",
        "sign_stability",
        "in_representation",
        "status",
        "rank",
    ]
    summary_columns = [
        "state",
        "status",
        "delta_norm",
        "n_cells0",
        "n_cells1",
        "n_samples0",
        "n_samples1",
        "contribution_sum",
        "reconstruction_error",
        "representation_status",
        "n_boot",
        "interval_scope",
    ]
    result = {
        "schema_version": "scgeo.explain_shift.v1",
        "source_params": deepcopy(dict(params)),
        "params": {
            **deepcopy(dict(params)),
            "shift_key": shift_key,
            "layer": layer,
            "loadings_key": loadings_key,
            "feature_scale": scale,
            "states": selected_states,
            "local_key": local_key,
            "representation_key": representation_key,
            "min_cells": min_cells,
            "min_samples": min_samples,
            "n_boot": n_boot,
            "paired": bool(paired),
            "seed": int(seed),
            "mapping_rtol": float(mapping_rtol),
            "mapping_atol": float(mapping_atol),
            "zero_tol": float(zero_tol),
            "store_key": store_key,
        },
        "mapping": mapping,
        "summary": pd.DataFrame(summaries, columns=summary_columns),
        "genes": pd.concat(gene_tables, ignore_index=True)
        if gene_tables
        else pd.DataFrame(columns=gene_columns),
        "programs": pd.DataFrame(
            program_rows,
            columns=[
                "state",
                "program",
                "n_genes",
                "expression_delta",
                "contribution",
                "ci_low",
                "ci_high",
                "status",
            ],
        ),
        "module_membership": pd.DataFrame(
            members, columns=["program", "gene", "weight"]
        ),
        "regulators": pd.DataFrame(
            regulator_rows,
            columns=[
                "state",
                "regulator",
                "n_targets",
                "signed_target_delta",
                "target_contribution_sum",
                "source",
                "evidence_type",
                "status",
            ],
        ),
        "regulon_membership": pd.DataFrame(
            targets, columns=["regulator", "gene", "weight", "source"]
        ),
        "samples": pd.DataFrame(
            sample_rows,
            columns=["state", "condition", "sample", "n_cells", "shift_score"],
        ),
        "cells": pd.DataFrame(
            cell_rows,
            columns=[
                "obs_position",
                "obs_name",
                "state",
                "condition",
                "sample",
                "shift_score",
                "eligible",
            ],
        ),
        "local_summary": pd.DataFrame(
            local_summaries,
            columns=summary_columns
            + [
                "local_state",
                "projection_on_parent_axis",
                "parent_representation_status",
            ],
        ),
        "local_genes": pd.concat(local_gene_tables, ignore_index=True)
        if local_gene_tables
        else pd.DataFrame(
            columns=gene_columns + ["local_state", "parent_axis_contribution"]
        ),
        "composition": pd.DataFrame(
            composition_rows,
            columns=[
                "state",
                "local_state",
                "condition",
                "sample",
                "n_cells",
                "fraction",
            ],
        ),
        "state_results": state_results,
        "warnings": warnings,
    }
    result["input_signature"] = _input_signature(
        adata,
        result["cells"]["obs_position"].to_numpy(dtype=int),
        layer,
        params["rep"],
    )
    # Commit the complete result only after all validation and calculations pass.
    adata.uns.setdefault("scgeo", {})[store_key] = result
    return result
