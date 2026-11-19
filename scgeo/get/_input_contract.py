"""Read-only, explicit input contracts; no estimators or data repair."""
from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
from scipy import sparse


_REQUIRED = {"purpose", "state_key", "condition_key", "group0", "group1", "sample_key",
             "reps", "min_cells", "min_samples_per_condition", "methods"}
_OPTIONAL = {"design", "bootstrap_unit", "embedding_key", "projected_velocity", "paga",
             "evidence", "k_values", "representation_provenance"}
_METHODS = {"robust_shift", "representation_stability", "local_geometry_stability", "plotting"}


def input_contract(adata, *, config):
    """Inspect explicitly configured AnnData inputs without modifying or fitting them.

    Parameters
    ----------
    adata : AnnData
        Only obs, obsm and uns are inspected; X, layers and raw are not read.
    config : dict
        JSON-compatible configuration with purpose (descriptive/sample_inference),
        state_key, condition_key, group0, group1, sample_key (or None), reps (exact
        obsm keys), methods, min_cells and min_samples_per_condition. Optional
        design='independent', bootstrap_unit, k_values, embedding_key,
        representation_provenance, projected_velocity, paga and evidence are
        described in docs/JUPYTERHUB_PREFLIGHT.md. No keys or groups are inferred.

    Returns
    -------
    dict
        JSON-safe checks, inventory and per-state coverage. ok means the requested
        input contract passed; excluded_states remain ineligible for sample-level
        interpretation. Eligibility is a coverage screen, not significance or a
        guarantee of biological independence. Provenance assertions are checked
        for consistency, not authenticated. No results are stored in adata.
    """
    checks = []
    report = {"schema_version": 1, "ok": False, "checks": checks, "state_coverage": [],
              "eligible_states": [], "excluded_states": [], "sample_inference_eligible": False}

    def issue(code, message, severity="error"):
        checks.append({"code": code, "severity": severity, "message": message})

    def finish():
        report["ok"] = not any(c["severity"] == "error" for c in checks)
        report["sample_inference_eligible"] = bool(
            report["ok"] and config.get("purpose") == "sample_inference" and report["eligible_states"])
        return report

    if not isinstance(config, Mapping):
        issue("config", "config must be a JSON-compatible mapping.")
        return report
    try:
        report["config"] = json.loads(json.dumps(config, allow_nan=False))
    except (TypeError, ValueError):
        issue("config", "config must contain only finite, JSON-compatible values.")
        return report
    config = report["config"]
    missing, unknown = _REQUIRED - config.keys(), config.keys() - _REQUIRED - _OPTIONAL
    if missing or unknown:
        issue("config", f"Missing config keys: {sorted(missing)}; unknown keys: {sorted(unknown)}.")
        return finish()
    if config["purpose"] not in ("descriptive", "sample_inference"):
        issue("purpose", "purpose must be descriptive or sample_inference.")
    for key in ("min_cells", "min_samples_per_condition"):
        if type(config[key]) is not int or config[key] < 1:
            issue("threshold", f"{key} must be an explicit positive integer.")
    for key in ("reps", "methods"):
        value = config[key]
        if (not isinstance(value, list) or not value
                or any(not isinstance(x, str) or not x for x in value)
                or len(set(value)) != len(value)):
            issue("config", f"{key} must be a nonempty list of distinct strings.")
    if any(c["severity"] == "error" for c in checks):
        return finish()
    for key in ("state_key", "condition_key", "sample_key"):
        if not (key == "sample_key" and config[key] is None) and not isinstance(config[key], str):
            issue("config", f"{key} must be an obs column name (sample_key may be None).")
    if any(c["severity"] == "error" for c in checks):
        return finish()
    methods, reps = set(config["methods"]), config["reps"]
    if methods - _METHODS:
        issue("methods", f"Unsupported methods: {sorted(methods - _METHODS)}.")
    inference = config["purpose"] == "sample_inference"
    sample_key = config["sample_key"]
    if inference and (config.get("design") != "independent" or sample_key is None):
        issue("sample_design", "Sample inference requires sample_key and an explicit independent design declaration.")
    if inference and config["min_samples_per_condition"] < 2:
        issue("sample_threshold", "Sample inference requires at least two independent samples per condition per state.")
    bootstrap = config.get("bootstrap_unit", "auto")
    if bootstrap not in ("auto", "cell", "sample"):
        issue("bootstrap", "bootstrap_unit must be auto, cell or sample.")
    if (sample_key is not None and bootstrap == "cell") or (bootstrap == "sample" and sample_key is None):
        issue("bootstrap", "Sample identities and bootstrap_unit are incompatible; cell resampling is not sample inference.")
    if methods & {"representation_stability", "local_geometry_stability"} and len(reps) < 2:
        issue("representations", "Stability requires at least two explicitly selected representations.")
    issue("interpretation", "Plots and geometry agreement are descriptive. Coverage does not establish significance, "
          "independence, a biological effect, or causal dynamics.", "warning")
    provenance = config.get("representation_provenance", {})
    if not isinstance(provenance, dict) or any(not isinstance(provenance.get(r), str) or not provenance[r].strip() for r in reps):
        issue("representation_provenance", "Describe the origin and dependence of every selected representation; "
              "related embeddings are sensitivity checks, not independent replication.", "warning")
    if not all(hasattr(adata, attr) for attr in ("obs", "obsm", "uns", "n_obs")):
        issue("anndata", "Expected AnnData-compatible obs, obsm, uns and n_obs.")
        return finish()
    store = adata.uns.get("scgeo", {})
    report["inventory"] = {"n_obs": int(adata.n_obs), "obs": list(map(str, adata.obs.columns)),
                           "obsm": list(map(str, adata.obsm.keys())),
                           "scgeo_evidence": list(map(str, store.keys())) if isinstance(store, Mapping) else []}
    if not adata.obs.columns.is_unique or not adata.obs.index.is_unique:
        issue("observation_identity", "Observation names and metadata column names must be unique.")
    labels = {}
    for key in (config["state_key"], config["condition_key"], sample_key):
        if key is None:
            continue
        if not isinstance(key, str) or key not in adata.obs or not adata.obs.columns.is_unique:
            issue("labels", f"Missing or invalid obs key: {key!r}.")
            continue
        values = adata.obs[key]
        if values.isna().any():
            issue("missing_labels", f"obs[{key!r}] contains missing labels; no labels are filled.")
            continue
        # The existing analysis methods stringify labels. Detect identity loss
        # before using that same convention for this report.
        try:
            unique = list(pd.unique(values))
        except (TypeError, ValueError):
            issue("labels", f"obs[{key!r}] must contain scalar, hashable labels.")
            continue
        if len({str(x) for x in unique}) != len(unique):
            issue("label_collision", f"obs[{key!r}] has distinct labels with the same string representation.")
            continue
        labels[key] = values.astype(str).to_numpy()
    group0, group1 = config["group0"], config["group1"]
    if (not isinstance(group0, (str, int, float, bool)) or not isinstance(group1, (str, int, float, bool))
            or str(group0) == str(group1)):
        issue("groups", "group0 and group1 must be distinct scalar labels.")
    elif config["condition_key"] in labels:
        raw = adata.obs[config["condition_key"]]
        if not (raw == group0).any() or not (raw == group1).any():
            issue("groups", "Both explicit groups must match actual condition labels without coercion.")
    shapes = {}

    def matrix(key, *, display=False):
        if not isinstance(key, str) or key not in adata.obsm:
            issue("matrix_missing", f"Exact obsm key {key!r} is unavailable; no fallback was used.")
            return
        value = adata.obsm[key]
        shape = getattr(value, "shape", ())
        if len(shape) != 2 or shape[0] != adata.n_obs or shape[1] < (2 if display else 1):
            issue("matrix_shape", f"obsm[{key!r}] has incompatible shape {shape}.")
            return
        if (display and sparse.issparse(value)) or np.iscomplexobj(value):
            issue("matrix_type", f"obsm[{key!r}] must use real coordinates; plotting embeddings must be dense.")
            return
        try:
            # Do not densify sparse coordinates or touch expression matrices.
            finite = np.isfinite(value.data if sparse.issparse(value) else np.asarray(value)).all()
            if not finite:
                issue("nonfinite", f"obsm[{key!r}] contains nonfinite coordinates; none were removed or imputed.")
        except TypeError:
            issue("matrix_type", f"obsm[{key!r}] must contain numeric coordinates.")
        shapes[key] = list(map(int, shape))

    for rep in reps:
        matrix(rep)
    embedding = config.get("embedding_key")
    if embedding is not None:
        matrix(embedding, display=True)
    if "local_geometry_stability" in methods:
        ks = config.get("k_values")
        if (not isinstance(ks, list) or not ks
                or any(type(k) is not int or k < 1 or k >= adata.n_obs for k in ks)):
            issue("local_k", "Supply positive integer k_values with k < n_obs for local geometry.")
    report["matrix_shapes"] = shapes
    velocity = config.get("projected_velocity")
    if velocity is not None:
        if (not isinstance(velocity, dict) or set(velocity) != {"key", "basis", "provenance_uns_key"}
                or any(not isinstance(v, str) for v in velocity.values())):
            issue("velocity_config", "projected_velocity requires key, basis and provenance_uns_key.")
        else:
            key, basis = velocity["key"], velocity["basis"]
            matrix(key, display=True)
            matrix(basis, display=True)
            if key in shapes and basis in shapes and shapes[key] != shapes[basis]:
                issue("velocity_shape", "Projected velocity must exactly match its coordinate basis shape.")
            if basis != embedding or not isinstance(basis, str) or key != "velocity_" + basis.removeprefix("X_"):
                issue("velocity_basis", "Projected velocity key and basis must match the configured plotting embedding.")
            origin = adata.uns.get(velocity["provenance_uns_key"], {})
            if (not isinstance(origin, Mapping) or origin.get("basis") != basis
                    or origin.get("velocity_key") != key
                    or not isinstance(origin.get("source"), str) or not origin["source"].strip()):
                issue("velocity_provenance", "Stored projection provenance must identify basis, velocity_key and source. "
                      "Compatible array shapes alone do not establish matching coordinates.")
    paga = config.get("paga")
    if paga is not None:
        try:
            if not isinstance(paga, dict) or set(paga) != {"key", "node_order"}:
                raise ValueError("paga requires key and explicit node_order")
            payload = adata.uns[paga["key"]]
            if (payload.get("groups") != config["state_key"] or not isinstance(payload.get("source"), str)
                    or not payload["source"].strip()):
                raise ValueError("stored PAGA groups and source provenance are required")
            order = paga["node_order"]
            if not isinstance(order, list) or not order or len(set(order)) != len(order):
                raise ValueError("PAGA node_order must be a nonempty list of distinct labels")
            stored_order = payload.get("node_order")
            if stored_order is None or list(stored_order) != order:
                raise ValueError("explicit order must match stored node_order; current categorical order is not provenance")
            from ..pl._paga_shift_map import _graph
            observed = labels.get(config["state_key"], [])
            names, edges, source = _graph(adata, config["state_key"], paga["key"], order, observed)
            if set(names) != set(observed):
                raise ValueError("PAGA nodes do not match observed states")
            report["paga"] = {"node_order": names, "n_edges": len(edges), "order_source": source}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issue("paga", f"PAGA contract: {exc}")
    evidence = config.get("evidence", {})
    if not isinstance(evidence, dict):
        issue("evidence", "evidence must map stored ScGeo keys to method/params contracts.")
    else:
        schemas = {"robust_shift": {"global": dict},
                   "representation_stability": {"per_rep_state": pd.DataFrame, "consensus_state": pd.DataFrame},
                   "local_geometry_stability": {"pair_summary": pd.DataFrame, "state_pair_summary": pd.DataFrame}}
        for key, specification in evidence.items():
            if (not isinstance(specification, dict) or set(specification) != {"method", "params"}
                    or not isinstance(specification["method"], str) or specification["method"] not in schemas
                    or not isinstance(specification["params"], dict) or not specification["params"]):
                issue("evidence_config", f"Declare method and nonempty expected params for evidence {key!r}.")
                continue
            expected = specification["params"]
            result = store.get(key) if isinstance(store, Mapping) else None
            if not isinstance(result, Mapping) or not isinstance(result.get("params"), Mapping):
                issue("evidence_missing", f"Stored evidence {key!r} with params is unavailable; no analysis was run.")
            else:
                required = dict(schemas[specification["method"]])
                if specification["method"] == "robust_shift" and result["params"].get("by") is not None:
                    required["by"] = dict
                if any(not isinstance(result.get(field), kind) for field, kind in required.items()):
                    issue("evidence_schema", f"Stored {key!r} is missing expected result containers for {specification['method']}.")
                for name, value in expected.items():
                    actual = result["params"].get(name)
                    try:
                        matches = name in result["params"] and np.array_equal(actual, value)
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        issue("evidence_mismatch", f"Stored {key!r} parameter {name!r} does not match the configuration.")
        if evidence:
            issue("evidence_freshness", "Matching stored params cannot prove that results were computed from the current cells. "
                  "Retain the analysis input digest and source commit separately.", "warning")
    needed = [config["state_key"], config["condition_key"]] + ([sample_key] if sample_key else [])
    if any(k not in labels for k in needed):
        return finish()
    states, conditions = labels[config["state_key"]], labels[config["condition_key"]]
    groups = (str(group0), str(group1))
    samples = labels.get(sample_key)
    if samples is not None:
        mixed = any(len(set(conditions[samples == s])) > 1 for s in pd.unique(samples))
        if mixed and (inference or config.get("design") == "independent"):
            issue("sample_condition", "A sample occurs in multiple conditions; the declared independent design is incompatible.")
        totals = [len(pd.unique(samples[conditions == group])) for group in groups]
        report["n_samples_per_condition"] = totals
        if totals == [2, 2]:
            issue("small_n", "Two independent samples per condition yield only six treatment-label assignments. "
                  "Bootstrap stability is not a treatment-label significance test.", "warning")
    for state in pd.unique(states):
        counts = [int(((states == state) & (conditions == g)).sum()) for g in groups]
        ns = ([len(pd.unique(samples[(states == state) & (conditions == g)])) for g in groups]
              if samples is not None else [None, None])
        status = ("missing_condition" if min(counts) == 0 else
                  "insufficient_cells" if min(counts) < config["min_cells"] else
                  "insufficient_samples" if samples is None or min(ns) < config["min_samples_per_condition"] else "eligible")
        row = {"state": str(state), "n_cells": counts, "n_samples": ns, "status": status,
               "sample_counts": []}
        if samples is not None:
            row["sample_counts"] = [{"sample": str(s), "condition": g,
                                      "n_cells": int(((states == state) & (conditions == g) & (samples == s)).sum())}
                                     for g in groups for s in pd.unique(samples[conditions == g])]
        if "local_geometry_stability" in methods and isinstance(config.get("k_values"), list):
            n_state_samples = len(pd.unique(samples[states == state])) if samples is not None else None
            row["local_k_coverage"] = {
                str(k): bool((states == state).sum() >= max(2, k) and (n_state_samples is None or n_state_samples >= 2))
                for k in config["k_values"] if type(k) is int}
        report["state_coverage"].append(row)
        report["eligible_states" if status == "eligible" else "excluded_states"].append(str(state))
    if report["excluded_states"]:
        issue("state_coverage", "Some states fail the configured sample-inference coverage screen; retain their explicit "
              "status even when descriptive plots or robust_shift estimates are available.", "warning")
    if inference and not report["eligible_states"]:
        issue("no_eligible_states", "No state passes the declared sample-inference coverage requirements.")
    return finish()
