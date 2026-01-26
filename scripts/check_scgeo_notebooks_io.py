#!/usr/bin/env python3
"""Check ScGeo notebooks for IO/contract correctness.

This is a lightweight, notebook-friendly contract checker.

It *does not execute* notebooks.
Instead it performs a best-effort static pass:
- Track (per AnnData variable) which keys are expected to exist after each tl call,
  based on scgeo_io_manifest.json (diff before/after tl calls).
- Add a small set of heuristics for common external writes (Scanpy UMAP/PCA)
  and a few tl functions that are usually skipped by the IO manifest generator
  (mapping, vector alignment), so plots don't produce false negatives.
- For each scgeo.pl.* call, assert required keys exist in the tracked state.

Usage:
  python scripts/check_scgeo_notebooks_io.py \
    --io-manifest scgeo_io_manifest.json \
    --glob "notebooks/*.ipynb" [--fail-fast]
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import nbformat


# -----------------------------
# Data structures


@dataclass
class Finding:
    level: str  # 'WARN' | 'FAIL'
    path: str
    cell: int
    lineno: int
    call: str
    message: str


class State:
    """Tracked key state for a single AnnData variable."""

    def __init__(self):
        self.obs: Set[str] = set()
        self.obsm: Set[str] = set()
        self.layers: Set[str] = set()
        self.uns: Set[str] = set()

    def apply_writes(self, writes: Dict[str, List[str]]):
        self.obs |= set(writes.get("obs_cols_added", []))
        self.obsm |= set(writes.get("obsm_keys_added", []))
        self.layers |= set(writes.get("layers_keys_added", []))
        self.uns |= set(writes.get("uns_keys_added", []))


# -----------------------------
# Helpers


def _iter_ipynb_paths(globs: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pat in globs:
        paths.extend(glob.glob(pat))
    out: List[str] = []
    for p in sorted(set(paths)):
        if ".ipynb_checkpoints" in p:
            continue
        if Path(p).name.startswith("."):
            continue
        if p.endswith(".ipynb"):
            out.append(p)
    return out


def _parse(src: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _call_fullname(fn: ast.AST) -> Optional[str]:
    """Return dotted fullname for calls like sg.tl.shift(...) / sc.tl.umap(...)."""
    parts: List[str] = []
    cur = fn
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    return ".".join(reversed(parts))


def _first_arg_name(node: ast.Call) -> Optional[str]:
    if not node.args:
        return None
    a0 = node.args[0]
    if isinstance(a0, ast.Name):
        return a0.id
    return None


def _kw(node: ast.Call, key: str) -> Optional[Any]:
    for kw in node.keywords:
        if kw.arg == key:
            if isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


def _const_str(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


# -----------------------------
# IO manifest


def load_io_manifest(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping: scgeo.tl.NAME -> writes dict."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, List[str]]] = {}
    for k, v in (m.get("tl") or {}).items():
        if isinstance(v, dict) and "writes" in v:
            out[k] = v["writes"]
    return out


# -----------------------------
# Requirements for pl functions


def pl_requirements(call_name: str, node: ast.Call) -> List[Tuple[str, str]]:
    """Return list of required (space, key) pairs.

    space in {'obs','obsm','uns','layers'}.
    """
    req: List[Tuple[str, str]] = []

    if call_name in {"sg.pl.score_umap", "scgeo.pl.score_umap"}:
        # score_umap delegates to score_embedding(basis='umap')
        req.append(("obsm", "X_umap"))
        # score_key is positional arg 1
        if len(node.args) >= 2:
            s = _const_str(node.args[1])
            if s:
                req.append(("obs", s))

    elif call_name in {"sg.pl.highlight_topk_cells", "scgeo.pl.highlight_topk_cells"}:
        req.append(("obsm", "X_umap"))
        # score_key positional
        if len(node.args) >= 2:
            s = _const_str(node.args[1])
            if s:
                req.append(("obs", s))

    elif call_name in {"sg.pl.mapping_qc_panel", "scgeo.pl.mapping_qc_panel"}:
        req.append(("obsm", "X_umap"))
        pred = _kw(node, "pred_key") or "map_pred"
        conf = _kw(node, "conf_key") or "map_confidence"
        ood = _kw(node, "ood_key") or "map_ood_score"
        req += [("obs", str(pred)), ("obs", str(conf)), ("obs", str(ood))]

    elif call_name in {"sg.pl.mapping_confidence_umap", "scgeo.pl.mapping_confidence_umap"}:
        req.append(("obsm", "X_umap"))
        conf = _kw(node, "conf_key") or "map_confidence"
        req.append(("obs", str(conf)))

    elif call_name in {"sg.pl.ood_cells", "scgeo.pl.ood_cells"}:
        req.append(("obsm", "X_umap"))
        ood = _kw(node, "ood_key") or "map_ood_score"
        req.append(("obs", str(ood)))

    elif call_name in {"sg.pl.consensus_subspace_panel", "scgeo.pl.consensus_subspace_panel"}:
        # default store_key='consensus_subspace', score_key='cs_score'
        req.append(("obsm", "X_cs"))
        score = _kw(node, "score_key") or "cs_score"
        req.append(("obs", str(score)))

    return req


# -----------------------------
# Writes inference


def apply_known_writes(states: Dict[str, State], call_name: str, node: ast.Call):
    """Apply heuristic writes for non-scgeo or skipped tl."""
    var = _first_arg_name(node)
    if not var:
        return
    st = states.setdefault(var, State())

    # --- Scanpy basics ---
    if call_name in {"sc.tl.umap", "scanpy.tl.umap"}:
        st.obsm.add("X_umap")
        return

    if call_name in {"sc.pp.pca", "sc.tl.pca", "scanpy.pp.pca", "scanpy.tl.pca"}:
        st.obsm.add("X_pca")
        return

    # --- ScGeo TL functions often skipped by IO generator ---
    if call_name in {"sg.tl.map_query_to_ref", "scgeo.tl.map_query_to_ref",
                     "sg.tl.map_query_to_ref_pool", "scgeo.tl.map_query_to_ref_pool",
                     "sg.tl.map_query_to_ref_pool_census", "scgeo.tl.map_query_to_ref_pool_census"}:
        pred = _kw(node, "pred_key") or "scgeo_pred"
        conf = _kw(node, "conf_key") or "scgeo_conf"
        ood = _kw(node, "ood_key") or "scgeo_ood"
        reject = _kw(node, "reject_key") or "scgeo_reject"
        st.obs |= {str(pred), str(conf), str(ood), str(reject)}
        # mapping plots use umap in notebooks typically
        return

    if call_name in {"sg.tl.align_vectors", "scgeo.tl.align_vectors"}:
        obs_key = _kw(node, "obs_key") or "scgeo_align"
        st.obs.add(str(obs_key))
        return

    if call_name in {"sg.tl.projection_disagreement", "scgeo.tl.projection_disagreement"}:
        obs_key = _kw(node, "obs_key") or "scgeo_disagree"
        st.obs.add(str(obs_key))
        return

    if call_name in {"sg.tl.velocity_delta_alignment", "scgeo.tl.velocity_delta_alignment"}:
        obs_key = _kw(node, "obs_key") or "scgeo_vel_delta_align"
        st.obs.add(str(obs_key))
        return

    if call_name in {"sg.tl.consensus_subspace", "scgeo.tl.consensus_subspace"}:
        prefix = _kw(node, "obs_key_prefix") or "cs"
        st.obs.add(f"{prefix}_score")
        st.obsm.add("X_cs")
        return


# -----------------------------
# Main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io-manifest", required=True)
    ap.add_argument("--glob", action="append", default=["notebooks/*.ipynb"])
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    io = load_io_manifest(args.io_manifest)
    paths = _iter_ipynb_paths(args.glob)
    if not paths:
        print(f"[ERROR] No notebooks matched: {args.glob}")
        return 2

    findings: List[Finding] = []

    for path in paths:
        nb = nbformat.read(path, as_version=4)
        states: Dict[str, State] = {}

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            src = cell.source or ""
            tree = _parse(src)
            if tree is None:
                # syntax checked elsewhere
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = _call_fullname(node.func)
                if not fn:
                    continue

                # Normalize sg.* -> scgeo.* for manifest lookup, but keep original for messages
                scgeo_name = fn.replace("sg.", "scgeo.")

                # Apply writes from io manifest (only for scgeo.tl.*)
                if scgeo_name.startswith("scgeo.tl."):
                    var = _first_arg_name(node)
                    if var and scgeo_name in io:
                        states.setdefault(var, State()).apply_writes(io[scgeo_name])
                    else:
                        # heuristic writes for skipped tl
                        apply_known_writes(states, scgeo_name, node)

                # Also track a few external writes
                if fn.startswith("sc.") or fn.startswith("scanpy."):
                    apply_known_writes(states, fn, node)

                # Check pl requirements
                if fn.startswith("sg.pl.") or fn.startswith("scgeo.pl."):
                    var = _first_arg_name(node)
                    if not var:
                        continue
                    st = states.setdefault(var, State())
                    for space, key in pl_requirements(fn, node):
                        have = getattr(st, space)
                        if key not in have:
                            findings.append(
                                Finding(
                                    level="FAIL",
                                    path=path,
                                    cell=i,
                                    lineno=getattr(node, "lineno", 1),
                                    call=fn,
                                    message=f"missing {space} '{key}' (likely wrong call order or missing step)",
                                )
                            )
                            if args.fail_fast:
                                break

            if args.fail_fast and any(f.level == "FAIL" and f.path == path for f in findings):
                break

    if findings:
        for f in findings:
            print(f"[{f.level}] {f.path} cell={f.cell} line={f.lineno}  {f.call}: {f.message}")
        n_fail = sum(1 for f in findings if f.level == "FAIL")
        if n_fail:
            return 1

    print("No IO contract failures found (heuristics enabled).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
