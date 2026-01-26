#!/usr/bin/env python3
"""
Check ScGeo notebooks for API-manifest consistency.

- Scans .ipynb cells for calls like: sg.tl.xxx(...), sg.pl.yyy(...)
- Flags unknown keyword arguments by comparing to api_manifest.json
- Optionally ignores .ipynb_checkpoints and hidden folders

Usage:
  python scripts/check_scgeo_notebooks.py \
      --manifest api_manifest.json \
      notebooks/pbmc_ingest_scgeo_demo.ipynb notebooks/scvelo_velocitybasics_scgeo_demo.ipynb

Or check all notebooks:
  python scripts/check_scgeo_notebooks.py --manifest api_manifest.json --glob "notebooks/*.ipynb"
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nbformat


@dataclass
class Finding:
    path: str
    cell: int
    lineno: int
    call: str
    unknown_kwargs: List[str]
    missing_required: List[str]


def _load_manifest(manifest_path: str) -> Dict[str, Dict[str, object]]:
    """
    Returns:
      api[name] = {"allowed_kwargs": set[str], "required_positional": list[str], "signature": str}
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    api: Dict[str, Dict[str, object]] = {}
    for section in ("tl", "pl"):
        for item in m.get(section, []):
            name = item["name"]  # e.g. scgeo.pl.density_overlap
            sig = item.get("signature", "")
            # Parse signature text lightly with ast? Too brittle.
            # We instead extract kwargs from "(...)" using a simple tokenizer:
            allowed = _extract_param_names_from_signature(sig)
            req = _extract_required_from_signature(sig)
            api[name] = {"allowed_kwargs": allowed, "required": req, "signature": sig}
    return api


def _extract_param_names_from_signature(sig: str) -> set[str]:
    """
    Best-effort extraction of parameter names from the signature string.
    Works for the api_manifest.json you generate (pretty normal formatting).
    """
    # signature is like: "(adata, *, store_key: 'str'='density_overlap', level: 'str'='by', ...)"
    if not sig.startswith("(") or ")" not in sig:
        return set()
    inside = sig[sig.find("(") + 1 : sig.rfind(")")]
    # remove type annotations and defaults
    params = []
    depth = 0
    buf = []
    for ch in inside:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            params.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        params.append("".join(buf).strip())

    names: set[str] = set()
    for p in params:
        if not p or p in {"*", "/"}:
            continue
        # remove leading * or **
        p = p.lstrip("*").strip()
        if not p:
            continue
        # name is before ":" or "="
        name = p.split(":")[0].split("=")[0].strip()
        # ignore positional-only marker if any
        if name and name not in {"self"}:
            names.add(name)
    return names


def _extract_required_from_signature(sig: str) -> List[str]:
    """
    Identify required parameters by signature text:
    - before '*' are positional params; those without defaults are "required-ish"
    - after '*' all are keyword-only; those without defaults are required
    This is conservative; we only use it for a helpful hint.
    """
    if not sig.startswith("(") or ")" not in sig:
        return []
    inside = sig[sig.find("(") + 1 : sig.rfind(")")]
    parts = []
    depth = 0
    buf = []
    for ch in inside:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())

    req: List[str] = []
    for p in parts:
        if not p or p in {"*", "/"}:
            continue
        p2 = p.lstrip("*").strip()
        if not p2:
            continue
        name = p2.split(":")[0].split("=")[0].strip()
        has_default = "=" in p2
        # treat "adata" as required positional always; otherwise only if no default
        if name and (name == "adata" or not has_default):
            req.append(name)
    return req


def _iter_ipynb_paths(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.glob:
        for pat in args.glob:
            paths.extend(glob.glob(pat))
    paths.extend(args.paths or [])
    # normalize & filter
    out: List[str] = []
    for p in paths:
        p = str(Path(p))
        if args.ignore_checkpoints and ".ipynb_checkpoints" in p:
            continue
        if args.ignore_hidden and any(part.startswith(".") for part in Path(p).parts):
            # keep visible notebooks only
            continue
        if p.endswith(".ipynb") and os.path.exists(p):
            out.append(p)
    return sorted(set(out))


def _find_scgeo_calls(cell_source: str) -> List[Tuple[str, ast.Call]]:
    """
    Returns list of (call_name, ast.Call) where call_name is 'sg.pl.x' or 'sg.tl.y'
    """
    try:
        tree = ast.parse(cell_source)
    except SyntaxError:
        return []

    hits: List[Tuple[str, ast.Call]] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # match sg.pl.xxx(...) or sg.tl.xxx(...)
            fn = node.func
            call_name: Optional[str] = None

            # Attribute(Attribute(Name('sg'),'pl'),'delta_rank') => sg.pl.delta_rank
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute):
                mid = fn.value
                if isinstance(mid.value, ast.Name) and mid.value.id == "sg":
                    if mid.attr in {"pl", "tl"}:
                        call_name = f"sg.{mid.attr}.{fn.attr}"

            if call_name:
                hits.append((call_name, node))
            self.generic_visit(node)

    V().visit(tree)
    return hits


def _kwarg_names(call: ast.Call) -> List[str]:
    out: List[str] = []
    for kw in call.keywords:
        if kw.arg is None:
            # **kwargs
            continue
        out.append(kw.arg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to api_manifest.json")
    ap.add_argument("--glob", action="append", default=[], help="Glob patterns, e.g. notebooks/*.ipynb")
    ap.add_argument("paths", nargs="*", help="Explicit .ipynb paths")
    ap.add_argument("--ignore-checkpoints", action="store_true", default=True)
    ap.add_argument("--no-ignore-checkpoints", dest="ignore_checkpoints", action="store_false")
    ap.add_argument("--ignore-hidden", action="store_true", default=True)
    ap.add_argument("--no-ignore-hidden", dest="ignore_hidden", action="store_false")
    args = ap.parse_args()

    api = _load_manifest(args.manifest)
    paths = _iter_ipynb_paths(args)

    findings: List[Finding] = []

    for path in paths:
        nb = nbformat.read(path, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            src = cell.source or ""
            calls = _find_scgeo_calls(src)
            if not calls:
                continue

            for call_name, node in calls:
                # resolve to scgeo.pl.x / scgeo.tl.y
                scgeo_name = call_name.replace("sg.", "scgeo.")
                if scgeo_name not in api:
                    continue

                allowed = api[scgeo_name]["allowed_kwargs"]
                required = api[scgeo_name]["required"]
                kws = _kwarg_names(node)

                unknown = sorted([k for k in kws if k not in allowed])
                missing = []  # conservative: only check if they used only kwargs
                if node.args == []:
                    for r in required:
                        if r not in kws and r != "adata":
                            missing.append(r)

                if unknown or missing:
                    findings.append(
                        Finding(
                            path=path,
                            cell=i,
                            lineno=getattr(node, "lineno", 1),
                            call=call_name,
                            unknown_kwargs=unknown,
                            missing_required=missing,
                        )
                    )

    if findings:
        for f in findings:
            print(
                f"[{f.path}] cell={f.cell} line={f.lineno}  {f.call}  "
                f"unknown_kwargs={f.unknown_kwargs} missing_required={f.missing_required}"
            )
        print(f"Found {len(findings)} API mismatches.")
        raise SystemExit(1)

    print("No API mismatches found.")


if __name__ == "__main__":
    main()
