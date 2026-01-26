#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


ROOT = Path(".")
API_PATH = ROOT / "api_manifest.json"
IO_PATH = ROOT / "scgeo_io_manifest.json"

DOCS_API_DIR = ROOT / "docs" / "api"
DOCS_CONTRACTS_DIR = ROOT / "docs" / "contracts"
DOCS_API_DIR.mkdir(parents=True, exist_ok=True)
DOCS_CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> Any:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _normalize_io_manifest(io_raw: Any) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - list of entries: [{"function": "...", "mode": "read|write", "path": "...", "type": "...", ...}, ...]
      - dict keyed by module or function -> list entries

    Returns a flat list of entries.
    """
    if io_raw is None:
        return []

    if isinstance(io_raw, list):
        return [x for x in io_raw if isinstance(x, dict)]

    # Some variants: {"tl.shift": [...], ...} or {"tl": [...], "pl": [...]}
    if isinstance(io_raw, dict):
        flat: List[Dict[str, Any]] = []
        for _, v in io_raw.items():
            if isinstance(v, list):
                flat.extend([x for x in v if isinstance(x, dict)])
            elif isinstance(v, dict):
                # nested dict
                for _, vv in v.items():
                    if isinstance(vv, list):
                        flat.extend([x for x in vv if isinstance(x, dict)])
        return flat

    return []


def _index_io(io_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Index IO entries by function name as used in api_manifest.
    We match on:
      - exact function name, or
      - suffix match if IO stores fully-qualified name (e.g., "tl.shift")
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for e in io_entries:
        fn = str(e.get("function", "")).strip()
        if not fn:
            continue
        by_name.setdefault(fn, []).append(e)
    return by_name


def _match_io_for_function(
    fn_name: str, section: str, io_by_name: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Try to find IO entries for this function.
    - First try exact match (fn_name)
    - Then try "section.fn_name" (e.g., "tl.shift")
    - Then try any key that endswith ".fn_name"
    """
    if fn_name in io_by_name:
        return io_by_name[fn_name]

    fq = f"{section}.{fn_name}"
    if fq in io_by_name:
        return io_by_name[fq]

    # fallback suffix match
    hits: List[Dict[str, Any]] = []
    suffix = f".{fn_name}"
    for k, v in io_by_name.items():
        if k.endswith(suffix):
            hits.extend(v)
    return hits


def _fmt_io_block(io_entries: List[Dict[str, Any]]) -> str:
    """
    Render Reads/Writes blocks from IO manifest entries.
    Expected keys: mode, path, type, note (optional)
    """
    reads: List[Tuple[str, str, str]] = []
    writes: List[Tuple[str, str, str]] = []

    for e in io_entries:
        mode = str(e.get("mode", "")).lower().strip()
        path = str(e.get("path", "")).strip()
        typ = str(e.get("type", "")).strip()
        note = str(e.get("note", "")).strip() or str(e.get("notes", "")).strip()

        if not path and e.get("key"):
            # tolerate alternate field name
            path = str(e.get("key")).strip()

        if mode == "read":
            reads.append((path, typ, note))
        elif mode == "write":
            writes.append((path, typ, note))

    lines: List[str] = []
    lines.append("### I/O contract")

    if reads:
        lines.append("**Reads**")
        for path, typ, note in reads:
            tail = f" — {note}" if note else ""
            lines.append(f"- `{path}` ({typ or 'unknown'}){tail}")
        lines.append("")

    if writes:
        lines.append("**Writes**")
        for path, typ, note in writes:
            tail = f" — {note}" if note else ""
            lines.append(f"- `{path}` ({typ or 'unknown'}){tail}")
        lines.append("")

    if not reads and not writes:
        lines.append("_No I/O entries found in `scgeo_io_manifest.json`._\n")

    return "\n".join(lines).rstrip() + "\n"


def render_section(section_name: str, items: List[Dict[str, Any]], io_by_name: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = [f"# scgeo.{section_name}\n"]
    lines.append(
        "> Auto-generated from `api_manifest.json` + `scgeo_io_manifest.json`.\n"
        "> If anything here is wrong, fix the manifests or the generator—not the generated files.\n"
    )

    for it in items:
        name = str(it.get("name", "")).strip()
        sig = str(it.get("signature", "")).strip()
        doc = str(it.get("doc", "")).strip()

        lines += [
            f"## `{name}`\n",
            f"**Signature**  \n`{sig}`\n",
        ]
        if doc:
            lines += [f"**Docstring**  \n{doc}\n"]

        io_entries = _match_io_for_function(name, section_name, io_by_name)
        lines.append(_fmt_io_block(io_entries))

        lines += [
            "---\n",
        ]

    return "\n".join(lines).rstrip() + "\n"


def render_io_manifest_table(io_entries: List[Dict[str, Any]]) -> str:
    """
    Generate docs/contracts/io_manifest.md as a compact searchable reference.
    """
    lines: List[str] = ["# ScGeo I/O manifest\n"]
    lines.append("> Auto-generated from `scgeo_io_manifest.json`.\n")

    # Normalize columns
    rows: List[Tuple[str, str, str, str, str]] = []
    for e in io_entries:
        fn = str(e.get("function", "")).strip()
        mode = str(e.get("mode", "")).strip()
        path = str(e.get("path", "")).strip() or str(e.get("key", "")).strip()
        typ = str(e.get("type", "")).strip()
        note = str(e.get("note", "")).strip() or str(e.get("notes", "")).strip()
        rows.append((fn, mode, path, typ, note))

    # Sort for readability
    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    lines.append("| Function | Mode | Path | Type | Notes |\n")
    lines.append("|---|---:|---|---|---|\n")
    for fn, mode, path, typ, note in rows:
        # escape pipes
        note = note.replace("|", "\\|")
        lines.append(f"| `{fn}` | {mode} | `{path}` | {typ or ''} | {note} |\n")

    return "".join(lines)


def main() -> None:
    api = _load_json(API_PATH)
    io_raw = _load_json(IO_PATH)

    io_entries = _normalize_io_manifest(io_raw)
    io_by_name = _index_io(io_entries)

    # Generate per-section docs
    for section in ["tl", "pl", "get", "pp", "data", "bench"]:
        items = api.get(section, [])
        if not items:
            # still create a stub so links don't break
            (DOCS_API_DIR / f"{section}.md").write_text(
                f"# scgeo.{section}\n\n_No entries found in `api_manifest.json` for `{section}`._\n",
                encoding="utf-8",
            )
            continue

        md = render_section(section, items, io_by_name)
        (DOCS_API_DIR / f"{section}.md").write_text(md, encoding="utf-8")

    # Generate IO manifest summary table
    io_md = render_io_manifest_table(io_entries)
    (DOCS_CONTRACTS_DIR / "io_manifest.md").write_text(io_md, encoding="utf-8")

    print("Wrote:")
    for section in ["tl", "pl", "get", "pp", "data", "bench"]:
        print(f"  - {DOCS_API_DIR / (section + '.md')}")
    print(f"  - {DOCS_CONTRACTS_DIR / 'io_manifest.md'}")


if __name__ == "__main__":
    main()
