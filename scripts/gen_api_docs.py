#!/usr/bin/env python3
"""Generate the complete API reference and module pages from checked manifests.

Run after rebuild_manifests.py. --check detects stale generated pages without
writing files. Curated scientific notes live in docs/contracts/api_notes.json;
signatures and exported names always come from the runtime-derived manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ("tl", "pl", "get", "pp", "data", "bench")
MODULES = {
    "tl": ("Analysis", "Measure displacement, representation stability, local geometry, distributional change and dynamics agreement."),
    "pl": ("Figures", "Visualize stored results and assemble evidence panels, reports and figure-source exports."),
    "get": ("Reports and tables", "Retrieve analysis tables and combine available evidence into a state-level report."),
    "pp": ("Reference preparation", "Build reference pools and query-to-reference neighbor graphs. Approximate reference-pool construction requires pynndescent."),
    "data": ("Census access", "Access CELLxGENE Census data and embeddings with the optional Census dependencies. Data requests require network access; importing ScGeo does not download data."),
    "bench": ("Synthetic benchmarks", "Simulate controlled perturbations and evaluate stored outputs against known truth. The benchmark profiler requires Linux, WSL or macOS."),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor(name: str) -> str:
    return name.replace(".", "").lower()


def _signature(name: str, signature: str) -> str:
    """Wrap a literal signature at top-level commas without changing its text."""
    if len(name + signature) <= 100:
        return name + signature
    parts, start, depth = [], 1, 0
    quote, escaped, end = None, False, None
    for index, char in enumerate(signature):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                parts.append(signature[start:index].strip())
                end = index
                break
        elif char == "," and depth == 1:
            parts.append(signature[start:index].strip())
            start = index + 1
    if end is None:
        raise ValueError(f"Cannot format signature: {name}{signature}")
    return name + "(\n" + "\n".join("    " + part + "," for part in parts if part) + "\n" + signature[end:]


def _keys(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) or "—"


def _io_summary(name: str, io: dict) -> list[str]:
    entry = io.get("tl", {}).get(name)
    if entry is None:
        skipped = io.get("skipped", {}).get(name)
        if skipped is None:
            return []
        return ["**I/O probe:** requires workflow-specific inputs. The small synthetic probe did not execute this function.",
                "", f"Probe note: {skipped.get('note', 'not assessed')}"]
    lines = ["**Observed AnnData keys in the synthetic I/O probe**", "",
             "| Field | Added | Present before and after | Removed |",
             "|---|---|---|---|"]
    for field, values in entry.get("writes", {}).items():
        lines.append(f"| `{field}` | {_keys(values.get('added', []))} | {_keys(values.get('touched', []))} | {_keys(values.get('removed', []))} |")
    return lines


def _entry(row: dict, notes: dict, io: dict, *, prefix: str) -> str:
    name = row["name"]
    curated = notes.get(name, {})
    summary = curated.get("summary") or row.get("doc") or "Public ScGeo helper; inspect the signature and source for supported arguments."
    lines = [f"### `{name}`", "", summary, "", "```text", _signature(name, row["signature"]), "```", ""]
    if row.get("source"):
        lines += [f"[View implementation]({prefix}{row['source']})", ""]
    extra = curated.get("notes", [])
    if extra:
        lines += [f"- {note}" for note in extra] + [""]
    lines += _io_summary(name, io)
    if row.get("docstring"):
        lines += ["", "<details>", "<summary>Parameters, returns and usage notes</summary>", "",
                  "```text", row["docstring"], "```", "", "</details>"]
    return "\n".join(lines).rstrip() + "\n"


def _module(section: str, rows: list[dict], notes: dict, io: dict, *, prefix: str) -> str:
    label, purpose = MODULES[section]
    lines = [f"## scgeo.{section}", "", f"**{label}.** {purpose}", ""]
    lines += [f"- [`{row['name'].split('.')[-1]}`](#{_anchor(row['name'])})" for row in rows]
    lines += [""] + [_entry(row, notes, io, prefix=prefix) for row in rows]
    return "\n".join(lines).rstrip() + "\n"


def expected_outputs(root: Path = ROOT) -> dict[Path, str]:
    api = _load(root / "api_manifest.json")
    io = _load(root / "scgeo_io_manifest.json")
    notes = _load(root / "docs/contracts/api_notes.json")
    names = {row["name"] for section in NAMESPACES for row in api[section]}
    stale = sorted(set(notes) - names)
    if stale:
        raise ValueError(f"Curated notes refer to missing public APIs: {stale}")
    version = api["scgeo_version"]
    overview = ["# API reference", "", f"Public API for **ScGeo {version}** · **{len(names)} exported callables**.", "",
                "This reference describes the current checkout. Plotting additions marked [unreleased in the changelog](../CHANGELOG.md) require that development checkout; they are not included in the published 0.1.0 distribution.", "",
                "[Quick start](QUICKSTART.md) · [Installation](INSTALLATION.md) · [Interpretation](RESULT_INTERPRETATION.md)", "",
                "Start with `tl.robust_shift`, `get.state_report` and `pl.state_evidence_panel`. Add `tl.representation_stability` and `tl.local_geometry_stability` when the required representations are available.", "",
                "| Namespace | Purpose | Browse separately |", "|---|---|---|"]
    overview += [f"| [`scgeo.{section}`](#{_anchor('scgeo.' + section)}) | {MODULES[section][0]} | [{len(api[section])} callables](api/{section}.md) |" for section in NAMESPACES]
    overview += ["", "## Reading this reference", "",
                 "Signatures, type annotations and defaults are generated from the package's public exports. Expand each function's usage notes for its docstring. Use `help(scgeo.tl.robust_shift)` (or another public symbol) for the documentation of your installed version.", "",
                 "The I/O tables describe key presence observed on a small synthetic input, not every supported workflow. A skipped probe means that domain-specific inputs are needed; it does not mean the API is unavailable. The legacy manifest field `touched` means a key exists before and after the call, not that its values were changed. Nested values and read-only access are not inferred from this probe.", "",
                 "For regeneration and validation, see [API and I/O provenance](contracts/README.md). These development manifests do not replace the paper's frozen analysis records.", ""]
    overview += [_module(section, api[section], notes, io, prefix="../") for section in NAMESPACES]
    output = {root / "docs/api_reference.md": "\n".join(overview).rstrip() + "\n"}
    index = ["# Browse the ScGeo API", "", f"These module pages and the [complete API reference](../api_reference.md) are generated together for ScGeo {version}.", "",
             "The current checkout includes [unreleased plotting additions](../../CHANGELOG.md); use the [plotting guide](../PLOTTING.md) for installation and examples.", "",
             "| Module | Purpose | Callables |", "|---|---|---|"]
    for section in NAMESPACES:
        rows = api[section]
        index.append(f"| [scgeo.{section}]({section}.md) | {MODULES[section][0]} | {len(rows)} |")
        header = [f"# scgeo.{section}", "", f"ScGeo {version} · [Complete reference](../api_reference.md) · [All modules](README.md) · [Quick start](../QUICKSTART.md)", "",
                  "Current-checkout reference: [unreleased plotting additions](../../CHANGELOG.md) require the development checkout described in the [plotting guide](../PLOTTING.md).", "",
                  "Signatures and docstrings match the public exports. I/O tables record a small synthetic probe; keys present before and after are not necessarily modified. [How this reference is generated](../contracts/README.md).", ""]
        body = _module(section, rows, notes, io, prefix="../../").split("\n\n", 1)[1]
        output[root / f"docs/api/{section}.md"] = "\n".join(header) + body
    output[root / "docs/api/README.md"] = "\n".join(index).rstrip() + "\n"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated pages are stale; do not write files")
    args = parser.parse_args()
    outputs = expected_outputs()
    stale = [str(path.relative_to(ROOT)) for path, text in outputs.items()
             if not path.exists() or path.read_text(encoding="utf-8") != text]
    if args.check:
        if stale:
            raise SystemExit("Stale generated API docs: " + ", ".join(stale))
        print(f"PASS: {len(outputs)} generated API pages match the manifests and curated notes.")
        return
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(outputs)} API pages from current manifests and curated notes.")


if __name__ == "__main__":
    main()
