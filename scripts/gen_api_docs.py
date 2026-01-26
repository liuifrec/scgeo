import json
from pathlib import Path

API = Path("api_manifest.json")
IO  = Path("scgeo_io_manifest.json")
OUT = Path("docs/api")
OUT.mkdir(parents=True, exist_ok=True)

def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None

def _io_lookup():
    """
    Supports 2 IO formats:
    (1) Old: flat list of dicts with keys: function, mode, path, type, note
    (2) New v2: dict with keys: tl, skipped (each maps full fn -> entry)
    """
    io = _load_json(IO)
    if io is None:
        return {"format": "none", "lookup": {}, "flat": []}

    # v2 format
    if isinstance(io, dict) and ("tl" in io or "skipped" in io):
        lk = {}
        for section in ["tl", "pl", "pp", "get", "bench", "data"]:
            if section in io and isinstance(io[section], dict):
                lk.update(io[section])
        if "skipped" in io and isinstance(io["skipped"], dict):
            lk.update(io["skipped"])
        return {"format": "v2", "lookup": lk, "flat": []}

    # old flat list format
    if isinstance(io, list):
        return {"format": "flat", "lookup": {}, "flat": io}

    # unknown
    return {"format": "unknown", "lookup": {}, "flat": []}

def _fmt_writes_v2(writes: dict) -> str:
    def fmt_block(title, d):
        if not d:
            return f"**{title}**\n- —\n"
        lines = [f"**{title}**"]
        for k in ["obs_cols", "obsm_keys", "layers_keys", "uns_keys"]:
            if k not in d:
                continue
            added   = d[k].get("added", [])
            touched = d[k].get("touched", [])
            removed = d[k].get("removed", [])
            lines.append(f"- `{k}`:")
            lines.append(f"  - added: {', '.join([f'`{x}`' for x in added]) if added else '—'}")
            lines.append(f"  - touched: {', '.join([f'`{x}`' for x in touched]) if touched else '—'}")
            lines.append(f"  - removed: {', '.join([f'`{x}`' for x in removed]) if removed else '—'}")
        return "\n".join(lines) + "\n"

    return fmt_block("Writes / touches (key-level)", writes)

def _fmt_io_for_function(fn_full: str, io_state: dict) -> str:
    """
    Render IO contract text for one function.
    """
    if io_state["format"] == "v2":
        entry = io_state["lookup"].get(fn_full)
        if entry is None:
            return "_No I/O entry in `scgeo_io_manifest.json` for this function._\n"
        ok = entry.get("ok", False)
        note = entry.get("note", "")
        sig = entry.get("signature", "")
        writes = entry.get("writes", {})

        lines = []
        lines.append(f"**Manifest status:** `{'ok' if ok else 'skipped'}`")
        if sig:
            lines.append(f"**Probed signature:** `{sig}`")
        if note:
            lines.append(f"**Note:** {note}")
        lines.append("")
        lines.append(_fmt_writes_v2(writes))
        return "\n".join(lines).strip() + "\n"

    if io_state["format"] == "flat":
        # old behavior: filter list entries by function name
        entries = [e for e in io_state["flat"] if e.get("function") == fn_full]
        if not entries:
            return "_No I/O entries found in `scgeo_io_manifest.json`._\n"
        reads = [e for e in entries if e.get("mode") == "read"]
        writes = [e for e in entries if e.get("mode") == "write"]

        lines = []
        if reads:
            lines.append("**Reads**")
            for e in reads:
                lines.append(f"- `{e.get('path','')}`")
            lines.append("")
        if writes:
            lines.append("**Writes**")
            for e in writes:
                lines.append(f"- `{e.get('path','')}`")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    return "_No I/O entries found in `scgeo_io_manifest.json`._\n"

def render_section(section_name, items, io_state):
    lines = [f"# scgeo.{section_name}\n"]
    for it in items:
        name = it["name"]  # e.g. "scgeo.tl.shift"
        sig = it.get("signature", "")
        doc = (it.get("doc", "") or "").strip()

        lines += [
            f"## `{name}`\n",
            f"**Signature**  \n`{sig}`\n",
        ]
        if doc:
            lines += [f"**Docstring**  \n{doc}\n"]

        lines += ["### I/O contract\n"]
        lines += [_fmt_io_for_function(name, io_state)]
        lines += ["---\n"]

    return "\n".join(lines)

def main():
    api = _load_json(API)
    io_state = _io_lookup()

    for section in ["tl", "pl", "get", "pp", "data", "bench"]:
        if section in api:
            md = render_section(section, api[section], io_state)
            (OUT / f"{section}.md").write_text(md)

    print("Wrote:")
    for section in ["tl", "pl", "get", "pp", "data", "bench"]:
        p = OUT / f"{section}.md"
        if p.exists():
            print(f"  - {p}")

if __name__ == "__main__":
    main()
