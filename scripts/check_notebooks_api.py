import ast
import json
from pathlib import Path

import nbformat


def parse_allowed_kwargs(signature: str) -> set[str]:
    """
    Very lightweight parser: extracts param names from a function signature string.
    Assumes the signature is already 'clean' (like your api_manifest).
    """
    inside = signature[signature.find("(") + 1 : signature.rfind(")")]
    parts = []
    depth = 0
    cur = ""
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    out = set()
    for p in parts:
        if not p or p in {"*", "/"}:
            continue
        name = p.split(":")[0].split("=")[0].strip()
        if name and name not in {"self"}:
            out.add(name)
    return out


def build_api_map(manifest_path: Path) -> dict[tuple[str, str], set[str]]:
    m = json.loads(manifest_path.read_text())
    api = {}
    for sec in ("tl", "pl"):
        for e in m.get(sec, []):
            func = e["name"].split(".")[-1]
            api[(sec, func)] = parse_allowed_kwargs(e["signature"])
    return api


class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []  # (sec, func, kwargs, lineno)

    def visit_Call(self, node: ast.Call):
        # detect sg.tl.xxx(...) or sg.pl.xxx(...)
        sec = None
        func = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            inner = node.func.value  # sg.tl or sg.pl
            if isinstance(inner.value, ast.Name) and inner.value.id == "sg":
                if inner.attr in {"tl", "pl"}:
                    sec = inner.attr
                    func = node.func.attr

        if sec and func:
            kw = [k.arg for k in node.keywords if k.arg is not None]
            self.calls.append((sec, func, kw, getattr(node, "lineno", None)))

        self.generic_visit(node)


def main():
    repo = Path(".")
    manifest = repo / "api_manifest.json"
    api = build_api_map(manifest)

    notebooks = list(repo.glob("notebooks/**/*.ipynb")) + list(repo.glob("**/*.ipynb"))
    notebooks = sorted({p.resolve() for p in notebooks})

    bad = 0
    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        for ci, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            try:
                tree = ast.parse(cell.source)
            except SyntaxError:
                continue

            v = CallVisitor()
            v.visit(tree)

            for sec, func, kwargs, lineno in v.calls:
                allowed = api.get((sec, func))
                if allowed is None:
                    # function not present in manifest
                    continue
                unknown = [k for k in kwargs if k not in allowed]
                if unknown:
                    bad += 1
                    print(
                        f"[{nb_path}] cell={ci} line={lineno}  sg.{sec}.{func}  "
                        f"unknown_kwargs={unknown}"
                    )

    if bad:
        raise SystemExit(f"Found {bad} API mismatches.")
    print("OK: no API mismatches found.")


if __name__ == "__main__":
    main()
