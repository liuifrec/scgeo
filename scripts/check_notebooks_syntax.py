#!/usr/bin/env python
import argparse, ast, glob
import nbformat

def iter_code_cells(nb):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            yield i, cell.source

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No notebooks matched: {args.glob}")

    n_fail = 0
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        for ci, src in iter_code_cells(nb):
            try:
                ast.parse(src)
            except SyntaxError as e:
                n_fail += 1
                print(f"[FAIL] {p} cell={ci} SyntaxError: {e.msg} (line {e.lineno}:{e.offset})")
                if args.fail_fast:
                    raise SystemExit(1)

    if n_fail == 0:
        print("No syntax errors found.")
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
