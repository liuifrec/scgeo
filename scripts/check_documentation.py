#!/usr/bin/env python3
"""Check local documentation targets, PyPI-safe README links and citation fields.

This is an offline documentation check, not an external URL availability test.
Optionally pass --companion to compare the two repositories' article citations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
LINKS = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)|<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.I)


def article(repo: Path) -> dict:
    citation = yaml.safe_load((repo / "CITATION.cff").read_text(encoding="utf-8"))
    paper = citation["preferred-citation"]
    assert citation["type"] == "software" and paper["type"] == "article"
    published = date.fromisoformat(paper["date-published"])
    assert (published.year, published.month) == (paper["year"], paper["month"])
    bib = (repo / "CITATION.bib").read_text(encoding="utf-8")
    bib_fields = {}
    for line in bib.splitlines():
        match = re.match(r"\s*(\w+)\s*=\s*\{(.*)\},?\s*$", line)
        if match:
            bib_fields[match.group(1)] = match.group(2).replace("{", "").replace("}", "")
    expected_bib = {
        "title": paper["title"], "journal": paper["journal"], "year": str(paper["year"]),
        "volume": str(paper["volume"]), "number": str(paper["issue"]),
        "pages": str(paper["start"]), "doi": paper["doi"], "url": paper["url"],
        "author": " and ".join(f"{a['family-names']}, {a['given-names']}" for a in paper["authors"]),
    }
    assert bib_fields == expected_bib, f"{repo.name}: BibTeX and CFF fields differ"
    ris = defaultdict(list)
    for line in (repo / "CITATION.ris").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z0-9]{2})  - ?(.*)", line)
        assert match, f"{repo.name}: malformed RIS line: {line!r}"
        ris[match.group(1)].append(match.group(2))
    expected_ris = {
        "TY": ["JOUR"], "AU": [f"{a['family-names']}, {a['given-names']}" for a in paper["authors"]],
        "TI": [paper["title"]], "JO": [paper["journal"]], "PY": [str(paper["year"])],
        "DA": [published.strftime("%Y/%m/%d")],
        "VL": [str(paper["volume"])], "IS": [str(paper["issue"])], "SP": [str(paper["start"])],
        "DO": [paper["doi"]], "UR": [paper["url"]], "ER": [""],
    }
    assert dict(ris) == expected_ris, f"{repo.name}: RIS and CFF fields differ"
    assert paper["doi"] == "10.1093/bioadv/vbag266", f"{repo.name}: unexpected paper DOI"
    return paper


def local_target(raw: str, source: Path, repositories: dict[str, Path]) -> Path | None:
    target = raw.strip().split()[0].strip("<>")
    url = urlsplit(target)
    if not url.scheme:
        return (source.parent / unquote(url.path)).resolve() if url.path else None
    parts = [unquote(x) for x in url.path.split("/") if x]
    if url.netloc == "github.com" and len(parts) >= 4 and parts[0] == "liuifrec":
        if parts[1] in repositories and parts[2] in {"blob", "tree"}:
            return repositories[parts[1]].joinpath(*parts[4:])
    if url.netloc == "raw.githubusercontent.com" and len(parts) >= 3 and parts[0] == "liuifrec":
        if parts[1] in repositories:
            return repositories[parts[1]].joinpath(*parts[3:])
    return None


def check_links(repo: Path, repositories: dict[str, Path]) -> int:
    checked = 0
    paths = [repo / "README.md", repo / "CHANGELOG.md"] + sorted((repo / "docs").rglob("*.md"))
    paths += sorted((repo / "notebooks").rglob("README.md")) if (repo / "notebooks").exists() else []
    for path in paths:
        if not path.exists():
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for match in LINKS.finditer(text):
            raw = match.group(1) or match.group(2)
            if repo.name == "scgeo" and path.name == "README.md" and path.parent == repo:
                assert urlsplit(raw).scheme == "https", f"PyPI README requires an absolute URL: {raw}"
            target = local_target(raw, path, repositories)
            if target is not None:
                assert target.exists(), f"{path.relative_to(repo)}: missing target {raw}"
                checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path)
    args = parser.parse_args()
    repositories = {"scgeo": ROOT}
    if args.companion:
        repositories["scgeo-notebooks"] = args.companion.resolve()
    expected = None
    for name, repo in repositories.items():
        paper = article(repo)
        if expected is not None:
            assert paper == expected, "The repositories' preferred article citations differ"
        expected = paper
        count = check_links(repo, repositories)
        print(f"PASS: {name}: {count} local/owned-repository link targets; CFF, BibTeX and RIS agree.")


if __name__ == "__main__":
    main()
