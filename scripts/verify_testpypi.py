"""Download TestPyPI artifacts only after their metadata matches the tested files.

No package code is executed here. SHA256 checks are repeated on downloaded bytes.
Network polling is bounded because the index may take time to expose an upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def verify(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest["version"]
    if manifest["name"] != "scgeo" or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Unexpected package identity")
    expected = manifest["files"]
    filenames = {f"scgeo-{version}-py3-none-any.whl", f"scgeo-{version}.tar.gz"}
    if set(expected) != filenames or not all(re.fullmatch(r"[0-9a-f]{64}", v) for v in expected.values()):
        raise ValueError("Invalid expected distribution hashes")
    endpoint = f"https://test.pypi.org/pypi/scgeo/{version}/json"
    for attempt in range(12):
        try:
            with urlopen(endpoint, timeout=20) as response:
                record = json.load(response)
            if record["info"]["name"].lower() != "scgeo" or record["info"]["version"] != version:
                raise ValueError("TestPyPI returned the wrong release")
            published = {f["filename"]: f for f in record["urls"]}
            if not filenames.issubset(published):
                raise URLError("Uploaded files have not all appeared yet")
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 11:
                raise
            time.sleep(10)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in expected.items():
        entry = published[name]
        if entry.get("yanked") or entry["digests"]["sha256"] != digest:
            raise ValueError(f"Index hash mismatch or yanked artifact: {name}")
        url = urlparse(entry["url"])
        if url.scheme != "https" or url.hostname not in {"test-files.pythonhosted.org", "files.pythonhosted.org"}:
            raise ValueError("Unexpected distribution host")
        with urlopen(entry["url"], timeout=60) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Downloaded bytes do not match tested artifact: {name}")
        (destination / name).write_bytes(data)
        print(f"PASS: TestPyPI {name} matches tested SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    verify(args.manifest, args.destination)
