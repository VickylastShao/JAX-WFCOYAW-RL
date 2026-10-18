#!/usr/bin/env python3
"""Verify every distributed file against the frozen package-wide inventory."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append({"path": name, "reason": "unsafe_path"})
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({"path": name, "reason": "missing_or_outside_package"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            failures.append({"path": name, "reason": "identity_mismatch"})
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and p.suffix != ".pyc"
              and str(p.relative_to(root)) not in manifest["excluded_metadata_files"]}
    unlisted = sorted(actual - set(manifest["files"]))
    if unlisted:
        failures.append({"reason": "unlisted_files", "paths": unlisted})
    print(json.dumps({"status": "pass" if not failures else "failed",
                      "checked_files": len(manifest["files"]), "failures": failures,
                      "scope": "file identity only; no LES or actor execution"}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
