#!/usr/bin/env python3
"""Verify package-relative SHA-256 and byte counts using Python stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def package_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("manifest path must be a relative POSIX string")
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(s in (".", "..") for s in p.parts):
        raise ValueError("unsafe manifest path: " + relative)
    resolved = root.joinpath(*p.parts).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes package root: " + relative)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest: Path) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest requires a nonempty files mapping")
    failures = []
    size_total = 0
    for name, expected in sorted(files.items()):
        path = package_path(root, name)
        if not isinstance(expected, dict):
            raise ValueError("invalid record: " + name)
        count, digest = expected.get("bytes"), expected.get("sha256")
        if type(count) is not int or count < 0 or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid size/hash contract: " + name)
        size_total += count
        if not path.is_file():
            failures.append({"path": name, "reason": "missing_file"})
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        if actual_size != count or actual_sha != digest:
            failures.append({"path": name, "reason": "identity_mismatch", "expected_bytes": count,
                             "actual_bytes": actual_size, "expected_sha256": digest, "actual_sha256": actual_sha})
    return {"status": "pass" if not failures else "fail", "files": len(files),
            "total_bytes": size_total, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, help="Default: ROOT/DATA_MANIFEST.json")
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve(), args.manifest or args.root / "DATA_MANIFEST.json")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] == "pass" else 2
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
