#!/usr/bin/env python3
"""Build a complete file inventory for the Mole/JAX cold archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


EXCLUDED = {"archive_manifest.json", "SHA256SUMS", "full_validation.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived_chunk_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for manifest_path in sorted(root.glob("gpu*/results/mole_plane_archive_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = manifest_path.parent
        for chunk in manifest.get("chunks", []):
            relative = str((archive / chunk["path"]).relative_to(root))
            hashes[relative] = chunk["sha256"]
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    known_chunks = archived_chunk_hashes(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative in EXCLUDED:
            continue
        digest = known_chunks.get(relative) or sha256(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "hash_source": "verified_plane_manifest" if relative in known_chunks else "local_file_read",
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "mole_jax_system_efficiency_complete_cold_archive",
        "experiment_id": "mole_jax_system_efficiency_20260806",
        "generated_at": datetime.now().astimezone().isoformat(),
        "archive_root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "plane_chunk_count": len(known_chunks),
        "files": files,
    }
    manifest = root / "archive_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = [f"{item['sha256']}  {item['path']}" for item in files]
    (root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes", "plane_chunk_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
