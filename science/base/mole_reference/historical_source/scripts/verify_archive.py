#!/usr/bin/env python3
"""Validate the Mole/JAX cold archive structure and optional payload hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARCHIVES = {
    "gpu205": "mole_plane_archive_224d_gpu205_s20260831_v1",
    "gpu206": "mole_plane_archive_224d_gpu206_s20260832_v1",
}
EXPECTED_PLANES = 2_355_700
EXPECTED_CHUNKS = 118


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(root: Path, host: str, name: str, full_hash: bool) -> dict:
    archive = root / host / "results" / name
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    failures: list[str] = []
    if manifest.get("status") != "complete":
        failures.append("manifest status is not complete")
    if manifest.get("completed_planes") != EXPECTED_PLANES:
        failures.append("completed plane count mismatch")
    if manifest.get("target_planes") != EXPECTED_PLANES:
        failures.append("target plane count mismatch")
    if len(chunks) != EXPECTED_CHUNKS:
        failures.append("chunk count mismatch")

    cursor = 0
    checked_bytes = 0
    for index, chunk in enumerate(chunks):
        path = archive / chunk["path"]
        if chunk.get("index") != index or chunk.get("start") != cursor:
            failures.append(f"chunk {index} is not contiguous")
        cursor = int(chunk.get("stop", -1))
        if not path.is_file():
            failures.append(f"missing {chunk['path']}")
            continue
        size = path.stat().st_size
        checked_bytes += size
        if size != chunk.get("bytes"):
            failures.append(f"size mismatch: {chunk['path']}")
        if full_hash and sha256(path) != chunk.get("sha256"):
            failures.append(f"hash mismatch: {chunk['path']}")
    if cursor != EXPECTED_PLANES:
        failures.append("final chunk cursor mismatch")

    resume = manifest.get("resume_state", {})
    resume_path = archive / resume.get("path", "")
    if not resume_path.is_file():
        failures.append("final resume state is missing")
    elif full_hash and sha256(resume_path) != resume.get("sha256"):
        failures.append("final resume state hash mismatch")

    return {
        "host": host,
        "archive": str(archive),
        "status": "pass" if not failures else "fail",
        "full_hash": full_hash,
        "chunks": len(chunks),
        "completed_planes": manifest.get("completed_planes"),
        "checked_bytes": checked_bytes,
        "manifest_sha256": sha256(manifest_path),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.archive_root.resolve()
    results = [
        validate_archive(root, host, name, args.full_hash)
        for host, name in ARCHIVES.items()
    ]
    required = [
        root / "gpu205" / "audit",
        root / "gpu206" / "audit",
        root / "gpu205" / "codes" / "jax_mole_microbatch_distributed.py",
        root / "gpu206" / "codes" / "jax_mole_microbatch_distributed.py",
        root / "gpu205" / "tools" / "run_mole_jax_system_efficiency_stage.sh",
        root / "gpu205" / "audit" / "archive_environment_gpu205_20260807.txt",
        root / "gpu206" / "audit" / "archive_environment_gpu206_20260807.txt",
        root / "gpu205" / "results" / "mole_jax_system_efficiency_20260806",
        root / "gpu206" / "results" / "mole_jax_system_efficiency_20260806",
        root / "references" / "wind-energy-papers" / "mole-code",
        root / "references" / "wind-energy-papers" / "mole-data",
        root / "local_project" / "project_evidence_snapshot_20260807.tar.gz",
        root / "repro_pack" / "mole_jax_system_efficiency_20260806" / "EXPERIMENT_RECORD.md",
        root / "archive_manifest.json",
        root / "SHA256SUMS",
    ]
    missing = [str(path) for path in required if not path.exists()]
    status = "pass" if all(item["status"] == "pass" for item in results) and not missing else "fail"
    payload = {
        "purpose": "mole_jax_system_efficiency_cold_archive_validation",
        "status": status,
        "archive_root": str(root),
        "full_hash": args.full_hash,
        "archives": results,
        "missing_required_paths": missing,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
