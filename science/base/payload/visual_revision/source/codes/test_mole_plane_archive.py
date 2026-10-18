#!/usr/bin/env python3
"""Contract tests for the chunked Mole inlet-plane archive reader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from mole_plane_archive import PlaneArchive, sha256_file


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        chunks_dir = root / "chunks"
        chunks_dir.mkdir()
        values = np.arange(10 * 3 * 2 * 4, dtype=np.float32).reshape(
            10, 3, 2, 4
        )
        chunks = []
        for index, (start, stop) in enumerate(((0, 4), (4, 10))):
            path = chunks_dir / f"chunk_{index}.npy"
            np.save(path, values[start:stop], allow_pickle=False)
            chunks.append(
                {
                    "index": index,
                    "start": start,
                    "stop": stop,
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "purpose": "mole_precursor_inlet_planes",
            "status": "complete",
            "dtype": "float32",
            "plane_shape": [3, 2, 4],
            "target_planes": 10,
            "completed_planes": 10,
            "member_offsets": [1, 5],
            "chunks": chunks,
            "source_state": {"sha256": "test", "completed_steps": 100},
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        archive = PlaneArchive(root)
        assert archive.member_offsets == (1, 5)
        assert archive.source_completed_steps == 100
        assert np.array_equal(archive.read(2, 6), values[2:8])
        assert np.array_equal(archive.read(4, 6), values[4:10])
        assert archive.verify(hashes=True)["chunks"] == 2
        try:
            archive.read(9, 2)
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range archive read did not fail")

        extension_values = np.arange(
            6 * 3 * 2 * 4, dtype=np.float32
        ).reshape(6, 3, 2, 4) + 1000
        extension_dir = root / "extension"
        extension_dir.mkdir()
        extension_path = extension_dir / "chunk.npy"
        np.save(extension_path, extension_values, allow_pickle=False)
        manifest["resume_state"] = {
            "sha256": "terminal-state",
            "completed_steps": 110,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        extension_manifest = {
            "schema_version": 1,
            "purpose": "mole_precursor_inlet_plane_extension",
            "status": "complete",
            "dtype": "float32",
            "plane_shape": [3, 2, 4],
            "base_manifest_sha256": sha256_file(root / "manifest.json"),
            "start_plane": 10,
            "completed_planes": 16,
            "target_planes": 16,
            "source_state": {
                "sha256": "terminal-state",
                "completed_steps": 110,
            },
            "chunks": [
                {
                    "index": 0,
                    "start": 10,
                    "stop": 16,
                    "path": str(extension_path.relative_to(root)),
                    "bytes": extension_path.stat().st_size,
                    "sha256": sha256_file(extension_path),
                }
            ],
        }
        (root / "extension_manifest.json").write_text(
            json.dumps(extension_manifest), encoding="utf-8"
        )
        extended = PlaneArchive(root)
        assert extended.completed_planes == 16
        assert extended.extension_manifest_sha256 is not None
        assert np.array_equal(extended.read(8, 5), np.concatenate(
            (values[8:10], extension_values[:3]), axis=0
        ))

        # A resumed consumer never reads the discarded prefix. The retained
        # tail and extension remain valid after the full base is cold-archived.
        (root / chunks[0]["path"]).unlink()
        tail = PlaneArchive(root)
        assert np.array_equal(tail.read(4, 12), np.concatenate(
            (values[4:10], extension_values), axis=0
        ))
        try:
            tail.read(0, 1)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("discarded archive prefix unexpectedly readable")
    print("plane archive contract tests: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
