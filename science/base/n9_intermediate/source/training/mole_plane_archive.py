#!/usr/bin/env python3
"""Read and validate chunked Mole precursor inlet-plane archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PlaneArchive:
    """Random-access reader for contiguous time-major inlet-plane chunks."""

    def __init__(self, root: Path, *, require_complete: bool = True):
        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_NAME
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported plane-archive schema: {self.manifest_path}"
            )
        if self.manifest.get("purpose") != "mole_precursor_inlet_planes":
            raise ValueError(f"unexpected archive purpose: {self.manifest_path}")
        if require_complete and self.manifest.get("status") != "complete":
            raise ValueError(f"plane archive is not complete: {self.root}")
        if self.manifest.get("dtype") != "float32":
            raise ValueError("plane archive must use float32")
        self.plane_shape = tuple(int(x) for x in self.manifest["plane_shape"])
        self.completed_planes = int(self.manifest["completed_planes"])
        self.target_planes = int(self.manifest["target_planes"])
        self.member_offsets = tuple(
            int(value) for value in self.manifest.get("member_offsets", [])
        )
        self.chunks = list(self.manifest.get("chunks", []))
        self._validate_index()

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.manifest_path)

    @property
    def source_state_sha256(self) -> str:
        return str(self.manifest["source_state"]["sha256"])

    @property
    def source_completed_steps(self) -> int:
        return int(self.manifest["source_state"]["completed_steps"])

    def _validate_index(self) -> None:
        expected_start = 0
        for chunk in self.chunks:
            start = int(chunk["start"])
            stop = int(chunk["stop"])
            if start != expected_start or stop <= start:
                raise ValueError("plane archive chunks are not contiguous")
            expected_start = stop
        if expected_start != self.completed_planes:
            raise ValueError(
                "chunk index does not match completed plane count"
            )
        if self.completed_planes > self.target_planes:
            raise ValueError("completed planes exceed archive target")
        if (
            self.manifest.get("status") == "complete"
            and self.completed_planes != self.target_planes
        ):
            raise ValueError("complete archive has an incomplete plane count")

    def verify(self, *, hashes: bool = False) -> dict[str, Any]:
        checked_bytes = 0
        for chunk in self.chunks:
            path = self.root / chunk["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(chunk["bytes"]):
                raise ValueError(f"chunk size mismatch: {path}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (
                int(chunk["stop"]) - int(chunk["start"]),
                *self.plane_shape,
            )
            if values.shape != expected_shape or values.dtype != np.float32:
                raise ValueError(f"chunk array contract mismatch: {path}")
            if hashes and sha256_file(path) != chunk["sha256"]:
                raise ValueError(f"chunk hash mismatch: {path}")
            checked_bytes += path.stat().st_size
        return {
            "chunks": len(self.chunks),
            "completed_planes": self.completed_planes,
            "checked_bytes": checked_bytes,
            "hashes_checked": hashes,
        }

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 1 or start + count > self.completed_planes:
            raise IndexError(
                f"requested planes [{start}, {start + count}) outside "
                f"[0, {self.completed_planes})"
            )
        result = np.empty((count, *self.plane_shape), dtype=np.float32)
        request_stop = start + count
        written = 0
        for chunk in self.chunks:
            chunk_start = int(chunk["start"])
            chunk_stop = int(chunk["stop"])
            overlap_start = max(start, chunk_start)
            overlap_stop = min(request_stop, chunk_stop)
            if overlap_start >= overlap_stop:
                continue
            values = np.load(
                self.root / chunk["path"],
                mmap_mode="r",
                allow_pickle=False,
            )
            source_start = overlap_start - chunk_start
            source_stop = overlap_stop - chunk_start
            length = overlap_stop - overlap_start
            result[written : written + length] = values[
                source_start:source_stop
            ]
            written += length
            if written == count:
                break
        if written != count:
            raise RuntimeError("plane archive index did not satisfy request")
        return result
