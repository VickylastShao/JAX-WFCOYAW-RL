#!/usr/bin/env python3
"""Read and validate chunked Mole precursor inlet-plane archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_NAME = "manifest.json"
EXTENSION_MANIFEST_NAME = "extension_manifest.json"
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
        self.base_completed_planes = int(self.manifest["completed_planes"])
        self.completed_planes = self.base_completed_planes
        self.target_planes = int(self.manifest["target_planes"])
        self.member_offsets = tuple(
            int(value) for value in self.manifest.get("member_offsets", [])
        )
        self.chunks = list(self.manifest.get("chunks", []))
        self._validate_index()
        self.extension_manifest_path = self.root / EXTENSION_MANIFEST_NAME
        self.extension_manifest: dict[str, Any] | None = None
        if self.extension_manifest_path.is_file():
            self.extension_manifest = json.loads(
                self.extension_manifest_path.read_text(encoding="utf-8")
            )
            self._load_extension_index(require_complete=require_complete)

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.manifest_path)

    @property
    def extension_manifest_sha256(self) -> str | None:
        if self.extension_manifest is None:
            return None
        return sha256_file(self.extension_manifest_path)

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
        if expected_start != self.base_completed_planes:
            raise ValueError(
                "chunk index does not match completed plane count"
            )
        if self.base_completed_planes > self.target_planes:
            raise ValueError("completed planes exceed archive target")
        if (
            self.manifest.get("status") == "complete"
            and self.base_completed_planes != self.target_planes
        ):
            raise ValueError("complete archive has an incomplete plane count")

    def _load_extension_index(self, *, require_complete: bool) -> None:
        assert self.extension_manifest is not None
        extension = self.extension_manifest
        if extension.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported plane-archive extension schema")
        if extension.get("purpose") != "mole_precursor_inlet_plane_extension":
            raise ValueError("unexpected plane-archive extension purpose")
        if require_complete and extension.get("status") != "complete":
            raise ValueError("plane archive extension is not complete")
        if extension.get("dtype") != "float32":
            raise ValueError("plane archive extension must use float32")
        if tuple(int(x) for x in extension["plane_shape"]) != self.plane_shape:
            raise ValueError("plane archive extension shape mismatch")
        if extension.get("base_manifest_sha256") != self.manifest_sha256:
            raise ValueError("plane archive extension base hash mismatch")
        if int(extension["start_plane"]) != self.base_completed_planes:
            raise ValueError("plane archive extension does not start at base end")
        base_resume = self.manifest.get("resume_state")
        extension_source = extension.get("source_state")
        if not isinstance(base_resume, dict) or not isinstance(extension_source, dict):
            raise ValueError("plane archive extension lacks source-state binding")
        if (
            extension_source.get("sha256") != base_resume.get("sha256")
            or int(extension_source.get("completed_steps", -1))
            != int(base_resume.get("completed_steps", -2))
        ):
            raise ValueError("plane archive extension source state mismatch")

        expected_start = self.base_completed_planes
        extension_chunks = list(extension.get("chunks", []))
        for chunk in extension_chunks:
            start = int(chunk["start"])
            stop = int(chunk["stop"])
            if start != expected_start or stop <= start:
                raise ValueError("plane archive extension chunks are not contiguous")
            expected_start = stop
        extension_completed = int(extension["completed_planes"])
        extension_target = int(extension["target_planes"])
        if expected_start != extension_completed:
            raise ValueError("extension index does not match completed plane count")
        if extension_completed > extension_target:
            raise ValueError("extension completed planes exceed target")
        if extension.get("status") == "complete" and extension_completed != extension_target:
            raise ValueError("complete extension has an incomplete plane count")
        self.chunks.extend(extension_chunks)
        self.completed_planes = extension_completed
        self.target_planes = extension_target

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
