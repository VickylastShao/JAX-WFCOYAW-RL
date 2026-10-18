#!/usr/bin/env python3
"""Generate a resumable, chunked archive of Mole precursor inlet planes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

from generate_mole_precursor_resumable import (
    load_state,
    metadata_path,
    save_state_atomic,
    sha256,
)
from jax_mole_compact_sac_benchmark import (
    build_flow_functions,
    make_article_224d_precursor_case,
    make_precursor_case,
    make_wide_article_224d_precursor_case,
)
from jax_mole_continuous_sac import make_precursor_plane_generator
from jax_mole_layouts import canonical_scale_layouts
from mole_plane_archive import MANIFEST_NAME, SCHEMA_VERSION, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-planes", type=int, required=True)
    parser.add_argument("--chunk-planes", type=int, default=20000)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--offset-steps", type=int, default=20000)
    parser.add_argument("--first-offset-steps", type=int, default=20000)
    parser.add_argument("--pressure-corrections", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("public_repo_14d", "article_224d_inferred"),
        default="article_224d_inferred",
    )
    parser.add_argument(
        "--target-turbines",
        type=int,
        choices=(9, 25, 50, 81, 100),
        help=(
            "Widen article_224d_inferred to the canonical target layout. "
            "Omit for the legacy 7D-span precursor."
        ),
    )
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument(
        "--minimum-free-gib-after", type=float, default=20.0
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_source_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    if metadata.get("npz_sha256") != sha256(path):
        raise ValueError("source state hash does not match its metadata")
    return metadata


def _implementation_hashes() -> dict[str, str]:
    code_root = Path(__file__).resolve().parent
    names = (
        "generate_mole_plane_archive.py",
        "generate_mole_precursor_resumable.py",
        "jax_mole_compact_sac_benchmark.py",
        "jax_mole_continuous_sac.py",
        "jax_mole_layouts.py",
        "mole_plane_archive.py",
    )
    return {name: sha256_file(code_root / name) for name in names}


def _build_case_and_provenance(
    args: argparse.Namespace,
    source: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if args.target_turbines is not None:
        if args.profile != "article_224d_inferred":
            raise ValueError(
                "--target-turbines requires --profile article_224d_inferred"
            )
        layout = canonical_scale_layouts()[args.target_turbines]
        case = make_wide_article_224d_precursor_case(
            layout,
            1,
            args.pressure_corrections,
            "incompact3d_staggered",
        )
        provenance = {
            "target_turbines": args.target_turbines,
            "target_layout_name": layout.name,
            "target_layout_sha256": layout.payload_sha256,
            "target_domain_lz_m": layout.domain_lz_m,
            "target_nz": layout.nz,
        }
    else:
        case_factory = (
            make_precursor_case
            if args.profile == "public_repo_14d"
            else make_article_224d_precursor_case
        )
        case = case_factory(
            1, args.pressure_corrections, "incompact3d_staggered"
        )
        provenance = {
            "target_turbines": None,
            "target_layout_name": "legacy_mole_7D_span",
            "target_layout_sha256": None,
            "target_domain_lz_m": case.domain_lz,
            "target_nz": case.nz,
        }

    source_case = source.get("case")
    if not isinstance(source_case, dict):
        raise ValueError("source metadata does not contain a case contract")
    source_solver = source_case.get("compact_solver", "dense_inverse")
    if source_solver not in {"dense_inverse", "cute_fused_41"}:
        raise ValueError(f"unsupported source compact solver: {source_solver!r}")
    case = replace(case, compact_solver=source_solver)

    expected_case = asdict(case)
    required_keys = {
        "batch",
        "nx",
        "ny",
        "nz",
        "domain_lx",
        "domain_ly",
        "domain_lz",
        "dt",
        "pressure_corrections",
        "pressure_projection",
        "compact_solver",
    }
    missing = sorted(required_keys - source_case.keys())
    if missing:
        raise ValueError(f"source case contract lacks required keys: {missing}")
    mismatches = {
        key: {"source": source_case[key], "expected": expected_case[key]}
        for key in sorted(source_case.keys() & expected_case.keys())
        if source_case[key] != expected_case[key]
    }
    if mismatches:
        raise ValueError(f"source case does not match requested archive: {mismatches}")

    source_provenance = source.get("provenance")
    if args.target_turbines is not None and source_provenance != provenance:
        raise ValueError(
            "source provenance does not match requested target layout: "
            f"{source_provenance!r} != {provenance!r}"
        )
    return case, provenance


def _immutable_contract(
    args: argparse.Namespace,
    source: dict[str, Any],
    case: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    offsets = [
        args.first_offset_steps + member * args.offset_steps
        for member in range(args.members)
    ]
    return {
        "target_planes": args.target_planes,
        "chunk_planes": args.chunk_planes,
        "members": args.members,
        "member_offsets": offsets,
        "profile": args.profile,
        "target_turbines": args.target_turbines,
        "target_layout_name": provenance["target_layout_name"],
        "target_layout_sha256": provenance["target_layout_sha256"],
        "plane_shape": [3, case.ny, case.nz],
        "pressure_corrections": args.pressure_corrections,
        "compact_solver": case.compact_solver,
        "seed": args.seed,
        "source_sha256": source["npz_sha256"],
        "source_completed_steps": int(source["completed_steps"]),
        "implementation_sha256": _implementation_hashes(),
    }


def _remove_orphans(root: Path, manifest: dict[str, Any]) -> None:
    referenced = {str(chunk["path"]) for chunk in manifest.get("chunks", [])}
    resume = manifest.get("resume_state")
    if resume and resume.get("archive_relative"):
        referenced.add(str(resume["path"]))
        referenced.add(str(Path(resume["path"]).with_suffix(".meta.json")))
    for directory in (root / "chunks", root / "states"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            relative = str(path.relative_to(root))
            if relative not in referenced:
                path.unlink()


def main() -> int:
    args = parse_args()
    if min(
        args.target_planes,
        args.chunk_planes,
        args.members,
        args.offset_steps,
    ) < 1 or args.first_offset_steps < 0:
        raise ValueError(
            "plane/member counts and spacing must be positive; the first "
            "offset may be zero"
        )
    source_metadata = _load_source_metadata(args.source_state)
    case, provenance = _build_case_and_provenance(args, source_metadata)
    contract = _immutable_contract(
        args, source_metadata, case, provenance
    )
    if args.target_planes <= max(contract["member_offsets"]):
        raise ValueError("target archive does not extend beyond member offsets")

    functions = build_flow_functions(
        case, periodic_x=True, include_turbines=False
    )
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    (args.archive_dir / "chunks").mkdir(exist_ok=True)
    (args.archive_dir / "states").mkdir(exist_ok=True)
    manifest_path = args.archive_dir / MANIFEST_NAME

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise ValueError("existing archive contract does not match arguments")
        _remove_orphans(args.archive_dir, manifest)
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "mole_precursor_inlet_planes",
            "status": "partial",
            "created_at": _now(),
            "updated_at": _now(),
            "dtype": "float32",
            "plane_shape": [3, case.ny, case.nz],
            "bytes_per_plane": 3 * case.ny * case.nz * 4,
            "target_planes": args.target_planes,
            "completed_planes": 0,
            "chunk_planes": args.chunk_planes,
            "chunks": [],
            "member_offsets": contract["member_offsets"],
            "source_state": {
                "path": str(args.source_state),
                "sha256": source_metadata["npz_sha256"],
                "completed_steps": int(source_metadata["completed_steps"]),
                "seed": args.seed,
            },
            "resume_state": {
                "path": str(args.source_state),
                "sha256": source_metadata["npz_sha256"],
                "completed_steps": int(source_metadata["completed_steps"]),
                "archive_relative": False,
            },
            "case": asdict(case),
            "provenance": provenance,
            "implementation_sha256": contract["implementation_sha256"],
            "contract": contract,
            "environment": {
                "jax": jax.__version__,
                "jaxlib": getattr(jax.lib, "__version__", "unknown"),
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
            },
            "evidence_boundary": (
                "Two base trajectories with time offsets do not establish "
                "32 statistically independent precursor realizations."
            ),
        }
        _write_json_atomic(manifest_path, manifest)

    completed = int(manifest["completed_planes"])
    if manifest.get("status") == "complete":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    remaining_payload = (
        args.target_planes - completed
    ) * int(manifest["bytes_per_plane"])
    free_bytes = shutil.disk_usage(args.archive_dir).free
    reserve = int(args.minimum_free_gib_after * 1024**3)
    if free_bytes - remaining_payload < reserve:
        raise RuntimeError(
            f"insufficient disk: free={free_bytes}, remaining_payload="
            f"{remaining_payload}, required_reserve={reserve}"
        )

    resume = manifest["resume_state"]
    resume_path = Path(resume["path"])
    if resume.get("archive_relative"):
        resume_path = args.archive_dir / resume_path
    state, absolute_steps, _, boundary_restored = load_state(
        resume_path,
        case,
        args.seed,
        provenance if source_metadata.get("provenance") is not None else None,
    )
    expected_absolute = int(source_metadata["completed_steps"]) + completed
    if absolute_steps != expected_absolute:
        raise ValueError(
            f"resume state is at {absolute_steps}, expected {expected_absolute}"
        )
    if not boundary_restored:
        raise ValueError("resume state lacks boundary pressure history")

    generators: dict[int, Any] = {}
    chunks_this_run = 0
    while completed < args.target_planes:
        if args.max_chunks is not None and chunks_this_run >= args.max_chunks:
            break
        length = min(args.chunk_planes, args.target_planes - completed)
        if length not in generators:
            generators[length] = jax.jit(
                make_precursor_plane_generator(functions, length),
                donate_argnums=(0,),
            )
        started = time.perf_counter()
        state, device_planes = generators[length](state)
        device_planes.block_until_ready()
        generation_seconds = time.perf_counter() - started
        planes = np.asarray(device_planes[0], dtype=np.float32)
        if planes.shape != (length, 3, case.ny, case.nz):
            raise RuntimeError(f"unexpected generated plane shape: {planes.shape}")
        if not np.isfinite(planes).all():
            raise FloatingPointError("generated inlet planes contain non-finite values")

        start = completed
        stop = completed + length
        chunk_relative = Path("chunks") / f"planes_{start:09d}_{stop:09d}.npy"
        chunk_path = args.archive_dir / chunk_relative
        write_started = time.perf_counter()
        _write_npy_atomic(chunk_path, planes)
        chunk_hash = sha256_file(chunk_path)
        write_seconds = time.perf_counter() - write_started

        next_state_relative = Path("states") / f"state_{stop:09d}.npz"
        next_state_path = args.archive_dir / next_state_relative
        checkpoint = save_state_atomic(
            next_state_path,
            state,
            case,
            args.seed,
            int(source_metadata["completed_steps"]) + stop,
            provenance=provenance,
        )
        previous_resume = manifest["resume_state"]
        manifest["chunks"].append(
            {
                "index": len(manifest["chunks"]),
                "start": start,
                "stop": stop,
                "path": str(chunk_relative),
                "bytes": chunk_path.stat().st_size,
                "sha256": chunk_hash,
                "generation_seconds": generation_seconds,
                "write_and_hash_seconds": write_seconds,
                "mean_u_m_per_s": float(np.mean(planes[:, 0])),
                "rms_first_last_m_per_s": float(
                    np.sqrt(np.mean((planes[-1] - planes[0]) ** 2))
                ),
            }
        )
        completed = stop
        manifest["completed_planes"] = completed
        manifest["status"] = (
            "complete" if completed == args.target_planes else "partial"
        )
        manifest["updated_at"] = _now()
        manifest["resume_state"] = {
            "path": str(next_state_relative),
            "sha256": checkpoint["npz_sha256"],
            "completed_steps": checkpoint["completed_steps"],
            "archive_relative": True,
        }
        _write_json_atomic(manifest_path, manifest)
        if previous_resume.get("archive_relative"):
            previous_path = args.archive_dir / previous_resume["path"]
            previous_path.unlink(missing_ok=True)
            metadata_path(previous_path).unlink(missing_ok=True)
        chunks_this_run += 1
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "completed_planes": completed,
                    "target_planes": args.target_planes,
                    "chunk": str(chunk_relative),
                    "generation_seconds": generation_seconds,
                    "write_and_hash_seconds": write_seconds,
                    "free_bytes": shutil.disk_usage(args.archive_dir).free,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "completed_planes": completed,
                "target_planes": args.target_planes,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
