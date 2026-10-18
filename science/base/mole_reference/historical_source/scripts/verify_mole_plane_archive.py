#!/usr/bin/env python3
"""Regenerate inlet planes and compare them with a stored archive prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from generate_mole_precursor_resumable import load_state, sha256
from jax_mole_compact_sac_benchmark import (
    build_flow_functions,
    make_article_224d_precursor_case,
    make_precursor_case,
)
from jax_mole_continuous_sac import make_precursor_plane_generator
from mole_plane_archive import PlaneArchive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--planes", type=int, default=100)
    parser.add_argument("--pressure-corrections", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("public_repo_14d", "article_224d_inferred"),
        default="article_224d_inferred",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.planes < 2:
        raise ValueError("--planes must be at least two")

    archive = PlaneArchive(args.archive_dir)
    source_state_sha256 = sha256(args.source_state)
    if archive.source_state_sha256 != source_state_sha256:
        raise ValueError("archive source hash does not match source-state file")
    case_factory = (
        make_precursor_case
        if args.profile == "public_repo_14d"
        else make_article_224d_precursor_case
    )
    case = case_factory(
        1, args.pressure_corrections, "incompact3d_staggered"
    )
    functions = build_flow_functions(
        case, periodic_x=True, include_turbines=False
    )
    state, completed_steps, _, boundary_restored = load_state(
        args.source_state, case, args.seed
    )
    if completed_steps != archive.source_completed_steps:
        raise ValueError("archive and source state step counts differ")
    generate = jax.jit(
        make_precursor_plane_generator(functions, args.planes)
    )
    final_state, generated = generate(state)
    generated.block_until_ready()
    generated_host = np.asarray(generated[0], dtype=np.float32)
    stored = archive.read(0, args.planes)
    difference = stored - generated_host
    max_abs = float(np.max(np.abs(difference)))
    relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(generated_host.ravel()), 1.0e-20)
    )
    checks = {
        "archive_integrity": archive.verify(hashes=True)["chunks"] > 0,
        "boundary_pressure_history_restored": boundary_restored,
        "source_advanced_exact_count": (
            int(np.asarray(final_state.step_index))
            - int(np.asarray(state.step_index))
            == args.planes
        ),
        "stored_shape_matches": stored.shape == generated_host.shape,
        "bitwise_equal": bool(np.array_equal(stored, generated_host)),
        "finite": bool(np.isfinite(stored).all()),
        "time_resolved": bool(
            np.sqrt(np.mean((stored[-1] - stored[0]) ** 2)) > 1.0e-6
        ),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "planes": args.planes,
        "max_abs_difference": max_abs,
        "relative_l2_difference": relative_l2,
        "archive_manifest_sha256": archive.manifest_sha256,
        "source_state_sha256": source_state_sha256,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
