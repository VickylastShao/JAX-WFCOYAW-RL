#!/usr/bin/env python3
"""Generate a source-configured Mole precursor with resumable AB3 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import (
    CompactMoleCase,
    FlowState,
    build_flow_functions,
    make_article_224d_precursor_case,
    make_initial_flow,
    make_precursor_case,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(state_path: Path) -> Path:
    return state_path.with_suffix(".meta.json")


def device_memory_stats() -> dict[str, int | float | str | None]:
    stats = jax.devices()[0].memory_stats() or {}
    keys = (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "largest_free_block_bytes",
    )
    return {
        key: (
            int(stats[key])
            if key in stats and stats[key] is not None
            else None
        )
        for key in keys
    }


def save_state_atomic(
    path: Path,
    state: FlowState,
    case: CompactMoleCase,
    seed: int,
    completed_steps: int,
    diagnostic_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            velocity=np.asarray(state.velocity, dtype=np.float32),
            rhs_previous=np.asarray(state.rhs_previous, dtype=np.float32),
            rhs_previous_2=np.asarray(
                state.rhs_previous_2, dtype=np.float32
            ),
            filtered_disk_speed=np.asarray(
                state.filtered_disk_speed, dtype=np.float32
            ),
            bottom_pressure_gradient=np.asarray(
                state.bottom_pressure_gradient, dtype=np.float32
            ),
            open_x_pressure_gradient=np.asarray(
                state.open_x_pressure_gradient, dtype=np.float32
            ),
            step_index=np.asarray(state.step_index, dtype=np.int32),
        )
    os.replace(temporary, path)
    payload = {
        "schema_version": 3,
        "purpose": "resumable_source_aligned_mole_precursor",
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": seed,
        "completed_steps": completed_steps,
        "simulated_seconds": completed_steps * case.dt,
        "case": asdict(case),
        "shape": list(state.velocity.shape),
        "dtype": str(state.velocity.dtype),
        "state_fields": [
            "velocity",
            "rhs_previous",
            "rhs_previous_2",
            "filtered_disk_speed",
            "bottom_pressure_gradient",
            "open_x_pressure_gradient",
            "step_index",
        ],
        "npz_bytes": path.stat().st_size,
        "npz_sha256": sha256(path),
    }
    if diagnostic_history is not None:
        payload["diagnostic_history"] = diagnostic_history
    metadata = metadata_path(path)
    temporary_metadata = metadata.with_name(metadata.name + ".tmp")
    temporary_metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata)
    return payload


def load_state(
    path: Path,
    case: CompactMoleCase,
    seed: int,
) -> tuple[FlowState, int, list[dict[str, Any]], bool]:
    metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    if metadata["seed"] != seed:
        raise ValueError("resume seed does not match checkpoint seed")
    if metadata["npz_sha256"] != sha256(path):
        raise ValueError("resume checkpoint hash does not match metadata")
    expected_shape = (case.batch, 3, case.nx, case.ny, case.nz)
    if tuple(metadata["shape"]) != expected_shape:
        raise ValueError(
            f"resume shape {metadata['shape']} does not match {expected_shape}"
        )
    with np.load(path, allow_pickle=False) as archive:
        boundary_pressure_history_restored = {
            "bottom_pressure_gradient",
            "open_x_pressure_gradient",
        }.issubset(archive.files)
        bottom_pressure_gradient = (
            archive["bottom_pressure_gradient"]
            if "bottom_pressure_gradient" in archive.files
            else np.zeros(
                (case.batch, 2, case.nx, case.nz), dtype=np.float32
            )
        )
        open_x_pressure_gradient = (
            archive["open_x_pressure_gradient"]
            if "open_x_pressure_gradient" in archive.files
            else np.zeros(
                (case.batch, 2, 2, case.ny, case.nz), dtype=np.float32
            )
        )
        state = FlowState(
            velocity=jnp.asarray(archive["velocity"]),
            rhs_previous=jnp.asarray(archive["rhs_previous"]),
            rhs_previous_2=jnp.asarray(archive["rhs_previous_2"]),
            filtered_disk_speed=jnp.asarray(
                archive["filtered_disk_speed"]
            ),
            bottom_pressure_gradient=jnp.asarray(
                bottom_pressure_gradient
            ),
            open_x_pressure_gradient=jnp.asarray(
                open_x_pressure_gradient
            ),
            step_index=jnp.asarray(archive["step_index"]),
        )
    return (
        state,
        int(metadata["completed_steps"]),
        list(metadata.get("diagnostic_history", [])),
        boundary_pressure_history_restored,
    )


def _normalized_energy_spectrum(values: jax.Array, axis: int) -> list[float]:
    fluctuations = values - jnp.mean(
        values, axis=(-2, -1), keepdims=True
    )
    coefficients = jnp.fft.rfft(fluctuations, axis=axis, norm="ortho")
    if axis == -1:
        energy = jnp.mean(jnp.abs(coefficients) ** 2, axis=(0, 1))
    elif axis == -2:
        energy = jnp.mean(jnp.abs(coefficients) ** 2, axis=(0, 2))
    else:
        raise ValueError(f"unsupported spectrum axis: {axis}")
    normalized = energy / jnp.maximum(jnp.sum(energy), 1.0e-20)
    return np.asarray(normalized, dtype=np.float64).tolist()


def precursor_metrics(
    state: FlowState,
    case: CompactMoleCase,
    functions: dict[str, Any],
) -> dict[str, Any]:
    velocity = state.velocity
    hub_index = int(round(case.hub_height / case.dy))
    hub_u = velocity[:, 0, :, hub_index, :]
    mean_hub_speed = jnp.mean(hub_u, axis=(-2, -1))
    turbulence_intensity = jnp.std(hub_u, axis=(-2, -1)) / jnp.maximum(
        jnp.abs(mean_hub_speed), 1.0e-6
    )
    hub_velocity = velocity[:, :, :, hub_index, :]
    hub_fluctuation = hub_velocity - jnp.mean(
        hub_velocity, axis=(-2, -1), keepdims=True
    )
    turbulent_kinetic_energy = np.float32(0.5) * jnp.mean(
        jnp.sum(hub_fluctuation**2, axis=1)
    )
    mean_streamwise_profile = jnp.mean(
        velocity[:, 0], axis=(0, 1, 3)
    )
    wall_index = int(np.floor(case.wall_sampling_dy))
    wall_fraction = np.float32(case.wall_sampling_dy - wall_index)
    sampled = (
        (np.float32(1.0) - wall_fraction)
        * velocity[:, :, :, wall_index, :]
        + wall_fraction * velocity[:, :, :, wall_index + 1, :]
    )
    wall_speed = jnp.sqrt(sampled[:, 0] ** 2 + sampled[:, 2] ** 2)
    wall_delta = case.wall_sampling_dy * case.dy
    wall_factor = (
        case.von_karman / np.log(wall_delta / case.roughness_length)
    ) ** 2
    friction_velocity_proxy = np.sqrt(wall_factor) * wall_speed
    divergence = functions["divergence"](velocity)
    bottom_wall_velocity = velocity[:, :, :, 0, :]
    top_normal_velocity = velocity[:, 1, :, -1, :]
    top_tangential_delta = (
        velocity[:, (0, 2), :, -1, :]
        - velocity[:, (0, 2), :, -2, :]
    )
    bottom_pressure_history = state.bottom_pressure_gradient
    open_x_pressure_history = state.open_x_pressure_gradient
    return {
        "finite": bool(jnp.all(jnp.isfinite(velocity))),
        "mean_hub_speed_m_per_s": float(jnp.mean(mean_hub_speed)),
        "mean_hub_turbulence_intensity": float(
            jnp.mean(turbulence_intensity)
        ),
        "min_hub_turbulence_intensity": float(
            jnp.min(turbulence_intensity)
        ),
        "hub_turbulent_kinetic_energy_m2_per_s2": float(
            turbulent_kinetic_energy
        ),
        "mean_streamwise_profile_m_per_s": np.asarray(
            mean_streamwise_profile, dtype=np.float64
        ).tolist(),
        "hub_streamwise_spectrum_normalized": _normalized_energy_spectrum(
            hub_u, axis=-2
        ),
        "hub_spanwise_spectrum_normalized": _normalized_energy_spectrum(
            hub_u, axis=-1
        ),
        "mean_wall_friction_velocity_proxy_m_per_s": float(
            jnp.mean(friction_velocity_proxy)
        ),
        "std_wall_friction_velocity_proxy_m_per_s": float(
            jnp.std(friction_velocity_proxy)
        ),
        "interior_divergence_rms": float(
            jnp.sqrt(jnp.mean(divergence[:, 3:-3, 3:-3] ** 2))
        ),
        "bottom_wall_velocity_abs_max_m_per_s": float(
            jnp.max(jnp.abs(bottom_wall_velocity))
        ),
        "bottom_wall_velocity_rms_m_per_s": float(
            jnp.sqrt(jnp.mean(bottom_wall_velocity**2))
        ),
        "top_normal_velocity_abs_max_m_per_s": float(
            jnp.max(jnp.abs(top_normal_velocity))
        ),
        "top_tangential_adjacent_difference_rms_m_per_s": float(
            jnp.sqrt(jnp.mean(top_tangential_delta**2))
        ),
        "bottom_pressure_history_abs_max_m_per_s": float(
            jnp.max(jnp.abs(bottom_pressure_history))
        ),
        "bottom_pressure_history_rms_m_per_s": float(
            jnp.sqrt(jnp.mean(bottom_pressure_history**2))
        ),
        "open_x_pressure_history_abs_max_m_per_s": float(
            jnp.max(jnp.abs(open_x_pressure_history))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--chunk-steps", type=int, default=20000)
    parser.add_argument("--pressure-corrections", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("public_repo_14d", "article_224d_inferred"),
        default="public_repo_14d",
        help=(
            "Precursor geometry. The 224D node count is inferred by preserving "
            "the public periodic-grid spacing because the final input is private."
        ),
    )
    parser.add_argument(
        "--compact-stdout",
        action="store_true",
        help="Print scalar progress summaries instead of full spectra.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch < 1 or args.target_steps < 1 or args.chunk_steps < 1:
        raise ValueError("batch, target steps and chunk steps must be positive")
    case_factory = (
        make_precursor_case
        if args.profile == "public_repo_14d"
        else make_article_224d_precursor_case
    )
    case = case_factory(
        args.batch,
        args.pressure_corrections,
        "incompact3d_staggered",
    )
    functions = build_flow_functions(
        case, periodic_x=True, include_turbines=False
    )
    if args.state.is_file():
        (
            state,
            completed_steps,
            diagnostic_history,
            boundary_pressure_history_restored,
        ) = load_state(
            args.state, case, args.seed
        )
    else:
        state = make_initial_flow(case, functions, args.seed)
        completed_steps = 0
        diagnostic_history = []
        boundary_pressure_history_restored = True
    if completed_steps > args.target_steps:
        raise ValueError("checkpoint is already beyond target steps")

    zero_yaw = jnp.zeros((case.batch, 3), dtype=jnp.float32)
    zero_weights = jnp.zeros(
        (case.batch, 3, case.nx, case.ny, case.nz), dtype=jnp.float32
    )

    def make_runner(length: int):
        def body(current: FlowState, _: None):
            next_state, _ = functions["advance"](
                current, zero_yaw, zero_weights, None
            )
            return next_state, None

        return jax.jit(
            lambda initial: jax.lax.scan(
                body, initial, xs=None, length=length
            )[0]
        )

    timings = []
    while completed_steps < args.target_steps:
        length = min(args.chunk_steps, args.target_steps - completed_steps)
        runner = make_runner(length)
        started = time.perf_counter()
        state = runner(state)
        state.velocity.block_until_ready()
        elapsed = time.perf_counter() - started
        completed_steps += length
        diagnostics_started = time.perf_counter()
        metrics = precursor_metrics(state, case, functions)
        diagnostics_seconds = time.perf_counter() - diagnostics_started
        diagnostic_row = {
            "completed_steps": completed_steps,
            "simulated_seconds": completed_steps * case.dt,
            "metrics": metrics,
        }
        diagnostic_history.append(diagnostic_row)
        checkpoint = save_state_atomic(
            args.state,
            state,
            case,
            args.seed,
            completed_steps,
            diagnostic_history,
        )
        row = {
            "completed_steps": completed_steps,
            "chunk_steps": length,
            "wallclock_seconds": elapsed,
            "diagnostics_seconds": diagnostics_seconds,
            "metrics": metrics,
            "state_sha256": checkpoint["npz_sha256"],
        }
        timings.append(row)
        stdout_row = row
        if args.compact_stdout:
            stdout_row = {
                "completed_steps": completed_steps,
                "chunk_steps": length,
                "wallclock_seconds": elapsed,
                "diagnostics_seconds": diagnostics_seconds,
                "steps_per_second": length / elapsed,
                "finite": metrics["finite"],
                "mean_hub_speed_m_per_s": metrics[
                    "mean_hub_speed_m_per_s"
                ],
                "mean_hub_turbulence_intensity": metrics[
                    "mean_hub_turbulence_intensity"
                ],
                "interior_divergence_rms": metrics[
                    "interior_divergence_rms"
                ],
                "bottom_wall_velocity_abs_max_m_per_s": metrics[
                    "bottom_wall_velocity_abs_max_m_per_s"
                ],
                "top_normal_velocity_abs_max_m_per_s": metrics[
                    "top_normal_velocity_abs_max_m_per_s"
                ],
                "top_tangential_adjacent_difference_rms_m_per_s": metrics[
                    "top_tangential_adjacent_difference_rms_m_per_s"
                ],
                "bottom_pressure_history_abs_max_m_per_s": metrics[
                    "bottom_pressure_history_abs_max_m_per_s"
                ],
                "state_sha256": checkpoint["npz_sha256"],
            }
        print(json.dumps(stdout_row, sort_keys=True), flush=True)

    metrics = diagnostic_history[-1]["metrics"]
    checks = {
        "target_reached": completed_steps == args.target_steps,
        "state_is_finite": metrics["finite"],
        "divergence_is_bounded": metrics["interior_divergence_rms"] < 0.05,
    }
    result = {
        "schema_version": 1,
        "purpose": "resumable_source_aligned_mole_precursor_run",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "configuration": {
            "precursor_profile": args.profile,
            "batch": args.batch,
            "seed": args.seed,
            "target_steps": args.target_steps,
            "chunk_steps": args.chunk_steps,
            "case": asdict(case),
            "boundary_pressure_history_restored": (
                boundary_pressure_history_restored
            ),
        },
        "metrics": metrics,
        "chunks_executed_this_invocation": timings,
        "diagnostic_history": diagnostic_history,
        "state": {
            "path": str(args.state),
            "metadata": str(metadata_path(args.state)),
            "sha256": sha256(args.state),
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": getattr(jax.lib, "__version__", "unknown"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_memory_stats": device_memory_stats(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.compact_stdout:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": result["status"],
                    "completed_steps": completed_steps,
                    "state_sha256": result["state"]["sha256"],
                    "device_memory_stats": result["environment"][
                        "device_memory_stats"
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
