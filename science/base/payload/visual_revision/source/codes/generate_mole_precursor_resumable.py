#!/usr/bin/env python3
"""Generate a source-configured Mole precursor with resumable AB3 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
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
    make_wide_article_224d_precursor_case,
)
from jax_mole_layouts import canonical_scale_layouts


@dataclass(frozen=True)
class PrecursorSpeedCondition:
    """Named, immutable references for one precursor operating condition."""

    condition_id: str
    source_precursor_reference_velocity_u1_m_per_s: float
    source_precursor_reference_velocity_u2_m_per_s: float
    case_config_inflow_hub_speed_m_per_s: float
    precursor_plane_advection_speed_m_per_s: float
    friction_velocity_m_per_s: float
    friction_velocity_scaling_rule: str


_PRECURSOR_SPEED_CONDITIONS = {
    condition.condition_id: condition
    for condition in (
        PrecursorSpeedCondition(
            condition_id="accepted8_v1",
            source_precursor_reference_velocity_u1_m_per_s=8.0,
            source_precursor_reference_velocity_u2_m_per_s=8.0,
            case_config_inflow_hub_speed_m_per_s=8.0,
            precursor_plane_advection_speed_m_per_s=8.0,
            friction_velocity_m_per_s=0.442,
            friction_velocity_scaling_rule="accepted_baseline_u_star_0.442",
        ),
        PrecursorSpeedCondition(
            condition_id="speed6_v1",
            source_precursor_reference_velocity_u1_m_per_s=6.0,
            source_precursor_reference_velocity_u2_m_per_s=6.0,
            case_config_inflow_hub_speed_m_per_s=6.0,
            precursor_plane_advection_speed_m_per_s=6.0,
            friction_velocity_m_per_s=0.3315,
            friction_velocity_scaling_rule="0.442*(6.0/8.0)",
        ),
        PrecursorSpeedCondition(
            condition_id="speed10_v1",
            source_precursor_reference_velocity_u1_m_per_s=10.0,
            source_precursor_reference_velocity_u2_m_per_s=10.0,
            case_config_inflow_hub_speed_m_per_s=10.0,
            precursor_plane_advection_speed_m_per_s=10.0,
            friction_velocity_m_per_s=0.5525,
            friction_velocity_scaling_rule="0.442*(10.0/8.0)",
        ),
    )
}


def precursor_speed_condition(condition_id: str) -> PrecursorSpeedCondition:
    try:
        return _PRECURSOR_SPEED_CONDITIONS[condition_id]
    except KeyError as exc:
        raise ValueError(f"unknown precursor speed condition: {condition_id}") from exc


def apply_precursor_speed_condition(
    case: CompactMoleCase,
    condition: PrecursorSpeedCondition,
) -> CompactMoleCase:
    return replace(
        case,
        inflow_hub_speed=condition.case_config_inflow_hub_speed_m_per_s,
        precursor_advection_speed=(
            condition.precursor_plane_advection_speed_m_per_s
        ),
        friction_velocity=condition.friction_velocity_m_per_s,
    )


def validate_resume_case_contract(
    saved_case: dict[str, Any],
    requested_case: CompactMoleCase,
    condition_id: str,
) -> None:
    """Fail closed on case drift for newly introduced speed conditions."""

    if condition_id == "accepted8_v1":
        return
    requested = asdict(requested_case)
    mismatches = {
        key: {"saved": saved_case.get(key), "requested": value}
        for key, value in requested.items()
        if saved_case.get(key) != value
    }
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(f"immutable precursor case mismatch: {fields}")


def validate_resume_source_contract(
    saved_source_sha256: str | None,
    requested_source_sha256: str,
    condition_id: str,
) -> None:
    """Bind new speed-condition checkpoints to the exact generator source."""

    if condition_id == "accepted8_v1":
        return
    if saved_source_sha256 is None:
        raise ValueError("resume checkpoint lacks generator source identity")
    if saved_source_sha256 != requested_source_sha256:
        raise ValueError("resume generator source does not match checkpoint")


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
    provenance: dict[str, Any] | None = None,
    speed_condition: PrecursorSpeedCondition | None = None,
    generator_source_sha256: str | None = None,
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
    if provenance is not None:
        payload["provenance"] = provenance
    if speed_condition is not None:
        payload["speed_condition"] = asdict(speed_condition)
    if generator_source_sha256 is not None:
        payload["generator_source_sha256"] = generator_source_sha256
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
    expected_provenance: dict[str, Any] | None = None,
    expected_speed_condition: PrecursorSpeedCondition | None = None,
    expected_generator_source_sha256: str | None = None,
) -> tuple[FlowState, int, list[dict[str, Any]], bool]:
    metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    if metadata["seed"] != seed:
        raise ValueError("resume seed does not match checkpoint seed")
    if metadata["npz_sha256"] != sha256(path):
        raise ValueError("resume checkpoint hash does not match metadata")
    checkpoint_solver = metadata.get("case", {}).get(
        "compact_solver", "dense_inverse"
    )
    if checkpoint_solver != case.compact_solver:
        raise ValueError(
            "resume compact solver does not match requested solver: "
            f"{checkpoint_solver!r} != {case.compact_solver!r}"
        )
    if (
        expected_provenance is not None
        and metadata.get("provenance") != expected_provenance
    ):
        raise ValueError("resume provenance does not match requested layout")
    if expected_speed_condition is not None:
        saved_speed_condition = metadata.get("speed_condition")
        if (
            saved_speed_condition is None
            and expected_speed_condition.condition_id != "accepted8_v1"
        ):
            raise ValueError("resume checkpoint lacks the requested speed condition")
        if (
            saved_speed_condition is not None
            and saved_speed_condition != asdict(expected_speed_condition)
        ):
            raise ValueError("resume speed condition does not match requested condition")
        validate_resume_case_contract(
            metadata.get("case", {}),
            case,
            expected_speed_condition.condition_id,
        )
        if expected_generator_source_sha256 is not None:
            validate_resume_source_contract(
                metadata.get("generator_source_sha256"),
                expected_generator_source_sha256,
                expected_speed_condition.condition_id,
            )
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
        "--speed-condition",
        choices=tuple(_PRECURSOR_SPEED_CONDITIONS),
        default="accepted8_v1",
        help=(
            "Named immutable precursor speed references. The default preserves "
            "the accepted 8 m/s factory and legacy resume behavior."
        ),
    )
    parser.add_argument(
        "--compact-solver",
        choices=("dense_inverse", "cute_fused_41"),
        default="dense_inverse",
        help=(
            "Vertical compact derivative implementation. cute_fused_41 "
            "requires the pinned CuTe container and must start a new state."
        ),
    )
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
        "--target-turbines",
        type=int,
        choices=(9, 25, 50, 81, 100),
        help=(
            "Widen article_224d_inferred to the canonical target layout. "
            "Omit for the legacy 7D precursor."
        ),
    )
    parser.add_argument(
        "--compact-stdout",
        action="store_true",
        help="Print scalar progress summaries instead of full spectra.",
    )
    return parser.parse_args()


def main() -> int:
    invocation_started = time.perf_counter()
    args = parse_args()
    generator_source_sha256 = sha256(Path(__file__).resolve())
    speed_condition = precursor_speed_condition(args.speed_condition)
    if args.batch < 1 or args.target_steps < 1 or args.chunk_steps < 1:
        raise ValueError("batch, target steps and chunk steps must be positive")
    if args.target_turbines is not None:
        if args.profile != "article_224d_inferred":
            raise ValueError(
                "--target-turbines requires --profile article_224d_inferred"
            )
        target_layout = canonical_scale_layouts()[args.target_turbines]
        case = make_wide_article_224d_precursor_case(
            target_layout,
            args.batch,
            args.pressure_corrections,
            "incompact3d_staggered",
        )
        provenance = {
            "target_turbines": args.target_turbines,
            "target_layout_name": target_layout.name,
            "target_layout_sha256": target_layout.payload_sha256,
            "target_domain_lz_m": target_layout.domain_lz_m,
            "target_nz": target_layout.nz,
        }
    else:
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
        provenance = {
            "target_turbines": None,
            "target_layout_name": "legacy_mole_7D_span",
            "target_layout_sha256": None,
            "target_domain_lz_m": case.domain_lz,
            "target_nz": case.nz,
        }
    case = apply_precursor_speed_condition(case, speed_condition)
    case = replace(case, compact_solver=args.compact_solver)
    functions_started = time.perf_counter()
    functions = build_flow_functions(
        case, periodic_x=True, include_turbines=False
    )
    functions_build_seconds = time.perf_counter() - functions_started
    state_setup_started = time.perf_counter()
    if args.state.is_file():
        (
            state,
            completed_steps,
            diagnostic_history,
            boundary_pressure_history_restored,
        ) = load_state(
            args.state,
            case,
            args.seed,
            provenance,
            speed_condition,
            generator_source_sha256,
        )
    else:
        state = make_initial_flow(case, functions, args.seed)
        completed_steps = 0
        diagnostic_history = []
        boundary_pressure_history_restored = True
    state_setup_seconds = time.perf_counter() - state_setup_started
    if completed_steps > args.target_steps:
        raise ValueError("checkpoint is already beyond target steps")

    zero_yaw = jnp.zeros((case.batch, 3), dtype=jnp.float32)
    # Turbine forcing is disabled for precursor runs, so a scalar placeholder
    # avoids allocating an otherwise unused farm-sized weight tensor.
    zero_weights = jnp.zeros((1,), dtype=jnp.float32)

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
        compile_started = time.perf_counter()
        compiled_runner = runner.lower(state).compile()
        compile_seconds = time.perf_counter() - compile_started
        execution_started = time.perf_counter()
        state = compiled_runner(state)
        state.velocity.block_until_ready()
        execution_seconds = time.perf_counter() - execution_started
        elapsed = compile_seconds + execution_seconds
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
        checkpoint_started = time.perf_counter()
        checkpoint = save_state_atomic(
            args.state,
            state,
            case,
            args.seed,
            completed_steps,
            diagnostic_history,
            provenance,
            speed_condition,
            generator_source_sha256,
        )
        checkpoint_io_seconds = time.perf_counter() - checkpoint_started
        row = {
            "completed_steps": completed_steps,
            "chunk_steps": length,
            "wallclock_seconds": elapsed,
            "compile_seconds": compile_seconds,
            "execution_seconds": execution_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "checkpoint_io_seconds": checkpoint_io_seconds,
            "checkpoint_npz_bytes": checkpoint["npz_bytes"],
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
                "compile_seconds": compile_seconds,
                "execution_seconds": execution_seconds,
                "diagnostics_seconds": diagnostics_seconds,
                "checkpoint_io_seconds": checkpoint_io_seconds,
                "checkpoint_npz_bytes": checkpoint["npz_bytes"],
                "steps_per_second": length / execution_seconds,
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
            "provenance": provenance,
            "batch": args.batch,
            "seed": args.seed,
            "target_steps": args.target_steps,
            "chunk_steps": args.chunk_steps,
            "case": asdict(case),
            "speed_condition": asdict(speed_condition),
            "boundary_pressure_history_restored": (
                boundary_pressure_history_restored
            ),
            "generator_source_sha256": generator_source_sha256,
        },
        "metrics": metrics,
        "chunks_executed_this_invocation": timings,
        "diagnostic_history": diagnostic_history,
        "state": {
            "path": str(args.state),
            "metadata": str(metadata_path(args.state)),
            "sha256": sha256(args.state),
            "bytes": args.state.stat().st_size,
        },
        "environment": {
            "jax": jax.__version__,
            "jaxlib": getattr(jax.lib, "__version__", "unknown"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_memory_stats": device_memory_stats(),
        },
        "invocation_timing": {
            "functions_build_seconds": functions_build_seconds,
            "state_setup_seconds": state_setup_seconds,
            "elapsed_before_result_write_seconds": (
                time.perf_counter() - invocation_started
            ),
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
