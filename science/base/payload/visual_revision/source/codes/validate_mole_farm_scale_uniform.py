#!/usr/bin/env python3
"""Computational-only uniform-inlet physics gate for scalable Mole farms."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import (
    FlowState,
    JAX_REAL_DTYPE,
    NP_REAL_DTYPE,
    build_flow_functions,
    case_metadata,
    environment_metadata,
    make_initial_flow,
    make_scaled_mole_training_case,
)
from jax_mole_layouts import canonical_scale_layouts


IMPLEMENTATION_FILES = (
    "validate_mole_farm_scale_uniform.py",
    "jax_mole_layouts.py",
    "jax_mole_compact_sac_benchmark.py",
    "incompact3d_filter.py",
    "incompact3d_staggered.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turbines", type=int, choices=(3, 9, 25, 50, 81, 100), default=9
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--average-start", type=int, default=500)
    parser.add_argument(
        "--mass-flow-reference",
        type=float,
        help="matched N=3 maximum inlet/outlet relative flux mismatch",
    )
    parser.add_argument(
        "--mass-flow-degradation-tolerance", type=float, default=1.0e-3
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yaw_cases(rows: int, columns: int) -> dict[str, np.ndarray]:
    count = rows * columns
    zero = np.zeros((count,), dtype=np.float32)
    positive = zero.copy()
    positive[:columns] = 20.0
    negative = -positive
    asymmetric = zero.copy()
    pattern = np.asarray((20.0, -15.0, 10.0), dtype=np.float32)
    asymmetric[:columns] = np.resize(pattern, columns)
    if rows > 1:
        second_pattern = np.asarray((-8.0, 5.0, -3.0), dtype=np.float32)
        asymmetric[columns : 2 * columns] = np.resize(
            second_pattern, columns
        )
    return {
        "zero": zero,
        "positive_upstream": positive,
        "negative_upstream": negative,
        "asymmetric": asymmetric,
    }


def main() -> int:
    args = parse_args()
    if not 0 <= args.average_start < args.steps:
        raise ValueError("average-start must be in [0, steps)")
    layout = canonical_scale_layouts()[args.turbines]
    case = make_scaled_mole_training_case(layout, batch=1)
    functions = build_flow_functions(
        case, periodic_x=False, include_turbines=True
    )
    initial_velocity = jnp.broadcast_to(
        functions["base_velocity"],
        (1, 3, case.nx, case.ny, case.nz),
    )
    initial_state = make_initial_flow(
        case,
        functions,
        seed=20260807,
        precursor_velocity=initial_velocity,
    )
    inlet = jnp.zeros(
        (1, 3, case.ny, case.nz), dtype=JAX_REAL_DTYPE
    ).at[:, 0].set(NP_REAL_DTYPE(case.inflow_hub_speed))

    def integrate(state: FlowState, yaw: jax.Array):
        weights = functions["disk_weights"](yaw)

        def body(current: FlowState, _: None):
            return functions["advance"](current, yaw, weights, inlet)

        return jax.lax.scan(body, state, xs=None, length=args.steps)

    compiled = jax.jit(integrate)
    results: dict[str, dict[str, Any]] = {}
    rows = layout.rows
    columns = layout.columns
    for name, host_yaw in yaw_cases(rows, columns).items():
        yaw = jnp.asarray(host_yaw[None], dtype=JAX_REAL_DTYPE)
        started = time.perf_counter()
        final_state, power_trace = compiled(initial_state, yaw)
        jax.block_until_ready((final_state, power_trace))
        wallclock_seconds = time.perf_counter() - started
        divergence = functions["diagnostic_divergence"](
            functions["divergence"](final_state.velocity)
        )
        state_finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(final_state)]
            )
        )
        mass_in = jnp.mean(final_state.velocity[:, 0, 0], axis=(-2, -1))
        mass_out = jnp.mean(final_state.velocity[:, 0, -1], axis=(-2, -1))
        compact = {
            "state_finite": state_finite,
            "divergence_rms": jnp.sqrt(jnp.mean(divergence**2)),
            "peak_abs_velocity_m_per_s": jnp.max(jnp.abs(final_state.velocity)),
            "mass_flow_relative_error": jnp.abs(mass_out - mass_in)
            / jnp.maximum(jnp.abs(mass_in), NP_REAL_DTYPE(1.0e-6)),
            "power_mean_watts": jnp.mean(
                power_trace[args.average_start :], axis=0
            )[0],
            "filtered_disk_speed_m_per_s": final_state.filtered_disk_speed[0],
        }
        host = jax.tree.map(np.asarray, compact)
        power = host["power_mean_watts"]
        row_power = power.reshape(rows, columns).mean(axis=1)
        results[name] = {
            "yaw_degrees": host_yaw.tolist(),
            "wallclock_seconds_including_compile_if_first": wallclock_seconds,
            "state_finite": bool(host["state_finite"]),
            "divergence_rms": float(host["divergence_rms"]),
            "peak_abs_velocity_m_per_s": float(
                host["peak_abs_velocity_m_per_s"]
            ),
            "mass_flow_relative_error": float(
                host["mass_flow_relative_error"][0]
            ),
            "farm_power_mw": float(np.sum(power) / 1.0e6),
            "power_by_turbine_mw": (power / 1.0e6).tolist(),
            "mean_power_by_row_mw": (row_power / 1.0e6).tolist(),
            "filtered_disk_speed_m_per_s": host[
                "filtered_disk_speed_m_per_s"
            ].tolist(),
        }

    zero_rows = np.asarray(results["zero"]["mean_power_by_row_mw"])
    positive_rows = np.asarray(
        results["positive_upstream"]["mean_power_by_row_mw"]
    )
    upstream_self_loss_percent = 100.0 * (
        positive_rows[0] / zero_rows[0] - 1.0
    )
    positive_total = results["positive_upstream"]["farm_power_mw"]
    negative_total = results["negative_upstream"]["farm_power_mw"]
    sign_symmetry_relative_percent = 100.0 * abs(
        positive_total - negative_total
    ) / max(abs(positive_total), abs(negative_total), 1.0e-12)
    maximum_mass_flow_relative_error = max(
        result["mass_flow_relative_error"] for result in results.values()
    )
    checks = {
        "all_states_finite": all(
            result["state_finite"] for result in results.values()
        ),
        "all_divergence_rms_below_1e_5": all(
            result["divergence_rms"] < 1.0e-5 for result in results.values()
        ),
        "all_peak_velocities_below_50_m_per_s": all(
            result["peak_abs_velocity_m_per_s"] < 50.0
            for result in results.values()
        ),
        "all_turbine_powers_nonnegative": all(
            min(result["power_by_turbine_mw"]) >= 0.0
            for result in results.values()
        ),
        "zero_yaw_row_power_is_nonincreasing": bool(
            np.all(np.diff(zero_rows) <= 1.0e-6)
        ),
        "upstream_yaw_has_self_loss": bool(
            upstream_self_loss_percent < -1.0
        ),
        "positive_negative_yaw_sign_symmetry_within_0_1_percent": (
            bool(sign_symmetry_relative_percent < 0.1)
        ),
    }
    if args.mass_flow_reference is not None:
        checks["mass_flow_not_degraded_from_matched_n3"] = bool(
            maximum_mass_flow_relative_error
            <= args.mass_flow_reference + args.mass_flow_degradation_tolerance
        )
    code_dir = Path(__file__).resolve().parent
    output = {
        "schema_version": 1,
        "purpose": "mole_farm_scale_uniform_inlet_computational_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "environment": environment_metadata(),
        "case": case_metadata(case),
        "configuration": {
            "steps": args.steps,
            "average_start": args.average_start,
            "inlet": "uniform streamwise 8 m/s; computational-only",
            "case_execution": "four yaw cases run sequentially with one compiled graph",
            "mass_flow_reference": args.mass_flow_reference,
            "mass_flow_degradation_tolerance": (
                args.mass_flow_degradation_tolerance
            ),
        },
        "derived_metrics": {
            "upstream_self_loss_percent": float(upstream_self_loss_percent),
            "positive_negative_farm_power_relative_difference_percent": (
                float(sign_symmetry_relative_percent)
            ),
            "maximum_mass_flow_relative_error": float(
                maximum_mass_flow_relative_error
            ),
            "absolute_mass_flow_error_below_1e_5": bool(
                maximum_mass_flow_relative_error < 1.0e-5
            ),
        },
        "cases": results,
        "implementation_sha256": {
            name: file_sha256(code_dir / name) for name in IMPLEMENTATION_FILES
        },
        "evidence_boundary": (
            "This is a deterministic uniform-inlet numerical and qualitative "
            "physics gate. It does not validate a wide turbulent precursor, "
            "closed-loop control gain, or farm-scale RL training."
        ),
    }
    encoded_without_hash = json.dumps(output, indent=2, sort_keys=True)
    output["payload_sha256_without_hash_field"] = hashlib.sha256(
        encoded_without_hash.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
