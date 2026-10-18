#!/usr/bin/env python3
"""Benchmark same-fidelity farm LES collection with serial microbatch waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import (
    JAX_REAL_DTYPE,
    NP_REAL_DTYPE,
    build_flow_functions,
    environment_metadata,
    make_initial_flow,
    make_scaled_mole_training_case,
)
from jax_mole_layouts import canonical_scale_layouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turbines", type=int, choices=(3, 9, 25, 50, 81, 100), default=81
    )
    parser.add_argument("--logical-environments-per-device", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--les-steps-per-interaction", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pressure-corrections", type=int, default=1)
    parser.add_argument(
        "--compact-solver",
        choices=("dense_inverse", "cute_fused_41"),
        default="dense_inverse",
    )
    parser.add_argument(
        "--rhs-assembly", choices=("tensor", "componentwise"), default="tensor"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def memory_stats() -> dict[str, int | None]:
    stats = jax.devices()[0].memory_stats() or {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    return {key: int(stats[key]) if stats.get(key) is not None else None for key in keys}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    positive = (
        args.logical_environments_per_device,
        args.microbatch,
        args.les_steps_per_interaction,
        args.warmups,
        args.repeats,
    )
    if min(positive) < 1:
        raise ValueError("all benchmark counts must be positive")
    if args.logical_environments_per_device % args.microbatch:
        raise ValueError("microbatch must divide logical environments per device")
    waves = args.logical_environments_per_device // args.microbatch

    layout = canonical_scale_layouts()[args.turbines]
    case = make_scaled_mole_training_case(
        layout,
        batch=args.microbatch,
        pressure_corrections=args.pressure_corrections,
        compact_solver=args.compact_solver,
        rhs_assembly=args.rhs_assembly,
    )
    functions = build_flow_functions(case, periodic_x=False, include_turbines=True)
    initial_velocity = jnp.broadcast_to(
        functions["base_velocity"],
        (case.batch, 3, case.nx, case.ny, case.nz),
    )
    state = make_initial_flow(
        case,
        functions,
        seed=20260809,
        precursor_velocity=initial_velocity,
    )
    yaw = jnp.zeros((case.batch, case.num_turbines), dtype=JAX_REAL_DTYPE)
    inlet = jnp.zeros(
        (case.batch, 3, case.ny, case.nz), dtype=JAX_REAL_DTYPE
    ).at[:, 0].set(NP_REAL_DTYPE(case.inflow_hub_speed))

    def one_interaction(initial, current_yaw):
        # The production SAC path computes dynamic-yaw disk weights inside
        # the compiled control interaction and reuses them across LES steps.
        weights = functions["disk_weights"](current_yaw)

        def body(current, _):
            next_state, _ = functions["advance"](
                current, current_yaw, weights, inlet
            )
            return next_state, None

        return jax.lax.scan(
            body, initial, xs=None, length=args.les_steps_per_interaction
        )[0]

    compiled = jax.jit(one_interaction, donate_argnums=(0,))
    compile_started = time.perf_counter()
    state = compiled(state, yaw)
    jax.block_until_ready(state)
    compile_and_first_seconds = time.perf_counter() - compile_started
    for _ in range(args.warmups - 1):
        state = compiled(state, yaw)
        jax.block_until_ready(state)

    samples = []
    for repeat in range(args.repeats):
        started = time.perf_counter()
        for _ in range(waves):
            state = compiled(state, yaw)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - started
        samples.append(elapsed)
        print(
            json.dumps(
                {
                    "repeat": repeat + 1,
                    "wave_seconds": elapsed,
                    "logical_interactions_per_second": (
                        args.logical_environments_per_device / elapsed
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    divergence = functions["diagnostic_divergence"](
        functions["divergence"](state.velocity)
    )
    diagnostics = jax.device_get(
        {
            "finite": jnp.all(jnp.isfinite(state.velocity)),
            "divergence_rms": jnp.sqrt(jnp.mean(divergence**2)),
            "peak_abs_velocity_m_per_s": jnp.max(jnp.abs(state.velocity)),
        }
    )
    median_wave_seconds = statistics.median(samples)
    checks = {
        "gpu_backend": jax.default_backend() == "gpu",
        "one_visible_device": len(jax.local_devices()) == 1,
        "state_is_finite": bool(diagnostics["finite"]),
        "divergence_rms_below_1e_5": float(diagnostics["divergence_rms"]) < 1.0e-5,
        "peak_velocity_below_50_m_per_s": (
            float(diagnostics["peak_abs_velocity_m_per_s"]) < 50.0
        ),
    }
    code_dir = Path(__file__).resolve().parent
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "same_fidelity_farm_les_collector_microbatch_gate",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "environment": environment_metadata(),
        "layout": {**asdict(layout), "layout_sha256": layout.payload_sha256},
        "configuration": {
            "logical_environments_per_device": args.logical_environments_per_device,
            "microbatch": args.microbatch,
            "serial_microbatch_waves": waves,
            "les_steps_per_interaction": args.les_steps_per_interaction,
            "pressure_corrections": args.pressure_corrections,
            "compact_solver": args.compact_solver,
            "rhs_assembly": args.rhs_assembly,
            "precision": str(JAX_REAL_DTYPE),
            "case": asdict(case),
        },
        "timing": {
            "compile_and_first_interaction_seconds": compile_and_first_seconds,
            "wave_seconds": samples,
            "median_wave_seconds": median_wave_seconds,
            "median_logical_interactions_per_second_per_device": (
                args.logical_environments_per_device / median_wave_seconds
            ),
            "projected_eight_device_interactions_per_second": (
                8.0 * args.logical_environments_per_device / median_wave_seconds
            ),
        },
        "device_memory": memory_stats(),
        "diagnostics": {
            key: bool(value) if key == "finite" else float(value)
            for key, value in diagnostics.items()
        },
        "implementation_sha256": {
            name: file_sha256(code_dir / name)
            for name in (
                "benchmark_mole_farm_scale_collector.py",
                "jax_mole_layouts.py",
                "jax_mole_compact_sac_benchmark.py",
                "incompact3d_filter.py",
                "incompact3d_staggered.py",
            )
        },
        "evidence_boundary": (
            "This computational gate measures farm-LES collection only with a "
            "uniform inlet. Dynamic-yaw disk weights are generated inside each "
            "compiled interaction as in the production collector. It excludes "
            "archive I/O, policy inference, SAC updates, synchronization, resets, "
            "checkpoints and evaluation, and cannot support a control-performance "
            "or end-to-end training-time claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
