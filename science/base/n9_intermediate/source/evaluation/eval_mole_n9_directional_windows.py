#!/usr/bin/env python3
"""Evaluate an N=9 direction-aware Graph-SAC actor on paired archive windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from eval_mole_n9_locked_windows import (
    make_block_scan,
    parse_offsets,
    reset_controller_input,
    run_blocks,
    summarize_metrics,
)
from jax_mole_compact_sac_benchmark import (
    layout_contract,
    make_scaled_mole_training_case,
)
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    initialize_continuous_training,
    initialize_evaluation_from_training,
    load_actor_checkpoint,
    make_evaluation_step,
    protocol_metadata,
)
from jax_mole_direction_geometry import (
    common_direction_center,
    rotate_layout_to_flow_frame,
)
from jax_mole_graph_sac_directional import (
    append_direction_branch_condition,
    append_direction_condition,
    build_directional_graph,
    directional_edge_features,
    directional_graph_actor_deterministic,
    equivariant_directional_graph_actor_deterministic,
    flow_responsive_steering_branch,
)
from jax_mole_graph_sac_v2 import (
    build_wake_graph,
    graph_v2_actor_deterministic,
)
from jax_mole_layouts import canonical_scale_layouts
from mole_plane_archive import PlaneArchive
from mole_n9_wp5_contract import parse_cumulative_interactions


DIRECTION_PROTOCOL_ANGLES_DEG = (
    -30.0,
    -15.0,
    -12.5,
    -10.0,
    -7.5,
    -5.0,
    -2.5,
    0.0,
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    30.0,
)
LAYOUT_SHA256 = (
    "50e9840b74f129fc8672b5f2551f49a3ce9d76a6b9ba814a11ecf6bdc397c3fc"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_window_offsets(
    offsets: tuple[int, ...], window_planes: int, completed_planes: int
) -> tuple[tuple[int, int], ...]:
    """Return sorted intervals after bounds and non-overlap checks."""

    intervals = tuple(
        sorted((offset, offset + window_planes) for offset in offsets)
    )
    if intervals[-1][1] > completed_planes:
        raise ValueError("an evaluation window exceeds the archive")
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("evaluation windows overlap")
    return intervals


def cumulative_controller_summaries(
    metrics: dict[str, np.ndarray], points: tuple[int, ...]
) -> dict[str, dict[str, Any]]:
    """Summarize prefixes without treating them as independent evaluations."""

    summaries: dict[str, dict[str, Any]] = {}
    for point in points:
        prefix = {name: values[:point] for name, values in metrics.items()}
        if any(values.shape[0] != point for values in prefix.values()):
            raise ValueError("a cumulative point exceeds collected interactions")
        summaries[str(point)] = summarize_metrics(prefix)
    return summaries


def make_policy_adapter(
    actor_metadata: dict[str, Any],
    target_angle_deg: float,
    canonical_case: Any,
    target_case: Any,
    frame_stack: int,
    steering_branch: float | None = None,
    steering_branch_mode: str = "fixed",
) -> tuple[Any, dict[str, Any]]:
    """Build the deterministic policy closure from frozen checkpoint metadata."""

    architecture = actor_metadata.get("policy_architecture")
    conditioned = bool(actor_metadata.get("direction_condition", False))
    branch_conditioned = bool(
        actor_metadata.get("steering_branch_condition", False)
    )
    graph_geometry = actor_metadata.get("graph_geometry", "case")
    if architecture not in {"graph_sac_v2", "graph_sac_directional"}:
        raise ValueError(f"unsupported policy architecture: {architecture}")
    if architecture == "graph_sac_directional" and not conditioned:
        raise ValueError("directional Graph-SAC checkpoint lacks its condition")
    if branch_conditioned and architecture != "graph_sac_directional":
        raise ValueError("steering branch requires directional Graph-SAC")
    if steering_branch_mode not in {"fixed", "flow-responsive"}:
        raise ValueError("unsupported steering branch mode")
    if (
        branch_conditioned
        and steering_branch_mode == "fixed"
        and steering_branch not in (-1.0, 1.0)
    ):
        raise ValueError("fixed branch-conditioned checkpoint requires branch -1 or +1")
    if (
        branch_conditioned
        and steering_branch_mode == "flow-responsive"
        and steering_branch is not None
    ):
        raise ValueError("flow-responsive branch mode does not accept a fixed branch")
    if not branch_conditioned and steering_branch is not None:
        raise ValueError("steering branch supplied to an unconditioned checkpoint")
    if not branch_conditioned and steering_branch_mode != "fixed":
        raise ValueError("flow-responsive mode requires a branch-conditioned checkpoint")
    if graph_geometry not in {"canonical", "case"}:
        raise ValueError(f"unsupported graph geometry: {graph_geometry}")

    graph_case = canonical_case if graph_geometry == "canonical" else target_case
    if architecture == "graph_sac_v2":
        graph = build_wake_graph(
            graph_case.turbine_positions_m,
            diameter_m=graph_case.diameter,
            wind_vector_xz=(1.0, 0.0),
            condition_dim=(2 if conditioned else 0),
        )

        def deterministic(actor: Any, observations: Any, _policy_mode: str) -> Any:
            policy_observation = (
                append_direction_condition(observations, target_angle_deg)
                if conditioned
                else observations
            )
            return graph_v2_actor_deterministic(
                actor, policy_observation, graph, frame_stack
            )

        graph_summary = {
            "type": "sparse_wake_graph_v2",
            "geometry": graph_geometry,
            "edge_count": int(graph.edge_sources.shape[0]),
            "conditioned": conditioned,
        }
    else:
        graph = build_directional_graph(
            canonical_case.turbine_positions_m, canonical_case.diameter
        )

        actor_deterministic = (
            equivariant_directional_graph_actor_deterministic
            if actor_metadata.get("reflection_equivariant_actor", False)
            else directional_graph_actor_deterministic
        )

        def deterministic(actor: Any, observations: Any, _policy_mode: str) -> Any:
            selected_branch = (
                flow_responsive_steering_branch(
                    observations,
                    graph,
                    target_angle_deg,
                    frame_stack=frame_stack,
                )
                if branch_conditioned and steering_branch_mode == "flow-responsive"
                else steering_branch
            )
            policy_observation = (
                append_direction_branch_condition(
                    observations, target_angle_deg, selected_branch
                )
                if branch_conditioned
                else append_direction_condition(observations, target_angle_deg)
            )
            return actor_deterministic(
                actor, policy_observation, graph, frame_stack
            )

        probe = (
            append_direction_branch_condition(
                jnp.zeros((1, 702), dtype=jnp.float32),
                target_angle_deg,
                (
                    0.0
                    if steering_branch_mode == "flow-responsive"
                    else steering_branch
                ),
            )
            if branch_conditioned
            else append_direction_condition(
                jnp.zeros((1, 702), dtype=jnp.float32), target_angle_deg
            )
        )
        edge_features = np.asarray(directional_edge_features(probe, graph))[0]
        graph_summary = {
            "type": "complete_directional_graph",
            "geometry": "canonical",
            "edge_count": int(graph.edge_sources.shape[0]),
            "conditioned": True,
            "steering_branch_conditioned": branch_conditioned,
            "steering_branch_mode": steering_branch_mode,
            "steering_branch": (
                steering_branch if steering_branch_mode == "fixed" else None
            ),
            "flow_selector": (
                {
                    "direction_deadband_deg": 2.5,
                    "barycenter_deadband_d": 0.02,
                    "upstream_probe_x_over_d": [-1.5, -1.0],
                    "symmetric_fallback_branch": 0.0,
                }
                if branch_conditioned and steering_branch_mode == "flow-responsive"
                else None
            ),
            "reflection_equivariant_actor": bool(
                actor_metadata.get("reflection_equivariant_actor", False)
            ),
            "downstream_edge_count": int(np.sum(edge_features[:, 3] > 0.5)),
            "mean_overlap": float(np.mean(edge_features[:, 4])),
        }
    return deterministic, graph_summary


def scale_policy_adapter(deterministic: Any, action_scale: float) -> Any:
    """Scale normalized absolute-yaw targets toward the zero-yaw baseline."""

    if not math.isfinite(action_scale) or not 0.0 <= action_scale <= 1.0:
        raise ValueError("action scale must be finite and in [0, 1]")

    def scaled(actor: Any, observations: Any, policy_mode: str) -> Any:
        return action_scale * deterministic(actor, observations, policy_mode)

    return scaled


def validate_checkpoint_training_layout(
    actor_metadata: dict[str, Any], canonical_layout: Any, center: tuple[float, float]
) -> None:
    training_angle = float(actor_metadata.get("relative_wind_angle_deg", 0.0))
    metadata_center = actor_metadata.get("direction_common_center_xz_m")
    if metadata_center is not None and tuple(float(value) for value in metadata_center) != center:
        raise ValueError("checkpoint direction center differs from declared protocol")
    training_layout = rotate_layout_to_flow_frame(
        canonical_layout, training_angle, center
    )
    training_case = make_scaled_mole_training_case(training_layout, batch=1)
    if actor_metadata.get("layout_contract") != layout_contract(training_case):
        raise ValueError("checkpoint training-layout contract is inconsistent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-wind-angle-deg", type=float, required=True)
    parser.add_argument(
        "--declared-directions",
        default=",".join(str(value) for value in DIRECTION_PROTOCOL_ANGLES_DEG),
        help="Comma-separated direction set used to derive the common layout center.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale deterministic normalized yaw targets toward zero yaw.",
    )
    parser.add_argument("--offsets", default="600000,900000")
    parser.add_argument("--burnin-interactions", type=int, default=100)
    parser.add_argument("--eval-interactions", type=int, default=200)
    parser.add_argument("--block-interactions", type=int, default=25)
    parser.add_argument(
        "--cumulative-interactions",
        default="",
        help=(
            "Optional comma-separated block boundaries ending at eval-interactions; "
            "writes descriptive cumulative summaries without changing selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--steering-branch",
        type=float,
        choices=(-1.0, 1.0),
        help="Required discrete branch for a branch-conditioned Actor.",
    )
    parser.add_argument(
        "--steering-branch-mode",
        choices=("fixed", "flow-responsive"),
        default="fixed",
        help="Use a fixed branch or derive it from direction, probes and yaw memory.",
    )
    parser.add_argument(
        "--force-reflection-equivariant-actor",
        action="store_true",
        help="Diagnostic only: group-average a legacy directional Actor at inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        not math.isfinite(args.relative_wind_angle_deg)
        or args.relative_wind_angle_deg not in DIRECTION_PROTOCOL_ANGLES_DEG
    ):
        raise ValueError("angle is outside the declared direction protocol")
    if not math.isfinite(args.action_scale) or not 0.0 <= args.action_scale <= 1.0:
        raise ValueError("action scale must be finite and in [0, 1]")
    offsets = parse_offsets(args.offsets)
    counts = (
        args.burnin_interactions,
        args.eval_interactions,
        args.block_interactions,
    )
    if min(counts) < 1:
        raise ValueError("interaction counts must be positive")
    if (
        args.burnin_interactions % args.block_interactions
        or args.eval_interactions % args.block_interactions
    ):
        raise ValueError("block size must divide burn-in and evaluation")
    cumulative_points = parse_cumulative_interactions(
        args.cumulative_interactions,
        args.eval_interactions,
        args.block_interactions,
    )

    declared_directions = tuple(
        float(value.strip())
        for value in args.declared_directions.split(",")
        if value.strip()
    )
    if (
        not declared_directions
        or len(set(declared_directions)) != len(declared_directions)
        or not all(math.isfinite(value) for value in declared_directions)
        or args.relative_wind_angle_deg not in declared_directions
    ):
        raise ValueError("relative direction must belong to a finite unique declared set")
    canonical_layout = canonical_scale_layouts()[9]
    if canonical_layout.payload_sha256 != LAYOUT_SHA256:
        raise ValueError("canonical N=9 layout hash changed")
    center = common_direction_center(canonical_layout, declared_directions)
    target_layout = rotate_layout_to_flow_frame(
        canonical_layout, args.relative_wind_angle_deg, center
    )
    canonical_case = make_scaled_mole_training_case(canonical_layout, batch=1)
    case = make_scaled_mole_training_case(target_layout, batch=1)

    archive = PlaneArchive(args.plane_archive, require_complete=True)
    if archive.plane_shape != (3, case.ny, case.nz):
        raise ValueError("archive plane shape does not match N=9 case")
    provenance = archive.manifest.get("provenance", archive.manifest.get("contract", {}))
    if (
        provenance.get("target_layout_sha256") != LAYOUT_SHA256
        or provenance.get("target_turbines") != 9
    ):
        raise ValueError("archive provenance does not match canonical N=9")

    config = ContinuousControlConfig(inlet_mode="time_planes")
    window_planes = (
        args.burnin_interactions + args.eval_interactions
    ) * config.les_steps_per_interaction
    validate_window_offsets(offsets, window_planes, archive.completed_planes)

    actor, actor_metadata = load_actor_checkpoint(args.checkpoint)
    if args.force_reflection_equivariant_actor:
        actor_metadata = dict(actor_metadata)
        actor_metadata["reflection_equivariant_actor"] = True
    if actor_metadata.get("target_turbines") != 9:
        raise ValueError("checkpoint is not an N=9 policy")
    if actor_metadata.get("action_parameterization") != "per_turbine_target":
        raise ValueError("checkpoint action contract differs from protocol")
    validate_checkpoint_training_layout(actor_metadata, canonical_layout, center)
    deterministic, graph_summary = make_policy_adapter(
        actor_metadata,
        args.relative_wind_angle_deg,
        canonical_case,
        case,
        config.frame_stack,
        args.steering_branch,
        args.steering_branch_mode,
    )
    deterministic = scale_policy_adapter(deterministic, args.action_scale)
    policy_diagnostic_fn = None
    if args.steering_branch_mode == "flow-responsive":
        selector_graph = build_directional_graph(
            canonical_case.turbine_positions_m, canonical_case.diameter
        )

        def policy_diagnostic_fn(
            _actor: Any, observations: Any, _policy_mode: str
        ) -> Any:
            return flow_responsive_steering_branch(
                observations,
                selector_graph,
                args.relative_wind_angle_deg,
                frame_stack=config.frame_stack,
            )

    first_plane = archive.read(offsets[0], 1)[0]
    _, _, precursor_functions, farm_functions = initialize_continuous_training(
        case,
        config,
        None,
        args.seed,
        256,
        3.0e-6,
        8,
        initial_inlet=jnp.asarray(first_plane[None]),
        action_parameterization="per_turbine_target",
    )
    burn_step = make_evaluation_step(
        case, config, precursor_functions, farm_functions, "zero"
    )
    zero_step = make_evaluation_step(
        case, config, precursor_functions, farm_functions, "zero"
    )
    sac_step = make_evaluation_step(
        case,
        config,
        precursor_functions,
        farm_functions,
        "sac",
        actor=actor,
        action_parameterization="per_turbine_target",
        actor_deterministic_fn=deterministic,
        actor_diagnostic_fn=policy_diagnostic_fn,
    )
    burn_scan = make_block_scan(burn_step, args.block_interactions)
    controller_scans = {
        "zero": make_block_scan(zero_step, args.block_interactions),
        "sac": make_block_scan(sac_step, args.block_interactions),
    }
    planes_per_block = (
        args.block_interactions * config.les_steps_per_interaction
    )
    burn_blocks = args.burnin_interactions // args.block_interactions
    eval_blocks = args.eval_interactions // args.block_interactions

    windows = []
    total_started = time.perf_counter()
    for window_index, offset in enumerate(offsets):
        initial_plane = archive.read(offset, 1)[0]
        training, _, _, _ = initialize_continuous_training(
            case,
            config,
            None,
            args.seed + window_index,
            256,
            3.0e-6,
            8,
            initial_inlet=jnp.asarray(initial_plane[None]),
            action_parameterization="per_turbine_target",
        )
        initial = initialize_evaluation_from_training(training)
        burn_started = time.perf_counter()
        common, eval_cursor, _ = run_blocks(
            initial,
            archive,
            offset,
            burn_blocks,
            planes_per_block,
            burn_scan,
        )
        burn_seconds = time.perf_counter() - burn_started
        common = reset_controller_input(common)
        controllers = {}
        cumulative_controllers = {}
        for name, scan in controller_scans.items():
            started = time.perf_counter()
            _, final_cursor, metrics = run_blocks(
                common,
                archive,
                eval_cursor,
                eval_blocks,
                planes_per_block,
                scan,
            )
            summary = summarize_metrics(metrics)
            summary["wallclock_seconds"] = time.perf_counter() - started
            summary["final_plane_cursor"] = final_cursor
            controllers[name] = summary
            cumulative_controllers[name] = cumulative_controller_summaries(
                metrics, cumulative_points
            )
        zero_power = controllers["zero"]["mean_farm_power_mw"]
        sac_power = controllers["sac"]["mean_farm_power_mw"]
        cumulative = {}
        for point in cumulative_points:
            key = str(point)
            zero_prefix = cumulative_controllers["zero"][key]
            sac_prefix = cumulative_controllers["sac"][key]
            cumulative[key] = {
                "interactions": point,
                "zero": zero_prefix,
                "sac": sac_prefix,
                "sac_paired_gain_percent": 100.0
                * (
                    sac_prefix["mean_farm_power_mw"]
                    / zero_prefix["mean_farm_power_mw"]
                    - 1.0
                ),
            }
        windows.append(
            {
                "window_index": window_index,
                "start_plane": offset,
                "burnin_final_plane": eval_cursor,
                "burnin_wallclock_seconds": burn_seconds,
                "controllers": controllers,
                "cumulative": cumulative,
                "sac_paired_gain_percent": 100.0 * (sac_power / zero_power - 1.0),
            }
        )

    gains = np.asarray([window["sac_paired_gain_percent"] for window in windows])
    saturation = np.asarray(
        [
            window["controllers"]["sac"]["action_saturation_fraction"]
            for window in windows
        ]
    )
    checks = {
        "gpu_backend": jax.default_backend() == "gpu",
        "archive_complete": archive.manifest.get("status") == "complete",
        "canonical_layout_contract": canonical_layout.payload_sha256 == LAYOUT_SHA256,
        "all_metrics_finite": all(
            controller["finite"]
            for window in windows
            for controller in window["controllers"].values()
        ),
        "yaw_bounds_respected": all(
            controller["max_abs_yaw_deg"] <= config.max_yaw_angle_deg + 1.0e-6
            for window in windows
            for controller in window["controllers"].values()
        ),
        "zero_controller_is_zero": all(
            window["controllers"]["zero"]["max_abs_yaw_deg"] == 0.0
            for window in windows
        ),
    }
    result = {
        "schema_version": 1,
        "purpose": "mole_n9_directional_paired_window_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "configuration": {
            "relative_wind_angle_deg": args.relative_wind_angle_deg,
            "action_scale": args.action_scale,
            "offsets": list(offsets),
            "burnin_interactions": args.burnin_interactions,
            "eval_interactions": args.eval_interactions,
            "block_interactions": args.block_interactions,
            "cumulative_interactions": list(cumulative_points),
            "seed": args.seed,
            "steering_branch": args.steering_branch,
            "steering_branch_mode": args.steering_branch_mode,
            "forced_reflection_equivariant_actor": (
                args.force_reflection_equivariant_actor
            ),
        },
        "direction_protocol": {
            "declared_directions_deg": list(declared_directions),
            "common_center_xz_m": list(center),
            "canonical_layout_sha256": canonical_layout.payload_sha256,
            "target_layout_sha256": target_layout.payload_sha256,
            "target_turbine_positions_m": target_layout.turbine_positions_m,
            "graph": graph_summary,
        },
        "archive": {
            "path": str(args.plane_archive),
            "manifest_sha256": archive.manifest_sha256,
            "source_state_sha256": archive.source_state_sha256,
            "completed_planes": archive.completed_planes,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "metadata": actor_metadata,
        },
        "protocol": protocol_metadata(case, config, "per_turbine_target"),
        "windows": windows,
        "summary": {
            "window_count": len(windows),
            "mean_sac_paired_gain_percent": float(np.mean(gains)),
            "median_sac_paired_gain_percent": float(np.median(gains)),
            "min_sac_paired_gain_percent": float(np.min(gains)),
            "max_sac_paired_gain_percent": float(np.max(gains)),
            "positive_window_count": int(np.sum(gains > 0.0)),
            "mean_action_saturation_fraction": float(np.mean(saturation)),
            "max_action_saturation_fraction": float(np.max(saturation)),
            "total_wallclock_seconds": time.perf_counter() - total_started,
        },
        "evidence_boundary": (
            "One declared direction and repeated windows from one precursor "
            "trajectory; paired zero-yaw comparison, not an independent-seed claim."
        ),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    result["payload_sha256_without_hash_field"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
