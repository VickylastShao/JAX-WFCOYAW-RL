#!/usr/bin/env python3
"""Capture or evaluate the frozen N=9 WP6 controller mechanisms."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from eval_mole_n9_directional_windows import (
    LAYOUT_SHA256,
    file_sha256,
    make_policy_adapter,
    validate_checkpoint_training_layout,
)
from eval_mole_n9_locked_windows import reset_controller_input, summarize_metrics
from jax_mole_compact_sac_benchmark import make_scaled_mole_training_case
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    initialize_continuous_training,
    initialize_evaluation_from_training,
    load_actor_checkpoint,
    make_evaluation_step,
)
from jax_mole_direction_geometry import common_direction_center, rotate_layout_to_flow_frame
from jax_mole_layouts import canonical_scale_layouts
from mole_n9_wp6_contract import (
    STATIC_LOOKUP_GROUPS,
    WP4_TRAINING_DIRECTIONS_DEG,
    WP5_DEVELOPMENT_OFFSETS,
    build_action_trace,
    payload_sha256_without_hash,
    validate_action_replay_binding,
)
from mole_n9_wp7_contract import (
    MIDPOINT_DIRECTIONS_DEG,
    WP7_OFFSETS,
    averaged_development_replay,
    interpolate_direction_mapping,
    interpolate_static_group_targets,
)
from mole_plane_archive import PlaneArchive


SELECTED_ACTOR_SHA256 = "a0474f942c93a296121cfff7c317e861dfbb05369f5cbd42c39aaaca495703b9"
NEUTRAL_ACTOR_SHA256 = "04a0425a0aeb24c457f602aaf661adeff4531e5f4c1b2f3415ff363d1dd0f81c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "evaluate"))
    parser.add_argument(
        "--evaluation-stage", choices=("wp6", "wp7-a1", "wp7-a2"), default="wp6"
    )
    parser.add_argument("--plane-archive", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--neutral-checkpoint", type=Path)
    parser.add_argument("--freeze-bundle", type=Path)
    parser.add_argument("--static-lookup", type=Path)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--relative-wind-angle-deg", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--burnin-interactions", type=int, default=150)
    parser.add_argument("--eval-interactions", type=int, default=200)
    parser.add_argument("--block-interactions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026082101)
    return parser.parse_args()


def angle_key(value: float) -> str:
    return f"{value:+05.1f}"


def load_hashed_json(path: Path, purpose: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("purpose") != purpose:
        raise ValueError(f"unexpected purpose in {path}")
    stored = payload.get("payload_sha256_without_hash_field")
    if stored != payload_sha256_without_hash(payload):
        raise ValueError(f"payload hash mismatch in {path}")
    return payload


def make_scan(step: Callable[..., Any], interactions: int, *, capture_observation: bool = False):
    @jax.jit
    def run(initial: Any, inlet_planes: jax.Array, plane_start: jax.Array):
        def body(state: Any, _: None):
            next_state, metrics = step(state, inlet_planes, plane_start)
            if capture_observation:
                metrics = {**metrics, "next_observation": next_state.observation}
            return next_state, metrics

        return jax.lax.scan(body, initial, xs=None, length=interactions)

    return run


def make_runtime_action_scan(step: Callable[..., Any], interactions: int):
    @jax.jit
    def run(
        initial: Any,
        inlet_planes: jax.Array,
        plane_start: jax.Array,
        normalized_actions: jax.Array,
    ):
        def body(state: Any, action: jax.Array):
            return step(state, inlet_planes, plane_start, action)

        return jax.lax.scan(body, initial, normalized_actions)

    return run


def run_blocks(
    state: Any,
    archive: PlaneArchive,
    cursor: int,
    blocks: int,
    planes_per_block: int,
    run_block: Callable[..., Any],
    *,
    external_actions: np.ndarray | None = None,
) -> tuple[Any, int, dict[str, np.ndarray]]:
    traces: dict[str, list[np.ndarray]] = {}
    block_interactions = None
    if external_actions is not None:
        if external_actions.ndim != 3 or external_actions.shape[1:] != (1, 9):
            raise ValueError("external actions must have shape [T, 1, 9]")
        if external_actions.shape[0] % blocks:
            raise ValueError("external action length is not divisible by blocks")
        block_interactions = external_actions.shape[0] // blocks
    for block_index in range(blocks):
        host_planes = archive.read(cursor, planes_per_block)
        plane_start = jnp.asarray(int(np.asarray(state.farm_flow.step_index)), dtype=jnp.int32)
        arguments: list[Any] = [state, jnp.asarray(host_planes[None]), plane_start]
        if external_actions is not None:
            start = block_index * block_interactions
            arguments.append(jnp.asarray(external_actions[start : start + block_interactions]))
        state, metrics = run_block(*arguments)
        jax.block_until_ready(metrics["farm_power_mw"])
        for name, value in metrics.items():
            traces.setdefault(name, []).append(np.asarray(value))
        cursor += planes_per_block
    return state, cursor, {
        name: np.concatenate(parts, axis=0) for name, parts in traces.items()
    }


def controller_summary(metrics: dict[str, np.ndarray], control_dt_seconds: float) -> dict[str, Any]:
    summary = summarize_metrics(metrics)
    yaw = np.asarray(metrics["yaw_deg"], dtype=np.float64)
    initial = np.zeros_like(yaw[:1])
    rates = np.abs(np.diff(np.concatenate((initial, yaw), axis=0), axis=0)) / control_dt_seconds
    summary["peak_yaw_rate_deg_per_s"] = float(np.max(rates))
    summary["mean_yaw_travel_deg_per_turbine"] = summary.pop(
        "mean_total_yaw_travel_deg_per_turbine"
    )
    yaw_2d = yaw[:, 0, :]
    centered = yaw_2d - np.mean(yaw_2d, axis=0, keepdims=True)
    scale = np.sqrt(np.sum(centered * centered, axis=0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((9, 9), dtype=np.float64),
        where=denominator > 1.0e-12,
    )
    summary["yaw_action_correlation"] = correlation.tolist()
    controlled = yaw_2d[:, :6]
    active = np.abs(controlled) >= 0.5
    positive = np.all(np.where(active, controlled > 0.0, True), axis=1)
    negative = np.all(np.where(active, controlled < 0.0, True), axis=1)
    summary["same_side_yaw_mode_fraction"] = float(
        np.mean(np.any(active, axis=1) & (positive | negative))
    )
    mode = np.where(np.mean(controlled, axis=1) > 0.5, 1, np.where(np.mean(controlled, axis=1) < -0.5, -1, 0))
    summary["yaw_mode_switch_count"] = int(np.sum(mode[1:] != mode[:-1]))
    action_signal = np.mean(np.abs(controlled), axis=1)
    power_signal = np.mean(
        np.asarray(metrics["turbine_power_mw"], dtype=np.float64)[:, 0, 3:], axis=1
    )
    lag_correlations = []
    for lag in range(31):
        action_slice = action_signal[: action_signal.size - lag or None]
        power_slice = power_signal[lag:]
        if (
            action_slice.size < 3
            or np.std(action_slice) <= 1.0e-12
            or np.std(power_slice) <= 1.0e-12
        ):
            lag_correlations.append(0.0)
        else:
            lag_correlations.append(float(np.corrcoef(action_slice, power_slice)[0, 1]))
    best_lag = int(np.argmax(np.abs(lag_correlations)))
    summary["action_to_downstream_power_lag_interactions"] = best_lag
    summary["action_to_downstream_power_lag_seconds"] = best_lag * control_dt_seconds
    summary["action_to_downstream_power_peak_correlation"] = lag_correlations[best_lag]
    summary["action_to_downstream_power_lag_curve"] = lag_correlations
    return summary


def validate_archive(archive: PlaneArchive, case: Any) -> None:
    provenance = archive.manifest.get("provenance", archive.manifest.get("contract", {}))
    if archive.plane_shape != (3, case.ny, case.nz):
        raise ValueError("archive plane shape differs from N=9 case")
    if provenance.get("target_layout_sha256") != LAYOUT_SHA256:
        raise ValueError("archive layout identity differs from frozen N=9 layout")
    if provenance.get("target_turbines") != 9:
        raise ValueError("archive turbine count differs from N=9")


def make_common_state(
    case: Any,
    config: ContinuousControlConfig,
    archive: PlaneArchive,
    offset: int,
    seed: int,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    burn_scan: Callable[..., Any],
    burn_blocks: int,
    planes_per_block: int,
) -> tuple[Any, int, float]:
    first_plane = archive.read(offset, 1)[0]
    training, _, _, _ = initialize_continuous_training(
        case,
        config,
        None,
        seed,
        256,
        3.0e-6,
        8,
        initial_inlet=jnp.asarray(first_plane[None]),
        action_parameterization="per_turbine_target",
    )
    initial = initialize_evaluation_from_training(training)
    started = time.perf_counter()
    common, cursor, _ = run_blocks(
        initial, archive, offset, burn_blocks, planes_per_block, burn_scan
    )
    return reset_controller_input(common), cursor, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    is_wp7 = args.evaluation_stage.startswith("wp7")
    if is_wp7 and args.phase != "evaluate":
        raise ValueError("WP7 is evaluation-only")
    allowed_directions = (
        MIDPOINT_DIRECTIONS_DEG
        if args.evaluation_stage == "wp7-a2"
        else WP4_TRAINING_DIRECTIONS_DEG
    )
    if args.relative_wind_angle_deg not in allowed_directions:
        raise ValueError(
            f"{args.evaluation_stage} does not admit direction {args.relative_wind_angle_deg}"
        )
    evaluation_offsets = WP7_OFFSETS if is_wp7 else WP5_DEVELOPMENT_OFFSETS
    if min(args.burnin_interactions, args.eval_interactions, args.block_interactions) < 1:
        raise ValueError("interaction counts must be positive")
    if args.burnin_interactions % args.block_interactions or args.eval_interactions % args.block_interactions:
        raise ValueError("block size must divide burn-in and evaluation lengths")
    if file_sha256(args.selected_checkpoint) != SELECTED_ACTOR_SHA256:
        raise ValueError("selected Actor SHA-256 differs from frozen WP5 identity")
    if args.phase == "evaluate":
        if not args.neutral_checkpoint or not args.freeze_bundle or not args.static_lookup:
            raise ValueError("evaluate phase requires neutral checkpoint, freeze bundle and static lookup")
        if file_sha256(args.neutral_checkpoint) != NEUTRAL_ACTOR_SHA256:
            raise ValueError("neutral Actor SHA-256 differs from preregistration")

    canonical_layout = canonical_scale_layouts()[9]
    if canonical_layout.payload_sha256 != LAYOUT_SHA256:
        raise ValueError("canonical N=9 layout identity changed")
    center = common_direction_center(canonical_layout, WP4_TRAINING_DIRECTIONS_DEG)
    target_layout = rotate_layout_to_flow_frame(
        canonical_layout, args.relative_wind_angle_deg, center
    )
    canonical_case = make_scaled_mole_training_case(canonical_layout, batch=1)
    case = make_scaled_mole_training_case(target_layout, batch=1)
    archive = PlaneArchive(args.plane_archive, require_complete=True)
    validate_archive(archive, case)
    config = ContinuousControlConfig(inlet_mode="time_planes")
    planes_per_interaction = config.les_steps_per_interaction
    planes_per_block = args.block_interactions * planes_per_interaction
    burn_blocks = args.burnin_interactions // args.block_interactions
    eval_blocks = args.eval_interactions // args.block_interactions
    required_planes = (args.burnin_interactions + args.eval_interactions) * planes_per_interaction
    if max(evaluation_offsets) + required_planes > archive.completed_planes:
        raise ValueError(f"a {args.evaluation_stage} window exceeds the archive")

    selected_actor, selected_metadata = load_actor_checkpoint(args.selected_checkpoint)
    validate_checkpoint_training_layout(selected_metadata, canonical_layout, center)
    selected_policy, graph_summary = make_policy_adapter(
        selected_metadata,
        args.relative_wind_angle_deg,
        canonical_case,
        case,
        config.frame_stack,
    )
    first_plane = archive.read(evaluation_offsets[0], 1)[0]
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
    zero_step = make_evaluation_step(case, config, precursor_functions, farm_functions, "zero")
    selected_step = make_evaluation_step(
        case,
        config,
        precursor_functions,
        farm_functions,
        "sac",
        actor=selected_actor,
        action_parameterization="per_turbine_target",
        actor_deterministic_fn=selected_policy,
    )
    burn_scan = make_scan(zero_step, args.block_interactions)
    zero_scan = make_scan(
        zero_step,
        args.block_interactions,
        capture_observation=args.phase == "capture",
    )
    selected_scan = make_scan(selected_step, args.block_interactions)

    scans: dict[str, Any] = {"zero_yaw": zero_scan, "selected_realtime_policy": selected_scan}
    external_scan = None
    probe_width = config.frame_stack * 9 * 77
    bundle = None
    static_lookup = None
    neutral_actor = None
    if args.phase == "evaluate":
        bundle = load_hashed_json(args.freeze_bundle, "mole_n9_wp6_actor_dependent_freeze")
        if bundle.get("formal_wp6_execution_allowed") is not True:
            raise ValueError("freeze bundle does not allow formal WP6 execution")
        if bundle["selected_actor"]["sha256"] != SELECTED_ACTOR_SHA256:
            raise ValueError("freeze bundle selected Actor differs from WP5")
        static_lookup = load_hashed_json(args.static_lookup, "mole_n9_static_les_lookup_final")
        neutral_actor, neutral_metadata = load_actor_checkpoint(args.neutral_checkpoint)
        validate_checkpoint_training_layout(neutral_metadata, canonical_layout, center)
        neutral_policy, _ = make_policy_adapter(
            neutral_metadata,
            args.relative_wind_angle_deg,
            canonical_case,
            case,
            config.frame_stack,
        )
        neutral_step = make_evaluation_step(
            case,
            config,
            precursor_functions,
            farm_functions,
            "sac",
            actor=neutral_actor,
            action_parameterization="per_turbine_target",
            actor_deterministic_fn=neutral_policy,
        )
        probe_mean = jnp.asarray(
            interpolate_direction_mapping(
                bundle["direction_probe_means"], args.relative_wind_angle_deg
            )
        )

        def ablated_policy(actor: Any, observations: jax.Array, policy_mode: str) -> jax.Array:
            replaced = observations.at[..., :probe_width].set(
                jnp.broadcast_to(probe_mean, observations[..., :probe_width].shape)
            )
            return selected_policy(actor, replaced, policy_mode)

        ablated_step = make_evaluation_step(
            case,
            config,
            precursor_functions,
            farm_functions,
            "sac",
            actor=selected_actor,
            action_parameterization="per_turbine_target",
            actor_deterministic_fn=ablated_policy,
        )
        runtime_step = make_evaluation_step(
            case,
            config,
            precursor_functions,
            farm_functions,
            "external_action",
            action_parameterization="per_turbine_target",
            runtime_external_action=True,
        )
        scans.update(
            {
                "neutral_actor_step32768": make_scan(neutral_step, args.block_interactions),
                "selected_probe_mean_ablation": make_scan(ablated_step, args.block_interactions),
            }
        )
        external_scan = make_runtime_action_scan(runtime_step, args.block_interactions)

    windows: list[dict[str, Any]] = []
    action_traces: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for window_index, offset in enumerate(evaluation_offsets):
        common, cursor, burn_seconds = make_common_state(
            case,
            config,
            archive,
            offset,
            args.seed + window_index,
            precursor_functions,
            farm_functions,
            burn_scan,
            burn_blocks,
            planes_per_block,
        )
        controller_metrics: dict[str, dict[str, np.ndarray]] = {}
        timings: dict[str, float] = {"burnin": burn_seconds}
        for controller_id, scan in scans.items():
            started = time.perf_counter()
            _, final_cursor, metrics = run_blocks(
                common, archive, cursor, eval_blocks, planes_per_block, scan
            )
            if final_cursor != cursor + args.eval_interactions * planes_per_interaction:
                raise ValueError("controller consumed an unexpected inlet interval")
            timings[controller_id] = time.perf_counter() - started
            controller_metrics[controller_id] = metrics

        if args.phase == "capture":
            actions = controller_metrics["selected_realtime_policy"]["normalized_action"][:, 0, :]
            trace = build_action_trace(
                actions,
                source_archive_manifest_sha256=archive.manifest_sha256,
                source_state_sha256=archive.source_state_sha256,
                source_actor_sha256=SELECTED_ACTOR_SHA256,
                declared_direction_deg=args.relative_wind_angle_deg,
                source_window_start_plane=offset,
            )
            trace["source_host_label"] = args.host_label
            trace["payload_sha256_without_hash_field"] = payload_sha256_without_hash(trace)
            action_traces.append(trace)
            observations = controller_metrics["zero_yaw"].pop("next_observation")[:, 0, :probe_width]
            probe_sum = np.sum(observations.astype(np.float64), axis=0)
            zero_power = float(np.mean(controller_metrics["zero_yaw"]["farm_power_mw"]))
            selected_power = float(np.mean(controller_metrics["selected_realtime_policy"]["farm_power_mw"]))
            windows.append(
                {
                    "start_plane": offset,
                    "probe_sum": probe_sum.tolist(),
                    "probe_count": int(observations.shape[0]),
                    "controllers": {
                        name: controller_summary(metrics, config.control_dt_seconds)
                        for name, metrics in controller_metrics.items()
                    },
                    "selected_gain_percent": 100.0 * (selected_power / zero_power - 1.0),
                    "timing_seconds": timings,
                }
            )
            continue

        direction_actions = interpolate_direction_mapping(
            bundle["direction_mean_actions"], args.relative_wind_angle_deg
        )
        grouped_targets = interpolate_static_group_targets(
            static_lookup["lookup_table"], args.relative_wind_angle_deg
        )
        static_yaw = np.zeros(9, dtype=np.float32)
        for target, group in zip(grouped_targets, STATIC_LOOKUP_GROUPS):
            static_yaw[list(group)] = np.float32(target)
        if is_wp7:
            replay_actions = averaged_development_replay(
                bundle["action_traces"], args.relative_wind_angle_deg
            )
            replay_binding = {
                "status": "pass",
                "source": "four_frozen_development_traces_samplewise_mean",
                "midpoint_rule": "piecewise_linear_adjacent_training_direction_replay_means",
                "target_archive_manifest_sha256": archive.manifest_sha256,
                "target_state_sha256": archive.source_state_sha256,
            }
        else:
            other_host = "rtx99" if args.host_label == "rtx98" else "rtx98"
            replay_trace = next(
                trace
                for trace in bundle["action_traces"]
                if trace["source_host_label"] == other_host
                and float(trace["declared_direction_deg"]) == args.relative_wind_angle_deg
                and int(trace["source_window_start_plane"]) == offset
            )
            replay_binding = validate_action_replay_binding(
                replay_trace,
                target_archive_manifest_sha256=archive.manifest_sha256,
                target_state_sha256=archive.source_state_sha256,
                expected_trace_sha256=replay_trace["payload_sha256_without_hash_field"],
            )
            replay_actions = np.asarray(replay_trace["actions"], dtype=np.float32)
        external_actions = {
            "static_les_lookup": np.broadcast_to(static_yaw / 40.0, (args.eval_interactions, 1, 9)).copy(),
            "selected_policy_direction_mean": np.broadcast_to(direction_actions, (args.eval_interactions, 1, 9)).copy(),
            "cross_precursor_action_replay": replay_actions[:, None, :],
        }
        for controller_id, actions in external_actions.items():
            started = time.perf_counter()
            _, final_cursor, metrics = run_blocks(
                common,
                archive,
                cursor,
                eval_blocks,
                planes_per_block,
                external_scan,
                external_actions=actions,
            )
            if final_cursor != cursor + args.eval_interactions * planes_per_interaction:
                raise ValueError("external controller consumed an unexpected inlet interval")
            timings[controller_id] = time.perf_counter() - started
            controller_metrics[controller_id] = metrics
        zero_power = float(np.mean(controller_metrics["zero_yaw"]["farm_power_mw"]))
        controllers = {}
        for name, metrics in controller_metrics.items():
            power = float(np.mean(metrics["farm_power_mw"]))
            summary = controller_summary(metrics, config.control_dt_seconds)
            summary["gain_percent"] = 100.0 * (power / zero_power - 1.0)
            controllers[name] = summary
        windows.append(
            {
                "start_plane": offset,
                "controllers": controllers,
                "cross_replay_binding": replay_binding,
                "timing_seconds": timings,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "mole_n9_wp6_source_capture"
            if args.phase == "capture"
            else "mole_n9_wp7_formal_direction_result"
            if is_wp7
            else "mole_n9_wp6_formal_host_direction_result"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "phase": args.phase,
        "identity": {
            "host_label": args.host_label,
            "image_digest": args.image_digest,
            "archive_manifest_sha256": archive.manifest_sha256,
            "source_state_sha256": archive.source_state_sha256,
            "selected_actor_sha256": SELECTED_ACTOR_SHA256,
            "neutral_actor_sha256": NEUTRAL_ACTOR_SHA256 if args.phase == "evaluate" else None,
            "canonical_layout_sha256": LAYOUT_SHA256,
            "target_layout_sha256": target_layout.payload_sha256,
        },
        "configuration": {
            "relative_wind_angle_deg": args.relative_wind_angle_deg,
            "evaluation_stage": args.evaluation_stage,
            "offsets": list(evaluation_offsets),
            "burnin_interactions": args.burnin_interactions,
            "eval_interactions": args.eval_interactions,
            "block_interactions": args.block_interactions,
            "direction_weighting": "equal_direction",
            "action_parameterization": "per_turbine_target",
            "max_yaw_angle_deg": config.max_yaw_angle_deg,
            "max_yaw_rate_deg_per_s": config.max_yaw_speed_deg_per_s,
            "graph": graph_summary,
        },
        "windows": windows,
        "action_traces": action_traces if args.phase == "capture" else None,
        "wallclock_seconds": time.perf_counter() - total_started,
        "selection_performed": False,
        "sealed_material_accessed": is_wp7,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "evidence_boundary": (
            "Development-set WP6 source capture only; no controller comparison claim."
            if args.phase == "capture"
            else "Independent seed20260904 holdout evaluation; no retraining or reselection."
            if is_wp7
            else "Development-set descriptive WP6 comparison; no inferential interval or sealed material."
        ),
    }
    result["payload_sha256_without_hash_field"] = payload_sha256_without_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output), "wallclock_seconds": result["wallclock_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
