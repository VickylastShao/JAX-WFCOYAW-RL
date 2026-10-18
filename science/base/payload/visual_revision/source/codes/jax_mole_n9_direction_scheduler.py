#!/usr/bin/env python3
"""Episode-exact direction scheduling for the N=9 distributed collector."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from jax_mole_n9_million_protocol import (
    ENVIRONMENTS_PER_HOST,
    GLOBAL_ENVIRONMENTS,
    RESETS_PER_SUPERCYCLE,
    SEALED_MIDPOINT_DIRECTIONS_DEG,
    TRAIN_DIRECTIONS_DEG,
    build_direction_schedule,
)


class CollectionSegment(NamedTuple):
    steps: int
    reset_index: int
    episode_position_start: int
    reset_before: bool


def archive_member_bounds(
    member_offsets: tuple[int, ...], completed_planes: int
) -> tuple[tuple[int, int], ...]:
    """Return strict, non-overlapping member intervals from archive offsets."""

    if completed_planes <= 0 or not member_offsets:
        raise ValueError("archive must contain at least one member")
    offsets = tuple(int(value) for value in member_offsets)
    if offsets[0] < 0 or any(
        left >= right for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("archive member offsets must be strictly increasing")
    if offsets[-1] >= completed_planes:
        raise ValueError("archive member offset exceeds completed planes")
    stops = (*offsets[1:], int(completed_planes))
    return tuple(zip(offsets, stops))


def episode_plane_window_start(
    member_start: int,
    member_stop: int,
    required_planes: int,
    *,
    schedule_seed: int,
    global_environment: int,
    episode_index: int,
) -> int:
    """Select a reproducible contiguous inlet window without Python RNG state."""

    if member_start < 0 or member_stop <= member_start:
        raise ValueError("invalid archive member interval")
    if required_planes <= 0 or required_planes > member_stop - member_start:
        raise ValueError("archive member is too short for one complete episode")
    if global_environment < 0 or episode_index < 0:
        raise ValueError("environment and episode indices cannot be negative")

    mask = (1 << 64) - 1
    value = (
        int(schedule_seed)
        ^ ((int(global_environment) + 1) * 0x9E3779B97F4A7C15)
        ^ ((int(episode_index) + 1) * 0xBF58476D1CE4E5B9)
    ) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    available_starts = member_stop - member_start - required_planes + 1
    return member_start + value % available_starts


def local_directions_for_reset(
    reset_index: int,
    process_id: int,
    *,
    local_environments: int = ENVIRONMENTS_PER_HOST,
    schedule_seed: int = 2026081802,
) -> np.ndarray:
    if reset_index < 0:
        raise ValueError("reset_index cannot be negative")
    if (
        local_environments <= 0
        or GLOBAL_ENVIRONMENTS % local_environments
    ):
        raise ValueError(
            "local_environments must be a positive divisor of the global 32"
        )
    process_count = GLOBAL_ENVIRONMENTS // local_environments
    if not 0 <= process_id < process_count:
        raise ValueError(
            f"process_id must be in [0, {process_count}) for "
            f"local_environments={local_environments}"
        )
    row = build_direction_schedule(
        supercycles=1, schedule_seed=schedule_seed
    )[reset_index % RESETS_PER_SUPERCYCLE]
    start = process_id * local_environments
    return row[start : start + local_environments].copy()


def split_collection_segments(
    episode_position: int,
    collection_steps: int,
    *,
    episode_interactions: int = 500,
    reset_index: int = 0,
    reset_before_start: bool = False,
) -> tuple[CollectionSegment, ...]:
    """Split one outer collection without allowing any segment to cross reset."""

    if not 0 <= episode_position < episode_interactions:
        raise ValueError("episode_position must lie inside the episode")
    if collection_steps <= 0 or episode_interactions <= 0 or reset_index < 0:
        raise ValueError("collection/reset arguments must be positive")
    remaining = collection_steps
    position = episode_position
    current_reset = reset_index
    reset_before = bool(reset_before_start)
    segments: list[CollectionSegment] = []
    while remaining:
        steps = min(remaining, episode_interactions - position)
        segments.append(
            CollectionSegment(
                steps=steps,
                reset_index=current_reset,
                episode_position_start=position,
                reset_before=reset_before,
            )
        )
        remaining -= steps
        position += steps
        reset_before = False
        if position == episode_interactions and remaining:
            current_reset += 1
            position = 0
            reset_before = True
    return tuple(segments)


def count_global_direction_transitions(
    interactions_per_environment: int,
    *,
    episode_interactions: int = 500,
    schedule_seed: int = 2026081802,
) -> dict[float, int]:
    """Count exact Replay transitions implied by the frozen schedule."""

    if interactions_per_environment <= 0:
        raise ValueError("interactions_per_environment must be positive")
    counts = {float(direction): 0 for direction in TRAIN_DIRECTIONS_DEG}
    remaining = interactions_per_environment
    reset_index = 0
    while remaining:
        episode_steps = min(remaining, episode_interactions)
        row = build_direction_schedule(
            supercycles=1, schedule_seed=schedule_seed
        )[reset_index % RESETS_PER_SUPERCYCLE]
        for direction in row:
            counts[float(direction)] += episode_steps
        remaining -= episode_steps
        reset_index += 1
    return counts


def assert_training_direction(direction_deg: float) -> None:
    value = np.float32(direction_deg)
    if value in SEALED_MIDPOINT_DIRECTIONS_DEG:
        raise ValueError("sealed midpoint direction cannot enter training")
    if value not in TRAIN_DIRECTIONS_DEG:
        raise ValueError("direction is outside the frozen training set")


def validate_formal_direction_protocol(args: Any) -> None:
    """Fail closed unless all frozen N=9 collection controls are exact."""

    if not getattr(args, "n9_million_direction_schedule", False):
        return
    performance_benchmark = bool(
        getattr(args, "h20_eight_process_performance_benchmark", False)
    )
    h20_formal_training = bool(
        getattr(args, "h20_eight_process_formal_training", False)
    )
    if performance_benchmark and h20_formal_training:
        raise ValueError("H20 performance and formal modes are mutually exclusive")
    h20_topology = performance_benchmark or h20_formal_training
    expected = {
        "microbatch": 1,
        "collection_steps": 16,
        "target_turbines": 9,
        "action_parameterization": "per_turbine_target",
        "episode_interactions": 500,
        "reset_interactions": 150,
        "burnin_interactions": 0,
        "inlet_mode": "time_planes",
        "replay_mode": "global",
        "algorithm": "graph_sac_directional",
        "direction_condition": True,
        "steering_branch_condition": False,
        "graph_geometry": "canonical",
        "reflection_equivariant_actor": True,
        "sample_size": 32 if h20_topology else 128,
        "learning_rate": 3.0e-4,
        "initial_alpha": 0.1,
        "actor_learning_rate_scale": 1.0,
        "graph_actor_warmup_updates": 0 if performance_benchmark else 4096,
        "graph_actor_minimum_replay_size": 0 if performance_benchmark else 32768,
        "graph_actor_update_period": 32,
        "entropy_scale": 1.0 / 9.0,
        "conservative_actor_init": True,
        "random_action_transitions": 0 if performance_benchmark else 32768,
        "replay_capacity": 1_000_448,
        "minimum_replay_size": 0 if performance_benchmark else 32768,
        "n9_direction_schedule_seed": 2026081802,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(args, name, None)}
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if h20_topology:
        expected_topology = (8, 4)
    else:
        expected_topology = (2, 16)
    actual_topology = (
        getattr(args, "num_processes", None),
        getattr(args, "local_environments", None),
    )
    if actual_topology != expected_topology:
        mismatches["process_topology"] = {
            "expected": expected_topology,
            "actual": actual_topology,
        }
    scale = getattr(args, "neutral_actor_initial_scale", None)
    if scale is None or not np.isclose(scale, 0.078329048, rtol=0.0, atol=1e-9):
        mismatches["neutral_actor_initial_scale"] = {
            "expected": 0.078329048,
            "actual": scale,
        }
    if getattr(args, "initial_actor_checkpoint", None) is not None:
        mismatches["initial_actor_checkpoint"] = {
            "expected": None,
            "actual": str(args.initial_actor_checkpoint),
        }
    if getattr(args, "plane_archive", None) is None:
        mismatches["plane_archive"] = {
            "expected": "a completed archive path",
            "actual": None,
        }
    if h20_formal_training:
        if not getattr(args, "episode_random_plane_windows", False):
            mismatches["episode_random_plane_windows"] = {
                "expected": True,
                "actual": False,
            }
        if getattr(args, "plane_window_seed", None) != 2026082201:
            mismatches["plane_window_seed"] = {
                "expected": 2026082201,
                "actual": getattr(args, "plane_window_seed", None),
            }
    if mismatches:
        raise ValueError(f"formal N=9 direction protocol mismatch: {mismatches}")
