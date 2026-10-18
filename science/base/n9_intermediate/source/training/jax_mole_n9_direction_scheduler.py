#!/usr/bin/env python3
"""Episode-exact direction scheduling for the N=9 distributed collector."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from jax_mole_n9_million_protocol import (
    ENVIRONMENTS_PER_HOST,
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


def local_directions_for_reset(
    reset_index: int,
    process_id: int,
    *,
    schedule_seed: int = 2026081802,
) -> np.ndarray:
    if reset_index < 0:
        raise ValueError("reset_index cannot be negative")
    if process_id not in (0, 1):
        raise ValueError("process_id must be 0 or 1")
    row = build_direction_schedule(
        supercycles=1, schedule_seed=schedule_seed
    )[reset_index % RESETS_PER_SUPERCYCLE]
    start = process_id * ENVIRONMENTS_PER_HOST
    return row[start : start + ENVIRONMENTS_PER_HOST].copy()


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
    expected = {
        "num_processes": 2,
        "local_environments": 16,
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
        "sample_size": 128,
        "learning_rate": 3.0e-4,
        "initial_alpha": 0.1,
        "actor_learning_rate_scale": 1.0,
        "graph_actor_warmup_updates": 4096,
        "graph_actor_minimum_replay_size": 32768,
        "graph_actor_update_period": 32,
        "entropy_scale": 1.0 / 9.0,
        "conservative_actor_init": True,
        "random_action_transitions": 32768,
        "replay_capacity": 1_000_448,
        "n9_direction_schedule_seed": 2026081802,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(args, name, None)}
        for name, value in expected.items()
        if getattr(args, name, None) != value
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
    if mismatches:
        raise ValueError(f"formal N=9 direction protocol mismatch: {mismatches}")
