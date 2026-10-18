#!/usr/bin/env python3
"""Pure helpers for the preregistered N=9 WP7 holdout evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


TRAINING_DIRECTIONS_DEG = (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)
MIDPOINT_DIRECTIONS_DEG = (-8.75, -6.25, -3.75, -1.25, 1.25, 3.75, 6.25, 8.75)
WP7_OFFSETS = (0, 40_000)


def angle_key(value: float) -> str:
    return f"{value:+05.1f}"


def interpolation_bracket(direction_deg: float) -> tuple[float, float, float]:
    """Return the frozen adjacent training directions and upper weight."""

    direction = float(direction_deg)
    if direction in TRAINING_DIRECTIONS_DEG:
        return direction, direction, 0.0
    if direction not in MIDPOINT_DIRECTIONS_DEG:
        raise ValueError(f"direction {direction} is outside the frozen WP7 matrix")
    for lower, upper in zip(TRAINING_DIRECTIONS_DEG[:-1], TRAINING_DIRECTIONS_DEG[1:]):
        if lower < direction < upper:
            weight = (direction - lower) / (upper - lower)
            return lower, upper, weight
    raise AssertionError("unreachable interpolation bracket")


def interpolate_direction_mapping(
    mapping: Mapping[str, Sequence[float]], direction_deg: float
) -> np.ndarray:
    lower, upper, weight = interpolation_bracket(direction_deg)
    lower_value = np.asarray(mapping[angle_key(lower)], dtype=np.float32)
    if lower == upper:
        return lower_value.copy()
    upper_value = np.asarray(mapping[angle_key(upper)], dtype=np.float32)
    if lower_value.shape != upper_value.shape:
        raise ValueError("adjacent frozen values have different shapes")
    return ((1.0 - weight) * lower_value + weight * upper_value).astype(np.float32)


def interpolate_static_group_targets(
    lookup_table: Sequence[Mapping[str, Any]], direction_deg: float
) -> np.ndarray:
    by_direction = {
        angle_key(float(row["relative_wind_angle_deg"])): row["grouped_target_yaw_deg"]
        for row in lookup_table
    }
    return interpolate_direction_mapping(by_direction, direction_deg)


def averaged_development_replay(
    action_traces: Sequence[Mapping[str, Any]], direction_deg: float
) -> np.ndarray:
    """Freeze a source-independent no-feedback replay for the WP7 target seed.

    At a training direction, all four development traces (two archives by two
    windows) are averaged sample by sample. At a sealed midpoint, the two
    adjacent training-direction averages are linearly interpolated. This rule
    uses development material only and is fixed before the holdout is opened.
    """

    def direction_average(direction: float) -> np.ndarray:
        selected = [
            np.asarray(trace["actions"], dtype=np.float32)
            for trace in action_traces
            if float(trace["declared_direction_deg"]) == direction
        ]
        if len(selected) != 4:
            raise ValueError(
                f"expected four frozen development traces at {direction}, got {len(selected)}"
            )
        shape = selected[0].shape
        if shape != (200, 9) or any(item.shape != shape for item in selected):
            raise ValueError("development action traces must all have shape [200, 9]")
        return np.mean(np.stack(selected), axis=0, dtype=np.float64).astype(np.float32)

    lower, upper, weight = interpolation_bracket(direction_deg)
    lower_trace = direction_average(lower)
    if lower == upper:
        return lower_trace
    upper_trace = direction_average(upper)
    return ((1.0 - weight) * lower_trace + weight * upper_trace).astype(np.float32)

