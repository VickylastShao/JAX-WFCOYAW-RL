#!/usr/bin/env python3
"""Geometry and reflection contracts for direction-conditioned Mole cases."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

import numpy as np

from jax_mole_layouts import MoleFarmLayout


def flow_frame_relative_positions(
    layout: MoleFarmLayout, relative_wind_angle_deg: float
) -> np.ndarray:
    """Return centered turbine x/z coordinates in the aligned-flow frame."""

    if not math.isfinite(relative_wind_angle_deg):
        raise ValueError("relative wind angle must be finite")
    positions = np.asarray(layout.turbine_positions_m, dtype=np.float64)
    horizontal = positions[:, (0, 2)]
    centered = horizontal - horizontal.mean(axis=0, keepdims=True)
    radians = math.radians(relative_wind_angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.stack(
        (
            cosine * centered[:, 0] + sine * centered[:, 1],
            -sine * centered[:, 0] + cosine * centered[:, 1],
        ),
        axis=-1,
    )


def common_direction_center(
    layout: MoleFarmLayout, angles_deg: Iterable[float]
) -> tuple[float, float]:
    """Find one x/z center that keeps every rotated layout inside the domain."""

    angles = tuple(float(value) for value in angles_deg)
    if not angles or not all(math.isfinite(value) for value in angles):
        raise ValueError("angles must contain finite values")
    diameter = layout.diameter_m
    lower_x = -math.inf
    upper_x = math.inf
    lower_z = -math.inf
    upper_z = math.inf
    for angle in angles:
        relative = flow_frame_relative_positions(layout, angle)
        lower_x = max(lower_x, 2.0 * diameter - float(relative[:, 0].min()))
        upper_x = min(
            upper_x,
            layout.domain_lx_m
            - 3.0 * diameter
            - float(relative[:, 0].max()),
        )
        lower_z = max(lower_z, diameter - float(relative[:, 1].min()))
        upper_z = min(
            upper_z,
            layout.domain_lz_m
            - diameter
            - float(relative[:, 1].max()),
        )
    if lower_x > upper_x or lower_z > upper_z:
        raise ValueError("no common center keeps all direction layouts in the domain")
    positions = np.asarray(layout.turbine_positions_m, dtype=np.float64)
    preferred_x = float(np.clip(positions[:, 0].mean(), lower_x, upper_x))
    preferred_z = float(np.clip(positions[:, 2].mean(), lower_z, upper_z))
    lower_index = math.ceil(lower_z / layout.dz_m)
    upper_index = math.floor(upper_z / layout.dz_m)
    if lower_index > upper_index:
        raise ValueError("no periodic z-grid point lies in the common center range")
    preferred_index = int(round(preferred_z / layout.dz_m))
    center_index = int(np.clip(preferred_index, lower_index, upper_index))
    return preferred_x, float(center_index * layout.dz_m)


def rotate_layout_to_flow_frame(
    layout: MoleFarmLayout,
    relative_wind_angle_deg: float,
    center_xz_m: tuple[float, float],
) -> MoleFarmLayout:
    """Rotate a physical layout while retaining an aligned computational inlet."""

    if len(center_xz_m) != 2 or not all(math.isfinite(value) for value in center_xz_m):
        raise ValueError("center_xz_m must contain two finite values")
    relative = flow_frame_relative_positions(layout, relative_wind_angle_deg)
    original = np.asarray(layout.turbine_positions_m, dtype=np.float64)
    positions = tuple(
        (
            float(center_xz_m[0] + relative[index, 0]),
            float(original[index, 1]),
            float(center_xz_m[1] + relative[index, 1]),
        )
        for index in range(layout.num_turbines)
    )
    angle_tag = (
        f"{relative_wind_angle_deg:+06.2f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
    rotated = replace(
        layout,
        name=f"{layout.name}_flow_angle_{angle_tag}",
        turbine_positions_m=positions,
    )
    rotated.validate()
    return rotated


def reflection_permutation(
    positive_layout: MoleFarmLayout,
    negative_layout: MoleFarmLayout,
    reflection_center_z_m: float,
    *,
    tolerance_m: float = 1.0e-7,
) -> tuple[int, ...]:
    """Map each positive-layout turbine index to its reflected negative index."""

    if positive_layout.num_turbines != negative_layout.num_turbines:
        raise ValueError("reflection layouts must contain the same turbine count")
    if not math.isfinite(reflection_center_z_m) or tolerance_m <= 0.0:
        raise ValueError("reflection center and tolerance must be valid")
    reflected = np.asarray(
        positive_layout.turbine_positions_m, dtype=np.float64
    ).copy()
    reflected[:, 2] = 2.0 * reflection_center_z_m - reflected[:, 2]
    target = np.asarray(negative_layout.turbine_positions_m, dtype=np.float64)
    distances = np.linalg.norm(reflected[:, None, :] - target[None, :, :], axis=-1)
    permutation = tuple(int(index) for index in np.argmin(distances, axis=1))
    if tuple(sorted(permutation)) != tuple(range(positive_layout.num_turbines)):
        raise ValueError("reflected turbine matching is not bijective")
    if max(distances[row, column] for row, column in enumerate(permutation)) > tolerance_m:
        raise ValueError("layouts are not reflections within tolerance")
    return permutation


def mirror_turbine_values(
    values: np.ndarray, permutation: tuple[int, ...]
) -> np.ndarray:
    """Reflect arrays whose final axis indexes turbines."""

    array = np.asarray(values)
    if array.shape[-1] != len(permutation):
        raise ValueError("value width does not match reflection permutation")
    mirrored = np.empty_like(array)
    mirrored[..., np.asarray(permutation)] = array
    return mirrored


def mirror_yaw_values(
    yaw_values: np.ndarray, permutation: tuple[int, ...]
) -> np.ndarray:
    """Reflect turbine-indexed yaw values and reverse their signed angle."""

    return -mirror_turbine_values(yaw_values, permutation)


def _reflection_indices(size: int, center_index: int) -> np.ndarray:
    if size < 1 or not 0 <= center_index < size:
        raise ValueError("reflection center index must lie on the periodic grid")
    return (2 * center_index - np.arange(size)) % size


def mirror_inlet_planes(
    planes: np.ndarray, *, center_z_index: int = 0
) -> np.ndarray:
    """Reflect periodic z planes and negate the spanwise velocity component."""

    array = np.asarray(planes)
    if array.ndim < 3 or array.shape[-3] != 3:
        raise ValueError("planes must have trailing shape (3, ny, nz)")
    z_indices = _reflection_indices(array.shape[-1], center_z_index)
    mirrored = np.take(array, z_indices, axis=-1).copy()
    mirrored[..., 2, :, :] *= -1
    return mirrored


def mirror_vector_field(
    field: np.ndarray, *, center_z_index: int = 0
) -> np.ndarray:
    """Reflect a velocity-like field with trailing shape (3, ..., nz)."""

    array = np.asarray(field)
    if array.ndim < 3 or array.shape[-4] != 3:
        raise ValueError("vector field must have component axis four from the end")
    z_indices = _reflection_indices(array.shape[-1], center_z_index)
    mirrored = np.take(array, z_indices, axis=-1).copy()
    component = [slice(None)] * mirrored.ndim
    component[-4] = 2
    mirrored[tuple(component)] *= -1
    return mirrored


def mirror_stacked_observation(
    observation: np.ndarray,
    permutation: tuple[int, ...],
    *,
    frame_stack: int = 1,
    lateral_probes: int = 7,
    streamwise_probes: int = 11,
) -> np.ndarray:
    """Reflect the Mole probe/yaw observation into the opposite direction case."""

    array = np.asarray(observation)
    num_turbines = len(permutation)
    probe_width = frame_stack * num_turbines * lateral_probes * streamwise_probes
    if array.shape[-1] != probe_width + num_turbines:
        raise ValueError("observation width does not match the reflection contract")
    probes = array[..., :probe_width].reshape(
        *array.shape[:-1],
        frame_stack,
        num_turbines,
        lateral_probes,
        streamwise_probes,
    )
    mirrored_probes = np.empty_like(probes)
    mirrored_probes[..., np.asarray(permutation), :, :] = probes[..., :, ::-1, :]
    mirrored_yaw = mirror_yaw_values(array[..., probe_width:], permutation)
    return np.concatenate(
        (mirrored_probes.reshape(*array.shape[:-1], probe_width), mirrored_yaw),
        axis=-1,
    )
