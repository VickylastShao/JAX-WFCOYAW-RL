#!/usr/bin/env python3
"""JAX workload reproduction of the Mole et al. three-turbine LES case.

This program reproduces the public computational workload and control
interface, not the unpublished turbulent inflow realizations or the paper's
4.30% control result. The reproduced elements are:

* 19D x 4D x 7D domain with a 193 x 41 x 72 grid;
* three NREL-5MW-scale actuator disks separated by 5D;
* D=126 m, hub height 90 m, dt_LES=0.2 s;
* super-Gaussian disk smearing and the public Winc3D thrust/power equations;
* 77 streamwise-velocity probes per turbine;
* 50 LES steps per 10 s control interaction;
* sixth-order explicit finite differences, an AB3 time integrator, and a
  modified-wave-number FFT pressure projection; and
* fully compiled JAX rollouts with batched independent environments.

Winc3D uses sixth-order *compact* finite differences, mixed non-periodic
boundaries, and precursor-generated atmospheric turbulence. This JAX workload
uses explicit sixth-order stencils, a periodic pressure projection, boundary
relaxation, and seeded synthetic perturbations. It is therefore suitable for
measuring JAX execution feasibility and workload-level speed, but not for
claiming solver equivalence or physical-result reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


PAPER_DOI = "10.1038/s44172-026-00667-8"
PAPER_REPORTED_TRAINING_SECONDS = 18.5 * 3600.0
PAPER_REPORTED_INTERACTIONS = 1_000_000
PAPER_LES_STEPS_PER_INTERACTION = 50
WIND_RL_COMMIT = "98d056e64e655a09d8d125cca47058ac3011d8e7"
INCOMPACT3D_SMARTREDIS_COMMIT = (
    "2c7c85bc3140d3ae983a65e44dbb50582d984709"
)


@dataclass(frozen=True)
class MoleCase:
    """Fixed physical and numerical parameters for one benchmark shape."""

    batch: int = 1
    rl_steps: int = 1
    les_steps_per_rl: int = PAPER_LES_STEPS_PER_INTERACTION
    nx: int = 193
    ny: int = 41
    nz: int = 72
    diameter: float = 126.0
    hub_height: float = 90.0
    inflow_hub_speed: float = 7.5
    dt: float = 0.2
    density: float = 1.2
    thrust_coefficient: float = 0.75
    induction_factor: float = 0.17095
    relaxation_time: float = 1.1291
    molecular_viscosity: float = 1.0 / 66667.0
    smagorinsky_constant: float = 0.14
    von_karman: float = 0.4
    roughness_length: float = 0.05
    friction_velocity: float = 0.442
    wall_damping_power: float = 3.0
    synthetic_turbulence_intensity: float = 0.05

    @property
    def lx(self) -> float:
        return 19.0 * self.diameter

    @property
    def ly(self) -> float:
        return 4.0 * self.diameter

    @property
    def lz(self) -> float:
        return 7.0 * self.diameter

    @property
    def dx(self) -> float:
        return self.lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.ly / (self.ny - 1)

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def cells_per_environment(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def total_les_steps(self) -> int:
        return self.rl_steps * self.les_steps_per_rl

    @property
    def turbine_x(self) -> tuple[float, float, float]:
        return (
            2.0 * self.diameter,
            7.0 * self.diameter,
            12.0 * self.diameter,
        )

    @property
    def turbine_z(self) -> float:
        return 3.5 * self.diameter


class FlowState(NamedTuple):
    velocity: Array
    rhs_previous: Array
    rhs_previous_2: Array
    filtered_disk_speed: Array
    step_index: Array


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _device_memory_stats(device: jax.Device) -> dict[str, Any]:
    try:
        stats = device.memory_stats()
    except Exception as exc:  # pragma: no cover - backend-dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if stats is None:
        return {"available": False}
    return {
        "available": True,
        **{
            str(key): value
            for key, value in stats.items()
            if isinstance(value, (bool, int, float, str))
        },
    }


def environment_metadata() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in devices],
        "device_kinds": [
            getattr(device, "device_kind", "unknown") for device in devices
        ],
        "memory_stats": [
            _device_memory_stats(device) for device in devices
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "xla_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
    }


def _sixth_derivative_periodic(
    values: Array,
    axis: int,
    spacing: float,
) -> Array:
    """Sixth-order explicit centered first derivative."""

    return (
        -jnp.roll(values, 3, axis=axis)
        + 9.0 * jnp.roll(values, 2, axis=axis)
        - 45.0 * jnp.roll(values, 1, axis=axis)
        + 45.0 * jnp.roll(values, -1, axis=axis)
        - 9.0 * jnp.roll(values, -2, axis=axis)
        + jnp.roll(values, -3, axis=axis)
    ) / np.float32(60.0 * spacing)


def _modified_wavenumbers(size: int, spacing: float, real: bool) -> Array:
    frequencies = (
        jnp.fft.rfftfreq(size, d=spacing)
        if real
        else jnp.fft.fftfreq(size, d=spacing)
    )
    theta = (
        np.float32(2.0 * np.pi * spacing)
        * frequencies.astype(jnp.float32)
    )
    modified = (
        45.0 * jnp.sin(theta)
        - 9.0 * jnp.sin(2.0 * theta)
        + jnp.sin(3.0 * theta)
    ) / np.float32(30.0 * spacing)
    if size % 2 == 0:
        nyquist_index = size // 2
        modified = modified.at[nyquist_index].set(0.0)
    return modified


def _rms(values: Array) -> float:
    return float(
        jnp.sqrt(jnp.mean(jnp.square(values))).block_until_ready()
    )


def build_case_functions(
    case: MoleCase,
) -> dict[str, Callable[..., Any] | Array]:
    """Build fixed-shape Mole-workload operators."""

    x = jnp.arange(case.nx, dtype=jnp.float32) * np.float32(case.dx)
    y = jnp.arange(case.ny, dtype=jnp.float32) * np.float32(case.dy)
    z = jnp.arange(case.nz, dtype=jnp.float32) * np.float32(case.dz)
    x_grid = x[None, None, :, None, None]
    y_grid = y[None, None, None, :, None]
    z_grid = z[None, None, None, None, :]

    kx = _modified_wavenumbers(
        case.nx, case.dx, real=False
    )[None, :, None, None]
    ky = _modified_wavenumbers(
        case.ny, case.dy, real=False
    )[None, None, :, None]
    kz = _modified_wavenumbers(
        case.nz, case.dz, real=True
    )[None, None, None, :]
    k_squared = kx * kx + ky * ky + kz * kz
    inverse_k_squared = jnp.where(
        k_squared > 0.0,
        1.0 / k_squared,
        0.0,
    )

    filter_width = np.float32(
        (case.dx * case.dy * case.dz) ** (1.0 / 3.0)
    )
    cell_volume = np.float32(case.dx * case.dy * case.dz)
    disk_area = np.float32(np.pi * case.diameter**2 / 4.0)
    ct_prime = np.float32(
        case.thrust_coefficient / (1.0 - case.induction_factor) ** 2
    )
    relaxation_alpha = np.float32(
        (case.dt / case.relaxation_time)
        / (1.0 + case.dt / case.relaxation_time)
    )

    raw_profile = (
        np.float32(case.friction_velocity / case.von_karman)
        * jnp.log(
            (y + np.float32(case.roughness_length))
            / np.float32(case.roughness_length)
        )
    )
    hub_profile = np.float32(
        case.friction_velocity / case.von_karman
        * math.log(
            (case.hub_height + case.roughness_length)
            / case.roughness_length
        )
    )
    streamwise_profile = (
        raw_profile
        * np.float32(case.inflow_hub_speed)
        / hub_profile
    )
    base_velocity = jnp.zeros(
        (1, 3, case.nx, case.ny, case.nz),
        dtype=jnp.float32,
    )
    base_velocity = base_velocity.at[:, 0].set(
        streamwise_profile[None, None, :, None]
    )

    wall_coordinate = y + np.float32(case.roughness_length)
    wall_scale = (
        np.float32(case.von_karman)
        * wall_coordinate
        / filter_width
    )
    damping_power = np.float32(case.wall_damping_power)
    smagorinsky = (
        np.float32(case.smagorinsky_constant) ** (-damping_power)
        + wall_scale ** (-damping_power)
    ) ** (-1.0 / damping_power)
    smagorinsky = smagorinsky[None, None, None, :, None]

    turbine_x = jnp.asarray(case.turbine_x, dtype=jnp.float32)
    turbine_y = jnp.full(
        (3,), np.float32(case.hub_height), dtype=jnp.float32
    )
    turbine_z = jnp.full(
        (3,), np.float32(case.turbine_z), dtype=jnp.float32
    )

    sensor_x_relative, sensor_z_relative = np.meshgrid(
        np.linspace(-2.0, 3.0, 11),
        np.linspace(-1.0, 1.0, 7),
    )
    sensor_x = (
        turbine_x[:, None]
        + jnp.asarray(
            sensor_x_relative.reshape(1, -1) * case.diameter,
            dtype=jnp.float32,
        )
    )
    sensor_y = jnp.full_like(sensor_x, np.float32(case.hub_height))
    sensor_z = (
        turbine_z[:, None]
        + jnp.asarray(
            sensor_z_relative.reshape(1, -1) * case.diameter,
            dtype=jnp.float32,
        )
    )
    sensor_x_index = jnp.clip(
        jnp.rint(sensor_x / np.float32(case.dx)).astype(jnp.int32),
        0,
        case.nx - 1,
    ).reshape(-1)
    sensor_y_index = jnp.clip(
        jnp.rint(sensor_y / np.float32(case.dy)).astype(jnp.int32),
        0,
        case.ny - 1,
    ).reshape(-1)
    sensor_z_index = jnp.mod(
        jnp.rint(sensor_z / np.float32(case.dz)).astype(jnp.int32),
        case.nz,
    ).reshape(-1)

    def derivative(values: Array, direction: int) -> Array:
        axes = (-3, -2, -1)
        spacings = (case.dx, case.dy, case.dz)
        return _sixth_derivative_periodic(
            values,
            axes[direction],
            spacings[direction],
        )

    def spectral_divergence(velocity: Array) -> Array:
        velocity_hat = jnp.fft.rfftn(
            velocity,
            axes=(-3, -2, -1),
        )
        divergence_hat = (
            1j * kx * velocity_hat[:, 0]
            + 1j * ky * velocity_hat[:, 1]
            + 1j * kz * velocity_hat[:, 2]
        )
        return jnp.fft.irfftn(
            divergence_hat,
            s=(case.nx, case.ny, case.nz),
            axes=(-3, -2, -1),
        )

    def project(velocity: Array) -> Array:
        velocity_hat = jnp.fft.rfftn(
            velocity,
            axes=(-3, -2, -1),
        )
        divergence_hat = (
            1j * kx * velocity_hat[:, 0]
            + 1j * ky * velocity_hat[:, 1]
            + 1j * kz * velocity_hat[:, 2]
        )
        potential_hat = -divergence_hat * inverse_k_squared
        correction_hat = jnp.stack(
            (
                1j * kx * potential_hat,
                1j * ky * potential_hat,
                1j * kz * potential_hat,
            ),
            axis=1,
        )
        correction = jnp.fft.irfftn(
            correction_hat,
            s=(case.nx, case.ny, case.nz),
            axes=(-3, -2, -1),
        )
        return velocity - correction

    def apply_boundary_relaxation(velocity: Array) -> Array:
        # The public precursor planes are unavailable. Relaxing six inlet
        # planes to the seeded ABL state supplies a deterministic workload
        # without pretending to reproduce those realizations.
        inlet_weight = jnp.asarray(
            [1.0, 0.85, 0.65, 0.45, 0.25, 0.1],
            dtype=jnp.float32,
        )[None, None, :, None, None]
        inlet = velocity[:, :, :6]
        target = jnp.broadcast_to(
            base_velocity[:, :, :6],
            inlet.shape,
        )
        velocity = velocity.at[:, :, :6].set(
            inlet * (1.0 - inlet_weight) + target * inlet_weight
        )
        velocity = velocity.at[:, :, -1].set(velocity[:, :, -2])
        velocity = velocity.at[:, :, :, 0, :].set(0.0)
        top = velocity[:, :, :, -2, :]
        top = top.at[:, 1].set(0.0)
        velocity = velocity.at[:, :, :, -1, :].set(top)
        return velocity

    def disk_weights(yaw_degrees: Array) -> Array:
        yaw = jnp.deg2rad(yaw_degrees)[:, :, None, None, None]
        normal_x = jnp.cos(yaw)
        normal_z = -jnp.sin(yaw)
        delta_x = x_grid - turbine_x[None, :, None, None, None]
        delta_y = y_grid - turbine_y[None, :, None, None, None]
        delta_z = z_grid - turbine_z[None, :, None, None, None]
        delta_z = jnp.mod(
            delta_z + np.float32(0.5 * case.lz),
            np.float32(case.lz),
        ) - np.float32(0.5 * case.lz)
        normal_distance = delta_x * normal_x + delta_z * normal_z
        projected_x = delta_x - normal_distance * normal_x
        projected_z = delta_z - normal_distance * normal_z
        radial_distance = jnp.sqrt(
            projected_x * projected_x
            + delta_y * delta_y
            + projected_z * projected_z
        )
        grid_normal_spacing = jnp.sqrt(
            (np.float32(case.dx) * normal_x) ** 2
            + (np.float32(case.dz) * normal_z) ** 2
        )
        disk_thickness = jnp.maximum(
            np.float32(case.diameter / 8.0),
            1.5 * grid_normal_spacing,
        )
        weights = jnp.exp(
            -(normal_distance / (0.5 * disk_thickness)) ** 2
            -(radial_distance / np.float32(0.5 * case.diameter)) ** 8
        )
        return weights / jnp.sum(
            weights,
            axis=(-3, -2, -1),
            keepdims=True,
        )

    def disk_source(
        velocity: Array,
        yaw_degrees: Array,
        weights: Array,
        previous_filtered_speed: Array,
        first_step: Array,
    ) -> tuple[Array, Array, Array, Array]:
        yaw = jnp.deg2rad(yaw_degrees)
        normal = jnp.stack(
            (
                jnp.cos(yaw),
                jnp.zeros_like(yaw),
                -jnp.sin(yaw),
            ),
            axis=2,
        )
        normal_velocity = jnp.sum(
            velocity[:, None] * normal[:, :, :, None, None, None],
            axis=2,
        )
        disk_speed = jnp.sum(
            weights * normal_velocity,
            axis=(-3, -2, -1),
        )
        relaxed_speed = (
            np.float32(relaxation_alpha) * disk_speed
            + np.float32(1.0 - relaxation_alpha)
            * previous_filtered_speed
        )
        filtered_speed = jnp.where(
            first_step,
            disk_speed,
            relaxed_speed,
        )
        thrust = (
            np.float32(0.5 * case.density)
            * ct_prime
            * filtered_speed**2
            * disk_area
        )
        power = thrust * filtered_speed
        acceleration = -jnp.sum(
            (
                thrust[:, :, None, None, None, None]
                / np.float32(case.density * cell_volume)
            )
            * weights[:, :, None]
            * normal[:, :, :, None, None, None],
            axis=1,
        )
        return acceleration, filtered_speed, power, thrust

    def flow_rhs(
        velocity: Array,
        yaw_degrees: Array,
        weights: Array,
        previous_filtered_speed: Array,
        first_step: Array,
    ) -> tuple[Array, Array, Array, Array]:
        gradient = jnp.stack(
            tuple(derivative(velocity, direction) for direction in range(3)),
            axis=2,
        )
        advective = jnp.sum(
            velocity[:, None] * gradient,
            axis=2,
        )
        conservative_parts = []
        for direction in range(3):
            flux = velocity * velocity[:, direction : direction + 1]
            conservative_parts.append(derivative(flux, direction))
        conservative = sum(conservative_parts)
        advection = -0.5 * (advective + conservative)

        strain = 0.5 * (
            gradient + jnp.swapaxes(gradient, 1, 2)
        )
        strain_magnitude = jnp.sqrt(
            2.0 * jnp.sum(strain * strain, axis=(1, 2), keepdims=True)
            + 1.0e-20
        )
        eddy_viscosity = (
            (smagorinsky * filter_width) ** 2
            * strain_magnitude
        )
        effective_viscosity = (
            np.float32(case.molecular_viscosity) + eddy_viscosity
        )
        stress = 2.0 * effective_viscosity * strain
        diffusion = sum(
            derivative(stress[:, :, direction], direction)
            for direction in range(3)
        )
        acceleration, filtered_speed, power, thrust = disk_source(
            velocity,
            yaw_degrees,
            weights,
            previous_filtered_speed,
            first_step,
        )
        return advection + diffusion + acceleration, filtered_speed, power, thrust

    def les_step(
        state: FlowState,
        inputs: tuple[Array, Array],
    ) -> tuple[FlowState, tuple[Array, Array]]:
        yaw_degrees, weights = inputs
        rhs, filtered_speed, power, _ = flow_rhs(
            state.velocity,
            yaw_degrees,
            weights,
            state.filtered_disk_speed,
            state.step_index == 0,
        )
        euler = state.velocity + np.float32(case.dt) * rhs
        ab2 = state.velocity + np.float32(case.dt) * (
            1.5 * rhs - 0.5 * state.rhs_previous
        )
        ab3 = state.velocity + np.float32(case.dt) * (
            (23.0 / 12.0) * rhs
            - (16.0 / 12.0) * state.rhs_previous
            + (5.0 / 12.0) * state.rhs_previous_2
        )
        predicted = jnp.where(
            state.step_index == 0,
            euler,
            jnp.where(state.step_index == 1, ab2, ab3),
        )
        predicted = apply_boundary_relaxation(predicted)
        updated_velocity = project(predicted)
        next_state = FlowState(
            velocity=updated_velocity,
            rhs_previous=rhs,
            rhs_previous_2=state.rhs_previous,
            filtered_disk_speed=filtered_speed,
            step_index=state.step_index + 1,
        )
        return next_state, (power, filtered_speed)

    def probe_observations(velocity: Array) -> Array:
        observations = velocity[
            :,
            0,
            sensor_x_index,
            sensor_y_index,
            sensor_z_index,
        ]
        return observations.reshape(case.batch, 3, 77)

    def rollout(
        state: FlowState,
        yaw_schedule_degrees: Array,
    ) -> tuple[FlowState, tuple[Array, Array]]:
        def control_step(
            current: FlowState,
            yaw_degrees: Array,
        ) -> tuple[FlowState, tuple[Array, Array]]:
            weights = disk_weights(yaw_degrees)

            def fixed_control_les_step(
                inner_state: FlowState,
                _: None,
            ) -> tuple[FlowState, tuple[Array, Array]]:
                return les_step(
                    inner_state,
                    (yaw_degrees, weights),
                )

            final_state, (powers, _) = jax.lax.scan(
                fixed_control_les_step,
                current,
                xs=None,
                length=case.les_steps_per_rl,
            )
            mean_power = jnp.mean(powers, axis=0)
            probes = probe_observations(final_state.velocity)
            return final_state, (mean_power, probes)

        return jax.lax.scan(control_step, state, yaw_schedule_degrees)

    return {
        "base_velocity": base_velocity,
        "project": project,
        "spectral_divergence": spectral_divergence,
        "disk_weights": disk_weights,
        "disk_source": disk_source,
        "probe_observations": probe_observations,
        "rollout": rollout,
        "cell_volume": jnp.asarray(cell_volume),
        "disk_area": jnp.asarray(disk_area),
        "ct_prime": jnp.asarray(ct_prime),
    }


def make_initial_state(
    case: MoleCase,
    functions: dict[str, Callable[..., Any] | Array],
    seed: int,
) -> FlowState:
    key = jax.random.PRNGKey(seed)
    base = jnp.broadcast_to(
        functions["base_velocity"],
        (case.batch, 3, case.nx, case.ny, case.nz),
    )
    noise_scale = np.float32(
        case.inflow_hub_speed * case.synthetic_turbulence_intensity
    )
    perturbation = noise_scale * jax.random.normal(
        key,
        base.shape,
        dtype=jnp.float32,
    )
    perturbation = functions["project"](perturbation)
    velocity = functions["project"](base + perturbation)
    zeros = jnp.zeros_like(velocity)
    return FlowState(
        velocity=velocity,
        rhs_previous=zeros,
        rhs_previous_2=zeros,
        filtered_disk_speed=jnp.zeros((case.batch, 3), dtype=jnp.float32),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def make_yaw_schedule(case: MoleCase) -> Array:
    phase = jnp.linspace(
        0.0,
        2.0 * jnp.pi,
        case.rl_steps,
        endpoint=False,
        dtype=jnp.float32,
    )
    base = jnp.stack(
        (
            20.0 + 3.0 * jnp.sin(phase),
            13.0 + 2.0 * jnp.sin(phase - 0.4),
            -3.0 + 1.0 * jnp.sin(phase - 0.8),
        ),
        axis=1,
    )
    batch_offset = jnp.linspace(
        -0.5,
        0.5,
        case.batch,
        dtype=jnp.float32,
    )
    return base[:, None, :] + batch_offset[None, :, None]


def run_correctness(case: MoleCase, seed: int) -> dict[str, Any]:
    functions = build_case_functions(case)
    project = jax.jit(functions["project"])
    disk_weights = jax.jit(functions["disk_weights"])
    disk_source = jax.jit(functions["disk_source"])
    rollout = jax.jit(functions["rollout"])
    state = make_initial_state(case, functions, seed)
    state.velocity.block_until_ready()

    raw = jax.random.normal(
        jax.random.PRNGKey(seed + 1),
        state.velocity.shape,
        dtype=jnp.float32,
    )
    raw_divergence = _rms(functions["spectral_divergence"](raw))
    projected = project(raw)
    projected.block_until_ready()
    projected_divergence = _rms(
        functions["spectral_divergence"](projected)
    )
    projection_ratio = projected_divergence / max(
        raw_divergence, 1.0e-30
    )

    yaw = make_yaw_schedule(case)[0]
    weights = disk_weights(yaw)
    weights.block_until_ready()
    weight_sum_error = float(
        jnp.max(
            jnp.abs(
                jnp.sum(weights, axis=(-3, -2, -1)) - 1.0
            )
        ).block_until_ready()
    )
    acceleration, filtered_speed, power, thrust = disk_source(
        state.velocity,
        yaw,
        weights,
        state.filtered_disk_speed,
        jnp.asarray(True),
    )
    acceleration.block_until_ready()
    force_integral = (
        np.float32(case.density)
        * jnp.sum(acceleration, axis=(-3, -2, -1))
        * functions["cell_volume"]
    )
    normal = jnp.stack(
        (
            jnp.cos(jnp.deg2rad(yaw)),
            jnp.zeros_like(yaw),
            -jnp.sin(jnp.deg2rad(yaw)),
        ),
        axis=2,
    )
    expected_force = -jnp.sum(thrust[:, :, None] * normal, axis=1)
    force_relative_error = float(
        (
            jnp.max(jnp.abs(force_integral - expected_force))
            / jnp.maximum(jnp.max(jnp.abs(expected_force)), 1.0)
        ).block_until_ready()
    )
    power_identity_error = float(
        (
            jnp.max(jnp.abs(power - thrust * filtered_speed))
            / jnp.maximum(jnp.max(jnp.abs(power)), 1.0)
        ).block_until_ready()
    )

    final_state, (mean_powers, probes) = rollout(
        state,
        make_yaw_schedule(case),
    )
    final_state.velocity.block_until_ready()
    final_divergence = _rms(
        functions["spectral_divergence"](final_state.velocity)
    )
    finite = bool(
        np.asarray(
            jnp.all(jnp.isfinite(final_state.velocity)).block_until_ready()
        )
    )
    probe_shape = tuple(int(value) for value in probes.shape)
    expected_probe_shape = (case.rl_steps, case.batch, 3, 77)
    power_shape = tuple(int(value) for value in mean_powers.shape)
    expected_power_shape = (case.rl_steps, case.batch, 3)

    checks = {
        "case_grid_matches_paper": (
            case.nx,
            case.ny,
            case.nz,
        ) == (193, 41, 72),
        "case_domain_matches_paper": np.allclose(
            (case.lx, case.ly, case.lz),
            (
                19.0 * case.diameter,
                4.0 * case.diameter,
                7.0 * case.diameter,
            ),
        ),
        "three_turbines_are_5d_apart": np.allclose(
            np.diff(case.turbine_x),
            5.0 * case.diameter,
        ),
        "projection_reduces_divergence": projection_ratio < 1.0e-4,
        "disk_weights_are_normalized": weight_sum_error < 2.0e-5,
        "integrated_force_matches_thrust": force_relative_error < 2.0e-5,
        "power_equals_thrust_times_speed": power_identity_error < 2.0e-6,
        "rollout_is_finite": finite,
        "final_divergence_is_bounded": final_divergence < 2.0e-4,
        "probe_interface_matches_paper": probe_shape
        == expected_probe_shape,
        "power_interface_has_three_turbines": power_shape
        == expected_power_shape,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": {
            "raw_divergence_rms": raw_divergence,
            "projected_divergence_rms": projected_divergence,
            "projection_ratio": projection_ratio,
            "final_divergence_rms": final_divergence,
            "disk_weight_sum_max_abs_error": weight_sum_error,
            "integrated_force_relative_error": force_relative_error,
            "power_identity_relative_error": power_identity_error,
            "mean_turbine_power_watts": np.asarray(
                mean_powers[-1]
            ).tolist(),
            "filtered_disk_speed_m_per_s": np.asarray(
                final_state.filtered_disk_speed
            ).tolist(),
            "probe_shape": list(probe_shape),
            "power_shape": list(power_shape),
        },
    }


def run_benchmark(
    case: MoleCase,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    functions = build_case_functions(case)
    rollout = jax.jit(functions["rollout"])
    state = make_initial_state(case, functions, seed)
    state.velocity.block_until_ready()
    yaw_schedule = make_yaw_schedule(case)

    compile_start = time.perf_counter()
    compiled = rollout(state, yaw_schedule)
    compiled[0].velocity.block_until_ready()
    compile_and_first_seconds = time.perf_counter() - compile_start

    durations: list[float] = []
    final = compiled
    for _ in range(repeats):
        start = time.perf_counter()
        final = rollout(state, yaw_schedule)
        final[0].velocity.block_until_ready()
        durations.append(time.perf_counter() - start)

    median_seconds = float(np.median(durations))
    environment_les_steps = case.batch * case.total_les_steps
    les_steps_per_second = environment_les_steps / median_seconds
    interactions = case.batch * case.rl_steps
    interactions_per_second = interactions / median_seconds
    cell_updates_per_second = (
        environment_les_steps
        * case.cells_per_environment
        / median_seconds
    )
    final_state, (mean_powers, probes) = final
    final_divergence = _rms(
        functions["spectral_divergence"](final_state.velocity)
    )
    finite = bool(
        np.asarray(
            jnp.all(jnp.isfinite(final_state.velocity)).block_until_ready()
        )
    )
    paper_aggregate_les_steps_per_second = (
        PAPER_REPORTED_INTERACTIONS
        * PAPER_LES_STEPS_PER_INTERACTION
        / PAPER_REPORTED_TRAINING_SECONDS
    )
    workload_speed_ratio_to_reported_training = (
        les_steps_per_second / paper_aggregate_les_steps_per_second
    )
    projected_two_gpu_seconds = (
        PAPER_REPORTED_INTERACTIONS
        * PAPER_LES_STEPS_PER_INTERACTION
        / (2.0 * les_steps_per_second)
    )

    return {
        "status": (
            "pass"
            if finite and final_divergence < 2.0e-4
            else "fail"
        ),
        "metrics": {
            "compile_and_first_run_seconds": (
                compile_and_first_seconds
            ),
            "steady_run_seconds": durations,
            "steady_run_median_seconds": median_seconds,
            "les_steps_per_second": les_steps_per_second,
            "rl_interactions_per_second": interactions_per_second,
            "cell_updates_per_second": cell_updates_per_second,
            "seconds_per_les_step_per_environment": (
                median_seconds / environment_les_steps
            ),
            "final_divergence_rms": final_divergence,
            "rollout_is_finite": finite,
            "mean_turbine_power_watts": np.asarray(
                mean_powers[-1]
            ).tolist(),
            "probe_shape": list(probes.shape),
            "paper_reported_aggregate_les_steps_per_second": (
                paper_aggregate_les_steps_per_second
            ),
            "single_gpu_workload_speed_ratio_to_paper_training": (
                workload_speed_ratio_to_reported_training
            ),
            "projected_two_gpu_workload_seconds_for_1m_interactions": (
                projected_two_gpu_seconds
            ),
        },
    }


def case_metadata(case: MoleCase) -> dict[str, Any]:
    metadata = asdict(case)
    metadata.update(
        {
            "lx": case.lx,
            "ly": case.ly,
            "lz": case.lz,
            "dx": case.dx,
            "dy": case.dy,
            "dz": case.dz,
            "cells_per_environment": case.cells_per_environment,
            "total_les_steps": case.total_les_steps,
            "turbine_x": case.turbine_x,
            "turbine_y": case.hub_height,
            "turbine_z": case.turbine_z,
            "turbine_spacing_diameters": 5.0,
            "probes_per_turbine": 77,
        }
    )
    return metadata


def source_metadata() -> dict[str, Any]:
    return {
        "paper_doi": PAPER_DOI,
        "wind_rl_url": "https://github.com/admole/Wind-RL",
        "wind_rl_commit": WIND_RL_COMMIT,
        "incompact3d_url": "https://github.com/admole/Incompact3d",
        "incompact3d_branch": "smartRedis-coupling",
        "incompact3d_commit": INCOMPACT3D_SMARTREDIS_COMMIT,
        "zenodo_doi": "10.5281/zenodo.15705117",
        "paper_reported_training": {
            "seconds": PAPER_REPORTED_TRAINING_SECONDS,
            "interactions": PAPER_REPORTED_INTERACTIONS,
            "parallel_les_environments": 32,
            "les_steps_per_interaction": PAPER_LES_STEPS_PER_INTERACTION,
            "compute_nodes": 34,
            "cpu_description": (
                "dual AMD EPYC 7742 64-core processors per node"
            ),
        },
        "known_reproduction_limits": [
            "The paper's 32 precursor turbulent inflow realizations are not "
            "present in the public Wind-RL repository or Zenodo result bundle.",
            "The public Wind-RL default YAML does not equal all final values "
            "reported in Supplementary Table S1.",
            "The paper text states C0=1.4 while the tracked Winc3D input uses "
            "smagcst=0.14; this workload follows the executable input value.",
            "The JAX solver uses explicit rather than compact sixth-order "
            "finite differences and a periodic FFT projection.",
            "Projected end-to-end training time excludes SAC updates, replay "
            "buffer work, checkpointing, input I/O, and distributed overhead.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("check", "benchmark"),
        default="check",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rl-steps", type=int, default=1)
    parser.add_argument(
        "--les-steps-per-rl",
        type=int,
        default=PAPER_LES_STEPS_PER_INTERACTION,
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch < 1 or args.rl_steps < 1:
        raise ValueError("batch and rl-steps must be positive")
    if args.les_steps_per_rl < 1:
        raise ValueError("les-steps-per-rl must be positive")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    case = MoleCase(
        batch=args.batch,
        rl_steps=args.rl_steps,
        les_steps_per_rl=args.les_steps_per_rl,
    )
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    result = (
        run_correctness(case, args.seed)
        if args.mode == "check"
        else run_benchmark(case, args.seed, args.repeats)
    )
    payload = {
        "schema_version": 1,
        "purpose": "mole_case_jax_workload_reproduction_not_physical_replica",
        "mode": args.mode,
        "started_at": started_at,
        "source": source_metadata(),
        "case": case_metadata(case),
        "environment": environment_metadata(),
        "result": result,
    }
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    payload_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    payload["payload_sha256_without_hash_field"] = payload_hash
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
