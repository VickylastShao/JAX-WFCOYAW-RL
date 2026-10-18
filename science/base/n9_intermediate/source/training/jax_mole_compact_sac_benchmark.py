#!/usr/bin/env python3
"""Compact-difference Mole LES workload with precursor inflow and SAC.

This is a reproducible JAX implementation of the public computational
contract in Mole et al., not a bitwise port of WInc3D and not a reproduction
of the unpublished precursor realizations. It adds the three missing workload
components from the first feasibility benchmark:

* the WInc3D sixth-order tridiagonal compact first derivative;
* Incompact3d's half-staggered pressure projection for the farm domain;
* the source-aligned x/z compact spatial filter used by Mole's farm input;
* replay of turbulence evolved in a separate periodic precursor LES; and
* on-device SAC actor, twin critics, entropy temperature, replay buffer, and
  Polyak target updates inside the timed training iteration.

The precursor is physics-evolved and reproducible, but it is not measured
atmospheric data and is not one of the paper's unavailable inflow fields.
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
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.fft as jsp_fft
import numpy as np
import optax


REAL_DTYPE_NAME = os.environ.get(
    "MOLE_JAX_REAL_DTYPE", "float32"
).strip().lower()
if REAL_DTYPE_NAME not in {"float32", "float64"}:
    raise ValueError(
        "MOLE_JAX_REAL_DTYPE must be 'float32' or 'float64', got "
        f"{REAL_DTYPE_NAME!r}"
    )
if REAL_DTYPE_NAME == "float64":
    jax.config.update("jax_enable_x64", True)
NP_REAL_DTYPE = (
    np.float64 if REAL_DTYPE_NAME == "float64" else np.float32
)
JAX_REAL_DTYPE = (
    jnp.float64 if REAL_DTYPE_NAME == "float64" else jnp.float32
)

from jax_mole_case_benchmark import (
    INCOMPACT3D_SMARTREDIS_COMMIT,
    PAPER_DOI,
    PAPER_LES_STEPS_PER_INTERACTION,
    PAPER_REPORTED_INTERACTIONS,
    PAPER_REPORTED_TRAINING_SECONDS,
    WIND_RL_COMMIT,
)
from incompact3d_staggered import (
    periodic_x_staggered_divergence,
    periodic_x_staggered_gradient,
    remove_reference_divergence,
    solve_periodic_x_pressure_poisson,
    solve_pressure_poisson,
    staggered_divergence,
    staggered_gradient,
)
from incompact3d_filter import (
    mole_farm_all_direction_filter,
    mole_farm_spatial_filter,
    mole_precursor_spatial_filter,
)
from jax_mole_layouts import MoleFarmLayout


Array = jax.Array
PyTree = Any


@dataclass(frozen=True)
class CompactMoleCase:
    batch: int = 1
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
    boundary_layer_height: float = 504.0
    pressure_gradient_forcing: bool = True
    wall_damping_power: float = 3.0
    wall_sampling_dy: float = 2.2
    wall_sgs_model: bool = True
    convective_outflow: bool = True
    spatial_filter: bool = True
    farm_filter_all_directions: bool = False
    filter_coefficient: float = 0.49
    filter_solver: str = "dense"
    domain_lx: float = 2394.0
    domain_ly: float = 500.0
    domain_lz: float = 882.0
    x_periodic_grid: bool = False
    shifted_periodic_fringe: bool = True
    profile_relaxation_forcing: bool = False
    source_aligned_initial_noise: bool = True
    initialize_with_log_profile: bool = True
    precursor_advection_speed: float = 8.0
    precursor_dx: float = 12.25
    precursor_initial_ti: float = 0.10
    precursor_profile_relaxation_seconds: float = 100.0
    pressure_corrections: int = 1
    pressure_projection: str = "incompact3d_staggered"
    pressure_fft_mode: str = "sequential"
    compact_solver: str = "dense_inverse"
    staggered_y_solver: str = "dense"
    rhs_assembly: str = "tensor"

    @property
    def lx(self) -> float:
        return self.domain_lx

    @property
    def ly(self) -> float:
        return self.domain_ly

    @property
    def lz(self) -> float:
        return self.domain_lz

    @property
    def dx(self) -> float:
        return self.lx / (self.nx if self.x_periodic_grid else self.nx - 1)

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
    def turbine_x(self) -> tuple[float, float, float]:
        return (
            2.0 * self.diameter,
            7.0 * self.diameter,
            12.0 * self.diameter,
        )

    @property
    def turbine_z(self) -> float:
        return 3.5 * self.diameter

    @property
    def turbine_positions_m(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (x, self.hub_height, self.turbine_z) for x in self.turbine_x
        )

    @property
    def num_turbines(self) -> int:
        return len(self.turbine_positions_m)


@dataclass(frozen=True)
class ScalableCompactMoleCase(CompactMoleCase):
    """Compact Mole case carrying an explicit validated farm layout."""

    layout_name: str = "mole_rect_3x1"
    layout_sha256: str = ""
    configured_turbine_positions_m: tuple[
        tuple[float, float, float], ...
    ] = (
        (252.0, 90.0, 441.0),
        (882.0, 90.0, 441.0),
        (1512.0, 90.0, 441.0),
    )

    @property
    def turbine_positions_m(self) -> tuple[tuple[float, float, float], ...]:
        return self.configured_turbine_positions_m

    @property
    def turbine_x(self) -> tuple[float, ...]:
        return tuple(position[0] for position in self.turbine_positions_m)

    @property
    def num_turbines(self) -> int:
        return len(self.turbine_positions_m)


class FlowState(NamedTuple):
    velocity: Array
    rhs_previous: Array
    rhs_previous_2: Array
    filtered_disk_speed: Array
    bottom_pressure_gradient: Array
    open_x_pressure_gradient: Array
    step_index: Array


class ReplayState(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    position: Array
    size: Array


class SACState(NamedTuple):
    actor: PyTree
    critic: PyTree
    target_critic: PyTree
    log_alpha: Array
    actor_opt: PyTree
    critic_opt: PyTree
    alpha_opt: PyTree
    updates: Array


class TrainingState(NamedTuple):
    flow: FlowState
    observation: Array
    previous_yaw: Array
    replay: ReplayState
    sac: SACState
    key: Array


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_all_finite(tree: PyTree) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def environment_metadata() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "optax": optax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_kinds": [
            getattr(device, "device_kind", "unknown") for device in devices
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "mole_real_dtype": REAL_DTYPE_NAME,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "xla_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
    }


def _compact_periodic_wavenumber(size: int, spacing: float) -> Array:
    theta = (
        2.0
        * NP_REAL_DTYPE(np.pi)
        * jnp.fft.fftfreq(size).astype(JAX_REAL_DTYPE)
    )
    alpha = NP_REAL_DTYPE(1.0 / 3.0)
    a = NP_REAL_DTYPE(7.0 / (9.0 * spacing))
    b = NP_REAL_DTYPE(1.0 / (36.0 * spacing))
    return (
        2.0 * a * jnp.sin(theta) + 2.0 * b * jnp.sin(2.0 * theta)
    ) / (1.0 + 2.0 * alpha * jnp.cos(theta))


@lru_cache(maxsize=None)
def _nonperiodic_compact_inverse(size: int) -> np.ndarray:
    """Precompute the fixed WInc3D compact tridiagonal inverse."""

    alpha = NP_REAL_DTYPE(1.0 / 3.0)
    matrix = np.eye(size, dtype=NP_REAL_DTYPE)
    row = np.arange(size - 1)
    matrix[row + 1, row] = alpha
    matrix[row, row + 1] = alpha
    matrix[0, 1] = 2.0
    matrix[1, 0] = 0.25
    matrix[1, 2] = 0.25
    matrix[-2, -3] = 0.25
    matrix[-2, -1] = 0.25
    matrix[-1, -2] = 2.0
    return np.linalg.inv(matrix).astype(NP_REAL_DTYPE)


@lru_cache(maxsize=None)
def _nonperiodic_compact_diagonals(
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed WInc3D compact tridiagonal coefficients."""

    alpha = NP_REAL_DTYPE(1.0 / 3.0)
    lower = np.full((size,), alpha, dtype=NP_REAL_DTYPE)
    diagonal = np.ones((size,), dtype=NP_REAL_DTYPE)
    upper = np.full((size,), alpha, dtype=NP_REAL_DTYPE)
    lower[0] = 0.0
    lower[1] = 0.25
    lower[-2] = 0.25
    lower[-1] = 2.0
    upper[0] = 2.0
    upper[1] = 0.25
    upper[-2] = 0.25
    upper[-1] = 0.0
    return lower, diagonal, upper


def compact_first_derivative(
    values: Array,
    axis: int,
    spacing: float,
    periodic: bool,
    solver: str = "dense_inverse",
) -> Array:
    """WInc3D sixth-order compact derivative with tracked coefficients."""

    size = values.shape[axis]
    if size < 7:
        raise ValueError("compact derivative requires at least seven points")

    if periodic:
        wavenumber = _compact_periodic_wavenumber(size, spacing)
        shape = [1] * values.ndim
        shape[axis] = size
        multiplier = 1j * wavenumber.reshape(shape)
        transformed = jnp.fft.fft(values, axis=axis)
        return jnp.fft.ifft(multiplier * transformed, axis=axis).real

    moved = jnp.moveaxis(values, axis, -1)
    if solver == "cute_fused_41" and size == 41:
        if values.dtype != jnp.float32:
            raise ValueError("CuTe compact kernel currently supports float32 only")
        from cute_compact_derivative import compact_first_derivative_41_last_axis

        solution = compact_first_derivative_41_last_axis(moved, spacing)
        return jnp.moveaxis(solution, -1, axis)
    effective_solver = "dense_inverse" if solver == "cute_fused_41" else solver
    flat = moved.reshape((-1, size))
    inv_h = NP_REAL_DTYPE(1.0 / spacing)

    rhs = jnp.zeros_like(flat)
    rhs = rhs.at[:, 0].set(
        inv_h
        * (
            -2.5 * flat[:, 0]
            + 2.0 * flat[:, 1]
            + 0.5 * flat[:, 2]
        )
    )
    rhs = rhs.at[:, 1].set(
        NP_REAL_DTYPE(0.75) * inv_h * (flat[:, 2] - flat[:, 0])
    )
    rhs = rhs.at[:, 2:-2].set(
        NP_REAL_DTYPE(7.0 / 9.0)
        * inv_h
        * (flat[:, 3:-1] - flat[:, 1:-3])
        + NP_REAL_DTYPE(1.0 / 36.0)
        * inv_h
        * (flat[:, 4:] - flat[:, :-4])
    )
    rhs = rhs.at[:, -2].set(
        NP_REAL_DTYPE(0.75) * inv_h * (flat[:, -1] - flat[:, -3])
    )
    rhs = rhs.at[:, -1].set(
        inv_h
        * (
            2.5 * flat[:, -1]
            - 2.0 * flat[:, -2]
            - 0.5 * flat[:, -3]
        )
    )

    if effective_solver == "dense_inverse":
        inverse = jnp.asarray(
            _nonperiodic_compact_inverse(size), dtype=values.dtype
        )
        solution = rhs @ inverse.T
    elif effective_solver == "tridiagonal":
        lower, diagonal, upper = (
            jnp.asarray(value, dtype=values.dtype)
            for value in _nonperiodic_compact_diagonals(size)
        )
        solution = jax.lax.linalg.tridiagonal_solve(
            lower,
            diagonal,
            upper,
            rhs.T,
        ).T
    else:
        raise ValueError(f"unknown compact solver: {solver}")
    restored = solution.reshape(moved.shape)
    return jnp.moveaxis(restored, -1, axis)


def _compact_neumann_wavenumber(size: int, spacing: float) -> Array:
    theta = (
        NP_REAL_DTYPE(np.pi / size)
        * jnp.arange(size, dtype=JAX_REAL_DTYPE)
    )
    alpha = NP_REAL_DTYPE(1.0 / 3.0)
    a = NP_REAL_DTYPE(7.0 / (9.0 * spacing))
    b = NP_REAL_DTYPE(1.0 / (36.0 * spacing))
    return (
        2.0 * a * jnp.sin(theta) + 2.0 * b * jnp.sin(2.0 * theta)
    ) / (1.0 + 2.0 * alpha * jnp.cos(theta))


def _compact_rfft_wavenumber(size: int, spacing: float) -> Array:
    theta = (
        NP_REAL_DTYPE(2.0 * np.pi)
        * jnp.fft.rfftfreq(size).astype(JAX_REAL_DTYPE)
    )
    alpha = NP_REAL_DTYPE(1.0 / 3.0)
    a = NP_REAL_DTYPE(7.0 / (9.0 * spacing))
    b = NP_REAL_DTYPE(1.0 / (36.0 * spacing))
    return (
        2.0 * a * jnp.sin(theta) + 2.0 * b * jnp.sin(2.0 * theta)
    ) / (1.0 + 2.0 * alpha * jnp.cos(theta))


def _apply_velocity_boundaries(
    velocity: Array,
    inlet: Array | None,
    periodic_x: bool,
    outflow_boundary: Array | None = None,
) -> Array:
    if not periodic_x:
        if inlet is None:
            raise ValueError("non-periodic x requires an inlet plane")
        velocity = velocity.at[:, :, 0].set(inlet)
        if outflow_boundary is None:
            outlet = velocity[:, :, -2]
        else:
            outlet = outflow_boundary
        velocity = velocity.at[:, :, -1].set(outlet)
    velocity = velocity.at[:, :, :, 0, :].set(0.0)
    # XCompact3D's nclyn=1 pre-correction constrains only the wall-normal
    # component. The two tangential top-plane values remain prognostic.
    velocity = velocity.at[:, 1, :, -1, :].set(0.0)
    return velocity


def _apply_shifted_periodic_fringe(
    velocity: Array, case: CompactMoleCase
) -> Array:
    """Apply Case-ABL's shifted-periodic fringe for the precursor."""

    nshift = int((case.lz / 8.0) / case.dz)
    shifted = jnp.roll(velocity, shift=nshift, axis=-1)
    fringe_end = NP_REAL_DTYPE(case.lx * (2.0 / 3.0))
    fringe_length = NP_REAL_DTYPE(fringe_end / 6.0)
    fringe_start = NP_REAL_DTYPE(fringe_end - fringe_length)
    nfringe = int(case.nx * fringe_end / case.lx)
    x = jnp.arange(nfringe, dtype=velocity.dtype) * NP_REAL_DTYPE(case.dx)
    ramp_end = NP_REAL_DTYPE(fringe_end - fringe_length / 4.0)
    ramp = NP_REAL_DTYPE(0.5) * (
        1.0
        - jnp.cos(
            NP_REAL_DTYPE(4.0 * np.pi / 3.0)
            * (x - fringe_start)
            / fringe_length
        )
    )
    weight = jnp.where(
        x < fringe_start,
        0.0,
        jnp.where(x < ramp_end, ramp, jnp.where(x < fringe_end, 1.0, 0.0)),
    )
    weight = weight[None, None, :, None, None]
    target = velocity[:, :, case.nx - nfringe :]
    source = shifted[:, :, :nfringe]
    return velocity.at[:, :, case.nx - nfringe :].set(
        weight * source + (1.0 - weight) * target
    )


def _mixed_pressure_solver(
    divergence: Array,
    case: CompactMoleCase,
    periodic_x: bool,
) -> Array:
    """Solve Poisson with Neumann y, periodic z, and Neumann/periodic x."""

    if periodic_x:
        transformed_y = jsp_fft.dct(
            divergence,
            type=2,
            axis=-2,
            norm="ortho",
        )
        transformed = jnp.fft.rfftn(
            transformed_y,
            axes=(-3, -1),
        )
        kx = _compact_periodic_wavenumber(
            case.nx, case.dx
        )[:, None, None]
        ky = _compact_neumann_wavenumber(
            case.ny, case.dy
        )[None, :, None]
        kz = _compact_rfft_wavenumber(
            case.nz, case.dz
        )[None, None, :]
        k_squared = kx * kx + ky * ky + kz * kz
        potential_hat = jnp.where(
            k_squared > 1.0e-10,
            -transformed / k_squared,
            0.0,
        )
        potential_y = jnp.fft.irfftn(
            potential_hat,
            s=(case.nx, case.nz),
            axes=(-3, -1),
        )
        return jsp_fft.idct(
            potential_y,
            type=2,
            axis=-2,
            norm="ortho",
        )

    transformed_xy = jsp_fft.dctn(
        divergence,
        type=2,
        axes=(-3, -2),
        norm="ortho",
    )
    transformed = jnp.fft.rfft(transformed_xy, axis=-1)
    kx = _compact_neumann_wavenumber(
        case.nx, case.dx
    )[:, None, None]
    ky = _compact_neumann_wavenumber(
        case.ny, case.dy
    )[None, :, None]
    kz = _compact_rfft_wavenumber(
        case.nz, case.dz
    )[None, None, :]
    k_squared = kx * kx + ky * ky + kz * kz
    potential_hat = jnp.where(
        k_squared > 1.0e-10,
        -transformed / k_squared,
        0.0,
    )
    potential_xy = jnp.fft.irfft(
        potential_hat,
        n=case.nz,
        axis=-1,
    )
    return jsp_fft.idctn(
        potential_xy,
        type=2,
        axes=(-3, -2),
        norm="ortho",
    )


def build_flow_functions(
    case: CompactMoleCase,
    periodic_x: bool,
    include_turbines: bool,
) -> dict[str, Any]:
    if case.compact_solver not in {
        "dense_inverse",
        "tridiagonal",
        "cute_fused_41",
    }:
        raise ValueError(f"unknown compact solver: {case.compact_solver}")
    if case.staggered_y_solver not in {"dense", "cute_fused_41"}:
        raise ValueError(
            f"unknown staggered y solver: {case.staggered_y_solver}"
        )
    if case.rhs_assembly not in {
        "tensor",
        "componentwise",
        "componentwise_wall",
    }:
        raise ValueError(f"unknown RHS assembly: {case.rhs_assembly}")
    if case.filter_solver not in {"dense", "tridiagonal"}:
        raise ValueError(f"unknown filter solver: {case.filter_solver}")
    if case.pressure_projection not in {
        "incompact3d_staggered",
        "colocated_legacy",
    }:
        raise ValueError(
            f"unknown pressure projection: {case.pressure_projection}"
        )
    if case.pressure_fft_mode not in {"sequential", "fftn"}:
        raise ValueError(f"unknown pressure FFT mode: {case.pressure_fft_mode}")
    x = jnp.arange(case.nx, dtype=JAX_REAL_DTYPE) * NP_REAL_DTYPE(case.dx)
    y = jnp.arange(case.ny, dtype=JAX_REAL_DTYPE) * NP_REAL_DTYPE(case.dy)
    z = jnp.arange(case.nz, dtype=JAX_REAL_DTYPE) * NP_REAL_DTYPE(case.dz)
    x_grid = x[None, None, :, None, None]
    y_grid = y[None, None, None, :, None]
    z_grid = z[None, None, None, None, :]

    raw_profile = (
        NP_REAL_DTYPE(case.friction_velocity / case.von_karman)
        * jnp.log(
            (y + NP_REAL_DTYPE(case.roughness_length))
            / NP_REAL_DTYPE(case.roughness_length)
        )
    )
    streamwise_profile = raw_profile
    base_velocity = jnp.zeros(
        (1, 3, case.nx, case.ny, case.nz),
        dtype=JAX_REAL_DTYPE,
    )
    base_velocity = base_velocity.at[:, 0].set(
        streamwise_profile[None, None, :, None]
    )

    filter_width = NP_REAL_DTYPE(
        (case.dx * case.dy * case.dz) ** (1.0 / 3.0)
    )
    wall_coordinate = y + NP_REAL_DTYPE(case.roughness_length)
    wall_scale = (
        NP_REAL_DTYPE(case.von_karman)
        * wall_coordinate
        / filter_width
    )
    damping_power = NP_REAL_DTYPE(case.wall_damping_power)
    smagorinsky = (
        NP_REAL_DTYPE(case.smagorinsky_constant) ** (-damping_power)
        + wall_scale ** (-damping_power)
    ) ** (-1.0 / damping_power)
    smagorinsky = smagorinsky[None, None, None, :, None]

    cell_volume = NP_REAL_DTYPE(case.dx * case.dy * case.dz)
    disk_area = NP_REAL_DTYPE(np.pi * case.diameter**2 / 4.0)
    ct_prime = NP_REAL_DTYPE(
        case.thrust_coefficient
        / (1.0 - case.induction_factor) ** 2
    )
    relaxation_alpha = NP_REAL_DTYPE(
        (case.dt / case.relaxation_time)
        / (1.0 + case.dt / case.relaxation_time)
    )
    profile_relaxation = NP_REAL_DTYPE(
        case.dt / case.precursor_profile_relaxation_seconds
    )
    pressure_gradient_acceleration = NP_REAL_DTYPE(
        case.friction_velocity**2 / case.boundary_layer_height
        if case.pressure_gradient_forcing
        else 0.0
    )

    turbine_positions_np = np.asarray(
        case.turbine_positions_m, dtype=NP_REAL_DTYPE
    )
    if turbine_positions_np.shape != (case.num_turbines, 3):
        raise ValueError("turbine positions must have shape (N, 3)")
    if not np.all(np.isfinite(turbine_positions_np)):
        raise ValueError("turbine positions must be finite")
    turbine_positions = jnp.asarray(
        turbine_positions_np, dtype=JAX_REAL_DTYPE
    )
    turbine_x = turbine_positions[:, 0]
    turbine_y = turbine_positions[:, 1]
    turbine_z = turbine_positions[:, 2]

    sensor_x_relative, sensor_z_relative = np.meshgrid(
        np.linspace(-2.0, 3.0, 11),
        np.linspace(-1.0, 1.0, 7),
    )
    sensor_x = turbine_positions_np[:, 0, None] + (
        sensor_x_relative.reshape(1, -1) * case.diameter
    )
    sensor_y = np.broadcast_to(
        turbine_positions_np[:, 1, None], sensor_x.shape
    )
    sensor_z = turbine_positions_np[:, 2, None] + (
        sensor_z_relative.reshape(1, -1) * case.diameter
    )
    sensor_x_index_np = np.rint(sensor_x / case.dx).astype(np.int32)
    sensor_y_index_np = np.rint(sensor_y / case.dy).astype(np.int32)
    sensor_z_index_np = np.rint(sensor_z / case.dz).astype(np.int32)
    if np.any((sensor_x_index_np < 0) | (sensor_x_index_np >= case.nx)):
        raise ValueError("streamwise probe index crosses the LES domain")
    if np.any((sensor_y_index_np < 0) | (sensor_y_index_np >= case.ny)):
        raise ValueError("vertical probe index crosses the LES domain")
    if np.any((sensor_z_index_np < 0) | (sensor_z_index_np >= case.nz)):
        raise ValueError("spanwise probe index crosses the LES domain")
    for turbine_index in range(case.num_turbines):
        discrete_indices = set(
            zip(
                sensor_x_index_np[turbine_index].tolist(),
                sensor_y_index_np[turbine_index].tolist(),
                sensor_z_index_np[turbine_index].tolist(),
                strict=True,
            )
        )
        if len(discrete_indices) != 77:
            raise ValueError(
                f"turbine {turbine_index} has duplicate probe indices"
            )
    sensor_x_index = jnp.asarray(sensor_x_index_np.reshape(-1))
    sensor_y_index = jnp.asarray(sensor_y_index_np.reshape(-1))
    sensor_z_index = jnp.asarray(sensor_z_index_np.reshape(-1))

    def derivative(values: Array, direction: int) -> Array:
        axes = (-3, -2, -1)
        spacings = (case.dx, case.dy, case.dz)
        periodic = (periodic_x, False, True)
        return compact_first_derivative(
            values,
            axes[direction],
            spacings[direction],
            periodic[direction],
            case.compact_solver,
        )

    use_staggered_pressure = (
        case.pressure_projection == "incompact3d_staggered"
    )

    def divergence(velocity: Array) -> Array:
        if use_staggered_pressure:
            if periodic_x:
                return periodic_x_staggered_divergence(
                    velocity,
                    case.dx,
                    case.dy,
                    case.dz,
                    implementation=case.staggered_y_solver,
                )
            return staggered_divergence(
                velocity, case.dx, case.dy, case.dz
            )
        return sum(
            derivative(velocity[:, component], component)
            for component in range(3)
        )

    def project_with_boundary_history(
        velocity: Array,
        inlet: Array | None,
    ) -> tuple[Array, Array]:
        if use_staggered_pressure:
            corrected = velocity
            total_gradient = jnp.zeros_like(velocity)
            # FP32 iterative refinement recovers the residual lost in the
            # dense VP/PV applications without enabling slow consumer-GPU
            # FP64 throughout the LES.
            for _ in range(case.pressure_corrections):
                div = divergence(corrected)
                if periodic_x:
                    pressure = solve_periodic_x_pressure_poisson(
                        div,
                        case.dx,
                        case.dy,
                        case.dz,
                        implementation=case.pressure_fft_mode,
                    )
                    gradient = periodic_x_staggered_gradient(
                        pressure,
                        case.dx,
                        case.dy,
                        case.dz,
                        implementation=case.staggered_y_solver,
                    )
                else:
                    pressure = solve_pressure_poisson(
                        div, case.dx, case.dy, case.dz
                    )
                    gradient = staggered_gradient(
                        pressure, case.dx, case.dy, case.dz
                    )
                corrected = corrected - gradient
                total_gradient = total_gradient + gradient
            return corrected, total_gradient

        corrected = velocity
        total_gradient = jnp.zeros_like(velocity)
        for _ in range(case.pressure_corrections):
            div = divergence(corrected)
            potential = _mixed_pressure_solver(div, case, periodic_x)
            gradient = jnp.stack(
                tuple(derivative(potential, i) for i in range(3)),
                axis=1,
            )
            corrected = corrected - gradient
            total_gradient = total_gradient + gradient
            corrected = _apply_velocity_boundaries(
                corrected, inlet, periodic_x
            )
        return corrected, total_gradient

    def project(velocity: Array, inlet: Array | None) -> Array:
        corrected, _ = project_with_boundary_history(velocity, inlet)
        return corrected

    def disk_weights(yaw_degrees: Array) -> Array:
        yaw = jnp.deg2rad(yaw_degrees)[:, :, None, None, None]
        normal_x = jnp.cos(yaw)
        normal_z = -jnp.sin(yaw)
        delta_x = x_grid - turbine_x[None, :, None, None, None]
        delta_y = y_grid - turbine_y[None, :, None, None, None]
        delta_z = z_grid - turbine_z[None, :, None, None, None]
        delta_z = jnp.mod(
            delta_z + NP_REAL_DTYPE(0.5 * case.lz),
            NP_REAL_DTYPE(case.lz),
        ) - NP_REAL_DTYPE(0.5 * case.lz)
        normal_distance = delta_x * normal_x + delta_z * normal_z
        projected_x = delta_x - normal_distance * normal_x
        projected_z = delta_z - normal_distance * normal_z
        radial_distance = jnp.sqrt(
            projected_x**2 + delta_y**2 + projected_z**2
        )
        grid_normal_spacing = jnp.sqrt(
            (NP_REAL_DTYPE(case.dx) * normal_x) ** 2
            + (NP_REAL_DTYPE(case.dz) * normal_z) ** 2
        )
        disk_thickness = jnp.maximum(
            NP_REAL_DTYPE(case.diameter / 8.0),
            1.5 * grid_normal_spacing,
        )
        weights = jnp.exp(
            -(normal_distance / (0.5 * disk_thickness)) ** 2
            -(radial_distance / NP_REAL_DTYPE(0.5 * case.diameter)) ** 8
        )
        return weights / jnp.sum(
            weights, axis=(-3, -2, -1), keepdims=True
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
            weights * normal_velocity, axis=(-3, -2, -1)
        )
        relaxed_speed = (
            relaxation_alpha * disk_speed
            + (1.0 - relaxation_alpha) * previous_filtered_speed
        )
        filtered_speed = jnp.where(
            first_step, disk_speed, relaxed_speed
        )
        thrust = (
            NP_REAL_DTYPE(0.5 * case.density)
            * ct_prime
            * filtered_speed**2
            * disk_area
        )
        power = thrust * filtered_speed
        acceleration = -jnp.sum(
            (
                thrust[:, :, None, None, None, None]
                / NP_REAL_DTYPE(case.density * cell_volume)
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
        turbine_terms: tuple[Array, Array, Array] | None = None,
    ) -> tuple[Array, Array, Array]:
        directional_gradients = tuple(
            derivative(velocity, direction) for direction in range(3)
        )
        if case.rhs_assembly == "tensor":
            gradient = jnp.stack(directional_gradients, axis=2)
            advective = jnp.sum(
                velocity[:, None] * gradient, axis=2
            )
        else:
            advective = sum(
                velocity[:, direction : direction + 1]
                * directional_gradients[direction]
                for direction in range(3)
            )
        conservative = sum(
            derivative(
                velocity * velocity[:, direction : direction + 1],
                direction,
            )
            for direction in range(3)
        )
        advection = -0.5 * (advective + conservative)

        if case.rhs_assembly == "tensor":
            strain = 0.5 * (
                gradient + jnp.swapaxes(gradient, 1, 2)
            )
            strain_magnitude = jnp.sqrt(
                2.0
                * jnp.sum(
                    strain * strain, axis=(1, 2), keepdims=True
                )
                + 1.0e-20
            )
            eddy_viscosity = (
                (smagorinsky * filter_width) ** 2
                * strain_magnitude
            )
            stress = (
                2.0
                * (
                    NP_REAL_DTYPE(case.molecular_viscosity)
                    + eddy_viscosity
                )
                * strain
            )
            if case.wall_sgs_model:
                wall_delta = NP_REAL_DTYPE(case.wall_sampling_dy * case.dy)
                wall_index = int(math.floor(case.wall_sampling_dy))
                wall_fraction = NP_REAL_DTYPE(
                    case.wall_sampling_dy - wall_index
                )
                sampled = (
                    (1.0 - wall_fraction)
                    * velocity[:, :, :, wall_index, :]
                    + wall_fraction
                    * velocity[:, :, :, wall_index + 1, :]
                )
                wall_speed = jnp.sqrt(
                    sampled[:, 0] ** 2 + sampled[:, 2] ** 2
                )
                wall_factor = NP_REAL_DTYPE(
                    (
                        case.von_karman
                        / math.log(wall_delta / case.roughness_length)
                    )
                    ** 2
                )
                tau_x = -wall_factor * sampled[:, 0] * wall_speed
                tau_z = -wall_factor * sampled[:, 2] * wall_speed
                molecular_stress = (
                    NP_REAL_DTYPE(2.0 * case.molecular_viscosity)
                    * strain[:, :, :, :, 1, :]
                )
                stress = stress.at[:, :, :, :, 1, :].set(
                    molecular_stress
                )
                stress = stress.at[:, 0, 1, :, 1, :].add(-tau_x)
                stress = stress.at[:, 1, 0, :, 1, :].add(-tau_x)
                stress = stress.at[:, 2, 1, :, 1, :].add(-tau_z)
                stress = stress.at[:, 1, 2, :, 1, :].add(-tau_z)
            diffusion = sum(
                derivative(stress[:, :, direction], direction)
                for direction in range(3)
            )
        else:
            strain_squared = sum(
                directional_gradients[component][
                    :, component : component + 1
                ]
                ** 2
                for component in range(3)
            )
            for first in range(3):
                for second in range(first + 1, 3):
                    off_diagonal = NP_REAL_DTYPE(0.5) * (
                        directional_gradients[second][
                            :, first : first + 1
                        ]
                        + directional_gradients[first][
                            :, second : second + 1
                        ]
                    )
                    strain_squared = (
                        strain_squared
                        + NP_REAL_DTYPE(2.0) * off_diagonal**2
                    )
            strain_magnitude = jnp.sqrt(
                NP_REAL_DTYPE(2.0) * strain_squared + 1.0e-20
            )
            eddy_viscosity = (
                (smagorinsky[:, 0] * filter_width) ** 2
                * strain_magnitude
            )
            effective_viscosity = (
                NP_REAL_DTYPE(case.molecular_viscosity)
                + eddy_viscosity
            )

            wall_terms = None
            if (
                case.wall_sgs_model
                and case.rhs_assembly == "componentwise_wall"
            ):
                wall_delta = NP_REAL_DTYPE(case.wall_sampling_dy * case.dy)
                wall_index = int(math.floor(case.wall_sampling_dy))
                wall_fraction = NP_REAL_DTYPE(
                    case.wall_sampling_dy - wall_index
                )
                sampled = (
                    (1.0 - wall_fraction)
                    * velocity[:, :, :, wall_index, :]
                    + wall_fraction
                    * velocity[:, :, :, wall_index + 1, :]
                )
                wall_speed = jnp.sqrt(
                    sampled[:, 0] ** 2 + sampled[:, 2] ** 2
                )
                wall_factor = NP_REAL_DTYPE(
                    (
                        case.von_karman
                        / math.log(wall_delta / case.roughness_length)
                    )
                    ** 2
                )
                wall_terms = (
                    -wall_factor * sampled[:, 0] * wall_speed,
                    -wall_factor * sampled[:, 2] * wall_speed,
                )

            def stress_for_direction(direction: int) -> Array:
                transposed = jnp.stack(
                    tuple(
                        directional_gradients[component][:, direction]
                        for component in range(3)
                    ),
                    axis=1,
                )
                stress_component = effective_viscosity * (
                    directional_gradients[direction] + transposed
                )
                if wall_terms is None:
                    return stress_component
                molecular_stress = (
                    NP_REAL_DTYPE(case.molecular_viscosity)
                    * (directional_gradients[direction] + transposed)
                )
                stress_component = stress_component.at[:, :, :, 1, :].set(
                    molecular_stress[:, :, :, 1, :]
                )
                tau_x, tau_z = wall_terms
                if direction == 0:
                    stress_component = stress_component.at[:, 1, :, 1, :].add(
                        -tau_x
                    )
                elif direction == 1:
                    stress_component = stress_component.at[:, 0, :, 1, :].add(
                        -tau_x
                    )
                    stress_component = stress_component.at[:, 2, :, 1, :].add(
                        -tau_z
                    )
                else:
                    stress_component = stress_component.at[:, 1, :, 1, :].add(
                        -tau_z
                    )
                return stress_component

            diffusion = sum(
                derivative(
                    stress_for_direction(direction), direction
                )
                for direction in range(3)
            )
        if include_turbines:
            if turbine_terms is None:
                acceleration, filtered_speed, power, _ = disk_source(
                    velocity,
                    yaw_degrees,
                    weights,
                    previous_filtered_speed,
                    first_step,
                )
            else:
                acceleration, filtered_speed, power = turbine_terms
        else:
            acceleration = jnp.zeros_like(velocity)
            if case.profile_relaxation_forcing:
                horizontal_mean = jnp.mean(
                    velocity[:, 0], axis=(-3, -1), keepdims=True
                )
                target = base_velocity[:, 0, :1]
                streamwise_force = profile_relaxation * (
                    target - horizontal_mean
                )
                acceleration = acceleration.at[:, 0].set(
                    jnp.broadcast_to(
                        streamwise_force,
                        velocity[:, 0].shape,
                    )
                )
            filtered_speed = previous_filtered_speed
            power = jnp.zeros(
                (case.batch, case.num_turbines), dtype=velocity.dtype
            )
        acceleration = acceleration.at[:, 0].add(
            pressure_gradient_acceleration
        )
        return advection + diffusion + acceleration, filtered_speed, power

    def advance(
        state: FlowState,
        yaw_degrees: Array,
        weights: Array,
        inlet: Array | None,
    ) -> tuple[FlowState, Array]:
        velocity = state.velocity
        turbine_terms = None
        if include_turbines:
            acceleration, filtered_speed, power, _ = disk_source(
                velocity,
                yaw_degrees,
                weights,
                state.filtered_disk_speed,
                state.step_index == 0,
            )
            turbine_terms = (acceleration, filtered_speed, power)
        if case.spatial_filter and not periodic_x:
            if case.farm_filter_all_directions:
                velocity = mole_farm_all_direction_filter(
                    velocity, case.filter_coefficient, case.filter_solver
                )
            else:
                velocity = mole_farm_spatial_filter(
                    velocity, case.filter_coefficient, case.filter_solver
                )
        elif case.spatial_filter and periodic_x:
            velocity = mole_precursor_spatial_filter(
                velocity, case.filter_coefficient
            )
        if periodic_x and case.shifted_periodic_fringe:
            velocity = _apply_shifted_periodic_fringe(velocity, case)
        rhs, filtered_speed, power = flow_rhs(
            velocity,
            yaw_degrees,
            weights,
            state.filtered_disk_speed,
            state.step_index == 0,
            turbine_terms,
        )
        euler = velocity + NP_REAL_DTYPE(case.dt) * rhs
        ab2 = velocity + NP_REAL_DTYPE(case.dt) * (
            1.5 * rhs - 0.5 * state.rhs_previous
        )
        ab3 = velocity + NP_REAL_DTYPE(case.dt) * (
            NP_REAL_DTYPE(23.0 / 12.0) * rhs
            - NP_REAL_DTYPE(16.0 / 12.0) * state.rhs_previous
            + NP_REAL_DTYPE(5.0 / 12.0) * state.rhs_previous_2
        )
        predicted = jnp.where(
            state.step_index == 0,
            euler,
            jnp.where(state.step_index == 1, ab2, ab3),
        )
        # Case-ABL computes bxxn/bxyn/bxzn from the filtered old velocity
        # before int_time, then pre_correc applies those stored planes.
        outflow_speed = NP_REAL_DTYPE(0.5) * (
            jnp.max(velocity[:, 0, -2], axis=(-2, -1))
            + jnp.min(velocity[:, 0, -2], axis=(-2, -1))
        )
        outflow_courant = (
            outflow_speed[:, None, None, None]
            * NP_REAL_DTYPE(case.dt / case.dx)
        )
        outflow_boundary = velocity[:, :, -1] - outflow_courant * (
            velocity[:, :, -1] - velocity[:, :, -2]
        )
        if inlet is not None:
            inlet_streamwise_mean = jnp.mean(
                inlet[:, 0], axis=(-2, -1), keepdims=True
            )
            outlet_streamwise_mean = jnp.mean(
                outflow_boundary[:, 0], axis=(-2, -1), keepdims=True
            )
            outflow_boundary = outflow_boundary.at[:, 0].add(
                inlet_streamwise_mean - outlet_streamwise_mean
            )
        predicted = _apply_velocity_boundaries(
            predicted,
            inlet,
            periodic_x,
            (
                outflow_boundary
                if case.convective_outflow and not periodic_x
                else None
            ),
        )
        if not periodic_x:
            predicted = predicted.at[:, 1, 0].add(
                state.open_x_pressure_gradient[:, 0, 0]
            )
            predicted = predicted.at[:, 2, 0].add(
                state.open_x_pressure_gradient[:, 0, 1]
            )
            predicted = predicted.at[:, 1, -1].add(
                state.open_x_pressure_gradient[:, 1, 0]
            )
            predicted = predicted.at[:, 2, -1].add(
                state.open_x_pressure_gradient[:, 1, 1]
            )
        # XCompact3D applies the y-boundary pre-correction after x, so the
        # bottom-wall values overwrite corner contributions from open x.
        predicted = predicted.at[:, 0, :, 0].set(
            state.bottom_pressure_gradient[:, 0]
        )
        predicted = predicted.at[:, 1, :, 0].set(0.0)
        predicted = predicted.at[:, 2, :, 0].set(
            state.bottom_pressure_gradient[:, 1]
        )
        updated, pressure_gradient = project_with_boundary_history(
            predicted, inlet
        )
        bottom_pressure_gradient = jnp.stack(
            (
                pressure_gradient[:, 0, :, 0],
                pressure_gradient[:, 2, :, 0],
            ),
            axis=1,
        )
        if periodic_x:
            open_x_pressure_gradient = state.open_x_pressure_gradient
        else:
            open_x_pressure_gradient = jnp.stack(
                (
                    jnp.stack(
                        (
                            pressure_gradient[:, 1, 0],
                            pressure_gradient[:, 2, 0],
                        ),
                        axis=1,
                    ),
                    jnp.stack(
                        (
                            pressure_gradient[:, 1, -1],
                            pressure_gradient[:, 2, -1],
                        ),
                        axis=1,
                    ),
                ),
                axis=1,
            )
        next_state = FlowState(
            velocity=updated,
            rhs_previous=rhs,
            rhs_previous_2=state.rhs_previous,
            filtered_disk_speed=filtered_speed,
            bottom_pressure_gradient=bottom_pressure_gradient,
            open_x_pressure_gradient=open_x_pressure_gradient,
            step_index=state.step_index + 1,
        )
        return next_state, power

    def probes(velocity: Array) -> Array:
        values = velocity[
            :,
            0,
            sensor_x_index,
            sensor_y_index,
            sensor_z_index,
        ]
        return values.reshape(case.batch, case.num_turbines, 77)

    return {
        "base_velocity": base_velocity,
        "derivative": derivative,
        "divergence": divergence,
        "diagnostic_divergence": (
            remove_reference_divergence
            if use_staggered_pressure and not periodic_x
            else lambda value: value
        ),
        "pressure_projection": (
            (
                "incompact3d_staggered_periodic_x"
                if periodic_x
                else "incompact3d_staggered"
            )
            if use_staggered_pressure
            else "colocated_legacy"
        ),
        "case": case,
        "project": project,
        "project_with_boundary_history": project_with_boundary_history,
        "disk_weights": disk_weights,
        "disk_source": disk_source,
        "flow_rhs": flow_rhs,
        "advance": advance,
        "probes": probes,
    }


def make_initial_flow(
    case: CompactMoleCase,
    functions: dict[str, Any],
    seed: int,
    precursor_velocity: Array | None = None,
) -> FlowState:
    if precursor_velocity is None:
        if case.initialize_with_log_profile:
            base = jnp.broadcast_to(
                functions["base_velocity"],
                (case.batch, 3, case.nx, case.ny, case.nz),
            )
        else:
            base = jnp.zeros(
                (case.batch, 3, case.nx, case.ny, case.nz),
                dtype=JAX_REAL_DTYPE,
            )
        if case.source_aligned_initial_noise:
            noise = NP_REAL_DTYPE(case.precursor_initial_ti) * jax.random.uniform(
                jax.random.PRNGKey(seed),
                base.shape,
                dtype=JAX_REAL_DTYPE,
                minval=-1.0,
                maxval=1.0,
            )
            velocity = base.at[:, 0].set(
                base[:, 0] * (1.0 + noise[:, 0])
            )
            velocity = velocity.at[:, 1].set(noise[:, 1])
            velocity = velocity.at[:, 2].set(noise[:, 2])
        else:
            perturbation = (
                NP_REAL_DTYPE(
                    case.inflow_hub_speed * case.precursor_initial_ti
                )
                * jax.random.normal(
                    jax.random.PRNGKey(seed),
                    base.shape,
                    dtype=JAX_REAL_DTYPE,
                )
            )
            velocity = base + perturbation
        velocity = _apply_velocity_boundaries(
            velocity, None, periodic_x=True
        )
        velocity = functions["project"](velocity, None)
    else:
        velocity = precursor_velocity
    return FlowState(
        velocity=velocity,
        # Public Incompact3d initializes every AB3 RHS history layer with
        # the initial velocity. The startup Euler/AB2 branches do not use
        # these placeholders, but preserving them makes restart state exact.
        rhs_previous=jnp.copy(velocity),
        rhs_previous_2=jnp.copy(velocity),
        filtered_disk_speed=jnp.zeros(
            (case.batch, case.num_turbines), dtype=JAX_REAL_DTYPE
        ),
        bottom_pressure_gradient=jnp.zeros(
            (case.batch, 2, case.nx, case.nz), dtype=JAX_REAL_DTYPE
        ),
        open_x_pressure_gradient=jnp.zeros(
            (case.batch, 2, 2, case.ny, case.nz), dtype=JAX_REAL_DTYPE
        ),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def _precursor_inlet(
    precursor_velocity: Array,
    absolute_step: Array,
    case: CompactMoleCase,
) -> Array:
    precursor_nx = precursor_velocity.shape[2]
    displacement = (
        absolute_step.astype(JAX_REAL_DTYPE)
        * NP_REAL_DTYPE(
            case.precursor_advection_speed * case.dt / case.precursor_dx
        )
    )
    left = jnp.floor(displacement).astype(jnp.int32) % precursor_nx
    right = (left + 1) % precursor_nx
    fraction = displacement - jnp.floor(displacement)
    left_plane = jax.lax.dynamic_index_in_dim(
        precursor_velocity, left, axis=2, keepdims=False
    )
    right_plane = jax.lax.dynamic_index_in_dim(
        precursor_velocity, right, axis=2, keepdims=False
    )
    return (1.0 - fraction) * left_plane + fraction * right_plane


def generate_precursor(
    case: CompactMoleCase,
    seed: int,
    spinup_steps: int,
) -> tuple[Array, dict[str, float]]:
    functions = build_flow_functions(
        case, periodic_x=True, include_turbines=False
    )
    state = make_initial_flow(case, functions, seed)
    zero_yaw = jnp.zeros(
        (case.batch, case.num_turbines), dtype=JAX_REAL_DTYPE
    )
    zero_weights = jnp.zeros(
        (
            case.batch,
            case.num_turbines,
            case.nx,
            case.ny,
            case.nz,
        ),
        dtype=JAX_REAL_DTYPE,
    )

    def body(current: FlowState, _: None) -> tuple[FlowState, None]:
        next_state, _ = functions["advance"](
            current, zero_yaw, zero_weights, None
        )
        return next_state, None

    run = jax.jit(
        lambda initial: jax.lax.scan(
            body, initial, xs=None, length=spinup_steps
        )[0]
    )
    final_state = run(state)
    final_state.velocity.block_until_ready()
    velocity = final_state.velocity
    hub_index = int(round(case.hub_height / case.dy))
    hub_u = velocity[:, 0, :, hub_index, :]
    mean_hub_speed = jnp.mean(hub_u, axis=(-2, -1))
    turbulence_intensity = (
        jnp.std(hub_u, axis=(-2, -1))
        / jnp.maximum(jnp.abs(mean_hub_speed), 1.0e-6)
    )
    divergence = functions["divergence"](velocity)
    metrics = {
        "mean_hub_speed_m_per_s": float(jnp.mean(mean_hub_speed)),
        "mean_hub_turbulence_intensity": float(
            jnp.mean(turbulence_intensity)
        ),
        "min_hub_turbulence_intensity": float(
            jnp.min(turbulence_intensity)
        ),
        "velocity_standard_deviation_m_per_s": float(
            jnp.std(velocity)
        ),
        "interior_divergence_rms": float(
            jnp.sqrt(
                jnp.mean(divergence[:, 3:-3, 3:-3] ** 2)
            )
        ),
        "finite": bool(jnp.all(jnp.isfinite(velocity))),
    }
    return velocity, metrics


def init_mlp(
    key: Array,
    dimensions: tuple[int, ...],
    initialization: str = "xavier_uniform",
) -> tuple[dict[str, Array], ...]:
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(
        keys, dimensions[:-1], dimensions[1:], strict=True
    ):
        if initialization == "xavier_uniform":
            limit = NP_REAL_DTYPE(math.sqrt(6.0 / (fan_in + fan_out)))
        elif initialization == "torch_linear":
            # torch.nn.Linear.reset_parameters uses Kaiming uniform with
            # a=sqrt(5), which reduces to this bound.
            limit = NP_REAL_DTYPE(1.0 / math.sqrt(fan_in))
        else:
            raise ValueError(f"Unknown MLP initialization: {initialization}")
        if initialization == "torch_linear":
            weight_key, bias_key = jax.random.split(layer_key)
            bias = jax.random.uniform(
                bias_key,
                (fan_out,),
                minval=-limit,
                maxval=limit,
                dtype=JAX_REAL_DTYPE,
            )
        else:
            weight_key = layer_key
            bias = jnp.zeros((fan_out,), dtype=JAX_REAL_DTYPE)
        layers.append(
            {
                "w": jax.random.uniform(
                    weight_key,
                    (fan_in, fan_out),
                    minval=-limit,
                    maxval=limit,
                    dtype=JAX_REAL_DTYPE,
                ),
                "b": bias,
            }
        )
    return tuple(layers)


def mlp_apply(
    parameters: tuple[dict[str, Array], ...],
    inputs: Array,
    activate_last: bool = False,
    activation: str = "silu",
) -> Array:
    output = inputs
    for index, layer in enumerate(parameters):
        output = output @ layer["w"] + layer["b"]
        if index < len(parameters) - 1 or activate_last:
            if activation == "silu":
                output = jax.nn.silu(output)
            elif activation == "relu":
                output = jax.nn.relu(output)
            else:
                raise ValueError(f"Unknown MLP activation: {activation}")
    return output


def torchrl_biased_softplus(
    values: Array,
    bias: float = 1.0,
    minimum: float = 0.01,
    scale_lower_bound: float = 0.1,
) -> Array:
    """Reproduce TensorDict's ``biased_softplus_1.0`` scale mapping."""

    if bias <= minimum:
        raise ValueError("bias must be greater than minimum")
    shift = NP_REAL_DTYPE(math.log(math.expm1(bias - minimum)))
    scale = jax.nn.softplus(values + shift) + NP_REAL_DTYPE(minimum)
    return jnp.maximum(scale, NP_REAL_DTYPE(scale_lower_bound))


def actor_sample(
    actor: PyTree,
    observations: Array,
    key: Array,
    action_bound: float,
    policy_mode: str = "legacy",
) -> tuple[Array, Array]:
    if action_bound <= 0.0:
        raise ValueError("action_bound must be positive")
    activation = "relu" if policy_mode == "public_torchrl" else "silu"
    output = mlp_apply(actor, observations, activation=activation)
    mean, raw_scale = jnp.split(output, 2, axis=-1)
    if policy_mode == "public_torchrl":
        scale = torchrl_biased_softplus(raw_scale)
        log_scale = jnp.log(scale)
    elif policy_mode == "legacy":
        log_scale = jnp.clip(raw_scale, -5.0, 1.0)
        scale = jnp.exp(log_scale)
    else:
        raise ValueError(f"Unknown SAC policy mode: {policy_mode}")
    noise = jax.random.normal(key, mean.shape)
    pre_tanh = mean + scale * noise
    # TorchRL 0.3.1 TanhNormal uses SafeTanhTransform.  In float32 it
    # clamps the transformed action to +/- (1 - 1e-6), then evaluates the
    # density through the finite inverse transform.
    resolution = NP_REAL_DTYPE(np.finfo(NP_REAL_DTYPE).resolution)
    limit = NP_REAL_DTYPE(1.0) - resolution
    normalized = jnp.clip(jnp.tanh(pre_tanh), -limit, limit)
    actions = NP_REAL_DTYPE(action_bound) * normalized
    inverse = jnp.arctanh(normalized)
    inverse_limit = jnp.arctanh(limit)
    density_coordinate = jnp.where(
        jnp.isfinite(pre_tanh) & (jnp.abs(pre_tanh) <= inverse_limit),
        pre_tanh,
        inverse,
    )
    standardized = (density_coordinate - mean) / scale
    gaussian_log_probability = jnp.sum(
        -0.5
        * (
            standardized**2
            + NP_REAL_DTYPE(math.log(2.0 * math.pi))
        )
        - log_scale,
        axis=-1,
    )
    # Stable log-Jacobian of tanh at the finite inverse transform.
    correction = jnp.sum(
        2.0
        * (
            NP_REAL_DTYPE(math.log(2.0))
            - density_coordinate
            - jax.nn.softplus(-2.0 * density_coordinate)
        )
        + NP_REAL_DTYPE(math.log(action_bound)),
        axis=-1,
    )
    return actions, gaussian_log_probability - correction


def critic_apply(
    critic: PyTree,
    observations: Array,
    actions: Array,
    policy_mode: str = "legacy",
) -> Array:
    inputs = jnp.concatenate(
        (
            (actions, observations)
            if policy_mode == "public_torchrl"
            else (observations, actions)
        ),
        axis=-1,
    )
    activation = "relu" if policy_mode == "public_torchrl" else "silu"
    return jnp.stack(
        (
            mlp_apply(
                critic["q1"], inputs, activation=activation
            ).squeeze(-1),
            mlp_apply(
                critic["q2"], inputs, activation=activation
            ).squeeze(-1),
        ),
        axis=-1,
    )


def init_sac(
    key: Array,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    initial_alpha: float = 0.2,
    policy_mode: str = "legacy",
    entropy_learning_rate: float | None = None,
    actor_learning_rate_scale: float = 1.0,
) -> tuple[SACState, tuple[Any, Any, Any]]:
    actor_key, q1_key, q2_key = jax.random.split(key, 3)
    initialization = (
        "torch_linear"
        if policy_mode == "public_torchrl"
        else "xavier_uniform"
    )
    actor = init_mlp(
        actor_key,
        (observation_dim, hidden_dim, hidden_dim, 2 * action_dim),
        initialization,
    )
    critic = {
        "q1": init_mlp(
            q1_key,
            (observation_dim + action_dim, hidden_dim, hidden_dim, 1),
            initialization,
        ),
        "q2": init_mlp(
            q2_key,
            (observation_dim + action_dim, hidden_dim, hidden_dim, 1),
            initialization,
        ),
    }
    target_critic = jax.tree.map(jnp.copy, critic)
    if actor_learning_rate_scale <= 0.0:
        raise ValueError("actor_learning_rate_scale must be positive")
    actor_optimizer = optax.adam(
        learning_rate * actor_learning_rate_scale
    )
    critic_optimizer = optax.adam(learning_rate)
    alpha_optimizer = optax.adam(
        learning_rate
        if entropy_learning_rate is None
        else entropy_learning_rate
    )
    if initial_alpha <= 0.0:
        raise ValueError("initial_alpha must be positive")
    log_alpha = jnp.asarray(math.log(initial_alpha), dtype=JAX_REAL_DTYPE)
    state = SACState(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        log_alpha=log_alpha,
        actor_opt=actor_optimizer.init(actor),
        critic_opt=critic_optimizer.init(critic),
        alpha_opt=alpha_optimizer.init(log_alpha),
        updates=jnp.asarray(0, dtype=jnp.int32),
    )
    return state, (
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
    )


def init_replay(
    capacity: int,
    observation_dim: int,
    action_dim: int,
) -> ReplayState:
    return ReplayState(
        observations=jnp.zeros(
            (capacity, observation_dim), dtype=JAX_REAL_DTYPE
        ),
        actions=jnp.zeros((capacity, action_dim), dtype=JAX_REAL_DTYPE),
        rewards=jnp.zeros((capacity,), dtype=JAX_REAL_DTYPE),
        next_observations=jnp.zeros(
            (capacity, observation_dim), dtype=JAX_REAL_DTYPE
        ),
        dones=jnp.zeros((capacity,), dtype=JAX_REAL_DTYPE),
        position=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def replay_add(
    replay: ReplayState,
    observations: Array,
    actions: Array,
    rewards: Array,
    next_observations: Array,
    dones: Array,
) -> ReplayState:
    capacity = replay.observations.shape[0]
    batch = observations.shape[0]
    indices = (
        replay.position + jnp.arange(batch, dtype=jnp.int32)
    ) % capacity
    return ReplayState(
        observations=replay.observations.at[indices].set(observations),
        actions=replay.actions.at[indices].set(actions),
        rewards=replay.rewards.at[indices].set(rewards),
        next_observations=replay.next_observations.at[indices].set(
            next_observations
        ),
        dones=replay.dones.at[indices].set(dones),
        position=(replay.position + batch) % capacity,
        size=jnp.minimum(replay.size + batch, capacity),
    )


def sac_update(
    state: SACState,
    replay: ReplayState,
    key: Array,
    optimizers: tuple[Any, Any, Any],
    sample_size: int,
    action_bound: float,
    gamma: float,
    tau: float,
    target_entropy: float,
    axis_name: str | None = None,
    policy_mode: str = "legacy",
    reward_scale: float = 1.0,
    actor_update_period: int = 1,
    action_l2_coefficient: float = 0.0,
) -> tuple[SACState, dict[str, Array]]:
    if actor_update_period < 0:
        raise ValueError("actor_update_period cannot be negative")
    if action_l2_coefficient < 0.0:
        raise ValueError("action_l2_coefficient cannot be negative")
    actor_optimizer, critic_optimizer, alpha_optimizer = optimizers
    sample_key, target_key, actor_key = jax.random.split(key, 3)
    indices = jax.random.randint(
        sample_key,
        (sample_size,),
        minval=0,
        maxval=jnp.maximum(replay.size, 1),
    )
    observations = replay.observations[indices]
    actions = replay.actions[indices]
    rewards = replay.rewards[indices]
    next_observations = replay.next_observations[indices]
    dones = replay.dones[indices]
    alpha = jnp.exp(state.log_alpha)

    next_actions, next_log_probability = actor_sample(
        state.actor,
        next_observations,
        target_key,
        action_bound,
        policy_mode,
    )
    target_q = jnp.min(
        critic_apply(
            state.target_critic,
            next_observations,
            next_actions,
            policy_mode,
        ),
        axis=-1,
    )
    target = jax.lax.stop_gradient(
        NP_REAL_DTYPE(reward_scale) * rewards
        + NP_REAL_DTYPE(gamma)
        * (1.0 - dones)
        * (target_q - alpha * next_log_probability)
    )

    def critic_loss_fn(critic: PyTree) -> Array:
        q_values = critic_apply(
            critic, observations, actions, policy_mode
        )
        squared_error = (q_values - target[:, None]) ** 2
        if policy_mode == "public_torchrl":
            return jnp.mean(jnp.sum(squared_error, axis=-1))
        return jnp.mean(squared_error)

    critic_loss, critic_gradient = jax.value_and_grad(
        critic_loss_fn
    )(state.critic)
    if axis_name is not None:
        critic_gradient = jax.lax.pmean(
            critic_gradient, axis_name=axis_name
        )
    critic_updates, critic_opt = critic_optimizer.update(
        critic_gradient, state.critic_opt, state.critic
    )
    critic = optax.apply_updates(state.critic, critic_updates)
    critic_gradient_norm = optax.global_norm(critic_gradient)

    def actor_loss_fn(
        actor: PyTree,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        sampled_actions, log_probability = actor_sample(
            actor,
            observations,
            actor_key,
            action_bound,
            policy_mode,
        )
        actor_loss_critic = (
            state.critic if policy_mode == "public_torchrl" else critic
        )
        q_min = jnp.min(
            critic_apply(
                actor_loss_critic,
                observations,
                sampled_actions,
                policy_mode,
            ),
            axis=-1,
        )
        entropy_term = jnp.mean(
            jax.lax.stop_gradient(alpha) * log_probability
        )
        action_l2_mean = jnp.mean(
            (sampled_actions / NP_REAL_DTYPE(action_bound)) ** 2
        )
        loss = (
            entropy_term
            - jnp.mean(q_min)
            + NP_REAL_DTYPE(action_l2_coefficient) * action_l2_mean
        )
        return loss, (
            log_probability,
            q_min,
            sampled_actions,
            action_l2_mean,
        )

    (
        actor_loss,
        (log_probability, actor_q, sampled_actions, action_l2_mean),
    ), actor_gradient = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(state.actor)
    if axis_name is not None:
        actor_gradient = jax.lax.pmean(
            actor_gradient, axis_name=axis_name
        )
    actor_gradient_norm = optax.global_norm(actor_gradient)
    actor_updates, proposed_actor_opt = actor_optimizer.update(
        actor_gradient, state.actor_opt, state.actor
    )
    proposed_actor = optax.apply_updates(state.actor, actor_updates)

    def alpha_loss_fn(log_alpha: Array) -> Array:
        return -jnp.mean(
            log_alpha
            * jax.lax.stop_gradient(
                log_probability + NP_REAL_DTYPE(target_entropy)
            )
        )

    alpha_loss, alpha_gradient = jax.value_and_grad(alpha_loss_fn)(
        state.log_alpha
    )
    if axis_name is not None:
        alpha_gradient = jax.lax.pmean(
            alpha_gradient, axis_name=axis_name
        )
    alpha_updates, proposed_alpha_opt = alpha_optimizer.update(
        alpha_gradient, state.alpha_opt, state.log_alpha
    )
    proposed_log_alpha = optax.apply_updates(
        state.log_alpha, alpha_updates
    )
    if actor_update_period == 0:
        actor_updated = jnp.asarray(False)
    else:
        actor_updated = (
            (state.updates + 1) % actor_update_period == 0
        )
    actor = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_actor,
        state.actor,
    )
    actor_opt = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_actor_opt,
        state.actor_opt,
    )
    log_alpha = jnp.where(
        actor_updated, proposed_log_alpha, state.log_alpha
    )
    alpha_opt = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_alpha_opt,
        state.alpha_opt,
    )
    target_critic = jax.tree.map(
        lambda target_value, online_value: (
            (1.0 - NP_REAL_DTYPE(tau)) * target_value
            + NP_REAL_DTYPE(tau) * online_value
        ),
        state.target_critic,
        critic,
    )
    next_state = SACState(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        log_alpha=log_alpha,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        alpha_opt=alpha_opt,
        updates=state.updates + 1,
    )
    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(log_alpha),
        "actor_updated": actor_updated.astype(jnp.int32),
        "actor_gradient_norm": actor_gradient_norm,
        "critic_gradient_norm": critic_gradient_norm,
        "target_q_mean": jnp.mean(target_q),
        "target_q_std": jnp.std(target_q),
        "actor_q_mean": jnp.mean(actor_q),
        "entropy_term_mean": jnp.mean(
            jax.lax.stop_gradient(alpha) * log_probability
        ),
        "action_l2_mean": action_l2_mean,
        "sampled_action_abs_mean": jnp.mean(jnp.abs(sampled_actions)),
        "sampled_action_abs_max": jnp.max(jnp.abs(sampled_actions)),
    }
    return next_state, metrics


def sac_update_entropy_scaled(
    state: SACState,
    replay: ReplayState,
    key: Array,
    optimizers: tuple[Any, Any, Any],
    sample_size: int,
    action_bound: float,
    gamma: float,
    tau: float,
    target_entropy: float,
    entropy_scale: float,
    axis_name: str | None = None,
    policy_mode: str = "legacy",
    reward_scale: float = 1.0,
    actor_update_period: int = 1,
    action_l2_coefficient: float = 0.0,
) -> tuple[SACState, dict[str, Array]]:
    """SAC update with one consistent multiplier on joint policy entropy."""
    if actor_update_period < 0:
        raise ValueError("actor_update_period cannot be negative")
    if action_l2_coefficient < 0.0:
        raise ValueError("action_l2_coefficient cannot be negative")
    if not 0.0 <= entropy_scale < 1.0:
        raise ValueError("scaled SAC requires 0 <= entropy_scale < 1")
    actor_optimizer, critic_optimizer, alpha_optimizer = optimizers
    sample_key, target_key, actor_key = jax.random.split(key, 3)
    indices = jax.random.randint(
        sample_key,
        (sample_size,),
        minval=0,
        maxval=jnp.maximum(replay.size, 1),
    )
    observations = replay.observations[indices]
    actions = replay.actions[indices]
    rewards = replay.rewards[indices]
    next_observations = replay.next_observations[indices]
    dones = replay.dones[indices]
    alpha = jnp.exp(state.log_alpha)

    next_actions, next_log_probability = actor_sample(
        state.actor,
        next_observations,
        target_key,
        action_bound,
        policy_mode,
    )
    target_q = jnp.min(
        critic_apply(
            state.target_critic,
            next_observations,
            next_actions,
            policy_mode,
        ),
        axis=-1,
    )
    target = jax.lax.stop_gradient(
        NP_REAL_DTYPE(reward_scale) * rewards
        + NP_REAL_DTYPE(gamma)
        * (1.0 - dones)
        * (
            target_q
            - alpha
            * NP_REAL_DTYPE(entropy_scale)
            * next_log_probability
        )
    )

    def critic_loss_fn(critic: PyTree) -> Array:
        q_values = critic_apply(
            critic, observations, actions, policy_mode
        )
        squared_error = (q_values - target[:, None]) ** 2
        if policy_mode == "public_torchrl":
            return jnp.mean(jnp.sum(squared_error, axis=-1))
        return jnp.mean(squared_error)

    critic_loss, critic_gradient = jax.value_and_grad(
        critic_loss_fn
    )(state.critic)
    if axis_name is not None:
        critic_gradient = jax.lax.pmean(
            critic_gradient, axis_name=axis_name
        )
    critic_updates, critic_opt = critic_optimizer.update(
        critic_gradient, state.critic_opt, state.critic
    )
    critic = optax.apply_updates(state.critic, critic_updates)
    critic_gradient_norm = optax.global_norm(critic_gradient)

    def actor_loss_fn(
        actor: PyTree,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        sampled_actions, log_probability = actor_sample(
            actor,
            observations,
            actor_key,
            action_bound,
            policy_mode,
        )
        actor_loss_critic = (
            state.critic if policy_mode == "public_torchrl" else critic
        )
        q_min = jnp.min(
            critic_apply(
                actor_loss_critic,
                observations,
                sampled_actions,
                policy_mode,
            ),
            axis=-1,
        )
        entropy_term = jnp.mean(
            jax.lax.stop_gradient(alpha)
            * NP_REAL_DTYPE(entropy_scale)
            * log_probability
        )
        action_l2_mean = jnp.mean(
            (sampled_actions / NP_REAL_DTYPE(action_bound)) ** 2
        )
        loss = (
            entropy_term
            - jnp.mean(q_min)
            + NP_REAL_DTYPE(action_l2_coefficient) * action_l2_mean
        )
        return loss, (
            log_probability,
            q_min,
            sampled_actions,
            action_l2_mean,
        )

    (
        actor_loss,
        (log_probability, actor_q, sampled_actions, action_l2_mean),
    ), actor_gradient = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(state.actor)
    if axis_name is not None:
        actor_gradient = jax.lax.pmean(
            actor_gradient, axis_name=axis_name
        )
    actor_gradient_norm = optax.global_norm(actor_gradient)
    actor_updates, proposed_actor_opt = actor_optimizer.update(
        actor_gradient, state.actor_opt, state.actor
    )
    proposed_actor = optax.apply_updates(state.actor, actor_updates)

    def alpha_loss_fn(log_alpha: Array) -> Array:
        return -jnp.mean(
            log_alpha
            * jax.lax.stop_gradient(
                NP_REAL_DTYPE(entropy_scale)
                * (
                    log_probability
                    + NP_REAL_DTYPE(target_entropy)
                )
            )
        )

    alpha_loss, alpha_gradient = jax.value_and_grad(alpha_loss_fn)(
        state.log_alpha
    )
    if axis_name is not None:
        alpha_gradient = jax.lax.pmean(
            alpha_gradient, axis_name=axis_name
        )
    alpha_updates, proposed_alpha_opt = alpha_optimizer.update(
        alpha_gradient, state.alpha_opt, state.log_alpha
    )
    proposed_log_alpha = optax.apply_updates(
        state.log_alpha, alpha_updates
    )
    if actor_update_period == 0:
        actor_updated = jnp.asarray(False)
    else:
        actor_updated = (
            (state.updates + 1) % actor_update_period == 0
        )
    actor = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_actor,
        state.actor,
    )
    actor_opt = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_actor_opt,
        state.actor_opt,
    )
    log_alpha = jnp.where(
        actor_updated, proposed_log_alpha, state.log_alpha
    )
    alpha_opt = jax.tree.map(
        lambda proposed, current: jnp.where(
            actor_updated, proposed, current
        ),
        proposed_alpha_opt,
        state.alpha_opt,
    )
    target_critic = jax.tree.map(
        lambda target_value, online_value: (
            (1.0 - NP_REAL_DTYPE(tau)) * target_value
            + NP_REAL_DTYPE(tau) * online_value
        ),
        state.target_critic,
        critic,
    )
    next_state = SACState(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        log_alpha=log_alpha,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        alpha_opt=alpha_opt,
        updates=state.updates + 1,
    )
    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(log_alpha),
        "actor_updated": actor_updated.astype(jnp.int32),
        "actor_gradient_norm": actor_gradient_norm,
        "critic_gradient_norm": critic_gradient_norm,
        "target_q_mean": jnp.mean(target_q),
        "target_q_std": jnp.std(target_q),
        "actor_q_mean": jnp.mean(actor_q),
        "entropy_term_mean": jnp.mean(
            jax.lax.stop_gradient(alpha)
            * NP_REAL_DTYPE(entropy_scale)
            * log_probability
        ),
        "action_l2_mean": action_l2_mean,
        "sampled_action_abs_mean": jnp.mean(jnp.abs(sampled_actions)),
        "sampled_action_abs_max": jnp.max(jnp.abs(sampled_actions)),
    }
    return next_state, metrics


def normalized_observation(
    probes: Array,
    previous_yaw: Array,
    inflow_speed: float,
    action_bound: float,
) -> Array:
    normalized_probes = (
        probes.reshape((probes.shape[0], -1))
        / NP_REAL_DTYPE(inflow_speed)
        - 1.0
    )
    normalized_yaw = previous_yaw / NP_REAL_DTYPE(action_bound)
    return jnp.concatenate(
        (normalized_probes, normalized_yaw), axis=-1
    )


def make_training_iteration(
    case: CompactMoleCase,
    functions: dict[str, Any],
    optimizers: tuple[Any, Any, Any],
    les_steps_per_interaction: int,
    sample_size: int,
    action_bound: float,
    gamma: float,
    tau: float,
    sac_updates_per_interaction: int,
    axis_name: str | None = None,
) -> Callable[
    [TrainingState, Array],
    tuple[TrainingState, dict[str, Array]],
]:
    zero_losses = {
        "critic_loss": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "actor_loss": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "alpha_loss": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "alpha": jnp.asarray(0.2, dtype=JAX_REAL_DTYPE),
        "actor_updated": jnp.asarray(0, dtype=jnp.int32),
        "actor_gradient_norm": jnp.asarray(
            0.0, dtype=JAX_REAL_DTYPE
        ),
        "critic_gradient_norm": jnp.asarray(
            0.0, dtype=JAX_REAL_DTYPE
        ),
        "target_q_mean": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "target_q_std": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "actor_q_mean": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "entropy_term_mean": jnp.asarray(
            0.0, dtype=JAX_REAL_DTYPE
        ),
        "action_l2_mean": jnp.asarray(0.0, dtype=JAX_REAL_DTYPE),
        "sampled_action_abs_mean": jnp.asarray(
            0.0, dtype=JAX_REAL_DTYPE
        ),
        "sampled_action_abs_max": jnp.asarray(
            0.0, dtype=JAX_REAL_DTYPE
        ),
    }

    def iteration(
        training: TrainingState,
        precursor_velocity: Array,
    ) -> tuple[TrainingState, dict[str, Array]]:
        action_key, update_key, next_key = jax.random.split(
            training.key, 3
        )
        actions, _ = actor_sample(
            training.sac.actor,
            training.observation,
            action_key,
            action_bound,
        )
        weights = functions["disk_weights"](actions)

        def les_body(
            flow: FlowState,
            _: None,
        ) -> tuple[FlowState, Array]:
            inlet = _precursor_inlet(
                precursor_velocity, flow.step_index, case
            )
            return functions["advance"](
                flow, actions, weights, inlet
            )

        flow, powers = jax.lax.scan(
            les_body,
            training.flow,
            xs=None,
            length=les_steps_per_interaction,
        )
        mean_power = jnp.mean(powers, axis=0)
        probes = functions["probes"](flow.velocity)
        next_observation = normalized_observation(
            probes,
            actions,
            case.inflow_hub_speed,
            action_bound,
        )
        reference_power = NP_REAL_DTYPE(
            float(case.num_turbines)
            * 0.5
            * case.density
            * (
                case.thrust_coefficient
                / (1.0 - case.induction_factor) ** 2
            )
            * (np.pi * case.diameter**2 / 4.0)
            * case.inflow_hub_speed**3
        )
        reward = (
            jnp.sum(mean_power, axis=-1) / reference_power
        )
        dones = jnp.zeros((case.batch,), dtype=JAX_REAL_DTYPE)
        replay = replay_add(
            training.replay,
            training.observation,
            actions,
            reward,
            next_observation,
            dones,
        )

        def update_many(
            carry: tuple[SACState, Array],
            _: None,
        ) -> tuple[tuple[SACState, Array], dict[str, Array]]:
            current_sac, current_key = carry
            current_key, step_key = jax.random.split(current_key)
            next_sac, losses = sac_update(
                current_sac,
                replay,
                step_key,
                optimizers,
                sample_size,
                action_bound,
                gamma,
                tau,
                -float(case.num_turbines),
                axis_name,
            )
            return (next_sac, current_key), losses

        enough_samples = replay.size >= sample_size
        if sac_updates_per_interaction > 0:
            def perform_updates(
                operand: tuple[SACState, Array],
            ) -> tuple[SACState, Array, dict[str, Array]]:
                (updated_sac, updated_key), losses = jax.lax.scan(
                    update_many,
                    operand,
                    xs=None,
                    length=sac_updates_per_interaction,
                )
                return (
                    updated_sac,
                    updated_key,
                    jax.tree.map(lambda value: value[-1], losses),
                )

            def skip_updates(
                operand: tuple[SACState, Array],
            ) -> tuple[SACState, Array, dict[str, Array]]:
                return operand[0], operand[1], zero_losses

            sac, update_key, losses = jax.lax.cond(
                enough_samples,
                perform_updates,
                skip_updates,
                (training.sac, update_key),
            )
        else:
            sac = training.sac
            losses = zero_losses

        next_training = TrainingState(
            flow=flow,
            observation=next_observation,
            previous_yaw=actions,
            replay=replay,
            sac=sac,
            key=next_key,
        )
        actor_checksum = sum(
            jnp.sum(value) for value in jax.tree.leaves(sac.actor)
        )
        if axis_name is not None:
            actor_replica_delta = (
                jax.lax.pmax(actor_checksum, axis_name)
                - jax.lax.pmin(actor_checksum, axis_name)
            )
        else:
            actor_replica_delta = jnp.asarray(0.0, dtype=JAX_REAL_DTYPE)
        metrics = {
            **losses,
            "mean_total_power_watts": jnp.mean(
                jnp.sum(mean_power, axis=-1)
            ),
            "mean_reward": jnp.mean(reward),
            "replay_size": replay.size,
            "sac_updates": sac.updates,
            "actor_replica_checksum_delta": actor_replica_delta,
        }
        return next_training, metrics

    return iteration


def load_precursor(
    path: Path,
    case: CompactMoleCase,
) -> tuple[Array, dict[str, Any]]:
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing precursor or metadata: {path}, {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_hash = _sha256(path)
    if actual_hash != metadata["npz_sha256"]:
        raise ValueError(
            f"Precursor hash mismatch: {actual_hash} != "
            f"{metadata['npz_sha256']}"
        )
    with np.load(path) as archive:
        velocity = archive["velocity"]
    expected_suffix = (3, case.ny, case.nz)
    if (
        velocity.ndim != 5
        or velocity.shape[0] != case.batch
        or (velocity.shape[1], velocity.shape[3], velocity.shape[4])
        != expected_suffix
    ):
        raise ValueError(
            "Precursor shape "
            f"{velocity.shape} is incompatible with batch/y/z "
            f"{(case.batch, case.ny, case.nz)}"
        )
    if velocity.dtype != NP_REAL_DTYPE:
        raise ValueError(f"Precursor dtype must be float32, got {velocity.dtype}")
    return jnp.asarray(velocity), metadata


def make_precursor_case(
    batch: int,
    pressure_corrections: int,
    pressure_projection: str,
) -> CompactMoleCase:
    """Return the frozen Wind-RL precursor_Base configuration."""

    return CompactMoleCase(
        batch=batch,
        nx=144,
        ny=41,
        nz=72,
        inflow_hub_speed=8.0,
        domain_lx=1764.0,
        domain_ly=500.0,
        domain_lz=882.0,
        x_periodic_grid=True,
        precursor_initial_ti=0.10,
        shifted_periodic_fringe=True,
        profile_relaxation_forcing=False,
        spatial_filter=True,
        pressure_corrections=pressure_corrections,
        pressure_projection=pressure_projection,
    )


def make_article_224d_precursor_case(
    batch: int,
    pressure_corrections: int,
    pressure_projection: str,
) -> CompactMoleCase:
    """Return the inferred final-article 224D precursor configuration.

    The article states that the public 14D precursor was extended to 224D.
    It does not publish the final input file or node count.  Keeping the public
    periodic-grid spacing while multiplying both length and node count by 16
    gives 28,224 m and 2,304 nodes.  Callers must therefore label this profile
    as inferred rather than source-exact.
    """

    return CompactMoleCase(
        batch=batch,
        nx=2304,
        ny=41,
        nz=72,
        inflow_hub_speed=8.0,
        domain_lx=28224.0,
        domain_ly=500.0,
        domain_lz=882.0,
        x_periodic_grid=True,
        precursor_initial_ti=0.10,
        shifted_periodic_fringe=True,
        profile_relaxation_forcing=False,
        spatial_filter=True,
        pressure_corrections=pressure_corrections,
        pressure_projection=pressure_projection,
    )


def make_wide_article_224d_precursor_case(
    layout: MoleFarmLayout,
    batch: int,
    pressure_corrections: int = 1,
    pressure_projection: str = "incompact3d_staggered",
) -> CompactMoleCase:
    """Return an inferred 224D precursor widened to a target farm layout."""

    layout.validate()
    return CompactMoleCase(
        batch=batch,
        nx=2304,
        ny=layout.ny,
        nz=layout.nz,
        inflow_hub_speed=8.0,
        domain_lx=28224.0,
        domain_ly=layout.domain_ly_m,
        domain_lz=layout.domain_lz_m,
        x_periodic_grid=True,
        precursor_initial_ti=0.10,
        shifted_periodic_fringe=True,
        profile_relaxation_forcing=False,
        spatial_filter=True,
        pressure_corrections=pressure_corrections,
        pressure_projection=pressure_projection,
    )


def make_mole_training_case(
    batch: int,
    pressure_corrections: int = 1,
    pressure_projection: str = "incompact3d_staggered",
    compact_solver: str = "dense_inverse",
    rhs_assembly: str = "tensor",
) -> CompactMoleCase:
    """Return the generated Wind-RL three-turbine training case."""

    return CompactMoleCase(
        batch=batch,
        nx=193,
        ny=41,
        nz=72,
        inflow_hub_speed=8.0,
        domain_lx=2394.0,
        domain_ly=500.0,
        domain_lz=882.0,
        friction_velocity=0.354,
        pressure_gradient_forcing=False,
        wall_sampling_dy=1.5,
        spatial_filter=True,
        farm_filter_all_directions=True,
        precursor_initial_ti=0.5,
        initialize_with_log_profile=False,
        shifted_periodic_fringe=False,
        pressure_corrections=pressure_corrections,
        pressure_projection=pressure_projection,
        compact_solver=compact_solver,
        rhs_assembly=rhs_assembly,
    )


def make_scaled_mole_training_case(
    layout: MoleFarmLayout,
    batch: int,
    pressure_corrections: int = 1,
    pressure_projection: str = "incompact3d_staggered",
    compact_solver: str = "dense_inverse",
    rhs_assembly: str = "tensor",
) -> ScalableCompactMoleCase:
    """Return a Mole-numerics farm case for a validated explicit layout."""

    layout.validate()
    return ScalableCompactMoleCase(
        batch=batch,
        nx=layout.nx,
        ny=layout.ny,
        nz=layout.nz,
        diameter=layout.diameter_m,
        inflow_hub_speed=8.0,
        domain_lx=layout.domain_lx_m,
        domain_ly=layout.domain_ly_m,
        domain_lz=layout.domain_lz_m,
        friction_velocity=0.354,
        pressure_gradient_forcing=False,
        wall_sampling_dy=1.5,
        spatial_filter=True,
        farm_filter_all_directions=True,
        precursor_initial_ti=0.5,
        initialize_with_log_profile=False,
        shifted_periodic_fringe=False,
        pressure_corrections=pressure_corrections,
        pressure_projection=pressure_projection,
        compact_solver=compact_solver,
        rhs_assembly=rhs_assembly,
        layout_name=layout.name,
        layout_sha256=layout.payload_sha256,
        configured_turbine_positions_m=layout.turbine_positions_m,
    )


def case_metadata(case: CompactMoleCase) -> dict[str, Any]:
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
            "turbine_x": case.turbine_x,
            "turbine_y": case.hub_height,
            "turbine_z": case.turbine_z,
        }
    )
    if isinstance(case, ScalableCompactMoleCase):
        metadata.update(
            {
                "num_turbines": case.num_turbines,
                "turbine_positions_m": case.turbine_positions_m,
                "turbine_y": tuple(
                    position[1] for position in case.turbine_positions_m
                ),
                "turbine_z": tuple(
                    position[2] for position in case.turbine_positions_m
                ),
            }
        )
    metadata["layout_contract"] = layout_contract(case)
    return metadata


def layout_contract(case: CompactMoleCase) -> dict[str, Any]:
    """Return a stable farm-layout contract for results and checkpoints."""

    payload = {
        "name": (
            case.layout_name
            if isinstance(case, ScalableCompactMoleCase)
            else "mole_legacy_3x1"
        ),
        "declared_layout_sha256": (
            case.layout_sha256
            if isinstance(case, ScalableCompactMoleCase)
            else None
        ),
        "num_turbines": case.num_turbines,
        "turbine_positions_m": case.turbine_positions_m,
        "grid": [case.nx, case.ny, case.nz],
        "domain_m": [case.lx, case.ly, case.lz],
        "spacing_m": [case.dx, case.dy, case.dz],
        "probe_order": (
            "turbine-major; for each turbine numpy.meshgrid of "
            "x/D=-2:0.5:3 and z/D=-1:1/3:1, flattened in C order"
        ),
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return {
        **payload,
        "resolved_contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_payload(path: Path | None, payload: dict[str, Any]) -> None:
    encoded_without_hash = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    payload["payload_sha256_without_hash_field"] = hashlib.sha256(
        encoded_without_hash.encode("utf-8")
    ).hexdigest()
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    print(encoded)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")


def run_numerics_check(seed: int) -> dict[str, Any]:
    derivative_errors: list[dict[str, float]] = []
    for size in (17, 33, 65, 129):
        length = 2.0 * np.pi
        periodic_x = jnp.arange(size, dtype=JAX_REAL_DTYPE) * (
            NP_REAL_DTYPE(length / size)
        )
        periodic_values = jnp.sin(2.0 * periodic_x)
        periodic_truth = 2.0 * jnp.cos(2.0 * periodic_x)
        periodic_result = compact_first_derivative(
            periodic_values, 0, length / size, True
        )
        nonperiodic_x = jnp.linspace(
            0.0, length, size, dtype=JAX_REAL_DTYPE
        )
        nonperiodic_values = jnp.sin(nonperiodic_x)
        nonperiodic_truth = jnp.cos(nonperiodic_x)
        nonperiodic_result = compact_first_derivative(
            nonperiodic_values,
            0,
            length / (size - 1),
            False,
        )
        derivative_errors.append(
            {
                "size": size,
                "periodic_l2": float(
                    jnp.sqrt(
                        jnp.mean(
                            (periodic_result - periodic_truth) ** 2
                        )
                    )
                ),
                "nonperiodic_interior_l2": float(
                    jnp.sqrt(
                        jnp.mean(
                            (
                                nonperiodic_result[3:-3]
                                - nonperiodic_truth[3:-3]
                            )
                            ** 2
                        )
                    )
                ),
            }
        )

    # The finer FP32 grids are already round-off dominated. Estimate the
    # observed order from the last pair before that plateau.
    coarse = derivative_errors[0]["nonperiodic_interior_l2"]
    fine = derivative_errors[1]["nonperiodic_interior_l2"]
    observed_order = math.log(max(coarse, 1.0e-30) / max(fine, 1.0e-30), 2.0)

    case = CompactMoleCase(
        batch=1,
        nx=33,
        ny=17,
        nz=16,
        pressure_corrections=3,
    )
    functions = build_flow_functions(
        case, periodic_x=False, include_turbines=False
    )
    key = jax.random.PRNGKey(seed)
    velocity = jax.random.normal(
        key,
        (1, 3, case.nx, case.ny, case.nz),
        dtype=JAX_REAL_DTYPE,
    )
    inlet = jnp.zeros(
        (1, 3, case.ny, case.nz), dtype=JAX_REAL_DTYPE
    )
    velocity = _apply_velocity_boundaries(velocity, inlet, False)
    raw_divergence = functions["divergence"](velocity)
    projected = jax.jit(functions["project"])(velocity, inlet)
    projected.block_until_ready()
    projected_divergence = functions["divergence"](projected)
    raw_divergence = functions["diagnostic_divergence"](
        raw_divergence
    )
    projected_divergence = functions["diagnostic_divergence"](
        projected_divergence
    )
    raw_rms = float(
        jnp.sqrt(jnp.mean(raw_divergence[:, 3:-3, 3:-3] ** 2))
    )
    projected_rms = float(
        jnp.sqrt(
            jnp.mean(projected_divergence[:, 3:-3, 3:-3] ** 2)
        )
    )
    ratio = projected_rms / max(raw_rms, 1.0e-30)
    normal_boundary_max = float(
        jnp.max(
            jnp.stack(
                (
                    jnp.max(jnp.abs(projected[:, 0, 0])),
                    jnp.max(jnp.abs(projected[:, 1, :, 0])),
                    jnp.max(jnp.abs(projected[:, 1, :, -1])),
                )
            )
        )
    )
    precursor_case = CompactMoleCase(
        batch=1,
        nx=32,
        ny=17,
        nz=16,
        domain_lx=392.0,
        domain_ly=200.0,
        domain_lz=196.0,
        pressure_corrections=1,
        spatial_filter=False,
        shifted_periodic_fringe=False,
    )
    precursor_functions = build_flow_functions(
        precursor_case, periodic_x=True, include_turbines=False
    )
    precursor_velocity = jax.random.normal(
        jax.random.PRNGKey(seed + 1),
        (
            1,
            3,
            precursor_case.nx,
            precursor_case.ny,
            precursor_case.nz,
        ),
        dtype=JAX_REAL_DTYPE,
    )
    precursor_velocity = _apply_velocity_boundaries(
        precursor_velocity, None, True
    )
    precursor_before = precursor_functions["divergence"](
        precursor_velocity
    )
    precursor_projected = jax.jit(precursor_functions["project"])(
        precursor_velocity, None
    )
    precursor_projected.block_until_ready()
    precursor_after = precursor_functions["divergence"](
        precursor_projected
    )
    precursor_before_rms = float(
        jnp.sqrt(jnp.mean(precursor_before**2))
    )
    precursor_after_rms = float(
        jnp.sqrt(jnp.mean(precursor_after**2))
    )
    precursor_ratio = precursor_after_rms / max(
        precursor_before_rms, 1.0e-30
    )
    checks = {
        "periodic_compact_derivative_accurate": (
            derivative_errors[2]["periodic_l2"] < 2.0e-5
        ),
        "nonperiodic_compact_derivative_accurate": fine < 2.0e-4,
        "compact_interior_order_at_least_four": observed_order > 4.0,
        "mixed_projection_reduces_divergence": ratio < 0.25,
        "normal_velocity_boundaries_enforced": normal_boundary_max < 1.0e-6,
        "projection_is_finite": bool(jnp.all(jnp.isfinite(projected))),
        "periodic_precursor_projection_ratio_below_2e-6": (
            precursor_ratio < 2.0e-6
        ),
        "periodic_precursor_projection_is_finite": bool(
            jnp.all(jnp.isfinite(precursor_projected))
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": {
            "derivative_errors": derivative_errors,
            "observed_nonperiodic_interior_order_17_to_33": observed_order,
            "raw_interior_divergence_rms": raw_rms,
            "projected_interior_divergence_rms": projected_rms,
            "projection_ratio": ratio,
            "normal_boundary_max_abs_velocity": normal_boundary_max,
            "periodic_precursor_projection": {
                "pressure_projection": precursor_functions[
                    "pressure_projection"
                ],
                "before_rms": precursor_before_rms,
                "after_rms": precursor_after_rms,
                "ratio": precursor_ratio,
            },
        },
    }


def run_precursor_mode(args: argparse.Namespace) -> dict[str, Any]:
    case = make_precursor_case(
        args.batch,
        args.pressure_corrections,
        args.pressure_projection,
    )
    if args.precursor is None:
        raise ValueError("--precursor is required in precursor mode")
    started = time.perf_counter()
    velocity, metrics = generate_precursor(
        case, args.seed, args.precursor_steps
    )
    generation_seconds = time.perf_counter() - started
    host_velocity = np.asarray(velocity, dtype=NP_REAL_DTYPE)
    args.precursor.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.precursor, velocity=host_velocity)
    npz_hash = _sha256(args.precursor)
    checks = {
        "field_is_finite": metrics["finite"],
        "turbulence_is_nonzero": (
            metrics["min_hub_turbulence_intensity"] > 0.005
        ),
        "hub_speed_is_positive": metrics["mean_hub_speed_m_per_s"] > 1.0,
        "divergence_is_bounded": (
            metrics["interior_divergence_rms"] < 0.05
        ),
    }
    metadata = {
        "schema_version": 1,
        "purpose": "physics_evolved_precursor_les_not_paper_inflow",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "case": case_metadata(case),
        "seed": args.seed,
        "spinup_steps": args.precursor_steps,
        "simulated_spinup_seconds": args.precursor_steps * case.dt,
        "generation_wallclock_seconds": generation_seconds,
        "shape": list(host_velocity.shape),
        "dtype": str(host_velocity.dtype),
        "npz_bytes": args.precursor.stat().st_size,
        "npz_sha256": npz_hash,
        "checks": checks,
        "metrics": metrics,
        "environment": environment_metadata(),
    }
    metadata_path = args.precursor.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "precursor": str(args.precursor),
        "metadata": str(metadata_path),
        "checks": checks,
        "metrics": metrics,
        "generation_wallclock_seconds": generation_seconds,
        "npz_sha256": npz_hash,
    }


def run_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    case = make_mole_training_case(
        args.batch,
        args.pressure_corrections,
        args.pressure_projection,
    )
    if args.precursor is None:
        raise ValueError("--precursor is required in benchmark-e2e mode")
    precursor, precursor_metadata = load_precursor(args.precursor, case)
    functions = build_flow_functions(
        case, periodic_x=False, include_turbines=True
    )
    inlet = _precursor_inlet(
        precursor, jnp.asarray(0, dtype=jnp.int32), case
    )
    flow = make_initial_flow(case, functions, args.seed)
    initial_velocity = _apply_velocity_boundaries(
        flow.velocity, inlet, False
    )
    flow = flow._replace(velocity=initial_velocity)
    zero_yaw = jnp.zeros(
        (case.batch, case.num_turbines), dtype=JAX_REAL_DTYPE
    )
    observation = normalized_observation(
        functions["probes"](flow.velocity),
        zero_yaw,
        case.inflow_hub_speed,
        args.action_bound,
    )
    observation_dim = int(observation.shape[-1])
    sac, optimizers = init_sac(
        jax.random.PRNGKey(args.seed + 1),
        observation_dim,
        case.num_turbines,
        args.hidden_dim,
        args.learning_rate,
    )
    replay = init_replay(
        args.replay_capacity, observation_dim, case.num_turbines
    )
    training = TrainingState(
        flow=flow,
        observation=observation,
        previous_yaw=zero_yaw,
        replay=replay,
        sac=sac,
        key=jax.random.PRNGKey(args.seed + 2),
    )
    iteration = jax.jit(
        make_training_iteration(
            case,
            functions,
            optimizers,
            args.les_steps_per_interaction,
            args.sample_size,
            args.action_bound,
            args.gamma,
            args.tau,
            args.sac_updates,
        )
    )

    compile_started = time.perf_counter()
    training, metrics = iteration(training, precursor)
    jax.block_until_ready(metrics["mean_total_power_watts"])
    compile_seconds = time.perf_counter() - compile_started

    warmup_durations = []
    for _ in range(args.warmup_interactions):
        started = time.perf_counter()
        training, metrics = iteration(training, precursor)
        jax.block_until_ready(metrics["mean_total_power_watts"])
        warmup_durations.append(time.perf_counter() - started)

    actor_before = jax.tree.map(lambda value: value.copy(), training.sac.actor)
    measured_durations = []
    metric_history = []
    measured_started = time.perf_counter()
    for _ in range(args.measured_interactions):
        step_started = time.perf_counter()
        training, metrics = iteration(training, precursor)
        jax.block_until_ready(metrics["mean_total_power_watts"])
        measured_durations.append(time.perf_counter() - step_started)
        metric_history.append(
            {
                key: float(value)
                for key, value in metrics.items()
            }
        )
    measured_seconds = time.perf_counter() - measured_started
    actor_delta = float(
        jnp.sqrt(
            sum(
                jnp.sum((after - before) ** 2)
                for after, before in zip(
                    jax.tree.leaves(training.sac.actor),
                    jax.tree.leaves(actor_before),
                    strict=True,
                )
            )
        )
    )
    interactions = case.batch * args.measured_interactions
    environment_les_steps = (
        interactions * args.les_steps_per_interaction
    )
    interactions_per_second = interactions / measured_seconds
    les_steps_per_second = environment_les_steps / measured_seconds
    projected_seconds = (
        PAPER_REPORTED_INTERACTIONS / interactions_per_second
    )
    finite = bool(
        _tree_all_finite(
            (
                training.flow,
                training.sac.actor,
                training.sac.critic,
                training.sac.log_alpha,
            )
        )
    )
    divergence = functions["divergence"](training.flow.velocity)
    divergence = functions["diagnostic_divergence"](divergence)
    divergence_rms = float(
        jnp.sqrt(jnp.mean(divergence[:, 3:-3, 3:-3] ** 2))
    )
    updates_performed = int(training.sac.updates)
    update_expected = args.sac_updates > 0
    checks = {
        "training_state_is_finite": finite,
        "precursor_hash_was_verified": True,
        "replay_contains_transitions": int(training.replay.size) > 0,
        "sac_updates_executed_when_requested": (
            updates_performed > 0 if update_expected else True
        ),
        "actor_changed_when_updates_requested": (
            actor_delta > 0.0 if update_expected else True
        ),
        "flow_divergence_is_bounded": divergence_rms < 0.1,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": {
            "compile_and_first_interaction_seconds": compile_seconds,
            "warmup_interaction_seconds": warmup_durations,
            "measured_interaction_seconds": measured_durations,
            "measured_wallclock_seconds": measured_seconds,
            "aggregate_interactions_per_second": interactions_per_second,
            "aggregate_les_steps_per_second": les_steps_per_second,
            "projected_hours_for_one_million_interactions": (
                projected_seconds / 3600.0
            ),
            "paper_reported_end_to_end_hours": (
                PAPER_REPORTED_TRAINING_SECONDS / 3600.0
            ),
            "throughput_ratio_to_paper_reported_run": (
                (
                    PAPER_REPORTED_INTERACTIONS
                    / PAPER_REPORTED_TRAINING_SECONDS
                )
                and interactions_per_second
                / (
                    PAPER_REPORTED_INTERACTIONS
                    / PAPER_REPORTED_TRAINING_SECONDS
                )
            ),
            "sac_updates_per_interaction": args.sac_updates,
            "sac_updates_performed_total": updates_performed,
            "actor_parameter_l2_change_during_measurement": actor_delta,
            "final_alpha": float(jnp.exp(training.sac.log_alpha)),
            "final_interior_divergence_rms": divergence_rms,
            "peak_jax_memory": [
                device.memory_stats() for device in jax.devices()
            ],
            "last_training_metrics": metric_history[-1],
        },
        "precursor": {
            "path": str(args.precursor),
            "sha256": precursor_metadata["npz_sha256"],
            "metrics": precursor_metadata["metrics"],
        },
        "configuration": {
            "observation_dim": observation_dim,
            "action_dim": case.num_turbines,
            "hidden_dim": args.hidden_dim,
            "sample_size": args.sample_size,
            "replay_capacity": args.replay_capacity,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "tau": args.tau,
            "action_bound_degrees": args.action_bound,
            "les_steps_per_interaction": args.les_steps_per_interaction,
            "warmup_interactions": args.warmup_interactions,
            "measured_interactions": args.measured_interactions,
        },
    }


def source_metadata() -> dict[str, Any]:
    return {
        "paper_doi": PAPER_DOI,
        "wind_rl_commit": WIND_RL_COMMIT,
        "incompact3d_smartredis_commit": (
            INCOMPACT3D_SMARTREDIS_COMMIT
        ),
        "incompact3d_reference_commit": (
            "abb010e615cff520f949a210278945346995966c"
        ),
        "compact_coefficients": {
            "alpha": 1.0 / 3.0,
            "a_over_dx": 7.0 / 9.0,
            "b_over_dx": 1.0 / 36.0,
            "source": "Incompact3d src/schemes.f90 first_derivative",
        },
        "compact_filter": {
            "coefficient": 0.49,
            "farm_training": (
                "ifilter=1: Dirichlet x, wall/free-slip y, periodic z"
            ),
            "static_calibration": (
                "ifilter=2: Dirichlet x and periodic z"
            ),
            "precursor": (
                "ifilter=1: periodic x/z and Dirichlet/free-slip y"
            ),
            "source": "Incompact3d src/filters.f90",
        },
        "boundary_algorithms": {
            "outflow": "Case-ABL convective plane from old velocity",
            "mass_correction": "pre_correc outlet mean equals inlet mean",
            "wall_sgs": (
                "wall_sgs_noslip; precursor samples at 2.2 dy and farm "
                "training at 1.5 dy"
            ),
        },
        "evidence_boundary": [
            "The paper's 32 precursor inflow realizations are unavailable.",
            "The generated precursor is LES-evolved but uses reproducible "
            "random initialization and is not a private paper realization.",
            "Farm and precursor pressure paths use public Incompact3d VP/PV "
            "operators and half-staggered Poisson eigenvalues.",
            "A frozen periodic volume is replayed through Taylor advection; "
            "the paper consumed saved time-resolved precursor planes.",
            "Independent-host totals use one learner per GPU. The dedicated "
            "distributed benchmark instead synchronizes SAC gradients with "
            "jax.lax.pmean.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("numerics", "precursor", "benchmark-e2e"),
        required=True,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--pressure-corrections", type=int, default=1)
    parser.add_argument(
        "--pressure-projection",
        choices=("incompact3d_staggered", "colocated_legacy"),
        default="incompact3d_staggered",
    )
    parser.add_argument("--precursor", type=Path)
    parser.add_argument("--precursor-steps", type=int, default=500)
    parser.add_argument(
        "--les-steps-per-interaction",
        type=int,
        default=PAPER_LES_STEPS_PER_INTERACTION,
    )
    parser.add_argument("--warmup-interactions", type=int, default=16)
    parser.add_argument("--measured-interactions", type=int, default=20)
    parser.add_argument("--sac-updates", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-bound", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if args.pressure_corrections < 1:
        raise ValueError("--pressure-corrections must be positive")
    if args.precursor_steps < 1:
        raise ValueError("--precursor-steps must be positive")
    if args.sample_size < 1 or args.replay_capacity < args.sample_size:
        raise ValueError("replay capacity must be >= sample size >= 1")

    if args.mode == "numerics":
        result = run_numerics_check(args.seed)
        case = None
    elif args.mode == "precursor":
        result = run_precursor_mode(args)
        case = make_precursor_case(
            args.batch,
            args.pressure_corrections,
            args.pressure_projection,
        )
    else:
        result = run_end_to_end(args)
        case = make_mole_training_case(
            args.batch,
            args.pressure_corrections,
            args.pressure_projection,
        )

    payload = {
        "schema_version": 1,
        "purpose": "mole_compact_precursor_sac_end_to_end_benchmark",
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": source_metadata(),
        "environment": environment_metadata(),
        "case": case_metadata(case) if case is not None else None,
        "result": result,
    }
    write_payload(args.output, payload)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
