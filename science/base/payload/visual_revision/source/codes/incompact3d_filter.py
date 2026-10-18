"""Source-aligned Incompact3d compact spatial filters for JAX.

The coefficients and boundary closures follow ``filters.f90`` in the
public Incompact3d source used by Wind-RL.  Mole's farm configuration uses
Dirichlet closures in x, no filtering in y, and a periodic filter in z.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


@lru_cache(maxsize=None)
def dirichlet_filter_matrix(size: int, alpha: float = 0.49) -> np.ndarray:
    """Return A^-1 B for Incompact3d's two-sided Dirichlet filter."""

    if size < 7:
        raise ValueError("compact filter requires at least seven points")
    if not -0.5 < alpha < 0.5:
        raise ValueError("filter alpha must be in (-0.5, 0.5)")

    a = (11.0 + 10.0 * alpha) / 16.0
    b = 0.5 * (15.0 + 34.0 * alpha) / 32.0
    c = 0.5 * (-3.0 + 6.0 * alpha) / 16.0
    d = 0.5 * (1.0 - 2.0 * alpha) / 32.0

    alpha_2 = alpha
    a_2 = 1.0 / 8.0 + 3.0 * alpha / 4.0
    b_2 = 5.0 / 8.0 + 3.0 * alpha / 4.0
    c_2 = 3.0 / 8.0 + alpha / 4.0
    d_2 = -1.0 / 8.0 + alpha / 4.0

    alpha_3 = alpha
    a_3 = -1.0 / 32.0 + alpha / 16.0
    b_3 = 5.0 / 32.0 + 11.0 * alpha / 16.0
    c_3 = 11.0 / 16.0 + 5.0 * alpha / 8.0
    d_3 = 5.0 / 16.0 + 3.0 * alpha / 8.0
    e_3 = -5.0 / 32.0 + 5.0 * alpha / 16.0
    f_3 = 1.0 / 32.0 - alpha / 16.0

    lhs = np.eye(size, dtype=np.float64)
    lhs[1, 0] = alpha_2
    lhs[1, 2] = alpha
    for index in range(2, size - 2):
        lhs[index, index - 1] = alpha
        lhs[index, index + 1] = alpha
    lhs[-2, -3] = alpha
    lhs[-2, -1] = alpha_2

    rhs = np.zeros((size, size), dtype=np.float64)
    rhs[0, 0] = 1.0
    rhs[1, :4] = (a_2, b_2, c_2, d_2)
    rhs[2, :6] = (a_3, b_3, c_3, d_3, e_3, f_3)
    for index in range(3, size - 3):
        rhs[index, index] = a
        rhs[index, index - 1] = b
        rhs[index, index + 1] = b
        rhs[index, index - 2] = c
        rhs[index, index + 2] = c
        rhs[index, index - 3] = d
        rhs[index, index + 3] = d
    rhs[-3, -6:] = (f_3, e_3, d_3, c_3, b_3, a_3)
    rhs[-2, -4:] = (d_2, c_2, b_2, a_2)
    rhs[-1, -1] = 1.0
    return np.linalg.solve(lhs, rhs)


def filter_dirichlet(values: Array, axis: int, alpha: float = 0.49) -> Array:
    """Apply Incompact3d's nonperiodic compact filter along one axis."""

    size = values.shape[axis]
    operator = jnp.asarray(
        dirichlet_filter_matrix(size, alpha), dtype=values.dtype
    )
    moved = jnp.moveaxis(values, axis, -1)
    filtered = jnp.matmul(
        moved,
        operator.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.moveaxis(filtered, -1, axis)


def filter_dirichlet_tridiagonal(
    values: Array, axis: int, alpha: float = 0.49
) -> Array:
    """Apply the same Dirichlet filter without materializing A^-1."""

    size = values.shape[axis]
    if size < 7:
        raise ValueError("compact filter requires at least seven points")
    real_dtype = np.dtype(values.dtype).type
    alpha_value = real_dtype(alpha)
    a = real_dtype((11.0 + 10.0 * alpha) / 16.0)
    b = real_dtype(0.5 * (15.0 + 34.0 * alpha) / 32.0)
    c = real_dtype(0.5 * (-3.0 + 6.0 * alpha) / 16.0)
    d = real_dtype(0.5 * (1.0 - 2.0 * alpha) / 32.0)
    a2 = real_dtype(1.0 / 8.0 + 3.0 * alpha / 4.0)
    b2 = real_dtype(5.0 / 8.0 + 3.0 * alpha / 4.0)
    c2 = real_dtype(3.0 / 8.0 + alpha / 4.0)
    d2 = real_dtype(-1.0 / 8.0 + alpha / 4.0)
    a3 = real_dtype(-1.0 / 32.0 + alpha / 16.0)
    b3 = real_dtype(5.0 / 32.0 + 11.0 * alpha / 16.0)
    c3 = real_dtype(11.0 / 16.0 + 5.0 * alpha / 8.0)
    d3 = real_dtype(5.0 / 16.0 + 3.0 * alpha / 8.0)
    e3 = real_dtype(-5.0 / 32.0 + 5.0 * alpha / 16.0)
    f3 = real_dtype(1.0 / 32.0 - alpha / 16.0)

    moved = jnp.moveaxis(values, axis, -1)
    flat = moved.reshape((-1, size))
    rhs = jnp.zeros_like(flat)
    rhs = rhs.at[:, 0].set(flat[:, 0])
    rhs = rhs.at[:, 1].set(
        a2 * flat[:, 0]
        + b2 * flat[:, 1]
        + c2 * flat[:, 2]
        + d2 * flat[:, 3]
    )
    rhs = rhs.at[:, 2].set(
        a3 * flat[:, 0]
        + b3 * flat[:, 1]
        + c3 * flat[:, 2]
        + d3 * flat[:, 3]
        + e3 * flat[:, 4]
        + f3 * flat[:, 5]
    )
    rhs = rhs.at[:, 3:-3].set(
        a * flat[:, 3:-3]
        + b * (flat[:, 2:-4] + flat[:, 4:-2])
        + c * (flat[:, 1:-5] + flat[:, 5:-1])
        + d * (flat[:, :-6] + flat[:, 6:])
    )
    rhs = rhs.at[:, -3].set(
        f3 * flat[:, -6]
        + e3 * flat[:, -5]
        + d3 * flat[:, -4]
        + c3 * flat[:, -3]
        + b3 * flat[:, -2]
        + a3 * flat[:, -1]
    )
    rhs = rhs.at[:, -2].set(
        d2 * flat[:, -4]
        + c2 * flat[:, -3]
        + b2 * flat[:, -2]
        + a2 * flat[:, -1]
    )
    rhs = rhs.at[:, -1].set(flat[:, -1])

    lower = jnp.full((size,), alpha_value, dtype=values.dtype)
    upper = jnp.full((size,), alpha_value, dtype=values.dtype)
    diagonal = jnp.ones((size,), dtype=values.dtype)
    lower = lower.at[0].set(0.0).at[-1].set(0.0)
    upper = upper.at[0].set(0.0).at[-1].set(0.0)
    solution = jax.lax.linalg.tridiagonal_solve(
        lower, diagonal, upper, rhs.T
    ).T
    restored = solution.reshape(moved.shape)
    return jnp.moveaxis(restored, -1, axis)


def periodic_transfer_function(
    size: int, alpha: float = 0.49, dtype: jnp.dtype = jnp.float32
) -> Array:
    """Return the exact Fourier response of Incompact3d's periodic filter."""

    theta = 2.0 * np.pi * jnp.fft.rfftfreq(size).astype(dtype)
    alpha_value = jnp.asarray(alpha, dtype=dtype)
    a = (11.0 + 10.0 * alpha_value) / 16.0
    b = 0.5 * (15.0 + 34.0 * alpha_value) / 32.0
    c = 0.5 * (-3.0 + 6.0 * alpha_value) / 16.0
    d = 0.5 * (1.0 - 2.0 * alpha_value) / 32.0
    numerator = (
        a
        + 2.0 * b * jnp.cos(theta)
        + 2.0 * c * jnp.cos(2.0 * theta)
        + 2.0 * d * jnp.cos(3.0 * theta)
    )
    denominator = 1.0 + 2.0 * alpha_value * jnp.cos(theta)
    return numerator / denominator


def filter_periodic(values: Array, axis: int, alpha: float = 0.49) -> Array:
    """Apply Incompact3d's periodic compact filter along one axis."""

    size = values.shape[axis]
    response = periodic_transfer_function(size, alpha, values.dtype)
    shape = [1] * values.ndim
    shape[axis] = response.shape[0]
    transformed = jnp.fft.rfft(values, axis=axis)
    return jnp.fft.irfft(
        transformed * response.reshape(shape), n=size, axis=axis
    )


@lru_cache(maxsize=None)
def wall_free_slip_filter_matrix(
    size: int, alpha: float = 0.49, parity: int = 1
) -> np.ndarray:
    """Return the ``fily_21`` matrix for one velocity parity."""

    if size < 7:
        raise ValueError("compact filter requires at least seven points")
    if parity not in (0, 1):
        raise ValueError("parity must be 0 (normal) or 1 (tangential)")
    a = (11.0 + 10.0 * alpha) / 16.0
    b = 0.5 * (15.0 + 34.0 * alpha) / 32.0
    c = 0.5 * (-3.0 + 6.0 * alpha) / 16.0
    d = 0.5 * (1.0 - 2.0 * alpha) / 32.0
    a2 = 1.0 / 8.0 + 3.0 * alpha / 4.0
    b2 = 5.0 / 8.0 + 3.0 * alpha / 4.0
    c2 = 3.0 / 8.0 + alpha / 4.0
    d2 = -1.0 / 8.0 + alpha / 4.0
    a3 = -1.0 / 32.0 + alpha / 16.0
    b3 = 5.0 / 32.0 + 11.0 * alpha / 16.0
    c3 = 11.0 / 16.0 + 5.0 * alpha / 8.0
    d3 = 5.0 / 16.0 + 3.0 * alpha / 8.0
    e3 = -5.0 / 32.0 + 5.0 * alpha / 16.0
    f3 = 1.0 / 32.0 - alpha / 16.0

    lhs = np.eye(size, dtype=np.float64)
    lhs[1, 0] = alpha
    lhs[1, 2] = alpha
    for index in range(2, size - 1):
        lhs[index, index - 1] = alpha
        lhs[index, index + 1] = alpha
    lhs[-1, -2] = 2.0 * alpha if parity == 1 else 0.0

    rhs = np.zeros((size, size), dtype=np.float64)
    rhs[0, 0] = 1.0
    rhs[1, :4] = (a2, b2, c2, d2)
    rhs[2, :6] = (a3, b3, c3, d3, e3, f3)
    for index in range(3, size - 3):
        rhs[index, index] = a
        rhs[index, index - 1] = b
        rhs[index, index + 1] = b
        rhs[index, index - 2] = c
        rhs[index, index + 2] = c
        rhs[index, index - 3] = d
        rhs[index, index + 3] = d
    if parity == 1:
        rhs[-3, -6:] = (d, c, b, a, b + d, c)
        rhs[-2, -5:] = (d, c, b + d, a + c, b)
        rhs[-1, -4:] = (2.0 * d, 2.0 * c, 2.0 * b, a)
    else:
        rhs[-3, -6:] = (d, c, b, a, b - d, c)
        rhs[-2, -5:] = (d, c, b - d, a - c, b)
    return np.linalg.solve(lhs, rhs)


def filter_wall_free_slip(
    values: Array,
    axis: int,
    alpha: float = 0.49,
    parity: int = 1,
) -> Array:
    """Apply Incompact3d ``fily_21`` along the wall-normal axis."""

    operator = jnp.asarray(
        wall_free_slip_filter_matrix(values.shape[axis], alpha, parity),
        dtype=values.dtype,
    )
    moved = jnp.moveaxis(values, axis, -1)
    filtered = jnp.matmul(
        moved,
        operator.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.moveaxis(filtered, -1, axis)


def mole_farm_spatial_filter(
    velocity: Array, alpha: float = 0.49, solver: str = "dense"
) -> Array:
    """Apply Mole farm ``ifilter=2``: Dirichlet x then periodic z."""

    if solver == "dense":
        filtered_x = filter_dirichlet(velocity, axis=-3, alpha=alpha)
    elif solver == "tridiagonal":
        filtered_x = filter_dirichlet_tridiagonal(
            velocity, axis=-3, alpha=alpha
        )
    else:
        raise ValueError(f"unknown compact filter solver: {solver}")
    return filter_periodic(
        filtered_x,
        axis=-1,
        alpha=alpha,
    )


def mole_farm_all_direction_filter(
    velocity: Array, alpha: float = 0.49, solver: str = "dense"
) -> Array:
    """Apply Mole farm ``ifilter=1`` in x, wall-normal y and periodic z."""

    if solver == "dense":
        filtered = filter_dirichlet(velocity, axis=-3, alpha=alpha)
    elif solver == "tridiagonal":
        filtered = filter_dirichlet_tridiagonal(
            velocity, axis=-3, alpha=alpha
        )
    else:
        raise ValueError(f"unknown compact filter solver: {solver}")
    components = (
        filter_wall_free_slip(filtered[:, 0], -2, alpha, parity=1),
        filter_wall_free_slip(filtered[:, 1], -2, alpha, parity=0),
        filter_wall_free_slip(filtered[:, 2], -2, alpha, parity=1),
    )
    return filter_periodic(jnp.stack(components, axis=1), -1, alpha)


def mole_precursor_spatial_filter(
    velocity: Array, alpha: float = 0.49
) -> Array:
    """Apply precursor ``ifilter=1`` in periodic x, wall y and periodic z."""

    filtered = filter_periodic(velocity, axis=-3, alpha=alpha)
    components = (
        filter_wall_free_slip(filtered[:, 0], -2, alpha, parity=1),
        filter_wall_free_slip(filtered[:, 1], -2, alpha, parity=0),
        filter_wall_free_slip(filtered[:, 2], -2, alpha, parity=1),
    )
    return filter_periodic(jnp.stack(components, axis=1), -1, alpha)
