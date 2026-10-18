"""Incompact3d-compatible staggered pressure operators for JAX.

The velocity mesh has ``(nx, ny, nz)`` points. Pressure and divergence use
``(nx - 1, ny - 1, nz)`` in the open farm and ``(nx, ny - 1, nz)`` in the
periodic precursor. Coefficients follow the public Incompact3d
``interpolation`` and VP/PV routines for ``ifirstder=4, ipinter=1``.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import jax.scipy.fft as jsp_fft
import numpy as np


Array = jnp.ndarray


DERIVATIVE_ALPHA = 9.0 / 62.0
DERIVATIVE_A = 63.0 / 62.0
DERIVATIVE_B = 17.0 / 186.0
INTERPOLATION_ALPHA = 3.0 / 10.0
INTERPOLATION_A = 3.0 / 4.0
INTERPOLATION_B = 1.0 / 20.0


def _implicit_matrix(
    size: int,
    alpha: float,
    *,
    first_upper: float,
    last_lower: float,
    endpoint_diagonal: float,
) -> np.ndarray:
    matrix = np.eye(size, dtype=np.float64)
    rows = np.arange(size - 1)
    matrix[rows, rows + 1] = alpha
    matrix[rows + 1, rows] = alpha
    matrix[0, 0] = endpoint_diagonal
    matrix[-1, -1] = endpoint_diagonal
    matrix[0, 1] = first_upper
    matrix[-1, -2] = last_lower
    return matrix


@lru_cache(maxsize=None)
def nonperiodic_vp_matrix(
    size: int, spacing: float, derivative: bool
) -> np.ndarray:
    """Map a velocity-grid line of length ``size`` to ``size - 1``."""

    output_size = size - 1
    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B

    rhs = np.zeros((output_size, size), dtype=np.float64)
    if derivative:
        rhs[0, 0:3] = (-a - 2.0 * b, a + b, b)
        rhs[1, [0, 1, 2, 3]] = (-b, -a, a, b)
        for row in range(2, output_size - 2):
            rhs[row, [row - 1, row, row + 1, row + 2]] = (
                -b,
                -a,
                a,
                b,
            )
        row = output_size - 2
        rhs[row, [row - 1, row, row + 1, row + 2]] = (
            -b,
            -a,
            a,
            b,
        )
        row = output_size - 1
        rhs[row, [row - 1, row, row + 1]] = (-b, -a - b, a + 2.0 * b)
    else:
        rhs[0, 0:3] = (a, a + b, b)
        rhs[1, [0, 1, 2, 3]] = (b, a, a, b)
        for row in range(2, output_size - 1):
            rhs[row, [row - 1, row, row + 1, row + 2]] = (b, a, a, b)
        row = output_size - 1
        rhs[row, [row - 1, row, row + 1]] = (b, a + b, a)

    implicit = _implicit_matrix(
        output_size,
        alpha,
        first_upper=alpha,
        # In ``prepare``, row i uses b(i-1).  The public source modifies
        # b(nxm) for VP interpolation, which is intentionally outside the
        # last row's lower diagonal; both VP operators therefore retain it.
        last_lower=alpha,
        endpoint_diagonal=1.0 + alpha,
    )
    return np.linalg.solve(implicit, rhs)


@lru_cache(maxsize=None)
def nonperiodic_pv_matrix(
    pressure_size: int, spacing: float, derivative: bool
) -> np.ndarray:
    """Map a pressure-grid line to ``pressure_size + 1`` velocity points."""

    output_size = pressure_size + 1
    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B

    rhs = np.zeros((output_size, pressure_size), dtype=np.float64)
    if derivative:
        rhs[1, [0, 1, 2]] = (-a - b, a, b)
        for row in range(2, output_size - 2):
            rhs[row, [row - 2, row - 1, row, row + 1]] = (
                -b,
                -a,
                a,
                b,
            )
        row = output_size - 2
        rhs[row, [row - 2, row - 1, row]] = (-b, -a, a + b)
        implicit = _implicit_matrix(
            output_size,
            alpha,
            first_upper=0.0,
            last_lower=0.0,
            endpoint_diagonal=1.0,
        )
    else:
        rhs[0, [0, 1]] = (2.0 * a, 2.0 * b)
        rhs[1, [0, 1, 2]] = (a + b, a, b)
        for row in range(2, output_size - 2):
            rhs[row, [row - 2, row - 1, row, row + 1]] = (b, a, a, b)
        row = output_size - 2
        rhs[row, [row - 2, row - 1, row]] = (b, a, a + b)
        rhs[-1, [-2, -1]] = (2.0 * b, 2.0 * a)
        implicit = _implicit_matrix(
            output_size,
            alpha,
            first_upper=2.0 * alpha,
            last_lower=2.0 * alpha,
            endpoint_diagonal=1.0,
        )
    return np.linalg.solve(implicit, rhs)


@lru_cache(maxsize=None)
def periodic_vp_matrix(
    size: int, spacing: float, derivative: bool
) -> np.ndarray:
    """Periodic velocity-to-pressure operator, including the half-cell phase."""

    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B
    implicit = np.eye(size, dtype=np.float64)
    rhs = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        implicit[row, (row - 1) % size] = alpha
        implicit[row, (row + 1) % size] = alpha
        if derivative:
            rhs[row, row] -= a
            rhs[row, (row + 1) % size] += a
            rhs[row, (row - 1) % size] -= b
            rhs[row, (row + 2) % size] += b
        else:
            rhs[row, row] += a
            rhs[row, (row + 1) % size] += a
            rhs[row, (row - 1) % size] += b
            rhs[row, (row + 2) % size] += b
    return np.linalg.solve(implicit, rhs)


@lru_cache(maxsize=None)
def periodic_pv_matrix(
    size: int, spacing: float, derivative: bool
) -> np.ndarray:
    """Periodic pressure-to-velocity operator, including the half-cell phase."""

    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B
    implicit = np.eye(size, dtype=np.float64)
    rhs = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        implicit[row, (row - 1) % size] = alpha
        implicit[row, (row + 1) % size] = alpha
        if derivative:
            rhs[row, row] += a
            rhs[row, (row - 1) % size] -= a
            rhs[row, (row + 1) % size] += b
            rhs[row, (row - 2) % size] -= b
        else:
            rhs[row, row] += a
            rhs[row, (row - 1) % size] += a
            rhs[row, (row + 1) % size] += b
            rhs[row, (row - 2) % size] += b
    return np.linalg.solve(implicit, rhs)


def apply_matrix(values: Array, matrix: np.ndarray, axis: int) -> Array:
    moved = jnp.moveaxis(values, axis, -1)
    operator = jnp.asarray(matrix, dtype=values.dtype)
    transformed = jnp.matmul(
        moved,
        operator.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.moveaxis(transformed, -1, axis)


def _tridiagonal_solve(
    rhs: Array,
    axis: int,
    alpha: float,
    *,
    endpoint_diagonal: float,
    first_upper: float,
    last_lower: float,
) -> Array:
    moved = jnp.moveaxis(rhs, axis, 0)
    size = moved.shape[0]
    dtype = rhs.dtype
    lower = jnp.full((size,), alpha, dtype=dtype).at[0].set(0.0)
    upper = jnp.full((size,), alpha, dtype=dtype).at[-1].set(0.0)
    diagonal = jnp.ones((size,), dtype=dtype)
    diagonal = diagonal.at[0].set(endpoint_diagonal)
    diagonal = diagonal.at[-1].set(endpoint_diagonal)
    upper = upper.at[0].set(first_upper)
    lower = lower.at[-1].set(last_lower)
    flat = moved.reshape((size, -1))
    solution = jax.lax.linalg.tridiagonal_solve(
        lower, diagonal, upper, flat
    )
    return jnp.moveaxis(solution.reshape(moved.shape), 0, axis)


def _nonperiodic_vp(
    values: Array, axis: int, spacing: float, derivative: bool
) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    size = moved.shape[0]
    output_size = size - 1
    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B
    rhs = jnp.zeros((output_size,) + moved.shape[1:], dtype=values.dtype)
    if derivative:
        rhs = rhs.at[0].set(
            a * (moved[1] - moved[0])
            + b * (moved[2] - 2.0 * moved[0] + moved[1])
        )
        rhs = rhs.at[1].set(
            a * (moved[2] - moved[1]) + b * (moved[3] - moved[0])
        )
        rhs = rhs.at[2:-2].set(
            a * (moved[3:-2] - moved[2:-3])
            + b * (moved[4:-1] - moved[1:-4])
        )
        rhs = rhs.at[-2].set(
            a * (moved[-2] - moved[-3])
            + b * (moved[-1] - moved[-4])
        )
        rhs = rhs.at[-1].set(
            a * (moved[-1] - moved[-2])
            + b * (2.0 * moved[-1] - moved[-2] - moved[-3])
        )
    else:
        rhs = rhs.at[0].set(
            a * (moved[1] + moved[0]) + b * (moved[2] + moved[1])
        )
        rhs = rhs.at[1].set(
            a * (moved[2] + moved[1]) + b * (moved[3] + moved[0])
        )
        rhs = rhs.at[2:-1].set(
            a * (moved[3:-1] + moved[2:-2])
            + b * (moved[4:] + moved[1:-3])
        )
        rhs = rhs.at[-1].set(
            a * (moved[-1] + moved[-2])
            + b * (moved[-2] + moved[-3])
        )
    rhs = jnp.moveaxis(rhs, 0, axis)
    return _tridiagonal_solve(
        rhs,
        axis,
        alpha,
        endpoint_diagonal=1.0 + alpha,
        first_upper=alpha,
        last_lower=alpha,
    )


def _nonperiodic_pv(
    values: Array, axis: int, spacing: float, derivative: bool
) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    pressure_size = moved.shape[0]
    output_size = pressure_size + 1
    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B
    rhs = jnp.zeros((output_size,) + moved.shape[1:], dtype=values.dtype)
    if derivative:
        rhs = rhs.at[1].set(
            a * (moved[1] - moved[0]) + b * (moved[2] - moved[0])
        )
        rhs = rhs.at[2:-2].set(
            a * (moved[2:-1] - moved[1:-2])
            + b * (moved[3:] - moved[:-3])
        )
        rhs = rhs.at[-2].set(
            a * (moved[-1] - moved[-2])
            + b * (moved[-1] - moved[-3])
        )
        first_upper = 0.0
        last_lower = 0.0
    else:
        rhs = rhs.at[0].set(2.0 * a * moved[0] + 2.0 * b * moved[1])
        rhs = rhs.at[1].set(
            a * (moved[1] + moved[0]) + b * (moved[2] + moved[0])
        )
        rhs = rhs.at[2:-2].set(
            a * (moved[2:-1] + moved[1:-2])
            + b * (moved[3:] + moved[:-3])
        )
        rhs = rhs.at[-2].set(
            a * (moved[-1] + moved[-2])
            + b * (moved[-1] + moved[-3])
        )
        rhs = rhs.at[-1].set(
            2.0 * a * moved[-1] + 2.0 * b * moved[-2]
        )
        first_upper = 2.0 * alpha
        last_lower = 2.0 * alpha
    rhs = jnp.moveaxis(rhs, 0, axis)
    return _tridiagonal_solve(
        rhs,
        axis,
        alpha,
        endpoint_diagonal=1.0,
        first_upper=first_upper,
        last_lower=last_lower,
    )


def _periodic_staggered(
    values: Array,
    axis: int,
    spacing: float,
    derivative: bool,
    velocity_to_pressure: bool,
) -> Array:
    size = values.shape[axis]
    dtype = values.dtype
    theta = 2.0 * jnp.pi * jnp.fft.fftfreq(size).astype(dtype)
    if derivative:
        alpha = DERIVATIVE_ALPHA
        a = DERIVATIVE_A / spacing
        b = DERIVATIVE_B / spacing
        if velocity_to_pressure:
            numerator = (
                a * (jnp.exp(1j * theta) - 1.0)
                + b * (jnp.exp(2j * theta) - jnp.exp(-1j * theta))
            )
        else:
            numerator = (
                a * (1.0 - jnp.exp(-1j * theta))
                + b * (jnp.exp(1j * theta) - jnp.exp(-2j * theta))
            )
    else:
        alpha = INTERPOLATION_ALPHA
        a = INTERPOLATION_A
        b = INTERPOLATION_B
        if velocity_to_pressure:
            numerator = (
                a * (jnp.exp(1j * theta) + 1.0)
                + b * (jnp.exp(2j * theta) + jnp.exp(-1j * theta))
            )
        else:
            numerator = (
                a * (1.0 + jnp.exp(-1j * theta))
                + b * (jnp.exp(1j * theta) + jnp.exp(-2j * theta))
            )
    multiplier = numerator / (1.0 + 2.0 * alpha * jnp.cos(theta))
    shape = [1] * values.ndim
    shape[axis] = size
    transformed = jnp.fft.fft(values, axis=axis)
    return jnp.fft.ifft(
        transformed * multiplier.reshape(shape), axis=axis
    ).real


def staggered_divergence(
    velocity: Array,
    dx: float,
    dy: float,
    dz: float,
    implementation: str = "dense_high_precision",
) -> Array:
    """Reproduce Incompact3d ``divergence`` for the Mole boundary class."""

    nx, ny, nz = velocity.shape[-3:]
    if implementation == "dense_high_precision":
        dxu = apply_matrix(
            velocity[:, 0], nonperiodic_vp_matrix(nx, dx, True), -3
        )
        ixv = apply_matrix(
            velocity[:, 1], nonperiodic_vp_matrix(nx, dx, False), -3
        )
        ixw = apply_matrix(
            velocity[:, 2], nonperiodic_vp_matrix(nx, dx, False), -3
        )
        dxy = apply_matrix(
            dxu, nonperiodic_vp_matrix(ny, dy, False), -2
        ) + apply_matrix(ixv, nonperiodic_vp_matrix(ny, dy, True), -2)
        ixyw = apply_matrix(
            ixw, nonperiodic_vp_matrix(ny, dy, False), -2
        )
        return _periodic_staggered(
            dxy, -1, dz, False, True
        ) + _periodic_staggered(ixyw, -1, dz, True, True)
    if implementation != "tridiagonal_experimental":
        raise ValueError(f"unknown staggered implementation: {implementation}")
    dxu = _nonperiodic_vp(velocity[:, 0], -3, dx, True)
    ixv = _nonperiodic_vp(velocity[:, 1], -3, dx, False)
    ixw = _nonperiodic_vp(velocity[:, 2], -3, dx, False)
    dxy = _nonperiodic_vp(dxu, -2, dy, False) + _nonperiodic_vp(
        ixv, -2, dy, True
    )
    ixyw = _nonperiodic_vp(ixw, -2, dy, False)
    return _periodic_staggered(
        dxy, -1, dz, False, True
    ) + _periodic_staggered(ixyw, -1, dz, True, True)


def staggered_gradient(
    pressure: Array,
    dx: float,
    dy: float,
    dz: float,
    implementation: str = "dense_high_precision",
) -> Array:
    """Reproduce Incompact3d ``gradp`` from pressure to velocity mesh."""

    nxm, nym, nz = pressure.shape[-3:]
    if implementation == "dense_high_precision":
        izp = _periodic_staggered(pressure, -1, dz, False, False)
        dzp = _periodic_staggered(pressure, -1, dz, True, False)
        iyizp = apply_matrix(
            izp, nonperiodic_pv_matrix(nym, dy, False), -2
        )
        dyizp = apply_matrix(
            izp, nonperiodic_pv_matrix(nym, dy, True), -2
        )
        iydzp = apply_matrix(
            dzp, nonperiodic_pv_matrix(nym, dy, False), -2
        )
        gx = apply_matrix(
            iyizp, nonperiodic_pv_matrix(nxm, dx, True), -3
        )
        gy = apply_matrix(
            dyizp, nonperiodic_pv_matrix(nxm, dx, False), -3
        )
        gz = apply_matrix(
            iydzp, nonperiodic_pv_matrix(nxm, dx, False), -3
        )
        return jnp.stack((gx, gy, gz), axis=1)
    if implementation != "tridiagonal_experimental":
        raise ValueError(f"unknown staggered implementation: {implementation}")
    izp = _periodic_staggered(pressure, -1, dz, False, False)
    dzp = _periodic_staggered(pressure, -1, dz, True, False)
    iyizp = _nonperiodic_pv(izp, -2, dy, False)
    dyizp = _nonperiodic_pv(izp, -2, dy, True)
    iydzp = _nonperiodic_pv(dzp, -2, dy, False)
    gx = _nonperiodic_pv(iyizp, -3, dx, True)
    gy = _nonperiodic_pv(dyizp, -3, dx, False)
    gz = _nonperiodic_pv(iydzp, -3, dx, False)
    return jnp.stack((gx, gy, gz), axis=1)


def periodic_x_staggered_divergence(
    velocity: Array,
    dx: float,
    dy: float,
    dz: float,
    implementation: str = "dense",
) -> Array:
    """Divergence for periodic x/z and nonperiodic y precursor domains."""

    dxu = _periodic_staggered(
        velocity[:, 0], -3, dx, True, True
    )
    ixv = _periodic_staggered(
        velocity[:, 1], -3, dx, False, True
    )
    ixw = _periodic_staggered(
        velocity[:, 2], -3, dx, False, True
    )
    if implementation == "cute_fused_41":
        from cute_staggered_y import nonperiodic_vp

        dxy = nonperiodic_vp(dxu, -2, dy, False) + nonperiodic_vp(
            ixv, -2, dy, True
        )
        ixyw = nonperiodic_vp(ixw, -2, dy, False)
    elif implementation == "dense":
        dxy = apply_matrix(
            dxu, nonperiodic_vp_matrix(velocity.shape[-2], dy, False), -2
        ) + apply_matrix(
            ixv, nonperiodic_vp_matrix(velocity.shape[-2], dy, True), -2
        )
        ixyw = apply_matrix(
            ixw, nonperiodic_vp_matrix(velocity.shape[-2], dy, False), -2
        )
    else:
        raise ValueError(f"unknown periodic staggered y implementation: {implementation}")
    return _periodic_staggered(
        dxy, -1, dz, False, True
    ) + _periodic_staggered(ixyw, -1, dz, True, True)


def periodic_x_staggered_gradient(
    pressure: Array,
    dx: float,
    dy: float,
    dz: float,
    implementation: str = "dense",
) -> Array:
    """Pressure gradient for the periodic-x precursor pressure mesh."""

    nym = pressure.shape[-2]
    izp = _periodic_staggered(pressure, -1, dz, False, False)
    dzp = _periodic_staggered(pressure, -1, dz, True, False)
    if implementation == "cute_fused_41":
        from cute_staggered_y import nonperiodic_pv

        iyizp = nonperiodic_pv(izp, -2, dy, False)
        dyizp = nonperiodic_pv(izp, -2, dy, True)
        iydzp = nonperiodic_pv(dzp, -2, dy, False)
    elif implementation == "dense":
        iyizp = apply_matrix(
            izp, nonperiodic_pv_matrix(nym, dy, False), -2
        )
        dyizp = apply_matrix(
            izp, nonperiodic_pv_matrix(nym, dy, True), -2
        )
        iydzp = apply_matrix(
            dzp, nonperiodic_pv_matrix(nym, dy, False), -2
        )
    else:
        raise ValueError(f"unknown periodic staggered y implementation: {implementation}")
    gx = _periodic_staggered(iyizp, -3, dx, True, False)
    gy = _periodic_staggered(dyizp, -3, dx, False, False)
    gz = _periodic_staggered(iydzp, -3, dx, False, False)
    return jnp.stack((gx, gy, gz), axis=1)


def _modified_derivative(theta: Array, spacing: float) -> Array:
    numerator = (
        2.0 * (DERIVATIVE_A / spacing) * jnp.sin(0.5 * theta)
        + 2.0 * (DERIVATIVE_B / spacing) * jnp.sin(1.5 * theta)
    )
    return numerator / (1.0 + 2.0 * DERIVATIVE_ALPHA * jnp.cos(theta))


def _modified_interpolation(theta: Array) -> Array:
    numerator = (
        2.0 * INTERPOLATION_A * jnp.cos(0.5 * theta)
        + 2.0 * INTERPOLATION_B * jnp.cos(1.5 * theta)
    )
    return numerator / (
        1.0 + 2.0 * INTERPOLATION_ALPHA * jnp.cos(theta)
    )


def solve_pressure_poisson(
    divergence: Array, dx: float, dy: float, dz: float
) -> Array:
    """Solve the public ``poisson_11x`` spectral operator."""

    nxm, nym, nz = divergence.shape[-3:]
    dtype = divergence.dtype
    tx = jnp.pi * jnp.arange(nxm, dtype=dtype) / nxm
    ty = jnp.pi * jnp.arange(nym, dtype=dtype) / nym
    tz = 2.0 * jnp.pi * jnp.fft.rfftfreq(nz).astype(dtype)

    kx = _modified_derivative(tx, dx)
    ky = _modified_derivative(ty, dy)
    kz = _modified_derivative(tz, dz)
    qx = _modified_interpolation(tx)
    qy = _modified_interpolation(ty)
    qz = _modified_interpolation(tz)
    eigenvalue = (
        kx[:, None, None] ** 2
        * (qy[None, :, None] * qz[None, None, :]) ** 2
        + ky[None, :, None] ** 2
        * (qx[:, None, None] * qz[None, None, :]) ** 2
        + kz[None, None, :] ** 2
        * (qx[:, None, None] * qy[None, :, None]) ** 2
    )

    transformed_xy = jsp_fft.dctn(
        divergence, type=2, axes=(-3, -2), norm="ortho"
    )
    transformed = jnp.fft.rfft(transformed_xy, axis=-1)
    resolved_mode = eigenvalue > jnp.asarray(1.0e-14, dtype=dtype)
    safe_eigenvalue = jnp.where(resolved_mode, eigenvalue, 1.0)
    pressure_hat = jnp.where(
        resolved_mode,
        -transformed / safe_eigenvalue,
        0.0,
    )
    pressure_xy = jnp.fft.irfft(pressure_hat, n=nz, axis=-1)
    return jsp_fft.idctn(
        pressure_xy, type=2, axes=(-3, -2), norm="ortho"
    )


def solve_periodic_x_pressure_poisson(
    divergence: Array,
    dx: float,
    dy: float,
    dz: float,
    implementation: str = "sequential",
) -> Array:
    """Solve Incompact3d's periodic-x/z, nonperiodic-y Poisson operator."""

    nx, nym, nz = divergence.shape[-3:]
    dtype = divergence.dtype
    tx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx).astype(dtype)
    ty = jnp.pi * jnp.arange(nym, dtype=dtype) / nym
    tz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz).astype(dtype)

    kx = _modified_derivative(tx, dx)
    ky = _modified_derivative(ty, dy)
    kz = _modified_derivative(tz, dz)
    qx = _modified_interpolation(tx)
    qy = _modified_interpolation(ty)
    qz = _modified_interpolation(tz)
    eigenvalue = (
        kx[:, None, None] ** 2
        * (qy[None, :, None] * qz[None, None, :]) ** 2
        + ky[None, :, None] ** 2
        * (qx[:, None, None] * qz[None, None, :]) ** 2
        + kz[None, None, :] ** 2
        * (qx[:, None, None] * qy[None, :, None]) ** 2
    )

    transformed_y = jsp_fft.dct(
        divergence, type=2, axis=-2, norm="ortho"
    )
    if implementation == "sequential":
        transformed = jnp.fft.fft(
            jnp.fft.fft(transformed_y, axis=-3), axis=-1
        )
    elif implementation == "fftn":
        transformed = jnp.fft.fftn(transformed_y, axes=(-3, -1))
    else:
        raise ValueError(
            f"unknown periodic pressure FFT implementation: {implementation}"
        )
    resolved_mode = eigenvalue > jnp.asarray(1.0e-14, dtype=dtype)
    safe_eigenvalue = jnp.where(resolved_mode, eigenvalue, 1.0)
    pressure_hat = jnp.where(
        resolved_mode,
        -transformed / safe_eigenvalue,
        0.0,
    )
    if implementation == "sequential":
        pressure_y = jnp.fft.ifft(
            jnp.fft.ifft(pressure_hat, axis=-1), axis=-3
        ).real
    else:
        pressure_y = jnp.fft.ifftn(
            pressure_hat, axes=(-3, -1)
        ).real
    return jsp_fft.idct(pressure_y, type=2, axis=-2, norm="ortho")


def remove_reference_divergence(divergence: Array) -> Array:
    """Apply Incompact3d's open-boundary divergence diagnostic gauge."""

    return divergence - divergence[..., :1, :1, -1:]
