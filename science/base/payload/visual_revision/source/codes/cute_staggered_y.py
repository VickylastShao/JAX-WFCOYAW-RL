"""Optional CuTe kernels for Mole's fixed vertical VP/PV operators."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
import numpy as np
from cuda.bindings import driver as cuda


THREADS = 128
VP_INPUT = 41
VP_OUTPUT = 40
PV_INPUT = 40
PV_OUTPUT = 41
DERIVATIVE_ALPHA = 9.0 / 62.0
DERIVATIVE_A = 63.0 / 62.0
DERIVATIVE_B = 17.0 / 186.0
INTERPOLATION_ALPHA = 3.0 / 10.0
INTERPOLATION_A = 3.0 / 4.0
INTERPOLATION_B = 1.0 / 20.0


def _thomas_constants(
    size: int,
    alpha: float,
    *,
    endpoint_diagonal: float,
    first_upper: float,
    last_lower: float,
) -> tuple[tuple[float, ...], ...]:
    lower = np.full((size,), alpha, dtype=np.float32)
    diagonal = np.ones((size,), dtype=np.float32)
    upper = np.full((size,), alpha, dtype=np.float32)
    lower[0] = 0.0
    upper[-1] = 0.0
    diagonal[0] = endpoint_diagonal
    diagonal[-1] = endpoint_diagonal
    upper[0] = first_upper
    lower[-1] = last_lower
    cprime = np.zeros_like(upper)
    inverse_denominator = np.zeros_like(diagonal)
    inverse_denominator[0] = 1.0 / diagonal[0]
    cprime[0] = upper[0] * inverse_denominator[0]
    for index in range(1, size):
        inverse_denominator[index] = 1.0 / (
            diagonal[index] - lower[index] * cprime[index - 1]
        )
        cprime[index] = upper[index] * inverse_denominator[index]
    return tuple(
        tuple(float(value) for value in array)
        for array in (lower, cprime, inverse_denominator)
    )


VP_DERIVATIVE = _thomas_constants(
    VP_OUTPUT,
    DERIVATIVE_ALPHA,
    endpoint_diagonal=1.0 + DERIVATIVE_ALPHA,
    first_upper=DERIVATIVE_ALPHA,
    last_lower=DERIVATIVE_ALPHA,
)
VP_INTERPOLATION = _thomas_constants(
    VP_OUTPUT,
    INTERPOLATION_ALPHA,
    endpoint_diagonal=1.0 + INTERPOLATION_ALPHA,
    first_upper=INTERPOLATION_ALPHA,
    last_lower=INTERPOLATION_ALPHA,
)
PV_DERIVATIVE = _thomas_constants(
    PV_OUTPUT,
    DERIVATIVE_ALPHA,
    endpoint_diagonal=1.0,
    first_upper=0.0,
    last_lower=0.0,
)
PV_INTERPOLATION = _thomas_constants(
    PV_OUTPUT,
    INTERPOLATION_ALPHA,
    endpoint_diagonal=1.0,
    first_upper=2.0 * INTERPOLATION_ALPHA,
    last_lower=2.0 * INTERPOLATION_ALPHA,
)


@cute.kernel
def _vp_kernel(
    values: cute.Tensor,
    output: cute.Tensor,
    spacing: float,
    num_lines: int,
    derivative: cutlass.Constexpr[bool],
):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    line = block_index * block_size + thread_index
    if line < num_lines:
        rhs = cute.make_rmem_tensor((VP_OUTPUT,), values.element_type)
        solved = cute.make_rmem_tensor((VP_OUTPUT,), values.element_type)
        if cutlass.const_expr(derivative):
            alpha_a = cutlass.Float32(DERIVATIVE_A) / spacing
            alpha_b = cutlass.Float32(DERIVATIVE_B) / spacing
            lower, cprime, inverse = VP_DERIVATIVE
            rhs[0] = alpha_a * (values[line, 1] - values[line, 0]) + alpha_b * (
                values[line, 2]
                - cutlass.Float32(2.0) * values[line, 0]
                + values[line, 1]
            )
            rhs[1] = alpha_a * (values[line, 2] - values[line, 1]) + alpha_b * (
                values[line, 3] - values[line, 0]
            )
            for index in cutlass.range_constexpr(2, VP_OUTPUT - 2):
                rhs[index] = alpha_a * (
                    values[line, index + 1] - values[line, index]
                ) + alpha_b * (
                    values[line, index + 2] - values[line, index - 1]
                )
            rhs[VP_OUTPUT - 2] = alpha_a * (
                values[line, VP_INPUT - 2] - values[line, VP_INPUT - 3]
            ) + alpha_b * (
                values[line, VP_INPUT - 1] - values[line, VP_INPUT - 4]
            )
            rhs[VP_OUTPUT - 1] = alpha_a * (
                values[line, VP_INPUT - 1] - values[line, VP_INPUT - 2]
            ) + alpha_b * (
                cutlass.Float32(2.0) * values[line, VP_INPUT - 1]
                - values[line, VP_INPUT - 2]
                - values[line, VP_INPUT - 3]
            )
        else:
            alpha_a = cutlass.Float32(INTERPOLATION_A)
            alpha_b = cutlass.Float32(INTERPOLATION_B)
            lower, cprime, inverse = VP_INTERPOLATION
            rhs[0] = alpha_a * (values[line, 1] + values[line, 0]) + alpha_b * (
                values[line, 2] + values[line, 1]
            )
            rhs[1] = alpha_a * (values[line, 2] + values[line, 1]) + alpha_b * (
                values[line, 3] + values[line, 0]
            )
            for index in cutlass.range_constexpr(2, VP_OUTPUT - 1):
                rhs[index] = alpha_a * (
                    values[line, index + 1] + values[line, index]
                ) + alpha_b * (
                    values[line, index + 2] + values[line, index - 1]
                )
            rhs[VP_OUTPUT - 1] = alpha_a * (
                values[line, VP_INPUT - 1] + values[line, VP_INPUT - 2]
            ) + alpha_b * (
                values[line, VP_INPUT - 2] + values[line, VP_INPUT - 3]
            )
        solved[0] = rhs[0] * cutlass.Float32(inverse[0])
        for index in cutlass.range_constexpr(1, VP_OUTPUT):
            solved[index] = (
                rhs[index] - cutlass.Float32(lower[index]) * solved[index - 1]
            ) * cutlass.Float32(inverse[index])
        output[line, VP_OUTPUT - 1] = solved[VP_OUTPUT - 1]
        for index in cutlass.range_constexpr(VP_OUTPUT - 2, -1, -1):
            solved[index] = solved[index] - cutlass.Float32(cprime[index]) * solved[
                index + 1
            ]
            output[line, index] = solved[index]


@cute.kernel
def _pv_kernel(
    values: cute.Tensor,
    output: cute.Tensor,
    spacing: float,
    num_lines: int,
    derivative: cutlass.Constexpr[bool],
):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    line = block_index * block_size + thread_index
    if line < num_lines:
        rhs = cute.make_rmem_tensor((PV_OUTPUT,), values.element_type)
        solved = cute.make_rmem_tensor((PV_OUTPUT,), values.element_type)
        if cutlass.const_expr(derivative):
            alpha_a = cutlass.Float32(DERIVATIVE_A) / spacing
            alpha_b = cutlass.Float32(DERIVATIVE_B) / spacing
            lower, cprime, inverse = PV_DERIVATIVE
            rhs[0] = cutlass.Float32(0.0)
            rhs[1] = alpha_a * (values[line, 1] - values[line, 0]) + alpha_b * (
                values[line, 2] - values[line, 0]
            )
            for index in cutlass.range_constexpr(2, PV_OUTPUT - 2):
                rhs[index] = alpha_a * (
                    values[line, index] - values[line, index - 1]
                ) + alpha_b * (
                    values[line, index + 1] - values[line, index - 2]
                )
            rhs[PV_OUTPUT - 2] = alpha_a * (
                values[line, PV_INPUT - 1] - values[line, PV_INPUT - 2]
            ) + alpha_b * (
                values[line, PV_INPUT - 1] - values[line, PV_INPUT - 3]
            )
            rhs[PV_OUTPUT - 1] = cutlass.Float32(0.0)
        else:
            alpha_a = cutlass.Float32(INTERPOLATION_A)
            alpha_b = cutlass.Float32(INTERPOLATION_B)
            lower, cprime, inverse = PV_INTERPOLATION
            rhs[0] = cutlass.Float32(2.0) * (
                alpha_a * values[line, 0] + alpha_b * values[line, 1]
            )
            rhs[1] = alpha_a * (values[line, 1] + values[line, 0]) + alpha_b * (
                values[line, 2] + values[line, 0]
            )
            for index in cutlass.range_constexpr(2, PV_OUTPUT - 2):
                rhs[index] = alpha_a * (
                    values[line, index] + values[line, index - 1]
                ) + alpha_b * (
                    values[line, index + 1] + values[line, index - 2]
                )
            rhs[PV_OUTPUT - 2] = alpha_a * (
                values[line, PV_INPUT - 1] + values[line, PV_INPUT - 2]
            ) + alpha_b * (
                values[line, PV_INPUT - 1] + values[line, PV_INPUT - 3]
            )
            rhs[PV_OUTPUT - 1] = cutlass.Float32(2.0) * (
                alpha_a * values[line, PV_INPUT - 1]
                + alpha_b * values[line, PV_INPUT - 2]
            )
        solved[0] = rhs[0] * cutlass.Float32(inverse[0])
        for index in cutlass.range_constexpr(1, PV_OUTPUT):
            solved[index] = (
                rhs[index] - cutlass.Float32(lower[index]) * solved[index - 1]
            ) * cutlass.Float32(inverse[index])
        output[line, PV_OUTPUT - 1] = solved[PV_OUTPUT - 1]
        for index in cutlass.range_constexpr(PV_OUTPUT - 2, -1, -1):
            solved[index] = solved[index] - cutlass.Float32(cprime[index]) * solved[
                index + 1
            ]
            output[line, index] = solved[index]


@cute.jit
def _launch_vp(
    stream: cuda.CUstream,
    values: cute.Tensor,
    output: cute.Tensor,
    *,
    spacing: float,
    num_lines: int,
    derivative: cutlass.Constexpr[bool],
):
    _vp_kernel(values, output, spacing, num_lines, derivative).launch(
        grid=[(num_lines + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_pv(
    stream: cuda.CUstream,
    values: cute.Tensor,
    output: cute.Tensor,
    *,
    spacing: float,
    num_lines: int,
    derivative: cutlass.Constexpr[bool],
):
    _pv_kernel(values, output, spacing, num_lines, derivative).launch(
        grid=[(num_lines + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


def _call(values, axis, spacing, derivative, input_size, output_size, launch):
    moved = jnp.moveaxis(values, axis, -1)
    if moved.shape[-1] != input_size:
        raise ValueError(f"expected vertical size {input_size}")
    flat = moved.reshape((-1, input_size))
    output = cjax.cutlass_call(
        launch,
        output_shape_dtype=jax.ShapeDtypeStruct(
            (flat.shape[0], output_size), flat.dtype
        ),
        use_static_tensors=True,
        spacing=float(spacing),
        num_lines=flat.shape[0],
        derivative=bool(derivative),
    )(flat)
    restored = output.reshape(moved.shape[:-1] + (output_size,))
    return jnp.moveaxis(restored, -1, axis)


def nonperiodic_vp(values, axis, spacing, derivative):
    return _call(
        values, axis, spacing, derivative, VP_INPUT, VP_OUTPUT, _launch_vp
    )


def nonperiodic_pv(values, axis, spacing, derivative):
    return _call(
        values, axis, spacing, derivative, PV_INPUT, PV_OUTPUT, _launch_pv
    )
