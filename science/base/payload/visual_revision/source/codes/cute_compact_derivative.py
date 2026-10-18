"""Optional CuTe kernel for the fixed 41-point Mole vertical derivative."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
import numpy as np
from cuda.bindings import driver as cuda


LINE_SIZE = 41
THREADS = 128


def _thomas_constants() -> tuple[tuple[float, ...], ...]:
    alpha = np.float32(1.0 / 3.0)
    lower = np.full((LINE_SIZE,), alpha, dtype=np.float32)
    diagonal = np.ones((LINE_SIZE,), dtype=np.float32)
    upper = np.full((LINE_SIZE,), alpha, dtype=np.float32)
    lower[0] = 0.0
    lower[1] = 0.25
    lower[-2] = 0.25
    lower[-1] = 2.0
    upper[0] = 2.0
    upper[1] = 0.25
    upper[-2] = 0.25
    upper[-1] = 0.0
    cprime = np.zeros_like(upper)
    inverse_denominator = np.zeros_like(diagonal)
    inverse_denominator[0] = 1.0
    cprime[0] = upper[0]
    for index in range(1, LINE_SIZE):
        inverse_denominator[index] = 1.0 / (
            diagonal[index] - lower[index] * cprime[index - 1]
        )
        cprime[index] = upper[index] * inverse_denominator[index]
    return tuple(
        tuple(float(value) for value in array)
        for array in (lower, cprime, inverse_denominator)
    )


LOWER, CPRIME, INVERSE_DENOMINATOR = _thomas_constants()


@cute.kernel
def _compact_kernel(
    values: cute.Tensor,
    output: cute.Tensor,
    spacing: float,
    num_lines: int,
):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    line = block_index * block_size + thread_index
    if line < num_lines:
        rhs = cute.make_rmem_tensor((LINE_SIZE,), values.element_type)
        solved = cute.make_rmem_tensor((LINE_SIZE,), values.element_type)
        inv_h = cutlass.Float32(1.0) / spacing
        rhs[0] = inv_h * (
            cutlass.Float32(-2.5) * values[line, 0]
            + cutlass.Float32(2.0) * values[line, 1]
            + cutlass.Float32(0.5) * values[line, 2]
        )
        rhs[1] = cutlass.Float32(0.75) * inv_h * (
            values[line, 2] - values[line, 0]
        )
        for index in cutlass.range_constexpr(2, LINE_SIZE - 2):
            rhs[index] = (
                cutlass.Float32(7.0 / 9.0)
                * inv_h
                * (values[line, index + 1] - values[line, index - 1])
                + cutlass.Float32(1.0 / 36.0)
                * inv_h
                * (values[line, index + 2] - values[line, index - 2])
            )
        rhs[LINE_SIZE - 2] = cutlass.Float32(0.75) * inv_h * (
            values[line, LINE_SIZE - 1] - values[line, LINE_SIZE - 3]
        )
        rhs[LINE_SIZE - 1] = inv_h * (
            cutlass.Float32(2.5) * values[line, LINE_SIZE - 1]
            - cutlass.Float32(2.0) * values[line, LINE_SIZE - 2]
            - cutlass.Float32(0.5) * values[line, LINE_SIZE - 3]
        )

        solved[0] = rhs[0] * cutlass.Float32(INVERSE_DENOMINATOR[0])
        for index in cutlass.range_constexpr(1, LINE_SIZE):
            solved[index] = (
                rhs[index]
                - cutlass.Float32(LOWER[index]) * solved[index - 1]
            ) * cutlass.Float32(INVERSE_DENOMINATOR[index])
        output[line, LINE_SIZE - 1] = solved[LINE_SIZE - 1]
        for index in cutlass.range_constexpr(LINE_SIZE - 2, -1, -1):
            solved[index] = solved[index] - cutlass.Float32(CPRIME[index]) * solved[
                index + 1
            ]
            output[line, index] = solved[index]


@cute.jit
def _launch(
    stream: cuda.CUstream,
    values: cute.Tensor,
    output: cute.Tensor,
    *,
    spacing: float,
    num_lines: int,
):
    _compact_kernel(values, output, spacing, num_lines).launch(
        grid=[(num_lines + THREADS - 1) // THREADS, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


def compact_first_derivative_41_last_axis(
    values: jax.Array, spacing: float
) -> jax.Array:
    """Differentiate rows whose final dimension is the fixed 41-point line."""

    if values.shape[-1] != LINE_SIZE:
        raise ValueError(f"CuTe compact kernel requires size {LINE_SIZE}")
    flat = jnp.reshape(values, (-1, LINE_SIZE))
    call = cjax.cutlass_call(
        _launch,
        output_shape_dtype=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        use_static_tensors=True,
        spacing=float(spacing),
        num_lines=flat.shape[0],
    )
    return jnp.reshape(call(flat), values.shape)
