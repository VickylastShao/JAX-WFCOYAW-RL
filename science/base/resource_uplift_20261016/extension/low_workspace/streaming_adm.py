"""Full-support grouped ADM source for an isolated, frozen LES comparison.

No compact/spatial operator, physical source support, probe or state evolution is
changed. Weights are recomputed inside each LES substep. The only algebraic
reordering is the turbine-axis sum, so floating-point equality is measured, not
assumed. This module never modifies the imported frozen source file.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np


def streaming_disk_source(velocity, yaw_degrees, previous_filtered_speed,
                          first_step, case, group_size):
    """Return the same four ADM quantities with O(batch * group * grid) weights."""
    dtype = velocity.dtype
    scalar = np.float64 if dtype == jnp.float64 else np.float32
    batch, turbines = yaw_degrees.shape
    padded = ((turbines + group_size - 1) // group_size) * group_size
    positions = jnp.asarray(case.turbine_positions_m, dtype=dtype)
    positions = jnp.pad(positions, ((0, padded - turbines), (0, 0)))
    yaws = jnp.pad(yaw_degrees, ((0, 0), (0, padded - turbines)))
    previous = jnp.pad(previous_filtered_speed,
                       ((0, 0), (0, padded - turbines)))
    x_grid = (jnp.arange(case.nx, dtype=dtype) * scalar(case.dx))[None, None, :, None, None]
    y_grid = (jnp.arange(case.ny, dtype=dtype) * scalar(case.dy))[None, None, None, :, None]
    z_grid = (jnp.arange(case.nz, dtype=dtype) * scalar(case.dz))[None, None, None, None, :]
    alpha = scalar((case.dt / case.relaxation_time) /
                   (1.0 + case.dt / case.relaxation_time))
    ct_prime = scalar(case.thrust_coefficient / (1.0 - case.induction_factor) ** 2)
    disk_area = scalar(np.pi * case.diameter ** 2 / 4.0)
    cell_volume = scalar(case.dx * case.dy * case.dz)
    empty = jnp.zeros((batch, padded), dtype=dtype)

    def group(index, carry):
        acceleration, speeds, powers, thrusts = carry
        start = index * group_size
        position = jax.lax.dynamic_slice(positions, (start, 0), (group_size, 3))
        yaw_degrees_group = jax.lax.dynamic_slice(yaws, (0, start), (batch, group_size))
        old = jax.lax.dynamic_slice(previous, (0, start), (batch, group_size))
        yaw = jnp.deg2rad(yaw_degrees_group)[:, :, None, None, None]
        normal_x = jnp.cos(yaw)
        normal_z = -jnp.sin(yaw)
        delta_x = x_grid - position[None, :, 0, None, None, None]
        delta_y = y_grid - position[None, :, 1, None, None, None]
        delta_z = z_grid - position[None, :, 2, None, None, None]
        delta_z = jnp.mod(delta_z + scalar(0.5 * case.lz), scalar(case.lz)) - scalar(0.5 * case.lz)
        normal_distance = delta_x * normal_x + delta_z * normal_z
        projected_x = delta_x - normal_distance * normal_x
        projected_z = delta_z - normal_distance * normal_z
        radial_distance = jnp.sqrt(projected_x ** 2 + delta_y ** 2 + projected_z ** 2)
        normal_spacing = jnp.sqrt((scalar(case.dx) * normal_x) ** 2 +
                                  (scalar(case.dz) * normal_z) ** 2)
        thickness = jnp.maximum(scalar(case.diameter / 8.0), 1.5 * normal_spacing)
        weights = jnp.exp(-(normal_distance / (0.5 * thickness)) ** 2 -
                          (radial_distance / scalar(0.5 * case.diameter)) ** 8)
        weights = weights / jnp.sum(weights, axis=(-3, -2, -1), keepdims=True)
        # Last padded group remains numerically well defined before masking.
        valid = (start + jnp.arange(group_size)) < turbines
        weights = weights * valid[None, :, None, None, None]
        yaw2 = jnp.deg2rad(yaw_degrees_group)
        normal = jnp.stack((jnp.cos(yaw2), jnp.zeros_like(yaw2), -jnp.sin(yaw2)), axis=2)
        normal_velocity = jnp.sum(velocity[:, None] * normal[:, :, :, None, None, None], axis=2)
        disk_speed = jnp.sum(weights * normal_velocity, axis=(-3, -2, -1))
        filtered = jnp.where(first_step, disk_speed,
                             alpha * disk_speed + (1.0 - alpha) * old)
        thrust = scalar(0.5 * case.density) * ct_prime * filtered ** 2 * disk_area
        power = thrust * filtered
        group_acceleration = -jnp.sum(
            (thrust[:, :, None, None, None, None] / scalar(case.density * cell_volume)) *
            weights[:, :, None] * normal[:, :, :, None, None, None], axis=1)
        acceleration = acceleration + group_acceleration
        speeds = jax.lax.dynamic_update_slice(speeds, filtered, (0, start))
        powers = jax.lax.dynamic_update_slice(powers, power, (0, start))
        thrusts = jax.lax.dynamic_update_slice(thrusts, thrust, (0, start))
        return acceleration, speeds, powers, thrusts

    acceleration, speeds, powers, thrusts = jax.lax.fori_loop(
        0, padded // group_size, group,
        (jnp.zeros_like(velocity), empty, empty, empty))
    return acceleration, speeds[:, :turbines], powers[:, :turbines], thrusts[:, :turbines]


def build_streaming_flow_functions(core, case, *, group_size, periodic_x=False,
                                   include_turbines=True):
    """Privately compile the frozen factory after replacing only two closures.

    Reusing all other original AST nodes preserves the full advance operator,
    including pre-filter source sampling and pressure-boundary histories.
    """
    if not isinstance(group_size, int) or not 1 <= group_size <= case.num_turbines:
        raise ValueError("group_size must be an integer in [1, num_turbines]")
    original_text = textwrap.dedent(inspect.getsource(core.build_flow_functions))
    original = ast.parse(original_text)
    factory = original.body[0]
    replacements = {
        "disk_weights": ast.parse("def disk_weights(yaw_degrees):\n    return yaw_degrees\n").body[0],
        "disk_source": ast.parse(
            "def disk_source(velocity, yaw_degrees, weights, previous_filtered_speed, first_step):\n"
            "    return _streaming_source(velocity, yaw_degrees, previous_filtered_speed, first_step, case, _streaming_group)\n"
        ).body[0],
    }
    found = []
    unchanged = {}
    for i, node in enumerate(factory.body):
        if isinstance(node, ast.FunctionDef):
            if node.name in replacements:
                found.append(node.name)
                factory.body[i] = replacements[node.name]
            else:
                unchanged[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if sorted(found) != ["disk_source", "disk_weights"]:
        raise ValueError("unexpected frozen factory: could not identify both ADM closures")
    ast.fix_missing_locations(original)
    namespace = dict(core.__dict__)
    namespace.update(_streaming_source=streaming_disk_source, _streaming_group=group_size)
    exec(compile(original, "<private_full_support_streaming_adm>", "exec"), namespace)
    functions = namespace["build_flow_functions"](case, periodic_x, include_turbines)
    functions["streaming_adm_provenance"] = {
        "schema": "full_support_streaming_adm_v1", "group_size": group_size,
        "changed_nested_functions": sorted(found),
        "unchanged_nested_function_ast_sha256": unchanged,
        "original_factory_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
        "source_file_sha256": hashlib.sha256(open(core.__file__, "rb").read()).hexdigest(),
        "candidate_module_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "support": "entire original grid, original periodic-z distance; no geometric truncation",
        "tradeoff": "weights recomputed every LES substep; turbine reduction order changes",
    }
    return functions
