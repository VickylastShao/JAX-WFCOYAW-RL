#!/usr/bin/env python3
"""Fixed-shape direction-conditioned Graph-SAC for multi-direction N=9 control."""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_mole_compact_sac_benchmark import (
    JAX_REAL_DTYPE,
    NP_REAL_DTYPE,
    Array,
    PyTree,
    ReplayState,
    SACState,
    mlp_apply,
    torchrl_biased_softplus,
)
from jax_mole_graph_sac_v2 import _init_actor, _init_critic
from jax_mole_n9_auditable_replay import replay_bootstrap_discounts


class DirectionalWakeGraph(NamedTuple):
    """Canonical physical geometry with every non-self ordered turbine pair."""

    node_positions_xz_d: Array
    edge_sources: Array
    edge_targets: Array
    edge_displacements_xz_d: Array
    in_degree: Array
    reflection_permutation: Array

    @property
    def num_nodes(self) -> int:
        return int(self.node_positions_xz_d.shape[0])


def append_direction_condition(
    observations: Array | np.ndarray,
    relative_wind_angle_deg: float | Array,
) -> Array:
    """Append `[sin(theta), cos(theta)]` to every observation row."""

    values = jnp.asarray(observations, dtype=JAX_REAL_DTYPE)
    angle = jnp.asarray(relative_wind_angle_deg, dtype=JAX_REAL_DTYPE)
    radians = jnp.deg2rad(angle)
    condition = jnp.stack((jnp.sin(radians), jnp.cos(radians)), axis=-1)
    condition = jnp.broadcast_to(condition, (*values.shape[:-1], 2))
    return jnp.concatenate((values, condition), axis=-1)


def append_direction_branch_condition(
    observations: Array | np.ndarray,
    relative_wind_angle_deg: float | Array,
    steering_branch: float | Array,
) -> Array:
    """Append ``[sin(theta), cos(theta), steering_branch]`` to observations."""

    values = jnp.asarray(observations, dtype=JAX_REAL_DTYPE)
    angle = jnp.asarray(relative_wind_angle_deg, dtype=JAX_REAL_DTYPE)
    branch = jnp.asarray(steering_branch, dtype=JAX_REAL_DTYPE)
    radians = jnp.deg2rad(angle)
    sine, cosine, branch = jnp.broadcast_arrays(
        jnp.sin(radians), jnp.cos(radians), branch
    )
    condition = jnp.stack((sine, cosine, branch), axis=-1)
    condition = jnp.broadcast_to(condition, (*values.shape[:-1], 3))
    return jnp.concatenate((values, condition), axis=-1)


def append_direction_speed_condition(
    observations: Array | np.ndarray,
    relative_wind_angle_deg: float | Array,
    measured_upstream_speed_m_per_s: float | Array,
    *,
    normalization_center_m_per_s: float = 8.0,
    normalization_half_range_m_per_s: float = 2.0,
) -> Array:
    """Append ``[sin(theta), cos(theta), normalized upstream speed]``."""

    if normalization_half_range_m_per_s <= 0.0:
        raise ValueError("wind-speed normalization half range must be positive")
    values = jnp.asarray(observations, dtype=JAX_REAL_DTYPE)
    angle = jnp.asarray(relative_wind_angle_deg, dtype=JAX_REAL_DTYPE)
    speed = jnp.asarray(measured_upstream_speed_m_per_s, dtype=JAX_REAL_DTYPE)
    radians = jnp.deg2rad(angle)
    sine, cosine, speed = jnp.broadcast_arrays(
        jnp.sin(radians), jnp.cos(radians), speed
    )
    normalized_speed = (
        speed - NP_REAL_DTYPE(normalization_center_m_per_s)
    ) / NP_REAL_DTYPE(normalization_half_range_m_per_s)
    condition = jnp.stack((sine, cosine, normalized_speed), axis=-1)
    condition = jnp.broadcast_to(condition, (*values.shape[:-1], 3))
    return jnp.concatenate((values, condition), axis=-1)


def steering_branch_for_shard(
    process_id: int,
    shards_per_process: int,
    shard_index: int,
) -> float:
    """Assign alternating global branches without host-local imbalance."""

    if process_id < 0 or shards_per_process < 1:
        raise ValueError("process and shard counts are invalid")
    if not 0 <= shard_index < shards_per_process:
        raise ValueError("shard index is outside the local shard range")
    global_shard = process_id * shards_per_process + shard_index
    return 1.0 if global_shard % 2 == 0 else -1.0


def build_directional_graph(
    turbine_positions_m: tuple[tuple[float, float, float], ...],
    diameter_m: float,
) -> DirectionalWakeGraph:
    """Build a fixed complete directed graph from canonical physical positions."""

    positions = np.asarray(turbine_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("turbine positions must have shape (N, 3) with N >= 2")
    if diameter_m <= 0.0 or not np.all(np.isfinite(positions)):
        raise ValueError("diameter and turbine positions must be finite and valid")
    horizontal = positions[:, (0, 2)]
    centered = (horizontal - horizontal.mean(axis=0, keepdims=True)) / diameter_m
    sources = []
    targets = []
    displacements = []
    for target in range(positions.shape[0]):
        for source in range(positions.shape[0]):
            if source == target:
                continue
            sources.append(source)
            targets.append(target)
            displacements.append(centered[target] - centered[source])
    degree = np.full((positions.shape[0],), positions.shape[0] - 1, dtype=np.float32)
    reflected = centered.copy()
    reflected[:, 1] *= -1.0
    distances = np.linalg.norm(
        reflected[:, None, :] - centered[None, :, :], axis=-1
    )
    permutation = np.argmin(distances, axis=1)
    if (
        sorted(permutation.tolist()) != list(range(positions.shape[0]))
        or np.max(distances[np.arange(positions.shape[0]), permutation]) > 1.0e-6
        or not np.array_equal(permutation[permutation], np.arange(positions.shape[0]))
    ):
        raise ValueError("canonical turbine layout is not reflection symmetric")
    return DirectionalWakeGraph(
        node_positions_xz_d=jnp.asarray(centered, dtype=JAX_REAL_DTYPE),
        edge_sources=jnp.asarray(sources, dtype=jnp.int32),
        edge_targets=jnp.asarray(targets, dtype=jnp.int32),
        edge_displacements_xz_d=jnp.asarray(displacements, dtype=JAX_REAL_DTYPE),
        in_degree=jnp.asarray(degree, dtype=JAX_REAL_DTYPE),
        reflection_permutation=jnp.asarray(permutation, dtype=jnp.int32),
    )


def mirror_conditioned_observation(
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    lateral_probes: int = 7,
    streamwise_probes: int = 11,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    """Reflect probe/yaw/direction observations across the farm centerline."""

    values = jnp.asarray(observations, dtype=JAX_REAL_DTYPE)
    num_turbines = graph.num_nodes
    probe_width = (
        frame_stack * num_turbines * lateral_probes * streamwise_probes
    )
    base_width = probe_width + num_turbines
    condition_dim = values.shape[-1] - base_width
    if condition_dim not in (2, 3):
        raise ValueError("observation width does not match mirror contract")
    probes = values[..., :probe_width].reshape(
        *values.shape[:-1],
        frame_stack,
        num_turbines,
        lateral_probes,
        streamwise_probes,
    )
    permutation = graph.reflection_permutation
    mirrored_probes = jnp.take(probes, permutation, axis=-3)[..., ::-1, :]
    yaw = values[..., probe_width:base_width]
    mirrored_yaw = -jnp.take(yaw, permutation, axis=-1)
    condition = values[..., base_width:]
    mirrored_direction = jnp.stack(
        (-condition[..., 0], condition[..., 1]), axis=-1
    )
    if condition_scalar_kind not in ("branch", "speed"):
        raise ValueError("condition scalar must be 'branch' or 'speed'")
    if condition_dim == 3:
        scalar = condition[..., 2:3]
        mirrored_scalar = -scalar if condition_scalar_kind == "branch" else scalar
        mirrored_condition = jnp.concatenate(
            (mirrored_direction, mirrored_scalar), axis=-1
        )
    else:
        mirrored_condition = mirrored_direction
    return jnp.concatenate(
        (
            mirrored_probes.reshape(*values.shape[:-1], probe_width),
            mirrored_yaw,
            mirrored_condition,
        ),
        axis=-1,
    )


def mirror_action_values(actions: Array, graph: DirectionalWakeGraph) -> Array:
    """Map turbine-indexed signed yaw actions through the reflection."""

    values = jnp.asarray(actions, dtype=JAX_REAL_DTYPE)
    if values.shape[-1] != graph.num_nodes:
        raise ValueError("action width does not match reflection contract")
    return -jnp.take(values, graph.reflection_permutation, axis=-1)


def _split_conditioned_observation(
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[Array, Array, Array, Array | None]:
    base_width = frame_stack * graph.num_nodes * 7 * 11 + graph.num_nodes
    condition_dim = observations.shape[-1] - base_width
    if condition_dim not in (2, 3):
        raise ValueError(
            "directional observation must contain direction or direction/branch condition"
        )
    if condition_scalar_kind not in ("branch", "speed"):
        raise ValueError("condition scalar must be 'branch' or 'speed'")
    base = observations[..., :base_width]
    condition = observations[..., base_width:]
    sine = condition[..., 0]
    cosine = condition[..., 1]
    norm = jnp.sqrt(sine**2 + cosine**2)
    norm = jnp.maximum(norm, NP_REAL_DTYPE(1.0e-6))
    branch = condition[..., 2] if condition_dim == 3 else None
    return base, sine / norm, cosine / norm, branch


def upstream_velocity_barycenter_d(
    conditioned_observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    upstream_band_d: float = 0.5,
) -> Array:
    """Estimate the upstream-row lateral velocity barycenter in diameters."""

    if frame_stack <= 0 or upstream_band_d <= 0.0:
        raise ValueError("frame stack and upstream band must be positive")
    base, sine, cosine, _ = _split_conditioned_observation(
        conditioned_observations, graph, frame_stack
    )
    num_turbines = graph.num_nodes
    probe_width = frame_stack * num_turbines * 7 * 11
    probes = base[..., :probe_width].reshape(
        *base.shape[:-1], frame_stack, num_turbines, 7, 11
    )
    latest = probes[..., -1, :, :, :]

    # Observations store u/U_infinity - 1. The common U_infinity factor
    # cancels in the barycenter ratio.
    velocity = jnp.maximum(latest + NP_REAL_DTYPE(1.0), NP_REAL_DTYPE(1.0e-6))
    upstream_planes = jnp.take(
        velocity, jnp.asarray((1, 2), dtype=jnp.int32), axis=-1
    )
    lateral_weight = jnp.sum(upstream_planes, axis=-1)
    lateral_coordinate_d = jnp.linspace(
        NP_REAL_DTYPE(-1.0), NP_REAL_DTYPE(1.0), 7, dtype=JAX_REAL_DTYPE
    )
    turbine_denominator = jnp.maximum(
        jnp.sum(lateral_weight, axis=-1), NP_REAL_DTYPE(1.0e-6)
    )
    turbine_barycenter = jnp.sum(
        lateral_weight * lateral_coordinate_d, axis=-1
    ) / turbine_denominator

    positions = graph.node_positions_xz_d
    streamwise = (
        cosine[..., None] * positions[:, 0]
        + sine[..., None] * positions[:, 1]
    )
    first_row = streamwise <= (
        jnp.min(streamwise, axis=-1, keepdims=True)
        + NP_REAL_DTYPE(upstream_band_d)
    )
    row_weight = first_row.astype(JAX_REAL_DTYPE)
    return jnp.sum(turbine_barycenter * row_weight, axis=-1) / jnp.maximum(
        jnp.sum(row_weight, axis=-1), NP_REAL_DTYPE(1.0)
    )


def flow_responsive_steering_branch(
    observations: Array | np.ndarray,
    graph: DirectionalWakeGraph,
    relative_wind_angle_deg: float | Array,
    frame_stack: int = 1,
    direction_deadband_deg: float = 2.5,
    barycenter_deadband_d: float = 0.02,
) -> Array:
    """Select a reflection-odd steering mode from direction, flow and yaw memory."""

    if direction_deadband_deg < 0.0 or barycenter_deadband_d < 0.0:
        raise ValueError("selector deadbands cannot be negative")
    values = jnp.asarray(observations, dtype=JAX_REAL_DTYPE)
    angle = jnp.asarray(relative_wind_angle_deg, dtype=JAX_REAL_DTYPE)
    conditioned = append_direction_condition(values, angle)
    barycenter = upstream_velocity_barycenter_d(
        conditioned, graph, frame_stack=frame_stack
    )

    num_turbines = graph.num_nodes
    probe_width = frame_stack * num_turbines * 7 * 11
    previous_yaw = values[..., probe_width : probe_width + num_turbines]
    yaw_memory = jnp.sign(jnp.sum(previous_yaw, axis=-1))
    flow_branch = jnp.where(
        jnp.abs(barycenter) > NP_REAL_DTYPE(barycenter_deadband_d),
        -jnp.sign(barycenter),
        yaw_memory,
    )
    return jnp.where(
        jnp.abs(angle) > NP_REAL_DTYPE(direction_deadband_deg),
        jnp.sign(angle),
        flow_branch,
    ).astype(JAX_REAL_DTYPE)


def directional_node_observations(
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    """Tokenize probes/yaw and append direction-relative node geometry."""

    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    base, sine, cosine, branch = _split_conditioned_observation(
        observations, graph, frame_stack, condition_scalar_kind
    )
    num_turbines = graph.num_nodes
    probe_width = base.shape[-1] - num_turbines
    divisor = num_turbines * frame_stack
    if probe_width <= 0 or probe_width % divisor:
        raise ValueError("conditioned observation is incompatible with graph/frame stack")
    probes_per_frame = probe_width // divisor
    probes = base[..., :probe_width].reshape(
        *base.shape[:-1], frame_stack, num_turbines, probes_per_frame
    )
    probes = jnp.swapaxes(probes, -3, -2).reshape(
        *base.shape[:-1], num_turbines, frame_stack * probes_per_frame
    )
    yaw = base[..., probe_width:].reshape(*base.shape[:-1], num_turbines, 1)
    positions = graph.node_positions_xz_d
    streamwise = cosine[..., None] * positions[:, 0] + sine[..., None] * positions[:, 1]
    lateral = -sine[..., None] * positions[:, 0] + cosine[..., None] * positions[:, 1]
    direction_sine = jnp.broadcast_to(sine[..., None], streamwise.shape)
    direction_cosine = jnp.broadcast_to(cosine[..., None], streamwise.shape)
    geometry = jnp.stack(
        (streamwise, lateral, direction_sine, direction_cosine), axis=-1
    )
    if branch is not None:
        branch_feature = jnp.broadcast_to(branch[..., None], streamwise.shape)
        geometry = jnp.concatenate(
            (geometry, branch_feature[..., None]), axis=-1
        )
    return jnp.concatenate((probes, yaw, geometry), axis=-1)


def directional_edge_features(
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    """Return sample-specific fixed-shape wake features for all ordered pairs."""

    _, sine, cosine, _ = _split_conditioned_observation(
        observations, graph, frame_stack, condition_scalar_kind
    )
    displacement = graph.edge_displacements_xz_d
    streamwise = cosine[..., None] * displacement[:, 0] + sine[..., None] * displacement[:, 1]
    lateral = -sine[..., None] * displacement[:, 0] + cosine[..., None] * displacement[:, 1]
    distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    inverse_distance = jnp.broadcast_to(
        1.0 / jnp.sqrt(1.0 + distance**2), streamwise.shape
    )
    downstream = (streamwise > NP_REAL_DTYPE(1.0e-8)).astype(JAX_REAL_DTYPE)
    wake_width = NP_REAL_DTYPE(0.5) + NP_REAL_DTYPE(0.08) * jnp.maximum(
        streamwise, NP_REAL_DTYPE(0.0)
    )
    overlap = downstream * jnp.exp(
        -NP_REAL_DTYPE(0.5) * (lateral / wake_width) ** 2
    )
    return jnp.stack(
        (streamwise, lateral, inverse_distance, downstream, overlap), axis=-1
    )


def _directional_graph_body_apply(
    body: PyTree,
    nodes: Array,
    edge_features: Array,
    graph: DirectionalWakeGraph,
) -> Array:
    hidden = jax.nn.relu(nodes @ body["encoder"]["w"] + body["encoder"]["b"])
    for layer in body["layers"]:
        source_hidden = jnp.take(hidden, graph.edge_sources, axis=-2)
        message_input = jnp.concatenate((source_hidden, edge_features), axis=-1)
        messages = message_input @ layer["message"]["w"] + layer["message"]["b"]
        aggregate = jnp.zeros_like(hidden).at[..., graph.edge_targets, :].add(messages)
        degree = jnp.reshape(
            graph.in_degree,
            (1,) * (hidden.ndim - 2) + (graph.num_nodes, 1),
        )
        hidden = jax.nn.relu(
            hidden @ layer["self"]["w"] + layer["self"]["b"] + aggregate / degree
        )
    return hidden


def directional_graph_actor_parameters(
    actor: PyTree,
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[Array, Array]:
    nodes = directional_node_observations(
        observations, graph, frame_stack, condition_scalar_kind
    )
    edge_features = directional_edge_features(
        observations, graph, frame_stack, condition_scalar_kind
    )
    hidden = _directional_graph_body_apply(actor["body"], nodes, edge_features, graph)
    output = hidden @ actor["head"]["w"] + actor["head"]["b"]
    mean, raw_scale = jnp.split(output, 2, axis=-1)
    scale_lower_bound = 0.01 if "neutral_scale_contract" in actor else 0.1
    return mean.squeeze(-1), torchrl_biased_softplus(
        raw_scale, scale_lower_bound=scale_lower_bound
    ).squeeze(-1)


def equivariant_directional_graph_actor_parameters(
    actor: PyTree,
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[Array, Array]:
    """Symmetrize Actor parameters under the exact farm reflection group."""

    mean, scale = directional_graph_actor_parameters(
        actor, observations, graph, frame_stack, condition_scalar_kind
    )
    mirrored_observations = mirror_conditioned_observation(
        observations,
        graph,
        frame_stack,
        condition_scalar_kind=condition_scalar_kind,
    )
    mirrored_mean, mirrored_scale = directional_graph_actor_parameters(
        actor,
        mirrored_observations,
        graph,
        frame_stack,
        condition_scalar_kind,
    )
    reflected_mean = mirror_action_values(mirrored_mean, graph)
    reflected_scale = jnp.take(
        mirrored_scale, graph.reflection_permutation, axis=-1
    )
    return (
        NP_REAL_DTYPE(0.5) * (mean + reflected_mean),
        NP_REAL_DTYPE(0.5) * (scale + reflected_scale),
    )


def _actor_sample_from_parameters(
    mean: Array,
    scale: Array,
    key: Array,
    action_bound: float,
) -> tuple[Array, Array]:
    """Sample a squashed Gaussian after its parameters have been constructed."""

    if action_bound <= 0.0:
        raise ValueError("action_bound must be positive")
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    pre_tanh = mean + scale * noise
    resolution = NP_REAL_DTYPE(np.finfo(NP_REAL_DTYPE).resolution)
    limit = NP_REAL_DTYPE(1.0) - resolution
    normalized = jnp.clip(jnp.tanh(pre_tanh), -limit, limit)
    coordinate = jnp.where(
        jnp.isfinite(pre_tanh) & (jnp.abs(pre_tanh) <= jnp.arctanh(limit)),
        pre_tanh,
        jnp.arctanh(normalized),
    )
    standardized = (coordinate - mean) / scale
    gaussian = -NP_REAL_DTYPE(0.5) * (
        standardized**2 + NP_REAL_DTYPE(math.log(2.0 * math.pi))
    ) - jnp.log(scale)
    correction = NP_REAL_DTYPE(2.0) * (
        NP_REAL_DTYPE(math.log(2.0))
        - coordinate
        - jax.nn.softplus(-NP_REAL_DTYPE(2.0) * coordinate)
    ) + NP_REAL_DTYPE(math.log(action_bound))
    return NP_REAL_DTYPE(action_bound) * normalized, jnp.sum(
        gaussian - correction, axis=-1
    )


def directional_graph_actor_sample(
    actor: PyTree,
    observations: Array,
    key: Array,
    action_bound: float,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[Array, Array]:
    mean, scale = directional_graph_actor_parameters(
        actor, observations, graph, frame_stack, condition_scalar_kind
    )
    return _actor_sample_from_parameters(mean, scale, key, action_bound)


def equivariant_directional_graph_actor_sample(
    actor: PyTree,
    observations: Array,
    key: Array,
    action_bound: float,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[Array, Array]:
    mean, scale = equivariant_directional_graph_actor_parameters(
        actor, observations, graph, frame_stack, condition_scalar_kind
    )
    return _actor_sample_from_parameters(mean, scale, key, action_bound)


def directional_graph_actor_deterministic(
    actor: PyTree,
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    mean, _ = directional_graph_actor_parameters(
        actor, observations, graph, frame_stack, condition_scalar_kind
    )
    return jnp.tanh(mean)


def equivariant_directional_graph_actor_deterministic(
    actor: PyTree,
    observations: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    mean, _ = equivariant_directional_graph_actor_parameters(
        actor, observations, graph, frame_stack, condition_scalar_kind
    )
    return jnp.tanh(mean)


def _critic_one(
    critic: PyTree,
    observations: Array,
    actions: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int,
    condition_scalar_kind: Literal["branch", "speed"],
) -> Array:
    nodes = directional_node_observations(
        observations, graph, frame_stack, condition_scalar_kind
    )
    edge_features = directional_edge_features(
        observations, graph, frame_stack, condition_scalar_kind
    )
    nodes = jnp.concatenate((actions[..., None], nodes), axis=-1)
    hidden = _directional_graph_body_apply(critic["body"], nodes, edge_features, graph)
    node_value = jnp.mean(
        mlp_apply(critic["node_head"], hidden, activation="relu").squeeze(-1), axis=-1
    )
    global_value = mlp_apply(
        critic["global_head"], jnp.mean(hidden, axis=-2), activation="relu"
    ).squeeze(-1)
    source_hidden = jnp.take(hidden, graph.edge_sources, axis=-2)
    target_hidden = jnp.take(hidden, graph.edge_targets, axis=-2)
    source_action = jnp.take(actions, graph.edge_sources, axis=-1)[..., None]
    target_action = jnp.take(actions, graph.edge_targets, axis=-1)[..., None]
    edge_inputs = jnp.concatenate(
        (
            source_hidden,
            target_hidden,
            source_action,
            target_action,
            edge_features,
        ),
        axis=-1,
    )
    edge_value = jnp.mean(
        mlp_apply(critic["edge_head"], edge_inputs, activation="relu").squeeze(-1),
        axis=-1,
    )
    return (node_value + edge_value + global_value) / NP_REAL_DTYPE(3.0)


def directional_graph_critic_apply(
    critic: PyTree,
    observations: Array,
    actions: Array,
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> Array:
    return jnp.stack(
        tuple(
            _critic_one(
                critic[name],
                observations,
                actions,
                graph,
                frame_stack,
                condition_scalar_kind,
            )
            for name in ("q1", "q2")
        ),
        axis=-1,
    )


def init_directional_graph_sac(
    key: Array,
    observation_dim: int,
    graph: DirectionalWakeGraph,
    hidden_dim: int,
    learning_rate: float,
    *,
    frame_stack: int = 1,
    message_layers: int = 2,
    conservative: bool = False,
    neutral_initial_scale: float | None = None,
    initial_alpha: float = 0.1,
    entropy_learning_rate: float = 1.0e-5,
    actor_learning_rate_scale: float = 1.0,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[SACState, tuple[Any, Any, Any]]:
    num_turbines = graph.num_nodes
    condition_candidates = []
    for condition_dim in (2, 3):
        base_observation_dim = observation_dim - condition_dim
        if (
            base_observation_dim > num_turbines
            and (base_observation_dim - num_turbines)
            % (num_turbines * frame_stack)
            == 0
        ):
            condition_candidates.append((condition_dim, base_observation_dim))
    if len(condition_candidates) != 1:
        raise ValueError(
            "conditioned observation is incompatible with graph/frame stack"
        )
    condition_dim, base_observation_dim = condition_candidates[0]
    if condition_scalar_kind not in ("branch", "speed"):
        raise ValueError("condition scalar must be 'branch' or 'speed'")
    if initial_alpha <= 0.0 or actor_learning_rate_scale <= 0.0:
        raise ValueError("alpha and Actor learning-rate scale must be positive")
    raw_node_dim = (base_observation_dim - num_turbines) // num_turbines + 1
    node_dim = raw_node_dim + 4 + int(condition_dim == 3)
    edge_dim = 5
    actor_key, q1_key, q2_key = jax.random.split(key, 3)
    actor = _init_actor(
        actor_key,
        node_dim,
        edge_dim,
        hidden_dim,
        message_layers,
        conservative,
        neutral_initial_scale,
    )
    critic = {
        "q1": _init_critic(q1_key, node_dim, edge_dim, hidden_dim, message_layers),
        "q2": _init_critic(q2_key, node_dim, edge_dim, hidden_dim, message_layers),
    }
    target_critic = jax.tree.map(jnp.copy, critic)
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate * actor_learning_rate_scale),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adam(learning_rate)
    )
    alpha_optimizer = optax.adam(entropy_learning_rate)
    log_alpha = jnp.asarray(math.log(initial_alpha), dtype=JAX_REAL_DTYPE)
    return SACState(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        log_alpha=log_alpha,
        actor_opt=actor_optimizer.init(actor),
        critic_opt=critic_optimizer.init(critic),
        alpha_opt=alpha_optimizer.init(log_alpha),
        updates=jnp.asarray(0, dtype=jnp.int32),
    ), (actor_optimizer, critic_optimizer, alpha_optimizer)


def directional_graph_sac_update(
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
    graph: DirectionalWakeGraph,
    frame_stack: int = 1,
    axis_name: str | None = None,
    reward_scale: float = 1.0,
    actor_update_period: int = 1,
    actor_warmup_updates: int = 0,
    actor_minimum_replay_size: int = 0,
    action_l2_coefficient: float = 0.0,
    reflection_equivariant_actor: bool = False,
    condition_scalar_kind: Literal["branch", "speed"] = "branch",
) -> tuple[SACState, dict[str, Array]]:
    """One direction-conditioned Graph-SAC update."""

    if not 0.0 <= entropy_scale <= 1.0:
        raise ValueError("entropy_scale must be in [0, 1]")
    if (
        actor_update_period < 0
        or actor_warmup_updates < 0
        or actor_minimum_replay_size < 0
    ):
        raise ValueError("Actor update controls cannot be negative")
    if action_l2_coefficient < 0.0:
        raise ValueError("action_l2_coefficient cannot be negative")
    actor_optimizer, critic_optimizer, alpha_optimizer = optimizers
    sample_key, target_key, actor_key = jax.random.split(key, 3)
    indices = jax.random.randint(sample_key, (sample_size,), 0, jnp.maximum(replay.size, 1))
    observations = replay.observations[indices]
    actions = replay.actions[indices]
    rewards = replay.rewards[indices]
    next_observations = replay.next_observations[indices]
    dones = replay.dones[indices]
    alpha = jnp.exp(state.log_alpha)
    actor_sample = (
        equivariant_directional_graph_actor_sample
        if reflection_equivariant_actor
        else directional_graph_actor_sample
    )
    next_actions, next_log_probability = actor_sample(
        state.actor,
        next_observations,
        target_key,
        action_bound,
        graph,
        frame_stack,
        condition_scalar_kind,
    )
    target_q = jnp.min(
        directional_graph_critic_apply(
            state.target_critic,
            next_observations,
            next_actions,
            graph,
            frame_stack,
            condition_scalar_kind,
        ),
        axis=-1,
    )
    bootstrap_discounts = replay_bootstrap_discounts(
        replay, indices, gamma=gamma, sampled_dones=dones
    )
    target = jax.lax.stop_gradient(
        NP_REAL_DTYPE(reward_scale) * rewards
        + bootstrap_discounts
        * (target_q - alpha * NP_REAL_DTYPE(entropy_scale) * next_log_probability)
    )

    def critic_loss_fn(parameters: PyTree) -> Array:
        values = directional_graph_critic_apply(
            parameters,
            observations,
            actions,
            graph,
            frame_stack,
            condition_scalar_kind,
        )
        return jnp.mean(jnp.sum((values - target[:, None]) ** 2, axis=-1))

    critic_loss, critic_gradient = jax.value_and_grad(critic_loss_fn)(state.critic)
    if axis_name is not None:
        critic_gradient = jax.lax.pmean(critic_gradient, axis_name)
    critic_updates, critic_opt = critic_optimizer.update(
        critic_gradient, state.critic_opt, state.critic
    )
    critic = optax.apply_updates(state.critic, critic_updates)

    def actor_loss_fn(actor: PyTree):
        sampled_actions, log_probability = actor_sample(
            actor,
            observations,
            actor_key,
            action_bound,
            graph,
            frame_stack,
            condition_scalar_kind,
        )
        q_min = jnp.min(
            directional_graph_critic_apply(
                state.critic,
                observations,
                sampled_actions,
                graph,
                frame_stack,
                condition_scalar_kind,
            ),
            axis=-1,
        )
        action_l2_mean = jnp.mean((sampled_actions / NP_REAL_DTYPE(action_bound)) ** 2)
        loss = jnp.mean(
            jax.lax.stop_gradient(alpha)
            * NP_REAL_DTYPE(entropy_scale)
            * log_probability
            - q_min
        ) + NP_REAL_DTYPE(action_l2_coefficient) * action_l2_mean
        return loss, (log_probability, q_min, sampled_actions, action_l2_mean)

    after_warmup = (
        (state.updates + 1 > actor_warmup_updates)
        & (replay.size > actor_minimum_replay_size)
    )
    actor_updated = (
        jnp.asarray(False)
        if actor_update_period == 0
        else after_warmup
        & ((state.updates + 1 - actor_warmup_updates) % actor_update_period == 0)
    )

    def update_actor_and_alpha(_):
        (actor_loss, actor_aux), actor_gradient = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor)
        log_probability, actor_q, sampled_actions, action_l2_mean = actor_aux
        if axis_name is not None:
            actor_gradient = jax.lax.pmean(actor_gradient, axis_name)
        actor_updates, actor_opt = actor_optimizer.update(
            actor_gradient, state.actor_opt, state.actor
        )
        actor = optax.apply_updates(state.actor, actor_updates)
        alpha_loss = -jnp.mean(
            state.log_alpha
            * jax.lax.stop_gradient(
                NP_REAL_DTYPE(entropy_scale)
                * (log_probability + NP_REAL_DTYPE(target_entropy))
            )
        )
        alpha_gradient = jax.grad(
            lambda value: -jnp.mean(
                value
                * jax.lax.stop_gradient(
                    NP_REAL_DTYPE(entropy_scale)
                    * (log_probability + NP_REAL_DTYPE(target_entropy))
                )
            )
        )(state.log_alpha)
        if axis_name is not None:
            alpha_gradient = jax.lax.pmean(alpha_gradient, axis_name)
        alpha_updates, alpha_opt = alpha_optimizer.update(
            alpha_gradient, state.alpha_opt, state.log_alpha
        )
        log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)
        diagnostics = (
            actor_loss,
            alpha_loss,
            optax.global_norm(actor_gradient),
            jnp.mean(actor_q),
            jnp.mean(
                jax.lax.stop_gradient(alpha)
                * NP_REAL_DTYPE(entropy_scale)
                * log_probability
            ),
            action_l2_mean,
            jnp.mean(jnp.abs(sampled_actions)),
            jnp.max(jnp.abs(sampled_actions)),
        )
        return actor, actor_opt, log_alpha, alpha_opt, diagnostics

    def skip_actor_and_alpha(_):
        zero = jnp.asarray(0.0, dtype=NP_REAL_DTYPE)
        return (
            state.actor,
            state.actor_opt,
            state.log_alpha,
            state.alpha_opt,
            (zero, zero, zero, zero, zero, zero, zero, zero),
        )

    actor, actor_opt, log_alpha, alpha_opt, actor_diagnostics = jax.lax.cond(
        actor_updated,
        update_actor_and_alpha,
        skip_actor_and_alpha,
        operand=None,
    )
    (
        actor_loss,
        alpha_loss,
        actor_gradient_norm,
        actor_q_mean,
        entropy_term_mean,
        action_l2_mean,
        sampled_action_abs_mean,
        sampled_action_abs_max,
    ) = actor_diagnostics
    target_critic = jax.tree.map(
        lambda target_value, value: (
            (1.0 - NP_REAL_DTYPE(tau)) * target_value + NP_REAL_DTYPE(tau) * value
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
        "critic_gradient_norm": optax.global_norm(critic_gradient),
        "target_q_mean": jnp.mean(target_q),
        "target_q_std": jnp.std(target_q),
        "critic_target_mean": jnp.mean(target),
        "critic_target_std": jnp.std(target),
        "bootstrap_discount_mean": jnp.mean(bootstrap_discounts),
        "actor_q_mean": actor_q_mean,
        "entropy_term_mean": entropy_term_mean,
        "action_l2_mean": action_l2_mean,
        "sampled_action_abs_mean": sampled_action_abs_mean,
        "sampled_action_abs_max": sampled_action_abs_max,
    }
    return next_state, metrics
