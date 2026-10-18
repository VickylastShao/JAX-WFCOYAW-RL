#!/usr/bin/env python3
"""Permutation-equivariant Graph-SAC for scalable Mole wind-farm control."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

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
    init_mlp,
    mlp_apply,
    torchrl_biased_softplus,
)
from jax_mole_n9_auditable_replay import replay_bootstrap_discounts


class WakeGraph(NamedTuple):
    """Static sparse wake graph carried as a JAX pytree."""

    node_features: Array
    edge_sources: Array
    edge_targets: Array
    edge_features: Array
    in_degree: Array
    condition_dim: int = 0

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])


def build_wake_graph(
    turbine_positions_m: tuple[tuple[float, float, float], ...],
    diameter_m: float,
    wind_vector_xz: tuple[float, float] = (1.0, 0.0),
    lateral_tolerance_diameters: float = 0.6,
    condition_dim: int = 0,
) -> WakeGraph:
    """Build directed upstream edges from geometry and a horizontal flow vector."""

    positions = np.asarray(turbine_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("turbine positions must have shape (N, 3) with N > 0")
    if diameter_m <= 0.0 or lateral_tolerance_diameters <= 0.0:
        raise ValueError("diameter and lateral tolerance must be positive")
    if condition_dim < 0:
        raise ValueError("condition_dim cannot be negative")
    flow = np.asarray(wind_vector_xz, dtype=np.float64)
    if flow.shape != (2,) or not np.all(np.isfinite(flow)):
        raise ValueError("wind_vector_xz must contain two finite values")
    norm = float(np.linalg.norm(flow))
    if norm <= 0.0:
        raise ValueError("wind_vector_xz must be non-zero")
    flow /= norm
    cross = np.asarray((-flow[1], flow[0]), dtype=np.float64)
    horizontal = positions[:, (0, 2)]
    centered = horizontal - horizontal.mean(axis=0, keepdims=True)
    streamwise = centered @ flow / diameter_m
    lateral = centered @ cross / diameter_m

    sources: list[int] = []
    targets: list[int] = []
    features: list[tuple[float, float, float]] = []
    for target in range(positions.shape[0]):
        for source in range(positions.shape[0]):
            displacement = horizontal[target] - horizontal[source]
            downstream_d = float(displacement @ flow / diameter_m)
            lateral_d = float(displacement @ cross / diameter_m)
            if (
                downstream_d > 1.0e-8
                and abs(lateral_d) <= lateral_tolerance_diameters
            ):
                sources.append(source)
                targets.append(target)
                features.append(
                    (
                        downstream_d,
                        lateral_d,
                        1.0 / math.sqrt(1.0 + downstream_d * downstream_d),
                    )
                )

    source_array = np.asarray(sources, dtype=np.int32)
    target_array = np.asarray(targets, dtype=np.int32)
    feature_array = np.asarray(features, dtype=np.float32).reshape((-1, 3))
    in_degree = np.bincount(
        target_array, minlength=positions.shape[0]
    ).astype(np.float32)
    out_degree = np.bincount(
        source_array, minlength=positions.shape[0]
    ).astype(np.float32)
    normalizer = max(1.0, float(positions.shape[0] - 1))
    node_features = np.stack(
        (
            streamwise,
            lateral,
            in_degree / normalizer,
            out_degree / normalizer,
        ),
        axis=-1,
    ).astype(np.float32)
    return WakeGraph(
        node_features=jnp.asarray(node_features, dtype=JAX_REAL_DTYPE),
        edge_sources=jnp.asarray(source_array, dtype=jnp.int32),
        edge_targets=jnp.asarray(target_array, dtype=jnp.int32),
        edge_features=jnp.asarray(feature_array, dtype=JAX_REAL_DTYPE),
        in_degree=jnp.asarray(np.maximum(in_degree, 1.0), dtype=JAX_REAL_DTYPE),
        condition_dim=condition_dim,
    )


def graph_v2_node_observations(
    observations: Array,
    graph: WakeGraph,
    frame_stack: int = 1,
) -> Array:
    """Convert flattened probe histories to turbine tokens with geometry."""

    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    num_turbines = graph.num_nodes
    base = (
        observations[..., :-graph.condition_dim]
        if graph.condition_dim
        else observations
    )
    condition = (
        observations[..., -graph.condition_dim :]
        if graph.condition_dim
        else None
    )
    probe_width = base.shape[-1] - num_turbines
    divisor = num_turbines * frame_stack
    if probe_width <= 0 or probe_width % divisor:
        raise ValueError("observation width is incompatible with graph/frame stack")
    probes_per_frame = probe_width // divisor
    probes = base[..., :probe_width].reshape(
        *base.shape[:-1], frame_stack, num_turbines, probes_per_frame
    )
    probes = jnp.swapaxes(probes, -3, -2).reshape(
        *base.shape[:-1], num_turbines, frame_stack * probes_per_frame
    )
    yaw = base[..., probe_width:].reshape(
        *base.shape[:-1], num_turbines, 1
    )
    geometry = jnp.broadcast_to(
        graph.node_features,
        (*base.shape[:-1], *graph.node_features.shape),
    )
    parts = [probes, yaw, geometry]
    if condition is not None:
        parts.append(
            jnp.broadcast_to(
                condition[..., None, :],
                (*base.shape[:-1], num_turbines, graph.condition_dim),
            )
        )
    return jnp.concatenate(parts, axis=-1)


def _linear(key: Array, fan_in: int, fan_out: int) -> dict[str, Array]:
    return init_mlp(key, (fan_in, fan_out), "torch_linear")[0]


def _init_graph_body(
    key: Array,
    input_dim: int,
    edge_dim: int,
    hidden_dim: int,
    message_layers: int,
) -> PyTree:
    keys = iter(jax.random.split(key, 1 + 2 * message_layers))
    encoder = _linear(next(keys), input_dim, hidden_dim)
    layers = []
    for _ in range(message_layers):
        layers.append(
            {
                "self": _linear(next(keys), hidden_dim, hidden_dim),
                "message": _linear(
                    next(keys), hidden_dim + edge_dim, hidden_dim
                ),
            }
        )
    return {"encoder": encoder, "layers": tuple(layers)}


def _graph_body_apply(body: PyTree, nodes: Array, graph: WakeGraph) -> Array:
    hidden = jax.nn.relu(nodes @ body["encoder"]["w"] + body["encoder"]["b"])
    for layer in body["layers"]:
        source_hidden = jnp.take(hidden, graph.edge_sources, axis=-2)
        edge_features = jnp.broadcast_to(
            graph.edge_features,
            (*source_hidden.shape[:-1], graph.edge_features.shape[-1]),
        )
        message_input = jnp.concatenate((source_hidden, edge_features), axis=-1)
        messages = (
            message_input @ layer["message"]["w"] + layer["message"]["b"]
        )
        aggregate = jnp.zeros_like(hidden).at[..., graph.edge_targets, :].add(messages)
        degree = jnp.reshape(
            graph.in_degree,
            (1,) * (hidden.ndim - 2) + (graph.num_nodes, 1),
        )
        aggregate = aggregate / degree
        hidden = jax.nn.relu(
            hidden @ layer["self"]["w"] + layer["self"]["b"] + aggregate
        )
    return hidden


def _init_actor(
    key: Array,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    message_layers: int,
    conservative: bool,
    neutral_initial_scale: float | None = None,
) -> PyTree:
    body_key, head_key = jax.random.split(key)
    body = _init_graph_body(
        body_key, node_dim, edge_dim, hidden_dim, message_layers
    )
    head = _linear(head_key, hidden_dim, 2)
    if neutral_initial_scale is not None and not conservative:
        raise ValueError("neutral scale requires a zero-mean conservative head")
    if conservative:
        raw_scale_bias = -10.0
        if neutral_initial_scale is not None:
            minimum = 0.01
            if not minimum < neutral_initial_scale < 1.0:
                raise ValueError("neutral initial scale must be in (0.01, 1)")
            shift = math.log(math.expm1(1.0 - minimum))
            raw_scale_bias = math.log(
                math.expm1(neutral_initial_scale - minimum)
            ) - shift
        head = {
            "w": jnp.zeros_like(head["w"]),
            "b": jnp.asarray((0.0, raw_scale_bias), dtype=JAX_REAL_DTYPE),
        }
    actor = {"body": body, "head": head}
    if neutral_initial_scale is not None:
        # Empty tuple is a static PyTree marker, not an optimizable parameter.
        actor["neutral_scale_contract"] = ()
    return actor


def _init_critic(
    key: Array,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    message_layers: int,
) -> PyTree:
    body_key, node_key, edge_key, global_key = jax.random.split(key, 4)
    return {
        "body": _init_graph_body(
            body_key, node_dim + 1, edge_dim, hidden_dim, message_layers
        ),
        "node_head": init_mlp(
            node_key, (hidden_dim, hidden_dim, 1), "torch_linear"
        ),
        "edge_head": init_mlp(
            edge_key,
            (2 * hidden_dim + 2 + edge_dim, hidden_dim, 1),
            "torch_linear",
        ),
        "global_head": init_mlp(
            global_key, (hidden_dim, hidden_dim, 1), "torch_linear"
        ),
    }


def graph_v2_actor_parameters(
    actor: PyTree,
    observations: Array,
    graph: WakeGraph,
    frame_stack: int = 1,
) -> tuple[Array, Array]:
    nodes = graph_v2_node_observations(observations, graph, frame_stack)
    hidden = _graph_body_apply(actor["body"], nodes, graph)
    output = hidden @ actor["head"]["w"] + actor["head"]["b"]
    mean, raw_scale = jnp.split(output, 2, axis=-1)
    scale_lower_bound = 0.01 if "neutral_scale_contract" in actor else 0.1
    return mean.squeeze(-1), torchrl_biased_softplus(
        raw_scale, scale_lower_bound=scale_lower_bound
    ).squeeze(-1)


def graph_v2_actor_sample(
    actor: PyTree,
    observations: Array,
    key: Array,
    action_bound: float,
    graph: WakeGraph,
    frame_stack: int = 1,
) -> tuple[Array, Array]:
    if action_bound <= 0.0:
        raise ValueError("action_bound must be positive")
    mean, scale = graph_v2_actor_parameters(actor, observations, graph, frame_stack)
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
    gaussian = -0.5 * (
        standardized**2 + NP_REAL_DTYPE(math.log(2.0 * math.pi))
    ) - jnp.log(scale)
    correction = 2.0 * (
        NP_REAL_DTYPE(math.log(2.0))
        - coordinate
        - jax.nn.softplus(-2.0 * coordinate)
    ) + NP_REAL_DTYPE(math.log(action_bound))
    log_probability = jnp.sum(gaussian - correction, axis=-1)
    return NP_REAL_DTYPE(action_bound) * normalized, log_probability


def graph_v2_actor_deterministic(
    actor: PyTree,
    observations: Array,
    graph: WakeGraph,
    frame_stack: int = 1,
) -> Array:
    mean, _ = graph_v2_actor_parameters(actor, observations, graph, frame_stack)
    return jnp.tanh(mean)


def _critic_one(
    critic: PyTree,
    observations: Array,
    actions: Array,
    graph: WakeGraph,
    frame_stack: int,
) -> Array:
    nodes = graph_v2_node_observations(observations, graph, frame_stack)
    nodes = jnp.concatenate((actions[..., None], nodes), axis=-1)
    hidden = _graph_body_apply(critic["body"], nodes, graph)
    node_value = jnp.mean(
        mlp_apply(critic["node_head"], hidden, activation="relu").squeeze(-1),
        axis=-1,
    )
    global_value = mlp_apply(
        critic["global_head"], jnp.mean(hidden, axis=-2), activation="relu"
    ).squeeze(-1)
    if graph.edge_sources.shape[0] == 0:
        return 0.5 * (node_value + global_value)
    source_hidden = jnp.take(hidden, graph.edge_sources, axis=-2)
    target_hidden = jnp.take(hidden, graph.edge_targets, axis=-2)
    source_action = jnp.take(actions, graph.edge_sources, axis=-1)[..., None]
    target_action = jnp.take(actions, graph.edge_targets, axis=-1)[..., None]
    edge_features = jnp.broadcast_to(
        graph.edge_features,
        (*source_hidden.shape[:-1], graph.edge_features.shape[-1]),
    )
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


def graph_v2_critic_apply(
    critic: PyTree,
    observations: Array,
    actions: Array,
    graph: WakeGraph,
    frame_stack: int = 1,
) -> Array:
    return jnp.stack(
        tuple(
            _critic_one(
                critic[name], observations, actions, graph, frame_stack
            )
            for name in ("q1", "q2")
        ),
        axis=-1,
    )


def init_graph_v2_sac(
    key: Array,
    observation_dim: int,
    graph: WakeGraph,
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
) -> tuple[SACState, tuple[Any, Any, Any]]:
    num_turbines = graph.num_nodes
    base_observation_dim = observation_dim - graph.condition_dim
    if base_observation_dim <= num_turbines:
        raise ValueError("observation_dim must include probes and yaw")
    if (base_observation_dim - num_turbines) % (num_turbines * frame_stack):
        raise ValueError("observation_dim is incompatible with graph/frame stack")
    if initial_alpha <= 0.0 or actor_learning_rate_scale <= 0.0:
        raise ValueError("alpha and Actor learning-rate scale must be positive")
    raw_node_dim = (base_observation_dim - num_turbines) // num_turbines + 1
    node_dim = (
        raw_node_dim
        + graph.node_features.shape[-1]
        + graph.condition_dim
    )
    edge_dim = graph.edge_features.shape[-1]
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
        "q1": _init_critic(
            q1_key, node_dim, edge_dim, hidden_dim, message_layers
        ),
        "q2": _init_critic(
            q2_key, node_dim, edge_dim, hidden_dim, message_layers
        ),
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


def graph_v2_sac_update(
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
    graph: WakeGraph,
    frame_stack: int = 1,
    axis_name: str | None = None,
    reward_scale: float = 1.0,
    actor_update_period: int = 1,
    actor_warmup_updates: int = 0,
    actor_minimum_replay_size: int = 0,
    action_l2_coefficient: float = 0.0,
) -> tuple[SACState, dict[str, Array]]:
    """One Graph-SAC update with joint entropy and optional Actor warm-up."""

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
    indices = jax.random.randint(
        sample_key, (sample_size,), 0, jnp.maximum(replay.size, 1)
    )
    observations = replay.observations[indices]
    actions = replay.actions[indices]
    rewards = replay.rewards[indices]
    next_observations = replay.next_observations[indices]
    dones = replay.dones[indices]
    alpha = jnp.exp(state.log_alpha)

    next_actions, next_log_probability = graph_v2_actor_sample(
        state.actor,
        next_observations,
        target_key,
        action_bound,
        graph,
        frame_stack,
    )
    target_q = jnp.min(
        graph_v2_critic_apply(
            state.target_critic,
            next_observations,
            next_actions,
            graph,
            frame_stack,
        ),
        axis=-1,
    )
    bootstrap_discounts = replay_bootstrap_discounts(
        replay, indices, gamma=gamma, sampled_dones=dones
    )
    target = jax.lax.stop_gradient(
        NP_REAL_DTYPE(reward_scale) * rewards
        + bootstrap_discounts
        * (
            target_q
            - alpha * NP_REAL_DTYPE(entropy_scale) * next_log_probability
        )
    )

    def critic_loss_fn(parameters: PyTree) -> Array:
        values = graph_v2_critic_apply(
            parameters, observations, actions, graph, frame_stack
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
        sampled_actions, log_probability = graph_v2_actor_sample(
            actor, observations, actor_key, action_bound, graph, frame_stack
        )
        q_min = jnp.min(
            graph_v2_critic_apply(
                state.critic, observations, sampled_actions, graph, frame_stack
            ),
            axis=-1,
        )
        action_l2_mean = jnp.mean(
            (sampled_actions / NP_REAL_DTYPE(action_bound)) ** 2
        )
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
    if actor_update_period == 0:
        actor_updated = jnp.asarray(False)
    else:
        actor_updated = after_warmup & (
            (state.updates + 1 - actor_warmup_updates) % actor_update_period == 0
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
            (1.0 - NP_REAL_DTYPE(tau)) * target_value
            + NP_REAL_DTYPE(tau) * value
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
