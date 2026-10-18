#!/usr/bin/env python3
"""Critic-only Batch-Renormalized CrossQ for the Mole JAX trainer.

The actor intentionally retains the existing Mole/TorchRL contract. The
CrossQ critic removes target networks and evaluates replay and next-policy
batches in one joint forward pass so every normalization layer sees the same
mixture distribution. This corresponds to the critic-only normalization
ablation reported by Bhatt et al. (ICLR 2024).
"""

from __future__ import annotations

import math
from typing import Any

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
    actor_sample,
    init_mlp,
)


def _normalizer_parameters(width: int) -> dict[str, Array]:
    return {
        "scale": jnp.ones((width,), dtype=JAX_REAL_DTYPE),
        "bias": jnp.zeros((width,), dtype=JAX_REAL_DTYPE),
    }


def _normalizer_statistics(width: int) -> dict[str, Array]:
    return {
        "mean": jnp.zeros((width,), dtype=JAX_REAL_DTYPE),
        "variance": jnp.ones((width,), dtype=JAX_REAL_DTYPE),
        "steps": jnp.asarray(0, dtype=jnp.int32),
    }


def _init_crossq_critic(
    key: Array,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> tuple[PyTree, PyTree]:
    q1_key, q2_key = jax.random.split(key)
    widths = (observation_dim + action_dim, hidden_dim, hidden_dim)

    def initialize_one(network_key: Array) -> tuple[PyTree, PyTree]:
        parameters = {
            "layers": init_mlp(
                network_key,
                (
                    observation_dim + action_dim,
                    hidden_dim,
                    hidden_dim,
                    1,
                ),
                "torch_linear",
            ),
            "normalizers": tuple(
                _normalizer_parameters(width) for width in widths
            ),
        }
        statistics = tuple(
            _normalizer_statistics(width) for width in widths
        )
        return parameters, statistics

    q1_parameters, q1_statistics = initialize_one(q1_key)
    q2_parameters, q2_statistics = initialize_one(q2_key)
    return (
        {"q1": q1_parameters, "q2": q2_parameters},
        {"q1": q1_statistics, "q2": q2_statistics},
    )


def _global_batch_moments(
    values: Array, axis_name: str | None
) -> tuple[Array, Array]:
    mean = jnp.mean(values, axis=0)
    second_moment = jnp.mean(values * values, axis=0)
    if axis_name is not None:
        mean = jax.lax.pmean(mean, axis_name)
        second_moment = jax.lax.pmean(second_moment, axis_name)
    variance = jnp.maximum(second_moment - mean * mean, 0.0)
    return mean, variance


def _batch_renormalize(
    values: Array,
    parameters: PyTree,
    statistics: PyTree,
    *,
    training: bool,
    momentum: float,
    warmup_updates: int,
    axis_name: str | None,
    epsilon: float = 1.0e-3,
) -> tuple[Array, PyTree]:
    running_mean = statistics["mean"]
    running_variance = statistics["variance"]
    if not training:
        normalized = (values - running_mean) * jax.lax.rsqrt(
            running_variance + NP_REAL_DTYPE(epsilon)
        )
        return (
            normalized * parameters["scale"] + parameters["bias"],
            statistics,
        )

    batch_mean, batch_variance = _global_batch_moments(values, axis_name)
    batch_inverse_std = jax.lax.rsqrt(
        batch_variance + NP_REAL_DTYPE(epsilon)
    )
    normalized = (values - batch_mean) * batch_inverse_std

    running_std = jnp.sqrt(running_variance + NP_REAL_DTYPE(epsilon))
    batch_std = jnp.sqrt(batch_variance + NP_REAL_DTYPE(epsilon))
    ratio = jnp.clip(batch_std / running_std, 1.0 / 3.0, 3.0)
    offset = jnp.clip(
        (batch_mean - running_mean) / running_std, -5.0, 5.0
    )
    warmed_up = statistics["steps"] >= warmup_updates
    normalized = jnp.where(
        warmed_up,
        normalized * jax.lax.stop_gradient(ratio)
        + jax.lax.stop_gradient(offset),
        normalized,
    )

    one_minus_momentum = NP_REAL_DTYPE(1.0 - momentum)
    updated_statistics = {
        "mean": jax.lax.stop_gradient(
            NP_REAL_DTYPE(momentum) * running_mean
            + one_minus_momentum * batch_mean
        ),
        "variance": jax.lax.stop_gradient(
            NP_REAL_DTYPE(momentum) * running_variance
            + one_minus_momentum * batch_variance
        ),
        "steps": statistics["steps"] + 1,
    }
    return (
        normalized * parameters["scale"] + parameters["bias"],
        updated_statistics,
    )


def _critic_one(
    parameters: PyTree,
    statistics: PyTree,
    inputs: Array,
    *,
    training: bool,
    momentum: float,
    warmup_updates: int,
    axis_name: str | None,
) -> tuple[Array, PyTree]:
    output = inputs
    updated_statistics = []
    for index, layer in enumerate(parameters["layers"]):
        if index < len(parameters["normalizers"]):
            output, layer_statistics = _batch_renormalize(
                output,
                parameters["normalizers"][index],
                statistics[index],
                training=training,
                momentum=momentum,
                warmup_updates=warmup_updates,
                axis_name=axis_name,
            )
            updated_statistics.append(layer_statistics)
        output = output @ layer["w"] + layer["b"]
        if index < len(parameters["layers"]) - 1:
            output = jax.nn.relu(output)
    return output.squeeze(-1), tuple(updated_statistics)


def crossq_critic_apply(
    critic: PyTree,
    observations: Array,
    actions: Array,
    *,
    training: bool,
    momentum: float = 0.99,
    warmup_updates: int = 100_000,
    axis_name: str | None = None,
) -> tuple[Array, PyTree]:
    """Apply both critics and optionally update synchronized BRN statistics."""

    inputs = jnp.concatenate((actions, observations), axis=-1)
    values = []
    updated_statistics = {}
    for name in ("q1", "q2"):
        value, statistics = _critic_one(
            critic["params"][name],
            critic["statistics"][name],
            inputs,
            training=training,
            momentum=momentum,
            warmup_updates=warmup_updates,
            axis_name=axis_name,
        )
        values.append(value)
        updated_statistics[name] = statistics
    return jnp.stack(values, axis=-1), updated_statistics


def init_crossq(
    key: Array,
    observation_dim: int,
    action_dim: int,
    actor_hidden_dim: int,
    critic_hidden_dim: int,
    learning_rate: float,
    *,
    initial_alpha: float,
    entropy_learning_rate: float,
    adam_b1: float = 0.5,
) -> tuple[SACState, tuple[Any, Any, Any]]:
    """Initialize critic-only CrossQ while preserving the Mole actor format."""

    if initial_alpha <= 0.0:
        raise ValueError("initial_alpha must be positive")
    actor_key, critic_key = jax.random.split(key)
    actor = init_mlp(
        actor_key,
        (
            observation_dim,
            actor_hidden_dim,
            actor_hidden_dim,
            2 * action_dim,
        ),
        "torch_linear",
    )
    critic_parameters, critic_statistics = _init_crossq_critic(
        critic_key,
        observation_dim,
        action_dim,
        critic_hidden_dim,
    )
    critic = {
        "params": critic_parameters,
        "statistics": critic_statistics,
    }
    actor_optimizer = optax.adam(learning_rate, b1=adam_b1)
    critic_optimizer = optax.adam(learning_rate, b1=adam_b1)
    alpha_optimizer = optax.adam(entropy_learning_rate)
    log_alpha = jnp.asarray(
        math.log(initial_alpha), dtype=JAX_REAL_DTYPE
    )
    state = SACState(
        actor=actor,
        critic=critic,
        target_critic={},
        log_alpha=log_alpha,
        actor_opt=actor_optimizer.init(actor),
        critic_opt=critic_optimizer.init(critic_parameters),
        alpha_opt=alpha_optimizer.init(log_alpha),
        updates=jnp.asarray(0, dtype=jnp.int32),
    )
    return state, (
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
    )


def crossq_update(
    state: SACState,
    replay: ReplayState,
    key: Array,
    optimizers: tuple[Any, Any, Any],
    sample_size: int,
    action_bound: float,
    gamma: float,
    target_entropy: float,
    *,
    axis_name: str | None = None,
    policy_mode: str = "public_torchrl",
    momentum: float = 0.99,
    warmup_updates: int = 100_000,
    policy_delay: int = 3,
    critic_gradient_clip_norm: float = 0.0,
) -> tuple[SACState, dict[str, Array]]:
    """Run one UTD=1 CrossQ update with a joint current/next critic pass."""

    if policy_mode != "public_torchrl":
        raise ValueError("CrossQ currently requires public_torchrl policy mode")
    if policy_delay < 1:
        raise ValueError("policy_delay must be positive")
    if critic_gradient_clip_norm < 0.0:
        raise ValueError("critic_gradient_clip_norm cannot be negative")
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
    joint_observations = jnp.concatenate(
        (observations, next_observations), axis=0
    )
    joint_actions = jnp.concatenate((actions, next_actions), axis=0)

    def critic_loss_fn(
        critic_parameters: PyTree,
    ) -> tuple[Array, tuple[PyTree, Array, Array]]:
        critic = {
            "params": critic_parameters,
            "statistics": state.critic["statistics"],
        }
        joint_q, updated_statistics = crossq_critic_apply(
            critic,
            joint_observations,
            joint_actions,
            training=True,
            momentum=momentum,
            warmup_updates=warmup_updates,
            axis_name=axis_name,
        )
        current_q, next_q = jnp.split(joint_q, 2, axis=0)
        next_q_min = jnp.min(next_q, axis=-1)
        target = jax.lax.stop_gradient(
            rewards
            + NP_REAL_DTYPE(gamma)
            * (1.0 - dones)
            * (next_q_min - alpha * next_log_probability)
        )
        squared_error = (current_q - target[:, None]) ** 2
        loss = NP_REAL_DTYPE(0.5) * jnp.mean(
            jnp.sum(squared_error, axis=-1)
        )
        return loss, (
            updated_statistics,
            jnp.mean(current_q),
            jnp.mean(next_q_min),
        )

    (
        critic_loss,
        (critic_statistics, current_q_mean, next_q_mean),
    ), critic_gradient = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(state.critic["params"])
    if axis_name is not None:
        critic_gradient = jax.lax.pmean(critic_gradient, axis_name)
    critic_gradient_norm_raw = optax.global_norm(critic_gradient)
    if critic_gradient_clip_norm > 0.0:
        scale = jnp.minimum(
            1.0,
            NP_REAL_DTYPE(critic_gradient_clip_norm)
            / jnp.maximum(
                critic_gradient_norm_raw, NP_REAL_DTYPE(1.0e-12)
            ),
        )
        critic_gradient = jax.tree.map(
            lambda value: value * scale, critic_gradient
        )
    critic_gradient_norm_applied = optax.global_norm(critic_gradient)
    critic_updates, critic_opt = critic_optimizer.update(
        critic_gradient,
        state.critic_opt,
        state.critic["params"],
    )
    critic_parameters = optax.apply_updates(
        state.critic["params"], critic_updates
    )
    critic = {
        "params": critic_parameters,
        "statistics": critic_statistics,
    }

    def update_actor_and_alpha(
        operand: tuple[PyTree, Array, PyTree, PyTree]
    ) -> tuple[PyTree, Array, PyTree, PyTree, Array, Array]:
        actor, log_alpha, actor_opt, alpha_opt = operand

        def actor_loss_fn(actor_parameters: PyTree) -> tuple[Array, Array]:
            sampled_actions, log_probability = actor_sample(
                actor_parameters,
                observations,
                actor_key,
                action_bound,
                policy_mode,
            )
            q_values, _ = crossq_critic_apply(
                critic,
                observations,
                sampled_actions,
                training=False,
                momentum=momentum,
                warmup_updates=warmup_updates,
                axis_name=None,
            )
            loss = jnp.mean(
                jax.lax.stop_gradient(alpha) * log_probability
                - jnp.min(q_values, axis=-1)
            )
            return loss, log_probability

        (actor_loss, log_probability), actor_gradient = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor)
        if axis_name is not None:
            actor_gradient = jax.lax.pmean(actor_gradient, axis_name)
        updates, next_actor_opt = actor_optimizer.update(
            actor_gradient, actor_opt, actor
        )
        next_actor = optax.apply_updates(actor, updates)

        def alpha_loss_fn(value: Array) -> Array:
            return -jnp.mean(
                value
                * jax.lax.stop_gradient(
                    log_probability + NP_REAL_DTYPE(target_entropy)
                )
            )

        alpha_loss, alpha_gradient = jax.value_and_grad(alpha_loss_fn)(
            log_alpha
        )
        if axis_name is not None:
            alpha_gradient = jax.lax.pmean(alpha_gradient, axis_name)
        updates, next_alpha_opt = alpha_optimizer.update(
            alpha_gradient, alpha_opt, log_alpha
        )
        next_log_alpha = optax.apply_updates(log_alpha, updates)
        return (
            next_actor,
            next_log_alpha,
            next_actor_opt,
            next_alpha_opt,
            actor_loss,
            alpha_loss,
        )

    def skip_actor_and_alpha(
        operand: tuple[PyTree, Array, PyTree, PyTree]
    ) -> tuple[PyTree, Array, PyTree, PyTree, Array, Array]:
        actor, log_alpha, actor_opt, alpha_opt = operand
        zero = jnp.asarray(0.0, dtype=JAX_REAL_DTYPE)
        return actor, log_alpha, actor_opt, alpha_opt, zero, zero

    actor_updated = state.updates % policy_delay == 0
    (
        actor,
        log_alpha,
        actor_opt,
        alpha_opt,
        actor_loss,
        alpha_loss,
    ) = jax.lax.cond(
        actor_updated,
        update_actor_and_alpha,
        skip_actor_and_alpha,
        (state.actor, state.log_alpha, state.actor_opt, state.alpha_opt),
    )
    next_state = SACState(
        actor=actor,
        critic=critic,
        target_critic={},
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
        "current_q_mean": current_q_mean,
        "next_q_mean": next_q_mean,
        "brn_steps": critic_statistics["q1"][0]["steps"],
        "critic_gradient_norm_raw": critic_gradient_norm_raw,
        "critic_gradient_norm_applied": critic_gradient_norm_applied,
    }
    return next_state, metrics
