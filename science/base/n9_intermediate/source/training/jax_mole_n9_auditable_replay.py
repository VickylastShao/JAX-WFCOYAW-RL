#!/usr/bin/env python3
"""Auditable replay storage for the frozen N=9 million-step protocol."""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import JAX_REAL_DTYPE, Array


class AuditableReplayState(NamedTuple):
    # The first seven fields preserve the legacy ReplayState contract.
    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    position: Array
    size: Array
    raw_turbine_power_w: Array
    applied_yaw_deg: Array
    p0_mw_per_turbine: Array
    mean_power_mw: Array
    barrier_mw: Array
    relative_power_gain: Array
    reward_r0: Array
    reward_r1: Array
    direction_deg: Array
    episode_ids: Array
    sequence_ids: Array
    bootstrap_discounts: Array
    n_step_horizons: Array


AUDIT_ARRAY_FIELDS = tuple(
    name for name in AuditableReplayState._fields if name not in {"position", "size"}
)


def auditable_replay_theoretical_bytes(
    capacity: int,
    observation_dim: int,
    action_dim: int,
    num_turbines: int,
    *,
    itemsize: int = 4,
) -> int:
    """Return exact array payload bytes for the v1 auditable Replay."""

    if min(capacity, observation_dim, action_dim, num_turbines, itemsize) <= 0:
        raise ValueError("Replay dimensions and itemsize must be positive")
    # Two observations, one action, power/yaw vectors, ten float scalars,
    # and three int32 scalars. Position and size are two device scalars.
    elements_per_row = (
        2 * observation_dim + action_dim + 2 * num_turbines + 13
    )
    return capacity * elements_per_row * itemsize + 2 * itemsize


def init_auditable_replay(
    capacity: int,
    observation_dim: int,
    action_dim: int,
    num_turbines: int,
) -> AuditableReplayState:
    if min(capacity, observation_dim, action_dim, num_turbines) <= 0:
        raise ValueError("Replay dimensions must be positive")
    scalar = lambda dtype=JAX_REAL_DTYPE: jnp.zeros((capacity,), dtype=dtype)
    return AuditableReplayState(
        observations=jnp.zeros((capacity, observation_dim), JAX_REAL_DTYPE),
        actions=jnp.zeros((capacity, action_dim), JAX_REAL_DTYPE),
        rewards=scalar(),
        next_observations=jnp.zeros((capacity, observation_dim), JAX_REAL_DTYPE),
        dones=scalar(),
        position=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        raw_turbine_power_w=jnp.zeros((capacity, num_turbines), JAX_REAL_DTYPE),
        applied_yaw_deg=jnp.zeros((capacity, num_turbines), JAX_REAL_DTYPE),
        p0_mw_per_turbine=scalar(),
        mean_power_mw=scalar(),
        barrier_mw=scalar(),
        relative_power_gain=scalar(),
        reward_r0=scalar(),
        reward_r1=scalar(),
        direction_deg=scalar(),
        episode_ids=scalar(jnp.int32),
        sequence_ids=scalar(jnp.int32),
        bootstrap_discounts=scalar(),
        n_step_horizons=scalar(jnp.int32),
    )


def add_auditable_batch(
    replay: AuditableReplayState,
    **arrays: Array,
) -> AuditableReplayState:
    required = set(AUDIT_ARRAY_FIELDS)
    if set(arrays) != required:
        missing = sorted(required - set(arrays))
        extra = sorted(set(arrays) - required)
        raise ValueError(f"Replay batch fields differ: missing={missing}, extra={extra}")
    batch = int(arrays["observations"].shape[0])
    if batch <= 0 or any(int(value.shape[0]) != batch for value in arrays.values()):
        raise ValueError("Every Replay batch field must have the same positive length")
    capacity = replay.observations.shape[0]
    if batch > capacity:
        raise ValueError("A Replay batch cannot exceed capacity")
    indices = (replay.position + jnp.arange(batch, dtype=jnp.int32)) % capacity
    updates = {
        name: getattr(replay, name).at[indices].set(value)
        for name, value in arrays.items()
    }
    return replay._replace(
        **updates,
        position=(replay.position + batch) % capacity,
        size=jnp.minimum(replay.size + batch, capacity),
    )


def replay_bootstrap_discounts(
    replay: object,
    indices: Array,
    *,
    gamma: float,
    sampled_dones: Array,
) -> Array:
    """Return stored n-step discounts, or the legacy one-step discount."""

    if hasattr(replay, "bootstrap_discounts"):
        return replay.bootstrap_discounts[indices]
    return jnp.asarray(gamma, dtype=JAX_REAL_DTYPE) * (1.0 - sampled_dones)


def recompute_reward_audit(
    raw_turbine_power_w: np.ndarray,
    applied_yaw_deg: np.ndarray,
    p0_mw_per_turbine: np.ndarray,
    *,
    action_limit_deg: float = 40.0,
    barrier_scale_mw: float = 0.5,
    barrier_exponent: int = 26,
) -> dict[str, np.ndarray]:
    power = np.asarray(raw_turbine_power_w, dtype=np.float64)
    yaw = np.asarray(applied_yaw_deg, dtype=np.float64)
    p0 = np.asarray(p0_mw_per_turbine, dtype=np.float64)
    if power.shape != yaw.shape or power.ndim != 2:
        raise ValueError("Power and yaw must have shape [batch, turbine]")
    if p0.shape != (power.shape[0],) or np.any(p0 <= 0.0):
        raise ValueError("P0 must be positive with one value per transition")
    mean_power = power.mean(axis=1) / 1.0e6
    barrier = barrier_scale_mw * np.mean(
        (yaw / action_limit_deg) ** barrier_exponent, axis=1
    )
    relative_gain = (mean_power - p0) / p0
    return {
        "mean_power_mw": mean_power,
        "barrier_mw": barrier,
        "relative_power_gain": relative_gain,
        "reward_r0": mean_power - barrier,
        "reward_r1": relative_gain - barrier / p0,
    }


def active_replay_sha256(replay: AuditableReplayState) -> str:
    """Hash active Replay arrays in stable field order on the host."""

    size = int(np.asarray(replay.size))
    digest = hashlib.sha256()
    digest.update(f"auditable-replay-v1:{size}\n".encode())
    for name in AUDIT_ARRAY_FIELDS:
        value = np.ascontiguousarray(np.asarray(getattr(replay, name))[:size])
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
