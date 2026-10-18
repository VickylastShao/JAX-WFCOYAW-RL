#!/usr/bin/env python3
"""Frozen, backend-independent contracts for the N=9 million-step study.

This module contains protocol arithmetic only. The LES collector and JAX
learner integrate these contracts after their standalone tests pass.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import NamedTuple

import numpy as np


TRAIN_DIRECTIONS_DEG = np.asarray(
    (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0),
    dtype=np.float32,
)
P0_MW_PER_TURBINE = np.asarray(
    (
        1.5912005454301834,
        1.422188714146614,
        1.230438768863678,
        1.0754493325948715,
        1.0178109556436539,
        1.0754493325948715,
        1.230438768863678,
        1.422188714146614,
        1.5912005454301834,
    ),
    dtype=np.float32,
)
P0_TABLE_SHA256 = "118f1d5efdc8168bbc4c4a330bcaf12e4926237d2034a1c169a3c598044a8316"
SEALED_MIDPOINT_DIRECTIONS_DEG = np.asarray(
    (-8.75, -6.25, -3.75, -1.25, 1.25, 3.75, 6.25, 8.75),
    dtype=np.float32,
)
GLOBAL_ENVIRONMENTS = 32
ENVIRONMENTS_PER_HOST = 16
RESETS_PER_SUPERCYCLE = 9


class RewardComponents(NamedTuple):
    mean_power_mw: np.ndarray
    barrier_mw: np.ndarray
    relative_power_gain: np.ndarray
    r0: np.ndarray
    r1: np.ndarray


class NStepTargets(NamedTuple):
    returns: np.ndarray
    bootstrap_discounts: np.ndarray
    horizons: np.ndarray
    next_indices: np.ndarray
    valid: np.ndarray


def p0_mw_per_turbine(direction_deg: float | np.ndarray) -> np.ndarray:
    """Return the frozen development P0 for exact training directions."""

    directions = np.asarray(direction_deg, dtype=np.float32)
    matches = directions[..., None] == TRAIN_DIRECTIONS_DEG
    valid = np.sum(matches, axis=-1) == 1
    if not np.all(valid):
        invalid = directions[~valid] if directions.ndim else directions
        raise ValueError(f"direction has no frozen P0 value: {invalid}")
    return np.sum(matches * P0_MW_PER_TURBINE, axis=-1, dtype=np.float32)


def build_direction_schedule(
    supercycles: int = 1,
    *,
    schedule_seed: int = 2026081802,
) -> np.ndarray:
    """Return the deterministic 9-reset, 32-environment direction schedule.

    Every reset assigns four environments to five directions and three to the
    other four. Across a supercycle, every direction has 32 global episodes and
    16 episodes on each host.
    """

    if supercycles <= 0:
        raise ValueError("supercycles must be positive")
    cycle_rows: list[np.ndarray] = []
    for reset in range(RESETS_PER_SUPERCYCLE):
        host_zero: list[float] = []
        host_one: list[float] = []
        for direction_index, direction in enumerate(TRAIN_DIRECTIONS_DEG):
            residue = (direction_index - reset) % RESETS_PER_SUPERCYCLE
            if residue <= 4:
                host_zero.extend((float(direction),) * 2)
                host_one.extend((float(direction),) * 2)
            elif residue in (5, 7):
                host_zero.extend((float(direction),) * 2)
                host_one.append(float(direction))
            else:
                host_zero.append(float(direction))
                host_one.extend((float(direction),) * 2)
        if not (
            len(host_zero) == ENVIRONMENTS_PER_HOST
            and len(host_one) == ENVIRONMENTS_PER_HOST
        ):
            raise AssertionError("invalid host allocation in direction schedule")
        shared_order = TRAIN_DIRECTIONS_DEG[
            np.random.default_rng(
                np.random.SeedSequence([schedule_seed, reset, 2])
            ).permutation(RESETS_PER_SUPERCYCLE)
        ]

        def ordered_host(values: list[float], host: int) -> np.ndarray:
            counts = {
                float(direction): values.count(float(direction))
                for direction in TRAIN_DIRECTIONS_DEG
            }
            extras = np.asarray(
                [
                    float(direction)
                    for direction in shared_order
                    for _ in range(counts[float(direction)] - 1)
                ],
                dtype=np.float32,
            )
            extras = extras[
                np.random.default_rng(
                    np.random.SeedSequence([schedule_seed, reset, host])
                ).permutation(extras.size)
            ]
            return np.concatenate((shared_order, extras))

        host_zero_array = ordered_host(host_zero, 0)
        host_one_array = ordered_host(host_one, 1)
        cycle_rows.append(np.concatenate((host_zero_array, host_one_array)))
    cycle = np.stack(cycle_rows)
    return np.tile(cycle, (supercycles, 1))


def initial_exploration_sigma_raw(
    *,
    action_limit_deg: float = 40.0,
    initial_limit_deg: float = 10.0,
    num_actions: int = 9,
    familywise_tail_probability: float = 0.01,
) -> float:
    """Compute the raw Gaussian sigma for the frozen initial action envelope."""

    if not 0.0 < initial_limit_deg < action_limit_deg:
        raise ValueError("initial_limit_deg must be inside the action bound")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    if not 0.0 < familywise_tail_probability < 1.0:
        raise ValueError("familywise_tail_probability must be in (0, 1)")
    quantile = NormalDist().inv_cdf(
        1.0 - familywise_tail_probability / (2.0 * num_actions)
    )
    normalized_limit = math.atanh(initial_limit_deg / action_limit_deg)
    return normalized_limit / quantile


def reward_components(
    mean_power_w: np.ndarray,
    yaw_deg: np.ndarray,
    *,
    p0_mw_per_turbine: float | np.ndarray,
    action_limit_deg: float = 40.0,
    barrier_scale_mw: float = 0.5,
    barrier_exponent: int = 26,
) -> RewardComponents:
    """Compute the auditable R0 and P0-normalized R1 reward components."""

    power = np.asarray(mean_power_w, dtype=np.float64)
    yaw = np.asarray(yaw_deg, dtype=np.float64)
    if power.shape != yaw.shape or power.ndim < 1:
        raise ValueError("power and yaw must have matching turbine dimensions")
    p0 = np.asarray(p0_mw_per_turbine, dtype=np.float64)
    if np.any(~np.isfinite(p0)) or np.any(p0 <= 0.0):
        raise ValueError("p0_mw_per_turbine must be finite and positive")
    mean_power_mw = np.mean(power, axis=-1) / 1.0e6
    normalized_barrier = np.mean(
        (yaw / float(action_limit_deg)) ** barrier_exponent,
        axis=-1,
    )
    barrier_mw = float(barrier_scale_mw) * normalized_barrier
    relative_power_gain = (mean_power_mw - p0) / p0
    r0 = mean_power_mw - barrier_mw
    r1 = relative_power_gain - barrier_mw / p0
    return RewardComponents(
        mean_power_mw=mean_power_mw,
        barrier_mw=barrier_mw,
        relative_power_gain=relative_power_gain,
        r0=r0,
        r1=r1,
    )


def build_n_step_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_ids: np.ndarray,
    *,
    gamma: float,
    n_step: int,
) -> NStepTargets:
    """Construct n-step returns without crossing an episode/reset boundary."""

    reward_array = np.asarray(rewards, dtype=np.float64)
    done_array = np.asarray(dones, dtype=np.bool_)
    episode_array = np.asarray(episode_ids)
    if not (
        reward_array.ndim == done_array.ndim == episode_array.ndim == 1
        and reward_array.shape == done_array.shape == episode_array.shape
    ):
        raise ValueError("rewards, dones and episode_ids must be equal 1-D arrays")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if n_step <= 0:
        raise ValueError("n_step must be positive")

    count = reward_array.size
    returns = np.zeros(count, dtype=np.float64)
    discounts = np.zeros(count, dtype=np.float64)
    horizons = np.zeros(count, dtype=np.int32)
    next_indices = np.full(count, -1, dtype=np.int64)
    valid = np.zeros(count, dtype=np.bool_)

    for start in range(count):
        episode = episode_array[start]
        terminal = False
        for offset in range(n_step):
            index = start + offset
            if index >= count or episode_array[index] != episode:
                break
            returns[start] += (gamma**offset) * reward_array[index]
            horizons[start] += 1
            next_indices[start] = index
            if done_array[index]:
                terminal = True
                valid[start] = True
                break
        if not terminal and horizons[start] == n_step:
            valid[start] = True
            discounts[start] = gamma**n_step

    return NStepTargets(
        returns=returns,
        bootstrap_discounts=discounts,
        horizons=horizons,
        next_indices=next_indices,
        valid=valid,
    )


def actor_update_allowed(
    global_transitions: int,
    critic_updates: int,
    *,
    minimum_transitions: int = 32_768,
    minimum_critic_updates: int = 4_096,
) -> bool:
    """Return whether both frozen Actor warm-up gates have opened."""

    return (
        int(global_transitions) > minimum_transitions
        and int(critic_updates) > minimum_critic_updates
    )
