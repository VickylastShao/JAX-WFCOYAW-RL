#!/usr/bin/env python3
"""Deterministic R0-R3 branching from one auditable Replay prefix."""

from __future__ import annotations

from typing import Any

import numpy as np

from jax_mole_n9_auditable_replay import AUDIT_ARRAY_FIELDS
from jax_mole_n9_million_protocol import build_n_step_targets


def branch_auditable_arrays(
    arrays: dict[str, np.ndarray],
    *,
    reward_name: str,
    n_step: int,
    gamma: float,
    global_environments: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a finalized branch while preserving original insertion order."""

    finalized, _, metadata = branch_auditable_arrays_with_pending(
        arrays,
        reward_name=reward_name,
        n_step=n_step,
        gamma=gamma,
        global_environments=global_environments,
    )
    return finalized, metadata


def branch_auditable_arrays_with_pending(
    arrays: dict[str, np.ndarray],
    *,
    reward_name: str,
    n_step: int,
    gamma: float,
    global_environments: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build finalized n-step rows and retain incomplete per-env tails."""

    if set(arrays) != set(AUDIT_ARRAY_FIELDS):
        raise ValueError("input fields do not match auditable Replay v1")
    if reward_name not in {"reward_r0", "reward_r1"}:
        raise ValueError("reward_name must be reward_r0 or reward_r1")
    if n_step not in {1, 8, 16} or global_environments <= 0:
        raise ValueError("unsupported n-step or environment count")
    count = int(np.asarray(arrays["observations"]).shape[0])
    if count == 0 or any(np.asarray(value).shape[0] != count for value in arrays.values()):
        raise ValueError("all auditable fields must share a positive row count")

    sequence_ids = np.asarray(arrays["sequence_ids"], dtype=np.int64)
    if np.unique(sequence_ids).size != count:
        raise ValueError("sequence IDs must be globally unique")
    episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
    dones = np.asarray(arrays["dones"], dtype=np.float64)
    source_rewards = np.asarray(arrays[reward_name], dtype=np.float64)
    returns = np.zeros(count, dtype=np.float64)
    discounts = np.zeros(count, dtype=np.float64)
    horizons = np.zeros(count, dtype=np.int32)
    endpoints = np.full(count, -1, dtype=np.int64)
    valid = np.zeros(count, dtype=np.bool_)

    environment_ids = sequence_ids % global_environments
    for environment in np.unique(environment_ids):
        rows = np.flatnonzero(environment_ids == environment)
        order = rows[np.argsort(sequence_ids[rows], kind="stable")]
        targets = build_n_step_targets(
            source_rewards[order],
            dones[order],
            episode_ids[order],
            gamma=gamma,
            n_step=n_step,
        )
        selected = order[targets.valid]
        valid[selected] = True
        returns[selected] = targets.returns[targets.valid]
        discounts[selected] = targets.bootstrap_discounts[targets.valid]
        horizons[selected] = targets.horizons[targets.valid]
        endpoints[selected] = order[targets.next_indices[targets.valid]]

    selected_rows = np.flatnonzero(valid)
    if selected_rows.size == 0:
        raise ValueError("branch contains no finalized transitions")
    output = {
        name: np.array(np.asarray(value)[selected_rows], copy=True)
        for name, value in arrays.items()
    }
    output["rewards"] = returns[selected_rows].astype(
        np.asarray(arrays["rewards"]).dtype
    )
    output["next_observations"] = np.array(
        np.asarray(arrays["next_observations"])[endpoints[selected_rows]],
        copy=True,
    )
    output["dones"] = np.array(
        np.asarray(arrays["dones"])[endpoints[selected_rows]], copy=True
    )
    output["bootstrap_discounts"] = discounts[selected_rows].astype(
        np.asarray(arrays["bootstrap_discounts"]).dtype
    )
    output["n_step_horizons"] = horizons[selected_rows].astype(np.int32)
    metadata = {
        "source_rows": count,
        "finalized_rows": int(selected_rows.size),
        "pending_tail_rows": int(count - selected_rows.size),
        "reward_name": reward_name,
        "n_step": n_step,
        "gamma": float(gamma),
        "global_environments": global_environments,
    }
    pending_rows = np.flatnonzero(~valid)
    pending = {
        name: np.array(np.asarray(value)[pending_rows], copy=True)
        for name, value in arrays.items()
    }
    return output, pending, metadata


def advance_auditable_n_step(
    pending: dict[str, np.ndarray],
    new_arrays: dict[str, np.ndarray],
    *,
    reward_name: str,
    n_step: int,
    gamma: float,
    global_environments: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Finalize an online batch after prepending retained raw tails."""

    if set(pending) != set(AUDIT_ARRAY_FIELDS):
        raise ValueError("pending fields do not match auditable Replay v1")
    if set(new_arrays) != set(AUDIT_ARRAY_FIELDS):
        raise ValueError("new fields do not match auditable Replay v1")
    pending_count = int(np.asarray(pending["observations"]).shape[0])
    new_count = int(np.asarray(new_arrays["observations"]).shape[0])
    combined: dict[str, np.ndarray] = {}
    for name in AUDIT_ARRAY_FIELDS:
        old = np.asarray(pending[name])
        new = np.asarray(new_arrays[name])
        if old.shape[1:] != new.shape[1:] or old.dtype != new.dtype:
            raise ValueError(f"pending/new field contract differs for {name}")
        combined[name] = np.concatenate((old, new), axis=0)
    finalized, next_pending, metadata = branch_auditable_arrays_with_pending(
        combined,
        reward_name=reward_name,
        n_step=n_step,
        gamma=gamma,
        global_environments=global_environments,
    )
    return finalized, next_pending, {
        **metadata,
        "input_pending_rows": pending_count,
        "new_rows": new_count,
    }
