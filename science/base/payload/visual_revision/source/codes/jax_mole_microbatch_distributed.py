#!/usr/bin/env python3
"""Two-host Mole SAC benchmark with time-sliced LES microbatches.

Each host retains 16 independent environment states, but advances them in
small microbatches before performing the source-aligned distributed SAC update.
This avoids the poor per-environment efficiency of one batch-16 LES graph while
preserving the public 32-environment, 512-transition collector cadence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_mole_compact_sac_benchmark import (
    ReplayState,
    SACState,
    build_flow_functions,
    init_replay,
    init_sac,
    layout_contract,
    make_mole_training_case,
    make_scaled_mole_training_case,
    replay_add,
    sac_update,
    sac_update_entropy_scaled,
)
from jax_mole_layouts import canonical_scale_layouts
from jax_mole_direction_geometry import (
    common_direction_center,
    rotate_layout_to_flow_frame,
)
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    actor_deterministic,
    control_action_dim,
    initialize_continuous_training,
    initialize_evaluation_from_training,
    load_actor_checkpoint,
    load_initial_precursor,
    make_continuous_episode_reset,
    make_continuous_training_iteration,
    make_exact_svg_probe,
    make_evaluation_step,
    make_precursor_plane_generator,
    protocol_metadata,
    save_actor_checkpoint,
    slice_initial_precursor,
)
from jax_mole_crossq import crossq_update, init_crossq
from jax_mole_graph_sac_v2 import (
    WakeGraph,
    build_wake_graph,
    graph_v2_actor_sample,
    graph_v2_sac_update,
    init_graph_v2_sac,
)
from jax_mole_graph_sac_directional import (
    DirectionalWakeGraph,
    append_direction_branch_condition,
    append_direction_condition,
    build_directional_graph,
    directional_graph_actor_sample,
    directional_graph_sac_update,
    equivariant_directional_graph_actor_sample,
    init_directional_graph_sac,
    steering_branch_for_shard,
)
from mole_plane_archive import PlaneArchive
from jax_mole_n9_direction_scheduler import (
    archive_member_bounds,
    episode_plane_window_start,
    local_directions_for_reset,
    split_collection_segments,
    validate_formal_direction_protocol,
)
from jax_mole_n9_million_protocol import TRAIN_DIRECTIONS_DEG
from jax_mole_n9_million_protocol import (
    GLOBAL_ENVIRONMENTS,
    P0_MW_PER_TURBINE,
)
from jax_mole_n9_auditable_replay import (
    AUDIT_ARRAY_FIELDS,
    AuditableReplayState,
    add_auditable_batch,
    init_auditable_replay,
)
from mole_n9_replay_branching import advance_auditable_n_step
from mole_training_horizon import plane_horizon


IMPLEMENTATION_FILES = (
    "jax_mole_microbatch_distributed.py",
    "jax_mole_continuous_sac.py",
    "jax_mole_compact_sac_benchmark.py",
    "jax_mole_layouts.py",
    "mole_plane_archive.py",
    "incompact3d_staggered.py",
    "jax_mole_crossq.py",
    "jax_mole_graph_sac_v2.py",
    "jax_mole_graph_sac_directional.py",
    "jax_mole_direction_geometry.py",
    "jax_mole_n9_auditable_replay.py",
    "jax_mole_n9_direction_scheduler.py",
    "jax_mole_n9_million_protocol.py",
    "mole_n9_replay_branching.py",
    "mole_training_horizon.py",
)

FULL_CHECKPOINT_SCHEMA_VERSION = 2
COMPACT_REPLAY_FORMAT = "active_prefix_v1"
AUDITABLE_COMPACT_REPLAY_FORMAT = "auditable_active_prefix_v1"
REPLAY_ARRAY_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "dones",
)


class LearnerState(NamedTuple):
    replay: ReplayState
    sac: SACState
    key: jax.Array
    replay_checksum: jax.Array


class TransitionBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    dones: jax.Array


class AuditableTransitionBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    raw_turbine_power_w: jax.Array
    applied_yaw_deg: jax.Array
    p0_mw_per_turbine: jax.Array
    mean_power_mw: jax.Array
    barrier_mw: jax.Array
    relative_power_gain: jax.Array
    reward_r0: jax.Array
    reward_r1: jax.Array
    direction_deg: jax.Array
    episode_ids: jax.Array
    sequence_ids: jax.Array
    bootstrap_discounts: jax.Array
    n_step_horizons: jax.Array


class SVGGradientBatch(NamedTuple):
    observations: jax.Array
    action_gradients: jax.Array
    objectives: jax.Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument(
        "--local-device-id",
        type=int,
        default=0,
        help=(
            "Host-visible CUDA device ordinal assigned to this process. Keep "
            "the default when each process sees one GPU; use the process rank "
            "for one-node multi-process runs where every process sees all GPUs."
        ),
    )
    parser.add_argument(
        "--h20-eight-process-performance-benchmark",
        action="store_true",
        help=(
            "Enable the frozen 8-process x 4-environment XT-H20 profiling "
            "topology with immediate learner/Actor updates. This is a "
            "performance diagnostic, not a resumable scientific-training run."
        ),
    )
    parser.add_argument(
        "--h20-eight-process-formal-training",
        action="store_true",
        help=(
            "Enable the resumable 8-process x 4-environment XT-H20 formal "
            "training topology while retaining the frozen 32k Replay and "
            "Actor warm-up gates."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Write a JAX/XProf trace for one selected training iteration.",
    )
    parser.add_argument(
        "--profile-iteration",
        type=int,
        default=0,
        help="One-based iteration to trace; zero disables profiling.",
    )
    parser.add_argument(
        "--profile-process-id",
        type=int,
        default=0,
        help="Distributed process rank that records the XProf trace.",
    )
    parser.add_argument(
        "--precursor",
        type=Path,
        help=(
            "Complete precursor state for online generation, or an optional "
            "source-state integrity cross-check when --plane-archive is used."
        ),
    )
    parser.add_argument(
        "--plane-archive",
        type=Path,
        help=(
            "Completed chunked inlet-plane archive. Requires "
            "--inlet-mode time_planes and replaces online plane generation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--initial-actor-checkpoint",
        type=Path,
        help=(
            "Initialize a fresh directional Graph-SAC Actor from a validated "
            "actor-only checkpoint; Replay and Critic remain fresh."
        ),
    )
    parser.add_argument(
        "--checkpoint-milestones",
        default="",
        help=(
            "Comma-separated positive global interaction thresholds. Process "
            "0 writes Actor snapshots and every process writes a complete "
            "local state at each threshold. A fresh formal N=9 run also "
            "writes explicit step-0 Actor/full-state checkpoints."
        ),
    )
    parser.add_argument("--save-initial-checkpoint", action="store_true",
                        help="Save step-0 Actor and full state for any fresh layout.")
    parser.add_argument("--full-checkpoint-interval", type=int, default=0,
                        help="Maximum new global interactions between rolling full saves; zero disables.")
    parser.add_argument("--stop-request", type=Path,
                        help="Shared file requesting a full checkpoint and synchronized block-boundary pause.")
    parser.add_argument("--stop-after-seconds", type=float,
                        help="Request a synchronized recoverable pause after this elapsed process time.")
    parser.add_argument("--planned-global-interactions", type=int,
                        help="Require this exact final global budget and check the whole continuous-input horizon.")
    parser.add_argument("--local-environments", type=int, default=16)
    parser.add_argument(
        "--target-turbines",
        type=int,
        choices=(3, 9, 25, 50, 81, 100),
        default=3,
        help="Canonical farm layout. Scaled layouts require cached time planes.",
    )
    parser.add_argument(
        "--action-parameterization",
        choices=("per_turbine", "per_turbine_target", "n9_two_column"),
        default="per_turbine",
        help=(
            "Policy/Critic action representation. n9_two_column shares one "
            "action across each of the first two columns and fixes the third "
            "column at zero yaw rate."
        ),
    )
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--collection-steps", type=int, default=16)
    parser.add_argument("--burnin-interactions", type=int, default=0)
    parser.add_argument("--episode-interactions", type=int, default=500)
    parser.add_argument("--reset-interactions", type=int, default=150)
    parser.add_argument(
        "--inlet-mode",
        choices=("online", "replay_volume", "time_planes"),
        default="replay_volume",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("global", "local_stratified"),
        default="global",
        help=(
            "global all-gathers each collector batch before appending to an "
            "identical central-replay replica on every host"
        ),
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        choices=("sac", "crossq", "graph_sac_v2", "graph_sac_directional"),
        default="sac",
    )
    parser.add_argument(
        "--relative-wind-angle-deg",
        type=float,
        default=0.0,
        help="Rotate the farm into the aligned-flow frame by this relative angle.",
    )
    parser.add_argument(
        "--n9-million-direction-schedule",
        action="store_true",
        help="Enable the frozen per-episode nine-direction N=9 protocol.",
    )
    parser.add_argument(
        "--n9-direction-schedule-seed", type=int, default=2026081802
    )
    parser.add_argument(
        "--formal-reward-name",
        choices=("reward_r0", "reward_r1"),
        default="reward_r0",
        help="Selected learner reward stored alongside both audited rewards.",
    )
    parser.add_argument(
        "--formal-n-step",
        type=int,
        choices=(1, 8, 16),
        default=1,
        help="Frozen R0-R3 return horizon for the auditable Replay.",
    )
    parser.add_argument(
        "--plane-offset-add",
        type=int,
        default=0,
        help="Non-negative offset added to every archive member start cursor.",
    )
    parser.add_argument(
        "--episode-random-plane-windows",
        action="store_true",
        help=(
            "At each formal episode boundary, deterministically select a "
            "contiguous precursor window inside each archive member."
        ),
    )
    parser.add_argument(
        "--plane-window-seed",
        type=int,
        default=2026082201,
        help="Frozen deterministic seed for episode precursor-window selection.",
    )
    parser.add_argument(
        "--direction-condition",
        action="store_true",
        help="Append sin/cos relative wind direction to every Replay observation.",
    )
    parser.add_argument(
        "--steering-branch-condition",
        action="store_true",
        help=(
            "Append a balanced global +/-1 wake-steering branch after the "
            "direction condition. Requires directional Graph-SAC."
        ),
    )
    parser.add_argument(
        "--graph-geometry",
        choices=("case", "canonical"),
        default="case",
        help="Use the current rotated case or canonical physical layout for sparse Graph-SAC.",
    )
    parser.add_argument(
        "--reflection-equivariant-actor",
        action="store_true",
        help="Apply exact N=9 reflection group averaging to the directional Actor.",
    )
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=1_000_448)
    parser.add_argument("--minimum-replay-size", type=int, default=32000)
    parser.add_argument(
        "--random-action-transitions",
        type=int,
        default=32000,
        help="Global random-transition budget across all environments.",
    )
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        help=(
            "Base path for atomic process-specific full-state checkpoints; "
            "'.processN.pkl' is appended on each host."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Process-specific full-state checkpoint written by this process.",
    )
    parser.add_argument(
        "--resume-learner-only",
        action="store_true",
        help=(
            "Restore only the shared SAC learner and Replay while creating "
            "fresh LES actor states for a new conditioned wind direction."
        ),
    )
    parser.add_argument(
        "--resume-allow-implementation-change",
        action="store_true",
        help=(
            "Allow a resume checkpoint whose only configuration difference "
            "is implementation hashes. Intended for audited algorithm forks."
        ),
    )
    parser.add_argument(
        "--resume-allow-plane-archive-extension",
        action="store_true",
        help=(
            "Allow an audited complete continuation manifest to extend the "
            "base plane archive recorded by an older full-state checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-allow-diagnostic-change",
        action="store_true",
        help=(
            "Allow only explicitly recorded short-isolation overrides to "
            "differ from a resume checkpoint."
        ),
    )
    parser.add_argument(
        "--diagnostic-updates-per-iteration",
        type=int,
        default=-1,
        help="Override the derived learner updates per collector iteration.",
    )
    parser.add_argument(
        "--diagnostic-resume-alpha",
        type=float,
        help="After resume, replace alpha and reset only its optimizer state.",
    )
    parser.add_argument(
        "--diagnostic-reward-scale",
        type=float,
        default=1.0,
        help="Scale Replay rewards only when constructing SAC targets.",
    )
    parser.add_argument(
        "--diagnostic-actor-update-period",
        type=int,
        default=1,
        help="Update Actor and alpha every N Critic updates; zero freezes both.",
    )
    parser.add_argument(
        "--diagnostic-actor-learning-rate-scale",
        type=float,
        default=1.0,
        help="Multiply only the resumed SAC Actor optimizer learning rate.",
    )
    parser.add_argument(
        "--diagnostic-action-l2-coefficient",
        type=float,
        default=0.0,
        help="Penalty on mean squared sampled action normalized by its bound.",
    )
    parser.add_argument(
        "--diagnostic-graph-action-l2-coefficient",
        type=float,
        help="Override Graph-SAC action L2 only for an audited resume diagnostic.",
    )
    parser.add_argument(
        "--diagnostic-disable-reflection-equivariant-actor",
        action="store_true",
        help=(
            "Use the underlying directional Actor without reflection-group "
            "averaging only for an audited resume diagnostic."
        ),
    )
    parser.add_argument(
        "--diagnostic-entropy-scale",
        type=float,
        default=1.0,
        help=(
            "Scale joint log probability consistently in the SAC target, "
            "Actor loss and alpha loss. Use 3/N for Mole-scale extensions."
        ),
    )
    parser.add_argument(
        "--svg-gradient-samples-per-host",
        type=int,
        default=0,
        help="Exact 50-step action gradients computed per host and iteration.",
    )
    parser.add_argument(
        "--svg-actor-updates-per-iteration", type=int, default=0
    )
    parser.add_argument("--svg-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--svg-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--initial-alpha", type=float, default=10.0)
    parser.add_argument(
        "--actor-learning-rate-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--graph-actor-warmup-updates",
        type=int,
        default=0,
        help="Critic-only updates before Graph-SAC v2 starts Actor updates.",
    )
    parser.add_argument(
        "--graph-actor-minimum-replay-size",
        type=int,
        default=0,
        help=(
            "Replay transitions required before Graph-SAC Actor updates. "
            "This gate is applied in addition to --graph-actor-warmup-updates."
        ),
    )
    parser.add_argument(
        "--graph-actor-update-period",
        type=int,
        default=1,
        help="Apply one Graph-SAC Actor/alpha update every N Critic updates.",
    )
    parser.add_argument(
        "--graph-action-l2-coefficient",
        type=float,
        default=0.0,
        help="Graph-SAC v2 Actor action regularization coefficient.",
    )
    parser.add_argument(
        "--entropy-scale",
        type=float,
        default=1.0,
        help="Formal multiplier on joint SAC entropy; use 1/N for N actions.",
    )
    parser.add_argument(
        "--conservative-actor-init",
        action="store_true",
        help=(
            "Initialize flat SAC at zero mean and minimum policy scale. "
            "Disabled by default to preserve the existing baseline."
        ),
    )
    parser.add_argument(
        "--neutral-actor-initial-scale",
        type=float,
        help=(
            "Pre-tanh Gaussian standard deviation for the zero-mean Graph-SAC "
            "head. Requires --conservative-actor-init and rejects Actor warm starts."
        ),
    )
    parser.add_argument("--crossq-critic-hidden-dim", type=int, default=2048)
    parser.add_argument("--crossq-learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--crossq-entropy-learning-rate", type=float, default=1.0e-3
    )
    parser.add_argument("--crossq-adam-b1", type=float, default=0.5)
    parser.add_argument("--crossq-brn-momentum", type=float, default=0.99)
    parser.add_argument(
        "--crossq-brn-warmup-updates", type=int, default=100_000
    )
    parser.add_argument("--crossq-policy-delay", type=int, default=3)
    parser.add_argument(
        "--crossq-critic-gradient-clip-norm", type=float, default=0.0
    )
    parser.add_argument(
        "--compact-solver",
        choices=("dense_inverse", "cute_fused_41"),
        default="dense_inverse",
        help="Vertical compact derivative implementation for LES advancement.",
    )
    parser.add_argument(
        "--rhs-assembly",
        choices=("tensor", "componentwise"),
        default="tensor",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _copy_actor_into_state(state: Any, actor: Any) -> Any:
    actor_copy = jax.tree.map(jnp.copy, actor)
    return state._replace(sac=state.sac._replace(actor=actor_copy))


def _conservative_flat_actor(actor: Any) -> Any:
    """Set only the flat Actor output head to the zero-yaw prior."""

    layers = list(actor)
    output = layers[-1]
    action_dim = output["b"].shape[0] // 2
    layers[-1] = {
        "w": jnp.zeros_like(output["w"]),
        "b": jnp.concatenate(
            (
                jnp.zeros((action_dim,), dtype=output["b"].dtype),
                jnp.full(
                    (action_dim,), -10.0, dtype=output["b"].dtype
                ),
            )
        ),
    }
    return tuple(layers)


def _transition_batch(states: list[Any]) -> TransitionBatch:
    return TransitionBatch(
        observations=jnp.concatenate(
            [state.replay.observations for state in states], axis=0
        ),
        actions=jnp.concatenate(
            [state.replay.actions for state in states], axis=0
        ),
        rewards=jnp.concatenate(
            [state.replay.rewards for state in states], axis=0
        ),
        next_observations=jnp.concatenate(
            [state.replay.next_observations for state in states], axis=0
        ),
        dones=jnp.concatenate(
            [state.replay.dones for state in states], axis=0
        ),
    )


def _auditable_transition_batch(
    metrics: list[dict[str, Any]],
    *,
    gamma: float,
    reward_name: str = "reward_r0",
) -> AuditableTransitionBatch:
    """Build the one-step formal Replay batch from raw collector evidence."""

    if reward_name not in {"reward_r0", "reward_r1"}:
        raise ValueError("formal reward must be reward_r0 or reward_r1")

    def flatten(name: str) -> jax.Array:
        values = [item["raw_transitions"][name] for item in metrics]
        value = (
            jnp.stack(values, axis=1)
            if values[0].ndim == 1
            else jnp.concatenate(values, axis=1)
        )
        return value.reshape((-1, *value.shape[2:]))

    observations = flatten("observation")
    actions = flatten("action")
    next_observations = flatten("next_observation")
    dones = flatten("done")
    power = flatten("turbine_power_w")
    yaw = flatten("applied_yaw_deg")
    directions = flatten("direction_deg")
    episode_ids = flatten("episode_id").astype(jnp.int32)
    sequence_ids = flatten("sequence_id").astype(jnp.int32)
    direction_matches = (
        directions[:, None]
        == jnp.asarray(TRAIN_DIRECTIONS_DEG)[None, :]
    )
    p0 = jnp.sum(
        direction_matches
        * jnp.asarray(P0_MW_PER_TURBINE)[None, :],
        axis=1,
    )
    mean_power_mw = jnp.mean(power, axis=1) / np.float32(1.0e6)
    barrier_mw = np.float32(0.5) * jnp.mean(
        (yaw / np.float32(40.0)) ** 26, axis=1
    )
    relative_power_gain = (mean_power_mw - p0) / p0
    reward_r0 = mean_power_mw - barrier_mw
    reward_r1 = relative_power_gain - barrier_mw / p0
    selected_reward = reward_r0 if reward_name == "reward_r0" else reward_r1
    bootstrap_discounts = np.float32(gamma) * (
        np.float32(1.0) - dones
    )
    return AuditableTransitionBatch(
        observations=observations,
        actions=actions,
        rewards=selected_reward,
        next_observations=next_observations,
        dones=dones,
        raw_turbine_power_w=power,
        applied_yaw_deg=yaw,
        p0_mw_per_turbine=p0,
        mean_power_mw=mean_power_mw,
        barrier_mw=barrier_mw,
        relative_power_gain=relative_power_gain,
        reward_r0=reward_r0,
        reward_r1=reward_r1,
        direction_deg=directions,
        episode_ids=episode_ids,
        sequence_ids=sequence_ids,
        bootstrap_discounts=bootstrap_discounts,
        n_step_horizons=jnp.ones_like(episode_ids),
    )


def _empty_auditable_arrays_like(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.empty((0, *np.asarray(value).shape[1:]), dtype=value.dtype)
        for name, value in arrays.items()
    }


def _local_n_step_pending(
    arrays: dict[str, Any],
    *,
    process_id: int,
    local_environments: int,
    global_environments: int,
) -> dict[str, np.ndarray]:
    if set(arrays) != set(AUDIT_ARRAY_FIELDS):
        raise ValueError("checkpoint n-step pending fields are incomplete")
    host = {name: np.asarray(value) for name, value in arrays.items()}
    count = host["sequence_ids"].shape[0]
    if any(value.shape[0] != count for value in host.values()):
        raise ValueError("checkpoint n-step pending arrays have unequal rows")
    environment_ids = host["sequence_ids"].astype(np.int64) % global_environments
    first = process_id * local_environments
    selected = (environment_ids >= first) & (
        environment_ids < first + local_environments
    )
    return {
        name: np.array(value[selected], copy=True) for name, value in host.items()
    }


def _advance_auditable_transition_batch(
    pending: dict[str, np.ndarray] | None,
    transitions: AuditableTransitionBatch,
    *,
    reward_name: str,
    n_step: int,
    gamma: float,
    global_environments: int,
) -> tuple[
    AuditableTransitionBatch,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    raw = {
        name: np.asarray(jax.device_get(value))
        for name, value in transitions._asdict().items()
    }
    active_pending = (
        _empty_auditable_arrays_like(raw) if pending is None else pending
    )
    finalized, next_pending, metadata = advance_auditable_n_step(
        active_pending,
        raw,
        reward_name=reward_name,
        n_step=n_step,
        gamma=gamma,
        global_environments=global_environments,
    )
    device_arrays = {
        name: jax.device_put(value, device=jax.local_devices()[0])
        for name, value in finalized.items()
    }
    result = AuditableTransitionBatch(**device_arrays)
    jax.block_until_ready(result)
    return result, next_pending, metadata


def _required_plane_count(
    state: Any,
    case: Any,
    config: ContinuousControlConfig,
    collection_steps: int,
) -> int:
    start_interaction = int(np.asarray(state.transitions)) // case.batch
    reset_count = sum(
        (start_interaction + offset + 1) % config.episode_interactions == 0
        for offset in range(collection_steps)
    )
    intervals = collection_steps + reset_count * config.reset_interactions
    return intervals * config.les_steps_per_interaction


def _tree_to_host(tree: Any) -> Any:
    return jax.tree.map(lambda value: np.asarray(value), tree)


def _tree_to_device(tree: Any) -> Any:
    return jax.tree.map(lambda value: jnp.asarray(value), tree)


def _unreplicated_array_to_host(value: Any) -> np.ndarray:
    """Transfer a one-replica array to host before removing its replica axis."""
    host = np.asarray(jax.device_get(value))
    if host.ndim < 1 or host.shape[0] != 1:
        raise ValueError(
            "expected a leading singleton replica axis, got "
            f"shape={host.shape}"
        )
    return np.array(host[0], copy=True)


def _unreplicated_tree_to_host(tree: Any) -> Any:
    return jax.tree.map(_unreplicated_array_to_host, tree)


def _unreplicated_tree_to_local_device(tree: Any) -> Any:
    """Detach one distributed replica before use by local LES collectors."""
    device = jax.local_devices()[0]

    def copy_leaf(value: Any) -> jax.Array:
        host = _unreplicated_array_to_host(value)
        return jax.device_put(host, device=device)

    local = jax.tree.map(copy_leaf, tree)
    jax.block_until_ready(local)
    return local


def _tree_to_sharded_device(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: jax.device_put(np.expand_dims(np.asarray(value), 0)),
        tree,
    )


def _compact_replay_to_host(
    replay: ReplayState | AuditableReplayState,
) -> dict[str, Any]:
    """Serialize only initialized replay rows without device-side unreplication."""
    size = int(_unreplicated_array_to_host(replay.size).item())
    position = int(_unreplicated_array_to_host(replay.position).item())
    capacity = int(replay.observations.shape[1])
    if not 0 <= size <= capacity:
        raise ValueError(f"invalid replay size {size} for capacity {capacity}")
    if not 0 <= position < capacity:
        raise ValueError(
            f"invalid replay position {position} for capacity {capacity}"
        )
    if size < capacity and position != size:
        raise ValueError(
            "partially filled replay must use a contiguous active prefix: "
            f"position={position}, size={size}"
        )

    arrays: dict[str, np.ndarray] = {}
    full_allocation_bytes = 0
    stored_bytes = 0
    auditable = isinstance(replay, AuditableReplayState)
    array_fields = AUDIT_ARRAY_FIELDS if auditable else REPLAY_ARRAY_FIELDS
    for name in array_fields:
        value = getattr(replay, name)
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(
                f"replay field {name} lacks a singleton replica axis: "
                f"shape={value.shape}"
            )
        if value.shape[1] != capacity:
            raise ValueError(
                f"replay field {name} capacity {value.shape[1]} != {capacity}"
            )
        # np.asarray performs the device-to-host transfer. Slicing happens only
        # after that transfer, avoiding a capacity-sized GPU squeeze buffer.
        host_full = np.asarray(jax.device_get(value))
        active = np.array(host_full[0, :size], copy=True)
        full_allocation_bytes += int(host_full[0].nbytes)
        stored_bytes += int(active.nbytes)
        arrays[name] = active
        del host_full

    return {
        "format": (
            AUDITABLE_COMPACT_REPLAY_FORMAT
            if auditable
            else COMPACT_REPLAY_FORMAT
        ),
        "capacity": capacity,
        "size": size,
        "position": position,
        "arrays": arrays,
        "stored_bytes": stored_bytes,
        "full_allocation_bytes": full_allocation_bytes,
    }


def _compact_replay_to_sharded_device(
    payload: dict[str, Any],
) -> ReplayState | AuditableReplayState:
    replay_format = payload.get("format")
    if replay_format not in {
        COMPACT_REPLAY_FORMAT,
        AUDITABLE_COMPACT_REPLAY_FORMAT,
    }:
        raise ValueError(
            f"unsupported compact replay format: {payload.get('format')!r}"
        )
    capacity = int(payload["capacity"])
    size = int(payload["size"])
    position = int(payload["position"])
    if capacity <= 0 or not 0 <= size <= capacity:
        raise ValueError(
            f"invalid compact replay capacity/size: {capacity}/{size}"
        )
    if not 0 <= position < capacity:
        raise ValueError(
            f"invalid compact replay position {position} for {capacity}"
        )
    if size < capacity and position != size:
        raise ValueError(
            "partially filled compact replay is not an active prefix"
        )

    saved_arrays = payload["arrays"]
    restored: dict[str, jax.Array] = {}
    auditable = replay_format == AUDITABLE_COMPACT_REPLAY_FORMAT
    array_fields = AUDIT_ARRAY_FIELDS if auditable else REPLAY_ARRAY_FIELDS
    for name in array_fields:
        active = np.asarray(saved_arrays[name])
        if active.shape[0] != size:
            raise ValueError(
                f"compact replay field {name} has {active.shape[0]} rows, "
                f"expected {size}"
            )
        host_full = np.zeros(
            (1, capacity, *active.shape[1:]), dtype=active.dtype
        )
        host_full[0, :size] = active
        device_full = jax.device_put(host_full)
        jax.block_until_ready(device_full)
        restored[name] = device_full
        del host_full

    replay_type = AuditableReplayState if auditable else ReplayState
    return replay_type(
        **restored,
        position=jax.device_put(np.asarray([position], dtype=np.int32)),
        size=jax.device_put(np.asarray([size], dtype=np.int32)),
    )


def _sharded_learner_to_checkpoint(
    learner: LearnerState,
) -> dict[str, Any]:
    return {
        "replay": _compact_replay_to_host(learner.replay),
        "sac": _unreplicated_tree_to_host(learner.sac),
        "key": _unreplicated_array_to_host(learner.key),
        "replay_checksum": _unreplicated_array_to_host(
            learner.replay_checksum
        ),
    }


def _checkpoint_to_sharded_learner(payload: dict[str, Any]) -> LearnerState:
    replay_payload = payload["replay"]
    if (
        isinstance(replay_payload, dict)
        and replay_payload.get("format")
        in {COMPACT_REPLAY_FORMAT, AUDITABLE_COMPACT_REPLAY_FORMAT}
    ):
        return LearnerState(
            replay=_compact_replay_to_sharded_device(replay_payload),
            sac=_tree_to_sharded_device(payload["sac"]),
            key=_tree_to_sharded_device(payload["key"]),
            replay_checksum=_tree_to_sharded_device(
                payload["replay_checksum"]
            ),
        )

    # Schema v1 stored a full, unreplicated ReplayState.
    restored = LearnerState(**_tree_to_device(payload))
    sharded = jax.tree.map(lambda value: value[None], restored)
    del restored
    return sharded


def _process_checkpoint_path(base: Path, process_id: int) -> Path:
    suffix = base.suffix or ".pkl"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}.process{process_id}{suffix}")


def _milestone_checkpoint_base(base: Path, interactions: int) -> Path:
    return base.with_name(f"{base.stem}_step{interactions}{base.suffix}")


def _checkpoint_milestone_paths(
    actor_base: Path,
    full_base: Path,
    interactions: int,
    process_id: int,
) -> dict[str, Path]:
    paths = {
        "full_state_process_local": _process_checkpoint_path(
            _milestone_checkpoint_base(full_base, interactions), process_id
        )
    }
    if process_id == 0:
        paths["actor_process0_only"] = _milestone_checkpoint_base(
            actor_base, interactions
        )
    return paths


def _full_checkpoint_payload(
    *,
    args: argparse.Namespace,
    source_precursor_sha256: str,
    plane_archive_sha256: str | None,
    plane_archive_extension_sha256: str | None,
    case: Any,
    actor_states: list[Any],
    plane_source_states: list[Any],
    plane_cursors: list[int],
    sharded_learner: LearnerState,
    n_step_pending: dict[str, np.ndarray] | None,
    history: list[dict[str, Any]],
    global_interactions: int,
    completed_collector_iterations: int,
    algorithm_extension: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FULL_CHECKPOINT_SCHEMA_VERSION,
        "purpose": "mole_protocol_parity_full_training_state",
        "process_id": args.process_id,
        "num_processes": args.num_processes,
        "completed_collector_iterations_including_warmup": (
            completed_collector_iterations
        ),
        "global_interactions": global_interactions,
        "configuration": _full_checkpoint_configuration(
            args,
            source_precursor_sha256,
            plane_archive_sha256,
            case,
        ),
        "plane_archive_extension_sha256": plane_archive_extension_sha256,
        "actor_states": _tree_to_host(tuple(actor_states)),
        "plane_source_states": _tree_to_host(tuple(plane_source_states)),
        "plane_cursors": list(plane_cursors),
        "learner": _sharded_learner_to_checkpoint(sharded_learner),
        "n_step_pending": n_step_pending,
        "history": history,
        "algorithm_extension": algorithm_extension,
    }


def _save_full_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    code_dir = Path(__file__).resolve().parent
    return {
        name: _file_sha256(code_dir / name)
        for name in IMPLEMENTATION_FILES
    }


def _load_full_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") not in {
        1,
        FULL_CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported full-checkpoint schema: {path}")
    return payload


def _full_checkpoint_configuration(
    args: argparse.Namespace,
    source_precursor_sha256: str,
    plane_archive_sha256: str | None,
    case: Any,
) -> dict[str, Any]:
    return {
        "h20_eight_process_performance_benchmark": (
            args.h20_eight_process_performance_benchmark
        ),
        "h20_eight_process_formal_training": (
            args.h20_eight_process_formal_training
        ),
        "target_turbines": args.target_turbines,
        "action_parameterization": args.action_parameterization,
        "layout_contract": layout_contract(case),
        "local_environments": args.local_environments,
        "microbatch": args.microbatch,
        "collection_steps": args.collection_steps,
        "burnin_interactions": args.burnin_interactions,
        "episode_interactions": args.episode_interactions,
        "reset_interactions": args.reset_interactions,
        "inlet_mode": args.inlet_mode,
        "replay_mode": args.replay_mode,
        "formal_reward_name": args.formal_reward_name,
        "formal_n_step": args.formal_n_step,
        "replay_capacity": args.replay_capacity,
        "minimum_replay_size": args.minimum_replay_size,
        "random_action_transitions": args.random_action_transitions,
        "updates_per_transition": args.updates_per_transition,
        "algorithm": args.algorithm,
        "relative_wind_angle_deg": args.relative_wind_angle_deg,
        "plane_offset_add": args.plane_offset_add,
        "episode_random_plane_windows": args.episode_random_plane_windows,
        "plane_window_seed": args.plane_window_seed,
        "direction_condition": args.direction_condition,
        "steering_branch_condition": args.steering_branch_condition,
        "initial_actor_checkpoint_sha256": (
            _file_sha256(args.initial_actor_checkpoint)
            if args.initial_actor_checkpoint is not None
            else None
        ),
        "graph_geometry": args.graph_geometry,
        "reflection_equivariant_actor": args.reflection_equivariant_actor,
        "sample_size_per_host": args.sample_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "initial_alpha": args.initial_alpha,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "graph_actor_warmup_updates": args.graph_actor_warmup_updates,
        "graph_actor_minimum_replay_size": (
            args.graph_actor_minimum_replay_size
        ),
        "graph_actor_update_period": args.graph_actor_update_period,
        "graph_action_l2_coefficient": args.graph_action_l2_coefficient,
        "entropy_scale": args.entropy_scale,
        "conservative_actor_init": args.conservative_actor_init,
        "neutral_actor_initial_scale": args.neutral_actor_initial_scale,
        "crossq_critic_hidden_dim": args.crossq_critic_hidden_dim,
        "crossq_learning_rate": args.crossq_learning_rate,
        "crossq_entropy_learning_rate": (
            args.crossq_entropy_learning_rate
        ),
        "crossq_adam_b1": args.crossq_adam_b1,
        "crossq_brn_momentum": args.crossq_brn_momentum,
        "crossq_brn_warmup_updates": args.crossq_brn_warmup_updates,
        "crossq_policy_delay": args.crossq_policy_delay,
        "crossq_critic_gradient_clip_norm": (
            args.crossq_critic_gradient_clip_norm
        ),
        "compact_solver": args.compact_solver,
        "rhs_assembly": args.rhs_assembly,
        "seed": args.seed,
        "diagnostic_overrides": {
            "updates_per_iteration": args.diagnostic_updates_per_iteration,
            "resume_alpha": args.diagnostic_resume_alpha,
            "reward_scale": args.diagnostic_reward_scale,
            "actor_update_period": args.diagnostic_actor_update_period,
            "actor_learning_rate_scale": (
                args.diagnostic_actor_learning_rate_scale
            ),
            "action_l2_coefficient": (
                args.diagnostic_action_l2_coefficient
            ),
            "graph_action_l2_coefficient": (
                args.diagnostic_graph_action_l2_coefficient
            ),
            "disable_reflection_equivariant_actor": (
                args.diagnostic_disable_reflection_equivariant_actor
            ),
            "entropy_scale": args.diagnostic_entropy_scale,
        },
        "source_precursor_sha256": source_precursor_sha256,
        "plane_archive_sha256": plane_archive_sha256,
        "implementation_sha256": _implementation_sha256(),
        "protocol": protocol_metadata(case, ContinuousControlConfig(
            inlet_mode=args.inlet_mode,
            tau=(1.0 if args.algorithm == "crossq" else 0.005),
            sample_size=args.sample_size,
            minimum_replay_size=args.minimum_replay_size,
            random_action_transitions=(
                args.random_action_transitions
                * args.microbatch
                // (args.local_environments * args.num_processes)
            ),
            updates_per_transition=args.updates_per_transition,
            initial_alpha=args.initial_alpha,
            entropy_learning_rate=(
                args.crossq_entropy_learning_rate
                if args.algorithm == "crossq"
                else 1.0e-5
            ),
            episode_interactions=args.episode_interactions,
            reset_interactions=args.reset_interactions,
        ), args.action_parameterization),
    }


def _make_learner_step(
    optimizers: tuple[Any, Any, Any],
    config: ContinuousControlConfig,
    updates_per_iteration: int,
    replay_mode: str,
    action_dim: int,
    algorithm: str,
    crossq_brn_momentum: float,
    crossq_brn_warmup_updates: int,
    crossq_policy_delay: int,
    crossq_critic_gradient_clip_norm: float,
    reward_scale: float,
    actor_update_period: int,
    action_l2_coefficient: float,
    entropy_scale: float,
    wake_graph: WakeGraph | DirectionalWakeGraph | None,
    graph_actor_warmup_updates: int,
    graph_actor_minimum_replay_size: int,
    reflection_equivariant_actor: bool,
):
    def step(
        learner: LearnerState,
        transitions: TransitionBatch | AuditableTransitionBatch,
    ) -> tuple[LearnerState, dict[str, jax.Array]]:
        if replay_mode == "global":
            transitions = jax.tree.map(
                lambda value: jax.lax.all_gather(
                    value, "hosts", axis=0, tiled=True
                ),
                transitions,
            )
        transition_replica_max_abs_delta = jnp.max(
            jnp.stack(
                [
                    jnp.max(
                        jnp.abs(
                            jax.lax.pmax(value, "hosts")
                            - jax.lax.pmin(value, "hosts")
                        )
                    )
                    for value in jax.tree.leaves(transitions)
                ]
            )
        )
        if isinstance(learner.replay, AuditableReplayState):
            if not isinstance(transitions, AuditableTransitionBatch):
                raise TypeError("auditable Replay requires auditable transitions")
            replay = add_auditable_batch(
                learner.replay, **transitions._asdict()
            )
        else:
            replay = replay_add(
                learner.replay,
                transitions.observations,
                transitions.actions,
                transitions.rewards,
                transitions.next_observations,
                transitions.dones,
            )
        transition_checksum = sum(
            jnp.sum(value) for value in jax.tree.leaves(transitions)
        )
        replay_checksum = learner.replay_checksum + transition_checksum

        def update_once(
            carry: tuple[SACState, jax.Array],
            _: None,
        ) -> tuple[
            tuple[SACState, jax.Array], dict[str, jax.Array]
        ]:
            sac, key = carry
            key, update_key = jax.random.split(key)
            if algorithm == "crossq":
                sac, losses = crossq_update(
                    sac,
                    replay,
                    update_key,
                    optimizers,
                    config.sample_size,
                    1.0,
                    config.gamma,
                    -float(action_dim),
                    axis_name="hosts",
                    policy_mode=config.policy_mode,
                    momentum=crossq_brn_momentum,
                    warmup_updates=crossq_brn_warmup_updates,
                    policy_delay=crossq_policy_delay,
                    critic_gradient_clip_norm=(
                        crossq_critic_gradient_clip_norm
                    ),
                )
            elif algorithm == "graph_sac_v2":
                if wake_graph is None:
                    raise ValueError("Graph-SAC v2 requires a wake graph")
                sac, losses = graph_v2_sac_update(
                    sac,
                    replay,
                    update_key,
                    optimizers,
                    sample_size=config.sample_size,
                    action_bound=1.0,
                    gamma=config.gamma,
                    tau=config.tau,
                    target_entropy=-float(action_dim),
                    entropy_scale=entropy_scale,
                    graph=wake_graph,
                    frame_stack=config.frame_stack,
                    axis_name="hosts",
                    reward_scale=reward_scale,
                    actor_update_period=actor_update_period,
                    actor_warmup_updates=graph_actor_warmup_updates,
                    actor_minimum_replay_size=(
                        graph_actor_minimum_replay_size
                    ),
                    action_l2_coefficient=action_l2_coefficient,
                )
            elif algorithm == "graph_sac_directional":
                if wake_graph is None or not isinstance(
                    wake_graph, DirectionalWakeGraph
                ):
                    raise ValueError(
                        "directional Graph-SAC requires a directional graph"
                    )
                sac, losses = directional_graph_sac_update(
                    sac,
                    replay,
                    update_key,
                    optimizers,
                    sample_size=config.sample_size,
                    action_bound=1.0,
                    gamma=config.gamma,
                    tau=config.tau,
                    target_entropy=-float(action_dim),
                    entropy_scale=entropy_scale,
                    graph=wake_graph,
                    frame_stack=config.frame_stack,
                    axis_name="hosts",
                    reward_scale=reward_scale,
                    actor_update_period=actor_update_period,
                    actor_warmup_updates=graph_actor_warmup_updates,
                    actor_minimum_replay_size=(
                        graph_actor_minimum_replay_size
                    ),
                    action_l2_coefficient=action_l2_coefficient,
                    reflection_equivariant_actor=(
                        reflection_equivariant_actor
                    ),
                )
            else:
                update_function = (
                    sac_update
                    if entropy_scale == 1.0
                    else sac_update_entropy_scaled
                )
                update_kwargs = {
                    "axis_name": "hosts",
                    "policy_mode": config.policy_mode,
                    "reward_scale": reward_scale,
                    "actor_update_period": actor_update_period,
                    "action_l2_coefficient": action_l2_coefficient,
                }
                if entropy_scale != 1.0:
                    update_kwargs["entropy_scale"] = entropy_scale
                sac, losses = update_function(
                    sac,
                    replay,
                    update_key,
                    optimizers,
                    config.sample_size,
                    1.0,
                    config.gamma,
                    config.tau,
                    -float(action_dim),
                    **update_kwargs,
                )
            return (sac, key), losses

        enough_samples = replay.size >= config.minimum_replay_size

        def perform_updates(
            operand: tuple[SACState, jax.Array],
        ) -> tuple[SACState, jax.Array, dict[str, jax.Array]]:
            (sac, key), losses = jax.lax.scan(
                update_once,
                operand,
                xs=None,
                length=updates_per_iteration,
            )
            final_losses = jax.tree.map(
                lambda value: value[-1], losses
            )
            if algorithm != "crossq":
                final_losses["actor_updates_applied"] = jnp.sum(
                    losses["actor_updated"]
                )
            return (
                sac,
                key,
                final_losses,
            )

        zero_losses = {
            "critic_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "actor_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "alpha": jnp.exp(learner.sac.log_alpha),
        }
        if algorithm == "crossq":
            zero_losses.update(
                {
                    "actor_updated": jnp.asarray(0, dtype=jnp.int32),
                    "current_q_mean": jnp.asarray(0.0, dtype=jnp.float32),
                    "next_q_mean": jnp.asarray(0.0, dtype=jnp.float32),
                    "brn_steps": jnp.asarray(0, dtype=jnp.int32),
                    "critic_gradient_norm_raw": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "critic_gradient_norm_applied": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                }
            )
        else:
            zero_losses.update(
                {
                    "actor_updated": jnp.asarray(0, dtype=jnp.int32),
                    "actor_updates_applied": jnp.asarray(
                        0, dtype=jnp.int32
                    ),
                    "actor_gradient_norm": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "critic_gradient_norm": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "target_q_mean": jnp.asarray(0.0, dtype=jnp.float32),
                    "target_q_std": jnp.asarray(0.0, dtype=jnp.float32),
                    "actor_q_mean": jnp.asarray(0.0, dtype=jnp.float32),
                    "entropy_term_mean": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "action_l2_mean": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "sampled_action_abs_mean": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                    "sampled_action_abs_max": jnp.asarray(
                        0.0, dtype=jnp.float32
                    ),
                }
            )
            if algorithm in {"graph_sac_v2", "graph_sac_directional"}:
                zero_losses.update(
                    {
                        "critic_target_mean": jnp.asarray(
                            0.0, dtype=jnp.float32
                        ),
                        "critic_target_std": jnp.asarray(
                            0.0, dtype=jnp.float32
                        ),
                        "bootstrap_discount_mean": jnp.asarray(
                            0.0, dtype=jnp.float32
                        ),
                    }
                )

        def skip_updates(
            operand: tuple[SACState, jax.Array],
        ) -> tuple[SACState, jax.Array, dict[str, jax.Array]]:
            return operand[0], operand[1], zero_losses

        if updates_per_iteration > 0:
            sac, key, losses = jax.lax.cond(
                enough_samples,
                perform_updates,
                skip_updates,
                (learner.sac, learner.key),
            )
        else:
            sac, key, losses = (
                learner.sac,
                learner.key,
                zero_losses,
            )
        actor_checksum = sum(
            jnp.sum(value) for value in jax.tree.leaves(sac.actor)
        )
        actor_abs_max = jnp.max(
            jnp.stack(
                [jnp.max(jnp.abs(value)) for value in jax.tree.leaves(sac.actor)]
            )
        )
        critic_parameter_tree = (
            sac.critic["params"] if algorithm == "crossq" else sac.critic
        )
        critic_abs_max = jnp.max(
            jnp.stack(
                [
                    jnp.max(jnp.abs(value))
                    for value in jax.tree.leaves(critic_parameter_tree)
                ]
            )
        )
        parameters_finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(value))
                    for value in jax.tree.leaves(
                        (sac.actor, sac.critic, sac.target_critic, sac.log_alpha)
                    )
                ]
            )
        )
        replica_delta = (
            jax.lax.pmax(actor_checksum, "hosts")
            - jax.lax.pmin(actor_checksum, "hosts")
        )
        replay_checksum_delta = (
            jax.lax.pmax(replay_checksum, "hosts")
            - jax.lax.pmin(replay_checksum, "hosts")
        )
        critic_statistics_checksum = (
            sum(
                jnp.sum(value)
                for value in jax.tree.leaves(sac.critic["statistics"])
            )
            if algorithm == "crossq"
            else jnp.asarray(0.0, dtype=jnp.float32)
        )
        critic_statistics_replica_delta = (
            jax.lax.pmax(critic_statistics_checksum, "hosts")
            - jax.lax.pmin(critic_statistics_checksum, "hosts")
        )
        return (
            LearnerState(
                replay=replay,
                sac=sac,
                key=key,
                replay_checksum=replay_checksum,
            ),
            {
                **losses,
                "replay_size": replay.size,
                "sac_updates": sac.updates,
                "actor_replica_checksum_delta": replica_delta,
                "replay_replica_checksum_delta": replay_checksum_delta,
                "transition_replica_max_abs_delta": (
                    transition_replica_max_abs_delta
                ),
                "critic_statistics_replica_checksum_delta": (
                    critic_statistics_replica_delta
                ),
                "learner_transition_batch": jnp.asarray(
                    transitions.observations.shape[0], dtype=jnp.int32
                ),
                "actor_abs_max": actor_abs_max,
                "critic_abs_max": critic_abs_max,
                "parameters_finite": parameters_finite,
            },
        )

    return step


def _configuration_differs_only_in_implementation(
    saved: dict[str, Any], expected: dict[str, Any]
) -> bool:
    saved_semantics = {
        key: value for key, value in saved.items() if key != "implementation_sha256"
    }
    expected_semantics = {
        key: value
        for key, value in expected.items()
        if key != "implementation_sha256"
    }
    return saved_semantics == expected_semantics


def _normalize_saved_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Add defaults introduced after legacy per-turbine checkpoints."""

    normalized = dict(configuration)
    normalized.setdefault("action_parameterization", "per_turbine")
    normalized.setdefault("plane_offset_add", 0)
    normalized.setdefault("steering_branch_condition", False)
    normalized.setdefault("initial_actor_checkpoint_sha256", None)
    normalized.setdefault("graph_actor_update_period", 1)
    normalized.setdefault("formal_n_step", 1)
    diagnostics = dict(normalized.get("diagnostic_overrides", {}))
    diagnostics.setdefault("graph_action_l2_coefficient", None)
    diagnostics.setdefault("disable_reflection_equivariant_actor", False)
    normalized["diagnostic_overrides"] = diagnostics
    return normalized


def _configuration_differs_only_in_diagnostics(
    saved: dict[str, Any], expected: dict[str, Any]
) -> bool:
    ignored = {"implementation_sha256", "diagnostic_overrides"}
    saved_semantics = {
        key: value for key, value in saved.items() if key not in ignored
    }
    expected_semantics = {
        key: value for key, value in expected.items() if key not in ignored
    }
    return saved_semantics == expected_semantics


def _effective_graph_action_l2_coefficient(args: argparse.Namespace) -> float:
    override = args.diagnostic_graph_action_l2_coefficient
    return float(args.graph_action_l2_coefficient if override is None else override)


def _effective_reflection_equivariant_actor(args: argparse.Namespace) -> bool:
    return bool(
        args.reflection_equivariant_actor
        and not args.diagnostic_disable_reflection_equivariant_actor
    )


def _configuration_compatible_for_learner_only(
    saved: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Allow only the physical direction and inlet-window contract to change."""

    environment_keys = {
        "relative_wind_angle_deg",
        "plane_offset_add",
        "layout_contract",
        "protocol",
    }
    saved_learner = {
        key: value for key, value in saved.items() if key not in environment_keys
    }
    expected_learner = {
        key: value for key, value in expected.items() if key not in environment_keys
    }
    return saved_learner == expected_learner


def _make_svg_actor_step(
    actor_optimizer: Any,
    policy_mode: str,
    updates_per_iteration: int,
    gradient_clip_norm: float,
    loss_coefficient: float,
):
    def step(
        learner: LearnerState,
        samples: SVGGradientBatch,
    ) -> tuple[LearnerState, dict[str, jax.Array]]:
        samples = jax.tree.map(
            lambda value: jax.lax.all_gather(
                value, "hosts", axis=0, tiled=True
            ),
            samples,
        )
        raw_norms = jnp.linalg.norm(samples.action_gradients, axis=-1)
        scales = jnp.minimum(
            1.0,
            np.float32(gradient_clip_norm) / jnp.maximum(raw_norms, 1.0e-12),
        )
        clipped_gradients = samples.action_gradients * scales[:, None]

        def update_once(
            carry: tuple[Any, Any], _: None
        ) -> tuple[tuple[Any, Any], jax.Array]:
            actor, actor_opt = carry

            def loss_fn(parameters: Any) -> jax.Array:
                actions = actor_deterministic(
                    parameters, samples.observations, policy_mode
                )
                return -np.float32(loss_coefficient) * jnp.mean(
                    jnp.sum(
                        jax.lax.stop_gradient(clipped_gradients) * actions,
                        axis=-1,
                    )
                )

            loss, gradient = jax.value_and_grad(loss_fn)(actor)
            gradient = jax.lax.pmean(gradient, axis_name="hosts")
            updates, actor_opt = actor_optimizer.update(
                gradient, actor_opt, actor
            )
            return (optax.apply_updates(actor, updates), actor_opt), loss

        if updates_per_iteration > 0:
            (actor, actor_opt), losses = jax.lax.scan(
                update_once,
                (learner.sac.actor, learner.sac.actor_opt),
                xs=None,
                length=updates_per_iteration,
            )
            loss = losses[-1]
        else:
            actor = learner.sac.actor
            actor_opt = learner.sac.actor_opt
            loss = jnp.asarray(0.0, dtype=jnp.float32)
        sac = learner.sac._replace(actor=actor, actor_opt=actor_opt)
        actor_checksum = sum(
            jnp.sum(value) for value in jax.tree.leaves(actor)
        )
        return (
            learner._replace(sac=sac),
            {
                "svg_actor_loss": loss,
                "svg_objective_mean": jnp.mean(samples.objectives),
                "svg_gradient_norm_mean": jnp.mean(raw_norms),
                "svg_gradient_norm_max": jnp.max(raw_norms),
                "svg_gradient_finite": jnp.all(
                    jnp.isfinite(samples.action_gradients)
                ),
                "svg_actor_replica_checksum_delta": (
                    jax.lax.pmax(actor_checksum, "hosts")
                    - jax.lax.pmin(actor_checksum, "hosts")
                ),
            },
        )

    return step


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "jax_use_shardy_partitioner": bool(
            jax.config.jax_use_shardy_partitioner
        ),
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_devices": [str(device) for device in jax.local_devices()],
        "global_devices": [str(device) for device in jax.devices()],
    }


def main() -> int:
    process_started = time.perf_counter()
    args = parse_args()
    validate_formal_direction_protocol(args)
    if args.h20_eight_process_performance_benchmark and any(
        value is not None
        for value in (
            args.checkpoint,
            args.full_checkpoint,
            args.resume_checkpoint,
            args.initial_actor_checkpoint,
        )
    ):
        raise ValueError(
            "the H20 performance benchmark cannot read or write training checkpoints"
        )
    if (
        args.h20_eight_process_performance_benchmark
        and args.h20_eight_process_formal_training
    ):
        raise ValueError("H20 performance and formal modes are mutually exclusive")
    checkpoint_milestones = sorted(
        {
            int(value)
            for value in args.checkpoint_milestones.split(",")
            if value.strip()
        }
    )
    if any(value <= 0 for value in checkpoint_milestones):
        raise ValueError("--checkpoint-milestones values must be positive")
    if args.full_checkpoint_interval < 0:
        raise ValueError("--full-checkpoint-interval must be non-negative")
    if args.stop_after_seconds is not None and args.stop_after_seconds <= 0:
        raise ValueError("--stop-after-seconds must be positive")
    if (args.save_initial_checkpoint or args.full_checkpoint_interval
            or args.stop_request is not None or args.stop_after_seconds is not None):
        if args.full_checkpoint is None:
            raise ValueError("recoverable initial/periodic/stop behavior requires --full-checkpoint")
    if args.save_initial_checkpoint and args.checkpoint is None:
        raise ValueError("--save-initial-checkpoint requires --checkpoint")
    if args.planned_global_interactions is not None and args.episode_random_plane_windows:
        raise ValueError("whole continuous-input horizon currently excludes randomized episode windows")
    if (
        args.n9_million_direction_schedule
        and checkpoint_milestones
        and (args.checkpoint is None or args.full_checkpoint is None)
    ):
        raise ValueError(
            "formal N=9 milestones require both --checkpoint and "
            "--full-checkpoint"
        )
    if args.local_environments % args.microbatch:
        raise ValueError(
            "--microbatch must divide --local-environments exactly"
        )
    if args.burnin_interactions < 0:
        raise ValueError("--burnin-interactions must be non-negative")
    if args.episode_interactions <= 0:
        raise ValueError("--episode-interactions must be positive")
    if args.reset_interactions <= 0:
        raise ValueError("--reset-interactions must be positive")
    if not 0 <= args.svg_gradient_samples_per_host <= args.local_environments:
        raise ValueError(
            "--svg-gradient-samples-per-host must be between zero and "
            "--local-environments"
        )
    if args.svg_actor_updates_per_iteration < 0:
        raise ValueError("--svg-actor-updates-per-iteration cannot be negative")
    if args.svg_gradient_clip_norm <= 0.0:
        raise ValueError("--svg-gradient-clip-norm must be positive")
    if args.svg_loss_coefficient <= 0.0:
        raise ValueError("--svg-loss-coefficient must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.initial_alpha <= 0.0:
        raise ValueError("--initial-alpha must be positive")
    if args.actor_learning_rate_scale <= 0.0:
        raise ValueError("--actor-learning-rate-scale must be positive")
    if not 0.0 < args.entropy_scale <= 1.0:
        raise ValueError("--entropy-scale must be in (0, 1]")
    if (
        args.entropy_scale != 1.0
        and args.diagnostic_entropy_scale != 1.0
    ):
        raise ValueError(
            "formal and diagnostic entropy scales cannot both be changed"
        )
    if args.crossq_critic_hidden_dim <= 0:
        raise ValueError("--crossq-critic-hidden-dim must be positive")
    if args.crossq_learning_rate <= 0.0:
        raise ValueError("--crossq-learning-rate must be positive")
    if args.crossq_entropy_learning_rate <= 0.0:
        raise ValueError(
            "--crossq-entropy-learning-rate must be positive"
        )
    if not 0.0 < args.crossq_adam_b1 < 1.0:
        raise ValueError("--crossq-adam-b1 must be between zero and one")
    if not 0.0 < args.crossq_brn_momentum < 1.0:
        raise ValueError("--crossq-brn-momentum must be between zero and one")
    if args.crossq_brn_warmup_updates < 0:
        raise ValueError("--crossq-brn-warmup-updates cannot be negative")
    if args.crossq_policy_delay < 1:
        raise ValueError("--crossq-policy-delay must be positive")
    if args.crossq_critic_gradient_clip_norm < 0.0:
        raise ValueError(
            "--crossq-critic-gradient-clip-norm cannot be negative"
        )
    if args.diagnostic_updates_per_iteration == 0:
        raise ValueError(
            "--diagnostic-updates-per-iteration must be -1 or positive"
        )
    if args.diagnostic_updates_per_iteration < -1:
        raise ValueError(
            "--diagnostic-updates-per-iteration must be -1 or positive"
        )
    if (
        args.diagnostic_resume_alpha is not None
        and args.diagnostic_resume_alpha <= 0.0
    ):
        raise ValueError("--diagnostic-resume-alpha must be positive")
    if args.diagnostic_reward_scale <= 0.0:
        raise ValueError("--diagnostic-reward-scale must be positive")
    if args.diagnostic_actor_update_period < 0:
        raise ValueError(
            "--diagnostic-actor-update-period cannot be negative"
        )
    if args.diagnostic_actor_learning_rate_scale <= 0.0:
        raise ValueError(
            "--diagnostic-actor-learning-rate-scale must be positive"
        )
    if args.diagnostic_action_l2_coefficient < 0.0:
        raise ValueError(
            "--diagnostic-action-l2-coefficient cannot be negative"
        )
    if (
        args.diagnostic_graph_action_l2_coefficient is not None
        and args.diagnostic_graph_action_l2_coefficient < 0.0
    ):
        raise ValueError(
            "--diagnostic-graph-action-l2-coefficient cannot be negative"
        )
    if args.diagnostic_entropy_scale < 0.0:
        raise ValueError(
            "--diagnostic-entropy-scale cannot be negative"
        )
    sac_diagnostic_requested = (
        args.diagnostic_reward_scale != 1.0
        or args.diagnostic_actor_update_period != 1
        or args.diagnostic_actor_learning_rate_scale != 1.0
        or args.diagnostic_action_l2_coefficient != 0.0
        or args.diagnostic_graph_action_l2_coefficient is not None
        or args.diagnostic_disable_reflection_equivariant_actor
        or args.diagnostic_entropy_scale != 1.0
    )
    if args.algorithm == "crossq" and sac_diagnostic_requested:
        raise ValueError(
            "SAC-specific diagnostic controls do not support CrossQ"
        )
    diagnostic_requested = (
        args.diagnostic_updates_per_iteration != -1
        or args.diagnostic_resume_alpha is not None
        or sac_diagnostic_requested
    )
    if diagnostic_requested and not args.resume_allow_diagnostic_change:
        raise ValueError(
            "diagnostic overrides require --resume-allow-diagnostic-change"
        )
    if diagnostic_requested and args.resume_checkpoint is None:
        raise ValueError("diagnostic overrides require --resume-checkpoint")
    if args.resume_learner_only and args.resume_checkpoint is None:
        raise ValueError("--resume-learner-only requires --resume-checkpoint")
    if args.plane_offset_add < 0:
        raise ValueError("--plane-offset-add cannot be negative")
    if args.episode_random_plane_windows and args.plane_offset_add:
        raise ValueError(
            "--plane-offset-add cannot be combined with episode-random windows"
        )
    if args.episode_random_plane_windows and not args.n9_million_direction_schedule:
        raise ValueError(
            "episode-random plane windows require the formal N=9 schedule"
        )
    if (
        not args.n9_million_direction_schedule
        and (
            args.formal_reward_name != "reward_r0"
            or args.formal_n_step != 1
        )
    ):
        raise ValueError(
            "formal reward/n-step selection requires the N=9 protocol"
        )
    if args.formal_reward_name == "reward_r0" and args.formal_n_step != 1:
        raise ValueError("the declared R0 branch supports only one-step targets")
    svg_enabled = args.svg_gradient_samples_per_host > 0
    if svg_enabled != (args.svg_actor_updates_per_iteration > 0):
        raise ValueError(
            "SVG gradient samples and actor updates must both be enabled or disabled"
        )
    if args.algorithm == "crossq" and svg_enabled:
        raise ValueError("CrossQ and the SVG actor extension cannot be combined")
    graph_algorithms = {"graph_sac_v2", "graph_sac_directional"}
    if (
        args.algorithm not in {"sac", *graph_algorithms}
        and args.conservative_actor_init
    ):
        raise ValueError(
            "--conservative-actor-init supports SAC and Graph-SAC v2 only"
        )
    if args.neutral_actor_initial_scale is not None:
        if not (
            math.isfinite(args.neutral_actor_initial_scale)
            and 0.01 < args.neutral_actor_initial_scale < 1.0
        ):
            raise ValueError("--neutral-actor-initial-scale must be in (0.01, 1)")
        if not (
            args.conservative_actor_init and args.algorithm in graph_algorithms
        ):
            raise ValueError(
                "--neutral-actor-initial-scale requires conservative Graph-SAC"
            )
        if args.initial_actor_checkpoint is not None:
            raise ValueError(
                "neutral Actor initialization cannot load an Actor checkpoint"
            )
    if args.graph_actor_warmup_updates < 0:
        raise ValueError("--graph-actor-warmup-updates cannot be negative")
    if args.graph_actor_minimum_replay_size < 0:
        raise ValueError(
            "--graph-actor-minimum-replay-size cannot be negative"
        )
    if args.graph_actor_update_period < 0:
        raise ValueError("--graph-actor-update-period cannot be negative")
    if args.graph_action_l2_coefficient < 0.0:
        raise ValueError("--graph-action-l2-coefficient cannot be negative")
    if args.algorithm not in graph_algorithms and (
        args.graph_actor_warmup_updates
        or args.graph_actor_minimum_replay_size
    ):
        raise ValueError(
            "Graph-SAC Actor warm-up gates require a Graph-SAC algorithm"
        )
    if (
        args.algorithm not in graph_algorithms
        and args.graph_actor_update_period != 1
    ):
        raise ValueError(
            "--graph-actor-update-period requires a Graph-SAC algorithm"
        )
    if args.algorithm not in graph_algorithms and args.graph_action_l2_coefficient:
        raise ValueError(
            "--graph-action-l2-coefficient requires a Graph-SAC algorithm"
        )
    if args.algorithm in graph_algorithms and svg_enabled:
        raise ValueError("Graph-SAC cannot be combined with the SVG extension")
    if not math.isfinite(args.relative_wind_angle_deg):
        raise ValueError("--relative-wind-angle-deg must be finite")
    if args.direction_condition and args.algorithm not in graph_algorithms:
        raise ValueError("direction conditioning currently requires Graph-SAC")
    if args.algorithm == "graph_sac_directional" and not args.direction_condition:
        raise ValueError("directional Graph-SAC requires --direction-condition")
    if args.steering_branch_condition and not (
        args.algorithm == "graph_sac_directional"
        and args.direction_condition
        and args.reflection_equivariant_actor
    ):
        raise ValueError(
            "steering branch requires direction-conditioned, reflection-"
            "equivariant directional Graph-SAC"
        )
    if args.initial_actor_checkpoint is not None and not (
        args.algorithm == "graph_sac_directional"
        and args.steering_branch_condition
        and args.reflection_equivariant_actor
    ):
        raise ValueError(
            "initial Actor checkpoints currently require branch-conditioned "
            "reflection-equivariant directional Graph-SAC"
        )
    if args.algorithm == "graph_sac_directional" and args.graph_geometry != "canonical":
        raise ValueError("directional Graph-SAC requires canonical graph geometry")
    if args.reflection_equivariant_actor and args.algorithm != "graph_sac_directional":
        raise ValueError("reflection-equivariant Actor requires directional Graph-SAC")
    if (
        args.diagnostic_graph_action_l2_coefficient is not None
        and args.algorithm not in graph_algorithms
    ):
        raise ValueError("diagnostic Graph-SAC action L2 requires Graph-SAC")
    if args.diagnostic_disable_reflection_equivariant_actor and not (
        args.algorithm == "graph_sac_directional"
        and args.reflection_equivariant_actor
    ):
        raise ValueError(
            "diagnostic reflection disable requires an equivariant directional Actor"
        )
    effective_graph_action_l2_coefficient = (
        _effective_graph_action_l2_coefficient(args)
    )
    effective_reflection_equivariant_actor = (
        _effective_reflection_equivariant_actor(args)
    )
    effective_actor_update_period = (
        args.diagnostic_actor_update_period
        if args.diagnostic_actor_update_period != 1
        else (
            args.graph_actor_update_period
            if args.algorithm in graph_algorithms
            else 1
        )
    )
    global_environments = args.local_environments * args.num_processes
    if args.random_action_transitions % global_environments:
        raise ValueError(
            "--random-action-transitions must divide evenly across all "
            "global environments"
        )
    if args.inlet_mode == "time_planes" and args.microbatch != 1:
        raise ValueError(
            "time_planes currently requires --microbatch 1 so each source "
            "trajectory is advanced and audited independently"
        )
    if args.plane_archive is not None and args.inlet_mode != "time_planes":
        raise ValueError("--plane-archive requires --inlet-mode time_planes")
    if args.target_turbines != 3 and (
        args.inlet_mode != "time_planes" or args.plane_archive is None
    ):
        raise ValueError(
            "scaled farms require --inlet-mode time_planes and --plane-archive"
        )
    if (
        args.action_parameterization == "n9_two_column"
        and args.target_turbines != 9
    ):
        raise ValueError("n9_two_column requires --target-turbines 9")
    if svg_enabled and args.plane_archive is None:
        raise ValueError("exact SVG requires a cached time-plane archive")
    plane_archive = (
        PlaneArchive(args.plane_archive)
        if args.plane_archive is not None
        else None
    )
    if plane_archive is None and args.precursor is None:
        raise ValueError("--precursor is required without --plane-archive")
    plane_archive_sha256 = (
        plane_archive.manifest_sha256 if plane_archive is not None else None
    )
    plane_archive_extension_sha256 = (
        plane_archive.extension_manifest_sha256
        if plane_archive is not None
        else None
    )
    if args.resume_allow_plane_archive_extension and (
        args.resume_checkpoint is None or plane_archive_extension_sha256 is None
    ):
        raise ValueError(
            "--resume-allow-plane-archive-extension requires both a resume "
            "checkpoint and a complete archive extension manifest"
        )
    proxy_variables = tuple(
        name
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        if os.environ.get(name)
    )
    if proxy_variables:
        raise RuntimeError(
            "Unset proxy variables before JAX distributed initialization: "
            + ", ".join(proxy_variables)
        )
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=args.local_device_id,
    )

    layout = canonical_scale_layouts()[args.target_turbines]
    direction_center = None
    farm_layout = layout
    if args.target_turbines in (81, 100) and args.relative_wind_angle_deg == 0.0:
        # The large aligned farms do not fit the legacy +/-30 degree envelope.
        # Keep their declared geometry exactly, including boundary distances.
        # The N=9 direction protocol below remains unchanged.
        direction_center = tuple(
            np.asarray(layout.turbine_positions_m, dtype=np.float64)[:, (0, 2)]
            .mean(axis=0).tolist()
        )
    elif args.target_turbines != 3:
        direction_center = common_direction_center(
            layout,
            (
                TRAIN_DIRECTIONS_DEG
                if args.n9_million_direction_schedule
                else (-30.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 30.0)
            ),
        )
        farm_layout = rotate_layout_to_flow_frame(
            layout, args.relative_wind_angle_deg, direction_center
        )
    if args.target_turbines == 3:
        source_case = make_mole_training_case(
            args.local_environments,
            compact_solver=args.compact_solver,
        )
    else:
        source_case = make_scaled_mole_training_case(
            layout,
            args.local_environments,
            compact_solver=args.compact_solver,
        )
    source_precursor = None
    if plane_archive is None:
        assert args.precursor is not None
        source_precursor, source_metadata = load_initial_precursor(
            args.precursor, source_case
        )
    else:
        source_metadata = {
            "npz_sha256": plane_archive.source_state_sha256,
            "completed_steps": plane_archive.source_completed_steps,
            "loaded_state_format": "external_time_plane_archive",
            "boundary_pressure_history_loaded": False,
        }
        if args.precursor is not None:
            source_metadata_path = args.precursor.with_suffix(".meta.json")
            if (
                not args.precursor.is_file()
                or not source_metadata_path.is_file()
            ):
                raise FileNotFoundError(
                    f"missing archive source or metadata: {args.precursor}"
                )
            checked_metadata = json.loads(
                source_metadata_path.read_text(encoding="utf-8")
            )
            source_hash = _file_sha256(args.precursor)
            if source_hash != plane_archive.source_state_sha256:
                raise ValueError(
                    "plane archive source hash does not match --precursor"
                )
            if checked_metadata.get("npz_sha256") != source_hash:
                raise ValueError(
                    "precursor metadata hash does not match source"
                )
            if (
                int(checked_metadata["completed_steps"])
                != plane_archive.source_completed_steps
            ):
                raise ValueError(
                    "plane archive source step does not match precursor metadata"
                )
            source_metadata.update(checked_metadata)
    plane_cursors: list[int] = []
    plane_member_bounds: tuple[tuple[int, int], ...] = ()
    plane_window_required_planes = (
        (args.episode_interactions + args.reset_interactions) * 50
    )
    if plane_archive is not None:
        ensemble_source_sha256 = source_metadata.get(
            "source_state", {}
        ).get("sha256")
        if ensemble_source_sha256 is None:
            ensemble_source_sha256 = source_metadata["npz_sha256"]
        if ensemble_source_sha256 != plane_archive.source_state_sha256:
            raise ValueError(
                "precursor ensemble and plane archive do not share the same "
                "base source state"
            )
        if len(plane_archive.member_offsets) != args.local_environments:
            raise ValueError(
                "plane archive member count does not match local environments"
            )
        plane_member_bounds = archive_member_bounds(
            plane_archive.member_offsets, plane_archive.completed_planes
        )
        if args.episode_random_plane_windows:
            plane_cursors = [
                episode_plane_window_start(
                    start,
                    stop,
                    plane_window_required_planes,
                    schedule_seed=args.plane_window_seed,
                    global_environment=(
                        args.process_id * args.local_environments + index
                    ),
                    episode_index=0,
                )
                for index, (start, stop) in enumerate(plane_member_bounds)
            ]
        else:
            plane_cursors = [
                int(value) + args.plane_offset_add
                for value in plane_archive.member_offsets
            ]
        if min(plane_cursors) < 0:
            raise ValueError("plane archive member offset cannot be negative")
        if max(plane_cursors) >= plane_archive.completed_planes:
            raise ValueError("precursor member offset exceeds plane archive")
    if args.target_turbines == 3:
        case = make_mole_training_case(
            args.microbatch,
            compact_solver=args.compact_solver,
            rhs_assembly=args.rhs_assembly,
        )
    else:
        case = make_scaled_mole_training_case(
            farm_layout,
            args.microbatch,
            compact_solver=args.compact_solver,
            rhs_assembly=args.rhs_assembly,
        )
    if plane_archive is not None:
        expected_plane_shape = (3, case.ny, case.nz)
        if plane_archive.plane_shape != expected_plane_shape:
            raise ValueError(
                "plane archive shape does not match farm case: "
                f"{plane_archive.plane_shape} != {expected_plane_shape}"
            )
        provenance = plane_archive.manifest.get("provenance", {})
        if args.target_turbines != 3 and (
            provenance.get("target_turbines") != args.target_turbines
            or provenance.get("target_layout_sha256") != layout.payload_sha256
        ):
            raise ValueError("plane archive layout provenance does not match case")
    direction_cases: dict[float, Any] = {}
    if args.n9_million_direction_schedule:
        assert direction_center is not None
        direction_cases = {
            float(direction): make_scaled_mole_training_case(
                rotate_layout_to_flow_frame(
                    layout, float(direction), direction_center
                ),
                args.microbatch,
                compact_solver=args.compact_solver,
                rhs_assembly=args.rhs_assembly,
            )
            for direction in TRAIN_DIRECTIONS_DEG
        }
    direction_farm_functions = {
        direction: build_flow_functions(
            direction_case, periodic_x=False, include_turbines=True
        )
        for direction, direction_case in direction_cases.items()
    }
    action_dim = control_action_dim(case, args.action_parameterization)
    graph_positions = (
        layout.turbine_positions_m
        if args.graph_geometry == "canonical"
        else case.turbine_positions_m
    )
    if args.algorithm == "graph_sac_v2":
        wake_graph = build_wake_graph(
            graph_positions,
            diameter_m=case.diameter,
            wind_vector_xz=(1.0, 0.0),
            condition_dim=(2 if args.direction_condition else 0),
        )
    elif args.algorithm == "graph_sac_directional":
        wake_graph = build_directional_graph(
            layout.turbine_positions_m, case.diameter
        )
    else:
        wake_graph = None
    actor_state_shards = args.local_environments // args.microbatch
    global_actor_state_shards = actor_state_shards * args.num_processes
    if args.steering_branch_condition and global_actor_state_shards % 2:
        raise ValueError(
            "steering branches require an even number of global actor shards"
        )
    steering_branches: list[float | None] = []
    observation_suffixes: list[jax.Array | None] = []
    initial_actor_directions = (
        local_directions_for_reset(
            0,
            args.process_id,
            local_environments=args.local_environments,
            schedule_seed=args.n9_direction_schedule_seed,
        )
        if args.n9_million_direction_schedule
        else None
    )
    for shard in range(actor_state_shards):
        branch = (
            steering_branch_for_shard(
                args.process_id, actor_state_shards, shard
            )
            if args.steering_branch_condition
            else None
        )
        steering_branches.append(branch)
        if branch is not None:
            suffix = append_direction_branch_condition(
                jnp.zeros((0,), dtype=jnp.float32),
                args.relative_wind_angle_deg,
                branch,
            )
        elif args.direction_condition:
            suffix = append_direction_condition(
                jnp.zeros((0,), dtype=jnp.float32),
                (
                    float(initial_actor_directions[shard])
                    if initial_actor_directions is not None
                    else args.relative_wind_angle_deg
                ),
            )
        else:
            suffix = None
        observation_suffixes.append(suffix)
    global_steering_branches = (
        [
            steering_branch_for_shard(process, actor_state_shards, shard)
            for process in range(args.num_processes)
            for shard in range(actor_state_shards)
        ]
        if args.steering_branch_condition
        else []
    )
    random_action_transitions_per_shard = (
        args.random_action_transitions
        * args.microbatch
        // global_environments
    )
    config = ContinuousControlConfig(
        inlet_mode=args.inlet_mode,
        tau=(1.0 if args.algorithm == "crossq" else 0.005),
        sample_size=args.sample_size,
        minimum_replay_size=args.minimum_replay_size,
        random_action_transitions=random_action_transitions_per_shard,
        updates_per_transition=args.updates_per_transition,
        initial_alpha=args.initial_alpha,
        entropy_learning_rate=(
            args.crossq_entropy_learning_rate
            if args.algorithm == "crossq"
            else 1.0e-5
        ),
        episode_interactions=args.episode_interactions,
        reset_interactions=args.reset_interactions,
    )
    actor_replay_capacity = args.microbatch * args.collection_steps
    actor_states = []
    actor_cases = []
    actor_precursor_functions = []
    actor_farm_functions = []
    plane_source_states = []
    optimizers = None
    precursor_functions = None
    farm_functions = None
    for shard in range(actor_state_shards):
        start = shard * args.microbatch
        stop = start + args.microbatch
        if plane_archive is None:
            assert source_precursor is not None
            source_slice = slice_initial_precursor(
                source_precursor, start, stop
            )
            initial_inlet = None
        else:
            source_slice = None
            initial_inlet = jnp.asarray(
                plane_archive.read(plane_cursors[shard], 1)
            )
        actor_case = (
            direction_cases[float(initial_actor_directions[shard])]
            if initial_actor_directions is not None
            else case
        )
        (
            actor_state,
            shard_optimizers,
            shard_precursor_functions,
            shard_farm_functions,
        ) = initialize_continuous_training(
            actor_case,
            config,
            source_slice,
            args.seed + 1000 * args.process_id + shard,
            args.hidden_dim,
            args.learning_rate,
            actor_replay_capacity,
            initial_inlet=initial_inlet,
            action_parameterization=args.action_parameterization,
        )
        observation_suffix = observation_suffixes[shard]
        if observation_suffix is not None:
            condition_dim = int(observation_suffix.shape[-1])
            conditioned_observation = jnp.concatenate(
                (
                    actor_state.observation,
                    jnp.broadcast_to(
                        observation_suffix,
                        (*actor_state.observation.shape[:-1], condition_dim),
                    ),
                ),
                axis=-1,
            )
            actor_state = actor_state._replace(
                observation=conditioned_observation,
                replay=init_replay(
                    actor_replay_capacity,
                    int(conditioned_observation.shape[-1]),
                    action_dim,
                ),
            )
        actor_states.append(actor_state)
        actor_cases.append(actor_case)
        actor_precursor_functions.append(shard_precursor_functions)
        actor_farm_functions.append(shard_farm_functions)
        if args.inlet_mode == "time_planes" and plane_archive is None:
            if source_slice is None or not hasattr(
                source_slice, "rhs_previous"
            ):
                raise ValueError(
                    "time_planes requires complete AB3 precursor states"
                )
            plane_source_states.append(source_slice)
        if optimizers is None:
            optimizers = shard_optimizers
            precursor_functions = shard_precursor_functions
            farm_functions = shard_farm_functions
    assert optimizers is not None
    assert precursor_functions is not None
    assert farm_functions is not None
    if plane_archive is not None:
        gc.collect()

    plane_generators: dict[int, Any] = {}
    precursor_generation_seconds_total = 0.0
    precursor_planes_generated_total = 0

    def generate_time_planes(index: int, count: int) -> tuple[jax.Array, float]:
        nonlocal precursor_generation_seconds_total
        nonlocal precursor_planes_generated_total
        if args.inlet_mode != "time_planes":
            raise RuntimeError("plane generation requested outside time_planes")
        started = time.perf_counter()
        if plane_archive is not None:
            before = plane_cursors[index]
            host_planes = plane_archive.read(before, count)
            planes = jnp.asarray(host_planes[None])
            jax.block_until_ready(planes)
            plane_cursors[index] = before + count
        else:
            if count not in plane_generators:
                plane_generators[count] = jax.jit(
                    make_precursor_plane_generator(
                        precursor_functions, count
                    ),
                    donate_argnums=(0,),
                )
            before = int(np.asarray(plane_source_states[index].step_index))
            source, planes = plane_generators[count](
                plane_source_states[index]
            )
            jax.block_until_ready(planes)
            after = int(np.asarray(source.step_index))
            if after - before != count:
                raise RuntimeError(
                    f"precursor source advanced {after - before}, "
                    f"expected {count}"
                )
            plane_source_states[index] = source
        elapsed = time.perf_counter() - started
        precursor_generation_seconds_total += elapsed
        precursor_planes_generated_total += count * case.batch
        return planes, elapsed

    def peek_time_planes(index: int, count: int) -> jax.Array:
        if plane_archive is None:
            raise RuntimeError("SVG plane peek requires a cached archive")
        host_planes = plane_archive.read(plane_cursors[index], count)
        planes = jnp.asarray(host_planes[None])
        jax.block_until_ready(planes)
        return planes

    def peek_time_plane_batch(
        indices: list[int], count: int
    ) -> tuple[dict[int, jax.Array], float]:
        if plane_archive is None:
            raise RuntimeError("SVG plane peek requires a cached archive")
        started = time.perf_counter()
        host_planes = np.stack(
            [
                plane_archive.read(plane_cursors[index], count)
                for index in indices
            ],
            axis=0,
        )
        plane_batch = jnp.asarray(host_planes)
        jax.block_until_ready(plane_batch)
        return (
            {
                index: plane_batch[offset : offset + 1]
                for offset, index in enumerate(indices)
            },
            time.perf_counter() - started,
        )

    resume_payload = None
    resumed_global_interactions = 0
    resume_implementation_override = False
    resume_diagnostic_override = False
    resume_plane_archive_extension_override = False
    if args.resume_checkpoint is not None:
        resume_payload = _load_full_checkpoint(args.resume_checkpoint)
        saved_extension_sha256 = resume_payload.get(
            "plane_archive_extension_sha256"
        )
        if saved_extension_sha256 != plane_archive_extension_sha256:
            if (
                saved_extension_sha256 is None
                and plane_archive_extension_sha256 is not None
                and args.resume_allow_plane_archive_extension
            ):
                resume_plane_archive_extension_override = True
            else:
                raise ValueError(
                    "resume checkpoint plane archive extension does not match"
                )
        saved_configuration = _normalize_saved_configuration(
            resume_payload["configuration"]
        )
        expected_configuration = _full_checkpoint_configuration(
            args,
            source_metadata["npz_sha256"],
            plane_archive_sha256,
            case,
        )
        if resume_payload["process_id"] != args.process_id:
            raise ValueError("resume checkpoint process id does not match")
        if resume_payload["num_processes"] != args.num_processes:
            raise ValueError("resume checkpoint process count does not match")
        implementation_only_change = _configuration_differs_only_in_implementation(
            saved_configuration, expected_configuration
        )
        diagnostic_only_change = _configuration_differs_only_in_diagnostics(
            saved_configuration, expected_configuration
        )
        learner_only_compatible = _configuration_compatible_for_learner_only(
            saved_configuration, expected_configuration
        )
        if saved_configuration != expected_configuration and not (
            args.resume_allow_implementation_change and implementation_only_change
        ) and not (
            args.resume_allow_diagnostic_change and diagnostic_only_change
        ) and not (
            args.resume_learner_only and learner_only_compatible
        ):
            raise ValueError(
                "resume checkpoint configuration does not match current run"
            )
        resume_implementation_override = (
            saved_configuration != expected_configuration
            and implementation_only_change
        )
        resume_diagnostic_override = (
            saved_configuration != expected_configuration
            and diagnostic_only_change
        )
        resumed_global_interactions = int(
            resume_payload["global_interactions"]
        )

    input_horizon = None
    if args.planned_global_interactions is not None:
        if plane_archive is None:
            raise ValueError("declared whole-input budget requires a completed plane archive")
        input_horizon = plane_horizon(
            cursors=(list(map(int, resume_payload["plane_cursors"]))
                     if resume_payload is not None else plane_cursors),
            completed_planes=plane_archive.completed_planes,
            iterations=args.iterations,
            collection_steps=args.collection_steps,
            global_environments=global_environments,
            burnin_interactions=args.burnin_interactions,
            episode_interactions=config.episode_interactions,
            reset_interactions=config.reset_interactions,
            les_steps_per_interaction=config.les_steps_per_interaction,
            resumed_global_interactions=(resumed_global_interactions
                                         if resume_payload is not None else None),
            warmup_cursors=plane_cursors,
            planned_global_interactions=args.planned_global_interactions,
        )
        if args.process_id == 0:
            print(json.dumps({"phase": "whole_input_horizon", **input_horizon}), flush=True)

    resume_n_step_pending = None
    if resume_payload is not None and args.formal_n_step > 1:
        if "n_step_pending" not in resume_payload:
            raise ValueError(
                "n-step resume checkpoint does not contain pending raw tails"
            )
        resume_n_step_pending = _local_n_step_pending(
            resume_payload["n_step_pending"],
            process_id=args.process_id,
            local_environments=args.local_environments,
            global_environments=global_environments,
        )
    n_step_pending = (
        None
        if resume_n_step_pending is None
        else {
            name: np.array(value, copy=True)
            for name, value in resume_n_step_pending.items()
        }
    )

    burnin_seconds = 0.0
    if args.burnin_interactions > 0 and (
        resume_payload is None or args.resume_learner_only
    ):
        evaluation_step = make_evaluation_step(
            case,
            config,
            precursor_functions,
            farm_functions,
            "zero",
        )

        def burnin_scan(state, planes=None, plane_start=None):
            def body(current, _):
                next_state, metrics = evaluation_step(
                    current, planes, plane_start
                )
                return next_state, metrics["farm_power_mw"]

            return jax.lax.scan(
                body,
                state,
                xs=None,
                length=args.burnin_interactions,
            )

        run_burnin = jax.jit(burnin_scan, donate_argnums=(0,))
        burnin_started = time.perf_counter()
        for index, state in enumerate(actor_states):
            observation_suffix = observation_suffixes[index]
            evaluation = initialize_evaluation_from_training(state)
            if observation_suffix is not None:
                condition_dim = int(observation_suffix.shape[-1])
                evaluation = evaluation._replace(
                    observation=evaluation.observation[..., :-condition_dim]
                )
            if args.inlet_mode == "time_planes":
                plane_count = (
                    args.burnin_interactions * config.les_steps_per_interaction
                )
                planes, _ = generate_time_planes(index, plane_count)
                plane_start = jnp.asarray(
                    int(np.asarray(evaluation.farm_flow.step_index)),
                    dtype=jnp.int32,
                )
                evaluation, burnin_power = run_burnin(
                    evaluation, planes, plane_start
                )
            else:
                evaluation, burnin_power = run_burnin(evaluation)
            jax.block_until_ready(burnin_power)
            reset_observation = jnp.zeros_like(evaluation.observation)
            if observation_suffix is not None:
                reset_observation = jnp.concatenate(
                    (
                        reset_observation,
                        jnp.broadcast_to(
                            observation_suffix,
                            (*reset_observation.shape[:-1], condition_dim),
                        ),
                    ),
                    axis=-1,
                )
            actor_states[index] = state._replace(
                precursor_flow=evaluation.precursor_flow,
                farm_flow=evaluation.farm_flow,
                observation_frames=jnp.zeros_like(
                    evaluation.observation_frames
                ),
                observation=reset_observation,
                previous_yaw=jnp.zeros_like(evaluation.previous_yaw),
            )
        burnin_seconds = time.perf_counter() - burnin_started

    observation_dim = int(actor_states[0].observation.shape[-1])
    if args.algorithm == "crossq":
        learner_sac, learner_optimizers = init_crossq(
            jax.random.PRNGKey(args.seed + 9000),
            observation_dim,
            action_dim,
            args.hidden_dim,
            args.crossq_critic_hidden_dim,
            args.crossq_learning_rate,
            initial_alpha=config.initial_alpha,
            entropy_learning_rate=args.crossq_entropy_learning_rate,
            adam_b1=args.crossq_adam_b1,
        )
    elif args.algorithm == "graph_sac_v2":
        assert wake_graph is not None
        learner_sac, learner_optimizers = init_graph_v2_sac(
            jax.random.PRNGKey(args.seed + 9000),
            observation_dim,
            wake_graph,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            frame_stack=config.frame_stack,
            conservative=args.conservative_actor_init,
            neutral_initial_scale=args.neutral_actor_initial_scale,
            initial_alpha=config.initial_alpha,
            entropy_learning_rate=config.entropy_learning_rate,
            actor_learning_rate_scale=(
                args.actor_learning_rate_scale
                * args.diagnostic_actor_learning_rate_scale
            ),
        )
    elif args.algorithm == "graph_sac_directional":
        assert isinstance(wake_graph, DirectionalWakeGraph)
        learner_sac, learner_optimizers = init_directional_graph_sac(
            jax.random.PRNGKey(args.seed + 9000),
            observation_dim,
            wake_graph,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            frame_stack=config.frame_stack,
            conservative=args.conservative_actor_init,
            neutral_initial_scale=args.neutral_actor_initial_scale,
            initial_alpha=config.initial_alpha,
            entropy_learning_rate=config.entropy_learning_rate,
            actor_learning_rate_scale=(
                args.actor_learning_rate_scale
                * args.diagnostic_actor_learning_rate_scale
            ),
        )
    else:
        learner_sac, learner_optimizers = init_sac(
            jax.random.PRNGKey(args.seed + 9000),
            observation_dim,
            action_dim,
            args.hidden_dim,
            args.learning_rate,
            initial_alpha=config.initial_alpha,
            policy_mode=config.policy_mode,
            entropy_learning_rate=config.entropy_learning_rate,
            actor_learning_rate_scale=(
                args.actor_learning_rate_scale
                * args.diagnostic_actor_learning_rate_scale
            ),
        )
    initial_actor_checkpoint_sha256 = None
    initial_actor_metadata = None
    if args.initial_actor_checkpoint is not None:
        initial_actor_checkpoint_sha256 = _file_sha256(
            args.initial_actor_checkpoint
        )
        initial_actor, initial_actor_metadata = load_actor_checkpoint(
            args.initial_actor_checkpoint
        )
        required_metadata = {
            "policy_architecture": "graph_sac_directional",
            "target_turbines": args.target_turbines,
            "action_parameterization": args.action_parameterization,
            "direction_condition": True,
            "steering_branch_condition": True,
            "reflection_equivariant_actor": True,
            "graph_geometry": "canonical",
        }
        if any(
            initial_actor_metadata.get(key) != value
            for key, value in required_metadata.items()
        ):
            raise ValueError("initial Actor checkpoint metadata is incompatible")
        expected_leaves = jax.tree.leaves(learner_sac.actor)
        initial_leaves = jax.tree.leaves(initial_actor)
        if (
            jax.tree.structure(initial_actor)
            != jax.tree.structure(learner_sac.actor)
            or [value.shape for value in initial_leaves]
            != [value.shape for value in expected_leaves]
        ):
            raise ValueError("initial Actor parameter tree is incompatible")
        learner_sac = learner_sac._replace(
            actor=initial_actor,
            actor_opt=learner_optimizers[0].init(initial_actor),
        )
    if args.conservative_actor_init and args.algorithm == "sac":
        conservative_actor = _conservative_flat_actor(learner_sac.actor)
        learner_sac = learner_sac._replace(
            actor=conservative_actor,
            actor_opt=learner_optimizers[0].init(conservative_actor),
        )
        actor_states = [
            _copy_actor_into_state(state, conservative_actor)
            for state in actor_states
        ]
    if args.algorithm in graph_algorithms:
        actor_states = [
            _copy_actor_into_state(state, learner_sac.actor)
            for state in actor_states
        ]
    learner = LearnerState(
        replay=(
            init_auditable_replay(
                args.replay_capacity,
                observation_dim,
                action_dim,
                args.target_turbines,
            )
            if args.n9_million_direction_schedule
            else init_replay(
                args.replay_capacity, observation_dim, action_dim
            )
        ),
        sac=learner_sac,
        key=jax.random.PRNGKey(args.seed + 10000 + args.process_id),
        replay_checksum=jnp.asarray(0.0, dtype=jnp.float32),
    )
    if wake_graph is None:
        graph_actor_sample_fn = None
    elif args.algorithm == "graph_sac_directional":
        assert isinstance(wake_graph, DirectionalWakeGraph)

        def graph_actor_sample_fn(
            actor: Any,
            observations: jax.Array,
            key: jax.Array,
            action_bound: float,
            _policy_mode: str,
        ) -> tuple[jax.Array, jax.Array]:
            actor_sample = (
                equivariant_directional_graph_actor_sample
                if effective_reflection_equivariant_actor
                else directional_graph_actor_sample
            )
            return actor_sample(
                actor,
                observations,
                key,
                action_bound,
                wake_graph,
                config.frame_stack,
            )
    else:
        def graph_actor_sample_fn(
            actor: Any,
            observations: jax.Array,
            key: jax.Array,
            action_bound: float,
            _policy_mode: str,
        ) -> tuple[jax.Array, jax.Array]:
            return graph_v2_actor_sample(
                actor,
                observations,
                key,
                action_bound,
                wake_graph,
                config.frame_stack,
            )

    actor_collectors = []
    collector_cache: dict[Any, Any] = {}
    reset_cache: dict[float, Any] = {}
    if not args.n9_million_direction_schedule:
        for observation_suffix in observation_suffixes:
            suffix_key = (
                None
                if observation_suffix is None
                else tuple(float(value) for value in np.asarray(observation_suffix))
            )
            if suffix_key not in collector_cache:
                collector_cache[suffix_key] = jax.jit(
                    make_continuous_training_iteration(
                        case,
                        config,
                        optimizers,
                        precursor_functions,
                        farm_functions,
                        updates_per_iteration=0,
                        collection_steps_per_iteration=args.collection_steps,
                        action_parameterization=args.action_parameterization,
                        actor_sample_fn=graph_actor_sample_fn,
                        observation_suffix=observation_suffix,
                    ),
                    donate_argnums=(0,),
                )
            actor_collectors.append(collector_cache[suffix_key])

    def formal_direction_suffix(direction: float) -> jax.Array:
        return append_direction_condition(
            jnp.zeros((0,), dtype=jnp.float32), direction
        )

    def formal_actor_collector(direction: float, steps: int):
        key = (float(direction), int(steps))
        if key not in collector_cache:
            direction_case = direction_cases[float(direction)]
            collector_cache[key] = jax.jit(
                make_continuous_training_iteration(
                    direction_case,
                    config,
                    optimizers,
                    actor_precursor_functions[0],
                    direction_farm_functions[float(direction)],
                    updates_per_iteration=0,
                    collection_steps_per_iteration=steps,
                    action_parameterization=args.action_parameterization,
                    actor_sample_fn=graph_actor_sample_fn,
                    observation_suffix=formal_direction_suffix(direction),
                    return_raw_transitions=True,
                    reset_episode_in_collector=False,
                ),
                donate_argnums=(0,),
            )
        return collector_cache[key]

    def formal_episode_reset(direction: float):
        key = float(direction)
        if key not in reset_cache:
            reset_cache[key] = jax.jit(
                make_continuous_episode_reset(
                    direction_cases[key],
                    config,
                    actor_precursor_functions[0],
                    direction_farm_functions[key],
                    observation_suffix=formal_direction_suffix(key),
                ),
                donate_argnums=(0,),
            )
        return reset_cache[key]
    exact_svg_probe = (
        jax.jit(
            make_exact_svg_probe(
                case,
                config,
                precursor_functions,
                farm_functions,
                action_parameterization=args.action_parameterization,
            )
        )
        if svg_enabled
        else None
    )

    def collect_actor_state(index: int, state: Any):
        if args.n9_million_direction_schedule:
            start_interaction = int(np.asarray(state.transitions)) // case.batch
            episode_position = start_interaction % config.episode_interactions
            reset_index = start_interaction // config.episode_interactions
            segments = split_collection_segments(
                episode_position,
                args.collection_steps,
                episode_interactions=config.episode_interactions,
                reset_index=reset_index,
                reset_before_start=(
                    start_interaction > 0 and episode_position == 0
                ),
            )
            generated_seconds_total = 0.0
            generated_planes_total = 0
            farm_seconds_total = 0.0
            segment_metrics = []
            for segment in segments:
                direction = float(
                    local_directions_for_reset(
                        segment.reset_index,
                        args.process_id,
                        local_environments=args.local_environments,
                        schedule_seed=args.n9_direction_schedule_seed,
                    )[index]
                )
                if segment.reset_before:
                    if args.episode_random_plane_windows:
                        member_start, member_stop = plane_member_bounds[index]
                        plane_cursors[index] = episode_plane_window_start(
                            member_start,
                            member_stop,
                            plane_window_required_planes,
                            schedule_seed=args.plane_window_seed,
                            global_environment=(
                                args.process_id * args.local_environments + index
                            ),
                            episode_index=segment.reset_index,
                        )
                    reset_planes_count = (
                        config.reset_interactions
                        * config.les_steps_per_interaction
                    )
                    reset_planes, generated_seconds = generate_time_planes(
                        index, reset_planes_count
                    )
                    reset_start = jnp.asarray(
                        int(np.asarray(state.farm_flow.step_index)),
                        dtype=jnp.int32,
                    )
                    reset_started = time.perf_counter()
                    state = formal_episode_reset(direction)(
                        state, reset_planes, reset_start
                    )
                    jax.block_until_ready(state.farm_flow.velocity)
                    farm_seconds_total += time.perf_counter() - reset_started
                    generated_seconds_total += generated_seconds
                    generated_planes_total += reset_planes_count

                plane_count = (
                    segment.steps * config.les_steps_per_interaction
                )
                planes, generated_seconds = generate_time_planes(
                    index, plane_count
                )
                plane_start = jnp.asarray(
                    int(np.asarray(state.farm_flow.step_index)),
                    dtype=jnp.int32,
                )
                farm_started = time.perf_counter()
                state, metrics = formal_actor_collector(
                    direction, segment.steps
                )(state, planes, plane_start)
                jax.block_until_ready(metrics["mean_farm_power_mw"])
                farm_seconds_total += time.perf_counter() - farm_started
                generated_seconds_total += generated_seconds
                generated_planes_total += plane_count
                raw = dict(metrics["raw_transitions"])
                local_sequence = raw["sequence_id"].astype(jnp.int32)
                global_environment = np.int32(
                    args.process_id * args.local_environments + index
                )
                raw["direction_deg"] = jnp.full(
                    local_sequence.shape,
                    np.float32(direction),
                    dtype=jnp.float32,
                )
                raw["episode_id"] = (
                    local_sequence // np.int32(config.episode_interactions)
                ) * np.int32(GLOBAL_ENVIRONMENTS) + global_environment
                raw["sequence_id"] = (
                    local_sequence * np.int32(GLOBAL_ENVIRONMENTS)
                    + global_environment
                )
                metrics = dict(metrics)
                metrics["raw_transitions"] = raw
                segment_metrics.append((segment.steps, direction, metrics))

            total_steps = float(
                sum(steps for steps, _, _ in segment_metrics)
            )
            summary_names = (
                "mean_farm_power_mw",
                "mean_reward",
                "mean_angle_penalty",
                "mean_abs_yaw_deg",
                "mean_abs_yaw_speed_deg_per_s",
                "inlet_interval_delta_rms",
            )
            merged_metrics = {
                name: sum(
                    np.float32(steps) * metrics[name]
                    for steps, _, metrics in segment_metrics
                )
                / np.float32(total_steps)
                for name in summary_names
            }
            merged_metrics["episode_reset"] = jnp.max(
                jnp.stack(
                    [
                        metrics["episode_reset"]
                        for _, _, metrics in segment_metrics
                    ]
                )
            )
            merged_metrics["episode_reset_count"] = sum(
                np.float32(steps) * metrics["episode_reset"]
                for steps, _, metrics in segment_metrics
            )
            merged_metrics["raw_transitions"] = jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0),
                *[
                    metrics["raw_transitions"]
                    for _, _, metrics in segment_metrics
                ],
            )
            return (
                state,
                merged_metrics,
                generated_seconds_total,
                farm_seconds_total,
                generated_planes_total,
            )

        actor_collect = actor_collectors[index]
        generated_seconds = 0.0
        generated_planes = 0
        farm_started = time.perf_counter()
        if args.inlet_mode == "time_planes":
            generated_planes = _required_plane_count(
                state, case, config, args.collection_steps
            )
            planes, generated_seconds = generate_time_planes(
                index, generated_planes
            )
            plane_start = jnp.asarray(
                int(np.asarray(state.farm_flow.step_index)),
                dtype=jnp.int32,
            )
            farm_started = time.perf_counter()
            state, metrics = actor_collect(state, planes, plane_start)
        else:
            state, metrics = actor_collect(state)
        jax.block_until_ready(metrics["mean_farm_power_mw"])
        farm_seconds = time.perf_counter() - farm_started
        return (
            state,
            metrics,
            generated_seconds,
            farm_seconds,
            generated_planes,
        )

    def collect_svg_sample(
        index: int,
        state: Any,
        sac: SACState,
        planes: jax.Array | None = None,
    ):
        if exact_svg_probe is None:
            raise RuntimeError("exact SVG probe is disabled")
        if planes is None:
            planes = peek_time_planes(index, config.les_steps_per_interaction)
        plane_start = jnp.asarray(
            int(np.asarray(state.farm_flow.step_index)), dtype=jnp.int32
        )
        started = time.perf_counter()
        objective, _, action_gradient = exact_svg_probe(
            state.farm_flow,
            state.observation_frames,
            state.observation,
            state.previous_yaw,
            sac,
            planes,
            plane_start,
        )
        jax.block_until_ready(action_gradient)
        return (
            SVGGradientBatch(
                observations=jnp.copy(state.observation),
                action_gradients=action_gradient,
                objectives=jnp.asarray([objective], dtype=jnp.float32),
            ),
            time.perf_counter() - started,
        )
    global_transitions_per_iteration = (
        args.local_environments
        * args.num_processes
        * args.collection_steps
    )
    updates_per_iteration = (
        args.diagnostic_updates_per_iteration
        if args.diagnostic_updates_per_iteration >= 0
        else global_transitions_per_iteration * args.updates_per_transition
    )
    learner_step = jax.pmap(
        _make_learner_step(
            learner_optimizers,
            config,
            updates_per_iteration,
            args.replay_mode,
            action_dim,
            args.algorithm,
            args.crossq_brn_momentum,
            args.crossq_brn_warmup_updates,
            args.crossq_policy_delay,
            args.crossq_critic_gradient_clip_norm,
            args.diagnostic_reward_scale,
            effective_actor_update_period,
            (
                effective_graph_action_l2_coefficient
                if args.algorithm in graph_algorithms
                else args.diagnostic_action_l2_coefficient
            ),
            (
                args.diagnostic_entropy_scale
                if args.diagnostic_entropy_scale != 1.0
                else args.entropy_scale
            ),
            wake_graph,
            args.graph_actor_warmup_updates,
            args.graph_actor_minimum_replay_size,
            effective_reflection_equivariant_actor,
        ),
        axis_name="hosts",
        in_axes=(0, 0),
        out_axes=(0, 0),
        donate_argnums=(0,),
    )
    learner_collect_step = (
        jax.pmap(
            _make_learner_step(
                learner_optimizers,
                config,
                0,
                args.replay_mode,
                action_dim,
                args.algorithm,
                args.crossq_brn_momentum,
                args.crossq_brn_warmup_updates,
                args.crossq_policy_delay,
                args.crossq_critic_gradient_clip_norm,
                args.diagnostic_reward_scale,
                effective_actor_update_period,
                effective_graph_action_l2_coefficient,
                args.entropy_scale,
                wake_graph,
                args.graph_actor_warmup_updates,
                args.graph_actor_minimum_replay_size,
                effective_reflection_equivariant_actor,
            ),
            axis_name="hosts",
            in_axes=(0, 0),
            out_axes=(0, 0),
            donate_argnums=(0,),
        )
        if args.n9_million_direction_schedule
        else learner_step
    )

    def learner_step_for_current_replay(sharded_state: LearnerState):
        if not args.n9_million_direction_schedule:
            return learner_step
        replay_size = int(np.asarray(sharded_state.replay.size)[0])
        projected_size = min(
            replay_size + global_transitions_per_iteration,
            args.replay_capacity,
        )
        return (
            learner_collect_step
            if projected_size <= config.minimum_replay_size
            else learner_step
        )
    svg_actor_step = (
        jax.pmap(
            _make_svg_actor_step(
                learner_optimizers[0],
                config.policy_mode,
                args.svg_actor_updates_per_iteration,
                args.svg_gradient_clip_norm,
                args.svg_loss_coefficient,
            ),
            axis_name="hosts",
            in_axes=(0, 0),
            out_axes=(0, 0),
            donate_argnums=(0,),
        )
        if svg_enabled
        else None
    )
    sharded_learner = jax.tree.map(lambda value: value[None], learner)
    fresh_alpha_opt = jax.tree.map(
        jnp.copy, sharded_learner.sac.alpha_opt
    )
    del learner
    gc.collect()

    initial_checkpoint_records: dict[str, Any] = {}
    if (
        (args.n9_million_direction_schedule or args.save_initial_checkpoint)
        and resume_payload is None
        and args.checkpoint is not None
        and args.full_checkpoint is not None
    ):
        initial_paths = _checkpoint_milestone_paths(
            args.checkpoint, args.full_checkpoint, 0, args.process_id
        )
        initial_full_path = initial_paths["full_state_process_local"]
        initial_full_sha256 = _save_full_checkpoint_atomic(
            initial_full_path,
            _full_checkpoint_payload(
                args=args,
                source_precursor_sha256=source_metadata["npz_sha256"],
                plane_archive_sha256=plane_archive_sha256,
                plane_archive_extension_sha256=(
                    plane_archive_extension_sha256
                ),
                case=case,
                actor_states=actor_states,
                plane_source_states=plane_source_states,
                plane_cursors=plane_cursors,
                sharded_learner=sharded_learner,
                n_step_pending=n_step_pending,
                history=[],
                global_interactions=0,
                completed_collector_iterations=0,
                algorithm_extension={
                    "name": "fresh_precompile_full_state",
                    "training_algorithm": args.algorithm,
                    "formal_reward_name": args.formal_reward_name,
                    "formal_n_step": args.formal_n_step,
                },
            ),
        )
        initial_checkpoint_records.update(
            {
                "full_state_process_local": str(initial_full_path),
                "full_state_sha256": initial_full_sha256,
            }
        )
        if args.process_id == 0:
            initial_actor_path = initial_paths["actor_process0_only"]
            initial_sac = jax.tree.map(
                lambda value: value[0], sharded_learner.sac
            )
            initial_actor_state = actor_states[0]._replace(sac=initial_sac)
            save_actor_checkpoint(
                initial_actor_path,
                initial_actor_state,
                {
                    "generated_at": datetime.now(timezone.utc)
                    .astimezone()
                    .isoformat(),
                    "purpose": "mole_microbatch_distributed_sac_milestone",
                    "training_algorithm": args.algorithm,
                    "policy_architecture": (
                        "graph_sac_directional" if args.algorithm == "graph_sac_directional"
                        else "graph_sac_v2" if args.algorithm == "graph_sac_v2"
                        else "flat_mlp"
                    ),
                    "relative_wind_angle_deg": args.relative_wind_angle_deg,
                    "action_dim": action_dim,
                    "direction_condition": args.direction_condition,
                    "steering_branch_condition": (
                        args.steering_branch_condition
                    ),
                    "initial_actor_checkpoint_sha256": None,
                    "graph_geometry": args.graph_geometry,
                    "target_turbines": args.target_turbines,
                    "action_parameterization": args.action_parameterization,
                    "seed": args.seed,
                    "requested_global_interactions": 0,
                    "actual_global_interactions": 0,
                    "global_batch": global_environments,
                    "inlet_mode": config.inlet_mode,
                    "episode_interactions": config.episode_interactions,
                    "reset_interactions": config.reset_interactions,
                    "source_precursor_sha256": source_metadata["npz_sha256"],
                    "plane_archive_sha256": plane_archive_sha256,
                    "plane_archive_extension_sha256": (
                        plane_archive_extension_sha256
                    ),
                },
                case=case,
            )
            initial_checkpoint_records["actor_process0_only"] = str(
                initial_actor_path
            )
        jax.experimental.multihost_utils.sync_global_devices(
            "microbatch-initial-checkpoint-written"
        )

    precursor_seconds_before_compile = precursor_generation_seconds_total
    precursor_planes_before_compile = precursor_planes_generated_total
    warm_actor = _unreplicated_tree_to_local_device(
        sharded_learner.sac.actor
    )
    compile_actor_started = time.perf_counter()
    compile_actor_metrics: list[dict[str, jax.Array]] = []
    for index, state in enumerate(actor_states):
        state = _copy_actor_into_state(state, warm_actor)
        (
            actor_states[index],
            actor_metrics,
            _,
            _,
            _,
        ) = collect_actor_state(index, state)
        compile_actor_metrics.append(actor_metrics)
    del warm_actor
    compile_and_warm_actor_seconds = (
        time.perf_counter() - compile_actor_started
    )
    transitions = (
        _auditable_transition_batch(
            compile_actor_metrics,
            gamma=config.gamma,
            reward_name=args.formal_reward_name,
        )
        if args.n9_million_direction_schedule
        else _transition_batch(actor_states)
    )
    if args.n9_million_direction_schedule and args.formal_n_step > 1:
        assert isinstance(transitions, AuditableTransitionBatch)
        transitions, n_step_pending, _ = _advance_auditable_transition_batch(
            n_step_pending,
            transitions,
            reward_name=args.formal_reward_name,
            n_step=args.formal_n_step,
            gamma=config.gamma,
            global_environments=global_environments,
        )
    sharded_transitions = jax.tree.map(
        lambda value: value[None], transitions
    )
    compile_learner_started = time.perf_counter()
    compile_learner_step = learner_step_for_current_replay(sharded_learner)
    sharded_learner, learner_metrics = compile_learner_step(
        sharded_learner, sharded_transitions
    )
    jax.block_until_ready(learner_metrics["critic_loss"])
    compile_and_first_learner_seconds = (
        time.perf_counter() - compile_learner_started
    )
    compile_and_first_svg_seconds = 0.0
    if svg_enabled:
        assert svg_actor_step is not None
        preserved_learner = jax.tree.map(jnp.copy, sharded_learner)
        compile_svg_started = time.perf_counter()
        compile_sac = jax.tree.map(
            lambda value: value[0], sharded_learner.sac
        )
        compile_sample, _ = collect_svg_sample(
            0, actor_states[0], compile_sac
        )
        compile_batch = jax.tree.map(
            lambda value: jnp.repeat(
                value, args.svg_gradient_samples_per_host, axis=0
            ),
            compile_sample,
        )
        sharded_compile_batch = jax.tree.map(
            lambda value: value[None], compile_batch
        )
        sharded_learner, compile_svg_metrics = svg_actor_step(
            sharded_learner, sharded_compile_batch
        )
        jax.block_until_ready(compile_svg_metrics["svg_actor_loss"])
        compile_and_first_svg_seconds = (
            time.perf_counter() - compile_svg_started
        )
        sharded_learner = preserved_learner
        del preserved_learner
    discarded_resume_compile_precursor_seconds = 0.0
    discarded_resume_compile_precursor_planes = 0
    if resume_payload is not None:
        discarded_resume_compile_precursor_seconds = (
            precursor_generation_seconds_total
            - precursor_seconds_before_compile
        )
        discarded_resume_compile_precursor_planes = (
            precursor_planes_generated_total - precursor_planes_before_compile
        )
        precursor_generation_seconds_total = precursor_seconds_before_compile
        precursor_planes_generated_total = precursor_planes_before_compile
        sharded_learner = _checkpoint_to_sharded_learner(
            resume_payload["learner"]
        )
        if not args.resume_learner_only:
            actor_states = list(_tree_to_device(resume_payload["actor_states"]))
            plane_source_states = list(
                _tree_to_device(resume_payload.get("plane_source_states", ()))
            )
            plane_cursors = [
                int(value)
                for value in resume_payload.get("plane_cursors", plane_cursors)
            ]
            if plane_archive is not None and len(plane_cursors) != len(actor_states):
                raise ValueError("resume checkpoint plane cursor count mismatch")
        if args.diagnostic_resume_alpha is not None:
            resumed_sac = sharded_learner.sac._replace(
                log_alpha=jnp.full_like(
                    sharded_learner.sac.log_alpha,
                    np.float32(math.log(args.diagnostic_resume_alpha)),
                ),
                alpha_opt=fresh_alpha_opt,
            )
            sharded_learner = sharded_learner._replace(sac=resumed_sac)
        n_step_pending = (
            None
            if resume_n_step_pending is None
            else {
                name: np.array(value, copy=True)
                for name, value in resume_n_step_pending.items()
            }
        )
        gc.collect()
    training_interactions_before_loop = resumed_global_interactions + (
        0 if resume_payload is not None else global_transitions_per_iteration
    )

    history = []
    saved_milestones: list[dict[str, Any]] = []
    pending_milestones = list(checkpoint_milestones)
    terminated_early_nonfinite = False
    recoverable_pause_reason = None
    rolling_checkpoint_records = []
    last_full_saved_interactions = resumed_global_interactions
    rolling_sequence = 0
    for iteration in range(args.iterations):
        profile_active = (
            args.profile_dir is not None
            and args.profile_iteration == iteration + 1
            and args.profile_process_id == args.process_id
        )
        if profile_active:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(
                str(args.profile_dir), create_perfetto_link=False
            )
        started = time.perf_counter()
        iteration_precursor_seconds = 0.0
        iteration_farm_seconds = 0.0
        iteration_precursor_planes = 0
        iteration_svg_gradient_seconds = 0.0
        iteration_svg_plane_seconds = 0.0
        iteration_svg_actor_seconds = 0.0
        iteration_environment_farm_seconds: list[float] = []
        svg_samples: list[SVGGradientBatch] = []
        actor_collection_metrics: list[dict[str, jax.Array]] = []
        learner_sac_for_probe = (
            jax.tree.map(lambda value: value[0], sharded_learner.sac)
            if svg_enabled
            else None
        )
        actor_localization_started = time.perf_counter()
        actor = _unreplicated_tree_to_local_device(
            sharded_learner.sac.actor
        )
        actor_localization_seconds = (
            time.perf_counter() - actor_localization_started
        )
        selected_svg_indices = (
            {
                (
                    iteration * args.svg_gradient_samples_per_host + offset
                )
                % args.local_environments
                for offset in range(args.svg_gradient_samples_per_host)
            }
            if svg_enabled
            else set()
        )
        selected_svg_indices_ordered = sorted(selected_svg_indices)
        svg_planes: dict[int, jax.Array] = {}
        if svg_enabled:
            svg_planes, iteration_svg_plane_seconds = peek_time_plane_batch(
                selected_svg_indices_ordered,
                config.les_steps_per_interaction,
            )
        for index, state in enumerate(actor_states):
            actor_states[index] = _copy_actor_into_state(state, actor)
            if index in selected_svg_indices:
                assert learner_sac_for_probe is not None
                svg_sample, svg_seconds = collect_svg_sample(
                    index,
                    actor_states[index],
                    learner_sac_for_probe,
                    svg_planes[index],
                )
                svg_samples.append(svg_sample)
                iteration_svg_gradient_seconds += svg_seconds
            (
                actor_states[index],
                actor_metrics,
                generated_seconds,
                farm_seconds,
                generated_planes,
            ) = collect_actor_state(
                index, actor_states[index]
            )
            actor_collection_metrics.append(actor_metrics)
            iteration_precursor_seconds += generated_seconds
            iteration_farm_seconds += farm_seconds
            iteration_environment_farm_seconds.append(farm_seconds)
            iteration_precursor_planes += generated_planes
        collector_seconds = (
            iteration_precursor_seconds + iteration_farm_seconds
        )
        learner_started = time.perf_counter()
        transitions = (
            _auditable_transition_batch(
                actor_collection_metrics,
                gamma=config.gamma,
                reward_name=args.formal_reward_name,
            )
            if args.n9_million_direction_schedule
            else _transition_batch(actor_states)
        )
        n_step_metadata = None
        if args.n9_million_direction_schedule and args.formal_n_step > 1:
            assert isinstance(transitions, AuditableTransitionBatch)
            transitions, n_step_pending, n_step_metadata = (
                _advance_auditable_transition_batch(
                    n_step_pending,
                    transitions,
                    reward_name=args.formal_reward_name,
                    n_step=args.formal_n_step,
                    gamma=config.gamma,
                    global_environments=global_environments,
                )
            )
        sharded_transitions = jax.tree.map(
            lambda value: value[None], transitions
        )
        active_learner_step = learner_step_for_current_replay(
            sharded_learner
        )
        sharded_learner, learner_metrics = active_learner_step(
            sharded_learner, sharded_transitions
        )
        jax.block_until_ready(learner_metrics["critic_loss"])
        learner_seconds = time.perf_counter() - learner_started
        svg_metrics = {
            "svg_actor_loss": jnp.asarray([0.0], dtype=jnp.float32),
            "svg_objective_mean": jnp.asarray([0.0], dtype=jnp.float32),
            "svg_gradient_norm_mean": jnp.asarray([0.0], dtype=jnp.float32),
            "svg_gradient_norm_max": jnp.asarray([0.0], dtype=jnp.float32),
            "svg_gradient_finite": jnp.asarray([True]),
            "svg_actor_replica_checksum_delta": jnp.asarray(
                [0.0], dtype=jnp.float32
            ),
        }
        if svg_enabled:
            assert svg_actor_step is not None
            if len(svg_samples) != args.svg_gradient_samples_per_host:
                raise RuntimeError("incorrect local SVG sample count")
            svg_batch = jax.tree.map(
                lambda *values: jnp.concatenate(values, axis=0),
                *svg_samples,
            )
            sharded_svg_batch = jax.tree.map(
                lambda value: value[None], svg_batch
            )
            svg_actor_started = time.perf_counter()
            sharded_learner, svg_metrics = svg_actor_step(
                sharded_learner, sharded_svg_batch
            )
            jax.block_until_ready(svg_metrics["svg_actor_loss"])
            iteration_svg_actor_seconds = (
                time.perf_counter() - svg_actor_started
            )
        elapsed = time.perf_counter() - started
        if profile_active:
            jax.profiler.stop_trace()
        collection_metric_means = {
            key: float(
                np.mean(
                    [
                        np.asarray(metrics[key], dtype=np.float64)
                        for metrics in actor_collection_metrics
                    ]
                )
            )
            for key in (
                "mean_farm_power_mw",
                "mean_reward",
                "mean_angle_penalty",
                "mean_abs_yaw_deg",
                "mean_abs_yaw_speed_deg_per_s",
            )
        }
        row = {
            "iteration": iteration + 1,
            "wallclock_seconds": elapsed,
            "collector_seconds": collector_seconds,
            "precursor_generation_seconds": iteration_precursor_seconds,
            "farm_collection_seconds": iteration_farm_seconds,
            "actor_localization_seconds": actor_localization_seconds,
            "environment_farm_seconds": iteration_environment_farm_seconds,
            "environment_farm_seconds_median": float(
                np.median(iteration_environment_farm_seconds)
            ),
            "environment_farm_seconds_max": float(
                np.max(iteration_environment_farm_seconds)
            ),
            "precursor_planes_generated": iteration_precursor_planes,
            "learner_seconds": learner_seconds,
            **collection_metric_means,
            "episode_resets_per_environment": int(
                round(
                    float(
                        np.asarray(
                            actor_metrics.get(
                                "episode_reset_count",
                                actor_metrics["episode_reset"]
                                * args.collection_steps,
                            )
                        )
                    )
                )
            ),
            "physical_reset_interactions_per_environment": int(
                round(
                    float(
                        np.asarray(
                            actor_metrics.get(
                                "episode_reset_count",
                                actor_metrics["episode_reset"]
                                * args.collection_steps,
                            )
                        )
                    )
                )
            )
            * config.reset_interactions,
            "svg_gradient_seconds": iteration_svg_gradient_seconds,
            "svg_plane_seconds": iteration_svg_plane_seconds,
            "svg_actor_seconds": iteration_svg_actor_seconds,
            "svg_actor_loss": float(
                np.asarray(svg_metrics["svg_actor_loss"])[0]
            ),
            "svg_objective_mean": float(
                np.asarray(svg_metrics["svg_objective_mean"])[0]
            ),
            "svg_gradient_norm_mean": float(
                np.asarray(svg_metrics["svg_gradient_norm_mean"])[0]
            ),
            "svg_gradient_norm_max": float(
                np.asarray(svg_metrics["svg_gradient_norm_max"])[0]
            ),
            "svg_gradient_finite": bool(
                np.asarray(svg_metrics["svg_gradient_finite"])[0]
            ),
            "svg_actor_replica_checksum_delta": float(
                np.asarray(
                    svg_metrics["svg_actor_replica_checksum_delta"]
                )[0]
            ),
            "global_interactions_per_second": (
                global_transitions_per_iteration / elapsed
            ),
            "critic_loss": float(
                np.asarray(learner_metrics["critic_loss"])[0]
            ),
            "actor_loss": float(
                np.asarray(learner_metrics["actor_loss"])[0]
            ),
            "alpha": float(np.asarray(learner_metrics["alpha"])[0]),
            "actor_replica_checksum_delta": float(
                np.asarray(
                    learner_metrics["actor_replica_checksum_delta"]
                )[0]
            ),
            "replay_replica_checksum_delta": float(
                np.asarray(
                    learner_metrics["replay_replica_checksum_delta"]
                )[0]
            ),
            "transition_replica_max_abs_delta": float(
                np.asarray(
                    learner_metrics["transition_replica_max_abs_delta"]
                )[0]
            ),
            "learner_transition_batch": int(
                np.asarray(learner_metrics["learner_transition_batch"])[0]
            ),
            "formal_n_step": args.formal_n_step,
            "n_step_finalized_rows_local": (
                int(transitions.observations.shape[0])
                if args.n9_million_direction_schedule
                else None
            ),
            "n_step_pending_rows_local": (
                int(n_step_metadata["pending_tail_rows"])
                if n_step_metadata is not None
                else 0
            ),
            "replay_size": int(
                np.asarray(learner_metrics["replay_size"])[0]
            ),
            "actor_abs_max": float(
                np.asarray(learner_metrics["actor_abs_max"])[0]
            ),
            "critic_abs_max": float(
                np.asarray(learner_metrics["critic_abs_max"])[0]
            ),
            "parameters_finite": bool(
                np.asarray(learner_metrics["parameters_finite"])[0]
            ),
        }
        if args.algorithm == "crossq":
            row.update(
                {
                    "crossq_actor_updated": int(
                        np.asarray(learner_metrics["actor_updated"])[0]
                    ),
                    "crossq_current_q_mean": float(
                        np.asarray(learner_metrics["current_q_mean"])[0]
                    ),
                    "crossq_next_q_mean": float(
                        np.asarray(learner_metrics["next_q_mean"])[0]
                    ),
                    "crossq_brn_steps": int(
                        np.asarray(learner_metrics["brn_steps"])[0]
                    ),
                    "crossq_critic_gradient_norm_raw": float(
                        np.asarray(
                            learner_metrics["critic_gradient_norm_raw"]
                        )[0]
                    ),
                    "crossq_critic_gradient_norm_applied": float(
                        np.asarray(
                            learner_metrics[
                                "critic_gradient_norm_applied"
                            ]
                        )[0]
                    ),
                }
            )
        else:
            row.update(
                {
                    "sac_actor_updated": int(
                        np.asarray(learner_metrics["actor_updated"])[0]
                    ),
                    "sac_actor_updates_applied": int(
                        np.asarray(
                            learner_metrics["actor_updates_applied"]
                        )[0]
                    ),
                    "sac_actor_gradient_norm": float(
                        np.asarray(
                            learner_metrics["actor_gradient_norm"]
                        )[0]
                    ),
                    "sac_critic_gradient_norm": float(
                        np.asarray(
                            learner_metrics["critic_gradient_norm"]
                        )[0]
                    ),
                    "sac_target_q_mean": float(
                        np.asarray(learner_metrics["target_q_mean"])[0]
                    ),
                    "sac_target_q_std": float(
                        np.asarray(learner_metrics["target_q_std"])[0]
                    ),
                    "sac_actor_q_mean": float(
                        np.asarray(learner_metrics["actor_q_mean"])[0]
                    ),
                    "sac_entropy_term_mean": float(
                        np.asarray(
                            learner_metrics["entropy_term_mean"]
                        )[0]
                    ),
                    "sac_action_l2_mean": float(
                        np.asarray(learner_metrics["action_l2_mean"])[0]
                    ),
                    "sac_sampled_action_abs_mean": float(
                        np.asarray(
                            learner_metrics["sampled_action_abs_mean"]
                        )[0]
                    ),
                    "sac_sampled_action_abs_max": float(
                        np.asarray(
                            learner_metrics["sampled_action_abs_max"]
                        )[0]
                    ),
                }
            )
        history.append(row)
        if args.process_id == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        interactions_so_far = global_transitions_per_iteration * (
            iteration + 1
        ) + training_interactions_before_loop
        finite_row = row["parameters_finite"] and all(
            np.isfinite(value)
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        while (
            pending_milestones
            and interactions_so_far >= pending_milestones[0]
            and finite_row
        ):
            requested_milestone = pending_milestones.pop(0)
            milestone_record = {
                "requested_global_interactions": requested_milestone,
                "actual_global_interactions": interactions_so_far,
            }
            if args.full_checkpoint is not None:
                milestone_paths = _checkpoint_milestone_paths(
                    args.checkpoint,
                    args.full_checkpoint,
                    interactions_so_far,
                    args.process_id,
                )
                milestone_full_path = milestone_paths[
                    "full_state_process_local"
                ]
                milestone_full_sha256 = _save_full_checkpoint_atomic(
                    milestone_full_path,
                    _full_checkpoint_payload(
                        args=args,
                        source_precursor_sha256=source_metadata["npz_sha256"],
                        plane_archive_sha256=plane_archive_sha256,
                        plane_archive_extension_sha256=(
                            plane_archive_extension_sha256
                        ),
                        case=case,
                        actor_states=actor_states,
                        plane_source_states=plane_source_states,
                        plane_cursors=plane_cursors,
                        sharded_learner=sharded_learner,
                        n_step_pending=n_step_pending,
                        history=history,
                        global_interactions=interactions_so_far,
                        completed_collector_iterations=(
                            (
                                int(
                                    resume_payload[
                                        "completed_collector_iterations_including_warmup"
                                    ]
                                )
                                if resume_payload is not None
                                else 1
                            )
                            + iteration
                            + 1
                        ),
                        algorithm_extension={
                            "name": "formal_n9_milestone_state",
                            "training_algorithm": args.algorithm,
                            "formal_reward_name": args.formal_reward_name,
                            "formal_n_step": args.formal_n_step,
                        },
                    ),
                )
                milestone_record.update(
                    {
                        "full_state_process_local": str(milestone_full_path),
                        "full_state_sha256": milestone_full_sha256,
                    }
                )
            if args.process_id == 0 and args.checkpoint is not None:
                milestone_path = (
                    milestone_paths["actor_process0_only"]
                    if args.full_checkpoint is not None
                    else _milestone_checkpoint_base(
                        args.checkpoint, interactions_so_far
                    )
                )
                milestone_sac = jax.tree.map(
                    lambda value: value[0], sharded_learner.sac
                )
                milestone_state = actor_states[0]._replace(sac=milestone_sac)
                save_actor_checkpoint(
                    milestone_path,
                    milestone_state,
                    {
                        "generated_at": datetime.now(timezone.utc)
                        .astimezone()
                        .isoformat(),
                        "purpose": "mole_microbatch_distributed_sac_milestone",
                        "training_algorithm": args.algorithm,
                        "policy_architecture": (
                            "graph_sac_directional"
                            if args.algorithm == "graph_sac_directional"
                            else (
                                "graph_sac_v2"
                                if args.algorithm == "graph_sac_v2"
                                else "flat_mlp"
                            )
                        ),
                        "relative_wind_angle_deg": args.relative_wind_angle_deg,
                        "direction_condition": args.direction_condition,
                        "steering_branch_condition": (
                            args.steering_branch_condition
                        ),
                        "steering_branch_values": (
                            [-1.0, 1.0]
                            if args.steering_branch_condition
                            else None
                        ),
                        "initial_actor_checkpoint_sha256": (
                            initial_actor_checkpoint_sha256
                        ),
                        "graph_geometry": args.graph_geometry,
                        "target_turbines": args.target_turbines,
                        "action_parameterization": args.action_parameterization,
                        "action_dim": action_dim,
                        "seed": args.seed,
                        "requested_global_interactions": requested_milestone,
                        "actual_global_interactions": interactions_so_far,
                        "global_batch": (
                            args.local_environments * args.num_processes
                        ),
                        "inlet_mode": config.inlet_mode,
                        "episode_interactions": config.episode_interactions,
                        "reset_interactions": config.reset_interactions,
                        "source_precursor_sha256": source_metadata[
                            "npz_sha256"
                        ],
                        "plane_archive_sha256": plane_archive_sha256,
                        "plane_archive_extension_sha256": (
                            plane_archive_extension_sha256
                        ),
                    },
                    case=case,
                )
                milestone_record["actor_process0_only"] = str(milestone_path)
            saved_milestones.append(milestone_record)
            last_full_saved_interactions = interactions_so_far
            jax.experimental.multihost_utils.sync_global_devices(
                f"microbatch-milestone-{interactions_so_far}-written"
            )
        numeric_values = [
            value
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not row["parameters_finite"] or not all(
            np.isfinite(value) for value in numeric_values
        ):
            terminated_early_nonfinite = True
            break
        should_pause = False
        if args.stop_request is not None or args.stop_after_seconds is not None:
            requested = (args.stop_request is not None and args.stop_request.exists())
            expired = (args.stop_after_seconds is not None
                       and time.perf_counter() - process_started >= args.stop_after_seconds)
            flags = jax.experimental.multihost_utils.process_allgather(
                np.asarray([int(requested), int(expired)], dtype=np.int32), tiled=False
            )
            should_pause = bool(np.any(np.asarray(flags)))
            if should_pause:
                recoverable_pause_reason = ("stop_request" if np.any(np.asarray(flags)[..., 0])
                                            else "elapsed_budget")
        rolling_due = (args.full_checkpoint_interval > 0
                       and interactions_so_far - last_full_saved_interactions >= args.full_checkpoint_interval)
        if rolling_due or should_pause:
            checkpoint_started = time.perf_counter()
            slot = rolling_sequence % 2
            rolling_base = args.full_checkpoint.with_name(
                f"{args.full_checkpoint.stem}_rolling{slot}{args.full_checkpoint.suffix or '.pkl'}"
            )
            rolling_path = _process_checkpoint_path(rolling_base, args.process_id)
            payload = _full_checkpoint_payload(
                args=args, source_precursor_sha256=source_metadata["npz_sha256"],
                plane_archive_sha256=plane_archive_sha256,
                plane_archive_extension_sha256=plane_archive_extension_sha256,
                case=case, actor_states=actor_states,
                plane_source_states=plane_source_states, plane_cursors=plane_cursors,
                sharded_learner=sharded_learner, n_step_pending=n_step_pending,
                history=history, global_interactions=interactions_so_far,
                completed_collector_iterations=(
                    int(resume_payload["completed_collector_iterations_including_warmup"])
                    if resume_payload is not None else 1
                ) + iteration + 1,
                algorithm_extension={"name": "rolling_recoverable_full_state",
                                     "training_algorithm": args.algorithm},
            )
            digest = _save_full_checkpoint_atomic(rolling_path, payload)
            del payload
            record = {"path": str(rolling_path), "sha256": digest,
                      "global_interactions": interactions_so_far,
                      "plane_cursors": list(plane_cursors),
                      "process_id": args.process_id, "slot": slot,
                      "pause_reason": recoverable_pause_reason,
                      "checkpoint_io_seconds": time.perf_counter() - checkpoint_started}
            sidecar = rolling_path.with_suffix(rolling_path.suffix + ".json")
            temporary = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(temporary, sidecar)
            jax.experimental.multihost_utils.sync_global_devices(
                f"rolling-full-state-{interactions_so_far}-written"
            )
            rolling_checkpoint_records.append(record)
            last_full_saved_interactions = interactions_so_far
            rolling_sequence += 1
            if args.process_id == 0:
                print(json.dumps({"phase": "recoverable_checkpoint", **record}), flush=True)
        if should_pause:
            break

    times = np.asarray(
        [row["wallclock_seconds"] for row in history], dtype=np.float64
    )
    steady_times = times[-min(5, len(times)) :]
    steady_seconds = float(np.median(steady_times))
    steady_rate = global_transitions_per_iteration / steady_seconds
    final_metrics = {
        key: float(np.asarray(value)[0])
        for key, value in learner_metrics.items()
    }
    final_metrics.update(
        {
            key: history[-1][key]
            for key in (
                "svg_actor_loss",
                "svg_objective_mean",
                "svg_gradient_norm_mean",
                "svg_gradient_norm_max",
                "svg_gradient_finite",
                "svg_actor_replica_checksum_delta",
            )
        }
    )
    final_sac_for_checks = jax.tree.map(
        lambda value: value[0], sharded_learner.sac
    )
    formal_audit_checks = {}
    if args.n9_million_direction_schedule:
        assert isinstance(transitions, AuditableTransitionBatch)
        transition_directions = np.asarray(transitions.direction_deg)
        transition_horizons = np.asarray(transitions.n_step_horizons)
        transition_discounts = np.asarray(transitions.bootstrap_discounts)
        transition_dones = np.asarray(transitions.dones)
        formal_audit_checks = {
            "formal_auditable_replay_enabled": isinstance(
                sharded_learner.replay, AuditableReplayState
            ),
            "formal_p0_positive_finite": bool(
                np.all(np.isfinite(np.asarray(transitions.p0_mw_per_turbine)))
                and np.all(np.asarray(transitions.p0_mw_per_turbine) > 0.0)
            ),
            "formal_training_directions_exact": bool(
                np.all(np.isin(transition_directions, TRAIN_DIRECTIONS_DEG))
            ),
            "formal_selected_reward_exact": bool(
                args.formal_n_step > 1
                or np.array_equal(
                    np.asarray(transitions.rewards),
                    np.asarray(getattr(transitions, args.formal_reward_name)),
                )
            ),
            "formal_return_horizon_contract": bool(
                np.all(transition_horizons >= 1)
                and np.all(transition_horizons <= args.formal_n_step)
                and np.all(
                    transition_discounts
                    == np.where(
                        transition_dones != 0.0,
                        np.float32(0.0),
                        np.asarray(
                            config.gamma ** transition_horizons,
                            dtype=transition_discounts.dtype,
                        ),
                    )
                )
            ),
        }
    checks = {
        "cuda_backend": jax.default_backend() == "gpu",
        "two_global_devices": len(jax.devices()) == args.num_processes,
        "one_local_device": len(jax.local_devices()) == 1,
        "actor_replicas_synchronized": (
            final_metrics["actor_replica_checksum_delta"] < 1.0e-5
        ),
        "finite_metrics": all(
            np.isfinite(value) for value in final_metrics.values()
        ),
        "completed_without_early_nonfinite_termination": (
            not terminated_early_nonfinite
        ),
        "expected_local_transition_batch": (
            transitions.observations.shape[0] > 0
            and (
                args.formal_n_step > 1
                or transitions.observations.shape[0]
                == args.local_environments * args.collection_steps
            )
        ),
        "central_replay_transition_batch": (
            final_metrics["learner_transition_batch"]
            == (
                transitions.observations.shape[0] * args.num_processes
                if args.replay_mode == "global"
                else transitions.observations.shape[0]
            )
        ),
        "central_replay_replicas_synchronized": (
            final_metrics["transition_replica_max_abs_delta"] == 0.0
            if args.replay_mode == "global"
            else True
        ),
        "crossq_critic_statistics_replicated": (
            args.algorithm != "crossq"
            or final_metrics["critic_statistics_replica_checksum_delta"]
            < 1.0e-5
        ),
        "crossq_has_no_target_critic": (
            args.algorithm != "crossq"
            or len(final_sac_for_checks.target_critic) == 0
        ),
        "crossq_brn_steps_match_updates": (
            args.algorithm != "crossq"
            or int(final_metrics["brn_steps"])
            == int(final_metrics["sac_updates"])
        ),
        "sac_update_count_matches_replay_gate": (
            final_metrics["sac_updates"] > 0
            if updates_per_iteration > 0
            and final_metrics["replay_size"]
            > (
                config.minimum_replay_size
                if args.n9_million_direction_schedule
                else config.minimum_replay_size - 1
            )
            else final_metrics["sac_updates"] == 0
        ),
        "svg_gradients_finite": (
            not svg_enabled or bool(history[-1]["svg_gradient_finite"])
        ),
        "svg_actor_replicas_synchronized": (
            not svg_enabled
            or history[-1]["svg_actor_replica_checksum_delta"] < 1.0e-5
        ),
        "online_precursor_ab_history_preserved": (
            config.inlet_mode != "online"
            or source_metadata.get("loaded_state_format")
            == "complete_ab3_flow_state"
        ),
        "online_precursor_boundary_pressure_history_preserved": (
            config.inlet_mode != "online"
            or source_metadata.get("boundary_pressure_history_loaded", False)
        ),
        "cached_plane_archive_contract": (
            plane_archive is None
            or (
                plane_archive.source_state_sha256
                == source_metadata.get("source_state", {}).get(
                    "sha256", source_metadata["npz_sha256"]
                )
                and len(plane_cursors) == len(actor_states)
                and max(plane_cursors) <= plane_archive.completed_planes
                and (
                    not args.episode_random_plane_windows
                    or all(
                        start <= cursor <= stop
                        for cursor, (start, stop) in zip(
                            plane_cursors, plane_member_bounds
                        )
                    )
                )
            )
        ),
        "steering_branch_assignment_balanced": (
            not args.steering_branch_condition
            or global_steering_branches.count(1.0)
            == global_steering_branches.count(-1.0)
        ),
        **formal_audit_checks,
    }
    total_global_transitions = training_interactions_before_loop + (
        global_transitions_per_iteration * len(history)
    )
    checkpoint_metadata = {
        "generated_at": datetime.now(
            timezone.utc
        ).astimezone().isoformat(),
        "purpose": "mole_microbatch_distributed_sac",
        "training_algorithm": args.algorithm,
        "policy_architecture": (
            "graph_sac_directional"
            if args.algorithm == "graph_sac_directional"
            else (
                "graph_sac_v2"
                if args.algorithm == "graph_sac_v2"
                else "flat_mlp"
            )
        ),
        "relative_wind_angle_deg": args.relative_wind_angle_deg,
        "plane_offset_add": args.plane_offset_add,
        "direction_condition": args.direction_condition,
        "steering_branch_condition": args.steering_branch_condition,
        "steering_branch_values": (
            [-1.0, 1.0] if args.steering_branch_condition else None
        ),
        "local_actor_shard_steering_branches": steering_branches,
        "initial_actor_checkpoint_sha256": initial_actor_checkpoint_sha256,
        "initial_actor_checkpoint_metadata": initial_actor_metadata,
        "graph_geometry": args.graph_geometry,
        "reflection_equivariant_actor": effective_reflection_equivariant_actor,
        "configured_reflection_equivariant_actor": (
            args.reflection_equivariant_actor
        ),
        "direction_common_center_xz_m": direction_center,
        "global_transitions_including_compile_warmup": (
            total_global_transitions
        ),
        "actual_global_interactions": total_global_transitions,
        "learner_observation_dim": observation_dim,
        "resume_mode": (
            "learner_only"
            if args.resume_learner_only
            else ("full_state" if resume_payload is not None else "fresh")
        ),
        "global_batch": (
            args.local_environments * args.num_processes
        ),
        "local_environments": args.local_environments,
        "target_turbines": args.target_turbines,
        "action_parameterization": args.action_parameterization,
        "action_dim": action_dim,
        "layout_contract": layout_contract(case),
        "microbatch": args.microbatch,
        "compact_solver": args.compact_solver,
        "collection_steps_per_environment": args.collection_steps,
        "zero_yaw_burnin_interactions_per_environment": (
            args.burnin_interactions
        ),
        "episode_interactions": config.episode_interactions,
        "reset_interactions": config.reset_interactions,
        "updates_per_iteration": updates_per_iteration,
        "learning_rate": args.learning_rate,
        "initial_alpha": args.initial_alpha,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "graph_actor_warmup_updates": args.graph_actor_warmup_updates,
        "graph_actor_minimum_replay_size": (
            args.graph_actor_minimum_replay_size
        ),
        "graph_actor_update_period": effective_actor_update_period,
        "configured_graph_actor_update_period": (
            args.graph_actor_update_period
        ),
        "graph_action_l2_coefficient": effective_graph_action_l2_coefficient,
        "configured_graph_action_l2_coefficient": (
            args.graph_action_l2_coefficient
        ),
        "entropy_scale": args.entropy_scale,
        "conservative_actor_init": args.conservative_actor_init,
        "neutral_actor_initial_scale": args.neutral_actor_initial_scale,
        "diagnostic_overrides": {
            "updates_per_iteration": args.diagnostic_updates_per_iteration,
            "resume_alpha": args.diagnostic_resume_alpha,
            "reward_scale": args.diagnostic_reward_scale,
            "actor_update_period": args.diagnostic_actor_update_period,
            "actor_learning_rate_scale": (
                args.diagnostic_actor_learning_rate_scale
            ),
            "action_l2_coefficient": (
                args.diagnostic_action_l2_coefficient
            ),
            "graph_action_l2_coefficient": (
                args.diagnostic_graph_action_l2_coefficient
            ),
            "disable_reflection_equivariant_actor": (
                args.diagnostic_disable_reflection_equivariant_actor
            ),
            "entropy_scale": args.diagnostic_entropy_scale,
        },
        "random_action_transitions_global": args.random_action_transitions,
        "random_action_interactions_per_environment": (
            args.random_action_transitions // global_environments
        ),
        "random_action_transitions_per_actor_shard": (
            random_action_transitions_per_shard
        ),
        "source_precursor_sha256": source_metadata["npz_sha256"],
        "source_precursor_state_format": source_metadata.get(
            "loaded_state_format", "unknown"
        ),
        "source_precursor_boundary_pressure_history_loaded": (
            source_metadata.get("boundary_pressure_history_loaded", False)
        ),
        "inlet_mode": config.inlet_mode,
        "replay_mode": args.replay_mode,
        "formal_reward_name": args.formal_reward_name,
        "formal_n_step": args.formal_n_step,
        "replay_schema": (
            "auditable_replay_v1"
            if args.n9_million_direction_schedule
            else "legacy_replay_v1"
        ),
        "plane_archive_sha256": plane_archive_sha256,
        "plane_archive_extension_sha256": plane_archive_extension_sha256,
        "plane_archive_path": (
            str(args.plane_archive)
            if args.plane_archive is not None
            else None
        ),
        "algorithm_extension": {
            "name": (
                "graph_sac_directional_complete_pair_critic"
                if args.algorithm == "graph_sac_directional"
                else (
                    "graph_sac_v2_factorized_critic"
                    if args.algorithm == "graph_sac_v2"
                    else (
                        "crossq_critic_only_brn"
                        if args.algorithm == "crossq"
                        else (
                            "exact_svg_actor_auxiliary"
                            if svg_enabled
                            else (
                                "zero_yaw_conservative_initialization"
                                if args.conservative_actor_init
                                else "none"
                            )
                        )
                    )
                )
            ),
            "training_algorithm": args.algorithm,
            "policy_architecture": (
                "graph_sac_directional"
                if args.algorithm == "graph_sac_directional"
                else (
                    "graph_sac_v2"
                    if args.algorithm == "graph_sac_v2"
                    else "flat_mlp"
                )
            ),
            "graph_nodes": (
                wake_graph.num_nodes if wake_graph is not None else None
            ),
            "graph_edges": (
                int(wake_graph.edge_sources.shape[0])
                if wake_graph is not None
                else None
            ),
            "graph_actor_warmup_updates": args.graph_actor_warmup_updates,
            "graph_actor_minimum_replay_size": (
                args.graph_actor_minimum_replay_size
            ),
            "graph_actor_update_period": effective_actor_update_period,
            "configured_graph_actor_update_period": (
                args.graph_actor_update_period
            ),
            "graph_action_l2_coefficient": effective_graph_action_l2_coefficient,
            "configured_graph_action_l2_coefficient": (
                args.graph_action_l2_coefficient
            ),
            "crossq_critic_hidden_dim": args.crossq_critic_hidden_dim,
            "crossq_learning_rate": args.crossq_learning_rate,
            "crossq_entropy_learning_rate": (
                args.crossq_entropy_learning_rate
            ),
            "crossq_adam_b1": args.crossq_adam_b1,
            "crossq_brn_momentum": args.crossq_brn_momentum,
            "crossq_brn_warmup_updates": args.crossq_brn_warmup_updates,
            "crossq_policy_delay": args.crossq_policy_delay,
            "crossq_critic_gradient_clip_norm": (
                args.crossq_critic_gradient_clip_norm
            ),
            "gradient_samples_per_host": args.svg_gradient_samples_per_host,
            "actor_updates_per_iteration": (
                args.svg_actor_updates_per_iteration
            ),
            "gradient_clip_norm": args.svg_gradient_clip_norm,
            "loss_coefficient": args.svg_loss_coefficient,
        },
    }
    if (
        args.checkpoint is not None
        and args.process_id == 0
        and not terminated_early_nonfinite
    ):
        final_sac = jax.tree.map(
            lambda value: value[0], sharded_learner.sac
        )
        checkpoint_state = actor_states[0]._replace(sac=final_sac)
        save_actor_checkpoint(
            args.checkpoint,
            checkpoint_state,
            checkpoint_metadata,
            case=case,
        )
    full_checkpoint_path = None
    full_checkpoint_sha256 = None
    if args.full_checkpoint is not None and not terminated_early_nonfinite:
        full_checkpoint_path = _process_checkpoint_path(
            args.full_checkpoint, args.process_id
        )
        full_checkpoint_sha256 = _save_full_checkpoint_atomic(
            full_checkpoint_path,
            {
                "schema_version": FULL_CHECKPOINT_SCHEMA_VERSION,
                "purpose": "mole_protocol_parity_full_training_state",
                "process_id": args.process_id,
                "num_processes": args.num_processes,
                "completed_collector_iterations_including_warmup": (
                    (
                        int(
                            resume_payload[
                                "completed_collector_iterations_including_warmup"
                            ]
                        )
                        if resume_payload is not None
                        else 1
                    )
                    + len(history)
                ),
                "global_interactions": total_global_transitions,
                "configuration": _full_checkpoint_configuration(
                    args,
                    source_metadata["npz_sha256"],
                    plane_archive_sha256,
                    case,
                ),
                "plane_archive_extension_sha256": (
                    plane_archive_extension_sha256
                ),
                "actor_states": _tree_to_host(tuple(actor_states)),
                "plane_source_states": _tree_to_host(
                    tuple(plane_source_states)
                ),
                "plane_cursors": list(plane_cursors),
                "learner": _sharded_learner_to_checkpoint(
                    sharded_learner
                ),
                "n_step_pending": n_step_pending,
                "history": history,
                "algorithm_extension": checkpoint_metadata[
                    "algorithm_extension"
                ],
            },
        )
    jax.experimental.multihost_utils.sync_global_devices(
        "microbatch-checkpoint-written"
    )
    result = {
        "schema_version": 1,
        "purpose": "mole_microbatch_distributed_sac_benchmark",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": ("paused_at_block_boundary" if all(checks.values()) and recoverable_pause_reason
                   else "pass" if all(checks.values()) else "fail"),
        "recoverable_pause_reason": recoverable_pause_reason,
        "declared_global_interactions": args.planned_global_interactions,
        "declared_budget_completed": (args.planned_global_interactions is None
                                      or total_global_transitions == args.planned_global_interactions),
        "whole_input_horizon": input_horizon,
        "rolling_checkpoint_records": rolling_checkpoint_records,
        "checks": checks,
        "implementation_sha256": _implementation_sha256(),
        "protocol": protocol_metadata(
            case, config, args.action_parameterization
        ),
        "environment": _environment(),
        "configuration": {
            "h20_eight_process_performance_benchmark": (
                args.h20_eight_process_performance_benchmark
            ),
            "h20_eight_process_formal_training": (
                args.h20_eight_process_formal_training
            ),
            "training_algorithm": args.algorithm,
            "target_turbines": args.target_turbines,
            "action_parameterization": args.action_parameterization,
            "action_dim": action_dim,
            "layout_contract": layout_contract(case),
            "local_environments": args.local_environments,
            "microbatch": args.microbatch,
            "compact_solver": args.compact_solver,
            "rhs_assembly": args.rhs_assembly,
            "actor_state_shards": len(actor_states),
            "direction_condition": args.direction_condition,
            "steering_branch_condition": args.steering_branch_condition,
            "local_actor_shard_steering_branches": steering_branches,
            "global_steering_branch_counts": (
                {
                    "minus": global_steering_branches.count(-1.0),
                    "plus": global_steering_branches.count(1.0),
                }
                if args.steering_branch_condition
                else None
            ),
            "initial_actor_checkpoint": (
                str(args.initial_actor_checkpoint)
                if args.initial_actor_checkpoint is not None
                else None
            ),
            "initial_actor_checkpoint_sha256": (
                initial_actor_checkpoint_sha256
            ),
            "collection_steps_per_environment": args.collection_steps,
            "zero_yaw_burnin_interactions_per_environment": (
                args.burnin_interactions
            ),
            "episode_interactions": config.episode_interactions,
            "reset_interactions": config.reset_interactions,
            "global_transitions_per_iteration": (
                global_transitions_per_iteration
            ),
            "updates_per_iteration": updates_per_iteration,
            "replay_mode": args.replay_mode,
            "formal_reward_name": args.formal_reward_name,
            "formal_n_step": args.formal_n_step,
            "replay_schema": (
                "auditable_replay_v1"
                if args.n9_million_direction_schedule
                else "legacy_replay_v1"
            ),
            "sample_size_per_host": args.sample_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "initial_alpha": args.initial_alpha,
            "actor_learning_rate_scale": args.actor_learning_rate_scale,
            "graph_actor_warmup_updates": args.graph_actor_warmup_updates,
            "graph_actor_minimum_replay_size": (
                args.graph_actor_minimum_replay_size
            ),
            "graph_actor_update_period": effective_actor_update_period,
            "entropy_scale": args.entropy_scale,
            "conservative_actor_init": args.conservative_actor_init,
            "effective_global_sample_size": (
                args.sample_size * args.num_processes
            ),
            "replay_capacity_per_host": args.replay_capacity,
            "minimum_replay_size": args.minimum_replay_size,
            "random_action_transitions_global": args.random_action_transitions,
            "random_action_interactions_per_environment": (
                args.random_action_transitions // global_environments
            ),
            "random_action_transitions_per_actor_shard": (
                random_action_transitions_per_shard
            ),
            "source_precursor_sha256": source_metadata["npz_sha256"],
            "source_precursor_state_format": source_metadata.get(
                "loaded_state_format", "unknown"
            ),
            "source_precursor_boundary_pressure_history_loaded": (
                source_metadata.get("boundary_pressure_history_loaded", False)
            ),
            "inlet_mode": config.inlet_mode,
            "plane_archive_sha256": plane_archive_sha256,
            "plane_archive_extension_sha256": (
                plane_archive_extension_sha256
            ),
            "plane_archive_path": (
                str(args.plane_archive)
                if args.plane_archive is not None
                else None
            ),
            "episode_random_plane_windows": (
                args.episode_random_plane_windows
            ),
            "plane_window_seed": args.plane_window_seed,
            "plane_window_required_planes": plane_window_required_planes,
            "plane_member_bounds": [list(value) for value in plane_member_bounds],
            "plane_cursors": list(plane_cursors),
            "total_global_transitions_including_compile_warmup": (
                total_global_transitions
            ),
            "requested_iterations": args.iterations,
            "completed_iterations": len(history),
            "terminated_early_nonfinite": terminated_early_nonfinite,
            "resumed_from_full_checkpoint": (
                str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None
            ),
            "resumed_global_interactions": resumed_global_interactions,
            "checkpoint_milestones": checkpoint_milestones,
            "resume_implementation_override": (
                resume_implementation_override
            ),
            "resume_diagnostic_override": resume_diagnostic_override,
            "resume_plane_archive_extension_override": (
                resume_plane_archive_extension_override
            ),
            "diagnostic_updates_per_iteration": (
                args.diagnostic_updates_per_iteration
            ),
            "diagnostic_resume_alpha": args.diagnostic_resume_alpha,
            "diagnostic_reward_scale": args.diagnostic_reward_scale,
            "diagnostic_actor_update_period": (
                args.diagnostic_actor_update_period
            ),
            "diagnostic_actor_learning_rate_scale": (
                args.diagnostic_actor_learning_rate_scale
            ),
            "diagnostic_action_l2_coefficient": (
                args.diagnostic_action_l2_coefficient
            ),
            "diagnostic_graph_action_l2_coefficient": (
                args.diagnostic_graph_action_l2_coefficient
            ),
            "diagnostic_disable_reflection_equivariant_actor": (
                args.diagnostic_disable_reflection_equivariant_actor
            ),
            "effective_graph_action_l2_coefficient": (
                effective_graph_action_l2_coefficient
            ),
            "effective_reflection_equivariant_actor": (
                effective_reflection_equivariant_actor
            ),
            "diagnostic_entropy_scale": (
                args.diagnostic_entropy_scale
            ),
            "sac_actor_updates_applied_total": sum(
                row.get("sac_actor_updates_applied", 0)
                for row in history
            ),
            "svg_gradient_samples_per_host": (
                args.svg_gradient_samples_per_host
            ),
            "svg_actor_updates_per_iteration": (
                args.svg_actor_updates_per_iteration
            ),
            "svg_gradient_clip_norm": args.svg_gradient_clip_norm,
            "svg_loss_coefficient": args.svg_loss_coefficient,
            "crossq_critic_hidden_dim": args.crossq_critic_hidden_dim,
            "crossq_learning_rate": args.crossq_learning_rate,
            "crossq_entropy_learning_rate": (
                args.crossq_entropy_learning_rate
            ),
            "crossq_adam_b1": args.crossq_adam_b1,
            "crossq_brn_momentum": args.crossq_brn_momentum,
            "crossq_brn_warmup_updates": (
                args.crossq_brn_warmup_updates
            ),
            "crossq_policy_delay": args.crossq_policy_delay,
            "crossq_critic_gradient_clip_norm": (
                args.crossq_critic_gradient_clip_norm
            ),
        },
        "evidence_boundary": (
            "XT-H20 8-process performance benchmark only. It preserves the "
            "N=9 LES, 32-environment collector cadence, directional R1 reward, "
            "16-step Replay construction and Graph-SAC update, but disables "
            "the 32k random/critic-only warm-up so short profiling runs include "
            "Actor and Critic work. Its policy is not a scientific result and "
            "its checkpoint is intentionally disabled."
            if args.h20_eight_process_performance_benchmark
            else "Scientific training protocol; interpret with its recorded configuration."
        ),
        "timing": {
            "total_process_wallclock_seconds": (
                time.perf_counter() - process_started
            ),
            "zero_yaw_burnin_seconds": burnin_seconds,
            "compile_and_warm_actor_seconds": (
                compile_and_warm_actor_seconds
            ),
            "compile_and_first_learner_seconds": (
                compile_and_first_learner_seconds
            ),
            "compile_and_first_svg_seconds": (
                compile_and_first_svg_seconds
            ),
            "steady_state_iteration_seconds_median": steady_seconds,
            "steady_state_global_interactions_per_second": steady_rate,
            "steady_state_projected_hours_for_one_million_interactions": (
                1_000_000 / steady_rate / 3600.0
            ),
            "precursor_generation_seconds_total": (
                precursor_generation_seconds_total
            ),
            "precursor_planes_generated_total": (
                precursor_planes_generated_total
            ),
            "discarded_resume_compile_precursor_seconds": (
                discarded_resume_compile_precursor_seconds
            ),
            "discarded_resume_compile_precursor_planes": (
                discarded_resume_compile_precursor_planes
            ),
        },
        "final_metrics": final_metrics,
        "history": history,
        "checkpoint": {
            "initial": initial_checkpoint_records,
            "path_process0_only": (
                str(args.checkpoint)
                if args.checkpoint is not None
                else None
            ),
            "metadata": checkpoint_metadata,
            "milestones_process_local": saved_milestones,
            "milestones_process0_only": [
                record
                for record in saved_milestones
                if "actor_process0_only" in record
            ],
            "full_state_process_local": (
                str(full_checkpoint_path)
                if full_checkpoint_path is not None
                else None
            ),
            "full_state_sha256": full_checkpoint_sha256,
        },
    }
    encoded_without_hash = json.dumps(
        result, indent=2, sort_keys=True
    )
    result["payload_sha256_without_hash_field"] = hashlib.sha256(
        encoded_without_hash.encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    status = (0 if result["status"] == "pass" else
              75 if result["status"] == "paused_at_block_boundary" else 1)
    jax.distributed.shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
