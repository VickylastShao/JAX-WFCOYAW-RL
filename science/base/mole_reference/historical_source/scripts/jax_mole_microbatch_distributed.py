#!/usr/bin/env python3
"""Two-host Mole SAC benchmark with time-sliced LES microbatches.

Each host retains 16 independent environment states, but advances them in
small microbatches before performing the source-aligned distributed SAC update.
This avoids the poor per-environment efficiency of one batch-16 LES graph while
preserving the public 32-environment, 512-transition collector cadence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from jax_mole_compact_sac_benchmark import (
    ReplayState,
    SACState,
    init_replay,
    init_sac,
    make_mole_training_case,
    replay_add,
    sac_update,
)
from jax_mole_continuous_sac import (
    ContinuousControlConfig,
    initialize_continuous_training,
    initialize_evaluation_from_training,
    load_initial_precursor,
    make_continuous_training_iteration,
    make_evaluation_step,
    make_precursor_plane_generator,
    save_actor_checkpoint,
    slice_initial_precursor,
)
from mole_plane_archive import PlaneArchive


IMPLEMENTATION_FILES = (
    "jax_mole_microbatch_distributed.py",
    "jax_mole_continuous_sac.py",
    "jax_mole_compact_sac_benchmark.py",
    "mole_plane_archive.py",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--precursor", type=Path, required=True)
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
        "--checkpoint-milestones",
        default="",
        help=(
            "Comma-separated global interaction thresholds. When --checkpoint "
            "is set, process 0 writes an actor snapshot at the first iteration "
            "that reaches each threshold."
        ),
    )
    parser.add_argument("--local-environments", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--collection-steps", type=int, default=16)
    parser.add_argument("--burnin-interactions", type=int, default=0)
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
    parser.add_argument("--hidden-dim", type=int, default=256)
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


def _process_checkpoint_path(base: Path, process_id: int) -> Path:
    suffix = base.suffix or ".pkl"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}.process{process_id}{suffix}")


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
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported full-checkpoint schema: {path}")
    return payload


def _full_checkpoint_configuration(
    args: argparse.Namespace,
    source_precursor_sha256: str,
    plane_archive_sha256: str | None,
) -> dict[str, Any]:
    return {
        "local_environments": args.local_environments,
        "microbatch": args.microbatch,
        "collection_steps": args.collection_steps,
        "burnin_interactions": args.burnin_interactions,
        "inlet_mode": args.inlet_mode,
        "replay_mode": args.replay_mode,
        "replay_capacity": args.replay_capacity,
        "minimum_replay_size": args.minimum_replay_size,
        "random_action_transitions": args.random_action_transitions,
        "updates_per_transition": args.updates_per_transition,
        "sample_size_per_host": args.sample_size,
        "hidden_dim": args.hidden_dim,
        "rhs_assembly": args.rhs_assembly,
        "seed": args.seed,
        "source_precursor_sha256": source_precursor_sha256,
        "plane_archive_sha256": plane_archive_sha256,
        "implementation_sha256": _implementation_sha256(),
    }


def _make_learner_step(
    optimizers: tuple[Any, Any, Any],
    config: ContinuousControlConfig,
    updates_per_iteration: int,
    replay_mode: str,
):
    def step(
        learner: LearnerState,
        transitions: TransitionBatch,
    ) -> tuple[LearnerState, dict[str, jax.Array]]:
        if replay_mode == "global":
            transitions = jax.tree.map(
                lambda value: jax.lax.all_gather(
                    value, "hosts", axis=0, tiled=True
                ),
                transitions,
            )
        replay = replay_add(
            learner.replay,
            transitions.observations,
            transitions.actions,
            transitions.rewards,
            transitions.next_observations,
            transitions.dones,
        )
        transition_checksum = (
            jnp.sum(transitions.observations)
            + jnp.sum(transitions.actions)
            + jnp.sum(transitions.rewards)
            + jnp.sum(transitions.next_observations)
            + jnp.sum(transitions.dones)
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
            sac, losses = sac_update(
                sac,
                replay,
                update_key,
                optimizers,
                config.sample_size,
                1.0,
                config.gamma,
                config.tau,
                -3.0,
                "hosts",
                config.policy_mode,
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
            return (
                sac,
                key,
                jax.tree.map(lambda value: value[-1], losses),
            )

        zero_losses = {
            "critic_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "actor_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "alpha": jnp.exp(learner.sac.log_alpha),
        }

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
        critic_abs_max = jnp.max(
            jnp.stack(
                [jnp.max(jnp.abs(value)) for value in jax.tree.leaves(sac.critic)]
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
                "learner_transition_batch": jnp.asarray(
                    transitions.observations.shape[0], dtype=jnp.int32
                ),
                "actor_abs_max": actor_abs_max,
                "critic_abs_max": critic_abs_max,
                "parameters_finite": parameters_finite,
            },
        )

    return step


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "backend": jax.default_backend(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_devices": [str(device) for device in jax.local_devices()],
        "global_devices": [str(device) for device in jax.devices()],
    }


def main() -> int:
    process_started = time.perf_counter()
    args = parse_args()
    checkpoint_milestones = sorted(
        {
            int(value)
            for value in args.checkpoint_milestones.split(",")
            if value.strip()
        }
    )
    if any(value <= 0 for value in checkpoint_milestones):
        raise ValueError("--checkpoint-milestones values must be positive")
    if args.local_environments % args.microbatch:
        raise ValueError(
            "--microbatch must divide --local-environments exactly"
        )
    if args.burnin_interactions < 0:
        raise ValueError("--burnin-interactions must be non-negative")
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
    plane_archive = (
        PlaneArchive(args.plane_archive)
        if args.plane_archive is not None
        else None
    )
    plane_archive_sha256 = (
        plane_archive.manifest_sha256 if plane_archive is not None else None
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
        local_device_ids=0,
    )

    source_case = make_mole_training_case(args.local_environments)
    source_precursor, source_metadata = load_initial_precursor(
        args.precursor, source_case
    )
    plane_cursors: list[int] = []
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
        source_steps = source_metadata.get("source_steps")
        if source_steps is None:
            source_steps = [plane_archive.source_completed_steps] * (
                args.local_environments
            )
        if len(source_steps) != args.local_environments:
            raise ValueError(
                "precursor source-step count does not match local environments"
            )
        plane_cursors = [
            int(step) - plane_archive.source_completed_steps
            for step in source_steps
        ]
        if min(plane_cursors) < 0:
            raise ValueError("precursor member predates plane archive source")
        if max(plane_cursors) >= plane_archive.completed_planes:
            raise ValueError("precursor member offset exceeds plane archive")
    case = make_mole_training_case(
        args.microbatch,
        rhs_assembly=args.rhs_assembly,
    )
    actor_state_shards = args.local_environments // args.microbatch
    random_action_transitions_per_shard = (
        args.random_action_transitions
        * args.microbatch
        // global_environments
    )
    config = ContinuousControlConfig(
        inlet_mode=args.inlet_mode,
        sample_size=args.sample_size,
        minimum_replay_size=args.minimum_replay_size,
        random_action_transitions=random_action_transitions_per_shard,
        updates_per_transition=args.updates_per_transition,
    )
    actor_replay_capacity = args.microbatch * args.collection_steps
    actor_states = []
    plane_source_states = []
    optimizers = None
    precursor_functions = None
    farm_functions = None
    for shard in range(actor_state_shards):
        start = shard * args.microbatch
        stop = start + args.microbatch
        source_slice = slice_initial_precursor(source_precursor, start, stop)
        (
            actor_state,
            shard_optimizers,
            shard_precursor_functions,
            shard_farm_functions,
        ) = initialize_continuous_training(
            case,
            config,
            source_slice,
            args.seed + 1000 * args.process_id + shard,
            args.hidden_dim,
            3.0e-6,
            actor_replay_capacity,
        )
        actor_states.append(actor_state)
        if args.inlet_mode == "time_planes" and plane_archive is None:
            if not hasattr(source_slice, "rhs_previous"):
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

    resume_payload = None
    resumed_global_interactions = 0
    if args.resume_checkpoint is not None:
        resume_payload = _load_full_checkpoint(args.resume_checkpoint)
        saved_configuration = resume_payload["configuration"]
        expected_configuration = _full_checkpoint_configuration(
            args,
            source_metadata["npz_sha256"],
            plane_archive_sha256,
        )
        if resume_payload["process_id"] != args.process_id:
            raise ValueError("resume checkpoint process id does not match")
        if resume_payload["num_processes"] != args.num_processes:
            raise ValueError("resume checkpoint process count does not match")
        if saved_configuration != expected_configuration:
            raise ValueError(
                "resume checkpoint configuration does not match current run"
            )
        resumed_global_interactions = int(
            resume_payload["global_interactions"]
        )

    burnin_seconds = 0.0
    if args.burnin_interactions > 0 and resume_payload is None:
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
            evaluation = initialize_evaluation_from_training(state)
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
            actor_states[index] = state._replace(
                precursor_flow=evaluation.precursor_flow,
                farm_flow=evaluation.farm_flow,
                observation_frames=jnp.zeros_like(
                    evaluation.observation_frames
                ),
                observation=jnp.zeros_like(evaluation.observation),
                previous_yaw=jnp.zeros_like(evaluation.previous_yaw),
            )
        burnin_seconds = time.perf_counter() - burnin_started

    observation_dim = int(actor_states[0].observation.shape[-1])
    learner_sac, learner_optimizers = init_sac(
        jax.random.PRNGKey(args.seed + 9000),
        observation_dim,
        3,
        args.hidden_dim,
        3.0e-6,
        initial_alpha=config.initial_alpha,
        policy_mode=config.policy_mode,
        entropy_learning_rate=config.entropy_learning_rate,
    )
    learner = LearnerState(
        replay=init_replay(
            args.replay_capacity, observation_dim, 3
        ),
        sac=learner_sac,
        key=jax.random.PRNGKey(args.seed + 10000 + args.process_id),
        replay_checksum=jnp.asarray(0.0, dtype=jnp.float32),
    )
    actor_collect = jax.jit(
        make_continuous_training_iteration(
            case,
            config,
            optimizers,
            precursor_functions,
            farm_functions,
            updates_per_iteration=0,
            collection_steps_per_iteration=args.collection_steps,
        ),
        donate_argnums=(0,),
    )

    def collect_actor_state(index: int, state: Any):
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
    global_transitions_per_iteration = (
        args.local_environments
        * args.num_processes
        * args.collection_steps
    )
    updates_per_iteration = (
        global_transitions_per_iteration * args.updates_per_transition
    )
    learner_step = jax.pmap(
        _make_learner_step(
            learner_optimizers,
            config,
            updates_per_iteration,
            args.replay_mode,
        ),
        axis_name="hosts",
        in_axes=(0, 0),
        out_axes=(0, 0),
        donate_argnums=(0,),
    )
    sharded_learner = jax.tree.map(lambda value: value[None], learner)

    precursor_seconds_before_compile = precursor_generation_seconds_total
    precursor_planes_before_compile = precursor_planes_generated_total
    compile_actor_started = time.perf_counter()
    for index, state in enumerate(actor_states):
        (
            actor_states[index],
            actor_metrics,
            _,
            _,
            _,
        ) = collect_actor_state(index, state)
    compile_and_warm_actor_seconds = (
        time.perf_counter() - compile_actor_started
    )
    transitions = _transition_batch(actor_states)
    sharded_transitions = jax.tree.map(
        lambda value: value[None], transitions
    )
    compile_learner_started = time.perf_counter()
    sharded_learner, learner_metrics = learner_step(
        sharded_learner, sharded_transitions
    )
    jax.block_until_ready(learner_metrics["critic_loss"])
    compile_and_first_learner_seconds = (
        time.perf_counter() - compile_learner_started
    )
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
        restored_learner = LearnerState(
            **_tree_to_device(resume_payload["learner"])
        )
        sharded_learner = jax.tree.map(
            lambda value: value[None], restored_learner
        )
    training_interactions_before_loop = resumed_global_interactions + (
        0 if resume_payload is not None else global_transitions_per_iteration
    )

    history = []
    saved_milestones: list[dict[str, Any]] = []
    pending_milestones = list(checkpoint_milestones)
    terminated_early_nonfinite = False
    for iteration in range(args.iterations):
        started = time.perf_counter()
        iteration_precursor_seconds = 0.0
        iteration_farm_seconds = 0.0
        iteration_precursor_planes = 0
        actor = jax.tree.map(lambda value: value[0], sharded_learner.sac.actor)
        for index, state in enumerate(actor_states):
            actor_states[index] = _copy_actor_into_state(state, actor)
            (
                actor_states[index],
                actor_metrics,
                generated_seconds,
                farm_seconds,
                generated_planes,
            ) = collect_actor_state(
                index, actor_states[index]
            )
            iteration_precursor_seconds += generated_seconds
            iteration_farm_seconds += farm_seconds
            iteration_precursor_planes += generated_planes
        collector_seconds = (
            iteration_precursor_seconds + iteration_farm_seconds
        )
        learner_started = time.perf_counter()
        transitions = _transition_batch(actor_states)
        sharded_transitions = jax.tree.map(
            lambda value: value[None], transitions
        )
        sharded_learner, learner_metrics = learner_step(
            sharded_learner, sharded_transitions
        )
        jax.block_until_ready(learner_metrics["critic_loss"])
        learner_seconds = time.perf_counter() - learner_started
        elapsed = time.perf_counter() - started
        row = {
            "iteration": iteration + 1,
            "wallclock_seconds": elapsed,
            "collector_seconds": collector_seconds,
            "precursor_generation_seconds": iteration_precursor_seconds,
            "farm_collection_seconds": iteration_farm_seconds,
            "precursor_planes_generated": iteration_precursor_planes,
            "learner_seconds": learner_seconds,
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
            "learner_transition_batch": int(
                np.asarray(learner_metrics["learner_transition_batch"])[0]
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
        history.append(row)
        if args.process_id == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
            interactions_so_far = global_transitions_per_iteration * (
                iteration + 1
            ) + training_interactions_before_loop
            while (
                args.checkpoint is not None
                and pending_milestones
                and interactions_so_far >= pending_milestones[0]
                and row["parameters_finite"]
                and all(
                    np.isfinite(value)
                    for value in row.values()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
            ):
                requested_milestone = pending_milestones.pop(0)
                milestone_path = args.checkpoint.with_name(
                    f"{args.checkpoint.stem}_step{interactions_so_far}"
                    f"{args.checkpoint.suffix}"
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
                        "seed": args.seed,
                        "requested_global_interactions": requested_milestone,
                        "actual_global_interactions": interactions_so_far,
                        "global_batch": (
                            args.local_environments * args.num_processes
                        ),
                        "inlet_mode": config.inlet_mode,
                        "source_precursor_sha256": source_metadata[
                            "npz_sha256"
                        ],
                        "plane_archive_sha256": plane_archive_sha256,
                    },
                )
                saved_milestones.append(
                    {
                        "requested_global_interactions": requested_milestone,
                        "actual_global_interactions": interactions_so_far,
                        "path": str(milestone_path),
                    }
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
            transitions.observations.shape[0]
            == args.local_environments * args.collection_steps
        ),
        "central_replay_transition_batch": (
            final_metrics["learner_transition_batch"]
            == (
                global_transitions_per_iteration
                if args.replay_mode == "global"
                else args.local_environments * args.collection_steps
            )
        ),
        "central_replay_replicas_synchronized": (
            final_metrics["replay_replica_checksum_delta"] < 1.0e-5
            if args.replay_mode == "global"
            else True
        ),
        "sac_update_count_matches_replay_gate": (
            final_metrics["sac_updates"] > 0
            if updates_per_iteration > 0
            and final_metrics["replay_size"] >= config.minimum_replay_size
            else final_metrics["sac_updates"] == 0
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
            )
        ),
    }
    total_global_transitions = training_interactions_before_loop + (
        global_transitions_per_iteration * len(history)
    )
    checkpoint_metadata = {
        "generated_at": datetime.now(
            timezone.utc
        ).astimezone().isoformat(),
        "purpose": "mole_microbatch_distributed_sac",
        "global_transitions_including_compile_warmup": (
            total_global_transitions
        ),
        "global_batch": (
            args.local_environments * args.num_processes
        ),
        "local_environments": args.local_environments,
        "microbatch": args.microbatch,
        "collection_steps_per_environment": args.collection_steps,
        "zero_yaw_burnin_interactions_per_environment": (
            args.burnin_interactions
        ),
        "updates_per_iteration": updates_per_iteration,
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
        "plane_archive_sha256": plane_archive_sha256,
        "plane_archive_path": (
            str(args.plane_archive)
            if args.plane_archive is not None
            else None
        ),
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
        )
    full_checkpoint_path = None
    full_checkpoint_sha256 = None
    if args.full_checkpoint is not None and not terminated_early_nonfinite:
        full_checkpoint_path = _process_checkpoint_path(
            args.full_checkpoint, args.process_id
        )
        replicated_learner = jax.tree.map(
            lambda value: value[0], sharded_learner
        )
        full_checkpoint_sha256 = _save_full_checkpoint_atomic(
            full_checkpoint_path,
            {
                "schema_version": 1,
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
                ),
                "actor_states": _tree_to_host(tuple(actor_states)),
                "plane_source_states": _tree_to_host(
                    tuple(plane_source_states)
                ),
                "plane_cursors": list(plane_cursors),
                "learner": _tree_to_host(replicated_learner._asdict()),
                "history": history,
            },
        )
    jax.experimental.multihost_utils.sync_global_devices(
        "microbatch-checkpoint-written"
    )
    result = {
        "schema_version": 1,
        "purpose": "mole_microbatch_distributed_sac_benchmark",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "environment": _environment(),
        "configuration": {
            "local_environments": args.local_environments,
            "microbatch": args.microbatch,
            "rhs_assembly": args.rhs_assembly,
            "actor_state_shards": len(actor_states),
            "collection_steps_per_environment": args.collection_steps,
            "zero_yaw_burnin_interactions_per_environment": (
                args.burnin_interactions
            ),
            "global_transitions_per_iteration": (
                global_transitions_per_iteration
            ),
            "updates_per_iteration": updates_per_iteration,
            "replay_mode": args.replay_mode,
            "sample_size_per_host": args.sample_size,
            "effective_global_sample_size": (
                args.sample_size * args.num_processes
            ),
            "replay_capacity_per_host": args.replay_capacity,
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
            "plane_archive_path": (
                str(args.plane_archive)
                if args.plane_archive is not None
                else None
            ),
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
        },
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
            "path_process0_only": (
                str(args.checkpoint)
                if args.checkpoint is not None
                else None
            ),
            "metadata": checkpoint_metadata,
            "milestones_process0_only": saved_milestones,
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
    status = 0 if result["status"] == "pass" else 1
    jax.distributed.shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
