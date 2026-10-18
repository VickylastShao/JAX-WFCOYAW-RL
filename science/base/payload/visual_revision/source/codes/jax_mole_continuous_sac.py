#!/usr/bin/env python3
"""Mole-aligned LES inlet and SAC control utilities.

This module builds on ``jax_mole_compact_sac_benchmark`` but deliberately
keeps the earlier throughput benchmark unchanged. It supports an online
precursor LES, a Taylor-shifted replay of one pre-generated three-dimensional
precursor field, and explicit time-resolved precursor planes. The time-plane
path matches the public Mole dataflow: a precursor-only run emits planes and
the farm LES consumes them without advancing the precursor in the control step.

The published Mole et al. protocol is followed where it is observable:

* 50 LES steps per 10 s control interaction;
* 77 current streamwise probe averages and one previous yaw angle per turbine;
* differential yaw actions with a 1.0 deg/s speed bound and +/-40 deg bounds;
* mean per-turbine power in MW with the published large-angle penalty form;
* SAC gamma=0.99, batch size 256, two ReLU hidden layers of width 256,
  TorchRL-compatible biased-softplus TanhNormal parameterization, and one
  optimizer update per newly collected transition.

The published actor checkpoint and evaluation logs are available, but the
original turbulent precursor realizations are not. This code therefore tests
a source-aligned JAX retraining, not a bitwise reproduction.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_compact_sac_benchmark import (
    Array,
    CompactMoleCase,
    FlowState,
    PyTree,
    ReplayState,
    SACState,
    _apply_velocity_boundaries,
    actor_sample,
    build_flow_functions,
    case_metadata,
    critic_apply,
    init_replay,
    init_sac,
    load_precursor,
    make_article_224d_precursor_case,
    make_initial_flow,
    make_precursor_case,
    layout_contract,
    mlp_apply,
    _precursor_inlet,
    replay_add,
    sac_update,
)

PUBLISHED_STATIC_BO_YAW_DEG = (
    20.406089782714844,
    12.728775024414062,
    -3.5519638061523438,
)

ACTION_PARAMETERIZATIONS = (
    "per_turbine",
    "per_turbine_target",
    "n9_two_column",
)


@dataclass(frozen=True)
class ContinuousControlConfig:
    inlet_mode: str = "online"
    les_steps_per_interaction: int = 50
    les_scan_unroll: int = 1
    frame_stack: int = 1
    max_yaw_speed_deg_per_s: float = 1.0
    max_yaw_angle_deg: float = 40.0
    angle_penalty_scale: float = 0.5
    angle_penalty_exponent: int = 26
    gamma: float = 0.99
    tau: float = 0.005
    sample_size: int = 256
    minimum_replay_size: int = 32000
    random_action_transitions: int = 32000
    updates_per_transition: int = 1
    policy_mode: str = "public_torchrl"
    initial_alpha: float = 10.0
    entropy_learning_rate: float = 1.0e-5
    episode_interactions: int = 500
    reset_interactions: int = 150

    @property
    def control_dt_seconds(self) -> float:
        return self.les_steps_per_interaction * 0.2


class ContinuousTrainingState(NamedTuple):
    precursor_flow: FlowState
    farm_flow: FlowState
    observation_frames: Array
    observation: Array
    previous_yaw: Array
    replay: ReplayState
    sac: SACState
    key: Array
    transitions: Array


class ContinuousEvaluationState(NamedTuple):
    precursor_flow: FlowState
    farm_flow: FlowState
    observation_frames: Array
    observation: Array
    previous_yaw: Array
    interaction_index: Array


class FixedControllerPairEvaluationState(NamedTuple):
    precursor_flow: FlowState
    zero_farm_flow: FlowState
    static_farm_flow: FlowState
    interaction_index: Array


class IntervalDiagnostics(NamedTuple):
    inlet_mean_u: Array
    inlet_delta_rms: Array


def control_action_dim(
    case: CompactMoleCase,
    action_parameterization: str = "per_turbine",
) -> int:
    """Return the policy/Critic action width for one farm layout."""

    if action_parameterization in {"per_turbine", "per_turbine_target"}:
        return case.num_turbines
    if action_parameterization == "n9_two_column":
        if case.num_turbines != 9:
            raise ValueError("n9_two_column requires the N=9 layout")
        return 2
    raise ValueError(
        f"unknown action parameterization: {action_parameterization}"
    )


def expand_control_action(
    normalized_action: Array,
    case: CompactMoleCase,
    action_parameterization: str = "per_turbine",
) -> Array:
    """Map policy actions to one normalized yaw-rate command per turbine."""

    action_dim = control_action_dim(case, action_parameterization)
    if normalized_action.shape[-1] != action_dim:
        raise ValueError(
            "control action width does not match parameterization: "
            f"{normalized_action.shape[-1]} != {action_dim}"
        )
    if action_parameterization in {"per_turbine", "per_turbine_target"}:
        return normalized_action
    first_column = jnp.repeat(normalized_action[..., 0:1], 3, axis=-1)
    second_column = jnp.repeat(normalized_action[..., 1:2], 3, axis=-1)
    downstream_column = jnp.zeros_like(first_column)
    return jnp.concatenate(
        (first_column, second_column, downstream_column), axis=-1
    )


def apply_control_action(
    normalized_action: Array,
    previous_yaw: Array,
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    action_parameterization: str = "per_turbine",
) -> Array:
    """Apply a normalized rate or absolute target under physical yaw limits."""

    turbine_action = expand_control_action(
        normalized_action, case, action_parameterization
    )
    max_increment = np.float32(
        config.max_yaw_speed_deg_per_s * config.control_dt_seconds
    )
    if action_parameterization == "per_turbine_target":
        target_yaw = (
            turbine_action * np.float32(config.max_yaw_angle_deg)
        )
        yaw_increment = jnp.clip(
            target_yaw - previous_yaw,
            -max_increment,
            max_increment,
        )
    else:
        yaw_increment = turbine_action * max_increment
    return jnp.clip(
        previous_yaw + yaw_increment,
        -np.float32(config.max_yaw_angle_deg),
        np.float32(config.max_yaw_angle_deg),
    )


def resolve_external_normalized_action(
    state: ContinuousEvaluationState,
    action_provider: Callable[[ContinuousEvaluationState], Array],
    action_dim: int,
) -> Array:
    """Resolve one replay/fixed action without exposing target LES state."""

    normalized_action = jnp.asarray(action_provider(state), dtype=jnp.float32)
    expected_shape = state.previous_yaw.shape[:-1] + (action_dim,)
    if normalized_action.shape != expected_shape:
        raise ValueError(
            "external action shape differs from the controller contract: "
            f"{normalized_action.shape} != {expected_shape}"
        )
    return normalized_action


def actor_deterministic(
    actor: PyTree,
    observations: Array,
    policy_mode: str = "public_torchrl",
) -> Array:
    activation = "relu" if policy_mode == "public_torchrl" else "silu"
    output = mlp_apply(actor, observations, activation=activation)
    mean, _ = jnp.split(output, 2, axis=-1)
    return jnp.tanh(mean)


def normalize_probe_frames_by_upstream_speed(
    frames: Array,
    measured_upstream_speed_m_per_s: Array | float,
) -> Array:
    """Express probe velocities as local relative deficits ``u/U_up - 1``."""

    values = jnp.asarray(frames, dtype=jnp.float32)
    speed = jnp.asarray(
        measured_upstream_speed_m_per_s, dtype=jnp.float32
    )
    if speed.ndim > values.ndim:
        raise ValueError("upstream-speed rank exceeds probe-frame rank")
    speed = jnp.reshape(speed, (*speed.shape, *((1,) * (values.ndim - speed.ndim))))
    denominator = jnp.where(speed > 0.0, speed, jnp.nan)
    return values / denominator - np.float32(1.0)


def stacked_observation(
    frames: Array,
    previous_yaw: Array,
    case: CompactMoleCase,
    config: ContinuousControlConfig,
) -> Array:
    normalized_frames = (
        (frames - np.float32(6.0)) / np.float32(6.0)
    ).reshape((frames.shape[0], -1))
    normalized_yaw = (
        previous_yaw
        * np.float32(4.0 / config.max_yaw_angle_deg)
    )
    return jnp.concatenate((normalized_frames, normalized_yaw), axis=-1)


def _online_precursor_inlet(precursor_flow: FlowState) -> Array:
    return precursor_flow.velocity[:, :, 0]


def make_precursor_plane_generator(
    precursor_functions: dict[str, Any],
    steps: int,
) -> Callable[[FlowState], tuple[FlowState, Array]]:
    """Advance a precursor and emit one outlet plane after every LES step."""

    if steps < 1:
        raise ValueError("precursor plane count must be positive")
    precursor_case = precursor_functions["case"]
    zero_yaw = jnp.zeros(
        (precursor_case.batch, precursor_case.num_turbines),
        dtype=jnp.float32,
    )
    zero_weights = jnp.zeros(
        (
            precursor_case.batch,
            precursor_case.num_turbines,
            precursor_case.nx,
            precursor_case.ny,
            precursor_case.nz,
        ),
        dtype=jnp.float32,
    )

    def generate(initial: FlowState) -> tuple[FlowState, Array]:
        def body(current: FlowState, _: None):
            next_state, _ = precursor_functions["advance"](
                current, zero_yaw, zero_weights, None
            )
            return next_state, _online_precursor_inlet(next_state)

        final, time_major_planes = jax.lax.scan(
            body, initial, xs=None, length=steps
        )
        return final, jnp.swapaxes(time_major_planes, 0, 1)

    return generate


def _time_plane_marker(case: CompactMoleCase) -> FlowState:
    """Keep farm state small when the precursor is managed by the producer."""

    velocity = jnp.zeros((case.batch, 3, 1, 1, 1), dtype=jnp.float32)
    return FlowState(
        velocity=velocity,
        rhs_previous=jnp.zeros_like(velocity),
        rhs_previous_2=jnp.zeros_like(velocity),
        filtered_disk_speed=jnp.zeros(
            (case.batch, case.num_turbines), dtype=jnp.float32
        ),
        bottom_pressure_gradient=jnp.zeros(
            (case.batch, 2, 1, 1), dtype=jnp.float32
        ),
        open_x_pressure_gradient=jnp.zeros(
            (case.batch, 2, 2, 1, 1), dtype=jnp.float32
        ),
        step_index=jnp.asarray(0, dtype=jnp.int32),
    )


def _advance_coupled_interval(
    state: tuple[FlowState, FlowState],
    yaw: Array,
    disk_weights: Array,
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    inlet_planes: Array | None = None,
    inlet_plane_start_step: Array | None = None,
) -> tuple[
    FlowState,
    FlowState,
    Array,
    Array,
    IntervalDiagnostics,
]:
    if config.inlet_mode not in {"online", "replay_volume", "time_planes"}:
        raise ValueError(f"unknown inlet mode: {config.inlet_mode}")
    if config.les_scan_unroll < 1:
        raise ValueError("les_scan_unroll must be positive")
    if config.inlet_mode == "online":
        precursor_case = precursor_functions["case"]
        zero_yaw = jnp.zeros(
            (precursor_case.batch, precursor_case.num_turbines),
            dtype=jnp.float32,
        )
        zero_weights = jnp.zeros(
            (
                precursor_case.batch,
                precursor_case.num_turbines,
                precursor_case.nx,
                precursor_case.ny,
                precursor_case.nz,
            ),
            dtype=jnp.float32,
        )
    elif config.inlet_mode == "time_planes":
        if inlet_planes is None or inlet_plane_start_step is None:
            raise ValueError(
                "time_planes mode requires planes and their farm-step origin"
            )
        if inlet_planes.ndim != 5:
            raise ValueError(
                "time planes must have shape [batch, steps, 3, ny, nz]"
            )

    def body(
        carry: tuple[
            FlowState,
            FlowState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
        _: None,
    ) -> tuple[
        tuple[
            FlowState,
            FlowState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
        None,
    ]:
        (
            precursor_flow,
            farm_flow,
            power_sum,
            probe_sum,
            inlet_u_sum,
            first_inlet,
            last_inlet,
            step,
        ) = carry
        if config.inlet_mode == "online":
            precursor_flow, _ = precursor_functions["advance"](
                precursor_flow,
                zero_yaw,
                zero_weights,
                None,
            )
            inlet = _online_precursor_inlet(precursor_flow)
        elif config.inlet_mode == "replay_volume":
            inlet = _precursor_inlet(
                precursor_flow.velocity,
                farm_flow.step_index,
                case,
            )
        else:
            plane_index = farm_flow.step_index - inlet_plane_start_step
            inlet = jax.lax.dynamic_index_in_dim(
                inlet_planes,
                plane_index,
                axis=1,
                keepdims=False,
            )
        farm_flow, power = farm_functions["advance"](
            farm_flow,
            yaw,
            disk_weights,
            inlet,
        )
        probes = farm_functions["probes"](farm_flow.velocity)
        first_inlet = jnp.where(step == 0, inlet, first_inlet)
        return (
            precursor_flow,
            farm_flow,
            power_sum + power,
            probe_sum + probes,
            inlet_u_sum + jnp.mean(inlet[:, 0], axis=(-2, -1)),
            first_inlet,
            inlet,
            step + 1,
        ), None

    initial_power_sum = jnp.zeros(
        (case.batch, case.num_turbines), dtype=jnp.float32
    )
    initial_probe_sum = jnp.zeros(
        (case.batch, case.num_turbines, 77), dtype=jnp.float32
    )
    initial_inlet_u_sum = jnp.zeros(
        (case.batch,), dtype=jnp.float32
    )
    initial_inlet = jnp.zeros(
        (case.batch, 3, case.ny, case.nz), dtype=jnp.float32
    )
    (
        precursor_flow,
        farm_flow,
        power_sum,
        probe_sum,
        inlet_u_sum,
        first_inlet,
        last_inlet,
        _,
    ), _ = jax.lax.scan(
        body,
        (
            state[0],
            state[1],
            initial_power_sum,
            initial_probe_sum,
            initial_inlet_u_sum,
            initial_inlet,
            initial_inlet,
            jnp.asarray(0, dtype=jnp.int32),
        ),
        xs=None,
        length=config.les_steps_per_interaction,
        unroll=config.les_scan_unroll,
    )
    interval_length = np.float32(config.les_steps_per_interaction)
    return (
        precursor_flow,
        farm_flow,
        power_sum / interval_length,
        probe_sum / interval_length,
        IntervalDiagnostics(
            inlet_mean_u=inlet_u_sum / interval_length,
            inlet_delta_rms=jnp.sqrt(
                jnp.mean(
                    (last_inlet - first_inlet) ** 2,
                    axis=(1, 2, 3),
                )
            ),
        ),
    )


def _next_frames(frames: Array, averaged_probes: Array) -> Array:
    return jnp.concatenate(
        (frames[:, 1:], averaged_probes.reshape(frames.shape[0], 1, -1)),
        axis=1,
    )


def _advance_episode_reset(
    precursor_flow: FlowState,
    farm_flow: FlowState,
    initial_yaw: Array,
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    inlet_planes: Array | None = None,
    inlet_plane_start_step: Array | None = None,
) -> tuple[FlowState, FlowState]:
    steps_to_zero = int(
        math.ceil(
            config.max_yaw_angle_deg
            / (
                config.max_yaw_speed_deg_per_s
                * config.control_dt_seconds
            )
        )
    )
    if steps_to_zero > config.reset_interactions:
        raise ValueError("reset interval is too short for the yaw-rate limit")

    def body(
        carry: tuple[FlowState, FlowState],
        index: Array,
    ) -> tuple[tuple[FlowState, FlowState], None]:
        denominator = np.float32(max(1, steps_to_zero - 1))
        fraction = jnp.clip(
            np.float32(steps_to_zero - 1) - index.astype(jnp.float32),
            0.0,
            denominator,
        ) / denominator
        yaw = initial_yaw * fraction
        weights = farm_functions["disk_weights"](yaw)
        next_precursor, next_farm, _, _, _ = _advance_coupled_interval(
            carry,
            yaw,
            weights,
            case,
            config,
            precursor_functions,
            farm_functions,
            inlet_planes,
            inlet_plane_start_step,
        )
        return (next_precursor, next_farm), None

    (precursor_flow, farm_flow), _ = jax.lax.scan(
        body,
        (precursor_flow, farm_flow),
        jnp.arange(config.reset_interactions),
    )
    return precursor_flow, farm_flow


def make_continuous_episode_reset(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    *,
    observation_suffix: Array | None = None,
) -> Callable[..., ContinuousTrainingState]:
    """Build an explicit reset so a new static geometry owns the reset flow."""

    condition_dim = (
        0 if observation_suffix is None else int(observation_suffix.shape[-1])
    )
    if observation_suffix is not None and observation_suffix.ndim != 1:
        raise ValueError("observation_suffix must be a one-dimensional vector")

    def reset(
        state: ContinuousTrainingState,
        inlet_planes: Array | None = None,
        inlet_plane_start_step: Array | None = None,
    ) -> ContinuousTrainingState:
        reset_precursor, reset_farm = _advance_episode_reset(
            state.precursor_flow,
            state.farm_flow,
            state.previous_yaw,
            case,
            config,
            precursor_functions,
            farm_functions,
            inlet_planes,
            inlet_plane_start_step,
        )
        physical_observation = (
            state.observation[..., :-condition_dim]
            if condition_dim
            else state.observation
        )
        zero_observation = jnp.zeros_like(physical_observation)
        if observation_suffix is not None:
            suffix = jnp.broadcast_to(
                observation_suffix,
                (*zero_observation.shape[:-1], condition_dim),
            )
            zero_observation = jnp.concatenate(
                (zero_observation, suffix), axis=-1
            )
        return state._replace(
            precursor_flow=reset_precursor,
            farm_flow=reset_farm,
            observation_frames=jnp.zeros_like(state.observation_frames),
            observation=zero_observation,
            previous_yaw=jnp.zeros_like(state.previous_yaw),
        )

    return reset


def _reward(
    mean_power: Array,
    yaw: Array,
    config: ContinuousControlConfig,
) -> tuple[Array, Array, Array]:
    mean_turbine_power_mw = (
        jnp.mean(mean_power, axis=-1) / np.float32(1.0e6)
    )
    angle_penalty = jnp.mean(
        np.float32(config.angle_penalty_scale)
        * (
            yaw / np.float32(config.max_yaw_angle_deg)
        )
        ** config.angle_penalty_exponent,
        axis=-1,
    )
    return (
        mean_turbine_power_mw - angle_penalty,
        mean_turbine_power_mw,
        angle_penalty,
    )


def initialize_continuous_training(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_initial: Array | FlowState | None,
    seed: int,
    hidden_dim: int,
    learning_rate: float,
    replay_capacity: int,
    *,
    initial_inlet: Array | None = None,
    action_parameterization: str = "per_turbine",
) -> tuple[
    ContinuousTrainingState,
    tuple[Any, Any, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if config.inlet_mode not in {"online", "replay_volume", "time_planes"}:
        raise ValueError(f"unknown inlet mode: {config.inlet_mode}")
    external_time_planes = (
        config.inlet_mode == "time_planes" and initial_inlet is not None
    )
    if external_time_planes:
        if precursor_initial is not None:
            raise ValueError(
                "external time-plane initialization does not accept a "
                "precursor volume"
            )
        if initial_inlet.shape != (case.batch, 3, case.ny, case.nz):
            raise ValueError(
                "initial time plane must have shape "
                f"{(case.batch, 3, case.ny, case.nz)}"
            )
        precursor_flow = _time_plane_marker(case)
        precursor_functions = {
            "case": case,
            "external_time_planes": True,
        }
        inlet = jnp.asarray(initial_inlet, dtype=jnp.float32)
    else:
        if precursor_initial is None:
            raise ValueError(
                "precursor_initial is required without initial_inlet"
            )
        precursor_velocity = (
            precursor_initial.velocity
            if isinstance(precursor_initial, FlowState)
            else precursor_initial
        )
        precursor_nx = int(precursor_velocity.shape[2])
        if precursor_nx == 144:
            precursor_case_factory = make_precursor_case
        elif precursor_nx == 2304:
            precursor_case_factory = make_article_224d_precursor_case
        else:
            raise ValueError(
                "Unsupported precursor x dimension "
                f"{precursor_nx}; expected public 14D nx=144 or inferred "
                "article 224D nx=2304"
            )
        precursor_case = precursor_case_factory(
            case.batch,
            case.pressure_corrections,
            case.pressure_projection,
        )
        precursor_functions = build_flow_functions(
            precursor_case, periodic_x=True, include_turbines=False
        )
        if isinstance(precursor_initial, FlowState):
            precursor_flow = precursor_initial
        else:
            precursor_flow = make_initial_flow(
                precursor_case, precursor_functions, seed, precursor_initial
            )
        if config.inlet_mode in {"online", "time_planes"}:
            inlet = _online_precursor_inlet(precursor_flow)
        else:
            inlet = _precursor_inlet(
                precursor_flow.velocity,
                jnp.asarray(0, dtype=jnp.int32),
                case,
            )
    farm_functions = build_flow_functions(
        case, periodic_x=False, include_turbines=True
    )
    farm_flow = make_initial_flow(case, farm_functions, seed + 1)
    farm_velocity = _apply_velocity_boundaries(
        farm_flow.velocity, inlet, periodic_x=False
    )
    farm_flow = farm_flow._replace(velocity=farm_velocity)
    yaw = jnp.zeros(
        (case.batch, case.num_turbines), dtype=jnp.float32
    )
    probes = farm_functions["probes"](farm_flow.velocity).reshape(
        case.batch, -1
    )
    frames = jnp.repeat(
        probes[:, None, :], config.frame_stack, axis=1
    )
    observation = stacked_observation(frames, yaw, case, config)
    action_dim = control_action_dim(case, action_parameterization)
    sac, optimizers = init_sac(
        jax.random.PRNGKey(seed + 2),
        int(observation.shape[-1]),
        action_dim,
        hidden_dim,
        learning_rate,
        initial_alpha=config.initial_alpha,
        policy_mode=config.policy_mode,
        entropy_learning_rate=config.entropy_learning_rate,
    )
    replay = init_replay(
        replay_capacity, int(observation.shape[-1]), action_dim
    )
    state = ContinuousTrainingState(
        precursor_flow=(
            _time_plane_marker(case)
            if config.inlet_mode == "time_planes"
            else precursor_flow
        ),
        farm_flow=farm_flow,
        observation_frames=frames,
        observation=observation,
        previous_yaw=yaw,
        replay=replay,
        sac=sac,
        key=jax.random.PRNGKey(seed + 3),
        transitions=jnp.asarray(0, dtype=jnp.int32),
    )
    return state, optimizers, precursor_functions, farm_functions


def make_continuous_training_iteration(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    optimizers: tuple[Any, Any, Any],
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    updates_per_iteration: int,
    axis_name: str | None = None,
    collection_steps_per_iteration: int = 1,
    action_parameterization: str = "per_turbine",
    actor_sample_fn: Callable[..., tuple[Array, Array]] | None = None,
    observation_suffix: Array | None = None,
    return_raw_transitions: bool = False,
    reset_episode_in_collector: bool = True,
) -> Callable[
    [ContinuousTrainingState],
    tuple[ContinuousTrainingState, dict[str, Array]],
]:
    if collection_steps_per_iteration < 1:
        raise ValueError("collection_steps_per_iteration must be positive")
    sample_policy = actor_sample if actor_sample_fn is None else actor_sample_fn
    condition_dim = 0 if observation_suffix is None else int(observation_suffix.shape[-1])
    if observation_suffix is not None and observation_suffix.ndim != 1:
        raise ValueError("observation_suffix must be a one-dimensional vector")

    def condition_observation(observation: Array) -> Array:
        if observation_suffix is None:
            return observation
        suffix = jnp.broadcast_to(
            observation_suffix,
            (*observation.shape[:-1], condition_dim),
        )
        return jnp.concatenate((observation, suffix), axis=-1)
    zero_losses = {
        "critic_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "actor_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
        "alpha": jnp.asarray(config.initial_alpha, dtype=jnp.float32),
    }

    def iteration(
        training: ContinuousTrainingState,
        inlet_planes: Array | None = None,
        inlet_plane_start_step: Array | None = None,
    ) -> tuple[ContinuousTrainingState, dict[str, Array]]:
        def collect_once(
            current: ContinuousTrainingState,
            _: None,
        ) -> tuple[ContinuousTrainingState, dict[str, Array]]:
            action_key, next_key = jax.random.split(current.key)
            policy_action, _ = sample_policy(
                current.sac.actor,
                current.observation,
                action_key,
                1.0,
                config.policy_mode,
            )
            random_action = jax.random.uniform(
                action_key,
                policy_action.shape,
                minval=-1.0,
                maxval=1.0,
            )
            normalized_action = jnp.where(
                current.transitions < config.random_action_transitions,
                random_action,
                policy_action,
            )
            yaw = apply_control_action(
                normalized_action,
                current.previous_yaw,
                case,
                config,
                action_parameterization,
            )
            yaw_increment = yaw - current.previous_yaw
            weights = farm_functions["disk_weights"](yaw)
            (
                precursor_flow,
                farm_flow,
                mean_power,
                mean_probes,
                interval_diagnostics,
            ) = _advance_coupled_interval(
                (current.precursor_flow, current.farm_flow),
                yaw,
                weights,
                case,
                config,
                precursor_functions,
                farm_functions,
                inlet_planes,
                inlet_plane_start_step,
            )
            frames = _next_frames(current.observation_frames, mean_probes)
            next_observation = condition_observation(stacked_observation(
                frames, yaw, case, config
            ))
            reward, mean_turbine_power_mw, angle_penalty = _reward(
                mean_power, yaw, config
            )
            farm_power_mw = (
                mean_turbine_power_mw * np.float32(mean_power.shape[-1])
            )
            interaction_index = current.transitions // case.batch
            episode_done = (
                (interaction_index + 1) % config.episode_interactions
            ) == 0
            # TorchRL's StepCounter marks this boundary as truncated=True and
            # terminated=False. SAC therefore bootstraps through the time limit
            # even though the environment is reset before the next collection.
            terminated = jnp.zeros((case.batch,), dtype=jnp.float32)
            replay = replay_add(
                current.replay,
                current.observation,
                normalized_action,
                reward,
                next_observation,
                terminated,
            )
            advanced_training = current._replace(
                precursor_flow=precursor_flow,
                farm_flow=farm_flow,
                observation_frames=frames,
                observation=next_observation,
                previous_yaw=yaw,
                replay=replay,
                key=next_key,
                transitions=current.transitions + case.batch,
            )

            def reset_episode(
                state: ContinuousTrainingState,
            ) -> ContinuousTrainingState:
                reset_precursor, reset_farm = _advance_episode_reset(
                    state.precursor_flow,
                    state.farm_flow,
                    state.previous_yaw,
                    case,
                    config,
                    precursor_functions,
                    farm_functions,
                    inlet_planes,
                    inlet_plane_start_step,
                )
                return state._replace(
                    precursor_flow=reset_precursor,
                    farm_flow=reset_farm,
                    observation_frames=jnp.zeros_like(
                        state.observation_frames
                    ),
                    observation=(
                        condition_observation(
                            jnp.zeros_like(state.observation[..., :-condition_dim])
                        )
                        if condition_dim
                        else jnp.zeros_like(state.observation)
                    ),
                    previous_yaw=jnp.zeros_like(state.previous_yaw),
                )

            next_training = (
                jax.lax.cond(
                    episode_done,
                    reset_episode,
                    lambda state: state,
                    advanced_training,
                )
                if reset_episode_in_collector
                else advanced_training
            )
            metrics = {
                "mean_farm_power_mw": jnp.mean(farm_power_mw),
                "mean_reward": jnp.mean(reward),
                "mean_angle_penalty": jnp.mean(angle_penalty),
                "mean_abs_yaw_deg": jnp.mean(jnp.abs(yaw)),
                "mean_abs_yaw_speed_deg_per_s": jnp.mean(
                    jnp.abs(yaw_increment)
                )
                / np.float32(config.control_dt_seconds),
                "inlet_interval_delta_rms": jnp.mean(
                    interval_diagnostics.inlet_delta_rms
                ),
                "episode_reset": episode_done.astype(jnp.float32),
                **(
                    {
                        "raw_observation": current.observation,
                        "raw_action": normalized_action,
                        "raw_reward_r0": reward,
                        "raw_next_observation": next_observation,
                        "raw_done": episode_done.astype(jnp.float32),
                        "raw_turbine_power_w": mean_power,
                        "raw_applied_yaw_deg": yaw,
                        "raw_sequence_id": jnp.broadcast_to(
                            interaction_index.astype(jnp.int32),
                            (case.batch,),
                        ),
                    }
                    if return_raw_transitions
                    else {}
                ),
            }
            return next_training, metrics

        collected, collection_metrics = jax.lax.scan(
            collect_once,
            training,
            xs=None,
            length=collection_steps_per_iteration,
        )
        update_key = collected.key

        def update_once(
            carry: tuple[SACState, Array],
            _: None,
        ) -> tuple[tuple[SACState, Array], dict[str, Array]]:
            sac, key = carry
            key, step_key = jax.random.split(key)
            sac, losses = sac_update(
                sac,
                collected.replay,
                step_key,
                optimizers,
                config.sample_size,
                1.0,
                config.gamma,
                config.tau,
                -float(control_action_dim(case, action_parameterization)),
                axis_name,
                config.policy_mode,
            )
            return (sac, key), losses

        enough_samples = (
            collected.replay.size >= config.minimum_replay_size
        )

        def perform_updates(
            operand: tuple[SACState, Array],
        ) -> tuple[SACState, Array, dict[str, Array]]:
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

        def skip_updates(
            operand: tuple[SACState, Array],
        ) -> tuple[SACState, Array, dict[str, Array]]:
            return operand[0], operand[1], zero_losses

        if updates_per_iteration > 0:
            sac, update_key, losses = jax.lax.cond(
                enough_samples,
                perform_updates,
                skip_updates,
                (collected.sac, update_key),
            )
        else:
            sac = collected.sac
            losses = zero_losses

        next_training = collected._replace(sac=sac, key=update_key)
        actor_checksum = sum(
            jnp.sum(value) for value in jax.tree.leaves(sac.actor)
        )
        if axis_name is None:
            replica_delta = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            replica_delta = (
                jax.lax.pmax(actor_checksum, axis_name)
                - jax.lax.pmin(actor_checksum, axis_name)
            )
        summary_names = {
            "mean_farm_power_mw",
            "mean_reward",
            "mean_angle_penalty",
            "mean_abs_yaw_deg",
            "mean_abs_yaw_speed_deg_per_s",
            "inlet_interval_delta_rms",
            "episode_reset",
        }
        metrics = {
            **losses,
            **{
                name: jnp.mean(value)
                for name, value in collection_metrics.items()
                if name in summary_names
            },
            "replay_size": collected.replay.size,
            "sac_updates": sac.updates,
            "actor_replica_checksum_delta": replica_delta,
            **(
                {
                    "raw_transitions": {
                        name.removeprefix("raw_"): value
                        for name, value in collection_metrics.items()
                        if name.startswith("raw_")
                    }
                }
                if return_raw_transitions
                else {}
            ),
        }
        return next_training, metrics

    return iteration


def make_exact_svg_probe(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    action_parameterization: str = "per_turbine",
) -> Callable[..., tuple[Array, Array, Array]]:
    """Differentiate a true one-interval value target with respect to action."""

    if config.inlet_mode != "time_planes":
        raise ValueError("exact SVG probe currently requires time-plane inlet")

    def objective(
        normalized_action: Array,
        farm_flow: FlowState,
        observation_frames: Array,
        previous_yaw: Array,
        sac: SACState,
        inlet_planes: Array,
        inlet_plane_start_step: Array,
    ) -> Array:
        yaw = apply_control_action(
            normalized_action,
            previous_yaw,
            case,
            config,
            action_parameterization,
        )
        weights = farm_functions["disk_weights"](yaw)
        marker = _time_plane_marker(case)
        _, _, mean_power, mean_probes, _ = _advance_coupled_interval(
            (marker, farm_flow),
            yaw,
            weights,
            case,
            config,
            precursor_functions,
            farm_functions,
            inlet_planes,
            inlet_plane_start_step,
        )
        frames = _next_frames(observation_frames, mean_probes)
        next_observation = stacked_observation(frames, yaw, case, config)
        reward, _, _ = _reward(mean_power, yaw, config)
        next_action = actor_deterministic(
            sac.actor, next_observation, config.policy_mode
        )
        target_q = jnp.min(
            critic_apply(
                sac.target_critic,
                next_observation,
                next_action,
                config.policy_mode,
            ),
            axis=-1,
        )
        return jnp.mean(reward + np.float32(config.gamma) * target_q)

    rematerialized_objective = jax.checkpoint(objective, prevent_cse=False)

    def probe(
        farm_flow: FlowState,
        observation_frames: Array,
        observation: Array,
        previous_yaw: Array,
        sac: SACState,
        inlet_planes: Array,
        inlet_plane_start_step: Array,
    ) -> tuple[Array, Array, Array]:
        action = actor_deterministic(
            sac.actor, observation, config.policy_mode
        )
        value, action_gradient = jax.value_and_grad(
            rematerialized_objective, argnums=0
        )(
            action,
            farm_flow,
            observation_frames,
            previous_yaw,
            sac,
            inlet_planes,
            inlet_plane_start_step,
        )
        return value, action, action_gradient

    return probe


def initialize_evaluation_from_training(
    training: ContinuousTrainingState,
) -> ContinuousEvaluationState:
    return ContinuousEvaluationState(
        precursor_flow=training.precursor_flow,
        farm_flow=training.farm_flow,
        observation_frames=training.observation_frames,
        observation=training.observation,
        previous_yaw=training.previous_yaw,
        interaction_index=jnp.asarray(0, dtype=jnp.int32),
    )


def make_evaluation_step(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    controller: str,
    actor: PyTree | None = None,
    static_target: tuple[float, ...] = PUBLISHED_STATIC_BO_YAW_DEG,
    sac_action_scale: float = 1.0,
    mirror_sac_policy: bool = False,
    action_parameterization: str = "per_turbine",
    actor_deterministic_fn: Callable[..., Array] | None = None,
    actor_diagnostic_fn: Callable[..., Array] | None = None,
    external_action_provider: (
        Callable[[ContinuousEvaluationState], Array] | None
    ) = None,
    runtime_external_action: bool = False,
) -> Callable[
    [ContinuousEvaluationState],
    tuple[ContinuousEvaluationState, dict[str, Array]],
]:
    if controller not in {"zero", "static_bo", "sac", "external_action"}:
        raise ValueError(f"unknown controller: {controller}")
    if controller == "sac" and actor is None:
        raise ValueError("SAC evaluation requires actor parameters")
    if runtime_external_action and controller != "external_action":
        raise ValueError("runtime external action requires external_action controller")
    if runtime_external_action and external_action_provider is not None:
        raise ValueError("runtime external action cannot also use an action provider")
    if (
        controller == "external_action"
        and external_action_provider is None
        and not runtime_external_action
    ):
        raise ValueError("external-action evaluation requires an action provider")
    if controller != "external_action" and external_action_provider is not None:
        raise ValueError("external action provider requires external_action controller")
    if mirror_sac_policy and case.num_turbines != 3:
        raise ValueError(
            "mirror_sac_policy is only defined for the three-inline layout"
        )
    action_dim = control_action_dim(case, action_parameterization)
    if mirror_sac_policy and action_parameterization != "per_turbine":
        raise ValueError(
            "mirror_sac_policy requires per_turbine actions"
        )
    if controller == "static_bo" and len(static_target) != case.num_turbines:
        raise ValueError(
            "static target length must match the farm turbine count"
        )
    deterministic_policy = (
        actor_deterministic
        if actor_deterministic_fn is None
        else actor_deterministic_fn
    )

    target = jnp.asarray(static_target, dtype=jnp.float32)[None]
    def step(
        state: ContinuousEvaluationState,
        inlet_planes: Array | None = None,
        inlet_plane_start_step: Array | None = None,
        external_normalized_action: Array | None = None,
    ) -> tuple[ContinuousEvaluationState, dict[str, Array]]:
        if controller != "external_action" and external_normalized_action is not None:
            raise ValueError("runtime external action requires external_action controller")
        if controller == "zero":
            yaw = jnp.zeros_like(state.previous_yaw)
        elif controller == "static_bo":
            # The public BO evaluator sets alpha to the optimized angles and
            # applies zero actions, so the target is present throughout reset
            # and evaluation rather than reached through an action-rate ramp.
            yaw = jnp.broadcast_to(target, state.previous_yaw.shape)
        elif controller == "sac":
            actor_observation = state.observation
            if mirror_sac_policy:
                probe_width = (
                    config.frame_stack * case.num_turbines * 7 * 11
                )
                mirrored_probes = state.observation[:, :probe_width].reshape(
                    state.observation.shape[0],
                    config.frame_stack,
                    case.num_turbines,
                    7,
                    11,
                )[:, :, :, ::-1, :]
                actor_observation = jnp.concatenate(
                    (
                        mirrored_probes.reshape(
                            state.observation.shape[0], probe_width
                        ),
                        -state.observation[:, probe_width:],
                    ),
                    axis=-1,
                )
            normalized_action = deterministic_policy(
                actor, actor_observation, config.policy_mode
            )
            policy_diagnostic = (
                actor_diagnostic_fn(actor, actor_observation, config.policy_mode)
                if actor_diagnostic_fn is not None
                else None
            )
            if mirror_sac_policy:
                normalized_action = -normalized_action
            normalized_action = normalized_action * np.float32(
                sac_action_scale
            )
            yaw = apply_control_action(
                normalized_action,
                state.previous_yaw,
                case,
                config,
                action_parameterization,
            )
        else:
            if runtime_external_action:
                if external_normalized_action is None:
                    raise ValueError("runtime external action value is missing")
                normalized_action = jnp.asarray(
                    external_normalized_action, dtype=jnp.float32
                )
                expected_shape = state.previous_yaw.shape[:-1] + (action_dim,)
                if normalized_action.shape != expected_shape:
                    raise ValueError(
                        "runtime external action shape differs from the controller "
                        f"contract: {normalized_action.shape} != {expected_shape}"
                    )
            else:
                normalized_action = resolve_external_normalized_action(
                    state, external_action_provider, action_dim
                )
            yaw = apply_control_action(
                normalized_action,
                state.previous_yaw,
                case,
                config,
                action_parameterization,
            )
        weights = farm_functions["disk_weights"](yaw)
        (
            precursor_flow,
            farm_flow,
            mean_power,
            mean_probes,
            interval_diagnostics,
        ) = _advance_coupled_interval(
            (state.precursor_flow, state.farm_flow),
            yaw,
            weights,
            case,
            config,
            precursor_functions,
            farm_functions,
            inlet_planes,
            inlet_plane_start_step,
        )
        frames = _next_frames(state.observation_frames, mean_probes)
        observation = stacked_observation(frames, yaw, case, config)
        reward, mean_turbine_power_mw, angle_penalty = _reward(
            mean_power, yaw, config
        )
        farm_power_mw = (
            mean_turbine_power_mw * np.float32(mean_power.shape[-1])
        )
        next_state = ContinuousEvaluationState(
            precursor_flow=precursor_flow,
            farm_flow=farm_flow,
            observation_frames=frames,
            observation=observation,
            previous_yaw=yaw,
            interaction_index=state.interaction_index + 1,
        )
        metrics = {
            "farm_power_mw": farm_power_mw,
            "turbine_power_mw": mean_power / np.float32(1.0e6),
            "reward": reward,
            "angle_penalty": angle_penalty,
            "yaw_deg": yaw,
            "inlet_mean_u": interval_diagnostics.inlet_mean_u,
            "inlet_delta_rms": interval_diagnostics.inlet_delta_rms,
        }
        if controller == "sac" and actor_diagnostic_fn is not None:
            metrics["policy_diagnostic"] = policy_diagnostic
        if controller in {"sac", "external_action"}:
            metrics["normalized_action"] = normalized_action
        return next_state, metrics

    return step


def initialize_fixed_controller_pair_evaluation(
    common: ContinuousEvaluationState,
) -> FixedControllerPairEvaluationState:
    """Branch two farm states while retaining one controller-independent inlet."""

    return FixedControllerPairEvaluationState(
        precursor_flow=common.precursor_flow,
        zero_farm_flow=jax.tree.map(jnp.copy, common.farm_flow),
        static_farm_flow=jax.tree.map(jnp.copy, common.farm_flow),
        interaction_index=jnp.asarray(0, dtype=jnp.int32),
    )


def make_paired_fixed_evaluation_step(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    precursor_functions: dict[str, Any],
    farm_functions: dict[str, Any],
    static_target: tuple[float, ...] = PUBLISHED_STATIC_BO_YAW_DEG,
) -> Callable[
    [FixedControllerPairEvaluationState],
    tuple[FixedControllerPairEvaluationState, dict[str, Array]],
]:
    """Evaluate zero/static farms with one shared precursor trajectory.

    The online precursor has no coupling from either farm. Advancing it once
    and feeding the same inlet plane to two independent farm states is therefore
    numerically equivalent to separately advancing two copies initialized from
    the same complete precursor state.
    """

    if config.inlet_mode not in {"online", "replay_volume"}:
        raise ValueError(f"unknown inlet mode: {config.inlet_mode}")
    if config.les_scan_unroll < 1:
        raise ValueError("les_scan_unroll must be positive")
    if len(static_target) != case.num_turbines:
        raise ValueError(
            "static target length must match the farm turbine count"
        )

    zero_yaw = jnp.zeros(
        (case.batch, case.num_turbines), dtype=jnp.float32
    )
    static_yaw = jnp.broadcast_to(
        jnp.asarray(static_target, dtype=jnp.float32)[None], zero_yaw.shape
    )
    zero_weights = farm_functions["disk_weights"](zero_yaw)
    static_weights = farm_functions["disk_weights"](static_yaw)
    if config.inlet_mode == "online":
        precursor_case = precursor_functions["case"]
        precursor_zero_weights = jnp.zeros(
            (
                precursor_case.batch,
                3,
                precursor_case.nx,
                precursor_case.ny,
                precursor_case.nz,
            ),
            dtype=jnp.float32,
        )

    def body(carry: tuple[Any, ...], _: None):
        (
            precursor_flow,
            zero_farm_flow,
            static_farm_flow,
            zero_power_sum,
            static_power_sum,
            inlet_u_sum,
            first_inlet,
            last_inlet,
            step,
        ) = carry
        if config.inlet_mode == "online":
            precursor_flow, _ = precursor_functions["advance"](
                precursor_flow,
                zero_yaw,
                precursor_zero_weights,
                None,
            )
            inlet = _online_precursor_inlet(precursor_flow)
        else:
            inlet = _precursor_inlet(
                precursor_flow.velocity,
                zero_farm_flow.step_index,
                case,
            )
        zero_farm_flow, zero_power = farm_functions["advance"](
            zero_farm_flow, zero_yaw, zero_weights, inlet
        )
        static_farm_flow, static_power = farm_functions["advance"](
            static_farm_flow, static_yaw, static_weights, inlet
        )
        first_inlet = jnp.where(step == 0, inlet, first_inlet)
        return (
            precursor_flow,
            zero_farm_flow,
            static_farm_flow,
            zero_power_sum + zero_power,
            static_power_sum + static_power,
            inlet_u_sum + jnp.mean(inlet[:, 0], axis=(-2, -1)),
            first_inlet,
            inlet,
            step + 1,
        ), None

    def step(state: FixedControllerPairEvaluationState):
        initial_power_sum = jnp.zeros(
            (case.batch, case.num_turbines), dtype=jnp.float32
        )
        initial_inlet_u_sum = jnp.zeros((case.batch,), dtype=jnp.float32)
        initial_inlet = jnp.zeros(
            (case.batch, 3, case.ny, case.nz), dtype=jnp.float32
        )
        (
            precursor_flow,
            zero_farm_flow,
            static_farm_flow,
            zero_power_sum,
            static_power_sum,
            inlet_u_sum,
            first_inlet,
            last_inlet,
            _,
        ), _ = jax.lax.scan(
            body,
            (
                state.precursor_flow,
                state.zero_farm_flow,
                state.static_farm_flow,
                initial_power_sum,
                initial_power_sum,
                initial_inlet_u_sum,
                initial_inlet,
                initial_inlet,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            xs=None,
            length=config.les_steps_per_interaction,
            unroll=config.les_scan_unroll,
        )
        interval_length = np.float32(config.les_steps_per_interaction)
        zero_power = zero_power_sum / interval_length
        static_power = static_power_sum / interval_length
        next_state = FixedControllerPairEvaluationState(
            precursor_flow=precursor_flow,
            zero_farm_flow=zero_farm_flow,
            static_farm_flow=static_farm_flow,
            interaction_index=state.interaction_index + 1,
        )
        metrics = {
            "zero_farm_power_mw": jnp.sum(zero_power, axis=-1)
            / np.float32(1.0e6),
            "static_farm_power_mw": jnp.sum(static_power, axis=-1)
            / np.float32(1.0e6),
            "zero_turbine_power_mw": zero_power / np.float32(1.0e6),
            "static_turbine_power_mw": static_power / np.float32(1.0e6),
            "inlet_mean_u": inlet_u_sum / interval_length,
            "inlet_delta_rms": jnp.sqrt(
                jnp.mean((last_inlet - first_inlet) ** 2, axis=(1, 2, 3))
            ),
        }
        return next_state, metrics

    return step


def save_actor_checkpoint(
    path: Path,
    training: ContinuousTrainingState,
    metadata: dict[str, Any],
    *,
    case: CompactMoleCase | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actor = jax.tree.map(lambda value: np.asarray(value), training.sac.actor)
    checkpoint_metadata = dict(metadata)
    if case is not None:
        expected_contract = layout_contract(case)
        supplied_contract = checkpoint_metadata.get("layout_contract")
        if supplied_contract is not None and supplied_contract != expected_contract:
            raise ValueError("checkpoint metadata layout contract mismatch")
        checkpoint_metadata["layout_contract"] = expected_contract
        action_parameterization = checkpoint_metadata.get(
            "action_parameterization", "per_turbine"
        )
        expected_action_dim = control_action_dim(
            case, action_parameterization
        )
        policy_architecture = checkpoint_metadata.get(
            "policy_architecture", "flat_mlp"
        )
        if policy_architecture in {
            "graph_sac_v2",
            "graph_sac_directional",
        }:
            final_bias = actor["head"]["b"]
            if final_bias.shape != (2,) or expected_action_dim != case.num_turbines:
                raise ValueError(
                    "graph checkpoint must emit one shared-head action per turbine"
                )
        else:
            final_bias = actor[-1]["b"]
            if final_bias.shape != (2 * expected_action_dim,):
                raise ValueError(
                    "checkpoint actor output dimension does not match action "
                    "parameterization"
                )
        checkpoint_metadata["action_dim"] = expected_action_dim
    payload = {
        "schema_version": 1,
        "actor": actor,
        "metadata": checkpoint_metadata,
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_actor_checkpoint(
    path: Path,
    *,
    expected_case: CompactMoleCase | None = None,
    expected_action_parameterization: str | None = None,
) -> tuple[PyTree, dict[str, Any]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    actor = jax.tree.map(jnp.asarray, payload["actor"])
    metadata = payload["metadata"]
    if expected_case is not None:
        saved_action_parameterization = metadata.get(
            "action_parameterization", "per_turbine"
        )
        action_parameterization = (
            saved_action_parameterization
            if expected_action_parameterization is None
            else expected_action_parameterization
        )
        if saved_action_parameterization != action_parameterization:
            raise ValueError(
                "checkpoint action parameterization does not match expected"
            )
        expected_action_dim = control_action_dim(
            expected_case, action_parameterization
        )
        policy_architecture = metadata.get("policy_architecture", "flat_mlp")
        if policy_architecture in {
            "graph_sac_v2",
            "graph_sac_directional",
        }:
            final_bias = actor["head"]["b"]
            if final_bias.shape != (2,) or expected_action_dim != expected_case.num_turbines:
                raise ValueError(
                    "graph checkpoint must emit one shared-head action per turbine"
                )
        else:
            final_bias = actor[-1]["b"]
            if (
                final_bias.ndim != 1
                or final_bias.shape[0] != 2 * expected_action_dim
            ):
                raise ValueError(
                    "checkpoint actor output dimension does not match case"
                )
        saved_contract = metadata.get("layout_contract")
        expected_contract = layout_contract(expected_case)
        if saved_contract is None:
            if hasattr(expected_case, "layout_name"):
                raise ValueError(
                    "checkpoint lacks the layout contract required by a "
                    "scalable farm case"
                )
        elif saved_contract != expected_contract:
            raise ValueError("checkpoint layout contract does not match case")
    return actor, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_metadata(
    case: CompactMoleCase,
    config: ContinuousControlConfig,
    action_parameterization: str = "per_turbine",
) -> dict[str, Any]:
    inlet_semantics = {
        "online": (
            "direct outlet plane from an online periodic precursor LES; "
            "the one-eighth-span shifted-periodic fringe is applied inside "
            "the precursor solver"
        ),
        "replay_volume": (
            "Taylor-shifted planes sampled from one pre-generated periodic "
            "precursor volume; only the farm LES advances during training. "
            "This is a frozen-volume Taylor surrogate, not the public "
            "time-resolved precursor-plane sequence"
        ),
        "time_planes": (
            "time-indexed two-dimensional planes emitted by an independently "
            "advanced precursor LES; only the farm LES advances while each "
            "plane chunk is consumed"
        ),
    }
    return {
        "case": case_metadata(case),
        "control": asdict(config),
        "observation_dim": (
            config.frame_stack * 77 * case.num_turbines
            + case.num_turbines
        ),
        "action_dim": control_action_dim(case, action_parameterization),
        **(
            {
                "action_semantics": (
                    "normalized yaw velocity, integrated over each control "
                    "interval"
                )
            }
            if action_parameterization == "per_turbine"
            else (
                {
                    "action_parameterization": action_parameterization,
                    "action_semantics": (
                        "one normalized absolute yaw target per turbine; the "
                        "physical yaw follows each target under the configured "
                        "yaw-rate and angle limits"
                    ),
                }
                if action_parameterization == "per_turbine_target"
                else {
                    "action_parameterization": action_parameterization,
                    "action_semantics": (
                        "two normalized shared yaw velocities for the first and "
                        "second streamwise columns; the third column is fixed at zero"
                    ),
                }
            )
        ),
        "observation_normalization": {
            "streamwise_probe": "(interval-mean ux - 6 m/s) / 6 m/s",
            "yaw": "yaw_deg * 4 / max_yaw_angle_deg",
        },
        "inlet_semantics": inlet_semantics[config.inlet_mode],
        "limitations": [
            "The paper's 224 evaluation precursor realizations are "
            "unavailable.",
            "The published actor and evaluation logs are available on Zenodo, "
            "but the training critic and exact inlet realizations are not.",
            "The public_torchrl policy mode reproduces the public ReLU, "
            "biased-softplus TanhNormal parameterization, critic input order, "
            "loss reduction, and update ordering. Fixed-batch TorchRL 0.3.1 "
            "one-update parity is recorded separately in the audit outputs.",
            "Training episode boundaries are time-limit truncations, so SAC "
            "bootstraps through them as TorchRL 0.3.1 does.",
        ],
    }


def load_initial_precursor(
    path: Path,
    case: CompactMoleCase,
) -> tuple[Array | FlowState, dict[str, Any]]:
    velocity, metadata = load_precursor(path, case)
    with np.load(path, allow_pickle=False) as archive:
        state_fields = {
            "rhs_previous",
            "rhs_previous_2",
            "filtered_disk_speed",
            "step_index",
        }
        if not state_fields.issubset(archive.files):
            return velocity, {
                **metadata,
                "loaded_state_format": "velocity_only_ab_history_reset",
            }
        rhs_previous = archive["rhs_previous"]
        rhs_previous_2 = archive["rhs_previous_2"]
        filtered_disk_speed = archive["filtered_disk_speed"]
        step_index = archive["step_index"]
        boundary_history_loaded = {
            "bottom_pressure_gradient",
            "open_x_pressure_gradient",
        }.issubset(archive.files)
        if boundary_history_loaded:
            bottom_pressure_gradient = archive["bottom_pressure_gradient"]
            open_x_pressure_gradient = archive["open_x_pressure_gradient"]
        else:
            bottom_pressure_gradient = np.zeros(
                (case.batch, 2, velocity.shape[2], case.nz), dtype=np.float32
            )
            open_x_pressure_gradient = np.zeros(
                (case.batch, 2, 2, case.ny, case.nz), dtype=np.float32
            )
    expected_velocity_shape = tuple(velocity.shape)
    if rhs_previous.shape != expected_velocity_shape:
        raise ValueError("rhs_previous shape does not match precursor velocity")
    if rhs_previous_2.shape != expected_velocity_shape:
        raise ValueError(
            "rhs_previous_2 shape does not match precursor velocity"
        )
    if filtered_disk_speed.shape != (case.batch, case.num_turbines):
        raise ValueError(
            "filtered_disk_speed shape does not match precursor batch"
        )
    if step_index.ndim != 0 or int(step_index) < 2:
        raise ValueError("mature precursor step_index must be a scalar >= 2")
    if bottom_pressure_gradient.shape != (
        case.batch,
        2,
        velocity.shape[2],
        case.nz,
    ):
        raise ValueError("bottom pressure-gradient history shape mismatch")
    if open_x_pressure_gradient.shape != (
        case.batch,
        2,
        2,
        case.ny,
        case.nz,
    ):
        raise ValueError("open-x pressure-gradient history shape mismatch")
    for name, value in (
        ("rhs_previous", rhs_previous),
        ("rhs_previous_2", rhs_previous_2),
        ("filtered_disk_speed", filtered_disk_speed),
        ("bottom_pressure_gradient", bottom_pressure_gradient),
        ("open_x_pressure_gradient", open_x_pressure_gradient),
    ):
        if value.dtype != np.float32:
            raise ValueError(f"{name} must be float32, got {value.dtype}")
    return (
        FlowState(
            velocity=velocity,
            rhs_previous=jnp.asarray(rhs_previous),
            rhs_previous_2=jnp.asarray(rhs_previous_2),
            filtered_disk_speed=jnp.asarray(filtered_disk_speed),
            bottom_pressure_gradient=jnp.asarray(bottom_pressure_gradient),
            open_x_pressure_gradient=jnp.asarray(open_x_pressure_gradient),
            step_index=jnp.asarray(step_index, dtype=jnp.int32),
        ),
        {
            **metadata,
            "loaded_state_format": "complete_ab3_flow_state",
            "boundary_pressure_history_loaded": boundary_history_loaded,
        },
    )


def slice_initial_precursor(
    precursor: Array | FlowState,
    start: int,
    stop: int,
) -> Array | FlowState:
    if not isinstance(precursor, FlowState):
        return precursor[start:stop]

    def slice_batch(value: Array) -> Array:
        if value.ndim > 0 and value.shape[0] == precursor.velocity.shape[0]:
            return jnp.copy(value[start:stop])
        return jnp.copy(value)

    return jax.tree.map(slice_batch, precursor)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(encoded + "\n", encoding="utf-8")
