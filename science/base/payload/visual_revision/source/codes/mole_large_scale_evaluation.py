"""Learner-free initialization and Graph-SAC evaluation for canonical layouts."""
import jax
import jax.numpy as jnp

from jax_mole_compact_sac_benchmark import build_flow_functions, make_initial_flow
from jax_mole_continuous_sac import (
    ContinuousEvaluationState, _apply_velocity_boundaries, _time_plane_marker,
    stacked_observation, make_evaluation_step,
)
from jax_mole_graph_sac_v2 import graph_v2_actor_deterministic


def validate_actor_metadata(metadata, case, config, required_interactions):
    """Reject training contracts this frozen deterministic evaluator cannot run."""
    if metadata.get('policy_architecture') != 'graph_sac_v2':
        raise ValueError('frozen large-scale Graph-SAC v2 Actor required')
    if metadata.get('actual_global_interactions') != required_interactions:
        raise ValueError('Actor is not the predeclared final fixed-budget controller')
    if metadata.get('reflection_equivariant_actor', False):
        raise ValueError('reflection-equivariant Actor requires its evaluation wrapper')
    if metadata.get('direction_condition', False) or metadata.get('steering_branch_condition', False):
        raise ValueError('conditioned Actor requires its observation/evaluation wrapper')
    expected_width = case.num_turbines * (77 * config.frame_stack + 1)
    if metadata.get('learner_observation_dim') != expected_width:
        raise ValueError('Actor observation width does not match the fixed frame-stack contract')
    if metadata.get('compact_solver') != case.compact_solver:
        raise ValueError('Actor training compact solver differs from evaluation')


def initialize_evaluation(case, config, initial_inlet, seed):
    if config.inlet_mode != 'time_planes':
        raise ValueError('large-scale frozen evaluation requires cached time planes')
    if initial_inlet.shape != (case.batch, 3, case.ny, case.nz):
        raise ValueError('inlet shape does not match canonical layout')
    farm = build_flow_functions(case, periodic_x=False, include_turbines=True)
    flow = make_initial_flow(case, farm, seed + 1)
    velocity = _apply_velocity_boundaries(flow.velocity, initial_inlet, periodic_x=False)
    flow = flow._replace(velocity=velocity)
    yaw = jnp.zeros((case.batch, case.num_turbines), dtype=jnp.float32)
    probes = farm['probes'](velocity).reshape(case.batch, -1)
    frames = jnp.repeat(probes[:, None], config.frame_stack, axis=1)
    state = ContinuousEvaluationState(
        _time_plane_marker(case), flow, frames,
        stacked_observation(frames, yaw, case, config), yaw,
        jnp.asarray(0, dtype=jnp.int32))
    return state, {'case': case, 'external_time_planes': True}, farm


def scoring_boundary(state):
    """Match the existing trainer's explicit post-burnin zero-cache boundary."""
    return state._replace(observation_frames=jnp.zeros_like(state.observation_frames),
                          observation=jnp.zeros_like(state.observation),
                          previous_yaw=jnp.zeros_like(state.previous_yaw),
                          interaction_index=jnp.zeros_like(state.interaction_index))


def make_step(case, config, precursor, farm, *, actor=None, graph=None):
    if actor is None:
        return make_evaluation_step(case, config, precursor, farm, 'zero')
    if graph is None or graph.num_nodes != case.num_turbines:
        raise ValueError('matching Graph-SAC v2 graph required')
    policy = lambda parameters, observation, mode: graph_v2_actor_deterministic(
        parameters, observation, graph, config.frame_stack)
    return make_evaluation_step(case, config, precursor, farm, 'sac', actor=actor,
                                actor_deterministic_fn=policy,
                                action_parameterization='per_turbine')


def make_scan(step, interactions):
    if interactions < 1:
        raise ValueError('positive scan length required')
    return jax.jit(lambda state, planes, start: jax.lax.scan(
        lambda current, _: step(current, planes, start), state, None,
        length=interactions))
