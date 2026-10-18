"""Opt-in Actor information path, admitted only through an explicit training fork."""
import numpy as np


def downstream_pool_weights(nodes, sources, targets):
    reaches = np.zeros((nodes, nodes), dtype=bool)
    reaches[np.asarray(sources, int), np.asarray(targets, int)] = True
    for _ in range(nodes):
        reaches |= reaches.astype(int) @ reaches.astype(int) > 0
    if np.diag(reaches).any():
        raise ValueError('Downstream prototype requires an acyclic wake graph')
    counts = reaches.sum(axis=1, keepdims=True)
    return reaches.astype(np.float32)/np.maximum(counts, 1).astype(np.float32)


def init_feedback(actor):
    import jax.numpy as jnp
    weight = actor['head']['w']
    return {'w': jnp.zeros((weight.shape[0], 1), dtype=weight.dtype)}


def normalize_feedback_features(features):
    import jax.numpy as jnp
    # Clamp before sqrt so zero features have finite derivatives, including masked rows.
    return features/jnp.sqrt(jnp.maximum(jnp.mean(features**2, axis=-1, keepdims=True), 1.))


def feedback_actor_parameters(actor, feedback, observations, graph, mode, frame_stack=1):
    import jax.numpy as jnp
    from jax_mole_graph_sac_v2 import (graph_v2_node_observations, _graph_body_apply,
                                     bound_graph_v2_actor_parameters)
    if mode not in ('self', 'downstream'):
        raise ValueError('Use self-information control or downstream-information trial')
    nodes = graph_v2_node_observations(observations, graph, frame_stack)
    hidden = _graph_body_apply(actor['body'], nodes, graph)
    weights = downstream_pool_weights(graph.num_nodes, graph.edge_sources, graph.edge_targets)
    enabled = jnp.asarray(weights.sum(axis=1) > 0, dtype=hidden.dtype)
    if mode == 'self':
        weights = np.diag((weights.sum(axis=1) > 0).astype(np.float32))
    features = jnp.einsum('ij,...jh->...ih', jnp.asarray(weights), hidden)
    # Identical feature scaling and enabled rows in both arms isolate information routing.
    features = normalize_feedback_features(features)
    correction = (features @ feedback['w'])*enabled[:, None]
    output = hidden @ actor['head']['w'] + actor['head']['b']
    raw_mean, raw_scale = jnp.split(output, 2, axis=-1)
    mean, scale = bound_graph_v2_actor_parameters(raw_mean+correction, raw_scale,
        scale_lower_bound=.01 if 'neutral_scale_contract' in actor else .1)
    return mean.squeeze(-1), scale.squeeze(-1)
