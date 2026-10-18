"""Current-policy entropy terms for the historical block-16 SAC target."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_mole_graph_sac_v2 import graph_v2_actor_parameters


def internal_entropy_adjustment(actor, alpha, observations, pre_tanh,
                                clipped, valid, action_bound, graph, gamma=0.99):
    """Return -alpha times internal log-pi at offsets 16, 32 and 48.

    `pre_tanh` and `clipped` must come from the actual behavior sample record;
    the correction reproduces the historical graph_v2_actor_sample convention.
    """
    if action_bound <= 0 or not 0 < gamma <= 1:
        raise ValueError("invalid target configuration")
    if observations.ndim != 3 or observations.shape[1] != 3:
        raise ValueError("expected [batch, 3, observation_dim] internal observations")
    if pre_tanh.shape != clipped.shape or pre_tanh.shape[:2] != observations.shape[:2]:
        raise ValueError("pre-tanh and clip masks differ from internal slots")
    if valid.shape != observations.shape[:2]:
        raise ValueError("invalid three-slot mask")
    batch, slots = observations.shape[:2]
    flat = observations.reshape((batch * slots,) + observations.shape[2:])
    mean, scale = graph_v2_actor_parameters(actor, flat, graph)
    mean = mean.reshape(pre_tanh.shape)
    scale = scale.reshape(pre_tanh.shape)
    real = np.dtype(mean.dtype).type
    limit = real(1.0) - real(np.finfo(real).resolution)
    threshold = jnp.arctanh(limit)
    coordinate = jnp.where(clipped, jnp.sign(pre_tanh) * threshold, pre_tanh)
    standardized = (coordinate - mean) / scale
    gaussian = -0.5 * (standardized**2 + real(math.log(2 * math.pi))) - jnp.log(scale)
    correction = 2 * (real(math.log(2)) - coordinate - jax.nn.softplus(-2 * coordinate)) + real(math.log(action_bound))
    log_probability = jnp.sum(gaussian - correction, axis=-1)
    weights = jnp.asarray([gamma**16, gamma**32, gamma**48], dtype=log_probability.dtype)
    adjusted = -jnp.asarray(alpha, dtype=log_probability.dtype) * jnp.sum(
        jnp.where(valid, log_probability * weights[None, :], 0.0), axis=1)
    return adjusted, log_probability
