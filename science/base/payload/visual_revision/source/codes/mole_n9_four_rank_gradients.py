"""Two 128-sample gradients per rank, one optimizer step after global mean."""

import jax
import jax.numpy as jnp


def aggregate_microbatch_gradients(stacked_gradients, *, axis_name=None):
    """Average two local gradients in FP64, then four ranks, and cast back."""
    leaves = jax.tree.leaves(stacked_gradients)
    if not leaves or any(value.shape[0] != 2 for value in leaves):
        raise ValueError("expected exactly two 128-sample gradient trees")
    with jax.enable_x64():
        wide = jax.tree.map(lambda value: jnp.mean(value.astype(jnp.float64), axis=0),
                            stacked_gradients)
        if axis_name is not None:
            wide = jax.lax.pmean(wide, axis_name)
        return jax.tree.map(lambda value, source: value.astype(source.dtype),
                            wide, jax.tree.map(lambda value: value[0], stacked_gradients))
