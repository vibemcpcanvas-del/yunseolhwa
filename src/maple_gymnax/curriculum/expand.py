"""Zero-init dimension expansion for Dense layers (kernel shape: in x out)."""
from __future__ import annotations

import jax.numpy as jnp


def expand_dense_in(kernel, new_in: int):
    """Append zero rows so new input dims initially have no effect."""
    old_in, out = kernel.shape
    if new_in < old_in:
        raise ValueError("new_in must be >= current input size")
    pad = jnp.zeros((new_in - old_in, out), dtype=kernel.dtype)
    return jnp.concatenate([kernel, pad], axis=0)


def expand_dense_out(kernel, bias, new_out: int):
    """Append zero columns and biases for new output units (e.g. new actions)."""
    in_dim, old_out = kernel.shape
    if new_out < old_out:
        raise ValueError("new_out must be >= current output size")
    extra = new_out - old_out
    new_kernel = jnp.concatenate(
        [kernel, jnp.zeros((in_dim, extra), dtype=kernel.dtype)], axis=1
    )
    new_bias = jnp.concatenate([bias, jnp.zeros((extra,), dtype=bias.dtype)], axis=0)
    return new_kernel, new_bias
