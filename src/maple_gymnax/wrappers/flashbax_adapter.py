"""JIT-compatible Flashbax flat buffer adapter for Maple Gymnax environments.

Wraps ``flashbax.buffers.make_flat_buffer`` with a thin convenience API
that integrates cleanly into ``jax.lax.scan`` training loops.

Usage::

    from maple_gymnax.wrappers import FlashbaxAdapter

    adapter = FlashbaxAdapter(max_length=100_000, min_length=256, sample_batch_size=64)
    dummy_transition = {"obs": jnp.zeros(130), "action": jnp.array(0), ...}
    buffer_state = adapter.init(dummy_transition)

    # Inside jax.lax.scan body:
    buffer_state = adapter.add(buffer_state, transition)
    can = adapter.can_sample(buffer_state)
    batch = adapter.sample(buffer_state, rng_key)
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple

import chex
import jax
try:
    from flashbax.buffers import make_flat_buffer
except ImportError:
    make_flat_buffer = None


class FlashbaxAdapter:
    """JIT-compatible replay buffer adapter using flashbax's flat buffer.

    All public methods (``init``, ``add``, ``sample``, ``can_sample``) are
    pure functions over JAX arrays and can be freely used inside ``jax.jit``
    and ``jax.lax.scan`` without triggering Python side-effects.
    """

    def __init__(
        self,
        max_length: int = 100_000,
        min_length: int = 1,
        sample_batch_size: int = 256,
        add_sequences: bool = False,
        add_batch_size: int | None = None,
    ):
        # Auto-cap: flashbax requires sample_batch_size <= max_length
        sample_batch_size = min(sample_batch_size, max_length)
        kwargs: Dict[str, Any] = dict(
            max_length=max_length,
            min_length=min_length,
            sample_batch_size=sample_batch_size,
            add_sequences=add_sequences,
        )
        if add_batch_size is not None:
            kwargs["add_batch_size"] = add_batch_size
        self._buffer = make_flat_buffer(**kwargs)

    # ------------------------------------------------------------------
    # Public API — pure JAX, JIT-safe
    # ------------------------------------------------------------------

    def init(self, dummy_transition: Dict[str, Any]) -> Any:
        """Initialise buffer state from a dummy transition pytree.

        Args:
            dummy_transition: A single transition dict (e.g.
                ``{"obs": jnp.zeros(130), "action": jnp.array(0), ...}``).
                Shapes are used to pre-allocate the fixed-size buffer.

        Returns:
            A ``TrajectoryBufferState`` (pure JAX pytree, JIT-compatible).
        """
        return self._buffer.init(dummy_transition)

    def add(self, buffer_state: Any, transition: Dict[str, Any]) -> Any:
        """Append one transition to the buffer (ring-buffer overwrite when full).

        Args:
            buffer_state: Current ``TrajectoryBufferState``.
            transition: Transition dict matching the dummy pytree shape.

        Returns:
            Updated ``TrajectoryBufferState``.
        """
        return self._buffer.add(buffer_state, transition)

    def can_sample(self, buffer_state: Any) -> chex.Array:
        """Whether the buffer contains enough data to sample a batch.

        Returns:
            Boolean scalar JAX array.
        """
        return self._buffer.can_sample(buffer_state)

    def sample(self, buffer_state: Any, key: chex.PRNGKey) -> Any:
        """Uniformly sample a batch of transitions.

        Args:
            buffer_state: Current ``TrajectoryBufferState``.
            key: PRNG key for uniform sampling.

        Returns:
            A ``TransitionSample`` pytree with ``.experience`` containing
            an ``ExperiencePair(first, second)`` of (s_t, s_{t+1}).
        """
        return self._buffer.sample(buffer_state, key)
