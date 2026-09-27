"""Batched episodic rollout runner leveraging jax.vmap and jax.lax.scan."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import chex
import jax
import jax.numpy as jnp

from maple_gymnax.envs.lotus_phase1 import EnvParams


# Default random uniform policy for benchmarking / debugging
def _random_policy(obs: chex.Array, state: Any, key: chex.PRNGKey) -> chex.Array:
    """Uniform random action over Discrete(7)."""
    return jax.random.randint(key, shape=(), minval=0, maxval=7)


class RolloutRunner:
    """High-speed batched episode runner leveraging jax.vmap and jax.lax.scan.

    Args:
        env: A Gymnax-compatible environment (must have ``step`` or ``step_env``).
        params: Immutable environment parameters.
        policy_fn: Optional callable ``(obs, state, key) -> action``.
            Defaults to uniform random policy.
    """

    def __init__(
        self,
        env: Any,
        params: EnvParams,
        policy_fn: Optional[Callable[..., chex.Array]] = None,
    ):
        self.env = env
        self.params = params
        self.policy_fn = policy_fn or _random_policy

    def run(
        self,
        rng: chex.PRNGKey,
        initial_state: Any,
        num_steps: int,
    ) -> Tuple[Any, Dict[str, chex.Array]]:
        """Unrolls a trajectory of length num_steps using jax.lax.scan without host synchronization."""
        step_fn = self.env.step if hasattr(self.env, "step") else self.env.step_env
        policy = self.policy_fn

        def _step(carry, _):
            key, state, prev_obs = carry
            key, step_key, act_key = jax.random.split(key, 3)
            action = policy(prev_obs, state, act_key)
            obs, next_state, reward, done, info = step_fn(step_key, state, action, self.params)
            transition = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
            }
            return (key, next_state, obs), transition

        # Get initial observation for first policy call
        reset_fn = self.env.reset if hasattr(self.env, "reset") else self.env.reset_env
        init_obs = reset_fn(rng, self.params)[0]

        (final_key, final_state, _), traj = jax.lax.scan(
            _step, (rng, initial_state, init_obs), None, length=num_steps
        )
        return final_state, traj
