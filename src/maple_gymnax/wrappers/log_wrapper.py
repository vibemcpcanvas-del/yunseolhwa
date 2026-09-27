"""LogWrapper for episode metrics tracking with auto-reset, without host synchronization."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import chex
import flax
import jax
import jax.numpy as jnp

from maple_gymnax.envs.lotus_phase1 import EnvParams, EnvState


@flax.struct.dataclass
class LogEnvState:
    """State containing wrapped environment state and tracking metrics."""
    env_state: Any
    episode_returns: float = 0.0
    episode_lengths: int = 0
    returned_episode_returns: float = 0.0
    returned_episode_lengths: int = 0


class LogWrapper:
    """Tracks episode returns and lengths branch-free with auto-reset on done.

    When ``done=True``, automatically resets the environment so the agent
    continues collecting experience in ``jax.lax.scan`` training loops
    without dead steps.
    """

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, LogEnvState]:
        obs, state = (
            self._env.reset(key, params)
            if hasattr(self._env, "reset")
            else self._env.reset_env(key, params)
        )
        log_state = LogEnvState(
            env_state=state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
        )
        return obs, log_state

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, LogEnvState]:
        return self.reset(key, params)

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, Dict[str, Any]]:
        step_fn = self._env.step if hasattr(self._env, "step") else self._env.step_env
        key_step, key_reset = jax.random.split(key)

        obs, next_env_state, reward, done, info = step_fn(
            key_step, state.env_state, action, params
        )

        # Episode metric tracking
        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1

        returned_returns = jnp.where(done, new_returns, state.returned_episode_returns)
        returned_lengths = jnp.where(done, new_lengths, state.returned_episode_lengths)

        # Auto-reset: when done, replace env_state and obs with fresh reset
        reset_fn = self._env.reset if hasattr(self._env, "reset") else self._env.reset_env
        reset_obs, reset_state = reset_fn(key_reset, params)

        # Branch-free: select reset state/obs if done, otherwise keep stepping
        auto_env_state = jax.tree.map(
            lambda r, s: jnp.where(done, r, s), reset_state, next_env_state
        )
        auto_obs = jnp.where(done, reset_obs, obs)

        current_returns = jnp.where(done, 0.0, new_returns)
        current_lengths = jnp.where(done, 0, new_lengths)

        next_log_state = LogEnvState(
            env_state=auto_env_state,
            episode_returns=current_returns,
            episode_lengths=current_lengths,
            returned_episode_returns=returned_returns,
            returned_episode_lengths=returned_lengths,
        )
        info["returned_episode_returns"] = returned_returns
        info["returned_episode_lengths"] = returned_lengths
        info["returned_episode"] = done
        return auto_obs, next_log_state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, Dict[str, Any]]:
        return self.step(key, state, action, params)
