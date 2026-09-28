"""Observation flattening wrapper for Maple Gymnax environments."""

from typing import Any, Dict, Tuple, Union
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from maple_gymnax.envs.lotus_phase1 import EnvParams, EnvState, LotusPhase1Env


class FlattenObservationWrapper:
    """Flattens environment observations into a static 1D float32 tensor."""

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict[str, Any]]:
        step_fn = self._env.step if hasattr(self._env, "step") else self._env.step_env
        out = step_fn(key, state, action, params)
        if len(out) == 6:
            obs, next_state, reward, terminated, truncated, info = out
            done = jnp.logical_or(terminated, truncated)
        else:
            obs, next_state, reward, done, info = out
        flat_obs = obs.reshape(-1)
        return flat_obs, next_state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict[str, Any]]:
        return self.step(key, state, action, params)

    def reset(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        reset_fn = self._env.reset if hasattr(self._env, "reset") else self._env.reset_env
        obs, state = reset_fn(key, params)
        return obs.reshape(-1), state

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        return self.reset(key, params)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        orig_space = self._env.observation_space(params)
        flat_dim = int(jnp.prod(jnp.array(orig_space.shape)))
        low = float(jnp.min(orig_space.low))
        high = float(jnp.max(orig_space.high))
        return spaces.Box(low=low, high=high, shape=(flat_dim,), dtype=jnp.float32)

    def get_observation(self, state: Any, params: EnvParams) -> chex.Array:
        env_state = state.env_state if hasattr(state, "env_state") else state
        get_obs_fn = getattr(self._env, "get_observation", getattr(self._env, "get_obs", None))
        if get_obs_fn is None:
            raise AttributeError("Underlying environment has no get_observation or get_obs")
        obs = get_obs_fn(env_state, params)
        return obs.reshape(-1)

    def get_obs(self, state: Any, params: EnvParams) -> chex.Array:
        return self.get_observation(state, params)
