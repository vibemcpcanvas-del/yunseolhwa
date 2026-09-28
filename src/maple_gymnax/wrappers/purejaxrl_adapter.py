"""PureJaxRL adapter wrapper for Gymnax environments."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import chex
import jax.numpy as jnp

from maple_gymnax.envs.lotus_phase1 import EnvParams, EnvState


class PureJaxRLAdapterWrapper:
    """Adapts Gymnax step output to standard PureJaxRL 5-tuple (obs, state, r, done, info).

    LotusPhase1Env already returns a 5-tuple, so this wrapper is a transparent
    pass-through that ensures API consistency. If a future environment returns
    a 6-tuple ``(obs, state, reward, terminated, truncated, info)``, the wrapper
    merges ``terminated | truncated`` into a single ``done`` flag.
    """

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
        return obs, next_state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict[str, Any]]:
        return self.step(key, state, action, params)

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        if hasattr(self._env, "reset"):
            return self._env.reset(key, params)
        return self._env.reset_env(key, params)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        return self.reset(key, params)
