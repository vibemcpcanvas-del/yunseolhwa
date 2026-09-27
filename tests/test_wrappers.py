"""Unit tests for Maple Gymnax reinforcement learning wrappers."""

import pytest
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams, EnvState
from maple_gymnax.wrappers import (
    FlattenObservationWrapper,
    PureJaxRLAdapterWrapper,
    LogWrapper,
    LogEnvState,
    RolloutRunner,
    FlashbaxAdapter,
)


class TestFlattenObservationWrapper:
    """Validates 1D flattened observation tensor transformations."""

    def test_flatten_shape_and_dtype(self):
        env = LotusPhase1Env()
        wrapped = FlattenObservationWrapper(env)
        params = EnvParams()
        key = jax.random.PRNGKey(42)

        obs, state = wrapped.reset(key, params)
        assert obs.ndim == 1
        assert obs.shape == (130,)
        assert obs.dtype == jnp.float32

        k_step, k_act = jax.random.split(key)
        action = 0
        next_obs, next_state, reward, done, info = wrapped.step(k_step, state, action, params)
        assert next_obs.ndim == 1
        assert next_obs.shape == (130,)
        assert next_obs.dtype == jnp.float32

    def test_observation_space_query(self):
        env = LotusPhase1Env()
        wrapped = FlattenObservationWrapper(env)
        params = EnvParams()
        space = wrapped.observation_space(params)
        assert isinstance(space, spaces.Box)
        assert space.shape == (130,)

    def test_jit_and_vmap_compatibility(self):
        env = LotusPhase1Env()
        wrapped = FlattenObservationWrapper(env)
        params = EnvParams()

        @jax.jit
        def _step_jit(k, s, a):
            return wrapped.step(k, s, a, params)

        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key, params)
        next_obs, _, _, _, _ = _step_jit(key, state, 1)
        assert next_obs.shape == (130,)

        batch_size = 128
        keys = jax.random.split(key, batch_size)
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        v_reset = jax.jit(jax.vmap(wrapped.reset, in_axes=(0, None)))
        v_step = jax.jit(jax.vmap(wrapped.step, in_axes=(0, 0, 0, None)))

        b_obs, b_states = v_reset(keys, params)
        assert b_obs.shape == (batch_size, 130)

        b_next_obs, _, _, _, _ = v_step(keys, b_states, actions, params)
        assert b_next_obs.shape == (batch_size, 130)


class TestPureJaxRLAdapterWrapper:
    """Validates Gymnax to PureJaxRL 5-tuple interface adaptation."""

    def test_purejaxrl_5_tuple_contract(self):
        env = LotusPhase1Env()
        wrapped = PureJaxRLAdapterWrapper(env)
        params = EnvParams()
        key = jax.random.PRNGKey(101)

        reset_res = wrapped.reset(key, params)
        assert len(reset_res) == 2
        obs, state = reset_res

        step_res = wrapped.step(key, state, 2, params)
        assert len(step_res) == 5
        next_obs, next_state, reward, done, info = step_res
        assert isinstance(reward, (float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))
        assert isinstance(info, dict)


class TestLogWrapper:
    """Validates host-synchronization-free episode logging."""

    def test_log_state_initialization_and_accumulation(self):
        env = LotusPhase1Env()
        wrapped = LogWrapper(env)
        # Safe parameters avoiding instant lethal laser contact
        params = EnvParams(laser_damage=0.0, debris_spawn_prob=0.0)
        key = jax.random.PRNGKey(202)

        obs, log_state = wrapped.reset(key, params)
        assert isinstance(log_state, LogEnvState)
        assert log_state.episode_returns == 0.0
        assert log_state.episode_lengths == 0
        assert log_state.returned_episode_returns == 0.0
        assert log_state.returned_episode_lengths == 0

        # Step 10 times and verify length increments without death
        curr_state = log_state
        for _ in range(10):
            k_step, key = jax.random.split(key)
            obs, curr_state, r, d, info = wrapped.step(k_step, curr_state, 0, params)

        assert int(curr_state.episode_lengths) == 10
        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info


class TestRolloutRunner:
    """Validates jax.lax.scan high-throughput trajectory rollouts."""

    def test_unroll_scan_trajectory(self):
        env = LotusPhase1Env()
        wrapped = FlattenObservationWrapper(env)
        params = EnvParams()
        runner = RolloutRunner(wrapped, params)

        rng = jax.random.PRNGKey(303)
        k_reset, k_run = jax.random.split(rng)
        _, init_state = wrapped.reset(k_reset, params)

        num_steps = 60
        final_state, traj = jax.jit(runner.run, static_argnums=(2,))(k_run, init_state, num_steps)

        assert traj["obs"].shape == (num_steps, 130)
        assert traj["action"].shape == (num_steps,)
        assert traj["reward"].shape == (num_steps,)
        assert traj["done"].shape == (num_steps,)
        assert final_state.time == num_steps

    def test_rollout_runner_first_obs_matches_initial_state(self):
        """RolloutRunner의 첫 관측이 전달된 initial_state에서 파생되어야 한다."""
        env = LotusPhase1Env()
        params = EnvParams()
        key = jax.random.PRNGKey(0)
        _, distinct_state = env.reset_env(key, params)

        # initial_state를 명백히 구분되는 값으로 설정
        custom_state = distinct_state.replace(player_x=999.0, player_hp=42.0)
        expected_obs = env.get_obs(custom_state, params)

        runner = RolloutRunner(env, params)
        diff_key = jax.random.PRNGKey(777)  # run()에 전달되는 키는 initial_state와 무관해야 함
        _, traj = runner.run(diff_key, custom_state, num_steps=1)

        assert jnp.allclose(traj["obs"][0], expected_obs), (
            "첫 관측이 initial_state가 아닌 별도 reset에서 생성되었습니다."
        )


class TestFlashbaxAdapter:
    """Validates Flashbax buffer storage and uniform sampling."""

    def test_buffer_add_and_sample(self):
        adapter = FlashbaxAdapter(max_length=100, min_length=1, sample_batch_size=8)
        rng = jax.random.PRNGKey(404)

        # Init with dummy transition
        dummy = {
            "obs": jnp.zeros((130,), dtype=jnp.float32),
            "action": jnp.int32(0),
            "reward": jnp.float32(0.0),
            "done": jnp.array(False),
        }
        buffer_state = adapter.init(dummy)

        for i in range(20):
            k_obs, rng = jax.random.split(rng)
            buffer_state = adapter.add(buffer_state, {
                "obs": jax.random.normal(k_obs, shape=(130,)),
                "action": jnp.int32(i % 7),
                "reward": jnp.float32(1.0),
                "done": jnp.array(False),
            })

        assert adapter.can_sample(buffer_state)
        k_sample, rng = jax.random.split(rng)
        batch = adapter.sample(buffer_state, k_sample)
        # flashbax returns TransitionSample with .experience (ExperiencePair)
        first = batch.experience.first
        assert first["obs"].shape == (8, 130)
        assert first["action"].shape == (8,)
        assert first["reward"].shape == (8,)
        assert first["done"].shape == (8,)
