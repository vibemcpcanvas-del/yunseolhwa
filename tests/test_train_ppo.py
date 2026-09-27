"""Unit tests for train_ppo.py single-file PPO training pipeline."""

import os
import shutil
import tempfile
import pytest
import jax
import jax.numpy as jnp

from train_ppo import (
    PPOConfig,
    ActorCritic,
    Transition,
    _calculate_gae,
    make_train,
    save_checkpoint_orbax,
    load_checkpoint_orbax,
    evaluate_policy,
)


class TestPPONetwork:
    """Verifies ActorCritic neural network construction and forward propagation."""

    def test_actor_critic_shapes(self):
        obs_dim = 130
        action_dim = 7
        network = ActorCritic(action_dim=action_dim)
        rng = jax.random.PRNGKey(0)

        dummy_obs = jnp.zeros((16, obs_dim), dtype=jnp.float32)
        params = network.init(rng, dummy_obs)

        logits, value = network.apply(params, dummy_obs)
        assert logits.shape == (16, action_dim)
        assert value.shape == (16,)
        assert logits.dtype == jnp.float32
        assert value.dtype == jnp.float32


class TestGAECalculation:
    """Validates Generalized Advantage Estimation mathematical properties."""

    def test_gae_scan_shapes(self):
        num_steps = 8
        num_envs = 4
        rng = jax.random.PRNGKey(1)

        dummy_trans = Transition(
            done=jnp.zeros((num_steps, num_envs), dtype=bool),
            action=jnp.zeros((num_steps, num_envs), dtype=jnp.int32),
            value=jnp.ones((num_steps, num_envs), dtype=jnp.float32),
            reward=jnp.ones((num_steps, num_envs), dtype=jnp.float32),
            log_prob=jnp.zeros((num_steps, num_envs), dtype=jnp.float32),
            obs=jnp.zeros((num_steps, num_envs, 130), dtype=jnp.float32),
            info={},
        )
        last_val = jnp.ones(num_envs, dtype=jnp.float32)

        advantages, targets = _calculate_gae(dummy_trans, last_val, gamma=0.99, gae_lambda=0.95)
        assert advantages.shape == (num_steps, num_envs)
        assert targets.shape == (num_steps, num_envs)


class TestPPOTrainScan:
    """Validates end-to-end PPO single JIT compilation and checkpoint persistence."""

    def test_smoke_train_mode0_classic(self):
        config = PPOConfig(
            mode=0,
            num_envs=8,
            num_steps=8,
            num_updates=2,
            num_minibatches=2,
            update_epochs=2,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        try:
            train_fn = jax.jit(make_train(config))
            rng = jax.random.PRNGKey(42)
            final_state, metrics = train_fn(rng)

            assert final_state.train_state.params is not None
            assert "mean_return" in metrics

            # Checkpoint save and load
            ckpt_path = save_checkpoint_orbax(final_state.train_state.params, config, step=2)
            assert os.path.exists(ckpt_path)
            restored = load_checkpoint_orbax(ckpt_path)
            assert restored is not None

            # Evaluation via RolloutRunner
            eval_metrics = evaluate_policy(final_state.train_state.params, config, num_eval_steps=20)
            assert "reward" in eval_metrics
            assert "steps" in eval_metrics
        finally:
            if os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)

    def test_smoke_train_mode1_remastered(self):
        config = PPOConfig(
            mode=1,
            num_envs=8,
            num_steps=8,
            num_updates=2,
            num_minibatches=2,
            update_epochs=2,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        try:
            train_fn = jax.jit(make_train(config))
            rng = jax.random.PRNGKey(99)
            final_state, metrics = train_fn(rng)

            assert final_state.train_state.params is not None
            assert "mean_gauge" in metrics

            eval_metrics = evaluate_policy(final_state.train_state.params, config, num_eval_steps=20)
            assert eval_metrics["steps"] > 0
        finally:
            if os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)
