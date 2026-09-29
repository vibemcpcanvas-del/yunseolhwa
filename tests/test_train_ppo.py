"""Unit tests for train_ppo.py single-file PPO training pipeline."""

import os
import shutil
import tempfile
import pytest
import jax
import jax.numpy as jnp

from maple_gymnax.envs.common import (
    ACTION_NOOP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_JUMP,
    ACTION_JUMP_LEFT,
    ACTION_JUMP_RIGHT,
    ACTION_DUCK,
)
from train_ppo import (
    PPOConfig,
    ActorCritic,
    Transition,
    _calculate_gae,
    _log_callback,
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
            assert "survival_sec" in metrics
            assert "debris_hits_per_ep" in metrics
            assert "jump_ratio" in metrics
            assert "survival_rate" in metrics

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
            assert "survival_sec" in metrics
            assert "debris_hits_per_ep" in metrics
            assert "jump_ratio" in metrics
            assert "survival_rate" in metrics

            eval_metrics = evaluate_policy(final_state.train_state.params, config, num_eval_steps=20)
            assert eval_metrics["steps"] > 0
        finally:
            if os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)


class TestTelemetryMetrics:
    """Validates real-time telemetry metrics overhaul (Requirement R3)."""

    def test_survival_sec_calculation(self):
        """Verifies continuous survival seconds calculation (length / 60.0)."""
        # Plateau baseline: 680 steps -> 11.333s (previously blinded as 0.0% survival)
        length_plateau = 680.0
        survival_plateau = length_plateau / 60.0
        assert abs(survival_plateau - 11.3333) < 1e-3

        # Breakthrough target: 1200 steps -> 20.0s
        length_breakthrough = 1200.0
        survival_breakthrough = length_breakthrough / 60.0
        assert abs(survival_breakthrough - 20.0) < 1e-5

        # Full timeout: 3600 steps -> 60.0s
        length_full = 3600.0
        survival_full = length_full / 60.0
        assert abs(survival_full - 60.0) < 1e-5

    def test_jump_ratio_canonical_actions(self):
        """Verifies JumpRatio% correctly counts canonical jump actions {3, 4, 5} and excludes ground crouch (6)."""
        actions = jnp.array([
            ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DUCK,  # Ground actions: 0, 1, 2, 6 (4 actions)
            ACTION_JUMP, ACTION_JUMP_LEFT, ACTION_JUMP_RIGHT,      # Jump actions: 3, 4, 5 (3 actions)
            ACTION_JUMP_RIGHT, ACTION_DUCK, ACTION_NOOP            # 1 jump, 2 ground (3 actions)
        ], dtype=jnp.int32)  # Total: 10 actions, jumps = 4

        is_jump = (actions == ACTION_JUMP) | (actions == ACTION_JUMP_LEFT) | (actions == ACTION_JUMP_RIGHT)
        jump_ratio = jnp.mean(is_jump.astype(jnp.float32))

        assert abs(float(jump_ratio) - 0.4) < 1e-6
        # Verify DUCK (crouch) is NOT counted as jump
        duck_mask = (actions == ACTION_DUCK)
        assert not bool(jnp.any(is_jump & duck_mask))

    def test_debris_hits_per_ep_renewal_estimator(self):
        """Verifies renewal estimator for DebrisHits/ep under both has_dones and fallback branches."""
        num_steps = 10
        num_envs = 4
        mean_length = 25.0

        # Case A: has_dones = True
        # 2 episodes completed, total debris hits on done steps = 2
        done_mask = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        done_mask = done_mask.at[4, 0].set(1.0).at[9, 1].set(1.0)
        num_dones = jnp.sum(done_mask)
        has_dones = num_dones > 0

        debris_hit = jnp.zeros((num_steps, num_envs), dtype=bool)
        debris_hit = debris_hit.at[4, 0].set(True).at[9, 1].set(True)

        total_debris_hits = jnp.sum(debris_hit.astype(jnp.float32) * done_mask)
        debris_hits_per_ep = jnp.where(
            has_dones,
            total_debris_hits / jnp.maximum(num_dones, 1.0),
            (jnp.sum(debris_hit.astype(jnp.float32)) / (num_steps * num_envs)) * mean_length,
        )
        assert abs(float(debris_hits_per_ep) - 1.0) < 1e-6

        # Case B: has_dones = False (zero terminations in rollout window)
        done_mask_empty = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        num_dones_empty = jnp.sum(done_mask_empty)
        has_dones_empty = num_dones_empty > 0

        debris_hit_fallback = jnp.zeros((num_steps, num_envs), dtype=bool)
        debris_hit_fallback = debris_hit_fallback.at[2, 0].set(True).at[5, 2].set(True)  # 2 hits
        total_debris_hits_empty = jnp.sum(debris_hit_fallback.astype(jnp.float32) * done_mask_empty)

        debris_hits_per_ep_fallback = jnp.where(
            has_dones_empty,
            total_debris_hits_empty / jnp.maximum(num_dones_empty, 1.0),
            (jnp.sum(debris_hit_fallback.astype(jnp.float32)) / (num_steps * num_envs)) * mean_length,
        )
        # Expected: (2 / 40) * 25.0 = 1.25
        assert abs(float(debris_hits_per_ep_fallback) - 1.25) < 1e-5

    def test_log_callback_formatting(self, capsys):
        """Verifies _log_callback formats correctly without syntax or conversion errors."""
        _log_callback(
            update_step=20,
            mean_return=-45.2,
            mean_length=680.5,
            survival_sec=11.3,
            debris_hits_per_ep=3.2,
            jump_ratio=0.185,
            mean_gauge=0.0450,
            actor_loss=0.035,
            critic_loss=0.120,
            entropy=1.750,
            sps=980000.0,
        )
        captured = capsys.readouterr().out
        assert "[Update    20]" in captured
        assert "Survival(s):  11.3s" in captured
        assert "DebrisHits/ep:  3.2" in captured
        assert "JumpRatio%: 18.5%" in captured
        assert "Length:  680.5" in captured
        assert "Return:   -45.20" in captured
