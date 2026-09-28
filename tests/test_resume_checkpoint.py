"""Verification tests for Gate 1: Resumable Checkpoint State and Config Fingerprint."""

import os
import shutil
import tempfile
import pytest
import jax
import jax.numpy as jnp

from maple_gymnax.resume_state import save_resume_state, load_resume_state, HARD_KEYS
from train_ppo import PPOConfig, make_train_step, save_checkpoint_orbax


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_resume_ckpt_")
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


def test_resume_state_exact_continuation(tmp_dir):
    """Test 1: 8 continuous updates == 4 updates -> save -> restore -> 4 updates."""
    cfg_base = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        num_updates=8,
        num_minibatches=2,
        update_epochs=1,
        seed=123,
        checkpoint_dir=tmp_dir,
        checkpoint_interval=4,
    )

    init_fn, update_chunk_fn = make_train_step(cfg_base)
    jitted_init = jax.jit(init_fn)
    jitted_chunk = jax.jit(update_chunk_fn)

    master_rng = jax.random.PRNGKey(cfg_base.seed)
    rng_init, _ = jax.random.split(master_rng)

    # 1. Run Continuous 8 updates (as two 4-step chunks)
    state_cont = jitted_init(rng_init)
    state_cont, _ = jitted_chunk(state_cont, jnp.arange(0, 4))
    state_cont, _ = jitted_chunk(state_cont, jnp.arange(4, 8))

    # 2. Run 4 updates, save, restore, then run remaining 4 updates
    state_split = jitted_init(rng_init)
    state_split, _ = jitted_chunk(state_split, jnp.arange(0, 4))

    step_dir = os.path.join(tmp_dir, "mode_0", "step_4")
    save_checkpoint_orbax(state_split.train_state.params, cfg_base, 4, runner_state=state_split)

    # Restore in fresh state
    dummy_template = jitted_init(rng_init)
    restored_state, restored_step, _ = load_resume_state(
        step_dir,
        cfg_base,
        target_runner_state=dummy_template,
    )

    assert restored_step == 4

    # Run remaining 4 updates
    state_resumed, _ = jitted_chunk(restored_state, jnp.arange(4, 8))

    # 3. Assert params, last_obs, and env_state match exactly or within numerical precision
    def _assert_allclose(a, b, name=""):
        diff = jnp.max(jnp.abs(a - b))
        assert diff < 1e-4, f"Mismatch in {name}: max diff {diff}"

    jax.tree.map(_assert_allclose, state_cont.train_state.params, state_resumed.train_state.params)
    jax.tree.map(_assert_allclose, state_cont.train_state.opt_state, state_resumed.train_state.opt_state)
    jax.tree.map(_assert_allclose, state_cont.last_obs, state_resumed.last_obs)
    jax.tree.map(_assert_allclose, state_cont.env_state.env_state.player_hp, state_resumed.env_state.env_state.player_hp)


def test_resume_state_hard_keys_mismatch(tmp_dir):
    """Test 2: Changing HARD_KEYS must raise ValueError."""
    cfg = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        num_updates=4,
        num_minibatches=2,
        update_epochs=1,
        seed=42,
        checkpoint_dir=tmp_dir,
    )

    init_fn, _ = make_train_step(cfg)
    runner_state = init_fn(jax.random.PRNGKey(42))

    step_dir = os.path.join(tmp_dir, "step_4")
    save_resume_state(step_dir, runner_state, 4, cfg)

    # Test mismatch for each hard key
    mismatched_cfgs = [
        PPOConfig(mode=1, num_envs=4, num_steps=8, num_minibatches=2, update_epochs=1),
        PPOConfig(mode=0, num_envs=8, num_steps=8, num_minibatches=2, update_epochs=1),
        PPOConfig(mode=0, num_envs=4, num_steps=16, num_minibatches=2, update_epochs=1),
        PPOConfig(mode=0, num_envs=4, num_steps=8, gamma=0.95, num_minibatches=2, update_epochs=1),
    ]

    for bad_cfg in mismatched_cfgs:
        with pytest.raises(ValueError, match=r"HARD_KEYS mismatch"):
            load_resume_state(step_dir, bad_cfg, target_runner_state=runner_state)


def test_resume_state_precision_migration(tmp_dir):
    """Test 3: Precision downcast requires allow_precision_loss flag."""
    cfg_f32 = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        dtype="float32",
        checkpoint_dir=tmp_dir,
    )

    init_fn, _ = make_train_step(cfg_f32)
    runner_state = init_fn(jax.random.PRNGKey(42))

    step_dir = os.path.join(tmp_dir, "step_4")
    save_resume_state(step_dir, runner_state, 4, cfg_f32)

    cfg_f16 = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        dtype="float16",
        checkpoint_dir=tmp_dir,
    )

    # Without allow_precision_loss -> raises ValueError
    with pytest.raises(ValueError, match=r"requires --allow-precision-loss"):
        load_resume_state(step_dir, cfg_f16, target_runner_state=runner_state, allow_precision_loss=False)

    # With allow_precision_loss -> succeeds
    restored_state, step, _ = load_resume_state(
        step_dir, cfg_f16, target_runner_state=runner_state, allow_precision_loss=True
    )
    assert step == 4


def test_resume_state_num_updates_extension(tmp_dir):
    """Test 4: Extending num_updates logs extension and loads cleanly."""
    cfg_short = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        num_updates=100,
        checkpoint_dir=tmp_dir,
    )

    init_fn, _ = make_train_step(cfg_short)
    runner_state = init_fn(jax.random.PRNGKey(42))

    step_dir = os.path.join(tmp_dir, "step_50")
    save_resume_state(step_dir, runner_state, 50, cfg_short)

    cfg_long = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        num_updates=300,
        checkpoint_dir=tmp_dir,
    )

    restored_state, step, loaded_cfg = load_resume_state(
        step_dir, cfg_long, target_runner_state=runner_state
    )
    assert step == 50
    assert loaded_cfg.num_updates == 300
