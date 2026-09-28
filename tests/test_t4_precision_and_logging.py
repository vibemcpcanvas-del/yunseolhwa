"""Verification tests for Gate 4: T4 Precision, GPU Enforcement & Logging Defaults."""

import pytest
import jax
import jax.numpy as jnp

from cloud.cloud_manager import resolve_gpu_dtype
from train_ppo import PPOConfig, require_gpu, make_train_step


def test_resolve_gpu_dtype_t4_fallback():
    """Test 1: resolve_gpu_dtype downgrades bfloat16 to float16 only on T4."""
    assert resolve_gpu_dtype("T4", "bfloat16") == "float16"
    assert resolve_gpu_dtype("t4", "bfloat16") == "float16"
    assert resolve_gpu_dtype("A100", "bfloat16") == "bfloat16"
    assert resolve_gpu_dtype("L4", "bfloat16") == "bfloat16"
    assert resolve_gpu_dtype("T4", "float32") == "float32"
    assert resolve_gpu_dtype("T4", "float16") == "float16"


def test_require_gpu_enforcement():
    """Test 2: require_gpu(True) raises RuntimeError if running on non-GPU platform."""
    backend = jax.default_backend()
    if backend != "gpu":
        with pytest.raises(RuntimeError, match=r"GPU backend required"):
            require_gpu(True)
    else:
        require_gpu(True)

    # require_gpu(False) must never raise
    require_gpu(False)


def test_ppo_float16_precision_smoke():
    """Test 3: Smoke test PPO update with dtype='float16' computes finite losses without NaN/Inf."""
    cfg = PPOConfig(
        mode=0,
        num_envs=4,
        num_steps=8,
        num_updates=2,
        num_minibatches=2,
        update_epochs=1,
        dtype="float16",
        param_dtype="float32",
        seed=777,
    )

    init_fn, update_chunk_fn = make_train_step(cfg)
    jitted_init = jax.jit(init_fn)
    jitted_chunk = jax.jit(update_chunk_fn)

    master_rng = jax.random.PRNGKey(cfg.seed)
    rng_init, _ = jax.random.split(master_rng)

    runner_state = jitted_init(rng_init)
    next_runner_state, metrics = jitted_chunk(runner_state, jnp.arange(0, 2))

    # Assert losses and values are valid finite numbers
    for k, val in metrics.items():
        val_arr = jnp.asarray(val)
        assert jnp.all(jnp.isfinite(val_arr)), f"Metric {k} contains NaN or Inf: {val_arr}"

    # Assert model weights are finite
    def _assert_finite(leaf):
        assert jnp.all(jnp.isfinite(leaf)), "Model parameter leaf contains NaN/Inf"

    jax.tree.map(_assert_finite, next_runner_state.train_state.params)
