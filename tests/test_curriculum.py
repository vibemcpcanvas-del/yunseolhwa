import jax.numpy as jnp
import pytest

from maple_gymnax.curriculum import (
    BASELINES,
    STAGES,
    expand_dense_in,
    expand_dense_out,
    promote,
)


def _baselines(value):
    return {b: value for b in BASELINES}


def test_expand_in_preserves_old_outputs():
    k = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    k2 = expand_dense_in(k, 5)
    x_old = jnp.array([1.0, 2.0, 3.0])
    x_new = jnp.concatenate([x_old, jnp.array([9.0, 9.0])])
    assert k2.shape == (5, 2)
    assert jnp.allclose(x_old @ k, x_new @ k2)


def test_expand_out_zero_init_and_shapes():
    k = jnp.ones((3, 2), dtype=jnp.float32)
    b = jnp.ones((2,), dtype=jnp.float32)
    k2, b2 = expand_dense_out(k, b, 4)
    assert k2.shape == (3, 4) and b2.shape == (4,)
    assert jnp.all(k2[:, 2:] == 0) and jnp.all(b2[2:] == 0)
    assert jnp.all(k2[:, :2] == k)


def test_expand_rejects_shrinking():
    with pytest.raises(ValueError):
        expand_dense_in(jnp.ones((3, 2)), 2)
    with pytest.raises(ValueError):
        expand_dense_out(jnp.ones((3, 2)), jnp.ones((2,)), 1)


def test_promote_requires_margin_over_best_baseline():
    stage = STAGES[0]
    assert promote(stage, 0.95, _baselines(0.10))
    assert not promote(stage, 0.95, _baselines(0.80))


def test_promote_requires_absolute_goal_rate():
    stage = STAGES[0]
    assert not promote(stage, 0.50, _baselines(0.0))


def test_promote_requires_all_baselines():
    with pytest.raises(ValueError):
        promote(STAGES[0], 0.99, {"noop": 0.0})


def test_stage_dims_never_shrink():
    for a, b in zip(STAGES, STAGES[1:]):
        assert b.num_actions >= a.num_actions
        assert b.obs_dim >= a.obs_dim
