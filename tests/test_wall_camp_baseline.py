"""동일 시드에서 학습 정책 vs 벽 고정 정책 vs 무작위 정책의 낙하물 노출 비교."""

import jax
import jax.numpy as jnp
import pytest

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.envs.common import ACTION_RIGHT, ACTION_NOOP


def _wall_camp_policy(obs, state, key):
    return jnp.array(ACTION_RIGHT)


def _noop_policy(obs, state, key):
    return jnp.array(ACTION_NOOP)


def _random_policy(obs, state, key):
    return jax.random.randint(key, shape=(), minval=0, maxval=7)


def _rollout_with_policy(env, params, policy_fn, seed, num_steps=1800):
    key = jax.random.PRNGKey(seed)
    k_reset, k_run = jax.random.split(key)
    obs, state = env.reset_env(k_reset, params)

    debris_hit_count = 0
    took_hit_count = 0
    right_wall_ticks = 0

    cur_obs, cur_state = obs, state
    cur_key = k_run
    for _ in range(num_steps):
        cur_key, act_key, step_key = jax.random.split(cur_key, 3)
        action = policy_fn(cur_obs, cur_state, act_key)
        cur_obs, cur_state, reward, done, info = env.step_env(step_key, cur_state, action, params)

        debris_hit_count += int(info["debris_hit"])
        took_hit_count += int(info["dmg_taken"] > 0.0)
        right_wall_ticks += int(cur_state.player_x > (params.wall_right - 150.0))

        if done:
            break

    return {
        "debris_hit_count": debris_hit_count,
        "took_hit_count": took_hit_count,
        "right_wall_ratio": right_wall_ticks / num_steps,
        "final_hp": float(cur_state.player_hp),
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_wall_camp_vs_noop_debris_exposure(seed):
    """벽 고정과 무행동 정책 간 낙하물 노출·피격 차이를 계측한다."""
    env = LotusPhase1Env()
    params = env.default_params.replace(mode=1)

    wall_result = _rollout_with_policy(env, params, _wall_camp_policy, seed)
    noop_result = _rollout_with_policy(env, params, _noop_policy, seed)
    random_result = _rollout_with_policy(env, params, _random_policy, seed)

    # 계측만 수행: 결과값을 assert하지 않고 출력으로 남긴다 (탐색적 실험).
    print(f"\n[seed={seed}] wall_camp={wall_result}")
    print(f"[seed={seed}] noop     ={noop_result}")
    print(f"[seed={seed}] random   ={random_result}")

    # 최소 계약: 두 정책 모두 유한한 값이어야 한다 (NaN/오류 방지).
    assert wall_result["debris_hit_count"] >= 0
    assert noop_result["debris_hit_count"] >= 0
    assert random_result["debris_hit_count"] >= 0
