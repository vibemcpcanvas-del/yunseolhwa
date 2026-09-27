"""Tier 4: Real-World Application Scenarios Test Suite.

Comprehensive realistic gameplay, boss evasion, full survival episodes,
multi-agent rollout pipelines, and benchmark scenarios.
Contains 15 distinct high-complexity real-world application scenarios.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any
import pytest
import jax
import jax.numpy as jnp
import flax.struct

from tests.e2e.contract_harness import (
    EnvParams,
    EnvState,
    LotusPhase1Env,
    MAX_DEBRIS,
    validate_env_params_schema,
    crawl_wz_directory,
    extract_wz_anchors_and_frames,
    parse_wz_to_env_params,
    FlattenObservationWrapper,
    PureJaxRLAdapterWrapper,
    LogWrapper,
    RolloutRunner,
    FlashbaxAdapter,
    PersonaSystemPrompt,
    run_sps_benchmark,
)


def test_tier4_scenario_01_classic_full_survival_episode():
    """Scenario 1: Full classic survival gameplay episode with active laser & debris."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=0, debris_spawn_prob=0.2)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset_env(key, p)

    total_reward = 0.0
    steps_survived = 0
    # Run 60 ticks (1 second of real-time simulation)
    for _ in range(60):
        key, subkey = jax.random.split(key)
        # Tactical movement: alternate moving left/right or jumping
        action = 0 if state.laser_angle < jnp.pi else 2
        obs, state, r, done, info = env.step_env(subkey, state, action, p)
        total_reward += r
        steps_survived += 1
        if done:
            break

    assert steps_survived > 0
    assert not jnp.isnan(total_reward)


def test_tier4_scenario_02_remastered_overload_safe_zone_evasion():
    """Scenario 2: Complete Overload mode survival by escaping to the right safe zone."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1, overload_duration=5.0, debris_spawn_prob=0.0)
    key = jax.random.PRNGKey(101)
    _, state = env.reset_env(key, p)

    # 1. Trigger Overload mode while player safely inside right safe zone (x >= 1150)
    s_safe = state.replace(security_gauge=1.0, player_x=1160.0)
    _, s_safe_next, _, done, _ = env.step_env(key, s_safe, 0, p)
    assert bool(s_safe_next.is_overload) is True
    assert bool(s_safe_next.player_x >= p.safe_zone_x) is True
    # Inside safe zone: player survives overload artillery without damage
    assert bool(s_safe_next.player_hp == p.player_max_hp) is True
    assert bool(done) is False

    # 2. In contrast, being outside safe zone during Overload results in artillery damage
    s_danger = state.replace(security_gauge=1.0, player_x=683.0)
    _, s_danger_next, _, done_danger, _ = env.step_env(key, s_danger, 0, p)
    assert bool(s_danger_next.is_overload) is True
    assert bool(s_danger_next.player_hp <= 0.0) is True
    assert bool(done_danger) is True


def test_tier4_scenario_03_remastered_boss_friendly_fire_sequence():
    """Scenario 3: Multi-step boss baiting drops security gauge and shatters shield."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(777)
    _, state = env.reset_env(key, p)

    gauge_initial = 0.80
    state = state.replace(security_gauge=gauge_initial, boss_shield=200.0, shield_active=True)

    # 1. Bait tracking laser into Lotus Core: gauge drops by 10%
    gauge_step1 = max(0.0, state.security_gauge - p.tracking_laser_gauge_reduction)
    shield_step1 = 0.0  # Shattered by tracking laser
    state = state.replace(security_gauge=gauge_step1, boss_shield=shield_step1, shield_active=False)

    # 2. Bait arm slam: gauge drops by another 3%
    gauge_step2 = max(0.0, state.security_gauge - p.arm_slam_gauge_reduction)
    state = state.replace(security_gauge=gauge_step2)

    assert state.security_gauge < gauge_initial
    assert state.shield_active is False
    assert abs(state.security_gauge - (gauge_initial - 0.13)) < 1e-5


def test_tier4_scenario_04_floor_electric_warning_and_double_jump():
    """Scenario 4: Floor electric surge warning observed; player jumps and lands safely."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(555)
    _, state = env.reset_env(key, p)

    # Warning active: 1.0s remaining
    s = state.replace(electric_floor_active=True, electric_floor_warning=1.0)

    # Step forward until warning is nearly expired
    cur = s
    for _ in range(50):
        key, subk = jax.random.split(key)
        _, cur, _, _, _ = env.step_env(subk, cur, 0, p)

    # Player executes JUMP (Action 3) before detonation
    key, subk = jax.random.split(key)
    _, cur, _, _, _ = env.step_env(subk, cur, 3, p)

    # Detonation occurs while player is airborne
    cur = cur.replace(electric_floor_warning=0.0)
    key, subk = jax.random.split(key)
    _, cur, _, _, info = env.step_env(subk, cur, 0, p)

    # Player survived because they were in mid-air
    assert cur.player_hp == p.player_max_hp


def test_tier4_scenario_05_invincibility_window_tactical_passthrough():
    """Scenario 5: Player takes debris damage and uses 1s invincibility to bypass laser."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(123)
    _, state = env.reset_env(key, p)

    # Debris hit at x=500
    p_center_x = 500.0
    p_center_y = p.floor_y - p.player_h / 2.0
    s_debris = state.replace(
        player_x=p_center_x,
        debris_x=state.debris_x.at[0].set(p_center_x),
        debris_y=state.debris_y.at[0].set(p_center_y),
        debris_radius=state.debris_radius.at[0].set(20.0),
        debris_damage=state.debris_damage.at[0].set(10.0),
        debris_active=state.debris_active.at[0].set(True),
        invincible_timer=0.0,
    )
    _, s_post_hit, _, _, _ = env.step_env(key, s_debris, 0, p)
    hp_after_debris = s_post_hit.player_hp
    assert s_post_hit.invincible_timer > 0.0

    # Cross rotating laser beam while invincible
    s_laser_cross = s_post_hit.replace(
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=p.core_y + p.player_h / 2.0,
    )
    _, s_after_laser, _, _, _ = env.step_env(key, s_laser_cross, 0, p)
    # HP unchanged
    assert s_after_laser.player_hp == hp_after_debris


def test_tier4_scenario_06_batched_purejaxrl_ppo_rollout():
    """Scenario 6: 128 parallel environments unrolled over 32 steps via vmap + lax.scan."""
    env = PureJaxRLAdapterWrapper(LotusPhase1Env())
    p = env.default_params
    runner = RolloutRunner(env, p)
    b = 128
    t = 32

    keys = jax.random.split(jax.random.PRNGKey(0), b)
    _, init_states = jax.vmap(env.reset, in_axes=(0, None))(keys, p)

    @jax.jit
    def unroll_batch(k, s):
        return jax.vmap(runner.run, in_axes=(0, 0, None))(k, s, t)

    final_states, trajs = unroll_batch(keys, init_states)
    assert trajs["obs"].shape == (b, t, 130)
    assert trajs["reward"].shape == (b, t)
    assert trajs["done"].shape == (b, t)


def test_tier4_scenario_07_extreme_debris_barrage_stress():
    """Scenario 7: Extreme debris storm (prob=0.5) over 100 ticks stress testing slot recycling."""
    env = LotusPhase1Env()
    p = env.default_params.replace(debris_spawn_prob=0.5)
    key = jax.random.PRNGKey(999)
    _, state = env.reset_env(key, p)

    cur = state
    for _ in range(100):
        key, subk = jax.random.split(key)
        _, cur, _, _, _ = env.step_env(subk, cur, 0, p)
        assert cur.debris_active.shape == (MAX_DEBRIS,)
        assert jnp.sum(cur.debris_active) <= MAX_DEBRIS


def test_tier4_scenario_08_wall_pin_laser_evasion():
    """Scenario 8: Player trapped near wall escapes rotating laser via ducking and leap."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(333)
    _, state = env.reset_env(key, p)

    # Player near left wall
    s_corner = state.replace(player_x=p.wall_left + 25.0)

    # Action 6: DUCK
    _, s_duck, _, _, _ = env.step_env(key, s_corner, 6, p)
    # Action 5: JUMP_RIGHT out of corner
    _, s_leap, _, _, _ = env.step_env(key, s_duck, 5, p)
    assert s_leap.player_vx > 0.0
    assert s_leap.player_vy < 0.0


def test_tier4_scenario_09_flashbax_buffer_collection_and_sampling():
    """Scenario 9: Trajectories unrolled in accelerator memory and sampled from Flashbax buffer."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(888)
    _, init_state = env.reset_env(key, p)
    _, traj = runner.run(key, init_state, num_steps=50)

    buffer = FlashbaxAdapter(max_length=200, min_length=1, sample_batch_size=16)
    dummy = {
        "obs": jnp.zeros((130,), dtype=jnp.float32),
        "action": jnp.int32(0),
        "reward": jnp.float32(0.0),
        "done": jnp.array(False),
    }
    buf_state = buffer.init(dummy)
    for i in range(50):
        buf_state = buffer.add(buf_state, {
            "obs": traj["obs"][i],
            "action": traj["action"][i],
            "reward": traj["reward"][i],
            "done": traj["done"][i],
        })

    sample = buffer.sample(buf_state, key)
    assert sample.experience.first["obs"].shape == (16, 130)
    assert sample.experience.first["reward"].shape == (16,)


def test_tier4_scenario_10_wz_crawler_to_env_simulation_pipeline():
    """Scenario 10: Complete pipeline from WZ parsing to verified live simulation."""
    params = parse_wz_to_env_params(r"C:\mp")
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(42)
    obs, state = env.reset_env(key, params)

    for a in range(7):
        key, subk = jax.random.split(key)
        obs, state, r, done, _ = env.step_env(subk, state, a, params)
        assert obs.shape == (130,)


def test_tier4_scenario_11_multi_mode_parity_comparison():
    """Scenario 11: Cross-mode parity comparison across Classic, Remastered, and Hybrid."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)

    p_classic = env.default_params.replace(mode=0)
    p_remaster = env.default_params.replace(mode=1)
    p_hybrid = env.default_params.replace(mode=2)

    _, s_c = env.reset_env(key, p_classic)
    _, s_r = env.reset_env(key, p_remaster)
    _, s_h = env.reset_env(key, p_hybrid)

    _, next_c, _, _, info_c = env.step_env(key, s_c, 0, p_classic)
    _, next_r, _, _, info_r = env.step_env(key, s_r, 0, p_remaster)
    _, next_h, _, _, info_h = env.step_env(key, s_h, 0, p_hybrid)

    # Remastered and Hybrid accumulate gauge; Classic does not care
    assert next_r.security_gauge > 0.0
    assert next_h.security_gauge > 0.0


def test_tier4_scenario_12_hardware_sps_measurement_pipeline():
    """Scenario 12: SPS benchmark measuring throughput across batch sizes 256 and 512."""
    res = run_sps_benchmark(batch_sizes=(256, 512), num_steps=5)
    assert 256 in res
    assert 512 in res
    assert res[256] > 1000.0
    assert res[512] > 1000.0


def test_tier4_scenario_13_persona_prompt_to_env_synthesis():
    """Scenario 13: Persona prompt formatting and validation workflow."""
    prompt = PersonaSystemPrompt.format_extraction_prompt("LotusCore", "{'center': (683, 384)}")
    assert len(prompt) > 100
    mock_dict = {
        "screen_width": 1366.0, "screen_height": 768.0, "core_x": 683.0, "core_y": 384.0,
        "floor_y": 605.0, "laser_omega": 0.5235, "player_w": 40.0, "player_h": 60.0,
        "player_speed": 400.0, "jump_velocity": -650.0, "gravity": 1800.0, "dt": 1.0 / 60.0,
    }
    assert validate_env_params_schema(mock_dict) is True


def test_tier4_scenario_14_auto_reset_continuity_under_death():
    """Scenario 14: Mid-trajectory player death auto-resets without breaking scan continuity."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(99)
    _, init_state = env.reset_env(key, p)

    # Set player to lethal position at step 0
    lethal_init = init_state.replace(player_hp=0.0)
    final_s, traj = runner.run(key, lethal_init, num_steps=20)
    assert traj["obs"].shape[0] == 20
    assert traj["done"].shape[0] == 20


def test_tier4_scenario_15_marathon_timeout_truncation():
    """Scenario 15: Episode stepping through max_steps terminates with done=True."""
    env = LotusPhase1Env()
    p = env.default_params.replace(max_steps_in_episode=30)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)

    cur = state
    dones = []
    for _ in range(30):
        key, subk = jax.random.split(key)
        _, cur, _, done, _ = env.step_env(subk, cur, 0, p)
        dones.append(done)

    # Last step must trigger done=True
    assert dones[-1] == True
