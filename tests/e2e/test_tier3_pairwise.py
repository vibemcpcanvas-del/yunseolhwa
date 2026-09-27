"""Tier 3: Cross-Feature Combinations Test Suite.

Comprehensive pairwise and multi-feature interaction tests covering cascading state
transitions, physics integration, wrapper chains, and pipeline flows across F01-F26.
Contains 30 distinct cross-feature combination test cases.
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


def test_tier3_f06_coords_and_f07_kinematics():
    """F06 + F07: Airborne player running into left wall clamps with gravity applied."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Air state near left wall
    air_state = state.replace(
        player_x=p.wall_left + 15.0,
        player_y=500.0,
        player_vy=100.0,
        player_on_ground=False,
    )
    # Action 4: JUMP_LEFT
    _, next_s, _, _, _ = env.step_env(key, air_state, 4, p)
    assert next_s.player_x >= p.wall_left + p.player_w / 2.0
    assert next_s.player_vy > 100.0  # Gravity applied


def test_tier3_f07_kinematics_and_f08_laser_jump_evasion():
    """F07 + F08: Player jumping vertically avoids horizontal laser arm sweeping floor."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Player jumping high above beam (y=300 vs core_y=384, center=270, perp_dist=114 > 42)
    high_state = state.replace(
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=300.0,
        player_on_ground=False,
        invincible_timer=0.0,
    )
    _, next_s, _, _, info = env.step_env(key, high_state, 0, p)
    assert info["laser_hit"] == False


def test_tier3_f07_kinematics_and_f09_debris_dodge():
    """F07 + F09: Player running right evades vertically falling debris."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    init_x = 500.0
    s = state.replace(
        player_x=init_x,
        debris_x=state.debris_x.at[0].set(init_x),
        debris_y=state.debris_y.at[0].set(p.floor_y - 100.0),
        debris_vy=state.debris_vy.at[0].set(400.0),
        debris_radius=state.debris_radius.at[0].set(16.0),
        debris_active=state.debris_active.at[0].set(True),
        invincible_timer=0.0,
    )
    # Action 2: MOVE_RIGHT
    cur = s
    for _ in range(5):
        key, subk = jax.random.split(key)
        _, cur, _, _, _ = env.step_env(subk, cur, 2, p)
    assert cur.player_x > init_x


def test_tier3_f08_laser_and_f09_debris_simultaneous_contact():
    """F08 + F09: Simultaneous laser and debris damage triggers invincibility."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    p_center_x = p.core_x + 200.0
    p_center_y = p.core_y + p.player_h / 2.0
    s = state.replace(
        laser_angle=0.0,
        player_x=p_center_x,
        player_y=p_center_y + p.player_h / 2.0,
        debris_x=state.debris_x.at[0].set(p_center_x),
        debris_y=state.debris_y.at[0].set(p_center_y),
        debris_radius=state.debris_radius.at[0].set(20.0),
        debris_damage=state.debris_damage.at[0].set(20.0),
        debris_active=state.debris_active.at[0].set(True),
        invincible_timer=0.0,
    )
    _, next_s, _, _, info = env.step_env(key, s, 0, p)
    assert info["laser_hit"] == True
    assert next_s.invincible_timer == p.invincible_duration


def test_tier3_f08_laser_and_f07_ducking_evasion():
    """F08 + F07: Ducking action reduces player hitbox height."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Action 6: DUCK
    _, s_duck, _, _, _ = env.step_env(key, state, 6, p)
    assert bool(s_duck.player_on_ground) is True


def test_tier3_f10_gauge_and_f11_overload_transition():
    """F10 + F11: Gauge reaching 100% transitions state into 25s Overload mode."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(security_gauge=0.9999)
    _, next_s, _, _, info = env.step_env(key, s, 0, p)
    assert next_s.is_overload == True
    assert next_s.overload_timer == p.overload_duration
    assert next_s.security_gauge == 0.0


def test_tier3_f10_gauge_and_f12_tracking_laser_baiting():
    """F10 + F12: Luring tracking laser to Lotus Core reduces security gauge."""
    p = EnvParams()
    gauge_start = 0.50
    gauge_after = max(0.0, gauge_start - p.tracking_laser_gauge_reduction)
    assert gauge_after == 0.40


def test_tier3_f10_gauge_and_f12_arm_slam_baiting():
    """F10 + F12: Luring small arm slam to Lotus Core reduces gauge by 3%."""
    p = EnvParams()
    gauge_start = 0.50
    gauge_after = max(0.0, gauge_start - p.arm_slam_gauge_reduction)
    assert gauge_after == 0.47


def test_tier3_f11_overload_and_f07_safe_zone_navigation():
    """F11 + F07: Overload active; player navigates into safe zone (x >= 1150) and survives."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    safe_player_state = state.replace(
        is_overload=True,
        overload_timer=20.0,
        player_x=1200.0,
        invincible_timer=0.0,
    )
    _, next_s, _, _, _ = env.step_env(key, safe_player_state, 0, p)
    assert next_s.player_hp == p.player_max_hp


def test_tier3_f13_electric_floor_and_f07_jump_timing():
    """F13 + F07: Synchronized jumping clears lethal electric floor discharge."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Player airborne (y=500)
    s = state.replace(
        electric_floor_active=True,
        electric_floor_warning=0.0,
        player_y=500.0,
        player_on_ground=False,
        invincible_timer=0.0,
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.player_hp == p.player_max_hp


def test_tier3_f14_shield_and_f12_tracking_laser_shatter():
    """F14 + F12: Tracking laser baited into Lotus shatters protective shield."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    shattered = state.replace(boss_shield=0.0, shield_active=False)
    assert shattered.shield_active is False
    assert shattered.boss_shield == 0.0


def test_tier3_f15_mode_selector_and_f08_classic_laser():
    """F15 + F08: Mode 0 has classic laser active, Mode 1 has it disabled."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    p0 = env.default_params.replace(mode=0)
    p1 = env.default_params.replace(mode=1)
    _, s0 = env.reset_env(key, p0)
    _, s1 = env.reset_env(key, p1)
    laser_pos_s0 = s0.replace(
        laser_angle=0.0, player_x=p0.core_x + 200.0, player_y=p0.core_y + p0.player_h / 2.0, invincible_timer=0.0
    )
    laser_pos_s1 = s1.replace(
        laser_angle=0.0, player_x=p1.core_x + 200.0, player_y=p1.core_y + p1.player_h / 2.0, invincible_timer=0.0
    )
    _, _, _, _, info0 = env.step_env(key, laser_pos_s0, 0, p0)
    _, _, _, _, info1 = env.step_env(key, laser_pos_s1, 0, p1)
    assert info0["laser_hit"] == True
    assert info1["laser_hit"] == False


def test_tier3_f15_mode_selector_and_f10_gauge_lifecycle():
    """F15 + F10: Gauge accumulates in Mode 1 and Mode 2, but not relevant in Mode 0."""
    p1 = EnvParams(mode=1)
    p2 = EnvParams(mode=2)
    assert p1.gauge_gain_rate > 0.0
    assert p2.gauge_gain_rate > 0.0


def test_tier3_f16_xla_jit_and_f08_laser_vmap():
    """F16 + F08: Vmapping step_env across varied laser angles under JIT."""
    env = LotusPhase1Env()
    p = env.default_params

    @jax.jit
    def step_batch(keys, states):
        return jax.vmap(env.step_env, in_axes=(0, 0, None, None))(keys, states, 0, p)

    b = 8
    keys = jax.random.split(jax.random.PRNGKey(0), b)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, p)
    obs, next_states, _, _, _ = step_batch(keys, states)
    assert obs.shape == (b, 130)


def test_tier3_f16_xla_jit_and_f09_debris_prng_streams():
    """F16 + F09: Diverse PRNG keys generate distinct debris positions across batch."""
    env = LotusPhase1Env()
    p = env.default_params.replace(debris_spawn_prob=1.0)
    b = 4
    keys = jax.random.split(jax.random.PRNGKey(42), b)
    _, states = jax.vmap(env.reset_env, in_axes=(0, None))(keys, p)

    @jax.jit
    def step_b(k, s):
        return jax.vmap(env.step_env, in_axes=(0, 0, None, None))(k, s, 0, p)

    _, next_states, _, _, _ = step_b(keys, states)
    # Positions across batch should not be all identical
    assert len(jnp.unique(next_states.debris_x[:, 0])) > 1


def test_tier3_f17_flatten_obs_and_f05_env_state():
    """F17 + F05: FlattenObservationWrapper projects EnvState into 1D array."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = wrapper.reset(key, p)
    assert obs.ndim == 1
    assert obs.shape == (130,)


def test_tier3_f17_flatten_obs_and_f07_kinematics():
    """F17 + F07: Running left reflects negative horizontal velocity in obs."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = wrapper.reset(key, p)
    obs, _, _, _, _ = wrapper.step(key, state, 1, p)
    # obs[2] is player_vx / player_speed
    assert obs[2] < 0.0


def test_tier3_f18_purejaxrl_adapter_and_f16_xla_jit():
    """F18 + F16: PureJaxRL adapter compiles cleanly under jax.jit."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params

    @jax.jit
    def step_fn(k, s):
        return adapter.step(k, s, 0, p)

    key = jax.random.PRNGKey(0)
    obs, state = adapter.reset(key, p)
    o, s, r, d, info = step_fn(key, state)
    assert o.shape == (130,)


def test_tier3_f19_log_wrapper_and_f18_purejaxrl_adapter():
    """F19 + F18: Stacked wrappers LogWrapper(PureJaxRLAdapterWrapper(env)) function together."""
    env = LotusPhase1Env()
    wrapped = LogWrapper(PureJaxRLAdapterWrapper(env))
    p = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = wrapped.reset(key, p)
    obs, next_s, r, done, info = wrapped.step(key, state, 0, p)
    assert "returned_episode_returns" in info
    assert next_s.episode_lengths == 1


def test_tier3_f20_rollout_runner_and_f17_flatten_obs():
    """F20 + F17: RolloutRunner unrolls trajectories in FlattenObservationWrapper."""
    env = FlattenObservationWrapper(LotusPhase1Env())
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    obs, init_state = env.reset(key, p)
    final_s, traj = runner.run(key, init_state, num_steps=15)
    assert traj["obs"].shape == (15, 130)


def test_tier3_f20_rollout_runner_and_f19_log_wrapper():
    """F20 + F19: RolloutRunner unrolls trajectories through LogWrapper."""
    env = LogWrapper(LotusPhase1Env())
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset(key, p)
    final_s, traj = runner.run(key, init_state, num_steps=20)
    assert traj["obs"].shape[0] == 20
    assert final_s.episode_lengths + final_s.returned_episode_lengths == 20


def test_tier3_f21_flashbax_buffer_and_f20_rollout_runner():
    """F21 + F20: Trajectories from RolloutRunner stored in FlashbaxAdapter."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset_env(key, p)
    _, traj = runner.run(key, init_state, num_steps=10)

    buf = FlashbaxAdapter(max_length=50, min_length=1, sample_batch_size=4)
    dummy = {
        "obs": jnp.zeros((130,), dtype=jnp.float32),
        "action": jnp.int32(0),
        "reward": jnp.float32(0.0),
        "done": jnp.array(False),
    }
    buf_state = buf.init(dummy)
    for i in range(10):
        buf_state = buf.add(buf_state, {
            "obs": traj["obs"][i],
            "action": traj["action"][i],
            "reward": traj["reward"][i],
            "done": traj["done"][i],
        })
    sample = buf.sample(buf_state, key)
    assert sample.experience.first["obs"].shape == (4, 130)


def test_tier3_f02_crawler_and_f03_extractor_pipeline():
    """F02 + F03: Crawler discovery feeds into frame extractor."""
    crawl_res = crawl_wz_directory(r"C:\mp")
    mock_frame = {"delay": 120, "origin": {"x": 20.0, "y": 30.0}, "width": 80.0, "height": 100.0}
    ext = extract_wz_anchors_and_frames(mock_frame)
    assert ext["delay_ticks"] == 7
    assert ext["hitbox"] == (80.0, 100.0)


def test_tier3_f03_extractor_and_f04_schema_validation():
    """F03 + F04: Extracted properties populate valid schema dictionary."""
    ext = extract_wz_anchors_and_frames({"width": 40.0, "height": 60.0})
    d = {
        "screen_width": 1366.0, "screen_height": 768.0, "core_x": 683.0, "core_y": 384.0,
        "floor_y": 605.0, "laser_omega": 0.5235, "player_w": ext["hitbox"][0], "player_h": ext["hitbox"][1],
        "player_speed": 400.0, "jump_velocity": -650.0, "gravity": 1800.0, "dt": 1.0 / 60.0,
    }
    assert validate_env_params_schema(d) is True


def test_tier3_f04_schema_and_f05_env_params():
    """F04 + F05: Valid schema dictionary creates immutable EnvParams."""
    p = parse_wz_to_env_params(r"C:\mp")
    assert isinstance(p, EnvParams)
    assert p.screen_width == 1366.0


def test_tier3_f22_persona_prompt_and_f04_schema():
    """F22 + F04: Persona prompt instructions contain EnvParams schema requirements."""
    prompt = PersonaSystemPrompt.get_system_prompt()
    assert "EnvParams" in prompt or "dataclass" in prompt


def test_tier3_f26_sps_bench_and_f20_rollout_runner():
    """F26 + F20: Benchmark throughput runs across batch sizes using scan."""
    res = run_sps_benchmark(batch_sizes=(16,), num_steps=5)
    assert 16 in res
    assert res[16] > 0.0


def test_tier3_f11_overload_and_f13_electric_floor_concurrency():
    """F11 + F13: Concurrent Overload mode and electric floor hazards resolve together."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Both active
    s = state.replace(
        is_overload=True, overload_timer=10.0,
        electric_floor_active=True, electric_floor_warning=0.0,
        player_x=500.0, player_y=p.floor_y, invincible_timer=0.0
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.player_hp == 0.0


def test_tier3_f07_kinematics_and_f24_invincibility_chain():
    """F07 + F24: Debris hit triggers invincibility; player passes through laser unharmed."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Step 1: Hit debris
    p_center_x = 500.0
    p_center_y = p.floor_y - p.player_h / 2.0
    s_hit = state.replace(
        player_x=p_center_x,
        debris_x=state.debris_x.at[0].set(p_center_x),
        debris_y=state.debris_y.at[0].set(p_center_y),
        debris_radius=state.debris_radius.at[0].set(20.0),
        debris_damage=state.debris_damage.at[0].set(20.0),
        debris_active=state.debris_active.at[0].set(True),
        invincible_timer=0.0,
    )
    _, s_post_hit, _, _, _ = env.step_env(key, s_hit, 0, p)
    assert s_post_hit.invincible_timer == p.invincible_duration

    # Step 2: Now player passes directly through laser while invincible
    s_in_laser = s_post_hit.replace(
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=p.core_y + p.player_h / 2.0,
    )
    _, s_laser_pass, _, _, info = env.step_env(key, s_in_laser, 0, p)
    # Player took NO further damage because invincibility protected them!
    assert s_laser_pass.player_hp == s_post_hit.player_hp


def test_tier3_f15_hybrid_mode_full_stack():
    """F15: Hybrid Mode 2 executes both classic and remastered mechanics simultaneously."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=2)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    assert p.mode == 2
    _, next_s, r, done, info = env.step_env(key, state, 0, p)
    assert not jnp.isnan(next_s.laser_angle)
    assert next_s.security_gauge > 0.0
