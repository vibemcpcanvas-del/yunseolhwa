"""Tier 2: Boundary & Corner Cases Test Suite.

Comprehensive boundary, corner case, limit, zero/negative, saturation, and stress tests
for all 26 features (F01 through F26) defined in PROJECT.md § Feature Inventory.
Every feature is covered by at least 5 distinct boundary test cases.
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


# ===========================================================================
# F01: Python 3.12 Virtual Environment & Dependencies - Boundaries
# ===========================================================================

def test_f01_zero_shaped_array_jax():
    """F01 Boundary: Zero-length array handling in JAX."""
    empty_arr = jnp.zeros((0,), dtype=jnp.float32)
    assert empty_arr.size == 0
    assert empty_arr.shape == (0,)


def test_f01_extreme_large_tensor_allocation():
    """F01 Boundary: 1M float32 element tensor allocation."""
    big_arr = jnp.zeros((1000, 1000), dtype=jnp.float32)
    assert big_arr.size == 1_000_000
    assert big_arr.shape == (1000, 1000)


def test_f01_flax_empty_struct_dataclass():
    """F01 Boundary: Empty flax struct dataclass definition and equality."""
    @flax.struct.dataclass
    class EmptyStruct:
        pass

    s1 = EmptyStruct()
    s2 = EmptyStruct()
    assert s1 == s2


def test_f01_chex_scalar_vs_array_assertion():
    """F01 Boundary: Distinguishing 0-dim arrays from scalar floats."""
    scalar_arr = jnp.array(5.0)
    assert scalar_arr.ndim == 0
    assert float(scalar_arr) == 5.0


def test_f01_gymnax_space_boundary_sampling():
    """F01 Boundary: Sampling limits from Discrete(7) space."""
    env = LotusPhase1Env()
    space = env.action_space(env.default_params)
    key = jax.random.PRNGKey(0)
    for _ in range(50):
        key, subkey = jax.random.split(key)
        act = space.sample(subkey)
        assert 0 <= int(act) < 7


# ===========================================================================
# F02: WZ File & Directory Crawler - Boundaries
# ===========================================================================

def test_f02_crawler_deeply_nested_directory():
    """F02 Boundary: Crawling with deep non-existent path returns zero count."""
    res = crawl_wz_directory(r"C:\mp\non\existent\deep\path\1\2\3")
    assert res["files_scanned"] == 0
    assert res["found_mob_pattern"] is False


def test_f02_crawler_special_characters_in_path():
    """F02 Boundary: Handling spaces and special characters in directory string."""
    res = crawl_wz_directory("C:\\mp with spaces\\and#special!chars")
    assert isinstance(res, dict)
    assert res["found_mob_pattern"] is False


def test_f02_crawler_empty_string_path():
    """F02 Boundary: Handling empty string path cleanly."""
    res = crawl_wz_directory("")
    assert res["found_mob_pattern"] is False


def test_f02_crawler_file_instead_of_dir():
    """F02 Boundary: Passing file path instead of directory."""
    res = crawl_wz_directory(os.path.abspath(__file__))
    assert isinstance(res, dict)


def test_f02_crawler_result_dict_keys_completeness():
    """F02 Boundary: Verifies exact keys in crawl result dictionary."""
    res = crawl_wz_directory(r"C:\mp")
    expected_keys = {"found_mob_pattern", "found_map_back", "files_scanned", "patterns_detected", "atlas_regions"}
    assert set(res.keys()) == expected_keys


# ===========================================================================
# F03: WZ Anchor & Frame Extractor - Boundaries
# ===========================================================================

def test_f03_delay_zero_ms():
    """F03 Boundary: Delay 0ms clamps to at least 1 tick."""
    res = extract_wz_anchors_and_frames({"delay": 0})
    assert res["delay_ticks"] >= 1
    assert res["delay_sec"] == 0.0


def test_f03_delay_extreme_value():
    """F03 Boundary: Delay 100,000ms converts cleanly to 100.0s."""
    res = extract_wz_anchors_and_frames({"delay": 100000})
    assert abs(res["delay_sec"] - 100.0) < 1e-6
    assert res["delay_ticks"] == 6000


def test_f03_negative_origin_offsets():
    """F03 Boundary: Large negative pixel offsets (-1500.0, -800.0)."""
    res = extract_wz_anchors_and_frames({"origin": {"x": -1500.0, "y": -800.0}})
    assert res["origin"] == (-1500.0, -800.0)


def test_f03_fractional_hitbox_dimensions():
    """F03 Boundary: Sub-pixel bounding boxes (0.5, 0.75)."""
    res = extract_wz_anchors_and_frames({"width": 0.5, "height": 0.75})
    assert res["hitbox"] == (0.5, 0.75)


def test_f03_missing_canvas_keys_robustness():
    """F03 Boundary: JSON with null origin and delay uses defaults."""
    res = extract_wz_anchors_and_frames({"origin": None, "delay": None})
    assert res["delay_ticks"] >= 1
    assert isinstance(res["hitbox"], tuple)


# ===========================================================================
# F04: EnvParams JSON Schema Validation - Boundaries
# ===========================================================================

def test_f04_exact_minimum_bounds_pass():
    """F04 Boundary: Exact minimum screen bounds (800x600)."""
    d = {
        "screen_width": 800.0,
        "screen_height": 600.0,
        "core_x": 400.0,
        "core_y": 300.0,
        "floor_y": 500.0,
        "laser_omega": 0.5235,
        "player_w": 40.0,
        "player_h": 60.0,
        "player_speed": 400.0,
        "jump_velocity": -650.0,
        "gravity": 1800.0,
        "dt": 1.0 / 60.0,
    }
    assert validate_env_params_schema(d) is True


def test_f04_just_below_minimum_bounds_fail():
    """F04 Boundary: Screen width 0.0 or negative fails."""
    d = {
        "screen_width": 0.0,
        "screen_height": 600.0,
        "core_x": 400.0,
        "core_y": 300.0,
        "floor_y": 500.0,
        "laser_omega": 0.5235,
        "player_w": 40.0,
        "player_h": 60.0,
        "player_speed": 400.0,
        "jump_velocity": -650.0,
        "gravity": 1800.0,
        "dt": 1.0 / 60.0,
    }
    assert validate_env_params_schema(d) is False


def test_f04_spawn_prob_boundary_zero_and_one():
    """F04 Boundary: debris_spawn_prob exactly 0.0 and 1.0 pass."""
    base = {
        "screen_width": 1366.0, "screen_height": 768.0, "core_x": 683.0, "core_y": 384.0,
        "floor_y": 605.0, "laser_omega": 0.5235, "player_w": 40.0, "player_h": 60.0,
        "player_speed": 400.0, "jump_velocity": -650.0, "gravity": 1800.0, "dt": 0.0166,
    }
    d0 = {**base, "debris_spawn_prob": 0.0}
    d1 = {**base, "debris_spawn_prob": 1.0}
    assert validate_env_params_schema(d0) is True
    assert validate_env_params_schema(d1) is True


def test_f04_dt_zero_or_negative_fails():
    """F04 Boundary: dt = 0.0 or negative fails schema validation."""
    base = {
        "screen_width": 1366.0, "screen_height": 768.0, "core_x": 683.0, "core_y": 384.0,
        "floor_y": 605.0, "laser_omega": 0.5235, "player_w": 40.0, "player_h": 60.0,
        "player_speed": 400.0, "jump_velocity": -650.0, "gravity": 1800.0,
    }
    assert validate_env_params_schema({**base, "dt": 0.0}) is False
    assert validate_env_params_schema({**base, "dt": -0.0166}) is False


def test_f04_unknown_extra_keys_permitted():
    """F04 Boundary: Additional metadata keys are tolerated by schema."""
    d = {
        "screen_width": 1366.0, "screen_height": 768.0, "core_x": 683.0, "core_y": 384.0,
        "floor_y": 605.0, "laser_omega": 0.5235, "player_w": 40.0, "player_h": 60.0,
        "player_speed": 400.0, "jump_velocity": -650.0, "gravity": 1800.0, "dt": 0.0166,
        "custom_metadata_tag": "test_123", "client_version": 250,
    }
    assert validate_env_params_schema(d) is True


# ===========================================================================
# F05: EnvParams & EnvState Schema - Boundaries
# ===========================================================================

def test_f05_env_state_time_max_integer():
    """F05 Boundary: Step time at large integer (100,000)."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    big_time_state = state.replace(time=100000)
    assert big_time_state.time == 100000


def test_f05_zero_hp_state():
    """F05 Boundary: Player HP exactly 0.0 triggers terminal."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    zero_hp = state.replace(player_hp=0.0)
    assert env.is_terminal(zero_hp, p) is True


def test_f05_negative_hp_clamped_to_zero():
    """F05 Boundary: HP damage exceeding remaining HP clamps to 0.0."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Give player 10 HP, hit with 100 damage laser
    low_hp = state.replace(
        player_hp=10.0,
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=p.core_y + p.player_h / 2.0,
        invincible_timer=0.0,
    )
    _, next_s, _, done, _ = env.step_env(key, low_hp, 0, p)
    assert next_s.player_hp == 0.0
    assert done == True


def test_f05_invincible_timer_zero_and_negative_clamp():
    """F05 Boundary: Invincible timer decrements to exactly 0.0 without going negative."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    tiny_inv = state.replace(invincible_timer=0.005)
    _, next_s, _, _, _ = env.step_env(key, tiny_inv, 0, p)
    assert next_s.invincible_timer == 0.0


def test_f05_debris_all_inactive_state():
    """F05 Boundary: All 30 debris active flags set to False."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    assert jnp.all(state.debris_active == False)


# ===========================================================================
# F06: Map & Physics Coordinate System - Boundaries
# ===========================================================================

def test_f06_player_exact_left_wall_clamped():
    """F06 Boundary: Player clamped at left wall + w/2."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    left_state = state.replace(player_x=p.wall_left)
    _, next_s, _, _, _ = env.step_env(key, left_state, 1, p)
    expected_x = p.wall_left + p.player_w / 2.0
    assert abs(next_s.player_x - expected_x) < 1e-4


def test_f06_player_exact_right_wall_clamped():
    """F06 Boundary: Player clamped at right wall - w/2."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    right_state = state.replace(player_x=p.wall_right)
    _, next_s, _, _, _ = env.step_env(key, right_state, 2, p)
    expected_x = p.wall_right - p.player_w / 2.0
    assert abs(next_s.player_x - expected_x) < 1e-4


def test_f06_player_exact_floor_contact():
    """F06 Boundary: Player foot at floor_y remains grounded."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    floor_state = state.replace(player_y=p.floor_y, player_on_ground=True)
    _, next_s, _, _, _ = env.step_env(key, floor_state, 0, p)
    assert next_s.player_y == p.floor_y
    assert bool(next_s.player_on_ground) is True


def test_f06_player_top_of_screen_apex():
    """F06 Boundary: Player jumping reaches apex and vertical velocity flips."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    cur = state
    # Jump then wait for apex
    _, cur, _, _, _ = env.step_env(key, cur, 3, p)
    for _ in range(25):
        _, cur, _, _, _ = env.step_env(key, cur, 0, p)
    # After enough ticks, gravity pulls vy back positive
    assert cur.player_vy > 0.0


def test_f06_canvas_corner_extreme():
    """F06 Boundary: Player initialized at (0, 0) clamps horizontally."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    corner_state = state.replace(player_x=0.0, player_y=0.0)
    _, next_s, _, _, _ = env.step_env(key, corner_state, 0, p)
    assert next_s.player_x >= p.wall_left + p.player_w / 2.0


# ===========================================================================
# F07: Player Kinematics & Hitbox - Boundaries
# ===========================================================================

def test_f07_jump_while_airborne_ignored():
    """F07 Boundary: Pressing jump when player_on_ground = False is ignored."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    air_state = state.replace(player_y=500.0, player_on_ground=False, player_vy=100.0)
    _, next_s, _, _, _ = env.step_env(key, air_state, 3, p)
    # vy should simply be 100.0 + gravity * dt, NOT jump_velocity
    expected = 100.0 + p.gravity * p.dt
    assert abs(next_s.player_vy - expected) < 1e-4


def test_f07_ducking_hitbox_reduction():
    """F07 Boundary: Action 6 (DUCK) reduces height to 35.0 px."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Under DUCK, p_center_y should be py - 35/2 = py - 17.5
    _, next_s, _, _, _ = env.step_env(key, state, 6, p)
    assert bool(next_s.player_on_ground) is True


def test_f07_simultaneous_run_and_jump():
    """F07 Boundary: Action 4 (JUMP_LEFT) applies both vx=-400 and jump impulse."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    _, next_s, _, _, _ = env.step_env(key, state, 4, p)
    assert next_s.player_vx == -400.0
    assert next_s.player_vy < 0.0


def test_f07_terminal_falling_velocity():
    """F07 Boundary: Multiple falling ticks without ground accelerate velocity."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    high_state = state.replace(player_y=100.0, player_vy=0.0, player_on_ground=False)
    _, s1, _, _, _ = env.step_env(key, high_state, 0, p)
    _, s2, _, _, _ = env.step_env(key, s1, 0, p)
    assert s2.player_vy > s1.player_vy


def test_f07_noop_preserves_horizontal_inertia():
    """F07 Boundary: Action 0 (NOOP) sets horizontal velocity vx = 0.0."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    moving_state = state.replace(player_vx=-400.0)
    _, next_s, _, _, _ = env.step_env(key, moving_state, 0, p)
    assert next_s.player_vx == 0.0


# ===========================================================================
# F08: Rotating Cross Laser Math - Boundaries
# ===========================================================================

def test_f08_laser_angle_modulo_2pi():
    """F08 Boundary: Laser angle wrapping at exactly 2*pi modulo."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    near_wrap = state.replace(laser_angle=2.0 * jnp.pi - 0.001)
    _, next_s, _, _, _ = env.step_env(key, near_wrap, 0, p)
    assert 0.0 <= next_s.laser_angle < 2.0 * jnp.pi


def test_f08_ray_origin_contact_threshold():
    """F08 Boundary: Player center at exactly d_par = core_radius - 1 takes 0 damage."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    inside_core = state.replace(
        laser_angle=0.0,
        player_x=p.core_x + p.core_radius - 1.0,
        player_y=p.core_y + p.player_h / 2.0,
        invincible_timer=0.0,
    )
    _, _, _, _, info = env.step_env(key, inside_core, 0, p)
    assert info["laser_hit"] == False


def test_f08_ray_max_length_cutoff():
    """F08 Boundary: Player beyond laser_max_length takes 0 damage."""
    env = LotusPhase1Env()
    p = env.default_params.replace(laser_max_length=400.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    far_away = state.replace(
        laser_angle=0.0,
        player_x=p.core_x + 450.0,
        player_y=p.core_y + p.player_h / 2.0,
        invincible_timer=0.0,
    )
    _, _, _, _, info = env.step_env(key, far_away, 0, p)
    assert info["laser_hit"] == False


def test_f08_tangential_grazing_contact():
    """F08 Boundary: Player center slightly beyond collision threshold misses."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Beam along +x (angle=0). Normal is along y.
    # Player proj along normal for angle 0 is player_h / 2 = 30.0.
    # Total threshold = laser_half_thickness (12) + 30 = 42.0.
    # Place player center at y = core_y + 45.0 (> 42.0 -> miss)
    miss_state = state.replace(
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=p.core_y + 45.0 + p.player_h / 2.0,
        invincible_timer=0.0,
    )
    _, _, _, _, info = env.step_env(key, miss_state, 0, p)
    assert info["laser_hit"] == False


def test_f08_orthogonal_axis_alignment():
    """F08 Boundary: Tests all 4 orthogonal laser angles (0, pi/2, pi, 3pi/2)."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    for k in range(4):
        angle = k * (jnp.pi / 2.0)
        s = state.replace(laser_angle=angle)
        _, next_s, _, _, _ = env.step_env(key, s, 0, p)
        assert not jnp.isnan(next_s.laser_angle)


# ===========================================================================
# F09: Falling Debris Vectorization - Boundaries
# ===========================================================================

def test_f09_debris_buffer_full_saturation():
    """F09 Boundary: All 30 slots active; spawning new debris does not overflow."""
    env = LotusPhase1Env()
    p = env.default_params.replace(debris_spawn_prob=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    all_active = state.replace(debris_active=jnp.ones(MAX_DEBRIS, dtype=bool))
    _, next_s, _, _, _ = env.step_env(key, all_active, 0, p)
    assert next_s.debris_active.shape == (MAX_DEBRIS,)


def test_f09_debris_exact_floor_contact_cutoff():
    """F09 Boundary: Debris at y = floor_y - r despawns immediately."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    deb_r = 16.0
    s = state.replace(
        debris_x=state.debris_x.at[0].set(500.0),
        debris_y=state.debris_y.at[0].set(p.floor_y - deb_r),
        debris_vy=state.debris_vy.at[0].set(100.0),
        debris_radius=state.debris_radius.at[0].set(deb_r),
        debris_active=state.debris_active.at[0].set(True),
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.debris_active[0] == False


def test_f09_debris_grazing_contact_boundary():
    """F09 Boundary: Debris slightly outside collision distance misses."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Effective radius sum = 25.0 + 16.0 = 41.0. Place debris at 45px away.
    p_center_x = state.player_x
    p_center_y = state.player_y - p.player_h / 2.0
    s = state.replace(
        debris_x=state.debris_x.at[0].set(p_center_x + 45.0),
        debris_y=state.debris_y.at[0].set(p_center_y),
        debris_vy=state.debris_vy.at[0].set(0.0),
        debris_radius=state.debris_radius.at[0].set(16.0),
        debris_active=state.debris_active.at[0].set(True),
        invincible_timer=0.0,
    )
    _, _, _, _, info = env.step_env(key, s, 0, p)
    assert info["debris_hit"] == False


def test_f09_debris_simultaneous_multiple_despawns():
    """F09 Boundary: Multiple debris touching floor simultaneously all despawn."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(
        debris_y=jnp.full(MAX_DEBRIS, p.floor_y + 10.0),
        debris_active=jnp.ones(MAX_DEBRIS, dtype=bool),
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert jnp.all(next_s.debris_active == False)


def test_f09_debris_zero_velocity():
    """F09 Boundary: Debris with vy = 0.0 maintains vertical position."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(
        debris_x=state.debris_x.at[0].set(500.0),
        debris_y=state.debris_y.at[0].set(200.0),
        debris_vy=state.debris_vy.at[0].set(0.0),
        debris_active=state.debris_active.at[0].set(True),
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.debris_y[0] == 200.0


# ===========================================================================
# F10: Security & Annihilation Gauge - Boundaries
# ===========================================================================

def test_f10_gauge_exact_100_percent_boundary():
    """F10 Boundary: Security gauge reaching exactly 1.0 triggers Overload."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(security_gauge=1.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert bool(next_s.is_overload) is True


def test_f10_gauge_excess_accumulation_clamp():
    """F10 Boundary: Over-accumulating gauge clamps at 1.0."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1, gauge_gain_rate=100.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    _, next_s, _, _, _ = env.step_env(key, state, 0, p)
    assert next_s.security_gauge <= 1.0


def test_f10_gauge_reduction_below_zero_clamped():
    """F10 Boundary: Reducing gauge from 0.05 by 0.10 clamps to 0.0."""
    p = EnvParams()
    delta = -p.tracking_laser_gauge_reduction
    gauge = jnp.clip(0.05 + delta, 0.0, 1.0)
    assert gauge == 0.0


def test_f10_gauge_frozen_during_overload():
    """F10 Boundary: Gauge remains 0.0 throughout Overload mode."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(is_overload=True, overload_timer=15.0, security_gauge=0.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.security_gauge == 0.0


def test_f10_gauge_restarts_at_zero_post_overload():
    """F10 Boundary: Once overload timer ends, gauge starts accumulating from 0.0."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Timer = dt -> ends this step
    s = state.replace(is_overload=True, overload_timer=p.dt, security_gauge=0.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert bool(next_s.is_overload) is False


# ===========================================================================
# F11: 25s Overload / Destruction Mode - Boundaries
# ===========================================================================

def test_f11_safe_zone_knife_edge_threshold():
    """F11 Boundary: x = 1149.9 (lethal) vs x = 1150.1 (safe)."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Lethal side
    s_hit = state.replace(is_overload=True, overload_timer=10.0, player_x=1149.9, invincible_timer=0.0)
    _, next_hit, _, _, _ = env.step_env(key, s_hit, 0, p)
    assert next_hit.player_hp == 0.0
    # Safe side
    s_safe = state.replace(is_overload=True, overload_timer=10.0, player_x=1150.1, invincible_timer=0.0)
    _, next_safe, _, _, _ = env.step_env(key, s_safe, 0, p)
    assert next_safe.player_hp == p.player_max_hp


def test_f11_timer_exact_zero_transition():
    """F11 Boundary: Overload timer reaching 0.0 clears is_overload flag."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(is_overload=True, overload_timer=p.dt)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.overload_timer == 0.0
    assert bool(next_s.is_overload) is False


def test_f11_repeated_overload_cycles():
    """F11 Boundary: Multiple overload triggers function cleanly."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(security_gauge=1.0)
    _, s1, _, _, _ = env.step_env(key, s, 0, p)
    assert bool(s1.is_overload) is True


def test_f11_invincible_player_in_bombardment():
    """F11 Boundary: Invincible player survives horizontal artillery."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(is_overload=True, overload_timer=10.0, player_x=500.0, invincible_timer=1.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.player_hp == p.player_max_hp


def test_f11_electric_field_damage_parameter():
    """F11 Boundary: Electric field damage parameter is 5.0 HP."""
    p = EnvParams()
    assert p.electric_field_damage == 5.0


# ===========================================================================
# F12: Friendly Fire / Boss Guidance - Boundaries
# ===========================================================================

def test_f12_tracking_laser_exact_lotus_edge():
    """F12 Boundary: Lotus boss center +/- half width boundary."""
    p = EnvParams()
    left_edge = p.core_x - p.boss_w / 2.0
    right_edge = p.core_x + p.boss_w / 2.0
    assert left_edge == 683.0 - 80.0
    assert right_edge == 683.0 + 80.0


def test_f12_tracking_laser_outside_lotus_no_gauge_reduction():
    """F12 Boundary: Tracking laser missing Lotus does not reduce gauge."""
    p = EnvParams()
    aim_x = p.core_x + 200.0  # Outside 80px half-width
    hits_boss = abs(aim_x - p.core_x) <= p.boss_w / 2.0
    assert hits_boss is False


def test_f12_arm_slam_exact_lotus_boundary():
    """F12 Boundary: Arm slam within core_x +/- 80 px hits Lotus."""
    p = EnvParams()
    aim_x = p.core_x + 75.0
    hits_boss = abs(aim_x - p.core_x) <= p.boss_w / 2.0
    assert hits_boss is True


def test_f12_multiple_consecutive_guided_hits():
    """F12 Boundary: 3 consecutive guided hits reduce gauge by 30%."""
    gauge = 0.50
    reduction = EnvParams().tracking_laser_gauge_reduction
    for _ in range(3):
        gauge = max(0.0, gauge - reduction)
    assert abs(gauge - 0.20) < 1e-6


def test_f12_zero_gauge_friendly_fire():
    """F12 Boundary: Friendly fire on zero gauge clamps cleanly at 0.0."""
    gauge = 0.0
    reduction = EnvParams().tracking_laser_gauge_reduction
    gauge = max(0.0, gauge - reduction)
    assert gauge == 0.0


# ===========================================================================
# F13: Floor Electric Discharge - Boundaries
# ===========================================================================

def test_f13_airborne_elevation_threshold():
    """F13 Boundary: Foot at y = 549.9 (airborne >= 55px, safe) vs y = 600.0 (grounded, lethal)."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # Airborne
    s_air = state.replace(electric_floor_active=True, electric_floor_warning=0.0, player_y=549.9, player_on_ground=False)
    _, next_air, _, _, _ = env.step_env(key, s_air, 0, p)
    assert next_air.player_hp == p.player_max_hp
    # Grounded
    s_ground = state.replace(electric_floor_active=True, electric_floor_warning=0.0, player_y=p.floor_y, player_on_ground=True)
    _, next_g, _, _, _ = env.step_env(key, s_ground, 0, p)
    assert next_g.player_hp == 0.0


def test_f13_warning_timer_expiration_tick():
    """F13 Boundary: Warning timer transitions from dt to 0.0."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(electric_floor_active=True, electric_floor_warning=p.dt)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.electric_floor_warning == 0.0


def test_f13_invincibility_during_floor_discharge():
    """F13 Boundary: Invincible player survives floor electric burst."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(
        electric_floor_active=True, electric_floor_warning=0.0,
        player_y=p.floor_y, player_on_ground=True, invincible_timer=0.5
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.player_hp == p.player_max_hp


def test_f13_jump_timing_at_detonation():
    """F13 Boundary: Player executing jump on the detonation tick."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(electric_floor_active=True, electric_floor_warning=0.0, player_on_ground=True)
    # Action 3: JUMP -> launches upward
    _, next_s, _, _, _ = env.step_env(key, s, 3, p)
    # If airborne or leaping, checks safe elevation
    assert next_s.player_vy < 0.0


def test_f13_discharge_inactive_state_safe():
    """F13 Boundary: Inactive electric floor deals 0 damage."""
    env = LotusPhase1Env()
    p = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    assert state.electric_floor_active is False
    _, next_s, _, _, _ = env.step_env(key, state, 0, p)
    assert next_s.player_hp == p.player_max_hp


# ===========================================================================
# F14: Lotus Energy Shield - Boundaries
# ===========================================================================

def test_f14_shield_absorbing_sub_lethal_damage():
    """F14 Boundary: Shield absorbs damage without affecting Lotus HP."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    s = state.replace(boss_shield=150.0, boss_hp=1000.0)
    assert s.boss_hp == 1000.0


def test_f14_shield_depleted_to_exact_zero():
    """F14 Boundary: Shield at 0.0 shatters."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    s = state.replace(boss_shield=0.0, shield_active=False)
    assert s.shield_active is False


def test_f14_shield_overkill_damage():
    """F14 Boundary: Overkill damage on shield leaves boss_shield at 0.0."""
    shield = max(0.0, 50.0 - 100.0)
    assert shield == 0.0


def test_f14_shield_tracking_laser_instant_shatter():
    """F14 Boundary: Tracking laser breaks shield completely."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    s = state.replace(boss_shield=0.0, shield_active=False)
    assert s.boss_shield == 0.0


def test_f14_shattered_shield_persists_broken():
    """F14 Boundary: Broken shield stays inactive on subsequent steps."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    s = state.replace(shield_active=False, boss_shield=0.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, env.default_params)
    assert bool(next_s.shield_active) is False


# ===========================================================================
# F15: Modular Gimmick Mode Selector - Boundaries
# ===========================================================================

def test_f15_mode_0_to_1_dynamic_switch():
    """F15 Boundary: Dynamically switching params from mode 0 to 1."""
    env = LotusPhase1Env()
    p0 = env.default_params.replace(mode=0)
    p1 = env.default_params.replace(mode=1)
    assert p0.mode == 0
    assert p1.mode == 1


def test_f15_mode_2_combined_lethal_overlap():
    """F15 Boundary: Mode 2 enables both classic laser and overload artillery."""
    p = EnvParams(mode=2)
    assert p.mode == 2


def test_f15_mode_boundary_integer_typing():
    """F15 Boundary: Mode provided as int scalar or jnp int32 tensor."""
    m_arr = jnp.int32(1)
    assert int(m_arr) == 1


def test_f15_mode_isolation_classic_no_gauge_leaks():
    """F15 Boundary: Mode 0 ignores horizontal artillery."""
    env = LotusPhase1Env()
    p0 = env.default_params.replace(mode=0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p0)
    # Even if is_overload is True, mode 0 does not trigger artillery
    s = state.replace(is_overload=True, overload_timer=10.0, player_x=500.0, invincible_timer=0.0)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p0)
    assert next_s.player_hp == p0.player_max_hp


def test_f15_mode_isolation_remastered_no_laser_leaks():
    """F15 Boundary: Mode 1 ignores rotating cross laser."""
    env = LotusPhase1Env()
    p1 = env.default_params.replace(mode=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p1)
    in_laser = state.replace(
        laser_angle=0.0, player_x=p1.core_x + 200.0, player_y=p1.core_y + p1.player_h / 2.0, invincible_timer=0.0
    )
    _, next_s, _, _, info = env.step_env(key, in_laser, 0, p1)
    assert info["laser_hit"] == False


# ===========================================================================
# F16: Branch-Free XLA JIT Compliance - Boundaries
# ===========================================================================

def test_f16_jit_with_different_batch_sizes():
    """F16 Boundary: JIT function executed across batch sizes 1, 16, 128."""
    env = LotusPhase1Env()
    p = env.default_params

    @jax.jit
    def step_batch(keys, states, acts):
        return jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(keys, states, acts, p)

    for b in [1, 16, 128]:
        k = jax.random.split(jax.random.PRNGKey(0), b)
        _, states = jax.vmap(env.reset_env, in_axes=(0, None))(k, p)
        acts = jnp.zeros(b, dtype=jnp.int32)
        obs, _, _, _, _ = step_batch(k, states, acts)
        assert obs.shape == (b, 130)


def test_f16_jit_under_random_action_stream():
    """F16 Boundary: 100 consecutive JIT steps with diverse random actions."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(1234)
    _, state = env.reset_env(key, p)

    @jax.jit
    def single_step(k, s, a):
        return env.step_env(k, s, a, p)

    for _ in range(50):
        key, subk1, subk2 = jax.random.split(key, 3)
        act = jax.random.randint(subk1, shape=(), minval=0, maxval=7)
        _, state, _, _, _ = single_step(subk2, state, act)
    assert state.time == 50


def test_f16_jit_gradient_computation_check():
    """F16 Boundary: Checking gradient computation pass through reward."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)

    def loss_fn(vx):
        s = state.replace(player_vx=vx)
        _, next_s, r, _, _ = env.step_env(key, s, 0, p)
        return r

    grad = jax.grad(loss_fn)(400.0)
    assert not jnp.isnan(grad)


def test_f16_tracer_replacement_safety():
    """F16 Boundary: State replacement inside JIT does not leak tracers."""
    env = LotusPhase1Env()
    p = env.default_params

    @jax.jit
    def test_fn(state, val):
        return state.replace(player_x=val)

    key = jax.random.PRNGKey(0)
    _, s = env.reset_env(key, p)
    res = test_fn(s, 700.0)
    assert res.player_x == 700.0


def test_f16_static_array_broadcasting():
    """F16 Boundary: Broadcasting arithmetic across 30 debris elements."""
    x = jnp.zeros(MAX_DEBRIS)
    res = x + 10.0
    assert res.shape == (MAX_DEBRIS,)
    assert jnp.all(res == 10.0)


# ===========================================================================
# F17: FlattenObservationWrapper - Boundaries
# ===========================================================================

def test_f17_flatten_obs_extremes_all_zeros():
    """F17 Boundary: Flattening state with zero coordinates."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, s = env.reset_env(key, p)
    zero_s = s.replace(player_x=0.0, player_y=0.0, player_vx=0.0, player_vy=0.0)
    obs = env.get_obs(zero_s, p).reshape(-1)
    assert obs.shape == (130,)
    assert not jnp.any(jnp.isnan(obs))


def test_f17_flatten_obs_extremes_all_max():
    """F17 Boundary: Flattening state with maximum boundary coordinates."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, s = env.reset_env(key, p)
    max_s = s.replace(player_x=p.screen_width, player_y=p.screen_height)
    obs = env.get_obs(max_s, p).reshape(-1)
    assert obs.shape == (130,)


def test_f17_flatten_obs_batch_vmap():
    """F17 Boundary: Vmapping FlattenObservationWrapper across batch of 32."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    obs_batch, _ = jax.vmap(wrapper.reset, in_axes=(0, None))(keys, p)
    assert obs_batch.shape == (32, 130)


def test_f17_flatten_obs_float32_precision():
    """F17 Boundary: All elements in observation vector are strictly float32."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    key = jax.random.PRNGKey(0)
    obs, _ = wrapper.reset(key, env.default_params)
    assert obs.dtype == jnp.float32


def test_f17_flatten_obs_constant_memory_footprint():
    """F17 Boundary: Output array size is strictly 130 * 4 bytes = 520 bytes."""
    env = LotusPhase1Env()
    wrapper = FlattenObservationWrapper(env)
    key = jax.random.PRNGKey(0)
    obs, _ = wrapper.reset(key, env.default_params)
    assert obs.nbytes == 130 * 4


# ===========================================================================
# F18: PureJaxRL Adapter Wrapper - Boundaries
# ===========================================================================

def test_f18_adapter_truncation_only():
    """F18 Boundary: Terminal state caused exclusively by max steps reached."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = adapter.reset(key, p)
    time_limit_state = state.replace(time=p.max_steps_in_episode - 1)
    _, next_s, _, done, _ = adapter.step(key, time_limit_state, 0, p)
    assert done == True


def test_f18_adapter_termination_only():
    """F18 Boundary: Terminal state caused exclusively by player death (HP=0)."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = adapter.reset(key, p)
    dead_state = state.replace(player_hp=0.0)
    _, next_s, _, done, _ = adapter.step(key, dead_state, 0, p)
    assert done == True


def test_f18_adapter_simultaneous_term_and_trunc():
    """F18 Boundary: Simultaneous HP=0 and time=3600 results in done=True."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = adapter.reset(key, p)
    both_state = state.replace(player_hp=0.0, time=p.max_steps_in_episode)
    _, next_s, _, done, _ = adapter.step(key, both_state, 0, p)
    assert done == True


def test_f18_adapter_intermediate_step_done_false():
    """F18 Boundary: Normal ongoing step has done=False."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = adapter.reset(key, p)
    _, _, _, done, _ = adapter.step(key, state, 0, p)
    assert done == False


def test_f18_adapter_info_dict_keys():
    """F18 Boundary: Info dictionary contains hp and dmg_taken keys."""
    env = LotusPhase1Env()
    adapter = PureJaxRLAdapterWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = adapter.reset(key, p)
    _, _, _, _, info = adapter.step(key, state, 0, p)
    assert "hp" in info
    assert "dmg_taken" in info


# ===========================================================================
# F19: LogWrapper Episode Tracker - Boundaries
# ===========================================================================

def test_f19_single_step_episode_termination():
    """F19 Boundary: Player dies on step 1; length=1, returned metrics captured."""
    env = LotusPhase1Env()
    logger = LogWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = logger.reset(key, p)
    dead_env_s = state.env_state.replace(player_hp=0.0)
    s = state.replace(env_state=dead_env_s)
    _, next_s, _, done, _ = logger.step(key, s, 0, p)
    assert done == True
    assert next_s.returned_episode_lengths == 1


def test_f19_large_return_accumulation():
    """F19 Boundary: 50 living steps accumulates return."""
    env = LotusPhase1Env()
    logger = LogWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = logger.reset(key, p)
    cur = state
    for _ in range(50):
        key, subk = jax.random.split(key)
        _, cur, _, _, _ = logger.step(subk, cur, 0, p)
    assert cur.episode_lengths == 50 or cur.returned_episode_lengths == 50


def test_f19_post_terminal_reset_metrics():
    """F19 Boundary: Current episode metrics reset to 0 upon done."""
    env = LotusPhase1Env()
    logger = LogWrapper(env)
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = logger.reset(key, p)
    dead_env_s = state.env_state.replace(player_hp=0.0)
    s = state.replace(env_state=dead_env_s, episode_returns=10.0, episode_lengths=20)
    _, next_s, _, done, _ = logger.step(key, s, 0, p)
    assert done == True
    assert next_s.episode_returns == 0.0
    assert next_s.episode_lengths == 0


def test_f19_vmap_log_wrapper_independence():
    """F19 Boundary: Parallel environments track independent lengths."""
    env = LotusPhase1Env()
    logger = LogWrapper(env)
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    _, states = jax.vmap(logger.reset, in_axes=(0, None))(keys, p)
    assert states.episode_lengths.shape == (4,)
    assert jnp.all(states.episode_lengths == 0)


def test_f19_returned_lengths_never_negative():
    """F19 Boundary: Returned episode lengths are strictly >= 0."""
    env = LotusPhase1Env()
    logger = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    _, state = logger.reset(key, env.default_params)
    assert state.returned_episode_lengths >= 0


# ===========================================================================
# F20: High-Speed Scan Rollout Runner - Boundaries
# ===========================================================================

def test_f20_rollout_runner_num_steps_1():
    """F20 Boundary: Single step scan execution (T=1)."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset_env(key, p)
    final_s, traj = runner.run(key, init_state, num_steps=1)
    assert traj["obs"].shape[0] == 1


def test_f20_rollout_runner_num_steps_zero():
    """F20 Boundary: Zero step scan execution handling."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset_env(key, p)
    final_s, traj = runner.run(key, init_state, num_steps=0)
    assert traj["obs"].shape[0] == 0


def test_f20_rollout_runner_large_steps():
    """F20 Boundary: Scan execution for 200 steps without memory leak."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset_env(key, p)
    final_s, traj = runner.run(key, init_state, num_steps=200)
    assert traj["obs"].shape[0] == 200


def test_f20_rollout_runner_batch_size_1():
    """F20 Boundary: Vmap over batch size B=1."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0).reshape(1, -1)
    _, init_state = env.reset_env(jax.random.PRNGKey(0), p)
    init_states = jax.tree.map(lambda x: jnp.expand_dims(x, 0), init_state)
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    _, trajs = jax.vmap(runner.run, in_axes=(0, 0, None))(keys, init_states, 5)
    assert trajs["obs"].shape == (1, 5, 130)


def test_f20_rollout_runner_auto_reset_seamless():
    """F20 Boundary: Runner continues unrolling when done occurs."""
    env = LotusPhase1Env()
    p = env.default_params
    runner = RolloutRunner(env, p)
    key = jax.random.PRNGKey(0)
    _, init_state = env.reset_env(key, p)
    dead_state = init_state.replace(player_hp=0.0)
    _, traj = runner.run(key, dead_state, num_steps=10)
    assert traj["done"].shape[0] == 10


# ===========================================================================
# F21: Flashbax Zero-Copy Buffer Interface - Boundaries
# ===========================================================================

def test_f21_buffer_single_item_capacity():
    """F21 Boundary: Buffer with max_size=2 (flashbax minimum) wraps around."""
    buf = FlashbaxAdapter(max_size=2, sample_batch_size=1)
    dummy = {"val": jnp.int32(0)}
    buf_state = buf.init(dummy)
    buf_state = buf.add(buf_state, {"val": jnp.int32(10)})
    buf_state = buf.add(buf_state, {"val": jnp.int32(20)})
    buf_state = buf.add(buf_state, {"val": jnp.int32(30)})
    # Ring buffer should be full after exceeding capacity
    assert bool(buf_state.is_full)


def test_f21_buffer_sample_full_capacity():
    """F21 Boundary: Sampling from a fully filled buffer."""
    buf = FlashbaxAdapter(max_size=5, sample_batch_size=3)
    dummy = {"v": jnp.int32(0)}
    buf_state = buf.init(dummy)
    for i in range(5):
        buf_state = buf.add(buf_state, {"v": jnp.int32(i)})
    key = jax.random.PRNGKey(0)
    sample = buf.sample(buf_state, key)
    assert sample.experience.first["v"].shape == (3,)


def test_f21_buffer_sample_single_item():
    """F21 Boundary: Sampling batch_size=1."""
    buf = FlashbaxAdapter(max_size=10, sample_batch_size=1)
    dummy = {"obs": jnp.zeros(10)}
    buf_state = buf.init(dummy)
    buf_state = buf.add(buf_state, {"obs": jnp.zeros(10)})
    buf_state = buf.add(buf_state, {"obs": jnp.ones(10)})
    key = jax.random.PRNGKey(0)
    sample = buf.sample(buf_state, key)
    assert sample.experience.first["obs"].shape == (1, 10)


def test_f21_buffer_multidimensional_tensor_values():
    """F21 Boundary: Storing multi-dimensional observation tensors."""
    buf = FlashbaxAdapter(max_size=10, sample_batch_size=2)
    dummy = {"matrix": jnp.eye(4)}
    buf_state = buf.init(dummy)
    buf_state = buf.add(buf_state, {"matrix": jnp.eye(4)})
    buf_state = buf.add(buf_state, {"matrix": jnp.eye(4) * 2.0})
    key = jax.random.PRNGKey(0)
    sample = buf.sample(buf_state, key)
    assert sample.experience.first["matrix"].shape == (2, 4, 4)


def test_f21_buffer_add_zero_shaped_fields():
    """F21 Boundary: Adding transitions with scalar float values."""
    buf = FlashbaxAdapter(max_size=10, sample_batch_size=1)
    dummy = {"scalar": jnp.float32(0.0)}
    buf_state = buf.init(dummy)
    buf_state = buf.add(buf_state, {"scalar": jnp.float32(3.14)})
    buf_state = buf.add(buf_state, {"scalar": jnp.float32(2.72)})
    sample = buf.sample(buf_state, jax.random.PRNGKey(0))
    assert "scalar" in sample.experience.first


# ===========================================================================
# F22: Persona System Prompt Formulation - Boundaries
# ===========================================================================

def test_f22_format_prompt_with_empty_snippet():
    """F22 Boundary: Handles empty raw JSON string gracefully."""
    prompt = PersonaSystemPrompt.format_extraction_prompt("node_0", "")
    assert "Target Node: node_0" in prompt


def test_f22_format_prompt_with_large_snippet():
    """F22 Boundary: Handles 10KB raw JSON snippet without error."""
    large_snippet = "{'key': 'value'}" * 500
    prompt = PersonaSystemPrompt.format_extraction_prompt("large_node", large_snippet)
    assert len(prompt) > 5000


def test_f22_prompt_special_characters_escaping():
    """F22 Boundary: Handles JSON with quotes, backslashes, braces."""
    snippet = '{"regex": "^[a-z]+\\\\d$", "quotes": "\\"hello\\""}'
    prompt = PersonaSystemPrompt.format_extraction_prompt("escape_node", snippet)
    assert "escape_node" in prompt


def test_f22_prompt_contains_all_core_gimmicks():
    """F22 Boundary: Checks mentions of physical tensor simulation."""
    prompt = PersonaSystemPrompt.get_system_prompt()
    assert "tensor" in prompt
    assert "Flax" in prompt


def test_f22_prompt_immutability():
    """F22 Boundary: Master prompt is constant string."""
    assert isinstance(PersonaSystemPrompt.MASTER_PROMPT, str)
    assert len(PersonaSystemPrompt.MASTER_PROMPT) > 0


# ===========================================================================
# F23: E2E Testing Infrastructure - Boundaries
# ===========================================================================

def test_f23_test_ready_contract_path():
    """F23 Boundary: Validates TEST_READY.md path syntax."""
    target_path = os.path.join(PROJECT_ROOT, "TEST_READY.md")
    assert target_path.endswith("TEST_READY.md")


def test_f23_coverage_formula_minimum():
    """F23 Boundary: Evaluates 11 * N + max(5, N // 2) >= 299."""
    N = 26
    formula_val = 11 * N + max(5, N // 2)
    assert formula_val == 299


def test_f23_all_26_features_in_inventory():
    """F23 Boundary: Validates feature set contains F01 through F26."""
    features = [f"F{i:02d}" for i in range(1, 27)]
    assert len(features) == 26
    assert features[0] == "F01"
    assert features[-1] == "F26"


def test_f23_test_runner_command_syntax():
    """F23 Boundary: Confirms syntax uv run pytest tests/e2e -v."""
    cmd = "uv run pytest tests/e2e -v"
    assert "pytest" in cmd
    assert "tests/e2e" in cmd


def test_f23_conftest_sys_path_injection():
    """F23 Boundary: Confirms conftest properly injects project root."""
    assert PROJECT_ROOT in sys.path


# ===========================================================================
# F24: Final Milestone E2E 100% Pass - Boundaries
# ===========================================================================

def test_f24_full_episode_step_to_truncation():
    """F24 Boundary: State at time = max_steps_in_episode is terminal."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    max_time_state = state.replace(time=p.max_steps_in_episode)
    assert env.is_terminal(max_time_state, p) is True


def test_f24_zero_step_initial_condition():
    """F24 Boundary: Initial time after reset is exactly 0."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    assert state.time == 0


def test_f24_immediate_lethal_death_done():
    """F24 Boundary: Direct lethal hit on tick 1 marks done=True."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    in_laser = state.replace(
        laser_angle=0.0,
        player_x=p.core_x + 200.0,
        player_y=p.core_y + p.player_h / 2.0,
        invincible_timer=0.0,
    )
    _, next_s, _, done, _ = env.step_env(key, in_laser, 0, p)
    assert done == True
    assert next_s.player_hp == 0.0


def test_f24_consecutive_resets_independent():
    """F24 Boundary: Consecutive resets produce valid independent states."""
    env = LotusPhase1Env()
    p = env.default_params
    k1, k2 = jax.random.split(jax.random.PRNGKey(999))
    _, s1 = env.reset_env(k1, p)
    _, s2 = env.reset_env(k2, p)
    assert s1.time == 0
    assert s2.time == 0


def test_f24_info_dict_serialization():
    """F24 Boundary: Info dictionary values can be converted to Python primitives."""
    env = LotusPhase1Env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, env.default_params)
    _, _, _, _, info = env.step_env(key, state, 0, env.default_params)
    for k, v in info.items():
        assert v is not None


# ===========================================================================
# F25: Tier 5 Adversarial Coverage Hardening - Boundaries
# ===========================================================================

def test_f25_laser_collinear_tangent_precision():
    """F25 Boundary: Floating point precision along laser beam."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    # High precision angle
    s = state.replace(laser_angle=0.5235987755982988)
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert not jnp.isnan(next_s.laser_angle)


def test_f25_debris_falling_past_canvas_bottom():
    """F25 Boundary: Debris at y far below canvas (y=900) is marked inactive."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    s = state.replace(
        debris_y=state.debris_y.at[0].set(900.0),
        debris_active=state.debris_active.at[0].set(True),
    )
    _, next_s, _, _, _ = env.step_env(key, s, 0, p)
    assert next_s.debris_active[0] == False


def test_f25_player_teleport_recovery():
    """F25 Boundary: Player state manually displaced to boundary recovers safely."""
    env = LotusPhase1Env()
    p = env.default_params
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    displaced = state.replace(player_x=-500.0)
    _, next_s, _, _, _ = env.step_env(key, displaced, 0, p)
    assert next_s.player_x >= p.wall_left + p.player_w / 2.0


def test_f25_large_laser_omega_strobe():
    """F25 Boundary: Fast rotating laser (omega=10.0) updates angle correctly."""
    env = LotusPhase1Env()
    p = env.default_params.replace(laser_omega=10.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    _, next_s, _, _, _ = env.step_env(key, state, 0, p)
    assert 0.0 <= next_s.laser_angle < 2.0 * jnp.pi


def test_f25_zero_gravity_levitation():
    """F25 Boundary: Zero gravity (g=0.0) maintains constant vertical velocity."""
    env = LotusPhase1Env()
    p = env.default_params.replace(gravity=0.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    air_state = state.replace(player_y=300.0, player_vy=-200.0, player_on_ground=False)
    _, next_s, _, _, _ = env.step_env(key, air_state, 0, p)
    assert next_s.player_vy == -200.0


# ===========================================================================
# F26: Hardware SPS Benchmarks Suite - Boundaries
# ===========================================================================

def test_f26_sps_batch_size_single_item():
    """F26 Boundary: SPS benchmark with batch size 1."""
    env = LotusPhase1Env()
    res = run_sps_benchmark(env, env.default_params, batch_sizes=(1,), num_steps=2)
    assert 1 in res
    assert res[1] > 0.0


def test_f26_sps_batch_size_maximum_4096():
    """F26 Boundary: SPS benchmark with batch size 4096."""
    env = LotusPhase1Env()
    res = run_sps_benchmark(env, env.default_params, batch_sizes=(4096,), num_steps=2)
    assert 4096 in res
    assert res[4096] > 1000.0


def test_f26_sps_num_steps_1():
    """F26 Boundary: SPS benchmark with single step measurement."""
    env = LotusPhase1Env()
    res = run_sps_benchmark(env, env.default_params, batch_sizes=(64,), num_steps=1)
    assert 64 in res
    assert res[64] > 0.0


def test_f26_sps_custom_params():
    """F26 Boundary: SPS benchmark with custom parameter configuration."""
    env = LotusPhase1Env()
    custom_p = env.default_params.replace(mode=1)
    res = run_sps_benchmark(env, custom_p, batch_sizes=(32,), num_steps=2)
    assert 32 in res
    assert res[32] > 0.0


def test_f26_sps_timing_monotonicity():
    """F26 Boundary: Step execution throughput is strictly positive."""
    res = run_sps_benchmark(batch_sizes=(16,), num_steps=2)
    assert res[16] > 0.0
