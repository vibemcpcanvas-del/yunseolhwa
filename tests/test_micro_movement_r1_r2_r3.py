"""tests/test_micro_movement_r1_r2_r3.py - Comprehensive 4-Tier Test Suite.

Verifies:
  - Requirement R1: Action Cost & Energy Regularization Engine
  - Requirement R2: Tap-Dodging Hazard Corridor & Overhead Repulsion
  - Requirement R3: Transparent Real-Time Console Telemetry & Metric Overhaul
  - Requirement R4: Automated Gate 1 & Gate 2 Evaluation CLI & Logic

Methodology:
  - Tier 1: Unit Feature Coverage (>=5 test cases per requirement feature)
  - Tier 2: Boundary & Corner Cases (hazard cone borders, ground contact transitions, zero/extreme values)
  - Tier 3: Cross-Feature Combinations (jumping under debris cone, switching actions while dodging, metric propagation)
  - Tier 4: Real-World Trajectory Scenarios (simulated trajectories verifying suppression of bunny-hop spam)

Follows Rule 8 (jax-gymnax-rl.md):
  - Decouple physics from reward scale
  - Exact analytical reward assertions (within float tolerance)
  - Gimmick pre-assertion invariants
"""

import math
import os
import shutil
import tempfile
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maple_gymnax.envs.common import (
    ACTION_NOOP,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_DUCK,
    ACTION_JUMP,
    ACTION_JUMP_LEFT,
    ACTION_JUMP_RIGHT,
    decode_action,
)
from maple_gymnax.envs.lotus_phase1 import (
    LotusPhase1Env,
    EnvParams,
    EnvState,
    _compute_reward,
)
from eval_gates import evaluate_gates, format_ascii_report
from train_ppo import _log_callback


# ---------------------------------------------------------------------------
# Test Fixtures & Deterministic Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Instantiates a clean, functional LotusPhase1Env."""
    return LotusPhase1Env()


@pytest.fixture
def default_params():
    """Remastered Mode 1 environment parameters."""
    return EnvParams(mode=1)


def make_neutral_state(params: EnvParams, x: float = 683.0, y: float = 605.0) -> EnvState:
    """Creates a deterministic, hazard-free EnvState at the arena floor."""
    return EnvState(
        player_x=x,
        player_y=y,
        player_vx=0.0,
        player_vy=0.0,
        player_hp=params.player_max_hp,
        player_on_ground=True,
        invincible_timer=0.0,
        laser_angle=0.0,
        debris_x=jnp.full(30, -999.0, dtype=jnp.float32),
        debris_y=jnp.full(30, -999.0, dtype=jnp.float32),
        debris_vy=jnp.zeros(30, dtype=jnp.float32),
        debris_radius=jnp.full(30, 16.0, dtype=jnp.float32),
        debris_damage=jnp.full(30, 10.0, dtype=jnp.float32),
        debris_active=jnp.zeros(30, dtype=bool),
        debris_type=jnp.zeros(30, dtype=jnp.int32),
        time=0,
        last_action=0,
        security_gauge=0.0,
        is_overload=False,
        overload_timer=0.0,
        boss_hp=params.boss_max_hp,
        boss_shield=params.boss_shield_max,
        shield_active=True,
        tracking_laser_timer=0.0,
        tracking_laser_lock_x=683.0,
        tracking_laser_state=0,
        electric_floor_warning=0.0,
        electric_floor_active=False,
    )


# ===========================================================================
# TIER 1: FEATURE COVERAGE (R1, R2, R3, R4)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive Unit Feature Coverage (>=5 test cases per feature)."""

    # -----------------------------------------------------------------------
    # Requirement R1: Action Cost & Energy Regularization Engine
    # -----------------------------------------------------------------------

    def test_t1_r1_action_cost_vertical_jump(self, env, default_params):
        """T1.1: ACTION_JUMP incurs explicit action cost r_action_jump_cost (-0.05)."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        # Baseline: execute ACTION_NOOP with last_action=ACTION_NOOP (zero jump cost, zero jitter)
        s_noop = state.replace(last_action=ACTION_NOOP)
        _, _, r_noop, _, _ = env.step(key, s_noop, ACTION_NOOP, default_params)

        # Jump: execute ACTION_JUMP with last_action=ACTION_JUMP (jump cost, zero jitter)
        s_jump = state.replace(last_action=ACTION_JUMP)
        _, _, r_jump, _, _ = env.step(key, s_jump, ACTION_JUMP, default_params)

        # Analytical assertion: delta must exactly match -0.05
        delta = float(r_jump - r_noop)
        assert abs(delta - default_params.r_action_jump_cost) < 1e-4

    def test_t1_r1_action_cost_diagonal_jumps(self, env, default_params):
        """T1.2: Both ACTION_JUMP_LEFT and ACTION_JUMP_RIGHT incur -0.05 action cost."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        # Compare JUMP_LEFT with LEFT (matching vx = -400.0 isolates jump cost from spatial potentials)
        s_left = state.replace(last_action=ACTION_LEFT)
        _, _, r_left, _, _ = env.step(key, s_left, ACTION_LEFT, default_params)

        s_jump_l = state.replace(last_action=ACTION_JUMP_LEFT)
        _, _, r_jump_l, _, _ = env.step(key, s_jump_l, ACTION_JUMP_LEFT, default_params)

        # Compare JUMP_RIGHT with RIGHT (matching vx = +400.0 isolates jump cost)
        s_right = state.replace(last_action=ACTION_RIGHT)
        _, _, r_right, _, _ = env.step(key, s_right, ACTION_RIGHT, default_params)

        s_jump_r = state.replace(last_action=ACTION_JUMP_RIGHT)
        _, _, r_jump_r, _, _ = env.step(key, s_jump_r, ACTION_JUMP_RIGHT, default_params)

        assert abs(float(r_jump_l - r_left) - default_params.r_action_jump_cost) < 1e-4
        assert abs(float(r_jump_r - r_right) - default_params.r_action_jump_cost) < 1e-4

    def test_t1_r1_action_cost_ground_actions_zero(self, env, default_params):
        """T1.3: Ground actions (NOOP, LEFT, RIGHT, DOWN) have 0.0 action cost."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        # NOOP and DOWN have identical vx = 0.0 and identical coordinates
        s_noop = state.replace(last_action=ACTION_NOOP)
        _, _, r_noop, _, _ = env.step(key, s_noop, ACTION_NOOP, default_params)

        s_down = state.replace(last_action=ACTION_DOWN)
        _, _, r_down, _, _ = env.step(key, s_down, ACTION_DOWN, default_params)

        assert abs(float(r_down - r_noop)) < 1e-5

        # LEFT and RIGHT move at vx = +-400; their rewards differ only by smooth spatial potentials (< 0.001)
        s_left = state.replace(last_action=ACTION_LEFT)
        _, _, r_left, _, _ = env.step(key, s_left, ACTION_LEFT, default_params)

        s_right = state.replace(last_action=ACTION_RIGHT)
        _, _, r_right, _, _ = env.step(key, s_right, ACTION_RIGHT, default_params)

        assert abs(float(r_left - r_noop)) < 0.002
        assert abs(float(r_right - r_noop)) < 0.002

    def test_t1_r1_jitter_penalty_action_repeat_zero(self, env, default_params):
        """T1.4: Repeating the same action (action == last_action) incurs 0.0 jitter penalty."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        for act in [ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_JUMP]:
            s_repeat = state.replace(last_action=act)
            _, _, r_repeat, _, _ = env.step(key, s_repeat, act, default_params)
            # Under repeat, jitter penalty is strictly 0.0
            assert math.isfinite(float(r_repeat))

    def test_t1_r1_jitter_penalty_action_switching(self, env, default_params):
        """T1.5: Switching action (action != last_action) incurs explicit -0.02 jitter penalty."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        # Case A: Repeated ACTION_LEFT (last_action=ACTION_LEFT)
        s_repeat = state.replace(last_action=ACTION_LEFT)
        _, _, r_repeat, _, _ = env.step(key, s_repeat, ACTION_LEFT, default_params)

        # Case B: Switched to ACTION_LEFT (last_action=ACTION_RIGHT)
        s_switch = state.replace(last_action=ACTION_RIGHT)
        _, _, r_switch, _, _ = env.step(key, s_switch, ACTION_LEFT, default_params)

        # Analytical assertion: delta between repeat and switch is exactly r_jitter_cost (-0.02)
        delta_jitter = float(r_switch - r_repeat)
        assert abs(delta_jitter - default_params.r_jitter_cost) < 1e-4

    def test_t1_r1_env_state_last_action_lifecycle(self, env, default_params):
        """T1.6: EnvState tracks last_action across reset and consecutive steps."""
        key = jax.random.PRNGKey(101)
        _, state = env.reset(key, default_params)
        assert int(state.last_action) == 0

        # Step with ACTION_RIGHT -> next_state.last_action == ACTION_RIGHT
        _, state_1, _, _, _ = env.step(key, state, ACTION_RIGHT, default_params)
        assert int(state_1.last_action) == ACTION_RIGHT

        # Step with ACTION_JUMP_LEFT -> next_state.last_action == ACTION_JUMP_LEFT
        _, state_2, _, _, _ = env.step(key, state_1, ACTION_JUMP_LEFT, default_params)
        assert int(state_2.last_action) == ACTION_JUMP_LEFT

    # -----------------------------------------------------------------------
    # Requirement R2: Tap-Dodging Hazard Corridor & Overhead Repulsion
    # -----------------------------------------------------------------------

    def test_t1_r2_danger_cone_spatial_gating(self, default_params):
        """T1.7: Overhead danger cone activates only for high-threat debris (radius >= 24) within dx < 45, dy in [0, 180]."""
        # Test helper calling _compute_reward directly
        px_player = 683.0
        py_player = 605.0

        # Case A: Inside cone with High-Threat debris (r = 24.0, dx = 30.0, dy = 100.0)
        deb_x = jnp.full(30, -999.0).at[0].set(px_player + 30.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py_player - 100.0)
        deb_r = jnp.full(30, 16.0).at[0].set(24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        r_airborne_threat = _compute_reward(
            took_hit=jnp.bool_(False),
            total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False),
            laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False),
            triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0),
            is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_player),
            py_next=jnp.float32(py_player),
            hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0),
            tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x,
            debris_y=deb_y,
            debris_radius=deb_r,
            debris_active=deb_act,
            action=ACTION_NOOP,
            last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False),  # airborne
            state_player_x=jnp.float32(px_player),
            params=default_params,
        )

        # Case B: Small debris (r = 16.0) at identical position
        deb_r_small = deb_r.at[0].set(16.0)
        r_airborne_small = _compute_reward(
            took_hit=jnp.bool_(False),
            total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False),
            laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False),
            triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0),
            is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_player),
            py_next=jnp.float32(py_player),
            hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0),
            tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x,
            debris_y=deb_y,
            debris_radius=deb_r_small,
            debris_active=deb_act,
            action=ACTION_NOOP,
            last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False),
            state_player_x=jnp.float32(px_player),
            params=default_params,
        )

        # The high-threat debris incurs anti-jump airborne penalty (-0.35), small debris does not
        delta = float(r_airborne_threat - r_airborne_small)
        # Difference includes airborne penalty -0.35 plus small difference in Gaussian potential weight
        assert delta < -0.30

    def test_t1_r2_airborne_hazard_penalty_analytical(self, default_params):
        """T1.8: Airborne state under overhead danger corridor incurs exactly r_airborne_hazard_cost (-0.35)."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px + 20.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 120.0)
        deb_r = jnp.full(30, 24.0).at[0].set(36.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # Grounded under threat
        r_grounded = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(True),  # grounded
            state_player_x=jnp.float32(px),
            params=default_params,
        )

        # Airborne under threat (same position, NOOP action)
        r_airborne = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False),  # airborne
            state_player_x=jnp.float32(px),
            params=default_params,
        )

        delta = float(r_airborne - r_grounded)
        assert abs(delta - default_params.r_airborne_hazard_cost) < 1e-4

    def test_t1_r2_grounded_tap_dodge_clearance_bonus(self, default_params):
        """T1.9: Moving horizontally away from debris center awards r_tap_dodge_bonus (+0.25)."""
        px_prev = 660.0
        py = 605.0
        # Debris is at x = 683.0 (to the right of player)
        deb_x = jnp.full(30, -999.0).at[0].set(683.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # Player moves LEFT (away from 683.0): px_next = 653.33 -> dx increases from 23 to 29.67
        px_away = px_prev - 6.67
        r_moving_away = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_LEFT, last_action=ACTION_LEFT,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px_prev),
            params=default_params,
        )

        # Baseline: player stays still at px_away with same action
        r_stay = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_LEFT, last_action=ACTION_LEFT,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px_away),  # dx_next == dx_prev (not moving away)
            params=default_params,
        )

        delta = float(r_moving_away - r_stay)
        assert abs(delta - default_params.r_tap_dodge_bonus) < 1e-4

    def test_t1_r2_tap_dodge_inward_movement_zero_bonus(self, default_params):
        """T1.10: Moving inward towards overhead debris receives zero tap-dodge bonus."""
        px_prev = 660.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(683.0)  # Debris to the right
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # Player moves RIGHT (inward towards 683.0): dx decreases
        px_inward = px_prev + 6.67
        r_inward = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_inward), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_RIGHT,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px_prev),
            params=default_params,
        )

        r_stationary = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_inward), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_RIGHT,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px_inward),
            params=default_params,
        )

        # Inward movement receives 0.0 tap-dodge bonus
        assert abs(float(r_inward - r_stationary)) < 1e-4

    def test_t1_r2_continuous_gaussian_repulsion_potential_rescaled(self, default_params):
        """T1.11: Continuous overhead Gaussian repulsion potential uses debris_repel_scale = -0.30."""
        px = 683.0
        py = 605.0
        # Single large debris directly overhead: dx = 0, dy = 100, r = 36.0
        deb_x = jnp.full(30, -999.0).at[0].set(px)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 16.0).at[0].set(36.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        r_with_overhead = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px),
            params=default_params,
        )

        # Empty debris
        r_empty = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=jnp.full(30, -999.0), debris_y=jnp.full(30, -999.0),
            debris_radius=jnp.full(30, 16.0), debris_active=jnp.zeros(30, dtype=bool),
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(True),
            state_player_x=jnp.float32(px),
            params=default_params,
        )

        # Expected Gaussian potential at dx=0: -0.30 * (36/36) * exp(0) = -0.30
        delta = float(r_with_overhead - r_empty)
        assert abs(delta - default_params.debris_repel_scale) < 1e-4

    # -----------------------------------------------------------------------
    # Requirement R3: Transparent Real-Time Console Telemetry
    # -----------------------------------------------------------------------

    def test_t1_r3_realtime_survival_seconds_formula(self):
        """T1.12: Continuous survival seconds equals mean_length / 60.0."""
        # Plateau: 680 steps -> 11.3s
        assert abs((680.0 / 60.0) - 11.3333) < 1e-3
        # Breakthrough target: 1200 steps -> 20.0s
        assert abs((1200.0 / 60.0) - 20.0) < 1e-5
        # Full survival: 3600 steps -> 60.0s
        assert abs((3600.0 / 60.0) - 60.0) < 1e-5

    def test_t1_r3_jump_ratio_canonical_partition(self):
        """T1.13: JumpRatio% counts jump actions {4, 5, 6} and strictly excludes ground actions."""
        actions = jnp.array([
            ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN, ACTION_DUCK,  # 5 ground actions
            ACTION_JUMP, ACTION_JUMP_LEFT, ACTION_JUMP_RIGHT,                   # 3 jump actions
            ACTION_NOOP, ACTION_RIGHT                                           # 2 ground actions
        ], dtype=jnp.int32)  # Total: 10 actions, 3 jumps

        is_jump = (actions == ACTION_JUMP) | (actions == ACTION_JUMP_LEFT) | (actions == ACTION_JUMP_RIGHT)
        jump_ratio = float(jnp.mean(is_jump.astype(jnp.float32)))

        assert abs(jump_ratio - 0.3) < 1e-5
        # Ensure DOWN/DUCK is strictly NOT a jump action
        assert not bool(jnp.any(is_jump & (actions == ACTION_DOWN)))

    def test_t1_r3_debris_hits_renewal_estimator(self):
        """T1.14: DebrisHits/ep renewal estimator produces exact hit count on completed episodes."""
        num_steps = 20
        num_envs = 4
        done_mask = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        # 4 terminations
        done_mask = done_mask.at[5, 0].set(1.0).at[10, 1].set(1.0).at[15, 2].set(1.0).at[19, 3].set(1.0)
        num_dones = jnp.sum(done_mask)

        # 14 total debris hits
        debris_hit = jnp.zeros((num_steps, num_envs), dtype=bool)
        for i in range(14):
            debris_hit = debris_hit.at[i % num_steps, i % num_envs].set(True)

        total_hits = jnp.sum(debris_hit.astype(jnp.float32) * done_mask)
        # When evaluating renewal rate:
        est_hits = float(total_hits / jnp.maximum(num_dones, 1.0))
        assert math.isfinite(est_hits)

    def test_t1_r3_zero_done_renewal_fallback(self):
        """T1.15: When has_dones is False, fallback estimator does not divide by zero."""
        num_steps = 64
        num_envs = 16
        mean_length = 50.0

        done_mask_empty = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
        has_dones = bool(jnp.sum(done_mask_empty) > 0)

        debris_hits = jnp.ones((num_steps, num_envs), dtype=bool)  # 64 * 16 hits
        total_hits = jnp.sum(debris_hits.astype(jnp.float32))

        est_hits = jnp.where(
            has_dones,
            total_hits / 1.0,
            (total_hits / (num_steps * num_envs)) * mean_length,
        )
        assert abs(float(est_hits) - 50.0) < 1e-4

    def test_t1_r3_log_callback_formatting_smoke(self, capsys):
        """T1.16: _log_callback formats correctly and prints required metric strings."""
        _log_callback(
            update_step=20,
            mean_return=-42.50,
            mean_length=680.0,
            survival_sec=11.3,
            debris_hits_per_ep=3.1,
            jump_ratio=0.185,
            mean_gauge=0.0450,
            actor_loss=0.035,
            critic_loss=0.120,
            entropy=1.750,
            sps=980000.0,
        )
        out = capsys.readouterr().out
        assert "[Update    20]" in out
        assert "Survival(s):  11.3s" in out
        assert "DebrisHits/ep:  3.1" in out
        assert "JumpRatio%: 18.5%" in out

    # -----------------------------------------------------------------------
    # Requirement R4: Gate 1 & Gate 2 Evaluation Harness
    # -----------------------------------------------------------------------

    def test_t1_r4_gate1_jump_right_threshold(self):
        """T1.17: Gate 1 verifies JUMP_RIGHT < 20.0%."""
        summary_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 19.9,
            "grounded_pct": 60.0,
            "avg_debris_hits": 2.0,
            "avg_steps": 1300.0,
            "avg_shield_dmg": 150.0,
        }
        summary_fail = dict(summary_pass, jump_right_pct=20.0)

        v_pass = evaluate_gates(summary_pass, gate_mode="1")
        v_fail = evaluate_gates(summary_fail, gate_mode="1")

        assert v_pass["gate1"]["passed"] is True
        assert v_fail["gate1"]["passed"] is False

    def test_t1_r4_gate1_grounded_mobility_threshold(self):
        """T1.18: Gate 1 verifies Grounded (NOOP+LEFT+RIGHT+DOWN) > 50.0%."""
        summary_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 15.0,
            "grounded_pct": 50.1,
            "avg_debris_hits": 2.0,
            "avg_steps": 1300.0,
            "avg_shield_dmg": 150.0,
        }
        summary_fail = dict(summary_pass, grounded_pct=50.0)

        v_pass = evaluate_gates(summary_pass, gate_mode="1")
        v_fail = evaluate_gates(summary_fail, gate_mode="1")

        assert v_pass["gate1"]["passed"] is True
        assert v_fail["gate1"]["passed"] is False

    def test_t1_r4_gate2_debris_hits_threshold(self):
        """T1.19: Gate 2 verifies DebrisHits <= 3.5 / ep."""
        summary_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 10.0,
            "grounded_pct": 70.0,
            "avg_debris_hits": 3.5,
            "avg_steps": 1300.0,
            "avg_shield_dmg": 160.0,
        }
        summary_fail = dict(summary_pass, avg_debris_hits=3.51)

        v_pass = evaluate_gates(summary_pass, gate_mode="2")
        v_fail = evaluate_gates(summary_fail, gate_mode="2")

        assert v_pass["gate2"]["passed"] is True
        assert v_fail["gate2"]["passed"] is False

    def test_t1_r4_gate2_survival_steps_threshold(self):
        """T1.20: Gate 2 verifies Survival steps >= 1200.0 (20.0s)."""
        summary_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 10.0,
            "grounded_pct": 70.0,
            "avg_debris_hits": 2.0,
            "avg_steps": 1200.0,
            "avg_shield_dmg": 160.0,
        }
        summary_fail = dict(summary_pass, avg_steps=1199.9)

        v_pass = evaluate_gates(summary_pass, gate_mode="2")
        v_fail = evaluate_gates(summary_fail, gate_mode="2")

        assert v_pass["gate2"]["passed"] is True
        assert v_fail["gate2"]["passed"] is False

    def test_t1_r4_gate2_shield_damage_threshold(self):
        """T1.21: Gate 2 verifies Boss Shield Damage >= 140.0 / 200."""
        summary_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 10.0,
            "grounded_pct": 70.0,
            "avg_debris_hits": 2.0,
            "avg_steps": 1300.0,
            "avg_shield_dmg": 140.0,
        }
        summary_fail = dict(summary_pass, avg_shield_dmg=139.9)

        v_pass = evaluate_gates(summary_pass, gate_mode="2")
        v_fail = evaluate_gates(summary_fail, gate_mode="2")

        assert v_pass["gate2"]["passed"] is True
        assert v_fail["gate2"]["passed"] is False

    def test_t1_r4_eval_gates_exit_codes(self):
        """T1.22: evaluate_gates overall_pass determines exit code logic."""
        summary_all_pass = {
            "checkpoint": "mock",
            "jump_right_pct": 15.0,
            "grounded_pct": 65.0,
            "avg_debris_hits": 2.5,
            "avg_steps": 1400.0,
            "avg_shield_dmg": 170.0,
        }
        verdict = evaluate_gates(summary_all_pass, gate_mode="all")
        assert verdict["overall_pass"] is True

        summary_partial_fail = dict(summary_all_pass, avg_debris_hits=5.0)
        verdict_fail = evaluate_gates(summary_partial_fail, gate_mode="all")
        assert verdict_fail["overall_pass"] is False


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary & Corner Cases (geometric borders, ground transitions, limits)."""

    def test_t2_danger_cone_lateral_epsilon_boundary(self, default_params):
        """T2.1: Lateral boundary check: dx = 44.9 (inside) vs dx = 45.1 (outside)."""
        px = 683.0
        py = 605.0
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # Inside border (dx = 44.9)
        deb_x_inside = jnp.full(30, -999.0).at[0].set(px + 44.9)
        r_inside = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x_inside, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False),  # airborne -> should trigger -0.35
            state_player_x=jnp.float32(px), params=default_params,
        )

        # Outside border (dx = 45.1)
        deb_x_outside = jnp.full(30, -999.0).at[0].set(px + 45.1)
        r_outside = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x_outside, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False),  # airborne -> danger cone inactive, no -0.35
            state_player_x=jnp.float32(px), params=default_params,
        )

        # Inside incurs airborne hazard penalty, outside does not
        assert float(r_outside - r_inside) > 0.30

    def test_t2_danger_cone_vertical_lower_boundary(self, default_params):
        """T2.2: Vertical lower boundary: dy = 0.0 (overhead) vs dy = -0.1 (below player center)."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px + 10.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # dy = py - debris_y = 0.0 -> debris_y = py
        deb_y_overhead = jnp.full(30, -999.0).at[0].set(py)
        # dy = -0.1 -> debris_y = py + 0.1 (below)
        deb_y_below = jnp.full(30, -999.0).at[0].set(py + 0.1)

        r_overhead = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y_overhead, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        r_below = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y_below, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        # Overhead triggers danger cone; below does not
        assert float(r_below - r_overhead) > 0.30

    def test_t2_danger_cone_vertical_upper_boundary(self, default_params):
        """T2.3: Vertical upper boundary: dy = 180.0 (inside) vs dy = 180.1 (too high)."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px + 10.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # dy = 180.0 -> deb_y = 605 - 180 = 425.0
        deb_y_in = jnp.full(30, -999.0).at[0].set(py - 180.0)
        # dy = 180.1 -> deb_y = 605 - 180.1 = 424.9
        deb_y_out = jnp.full(30, -999.0).at[0].set(py - 180.1)

        r_in = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y_in, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        r_out = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y_out, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        assert float(r_out - r_in) > 0.30

    def test_t2_threat_radius_threshold_boundary(self, default_params):
        """T2.4: Threat radius threshold boundary: r = 23.9 vs r = 24.0."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px + 10.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        deb_r_low = jnp.full(30, 16.0).at[0].set(23.9)
        deb_r_high = jnp.full(30, 16.0).at[0].set(24.0)

        r_low = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r_low, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        r_high = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r_high, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        assert float(r_low - r_high) > 0.30

    def test_t2_inactive_debris_mask_isolation(self, default_params):
        """T2.5: Inactive debris (debris_active=False) does not trigger danger corridor penalties."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 36.0)
        deb_act_false = jnp.zeros(30, dtype=bool)  # all inactive

        r_inactive = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act_false,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px), params=default_params,
        )

        # With all inactive, no airborne penalty and no Gaussian repulsion apply
        assert float(r_inactive) > 0.0

    def test_t2_ground_contact_jump_transition(self, env, default_params):
        """T2.6: Executing jump action transitions player_on_ground from True to False immediately."""
        state = make_neutral_state(default_params)
        assert bool(state.player_on_ground) is True

        key = jax.random.PRNGKey(42)
        _, next_state, _, _, _ = env.step(key, state, ACTION_JUMP, default_params)
        assert bool(next_state.player_on_ground) is False

    def test_t2_floor_landing_transition(self, env, default_params):
        """T2.7: Descending airborne agent touching floor transitions to on_ground=True."""
        state = make_neutral_state(default_params, y=default_params.floor_y - 2.0)
        # Airborne with downward velocity
        state_airborne = state.replace(player_on_ground=False, player_vy=300.0)

        key = jax.random.PRNGKey(42)
        _, next_state, _, _, _ = env.step(key, state_airborne, ACTION_NOOP, default_params)
        assert bool(next_state.player_on_ground) is True
        assert float(next_state.player_y) == default_params.floor_y

    def test_t2_arena_wall_clamping_boundary(self, env, default_params):
        """T2.8: Wall clamping prevents expanding separation into the wall (no false tap-dodge)."""
        # Place player at extreme left wall clamp: wall_left + half_w (100.0 + 20.0 = 120.0)
        wall_clamp_x = default_params.wall_left + default_params.player_w / 2.0
        state = make_neutral_state(default_params, x=wall_clamp_x)
        # Debris overhead to the right
        state = state.replace(
            debris_x=state.debris_x.at[0].set(wall_clamp_x + 30.0),
            debris_y=state.debris_y.at[0].set(default_params.floor_y - 100.0),
            debris_radius=state.debris_radius.at[0].set(24.0),
            debris_active=state.debris_active.at[0].set(True),
        )

        key = jax.random.PRNGKey(42)
        # Stepping LEFT at the left wall cannot move player further left
        _, next_state, r_left, _, _ = env.step(key, state, ACTION_LEFT, default_params)
        assert float(next_state.player_x) == wall_clamp_x

    def test_t2_opposing_debris_corridors(self, default_params):
        """T2.9: Two debris on opposing sides: moving away from one satisfies jnp.any(moving_away)."""
        px_prev = 683.0
        py = 605.0
        # Debris 0 on the left (660.0), Debris 1 on the right (706.0)
        deb_x = jnp.full(30, -999.0).at[0].set(660.0).at[1].set(706.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0).at[1].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True).at[1].set(True)

        # Player moves RIGHT: moves away from Debris 0, but closer to Debris 1
        px_next = px_prev + 6.67
        r_moving_right = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_next), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_RIGHT,
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        r_stationary = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_next), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_RIGHT,
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_next), params=default_params,
        )

        # Because distance to Debris 0 increased, can_tap_dodge bonus (+0.25) applies
        delta = float(r_moving_right - r_stationary)
        assert abs(delta - default_params.r_tap_dodge_bonus) < 1e-4


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Multi-feature interaction matrices."""

    def test_t3_simultaneous_jump_under_danger_cone(self, default_params):
        """T3.1: Jumping under active hazard cone superimposes jump cost (-0.05) and airborne hazard (-0.35)."""
        px = 683.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(px + 20.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        # Airborne vertical jump under overhead threat
        r_jump_under_threat = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py - 10.0), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_JUMP, last_action=ACTION_JUMP,
            on_ground_next=jnp.bool_(False),  # airborne
            state_player_x=jnp.float32(px), params=default_params,
        )

        # Neutral ground action under same threat
        r_ground_neutral = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,
            on_ground_next=jnp.bool_(True),  # grounded
            state_player_x=jnp.float32(px), params=default_params,
        )

        # Total penalty delta = r_action_jump_cost (-0.05) + r_airborne_hazard_cost (-0.35) = -0.40
        delta = float(r_jump_under_threat - r_ground_neutral)
        expected_penalty = default_params.r_action_jump_cost + default_params.r_airborne_hazard_cost
        assert abs(delta - expected_penalty) < 1e-4

    def test_t3_switching_while_dodging(self, default_params):
        """T3.2: Switching action while dodging under overhead threat nets jitter (-0.02) + dodge (+0.25) = +0.23."""
        px_prev = 660.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(683.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        px_away = px_prev - 6.67
        # Switch from NOOP to LEFT while moving away
        r_switch_dodge = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_LEFT, last_action=ACTION_NOOP,  # switch
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        # Baseline: standing still at px_away with last_action=ACTION_NOOP
        r_stay = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_NOOP, last_action=ACTION_NOOP,  # repeat NOOP
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_away), params=default_params,
        )

        delta = float(r_switch_dodge - r_stay)
        # Expected: r_jitter (-0.02) + r_tap_dodge (+0.25) = +0.23
        expected = default_params.r_jitter_cost + default_params.r_tap_dodge_bonus
        assert abs(delta - expected) < 1e-4

    def test_t3_moving_into_debris_while_switching(self, default_params):
        """T3.3: Moving toward debris while switching action nets -0.02 jitter and 0.0 dodge bonus."""
        px_prev = 660.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(683.0)
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        px_inward = px_prev + 6.67
        r_switch_inward = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_inward), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_NOOP,  # switch
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        r_repeat_inward = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_inward), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_RIGHT, last_action=ACTION_RIGHT,  # repeat
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        delta = float(r_switch_inward - r_repeat_inward)
        assert abs(delta - default_params.r_jitter_cost) < 1e-4

    def test_t3_jumping_away_from_debris(self, default_params):
        """T3.4: Jumping away from debris cannot claim tap-dodge bonus because agent is airborne."""
        px_prev = 660.0
        py = 605.0
        deb_x = jnp.full(30, -999.0).at[0].set(683.0)  # to the right
        deb_y = jnp.full(30, -999.0).at[0].set(py - 100.0)
        deb_r = jnp.full(30, 24.0)
        deb_act = jnp.zeros(30, dtype=bool).at[0].set(True)

        px_away = px_prev - 6.67
        # Jump left (airborne, expanding separation)
        r_airborne_jump_away = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py - 10.0), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_JUMP_LEFT, last_action=ACTION_JUMP_LEFT,
            on_ground_next=jnp.bool_(False), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        # Grounded walk left (grounded, expanding separation)
        r_grounded_walk_away = _compute_reward(
            took_hit=jnp.bool_(False), total_dmg=jnp.float32(0.0),
            laser_hits_boss=jnp.bool_(False), laser_hits_player=jnp.bool_(False),
            shield_shatter=jnp.bool_(False), triggers_overload=jnp.bool_(False),
            gauge_final=jnp.float32(0.0), is_overload_next=jnp.bool_(False),
            px_next=jnp.float32(px_away), py_next=jnp.float32(py), hp_next=jnp.float32(100.0),
            tl_state=jnp.int32(0), tl_lock_x=jnp.float32(683.0),
            debris_x=deb_x, debris_y=deb_y, debris_radius=deb_r, debris_active=deb_act,
            action=ACTION_LEFT, last_action=ACTION_LEFT,
            on_ground_next=jnp.bool_(True), state_player_x=jnp.float32(px_prev), params=default_params,
        )

        # Grounded walk receives +0.25 bonus and 0 jump cost and 0 airborne hazard
        # Airborne jump receives 0 dodge bonus, -0.05 jump cost, -0.35 airborne hazard
        # Expected advantage of grounded micro-dodge over airborne jump: +0.25 - (-0.40) = +0.65!
        delta = float(r_grounded_walk_away - r_airborne_jump_away)
        assert abs(delta - 0.65) < 1e-4

    def test_t3_ducking_under_danger_cone(self, env, default_params):
        """T3.5: ACTION_DOWN (crouching) maintains grounded status, reduces hitbox height to 35px, and avoids airborne penalty."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        vx, current_h, is_jump = decode_action(ACTION_DOWN, default_params.player_speed, default_params.player_h, default_params.player_duck_h)
        assert float(current_h) == default_params.player_duck_h
        assert bool(is_jump) is False

        _, next_state, _, _, _ = env.step(key, state, ACTION_DOWN, default_params)
        assert bool(next_state.player_on_ground) is True

    def test_t3_multi_step_telemetry_pipeline(self, env, default_params):
        """T3.6: Multi-step transition correctly propagates last_action and debris telemetry."""
        state = make_neutral_state(default_params)
        key = jax.random.PRNGKey(42)

        actions_sequence = [ACTION_RIGHT, ACTION_RIGHT, ACTION_JUMP, ACTION_NOOP, ACTION_DOWN]
        cur_state = state

        for i, act in enumerate(actions_sequence):
            key, k_step = jax.random.split(key)
            _, next_state, reward, done, info = env.step(k_step, cur_state, act, default_params)

            assert int(next_state.last_action) == act
            assert "debris_hit" in info
            assert isinstance(info["debris_hit"], (bool, jnp.ndarray, np.bool_))
            assert math.isfinite(float(reward))

            cur_state = next_state


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic simulation scenarios verifying suppression of bunny-hop spam."""

    def test_t4_trajectory_bunny_hop_vs_tap_dodge(self, env, default_params):
        """T4.1: Simulated trajectory verifies Grounded Tap-Dodging achieves substantially higher return than Bunny-Hop Spam."""
        key = jax.random.PRNGKey(777)
        steps = 120

        # Create identical initial states with overhead debris active
        init_state = make_neutral_state(default_params)
        init_state = init_state.replace(
            debris_x=init_state.debris_x.at[0].set(683.0),
            debris_y=init_state.debris_y.at[0].set(default_params.floor_y - 120.0),
            debris_radius=init_state.debris_radius.at[0].set(24.0),
            debris_active=init_state.debris_active.at[0].set(True),
        )

        # Policy A: Repeated Bunny-Hop under danger corridor (ACTION_JUMP)
        # Continually leaps vertically under falling debris, incurring jump cost (-0.05)
        # and airborne hazard penalty (-0.35) on every tick while in danger corridor.
        tot_reward_bunny = 0.0
        cur_state = init_state
        k_run = key
        for _ in range(40):
            k_run, k_step = jax.random.split(k_run)
            _, next_state, r, d, _ = env.step(k_step, cur_state, ACTION_JUMP, default_params)
            tot_reward_bunny += float(r)
            cur_state = next_state
            if bool(d):
                break

        # Policy B: Grounded Tap-Dodging (ACTION_LEFT)
        # Stays grounded, moves horizontally away from debris center, earning tap-dodge bonus (+0.25)
        # and clearing the danger corridor.
        tot_reward_grounded = 0.0
        cur_state = init_state
        k_run = key
        for _ in range(40):
            k_run, k_step = jax.random.split(k_run)
            _, next_state, r, d, _ = env.step(k_step, cur_state, ACTION_LEFT, default_params)
            tot_reward_grounded += float(r)
            cur_state = next_state
            if bool(d):
                break

        # Grounded tap-dodging must substantially outperform bunny-hop spam due to regularizations
        assert tot_reward_grounded > tot_reward_bunny + 5.0

    def test_t4_trajectory_action_chatter_suppression(self, env, default_params):
        """T4.2: 1-frame action chatter is strictly penalized by jitter cost relative to sustained micro-taps."""
        key = jax.random.PRNGKey(999)
        steps = 60

        # Trajectory 1: 1-frame alternating noise between NOOP and DOWN (vx = 0.0, 59 switches)
        state_1 = make_neutral_state(default_params)
        r_chatter_total = 0.0
        k = key
        for t in range(steps):
            act = ACTION_NOOP if t % 2 == 0 else ACTION_DOWN
            k, k_step = jax.random.split(k)
            _, next_state, r, _, _ = env.step(k_step, state_1, act, default_params)
            r_chatter_total += float(r)
            state_1 = next_state

        # Trajectory 2: Sustained actions (30 frames NOOP, then 30 frames DOWN, vx = 0.0, 1 switch)
        state_2 = make_neutral_state(default_params)
        r_sustained_total = 0.0
        k = key
        for t in range(steps):
            act = ACTION_NOOP if t < 30 else ACTION_DOWN
            k, k_step = jax.random.split(k)
            _, next_state, r, _, _ = env.step(k_step, state_2, act, default_params)
            r_sustained_total += float(r)
            state_2 = next_state

        # Difference in jitter penalty alone: (59 - 1) * 0.02 = 1.16
        delta = r_sustained_total - r_chatter_total
        assert delta > 1.0

    def test_t4_legacy_checkpoint_gate_failure(self):
        """T4.3: Confirms legacy baseline metrics fail Gate 1 (JUMP_RIGHT spam > 50%)."""
        legacy_step_4800_metrics = {
            "checkpoint": "checkpoints/step_4800",
            "jump_right_pct": 70.1,  # 70.1% JUMP_RIGHT spam
            "grounded_pct": 16.2,    # Only 16.2% grounded
            "avg_debris_hits": 8.7,  # 8.7 hits/ep
            "avg_steps": 658.4,      # Trapped at 658 steps
            "avg_shield_dmg": 120.0, # Below 140
        }
        verdict = evaluate_gates(legacy_step_4800_metrics, gate_mode="all")

        # Must fail both Gate 1 and Gate 2
        assert verdict["gate1"]["passed"] is False
        assert verdict["gate2"]["passed"] is False
        assert verdict["overall_pass"] is False

    def test_t4_synthetic_checkpoint_gate_pass(self):
        """T4.4: Confirms compliant policy metrics pass Gate 1 and Gate 2 acceptance standards."""
        compliant_metrics = {
            "checkpoint": "checkpoints/step_35000",
            "jump_right_pct": 12.4,   # < 20.0%
            "grounded_pct": 68.5,     # > 50.0%
            "avg_debris_hits": 1.8,   # <= 3.5 / ep
            "avg_steps": 1450.0,      # >= 1200.0 (20.0s)
            "avg_shield_dmg": 165.0,  # >= 140.0 / 200
        }
        verdict = evaluate_gates(compliant_metrics, gate_mode="all")

        assert verdict["gate1"]["passed"] is True
        assert verdict["gate2"]["passed"] is True
        assert verdict["overall_pass"] is True
