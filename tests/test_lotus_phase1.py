"""Unit tests for MapleStory Lotus Phase 1 Gymnax Environment.

Tests verify:
1. JIT compilation of env.reset_env and env.step_env with zero concretization errors.
2. jax.vmap batched execution across 1,024, 2,048, and 4,096 parallel environments.
3. Separating Axis Theorem (SAT) AABB projection and laser raycast dot-product masking.
4. Static padded array debris physics, Bernoulli PRNG spawning, and Euclidean collision.
5. Remastered Lotus mechanics: security gauge, overload mode, safe zone evasion, electric floor.
6. Friendly-fire boss guidance and shield interaction.
7. Episode life cycle: HP depletion termination and step count truncation.
8. Observation and action space boundaries and normalization.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from maple_gymnax.envs.common import (
    ACTION_DOWN,
    ACTION_DUCK,
    ACTION_JUMP,
    ACTION_JUMP_LEFT,
    ACTION_JUMP_RIGHT,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    MAX_DEBRIS,
    MODE_CLASSIC,
    MODE_HYBRID,
    MODE_REMASTERED,
    compute_debris_collisions,
    compute_laser_collisions,
    decode_action,
    sat_aabb_projection,
)
from maple_gymnax.envs.lotus_phase1 import (
    EnvParams,
    EnvState,
    LotusPhase1Env,
)


# =============================================================================
# 1. JIT & VMAP Parallelism Tests
# =============================================================================

class TestJITAndVMap:
    """Verifies XLA JIT and high-throughput vmap parallelism."""

    def test_jit_reset_compilation(self):
        """Verifies env.reset_env compiles cleanly under jax.jit without tracer errors."""
        env = LotusPhase1Env()
        params = env.default_params

        jit_reset = jax.jit(env.reset_env)
        key = jax.random.PRNGKey(42)
        obs, state = jit_reset(key, params)

        assert obs.shape == (130,)
        assert state.player_hp == params.player_max_hp
        assert state.time == 0
        assert bool(state.player_on_ground) is True

    def test_jit_step_compilation(self):
        """Verifies env.step_env compiles cleanly under jax.jit without branch errors."""
        env = LotusPhase1Env()
        params = env.default_params

        jit_step = jax.jit(env.step_env)
        key = jax.random.PRNGKey(101)
        _, init_state = env.reset_env(key, params)

        key, step_key = jax.random.split(key)
        obs, next_state, reward, done, info = jit_step(step_key, init_state, ACTION_RIGHT, params)

        assert obs.shape == (130,)
        assert next_state.time == 1
        assert isinstance(reward, (float, jax.Array))
        assert isinstance(done, (bool, jax.Array))
        assert "laser_hit" in info

    @pytest.mark.parametrize("batch_size", [1024, 2048, 4096])
    def test_vmap_across_parallel_environments(self, batch_size: int):
        """Verifies jax.vmap across 1024, 2048, and 4096 parallel environments."""
        env = LotusPhase1Env()
        params = env.default_params

        v_reset = jax.jit(jax.vmap(env.reset_env, in_axes=(0, None)))
        v_step = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))

        master_key = jax.random.PRNGKey(777)
        keys = jax.random.split(master_key, batch_size)

        # Vectorized reset
        batch_obs, batch_states = v_reset(keys, params)
        assert batch_obs.shape == (batch_size, 130)
        assert batch_states.player_x.shape == (batch_size,)
        assert batch_states.debris_active.shape == (batch_size, MAX_DEBRIS)

        # Vectorized step with diverse actions
        actions = jax.random.randint(master_key, shape=(batch_size,), minval=0, maxval=7)
        step_keys = jax.random.split(keys[0], batch_size)
        next_obs, next_states, rewards, dones, info = v_step(step_keys, batch_states, actions, params)

        assert next_obs.shape == (batch_size, 130)
        assert next_states.player_x.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert info["laser_hit"].shape == (batch_size,)


# =============================================================================
# 2. Player Kinematics, Boundaries, and Hitbox Tests
# =============================================================================

class TestPlayerKinematics:
    """Verifies 2D continuous kinematics, jumping, and arena boundaries."""

    def test_horizontal_movement_velocities(self):
        """Verifies left and right actions assign +/- player_speed."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Action LEFT (1)
        _, s_left, _, _, _ = env.step_env(key, state, ACTION_LEFT, params)
        assert s_left.player_vx == -params.player_speed

        # Action RIGHT (2)
        _, s_right, _, _, _ = env.step_env(key, state, ACTION_RIGHT, params)
        assert s_right.player_vx == params.player_speed

        # Action NOOP (0)
        _, s_noop, _, _, _ = env.step_env(key, state, ACTION_NOOP, params)
        assert s_noop.player_vx == 0.0

    def test_jump_impulse_and_gravity_integration(self):
        """Verifies vertical jump velocity and downward gravity acceleration."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)
        assert bool(state.player_on_ground) is True

        # Execute JUMP (3)
        _, s_jump, _, _, _ = env.step_env(key, state, ACTION_JUMP, params)
        expected_vy = params.jump_velocity + params.gravity * params.dt
        assert abs(s_jump.player_vy - expected_vy) < 1e-4
        assert bool(s_jump.player_on_ground) is False

    def test_ducking_hitbox_reduction(self):
        """Verifies DUCK (6) reduces player hitbox height to duck_h."""
        params = EnvParams()
        vx, h_norm, is_jump = decode_action(ACTION_NOOP, params.player_speed, params.player_h, params.player_duck_h)
        assert h_norm == 60.0
        assert is_jump == False

        vx, h_duck, is_jump = decode_action(ACTION_DUCK, params.player_speed, params.player_h, params.player_duck_h)
        assert h_duck == 35.0
        assert vx == 0.0

    def test_floor_landing_snapping(self):
        """Verifies player falling downwards snaps to floor baseline with zero velocity."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        airborne = state.replace(player_y=params.floor_y - 2.0, player_vy=300.0, player_on_ground=False)
        _, landed, _, _, _ = env.step_env(key, airborne, ACTION_NOOP, params)

        assert landed.player_y == params.floor_y
        assert landed.player_vy == 0.0
        assert bool(landed.player_on_ground) is True

    def test_wall_boundary_clamping(self):
        """Verifies player x-coordinate clamps at stage wall boundaries."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Exceed left boundary
        near_left = state.replace(player_x=params.wall_left)
        _, clamped_left, _, _, _ = env.step_env(key, near_left, ACTION_LEFT, params)
        assert clamped_left.player_x == params.wall_left + params.player_w / 2.0

        # Exceed right boundary
        near_right = state.replace(player_x=params.wall_right)
        _, clamped_right, _, _, _ = env.step_env(key, near_right, ACTION_RIGHT, params)
        assert clamped_right.player_x == params.wall_right - params.player_w / 2.0


# =============================================================================
# 3. Rotating Cross Laser & SAT Raycast Tests
# =============================================================================

class TestLaserCollision:
    """Verifies rotating cross laser kinematics, SAT projection, and directional masking."""

    def test_laser_angular_velocity_rotation(self):
        """Verifies laser angle rotates at omega * dt per step."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        init_angle = state.laser_angle
        _, next_s, _, _, _ = env.step_env(key, state, ACTION_NOOP, params)
        expected = jnp.mod(init_angle + params.laser_omega * params.dt, 2.0 * jnp.pi)
        assert abs(next_s.laser_angle - expected) < 1e-5

    def test_laser_directional_ray_masking(self):
        """Verifies negative longitudinal dot product (behind beam origin) does not hit.

        angle=0: Arm 0 points (+1,0). Player at core_x-200 is BEHIND Arm 0.
        Arm 2 points (-1,0) and WOULD hit that player, so we use a true safe
        quadrant (diagonal) where no arm overlaps.
        """
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # angle=0: arms point at 0°, 90°, 180°, 270°. Diagonal 45° is between arms.
        test_behind_arm0 = state.replace(
            laser_angle=0.0,
            player_x=params.core_x + 200.0,
            player_y=params.core_y + 200.0,  # 45° diagonal — clear of all arms
            invincible_timer=0.0,
        )
        _, next_s_behind, _, _, info_behind = env.step_env(
            key, test_behind_arm0, ACTION_NOOP, params
        )
        assert info_behind["laser_hit"] == False
        assert next_s_behind.player_hp == params.player_max_hp

        # Additional: player directly LEFT of core at angle=0 is on Arm 2 path —
        # verify it IS hit (confirms directional masking works both ways).
        test_arm2_path = state.replace(
            laser_angle=0.0,
            player_x=params.core_x - 300.0,
            player_y=params.core_y + params.player_h / 2.0,
            invincible_timer=0.0,
        )
        _, next_s_arm2, _, _, info_arm2 = env.step_env(
            key, test_arm2_path, ACTION_NOOP, params
        )
        assert info_arm2["laser_hit"] == True

    def test_laser_core_origin_clearance(self):
        """Verifies player inside core clearance radius is protected from laser."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Player at 20px from core center (core_radius = 70.0)
        inside_core = state.replace(
            laser_angle=0.0,
            player_x=params.core_x + 20.0,
            player_y=params.core_y + params.player_h / 2.0,
            invincible_timer=0.0,
        )
        _, next_s, _, _, info = env.step_env(key, inside_core, ACTION_NOOP, params)
        assert info["laser_hit"] == False

    def test_direct_laser_hit_lethal_damage(self):
        """Verifies direct laser contact deals 100% lethal damage and ends episode.

        Physical contract assertions (done + hp) are decoupled from reward scale
        so this test remains valid under future reward reshaping.
        """
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Place player center directly on Arm 0 ray path
        in_path = state.replace(
            laser_angle=0.0,
            player_x=params.core_x + 300.0,
            player_y=params.core_y + params.player_h / 2.0,
            invincible_timer=0.0,
        )
        _, next_s, reward, done, info = env.step_env(key, in_path, ACTION_NOOP, params)
        assert info["laser_hit"] == True
        assert next_s.player_hp == 0.0
        assert done == True

    def test_laser_disabled_in_remastered_mode(self):
        """Verifies classic cross laser is inactive when mode=MODE_REMASTERED."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        in_path = state.replace(
            laser_angle=0.0,
            player_x=params.core_x + 300.0,
            player_y=params.core_y + params.player_h / 2.0,
            invincible_timer=0.0,
        )
        _, next_s, _, _, info = env.step_env(key, in_path, ACTION_NOOP, params)
        assert info["laser_hit"] == False
        assert next_s.player_hp == params.player_max_hp

    def test_sat_aabb_projection_mathematics(self):
        """Verifies exact SAT projection values at cardinal and diagonal angles."""
        w, h = 40.0, 60.0
        # At theta = 0: normal is along y (sin=0, cos=1), projection = h / 2 = 30.0
        proj_0 = sat_aabb_projection(w / 2.0, h / 2.0, jnp.array(0.0), jnp.array(1.0))
        assert abs(proj_0 - 30.0) < 1e-5

        # At theta = pi / 2: normal is along x (sin=1, cos=0), projection = w / 2 = 20.0
        proj_90 = sat_aabb_projection(w / 2.0, h / 2.0, jnp.array(1.0), jnp.array(0.0))
        assert abs(proj_90 - 20.0) < 1e-5


# =============================================================================
# 4. Falling Debris Gimmick & Euclidean Collision Tests
# =============================================================================

class TestDebrisPhysics:
    """Verifies static array management, PRNG slot allocation, and collision."""

    def test_static_debris_array_shapes(self):
        """Verifies all static debris tensors have shape (MAX_DEBRIS = 30,)."""
        env = LotusPhase1Env()
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, env.default_params)

        assert state.debris_x.shape == (MAX_DEBRIS,)
        assert state.debris_y.shape == (MAX_DEBRIS,)
        assert state.debris_vy.shape == (MAX_DEBRIS,)
        assert state.debris_radius.shape == (MAX_DEBRIS,)
        assert state.debris_damage.shape == (MAX_DEBRIS,)
        assert state.debris_active.shape == (MAX_DEBRIS,)
        assert state.debris_type.shape == (MAX_DEBRIS,)

    def test_debris_slot_allocation_order(self):
        """Verifies debris fills the lowest available inactive slot index."""
        env = LotusPhase1Env()
        params = env.default_params.replace(debris_spawn_prob=1.0)
        key = jax.random.PRNGKey(42)
        _, state = env.reset_env(key, params)

        assert jnp.sum(state.debris_active) == 0
        _, s1, _, _, _ = env.step_env(key, state, ACTION_NOOP, params)
        assert s1.debris_active[0] == True
        assert jnp.sum(s1.debris_active) == 1

    def test_debris_floor_despawn(self):
        """Verifies falling debris contacting the floor is deactivated."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Position debris near floor
        deb_y = state.debris_y.at[0].set(params.floor_y - 2.0)
        deb_vy = state.debris_vy.at[0].set(300.0)
        deb_r = state.debris_radius.at[0].set(16.0)
        deb_act = state.debris_active.at[0].set(True)
        custom_state = state.replace(debris_y=deb_y, debris_vy=deb_vy, debris_radius=deb_r, debris_active=deb_act)

        _, next_s, _, _, _ = env.step_env(key, custom_state, ACTION_NOOP, params)
        assert next_s.debris_active[0] == False

    def test_euclidean_collision_inflicts_damage(self):
        """Verifies collision with active debris inflicts damage and deactivates debris."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Place debris on player center
        p_cx = state.player_x
        p_cy = state.player_y - params.player_h / 2.0

        deb_x = state.debris_x.at[0].set(p_cx)
        deb_y = state.debris_y.at[0].set(p_cy - 5.0)
        deb_vy = state.debris_vy.at[0].set(0.0)
        deb_r = state.debris_radius.at[0].set(20.0)
        deb_dmg = state.debris_damage.at[0].set(25.0)
        deb_act = state.debris_active.at[0].set(True)

        colliding_state = state.replace(
            debris_x=deb_x, debris_y=deb_y, debris_vy=deb_vy,
            debris_radius=deb_r, debris_damage=deb_dmg, debris_active=deb_act,
            invincible_timer=0.0,
        )

        _, next_s, _, _, info = env.step_env(key, colliding_state, ACTION_NOOP, params)
        assert info["debris_hit"] == True
        assert next_s.player_hp == params.player_max_hp - 25.0
        assert next_s.debris_active[0] == False
        assert next_s.invincible_timer == params.invincible_duration

    def test_invincibility_window_prevents_subsequent_damage(self):
        """Verifies invincibility timer shields player from damage.

        Pre-asserts laser_hit==True to confirm the scenario is valid before
        checking that hp remains unchanged.
        """
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Player is invincible
        inv_state = state.replace(invincible_timer=0.8)
        # Laser path
        in_path = inv_state.replace(
            laser_angle=0.0,
            player_x=params.core_x + 200.0,
            player_y=params.core_y + params.player_h / 2.0,
        )
        _, next_s, _, _, info = env.step_env(key, in_path, ACTION_NOOP, params)
        # Pre-assert: scenario is valid — laser DID fire at player
        assert info["laser_hit"] == True
        # Core assertion: invincibility absorbs the hit
        assert next_s.player_hp == params.player_max_hp
        assert next_s.invincible_timer < 0.8


# =============================================================================
# 5. Remastered Gimmicks: Gauge, Overload, Safe Zone, Electric Floor
# =============================================================================

class TestRemasteredLotus:
    """Verifies April 2024 Remastered gimmicks and friendly-fire mechanics."""

    def test_natural_security_gauge_accumulation(self):
        """Verifies gauge climbs naturally at gauge_gain_rate * dt in Remastered mode."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED, gauge_gain_rate=0.008)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        assert state.security_gauge == 0.0
        _, next_s, _, _, _ = env.step_env(key, state, ACTION_NOOP, params)
        expected = params.gauge_gain_rate * params.dt
        assert abs(next_s.security_gauge - expected) < 1e-6

    def test_overload_trigger_and_reset(self):
        """Verifies gauge reaching 100% triggers Overload mode and resets gauge to 0."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        full_gauge = state.replace(security_gauge=1.0)
        _, next_s, _, _, _ = env.step_env(key, full_gauge, ACTION_NOOP, params)

        assert next_s.is_overload == True
        assert next_s.overload_timer == params.overload_duration
        assert next_s.security_gauge == 0.0

    def test_overload_safe_zone_evasion(self):
        """Verifies player outside safe zone receives artillery damage; inside safe zone survives."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Danger zone (x < 1150) during overload
        danger_s = state.replace(is_overload=True, overload_timer=20.0, player_x=600.0, invincible_timer=0.0)
        _, danger_next, _, done, _ = env.step_env(key, danger_s, ACTION_NOOP, params)
        assert danger_next.player_hp == 0.0
        assert done == True

        # Safe zone (x >= 1150) during overload
        safe_s = state.replace(is_overload=True, overload_timer=20.0, player_x=1200.0, invincible_timer=0.0)
        _, safe_next, _, done_safe, _ = env.step_env(key, safe_s, ACTION_NOOP, params)
        assert safe_next.player_hp == params.player_max_hp
        assert done_safe == False

    def test_electric_floor_lethal_burst_and_jump_evasion(self):
        """Verifies electric floor burst hits grounded player, but airborne player evades."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # Grounded player during detonation
        grounded_burst = state.replace(
            electric_floor_active=True,
            electric_floor_warning=0.0,
            player_y=params.floor_y,
            player_on_ground=True,
            invincible_timer=0.0,
        )
        _, grounded_next, _, _, _ = env.step_env(key, grounded_burst, ACTION_NOOP, params)
        assert grounded_next.player_hp == 0.0

        # Airborne jumping player during detonation
        airborne_burst = state.replace(
            electric_floor_active=True,
            electric_floor_warning=0.0,
            player_y=params.floor_y - 30.0,
            player_on_ground=False,
            invincible_timer=0.0,
        )
        _, airborne_next, _, _, _ = env.step_env(key, airborne_burst, ACTION_NOOP, params)
        assert airborne_next.player_hp == params.player_max_hp

    def test_friendly_fire_boss_guidance_parameters(self):
        """Verifies tracking laser and arm slam friendly fire reduction constants."""
        params = EnvParams()
        assert params.tracking_laser_gauge_reduction == 0.10
        assert params.arm_slam_gauge_reduction == 0.03
        assert params.tracking_laser_player_dmg == 15.0
        assert params.arm_slam_player_dmg == 5.0


# =============================================================================
# 6. Episode Termination and Observation Space Tests
# =============================================================================

class TestEpisodeLifecycle:
    """Verifies termination, truncation, and observation normalization."""

    def test_player_death_termination(self):
        """Verifies player HP reaching 0.0 sets is_terminal and done to True."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        dead_state = state.replace(player_hp=0.0)
        assert env.is_terminal(dead_state, params) is True

    def test_step_count_truncation(self):
        """Verifies reaching max_steps_in_episode (3600) sets done to True."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        timeout_state = state.replace(time=params.max_steps_in_episode)
        assert env.is_terminal(timeout_state, params) is True

    def test_observation_space_and_normalization(self):
        """Verifies 130-dim observation values fall strictly within [-1.0, 1.0]."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(123)
        obs, _ = env.reset_env(key, params)

        assert obs.shape == (130,)
        assert jnp.all(obs >= -1.0)
        assert jnp.all(obs <= 1.0)

    def test_extended_observation_shape_172(self):
        """Verifies extended observation returns 172 dimensions for Remastered RL (Rule 17)."""
        env = LotusPhase1Env()
        params = env.default_params
        key = jax.random.PRNGKey(123)
        _, state = env.reset_env(key, params)

        ext_obs = env.get_extended_obs(state, params)
        assert ext_obs.shape == (172,)

    def test_mode_observation_space_and_step_binding(self):
        """Verifies observation_space, reset_env, and step_env shapes match exactly across all modes."""
        env = LotusPhase1Env()
        key = jax.random.PRNGKey(999)

        # 1. Classic Mode (mode=0) -> 130-dim
        p_classic = env.default_params.replace(mode=0)
        assert p_classic.is_remastered is False
        assert env.observation_space(p_classic).shape == (130,)
        obs_c, state_c = env.reset_env(key, p_classic)
        assert obs_c.shape == (130,)
        next_obs_c, _, _, _, _ = env.step_env(key, state_c, 0, p_classic)
        assert next_obs_c.shape == (130,)

        # 2. Remastered Mode (mode=1) -> 172-dim
        p_remaster = env.default_params.replace(mode=1)
        assert p_remaster.is_remastered is True
        assert env.observation_space(p_remaster).shape == (172,)
        obs_r, state_r = env.reset_env(key, p_remaster)
        assert obs_r.shape == (172,)
        next_obs_r, _, _, _, _ = env.step_env(key, state_r, 0, p_remaster)
        assert next_obs_r.shape == (172,)

        # 3. Hybrid Mode (mode=2) -> 172-dim
        p_hybrid = env.default_params.replace(mode=2)
        assert p_hybrid.is_remastered is True
        assert env.observation_space(p_hybrid).shape == (172,)
        obs_h, state_h = env.reset_env(key, p_hybrid)
        assert obs_h.shape == (172,)
        next_obs_h, _, _, _, _ = env.step_env(key, state_h, 0, p_hybrid)
        assert next_obs_h.shape == (172,)

    def test_reward_shaping_gauge_and_safe_zone_alignment(self):
        """Verifies anti-reward-hacking shaping: gauge penalty and safe-zone positioning."""
        env = LotusPhase1Env()
        key = jax.random.PRNGKey(555)

        # In Remastered mode, higher gauge yields higher penalty
        p_remaster = env.default_params.replace(mode=1, gauge_gain_rate=0.0)
        _, s_low = env.reset_env(key, p_remaster)
        s_low = s_low.replace(security_gauge=0.0, invincible_timer=10.0)
        s_high = s_low.replace(security_gauge=0.8, invincible_timer=10.0)

        _, _, r_low, _, _ = env.step_env(key, s_low, 0, p_remaster)
        _, _, r_high, _, _ = env.step_env(key, s_high, 0, p_remaster)
        # Higher gauge has extra penalty (-0.08 * 0.8 = -0.064)
        assert r_low > r_high
        # Direct coefficient assertion: penalty difference must match r_gauge = -0.08 * gauge
        assert abs(float(r_low) - float(r_high) - 0.08 * 0.8) < 1e-4

        # During overload mode, standing in safe zone gives bonus vs danger zone
        s_safe = s_low.replace(
            is_overload=True,
            overload_timer=10.0,
            player_x=p_remaster.safe_zone_x + 50.0,
            invincible_timer=10.0,
        )
        s_danger = s_low.replace(
            is_overload=True,
            overload_timer=10.0,
            player_x=p_remaster.safe_zone_x - 200.0,
            invincible_timer=10.0,
        )
        _, _, r_safe, _, _ = env.step_env(key, s_safe, 0, p_remaster)
        _, _, r_danger, _, _ = env.step_env(key, s_danger, 0, p_remaster)
        # Safe zone (+0.3) vs danger zone (-0.3) -> difference = 0.6
        assert r_safe > r_danger
        assert (r_safe - r_danger) >= 0.35

    def test_boss_hit_reward_is_single_event_not_per_tick(self):
        """P0 회귀: 발사 지속 30틱 동안 전체 reward의 boss_hit 성분 합계가 정확히 50.0이어야 한다."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_REMASTERED)
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)

        # FIRING 진입 직전 상태로 설정: TRACKING(1) 상태, 타이머 0 임박, lock_x가 보스 판정 구간
        boss_center = params.core_x
        setup_state = state.replace(
            tracking_laser_state=1,
            tracking_laser_timer=0.001,
            tracking_laser_lock_x=boss_center,
            shield_active=True,
            boss_shield=params.boss_shield_max,
        )

        total_boss_hit_reward = 0.0
        cur = setup_state
        for _ in range(35):  # FIRING 지속시간(0.5s=30틱)보다 넉넉히
            key, subk = jax.random.split(key)
            obs, cur, reward, done, info = env.step_env(subk, cur, 0, params)
            if info["boss_hit_event"]:
                total_boss_hit_reward += 50.0

        assert total_boss_hit_reward == 50.0, (
            f"보스 적중 보상이 1회성이 아닙니다: 누적 {total_boss_hit_reward}"
        )


# =============================================================================
# 9. R1 & R2: Action Cost, Jitter Regularization & Threat-Gated Evasion Rewards
# =============================================================================

class TestMicroMovementRewardShapingR1R2:
    """Verifies analytical reward deltas for R1 and R2 micro-movement shaping."""

    def test_action_jump_cost_analytical_delta(self):
        """R1: Verifies jump action cost imparts exact -0.05 delta on actions {4, 5, 6} vs ground actions."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_CLASSIC)
        key = jax.random.PRNGKey(42)
        _, init_state = env.reset_env(key, params)

        # Baseline: Ground action (NOOP, 0) with matching last_action=0 (no jitter penalty)
        s_noop = init_state.replace(last_action=ACTION_NOOP, debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool))
        _, _, r_noop, _, _ = env.step_env(key, s_noop, ACTION_NOOP, params)

        # Jump action (JUMP, 4) with matching last_action=4 (no jitter penalty)
        s_jump = init_state.replace(last_action=ACTION_JUMP, debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool))
        _, _, r_jump, _, _ = env.step_env(key, s_jump, ACTION_JUMP, params)

        # Delta must match r_action_jump_cost = -0.05 analytically
        # r_noop - r_jump = 0.0 - (-0.05) = +0.05
        assert abs((float(r_noop) - float(r_jump)) - 0.05) < 1e-4

        # Jump Left (5) and Jump Right (6) also incur exact -0.05 action cost
        s_jl = init_state.replace(last_action=ACTION_JUMP_LEFT, debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool))
        _, _, r_jl, _, _ = env.step_env(key, s_jl, ACTION_JUMP_LEFT, params)
        assert abs((float(r_noop) - float(r_jl)) - 0.05) < 1e-4

        s_jr = init_state.replace(last_action=ACTION_JUMP_RIGHT, debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool))
        _, _, r_jr, _, _ = env.step_env(key, s_jr, ACTION_JUMP_RIGHT, params)
        assert abs((float(r_noop) - float(r_jr)) - 0.05) < 1e-4

    def test_jitter_regularization_analytical_delta(self):
        """R1: Verifies switching action incurs exact r_jitter_cost = -0.02 delta."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_CLASSIC)
        key = jax.random.PRNGKey(42)
        _, init_state = env.reset_env(key, params)

        # Compare repeating NOOP (last_action=0, action=0) vs switching to DOWN (last_action=0, action=3)
        # Both are ground actions with zero horizontal movement (vx=0, px_next=px), zero jump cost
        s_base = init_state.replace(last_action=ACTION_NOOP, debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool))
        _, _, r_same, _, _ = env.step_env(key, s_base, ACTION_NOOP, params)
        _, _, r_switch, _, _ = env.step_env(key, s_base, ACTION_DOWN, params)

        # Difference must match r_jitter_cost = -0.02
        # r_same - r_switch = 0.0 - (-0.02) = +0.02
        assert abs((float(r_same) - float(r_switch)) - 0.02) < 1e-4

    def test_overhead_airborne_hazard_penalty_analytical_delta(self):
        """R2: Verifies jumping under high-threat overhead debris incurs -0.35 anti-jump penalty."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_CLASSIC)
        key = jax.random.PRNGKey(42)
        _, init_state = env.reset_env(key, params)

        # Setup high-threat debris (radius=24.0, Type 1) directly overhead:
        # Player at floor_y=605.0, debris at x=player_x, y=500.0 (dy = 605 - 500 = 105 in [0, 180], dx = 0 < 45)
        deb_x = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(init_state.player_x)
        deb_y = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(500.0)
        deb_r = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(24.0)
        deb_act = jnp.zeros(MAX_DEBRIS, dtype=bool).at[0].set(True)

        s_hazard = init_state.replace(
            debris_x=deb_x,
            debris_y=deb_y,
            debris_radius=deb_r,
            debris_active=deb_act,
            debris_vy=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),  # stationary debris for pure reward isolation
            invincible_timer=10.0,  # avoid damage penalty
        )

        # When player remains grounded (NOOP, last_action=NOOP):
        # has_overhead_threat is True, but on_ground_next is True -> r_airborne_hazard = 0.0
        _, _, r_grounded, _, _ = env.step_env(key, s_hazard.replace(last_action=ACTION_NOOP), ACTION_NOOP, params)

        # When player jumps (JUMP, last_action=JUMP):
        # has_overhead_threat is True, on_ground_next is False -> r_airborne_hazard = -0.35, r_action_jump = -0.05
        _, _, r_jump, _, _ = env.step_env(key, s_hazard.replace(last_action=ACTION_JUMP), ACTION_JUMP, params)

        # Total delta between grounded and jumping under overhead hazard:
        # r_grounded - r_jump = 0.0 - (-0.05 + -0.35) = +0.40
        assert abs((float(r_grounded) - float(r_jump)) - 0.40) < 1e-4

        # Net airborne hazard penalty contribution is strictly 0.35:
        # (r_grounded - r_jump) - jump_cost = 0.40 - 0.05 = 0.35
        hazard_delta = (float(r_grounded) - float(r_jump)) - abs(params.r_action_jump_cost)
        assert abs(hazard_delta - abs(params.r_airborne_hazard_cost)) < 1e-4

    def test_tap_dodge_clearance_bonus_analytical_delta(self):
        """R2: Verifies grounded lateral movement expanding separation from overhead hazard awards +0.25 bonus."""
        env = LotusPhase1Env()
        params = env.default_params.replace(mode=MODE_CLASSIC)
        key = jax.random.PRNGKey(42)
        _, init_state = env.reset_env(key, params)

        # Setup overhead threat slightly to the left of the player:
        # Player at x=600.0, debris at x=590.0 (dx_prev = 10.0 < 45.0, dy = 105.0)
        p_x = 600.0
        deb_x = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(590.0)
        deb_y = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(500.0)
        deb_r = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(24.0)
        deb_act = jnp.zeros(MAX_DEBRIS, dtype=bool).at[0].set(True)

        s_dodge_setup = init_state.replace(
            player_x=p_x,
            debris_x=deb_x,
            debris_y=deb_y,
            debris_radius=deb_r,
            debris_active=deb_act,
            debris_vy=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            invincible_timer=10.0,
        )

        # Case 1: Moving away to the RIGHT (last_action=RIGHT, action=RIGHT)
        # vx = +400, dx_next = |606.67 - 590| = 16.67 > dx_prev (10.0)
        # on_ground_next = True -> can_tap_dodge = True -> r_tap_dodge = +0.25
        s_right = s_dodge_setup.replace(last_action=ACTION_RIGHT)
        _, _, r_dodge_away, _, _ = env.step_env(key, s_right, ACTION_RIGHT, params)

        # Case 2: Moving closer to the LEFT (last_action=LEFT, action=LEFT)
        # vx = -400, dx_next = |593.33 - 590| = 3.33 < dx_prev (10.0)
        # moving_away = False -> can_tap_dodge = False -> r_tap_dodge = 0.0
        s_left = s_dodge_setup.replace(last_action=ACTION_LEFT)
        _, _, r_move_toward, _, _ = env.step_env(key, s_left, ACTION_LEFT, params)

        # Case 3: Stationary (last_action=NOOP, action=NOOP)
        # dx_next = dx_prev = 10.0 (not > dx_prev) -> can_tap_dodge = False -> r_tap_dodge = 0.0
        s_noop = s_dodge_setup.replace(last_action=ACTION_NOOP)
        _, _, r_stationary, _, _ = env.step_env(key, s_noop, ACTION_NOOP, params)

        # Tap-dodge clearance bonus delta:
        # In classic mode, r_dodge_away has +0.25 bonus compared to stationary or moving toward
        assert abs((float(r_dodge_away) - float(r_move_toward)) - 0.25) < 1e-4
        assert abs((float(r_dodge_away) - float(r_stationary)) - 0.25) < 1e-4

    def test_debris_repel_scale_gaussian_potential_delta(self):
        """R2: Verifies continuous overhead Gaussian potential scales with debris_repel_scale = -0.30."""
        env = LotusPhase1Env()
        # Repulsion potential is active in Remastered / Hybrid mode (params.mode != MODE_CLASSIC)
        key = jax.random.PRNGKey(42)

        # Debris directly overhead at dx = 0 (exp(-0) = 1.0), large debris r = 36.0 (r/36 = 1.0)
        # overhead_weight = 1.0
        p_x = 683.0  # center arena
        deb_x = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(p_x)
        deb_y = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(500.0)
        deb_r = jnp.zeros(MAX_DEBRIS, dtype=jnp.float32).at[0].set(36.0)
        deb_act = jnp.zeros(MAX_DEBRIS, dtype=bool).at[0].set(True)

        params_scaled = env.default_params.replace(mode=MODE_REMASTERED, debris_repel_scale=-0.30)
        params_zero = env.default_params.replace(mode=MODE_REMASTERED, debris_repel_scale=0.0)

        _, init_state = env.reset_env(key, params_scaled)
        s_test = init_state.replace(
            player_x=p_x,
            debris_x=deb_x,
            debris_y=deb_y,
            debris_radius=deb_r,
            debris_active=deb_act,
            debris_vy=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            invincible_timer=10.0,
            last_action=ACTION_NOOP,
        )

        _, _, r_scaled, _, _ = env.step_env(key, s_test, ACTION_NOOP, params_scaled)
        _, _, r_zero, _, _ = env.step_env(key, s_test, ACTION_NOOP, params_zero)

        # Delta must exactly match params.debris_repel_scale * overhead_weight = -0.30 * 1.0 = -0.30
        assert abs((float(r_scaled) - float(r_zero)) - (-0.30)) < 1e-4

