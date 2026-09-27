"""MapleStory Lotus Phase 1 Gymnax Environment.

Implements Lotus Phase 1 (Classic rotating cross laser and falling debris,
Remastered April 2024 security gauge, overload mode, electric floor,
and friendly fire boss guidance, and Hybrid mode) running on JAX/XLA.

Features:
- Complete branch-free JIT compilation via jnp.where and jax.lax.select.
- Static padded array memory layout for falling debris (MAX_DEBRIS=30).
- Pure functional state transitions with immutable EnvParams and EnvState.
- Full compatibility with Gymnax 0.0.9 interface.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import chex
import flax.struct
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp

from maple_gymnax.envs.common import (
    MAX_DEBRIS,
    MODE_CLASSIC,
    MODE_REMASTERED,
    MODE_HYBRID,
    compute_laser_collisions,
    compute_debris_collisions,
    decode_action,
)


# ---------------------------------------------------------------------------
# 1. Environment Hyperparameters & State Schema
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class EnvParams:
    """Immutable environment hyperparameters for MapleStory Lotus Phase 1."""
    # Screen & Arena Dimensions
    screen_width: float = 1366.0
    screen_height: float = 768.0
    core_x: float = 683.0
    core_y: float = 384.0
    core_radius: float = 70.0
    floor_y: float = 605.0
    wall_left: float = 100.0
    wall_right: float = 1266.0
    safe_zone_x: float = 1150.0

    # Player Kinematics
    player_w: float = 40.0
    player_h: float = 60.0
    player_duck_h: float = 35.0
    player_speed: float = 400.0
    jump_velocity: float = -650.0
    gravity: float = 1800.0
    dt: float = 1.0 / 60.0
    player_max_hp: float = 100.0
    invincible_duration: float = 1.0

    # Rotating Cross Laser
    laser_omega: float = 0.5235
    laser_half_thickness: float = 12.0
    laser_damage: float = 100.0
    laser_max_length: float = 1200.0

    # Falling Debris Gimmick
    debris_spawn_prob: float = 0.15
    debris_min_vy: float = 200.0
    debris_max_vy: float = 450.0
    debris_player_eff_radius: float = 25.0

    # Episode Life Cycle
    max_steps_in_episode: int = 3600

    # Remastered Lotus Mechanics
    mode: int = flax.struct.field(pytree_node=False, default=0)  # 0: Classic, 1: Remastered, 2: Hybrid
    gauge_gain_rate: float = 0.008  # 0.8% per second in hard mode
    overload_duration: float = 25.0
    boss_max_hp: float = 1000.0
    boss_shield_max: float = 200.0
    boss_w: float = 160.0
    electric_field_damage: float = 5.0
    tracking_laser_player_dmg: float = 15.0
    tracking_laser_gauge_gain: float = 0.10
    tracking_laser_gauge_reduction: float = 0.10
    arm_slam_player_dmg: float = 5.0
    arm_slam_gauge_gain: float = 0.03
    arm_slam_gauge_reduction: float = 0.03

    # -----------------------------------------------------------------------
    # Convenience properties for PROJECT.md / schema.py contract compatibility
    # -----------------------------------------------------------------------
    @property
    def map_width(self) -> float:
        return self.screen_width

    @property
    def map_height(self) -> float:
        return self.screen_height

    @property
    def player_width(self) -> float:
        return self.player_w

    @property
    def player_height(self) -> float:
        return self.player_h

    @property
    def player_jump_impulse(self) -> float:
        return self.jump_velocity

    @property
    def jump_impulse(self) -> float:
        return abs(self.jump_velocity)

    @property
    def core_pos(self) -> Tuple[float, float]:
        return (self.core_x, self.core_y)

    @property
    def laser_thickness(self) -> float:
        return self.laser_half_thickness * 2.0

    @property
    def max_debris(self) -> int:
        return MAX_DEBRIS

    @property
    def is_remastered(self) -> bool:
        """True if Remastered (1) or Hybrid (2) mechanics are active."""
        return self.mode != MODE_CLASSIC


@flax.struct.dataclass
class EnvState:
    """Dynamic simulation state at a 60 Hz tick."""
    # Player Kinematic & Health State
    player_x: float
    player_y: float
    player_vx: float
    player_vy: float
    player_hp: float
    player_on_ground: bool
    invincible_timer: float

    # Classic Rotating Cross Laser State
    laser_angle: float

    # Falling Debris Static Padded Arrays (Capacity = 30)
    debris_x: chex.Array
    debris_y: chex.Array
    debris_vy: chex.Array
    debris_radius: chex.Array
    debris_damage: chex.Array
    debris_active: chex.Array
    debris_type: chex.Array

    # Temporal & Episode Tracking
    time: int

    # Remastered Gimmick States (April 2024 Remake)
    security_gauge: float = 0.0
    is_overload: bool = False
    overload_timer: float = 0.0
    boss_hp: float = 1000.0
    boss_shield: float = 200.0
    shield_active: bool = True
    tracking_laser_timer: float = 0.0
    tracking_laser_lock_x: float = 683.0
    tracking_laser_state: int = 0  # 0: idle, 1: tracking, 2: firing
    electric_floor_warning: float = 0.0
    electric_floor_active: bool = False


# ---------------------------------------------------------------------------
# 2. Pure Modular Helper Functions for Simulation Step (XLA Inlineable)
# ---------------------------------------------------------------------------

def _step_player_kinematics(
    state: EnvState,
    action: Union[int, chex.Array],
    params: EnvParams,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Pure functional player kinematic integration, jump mechanics, and arena boundary clamping."""
    vx, current_h, is_jump_action = decode_action(
        action=action,
        player_speed=params.player_speed,
        standard_h=params.player_h,
        duck_h=params.player_duck_h,
    )

    can_jump = state.player_on_ground & is_jump_action
    vy_start = jnp.where(can_jump, params.jump_velocity, state.player_vy)
    vy_next = vy_start + params.gravity * params.dt

    px_cand = state.player_x + vx * params.dt
    px_next = jnp.clip(
        px_cand,
        params.wall_left + params.player_w / 2.0,
        params.wall_right - params.player_w / 2.0,
    )

    py_cand = state.player_y + vy_next * params.dt
    hits_floor = py_cand >= params.floor_y
    py_next = jnp.where(hits_floor, params.floor_y, py_cand)
    vy_final = jnp.where(hits_floor, 0.0, vy_next)
    on_ground_next = hits_floor

    p_center_x = px_next
    p_center_y = py_next - current_h / 2.0
    return px_next, py_next, vx, vy_final, on_ground_next, p_center_x, p_center_y, current_h


def _step_classic_laser(
    state: EnvState,
    p_center_x: chex.Array,
    p_center_y: chex.Array,
    current_h: chex.Array,
    params: EnvParams,
) -> Tuple[chex.Array, chex.Array]:
    """Pure functional rotating cross laser state update and SAT AABB collision detection."""
    laser_angle_next = jnp.mod(state.laser_angle + params.laser_omega * params.dt, 2.0 * jnp.pi)
    classic_laser_hit = compute_laser_collisions(
        player_center_x=p_center_x,
        player_center_y=p_center_y,
        player_w=params.player_w,
        player_h=current_h,
        core_x=params.core_x,
        core_y=params.core_y,
        core_radius=params.core_radius,
        laser_max_length=params.laser_max_length,
        laser_angle=laser_angle_next,
        laser_half_thickness=params.laser_half_thickness,
    ) & (params.mode != MODE_REMASTERED)
    return laser_angle_next, classic_laser_hit


def _step_falling_debris(
    key_spawn: chex.PRNGKey,
    state: EnvState,
    p_center_x: chex.Array,
    p_center_y: chex.Array,
    params: EnvParams,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Pure functional falling debris spawning, floor cleanup, and vectorized Euclidean contact."""
    k_prob, k_x, k_vy, k_t = jax.random.split(key_spawn, 4)
    should_spawn = jax.random.bernoulli(k_prob, params.debris_spawn_prob)
    inactive = ~state.debris_active
    slot_idx = jnp.argmax(inactive)
    can_spawn = should_spawn & jnp.any(inactive)

    cand_x = jax.random.uniform(k_x, minval=params.wall_left + 50.0, maxval=params.wall_right - 50.0)
    cand_vy = jax.random.uniform(k_vy, minval=params.debris_min_vy, maxval=params.debris_max_vy)
    type_idx = jax.random.randint(k_t, shape=(), minval=0, maxval=3)
    radii = jnp.array([16.0, 24.0, 36.0])
    damages = jnp.array([10.0, 20.0, 30.0])

    slot_mask = (jnp.arange(MAX_DEBRIS) == slot_idx) & can_spawn
    cur_deb_x = jnp.where(slot_mask, cand_x, state.debris_x)
    cur_deb_y = jnp.where(slot_mask, 0.0, state.debris_y)
    cur_deb_vy = jnp.where(slot_mask, cand_vy, state.debris_vy)
    cur_deb_r = jnp.where(slot_mask, radii[type_idx], state.debris_radius)
    cur_deb_dmg = jnp.where(slot_mask, damages[type_idx], state.debris_damage)
    cur_deb_t = jnp.where(slot_mask, type_idx, state.debris_type)
    cur_deb_act = jnp.where(slot_mask, True, state.debris_active)

    deb_y_next = cur_deb_y + cur_deb_vy * params.dt
    hit_floor_deb = deb_y_next >= (params.floor_y - cur_deb_r)
    deb_act_fall = cur_deb_act & (~hit_floor_deb)

    deb_hit_player = compute_debris_collisions(
        player_center_x=p_center_x,
        player_center_y=p_center_y,
        player_eff_radius=params.debris_player_eff_radius,
        debris_x=cur_deb_x,
        debris_y=deb_y_next,
        debris_radius=cur_deb_r,
        debris_active=deb_act_fall,
    )
    deb_act_next = deb_act_fall & (~deb_hit_player)
    debris_damage_total = jnp.sum(jnp.where(deb_hit_player, cur_deb_dmg, 0.0))

    return cur_deb_x, deb_y_next, cur_deb_vy, cur_deb_r, cur_deb_dmg, deb_act_next, cur_deb_t, debris_damage_total, deb_hit_player


def _step_remastered_mechanics(
    state: EnvState,
    px_next: chex.Array,
    py_next: chex.Array,
    on_ground_next: chex.Array,
    time_next: chex.Array,
    params: EnvParams,
) -> Dict[str, Any]:
    """Pure functional Remastered Lotus mechanics (Tracking Laser FSM, Boss Shield, Gauge, Overload, Electric Floor)."""
    is_remastered_or_hybrid = jnp.bool_(params.mode != MODE_CLASSIC)
    is_classic = jnp.bool_(params.mode == MODE_CLASSIC)
    gain_rate = jnp.where(is_remastered_or_hybrid, params.gauge_gain_rate, 0.0)

    # 1. Tracking Laser FSM: 0: IDLE (cooldown), 1: TRACKING (aiming), 2: FIRING (locked)
    tl_timer_dec = state.tracking_laser_timer - params.dt
    idle_to_tracking = (state.tracking_laser_state == 0) & (tl_timer_dec <= 0.0) & is_remastered_or_hybrid
    tracking_to_firing = (state.tracking_laser_state == 1) & (tl_timer_dec <= 0.0) & is_remastered_or_hybrid
    firing_to_idle = (state.tracking_laser_state == 2) & (tl_timer_dec <= 0.0) & is_remastered_or_hybrid

    tl_state_next = jnp.where(
        is_classic,
        state.tracking_laser_state,
        jnp.where(
            idle_to_tracking,
            1,
            jnp.where(
                tracking_to_firing,
                2,
                jnp.where(firing_to_idle, 0, state.tracking_laser_state),
            ),
        ),
    )

    tl_timer_next = jnp.where(
        is_classic,
        state.tracking_laser_timer,
        jnp.where(
            idle_to_tracking,
            1.0,  # 1.0s tracking duration
            jnp.where(
                tracking_to_firing,
                0.5,  # 0.5s firing duration
                jnp.where(firing_to_idle, 6.0, jnp.maximum(0.0, tl_timer_dec)),
            ),
        ),
    )

    # Tracking follows player position; Firing maintains locked coordinate
    tl_lock_x_next = jnp.where(
        (tl_state_next == 1),
        px_next,
        state.tracking_laser_lock_x,
    )

    # Collision evaluation during FIRING state
    is_firing = (tl_state_next == 2) & is_remastered_or_hybrid
    just_entered_firing = tracking_to_firing
    boss_left = params.core_x - params.boss_w / 2.0
    boss_right = params.core_x + params.boss_w / 2.0
    laser_hits_boss = is_firing & (state.tracking_laser_lock_x >= boss_left) & (state.tracking_laser_lock_x <= boss_right)

    p_left = px_next - params.player_w / 2.0
    p_right = px_next + params.player_w / 2.0
    laser_hits_player = is_firing & (state.tracking_laser_lock_x >= p_left) & (state.tracking_laser_lock_x <= p_right)

    # Boss Shield shattering via friendly fire
    shield_shatter = laser_hits_boss & state.shield_active
    boss_shield_next = jnp.where(shield_shatter, 0.0, state.boss_shield)
    shield_active_next = jnp.where(shield_shatter, False, state.shield_active)

    # Gauge delta from friendly fire
    gated_hits_boss = laser_hits_boss & tracking_to_firing
    gated_hits_player = laser_hits_player & tracking_to_firing
    gauge_delta = (
        jnp.where(gated_hits_player, params.tracking_laser_gauge_gain, 0.0)
        - jnp.where(gated_hits_boss, params.tracking_laser_gauge_reduction, 0.0)
    )

    gauge_accum = state.security_gauge + gain_rate * params.dt + gauge_delta
    gauge_next_candidate = jnp.clip(gauge_accum, 0.0, 1.0)

    # Overload mode trigger at 100%
    triggers_overload = (gauge_next_candidate >= 1.0) & (~state.is_overload) & is_remastered_or_hybrid
    overload_timer_next = jnp.where(
        triggers_overload,
        params.overload_duration,
        jnp.maximum(0.0, state.overload_timer - params.dt),
    )
    is_overload_next = jnp.where(
        triggers_overload,
        True,
        jnp.where(overload_timer_next <= 0.0, False, state.is_overload),
    )
    gauge_final = jnp.where(is_overload_next, 0.0, gauge_next_candidate)

    # Remastered hazard: Horizontal artillery during Overload (1006-000)
    in_artillery_zone = (px_next < params.safe_zone_x) & is_overload_next & is_remastered_or_hybrid
    artillery_damage = jnp.where(in_artillery_zone, params.player_max_hp, 0.0)

    # Remastered hazard: Electric floor (F13) dynamic cycle
    ef_timer_dec = state.electric_floor_warning - params.dt
    ef_burst_expired = state.electric_floor_active & (ef_timer_dec <= -0.5)
    ef_periodic_spawn = (~state.electric_floor_active) & (time_next % 600 == 0) & is_remastered_or_hybrid

    electric_floor_active_next = jnp.where(
        ef_burst_expired,
        False,
        jnp.where(ef_periodic_spawn, True, state.electric_floor_active),
    )
    electric_floor_warning_next = jnp.where(
        ef_burst_expired,
        0.0,
        jnp.where(
            ef_periodic_spawn,
            2.0,  # 2.0s blue warning
            jnp.where(state.electric_floor_active, ef_timer_dec, state.electric_floor_warning),
        ),
    )

    electric_floor_detonates = state.electric_floor_active & (state.electric_floor_warning <= 0.0)
    electric_floor_hit = (
        electric_floor_detonates
        & on_ground_next
        & (py_next >= params.floor_y - 5.0)
        & is_remastered_or_hybrid
    )
    electric_floor_damage = jnp.where(electric_floor_hit, params.player_max_hp, 0.0)

    return {
        "tl_state_next": tl_state_next,
        "tl_timer_next": tl_timer_next,
        "tl_lock_x_next": tl_lock_x_next,
        "laser_hits_boss": laser_hits_boss,
        "laser_hits_player": laser_hits_player,
        "just_entered_firing": just_entered_firing,
        "shield_shatter": shield_shatter,
        "boss_shield_next": boss_shield_next,
        "shield_active_next": shield_active_next,
        "gauge_final": gauge_final,
        "triggers_overload": triggers_overload,
        "is_overload_next": is_overload_next,
        "overload_timer_next": overload_timer_next,
        "in_artillery_zone": in_artillery_zone,
        "artillery_damage": artillery_damage,
        "electric_floor_warning_next": electric_floor_warning_next,
        "electric_floor_active_next": electric_floor_active_next,
        "electric_floor_hit": electric_floor_hit,
        "electric_floor_damage": electric_floor_damage,
    }


def _compute_damage_and_health(
    state: EnvState,
    params: EnvParams,
    classic_laser_hit: chex.Array,
    debris_damage_total: chex.Array,
    in_artillery_zone: chex.Array,
    artillery_damage: chex.Array,
    electric_floor_hit: chex.Array,
    electric_floor_damage: chex.Array,
    laser_hits_player: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Pure functional damage aggregation, invincibility gating, and health update."""
    is_inv = state.invincible_timer > 0.0
    eff_laser_dmg = jnp.where(classic_laser_hit & (~is_inv), params.laser_damage, 0.0)
    eff_debris_dmg = jnp.where((debris_damage_total > 0.0) & (~is_inv), debris_damage_total, 0.0)
    eff_artillery_dmg = jnp.where(in_artillery_zone & (~is_inv), artillery_damage, 0.0)
    eff_elec_dmg = jnp.where(electric_floor_hit & (~is_inv), electric_floor_damage, 0.0)
    eff_tl_dmg = jnp.where(laser_hits_player & (~is_inv), params.tracking_laser_player_dmg, 0.0)
    total_dmg = eff_laser_dmg + eff_debris_dmg + eff_artillery_dmg + eff_elec_dmg + eff_tl_dmg

    hp_next = jnp.maximum(0.0, state.player_hp - total_dmg)
    took_hit = total_dmg > 0.0
    inv_timer_next = jnp.where(
        took_hit,
        params.invincible_duration,
        jnp.maximum(0.0, state.invincible_timer - params.dt),
    )
    return hp_next, inv_timer_next, took_hit, total_dmg


def _compute_reward(
    took_hit: chex.Array,
    laser_hits_boss: chex.Array,
    shield_shatter: chex.Array,
    triggers_overload: chex.Array,
    gauge_final: chex.Array,
    is_overload_next: chex.Array,
    px_next: chex.Array,
    hp_next: chex.Array,
    tl_state: chex.Array,
    tl_lock_x: chex.Array,
    params: EnvParams,
) -> chex.Array:
    """Computes shaped reward for Remastered gimmick-oriented play.

    Reward philosophy (v2): Shift agent strategy from passive wall-camping to
    active Friendly Fire exploitation.  Reduces base survival dominance and
    penalises wall-hugging outside of overload phases.

    Components:
    - Base survival: +0.03 per step (reduced from +0.1 to lower passive income)
    - Hit penalty: -100.0 on any damage
    - Friendly Fire boss hit: +50.0 (boosted from +10; primary learning signal)
    - Shield shatter:        +30.0 (boosted from +5; milestone event)
    - Overload penalty: -20.0 when security gauge reaches 100%
    - Anti-overload gauge suppression: -0.08 * gauge_final (stronger)
    - Safe zone positioning during Overload: +0.3 / -0.3
    - Health preservation bonus: +0.02 * (hp_next / player_max_hp)
    - Wall proximity penalty: -0.15 per step when |player_x - wall| < 150
      (disabled during overload, where wall-right camping IS correct)
    - Tracking laser bait shaping: +0.08 when tracking laser is in TRACKING
      state and the lock-x is within boss core hit window (guides baiting)
    """
    is_remastered_or_hybrid = params.mode != MODE_CLASSIC

    # --- Core Survival ---
    r_base = 0.03
    r_hit = jnp.where(took_hit, -100.0, 0.0)
    r_hp = 0.02 * (hp_next / params.player_max_hp)

    # --- Friendly Fire Gimmick (primary learning signal) ---
    r_boss_hit = jnp.where(laser_hits_boss, 50.0, 0.0)
    r_shield = jnp.where(shield_shatter, 30.0, 0.0)

    # --- Gauge & Overload Management ---
    r_overload = jnp.where(triggers_overload, -20.0, 0.0)
    r_gauge = jnp.where(is_remastered_or_hybrid, -0.08 * gauge_final, 0.0)

    # Safe zone positioning during overload (stronger signal)
    in_safe_zone = px_next >= params.safe_zone_x
    r_safe_zone = jnp.where(
        is_overload_next & is_remastered_or_hybrid,
        jnp.where(in_safe_zone, 0.3, -0.3),
        0.0,
    )

    # --- Anti-Wall-Camping Penalty ---
    # Penalise hugging left/right walls outside of overload phases.
    # During overload the agent SHOULD camp near the right wall (safe zone).
    wall_margin = 150.0
    near_left_wall = px_next < (params.wall_left + wall_margin)
    near_right_wall = px_next > (params.wall_right - wall_margin)
    near_any_wall = near_left_wall | near_right_wall
    r_wall = jnp.where(
        near_any_wall & (~is_overload_next) & is_remastered_or_hybrid,
        -0.15,
        0.0,
    )

    # --- Tracking Laser Bait Shaping ---
    # When the tracking laser is in TRACKING state (1), reward the player if
    # the lock-x is converging toward the boss core hit window.
    # Boss core hit window: [core_x - boss_w/2, core_x + boss_w/2]
    boss_left = params.core_x - params.boss_w / 2.0
    boss_right = params.core_x + params.boss_w / 2.0
    lock_in_boss_window = (tl_lock_x >= boss_left) & (tl_lock_x <= boss_right)
    is_tracking = tl_state == 1
    r_bait = jnp.where(
        is_tracking & lock_in_boss_window & is_remastered_or_hybrid,
        0.08,
        0.0,
    )

    return (
        r_base + r_hit + r_hp
        + r_boss_hit + r_shield
        + r_overload + r_gauge + r_safe_zone
        + r_wall + r_bait
    )


# ---------------------------------------------------------------------------
# 3. Gymnax LotusPhase1Env Environment Core
# ---------------------------------------------------------------------------

class LotusPhase1Env(environment.Environment):
    """MapleStory Lotus Phase 1 Functional Reinforcement Learning Environment."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict[str, Any]]:
        key_spawn, _ = jax.random.split(key)

        # 1. Player Kinematics
        px_next, py_next, vx, vy_final, on_ground_next, p_center_x, p_center_y, current_h = (
            _step_player_kinematics(state, action, params)
        )

        # 2. Classic Laser Mechanics
        laser_angle_next, classic_laser_hit = _step_classic_laser(
            state, p_center_x, p_center_y, current_h, params
        )

        # 3. Falling Debris Mechanics
        (
            cur_deb_x, deb_y_next, cur_deb_vy, cur_deb_r, cur_deb_dmg,
            deb_act_next, cur_deb_t, debris_damage_total, _
        ) = _step_falling_debris(key_spawn, state, p_center_x, p_center_y, params)

        # Timestep advance for periodic hazards and episode tracking
        time_next = state.time + 1

        # 4. Remastered Mechanics (Laser FSM, Shield, Gauge, Overload, Floor)
        rm = _step_remastered_mechanics(
            state, px_next, py_next, on_ground_next, time_next, params
        )

        # 5. Damage Aggregation & Health State
        hp_next, inv_timer_next, took_hit, total_dmg = _compute_damage_and_health(
            state,
            params,
            classic_laser_hit,
            debris_damage_total,
            rm["in_artillery_zone"],
            rm["artillery_damage"],
            rm["electric_floor_hit"],
            rm["electric_floor_damage"],
            rm["laser_hits_player"],
        )

        # 6. Temporal Tracking & Done Determination
        is_dead = hp_next <= 0.0
        is_timeout = time_next >= params.max_steps_in_episode
        done = is_dead | is_timeout

        # 7. Next Immutable State
        state_next = EnvState(
            player_x=px_next,
            player_y=py_next,
            player_vx=vx,
            player_vy=vy_final,
            player_hp=hp_next,
            player_on_ground=on_ground_next,
            invincible_timer=inv_timer_next,
            laser_angle=laser_angle_next,
            debris_x=cur_deb_x,
            debris_y=deb_y_next,
            debris_vy=cur_deb_vy,
            debris_radius=cur_deb_r,
            debris_damage=cur_deb_dmg,
            debris_active=deb_act_next,
            debris_type=cur_deb_t,
            time=time_next,
            security_gauge=rm["gauge_final"],
            is_overload=rm["is_overload_next"],
            overload_timer=rm["overload_timer_next"],
            boss_hp=state.boss_hp,
            boss_shield=rm["boss_shield_next"],
            shield_active=rm["shield_active_next"],
            tracking_laser_timer=rm["tl_timer_next"],
            tracking_laser_lock_x=rm["tl_lock_x_next"],
            tracking_laser_state=rm["tl_state_next"],
            electric_floor_warning=rm["electric_floor_warning_next"],
            electric_floor_active=rm["electric_floor_active_next"],
        )

        # 8. Aligned Shaped Reward (v2: anti-wall-camping + gimmick-oriented)
        gated_boss_hit = rm["laser_hits_boss"] & rm["just_entered_firing"]
        reward = _compute_reward(
            took_hit,
            gated_boss_hit,
            rm["shield_shatter"],
            rm["triggers_overload"],
            rm["gauge_final"],
            rm["is_overload_next"],
            px_next,
            hp_next,
            rm["tl_state_next"],
            rm["tl_lock_x_next"],
            params,
        )

        # 9. Dynamic Observation Matching Mode (130-dim or 142-dim)
        obs = self.get_observation(state_next, params)

        info = {
            "laser_hit": classic_laser_hit,
            "debris_hit": debris_damage_total > 0.0,
            "tracking_laser_hit_boss": rm["laser_hits_boss"],
            "tracking_laser_hit_player": rm["laser_hits_player"],
            "shield_shatter": rm["shield_shatter"],
            "dmg_taken": total_dmg,
            "hp": hp_next,
            "is_overload": rm["is_overload_next"],
            "security_gauge": rm["gauge_final"],
            "boss_hit_event": gated_boss_hit,
            "tracking_laser_hit_boss_tick": rm["laser_hits_boss"],
        }
        return obs, state_next, reward, done, info

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        key_x, key_angle = jax.random.split(key)
        start_x = jax.random.uniform(key_x, minval=300.0, maxval=1066.0)
        start_angle = jax.random.uniform(key_angle, minval=0.0, maxval=2.0 * jnp.pi)
        state = EnvState(
            player_x=start_x,
            player_y=params.floor_y,
            player_vx=0.0,
            player_vy=0.0,
            player_hp=params.player_max_hp,
            player_on_ground=True,
            invincible_timer=0.0,
            laser_angle=start_angle,
            debris_x=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            debris_y=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            debris_vy=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            debris_radius=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            debris_damage=jnp.zeros(MAX_DEBRIS, dtype=jnp.float32),
            debris_active=jnp.zeros(MAX_DEBRIS, dtype=bool),
            debris_type=jnp.zeros(MAX_DEBRIS, dtype=jnp.int32),
            time=0,
            security_gauge=0.0,
            is_overload=False,
            overload_timer=0.0,
            boss_hp=params.boss_max_hp,
            boss_shield=params.boss_shield_max,
            shield_active=True,
            tracking_laser_timer=0.0,
            tracking_laser_lock_x=params.core_x,
            tracking_laser_state=0,
            electric_floor_warning=0.0,
            electric_floor_active=False,
        )
        return self.get_observation(state, params), state

    def get_obs(self, state: Any, params: EnvParams) -> chex.Array:
        """Returns standard 130-dimensional flat normalized float32 observation tensor."""
        state = getattr(state, "env_state", state)
        p_norm = jnp.array([
            state.player_x / params.screen_width,
            state.player_y / params.screen_height,
            state.player_vx / params.player_speed,
            state.player_vy / 650.0,
            state.player_hp / params.player_max_hp,
            state.invincible_timer / params.invincible_duration,
        ], dtype=jnp.float32)

        laser_norm = jnp.array([
            jnp.cos(state.laser_angle),
            jnp.sin(state.laser_angle),
            jnp.cos(state.laser_angle + jnp.pi / 2.0),
            jnp.sin(state.laser_angle + jnp.pi / 2.0),
        ], dtype=jnp.float32)

        deb_norm = jnp.stack([
            state.debris_x / params.screen_width,
            state.debris_y / params.screen_height,
            state.debris_vy / params.debris_max_vy,
            state.debris_active.astype(jnp.float32),
        ], axis=-1).flatten()

        return jnp.concatenate([p_norm, laser_norm, deb_norm])

    def get_extended_obs(self, state: Any, params: EnvParams) -> chex.Array:
        """Returns extended 142-dimensional flat normalized float32 observation tensor for Remaster/Hybrid."""
        state = getattr(state, "env_state", state)
        base_obs = self.get_obs(state, params)
        remaster_features = jnp.array([
            state.security_gauge,
            jnp.where(state.is_overload, 1.0, 0.0),
            state.overload_timer / params.overload_duration,
            state.boss_hp / params.boss_max_hp,
            state.boss_shield / params.boss_shield_max,
            (params.core_x - state.player_x) / params.screen_width,
            state.tracking_laser_state / 3.0,
            (state.tracking_laser_lock_x - state.player_x) / params.screen_width,
            state.tracking_laser_timer / 2.0,
            state.electric_floor_warning / 2.0,
            jnp.where(state.electric_floor_active, 1.0, 0.0),
            0.0,  # slam_active placeholder
        ], dtype=jnp.float32)
        return jnp.concatenate([base_obs, remaster_features])

    def get_observation(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns observation matching active mode (130-dim for Classic, 142-dim for Remastered/Hybrid)."""
        return self.get_extended_obs(state, params) if params.is_remastered else self.get_obs(state, params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return (state.player_hp <= 0.0) | (state.time >= params.max_steps_in_episode)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(7)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_dim = 142 if params.is_remastered else 130
        return spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=jnp.float32)
