"""Pygame-based Interactive Policy Viewer for MapleStory Lotus Phase 1.

Visualizes trained policies or random agents in Classic (mode 0),
Remastered (mode 1), and Hybrid (mode 2) mechanics at 60 FPS.

Features:
- CLI flags for checkpoint path, mode selection, random dry-run, and action sampling.
- Real-time 60 FPS neon-themed rendering on 1366x768 display.
- Dynamic mode branching:
  * Classic: 4-arm rotating cross laser with SAT angle projection.
  * Remastered: Security gauge & boss shield HUD, player tracking laser FSM & friendly fire,
    overload horizontal artillery danger zone (x < 1150), and electric floor warning/detonation.
- OSD text overlay: Step, HP, cumulative reward, action name, playback speed, and PAUSED state.
- Interactive controls: Space (Pause/Resume), R (Reset), Esc (Exit), 1/2/3/4 (Speed 0.5x/1x/2x/4x).
- Host-side batch synchronization: Converts JAX simulation state into host NumPy arrays in a single
  device_get call per tick, eliminating scalar sync bottlenecks during rendering loops.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from maple_gymnax.envs.common import (
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
)
from maple_gymnax.envs.lotus_phase1 import EnvParams, EnvState, LotusPhase1Env


# ---------------------------------------------------------------------------
# 1. Neon Color Palette Constants
# ---------------------------------------------------------------------------
COLOR_BG = (10, 10, 20)
COLOR_FLOOR = (60, 65, 85)
COLOR_WALL = (40, 45, 60)
COLOR_SAFE_ZONE_FILL = (0, 255, 100, 30)
COLOR_SAFE_ZONE_LINE = (0, 255, 120)

COLOR_PLAYER_NORMAL = (0, 255, 136)
COLOR_PLAYER_DUCK = (255, 230, 0)
COLOR_PLAYER_INVINCIBLE = (240, 255, 255)
COLOR_PLAYER_CENTER = (255, 255, 255)

COLOR_CORE = (180, 50, 255)
COLOR_CORE_AURA = (140, 0, 230, 60)
COLOR_CORE_CENTER = (230, 180, 255)

COLOR_LASER_CORE = (255, 255, 255)
COLOR_LASER_BEAM = (255, 20, 60)
COLOR_LASER_GLOW = (255, 50, 80, 90)

COLOR_DEBRIS_SMALL = (0, 230, 255)      # Type 0 (r=16, cyan)
COLOR_DEBRIS_MEDIUM = (255, 160, 20)    # Type 1 (r=24, orange)
COLOR_DEBRIS_LARGE = (255, 40, 140)     # Type 2 (r=36, hot pink)

COLOR_TL_TRACKING = (255, 235, 50)     # Tracking state (dashed yellow)
COLOR_TL_FIRING = (255, 30, 30)        # Firing state (solid red beam)
COLOR_TL_CORE = (255, 255, 255)
COLOR_FRIENDLY_FIRE_HIT = (0, 180, 255)

COLOR_ELEC_WARNING = (60, 140, 255)
COLOR_ELEC_ACTIVE = (200, 235, 255)

COLOR_DANGER_FILL = (255, 20, 20, 45)
COLOR_DANGER_LINE = (255, 60, 20)

COLOR_HUD_BG = (25, 25, 35)
COLOR_HUD_BORDER = (160, 160, 180)
COLOR_SHIELD_BAR = (0, 150, 255)
COLOR_TEXT = (240, 240, 245)

ACTION_NAMES = [
    "IDLE (0)",
    "LEFT (1)",
    "RIGHT (2)",
    "JUMP (3)",
    "J+LEFT (4)",
    "J+RIGHT (5)",
    "DUCK (6)",
]


# ---------------------------------------------------------------------------
# 2. Host Simulation State Container (Zero-copy NumPy Conversion)
# ---------------------------------------------------------------------------
class HostEnvState:
    """CPU/NumPy mirrored simulation state to prevent per-scalar JAX synchronization."""

    def __init__(self, raw_state: EnvState):
        # Perform single host-device transfer for the entire state PyTree
        cpu_dict = jax.device_get(raw_state)

        # Player scalars
        self.player_x = float(cpu_dict.player_x)
        self.player_y = float(cpu_dict.player_y)
        self.player_vx = float(cpu_dict.player_vx)
        self.player_vy = float(cpu_dict.player_vy)
        self.player_hp = float(cpu_dict.player_hp)
        self.player_on_ground = bool(cpu_dict.player_on_ground)
        self.invincible_timer = float(cpu_dict.invincible_timer)

        # Classic Laser scalar
        self.laser_angle = float(cpu_dict.laser_angle)

        # Debris batch arrays (np.ndarray of shape (30,))
        self.debris_x = np.asarray(cpu_dict.debris_x, dtype=np.float32)
        self.debris_y = np.asarray(cpu_dict.debris_y, dtype=np.float32)
        self.debris_vy = np.asarray(cpu_dict.debris_vy, dtype=np.float32)
        self.debris_radius = np.asarray(cpu_dict.debris_radius, dtype=np.float32)
        self.debris_damage = np.asarray(cpu_dict.debris_damage, dtype=np.float32)
        self.debris_active = np.asarray(cpu_dict.debris_active, dtype=bool)
        self.debris_type = np.asarray(cpu_dict.debris_type, dtype=np.int32)

        # Time & Remastered mechanics
        self.time = int(cpu_dict.time)
        self.security_gauge = float(cpu_dict.security_gauge)
        self.is_overload = bool(cpu_dict.is_overload)
        self.overload_timer = float(cpu_dict.overload_timer)
        self.boss_hp = float(cpu_dict.boss_hp)
        self.boss_shield = float(cpu_dict.boss_shield)
        self.shield_active = bool(cpu_dict.shield_active)
        self.tracking_laser_timer = float(cpu_dict.tracking_laser_timer)
        self.tracking_laser_lock_x = float(cpu_dict.tracking_laser_lock_x)
        self.tracking_laser_state = int(cpu_dict.tracking_laser_state)
        self.electric_floor_warning = float(cpu_dict.electric_floor_warning)
        self.electric_floor_active = bool(cpu_dict.electric_floor_active)


# ---------------------------------------------------------------------------
# 3. Geometry & Drawing Helpers
# ---------------------------------------------------------------------------
def draw_dashed_vertical_line(
    surface: pygame.Surface,
    color: Tuple[int, int, int],
    x: float,
    y_start: float,
    y_end: float,
    dash_len: int = 8,
    gap_len: int = 6,
    width: int = 2,
) -> None:
    """Draws a vertical dashed line efficiently using Pygame draw.line."""
    cur_y = y_start
    ix = int(round(x))
    while cur_y < y_end:
        next_y = min(cur_y + dash_len, y_end)
        pygame.draw.line(surface, color, (ix, int(cur_y)), (ix, int(next_y)), width)
        cur_y += dash_len + gap_len


def draw_arena(surface: pygame.Surface, params: EnvParams) -> None:
    """Renders boundaries, floor platform, and Remastered safe-zone indicator."""
    w_left = int(params.wall_left)
    w_right = int(params.wall_right)
    floor_y = int(params.floor_y)
    safe_x = int(params.safe_zone_x)

    # Left & Right Wall Boundaries
    pygame.draw.line(surface, COLOR_WALL, (w_left, 0), (w_left, floor_y), 3)
    pygame.draw.line(surface, COLOR_WALL, (w_right, 0), (w_right, floor_y), 3)

    # Floor Platform (solid bar + top highlight)
    floor_rect = pygame.Rect(w_left, floor_y, w_right - w_left, int(params.screen_height - floor_y))
    pygame.draw.rect(surface, (20, 22, 32), floor_rect)
    pygame.draw.line(surface, COLOR_FLOOR, (w_left, floor_y), (w_right, floor_y), 4)

    # Mode 1/2 Safe Zone Background Tint (x >= safe_zone_x)
    if params.mode != MODE_CLASSIC:
        safe_width = w_right - safe_x
        safe_surf = pygame.Surface((safe_width, floor_y), pygame.SRCALPHA)
        safe_surf.fill(COLOR_SAFE_ZONE_FILL)
        surface.blit(safe_surf, (safe_x, 0))
        draw_dashed_vertical_line(surface, COLOR_SAFE_ZONE_LINE, safe_x, 0, floor_y, dash_len=10, gap_len=6, width=2)


def draw_boss_core(surface: pygame.Surface, state: HostEnvState, params: EnvParams) -> None:
    """Renders boss core hitbox, outer visual glow, and shield aura if active."""
    cx = int(params.core_x)
    cy = int(params.core_y)
    r_hit = int(params.core_radius)

    # Outer visual aura
    aura_surf = pygame.Surface((r_hit * 4, r_hit * 4), pygame.SRCALPHA)
    pygame.draw.circle(aura_surf, COLOR_CORE_AURA, (r_hit * 2, r_hit * 2), int(r_hit * 1.5))
    surface.blit(aura_surf, (cx - r_hit * 2, cy - r_hit * 2))

    # Core Solid Body
    pygame.draw.circle(surface, COLOR_CORE, (cx, cy), r_hit)
    pygame.draw.circle(surface, COLOR_CORE_CENTER, (cx, cy), r_hit // 2)

    # Shield Ring (Remastered mode)
    if params.mode != MODE_CLASSIC and state.shield_active:
        shield_ring_radius = r_hit + 12
        pygame.draw.circle(surface, COLOR_SHIELD_BAR, (cx, cy), shield_ring_radius, 3)


def draw_cross_laser(surface: pygame.Surface, state: HostEnvState, params: EnvParams) -> None:
    """Renders 4-arm rotating cross laser for Classic (0) and Hybrid (2) modes."""
    angle = state.laser_angle
    cx = params.core_x
    cy = params.core_y
    max_len = params.laser_max_length
    core_r = params.core_radius
    thickness = int(params.laser_half_thickness * 2.0)

    # Pre-render glowing beams
    for k in range(4):
        theta = angle + k * (math.pi / 2.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        start_x = cx + cos_t * core_r
        start_y = cy + sin_t * core_r
        end_x = cx + cos_t * max_len
        end_y = cy + sin_t * max_len

        # Wide colored beam
        pygame.draw.line(surface, COLOR_LASER_BEAM, (int(start_x), int(start_y)), (int(end_x), int(end_y)), thickness)
        # Inner hot white core line
        pygame.draw.line(surface, COLOR_LASER_CORE, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 3)


def draw_falling_debris(surface: pygame.Surface, state: HostEnvState) -> None:
    """Renders all active falling debris particles using NumPy vectorized arrays."""
    active_indices = np.where(state.debris_active)[0]
    for idx in active_indices:
        x = int(round(state.debris_x[idx]))
        y = int(round(state.debris_y[idx]))
        r = int(round(state.debris_radius[idx]))
        dtype = state.debris_type[idx]

        if dtype == 0:
            color = COLOR_DEBRIS_SMALL
        elif dtype == 1:
            color = COLOR_DEBRIS_MEDIUM
        else:
            color = COLOR_DEBRIS_LARGE

        # Circular debris with inner accent
        pygame.draw.circle(surface, color, (x, y), r)
        if r > 10:
            pygame.draw.circle(surface, (255, 255, 255), (x, y), max(2, r // 3))


def draw_player(surface: pygame.Surface, state: HostEnvState, params: EnvParams, current_action: int) -> None:
    """Renders the player AABB hitbox, center marker, ducking state, and invincibility blink."""
    # Invincibility flicker: blink every 6 frames
    if state.invincible_timer > 0.0:
        if (state.time // 6) % 2 == 0:
            return  # skip drawing frame for flicker effect

    px = state.player_x
    py = state.player_y
    is_duck = (current_action == ACTION_DUCK)

    box_w = params.player_w
    box_h = params.player_duck_h if is_duck else params.player_h

    # Foot coordinate py is bottom of box; top is py - box_h
    left = int(round(px - box_w / 2.0))
    top = int(round(py - box_h))
    width = int(round(box_w))
    height = int(round(box_h))

    color = COLOR_PLAYER_DUCK if is_duck else (
        COLOR_PLAYER_INVINCIBLE if state.invincible_timer > 0.0 else COLOR_PLAYER_NORMAL
    )

    rect = pygame.Rect(left, top, width, height)
    pygame.draw.rect(surface, color, rect, border_radius=4)
    pygame.draw.rect(surface, (255, 255, 255), rect, width=2, border_radius=4)

    # Player center dot
    center_y = int(round(py - box_h / 2.0))
    pygame.draw.circle(surface, COLOR_PLAYER_CENTER, (int(round(px)), center_y), 3)


def draw_remastered_hazards(surface: pygame.Surface, state: HostEnvState, params: EnvParams) -> None:
    """Renders Remastered-specific hazards: Tracking laser, Overload danger zone, Electric floor."""
    floor_y = int(params.floor_y)
    w_left = int(params.wall_left)

    # 1. Overload Horizontal Artillery Danger Zone (x < 1150)
    if state.is_overload:
        danger_w = int(params.safe_zone_x - w_left)
        danger_surf = pygame.Surface((danger_w, floor_y), pygame.SRCALPHA)
        danger_surf.fill(COLOR_DANGER_FILL)
        surface.blit(danger_surf, (w_left, 0))

        # Warning boundary line
        draw_dashed_vertical_line(
            surface, COLOR_DANGER_LINE, params.safe_zone_x, 0, floor_y, dash_len=12, gap_len=6, width=3
        )

    # 2. Electric Floor Warning & Detonation
    if state.electric_floor_active:
        ef_warn = state.electric_floor_warning
        w_right = int(params.wall_right)
        total_w = w_right - w_left

        if ef_warn > 0.0:
            # Blue warning pulsation
            alpha = int(70 + 40 * math.sin(ef_warn * 12.0))
            ef_surf = pygame.Surface((total_w, 24), pygame.SRCALPHA)
            ef_surf.fill((*COLOR_ELEC_WARNING, alpha))
            surface.blit(ef_surf, (w_left, floor_y - 12))
        else:
            # Lethal detonation burst
            ef_surf = pygame.Surface((total_w, 36), pygame.SRCALPHA)
            ef_surf.fill((*COLOR_ELEC_ACTIVE, 190))
            surface.blit(ef_surf, (w_left, floor_y - 18))
            # Lightning line
            pygame.draw.line(surface, (255, 255, 255), (w_left, floor_y), (w_right, floor_y), 3)

    # 3. Tracking Laser FSM Visualization
    tl_state = state.tracking_laser_state
    lock_x = state.tracking_laser_lock_x

    if tl_state == 1:
        # TRACKING: Yellow dashed guidance line targeting player
        draw_dashed_vertical_line(surface, COLOR_TL_TRACKING, lock_x, 0, floor_y, dash_len=8, gap_len=6, width=2)
        # Reticle at ground
        pygame.draw.circle(surface, COLOR_TL_TRACKING, (int(round(lock_x)), floor_y), 6, 2)

    elif tl_state == 2:
        # FIRING: Lethal vertical laser beam
        lx = int(round(lock_x))
        pygame.draw.line(surface, COLOR_TL_FIRING, (lx, 0), (lx, floor_y), 8)
        pygame.draw.line(surface, COLOR_TL_CORE, (lx, 0), (lx, floor_y), 3)

        # Friendly Fire Check against Boss Core
        boss_left = params.core_x - params.boss_w / 2.0
        boss_right = params.core_x + params.boss_w / 2.0
        if boss_left <= lock_x <= boss_right:
            # Target struck boss core!
            pygame.draw.circle(surface, COLOR_FRIENDLY_FIRE_HIT, (lx, int(params.core_y)), 30, 4)


def draw_hud(
    surface: pygame.Surface,
    state: HostEnvState,
    params: EnvParams,
    font_large: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    """Renders top HUD: Security Gauge, Boss Shield Bar, and Overload Banner."""
    if params.mode == MODE_CLASSIC:
        return

    scr_w = int(params.screen_width)
    bar_w = 340
    bar_h = 16
    start_x = scr_w // 2 - bar_w // 2

    # --- 1. Security Gauge Bar ---
    gauge_y = 12
    pygame.draw.rect(surface, COLOR_HUD_BG, (start_x, gauge_y, bar_w, bar_h))

    gauge_val = np.clip(state.security_gauge, 0.0, 1.0)
    fill_w = int(round(gauge_val * bar_w))

    # Color shifts smoothly from yellow/orange to fiery red
    red_ch = 255
    green_ch = int(220 * (1.0 - gauge_val))
    pygame.draw.rect(surface, (red_ch, green_ch, 20), (start_x, gauge_y, fill_w, bar_h))
    pygame.draw.rect(surface, COLOR_HUD_BORDER, (start_x, gauge_y, bar_w, bar_h), 1)

    txt_gauge = font_small.render(f"SECURITY GAUGE: {gauge_val * 100.0:5.1f}%", True, COLOR_TEXT)
    surface.blit(txt_gauge, (start_x + bar_w + 12, gauge_y))

    # Overload Alert
    if state.is_overload:
        txt_ol = font_large.render(f"⚡ OVERLOAD: {state.overload_timer:4.1f}s", True, (255, 50, 50))
        surface.blit(txt_ol, (start_x - txt_ol.get_width() - 16, gauge_y - 2))

    # --- 2. Boss Shield Bar ---
    shield_y = gauge_y + bar_h + 8
    pygame.draw.rect(surface, COLOR_HUD_BG, (start_x, shield_y, bar_w, bar_h))

    shield_ratio = np.clip(state.boss_shield / params.boss_shield_max, 0.0, 1.0)
    s_fill_w = int(round(shield_ratio * bar_w))
    s_color = COLOR_SHIELD_BAR if state.shield_active else (100, 100, 120)

    pygame.draw.rect(surface, s_color, (start_x, shield_y, s_fill_w, bar_h))
    pygame.draw.rect(surface, COLOR_HUD_BORDER, (start_x, shield_y, bar_w, bar_h), 1)

    shield_status = "ACTIVE" if state.shield_active else "BROKEN"
    txt_shield = font_small.render(f"BOSS SHIELD: {shield_status} ({state.boss_shield:3.0f})", True, COLOR_TEXT)
    surface.blit(txt_shield, (start_x + bar_w + 12, shield_y))


def draw_osd(
    surface: pygame.Surface,
    state: HostEnvState,
    params: EnvParams,
    action: int,
    cumulative_reward: float,
    speed_mult: float,
    paused: bool,
    font: pygame.font.Font,
) -> None:
    """Renders on-screen display (OSD) telemetry and controls."""
    scr_h = int(params.screen_height)

    # Bottom Telemetry Panel
    osd_y = scr_h - 48
    act_str = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"ACT_{action}"
    pause_str = " [PAUSED]" if paused else ""

    line1 = (
        f"STEP: {state.time:4d}/{params.max_steps_in_episode}  |  "
        f"HP: {state.player_hp:5.1f}/{params.player_max_hp:3.0f}  |  "
        f"REWARD: {cumulative_reward:8.2f}  |  "
        f"ACTION: {act_str}{pause_str}"
    )

    line2 = (
        f"POS: ({state.player_x:5.1f}, {state.player_y:5.1f})  |  "
        f"SPEED: {speed_mult:3.1f}x (1-4 keys)  |  "
        f"MODE: {params.mode} ({'Classic' if params.mode == 0 else ('Remastered' if params.mode == 1 else 'Hybrid')})  |  "
        f"[Space]: Pause  [R]: Reset  [Esc]: Exit"
    )

    surf1 = font.render(line1, True, (0, 255, 200))
    surf2 = font.render(line2, True, (180, 190, 210))

    surface.blit(surf1, (16, osd_y - 20))
    surface.blit(surf2, (16, osd_y + 4))


# ---------------------------------------------------------------------------
# 4. Checkpoint Loading & Policy Setup
# ---------------------------------------------------------------------------
def load_trained_policy(checkpoint_path: str, mode: int) -> Tuple[Any, Any]:
    """Restores model parameters from Orbax and instantiates ActorCritic."""
    import orbax.checkpoint as ocp
    from train_ppo import ActorCritic

    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint path not found: {abs_path}")

    action_dim = 7
    network = ActorCritic(action_dim=action_dim)

    # Initialize dummy target PyTree template to prevent cross-device topology mismatch
    # (e.g. restoring GPU-trained checkpoint on CPU)
    obs_dim = 142 if mode in (MODE_REMASTERED, MODE_HYBRID) else 130
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    dummy_params = network.init(jax.random.PRNGKey(0), dummy_obs)

    checkpointer = ocp.StandardCheckpointer()
    try:
        restored_params = checkpointer.restore(abs_path, target=dummy_params)
    except Exception:
        restored_params = checkpointer.restore(abs_path)
    checkpointer.close()

    print(f"[PolicyViewer] Successfully restored checkpoint from: {abs_path}")
    return restored_params, network


# ---------------------------------------------------------------------------
# 5. CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pygame Policy Viewer for MapleStory Lotus Phase 1 (Classic & Remastered)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to Orbax step checkpoint directory (e.g. checkpoints/mode_0/step_1000).",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        choices=[MODE_CLASSIC, MODE_REMASTERED, MODE_HYBRID],
        help="0: Classic, 1: Remastered, 2: Hybrid",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use uniform random policy instead of loading a checkpoint (dry-run mode).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=True,
        help="Use deterministic argmax action selection (default).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically from categorical policy distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master PRNG seed for environment reset and sampling.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target base frame rate (multiplied by speed factor).",
    )
    args = parser.parse_args()

    if not args.random and not args.checkpoint_path:
        parser.error("Either --checkpoint_path must be specified or --random flag must be set.")

    return args


# ---------------------------------------------------------------------------
# 6. Main Execution Loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_arguments()
    is_greedy = not args.stochastic

    # Initialize Environment & JIT Kernels
    env = LotusPhase1Env()
    params = EnvParams(mode=args.mode)
    jit_reset = jax.jit(env.reset_env)
    jit_step = jax.jit(env.step_env)
    jit_get_obs = jax.jit(env.get_observation)

    # Model & Inference Setup
    network = None
    policy_params = None
    jit_apply = None

    if not args.random:
        policy_params, network = load_trained_policy(args.checkpoint_path, args.mode)
        jit_apply = jax.jit(network.apply)
    else:
        print("[PolicyViewer] Running in --random dry-run mode (uniform random policy).")

    # Pygame Display Initialization
    pygame.init()
    pygame.font.init()

    win_w = int(params.screen_width)
    win_h = int(params.screen_height)
    screen = pygame.display.set_mode((win_w, win_h))
    mode_name = "Classic" if args.mode == 0 else ("Remastered" if args.mode == 1 else "Hybrid")
    pygame.display.set_caption(f"Lotus Phase 1 Viewer — Mode {args.mode} ({mode_name})")
    clock = pygame.time.Clock()

    font_osd = pygame.font.SysFont("Consolas, Menlo, monospace", 15)
    font_large = pygame.font.SysFont("Consolas, Menlo, monospace", 18, bold=True)
    font_small = pygame.font.SysFont("Consolas, Menlo, monospace", 13)

    # Master PRNG Key & Initial Reset
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = jit_reset(rng_reset, params)

    paused = False
    speed_mult = 1.0
    cumulative_reward = 0.0
    action = 0
    running = True

    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Force Reset
                    rng, rng_reset = jax.random.split(rng)
                    obs, state = jit_reset(rng_reset, params)
                    cumulative_reward = 0.0
                    action = 0
                elif event.key == pygame.K_1:
                    speed_mult = 0.5
                elif event.key == pygame.K_2:
                    speed_mult = 1.0
                elif event.key == pygame.K_3:
                    speed_mult = 2.0
                elif event.key == pygame.K_4:
                    speed_mult = 4.0

        if not running:
            break

        # 2. Simulation Step (if not paused)
        if not paused:
            # Action Selection
            if args.random:
                rng, rng_act = jax.random.split(rng)
                action = int(jax.random.randint(rng_act, shape=(), minval=0, maxval=7))
            else:
                obs_single = jit_get_obs(state, params)[None, :]
                logits, _ = jit_apply(policy_params, obs_single)
                logits = logits[0]

                if is_greedy:
                    action = int(jnp.argmax(logits))
                else:
                    rng, rng_act = jax.random.split(rng)
                    action = int(jax.random.categorical(rng_act, logits))

            # Environment Transition
            rng, rng_step = jax.random.split(rng)
            obs, state, reward, done, info = jit_step(rng_step, state, action, params)
            cumulative_reward += float(reward)

            # Auto-Reset on Terminal State
            if bool(done):
                rng, rng_reset = jax.random.split(rng)
                obs, state = jit_reset(rng_reset, params)
                cumulative_reward = 0.0
                action = 0

        # 3. Synchronize JAX State to Host NumPy in a Single Batch Call
        host_state = HostEnvState(state)

        # 4. Render Frame (Layered Neon Rendering)
        screen.fill(COLOR_BG)

        # Layer 1: Arena boundaries and safe zone
        draw_arena(screen, params)

        # Layer 2: Mode-specific Hazards
        if params.mode != MODE_REMASTERED:
            draw_cross_laser(screen, host_state, params)

        if params.mode != MODE_CLASSIC:
            draw_remastered_hazards(screen, host_state, params)

        # Layer 3: Boss Core
        draw_boss_core(screen, host_state, params)

        # Layer 4: Falling Debris
        draw_falling_debris(screen, host_state)

        # Layer 5: Player Hitbox
        draw_player(screen, host_state, params, current_action=action)

        # Layer 6: HUD Overlay
        draw_hud(screen, host_state, params, font_large, font_small)
        draw_osd(
            screen,
            host_state,
            params,
            action=action,
            cumulative_reward=cumulative_reward,
            speed_mult=speed_mult,
            paused=paused,
            font=font_osd,
        )

        pygame.display.flip()
        clock.tick(int(args.fps * speed_mult))

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
