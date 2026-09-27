"""Common constants, action spaces, and vectorized branch-free physics for MapleGymnax.

This module provides:
1. Static compile-time constants (MAX_DEBRIS=30, game modes, discrete action mapping).
2. Separating Axis Theorem (SAT) AABB projection onto arbitrary 2D normal vectors.
3. Vectorized orthogonal raycast distance + directional dot-product masking for 4-arm rotating laser.
4. Vectorized Euclidean distance collision detection for static padded debris arrays.
5. Branch-free kinematic integration helpers.
"""

from __future__ import annotations

from typing import Tuple, Union
import chex
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# 1. Compile-Time Static Constants
# ---------------------------------------------------------------------------

MAX_DEBRIS: int = 30

# Game Modes
MODE_CLASSIC: int = 0
MODE_REMASTERED: int = 1
MODE_HYBRID: int = 2

# Discrete Action Mapping (Gymnax Discrete(7))
ACTION_NOOP: int = 0
ACTION_LEFT: int = 1
ACTION_RIGHT: int = 2
ACTION_JUMP: int = 3
ACTION_JUMP_LEFT: int = 4
ACTION_JUMP_RIGHT: int = 5
ACTION_DUCK: int = 6

# Canonical Short Aliases
NOOP: int = ACTION_NOOP
LEFT: int = ACTION_LEFT
RIGHT: int = ACTION_RIGHT
JUMP: int = ACTION_JUMP
JUMP_LEFT: int = ACTION_JUMP_LEFT
JUMP_RIGHT: int = ACTION_JUMP_RIGHT
DUCK: int = ACTION_DUCK


# ---------------------------------------------------------------------------
# 2. Vectorized Branch-Free Collision & Projection Math
# ---------------------------------------------------------------------------

def sat_aabb_projection(
    half_w: Union[float, chex.Array],
    half_h: Union[float, chex.Array],
    sin_theta: chex.Array,
    cos_theta: chex.Array,
) -> chex.Array:
    """Separating Axis Theorem (SAT) projection of an AABB onto a line normal.

    Given an axis-aligned bounding box with half-extents (half_w, half_h),
    its projected radius onto a unit normal vector (-sin(theta), cos(theta)) is:
        r_proj = half_w * |sin(theta)| + half_h * |cos(theta)|

    Args:
        half_w: Half-width of the player AABB (w / 2.0).
        half_h: Half-height of the player AABB (h / 2.0).
        sin_theta: Sine of beam orientation angles.
        cos_theta: Cosine of beam orientation angles.

    Returns:
        Projected radius along the laser normal axis.
    """
    return half_w * jnp.abs(sin_theta) + half_h * jnp.abs(cos_theta)


def compute_laser_collisions(
    player_center_x: chex.Array,
    player_center_y: chex.Array,
    player_w: float,
    player_h: Union[float, chex.Array],
    core_x: float,
    core_y: float,
    core_radius: float,
    laser_max_length: float,
    laser_angle: chex.Array,
    laser_half_thickness: float,
) -> chex.Array:
    """Vectorized branch-free evaluation of 4-arm cross laser collision.

    Each beam k in {0, 1, 2, 3} radiates outward from (core_x, core_y) at:
        theta_k = laser_angle + k * (pi / 2)
    Directional vector: u_k = (cos(theta_k), sin(theta_k))
    Transverse normal:  n_k = (-sin(theta_k), cos(theta_k))

    Math formulation:
    - Longitudinal projection: d_par = (px - cx) * cos(theta_k) + (py - cy) * sin(theta_k)
    - Transverse distance:     d_perp = |-(px - cx) * sin(theta_k) + (py - cy) * cos(theta_k)|
    - Ray directional mask:    in_beam = (d_par >= core_radius) & (d_par <= laser_max_length)
    - SAT normal threshold:    d_thresh = laser_half_thickness + sat_aabb_projection(w/2, h/2, sin, cos)
    - Arm collision:           arm_hit = in_beam & (d_perp <= d_thresh)

    Args:
        player_center_x: Geometric center x of player hitbox.
        player_center_y: Geometric center y of player hitbox.
        player_w: Width of player hitbox.
        player_h: Height of player hitbox (reduced if ducking).
        core_x: Center x of boss core pivot.
        core_y: Center y of boss core pivot.
        core_radius: Inner origin clearance radius (laser starts at perimeter).
        laser_max_length: Outer bounds cutoff ray length.
        laser_angle: Base laser orientation angle in radians [0, 2*pi).
        laser_half_thickness: Half beam thickness in pixels.

    Returns:
        Boolean scalar (or batched bool array) indicating if any of the 4 arms hits the player.
    """
    k = jnp.arange(4)
    thetas = laser_angle + k * (jnp.pi / 2.0)
    cos_k = jnp.cos(thetas)
    sin_k = jnp.sin(thetas)

    rel_x = player_center_x - core_x
    rel_y = player_center_y - core_y

    # Scalar projection along ray axis
    d_par = rel_x * cos_k + rel_y * sin_k
    # Perpendicular distance to line of action
    d_perp = jnp.abs(-rel_x * sin_k + rel_y * cos_k)

    # Exact SAT projection of box extents onto laser normal
    r_proj = sat_aabb_projection(player_w / 2.0, player_h / 2.0, sin_k, cos_k)
    d_thresh = laser_half_thickness + r_proj

    # Directional ray validity mask + transverse proximity
    in_beam = (d_par >= core_radius) & (d_par <= laser_max_length)
    arm_hits = in_beam & (d_perp <= d_thresh)

    return jnp.any(arm_hits)


def compute_debris_collisions(
    player_center_x: chex.Array,
    player_center_y: chex.Array,
    player_eff_radius: float,
    debris_x: chex.Array,
    debris_y: chex.Array,
    debris_radius: chex.Array,
    debris_active: chex.Array,
) -> chex.Array:
    """Vectorized Euclidean distance collision detection against static padded debris arrays.

    Evaluates contact between player circular approximation and all static debris particles:
        dist_i = sqrt((px - deb_x_i)^2 + (py - deb_y_i)^2)
        collided_i = active_i & (dist_i < (player_eff_radius + deb_radius_i))

    Args:
        player_center_x: Geometric center x of player hitbox.
        player_center_y: Geometric center y of player hitbox.
        player_eff_radius: Effective collision radius of player (e.g. 25.0 px).
        debris_x: Array of shape (MAX_DEBRIS,) float32 x coordinates.
        debris_y: Array of shape (MAX_DEBRIS,) float32 y coordinates.
        debris_radius: Array of shape (MAX_DEBRIS,) float32 collision radii.
        debris_active: Array of shape (MAX_DEBRIS,) bool activity mask.

    Returns:
        Boolean array of shape (MAX_DEBRIS,) indicating collision for each slot.
    """
    dx = player_center_x - debris_x
    dy = player_center_y - debris_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    threshold = player_eff_radius + debris_radius
    return debris_active & (dist < threshold)


# ---------------------------------------------------------------------------
# 3. Branch-Free Kinematics & Decode Utilities
# ---------------------------------------------------------------------------

def decode_action(
    action: Union[int, chex.Array],
    player_speed: float,
    standard_h: float,
    duck_h: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Decodes discrete action into horizontal velocity, duck height, and jump flag.

    Actions:
        0: NOOP       -> vx = 0,    h = standard, jump = False
        1: LEFT       -> vx = -spd, h = standard, jump = False
        2: RIGHT      -> vx = +spd, h = standard, jump = False
        3: JUMP       -> vx = 0,    h = standard, jump = True
        4: JUMP_LEFT  -> vx = -spd, h = standard, jump = True
        5: JUMP_RIGHT -> vx = +spd, h = standard, jump = True
        6: DUCK       -> vx = 0,    h = duck_h,   jump = False

    Returns:
        (vx, current_h, is_jump_action)
    """
    is_left = (action == ACTION_LEFT) | (action == ACTION_JUMP_LEFT)
    is_right = (action == ACTION_RIGHT) | (action == ACTION_JUMP_RIGHT)
    vx = jnp.where(is_left, -player_speed, jnp.where(is_right, player_speed, 0.0))

    is_duck = action == ACTION_DUCK
    current_h = jnp.where(is_duck, duck_h, standard_h)

    is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)

    return vx, current_h, is_jump_action
