"""Tests for view_policy.py Pygame rendering pipeline and host state synchronization."""

import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Force dummy video and audio drivers for headless test environments
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.envs.common import MODE_CLASSIC, MODE_REMASTERED, MODE_HYBRID
from view_policy import (
    HostEnvState,
    draw_arena,
    draw_boss_core,
    draw_cross_laser,
    draw_falling_debris,
    draw_player,
    draw_remastered_hazards,
    draw_hud,
    draw_osd,
)


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    pygame.font.init()
    yield
    pygame.quit()


@pytest.mark.parametrize("mode", [MODE_CLASSIC, MODE_REMASTERED, MODE_HYBRID])
def test_host_env_state_and_rendering_pipeline(mode):
    """Verifies that HostEnvState converts properly and all rendering layers execute without error."""
    env = LotusPhase1Env()
    params = EnvParams(mode=mode)
    rng = jax.random.PRNGKey(42)

    surface = pygame.Surface((int(params.screen_width), int(params.screen_height)))
    font = pygame.font.SysFont("Consolas", 14)

    # Reset
    obs, state = env.reset_env(rng, params)

    # Run 60 simulation steps and test rendering pipeline
    for step in range(60):
        rng, rng_step = jax.random.split(rng)
        action = step % 7
        obs, state, reward, done, info = env.step_env(rng_step, state, action, params)

        # Batch host synchronization
        host_state = HostEnvState(state)

        # Ensure types are numpy/native scalars, not JAX DeviceArray
        assert isinstance(host_state.player_x, float)
        assert isinstance(host_state.debris_x, np.ndarray)
        assert host_state.debris_x.shape == (30,)

        # Render all layers
        surface.fill((10, 10, 20))
        draw_arena(surface, params)

        if params.mode != MODE_REMASTERED:
            draw_cross_laser(surface, host_state, params)

        if params.mode != MODE_CLASSIC:
            draw_remastered_hazards(surface, host_state, params)

        draw_boss_core(surface, host_state, params)
        draw_falling_debris(surface, host_state)
        draw_player(surface, host_state, params, current_action=action)
        draw_hud(surface, host_state, params, font, font)
        draw_osd(surface, host_state, params, action=action, cumulative_reward=10.0, speed_mult=1.0, paused=False, font=font)
        draw_osd(surface, host_state, params, action=action, cumulative_reward=10.0, speed_mult=1.0, paused=True, font=font)


def test_remastered_hazard_states():
    """Specifically tests active Remastered hazards: electric floor, overload, tracking laser."""
    env = LotusPhase1Env()
    params = EnvParams(mode=MODE_REMASTERED)
    rng = jax.random.PRNGKey(123)
    surface = pygame.Surface((int(params.screen_width), int(params.screen_height)))
    font = pygame.font.SysFont("Consolas", 14)

    obs, state = env.reset_env(rng, params)

    # Force Remastered hazard states to verify rendering branches
    state = state.replace(
        is_overload=True,
        overload_timer=24.5,
        electric_floor_active=True,
        electric_floor_warning=1.5,
        tracking_laser_state=1,  # TRACKING
        tracking_laser_lock_x=650.0,
    )
    host_state = HostEnvState(state)
    draw_remastered_hazards(surface, host_state, params)
    draw_hud(surface, host_state, params, font, font)

    # Force firing state with friendly fire hitting boss core
    state = state.replace(
        electric_floor_warning=0.0,  # DETONATING
        tracking_laser_state=2,      # FIRING
        tracking_laser_lock_x=683.0, # Boss center
        shield_active=False,
    )
    host_state = HostEnvState(state)
    draw_remastered_hazards(surface, host_state, params)
    draw_hud(surface, host_state, params, font, font)
