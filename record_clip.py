"""Headless Recorder: Generates animated GIF and keyframe screenshots of trained policy."""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from PIL import Image

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.wrappers.flatten_obs import FlattenObservationWrapper
from view_policy import (
    COLOR_BG,
    HostEnvState,
    draw_arena,
    draw_boss_core,
    draw_falling_debris,
    draw_hud,
    draw_osd,
    draw_player,
    draw_remastered_hazards,
    load_trained_policy,
)


def record_policy_clip(
    checkpoint_path: str = "checkpoints/mode_1/step_1000",
    mode: int = 1,
    num_frames: int = 240,  # 4 seconds at 60 FPS
    frame_stride: int = 3,  # Record every 3rd frame (20 FPS GIF for small size)
    output_gif: str = "policy_preview.gif",
    output_png: str = "policy_preview.png",
):
    print(f"[Recorder] Loading policy from {checkpoint_path}...")
    policy_params, network = load_trained_policy(checkpoint_path, mode)
    jit_apply = jax.jit(network.apply)

    env = LotusPhase1Env()
    params = EnvParams(mode=mode)
    jit_reset = jax.jit(env.reset_env)
    jit_step = jax.jit(env.step_env)
    jit_get_obs = jax.jit(env.get_observation)

    pygame.init()
    pygame.font.init()

    win_w = int(params.screen_width)
    win_h = int(params.screen_height)
    surface = pygame.Surface((win_w, win_h))

    font_osd = pygame.font.SysFont("Consolas, Menlo, monospace", 15)
    font_large = pygame.font.SysFont("Consolas, Menlo, monospace", 18, bold=True)
    font_small = pygame.font.SysFont("Consolas, Menlo, monospace", 13)

    rng = jax.random.PRNGKey(1004)
    rng, rng_reset = jax.random.split(rng)
    obs, state = jit_reset(rng_reset, params)

    frames = []
    cumulative_reward = 0.0
    action = 0

    print(f"[Recorder] Simulating and recording {num_frames} frames...")
    for frame_idx in range(num_frames):
        # 1. Greedy policy inference
        obs_single = jit_get_obs(state, params)[None, :]
        logits, _ = jit_apply(policy_params, obs_single)
        action = int(jnp.argmax(logits[0]))

        # 2. Step environment
        rng, rng_step = jax.random.split(rng)
        obs, state, reward, done, info = jit_step(rng_step, state, action, params)
        cumulative_reward += float(reward)

        if bool(done):
            rng, rng_reset = jax.random.split(rng)
            obs, state = jit_reset(rng_reset, params)
            cumulative_reward = 0.0
            action = 0

        # 3. Synchronize state
        host_state = HostEnvState(state)

        # 4. Render
        surface.fill(COLOR_BG)
        draw_arena(surface, params)
        draw_remastered_hazards(surface, host_state, params)
        draw_boss_core(surface, host_state, params)
        draw_falling_debris(surface, host_state)
        draw_player(surface, host_state, params, current_action=action)
        draw_hud(surface, host_state, params, font_large, font_small)
        draw_osd(
            surface,
            host_state,
            params,
            action=action,
            cumulative_reward=cumulative_reward,
            speed_mult=1.0,
            paused=False,
            font=font_osd,
        )

        # Save primary screenshot at 1 second mark
        if frame_idx == 60:
            view = pygame.surfarray.array3d(surface)
            view = np.transpose(view, (1, 0, 2))  # (W, H, C) -> (H, W, C)
            img = Image.fromarray(view)
            img.save(output_png)
            print(f"[Recorder] Saved preview screenshot to {output_png}")

        # Capture GIF frame
        if frame_idx % frame_stride == 0:
            view = pygame.surfarray.array3d(surface)
            view = np.transpose(view, (1, 0, 2))
            # Resize by 0.5 to keep GIF compact
            img = Image.fromarray(view)
            img_resized = img.resize((win_w // 2, win_h // 2), Image.Resampling.BILINEAR)
            frames.append(img_resized)

    # Save animated GIF
    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / (60 / frame_stride)),  # ~50ms per frame
            loop=0,
            optimize=True,
        )
        print(f"[Recorder] Successfully generated animated GIF: {output_gif} ({len(frames)} frames)")

    pygame.quit()


if __name__ == "__main__":
    record_policy_clip()
