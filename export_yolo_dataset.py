"""Synthetic YOLO Dataset Generator for MapleStory Lotus Phase 1.

Leverages JAX simulation and Pygame headless renderer to export perfectly labeled
training images and normalized bounding box annotation files (.txt) for YOLOv8/v11.

Classes:
  0: player
  1: boss_core
  2: debris_small  (radius 16px, 10% dmg)
  3: debris_medium (radius 24px, 20% dmg)
  4: debris_large  (radius 36px, 30% dmg)
  5: tracking_laser (lock vertical column)
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import argparse
import random
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from PIL import Image

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from view_policy import (
    COLOR_BG,
    HostEnvState,
    draw_arena,
    draw_boss_core,
    draw_falling_debris,
    draw_player,
    draw_remastered_hazards,
)


def export_yolo_dataset(
    output_dir: str = "dataset_yolo",
    num_frames: int = 1000,
    val_ratio: float = 0.2,
    mode: int = 1,
    seed: int = 2026,
):
    print(f"[*] Initializing Synthetic YOLO Dataset Generator...")
    print(f"    Target: {num_frames} frames -> {output_dir} (Train: {1-val_ratio:.0%}, Val: {val_ratio:.0%})")

    img_train_dir = os.path.join(output_dir, "images", "train")
    img_val_dir = os.path.join(output_dir, "images", "val")
    lbl_train_dir = os.path.join(output_dir, "labels", "train")
    lbl_val_dir = os.path.join(output_dir, "labels", "val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    env = LotusPhase1Env()
    params = EnvParams(mode=mode)
    jit_reset = jax.jit(env.reset_env)
    jit_step = jax.jit(env.step_env)

    pygame.init()
    win_w = int(params.screen_width)   # 1366
    win_h = int(params.screen_height)  # 768
    surface = pygame.Surface((win_w, win_h))

    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = jit_reset(rng_reset, params)

    random.seed(seed)
    saved_count = 0

    print(f"[*] Generating synthetic frames and ground truth bounding boxes...")

    for frame_idx in range(num_frames):
        # Step with semi-random exploratory action
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = int(jax.random.randint(rng_act, shape=(), minval=0, maxval=7))
        obs, state, reward, done, info = jit_step(rng_step, state, action, params)

        if bool(done):
            rng, rng_reset = jax.random.split(rng)
            obs, state = jit_reset(rng_reset, params)

        host_state = HostEnvState(state)

        # 1. Render frame
        surface.fill(COLOR_BG)
        draw_arena(surface, params, host_state)
        draw_remastered_hazards(surface, host_state, params)
        draw_boss_core(surface, host_state, params)
        draw_falling_debris(surface, host_state)
        draw_player(surface, host_state, params, current_action=action)

        # Convert pygame surface to image
        view = pygame.surfarray.array3d(surface)
        view = np.transpose(view, (1, 0, 2))  # (W, H, C) -> (H, W, C)
        img = Image.fromarray(view)

        # 2. Extract 100% accurate Ground Truth YOLO Bounding Boxes
        # Format: <class_id> <x_center/W> <y_center/H> <width/W> <height/H>
        yolo_boxes = []

        # Class 0: Player
        p_h = float(params.player_duck_h if action == 6 else params.player_h)
        p_cx = float(host_state.player_x) / win_w
        p_cy = float(host_state.player_y - p_h / 2.0) / win_h
        p_w = float(params.player_w) / win_w
        p_h_norm = p_h / win_h
        yolo_boxes.append(f"0 {p_cx:.6f} {p_cy:.6f} {p_w:.6f} {p_h_norm:.6f}")

        # Class 1: Boss Core
        b_cx = float(params.core_x) / win_w
        b_cy = float(params.core_y) / win_h
        b_w = float(params.core_radius * 2.0) / win_w
        b_h = float(params.core_radius * 2.0) / win_h
        yolo_boxes.append(f"1 {b_cx:.6f} {b_cy:.6f} {b_w:.6f} {b_h:.6f}")

        # Class 2, 3, 4: Falling Debris
        for i in range(len(host_state.debris_active)):
            if host_state.debris_active[i]:
                deb_x = float(host_state.debris_x[i])
                deb_y = float(host_state.debris_y[i])
                deb_r = float(host_state.debris_radius[i])
                deb_t = int(host_state.debris_type[i])  # 0: small, 1: med, 2: large
                cls_id = 2 + deb_t  # 2: small, 3: med, 4: large

                d_cx = deb_x / win_w
                d_cy = deb_y / win_h
                d_w = (deb_r * 2.0) / win_w
                d_h = (deb_r * 2.0) / win_h

                # Clip bounds
                if 0.0 <= d_cx <= 1.0 and 0.0 <= d_cy <= 1.0:
                    yolo_boxes.append(f"{cls_id} {d_cx:.6f} {d_cy:.6f} {d_w:.6f} {d_h:.6f}")

        # Class 5: Tracking Laser (during tracking or firing)
        if host_state.tracking_laser_state > 0:
            tl_x = float(host_state.tracking_laser_lock_x)
            tl_cx = tl_x / win_w
            tl_cy = 0.5
            tl_w = 40.0 / win_w  # ~40px beam width
            tl_h = 1.0
            yolo_boxes.append(f"5 {tl_cx:.6f} {tl_cy:.6f} {tl_w:.6f} {tl_h:.6f}")

        # Split train vs val
        is_val = random.random() < val_ratio
        sub_img_dir = img_val_dir if is_val else img_train_dir
        sub_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

        file_id = f"frame_{frame_idx:06d}"
        img_path = os.path.join(sub_img_dir, f"{file_id}.jpg")
        lbl_path = os.path.join(sub_lbl_dir, f"{file_id}.txt")

        # Save image (JPEG 90% quality)
        img.save(img_path, "JPEG", quality=90)

        # Save label txt
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_boxes) + "\n")

        saved_count += 1
        if (frame_idx + 1) % 200 == 0:
            print(f"    Exported {frame_idx + 1}/{num_frames} frames...")

    # Write data.yaml for Ultralytics YOLO training
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: player
  1: boss_core
  2: debris_small
  3: debris_medium
  4: debris_large
  5: tracking_laser
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    pygame.quit()
    print(f"\n[SUCCESS] Successfully exported {saved_count} synthetic labeled frames to '{output_dir}'!")
    print(f"          YOLO Config: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export synthetic labeled dataset for YOLOv8/v11")
    parser.add_argument("--output_dir", type=str, default="dataset_yolo", help="Target output dataset directory")
    parser.add_argument("--num_frames", type=int, default=500, help="Number of frames to generate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--mode", type=int, default=1, help="Mode (1: Remastered)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")

    args = parser.parse_args()
    export_yolo_dataset(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        val_ratio=args.val_ratio,
        mode=args.mode,
        seed=args.seed,
    )
