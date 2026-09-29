"""Detailed Greedy Evaluation of Remastered Mode 1 Checkpoint using RolloutRunner."""

import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.wrappers.flatten_obs import FlattenObservationWrapper
from maple_gymnax.wrappers.rollout_runner import RolloutRunner
from train_ppo import ActorCritic


def evaluate_checkpoint(checkpoint_path: str, num_episodes: int = 10, eval_steps: int = 1200):
    abs_path = os.path.abspath(checkpoint_path)
    print(f"[Eval] Loading checkpoint: {abs_path}")

    mode = 1  # Remastered
    env = LotusPhase1Env()
    params = EnvParams(mode=mode, max_steps_in_episode=eval_steps)
    wrapped = FlattenObservationWrapper(env)
    network = ActorCritic(action_dim=7)

    obs_dim = 142 if params.is_remastered else 130
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    dummy_params = network.init(jax.random.PRNGKey(0), dummy_obs)

    checkpointer = ocp.StandardCheckpointer()
    try:
        restored_params = checkpointer.restore(abs_path, target=dummy_params)
    except Exception:
        restored_params = checkpointer.restore(abs_path)
    checkpointer.close()

    def greedy_policy(obs, state, key):
        logits, _ = network.apply(restored_params, obs)
        return jnp.argmax(logits, axis=-1)

    runner = RolloutRunner(wrapped, params, policy_fn=greedy_policy)
    jit_run = jax.jit(runner.run, static_argnums=(2,))

    master_key = jax.random.PRNGKey(1004)
    results = []

    print(f"\n{'='*95}")
    print(f"Greedy Policy Evaluation (Remastered Mode 1, {num_episodes} Episodes, {eval_steps} Max Steps)")
    print(f"{'='*95}")
    print(f"{'Ep':>3} | {'Steps':>6} | {'Reward':>8} | {'Final HP':>8} | {'Boss Shield':>11} | {'Shield Dmg':>10} | {'Gauge':>6} | {'Survived':>8} | {'Timeout':>7} | {'Boss Hits':>9}")
    print(f"{'-'*95}")

    for ep in range(num_episodes):
        master_key, k_reset, k_run = jax.random.split(master_key, 3)
        init_obs, init_state = wrapped.reset(k_reset, params)
        final_state, traj = jit_run(k_run, init_state, eval_steps)

        # Host conversion
        tot_reward = float(jnp.sum(traj["reward"]))
        survived = bool(final_state.player_hp > 0.0)
        final_hp = float(final_state.player_hp)
        final_shield = float(final_state.boss_shield)
        shield_damage = float(params.boss_shield_max - final_shield)
        gauge = float(final_state.security_gauge)

        # Find actual termination step if dead before eval_steps
        dones = np.asarray(traj["done"])
        done_indices = np.where(dones)[0]
        survival_steps = int(done_indices[0] + 1) if len(done_indices) > 0 else eval_steps
        cleared_by_timeout = survived and (survival_steps >= eval_steps)
        boss_hit_events = int(jnp.sum(traj["info"]["boss_hit_event"])) if "boss_hit_event" in traj.get("info", {}) else None

        # Trajectory analytics
        actions = np.asarray(traj["action"])[:survival_steps]
        tl_player_hits = int(jnp.sum(traj["info"]["tracking_laser_hit_player"][:survival_steps])) if "tracking_laser_hit_player" in traj.get("info", {}) else 0
        tl_boss_ticks = int(jnp.sum(traj["info"]["tracking_laser_hit_boss"][:survival_steps])) if "tracking_laser_hit_boss" in traj.get("info", {}) else 0
        debris_hits = int(jnp.sum(traj["info"]["debris_hit"][:survival_steps])) if "debris_hit" in traj.get("info", {}) else 0
        
        # Clean vs Dirty baiting events
        if "boss_hit_event" in traj.get("info", {}) and "tracking_laser_hit_player" in traj.get("info", {}):
            boss_events = np.asarray(traj["info"]["boss_hit_event"][:survival_steps])
            p_hits = np.asarray(traj["info"]["tracking_laser_hit_player"][:survival_steps])
            clean_baits = int(np.sum(boss_events & ~p_hits))
            dirty_baits = int(np.sum(boss_events & p_hits))
        else:
            clean_baits = 0
            dirty_baits = 0

        res = {
            "episode": ep + 1,
            "steps": survival_steps,
            "reward": tot_reward,
            "hp": final_hp,
            "final_shield": final_shield,
            "shield_damage": shield_damage,
            "gauge": gauge,
            "survived": survived,
            "cleared_by_timeout": cleared_by_timeout,
            "boss_hit_events": boss_hit_events,
            "tl_player_hits": tl_player_hits,
            "tl_boss_ticks": tl_boss_ticks,
            "debris_hits": debris_hits,
            "clean_baits": clean_baits,
            "dirty_baits": dirty_baits,
            "actions": actions,
        }
        results.append(res)

        boss_hits_str = f"{boss_hit_events:4d}" if boss_hit_events is not None else " N/A"
        print(
            f"{ep+1:3d} | {survival_steps:6d} | {tot_reward:8.2f} | {final_hp:8.1f} | "
            f"{final_shield:10.1f} | {shield_damage:10.1f} | {gauge:5.3f} | {str(survived):>8} | {str(cleared_by_timeout):>7} | {boss_hits_str:>9}"
        )

    print(f"{'='*95}")

    # Summary Statistics
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    avg_hp = np.mean([r["hp"] for r in results])
    avg_shield_dmg = np.mean([r["shield_damage"] for r in results])
    avg_gauge = np.mean([r["gauge"] for r in results])
    survival_rate = np.mean([1.0 if r["survived"] else 0.0 for r in results]) * 100.0
    timeout_clear_rate = np.mean([1.0 if r["cleared_by_timeout"] else 0.0 for r in results]) * 100.0
    boss_hits = [r["boss_hit_events"] for r in results if r["boss_hit_events"] is not None]
    tot_tl_player_hits = sum(r["tl_player_hits"] for r in results)
    tot_clean_baits = sum(r["clean_baits"] for r in results)
    tot_dirty_baits = sum(r["dirty_baits"] for r in results)
    tot_debris_hits = sum(r["debris_hits"] for r in results)

    # Action distribution aggregation
    all_actions = np.concatenate([r["actions"] for r in results]) if results else np.array([])
    action_names = ["NOOP", "LEFT", "RIGHT", "DOWN", "JUMP", "JUMP_LEFT", "JUMP_RIGHT"]
    act_counts = np.bincount(all_actions, minlength=7)
    act_pcts = (act_counts / len(all_actions) * 100.0) if len(all_actions) > 0 else np.zeros(7)

    print(f"\n[Summary Across {num_episodes} Episodes]")
    print(f"  * Average Survival Steps   : {avg_steps:6.1f} / {eval_steps} ({avg_steps / eval_steps * 100:.1f}%)")
    print(f"  * Full Survival Rate       : {survival_rate:5.1f}%")
    print(f"  * Cleared by Timeout Rate  : {timeout_clear_rate:5.1f}%")
    if boss_hits:
        avg_boss_hits = np.mean(boss_hits)
        print(f"  * Average Boss Hit Events  : {avg_boss_hits:5.1f}")
    print(f"  * Average Cumulative Ret   : {avg_reward:8.2f}")
    print(f"  * Average Final HP         : {avg_hp:5.1f} / {params.player_max_hp:.0f}")
    print(f"  * Average Shield Damage    : {avg_shield_dmg:5.1f} / {params.boss_shield_max:.0f}")
    print(f"  * Average Security Gauge   : {avg_gauge:6.4f}")
    print(f"\n[Mode 1 Remastered Hazard Telemetry (Truthful - Rule 16)]")
    print(f"  * Tracking Laser Player Hits (Self-Dmg): {tot_tl_player_hits:5d} (Avg {tot_tl_player_hits/num_episodes:.1f}/ep)")
    print(f"  * Clean Bait Redirections (Clean)       : {tot_clean_baits:5d} (Avg {tot_clean_baits/num_episodes:.1f}/ep)")
    print(f"  * Dirty Bait Redirections (Self-Hit)    : {tot_dirty_baits:5d} (Avg {tot_dirty_baits/num_episodes:.1f}/ep)")
    print(f"  * Falling Debris Hits                   : {tot_debris_hits:5d} (Avg {tot_debris_hits/num_episodes:.1f}/ep)")
    print(f"\n[Action Distribution across {len(all_actions)} steps]")
    for i, name in enumerate(action_names):
        print(f"  * {name:<11}: {act_counts[i]:6d} ({act_pcts[i]:5.1f}%)")
    jump_pct = act_pcts[4] + act_pcts[5] + act_pcts[6]
    print(f"  * Total Jump Mobility (JUMP+J_L+J_R)   : {jump_pct:5.1f}%")
    print(f"{'='*95}\n")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed Evaluation of Lotus Phase 1 Checkpoint")
    parser.add_argument("checkpoint_path", nargs="?", default="checkpoints/step_35000", help="Path to checkpoint directory")
    parser.add_argument("--checkpoint_path", dest="checkpoint_path_opt", default=None, help="Optional flag for checkpoint path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_steps", type=int, default=1200, help="Max steps per episode")

    args = parser.parse_args()
    target_ckpt = args.checkpoint_path_opt if args.checkpoint_path_opt else args.checkpoint_path
    evaluate_checkpoint(target_ckpt, num_episodes=args.episodes, eval_steps=args.eval_steps)

