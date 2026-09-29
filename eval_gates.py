#!/usr/bin/env python3
"""eval_gates.py - Automated Gate 1 & Gate 2 Acceptance Verification CLI for Maple Gymnax Checkpoints.

Evaluates greedy policy rollouts for Remastered Lotus Phase 1 checkpoints and programmatically
verifies:
  - Gate 1 (Early Checkpoints step_4800+):
      * JUMP_RIGHT < 20.0%
      * Grounded (NOOP + LEFT + RIGHT + DOWN) > 50.0%
  - Gate 2 (Mature Checkpoints):
      * DebrisHits <= 3.5 / ep
      * Survival steps >= 1200 (20.0s at 60 Hz)
      * Boss Shield Damage >= 140.0 / 200

Exit Codes:
  0: Evaluated gates PASSED
  1: Evaluated gates FAILED or checkpoint evaluation error
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.wrappers.flatten_obs import FlattenObservationWrapper
from maple_gymnax.wrappers.rollout_runner import RolloutRunner
from train_ppo import ActorCritic

try:
    from maple_gymnax.envs.common import (
        ACTION_NOOP,
        ACTION_LEFT,
        ACTION_RIGHT,
        ACTION_DOWN,
        ACTION_JUMP,
        ACTION_JUMP_LEFT,
        ACTION_JUMP_RIGHT,
    )
except ImportError:
    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3
    ACTION_JUMP = 4
    ACTION_JUMP_LEFT = 5
    ACTION_JUMP_RIGHT = 6


# Action labels matching indices 0..6
ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "DOWN", "JUMP", "JUMP_LEFT", "JUMP_RIGHT"]
GROUND_ACTION_INDICES = [ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN]
JUMP_ACTION_INDICES = [ACTION_JUMP, ACTION_JUMP_LEFT, ACTION_JUMP_RIGHT]


def run_greedy_evaluation(
    checkpoint_path: str,
    num_episodes: int = 10,
    eval_steps: int = 1200,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs deterministic greedy evaluation of a checkpoint across specified episodes.
    
    Returns:
        (episode_results, summary_statistics)
    """
    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {abs_path}")

    if verbose:
        print(f"[eval_gates] Loading checkpoint: {abs_path}")

    mode = 1  # Remastered
    env = LotusPhase1Env()
    params = EnvParams(mode=mode, max_steps_in_episode=eval_steps)
    wrapped = FlattenObservationWrapper(env)
    network = ActorCritic(action_dim=7)

    obs_dim = 172 if params.is_remastered else 130
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
    results: List[Dict[str, Any]] = []

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

        dones = np.asarray(traj["done"])
        done_indices = np.where(dones)[0]
        survival_steps = int(done_indices[0] + 1) if len(done_indices) > 0 else eval_steps
        cleared_by_timeout = survived and (survival_steps >= eval_steps)
        boss_hit_events = int(jnp.sum(traj["info"]["boss_hit_event"])) if "boss_hit_event" in traj.get("info", {}) else 0

        # Trajectory slice up to survival steps
        actions = np.asarray(traj["action"])[:survival_steps]
        tl_player_hits = int(jnp.sum(traj["info"]["tracking_laser_hit_player"][:survival_steps])) if "tracking_laser_hit_player" in traj.get("info", {}) else 0
        tl_boss_ticks = int(jnp.sum(traj["info"]["tracking_laser_hit_boss"][:survival_steps])) if "tracking_laser_hit_boss" in traj.get("info", {}) else 0
        debris_hits = int(jnp.sum(traj["info"]["debris_hit"][:survival_steps])) if "debris_hit" in traj.get("info", {}) else 0

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

    # Aggregate summary
    avg_steps = float(np.mean([r["steps"] for r in results]))
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_hp = float(np.mean([r["hp"] for r in results]))
    avg_shield_dmg = float(np.mean([r["shield_damage"] for r in results]))
    avg_gauge = float(np.mean([r["gauge"] for r in results]))
    survival_rate = float(np.mean([1.0 if r["survived"] else 0.0 for r in results]) * 100.0)
    timeout_clear_rate = float(np.mean([1.0 if r["cleared_by_timeout"] else 0.0 for r in results]) * 100.0)
    tot_debris_hits = sum(r["debris_hits"] for r in results)
    avg_debris_hits = float(tot_debris_hits / num_episodes)
    tot_clean_baits = sum(r["clean_baits"] for r in results)
    tot_dirty_baits = sum(r["dirty_baits"] for r in results)
    tot_tl_player_hits = sum(r["tl_player_hits"] for r in results)

    all_actions = np.concatenate([r["actions"] for r in results]) if results else np.array([], dtype=int)
    total_actions = len(all_actions)
    act_counts = np.bincount(all_actions, minlength=7)[:7]
    act_pcts = (act_counts / total_actions * 100.0) if total_actions > 0 else np.zeros(7)

    jump_right_pct = float(act_pcts[ACTION_JUMP_RIGHT])
    grounded_pct = float(sum(act_pcts[idx] for idx in GROUND_ACTION_INDICES))
    total_jump_pct = float(sum(act_pcts[idx] for idx in JUMP_ACTION_INDICES))

    summary = {
        "checkpoint": abs_path,
        "num_episodes": num_episodes,
        "eval_steps": eval_steps,
        "total_actions": total_actions,
        "avg_steps": avg_steps,
        "avg_survival_sec": float(avg_steps / 60.0),
        "survival_rate_pct": survival_rate,
        "timeout_clear_rate_pct": timeout_clear_rate,
        "avg_reward": avg_reward,
        "avg_final_hp": avg_hp,
        "avg_shield_dmg": avg_shield_dmg,
        "avg_gauge": avg_gauge,
        "tot_debris_hits": tot_debris_hits,
        "avg_debris_hits": avg_debris_hits,
        "tot_clean_baits": tot_clean_baits,
        "tot_dirty_baits": tot_dirty_baits,
        "tot_tl_player_hits": tot_tl_player_hits,
        "act_counts": {ACTION_NAMES[i]: int(act_counts[i]) for i in range(7)},
        "act_pcts": {ACTION_NAMES[i]: float(act_pcts[i]) for i in range(7)},
        "jump_right_pct": jump_right_pct,
        "grounded_pct": grounded_pct,
        "total_jump_pct": total_jump_pct,
    }

    return results, summary


def evaluate_gates(summary: Dict[str, Any], gate_mode: str = "all") -> Dict[str, Any]:
    """Evaluates summary metrics against Gate 1 and Gate 2 criteria.
    
    Gate 1:
      - JUMP_RIGHT < 20.0%
      - Grounded (NOOP + LEFT + RIGHT + DOWN) > 50.0%
    
    Gate 2:
      - DebrisHits <= 3.5 / ep
      - Survival steps >= 1200.0 (20.0s)
      - ShieldDamage >= 140.0 / 200
    """
    jump_right_pct = summary["jump_right_pct"]
    grounded_pct = summary["grounded_pct"]
    avg_debris_hits = summary["avg_debris_hits"]
    avg_steps = summary["avg_steps"]
    avg_shield_dmg = summary["avg_shield_dmg"]

    # Gate 1 checks
    g1_jump_right_pass = bool(jump_right_pct < 20.0)
    g1_grounded_pass = bool(grounded_pct > 50.0)
    gate1_pass = g1_jump_right_pass and g1_grounded_pass

    # Gate 2 checks
    g2_debris_pass = bool(avg_debris_hits <= 3.5)
    g2_survival_pass = bool(avg_steps >= 1200.0)
    g2_shield_pass = bool(avg_shield_dmg >= 140.0)
    gate2_pass = g2_debris_pass and g2_survival_pass and g2_shield_pass

    # Overall pass logic based on gate_mode
    if gate_mode == "1":
        overall_pass = gate1_pass
    elif gate_mode == "2":
        overall_pass = gate2_pass
    elif gate_mode == "all":
        overall_pass = gate1_pass and gate2_pass
    elif gate_mode == "any":
        overall_pass = gate1_pass or gate2_pass
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")

    verdict = {
        "checkpoint": summary["checkpoint"],
        "gate_mode": gate_mode,
        "overall_pass": overall_pass,
        "gate1": {
            "passed": gate1_pass,
            "checks": {
                "jump_right_lt_20pct": {
                    "target": "< 20.0%",
                    "actual": f"{jump_right_pct:.1f}%",
                    "passed": g1_jump_right_pass,
                },
                "grounded_gt_50pct": {
                    "target": "> 50.0%",
                    "actual": f"{grounded_pct:.1f}%",
                    "passed": g1_grounded_pass,
                },
            },
        },
        "gate2": {
            "passed": gate2_pass,
            "checks": {
                "debris_hits_le_3_5": {
                    "target": "<= 3.5 / ep",
                    "actual": f"{avg_debris_hits:.2f} / ep",
                    "passed": g2_debris_pass,
                },
                "survival_steps_ge_1200": {
                    "target": ">= 1200.0 steps (20.0s)",
                    "actual": f"{avg_steps:.1f} steps ({summary.get('avg_survival_sec', avg_steps / 60.0):.1f}s)",
                    "passed": g2_survival_pass,
                },
                "shield_damage_ge_140": {
                    "target": ">= 140.0 / 200",
                    "actual": f"{avg_shield_dmg:.1f} / 200",
                    "passed": g2_shield_pass,
                },
            },
        },
    }
    return verdict


def format_ascii_report(summary: Dict[str, Any], verdict: Dict[str, Any]) -> str:
    """Formats a comprehensive ASCII evaluation and gate verification report."""
    lines = []
    bar = "=" * 80
    subbar = "-" * 80

    lines.append(bar)
    lines.append("  MAPLE GYMNAX LOTUS PHASE 1 - GATE ACCEPTANCE REPORT")
    lines.append(bar)
    lines.append(f"  Checkpoint  : {summary['checkpoint']}")
    lines.append(f"  Episodes    : {summary['num_episodes']} episodes | Horizon: {summary['eval_steps']} steps (max {summary['eval_steps']/60.0:.1f}s)")
    lines.append(f"  Gate Mode   : {verdict['gate_mode'].upper()}")
    lines.append(subbar)

    lines.append("  [Trajectory Summary]")
    lines.append(f"    * Average Survival       : {summary['avg_steps']:6.1f} steps ({summary['avg_survival_sec']:4.1f}s) | Timeout Clears: {summary['timeout_clear_rate_pct']:5.1f}%")
    lines.append(f"    * Average Cumulative Ret : {summary['avg_reward']:8.2f} | Final HP: {summary['avg_final_hp']:5.1f} / 100")
    lines.append(f"    * Boss Shield Damage     : {summary['avg_shield_dmg']:5.1f} / 200 | Final Gauge: {summary['avg_gauge']:6.4f}")
    lines.append(f"    * Falling Debris Hits    : {summary['tot_debris_hits']:5d} total (Avg {summary['avg_debris_hits']:.2f} hits/ep)")
    lines.append(f"    * Tracking Laser Player  : {summary['tot_tl_player_hits']:5d} (Avg {summary['tot_tl_player_hits']/summary['num_episodes']:.1f}/ep)")
    lines.append(f"    * Laser Bait Redirection : Clean {summary['tot_clean_baits']} vs Dirty {summary['tot_dirty_baits']}")
    lines.append(subbar)

    lines.append("  [Action Distribution Breakdown]")
    for name in ACTION_NAMES:
        cnt = summary["act_counts"][name]
        pct = summary["act_pcts"][name]
        category = "Grounded" if name in ["NOOP", "LEFT", "RIGHT", "DOWN"] else "Airborne"
        lines.append(f"    * {name:<11} : {cnt:6d} ({pct:5.1f}%) [{category}]")
    lines.append(f"    --> Total Grounded Mobility (NOOP+L+R+DOWN) : {summary['grounded_pct']:5.1f}%")
    lines.append(f"    --> Total Jump Mobility (JUMP+J_L+J_R)     : {summary['total_jump_pct']:5.1f}%")
    lines.append(subbar)

    lines.append("  [Gate Criteria Verification]")

    # Gate 1 table
    lines.append(f"  --- GATE 1: Behavioral Distribution Shift (Early Checkpoints step_4800+) ---")
    g1 = verdict["gate1"]
    for check_name, info in g1["checks"].items():
        status = "[ PASS ]" if info["passed"] else "[ FAIL ]"
        lines.append(f"    {status} {check_name:<25} | Actual: {info['actual']:<10} | Target: {info['target']}")
    g1_status = "[ GATE 1 PASSED ]" if g1["passed"] else "[ GATE 1 FAILED ]"
    lines.append(f"    ==> {g1_status}")
    lines.append("")

    # Gate 2 table
    lines.append(f"  --- GATE 2: Debris Evasion & Survival Breakthrough (Mature Checkpoints) ---")
    g2 = verdict["gate2"]
    for check_name, info in g2["checks"].items():
        status = "[ PASS ]" if info["passed"] else "[ FAIL ]"
        lines.append(f"    {status} {check_name:<25} | Actual: {info['actual']:<20} | Target: {info['target']}")
    g2_status = "[ GATE 2 PASSED ]" if g2["passed"] else "[ GATE 2 FAILED ]"
    lines.append(f"    ==> {g2_status}")
    lines.append(subbar)

    # Overall Verdict
    if verdict["overall_pass"]:
        lines.append("  OVERALL VERDICT: [ PASS ] (Exit Code 0)")
    else:
        lines.append("  OVERALL VERDICT: [ FAIL ] (Exit Code 1)")
    lines.append(bar)

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated Gate 1 & Gate 2 Acceptance Verification CLI for Maple Gymnax Checkpoints."
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        default=None,
        help="Path to checkpoint directory (e.g. checkpoints/step_4800)",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_opt",
        default=None,
        help="Explicit flag for checkpoint path",
    )
    parser.add_argument(
        "--gate",
        choices=["1", "2", "all", "any"],
        default="all",
        help="Which gate to enforce for exit code (default: 'all')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1200,
        help="Maximum rollout horizon steps per episode (default: 1200 = 20.0s)",
    )
    parser.add_argument(
        "--export_json",
        default=None,
        help="Optional path to export structured JSON verification results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate progress prints",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_target = args.checkpoint_opt or args.checkpoint_path
    if not checkpoint_target:
        # Default fallback to latest step_4800 if exists, or show error
        if os.path.exists("checkpoints/step_4800"):
            checkpoint_target = "checkpoints/step_4800"
        else:
            parser.error("checkpoint_path positional argument or --checkpoint flag is required.")

    try:
        results, summary = run_greedy_evaluation(
            checkpoint_path=checkpoint_target,
            num_episodes=args.episodes,
            eval_steps=args.eval_steps,
            verbose=not args.quiet,
        )
        verdict = evaluate_gates(summary, gate_mode=args.gate)
        report = format_ascii_report(summary, verdict)
        print(report)

        if args.export_json:
            export_payload = {
                "summary": summary,
                "verdict": verdict,
            }
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(export_payload, f, indent=2)
            print(f"[eval_gates] Exported JSON verification result to {args.export_json}")

        return 0 if verdict["overall_pass"] else 1

    except Exception as exc:
        print(f"[eval_gates] ERROR: Failed evaluating checkpoint {checkpoint_target}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
