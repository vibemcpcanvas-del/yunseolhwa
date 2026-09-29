## 2026-09-29T11:29:15Z

User request for orchestrator_3:
Task: Autonomous Micro-Movement & Threat-Gated Evasion: Eliminating jump-spam local minima via Action Cost, Jitter Regularization, and Tap-Dodging Corridor Rewards in Maple Gymnax Lotus Phase 1.
Working directory: c:/Users/ROCmAdmin/Documents/antigravity/bold-faraday
Branch: feat/debris-threat-obs
Integrity mode: development
Requirements:
- R1: Action Cost & Energy Regularization Engine in src/maple_gymnax/envs/lotus_phase1.py (_compute_reward, EnvState.last_action, r_action_jump = -0.05, r_jitter = -0.02 * (action != state.last_action))
- R2: Tap-Dodging Hazard Corridor & Overhead Repulsion (r_airborne_hazard = -0.35, r_tap_dodge = 0.25, Gaussian potential scaling -0.05 -> -0.30)
- R3: Transparent Real-Time Console Telemetry & Metric Overhaul in train_ppo.py (Survival(s), DebrisHits/ep, JumpRatio%)
- R4: Colab T4 Cloud Burst Training from Scratch (account_1, 16384 parallel envs, 35000 updates, SPS ~980,000, checkpoints synced every 1200 updates)
Acceptance Criteria: Gate 1 (Behavioral Distribution Shift) and Gate 2 (Debris Evasion & Survival Breakthrough)
Relevant Skills & Rules:
- Skill: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\skills\colab-burst-training\SKILL.md
- Rule: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\rules\jax-gymnax-rl.md
