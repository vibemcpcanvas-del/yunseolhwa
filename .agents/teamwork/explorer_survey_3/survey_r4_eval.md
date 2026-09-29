# Survey Report: Colab Cloud Burst Training (R4) & Gate 1/2 Evaluation Harness

**Target System**: Maple-Gymnax Lotus Phase 1 RL Training & Verification  
**Author**: `explorer_survey_3`  
**Date**: 2026-09-29  
**Parent**: `orchestrator_3`  

---

## 1. Executive Summary

This survey investigates **Requirement R4** (Colab T4 Cloud Burst Training from Scratch on `account_1`) and **Acceptance Criteria Gates 1 & 2** (Behavioral Distribution Shift and Debris Evasion / Survival Breakthrough).

### Key Takeaways
1. **Existing Cloud Burst Architecture is Complete & Operational**:
   - The repository possesses a fully integrated 5-account rotation, Cloudflare Quick Tunnel streaming, and Colab CLI automation pipeline (`run_colab_train.py`, `cloud/cloud_manager.py`, `cloud/colab_account_manager.py`, `cloud/colab_tunnel_bridge.py`, `cloud/colab_jax_train_job.py`).
   - `account_1` ("구글 메인 계정") is verified `READY` in `cloud/colab_accounts.json`.
2. **Clean-Slate Initialization Mechanism**:
   - `run_colab_train.py` defaults to `--resume_latest=True`. To satisfy R4's "training from scratch", `--no_resume` MUST be passed on invocation. This suppresses `resume_checkpoint/` inclusion in `upload_jax.tar`, compelling remote initialization at `step 0`.
3. **Checkpoint Sync Cadence (Every 1,200 Updates)**:
   - In `train_ppo.py`, when `checkpoint_interval_seconds > 0` (default 600.0s), saving defaults to time elapsed. To guarantee exact synchronization every 1,200 updates regardless of SPS fluctuations, `--checkpoint_interval 1200 --checkpoint_interval_seconds 0 --chunk_size 200` must be supplied (or `train_ppo.py` adjusted to trigger on either condition).
4. **Empirical Baseline Confirms Critical Jump-Spam Plateau**:
   - We evaluated existing checkpoints `step_4800` and `step_26400` using `eval_checkpoint.py`:
     - **At `step_4800`**: `JUMP_RIGHT` is **70.1%**, Total Jump Mobility is **83.0%**, Grounded actions are only **16.2%**, Debris hits are **8.7 hits/ep**, Survival is capped at **658.4 steps** (10.9s), and Boss Shield damage is **120.0 / 200**.
     - **At `step_26400`**: `JUMP_RIGHT` remains at **61.8%**, Total Jump Mobility is **77.2%**, Grounded actions are **15.2%**, Debris hits are **9.1 hits/ep**, Survival steps are **783.9 steps** (13.0s).
   - This provides empirical proof of the "bunny-hop" local minimum cited in `ORIGINAL_REQUEST.md`.
5. **Evaluation Harness Specification**:
   - `eval_checkpoint.py` accurately evaluates policy trajectories using JIT-compiled `RolloutRunner`, but lacks programmatic assertions, multi-checkpoint evaluation, exit codes, and automated Gate 1/2 pass-fail reporting.
   - We outline an automated evaluation harness (`eval_gates.py`) to systematically validate Gate 1 and Gate 2 against intermediate synced checkpoints.

---

## 2. Skill Review & Repository Asset Inventory

### 2.1 Skill Analysis: `colab-burst-training`
Defined in `.agents/skills/colab-burst-training/SKILL.md`:
- **5-Account Auto-Rotation**: Handles Colab quota exhaustion (`503 Quota Exhausted`) by automatically rotating credentials via WSL `Ubuntu-24.04-ROCmLab`.
- **Hybrid Precision**: Keeps simulation coordinates and physics collision projections in `float32`, while executing Actor-Critic MLP in `float16` on Tesla T4 (due to T4 lacking native `bfloat16` Tensor Core execution).
- **Cloudflare Quick Tunnel**: Zero-config reverse tunnel via `bin/cloudflared.exe` streaming checkpoints back to the host workstation at every chunk commit to protect against spot VM preemption.
- **Selective Packaging**: Packages strictly `src/`, `train_ppo.py`, `eval_checkpoint.py`, `pyproject.toml`, and optional resume state (~0.32 MB archive).

### 2.2 Repository Files & Scripts Inventory

| File Path | Role | Key Functionality |
| :--- | :--- | :--- |
| `run_colab_train.py` | Top-level CLI launcher | Entry point forwarding directly to `cloud.cloud_manager.main()` |
| `cloud/cloud_manager.py` | Master Cloud Orchestrator | Manages packaging, tunnel lifecycle, account rotation, Colab CLI execution, and resume step recovery |
| `cloud/colab_account_manager.py` | Combo Pool Manager | Tracks 5 Colab accounts, switches credentials in WSL, sweeps active sessions (`stop-all`) |
| `cloud/colab_accounts.json` | Account Registry | JSON registry of account IDs (`account_1` to `account_5`), status, and error logs |
| `cloud/colab_tunnel_bridge.py` | Tunnel & Data Bridge | Spawns `cloudflared.exe`, hosts HTTP endpoints (`/data`, `/script`, `/upload_checkpoint`, `/upload_chunk`), performs SHA256 integrity verification |
| `cloud/colab_jax_train_job.py` | Remote VM Worker Script | Runs inside Colab VM, installs dependencies, pulls tarball, generates `push_checkpoint.py`, invokes `train_ppo.py`, auto-unassigns VM |
| `cloud/colab_bridge.py` & `colab.bat` | Windows-to-WSL Bridge | Translates Windows drive paths (`C:\...`) into WSL mount paths (`/mnt/c/...`) and runs `/home/rocmuser/.local/bin/colab` |
| `eval_checkpoint.py` | Single-checkpoint Evaluator | Runs greedy argmax rollout across $N$ episodes using `RolloutRunner`, prints comprehensive ASCII telemetry table |
| `benchmarks/benchmark_sps.py` | Local SPS Benchmark | Evaluates environment simulation throughput across batch sizes (256 ~ 4,096) |
| `train_ppo.py` | PPO Training Kernel | Vectorized JAX/XLA PPO training loop with `make_train_step`, Orbax checkpointing, and `jax.debug.callback` |

---

## 3. Colab T4 Cloud Burst Training Pipeline (R4)

### 3.1 Triggering Training from Scratch on `account_1`

#### Account Verification & Cleanup
1. **Registry Verification**:
   ```bash
   python cloud/colab_account_manager.py list
   ```
   Verified Output:
   ```
   ★ [ACTIVE] account_1: 구글 메인 계정 | Status: READY | Last Error: None
     [IDLE] account_2: 구글 계정 2 | Status: READY | Last Error: None
     [IDLE] account_3: 구글 계정 3 | Status: READY | Last Error: None
     [IDLE] account_4: 구글 계정 4 | Status: READY | Last Error: None
     [IDLE] account_5: 구글 계정 5 | Status: READY | Last Error: None
   ```
2. **Pre-flight Session Sweep**:
   To prevent conflicts with existing dangling sessions:
   ```bash
   cloud\colab.bat stop -s jax_train
   ```
   or:
   ```bash
   python cloud/colab_account_manager.py stop-all jax_train
   ```

#### Invocation Command for R4
To enforce clean-slate initialization, 16,384 parallel environments, 35,000 updates, and checkpoints synced every 1,200 updates:
```bash
python run_colab_train.py \
  --accelerator gpu \
  --gpu_type T4 \
  --mode 1 \
  --num_envs 16384 \
  --num_steps 64 \
  --num_updates 35000 \
  --checkpoint_interval 1200 \
  --checkpoint_interval_seconds 0.0 \
  --chunk_size 200 \
  --dtype float16 \
  --no_resume
```

#### Why `--no_resume` is Mandatory:
In `cloud/cloud_manager.py`:
```python
parser.add_argument("--resume_latest", action="store_true", default=True, help="Auto-resume from latest verified checkpoint")
parser.add_argument("--no_resume", action="store_true", help="Start training fresh without loading previous checkpoint")
```
If `--no_resume` is omitted, `mgr.find_latest_verified_step(1)` detects existing checkpoints in `checkpoints/` (e.g. `step_26400`), packages them into `upload_jax.tar` as `resume_checkpoint/`, and `colab_jax_train_job.py` resumes from the old policy!
Passing `--no_resume` ensures `package_jax_codebase` creates a clean archive without any legacy checkpoints, triggering a true clean-slate initialization.

### 3.2 Real-Time Monitoring & Telemetry

1. **Console Telemetry**:
   - `colab_jax_train_job.py` passes `--log_interval 20` to `train_ppo.py`.
   - `train_ppo.py` triggers `_log_callback` via `jax.debug.callback` every 20 updates.
   - When Requirement R3 is implemented (by Milestone 3), the console output will print:
     `Survival(s)` (average survival seconds = length / 60.0), `DebrisHits/ep`, and `JumpRatio%`.
   - The remote output is piped back through the WSL Colab CLI session and printed in real-time to the host workstation console with the `[colab-jax]` prefix.

2. **Quota Monitoring & Auto-Rotation**:
   - `colab_jax_train_job.py` sets a 4-hour watchdog timer (`schedule_auto_unassign(14400)`).
   - If Colab terminates or runs out of compute quota (e.g., `503 Service Unavailable`, `Quota exceeded`), `cloud_manager.py` detects the termination keywords.
   - It captures the highest verified step from `JaxTunnelTransferManager`, gracefully stops the session, switches to `account_2` via `cam.switch_next_account()`, bundles the latest verified checkpoint, and re-launches the training job automatically.

### 3.3 Checkpoint Sync Architecture (Every 1,200 Updates)

#### Step-Based vs Time-Based Checkpointing in `train_ppo.py`
In `train_ppo.py` (lines 836–843):
```python
should_save = False
if checkpoint_interval_sec > 0:
    if elapsed_since_save >= checkpoint_interval_sec or current_step >= num_updates:
        should_save = True
else:
    if (current_step % config.checkpoint_interval == 0) or (current_step >= num_updates):
        should_save = True
```
- **Analysis**:
  - If `checkpoint_interval_seconds` is left at default `600.0`, checkpoints trigger strictly on 10-minute time intervals.
  - Setting `--checkpoint_interval_seconds 0.0` switches `train_ppo.py` to step-based mode: `current_step % config.checkpoint_interval == 0`.
  - With `--chunk_size 200` and `--checkpoint_interval 1200`, the JIT scan chunk (200 updates) divides 1,200 evenly. At `current_step = 1200, 2400, 3600, 4800, ...`, `should_save` evaluates to `True`.

#### Checkpoint Serialization & Streaming Protocol
1. **Orbax Persistence on VM**:
   - `save_checkpoint_orbax()` writes model weights and full `_resume_state` (TrainState, optimizer state, PRNG key, EnvState) into `/content/workspace/checkpoints/mode_1/step_{step}`.
2. **Push Trigger**:
   - `train_ppo.py` executes `--on_checkpoint_cmd`:
     `python3 /content/push_checkpoint.py step_{step} {step_dir} {TUNNEL_URL}`
3. **HTTP Streaming via Quick Tunnel**:
   - `push_checkpoint.py` archives `step_dir` into `/tmp/step_{step}.tar`, computes SHA256, and sends `POST {TUNNEL_URL}/upload_checkpoint?step=step_{step}&sha256={hash}`.
4. **Host Receipt & Verification**:
   - `JaxTunnelTransferManager` receives the stream, verifies `Content-Length` and SHA256 hash.
   - Extracts into host `checkpoints/` directory.
   - Adds the step number to `verified_steps`.
   - Returns `HTTP 200 CHECKPOINT_OK`.
   - Only 100% verified checkpoints are allowed to be resume candidates.

---

## 4. Empirical Baseline Metrics (Gate 1 & Gate 2 Checkpoints)

To ground our survey in empirical fact, we executed greedy evaluations of existing checkpoints `step_4800` and `step_26400` using `eval_checkpoint.py` with 10 episodes and 1,200 max steps.

### Baseline Evaluation Comparison Table

| Metric | Target Specification | Legacy `step_4800` (Measured) | Legacy `step_26400` (Measured) | Baseline Status |
| :--- | :--- | :--- | :--- | :--- |
| **`JUMP_RIGHT` Ratio** | **< 20.0%** (Gate 1) | **70.1%** (4,616 / 6,584) | **61.8%** (4,841 / 7,839) | ❌ **Severe Violation** (Dominates policy) |
| **Grounded Actions** (`LEFT`, `RIGHT`, `NOOP`) | **> 50.0%** (Gate 1) | **16.2%** (1,067 / 6,584) | **15.2%** (1,195 / 7,839) | ❌ **Severe Violation** (Bunny-hop spam) |
| **Total Jump Mobility** | Unconstrained / Minimized | **83.0%** | **77.2%** | ❌ Exploitation of zero-cost air actions |
| **Falling Debris Hits** | **$\le$ 3.5 hits/ep** (Gate 2) | **8.7 hits/ep** (87 hits / 10 ep) | **9.1 hits/ep** (91 hits / 10 ep) | ❌ **Severe Violation** (> 2.5x threshold) |
| **Average Survival Steps** | **> 1,200 steps** (Gate 2) | **658.4 steps** (10.9s) | **783.9 steps** (13.0s) | ❌ **Trapped in Plateau** (< 11~13s) |
| **Full Survival Rate** | 100.0% across 1,200s | 0.0% cleared by timeout | 0.0% cleared by timeout | ❌ All episodes terminate early |
| **Boss Shield Damage** | **$\ge$ 140 / 200** (Gate 2) | **120.0 / 200** | **160.0 / 200** | ⚠️ Marginally passes at step 26400 only |
| **Laser Redirections (Clean)** | High / Truthful | 0 (0.0/ep) | 0 (0.0/ep) | ❌ 100% dirty baiting (player self-damage) |

### Key Diagnostic Insights
1. **The 658-step / 11-second Death Trap**:
   In `step_4800`, the average survival is 658.4 steps, perfectly matching the user's reported "680-step / 11-second plateau". The agent survives initial easy patterns by bunny-hopping rightward, but when dense falling debris clusters collide with the central rotating laser, the agent lacks grounded deceleration control and is instantly annihilated.
2. **Jump-Right Artifact**:
   `JUMP_RIGHT` constitutes **70.1%** of all actions in `step_4800`. Grounded micro-movement (`NOOP`, `LEFT`, `RIGHT`) accounts for a combined **16.2%**, with `NOOP` at an astonishing 0.1%!
3. **Debris Vulnerability**:
   Debris hits average 8.7 to 9.1 hits per episode. Because jumping into overhead falling hazards was previously unpenalized, the policy continually leaps directly into falling blocks.
4. **Why Clean Slate is Required**:
   Even after 26,400 updates, the unregularized policy remains trapped in the same local minimum (61.8% `JUMP_RIGHT`, 9.1 debris hits/ep, 783 survival steps). Attempting to fine-tune from these legacy checkpoints inherits severe policy inertia. Clean-slate initialization with R1 & R2 regularizations is essential.

---

## 5. Evaluation Harness for Gate 1 & Gate 2 Acceptance Criteria

### 5.1 Acceptance Criteria Definitions

#### Gate 1: Behavioral Distribution Shift (Early Checkpoints `step_4800`+)
- **Condition 1**: `JUMP_RIGHT` action ratio drops from 53.7% / 70.1% to **under 20.0%** (`jump_right_pct < 20.0`).
- **Condition 2**: Grounded action ratio (`LEFT`, `RIGHT`, `NOOP`) exceeds **50.0%** across rollout steps (`grounded_pct > 50.0`).

#### Gate 2: Debris Evasion & Survival Breakthrough
- **Condition 1**: Debris hit count drops from 7.7~8.7 hits/ep to **$\le$ 3.5 hits/ep** (`avg_debris_hits <= 3.5`).
- **Condition 2**: Average episode survival steps exceed **1,200 steps** (20.0s), shattering the 680-step plateau (`avg_survival_steps >= 1200.0`).
- **Condition 3**: Laser baiting and boss shield destruction remain high ($\ge$ **140 / 200 shield damage**, `avg_shield_dmg >= 140.0`).

### 5.2 Analysis of Existing `eval_checkpoint.py`

#### Strengths:
- Accurately builds deterministic greedy argmax policy on top of JIT-compiled `RolloutRunner`.
- Correctly parses and counts action bincounts and hazard telemetry (`debris_hit`, `boss_hit_event`, `tracking_laser_hit_player`, `tracking_laser_hit_boss`).
- Produces human-readable per-episode and summary tables.

#### Deficiencies / Required Enhancements:
1. **No Programmatic Return Codes**: Returns `None`; cannot fail a build/test pipeline.
2. **No Automated Threshold Checks**: Does not calculate or evaluate Gate 1 or Gate 2 conditions against target thresholds.
3. **Single Checkpoint Only**: Requires manual invocation per checkpoint; cannot sweep `checkpoints/` to find the earliest passing checkpoint or evaluate all synced checkpoints.
4. **No Machine-Readable Output**: Does not dump JSON telemetry results for automated verification by other test suites.

### 5.3 Proposed Automated Gate Evaluation Script Design (`eval_gates.py`)

To fully automate Gate 1 and Gate 2 verification, we outline the exact architecture for an automated harness script, `eval_gates.py` (to be implemented in Milestone 4 or by test writers):

```python
"""Automated Gate 1 & Gate 2 Acceptance Verification for Maple-Gymnax Checkpoints."""

import argparse
import json
import os
import sys
import numpy as np

from eval_checkpoint import evaluate_checkpoint  # or refactored core evaluation logic


def verify_checkpoint_gates(
    checkpoint_path: str,
    episodes: int = 10,
    eval_steps: int = 1200,
    enforce_gate1: bool = True,
    enforce_gate2: bool = True,
) -> dict:
    """Evaluates checkpoint and performs strict Gate 1 & Gate 2 threshold assertions."""
    # 1. Run evaluation
    results, summary = run_greedy_evaluation(checkpoint_path, episodes, eval_steps)

    # 2. Extract Gate Metrics
    jump_right_pct = summary["act_pcts"][6]
    grounded_pct = summary["act_pcts"][0] + summary["act_pcts"][1] + summary["act_pcts"][2]
    avg_debris_hits = summary["avg_debris_hits"]
    avg_survival_steps = summary["avg_steps"]
    avg_shield_dmg = summary["avg_shield_dmg"]

    # 3. Gate 1 Checks
    gate1_jump_right_pass = jump_right_pct < 20.0
    gate1_grounded_pass = grounded_pct > 50.0
    gate1_pass = gate1_jump_right_pass and gate1_grounded_pass

    # 4. Gate 2 Checks
    gate2_debris_pass = avg_debris_hits <= 3.5
    gate2_survival_pass = avg_survival_steps >= 1200.0
    gate2_shield_pass = avg_shield_dmg >= 140.0
    gate2_pass = gate2_debris_pass and gate2_survival_pass and gate2_shield_pass

    verdict = {
        "checkpoint": checkpoint_path,
        "metrics": {
            "jump_right_pct": jump_right_pct,
            "grounded_pct": grounded_pct,
            "avg_debris_hits": avg_debris_hits,
            "avg_survival_steps": avg_survival_steps,
            "avg_shield_dmg": avg_shield_dmg,
        },
        "gate1": {
            "passed": gate1_pass,
            "jump_right_pass": gate1_jump_right_pass,
            "grounded_pass": gate1_grounded_pass,
        },
        "gate2": {
            "passed": gate2_pass,
            "debris_pass": gate2_debris_pass,
            "survival_pass": gate2_survival_pass,
            "shield_pass": gate2_shield_pass,
        },
        "all_passed": (not enforce_gate1 or gate1_pass) and (not enforce_gate2 or gate2_pass),
    }
    return verdict
```

#### CLI Interface Specifications for `eval_gates.py`:
- `python eval_gates.py --checkpoint checkpoints/step_4800 --gate 1` (validates Gate 1 on early checkpoint)
- `python eval_gates.py --checkpoint checkpoints/step_35000 --gate 1,2` (validates both gates on mature checkpoint)
- `python eval_gates.py --scan_dir checkpoints/ --export_json gate_results.json` (sweeps all synced checkpoints and outputs progression table)
- Exit codes: `0` on gate pass, `1` on gate failure.

---

## 6. Actionable Recommendations for Implementation & Verification

### Recommendation 1: Align `train_ppo.py` Checkpoint Interval Trigger
Currently, `train_ppo.py` uses `if checkpoint_interval_sec > 0: ... else: (current_step % config.checkpoint_interval == 0)`.
We recommend modifying line 837 in `train_ppo.py` during Milestone 3 (or via worker edit) to allow simultaneous step and time triggers:
```python
should_save = (
    (checkpoint_interval_sec > 0 and elapsed_since_save >= checkpoint_interval_sec)
    or (config.checkpoint_interval > 0 and current_step % config.checkpoint_interval == 0)
    or (current_step >= num_updates)
)
```
This guarantees that checkpoints are committed every 1,200 updates regardless of whether `checkpoint_interval_seconds` is passed. In the interim, passing `--checkpoint_interval_seconds 0.0` achieves the exact 1,200 update cadence.

### Recommendation 2: Cloud Burst Launch Runbook
When Milestones 1, 2, and 3 are completed and verified:
1. Ensure `cloud/colab_accounts.json` has `account_1` active.
2. Sweep any dangling remote sessions: `cloud\colab.bat stop -s jax_train`.
3. Launch training with:
   ```bash
   python run_colab_train.py --accelerator gpu --gpu_type T4 --mode 1 --num_envs 16384 --num_steps 64 --num_updates 35000 --checkpoint_interval 1200 --checkpoint_interval_seconds 0.0 --chunk_size 200 --dtype float16 --no_resume
   ```
4. Monitor terminal stdout for `[colab-jax] [Update ...]` logs and check `checkpoints/` as `step_1200`, `step_2400`, `step_3600`, `step_4800` stream in.

### Recommendation 3: Milestone Gate Verification Workflow
1. When `step_4800` is synced:
   Run Gate 1 verification:
   ```bash
   uv run python eval_checkpoint.py checkpoints/step_4800 --episodes 10 --eval_steps 1200
   ```
   Confirm `JUMP_RIGHT < 20%` and `Grounded > 50%`.
2. When training finishes or reaches mature steps ($\ge 12000$ to $35000$):
   Run Gate 2 verification:
   Confirm `Debris hits <= 3.5`, `Survival steps >= 1200`, `Shield damage >= 140 / 200`.

---
*Report generated and self-verified by explorer_survey_3.*
