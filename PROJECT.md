# Project: Autonomous Micro-Movement & Threat-Gated Evasion in Maple Gymnax Lotus Phase 1

## Architecture
- **Environment Core (`src/maple_gymnax/envs/`)**:
  - `lotus_phase1.py`: `Flax.struct.dataclass` immutable parameters (`EnvParams`) and dynamic tick state (`EnvState`). Vectorized, branch-free JAX transition step (`step_env`) and reset (`reset_env`).
  - `common.py`: Action constants (`ACTION_NOOP`, `ACTION_LEFT`, `ACTION_RIGHT`, `ACTION_JUMP`, `ACTION_JUMP_LEFT`, `ACTION_JUMP_RIGHT`, `ACTION_DUCK`/`ACTION_DOWN`), kinematics decoder (`decode_action`).
- **Reward Engine (`_compute_reward`)**:
  - Pure functional reward calculation.
  - Action Effort: Jump actions (`ACTION_JUMP`, `ACTION_JUMP_LEFT`, `ACTION_JUMP_RIGHT`) penalized by `-0.05`; ground actions (`NOOP`, `LEFT`, `RIGHT`, `DUCK`/`DOWN`) zero-cost (`0.0`).
  - Switching Jitter Regularization: `-0.02 * (action != last_action)` to discourage 1-frame chatter.
  - Threat-Gated Hazard Corridor: Under high-threat debris ($r \ge 24\text{px}$, $|\Delta x| < 45\text{px}$, $\Delta y \in [0, 180\text{px}]$), impose anti-jump penalty `r_airborne_hazard = -0.35` if airborne (`~player_on_ground`); award grounded clearance bonus `r_tap_dodge = 0.25` if moving horizontally away from debris center ($dx_{\text{next}} > dx_{\text{prev}}$).
  - Continuous Overhead Gaussian Potential: Scale $\phi_{\text{debris}}$ weight from `-0.05` to `-0.30`.
- **Training Telemetry & PPO Runner (`train_ppo.py`)**:
  - PureJaxRL vectorized PPO runner over $16,384$ parallel environments.
  - Real-time `Survival(s)` (average survival seconds = length / 60.0), replacing binary 60s survival.
  - Live `DebrisHits/ep` (renewal estimator: `total_hits / max(num_dones, 1.0)`) and `JumpRatio%` (frequency of jump actions in rollout batch) logged every 20 updates.
- **Cloud Burst Training Infrastructure (`cloud/`, `run_colab_train.py`)**:
  - Colab T4 GPU pool (`account_1`), 16,384 parallel environments, 35,000 updates, SPS ~980,000.
  - Clean-slate launch (`--no_resume`), intermediate checkpoint sync every 1,200 updates (`--checkpoint_interval 1200 --checkpoint_interval_seconds 0.0 --chunk_size 200`).
  - Automated evaluation harness (`eval_checkpoint.py` / `eval_gates.py`) validating Gate 1 (`JUMP_RIGHT < 20%`, `Grounded > 50%` at `step_4800`+) and Gate 2 (`DebrisHits <= 3.5`, `Survival >= 1200` steps / 20.0s, `ShieldDamage >= 140/200`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Action Cost & Energy Regularization (R1) | Jump effort cost (-0.05), ground zero cost, state.last_action tracking, jitter penalty (-0.02) | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Tap-Dodging Hazard Corridor & Overhead Repulsion (R2) | High-threat debris cone ($r \ge 24$, $|\Delta x| < 45$, $\Delta y \in [0, 180]$), anti-jump (-0.35), tap-dodge (+0.25), Gaussian potential (-0.30) | M2 | ORIGINAL_REQUEST §R2 |
| 3 | Telemetry & Metric Overhaul in PPO (R3) | Survival(s), DebrisHits/ep, JumpRatio% logged every 20 updates | M3 | ORIGINAL_REQUEST §R3 |
| 4 | Verification Test Suite & Regression Guard | Unit tests, analytical reward assertions, pytest configuration | M4 | E2E Testing Track |
| 5 | Colab T4 Cloud Burst Training & Gate 1/2 Verification (R4) | Launch clean-slate 35k update training, sync checkpoints every 1200 updates, verify Gate 1 & Gate 2 | M5 | ORIGINAL_REQUEST §R4, Gate 1 & 2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Action Cost & Energy Regularization Engine | Implement `last_action` in `EnvState`, reward hyperparameters in `EnvParams`, jump cost and jitter penalties in `_compute_reward` | None | PLANNED |
| M2 | Tap-Dodging Hazard Corridor & Overhead Repulsion | Implement high-threat overhead detection, airborne hazard penalty (-0.35), grounded tap-dodge bonus (+0.25), scale Gaussian potential to -0.30 | M1 | PLANNED |
| M3 | Transparent Telemetry & Metric Overhaul | Update `train_ppo.py` with Survival(s), DebrisHits/ep, JumpRatio%, --log_interval 20, watchdog compatibility | None | PLANNED |
| M4 | Comprehensive E2E Verification & Test Suite | Write `tests/test_micro_movement_r1_r2_r3.py`, fix `pyproject.toml` pythonpath, verify all unit & regression tests pass, publish `TEST_READY.md` | M1, M2, M3 | PLANNED |
| M5 | Colab T4 Cloud Burst Training & Gate 1 & 2 Acceptance | Launch clean-slate burst training on `account_1` with `--no_resume`, sync checkpoints every 1,200 updates, run automated Gate 1 & Gate 2 evaluations | M4 | PLANNED |

## Interface Contracts
### `src/maple_gymnax/envs/lotus_phase1.py` ↔ `train_ppo.py`
- `EnvState.last_action: int = 0`: Tracked across steps (`step_env` sets `state.replace(last_action=action)`).
- `_compute_reward()`: Pure functional JAX signature accepting `action`, `last_action`, `on_ground_next`, `px_next`, `px_prev`, `debris_x`, `debris_y`, `debris_radius`, `debris_active`.
- `traj_batch.info["debris_hit"]`: Boolean / float array indicating debris collision per environment step.

### `train_ppo.py` ↔ Checkpoint & Cloud Pipeline
- `train_ppo.py`: Saves Orbax checkpoints at `checkpoints/step_{step}` every 1,200 updates when `--checkpoint_interval 1200 --checkpoint_interval_seconds 0.0`.
- `eval_checkpoint.py`: Evaluates greedy policy on saved checkpoints and outputs empirical metrics (Survival steps, DebrisHits/ep, JumpRatio%, ShieldDamage).

## Code Layout
- `src/maple_gymnax/envs/common.py`: Action constants and kinematic decoders.
- `src/maple_gymnax/envs/lotus_phase1.py`: Core simulator, `EnvState`, `EnvParams`, `_compute_reward`.
- `train_ppo.py`: PPO training loop, vectorized environment rollouts, metrics logging callback.
- `tests/test_lotus_phase1.py`: Core environment and physics unit tests.
- `tests/test_micro_movement_r1_r2_r3.py`: New comprehensive test suite verifying R1, R2, R3 analytical properties.
- `cloud/`: Cloudflare tunnel bridge, Colab job runner, account manager.
- `run_colab_train.py`: Local launcher for Colab training bursts.
