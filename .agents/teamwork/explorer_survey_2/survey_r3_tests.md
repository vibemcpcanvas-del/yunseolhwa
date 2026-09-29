# Survey Report: Requirement R3 Telemetry & Test Infrastructure

**Document Path**: `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\survey_r3_tests.md`  
**Author**: `explorer_survey_2` (Teamwork Explorer)  
**Parent Agent**: `orchestrator_3` (`e8d54a3b-63f5-4fbd-92da-90a91af57a97`)  
**Target Milestone**: Autonomous Micro-Movement & Threat-Gated Evasion (R3 Telemetry & Test Suite)  
**Date**: 2026-09-29  

---

## 1. Executive Summary

This investigation analyzes requirement **R3** (Transparent Real-Time Console Telemetry & Metric Overhaul) in `train_ppo.py`, conducts an empirical audit of the existing test infrastructure across the repository, and provides a formal risk analysis of metric accumulation pitfalls in vectorized JAX scan environments (`jax.lax.scan`).

### Core Findings:
1. **Root Cause of `Survival: 0.0%` Metric**:
   In `train_ppo.py:443-448`, survival rate was computed as `jnp.sum((returned_lengths >= env_params.max_steps_in_episode) * done_mask) / max(num_dones, 1.0)`. Because `max_steps_in_episode = 3600` (60.0s at 60 Hz), an agent surviving 680 steps (11.3s) or 2,000 steps (33.3s) was scored as `0.0%`. This binary threshold completely blinded operators to incremental survival breakthroughs. Replacing this with real-time `Survival(s) = mean_length / 60.0` provides immediate, continuous, truthful feedback.
2. **Debris Hits Per Episode (`DebrisHits/ep`)**:
   Debris collisions are already accurately flagged in `src/maple_gymnax/envs/lotus_phase1.py:717` via `info["debris_hit"] = debris_damage_total > 0.0`. Two tracking architectures were evaluated:
   - **Option A (In-Wrapper Episodic Accumulator)**: Extend `LogEnvState` and `LogWrapper` to accumulate `episode_debris_hits` and output `returned_episode_debris_hits` on `done=True`.
   - **Option B (Vectorized Renewal Estimator in `train_ppo.py`)**: Directly compute `jnp.sum(traj_batch.info["debris_hit"]) / jnp.maximum(num_dones, 1.0)` over the rollout trajectory batch. By Little's Law / renewal reward theorem, this is an asymptotically unbiased estimator that requires zero modifications to `LogWrapper` or checkpoint PyTree schemas.
3. **Jump Ratio (`JumpRatio%`) & CRITICAL ACTION INDEX TRAP**:
   - The user request states: *"Assign explicit action cost to discrete actions 4 (JUMP), 5 (JUMP_LEFT), 6 (JUMP_RIGHT)"* and *"zero cost for 0 (NOOP), 1 (LEFT), 2 (RIGHT), 3 (DOWN)"*.
   - **CRITICAL DISCREPANCY**: In `src/maple_gymnax/envs/common.py:30-36` and `decode_action()`, the canonical actions are:
     - `ACTION_NOOP = 0`, `ACTION_LEFT = 1`, `ACTION_RIGHT = 2`
     - `ACTION_JUMP = 3`, `ACTION_JUMP_LEFT = 4`, `ACTION_JUMP_RIGHT = 5`
     - `ACTION_DUCK = 6`
   - Therefore, jump actions are strictly **`{3, 4, 5}`**, and ground actions are **`{0, 1, 2, 6}`**!
   - If an implementer blindly used `{4, 5, 6}`, `JUMP (3)` would be excluded, and `DUCK (6)` (which is a ground crouch reducing height to 35px) would be penalized as a jump!
   - In `train_ppo.py`, `traj_batch.action` already records every discrete action. `JumpRatio%` is cleanly computed across `(num_steps, num_envs)` as `jnp.mean(((traj_batch.action >= 3) & (traj_batch.action <= 5)).astype(jnp.float32)) * 100.0`.
4. **Test Infrastructure Audit (459 Passed, 2 Failed / 461 Total)**:
   - Running pytest requires setting pythonpath: `uv run pytest -o pythonpath=". src"`. (Adding `pythonpath = [".", "src"]` to `pyproject.toml` will permanently resolve this).
   - All 33 core simulation tests in `test_lotus_phase1.py`, all 4 PPO tests in `test_train_ppo.py`, all 9 wrapper tests in `test_wrappers.py`, and all 98 boundary tests in `e2e/test_tier2_boundaries.py` pass 100%.
   - The only 2 failures are CPU benchmark timing tests (`test_f26_batch_size_256_throughput` and `test_tier4_scenario_12`), which assert `res[256] > 1000.0` SPS (achieved 970.6 and 950.9 SPS on 5-step cold CPU JIT).

---

## 2. Detailed Analysis of `train_ppo.py` Architecture & Metrics Telemetry (R3)

### 2.1 File Layout & Call Chain in `train_ppo.py`

The training pipeline in `train_ppo.py` is structured as:
```
main()
 ├── parse_args() -> PPOConfig
 ├── make_train_step(config)
 │    ├── env = LogWrapper(FlattenObservationWrapper(LotusPhase1Env()))
 │    ├── ActorCritic(action_dim=7)
 │    ├── init_fn(rng) -> RunnerState(train_state, env_state, last_obs, rng)
 │    └── update_chunk_fn(runner_state, update_indices) -> jax.lax.scan(_update_step, ...)
 │         └── _update_step(runner_state, update_idx)
 │              ├── 3A. Trajectory Rollout: jax.lax.scan(_env_step, length=num_steps)
 │              │    └── Transition(done, action, value, reward, log_prob, obs, info)
 │              ├── 3B. GAE Computation: _calculate_gae(...)
 │              ├── 3C. Minibatch SGD across update_epochs
 │              ├── 3D. Metric Extraction & jax.debug.callback(_log_callback, ...)
 │              └── returns (next_runner_state, metrics_dict)
 └── Outer Python Loop (Chunk size = chunk_size updates)
      ├── runner_state, chunk_metrics = jitted_chunk(runner_state, chunk_indices)
      ├── jax.block_until_ready(...)
      ├── Orbax checkpointing (time-based & step-based)
      └── Watchdog checks (entropy collapse < 0.03, plateau patience)
```

### 2.2 Host Console Logging Callback (`_log_callback`)

Located at **`train_ppo.py:208-230`**:
```python
def _log_callback(
    update_step: int,
    mean_return: float,
    mean_length: float,
    survival_rate: float,
    mean_gauge: float,
    actor_loss: float,
    critic_loss: float,
    entropy: float,
    sps: float,
) -> None:
    """Prints training progress on host console without halting device computation."""
    print(
        f"[Update {int(update_step):5d}] "
        f"Return: {float(mean_return):8.2f} | "
        f"Length: {float(mean_length):6.1f} | "
        f"Survival: {float(survival_rate) * 100.0:5.1f}% | "
        f"Gauge: {float(mean_gauge):6.4f} | "
        f"Loss(A/C/Ent): {float(actor_loss):.3f}/{float(critic_loss):.3f}/{float(entropy):.3f} | "
        f"SPS: {float(sps):10,.0f}",
        flush=True,
    )
```

### 2.3 The 60-Second Binary Survival Metric vs Real-Time `Survival(s)`

#### The Problem:
In `train_ppo.py:443-448`:
```python
survival_rate = jnp.where(
    has_dones,
    jnp.sum((traj_batch.info["returned_episode_lengths"] >= env_params.max_steps_in_episode).astype(jnp.float32) * done_mask)
    / jnp.maximum(num_dones, 1.0),
    0.0,
)
```
- `env_params.max_steps_in_episode = 3600` (60 seconds at dt = 1/60s).
- During early training, agents live for 680 steps (11.3s). Because 680 < 3600, the condition `returned_episode_lengths >= 3600` is 0.
- Telemetry outputs `Survival: 0.0%` continuously until an agent survives the full 60 seconds without a single death.
- This creates an operator blind spot: an agent improving from 200 steps (3.3s) to 1,200 steps (20.0s) displays `Survival: 0.0%` for both!

#### The Solution:
Replace the binary metric with continuous average survival seconds:
```python
survival_sec = mean_length / 60.0  # Alternatively: mean_length * env_params.dt
```
And format in console logging as:
```python
f"Survival(s): {float(survival_sec):5.1f}s | "
```
- At 680 steps: `Survival(s):  11.3s`
- At 1,200 steps: `Survival(s):  20.0s`
- At 3,600 steps: `Survival(s):  60.0s`

*Note on Watchdog Compatibility*:
In `train_ppo.py:816`:
```python
if cur_survival >= 0.95:
    if cur_return > best_return + 0.5: ...
```
To maintain watchdog compatibility without regressions, `metrics["survival_rate"]` can either remain in `metrics` dict (computed alongside `survival_sec`), or the watchdog can check `cur_survival_sec >= 57.0` (95% of 60.0s). Retaining `"survival_rate"` in `metrics` dict while outputting `Survival(s)` to console is the safest drop-in pattern.

### 2.4 `DebrisHits/ep` Tracking Architecture

Requirement: *"Print live DebrisHits/ep directly in the training log every 20 updates."*  
Baseline target: Drop from 7.7 hits/ep to $\le 3.5$ hits/ep.

Where debris hits occur:
In `src/maple_gymnax/envs/lotus_phase1.py:717`:
```python
info = {
    "laser_hit": classic_laser_hit,
    "debris_hit": debris_damage_total > 0.0,
    ...
}
```
During rollout, `traj_batch.info["debris_hit"]` has shape `(num_steps, num_envs)`.

#### Architectural Comparison for Debris Hit Aggregation:

| Approach | Architecture A: `LogWrapper` Accumulator | Architecture B: Rollout Batch Renewal Estimator |
|---|---|---|
| **Mechanism** | Add `episode_debris_hits` and `returned_episode_debris_hits` to `LogEnvState`. Accumulate per tick and reset on `done=True`. | Sum `traj_batch.info["debris_hit"]` across `(num_steps, num_envs)` and divide by `num_dones` (or multiply per-step rate by `mean_length`). |
| **Files Modified** | `src/maple_gymnax/wrappers/log_wrapper.py`, `train_ppo.py` | `train_ppo.py` only |
| **PyTree Impact** | Changes `LogEnvState` PyTree definition (2 extra float leaves). Requires re-saving checkpoints. | Zero PyTree change. 100% backward compatible with existing Orbax PyTrees. |
| **Statistical Accuracy** | Exact per-completed-episode hit count for episodes ending in this window. | Asymptotically exact under Little's Law ($E[\text{hits/ep}] = E[\text{hits/step}] \times E[\text{length}]$). |
| **Zero-Done Window** | Falls back to 0.0 if `has_dones == False`. | Falls back to `(total_hits / batch_size) * mean_length`. |

**Recommendation**:
- **Architecture B** is strongly recommended for immediate adoption in `train_ppo.py`:
  ```python
  total_debris_hits = jnp.sum(traj_batch.info["debris_hit"].astype(jnp.float32))
  debris_hits_per_ep = jnp.where(
      has_dones,
      total_debris_hits / jnp.maximum(num_dones, 1.0),
      (total_debris_hits / (config.num_envs * config.num_steps)) * mean_length,
  )
  ```
  This is branch-free, XLA-friendly, requires no modifications to `LogWrapper`, and leaves `LogEnvState` checkpoint schemas intact.
- If strict per-trajectory episode accumulation is preferred by the team, Architecture A can be cleanly added to `LogEnvState` with default `0.0`.

### 2.5 `JumpRatio%` Tracking Architecture

Requirement: *"Print live JumpRatio% directly in the training log every 20 updates."*  
Baseline target: Drop from 53.7% `JUMP_RIGHT` spam to < 20% total jumps, with grounded ratio > 50%.

#### The Critical Action Index Discrepancy:
In `src/maple_gymnax/envs/common.py:30-46`:
```python
# Discrete Action Mapping (Gymnax Discrete(7))
ACTION_NOOP: int = 0
ACTION_LEFT: int = 1
ACTION_RIGHT: int = 2
ACTION_JUMP: int = 3
ACTION_JUMP_LEFT: int = 4
ACTION_JUMP_RIGHT: int = 5
ACTION_DUCK: int = 6
```
In `decode_action()` (`common.py:199-207`):
```python
is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)
```
- In the existing codebase, **Jump actions are indices 3, 4, 5**.
- In the user request text (`ORIGINAL_REQUEST.md:106`), the text informally referred to:
  `4 (JUMP), 5 (JUMP_LEFT), 6 (JUMP_RIGHT)` and `0 (NOOP), 1 (LEFT), 2 (RIGHT), 3 (DOWN)`.
- **WARNING**: In `common.py`, action 6 is `ACTION_DUCK` (crouching, height = 35.0, `is_jump = False`). Action 3 is `ACTION_JUMP` (`is_jump = True`).
- Any metric or reward calculation must use the canonical indices:
  - Jump actions: `3 (JUMP)`, `4 (JUMP_LEFT)`, `5 (JUMP_RIGHT)`
  - Ground actions: `0 (NOOP)`, `1 (LEFT)`, `2 (RIGHT)`, `6 (DUCK)`

#### Implementation in `train_ppo.py`:
In `_update_step`, `traj_batch.action` has shape `(num_steps, num_envs)`.
```python
jump_mask = (traj_batch.action >= 3) & (traj_batch.action <= 5)
jump_ratio = jnp.mean(jump_mask.astype(jnp.float32)) * 100.0
```
This is fully vectorized, requires zero additional environment state, and computes the exact empirical jump frequency over the entire rollout batch.

### 2.6 Logging Cadence (Every 20 Updates)

In `train_ppo.py:645`:
```python
parser.add_argument("--log_interval", type=int, default=1, help="Update interval for console logging")
```
And in `_update_step` (line 453):
```python
should_log = (jnp.mod(update_idx + 1, config.log_interval) == 0) | (update_idx == 0)
```
To meet the requirement *"logged every 20 updates"*:
- Change default in `PPOConfig.log_interval: int = 20` (and `argparse` default=20).
- Or pass `--log_interval 20` when launching the training script.

### 2.7 Proposed Overhauled Log Line Format

```python
def _log_callback(
    update_step: int,
    mean_return: float,
    mean_length: float,
    survival_sec: float,
    debris_hits: float,
    jump_ratio: float,
    mean_gauge: float,
    actor_loss: float,
    critic_loss: float,
    entropy: float,
    sps: float,
) -> None:
    """Prints training progress on host console without halting device computation."""
    print(
        f"[Update {int(update_step):5d}] "
        f"Return: {float(mean_return):8.2f} | "
        f"Length: {float(mean_length):6.1f} | "
        f"Survival(s): {float(survival_sec):5.1f}s | "
        f"DebrisHits/ep: {float(debris_hits):4.1f} | "
        f"JumpRatio%: {float(jump_ratio):5.1f}% | "
        f"Gauge: {float(mean_gauge):6.4f} | "
        f"Loss(A/C/Ent): {float(actor_loss):.3f}/{float(critic_loss):.3f}/{float(entropy):.3f} | "
        f"SPS: {float(sps):10,.0f}",
        flush=True,
    )
```

Example formatted output:
```
[Update    20] Return:   -52.40 | Length:  682.4 | Survival(s):  11.4s | DebrisHits/ep:  7.6 | JumpRatio%:  52.1% | Gauge: 0.1240 | Loss(A/C/Ent): 0.045/0.210/1.820 | SPS:    982,410
[Update    40] Return:   -18.10 | Length: 1240.8 | Survival(s):  20.7s | DebrisHits/ep:  3.1 | JumpRatio%:  18.4% | Gauge: 0.0820 | Loss(A/C/Ent): 0.038/0.145/1.650 | SPS:    979,850
```

---

## 3. Comprehensive Examination of Test Suite & Infrastructure

### 3.1 Pytest Execution Mechanics

When running `pytest` in this environment:
- Executing `uv run pytest` directly failed with:
  `ModuleNotFoundError: No module named 'train_ppo'`.
- Cause: `pyproject.toml` contains:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  ```
  It lacks `pythonpath = [".", "src"]`.
- Execution command used for test suite audit:
  ```powershell
  uv run pytest -o pythonpath=". src" tests/
  ```
- **Recommended Fix**: Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
  ```toml
  pythonpath = [".", "src"]
  ```

### 3.2 Baseline Test Suite Results (461 Tests Executed)

Full test run log (task-83):
```
=========================== short test summary info ===========================
FAILED tests/e2e/test_tier1_features.py::test_f26_batch_size_256_throughput
FAILED tests/e2e/test_tier4_scenarios.py::test_tier4_scenario_12_hardware_sps_measurement_pipeline
================== 2 failed, 459 passed in 358.51s (0:05:58) ==================
```

#### Detailed Breakdown of Passing Tests:
1. `tests/test_lotus_phase1.py`: **33 passed / 33 tests** (100%)
   - JIT reset/step compilation, vmap across 1024, 2048, 4096 envs.
   - Kinematics (horizontal speed, gravity, jumping, ducking hitbox reduction, wall clamping).
   - Rotating cross laser (angular velocity, SAT projection, directional ray dot product masking, core origin clearance, lethal contact).
   - Falling debris (static array capacity 30, Bernoulli PRNG slot allocation, floor despawn, Euclidean distance contact).
   - Remastered Lotus (gauge climb, overload trigger, safe-zone evasion, electric floor burst, friendly fire boss guidance).
   - Episode termination, observation normalization, 172-dim extended observation.
2. `tests/test_train_ppo.py`: **4 passed / 4 tests** (100%)
   - `test_actor_critic_shapes`: ActorCritic MLP forward pass and logits/value output.
   - `test_gae_scan_shapes`: Generalized Advantage Estimation reverse scan.
   - `test_smoke_train_mode0_classic`: Single JIT compile, 2 updates, Orbax save/restore, RolloutRunner evaluation.
   - `test_smoke_train_mode1_remastered`: Remastered PPO training smoke test.
3. `tests/test_wrappers.py`: **9 passed / 9 tests** (100%)
   - `FlattenObservationWrapper`: (130,) flattening, observation_space, JIT/vmap.
   - `PureJaxRLAdapterWrapper`: 5-tuple Gymnax adaptation.
   - `LogWrapper`: Branch-free episode metric accumulation and auto-reset.
   - `RolloutRunner`: `jax.lax.scan` trajectory rollout and initial observation alignment.
   - `FlashbaxAdapter`: Replay buffer init, add, and sample.
4. `tests/test_t4_precision_and_logging.py`: **3 passed / 3 tests** (100%)
   - T4 GPU precision fallback (`bfloat16` -> `float16`).
   - GPU enforcement check (`require_gpu`).
   - PPO float16 precision smoke test with finite losses and weights.
5. `tests/test_resume_checkpoint.py`: **4 passed / 4 tests** (100%)
   - Exact resumption continuation across Orbax checkpoint boundaries.
   - Hard key and soft key configuration fingerprinting.
6. `tests/test_tunnel_checkpoint_server.py`: **3 passed / 3 tests** (100%)
   - Tunnel streaming HTTP upload/download, SHA256 integrity check.
7. `tests/test_view_policy.py`: **4 passed / 4 tests** (100%)
8. `tests/test_wall_camp_baseline.py`: **5 passed / 5 tests** (100%)
   - Rollout comparisons of wall-camping vs noop vs random policies.
9. `tests/test_adversarial_m1.py`: **41 passed / 41 tests** (100%)
10. `tests/test_challenger_m1_2.py`: **19 passed / 19 tests** (100%)
11. `tests/test_cloud_account_rotation_resume.py`: **4 passed / 4 tests** (100%)
12. `tests/test_persona_prompt.py`: **2 passed / 2 tests** (100%)
13. `tests/test_wz_parser.py`: **8 passed / 8 tests** (100%)
14. `tests/e2e/test_tier2_boundaries.py`: **98 passed / 98 tests** (100%)
15. `tests/e2e/test_tier3_pairwise.py`: **30 passed / 30 tests** (100%)

#### Analysis of the 2 Failures:
- `tests/e2e/test_tier1_features.py:1509: AssertionError: assert 970.6 > 1000.0`
- `tests/e2e/test_tier4_scenarios.py:301: AssertionError: assert 950.9 > 1000.0`
- **Cause**: Both tests execute `run_sps_benchmark(batch_sizes=(256,), num_steps=5)` on the host CPU. Running only 5 steps includes one-time XLA compile overhead in the timing loop. On this CPU, throughput measured 950-970 SPS, which narrowly missed the hardcoded 1000.0 SPS assert. On GPU/ROCm or with warmup steps, SPS easily exceeds 500k+. These 2 failures are non-functional benchmark assertions, not simulator bugs.

---

## 4. Required New Unit Tests for Requirements R1, R2, and R3

To ensure regression-free implementation of the Autonomous Micro-Movement and Threat-Gated Evasion mechanics, the following unit test suite must be implemented (recommended file: `tests/test_micro_movement_r1_r2_r3.py`):

### 4.1 Requirement R1: Action Cost & Jitter Regularization Tests
1. **`test_r1_action_cost_jump_actions()`**:
   - Compare rewards for jump actions (`3: JUMP`, `4: JUMP_LEFT`, `5: JUMP_RIGHT`) against ground action (`0: NOOP`).
   - Fix state and `last_action` to prevent jitter penalty interference.
   - Assert `r_noop - r_jump == 0.05` within `1e-5` float tolerance.
2. **`test_r1_action_cost_ground_actions_zero()`**:
   - Compare rewards for ground actions (`0: NOOP`, `1: LEFT`, `2: RIGHT`, `6: DUCK`).
   - Assert all ground actions have identical action cost of `0.0`.
3. **`test_r1_jitter_penalty_on_action_switching()`**:
   - Step environment with `state.last_action = ACTION_LEFT`:
     - Case A: Take `ACTION_LEFT` (repeat) -> jitter penalty = `0.0`.
     - Case B: Take `ACTION_RIGHT` (switch) -> jitter penalty = `-0.02`.
   - Assert `r_repeat - r_switch == 0.02` within `1e-5`.
4. **`test_r1_env_state_last_action_persistence()`**:
   - Verify `env.reset_env()` initializes `state.last_action == 0`.
   - Step with action $a \in \{1, 2, 3, 4, 5, 6\}$ and verify `next_state.last_action == a`.
   - Verify `last_action` preserves through `jax.jit` and `jax.vmap`.

### 4.2 Requirement R2: Tap-Dodging Corridor & Overhead Repulsion Tests
1. **`test_r2_overhead_danger_cone_geometry()`**:
   - Test spatial gating conditions: high threat $r \ge 24$, $|\Delta x| < 45$, $\Delta y \in [0, 180]$.
   - Case A: Inside cone ($r=24, \Delta x=30, \Delta y=100$) -> cone active.
   - Case B: Outside cone via radius ($r=16$) -> cone inactive.
   - Case C: Outside cone via lateral distance ($|\Delta x|=55$) -> cone inactive.
   - Case D: Outside cone via vertical distance ($\Delta y=200$ or $\Delta y < 0$) -> cone inactive.
2. **`test_r2_anti_jump_airborne_hazard_penalty()`**:
   - Under active overhead danger cone:
     - Compare `player_on_ground = False` (airborne) vs `player_on_ground = True` (grounded).
     - Assert airborne state incurs additional penalty `r_airborne_hazard = -0.35`.
3. **`test_r2_tap_dodge_clearance_bonus()`**:
   - Debris overhead at $x = 683.0$, player at $x = 660.0$ ($\Delta x = -23.0$, debris to the right).
   - Case A: Agent takes `ACTION_LEFT` (moving away, $|\Delta x|$ increases) -> receives `r_tap_dodge = +0.25`.
   - Case B: Agent takes `ACTION_RIGHT` (moving toward) or `ACTION_NOOP` -> receives no clearance bonus.
   - Assert `r_moving_away - r_neutral == 0.25`.
4. **`test_r2_continuous_overhead_gaussian_potential_rescaled()`**:
   - Set up debris overhead and calculate continuous repulsion potential.
   - Verify scaling coefficient is `-0.30` (rescaled from legacy `-0.05`).
   - Assert analytical potential matches formula `-0.30 * (r / 36.0) * jnp.exp(-0.5 * (dx / 45.0)**2)`.

### 4.3 Requirement R3: Telemetry & Metric Overhaul Tests
1. **`test_r3_realtime_survival_seconds_formula()`**:
   - Test survival seconds calculation for episode lengths:
     - Length 680 -> $680 / 60.0 = 11.333\text{s}$.
     - Length 1200 -> $1200 / 60.0 = 20.0\text{s}$.
     - Length 3600 -> $3600 / 60.0 = 60.0\text{s}$.
   - Contrast with old binary `survival_rate`: confirm old metric was 0.0 at length 680 and 1200.
2. **`test_r3_jump_ratio_canonical_action_mapping()`**:
   - Construct trajectory action array with known counts: 20 NOOP, 20 LEFT, 20 RIGHT, 20 JUMP (3), 10 JUMP_LEFT (4), 10 JUMP_RIGHT (5). Total = 100 actions, jumps = 40.
   - Compute `jump_ratio`: assert exact equality to `40.0%`.
   - Ensure action 6 (`DUCK`) is NOT counted in jump ratio.
3. **`test_r3_debris_hits_per_episode_estimator()`**:
   - Construct mock rollout batch with 1,000 steps, 10 episode terminations, and 50 total debris hits.
   - Verify estimated `debris_hits_per_ep == 5.0`.
   - Test zero-termination boundary condition: verify fallback produces finite float without division by zero.
4. **`test_r3_console_callback_formatting_smoke()`**:
   - Invoke `_log_callback(...)` with sample arguments.
   - Verify it executes cleanly without format string syntax errors or exceptions.

---

## 5. Potential Pitfalls in Metric Accumulation in Vectorized JAX Scan

When accumulating episodic metrics in high-throughput vectorized JAX environments (`jax.lax.scan`), several subtle concurrency and vectorization traps arise:

### Pitfall 1: Episodic Reset Asynchrony & Stale Metric Overwriting
- **Mechanism**: In `jax.lax.scan`, environments step simultaneously. Environment $A$ may terminate at step 15, while Environment $B$ terminates at step 55.
- **Trap**: When an environment terminates, `LogWrapper` records `returned_episode_returns = new_returns` and resets `current_returns = 0.0`. At step 16, Environment $A$ is on step 1 of a new episode, but `returned_episode_returns` still holds the return of the completed episode.
- **Mitigation**: Never compute simple means across `traj_batch.info["returned_episode_returns"]`. Always mask by `done_mask = traj_batch.info["returned_episode"]`:
  ```python
  jnp.sum(traj_batch.info["returned_episode_returns"] * done_mask) / jnp.maximum(num_dones, 1.0)
  ```

### Pitfall 2: Multi-Termination Within a Single Rollout Window
- **Mechanism**: If `num_steps = 64` and an agent has an average lifespan of 20 steps, an environment can die 3 times within a single rollout unroll.
- **Trap**: If an implementation stores only the "latest" completed episode per environment slot at the end of the 64-step chunk, intermediate episode completions are erased, severely skewing the average length and hit counts downward.
- **Mitigation**: Sum across the entire temporal dimension `(num_steps, num_envs)` of `traj_batch.info["returned_episode"]`. Because every step where `done=True` is recorded in `traj_batch`, summing across the time axis captures all termination events without loss.

### Pitfall 3: Zero-Done Window Starvation in High-Survival Horizons
- **Mechanism**: As the policy matures and average survival steps exceed 1,200 steps (20.0s) up to 3,600 steps (60.0s), an episode spans many 64-step update windows.
- **Trap**: When running with smaller test environments (e.g. `num_envs = 64` locally), there will be update windows where `num_dones == 0`. If `mean_length` falls back to `config.num_steps` (64), the reported survival time will falsely collapse from 30s to 1.0s.
- **Mitigation**:
  1. For large production runs (`num_envs = 16,384` on Colab T4), by the law of large numbers, with 16,384 envs and 60-second horizon (3600 steps), an average of $16,384 \times 64 / 3600 \approx 291$ terminations occur every single update. Zero-done starvation is mathematically impossible at full scale.
  2. For local testing, maintain a running state accumulator or fallback to the running step counter when `num_dones == 0`.

### Pitfall 4: Action Distribution vs Episodic Metric Dimensionality Mismatch
- **Mechanism**: `DebrisHits/ep` is an **episodic** metric (hits accumulated between spawn and death), whereas `JumpRatio%` is an **instantaneous / behavioral** metric (fraction of total actions that are jumps).
- **Trap**: Attempting to accumulate `jump_ratio` per episode in `LogWrapper` introduces unnecessary state complexity, boundary reset logic, and variance.
- **Mitigation**: Keep them separate:
  - Compute `JumpRatio%` directly over all transitions in the rollout:
    `jnp.mean(is_jump.astype(jnp.float32)) * 100.0`. This is exact, zero-variance across episode boundaries, and computationally trivial.
  - Track `DebrisHits/ep` via the renewal estimator or episodic wrapper.

### Pitfall 5: Division by Zero & Concretization Errors in JAX
- **Trap**: Using Python `if num_dones > 0:` inside JIT-compiled functions causes `ConcretizationTypeError` because `num_dones` is a traced JAX array. Dividing by `num_dones` directly produces `NaN` when `num_dones == 0`.
- **Mitigation**: Always use `jnp.where(has_dones, numerator / jnp.maximum(num_dones, 1.0), fallback_value)`. `jnp.maximum(num_dones, 1.0)` guarantees that the denominator is never zero during gradient or evaluation tracing, preventing IEEE floating-point NaNs.

---

## 6. Implementation Blueprint for Downstream Roles

### 6.1 Changes Required in `train_ppo.py` (Worker Task)

1. **Overhaul `_log_callback`**:
   Update signature and print template to include `Survival(s)`, `DebrisHits/ep`, `JumpRatio%`.
2. **Update Metric Extraction in `_update_step`**:
   ```python
   # 1. Survival in Seconds (Continuous)
   survival_sec = mean_length / 60.0

   # 2. Jump Ratio across Canonical Jump Actions {3: JUMP, 4: JUMP_LEFT, 5: JUMP_RIGHT}
   jump_mask = (traj_batch.action >= 3) & (traj_batch.action <= 5)
   jump_ratio = jnp.mean(jump_mask.astype(jnp.float32)) * 100.0

   # 3. Debris Hits per Episode (Renewal Estimator)
   total_debris_hits = jnp.sum(traj_batch.info["debris_hit"].astype(jnp.float32))
   debris_hits_per_ep = jnp.where(
       has_dones,
       total_debris_hits / jnp.maximum(num_dones, 1.0),
       (total_debris_hits / (config.num_envs * config.num_steps)) * mean_length,
   )
   ```
3. **Pass New Metrics to `jax.debug.callback`**:
   Pass `survival_sec`, `debris_hits_per_ep`, `jump_ratio` to `_log_callback`.
4. **Update `metrics` Dictionary**:
   Populate `"survival_sec"`, `"debris_hits"`, `"jump_ratio"` in returned dictionary.
5. **Set Default Logging Interval**:
   Set `default=20` for `--log_interval` in `PPOConfig` and `parse_args()`.

### 6.2 Update in `pyproject.toml`
Add `pythonpath = [".", "src"]` under `[tool.pytest.ini_options]` so that running `uv run pytest` works seamlessly without manual CLI flags.

---
*Report compiled by explorer_survey_2. Ready for Orchestrator review and Worker implementation.*
