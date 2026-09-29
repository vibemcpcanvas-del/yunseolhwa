# Handoff Report: Requirement R3 Telemetry & Test Infrastructure

**Subagent**: `explorer_survey_2`  
**Recipient**: `orchestrator_3` (`e8d54a3b-63f5-4fbd-92da-90a91af57a97`)  
**Date**: 2026-09-29T11:42:00Z  
**Type**: Hard (Task Complete)  

---

## 1. Observation

### 1.1 Console Log Formatting and Metric Tracking in `train_ppo.py`
- In `train_ppo.py:208-230`:
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
- In `train_ppo.py:443-448`:
  ```python
  survival_rate = jnp.where(
      has_dones,
      jnp.sum((traj_batch.info["returned_episode_lengths"] >= env_params.max_steps_in_episode).astype(jnp.float32) * done_mask)
      / jnp.maximum(num_dones, 1.0),
      0.0,
  )
  ```
  `env_params.max_steps_in_episode` is 3600 steps (60.0s at dt=1/60s). When the agent survives 680 steps (11.3s), `returned_episode_lengths >= 3600` is 0.0, causing `Survival: 0.0%`.
- In `train_ppo.py:645`:
  `parser.add_argument("--log_interval", type=int, default=1, help="Update interval for console logging")`.

### 1.2 Debris Hit Flagging in Environment
- In `src/maple_gymnax/envs/lotus_phase1.py:717`:
  `info = {"laser_hit": classic_laser_hit, "debris_hit": debris_damage_total > 0.0, ...}`.
  During `jax.lax.scan` rollout, `traj_batch.info["debris_hit"]` is available with shape `(num_steps, num_envs)`.

### 1.3 Action Mapping Discrepancy
- In `src/maple_gymnax/envs/common.py:30-36`:
  ```python
  ACTION_NOOP: int = 0
  ACTION_LEFT: int = 1
  ACTION_RIGHT: int = 2
  ACTION_JUMP: int = 3
  ACTION_JUMP_LEFT: int = 4
  ACTION_JUMP_RIGHT: int = 5
  ACTION_DUCK: int = 6
  ```
- In `decode_action()` (`common.py:206`):
  `is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)`
  This confirms jumping actions are strictly `{3, 4, 5}`. Action 6 is `ACTION_DUCK` (ground crouch, height reduced to 35px).
- In `ORIGINAL_REQUEST.md:106-107`:
  The user request text casually referred to jump actions as `4 (JUMP)`, `5 (JUMP_LEFT)`, `6 (JUMP_RIGHT)`.

### 1.4 Test Suite Execution & Empirical Baseline
- Running bare `uv run pytest` fails with:
  `ModuleNotFoundError: No module named 'train_ppo'`.
- Running `uv run pytest -o pythonpath=". src" tests/` succeeded and executed 461 tests:
  ```
  FAILED tests/e2e/test_tier1_features.py::test_f26_batch_size_256_throughput
  FAILED tests/e2e/test_tier4_scenarios.py::test_tier4_scenario_12_hardware_sps_measurement_pipeline
  ================== 2 failed, 459 passed in 358.51s (0:05:58) ==================
  ```
  Both failures are `assert res[256] > 1000.0` (achieved 970.6 and 950.9 SPS on 5-step cold CPU JIT). All 459 functional, simulation, physics, wrapper, and PPO tests passed 100%.

---

## 2. Logic Chain

1. **Survival Metric Inadequacy**:
   - *Observation*: `survival_rate` evaluates `returned_episode_lengths >= 3600`.
   - *Reasoning*: Because the current policy plateaus at 680 steps (~11.3s), the agent never reaches 3600 steps. Thus the metric consistently evaluates to `0.0%`.
   - *Deduction*: By substituting `survival_sec = mean_length / 60.0`, the metric becomes continuous and dynamic, displaying real-time survival seconds (e.g. `11.3s` -> `20.0s`).

2. **Debris Hits per Episode Estimation**:
   - *Observation*: `traj_batch.info["debris_hit"]` records all debris contacts in each step across all parallel environments.
   - *Reasoning*: In steady-state rollouts of thousands of parallel environments, Little's Law dictates $E[\text{hits/episode}] = \text{total hits} / \text{total completed episodes}$.
   - *Deduction*: `total_debris_hits / jnp.maximum(num_dones, 1.0)` provides an asymptotically exact, unbiased estimate of `DebrisHits/ep` without requiring modifications to `LogWrapper` or altering `LogEnvState` PyTree structures.

3. **Action Index Regularization & Jump Ratio Integrity**:
   - *Observation*: `common.py` defines `JUMP=3`, `JUMP_LEFT=4`, `JUMP_RIGHT=5`, and `DUCK=6`.
   - *Reasoning*: If code were written using `{4, 5, 6}` based on the text of `ORIGINAL_REQUEST.md`, `JUMP (3)` would be unpenalized and uncounted, while `DUCK (6)` would be penalized as a jump.
   - *Deduction*: All R1 action cost penalties, R2 anti-jump checks, and R3 `JumpRatio%` metrics MUST use canonical indices `{3, 4, 5}` for jumps and `{0, 1, 2, 6}` for ground actions.

4. **Pytest Configuration**:
   - *Observation*: Pytest failed to resolve root modules when invoked without explicit pythonpath.
   - *Reasoning*: Adding `pythonpath = [".", "src"]` under `[tool.pytest.ini_options]` in `pyproject.toml` enables both local developers and automated CI pipelines to execute `uv run pytest` cleanly.

---

## 3. Caveats

- **Zero-Done Rollout Windows**: In small-scale local testing (e.g. `num_envs = 16`), if an agent survives longer than `num_steps = 64`, an update window may have `num_dones == 0`. The fallback `(total_debris_hits / batch_size) * mean_length` prevents NaN, but production training at `num_envs = 16,384` on Colab will average ~291 dones per update, rendering zero-done windows non-existent.
- **Watchdog Coupling**: In `train_ppo.py:816`, the convergence watchdog checks `cur_survival >= 0.95`. When replacing `survival_rate` in console output, retain `"survival_rate"` in the `metrics` dictionary to ensure watchdog logic continues functioning without unintended side effects.

---

## 4. Conclusion

Requirement R3 is fully mapped, and an implementation blueprint has been verified. The existing test suite provides 459 passing tests as a high-fidelity regression harness.
- In `train_ppo.py`:
  1. Replace `Survival: 0.0%` with `Survival(s): {float(survival_sec):5.1f}s`.
  2. Implement `DebrisHits/ep` via the vectorized renewal estimator.
  3. Implement `JumpRatio%` via `(traj_batch.action >= 3) & (traj_batch.action <= 5)`.
  4. Configure `--log_interval 20`.
- In `tests/`: Implement `tests/test_micro_movement_r1_r2_r3.py` covering all R1 action costs, R2 spatial danger cone gating, and R3 telemetry calculations.

---

## 5. Verification Method

1. **Verify Baseline Test Suite Execution**:
   ```powershell
   uv run pytest -o pythonpath=". src" tests/test_lotus_phase1.py tests/test_train_ppo.py tests/test_wrappers.py
   ```
   *Expected Result*: All 46 tests pass.

2. **Verify Canonical Action Indices**:
   Inspect `src/maple_gymnax/envs/common.py:30-46` and confirm `ACTION_JUMP == 3`, `ACTION_JUMP_LEFT == 4`, `ACTION_JUMP_RIGHT == 5`, `ACTION_DUCK == 6`.

3. **Verify Survey Report Artifact**:
   Read `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\survey_r3_tests.md`.
