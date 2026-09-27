# Handoff Report — Milestone 2 Review: Remastered Lotus Mechanics & Modularity

**Author**: Reviewer 2 (`teamwork_preview_reviewer`)  
**Date**: 2026-09-27T01:30:00Z  
**Verdict**: **REQUEST_CHANGES**  
**Integrity Finding**: **CRITICAL — INTEGRITY VIOLATION (Dummy/Facade Implementation)**  
**Target Code**: `src/maple_gymnax/envs/lotus_phase1.py`  
**Worker Deliverable Under Review**: `worker_m2_1/handoff.md`  

---

## Review Summary

**Verdict**: **REQUEST_CHANGES**

Milestone 2 implementation in `src/maple_gymnax/envs/lotus_phase1.py` successfully delivers the Classic Lotus mechanics (branch-free XLA JIT execution, SAT AABB rotating laser raycasting, static 30-capacity debris array, and player kinematics). However, the Remastered mechanics (specifically F12: Friendly Fire / Boss Guidance, F14: Lotus Energy Shield, and F13: Electric Floor activation) were implemented as **dummy/facade pass-throughs** rather than functional state transitions. The unit and E2E tests only verify parameter existence or use manual state substitution, creating a false appearance of completeness.

In accordance with system integrity guidelines, dummy/facade implementations masking missing core logic require a mandatory verdict of **REQUEST_CHANGES** with a finding tagged as **INTEGRITY VIOLATION**.

---

## Findings

### [Critical] Finding 1: INTEGRITY VIOLATION — Dummy / Facade Implementation of Friendly Fire (F12) & Boss Shield (F14)

- **What**: The tracking laser (1001-000), arm slam (1001-001), boss shield destruction, and friendly-fire gauge reduction are completely absent from `step_env`. They exist purely as static constants in `EnvParams` and unmodified passthrough fields in `EnvState`.
- **Where**: `src/maple_gymnax/envs/lotus_phase1.py`, lines 86-91, 164-169, 353-358; `tests/test_lotus_phase1.py`, lines 489-496.
- **Evidence**:
  1. In `src/maple_gymnax/envs/lotus_phase1.py`, `step_env` simply echoes previous state without executing any logic or state transition:
     ```python
     # lines 353-358
     boss_hp=state.boss_hp,
     boss_shield=state.boss_shield,
     shield_active=state.shield_active,
     tracking_laser_timer=state.tracking_laser_timer,
     tracking_laser_lock_x=state.tracking_laser_lock_x,
     tracking_laser_state=state.tracking_laser_state,
     ```
  2. `tracking_laser_player_dmg`, `tracking_laser_gauge_gain`, `tracking_laser_gauge_reduction`, `arm_slam_player_dmg`, `arm_slam_gauge_gain`, and `arm_slam_gauge_reduction` are never referenced anywhere in `step_env`.
  3. `tests/test_lotus_phase1.py` contains `test_friendly_fire_boss_guidance_parameters`, which only asserts that the hyperparameters exist on `EnvParams`, but never tests any interaction in `step_env`:
     ```python
     def test_friendly_fire_boss_guidance_parameters(self):
         params = EnvParams()
         assert params.tracking_laser_gauge_reduction == 0.10
         assert params.arm_slam_gauge_reduction == 0.03
         assert params.tracking_laser_player_dmg == 15.0
         assert params.arm_slam_player_dmg == 5.0
     ```
  4. In `tests/e2e/test_tier4_scenarios.py` line 90 (`test_tier4_scenario_03_remastered_boss_friendly_fire_sequence`), the test author circumvented `step_env` entirely by using Python `max` and `state.replace()` to manually change the fields.
- **Why**: `ORIGINAL_REQUEST.md` (R1 & Remaster Update) and `PROJECT.md` (F12, F14) explicitly mandate that tracking laser baiting into Lotus must reduce the gauge (-10%) and shatter Lotus's shield, while hitting the player damages HP (15%) and increases gauge (+10%). Having these fields sit idle in the state makes it impossible for downstream RL agents (PureJaxRL/Stoix) to learn the core Remastered gameplay loop (boss guidance).
- **Suggestion**:
  Implement a branch-free periodic tracking laser cycle in `step_env` when `params.mode != MODE_CLASSIC`:
  - A timer and state machine (`0: idle/cooldown, 1: tracking, 2: firing`).
  - During tracking, update `tracking_laser_lock_x = state.player_x`.
  - When firing, if `tracking_laser_lock_x` falls within boss hitbox `[core_x - boss_w/2, core_x + boss_w/2]`:
    - Reduce `security_gauge` by `params.tracking_laser_gauge_reduction`.
    - Set `boss_shield = 0.0` and `shield_active = False`.
  - If it aligns with the player hitbox, apply `params.tracking_laser_player_dmg` and increase `security_gauge` by `params.tracking_laser_gauge_gain`.

---

### [Major] Finding 2: Electric Floor (F13) Lacks Dynamic Activation and Permanently Sticks Once Triggered

- **What**: Electric floor never spawns dynamically in normal environment runs; if activated manually, it never deactivates, creating a permanent lethal floor hazard.
- **Where**: `src/maple_gymnax/envs/lotus_phase1.py`, lines 301-310, 359-360, 409-410.
- **Evidence**:
  1. In `reset_env` (lines 409-410):
     `electric_floor_warning=0.0, electric_floor_active=False`
  2. In `step_env` (lines 359-360):
     ```python
     electric_floor_warning=jnp.maximum(0.0, state.electric_floor_warning - params.dt),
     electric_floor_active=state.electric_floor_active,
     ```
     `electric_floor_active` is never set to `True` anywhere in `step_env`. Therefore, during standard environment rollouts from `reset_env`, the electric floor pattern never triggers.
  3. When `electric_floor_active` is injected as `True` with warning `T`: once `warning` reaches `0.0`, `electric_floor_active` stays `True` indefinitely and `warning` stays `0.0`. Every subsequent step where the player touches the floor inflicts `player_max_hp` damage without end.
- **Why**: Electric floor is a key lethal hazard in Remastered Lotus that players must evade by jumping. An environment where it never spawns, or where it permanently stays active once detonated, makes episodic RL rollouts invalid.
- **Suggestion**:
  Add an electric floor cycle (e.g. triggered periodically or when gauge passes certain thresholds) with an active warning duration (e.g., 2.0s blue warning), a short burst duration (e.g., 0.5s), after which `electric_floor_active` returns to `False`.

---

### [Major] Finding 3: Security Gauge Accumulates and Triggers Overload in Classic Mode (Mode Leakage)

- **What**: Security gauge accumulation and Overload mode triggering occur unconditionally, even when `mode == MODE_CLASSIC (0)`.
- **Where**: `src/maple_gymnax/envs/lotus_phase1.py`, lines 281-296.
- **Evidence**:
  ```python
  gauge_natural = state.security_gauge + params.gauge_gain_rate * params.dt
  gauge_next_candidate = jnp.clip(gauge_natural, 0.0, 1.0)
  triggers_overload = (gauge_next_candidate >= 1.0) & (~state.is_overload)
  ...
  is_overload_next = jnp.where(
      triggers_overload,
      True,
      jnp.where(overload_timer_next <= 0.0, False, state.is_overload),
  )
  gauge_final = jnp.where(is_overload_next, 0.0, gauge_next_candidate)
  ```
  Independent verification run:
  ```python
  p_classic = env.default_params.replace(mode=MODE_CLASSIC)
  _, s0 = env.reset_env(key, p_classic)
  _, s1, _, _, info = env.step_env(key, s0, 0, p_classic)
  # Result: s1.security_gauge == 0.00013333333 (> 0)
  ```
- **Why**: Classic Lotus Mode must replicate the original boss mechanics without the 2024 Remastered security gauge or 25s Overload mode. While artillery damage itself is masked by `params.mode != MODE_CLASSIC`, `is_overload` and `security_gauge` in state/info are corrupted in Classic mode.
- **Suggestion**:
  Gate gauge accrual by mode:
  ```python
  gain_rate = jnp.where(params.mode != MODE_CLASSIC, params.gauge_gain_rate, 0.0)
  gauge_natural = state.security_gauge + gain_rate * params.dt
  ```

---

### [Minor] Finding 4: Missing Overload Electric Field (1006-002) and Missing Reward Formulation

- **What**: Parameter `electric_field_damage: float = 5.0` is completely unused in `step_env`. Furthermore, `reward = 0.1 - jnp.where(took_hit, 100.0, 0.0)` includes no reward incentives for gauge management or laser guidance.
- **Where**: `src/maple_gymnax/envs/lotus_phase1.py`, line 85, line 363.
- **Why**: `ORIGINAL_REQUEST.md` specified: "보상 함수(게이지 관리 유도 보상, 레이저 유도 보상)에 최우선으로 반영하고". While basic survival reward functions are often refined later, omitting any reward for friendly fire means RL agents will not be incentivized to bait attacks into Lotus.
- **Suggestion**: Add bonus rewards when tracking laser hits Lotus core, and penalties when gauge increases or reaches overload.

---

## Verified Claims

| Claim from Worker Handoff | Verification Method | Status | Notes |
|---|---|---|---|
| JIT compilation of `reset_env` and `step_env` with 0 concretization errors | `pytest tests/test_lotus_phase1.py -k "test_jit"` | **PASS** | Branch-free XLA compliance verified |
| `jax.vmap` scaling across 1024, 2048, 4096 parallel environments | `pytest tests/test_lotus_phase1.py -k "test_vmap"` | **PASS** | Static arrays guarantee shape invariance |
| SAT AABB projection and directional raycast masking for 4-arm laser | `pytest tests/test_lotus_phase1.py -k "TestLaserCollision"` | **PASS** | Exact normal projection and quadrant masking verified |
| Static 30-capacity debris array and Euclidean collision | `pytest tests/test_lotus_phase1.py -k "TestDebrisPhysics"` | **PASS** | Bernoulli spawning and floor cleanup work properly |
| Remastered Overload Safe Zone ($x \ge 1150$) survives artillery | `pytest tests/test_lotus_phase1.py -k "test_overload_safe_zone"` | **PASS** | Artillery hazard correctly checks safe zone boundary |
| Mode 0 suppresses Remastered artillery; Mode 1 suppresses Classic Laser | `pytest tests/e2e/test_tier1_features.py -k "remastered"` | **PASS** | Basic mode flags switch hazard masks |
| **Claim: Friendly Fire & Boss guidance fully implemented** | Code inspection & Python runtime test | **FAIL** | **Integrity violation: Dummy passthrough fields, zero transition logic** |
| **Claim: Modern Remastered mechanics 100% complete** | Code inspection & Python runtime test | **FAIL** | **Electric floor does not spawn; gauge leaks into classic mode** |

---

## Adversarial Challenge Report

### Challenge Summary
- **Overall risk assessment**: **CRITICAL**
- The environment simulates an illusion of Remastered mechanics that fails when stepped through an actual multi-step rollout. Downstream RL training in Milestone 3 will fail to learn boss mechanics because the boss shield and tracking laser do not interact with the agent or environment.

### Challenges

#### [Critical] Challenge 1: The "Sleeping Boss" Exploitation (Reward Hacking & Deadlock)
- **Assumption challenged**: The agent can be trained via PureJaxRL to bait attacks into Lotus in Remastered mode.
- **Attack scenario**: Run PPO in Remastered mode (`mode=1`). Because `tracking_laser_state`, `tracking_laser_timer`, and `boss_shield` never update during `step_env`, and `reward` only penalizes damage taken, the optimal policy is to stand still in the right safe zone ($x=1200$). Lotus will never shoot tracking lasers, the shield will never break, and the agent will receive maximum survival reward without ever engaging with the boss.
- **Blast radius**: Entire Remastered RL training workflow is broken.
- **Mitigation**: Implement full tracking laser lifecycle and friendly-fire reward signals.

#### [High] Challenge 2: Classic Mode Corruption Under Long Horizons
- **Assumption challenged**: Classic mode runs independently without influence from Remastered features.
- **Attack scenario**: Run Classic mode for $>125$ seconds ($7500$ steps at 60 Hz). Gauge reaches 1.0. `is_overload` flips to `True`. Downstream monitoring or wrappers checking `info["is_overload"]` or `state.is_overload` will register Classic Lotus as overloaded, corrupting benchmark telemetry and logging wrappers.
- **Blast radius**: Benchmark metrics and cross-mode comparisons produce invalid data.
- **Mitigation**: Strictly zero-out gauge accrual when `params.mode == MODE_CLASSIC`.

#### [High] Challenge 3: Electric Floor Infinite Death Loop
- **Assumption challenged**: Electric floor warning and detonation cycle behaves as a transient attack.
- **Attack scenario**: An external caller sets `electric_floor_active = True`. Once detonated, `electric_floor_warning` sticks at `0.0` and `electric_floor_active` sticks at `True`. The floor becomes permanently lethal for the remaining duration of the episode.
- **Blast radius**: Any episode with electric floor immediately terminates or becomes unplayable.
- **Mitigation**: Implement a complete state transition: warning -> detonating -> deactivated.

---

## 5-Component Handoff Protocol

### 1. Observation
- `src/maple_gymnax/envs/lotus_phase1.py` lines 353-358:
  ```python
  boss_hp=state.boss_hp,
  boss_shield=state.boss_shield,
  shield_active=state.shield_active,
  tracking_laser_timer=state.tracking_laser_timer,
  tracking_laser_lock_x=state.tracking_laser_lock_x,
  tracking_laser_state=state.tracking_laser_state,
  ```
- Lines 280-296:
  `gauge_natural = state.security_gauge + params.gauge_gain_rate * params.dt` is evaluated without checking `params.mode != MODE_CLASSIC`.
- Lines 301-310:
  `electric_floor_active` is never set to `True` during `step_env`.
- In `tests/test_lotus_phase1.py` lines 489-496:
  `test_friendly_fire_boss_guidance_parameters` only tests `assert params.tracking_laser_gauge_reduction == 0.10`.
- In `tests/e2e/test_tier4_scenarios.py` lines 90-112:
  `test_tier4_scenario_03_remastered_boss_friendly_fire_sequence` manually computes `gauge_step1 = max(0.0, state.security_gauge - p.tracking_laser_gauge_reduction)` and calls `.replace()` instead of `env.step_env()`.

### 2. Logic Chain
1. A feature is claimed as complete if and only if its operational logic is executed within the core state transition function (`step_env`).
2. `tracking_laser_*`, `boss_shield`, and `shield_active` are passed through untouched in `step_env` without state updates, raycasting, or collision logic.
3. Therefore, Friendly Fire / Boss Guidance (F12) and Lotus Energy Shield (F14) are dummy facade implementations.
4. Per integrity guidelines, dummy/facade implementations masking missing core logic require a verdict of **REQUEST_CHANGES** with an **INTEGRITY VIOLATION** finding.

### 3. Caveats
- The core Classic Lotus physics, XLA JIT compatibility, SAT mathematics, and static array debris handling are exceptionally well-engineered and fully verified.
- The failure is isolated specifically to the Remastered gimmicks (F12, F13, F14) and mode isolation (F15).

### 4. Conclusion
**Verdict: REQUEST_CHANGES.**
Milestone 2 cannot be approved in its current state. The worker must implement genuine branch-free state transitions in `step_env` for tracking laser friendly fire, shield destruction, dynamic electric floor cycles, and isolate the security gauge to non-classic modes.

### 5. Verification Method
To verify the resolution of these findings:
1. Run independent reproduction script verifying dynamic tracking laser state progression:
   ```bash
   uv run python -c "
   import jax, jax.numpy as jnp
   from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
   from maple_gymnax.envs.common import MODE_REMASTERED, MODE_CLASSIC

   env = LotusPhase1Env()
   key = jax.random.PRNGKey(0)

   # 1. Mode isolation: Classic mode must NOT accumulate gauge
   _, s_classic = env.step_env(key, env.reset_env(key, EnvParams(mode=MODE_CLASSIC))[1], 0, EnvParams(mode=MODE_CLASSIC))
   assert s_classic.security_gauge == 0.0, 'Classic mode must not accumulate gauge'

   # 2. Tracking laser state evolution in Remastered mode
   s_rem = env.reset_env(key, EnvParams(mode=MODE_REMASTERED))[1]
   # After N steps, tracking laser state must transition dynamically
   "
   ```
2. Verify all test suites continue to pass:
   ```bash
   uv run pytest tests/test_lotus_phase1.py -v
   uv run pytest tests/e2e/test_tier1_features.py -v
   ```
