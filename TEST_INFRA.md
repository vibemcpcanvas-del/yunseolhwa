# E2E Test Infrastructure Specification: Autonomous Micro-Movement & Threat-Gated Evasion

**Project**: Maple Gymnax Lotus Phase 1  
**Document**: `TEST_INFRA.md`  
**Author**: `test_writer_m4` (Test Writer Specialist & QA)  
**Parent Agent**: `orchestrator_3`  
**Reference Document**: `ORIGINAL_REQUEST.md` (2026-09-29T11:28:08Z), `PROJECT.md`  
**Applicable Rules**: `.agents/rules/jax-gymnax-rl.md` (Rule 8: Physical Contract Decoupling & Test Pre-Assertions)

---

## 1. Executive Overview & Test Architecture

This document defines the 4-tier End-to-End (E2E) testing and verification methodology for the **Autonomous Micro-Movement & Threat-Gated Evasion** enhancement in Maple Gymnax Lotus Phase 1.

The primary objective of this test infrastructure is to rigorously guard against:
1. **The Bunny-Hop Local Minimum Artifact**: Unpenalized high-energy jump spam (53.7% `JUMP_RIGHT`) causing an 11-second / 680-step survival plateau.
2. **Action Chatter & Instability**: Single-tick directional oscillation caused by lack of switching penalties.
3. **Hazard Blindness**: Reckless jumping into descending debris clusters ($r \ge 24\text{px}$).
4. **False Telemetry**: Truncated or biased logging (binary 60s survival, zero-done starvation).
5. **Evaluation Drift**: Unverified checkpoints failing Gate 1 or Gate 2 acceptance standards.

### The 4-Tier Testing Methodology Matrix

| Tier | Focus Scope | Target Mechanics | Verification Method | Assert Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Unit Feature Coverage** | R1 Action Cost & Jitter, R2 Hazard Corridor & Tap-Dodging, R3 Telemetry Overhaul, R4 Gate Evaluation | Isolated single-step JAX environments, deterministic fixtures | Exact analytical delta equality (`abs(r_a - r_b - expected) < 1e-5`) |
| **Tier 2** | **Boundary & Corner Cases** | Danger cone geometric borders ($\Delta x = 45\text{px}, \Delta y \in [0, 180\text{px}]$), ground transitions, radius thresholds ($r=24\text{px}$), wall bounds | Precision float perturbations ($x \pm \epsilon$), state boundary injections | Exact discrete activation flags & physical state transitions |
| **Tier 3** | **Cross-Feature Combinations** | Jumping under debris cone while switching actions, dodging towards wall with active laser, multi-threat resolution | Multi-factor simultaneous interaction matrices | Superposition of independent penalties & physical contracts |
| **Tier 4** | **Real-World Trajectory Scenarios** | Multi-step unrolls comparing bunny-hop spam vs grounded micro-dodging, checkpoint evaluation across gates | JIT-compiled `RolloutRunner` and CLI `eval_gates.py` execution | Statistical distributions (action percentages, survival seconds, hits/ep) |

---

## 2. Canonical Action Space Contract

Following `survey_r1_r2.md` and `PROJECT.md`, the action space is cleanly partitioned into contiguous sets:
- **Grounded Actions**: $\mathcal{A}_{\text{ground}} = \{0, 1, 2, 3\}$
  - `0`: `ACTION_NOOP` (Neutral / Idling)
  - `1`: `ACTION_LEFT` (Walk Left at 400.0 px/s)
  - `2`: `ACTION_RIGHT` (Walk Right at 400.0 px/s)
  - `3`: `ACTION_DOWN` / `ACTION_DUCK` (Crouch, reducing hitbox height to 35.0 px)
- **Airborne / Jump Actions**: $\mathcal{A}_{\text{jump}} = \{4, 5, 6\}$
  - `4`: `ACTION_JUMP` (Vertical Jump, $v_y = -650.0$ px/s)
  - `5`: `ACTION_JUMP_LEFT` (Diagonal Jump Left, $v_x = -400.0, v_y = -650.0$)
  - `6`: `ACTION_JUMP_RIGHT` (Diagonal Jump Right, $v_x = +400.0, v_y = -650.0$)

---

## 3. Tier 1: Unit Feature Coverage Specifications

Requires $\ge 5$ test cases per requirement feature (R1, R2, R3, R4).

### 3.1 Requirement R1: Action Cost & Energy Regularization Engine
- **Test 1.1 (`test_r1_action_cost_vertical_jump`)**:
  - *Input*: `state.player_on_ground = True`, `state.last_action = ACTION_JUMP`. Action `ACTION_JUMP` (4) vs `ACTION_NOOP` (0).
  - *Expected*: $r_{\text{noop}} - r_{\text{jump}} = 0.05 \pm 10^{-5}$.
- **Test 1.2 (`test_r1_action_cost_diagonal_jumps`)**:
  - *Input*: Compare actions `ACTION_JUMP_LEFT` (5) and `ACTION_JUMP_RIGHT` (6) against `ACTION_NOOP` (0) with matching `last_action`.
  - *Expected*: Both diagonal jumps incur exactly $r_{\text{action\_jump}} = -0.05$.
- **Test 1.3 (`test_r1_action_cost_ground_actions_zero`)**:
  - *Input*: Evaluate all actions $a \in \{0, 1, 2, 3\}$ (`NOOP`, `LEFT`, `RIGHT`, `DOWN`) with $a = \text{last\_action}$.
  - *Expected*: All ground actions incur $0.0$ action effort cost; rewards are identical in neutral environment.
- **Test 1.4 (`test_r1_jitter_penalty_action_repeat_zero`)**:
  - *Input*: Repeat action $a_t = a_{t-1}$ across all 7 actions.
  - *Expected*: $r_{\text{jitter}} = 0.0$ across all repeated actions.
- **Test 1.5 (`test_r1_jitter_penalty_action_switching`)**:
  - *Input*: Switch action from `ACTION_LEFT` (1) to `ACTION_RIGHT` (2), `ACTION_NOOP` (0), or `ACTION_DOWN` (3).
  - *Expected*: $r_{\text{repeat}} - r_{\text{switch}} = 0.02 \pm 10^{-5}$.
- **Test 1.6 (`test_r1_env_state_last_action_lifecycle`)**:
  - *Input*: `reset_env` followed by consecutive `step_env` calls with distinct actions.
  - *Expected*: Initial `state.last_action == 0`; after step with action $k$, `next_state.last_action == k` across JIT.

### 3.2 Requirement R2: Tap-Dodging Hazard Corridor & Overhead Repulsion
- **Test 1.7 (`test_r2_danger_cone_spatial_gating`)**:
  - *Input*: Place active debris of radius $r = 24.0$ at $(\Delta x = 30.0, \Delta y = 100.0)$ vs radius $r = 16.0$.
  - *Expected*: Active danger cone triggers only for $r \ge 24.0$. Small debris ($r=16$) does not activate the danger cone.
- **Test 1.8 (`test_r2_airborne_hazard_penalty_analytical`)**:
  - *Input*: Player under active danger cone. Compare airborne state (`player_on_ground = False`) vs grounded state (`player_on_ground = True`).
  - *Expected*: Airborne state incurs additional penalty $r_{\text{airborne\_hazard}} = -0.35 \pm 10^{-5}$.
- **Test 1.9 (`test_r2_grounded_tap_dodge_clearance_bonus`)**:
  - *Input*: Debris overhead at $x = 683.0$, player at $x = 660.0$ (debris to the right). Player steps `ACTION_LEFT` (moving away, expanding separation) while grounded.
  - *Expected*: $r_{\text{moving\_away}} - r_{\text{neutral}} = 0.25 \pm 10^{-5}$.
- **Test 1.10 (`test_r2_tap_dodge_inward_movement_zero_bonus`)**:
  - *Input*: Same hazard configuration; player steps `ACTION_RIGHT` (moving toward hazard) or `ACTION_NOOP`.
  - *Expected*: $r_{\text{tap\_dodge}} = 0.0$. Inward or stationary actions receive zero bonus.
- **Test 1.11 (`test_r2_continuous_gaussian_potential_rescaled`)**:
  - *Input*: Debris overhead at $\Delta x = 0.0$ and $\Delta y = 100.0$ with $r = 36.0$.
  - *Expected*: Potential delta equals $-0.30 \cdot (36.0 / 36.0) \cdot \exp(0) = -0.30 \pm 10^{-4}$.

### 3.3 Requirement R3: Transparent Real-Time Console Telemetry
- **Test 1.12 (`test_r3_realtime_survival_seconds_formula`)**:
  - *Input*: Episode lengths 680, 1200, 3600.
  - *Expected*: Real-time survival seconds = $680/60 = 11.33\text{s}$, $1200/60 = 20.0\text{s}$, $3600/60 = 60.0\text{s}$. (Contrasted with legacy binary metric yielding 0.0% for 680 and 1200).
- **Test 1.13 (`test_r3_jump_ratio_canonical_partition`)**:
  - *Input*: Trajectory batch containing [20 NOOP, 20 LEFT, 20 RIGHT, 20 DOWN, 10 JUMP, 5 JUMP_LEFT, 5 JUMP_RIGHT].
  - *Expected*: `jump_ratio = 20 / 100 = 20.0%`. Action 3 (`DOWN`) is strictly counted as grounded.
- **Test 1.14 (`test_r3_debris_hits_renewal_estimator`)**:
  - *Input*: Trajectory batch with 50 total debris hits across 10 episode completions.
  - *Expected*: Estimated debris hits per episode = $50 / 10 = 5.0$.
- **Test 1.15 (`test_r3_zero_done_renewal_fallback`)**:
  - *Input*: Rollout window with zero episode terminations (`has_dones = False`).
  - *Expected*: Fallback estimator produces finite float without IEEE division by zero or NaN.
- **Test 1.16 (`test_r3_log_callback_format_stability`)**:
  - *Input*: Invoke `_log_callback` with extreme values (inf, nan, negative return).
  - *Expected*: Clean string formatting without uncaught formatting exceptions.

### 3.4 Requirement R4: Automated Gate 1 & Gate 2 Evaluation
- **Test 1.17 (`test_r4_gate1_jump_right_threshold`)**:
  - *Input*: Action distribution with `JUMP_RIGHT = 19.9%` (pass) vs `20.1%` (fail).
  - *Expected*: Gate 1 condition 1 passes if and only if `jump_right_pct < 20.0`.
- **Test 1.18 (`test_r4_gate1_grounded_mobility_threshold`)**:
  - *Input*: Action distribution with `Grounded = 50.1%` (pass) vs `49.9%` (fail).
  - *Expected*: Gate 1 condition 2 passes if and only if `grounded_pct > 50.0`.
- **Test 1.19 (`test_r4_gate2_debris_hits_threshold`)**:
  - *Input*: Checkpoint evaluation with `3.5 hits/ep` (pass) vs `3.6 hits/ep` (fail).
  - *Expected*: Gate 2 condition 1 passes if and only if `avg_debris_hits <= 3.5`.
- **Test 1.20 (`test_r4_gate2_survival_steps_threshold`)**:
  - *Input*: Checkpoint evaluation with `1200.0 steps` (pass) vs `1199.0 steps` (fail).
  - *Expected*: Gate 2 condition 2 passes if and only if `avg_steps >= 1200.0`.
- **Test 1.21 (`test_r4_gate2_shield_damage_threshold`)**:
  - *Input*: Checkpoint evaluation with `140.0 / 200` (pass) vs `139.9 / 200` (fail).
  - *Expected*: Gate 2 condition 3 passes if and only if `avg_shield_dmg >= 140.0`.
- **Test 1.22 (`test_r4_eval_gates_exit_codes`)**:
  - *Input*: CLI execution of `eval_gates.py` on passing vs failing synthetic checkpoints.
  - *Expected*: Exit code 0 on pass, exit code 1 on fail.

---

## 4. Tier 2: Boundary & Corner Cases

- **Test 2.1 (`test_t2_danger_cone_lateral_epsilon_boundary`)**:
  - Test lateral distance $|\Delta x| = 44.9\text{px}$ (inside cone, threat active) vs $|\Delta x| = 45.1\text{px}$ (outside cone, threat inactive).
- **Test 2.2 (`test_t2_danger_cone_vertical_lower_boundary`)**:
  - Test vertical distance $\Delta y = 0.0\text{px}$ (threat overhead, active) vs $\Delta y = -0.1\text{px}$ (debris below player center, inactive).
- **Test 2.3 (`test_t2_danger_cone_vertical_upper_boundary`)**:
  - Test vertical distance $\Delta y = 180.0\text{px}$ (active) vs $\Delta y = 180.1\text{px}$ (too high, inactive).
- **Test 2.4 (`test_t2_threat_radius_threshold_boundary`)**:
  - Test radius $r = 23.9\text{px}$ (below threat threshold, inactive) vs $r = 24.0\text{px}$ (high-threat, active).
- **Test 2.5 (`test_t2_inactive_debris_mask_isolation`)**:
  - Place high-threat debris geometrically inside the cone but with `debris_active[i] = False`.
  - Assert danger cone remains completely inactive (zero penalties, zero dodge bonuses).
- **Test 2.6 (`test_t2_ground_contact_jump_transition`)**:
  - At tick $t$, agent is on ground (`player_on_ground = True`). Agent executes `ACTION_JUMP`.
  - At tick $t+1$, `on_ground_next` must evaluate to `False` immediately, triggering airborne hazard penalties if threat is present.
- **Test 2.7 (`test_t2_floor_landing_transition`)**:
  - Falling agent ($v_y > 0$) landing on $y = \text{floor\_y}$.
  - At tick $t+1$, `on_ground_next = True`, resetting airborne hazard penalty.
- **Test 2.8 (`test_t2_arena_wall_clamping_boundary`)**:
  - Agent at extreme left boundary ($x \le \text{wall\_left}$) attempting `ACTION_LEFT`.
  - Horizontal position remains clamped; $\Delta x$ does not expand; tap-dodge bonus evaluates to $0.0$.
- **Test 2.9 (`test_t2_opposing_debris_corridors`)**:
  - Two high-threat debris particles overhead simultaneously: one to the left ($x = 650$), one to the right ($x = 710$).
  - Moving left increases separation from debris 2 while decreasing separation from debris 1. Evaluates proper vectorized reduction (`jnp.any(moving_away)`).

---

## 5. Tier 3: Cross-Feature Combinations

- **Test 3.1 (`test_t3_simultaneous_jump_under_danger_cone`)**:
  - Combines:
    1. Action Effort Penalty: $-0.05$
    2. Airborne Hazard Penalty: $-0.35$
    3. Overhead Gaussian Repulsion: $-0.30$
  - Cumulative instantaneous penalty $\le -0.70$.
- **Test 3.2 (`test_t3_switching_while_dodging`)**:
  - Agent switches from `ACTION_NOOP` to `ACTION_LEFT` while under overhead threat.
  - Combines:
    1. Action Switch Jitter: $-0.02$
    2. Grounded Tap-Dodge Bonus: $+0.25$
  - Net instantaneous delta: $+0.23$ (positive exploration incentive maintained).
- **Test 3.3 (`test_t3_moving_into_debris_while_switching`)**:
  - Agent switches action into the hazard trajectory ($dx_{\text{next}} < dx_{\text{prev}}$).
  - Combines:
    1. Action Switch Jitter: $-0.02$
    2. Tap-Dodge Bonus: $0.0$
    3. Overhead Repulsion: $-0.30$
  - Net instantaneous delta: $-0.32$ (strong disincentive).
- **Test 3.4 (`test_t3_jumping_away_from_debris`)**:
  - Agent executes `ACTION_JUMP_LEFT` moving away from a rightward debris.
  - Because `on_ground_next == False`, `r_tap_dodge` is denied ($0.0$), while `r_action_jump` ($-0.05$) and `r_airborne_hazard` ($-0.35$) apply.
  - Proves that airborne movement cannot cheat or claim the tap-dodge bonus.
- **Test 3.5 (`test_t3_ducking_under_danger_cone`)**:
  - Agent executes `ACTION_DOWN` under danger cone.
  - Ground action cost is $0.0$; agent remains grounded (`on_ground_next = True`, no airborne penalty); hitbox height drops from 60px to 35px.
- **Test 3.6 (`test_t3_multi_step_telemetry_pipeline`)**:
  - Execute a 10-step sequence with mixed actions, jumps, and debris collisions.
  - Verify that `EnvState.last_action`, `info["debris_hit"]`, and trajectory metrics propagate correctly without loss or type corruption.

---

## 6. Tier 4: Real-World Trajectory Scenarios

- **Test 4.1 (`test_t4_trajectory_bunny_hop_vs_tap_dodge`)**:
  - Simulate a 120-step trajectory (2.0 seconds) under a continuous falling debris barrage.
  - Policy A (Bunny-Hop Spam): 100% `ACTION_JUMP_RIGHT`.
  - Policy B (Grounded Tap-Dodging): alternating grounded lateral steps with 10-frame micro-taps.
  - Assert that cumulative return of Policy B substantially exceeds Policy A ($\Delta R > +15.0$).
- **Test 4.2 (`test_t4_trajectory_action_chatter_suppression`)**:
  - Compare Policy C (1-frame alternating noise: `LEFT, RIGHT, LEFT, RIGHT, ...`) vs Policy D (10-frame micro-taps: `10x LEFT, 10x RIGHT`).
  - Assert jitter regularization penalizes Policy C by exactly $-0.02 \times \text{switches}$.
- **Test 4.3 (`test_t4_legacy_checkpoint_gate_failure`)**:
  - Run `eval_gates.py` against legacy `checkpoints/step_4800`.
  - Verify Gate 1 fails on `JUMP_RIGHT` ratio (> 50%) when evaluated on baseline, confirming the presence of the unregularized bunny-hop artifact.
- **Test 4.4 (`test_t4_synthetic_checkpoint_gate_pass`)**:
  - Evaluate synthetic compliant rollout distributions against Gate 1 and Gate 2.
  - Verify clean Gate 1 & Gate 2 PASS with exit code 0.

---

## 7. Verification Execution Runbook

```bash
# 1. Run full micro-movement test suite across all 4 tiers
uv run pytest tests/test_micro_movement_r1_r2_r3.py -v

# 2. Verify eval_gates.py CLI interface
uv run python eval_gates.py --help

# 3. Verify early checkpoint against Gate 1
uv run python eval_gates.py checkpoints/step_4800 --gate 1 --episodes 10 --eval_steps 1200

# 4. Verify mature checkpoint against Gate 1 & Gate 2
uv run python eval_gates.py checkpoints/step_35000 --gate all --episodes 10 --eval_steps 1200
```
