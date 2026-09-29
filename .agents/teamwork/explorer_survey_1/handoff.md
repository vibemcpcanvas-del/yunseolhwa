# Handoff Report: R1 & R2 Investigation & Specification
## Autonomous Micro-Movement & Threat-Gated Evasion in Maple Gymnax Lotus Phase 1

- **Sender**: `explorer_survey_1` (Explorer Subagent)
- **Recipient**: `orchestrator_3` (ID: `e8d54a3b-63f5-4fbd-92da-90a91af57a97`)
- **Date**: 2026-09-29T11:37:00Z
- **Working Directory**: `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_1`
- **Full Survey Report**: `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_1\survey_r1_r2.md`

---

## 1. Observation

1. **Current Action Space Mismatch (`src/maple_gymnax/envs/common.py:30-36`)**:
   ```python
   ACTION_NOOP: int = 0
   ACTION_LEFT: int = 1
   ACTION_RIGHT: int = 2
   ACTION_JUMP: int = 3
   ACTION_JUMP_LEFT: int = 4
   ACTION_JUMP_RIGHT: int = 5
   ACTION_DUCK: int = 6
   ```
   Whereas the authoritative user request specifies:
   `0 NOOP, 1 LEFT, 2 RIGHT, 3 DOWN, 4 JUMP, 5 JUMP_LEFT, 6 JUMP_RIGHT`.
   In the existing code, jump actions are $\{3, 4, 5\}$ and duck is $6$. In the target specification, ground actions $\{0, 1, 2, 3\}$ and jump actions $\{4, 5, 6\}$ are contiguous blocks.
   Downstream impact: `tests/e2e/test_tier1_features.py` lines 379 and 1468 used integer literal `3` expecting jump, and line 781 used integer literal `6` expecting duck. `tests/test_lotus_phase1.py` uses symbolic imports (`ACTION_JUMP`, `ACTION_DUCK`) and has zero regressions when constants change.

2. **`EnvState` Missing `last_action` (`src/maple_gymnax/envs/lotus_phase1.py:138-177`)**:
   `EnvState` currently contains kinematic, classic laser, debris, time, and remastered fields, but does NOT contain `last_action`.
   In Python dataclasses with Flax, non-default fields precede default fields. Adding `last_action: int = 0` among default fields (e.g. line 164) preserves 100% backward compatibility with existing keyword instantiations.

3. **Missing Arguments in `_compute_reward` (`src/maple_gymnax/envs/lotus_phase1.py:471-490`)**:
   `_compute_reward` currently accepts 17 arguments. It does NOT receive `action`, `last_action`, `on_ground_next`, or `state.player_x`. Consequently, action effort penalties, switching jitter, airborne hazard gating, and directional tap-dodging cannot currently be evaluated.

4. **Debris Tracking & Current Overhead Repulsion (`src/maple_gymnax/envs/lotus_phase1.py:547-564`)**:
   Debris is tracked via static padded arrays of capacity `MAX_DEBRIS = 30` with radii $16.0$ (small), $24.0$ (medium), and $36.0$ (large).
   Current continuous overhead repulsion is implemented as:
   ```python
   dx_deb = jnp.abs(px_next - debris_x)
   dy_deb = py_next - debris_y
   is_overhead = (debris_active & (debris_radius >= 24.0) & (dy_deb > 0.0) & (dy_deb < 180.0))
   overhead_weight = jnp.where(is_overhead, (debris_radius / 36.0) * jnp.exp(-0.5 * (dx_deb / 45.0) ** 2), 0.0)
   r_debris_repel = jnp.where(is_remastered_or_hybrid, -0.05 * jnp.sum(overhead_weight), 0.0)
   ```
   The coefficient is currently `-0.05`. R2 specifies scaling this to `-0.30`.

5. **Baseline Test Execution**:
   Running `uv run pytest tests/test_lotus_phase1.py` via `run_command` (task-86) completed with exit code 0:
   `============================= 33 passed in 27.97s =============================`.

---

## 2. Logic Chain

1. **Root Cause of 680-step / 11-second Plateau**:
   Zero-cost jump actions in high-velocity platformers lead to parabolic bunny-hopping because ballistic elevation trivially avoids ground hazards. However, parabolic arcs remove steering authority and place the player directly into descending debris corridors, causing 7.7 hits/ep.
2. **From Observation 1 to Action Partitioning**:
   Remapping the action space in `common.py` such that $\{0, 1, 2, 3\}$ are ground actions and $\{4, 5, 6\}$ are jump actions allows clean, branch-free JAX vectorization:
   `is_jump = action >= 4` and `r_action_jump = jnp.where(is_jump, -0.05, 0.0)`.
3. **From Observation 2 to State Tracking**:
   Adding `last_action: int = 0` to `EnvState` and recording `last_action=action` in `step_env` allows computing the Markovian action transition:
   `r_jitter = jnp.where(action != state.last_action, -0.02, 0.0)`.
   This penalizes 1-frame chatter and encourages 5–10 frame micro-taps.
4. **From Observation 3 & 4 to Threat Corridor Control (R2)**:
   Under high-threat debris ($r \ge 24\text{px}$, $|\Delta x| < 45\text{px}$, $\Delta y \in [0, 180\text{px}]$):
   - Jumping is penalized with $r_{\text{airborne\_hazard}} = -0.35$ if `jnp.logical_not(on_ground_next)`.
   - Grounded lateral motion away from debris center ($dx_{\text{next}} > dx_{\text{prev}}$) is rewarded with $r_{\text{tap\_dodge}} = +0.25$.
   - The net reward delta between jumping and grounded micro-movement under threat is:
     $$\Delta r = (+0.25) - (-0.05 - 0.35) = +0.65 \text{ per tick}$$
   - Scaling $\phi_{\text{debris}}$ from $-0.05$ to $-0.30$ provides a smooth continuous gradient driving the policy away before entering the discrete danger corridor.

---

## 3. Caveats

1. **Downstream Hardcoded Integers in E2E Tests**:
   `tests/e2e/test_tier1_features.py` contains 2 instances where action 3 was hardcoded as jump (lines 379, 1468) and 1 instance where action 6 was hardcoded as duck (line 781). The implementer must update these tests to use `ACTION_JUMP` and `ACTION_DOWN` (or `4` and `3`).
2. **Backward Compatibility Alias**:
   Existing code and imports expect `ACTION_DUCK`. We must maintain `ACTION_DUCK = ACTION_DOWN = 3` so existing code does not raise `ImportError` or `AttributeError`.

---

## 4. Conclusion

The technical path to fulfill R1 and R2 is fully established, mathematically verified, and documented with exact code diffs in `survey_r1_r2.md`:
1. Update `src/maple_gymnax/envs/common.py` to establish the $\{0: \text{NOOP}, 1: \text{LEFT}, 2: \text{RIGHT}, 3: \text{DOWN/DUCK}, 4: \text{JUMP}, 5: \text{JUMP\_LEFT}, 6: \text{JUMP\_RIGHT}\}$ action space.
2. Extend `EnvParams` with default reward hyperparameters and `EnvState` with `last_action: int = 0`.
3. Update `_compute_reward()` in `src/maple_gymnax/envs/lotus_phase1.py` with pure JAX, branch-free formulations of `r_action_jump`, `r_jitter`, `r_airborne_hazard`, `r_tap_dodge`, and the scaled `r_debris_repel` ($-0.30$).
4. Wire `last_action=action` into `step_env` and `last_action=0` into `reset_env`.

---

## 5. Verification Method

To independently verify the implementation:
1. **Core Unit & JIT Tests**:
   `uv run pytest tests/test_lotus_phase1.py`
   Must pass all 33 tests with zero JIT compilation failures.
2. **Analytical Reward Assertions**:
   Add test methods to `tests/test_lotus_phase1.py` asserting exact analytical reward differences:
   - Action cost: `assert abs((r_ground - r_jump) - 0.05) < 1e-4`
   - Jitter penalty: `assert abs((r_same - r_switch) - 0.02) < 1e-4`
   - Airborne hazard penalty: `assert abs((r_grounded - r_airborne) - 0.35) < 1e-4`
   - Tap-dodge clearance bonus: `assert abs((r_dodge - r_stationary) - 0.25) < 1e-4`
3. **E2E Feature Suite**:
   `uv run pytest tests/e2e/test_tier1_features.py`
   Confirm all kinematic and action space assertions pass.
