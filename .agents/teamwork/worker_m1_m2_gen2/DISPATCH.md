## 2026-09-29T11:56:34Z

You are worker_m1_m2_gen2, a replacement Worker subagent in a Teamwork hierarchy.
Your predecessor (worker_m1_m2) was interrupted by an execution environment token error right before implementing code changes.
Your working directory is: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m1_m2_gen2
Your parent is orchestrator_3 (conv ID: e8d54a3b-63f5-4fbd-92da-90a91af57a97).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY FIRST STEP:
Read the authoritative user request at:
c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
Specifically section dated 2026-09-29T11:28:08Z.

Also read:
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\PROJECT.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_1\survey_r1_r2.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\rules\jax-gymnax-rl.md

YOUR MISSION & WRITE SCOPE:
Implement Milestone 1 and Milestone 2 (Requirements R1 and R2).
You exclusively own these files:
- `src/maple_gymnax/envs/common.py`
- `src/maple_gymnax/envs/lotus_phase1.py`
- `src/maple_gymnax/envs/__init__.py`
- `tests/test_lotus_phase1.py`
- `tests/e2e/test_tier1_features.py` (only the 3 lines with hardcoded action integers)

Implementation tasks (refer to survey_r1_r2.md for exact line numbers and code snippets):
1. In `src/maple_gymnax/envs/common.py`:
   - Remap action constants cleanly per survey_r1_r2.md:
     ACTION_NOOP = 0
     ACTION_LEFT = 1
     ACTION_RIGHT = 2
     ACTION_DOWN = 3
     ACTION_DUCK = 3 (backward compatibility alias)
     ACTION_JUMP = 4
     ACTION_JUMP_LEFT = 5
     ACTION_JUMP_RIGHT = 6
   - Update canonical short aliases:
     NOOP = ACTION_NOOP
     LEFT = ACTION_LEFT
     RIGHT = ACTION_RIGHT
     DOWN = ACTION_DOWN
     DUCK = ACTION_DOWN
     JUMP = ACTION_JUMP
     JUMP_LEFT = ACTION_JUMP_LEFT
     JUMP_RIGHT = ACTION_JUMP_RIGHT
   - In `decode_action`: update `is_duck = action == ACTION_DOWN`, `is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)` (or `action >= ACTION_JUMP`).
2. In `src/maple_gymnax/envs/__init__.py`:
   - Export `ACTION_DOWN` and `DOWN` alongside existing exports.
3. In `src/maple_gymnax/envs/lotus_phase1.py`:
   - `EnvParams`: Add reward hyperparameters with default values:
     `r_action_jump_cost: float = -0.05`
     `r_jitter_cost: float = -0.02`
     `r_airborne_hazard_cost: float = -0.35`
     `r_tap_dodge_bonus: float = 0.25`
     `debris_repel_scale: float = -0.30`
   - `EnvState`: Add `last_action: int = 0` among default fields (e.g. line 164 after `time: int`).
   - `_compute_reward`: Add arguments `action: Union[int, chex.Array], last_action: Union[int, chex.Array], on_ground_next: chex.Array, state_player_x: chex.Array`. Implement:
     - `is_jump_action = (action == 4) | (action == 5) | (action == 6)` (or `action >= 4`)
     - `r_action_jump = jnp.where(is_jump_action, params.r_action_jump_cost, 0.0)`
     - `r_jitter = jnp.where(action != last_action, params.r_jitter_cost, 0.0)`
     - High-threat overhead danger cone check ($r \ge 24$, $|\Delta x_{\text{next}}| < 45\text{px}$, $\Delta y \in [0, 180\text{px}]$):
       `is_threat_overhead = (debris_active & (debris_radius >= 24.0) & (dy_deb > 0.0) & (dy_deb < 180.0) & (dx_deb_next < 45.0))`
       `has_overhead_threat = jnp.any(is_threat_overhead)`
     - `r_airborne_hazard = jnp.where(has_overhead_threat & jnp.logical_not(on_ground_next), params.r_airborne_hazard_cost, 0.0)`
     - Grounded tap-dodge check: moving away horizontally ($dx_{\text{next}} > dx_{\text{prev}}$) while grounded (`on_ground_next`):
       `moving_away = is_threat_overhead & (dx_deb_next > dx_deb_prev)`
       `can_tap_dodge = on_ground_next & jnp.any(moving_away)`
       `r_tap_dodge = jnp.where(can_tap_dodge, params.r_tap_dodge_bonus, 0.0)`
     - Continuous overhead Gaussian repulsion potential: scale to `-0.30` (`params.debris_repel_scale * jnp.sum(overhead_weight)`).
     - Sum all reward components into total return.
   - `step_env`: Pass `last_action=action` to `state_next = EnvState(...)` and pass `action, state.last_action, on_ground_next, state.player_x` to `_compute_reward(...)`.
   - `reset_env`: Pass `last_action=0` to initial `EnvState(...)`.
4. In `tests/test_lotus_phase1.py`:
   - Add unit tests verifying analytical reward deltas (action jump cost -0.05, jitter -0.02, airborne hazard -0.35, tap dodge +0.25, Gaussian potential -0.30).
5. In `tests/e2e/test_tier1_features.py`:
   - Update lines 379, 1468 (which used raw integer `3` expecting jump) to `ACTION_JUMP`, and line 781 (which used `6` expecting duck) to `ACTION_DOWN`.
6. Run verification:
   `uv run pytest tests/test_lotus_phase1.py tests/e2e/test_tier1_features.py`
   Ensure all tests pass and JIT compilation succeeds with zero ConcretizationTypeError.
