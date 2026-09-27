## 2026-09-27T01:14:47+09:00

You are the Milestone 2 Worker (teamwork_preview_worker).
Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m2_1\
Project Root: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
Original Request: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
Project Spec: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\PROJECT.md
Gymnax Blueprint: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\gymnax_env_spec.md
Survey 2 Handoff: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\handoff.md
Reference Engine: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\tests\e2e\contract_harness.py

You MUST read ORIGINAL_REQUEST.md, PROJECT.md, and gymnax_env_spec.md first.

Your exclusive write ownership:
- src/maple_gymnax/envs/__init__.py
- src/maple_gymnax/envs/common.py
- src/maple_gymnax/envs/lotus_phase1.py
- tests/test_lotus_phase1.py

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope of Milestone 2:
1. Implement `src/maple_gymnax/envs/common.py`:
   - Static constants (`MAX_DEBRIS: int = 30`, `MODE_CLASSIC = 0`, `MODE_REMASTERED = 1`, `MODE_HYBRID = 2`, action definitions: NOOP=0, LEFT=1, RIGHT=2, JUMP=3, JUMP_LEFT=4, JUMP_RIGHT=5, DUCK=6).
   - Vectorized branch-free math: SAT AABB projection, orthogonal raycast distance + directional dot product masking for 4-arm rotating laser, Euclidean distance collision detection for static padded debris arrays.
2. Implement `src/maple_gymnax/envs/lotus_phase1.py`:
   - Implement `LotusPhase1Env(Environment)` using `flax.struct.dataclass` for `EnvState`.
   - `EnvState` must encapsulate: player position (x, y), velocity (vx, vy), hp, invincibility timer, laser angle, debris arrays of size 30 (x, y, vy, radius, damage, active, type), remastered state (security gauge, overload timer, is_overload, boss shield hp, tracking laser angle/timer, floor electric charge/active), step count.
   - `reset_env(key, params)`: deterministic reset returning `(obs, state)`.
   - `step_env(key, state, action, params)`: strictly branch-free state transition using `jnp.where` and `jax.lax.select` (zero Python if/else). Supports Classic mode (rotating laser + debris), Remastered mode (security gauge natural increase, friendly fire boss guidance via tracking laser & arm slam reducing gauge, 25s overload barrage, electric floor, shield), and Hybrid mode.
   - Observation spaces: normalized 130-dim or 142-dim Box space. Action space: Discrete(7).
3. Implement `tests/test_lotus_phase1.py`:
   - Test JIT compilation of `env.reset` and `env.step` with zero errors.
   - Test `jax.vmap` across 1,024 to 4,096 parallel environments.
   - Test laser collision with directional dot product masking.
   - Test falling debris Euclidean collision and active mask updating.
   - Test remastered gauge accumulation, friendly-fire boss guidance, and overload transitions.
   - Test player death termination and step truncation.
4. Run verification commands:
   `uv run pytest tests/test_lotus_phase1.py -v`
   `uv run pytest tests/e2e/test_tier1_features.py -k "lotus or gymnax or laser or debris" -v`
5. Write a comprehensive 5-component handoff report in `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m2_1\handoff.md`.
6. Send message to parent upon completion.
