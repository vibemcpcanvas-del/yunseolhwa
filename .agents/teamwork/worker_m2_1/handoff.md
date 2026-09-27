# Handoff Report — Milestone 2: Core Gymnax Lotus Phase 1 Environment

**Author**: Milestone 2 Worker (`teamwork_preview_worker`)  
**Date**: 2026-09-27T01:25:30Z  
**Type**: Hard Handoff (Milestone 2 Implementation and Verification Complete)  
**Deliverable**:
- `src/maple_gymnax/envs/__init__.py`
- `src/maple_gymnax/envs/common.py`
- `src/maple_gymnax/envs/lotus_phase1.py`
- `tests/test_lotus_phase1.py`

---

## 1. Observation

1. **Implementation Files**:
   - `src/maple_gymnax/envs/common.py`: Implemented compile-time static constants (`MAX_DEBRIS = 30`, `MODE_CLASSIC = 0`, `MODE_REMASTERED = 1`, `MODE_HYBRID = 2`, action enums `ACTION_NOOP=0`, `ACTION_LEFT=1`, `ACTION_RIGHT=2`, `ACTION_JUMP=3`, `ACTION_JUMP_LEFT=4`, `ACTION_JUMP_RIGHT=5`, `ACTION_DUCK=6`). Implemented `sat_aabb_projection`, `compute_laser_collisions` (orthogonal raycast distance + directional dot-product masking + SAT normal projection across 4 orthogonal arms), `compute_debris_collisions` (vectorized Euclidean distance against static padded debris tensors), and `decode_action`.
   - `src/maple_gymnax/envs/lotus_phase1.py`: Implemented `EnvParams(flax.struct.dataclass)` with physical defaults and contract aliases (`map_width`, `core_pos`, `player_width`, `jump_impulse`, `laser_thickness`, etc.), `EnvState(flax.struct.dataclass)` encapsulating all kinematics, 30-capacity debris tensors, classic laser angle, and Remastered states (security gauge, overload timer, boss HP/shield, tracking laser, electric floor). Implemented `LotusPhase1Env(environment.Environment)` with branch-free `reset_env`, `step_env`, `get_obs`, `get_extended_obs`, `is_terminal`, `action_space`, and `observation_space`.
   - `src/maple_gymnax/envs/__init__.py`: Exposed public environment classes, states, parameters, constants, and math functions.
   - `tests/test_lotus_phase1.py`: Implemented 30 test cases across 6 test classes (`TestJITAndVMap`, `TestPlayerKinematics`, `TestLaserCollision`, `TestDebrisPhysics`, `TestRemasteredLotus`, `TestEpisodeLifecycle`).

2. **Test Command Results**:
   - Execution of unit test suite:
     ```
     uv run pytest tests/test_lotus_phase1.py -v
     ============================= 30 passed in 13.52s =============================
     ```
     Verifying JIT compilation of `env.reset_env` and `env.step_env`, and `jax.vmap` scaling across 1,024, 2,048, and 4,096 parallel environments without memory errors or shape mismatches.
   - Execution of Tier 1 keyword test verification:
     ```
     uv run pytest tests/e2e/test_tier1_features.py -k "lotus or gymnax or laser or debris" -v
     ====================== 7 passed, 123 deselected in 4.35s ======================
     ```
   - Execution of full Tier 1 feature suite:
     ```
     uv run pytest tests/e2e/test_tier1_features.py -v
     ============================ 130 passed in 39.32s =============================
     ```
   - Execution of the full project test suite:
     ```
     uv run pytest tests/test_lotus_phase1.py tests/test_wz_parser.py tests/e2e/test_tier1_features.py tests/e2e/test_tier2_boundaries.py tests/e2e/test_tier3_pairwise.py tests/e2e/test_tier4_scenarios.py -q
     361 passed in 85.78s (0:01:25)
     ```
   - Python bytecode compilation:
     ```
     uv run python -m py_compile src/maple_gymnax/envs/__init__.py src/maple_gymnax/envs/common.py src/maple_gymnax/envs/lotus_phase1.py tests/test_lotus_phase1.py
     Exit code 0 (clean compilation)
     ```

3. **Dynamic Resolution**:
   - Verified that `tests/e2e/contract_harness.py` dynamically imported and bound `LotusPhase1Env`, `EnvParams`, and `EnvState` from `src/maple_gymnax/envs/lotus_phase1.py` with 100% pass rate across all contract and boundary tests.

---

## 2. Logic Chain

1. **Branch-Free XLA JIT Compliance**:
   - *Observation*: Dynamic Python conditionals on tracer values trigger `jax.errors.ConcretizationTypeError`.
   - *Implementation*: Every branch in `step_env` (horizontal movement, jump impulse, floor snapping, wall clamping, laser hit, debris collision, overload trigger, artillery hazard, electric floor, damage application, and termination) is executed exclusively via `jnp.where` and logical bitwise operators.
   - *Result*: `jax.jit(env.reset_env)` and `jax.jit(env.step_env)` compile without any concretization errors.

2. **Static Padded Array Guarantees**:
   - *Observation*: Dynamic list sizing or passing dynamic shape arguments triggers XLA graph invalidation and shape recompilation.
   - *Implementation*: Defined `MAX_DEBRIS: int = 30` as a compile-time static integer constant. Debris arrays have fixed shape `(30,)` initialized via `jnp.zeros(MAX_DEBRIS, ...)` and updated via PRNG slot masking `(jnp.arange(MAX_DEBRIS) == slot_idx) & can_spawn`.
   - *Result*: State shape invariance verified across arbitrary step unrolls and batched `jax.vmap` up to 4,096 instances.

3. **Rotating Cross Laser Physics**:
   - *Observation*: Laser has 4 orthogonal beams rotating at $\omega = 0.5235$ rad/s originating from core center $(683.0, 384.0)$.
   - *Implementation*: Longitudinal ray projection $d_{\parallel, k} = \mathbf{v} \cdot \mathbf{u}_k$ is masked by $[R_{core}, L_{max}]$ to ensure ray origin starts at core perimeter and stops behind core; transverse distance $d_{\perp, k}$ is checked against laser half-thickness plus exact SAT AABB projection radius $r_{proj, k} = \frac{w}{2}|\sin\theta_k| + \frac{h}{2}|\cos\theta_k|$.
   - *Result*: Directional masking, core clearance, and lethal collision verified across unit and E2E tests.

4. **Remastered Lotus Mechanics**:
   - *Observation*: April 2024 Remaster specifies security gauge accumulation ($0.8\%$/s), Overload mode at 100% lasting 25s, right safe zone ($x \ge 1150$), lethal electric floor, and friendly fire boss guidance.
   - *Implementation*: Mode switching (`MODE_CLASSIC = 0`, `MODE_REMASTERED = 1`, `MODE_HYBRID = 2`) is integrated without Python if/else branches (`params.mode != MODE_REMASTERED`, `params.mode != MODE_CLASSIC`).
   - *Result*: All Remastered features pass unit tests and scenario tests.

---

## 3. Caveats

- In `LotusPhase1Env`, `get_obs` returns the standard 130-dimensional normalized observation tensor expected by downstream wrappers and existing E2E tests. An auxiliary `get_extended_obs` method is provided for 142-dimensional Remastered/Hybrid observations containing security gauge, overload timer, boss HP/shield, and tracking laser states.
- The default game mode in `EnvParams` is `MODE_CLASSIC` (0) to preserve baseline compatibility with benchmark contracts; setting `mode=MODE_REMASTERED` (1) or `mode=MODE_HYBRID` (2) activates the modern Remastered mechanics seamlessly.

---

## 4. Conclusion

Milestone 2 is 100% complete. The Gymnax MapleStory Lotus Phase 1 environment is fully implemented with genuine physics, SAT raycasting, static array debris management, and modern Remastered mechanics. All 30 unit tests in `tests/test_lotus_phase1.py` and all 361 tests across the entire repository pass with zero errors. The implementation is ready for Milestone 3 (RL Wrappers & Rollout Runner).

---

## 5. Verification Method

To independently reproduce and verify this milestone:

```bash
# 1. Run unit test suite for Lotus Phase 1
uv run pytest tests/test_lotus_phase1.py -v

# 2. Run keyword-filtered Tier 1 tests
uv run pytest tests/e2e/test_tier1_features.py -k "lotus or gymnax or laser or debris" -v

# 3. Run all Tier 1 feature tests
uv run pytest tests/e2e/test_tier1_features.py -v

# 4. Run entire project test suite
uv run pytest tests/test_lotus_phase1.py tests/test_wz_parser.py tests/e2e/ -q
```

**Invalidation Conditions**:
- Any occurrence of `jax.errors.ConcretizationTypeError` during `jax.jit(env.step_env)`.
- Any `ShapeError` or shape mismatch during `jax.vmap` at batch sizes 1024, 2048, or 4096.
- Any non-zero exit code or test assertion failure in `tests/test_lotus_phase1.py`.
