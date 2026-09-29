# BRIEFING — 2026-09-29T11:56:34Z

## Mission
Implement Milestone 1 (Action Remapping R1) and Milestone 2 (Anti-Jitter / Anti-Jumping Reward Shaping R2) for MapleGymnax Lotus Phase 1 environment.

## 🔒 My Identity
- Archetype: worker
- Roles: implementer, qa, specialist
- Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m1_m2_gen2
- Original parent: e8d54a3b-63f5-4fbd-92da-90a91af57a97
- Milestone: Milestone 1 & Milestone 2 (R1 & R2)

## 🔒 Key Constraints
- Minimal change principle: only modify designated files (`src/maple_gymnax/envs/common.py`, `src/maple_gymnax/envs/lotus_phase1.py`, `src/maple_gymnax/envs/__init__.py`, `tests/test_lotus_phase1.py`, `tests/e2e/test_tier1_features.py`).
- Remap action constants cleanly per survey_r1_r2.md: NOOP=0, LEFT=1, RIGHT=2, DOWN=3, DUCK=3, JUMP=4, JUMP_LEFT=5, JUMP_RIGHT=6.
- Maintain backward compatibility for ACTION_DUCK / DUCK.
- Implement reward shaping hyperparameters and components with real JAX vectorization (no Python if/else on tracer values, no ConcretizationTypeError).
- Do not cheat, hardcode test outputs, or create dummy implementations.

## Current Parent
- Conversation ID: e8d54a3b-63f5-4fbd-92da-90a91af57a97
- Updated: 2026-09-29T11:56:34Z

## Task Summary
- **What to build**: Action remapping (R1) and reward shaping against jump addiction/jitter (R2).
- **Success criteria**: Tests in `tests/test_lotus_phase1.py` and `tests/e2e/test_tier1_features.py` pass; JIT runs cleanly; analytical reward deltas verified.
- **Interface contracts**: PROJECT.md, survey_r1_r2.md
- **Code layout**: `src/maple_gymnax/envs/`, `tests/`

## Key Decisions Made
- Starting task according to survey_r1_r2.md.

## Artifact Index
- `.agents/teamwork/worker_m1_m2_gen2/DISPATCH.md` — Assignment
- `.agents/teamwork/worker_m1_m2_gen2/BRIEFING.md` — Situational awareness
- `.agents/teamwork/worker_m1_m2_gen2/progress.md` — Liveness and progress tracking

## Change Tracker
- **Files modified**: None yet
- **Build status**: Not started
- **Pending issues**: None

## Quality Status
- **Build/test result**: Not run yet
- **Lint status**: Not run yet
- **Tests added/modified**: None yet

## Loaded Skills
- None
