# BRIEFING — 2026-09-29T11:43:00Z

## Mission
Implement Milestone 1 (Action Space Clean Remap R1) and Milestone 2 (Anti-Jump Spam & Tap-Dodge Ground Hazard Shaping R2).

## 🔒 My Identity
- Archetype: worker
- Roles: implementer, qa
- Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m1_m2
- Original parent: e8d54a3b-63f5-4fbd-92da-90a91af57a97
- Milestone: M1_M2

## 🔒 Key Constraints
- Exclusive write scope:
  - `src/maple_gymnax/envs/common.py`
  - `src/maple_gymnax/envs/lotus_phase1.py`
  - `src/maple_gymnax/envs/__init__.py`
  - `tests/test_lotus_phase1.py`
  - `tests/e2e/test_tier1_features.py` (only the 3 lines with hardcoded action integers)
- DO NOT CHEAT: Genuine logic only, no hardcoding, no facades.
- Pure JAX functional compatibility: No ConcretizationTypeError under `jax.jit`.
- Preserve backward compatibility alias `ACTION_DUCK = 3`.
- Verification command: `uv run pytest -o pythonpath=". src" tests/test_lotus_phase1.py tests/e2e/test_tier1_features.py`

## Current Parent
- Conversation ID: e8d54a3b-63f5-4fbd-92da-90a91af57a97
- Updated: not yet

## Task Summary
- **What to build**: Action remapping (DOWN=3, JUMP=4, JUMP_LEFT=5, JUMP_RIGHT=6) with aliases, EnvParams/EnvState extensions, anti-jump cost (-0.05), jitter cost (-0.02), airborne hazard cost (-0.35), grounded tap-dodge bonus (+0.25), scaled Gaussian repulsion potential (-0.30), unit tests and test updates.
- **Success criteria**: All tests pass, JIT compilation succeeds cleanly.
- **Interface contracts**: PROJECT.md, survey_r1_r2.md
- **Code layout**: src/maple_gymnax/envs/, tests/

## Change Tracker
- **Files modified**: [TBD]
- **Build status**: [TBD]
- **Pending issues**: [TBD]

## Quality Status
- **Build/test result**: [TBD]
- **Lint status**: [TBD]
- **Tests added/modified**: [TBD]

## Key Decisions Made
- [TBD]

## Artifact Index
- DISPATCH.md — Assignment instructions
- progress.md — Liveness heartbeat & step progress
- handoff.md — Final self-contained 5-component report
