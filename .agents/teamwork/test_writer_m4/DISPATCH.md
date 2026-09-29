## 2026-09-29T11:42:29Z
You are test_writer_m4, a Test Writer subagent in a Teamwork hierarchy.
Your working directory is: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\test_writer_m4
Your parent is orchestrator_3 (conv ID: e8d54a3b-63f5-4fbd-92da-90a91af57a97).

MANDATORY FIRST STEP:
Read the authoritative user request at:
c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
Specifically section dated 2026-09-29T11:28:08Z.

Also read:
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\PROJECT.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\survey_r3_tests.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_3\survey_r4_eval.md

YOUR MISSION & WRITE SCOPE:
Design and build the comprehensive E2E Verification & Test Suite for the project (Milestone 4).
You exclusively own these files:
- `TEST_INFRA.md` at project root
- `tests/test_micro_movement_r1_r2_r3.py`
- `eval_gates.py` at project root
- `TEST_READY.md` at project root (publish when complete)

Requirements:
1. Create `TEST_INFRA.md` following the 4-tier methodology:
   - Tier 1: Feature coverage (>=5 test cases per feature for R1 Action Cost/Jitter, R2 Hazard Corridor/Tap-Dodging, R3 Telemetry, R4 Gate 1 & 2 evaluation).
   - Tier 2: Boundary & corner cases (hazard cone borders, ground contact transitions, zero/extreme values).
   - Tier 3: Cross-feature combinations (simultaneous jumping under debris cone, switching actions while dodging, metric propagation).
   - Tier 4: Real-world application scenarios (simulated trajectories verifying suppression of bunny-hop spam).
2. Author `tests/test_micro_movement_r1_r2_r3.py`:
   - Implement all 4 tiers of tests covering R1, R2, R3.
   - Use clean, deterministic test fixtures with Gymnax / JAX.
3. Author `eval_gates.py`:
   - Automated CLI evaluation tool that loads a checkpoint (using `eval_checkpoint.py` infrastructure or direct JAX rollout), evaluates it for greedy policy, and programmatically checks:
     - Gate 1: `JUMP_RIGHT < 20%`, `Grounded (NOOP+LEFT+RIGHT+DOWN) > 50%` (for early checkpoints `step_4800`+)
     - Gate 2: `DebrisHits <= 3.5 / ep`, `Survival steps >= 1200` (20.0s), `ShieldDamage >= 140 / 200` (for mature checkpoints)
     - Returns exit code 0 if gate passes, exit code 1 if failed, with structured ASCII table report.
4. Execute `uv run pytest tests/test_micro_movement_r1_r2_r3.py` and `uv run python eval_gates.py --help`.
5. When all tests pass, publish `TEST_READY.md` at project root summarizing test runner command, tier coverage counts, and feature checklist.

OUTPUT REQUIREMENTS:
- Update progress in `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\test_writer_m4\progress.md`.
- Document your artifacts and test verification in `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\test_writer_m4\handoff.md`.
- Send completion message to orchestrator_3 via `send_message`.
