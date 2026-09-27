# Plan — MapleStory Lotus Phase 1 Gymnax & WZ Pipeline

## Objective
Coordinate the end-to-end development of the MapleStory Lotus Phase 1 Gymnax environment, WZ client parser pipeline, RL framework wrappers, persona system prompt, test suite, and hardware-tailored SPS benchmarks according to the hardware profile and acceptance criteria.

## Phase 0: Survey & Spec Mining (3 Parallel Explorers)
1. **Explorer 1 (WZ Data & Parser Spec)**: Probe `C:\mp`, `wz_json_restorer.py`, `BossSuu.img.json`, `bossSuu.img.json` map atlas, existing directory structure, physics parameters, frame delays, and hitboxes.
2. **Explorer 2 (Gymnax/JAX Architecture & Physical Spec)**: Map out Gymnax `EnvParams`/`EnvState` structure, branch-free JAX/XLA design, laser rotation math, falling debris vectorization, collision detection math, and reward/termination logic.
3. **Explorer 3 (RL Wrappers, Stoix/PureJaxRL/Flashbax & Hardware Benchmarks)**: Investigate Python environment (Python 3.12, JAX, Flax, Gymnax, Pytest), wrapper specs (`FlattenObservationWrapper`, `AutoResetWrapper`), rollout runner, and SPS benchmark design for Ryzen 5600X & RX 6600 XT.

## Phase 1: PROJECT.md & Feature Inventory
- Synthesize findings from Explorers into `PROJECT.md` at project root.
- Define Feature Inventory, Milestones, Interface Contracts, and Code Layout.
- Verify every feature is assigned to a milestone.

## Phase 2: Dual Track Dispatch
- **E2E Testing Track**: Spawn E2E Testing Orchestrator to build requirement-driven test infrastructure and test suites (Tiers 1-4) covering all features.
- **Implementation Track**: Spawn sub-orchestrators for milestones:
  - M1: WZ Client Data Parser Pipeline (`src/maple_gymnax/parser/wz_parser.py`)
  - M2: Gymnax Lotus Phase 1 Simulator Core (`src/maple_gymnax/envs/lotus_phase1.py`)
  - M3: RL Framework Wrappers & High-speed Rollout Runner (`src/maple_gymnax/wrappers/`)
  - M4: Persona Control System Prompt (`src/maple_gymnax/prompts/persona_system_prompt.py`)

## Phase 3: Final Milestone & E2E Validation
- Wait for `TEST_READY.md`.
- Final Milestone Sub-Orchestrator:
  - Phase 1: Pass 100% of E2E tests (Tiers 1-4).
  - Phase 2: Adversarial Coverage Hardening (Tier 5) via Challenger -> Worker -> Reviewer loop.
  - Phase 3: SPS Benchmark execution across batch sizes (256, 512, 1024, 2048, 4096).

## Phase 4: Final Synthesis & Reporting
- Verify all acceptance criteria.
- Produce comprehensive final report for parent / user.
