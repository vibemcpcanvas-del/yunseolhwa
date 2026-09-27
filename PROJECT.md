# Project: MapleStory Lotus Phase 1 Gymnax Environment & WZ Pipeline

## Architecture
This project implements a high-performance functional reinforcement learning simulator for MapleStory Lotus Phase 1 (both Classic and April 2024 Remastered mechanics) running on JAX/XLA, accompanied by a WZ client data parser, RL wrappers (PureJaxRL, Stoix, Flashbax), an agent persona system prompt, and hardware-optimized SPS benchmarks.

```
                              [C:\mp WZ Data / Restored_Data]
                                             │
                                             ▼
                             [WZ Parser: wz_parser.py]
                                             │
                                             ▼ (EnvParams JSON Schema)
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
        [Classic Lotus Gimmick]                           [Remastered Lotus Gimmick]
        - Rotating Cross Laser (4 beams,                  - Security & Annihilation Gauge
          orthogonal distance + dot product)              - 25s Overload/Destruction Mode
        - Falling Debris (30 static array,                - Friendly Fire / Boss Guidance
          Euclidean collision)                            - Electric Floor & Boss Shield
                    └────────────────────────┬────────────────────────┘
                                             ▼
                        [LotusPhase1Env (Gymnax Core)]
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [RL Wrappers & Rollout]                        [E2E Testing Suite (Tiers 1-4)]
          - FlattenObservationWrapper                    - 100% Functional Coverage
          - PureJaxRLAdapterWrapper                      - Boundary & Stress Tests
          - LogWrapper & Flashbax Zero-Copy              - Pairwise & Real-world Workloads
          - jax.vmap + jax.lax.scan Runner                           │
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                                [Milestone 5: Final Pass]
                                - 100% E2E Test Suite Pass
                                - Tier 5 Adversarial Hardening
                                - Hardware SPS Benchmarks (Ryzen 5600X)
```

## Code Layout
```
bold-faraday/
├── pyproject.toml                                # Project metadata, dependencies (uv)
├── src/
│   └── maple_gymnax/
│       ├── __init__.py
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── schema.py                         # EnvParams JSON Schema & dataclass
│       │   └── wz_parser.py                      # WZ data extraction & restoration pipeline
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── common.py                         # Math utilities, SAT AABB, static constants
│       │   └── lotus_phase1.py                   # Gymnax Lotus Phase 1 environment core
│       ├── wrappers/
│       │   ├── __init__.py
│       │   ├── flatten_obs.py                    # Static 1D observation flattener
│       │   ├── purejaxrl_adapter.py              # 6-tuple to 5-tuple step API adapter
│       │   ├── log_wrapper.py                    # Zero-overhead episode metric logger
│       │   ├── rollout_runner.py                 # jax.vmap + jax.lax.scan high-speed runner
│       │   └── flashbax_adapter.py               # Flashbax Pytree zero-copy buffer
│       └── prompts/
│           ├── __init__.py
│           └── persona_system_prompt.py          # LLM persona control system prompt & templates
├── tests/
│   ├── conftest.py                               # Pytest fixtures & environment setup
│   ├── test_wz_parser.py                         # Unit tests for WZ data parser & schema
│   ├── test_lotus_phase1.py                      # Unit tests for JAX/XLA simulator dynamics
│   ├── test_wrappers.py                          # Unit tests for RL wrappers & rollout runner
│   ├── test_persona_prompt.py                    # Unit tests for persona prompt generation
│   └── e2e/                                      # Opaque-box E2E testing track suite
│       ├── test_tier1_features.py                # Tier 1: Feature coverage
│       ├── test_tier2_boundaries.py              # Tier 2: Boundary & corner cases
│       ├── test_tier3_pairwise.py                # Tier 3: Cross-feature combinations
│       └── test_tier4_scenarios.py               # Tier 4: Real-world application scenarios
└── benchmarks/
    └── benchmark_sps.py                          # Hardware SPS benchmarks (B=256..4096)
```

## Feature Inventory
Every feature identified during survey is assigned to a milestone below:

| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F01 | Python 3.12 Virtual Environment | Setup isolated uv virtual environment with jax, flax, gymnax, flashbax, pytest | M1 | Survey |
| F02 | WZ File & Directory Crawler | Crawl `C:\mp` and `C:\mp\Restored_Data` for BossSuu and map atlas data | M1 | Survey |
| F03 | WZ Anchor & Frame Extractor | Parse pixel anchor offsets, frame delay (ms->s), and bounding boxes | M1 | Survey |
| F04 | EnvParams JSON Schema Validation | Generate and validate structured `EnvParams` JSON against Draft 2020-12 schema | M1 | Survey |
| F05 | EnvParams & EnvState Schema | `flax.struct.dataclass` immutable parameters and dynamic state definition | M2 | Survey |
| F06 | Map & Physics Coordinate System | 1366x768 canvas, core center (683, 384), floor platform y=605, wall boundaries | M2 | Survey |
| F07 | Player Kinematics & Hitbox | Continuous position, velocity, gravity, jump impulse, 40x60 hitbox, 7 discrete actions | M2 | Survey |
| F08 | Rotating Cross Laser Math | 4 orthogonal beams, $\omega = 0.5235$ rad/s, SAT AABB projection + dot product masking | M2 | Survey |
| F09 | Falling Debris Vectorization | Static array of 30, PRNG slot allocation, Euclidean distance collision detection | M2 | Survey |
| F10 | Security & Annihilation Gauge | 0.6%~2.0%/s natural gain, friendly fire delta, overload trigger at 100% | M2 | Remaster Update |
| F11 | 25s Overload / Destruction Mode | Pattern 1006-000 horizontal barrage + 1006-002 electric field with safe zone | M2 | Remaster Update |
| F12 | Friendly Fire / Boss Guidance | Tracking laser (1001-000) & arm slam (1001-001) baited to hit Lotus reduces gauge | M2 | Remaster Update |
| F13 | Floor Electric Discharge | Blue electric warning -> lethal burst avoided by jumping/airborne | M2 | Remaster Update |
| F14 | Lotus Energy Shield | Protective shield broken by baited tracking laser hits | M2 | Remaster Update |
| F15 | Modular Gimmick Mode Selector | Support Classic (0), Remastered (1), and Hybrid (2) without code branching | M2 | Remaster Update |
| F16 | Branch-Free XLA JIT Compliance | Zero Python if/else conditionals, pure jnp.where and jax.lax.select | M2 | Survey |
| F17 | FlattenObservationWrapper | Static 1D normalized float array flattener (130-dim classic / 142-dim remastered) | M3 | Survey |
| F18 | PureJaxRL Adapter Wrapper | Bridge Gymnax 0.0.9 6-tuple step return to standard 5-tuple `(obs, state, r, done, info)` | M3 | Survey |
| F19 | LogWrapper Episode Tracker | Zero-overhead branch-free episode return and length tracking | M3 | Survey |
| F20 | High-Speed Scan Rollout Runner | `jax.vmap` + `jax.lax.scan` batched trajectory unrolling in accelerator memory | M3 | Survey |
| F21 | Flashbax Zero-Copy Buffer Interface | Pytree replay buffer integration without host memory transfers | M3 | Survey |
| F22 | Persona System Prompt Formulation | Master prompt for LLMs to transform WZ data to physical tensors & mock code | M4 | Survey |
| F23 | E2E Testing Infrastructure | Standalone requirement-driven test runner and harness (Tiers 1-4) | E2E Track | Survey |
| F24 | Final Milestone E2E 100% Pass | All E2E test cases pass with exit code 0 | M5 | Acceptance Criteria |
| F25 | Tier 5 Adversarial Coverage Hardening | White-box adversarial testing via Challenger -> Worker -> Reviewer loop | M5 | Dual Track Spec |
| F26 | Hardware SPS Benchmarks Suite | SPS measurements on Ryzen 5600X CPU and RX 6600 XT (B=256, 512, 1024, 2048, 4096) | M5 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| E2E | E2E Testing Track | Design test infrastructure, Tiers 1-4 test cases, publish `TEST_READY.md` | none | DONE |
| M1 | WZ Parser Pipeline & Env Setup | Python 3.12 uv env, `schema.py`, `wz_parser.py`, `test_wz_parser.py` | none | DONE |
| M2 | Core Gymnax Lotus Phase 1 Env | `lotus_phase1.py` with classic & remastered mechanics, `test_lotus_phase1.py` | M1 | IN_PROGRESS |
| M3 | RL Wrappers & Rollout Runner | `flatten_obs.py`, `purejaxrl_adapter.py`, `log_wrapper.py`, `rollout_runner.py`, `flashbax_adapter.py` | M2 | PLANNED |
| M4 | Persona System Prompt | `persona_system_prompt.py`, prompt templates, `test_persona_prompt.py` | M1, M2 | PLANNED |
| M5 | Final E2E Pass, Hardening & SPS | Pass 100% E2E tests (Tiers 1-4), Tier 5 adversarial hardening, `benchmark_sps.py` | E2E, M3, M4 | PLANNED |

## Interface Contracts

### 1. WZ Parser ↔ Gymnax Env (`schema.py`)
- `wz_parser.parse_wz_to_env_params(wz_dir: str) -> EnvParams`
- Generates a dictionary / JSON conforming to `EnvParams` schema with fields:
  - `map_width: float = 1366.0`, `map_height: float = 768.0`
  - `core_pos: Tuple[float, float] = (683.0, 384.0)`, `core_radius: float = 120.0`
  - `floor_y: float = 605.0`
  - `laser_omega: float = 0.5235`, `laser_thickness: float = 20.0`
  - `player_width: float = 40.0`, `player_height: float = 60.0`, `player_speed: float = 400.0`
  - `gravity: float = 1800.0`, `jump_impulse: float = 650.0`, `dt: float = 1/60`
  - `max_debris: int = 30` (static compile-time constant)
  - Remastered fields: `gauge_gain_rate: float = 0.008`, `overload_duration: float = 25.0`

### 2. Gymnax Env ↔ RL Wrappers
- `LotusPhase1Env.step_env(key, state, action, params) -> (obs, state, reward, terminated, info)`
- `LotusPhase1Env.reset_env(key, params) -> (obs, state)`
- `PureJaxRLAdapterWrapper.step(key, state, action, params) -> (obs, state, reward, done, info)`
  - Where `done = jnp.logical_or(terminated, truncated)`
- `FlattenObservationWrapper.step(key, state, action, params) -> (flat_obs, state, reward, ...)`
  - `flat_obs` shape is statically `(130,)` for classic mode or `(142,)` for unified/remastered mode.
- `RolloutRunner.run(rng, initial_state, num_steps) -> (final_state, trajectory)`
  - Trajectory contains batched `(obs, action, reward, done)`.

### 3. E2E Test Runner ↔ Implementation Modules
- Test Runner Invocation: `uv run pytest tests/e2e -v`
- Pass/Fail Semantics: Exit code 0, all asserts succeed across Tiers 1-4.
- Zero private coupling: tests invoke public API (`LotusPhase1Env`, `wz_parser`, wrappers).
