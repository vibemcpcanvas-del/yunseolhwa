# BRIEFING — 2026-09-26T15:37:30Z

## Mission
Investigate Python environment, RL framework wrappers (PureJaxRL, Stoix/Stoa, Flashbax), Persona prompt specification, and SPS benchmark design for Maple Gymnax Lotus Phase 1 (Classic & Remastered).

## 🔒 My Identity
- Archetype: explorer
- Roles: teamwork_preview_explorer
- Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_3\
- Original parent: d30c9047-58e0-49d2-8a5c-3a4baedf9ed6
- Milestone: survey_rl_wrappers_and_benchmarks

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Scope: Workspace python environment, RL Framework Wrappers (PureJaxRL, Stoix/Stoa, Flashbax), Persona System Prompt, SPS Benchmark design
- Deliverables: rl_wrappers_bench_spec.md, handoff.md, send_message to parent

## Current Parent
- Conversation ID: d30c9047-58e0-49d2-8a5c-3a4baedf9ed6
- Updated: 2026-09-26T15:36:21Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (Classic R1-R4 & Remastered Lotus April 2024 update lines 54-78)
  - Windows host environment: Python 3.14.7, uv 0.12.6, Python 3.12 cached at `AppData\Roaming\uv\python`
  - WSL2 `Ubuntu-24.04-ROCmLab`: Python 3.12.3, ROCm `/dev/kfd` status
  - Gymnax 0.0.9 architecture (`Environment.step` 6-tuple return, built-in auto-reset)
  - Flashbax 0.1.3 Pytree replay buffer zero-copy workflow
  - Empirical Ryzen 5 5600X CPU benchmark: CartPole achieved 2.97M SPS with `jax.lax.scan` at B=4096
- **Key findings**:
  - `.venv` needs creation with `uv venv --python 3.12 .venv`.
  - Dependencies (`jax`, `flax`, `gymnax`, `flashbax`, `pytest`) resolve in 848ms without conflicts.
  - Unified 109-dim observation vector designed for `FlattenObservationWrapper`, accommodating both Classic (cross laser, debris) and Remastered (gauge, friendly fire, overload mode, shield).
  - High-speed rollout runner architecture combining `jax.vmap` and `jax.lax.scan` defined.
  - Full Persona System Prompt module and SPS benchmark suite designed.
- **Unexplored areas**:
  - None within Explorer 3 scope.

## Key Decisions Made
- Unified static 109-dimensional observation vector adopted to support both Classic and Remastered mechanics without changing network dimensions.
- `PureJaxRLAdapterWrapper` specified to convert Gymnax 0.0.9 6-tuple return into 5-tuple for PureJaxRL.
- `jax.lax.scan` selected as primary rollout engine due to proven ~3M SPS performance on Ryzen 5600X CPU.

## Artifact Index
- `rl_wrappers_bench_spec.md` — Comprehensive specification for RL wrappers, rollout runner, Stoix, Flashbax, Persona Prompt, and SPS Benchmark.
- `handoff.md` — 5-component handoff report.
- `progress.md` — Liveness heartbeat.
- `DISPATCH.md` — Dispatch message logs.
