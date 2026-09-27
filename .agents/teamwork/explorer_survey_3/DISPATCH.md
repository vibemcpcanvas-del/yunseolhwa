## 2026-09-26T15:27:20Z

You are Survey Explorer 3 (teamwork_preview_explorer).
Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_3\
Original Request: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
You MUST read ORIGINAL_REQUEST.md first.
Your objective is to investigate:
1. Workspace python environment: Check installed python / uv / jax / flax / gymnax versions and dependencies in bold-faraday. Check if uv venv --python 3.12 exists or needs creation.
2. RL Framework Wrappers:
   - FlattenObservationWrapper for PureJaxRL compatibility (flattening player coordinates, hp, laser angle, debris positions/masks into 1D float array).
   - Rollout runner using jax.vmap and jax.lax.scan for high-speed batched episode collection.
   - Stoix/Stoa adapter interface (AutoResetWrapper, episode metric logger).
   - Flashbax Pytree replay buffer zero-copy interface design.
3. Persona System Prompt requirements (src/maple_gymnax/prompts/persona_system_prompt.py): structure for prompting LLMs to transform unstructured client data into physical tensors and Gymnax code.
4. SPS Benchmark design (benchmarks/benchmark_sps.py): measuring steps per second across batch sizes (256, 512, 1024, 2048, 4096) on Ryzen 5600X CPU and RX 6600 XT GPU (WSL2 ROCmLab).
5. Output requirements: Write your comprehensive findings to c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_3\rl_wrappers_bench_spec.md and handoff.md.
6. Send a message to parent when done.

## 2026-09-26T15:36:21Z

**Context**: ORIGINAL_REQUEST.md updated with Critical Domain Knowledge (MapleStory Lotus Remaster April 2024).
**Content**: MapleStory Lotus was officially remastered. Patterns 1001-1009 and destruction/overload in C:\mp\Restored_Data\Mob\BossPattern\BossSuu.img.json correspond to this remaster:
1. Security & Annihilation Gauge (natural increase, 100% -> 25s Overload/Destruction mode with horizontal bombardment 1006-000 & electric field 1006-002).
2. Friendly fire / boss guidance: Tracking laser (1001-000) & small arm slam (1001-001) hitting player raises gauge; hitting Lotus lowers gauge & breaks shield.
3. Floor electric discharge (jump avoidance).
4. Shield generation.
Modular env design must support both classic (rotating cross laser + falling debris) and remastered (gauge + friendly fire + overload).
Please re-read ORIGINAL_REQUEST.md lines 54-78.
**Action**: Incorporate remastered state observation dimensions in FlattenObservationWrapper and ensure persona prompts cover both classic and remastered mechanics.
