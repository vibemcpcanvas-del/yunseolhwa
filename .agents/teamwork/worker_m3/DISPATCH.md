## 2026-09-29T11:42:29Z
You are worker_m3, a Worker subagent in a Teamwork hierarchy.
Your working directory is: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m3
Your parent is orchestrator_3 (conv ID: e8d54a3b-63f5-4fbd-92da-90a91af57a97).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY FIRST STEP:
Read the authoritative user request at:
c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
Specifically section dated 2026-09-29T11:28:08Z.

Also read:
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\PROJECT.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_2\survey_r3_tests.md
- c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\rules\jax-gymnax-rl.md

YOUR MISSION & WRITE SCOPE:
Implement Milestone 3 (Requirement R3): Transparent Real-Time Console Telemetry & Metric Overhaul.
You exclusively own these files:
- `train_ppo.py`
- `pyproject.toml`
- `tests/test_train_ppo.py`

Implementation tasks:
1. In `pyproject.toml`:
   - Add `pythonpath = [".", "src"]` under `[tool.pytest.ini_options]` so `uv run pytest` runs cleanly out of the box without specifying `-o pythonpath=...`.
2. In `train_ppo.py`:
   - Replace the 60-second binary `Survival: 0.0%` with real-time `Survival(s)` (average survival seconds = length / 60.0). Retain `survival_rate` in the metrics dict for watchdog compatibility.
   - Compute `DebrisHits/ep` via the renewal estimator:
     `total_debris_hits = jnp.sum(traj_batch.info["debris_hit"].astype(jnp.float32) * done_mask)`
     `debris_hits_per_ep = jnp.where(has_dones, total_debris_hits / jnp.maximum(num_dones, 1.0), (jnp.sum(traj_batch.info["debris_hit"].astype(jnp.float32)) / (config.num_steps * config.num_envs)) * mean_length)`
   - Compute `JumpRatio%` via:
     `is_jump = (traj_batch.action >= 4) & (traj_batch.action <= 6)` (or matching `common.py` jump actions).
     `jump_ratio = jnp.mean(is_jump.astype(jnp.float32))`
   - In `_log_callback`:
     Update format to include:
     `Length: {float(mean_length):6.1f} | Survival(s): {float(survival_sec):5.1f}s | DebrisHits/ep: {float(debris_hits_per_ep):4.1f} | JumpRatio%: {float(jump_ratio) * 100.0:4.1f}% | ...`
   - Ensure `--log_interval 20` is supported and configured as the default or recommended setting.
3. In `tests/test_train_ppo.py`:
   - Add or update tests asserting the presence and correct calculation of `survival_sec`, `debris_hits_per_ep`, and `jump_ratio`.
4. Run verification:
   - Run `uv run pytest tests/test_train_ppo.py`
   - Run `uv run python train_ppo.py --help`
   Ensure everything executes cleanly with zero syntax or compilation errors.

OUTPUT REQUIREMENTS:
- Update progress in `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m3\progress.md`.
- Document your changes and verification commands in `c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m3\handoff.md`.
- Send completion message to orchestrator_3 via `send_message`.
