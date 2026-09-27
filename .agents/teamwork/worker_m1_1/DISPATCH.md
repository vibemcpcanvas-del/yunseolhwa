## 2026-09-26T15:42:38Z
You are the Milestone 1 Worker (teamwork_preview_worker).
Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\worker_m1_1\
Project Root: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
Original Request: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\ORIGINAL_REQUEST.md
Project Spec: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\PROJECT.md
Survey 1 Report: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_1\wz_spec_report.md
Survey 1 Handoff: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.agents\teamwork\explorer_survey_1\handoff.md

You MUST read ORIGINAL_REQUEST.md, PROJECT.md, and the survey report first.
Your exclusive write ownership:
- pyproject.toml
- src/maple_gymnax/__init__.py
- src/maple_gymnax/parser/__init__.py
- src/maple_gymnax/parser/schema.py
- src/maple_gymnax/parser/wz_parser.py
- tests/conftest.py
- tests/test_wz_parser.py

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope of Milestone 1:
1. Setup Python 3.12 environment using uv (uv venv --python 3.12 .venv) and install dependencies: jax, flax, gymnax, flashbax, pytest, chex, optax. Create pyproject.toml with project metadata and dependencies.
2. Implement src/maple_gymnax/parser/schema.py:
   - Define EnvParams dataclass (flax.struct.dataclass compatible and serializable) and JSON schema for Lotus Phase 1 (both classic and remastered fields).
   - Include validation functions against the Draft 2020-12 schema documented in survey report.
3. Implement src/maple_gymnax/parser/wz_parser.py:
   - Crawler for C:\mp and C:\mp\Restored_Data (BossPattern BossSuu.img.json and Map bossSuu.img.json).
   - Parse anchor offsets, frame delays (ms -> s conversion: delay / 1000.0 or 60Hz tick conversion), hitboxes, skill metadata.
   - Handle extraction failures / empty canvas nodes gracefully by synthesizing verified client physical defaults from survey (Canvas 1366x768, Core center 683, 384, floor y 605, player hitbox 40x60, laser omega 0.5235, etc.).
   - Provide CLI and function parse_wz_to_env_params(wz_dir: str = 'C:\\mp') -> EnvParams and save_env_params_json(...).
4. Implement comprehensive unit tests in tests/test_wz_parser.py:
   - Test schema validation, WZ parsing from C:\mp, frame delay conversion, hitbox extraction, fallback synthesis.
5. Run the unit tests using uv run pytest tests/test_wz_parser.py -v. Ensure all tests pass.
6. Write a complete handoff report in your working directory (handoff.md) with Observation, Logic Chain, Caveats, Conclusion, and Verification Method (including exact test commands and passing output).
7. Send message to parent when done.
