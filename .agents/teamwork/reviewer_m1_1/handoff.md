# Milestone 1 Review Handoff Report: WZ Parser Pipeline & Env Setup

**Agent ID**: `reviewer_m1_1` (teamwork_preview_reviewer)  
**Parent Agent ID**: `d30c9047-58e0-49d2-8a5c-3a4baedf9ed6`  
**Date**: 2026-09-26T16:05:00Z  
**Target Milestone**: M1 (WZ Parser Pipeline & Env Setup)  
**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

1. **Test Suite Execution**:
   - Command: `uv run pytest tests/test_wz_parser.py -v`
   - Output verbatim:
     ```
     ============================= test session starts =============================
     platform win32 -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.venv\Scripts\python.exe
     cachedir: .pytest_cache
     rootdir: C:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
     configfile: pyproject.toml
     collecting ... collected 22 items

     tests/test_wz_parser.py::TestSchemaValidation::test_default_env_params_valid PASSED [  4%]
     tests/test_wz_parser.py::TestSchemaValidation::test_schema_rejects_missing_required_field PASSED [  9%]
     tests/test_wz_parser.py::TestSchemaValidation::test_schema_rejects_invalid_map_dimensions PASSED [ 13%]
     tests/test_wz_parser.py::TestSchemaValidation::test_schema_rejects_invalid_difficulty_enum PASSED [ 18%]
     tests/test_wz_parser.py::TestSchemaValidation::test_schema_rejects_invalid_beam_count PASSED [ 22%]
     tests/test_wz_parser.py::TestFlaxDataclassCompatibility::test_jax_tree_flatten_unflatten PASSED [ 27%]
     tests/test_wz_parser.py::TestFlaxDataclassCompatibility::test_env_params_property_accessors PASSED [ 31%]
     tests/test_wz_parser.py::TestSerializationRoundtrip::test_dict_roundtrip PASSED [ 36%]
     tests/test_wz_parser.py::TestSerializationRoundtrip::test_json_roundtrip PASSED [ 40%]
     tests/test_wz_parser.py::TestSerializationRoundtrip::test_file_save_and_load PASSED [ 45%]
     tests/test_wz_parser.py::TestUnitConversions::test_ms_to_seconds PASSED  [ 50%]
     tests/test_wz_parser.py::TestUnitConversions::test_ms_to_ticks PASSED    [ 54%]
     tests/test_wz_parser.py::TestUnitConversions::test_ticks_to_seconds PASSED [ 59%]
     tests/test_wz_parser.py::TestUnitConversions::test_seconds_to_ticks PASSED [ 63%]
     tests/test_wz_parser.py::TestWZRestorationAndAtlas::test_restore_node PASSED [ 68%]
     tests/test_wz_parser.py::TestWZRestorationAndAtlas::test_parse_spine_atlas PASSED [ 72%]
     tests/test_wz_parser.py::TestWZParserWithRealData::test_discover_files PASSED [ 77%]
     tests/test_wz_parser.py::TestWZParserWithRealData::test_pattern_and_atlas_extraction PASSED [ 81%]
     tests/test_wz_parser.py::TestWZParserWithRealData::test_extract_env_params_synthesis PASSED [ 86%]
     tests/test_wz_parser.py::TestWZParserWithRealData::test_difficulty_rate_variations PASSED [ 90%]
     tests/test_wz_parser.py::TestWZParserWithRealData::test_custom_overrides PASSED [ 95%]
     tests/test_wz_parser.py::TestFallbackGracefulDegradation::test_missing_directory_fallback PASSED [100%]

     ============================= 22 passed in 0.52s ==============================
     ```

2. **JAX JIT Tracing Failure on Convenience Properties**:
   - Location: `src/maple_gymnax/parser/schema.py`, lines 276-346:
     ```python
     @property
     def core_pos(self) -> Tuple[float, float]:
         return (float(self.core_x), float(self.core_y))

     @property
     def laser_omega(self) -> float:
         return float(self.classic_laser.angular_velocity)

     @property
     def laser_thickness(self) -> float:
         return float(self.classic_laser.beam_thickness)

     @property
     def jump_impulse(self) -> float:
         return abs(float(self.player_jump_impulse))

     @property
     def max_debris(self) -> int:
         return int(self.debris_params.max_debris)

     @property
     def gauge_gain_rate(self) -> float:
         diff = self.remastered_params.difficulty
         return float(getattr(self.remastered_params.gauge_natural_rates, diff, 0.008))
     ```
   - Test command:
     ```bash
     uv run python -c "import jax; from maple_gymnax.parser import EnvParams; p = EnvParams(); fn = jax.jit(lambda params: params.laser_omega * 2); fn(p)"
     ```
   - Verbatim error output:
     ```
     jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]
     The problem arose with the `float` function. If trying to convert the data type of a value, try using `x.astype(float)` or `jnp.array(x, float)` instead.
     The error occurred while tracing the function <lambda> for jit. This concrete value was not available in Python because it depends on the value of the argument params.classic_laser.angular_velocity.
     ```
   - Result: All convenience property accessors (`core_pos`, `laser_omega`, `laser_thickness`, `jump_impulse`, `max_debris`, `gauge_gain_rate`, `screen_width`, `screen_height`, `player_w`, `player_h`, `jump_velocity`, `laser_half_thickness`, `laser_damage`, `overload_duration`) fail unconditionally when invoked within `@jax.jit` functions.

3. **Dynamic Tracer Failure on Static Array Allocation (`max_debris`)**:
   - Location: `src/maple_gymnax/parser/schema.py`, line 203:
     ```python
     @struct.dataclass
     class DebrisParams:
         max_debris: int = 30
         spawn_interval_ticks: int = 15
         types: Tuple[DebrisTypeParams, ...] = struct.field(default_factory=_default_debris_types)
     ```
   - Contract requirement (`PROJECT.md` line 134):
     `max_debris: int = 30 (static compile-time constant)`
   - Test command:
     ```bash
     uv run python -c "import jax, jax.numpy as jnp; from maple_gymnax.parser import EnvParams; p = EnvParams(); fn = jax.jit(lambda params: jnp.zeros((params.max_debris,))); fn(p)"
     ```
   - Verbatim error output:
     ```
     ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[]
     ```
     Even without the `int(...)` cast, JAX raises:
     ```
     TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int32[]),).
     ```
     Because `max_debris` is not defined with `struct.field(pytree_node=False, default=30)`.

4. **Real Asset Extraction & CLI Verification**:
   - Paths `C:\mp\Restored_Data\Mob\BossPattern\_Canvas\_Canvas_012\BossSuu.img.json` and `C:\mp\Restored_Data\Map\Back\Back_000\bossSuu.img.json` exist and were successfully discovered by `WZParser.discover_files()`.
   - Discovered 10 patterns (`1000` through `1009`), UI sections `default`, `destruction`, `overload`, and 124 Spine texture atlas regions.
   - CLI command execution:
     ```bash
     uv run python -m maple_gymnax.parser.wz_parser --wz-dir C:\mp --output env_params_cli.json
     ```
     Output included a benign `RuntimeWarning`:
     `<frozen runpy>:128: RuntimeWarning: 'maple_gymnax.parser.wz_parser' found in sys.modules after import of package 'maple_gymnax.parser', but prior to execution of 'maple_gymnax.parser.wz_parser'`.

5. **Integrity Violation Assessment**:
   - Checked for hardcoded test results, facade logic, bypasses, and fabricated outputs.
   - Result: **NO INTEGRITY VIOLATION DETECTED**. The implementation contains genuine recursive WZ node parsing, Draft 2020-12 schema validation, and real client file crawling. The failures observed are genuine software defects in JAX tracer handling and static field annotations.

---

## 2. Logic Chain

1. **Step 1 (Interface Contract Inspection)**:
   `PROJECT.md` Section 1 specifies the interface contract between `wz_parser` / `schema.py` and downstream Milestone 2 `lotus_phase1.py` (`LotusPhase1Env`). In Gymnax, `step_env` and `reset_env` are executed under `@jax.jit`.
2. **Step 2 (Tracing Analysis)**:
   Observation 2 directly proves that calling Python's built-in `float(...)` or `int(...)` inside property accessors on `EnvParams` raises `ConcretizationTypeError` as soon as JAX traces `EnvParams`. Any simulation code in Milestone 2 referencing `params.laser_omega`, `params.core_pos`, or `params.jump_impulse` will crash during XLA compilation.
3. **Step 3 (Static Constant Analysis)**:
   Observation 3 proves that `max_debris` is treated as a dynamic array leaf by Flax PyTree rather than a static compile-time constant. Because R1 requires a `Static Padded Array` of 30 falling debris hazards, allocating or slicing this array using `params.max_debris` inside JIT crashes with `TypeError`.
4. **Step 4 (Test Gap Analysis)**:
   Observation 1 and `tests/test_wz_parser.py` show that while 22 unit tests exist, none of them execute `jax.jit` functions that read `EnvParams` properties or allocate static shapes. This gap masked the defect during the worker's internal verification.
5. **Step 5 (Verdict Deduction)**:
   Because these defects will immediately block Milestone 2 implementation and fail Acceptance Criteria R1 ("ConcretizationTypeError 원천 방지"), the changes must be requested before Milestone 1 is approved.

---

## 3. Caveats

1. **No Integrity Violations**: The code was constructed in good faith with substantial depth and genuine logic.
2. **JAX Host Execution**: Outside of JIT compilation (e.g. In standard Python scripts or non-JIT testing), `EnvParams` and `WZParser` behave as expected. The defects specifically emerge under JAX tracing/compilation.

---

## 4. Conclusion

**Verdict: REQUEST_CHANGES**

### Findings Summary

| # | Severity | Category | Description | Fix Required |
|---|---|---|---|---|
| 1 | **Critical** | Correctness / JAX | `EnvParams` property getters call `float()`, `int()`, `abs(float())` causing `ConcretizationTypeError` under `@jax.jit` | Remove `float()`/`int()` casts; return raw attributes (e.g. `self.classic_laser.angular_velocity`, `-self.player_jump_impulse`) |
| 2 | **Major** | Conformance / JAX | `max_debris`, `spawn_interval_ticks`, and `beam_count` are dynamic PyTree leaves, breaking static shape allocation | Mark compile-time constants with `struct.field(pytree_node=False, default=...)` |
| 3 | **Major** | Quality / Testing | Test suite lacks JIT verification tests | Add test case running a `@jax.jit` compiled function accessing all `EnvParams` properties and static shapes |
| 4 | **Minor** | Hygiene | RuntimeWarning on CLI execution via `-m maple_gymnax.parser.wz_parser` | Clean up import chain between `__init__.py` and `wz_parser.py` |
| 5 | **Minor** | Schema | Missing `minimum: 0` on `dt` and `gravity` in JSON Schema | Add positive number constraints to `dt` and `gravity` in `LOTUS_PHASE1_SCHEMA` |

---

## 5. Verification Method

To verify resolution of these findings:

1. **JIT Property & Shape Test**:
   ```bash
   uv run python -c "import jax, jax.numpy as jnp; from maple_gymnax.parser import EnvParams; p = EnvParams(); fn = jax.jit(lambda params: (params.core_pos[0] + params.laser_omega + params.jump_impulse + params.gauge_gain_rate, jnp.zeros((params.max_debris,)))); res = fn(p); print('JIT Verification Passed:', res[0], res[1].shape)"
   ```
   *Expected Output*: `JIT Verification Passed: <numeric_value> (30,)` with exit code 0 and zero `ConcretizationTypeError`.

2. **Existing Unit Test Suite**:
   ```bash
   uv run pytest tests/test_wz_parser.py -v
   ```
   *Expected Output*: All tests pass (including new JIT verification tests).
