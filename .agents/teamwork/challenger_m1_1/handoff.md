# Milestone 1 Adversarial Challenger Handoff Report

**Agent ID**: `challenger_m1_1` (teamwork_preview_challenger)  
**Parent Agent ID**: `d30c9047-58e0-49d2-8a5c-3a4baedf9ed6`  
**Date**: 2026-09-26T16:05:00Z  
**Target Milestone**: M1 (WZ Parser Pipeline & Env Setup)  
**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

### 1.1 Test Suite Execution
An adversarial test suite comprising 41 test cases was created in `tests/test_adversarial_m1.py` targeting three primary dimensions:
1. Malformed JSON inputs, missing required fields, boundary values (NaN/Inf, extreme canvas, negative frame delay/dt).
2. WZ crawler robustness (non-existent directories, corrupted JSONs, empty 0-byte files, non-dict roots, malformed Spine atlas).
3. JAX PyTree serialization edge cases (tuple vs list, int vs float conversions, tracer concretization under `@jax.jit` and `jax.vmap`).

Execution command:
```bash
uv run pytest tests/test_wz_parser.py tests/test_adversarial_m1.py -v
```
Output verbatim:
```
============================= test session starts =============================
platform win32 -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\ROCmAdmin\Documents\antigravity\bold-faraday\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
configfile: pyproject.toml
collecting ... collected 63 items

tests/test_wz_parser.py::TestSchemaValidation::test_default_env_params_valid PASSED [  1%]
tests/test_wz_parser.py::TestSchemaValidation::test_schema_rejects_missing_required_field PASSED [  3%]
...
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_missing_every_required_top_level_field PASSED [ 36%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[] PASSED [ 38%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[{] PASSED [ 39%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[{'invalid': 1}] PASSED [ 41%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[null] PASSED [ 42%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[123] PASSED [ 44%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json["just_a_string"] PASSED [ 46%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[[1, 2, 3]] PASSED [ 47%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_malformed_json_strings_in_from_json[{"map_width": 1366,}] PASSED [ 49%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_none_subsections[classic_laser] PASSED [ 50%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_none_subsections[debris_params] PASSED [ 52%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_none_subsections[remastered_params] PASSED [ 53%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[bad_core_pos0] PASSED [ 55%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[bad_core_pos1] PASSED [ 57%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[bad_core_pos2] PASSED [ 58%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[bad_core_pos3] PASSED [ 60%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[None] PASSED [ 61%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[123] PASSED [ 63%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[6] PASSED [ 65%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_core_pos[abc] PASSED [ 66%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_string_core_pos_silent_corruption PASSED [ 68%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_from_dict_with_malformed_debris_items PASSED [ 69%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_nan_values_bypass_schema_validation PASSED [ 71%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_inf_values_bypass_schema_validation PASSED [ 73%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_unconstrained_negative_fields_in_schema PASSED [ 74%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_inverted_wall_boundaries_pass_schema PASSED [ 76%]
tests/test_adversarial_m1.py::TestAdversarialSchemaAndInputs::test_conversion_functions_nan_inf_behavior PASSED [ 77%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_crawler_on_nonexistent_directory PASSED [ 79%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_crawler_on_empty_wz_files PASSED [ 80%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_crawler_on_corrupted_json_syntax PASSED [ 82%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_crawler_on_non_dict_json_root PASSED [ 84%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_crawler_when_wz_dir_is_a_file PASSED [ 85%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_parse_spine_atlas_adversarial_inputs PASSED [ 87%]
tests/test_adversarial_m1.py::TestAdversarialWZCrawler::test_restore_node_adversarial_inputs PASSED [ 88%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_pytree_treedef_mismatch_tuple_vs_list PASSED [ 90%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_critical_concretization_type_error_on_all_property_accessors PASSED [ 92%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_critical_static_array_dimension_concretization_failure PASSED [ 93%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_to_dict_crashes_under_jax_jit PASSED [ 95%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_direct_field_access_succeeds_under_jax_jit PASSED [ 96%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_vmap_with_env_params_under_jit PASSED [ 98%]
tests/test_adversarial_m1.py::TestAdversarialJAXPyTreeAndJIT::test_tree_map_preserves_pytree_structure PASSED [100%]

============================= 63 passed in 2.08s ==============================
```

### 1.2 Specific Critical Findings Directly Observed

#### Observation A: ConcretizationTypeError on All 14 Convenience Property Accessors inside `@jax.jit`
In `src/maple_gymnax/parser/schema.py`, lines 276-346, the convenience property accessors (`core_pos`, `laser_omega`, `laser_thickness`, `jump_impulse`, `max_debris`, `gauge_gain_rate`, `screen_width`, `screen_height`, `player_w`, `player_h`, `jump_velocity`, `laser_half_thickness`, `laser_damage`, `overload_duration`) wrap their values with Python native `float(...)` and `int(...)` casts:
```python
284:    return float(self.classic_laser.angular_velocity)
279:    return (float(self.core_x), float(self.core_y))
294:    return abs(float(self.player_jump_impulse))
299:    return int(self.debris_params.max_debris)
```
When `EnvParams` is passed as an argument to any `@jax.jit`-compiled function (e.g. Gymnax `env.step(key, state, action, params)` or `env.reset(key, params)`), JAX converts PyTree leaves into abstract tracer arrays. Executing `@jax.jit(lambda params: params.laser_omega)(params)` produces verbatim:
```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]
The problem arose with the `float` function. If trying to convert the data type of a value, try using `x.astype(float)` or `jnp.array(x, float)` instead.
The error occurred while tracing the function <lambda> for jit. This concrete value was not available in Python because it depends on the value of the argument params.classic_laser.angular_velocity.
```
Empirical test `test_critical_concretization_type_error_on_all_property_accessors` confirms that **14 out of 14 (100%)** property accessors raise this fatal error when accessed under `@jax.jit`.

#### Observation B: `max_debris` Traced as Dynamic Leaf, Breaking Static Padded Array Allocation
In `src/maple_gymnax/parser/schema.py`, line 203:
```python
@struct.dataclass
class DebrisParams:
    max_debris: int = 30
    spawn_interval_ticks: int = 15
    types: Tuple[DebrisTypeParams, ...] = struct.field(default_factory=_default_debris_types)
```
`max_debris` is declared as a dynamic PyTree leaf without `struct.field(pytree_node=False)`.
When attempting to allocate static debris buffers inside `@jax.jit` (`jnp.zeros((params.debris_params.max_debris, 4))`), JAX crashes verbatim with:
```
TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int32[]), 4).
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
```
This violates `PROJECT.md` line 134: `max_debris: int = 30 (static compile-time constant)`.

#### Observation C: Silent String Indexing / Coordinate Corruption in `from_dict`
In `src/maple_gymnax/parser/schema.py`, lines 402-404:
```python
if "core_pos" in data:
    pos = data["core_pos"]
    core_x, core_y = float(pos[0]), float(pos[1])
```
If a user or caller passes `"core_pos": "683,384"` as a string, `pos[0]` evaluates to `'6'` and `pos[1]` evaluates to `'8'`. `from_dict` silently sets `core_x = 6.0` and `core_y = 8.0` without raising any error, displacing the boss core from the center `(683, 384)` to the top-left boundary.

#### Observation D: Schema Lacks NaN / Inf and Lower Bound Guards
In `LOTUS_PHASE1_SCHEMA`:
- `float('nan')` and `float('inf')` pass validation because in Python `jsonschema`, comparison operations with NaN (`nan < minimum`) evaluate to `False`. Serializing parameters containing `NaN` produces non-standard RFC 8259 JSON (`"map_width": NaN`).
- `dt`, `player_width`, `player_height`, `player_speed`, `classic_laser.beam_thickness`, `debris_params.max_debris`, and `spawn_interval_ticks` have no `"minimum": 0.0` or `"exclusiveMinimum": 0.0`. Negative `dt` (`-0.0166667`) and zero `dt` pass validation; `dt=0.0` triggers `ZeroDivisionError` in `ms_to_ticks`.

---

## 2. Logic Chain

1. **Premise 1 (Downstream Gymnax Contract)**:
   In `PROJECT.md` Section 1 and `ORIGINAL_REQUEST.md` R1, Gymnax environment execution requires pure functional compilation under `@jax.jit(env.step)` and `jax.vmap`. The defined interface contract relies on `EnvParams` exposing `core_pos`, `laser_omega`, `laser_thickness`, `player_w`, `max_debris`, etc.
2. **Premise 2 (Empirical Reproduction of Observation A & B)**:
   Observations A and B empirically prove that:
   - Accessing any of the 14 convenience property accessors inside `@jax.jit` immediately fails with `jax.errors.ConcretizationTypeError`.
   - Allocating static buffers for falling debris inside `@jax.jit` using `params.debris_params.max_debris` immediately fails with `TypeError: Shapes must be 1D sequences of concrete values`.
3. **Premise 3 (Direct Blocker for Milestone 2)**:
   Milestone 2 (`LotusPhase1Env`) cannot implement JIT-compiled state transitions or static padded array buffers without encountering these exact runtime errors if it relies on the current `schema.py`.
4. **Premise 4 (Coordinate Corruption in Premise C)**:
   Observation C proves that `from_dict` performs unsafely typed indexing on `core_pos`, which silently corrupts spatial parameters when string inputs are supplied.
5. **Deductive Conclusion**:
   Therefore, Milestone 1 cannot be approved as-is. Changes must be requested to harden `schema.py` before downstream Milestone 2 can safely proceed.

---

## 3. Caveats

1. **Crawler Robustness Verified**:
   Adversarial testing confirmed that `WZParser` exhibits outstanding resilience when crawling non-existent directories, 0-byte empty JSON files, corrupted syntax, and non-dict roots. In all such cases, it logs clear warnings and cleanly falls back to canonical physical defaults.
2. **Direct Field Access Works**:
   When bypassing the property accessors and accessing fields directly (e.g., `params.classic_laser.angular_velocity` instead of `params.laser_omega`, and `params.player_width` instead of `params.player_w`), arithmetic tracing under `@jax.jit` succeeds. The defect is confined to the property accessor implementations and dataclass field declarations in `schema.py`.

---

## 4. Conclusion

**Verdict: REQUEST_CHANGES**

The worker must apply the following specific fixes to `src/maple_gymnax/parser/schema.py`:

1. **Fix Property Accessors (Lines 276-346)**:
   Remove all Python native `float(...)` and `int(...)` casts from property accessors so they return the raw JAX-traced leaf values. For `jump_impulse`, return `-self.player_jump_impulse` or `jnp.abs(self.player_jump_impulse)` instead of `abs(float(...))`. For `core_pos`, return `(self.core_x, self.core_y)`.
2. **Mark Compile-Time Constants as Static Metadata (Line 203)**:
   In `DebrisParams`, declare `max_debris: int = struct.field(pytree_node=False, default=30)`. Consider doing the same for `beam_count: int = struct.field(pytree_node=False, default=4)` in `ClassicLaserParams`.
3. **Guard `core_pos` in `from_dict` (Line 402)**:
   Validate that `isinstance(pos, (list, tuple)) and len(pos) >= 2` before indexing, preventing silent character slicing when strings are passed.
4. **Harden `LOTUS_PHASE1_SCHEMA` and `validate_env_params_dict`**:
   Add `"exclusiveMinimum": 0.0` for `dt`, and `"minimum": 0.0` for `player_width`, `player_height`, `player_speed`, `beam_thickness`, `max_debris`, and `spawn_interval_ticks`. Add explicit rejection of `math.isnan()` / `math.isinf()` in `validate_env_params_dict`.
5. **Handle `None` Subsections in `from_dict` (Lines 413, 426, 440)**:
   Use `data.get("classic_laser") or {}` instead of `data.get("classic_laser", {})` to avoid `AttributeError` when explicit `None` is provided.

---

## 5. Verification Method

To independently verify these findings and reproduce the test results:

1. **Run the Full Suite (63 tests)**:
   ```bash
   uv run pytest tests/test_wz_parser.py tests/test_adversarial_m1.py -v
   ```
   *Pass Condition*: All 63 tests pass with exit code 0.

2. **Verify Property Concretization Failure**:
   ```bash
   uv run python -c "import jax; from maple_gymnax.parser.schema import EnvParams; p = EnvParams(); jax.jit(lambda params: params.laser_omega)(p)"
   ```
   *Expected Error*: `jax.errors.ConcretizationTypeError`

3. **Verify Static Array Allocation Failure with `max_debris`**:
   ```bash
   uv run python -c "import jax, jax.numpy as jnp; from maple_gymnax.parser.schema import EnvParams; p = EnvParams(); jax.jit(lambda params: jnp.zeros((params.debris_params.max_debris, 4)))(p)"
   ```
   *Expected Error*: `TypeError: Shapes must be 1D sequences of concrete values`
