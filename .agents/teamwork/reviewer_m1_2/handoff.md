# Milestone 1 Review Handoff Report: WZ Parser Pipeline & Env Setup

**Agent ID**: `reviewer_m1_2` (teamwork_preview_reviewer / critic)  
**Parent Agent ID**: `d30c9047-58e0-49d2-8a5c-3a4baedf9ed6`  
**Date**: 2026-09-26T16:05:00Z  
**Target Milestone**: M1 (WZ Parser Pipeline & Env Setup)  
**Verdict**: **REQUEST_CHANGES**

---

## Review Summary

**Verdict**: **REQUEST_CHANGES**

The Milestone 1 implementation in `src/maple_gymnax/parser/` successfully establishes a working Python 3.12 environment, valid Draft 2020-12 schema validation, authentic WZ client file crawling against `C:\mp`, and JSON roundtrip serialization.

However, an adversarial inspection of JAX / XLA functional compliance revealed **two blocking technical defects** that directly violate Requirement R1 ("`ConcretizationTypeError` 원천 방지"):
1. **Critical Finding**: All 14 convenience property accessors on `EnvParams` (`core_pos`, `laser_omega`, `laser_thickness`, `jump_impulse`, `max_debris`, `gauge_gain_rate`, etc.) execute Python `float(...)` and `int(...)` casts on PyTree fields. When `EnvParams` is passed into any `@jax.jit` compiled function (such as Gymnax's `step_env` or `reset_env`), JAX transforms these fields into Tracer objects (`Traced<ShapedArray(...)>`), immediately raising `jax.errors.ConcretizationTypeError`.
2. **Major Finding**: `max_debris` in `DebrisParams` is defined as a dynamic PyTree leaf (`max_debris: int = 30`) instead of a static field (`struct.field(pytree_node=False, default=30)`). In JAX JIT, shape dimensions must be concrete integers. Attempting to allocate static padded arrays of debris (e.g. `jnp.zeros((params.max_debris, 2))`) fails compilation.

These defects must be resolved before Milestone 2 begins to avoid blocking environment simulation.

---

## 1. Observation

### 1.1 Test Suite Verification
Executed the project unit tests in the uv Python 3.12 virtual environment:
```bash
uv run pytest tests/test_wz_parser.py -v
```
Verbatim test output:
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

============================= 22 passed in 0.47s ==============================
```

### 1.2 `ConcretizationTypeError` Under `jax.jit`
Tested property accessors of `EnvParams` inside `@jax.jit`:
```bash
uv run python -c "import jax; from maple_gymnax.parser import EnvParams; p = EnvParams(); fn = jax.jit(lambda params: params.laser_omega * 2); fn(p)"
```
Verbatim error output:
```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]
The problem arose with the `float` function. If trying to convert the data type of a value, try using `x.astype(float)` or `jnp.array(x, float)` instead.
The error occurred while tracing the function <lambda> for jit. This concrete value was not available in Python because it depends on the value of the argument params.classic_laser.angular_velocity.

See https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError
```
Tested comparing `params.dt` (direct field) vs `params.laser_omega` (property accessor):
```bash
uv run python -c "import jax; from maple_gymnax.parser import parse_wz_to_env_params; p = parse_wz_to_env_params();
@jax.jit
def f1(p): return p.dt
print('p.dt passed:', f1(p))

@jax.jit
def f2(p): return p.laser_omega
try:
    print('p.laser_omega passed:', f2(p))
except Exception as e:
    print('p.laser_omega FAILED:', type(e).__name__)
"
```
Verbatim result:
```
p.dt passed: 0.016666668
p.laser_omega FAILED: ConcretizationTypeError
```
Exact code location: `src/maple_gymnax/parser/schema.py`, lines 276-346:
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

    @property
    def screen_width(self) -> float:
        return float(self.map_width)

    @property
    def screen_height(self) -> float:
        return float(self.map_height)

    @property
    def player_w(self) -> float:
        return float(self.player_width)

    @property
    def player_h(self) -> float:
        return float(self.player_height)

    @property
    def jump_velocity(self) -> float:
        return float(self.player_jump_impulse)

    @property
    def laser_half_thickness(self) -> float:
        return float(self.classic_laser.beam_thickness) / 2.0

    @property
    def laser_damage(self) -> float:
        return float(self.classic_laser.damage)

    @property
    def overload_duration(self) -> float:
        return float(self.remastered_params.overload_duration)
```

### 1.3 Static Array Allocation Failure on `max_debris`
Tested using `params.max_debris` for static array dimension inside `@jax.jit`:
```bash
uv run python -c "import jax, jax.numpy as jnp; from maple_gymnax.parser import EnvParams; p = EnvParams();
@jax.jit
def test_static_shape(params):
    arr = jnp.zeros((params.max_debris, 2))
    return arr

test_static_shape(p)
"
```
Verbatim error output:
```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[]
The problem arose with the `int` function. If trying to convert the data type of a value, try using `x.astype(int)` or `jnp.array(x, int)` instead.
The error occurred while tracing the function test_static_shape at <string>:2 for jit. This concrete value was not available in Python because it depends on the value of the argument params.debris_params.max_debris.
```
Exact code location: `src/maple_gymnax/parser/schema.py`, lines 201-206:
```python
@struct.dataclass
class DebrisParams:
    """Parameters for Falling Debris Barrage."""
    max_debris: int = 30
    spawn_interval_ticks: int = 15
    types: Tuple[DebrisTypeParams, ...] = struct.field(default_factory=_default_debris_types)
```
In contrast, when `max_debris` is marked `max_debris: int = struct.field(pytree_node=False, default=30)` and `self.debris_params.max_debris` is returned directly without `int()`, `jnp.zeros((params.max_debris, 2))` succeeds with shape `(30, 2)`.

### 1.4 Real Asset Verification Against `C:\mp`
Executed file discovery and extraction on the physical host filesystem:
- Path 1: `C:\mp\Restored_Data\Mob\BossPattern\_Canvas\_Canvas_012\BossSuu.img.json` (Size: 16,983 bytes). Discovered root patterns: `['1000', '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009']`. UI keys: `['default', 'destruction', 'overload']`.
- Path 2: `C:\mp\Restored_Data\Map\Back\Back_000\bossSuu.img.json` (Size: 43,343 bytes). Discovered Spine atlas `Swoo_Bossmap_Phase1.atlas`.
- Spine atlas parsed 124 sprite regions (`02_upper ligh10_1`, `03_blue_light01_1`, etc.).
- Output verified against MapleStory Lotus specifications:
  - Canvas: 1366.0 x 768.0
  - Core Center: (683.0, 384.0)
  - Floor Y: 605.0
  - Laser $\omega$: 0.523598775 rad/s ($\pi/6$)
  - Gauge natural gain rates: Normal (0.006), Hard (0.008), Extreme (0.020)
  - Overload duration: 25.0s

### 1.5 CLI Execution
Executed the CLI command:
```bash
uv run python -m maple_gymnax.parser.wz_parser --wz-dir C:\mp --output env_params_cli.json
```
Verbatim output:
```
<frozen runpy>:128: RuntimeWarning: 'maple_gymnax.parser.wz_parser' found in sys.modules after import of package 'maple_gymnax.parser', but prior to execution of 'maple_gymnax.parser.wz_parser'; this may result in unpredictable behaviour
[*] Crawling WZ assets in: C:\mp
[+] Discovered 10 patterns in BossSuu.img.json: ['1000', '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009']
[+] UI sections: ['default', 'destruction', 'overload']
[+] Spine atlas regions: 124
[+] Successfully exported validated EnvParams to: env_params_cli.json
```

### 1.6 Integrity Violation Audit
Audited codebase against adversarial integrity criteria:
- Hardcoded test results: None. The physical numbers match the MapleStory domain specifications established in `PROJECT.md` and `ORIGINAL_REQUEST.md`.
- Dummy or facade implementations: None. WZ node restoration, Spine atlas parsing, schema validation, and CLI routines implement real parsing logic.
- Bypassed core work: None. Real files in `C:\mp` are loaded and parsed.
- Fabricated verification outputs: None. Verified that reported test passes and command outputs are genuine.
- Self-certification: The test suite passed in eager mode, but missed JIT tracing on property accessors. This is an authentic technical oversight, not an integrity violation.

---

## 2. Logic Chain

1. **Premise 1 (JAX / Gymnax Execution Model)**:
   In Gymnax environments, `step_env` and `reset_env` are executed inside `@jax.jit`. Every field in an `EnvParams` PyTree passed to a JIT function is represented as an abstract JAX Tracer (`Traced<ShapedArray(...)>`).
2. **Premise 2 (Python Built-in Cast Failure)**:
   Observation 1.2 demonstrates that calling Python `float(tracer)` or `int(tracer)` unconditionally raises `jax.errors.ConcretizationTypeError`. Because all 14 convenience properties in `schema.py` wrap fields in `float(...)` and `int(...)`, invoking any convenience property (`params.laser_omega`, `params.core_pos`, `params.player_w`, etc.) inside JIT immediately crashes compilation.
3. **Premise 3 (Static Shape Requirement)**:
   Observation 1.3 demonstrates that `max_debris` is registered as a dynamic PyTree leaf rather than a static non-pytree field. Requirement R1 specifies a `Static Padded Array` of 30 falling debris hazards. Because JAX requires concrete integers for array shapes, allocating or slicing debris arrays using `params.max_debris` fails inside JIT.
4. **Premise 4 (Test Blind Spot)**:
   Observation 1.1 reveals that `tests/test_wz_parser.py` only tested `jax.tree_util.tree_flatten` and `tree_unflatten` in eager host Python. In eager mode, the leaves are concrete Python floats, so `float(...)` succeeds. The tests did not execute any `@jax.jit` function that reads convenience properties, leaving this critical defect undetected.
5. **Conclusion**:
   Downstream Milestone 2 (`lotus_phase1.py`) will immediately fail XLA JIT compilation if it references any `EnvParams` property accessors or uses `params.max_debris` for static array sizing. Therefore, Milestone 1 cannot be approved until these two issues are corrected.

---

## 3. Findings

### [Critical] Finding 1: `ConcretizationTypeError` on Property Accessors Under `jax.jit`
- **What**: Every property accessor on `EnvParams` fails with `ConcretizationTypeError` inside JIT.
- **Where**: `src/maple_gymnax/parser/schema.py`, lines 276-346 (`core_pos`, `laser_omega`, `laser_thickness`, `jump_impulse`, `max_debris`, `gauge_gain_rate`, `screen_width`, `screen_height`, `player_w`, `player_h`, `jump_velocity`, `laser_half_thickness`, `laser_damage`, `overload_duration`).
- **Why**: Wrapping PyTree fields in Python built-in `float(...)` or `int(...)` attempts to convert abstract JAX tracers to concrete Python types.
- **Suggestion**: Remove `float(...)` and `int(...)` casts from all property accessors. Return attributes directly:
  ```python
  @property
  def core_pos(self) -> Tuple[Any, Any]:
      return (self.core_x, self.core_y)

  @property
  def laser_omega(self) -> Any:
      return self.classic_laser.angular_velocity

  @property
  def laser_thickness(self) -> Any:
      return self.classic_laser.beam_thickness

  @property
  def jump_impulse(self) -> Any:
      return abs(self.player_jump_impulse)

  @property
  def gauge_gain_rate(self) -> Any:
      diff = self.remastered_params.difficulty
      return getattr(self.remastered_params.gauge_natural_rates, diff, 0.008)

  @property
  def screen_width(self) -> Any:
      return self.map_width

  @property
  def screen_height(self) -> Any:
      return self.map_height

  @property
  def player_w(self) -> Any:
      return self.player_width

  @property
  def player_h(self) -> Any:
      return self.player_height

  @property
  def jump_velocity(self) -> Any:
      return self.player_jump_impulse

  @property
  def laser_half_thickness(self) -> Any:
      return self.classic_laser.beam_thickness / 2.0

  @property
  def laser_damage(self) -> Any:
      return self.classic_laser.damage

  @property
  def overload_duration(self) -> Any:
      return self.remastered_params.overload_duration
  ```

### [Major] Finding 2: `max_debris` is Traced as Dynamic Leaf Instead of Static Field
- **What**: `max_debris` in `DebrisParams` is registered as a dynamic PyTree leaf rather than static metadata (`pytree_node=False`).
- **Where**: `src/maple_gymnax/parser/schema.py`, line 203.
- **Why**: R1 requires a `Static Padded Array` of 30 debris items. JAX array constructors (`jnp.zeros`, `jnp.empty`) require concrete shape dimensions. Dynamic tracers cannot be used as shape dimensions.
- **Suggestion**: Change `max_debris` definition in `DebrisParams` to:
  ```python
  max_debris: int = struct.field(pytree_node=False, default=30)
  ```
  And update the property accessor:
  ```python
  @property
  def max_debris(self) -> int:
      return self.debris_params.max_debris
  ```

### [Major] Finding 3: Missing JIT Property Tracing in Unit Test Suite
- **What**: `tests/test_wz_parser.py` lacks unit tests verifying that `EnvParams` property accessors work under `@jax.jit`.
- **Where**: `tests/test_wz_parser.py`, `TestFlaxDataclassCompatibility`.
- **Why**: Allows JIT regression to go undetected.
- **Suggestion**: Add a dedicated test `test_env_params_properties_inside_jit(default_env_params)` that evaluates all property accessors and static shape construction inside a `@jax.jit` function.

---

## 4. Challenge Report

### Challenge Summary
**Overall risk assessment**: **HIGH** (will block Milestone 2 simulation without code fix).

### Challenges

#### [Critical] Challenge 1: Functional RL Simulation Call Sites
- **Assumption challenged**: That passing `EnvParams` to JAX functions works simply because `tree_flatten`/`tree_unflatten` succeeds.
- **Attack scenario**: Downstream `LotusPhase1Env.step_env` accesses `params.laser_omega` to compute laser orientation or `params.core_pos` for collision offset under `jax.jit`.
- **Blast radius**: `ConcretizationTypeError` crashes simulation immediately on first `env.step()`.
- **Mitigation**: Remove `float()` casts in property accessors as described in Finding 1.

#### [High] Challenge 2: Debris Static Array Sizing
- **Assumption challenged**: That `params.max_debris` can be used to size arrays in `reset_env` or `step_env`.
- **Attack scenario**: `debris_y = jnp.zeros((params.max_debris,))` called inside JIT.
- **Blast radius**: `TypeError: Shapes must be 1D sequences of concrete values`.
- **Mitigation**: Mark `max_debris` with `pytree_node=False`.

---

## 5. Verified Claims

| Claim | Method | Result |
|---|---|---|
| Python 3.12 environment with `jax`, `flax`, `gymnax`, `flashbax` | `uv run python -c "import jax, flax, gymnax, flashbax"` | PASS |
| Draft 2020-12 Schema validation (`LOTUS_PHASE1_SCHEMA`) | Tested valid params & invalid inputs with `jsonschema.validate` | PASS |
| WZ Node Restoration (`restore_node`) | Unit tests with Canvas, SubProperty, UOL, Vector nodes | PASS |
| Spine Atlas parsing (`parse_spine_atlas`) | Parsed 124 regions from `Swoo_Bossmap_Phase1.atlas` | PASS |
| Discovery of real `C:\mp` assets | Verified paths, pattern IDs 1000..1009, UI keys | PASS |
| JSON Serialization roundtrip (`to_json`, `from_json`, file I/O) | `test_dict_roundtrip`, `test_json_roundtrip`, `test_file_save_and_load` | PASS |
| PyTree Flattening & Unflattening | `jax.tree_util.tree_flatten` & `tree_unflatten` | PASS |
| **Property accessors inside JIT** | Evaluated all 14 property accessors inside `@jax.jit` | **FAIL** (`ConcretizationTypeError`) |
| **Static shape allocation with `max_debris` inside JIT** | `jnp.zeros((params.max_debris, 2))` inside `@jax.jit` | **FAIL** (`ConcretizationTypeError`) |

---

## 6. Caveats

1. **Integrity Confirmation**: The worker implemented genuine parsing, schema validation, and file discovery. There is no evidence of cheating or facade code. The issues are genuine JAX tracer bugs.
2. **Eager Execution**: Outside of `@jax.jit`, the implementation runs without error. The defect is specific to JAX compilation and tracing.

---

## 7. Conclusion

**Verdict: REQUEST_CHANGES**

Milestone 1 satisfies file structure, schema validation, real WZ crawling, and serialization requirements. However, it fails JAX JIT execution due to `float(...)`/`int(...)` casts on PyTree fields and a non-static `max_debris` definition.

The following changes are required from the worker:
1. In `src/maple_gymnax/parser/schema.py`:
   - Remove `float(...)` and `int(...)` casts in all 14 property accessors of `EnvParams`.
   - In `DebrisParams`, define `max_debris: int = struct.field(pytree_node=False, default=30)`.
2. In `tests/test_wz_parser.py`:
   - Add a test function verifying that all property accessors and static array creation (`jnp.zeros((params.max_debris, 2))`) succeed inside a `@jax.jit` compiled function.

---

## 8. Verification Method

To verify the requested fix:

1. **Verify Property Accessors in JIT**:
   ```bash
   uv run python -c "import jax, jax.numpy as jnp; from maple_gymnax.parser import parse_wz_to_env_params; p = parse_wz_to_env_params();
   @jax.jit
   def test_jit(params):
       cx, cy = params.core_pos
       omega = params.laser_omega
       floor = params.floor_y
       gain = params.gauge_gain_rate
       arr = jnp.zeros((params.max_debris, 2))
       return cx + cy + omega + floor + gain + jnp.sum(arr)
   print('JIT result:', test_jit(p))"
   ```
   *Pass Condition*: Output `JIT result: ...` with zero exceptions.
   *Fail Condition*: Any `ConcretizationTypeError` or `TypeError`.

2. **Run Full Test Suite**:
   ```bash
   uv run pytest tests/test_wz_parser.py -v
   ```
   *Pass Condition*: All unit tests pass with exit code 0.
