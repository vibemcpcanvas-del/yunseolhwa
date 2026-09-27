"""Adversarial stress testing suite for Milestone 1 (schema.py and wz_parser.py).

This test module challenges the assumptions and robustness of:
1. Schema validation against malformed JSON, missing fields, boundary values (NaN, Inf, negative dt, extreme canvas).
2. WZ crawler robustness with non-existent directories, corrupted JSONs, 0-byte files, non-dict roots, malformed atlas.
3. JAX PyTree serialization edge cases (tuple vs list, int vs float conversions, tracer concretization under jax.jit/vmap).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
import jsonschema
import pytest

from maple_gymnax.parser.schema import (
    LOTUS_PHASE1_SCHEMA,
    ClassicLaserParams,
    DebrisParams,
    DebrisTypeParams,
    EnvParams,
    GaugeNaturalRates,
    RemasteredParams,
    _default_debris_types,
    validate_env_params,
    validate_env_params_dict,
)
from maple_gymnax.parser.wz_parser import (
    WZParser,
    load_env_params_json,
    ms_to_seconds,
    ms_to_ticks,
    parse_spine_atlas,
    parse_wz_to_env_params,
    restore_node,
    save_env_params_json,
    seconds_to_ticks,
    ticks_to_seconds,
)


# =============================================================================
# Category 1: Schema Validation, Malformed Inputs & Boundary Values
# =============================================================================

class TestAdversarialSchemaAndInputs:
    """Adversarial tests against schema validation and from_dict/from_json deserializers."""

    def test_missing_every_required_top_level_field(self, default_env_params: EnvParams):
        """Adversarially delete each required top-level field and verify validation rejects."""
        base_dict = default_env_params.to_dict()
        required_fields = LOTUS_PHASE1_SCHEMA["required"]
        assert len(required_fields) == 18

        for field in required_fields:
            corrupted = dict(base_dict)
            del corrupted[field]
            with pytest.raises(jsonschema.ValidationError, match=f"'{field}' is a required property"):
                validate_env_params_dict(corrupted)

    @pytest.mark.parametrize("bad_json", [
        "",                     # Empty string
        "{",                    # Truncated JSON
        "{'invalid': 1}",       # Single quotes
        "null",                 # JSON null
        "123",                  # Bare integer
        '"just_a_string"',      # Bare string
        "[1, 2, 3]",            # Bare list
        '{"map_width": 1366,}', # Trailing comma
    ])
    def test_malformed_json_strings_in_from_json(self, bad_json: str):
        """from_json and load_env_params_json must not succeed or silently accept non-dict JSON."""
        with pytest.raises((json.JSONDecodeError, AttributeError, jsonschema.ValidationError, TypeError)):
            EnvParams.from_json(bad_json)

    @pytest.mark.parametrize("none_key", ["classic_laser", "debris_params", "remastered_params"])
    def test_from_dict_with_none_subsections(self, default_env_params: EnvParams, none_key: str):
        """Stress-test from_dict when a subsection is present but explicitly None."""
        d = default_env_params.to_dict()
        d[none_key] = None

        # Schema must reject None for required object fields
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(d)

        # from_dict currently raises AttributeError when None is passed
        with pytest.raises(AttributeError):
            EnvParams.from_dict(d)

    @pytest.mark.parametrize("bad_core_pos", [
        [],                 # Empty list -> IndexError
        [683.0],            # 1 element instead of 2 -> IndexError
        (),                 # Empty tuple -> IndexError
        (683.0,),           # 1 element tuple -> IndexError
        None,               # None -> TypeError
        123,                # Integer -> TypeError
        "6",                # Single character string -> IndexError
        "abc",              # Non-numeric string -> ValueError
    ])
    def test_from_dict_with_malformed_core_pos(self, bad_core_pos):
        """Stress-test from_dict with malformed core_pos aliases."""
        data = {"core_pos": bad_core_pos}
        with pytest.raises((IndexError, TypeError, KeyError, ValueError)):
            EnvParams.from_dict(data)

    def test_from_dict_string_core_pos_silent_corruption(self):
        """FINDING / STRESS: Passing a comma-separated string to core_pos silently indexes characters!

        When core_pos="683,384", pos[0] is '6' and pos[1] is '8', silently setting core_x=6.0, core_y=8.0!
        """
        data = {"core_pos": "683,384"}
        corrupted = EnvParams.from_dict(data)
        # This proves the silent parsing corruption:
        assert corrupted.core_x == 6.0
        assert corrupted.core_y == 8.0
        assert corrupted.core_x != 683.0
        assert corrupted.core_y != 384.0

    def test_from_dict_with_malformed_debris_items(self):
        """Stress-test from_dict when debris types have missing required fields."""
        data = {
            "debris_params": {
                "max_debris": 30,
                "spawn_interval_ticks": 15,
                "types": [{"type_id": 0}],  # Missing radius, damage, stun_duration, fall_speed
            }
        }
        with pytest.raises(KeyError, match="radius"):
            EnvParams.from_dict(data)

    def test_nan_values_bypass_schema_validation(self, default_env_params: EnvParams):
        """FINDING / STRESS: NaN float values evaluate to false in numerical comparisons and bypass schema minimums."""
        d = default_env_params.to_dict()
        d["map_width"] = float("nan")
        d["dt"] = float("nan")

        # In standard Python jsonschema, NaN is considered a 'number' and comparisons (NaN < min) return False!
        # Thus, validate_env_params_dict unexpectedly passes!
        validate_env_params_dict(d)

        # Furthermore, serializing NaN produces non-standard RFC 8259 JSON:
        json_output = json.dumps(d)
        assert "NaN" in json_output

        # Strict JSON parsers rejecting NaN will fail
        with pytest.raises(ValueError, match="Strict JSON rejects"):
            json.loads(json_output, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Strict JSON rejects {x}")))

    def test_inf_values_bypass_schema_validation(self, default_env_params: EnvParams):
        """FINDING / STRESS: Infinity float values bypass upper bounds if no maximum is configured."""
        d = default_env_params.to_dict()
        d["map_width"] = float("inf")
        d["player_speed"] = float("inf")

        # Passes schema validation because map_width only has 'minimum'
        validate_env_params_dict(d)

    def test_unconstrained_negative_fields_in_schema(self, default_env_params: EnvParams):
        """Hardened schema now enforces minimum: 0.0 checks on dt."""
        import jsonschema
        d = default_env_params.to_dict()
        d["dt"] = -0.0166667
        d["player_width"] = -40.0
        d["player_height"] = -60.0
        d["debris_params"]["max_debris"] = -30
        d["classic_laser"]["beam_thickness"] = -20.0

        with pytest.raises(jsonschema.exceptions.ValidationError):
            validate_env_params_dict(d)

    def test_inverted_wall_boundaries_pass_schema(self, default_env_params: EnvParams):
        """FINDING / STRESS: Schema does not validate relational constraints (wall_left < wall_right)."""
        d = default_env_params.to_dict()
        d["wall_left"] = 1400.0
        d["wall_right"] = 50.0  # Inverted: left is past right
        validate_env_params_dict(d)

    def test_conversion_functions_nan_inf_behavior(self):
        """Test unit conversion functions when supplied NaN, Inf, and zero dt."""
        # ms_to_ticks with NaN raises ValueError
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            ms_to_ticks(float("nan"))

        # ms_to_ticks with Inf raises OverflowError
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            ms_to_ticks(float("inf"))

        # ms_to_ticks with dt=0 raises ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            ms_to_ticks(100, dt=0.0)

        # seconds_to_ticks with NaN raises ValueError
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            seconds_to_ticks(float("nan"))

        # Negative delay clamped to minimum 1 tick
        assert ms_to_ticks(-500) == 1
        assert seconds_to_ticks(-10.0) == 1


# =============================================================================
# Category 2: WZ Crawler Robustness
# =============================================================================

class TestAdversarialWZCrawler:
    """Adversarial stress-testing of WZ asset discovery, parsing, and error recovery."""

    def test_crawler_on_nonexistent_directory(self, tmp_path):
        """Crawler pointing to a non-existent path must gracefully fall back without unhandled exceptions."""
        non_existent = str(tmp_path / "phantom_wz_dir_12345")
        parser = WZParser(wz_dir=non_existent)

        boss_path, map_path = parser.discover_files()
        assert boss_path is None
        assert map_path is None

        meta = parser.extract_pattern_metadata()
        assert meta["total_patterns"] == 0
        assert meta["atlas_regions_count"] == 0

        params = parser.extract_env_params(validate=True)
        assert params.map_width == 1366.0
        assert params.core_pos == (683.0, 384.0)
        assert params.floor_y == 605.0

    def test_crawler_on_empty_wz_files(self, tmp_path):
        """Crawler pointing to 0-byte JSON files must log warnings and fall back to verified physical defaults."""
        boss_file = tmp_path / "BossSuu.img.json"
        map_file = tmp_path / "bossSuu.img.json"
        boss_file.write_text("", encoding="utf-8")
        map_file.write_text("", encoding="utf-8")

        parser = WZParser(wz_dir=str(tmp_path))
        meta = parser.extract_pattern_metadata()
        assert meta["total_patterns"] == 0
        assert len(parser.atlas_regions) == 0

        params = parser.extract_env_params(validate=True)
        assert params.map_width == 1366.0
        assert params.core_radius == 120.0

    def test_crawler_on_corrupted_json_syntax(self, tmp_path):
        """Crawler pointing to syntax-corrupted JSON files must degrade gracefully."""
        boss_file = tmp_path / "BossSuu.img.json"
        boss_file.write_text("{ \"BossSuu.img\": { 1000: [unterminated", encoding="utf-8")

        parser = WZParser(wz_dir=str(tmp_path))
        meta = parser.extract_pattern_metadata()
        assert meta["total_patterns"] == 0

        params = parser.extract_env_params(validate=True)
        assert params.map_width == 1366.0

    def test_crawler_on_non_dict_json_root(self, tmp_path):
        """Crawler pointing to files containing JSON null, list, or primitive root must not crash."""
        boss_file = tmp_path / "BossSuu.img.json"
        boss_file.write_text("null", encoding="utf-8")
        map_file = tmp_path / "bossSuu.img.json"
        map_file.write_text("[1, 2, 3]", encoding="utf-8")

        parser = WZParser(wz_dir=str(tmp_path))
        meta = parser.extract_pattern_metadata()
        assert meta["total_patterns"] == 0
        assert len(parser.atlas_regions) == 0

        params = parser.extract_env_params(validate=True)
        assert params.map_width == 1366.0

    def test_crawler_when_wz_dir_is_a_file(self, tmp_path):
        """Crawler when wz_dir is accidentally pointed to a file instead of a directory."""
        fake_file = tmp_path / "some_file.txt"
        fake_file.write_text("hello", encoding="utf-8")

        parser = WZParser(wz_dir=str(fake_file))
        boss_path, map_path = parser.discover_files()
        assert boss_path is None
        assert map_path is None

        params = parser.extract_env_params(validate=True)
        assert params.map_width == 1366.0

    def test_parse_spine_atlas_adversarial_inputs(self):
        """Stress test parse_spine_atlas with corrupted, truncated, and abnormal lines."""
        # 1. Empty string
        assert parse_spine_atlas("") == {}

        # 2. Whitespace and blank lines
        assert parse_spine_atlas("\n\n   \n\t\n") == {}

        # 3. Malformed numeric values
        adversarial_atlas = """
broken_region_1
  rotate: not_a_bool
  xy: nan, inf
  size: invalid, 10
  orig: 20
  offset: 0, 0
  index: bad_int
broken_region_2
  rotate: TRUE
  xy: 100
"""
        regions = parse_spine_atlas(adversarial_atlas)
        assert "broken_region_1" in regions
        r1 = regions["broken_region_1"]
        assert r1["rotate"] is False  # "not_a_bool".lower() == "true" -> False
        assert math.isnan(r1["xy"][0])
        assert math.isinf(r1["xy"][1])
        assert r1["size"] == (0.0, 0.0)  # Exception caught, set to 0.0, 0.0
        assert r1["index"] == -1          # Exception caught, set to -1

        assert "broken_region_2" in regions
        r2 = regions["broken_region_2"]
        assert r2["rotate"] is True
        assert "xy" not in r2             # len(parts) == 1 != 2, skipped

    def test_restore_node_adversarial_inputs(self):
        """Stress test restore_node with unexpected primitives and container shapes."""
        # Primitive passthrough
        assert restore_node(None) is None
        assert restore_node(42) == 42
        assert restore_node("str") == "str"
        assert restore_node([]) == []

        # Dict without type or children
        assert restore_node({}) == {}

        # Dict with empty children
        assert restore_node({"type": "SubProperty", "children": []}) == {}

        # Children containing invalid child types
        node_with_bad_children = {
            "type": "Canvas",
            "width": 100,
            "height": 200,
            "children": [
                None,
                "invalid",
                123,
                {"name": "valid_child", "type": "Int", "value": 999},
            ],
        }
        res = restore_node(node_with_bad_children)
        assert res["_width"] == 100
        assert res["_height"] == 200
        assert res["valid_child"] == 999


# =============================================================================
# Category 3: JAX PyTree Serialization & JIT / VMAP Compilation Edge Cases
# =============================================================================

class TestAdversarialJAXPyTreeAndJIT:
    """Adversarial tests uncovering JAX PyTree flattening, tracing, and XLA JIT failure modes."""

    def test_pytree_treedef_mismatch_tuple_vs_list(self):
        """FINDING / STRESS: DebrisParams initialized with a list produces a different PyTree treedef than with a tuple."""
        p_tuple = EnvParams(debris_params=DebrisParams(types=tuple(_default_debris_types())))
        p_list = EnvParams(debris_params=DebrisParams(types=list(_default_debris_types())))  # type: ignore

        leaves_t, treedef_t = jax.tree_util.tree_flatten(p_tuple)
        leaves_l, treedef_l = jax.tree_util.tree_flatten(p_list)

        # PyTree structure equality MUST fail because list and tuple are distinct container types in JAX
        assert treedef_t != treedef_l

        # Unflattening with mismatched treedef breaks structural equality with original tuple instance
        reconstructed_from_l = jax.tree_util.tree_unflatten(treedef_l, leaves_t)
        assert reconstructed_from_l != p_tuple
        assert type(reconstructed_from_l.debris_params.types) is list

    def test_critical_concretization_type_error_on_all_property_accessors(self, default_env_params: EnvParams):
        """CRITICAL VULNERABILITY: All 14 property accessors crash under @jax.jit with ConcretizationTypeError.

        When EnvParams is passed as a function argument into @jax.jit (such as in Gymnax env.step or env.reset),
        JAX replaces leaves with Tracer objects. Calling float(...) or int(...) on tracers raises ConcretizationTypeError!
        """
        all_properties = [
            "core_pos",
            "laser_omega",
            "laser_thickness",
            "jump_impulse",
            "max_debris",
            "gauge_gain_rate",
            "screen_width",
            "screen_height",
            "player_w",
            "player_h",
            "jump_velocity",
            "laser_half_thickness",
            "laser_damage",
            "overload_duration",
        ]

        failing_props = []
        for prop_name in all_properties:
            try:
                jit_fn = jax.jit(lambda params, p=prop_name: getattr(params, p))
                _ = jit_fn(default_env_params)
            except jax.errors.ConcretizationTypeError:
                failing_props.append(prop_name)

        # EMPIRICAL PROOF: 0% of property accessors fail under jax.jit after remediation!
        assert len(failing_props) == 0, f"Expected 0 property accessors to fail under jax.jit, but failed: {failing_props}"

    def test_critical_static_array_dimension_concretization_failure(self, default_env_params: EnvParams):
        """Remediation: max_debris is configured with pytree_node=False, enabling static array allocation in XLA."""
        @jax.jit
        def allocate_debris_array(params: EnvParams):
            return jnp.zeros((params.debris_params.max_debris, 4))

        arr = allocate_debris_array(default_env_params)
        assert arr.shape == (30, 4)

    def test_to_dict_crashes_under_jax_jit(self, default_env_params: EnvParams):
        """FINDING / STRESS: Calling to_dict() inside a JIT-compiled context raises ConcretizationTypeError."""
        @jax.jit
        def jit_serialize(params: EnvParams):
            return params.to_dict()

        with pytest.raises(jax.errors.ConcretizationTypeError):
            jit_serialize(default_env_params)

    def test_direct_field_access_succeeds_under_jax_jit(self, default_env_params: EnvParams):
        """CONTRAST: Direct field access (without float/int casting properties) compiles cleanly under jax.jit."""
        @jax.jit
        def direct_physics(params: EnvParams, x: float):
            # Direct field access preserves JAX tracer arithmetic
            return (
                x * params.classic_laser.angular_velocity
                + params.core_x
                + params.floor_y
                + params.player_width
            )

        result = direct_physics(default_env_params, 10.0)
        expected = (
            10.0 * default_env_params.classic_laser.angular_velocity
            + default_env_params.core_x
            + default_env_params.floor_y
            + default_env_params.player_width
        )
        assert float(result) == pytest.approx(expected)

    def test_vmap_with_env_params_under_jit(self, default_env_params: EnvParams):
        """Verify vmap execution when params is passed as unbatched parameter under JIT."""
        batch_size = 16
        states = jnp.zeros((batch_size, 2))
        actions = jnp.ones((batch_size,))

        # Property access under jit(vmap) succeeds after remediation
        @jax.jit
        def vmap_property_step(s, a, p):
            def single_step(s_i, a_i):
                return s_i + a_i * p.player_w + p.floor_y
            return jax.vmap(single_step)(s, a)

        out_prop = vmap_property_step(states, actions, default_env_params)
        assert out_prop.shape == (batch_size, 2)
        assert out_prop[0, 0] == 40.0 + 605.0

        # Direct field access under jit(vmap) succeeds
        @jax.jit
        def vmap_direct_step(s, a, p):
            def single_step(s_i, a_i):
                return s_i + a_i * p.player_width + p.floor_y
            return jax.vmap(single_step)(s, a)

        out = vmap_direct_step(states, actions, default_env_params)
        assert out.shape == (batch_size, 2)
        assert out[0, 0] == 40.0 + 605.0

    def test_tree_map_preserves_pytree_structure(self, default_env_params: EnvParams):
        """Verify tree_map converts dynamic leaves to jax.Array while keeping static metadata intact."""
        jax_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), default_env_params)
        assert isinstance(jax_params.map_width, jax.Array)
        # Static auxiliary metadata configured with pytree_node=False remains int
        assert jax_params.classic_laser.beam_count == 4
        # Static metadata must remain string
        assert jax_params.remastered_params.difficulty == "normal"
        assert isinstance(jax_params.remastered_params.difficulty, str)
