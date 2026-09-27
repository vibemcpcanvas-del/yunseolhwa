"""Unit tests for MapleStory WZ Client Data Parser and EnvParams Schema.

Tests verify:
- Draft 2020-12 JSON Schema validation and edge cases
- Flax dataclass and JAX PyTree compatibility
- Unit conversion utilities (ms -> sec, ms -> ticks, sec -> ticks)
- WZ node restoration and Spine atlas parsing
- Real C:\\mp crawling, pattern extraction, and atlas verification
- Synthesis of canonical client physical defaults and custom overrides
- JSON serialization/deserialization file roundtrip
"""

from __future__ import annotations

import os
from typing import Any, Dict
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
# 1. Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Tests for Draft 2020-12 JSON schema validation."""

    def test_default_env_params_valid(self, default_env_params: EnvParams):
        """Default EnvParams should strictly pass schema validation."""
        validate_env_params(default_env_params)
        raw_dict = default_env_params.to_dict()
        validate_env_params_dict(raw_dict)

    def test_schema_rejects_missing_required_field(self, default_env_params: EnvParams):
        """Schema should fail when a required field is missing."""
        data = default_env_params.to_dict()
        del data["floor_y"]
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)

    def test_schema_rejects_invalid_map_dimensions(self, default_env_params: EnvParams):
        """Schema enforces map_width >= 800 and map_height >= 600."""
        data = default_env_params.to_dict()
        data["map_width"] = 400.0  # Below minimum 800.0
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)

    def test_schema_rejects_invalid_difficulty_enum(self, default_env_params: EnvParams):
        """Schema enforces difficulty in ['normal', 'hard', 'extreme']."""
        data = default_env_params.to_dict()
        data["remastered_params"]["difficulty"] = "super_hard"
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)

    def test_schema_rejects_invalid_beam_count(self, default_env_params: EnvParams):
        """Schema enforces classic laser beam_count in [1, 8]."""
        data = default_env_params.to_dict()
        data["classic_laser"]["beam_count"] = 0
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)

    def test_schema_rejects_negative_gravity_and_dt(self, default_env_params: EnvParams):
        """Schema enforces gravity >= 0 and dt >= 0."""
        data = default_env_params.to_dict()
        data["gravity"] = -100.0
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)

        data = default_env_params.to_dict()
        data["dt"] = -0.016
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(data)


# =============================================================================
# 2. Flax Dataclass & JAX PyTree Compatibility Tests
# =============================================================================

class TestFlaxDataclassCompatibility:
    """Tests confirming EnvParams functions properly as a Flax PyTree."""

    def test_jax_tree_flatten_unflatten(self, default_env_params: EnvParams):
        """EnvParams must be flattenable and unflattenable without JAX error."""
        leaves, tree_def = jax.tree_util.tree_flatten(default_env_params)
        assert len(leaves) > 0

        reconstructed = jax.tree_util.tree_unflatten(tree_def, leaves)
        assert reconstructed == default_env_params
        assert reconstructed.core_pos == default_env_params.core_pos
        assert reconstructed.remastered_params.difficulty == "normal"

    def test_env_params_property_accessors(self, default_env_params: EnvParams):
        """Verify convenience property accessors conforming to PROJECT.md."""
        assert default_env_params.core_pos == (683.0, 384.0)
        assert default_env_params.laser_omega == pytest.approx(0.523598775)
        assert default_env_params.laser_thickness == 20.0
        assert default_env_params.laser_half_thickness == 10.0
        assert default_env_params.laser_damage == 1.0
        assert default_env_params.screen_width == 1366.0
        assert default_env_params.screen_height == 768.0
        assert default_env_params.player_w == 40.0
        assert default_env_params.player_h == 60.0
        assert default_env_params.jump_velocity == -650.0
        assert default_env_params.jump_impulse == 650.0
        assert default_env_params.max_debris == 30
        assert default_env_params.gauge_gain_rate == 0.006
        assert default_env_params.overload_duration == 25.0

    def test_property_accessors_inside_jax_jit(self, default_env_params: EnvParams):
        """All 14 property accessors execute inside @jax.jit without ConcretizationTypeError."""
        @jax.jit
        def eval_props(p: EnvParams):
            return (
                p.core_pos[0],
                p.core_pos[1],
                p.laser_omega,
                p.laser_thickness,
                p.laser_half_thickness,
                p.laser_damage,
                p.screen_width,
                p.screen_height,
                p.player_w,
                p.player_h,
                p.jump_velocity,
                p.jump_impulse,
                p.max_debris,
                p.gauge_gain_rate,
                p.overload_duration,
            )

        res = eval_props(default_env_params)
        assert float(res[0]) == 683.0
        assert float(res[1]) == 384.0
        assert float(res[2]) == pytest.approx(0.523598775)
        assert float(res[3]) == 20.0
        assert float(res[4]) == 10.0
        assert float(res[5]) == 1.0
        assert float(res[6]) == 1366.0
        assert float(res[7]) == 768.0
        assert float(res[8]) == 40.0
        assert float(res[9]) == 60.0
        assert float(res[10]) == -650.0
        assert float(res[11]) == 650.0
        assert int(res[12]) == 30
        assert float(res[13]) == pytest.approx(0.006)
        assert float(res[14]) == 25.0

    def test_static_array_allocation_inside_jax_jit(self, default_env_params: EnvParams):
        """Static array allocation jnp.zeros((params.max_debris, 2)) succeeds inside @jax.jit."""
        @jax.jit
        def alloc_fn(params: EnvParams):
            debris_arr = jnp.zeros((params.max_debris, 2))
            laser_arr = jnp.zeros((params.classic_laser.beam_count,))
            return debris_arr, laser_arr

        d_arr, l_arr = alloc_fn(default_env_params)
        assert d_arr.shape == (30, 2)
        assert l_arr.shape == (4,)

    def test_jax_vmap_across_batch_of_params(self, default_env_params: EnvParams):
        """jax.vmap across batch of params functions without shape mismatch."""
        batch_size = 16
        batched_params = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + jnp.asarray(x).shape),
            default_env_params,
        )

        @jax.jit
        @jax.vmap
        def batched_physics(p: EnvParams):
            cx, cy = p.core_pos
            ray_length = p.laser_omega * 10.0 + p.laser_thickness
            kinematics = p.player_w * 0.5 + p.screen_width - p.jump_impulse
            gauge = p.gauge_gain_rate * p.overload_duration
            return jnp.stack([cx, cy, ray_length, kinematics, gauge])

        out = batched_physics(batched_params)
        assert out.shape == (batch_size, 5)
        assert jnp.all(out[:, 0] == 683.0)
        assert jnp.all(out[:, 1] == 384.0)


# =============================================================================
# 3. Serialization Roundtrip Tests
# =============================================================================

class TestSerializationRoundtrip:
    """Tests for dict and JSON roundtrip serialization."""

    def test_dict_roundtrip(self, default_env_params: EnvParams):
        d = default_env_params.to_dict()
        reconstructed = EnvParams.from_dict(d)
        assert reconstructed == default_env_params

    def test_json_roundtrip(self, default_env_params: EnvParams):
        json_str = default_env_params.to_json()
        reconstructed = EnvParams.from_json(json_str)
        assert reconstructed == default_env_params

    def test_file_save_and_load(self, tmp_path, default_env_params: EnvParams):
        file_path = str(tmp_path / "env_params_test.json")
        save_env_params_json(default_env_params, file_path)
        assert os.path.isfile(file_path)

        loaded = load_env_params_json(file_path, validate=True)
        assert loaded == default_env_params


# =============================================================================
# 4. Unit Conversion Utilities Tests
# =============================================================================

class TestUnitConversions:
    """Tests for frame delay and tick conversion functions."""

    def test_ms_to_seconds(self):
        assert ms_to_seconds(1000) == 1.0
        assert ms_to_seconds(120) == pytest.approx(0.12)
        assert ms_to_seconds(0) == 0.0

    def test_ms_to_ticks(self):
        assert ms_to_ticks(1000, dt=1.0 / 60.0) == 60
        assert ms_to_ticks(16.6667, dt=1.0 / 60.0) == 1
        assert ms_to_ticks(0, dt=1.0 / 60.0) == 1  # Clamped to at least 1 tick
        assert ms_to_ticks(500, dt=1.0 / 60.0) == 30

    def test_ticks_to_seconds(self):
        assert ticks_to_seconds(60, dt=1.0 / 60.0) == pytest.approx(1.0)
        assert ticks_to_seconds(30, dt=1.0 / 60.0) == pytest.approx(0.5)

    def test_seconds_to_ticks(self):
        assert seconds_to_ticks(1.0, dt=1.0 / 60.0) == 60
        assert seconds_to_ticks(25.0, dt=1.0 / 60.0) == 1500
        assert seconds_to_ticks(0.0, dt=1.0 / 60.0) == 1


# =============================================================================
# 5. WZ Node Restoration & Atlas Parsing Tests
# =============================================================================

class TestWZRestorationAndAtlas:
    """Tests for raw node restoration and Spine texture atlas extraction."""

    def test_restore_node(self, sample_raw_wz_node: Dict[str, Any]):
        restored = restore_node(sample_raw_wz_node)
        assert "000" in restored
        sub = restored["000"]
        assert "0" in sub
        canvas_0 = sub["0"]
        assert canvas_0["_width"] == 120
        assert canvas_0["_height"] == 200
        assert canvas_0["delay"] == 120
        assert canvas_0["origin"] == [60, 100]
        assert sub["uol_ref"]["_target"] == "../0"

    def test_parse_spine_atlas(self, sample_spine_atlas_text: str):
        regions = parse_spine_atlas(sample_spine_atlas_text)
        assert len(regions) == 2
        assert "02_upper ligh10_1" in regions
        assert "03_blue_light01_1" in regions

        r1 = regions["02_upper ligh10_1"]
        assert r1["rotate"] is False
        assert r1["xy"] == (80.0, 2005.0)
        assert r1["size"] == (35.0, 30.0)
        assert r1["index"] == -1

        r2 = regions["03_blue_light01_1"]
        assert r2["rotate"] is True


# =============================================================================
# 6. Real C:\mp Asset Crawling & Synthesis Tests
# =============================================================================

class TestWZParserWithRealData:
    """Tests running WZParser directly against the local C:\\mp assets."""

    @pytest.fixture(autouse=True)
    def skip_if_no_c_mp(self, mp_dir: str):
        if not os.path.isdir(mp_dir):
            pytest.skip(f"WZ directory {mp_dir} not available on this machine")

    def test_discover_files(self, mp_dir: str):
        parser = WZParser(wz_dir=mp_dir)
        boss_path, map_path = parser.discover_files()
        assert boss_path is not None and os.path.isfile(boss_path)
        assert map_path is not None and os.path.isfile(map_path)
        assert "BossSuu.img.json" in boss_path
        assert "bossSuu.img.json" in map_path

    def test_pattern_and_atlas_extraction(self, mp_dir: str):
        parser = WZParser(wz_dir=mp_dir)
        meta = parser.extract_pattern_metadata()

        # Check discovered patterns 1000..1009
        patterns = meta["patterns"]
        assert meta["total_patterns"] >= 10
        for pid in ("1000", "1001", "1006"):
            assert pid in patterns

        # Check tracking laser 1001 sub-actions
        assert "000" in patterns["1001"]["sub_actions"]
        assert "001" in patterns["1001"]["sub_actions"]

        # Check overload horizontal bombardment 1006 sub-actions
        assert "000" in patterns["1006"]["sub_actions"]
        assert "002" in patterns["1006"]["sub_actions"]

        # Check UI components
        assert "default" in meta["ui_keys"]
        assert "destruction" in meta["ui_keys"]
        assert "overload" in meta["ui_keys"]

        # Check Spine atlas regions
        assert len(parser.atlas_regions) == 124

    def test_extract_env_params_synthesis(self, mp_dir: str):
        parser = WZParser(wz_dir=mp_dir)
        params = parser.extract_env_params(difficulty="normal", validate=True)

        # Verify geometric and physical bounds
        assert params.map_width == 1366.0
        assert params.map_height == 768.0
        assert params.core_pos == (683.0, 384.0)
        assert params.core_radius == 120.0
        assert params.floor_y == 605.0
        assert params.wall_left == 50.0
        assert params.wall_right == 1316.0
        assert params.safe_zone_x == 1150.0

        # Player
        assert params.player_width == 40.0
        assert params.player_height == 60.0
        assert params.player_speed == 400.0

        # Laser & Debris
        assert params.classic_laser.beam_count == 4
        assert params.classic_laser.angular_velocity == pytest.approx(0.523598775)
        assert len(params.debris_params.types) == 4
        assert params.debris_params.max_debris == 30

        # Remastered mechanics
        assert params.remastered_params.enabled is True
        assert params.remastered_params.difficulty == "normal"
        assert params.gauge_gain_rate == 0.006

    def test_difficulty_rate_variations(self, mp_dir: str):
        parser = WZParser(wz_dir=mp_dir)
        p_normal = parser.extract_env_params(difficulty="normal")
        p_hard = parser.extract_env_params(difficulty="hard")
        p_extreme = parser.extract_env_params(difficulty="extreme")

        assert p_normal.gauge_gain_rate == 0.006
        assert p_hard.gauge_gain_rate == 0.008
        assert p_extreme.gauge_gain_rate == 0.020

    def test_custom_overrides(self, mp_dir: str):
        overrides = {
            "floor_y": 610.0,
            "classic_laser": {"beam_thickness": 30.0},
        }
        params = parse_wz_to_env_params(wz_dir=mp_dir, custom_overrides=overrides, validate=True)
        assert params.floor_y == 610.0
        assert params.laser_thickness == 30.0
        assert params.laser_omega == pytest.approx(0.523598775)


# =============================================================================
# 7. Fallback & Graceful Degradation Tests
# =============================================================================

class TestFallbackGracefulDegradation:
    """Tests confirming fallback to synthesized defaults on missing directories."""

    def test_missing_directory_fallback(self, tmp_path):
        """Non-existent directory should fall back cleanly to validated defaults."""
        non_existent_dir = str(tmp_path / "does_not_exist")
        parser = WZParser(wz_dir=non_existent_dir)
        params = parser.extract_env_params(validate=True)

        assert params.map_width == 1366.0
        assert params.core_pos == (683.0, 384.0)
        assert params.floor_y == 605.0
        assert params.remastered_params.enabled is True
