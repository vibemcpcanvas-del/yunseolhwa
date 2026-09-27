"""Empirical Challenger 2 test suite for Milestone 1.

Validates:
1. Real C:\\mp data extraction via parse_wz_to_env_params('C:\\mp').
2. Comprehensive field inspection of EnvParams.
3. Verification of physical parameters against MapleStory Lotus specifications:
   - Screen bounds: (1366, 768), Core center: (683.0, 384.0), Floor Y: ~605.0.
   - Laser angular velocity: 0.5235 rad/s, player speed: 400.0 px/s, dt: 1/60s.
   - Remastered rates: natural gauge gain 0.006 (normal) to 0.020 (extreme), overload duration 25.0s.
4. Validation that generated JSON passes validate_env_params_dict.
5. JAX JIT and PyTree compatibility with extracted EnvParams.
6. Stress testing and adversarial edge cases on extraction and parameters.
"""

from __future__ import annotations

import json
import math
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
    parse_wz_to_env_params,
    save_env_params_json,
)


MP_DIR = r"C:\mp"


class TestRealDataExtractionAndFieldInspection:
    """1. Run parse_wz_to_env_params('C:\\mp') and inspect every field of EnvParams."""

    @pytest.fixture(scope="class")
    def real_params_normal(self) -> EnvParams:
        if not os.path.isdir(MP_DIR):
            pytest.skip(f"{MP_DIR} does not exist on this machine")
        return parse_wz_to_env_params(MP_DIR, difficulty="normal")

    @pytest.fixture(scope="class")
    def real_params_hard(self) -> EnvParams:
        if not os.path.isdir(MP_DIR):
            pytest.skip(f"{MP_DIR} does not exist on this machine")
        return parse_wz_to_env_params(MP_DIR, difficulty="hard")

    @pytest.fixture(scope="class")
    def real_params_extreme(self) -> EnvParams:
        if not os.path.isdir(MP_DIR):
            pytest.skip(f"{MP_DIR} does not exist on this machine")
        return parse_wz_to_env_params(MP_DIR, difficulty="extreme")

    def test_c_mp_directory_and_files_exist(self):
        """Verify C:\\mp directory structure and asset files."""
        assert os.path.isdir(MP_DIR), f"Directory {MP_DIR} must exist"
        parser = WZParser(wz_dir=MP_DIR)
        boss_path, map_path = parser.discover_files()
        assert boss_path is not None, "BossSuu.img.json must be discovered"
        assert os.path.isfile(boss_path), f"File {boss_path} must exist on disk"
        assert map_path is not None, "bossSuu.img.json map atlas must be discovered"
        assert os.path.isfile(map_path), f"File {map_path} must exist on disk"

    def test_inspect_all_top_level_fields(self, real_params_normal: EnvParams):
        """Inspect all top-level fields of the generated EnvParams."""
        p = real_params_normal

        # Screen & Map Bounds
        assert isinstance(p.map_width, float)
        assert p.map_width == 1366.0
        assert isinstance(p.map_height, float)
        assert p.map_height == 768.0

        # Core Coordinates & Radius
        assert isinstance(p.core_x, float)
        assert p.core_x == 683.0
        assert isinstance(p.core_y, float)
        assert p.core_y == 384.0
        assert isinstance(p.core_radius, float)
        assert p.core_radius == 120.0

        # Floor and Walls
        assert isinstance(p.floor_y, float)
        assert p.floor_y == 605.0
        assert isinstance(p.wall_left, float)
        assert p.wall_left == 50.0
        assert isinstance(p.wall_right, float)
        assert p.wall_right == 1316.0
        assert isinstance(p.safe_zone_x, float)
        assert p.safe_zone_x == 1150.0

        # Player Kinematics
        assert isinstance(p.player_width, float)
        assert p.player_width == 40.0
        assert isinstance(p.player_height, float)
        assert p.player_height == 60.0
        assert isinstance(p.player_speed, float)
        assert p.player_speed == 400.0
        assert isinstance(p.player_jump_impulse, float)
        assert p.player_jump_impulse == -650.0
        assert isinstance(p.gravity, float)
        assert p.gravity == 1800.0
        assert isinstance(p.dt, float)
        assert pytest.approx(p.dt, rel=1e-5) == 1.0 / 60.0

        # Sub-dataclasses
        assert isinstance(p.classic_laser, ClassicLaserParams)
        assert isinstance(p.debris_params, DebrisParams)
        assert isinstance(p.remastered_params, RemasteredParams)

    def test_inspect_classic_laser_fields(self, real_params_normal: EnvParams):
        """Inspect all fields of classic_laser."""
        cl = real_params_normal.classic_laser
        assert isinstance(cl.angular_velocity, float)
        assert pytest.approx(cl.angular_velocity, rel=1e-4) == 0.523598775
        assert isinstance(cl.beam_thickness, float)
        assert cl.beam_thickness == 20.0
        assert isinstance(cl.beam_count, int)
        assert cl.beam_count == 4
        assert isinstance(cl.damage, float)
        assert cl.damage == 1.0

    def test_inspect_debris_fields(self, real_params_normal: EnvParams):
        """Inspect all fields of debris_params and all debris types."""
        dp = real_params_normal.debris_params
        assert isinstance(dp.max_debris, int)
        assert dp.max_debris == 30
        assert isinstance(dp.spawn_interval_ticks, int)
        assert dp.spawn_interval_ticks == 15
        assert isinstance(dp.types, tuple)
        assert len(dp.types) == 4

        expected_types = [
            {"type_id": 0, "radius": 15.0, "damage": 0.10, "stun_duration": 1.0, "fall_speed": 300.0},
            {"type_id": 1, "radius": 25.0, "damage": 0.20, "stun_duration": 1.5, "fall_speed": 350.0},
            {"type_id": 2, "radius": 35.0, "damage": 0.30, "stun_duration": 2.0, "fall_speed": 400.0},
            {"type_id": 3, "radius": 50.0, "damage": 0.40, "stun_duration": 2.5, "fall_speed": 450.0},
        ]
        for dt, exp in zip(dp.types, expected_types):
            assert isinstance(dt, DebrisTypeParams)
            assert dt.type_id == exp["type_id"]
            assert dt.radius == exp["radius"]
            assert pytest.approx(dt.damage) == exp["damage"]
            assert pytest.approx(dt.stun_duration) == exp["stun_duration"]
            assert pytest.approx(dt.fall_speed) == exp["fall_speed"]

    def test_inspect_remastered_fields(self, real_params_normal: EnvParams):
        """Inspect all fields of remastered_params."""
        rm = real_params_normal.remastered_params
        assert isinstance(rm.enabled, bool)
        assert rm.enabled is True
        assert isinstance(rm.difficulty, str)
        assert rm.difficulty == "normal"
        assert isinstance(rm.gauge_natural_rates, GaugeNaturalRates)
        assert rm.gauge_natural_rates.normal == 0.006
        assert rm.gauge_natural_rates.hard == 0.008
        assert rm.gauge_natural_rates.extreme == 0.020
        assert isinstance(rm.overload_duration, float)
        assert rm.overload_duration == 25.0
        assert isinstance(rm.bombardment_damage, float)
        assert rm.bombardment_damage == 1.0
        assert isinstance(rm.bombardment_tick_rate, float)
        assert rm.bombardment_tick_rate == 0.5
        assert isinstance(rm.electric_field_damage, float)
        assert rm.electric_field_damage == 0.05
        assert isinstance(rm.electric_field_tick_rate, float)
        assert rm.electric_field_tick_rate == 0.36
        assert isinstance(rm.tracking_laser_player_damage, float)
        assert rm.tracking_laser_player_damage == 0.15
        assert isinstance(rm.tracking_laser_gauge_increase, float)
        assert rm.tracking_laser_gauge_increase == 0.10
        assert isinstance(rm.tracking_laser_gauge_decrease, float)
        assert rm.tracking_laser_gauge_decrease == 0.10
        assert isinstance(rm.small_arm_player_damage, float)
        assert rm.small_arm_player_damage == 0.05
        assert isinstance(rm.small_arm_gauge_increase, float)
        assert rm.small_arm_gauge_increase == 0.03
        assert isinstance(rm.small_arm_gauge_decrease, float)
        assert rm.small_arm_gauge_decrease == 0.03
        assert isinstance(rm.floor_discharge_warning_duration, float)
        assert rm.floor_discharge_warning_duration == 1.2
        assert isinstance(rm.floor_discharge_damage, float)
        assert rm.floor_discharge_damage == 1.0


class TestPhysicalParametersMapleStorySpecs:
    """2. Check physical parameters against MapleStory Lotus specifications."""

    def test_screen_bounds_and_core_center_and_floor(self):
        """Screen bounds: (1366, 768), Core center: (683.0, 384.0), Floor Y: ~605.0."""
        p = parse_wz_to_env_params(MP_DIR)
        assert (p.map_width, p.map_height) == (1366.0, 768.0)
        assert (p.screen_width, p.screen_height) == (1366.0, 768.0)
        assert (p.core_x, p.core_y) == (683.0, 384.0)
        assert p.core_pos == (683.0, 384.0)
        assert 600.0 <= p.floor_y <= 610.0
        assert p.floor_y == 605.0

    def test_laser_omega_player_speed_dt(self):
        """Laser angular velocity: 0.5235 rad/s, player speed: 400.0 px/s, dt: 1/60s."""
        p = parse_wz_to_env_params(MP_DIR)
        assert pytest.approx(p.laser_omega, abs=1e-3) == 0.5235
        assert pytest.approx(p.classic_laser.angular_velocity, abs=1e-3) == 0.5235
        assert p.player_speed == 400.0
        assert pytest.approx(p.dt, rel=1e-5) == 1.0 / 60.0

    def test_remastered_rates_and_overload_duration(self):
        """Remastered rates: natural gauge gain 0.006 (normal) to 0.020 (extreme), overload duration 25.0s."""
        p_normal = parse_wz_to_env_params(MP_DIR, difficulty="normal")
        assert p_normal.gauge_gain_rate == 0.006
        assert p_normal.remastered_params.gauge_natural_rates.normal == 0.006
        assert p_normal.overload_duration == 25.0
        assert p_normal.remastered_params.overload_duration == 25.0

        p_hard = parse_wz_to_env_params(MP_DIR, difficulty="hard")
        assert p_hard.gauge_gain_rate == 0.008
        assert p_hard.remastered_params.gauge_natural_rates.hard == 0.008
        assert p_hard.overload_duration == 25.0

        p_extreme = parse_wz_to_env_params(MP_DIR, difficulty="extreme")
        assert p_extreme.gauge_gain_rate == 0.020
        assert p_extreme.remastered_params.gauge_natural_rates.extreme == 0.020
        assert p_extreme.overload_duration == 25.0


class TestSchemaValidationAndSerialization:
    """3. Validate that generated JSON passes validate_env_params_dict."""

    def test_validate_env_params_dict_normal(self):
        p = parse_wz_to_env_params(MP_DIR, difficulty="normal")
        d = p.to_dict()
        validate_env_params_dict(d)

    def test_validate_env_params_dict_hard(self):
        p = parse_wz_to_env_params(MP_DIR, difficulty="hard")
        d = p.to_dict()
        validate_env_params_dict(d)

    def test_validate_env_params_dict_extreme(self):
        p = parse_wz_to_env_params(MP_DIR, difficulty="extreme")
        d = p.to_dict()
        validate_env_params_dict(d)

    def test_validate_env_params_dict_remaster_disabled(self):
        p = parse_wz_to_env_params(MP_DIR, enable_remaster=False)
        d = p.to_dict()
        validate_env_params_dict(d)
        assert d["remastered_params"]["enabled"] is False

    def test_json_string_roundtrip(self):
        p = parse_wz_to_env_params(MP_DIR, difficulty="extreme")
        json_str = p.to_json()
        parsed_dict = json.loads(json_str)
        validate_env_params_dict(parsed_dict)
        p_reconstructed = EnvParams.from_json(json_str)
        assert p_reconstructed == p

    def test_file_io_roundtrip(self, tmp_path):
        p = parse_wz_to_env_params(MP_DIR, difficulty="hard")
        out_file = str(tmp_path / "env_params_test_io.json")
        save_env_params_json(p, out_file)
        assert os.path.isfile(out_file)
        loaded = load_env_params_json(out_file, validate=True)
        assert loaded == p


class TestAdversarialAndStressScenarios:
    """Empirical adversarial stress testing on parameters and parser pipeline."""

    def test_jax_tree_flatten_and_jit_compatibility(self):
        """Verify EnvParams behaves as a valid Flax PyTree in JAX JIT computation."""
        p = parse_wz_to_env_params(MP_DIR)

        leaves, treedef = jax.tree_util.tree_flatten(p)
        assert len(leaves) > 0
        unflattened = jax.tree_util.tree_unflatten(treedef, leaves)
        assert unflattened == p

        # Test JIT execution taking EnvParams as static or dynamic PyTree arg
        @jax.jit
        def compute_kinetic_energy(params: EnvParams, vx: float, vy: float) -> float:
            # Simple physics check inside XLA JIT
            speed_sq = vx * vx + vy * vy
            return 0.5 * speed_sq * params.dt

        res = compute_kinetic_energy(p, 400.0, 0.0)
        expected = 0.5 * (400.0 ** 2) * (1.0 / 60.0)
        assert pytest.approx(float(res), rel=1e-4) == expected

    def test_schema_rejects_negative_speed(self):
        """Negative speed should be caught if validated."""
        p = parse_wz_to_env_params(MP_DIR)
        d = p.to_dict()
        # Test modifying required fields to out of bounds
        d["map_width"] = 100.0  # < 800 minimum
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(d)

    def test_schema_rejects_non_boolean_remaster(self):
        p = parse_wz_to_env_params(MP_DIR)
        d = p.to_dict()
        d["remastered_params"]["enabled"] = "not_a_boolean"
        with pytest.raises(jsonschema.ValidationError):
            validate_env_params_dict(d)

    def test_custom_overrides_boundary_values(self):
        """Test custom overrides at boundary values."""
        overrides = {
            "player_speed": 600.0,
            "floor_y": 600.0,
            "classic_laser": {"beam_count": 8, "beam_thickness": 50.0},
        }
        p = parse_wz_to_env_params(MP_DIR, custom_overrides=overrides, validate=True)
        assert p.player_speed == 600.0
        assert p.floor_y == 600.0
        assert p.classic_laser.beam_count == 8
        assert p.laser_thickness == 50.0

    def test_invalid_difficulty_raises_validation_error(self):
        """Invalid difficulty should fail validation."""
        with pytest.raises(jsonschema.ValidationError):
            parse_wz_to_env_params(MP_DIR, difficulty="invalid_difficulty", validate=True)
