"""EnvParams Flax Dataclass and JSON Schema for MapleStory Lotus Phase 1.

This module defines the immutable environment parameters (EnvParams) compatible
with JAX/Flax PyTrees, Draft 2020-12 JSON Schema validation, and bidirectional
serialization/deserialization.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from flax import struct
import jsonschema


# =============================================================================
# JSON Schema (Draft 2020-12) Definition
# =============================================================================

LOTUS_PHASE1_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LotusPhase1EnvParams",
    "description": "Immutable Environment Parameters for MapleStory Lotus Phase 1 Gymnax Simulator",
    "type": "object",
    "required": [
        "map_width",
        "map_height",
        "core_x",
        "core_y",
        "core_radius",
        "floor_y",
        "wall_left",
        "wall_right",
        "safe_zone_x",
        "player_width",
        "player_height",
        "player_speed",
        "player_jump_impulse",
        "gravity",
        "dt",
        "classic_laser",
        "debris_params",
        "remastered_params",
    ],
    "properties": {
        "map_width": {"type": "number", "default": 1366.0, "minimum": 800.0},
        "map_height": {"type": "number", "default": 768.0, "minimum": 600.0},
        "core_x": {"type": "number", "default": 683.0},
        "core_y": {"type": "number", "default": 384.0},
        "core_radius": {"type": "number", "default": 120.0, "minimum": 10.0},
        "floor_y": {"type": "number", "default": 605.0},
        "wall_left": {"type": "number", "default": 50.0},
        "wall_right": {"type": "number", "default": 1316.0},
        "safe_zone_x": {"type": "number", "default": 1150.0},
        "player_width": {"type": "number", "default": 40.0},
        "player_height": {"type": "number", "default": 60.0},
        "player_speed": {"type": "number", "default": 400.0},
        "player_jump_impulse": {"type": "number", "default": -650.0},
        "gravity": {"type": "number", "default": 1800.0, "minimum": 0.0},
        "dt": {"type": "number", "default": 0.016666667, "minimum": 0.0},
        "classic_laser": {
            "type": "object",
            "required": ["angular_velocity", "beam_thickness", "beam_count", "damage"],
            "properties": {
                "angular_velocity": {"type": "number", "default": 0.523598775},
                "beam_thickness": {"type": "number", "default": 20.0},
                "beam_count": {"type": "integer", "default": 4, "minimum": 1, "maximum": 8},
                "damage": {"type": "number", "default": 1.0},
            },
        },
        "debris_params": {
            "type": "object",
            "required": ["max_debris", "spawn_interval_ticks", "types"],
            "properties": {
                "max_debris": {"type": "integer", "default": 30},
                "spawn_interval_ticks": {"type": "integer", "default": 15},
                "types": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type_id", "radius", "damage", "stun_duration", "fall_speed"],
                        "properties": {
                            "type_id": {"type": "integer"},
                            "radius": {"type": "number"},
                            "damage": {"type": "number"},
                            "stun_duration": {"type": "number"},
                            "fall_speed": {"type": "number"},
                        },
                    },
                },
            },
        },
        "remastered_params": {
            "type": "object",
            "required": [
                "enabled",
                "difficulty",
                "gauge_natural_rates",
                "overload_duration",
                "bombardment_damage",
                "bombardment_tick_rate",
                "electric_field_damage",
                "electric_field_tick_rate",
                "tracking_laser_player_damage",
                "tracking_laser_gauge_increase",
                "tracking_laser_gauge_decrease",
                "small_arm_player_damage",
                "small_arm_gauge_increase",
                "small_arm_gauge_decrease",
                "floor_discharge_warning_duration",
                "floor_discharge_damage",
            ],
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "difficulty": {
                    "type": "string",
                    "enum": ["normal", "hard", "extreme"],
                    "default": "normal",
                },
                "gauge_natural_rates": {
                    "type": "object",
                    "required": ["normal", "hard", "extreme"],
                    "properties": {
                        "normal": {"type": "number", "default": 0.006},
                        "hard": {"type": "number", "default": 0.008},
                        "extreme": {"type": "number", "default": 0.020},
                    },
                },
                "overload_duration": {"type": "number", "default": 25.0},
                "bombardment_damage": {"type": "number", "default": 1.0},
                "bombardment_tick_rate": {"type": "number", "default": 0.5},
                "electric_field_damage": {"type": "number", "default": 0.05},
                "electric_field_tick_rate": {"type": "number", "default": 0.36},
                "tracking_laser_player_damage": {"type": "number", "default": 0.15},
                "tracking_laser_gauge_increase": {"type": "number", "default": 0.10},
                "tracking_laser_gauge_decrease": {"type": "number", "default": 0.10},
                "small_arm_player_damage": {"type": "number", "default": 0.05},
                "small_arm_gauge_increase": {"type": "number", "default": 0.03},
                "small_arm_gauge_decrease": {"type": "number", "default": 0.03},
                "floor_discharge_warning_duration": {"type": "number", "default": 1.2},
                "floor_discharge_damage": {"type": "number", "default": 1.0},
            },
        },
    },
}


# =============================================================================
# Flax Dataclass Definitions
# =============================================================================

@struct.dataclass
class ClassicLaserParams:
    """Parameters for Classic Rotating Cross Laser."""
    angular_velocity: float = 0.523598775  # pi / 6 rad/s (30 deg/s)
    beam_thickness: float = 20.0
    beam_count: int = struct.field(pytree_node=False, default=4)
    damage: float = 1.0


@struct.dataclass
class DebrisTypeParams:
    """Individual debris hazard type properties."""
    type_id: int
    radius: float
    damage: float
    stun_duration: float
    fall_speed: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_id": int(self.type_id),
            "radius": float(self.radius),
            "damage": float(self.damage),
            "stun_duration": float(self.stun_duration),
            "fall_speed": float(self.fall_speed),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DebrisTypeParams:
        return cls(
            type_id=int(data["type_id"]),
            radius=float(data["radius"]),
            damage=float(data["damage"]),
            stun_duration=float(data["stun_duration"]),
            fall_speed=float(data["fall_speed"]),
        )


def _default_debris_types() -> Tuple[DebrisTypeParams, ...]:
    return (
        DebrisTypeParams(type_id=0, radius=15.0, damage=0.10, stun_duration=1.0, fall_speed=300.0),
        DebrisTypeParams(type_id=1, radius=25.0, damage=0.20, stun_duration=1.5, fall_speed=350.0),
        DebrisTypeParams(type_id=2, radius=35.0, damage=0.30, stun_duration=2.0, fall_speed=400.0),
        DebrisTypeParams(type_id=3, radius=50.0, damage=0.40, stun_duration=2.5, fall_speed=450.0),
    )


@struct.dataclass
class DebrisParams:
    """Parameters for Falling Debris Barrage."""
    max_debris: int = struct.field(pytree_node=False, default=30)
    spawn_interval_ticks: int = struct.field(pytree_node=False, default=15)
    types: Tuple[DebrisTypeParams, ...] = struct.field(default_factory=_default_debris_types)


@struct.dataclass
class GaugeNaturalRates:
    """Natural accumulation rate of Annihilation Gauge per second by difficulty."""
    normal: float = 0.006  # 0.6% / s
    hard: float = 0.008    # 0.8% / s
    extreme: float = 0.020 # 2.0% / s

    def to_dict(self) -> Dict[str, float]:
        return {
            "normal": float(self.normal),
            "hard": float(self.hard),
            "extreme": float(self.extreme),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GaugeNaturalRates:
        return cls(
            normal=float(data.get("normal", 0.006)),
            hard=float(data.get("hard", 0.008)),
            extreme=float(data.get("extreme", 0.020)),
        )


@struct.dataclass
class RemasteredParams:
    """Parameters for Remastered Lotus Mechanics (April 2024)."""
    enabled: bool = True
    difficulty: str = struct.field(pytree_node=False, default="normal")
    gauge_natural_rates: GaugeNaturalRates = GaugeNaturalRates()
    overload_duration: float = 25.0
    bombardment_damage: float = 1.0
    bombardment_tick_rate: float = 0.5
    electric_field_damage: float = 0.05
    electric_field_tick_rate: float = 0.36
    tracking_laser_player_damage: float = 0.15
    tracking_laser_gauge_increase: float = 0.10
    tracking_laser_gauge_decrease: float = 0.10
    small_arm_player_damage: float = 0.05
    small_arm_gauge_increase: float = 0.03
    small_arm_gauge_decrease: float = 0.03
    floor_discharge_warning_duration: float = 1.2
    floor_discharge_damage: float = 1.0


@struct.dataclass
class EnvParams:
    """Immutable Environment Parameters for Lotus Phase 1 Gymnax Simulator."""
    map_width: float = 1366.0
    map_height: float = 768.0
    core_x: float = 683.0
    core_y: float = 384.0
    core_radius: float = 120.0
    floor_y: float = 605.0
    wall_left: float = 50.0
    wall_right: float = 1316.0
    safe_zone_x: float = 1150.0
    player_width: float = 40.0
    player_height: float = 60.0
    player_speed: float = 400.0
    player_jump_impulse: float = -650.0
    gravity: float = 1800.0
    dt: float = 1.0 / 60.0

    classic_laser: ClassicLaserParams = ClassicLaserParams()
    debris_params: DebrisParams = DebrisParams()
    remastered_params: RemasteredParams = RemasteredParams()

    # Convenience properties for PROJECT.md interface contracts
    @property
    def core_pos(self) -> Tuple[Any, Any]:
        """Core anchor coordinate (x, y)."""
        return (self.core_x, self.core_y)

    @property
    def laser_omega(self) -> Any:
        """Rotating laser angular velocity in rad/s."""
        return self.classic_laser.angular_velocity

    @property
    def laser_thickness(self) -> Any:
        """Rotating laser beam thickness in pixels."""
        return self.classic_laser.beam_thickness

    @property
    def jump_impulse(self) -> Any:
        """Magnitude of player jump impulse."""
        return abs(self.player_jump_impulse)

    @property
    def max_debris(self) -> int:
        """Maximum number of concurrent falling debris hazards."""
        return self.debris_params.max_debris

    @property
    def gauge_gain_rate(self) -> Any:
        """Natural gauge gain rate for active difficulty."""
        diff = self.remastered_params.difficulty
        return getattr(self.remastered_params.gauge_natural_rates, diff, 0.008)

    @property
    def screen_width(self) -> Any:
        """Alias for map_width."""
        return self.map_width

    @property
    def screen_height(self) -> Any:
        """Alias for map_height."""
        return self.map_height

    @property
    def player_w(self) -> Any:
        """Alias for player_width."""
        return self.player_width

    @property
    def player_h(self) -> Any:
        """Alias for player_height."""
        return self.player_height

    @property
    def jump_velocity(self) -> Any:
        """Alias for player_jump_impulse."""
        return self.player_jump_impulse

    @property
    def laser_half_thickness(self) -> Any:
        """Half of laser beam thickness."""
        return self.classic_laser.beam_thickness / 2.0

    @property
    def laser_damage(self) -> Any:
        """Laser contact damage."""
        return self.classic_laser.damage

    @property
    def overload_duration(self) -> Any:
        """Overload mode duration in seconds."""
        return self.remastered_params.overload_duration

    @property
    def is_remastered(self) -> bool:
        """True if remastered mechanics are active."""
        return bool(self.remastered_params.enabled)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize EnvParams to a dictionary conforming to LOTUS_PHASE1_SCHEMA."""
        return {
            "map_width": float(self.map_width),
            "map_height": float(self.map_height),
            "core_x": float(self.core_x),
            "core_y": float(self.core_y),
            "core_radius": float(self.core_radius),
            "floor_y": float(self.floor_y),
            "wall_left": float(self.wall_left),
            "wall_right": float(self.wall_right),
            "safe_zone_x": float(self.safe_zone_x),
            "player_width": float(self.player_width),
            "player_height": float(self.player_height),
            "player_speed": float(self.player_speed),
            "player_jump_impulse": float(self.player_jump_impulse),
            "gravity": float(self.gravity),
            "dt": float(self.dt),
            "classic_laser": {
                "angular_velocity": float(self.classic_laser.angular_velocity),
                "beam_thickness": float(self.classic_laser.beam_thickness),
                "beam_count": int(self.classic_laser.beam_count),
                "damage": float(self.classic_laser.damage),
            },
            "debris_params": {
                "max_debris": int(self.debris_params.max_debris),
                "spawn_interval_ticks": int(self.debris_params.spawn_interval_ticks),
                "types": [t.to_dict() for t in self.debris_params.types],
            },
            "remastered_params": {
                "enabled": bool(self.remastered_params.enabled),
                "difficulty": str(self.remastered_params.difficulty),
                "gauge_natural_rates": self.remastered_params.gauge_natural_rates.to_dict(),
                "overload_duration": float(self.remastered_params.overload_duration),
                "bombardment_damage": float(self.remastered_params.bombardment_damage),
                "bombardment_tick_rate": float(self.remastered_params.bombardment_tick_rate),
                "electric_field_damage": float(self.remastered_params.electric_field_damage),
                "electric_field_tick_rate": float(self.remastered_params.electric_field_tick_rate),
                "tracking_laser_player_damage": float(self.remastered_params.tracking_laser_player_damage),
                "tracking_laser_gauge_increase": float(self.remastered_params.tracking_laser_gauge_increase),
                "tracking_laser_gauge_decrease": float(self.remastered_params.tracking_laser_gauge_decrease),
                "small_arm_player_damage": float(self.remastered_params.small_arm_player_damage),
                "small_arm_gauge_increase": float(self.remastered_params.small_arm_gauge_increase),
                "small_arm_gauge_decrease": float(self.remastered_params.small_arm_gauge_decrease),
                "floor_discharge_warning_duration": float(self.remastered_params.floor_discharge_warning_duration),
                "floor_discharge_damage": float(self.remastered_params.floor_discharge_damage),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvParams:
        """Construct EnvParams from a dictionary, supporting both strict schema and legacy aliases."""
        # Handle core coordinates alias
        core_x = float(data.get("core_x", 683.0))
        core_y = float(data.get("core_y", 384.0))
        if "core_pos" in data:
            pos = data["core_pos"]
            core_x, core_y = float(pos[0]), float(pos[1])

        # Handle jump impulse alias
        player_jump_impulse = float(data.get("player_jump_impulse", -650.0))
        if "jump_impulse" in data and "player_jump_impulse" not in data:
            val = float(data["jump_impulse"])
            player_jump_impulse = -abs(val)

        # Parse classic laser
        cl_data = data.get("classic_laser", {})
        laser_omega = float(data.get("laser_omega", cl_data.get("angular_velocity", 0.523598775)))
        laser_thickness = float(data.get("laser_thickness", cl_data.get("beam_thickness", 20.0)))
        beam_count = int(cl_data.get("beam_count", 4))
        laser_damage = float(cl_data.get("damage", 1.0))
        classic_laser = ClassicLaserParams(
            angular_velocity=laser_omega,
            beam_thickness=laser_thickness,
            beam_count=beam_count,
            damage=laser_damage,
        )

        # Parse debris params
        db_data = data.get("debris_params", {})
        max_debris = int(data.get("max_debris", db_data.get("max_debris", 30)))
        spawn_interval = int(db_data.get("spawn_interval_ticks", 15))
        if "types" in db_data and db_data["types"]:
            types_tuple = tuple(DebrisTypeParams.from_dict(t) for t in db_data["types"])
        else:
            types_tuple = _default_debris_types()
        debris_params = DebrisParams(
            max_debris=max_debris,
            spawn_interval_ticks=spawn_interval,
            types=types_tuple,
        )

        # Parse remastered params
        rm_data = data.get("remastered_params", {})
        gnr_data = rm_data.get("gauge_natural_rates", {})
        gauge_natural_rates = GaugeNaturalRates.from_dict(gnr_data)
        if "gauge_gain_rate" in data:
            # Override current difficulty rate if top-level override is given
            diff = rm_data.get("difficulty", "normal")
            if diff == "hard":
                gauge_natural_rates = GaugeNaturalRates(
                    normal=gauge_natural_rates.normal,
                    hard=float(data["gauge_gain_rate"]),
                    extreme=gauge_natural_rates.extreme,
                )
            elif diff == "extreme":
                gauge_natural_rates = GaugeNaturalRates(
                    normal=gauge_natural_rates.normal,
                    hard=gauge_natural_rates.hard,
                    extreme=float(data["gauge_gain_rate"]),
                )
            else:
                gauge_natural_rates = GaugeNaturalRates(
                    normal=float(data["gauge_gain_rate"]),
                    hard=gauge_natural_rates.hard,
                    extreme=gauge_natural_rates.extreme,
                )

        remastered_params = RemasteredParams(
            enabled=bool(rm_data.get("enabled", True)),
            difficulty=str(rm_data.get("difficulty", "normal")),
            gauge_natural_rates=gauge_natural_rates,
            overload_duration=float(data.get("overload_duration", rm_data.get("overload_duration", 25.0))),
            bombardment_damage=float(rm_data.get("bombardment_damage", 1.0)),
            bombardment_tick_rate=float(rm_data.get("bombardment_tick_rate", 0.5)),
            electric_field_damage=float(rm_data.get("electric_field_damage", 0.05)),
            electric_field_tick_rate=float(rm_data.get("electric_field_tick_rate", 0.36)),
            tracking_laser_player_damage=float(rm_data.get("tracking_laser_player_damage", 0.15)),
            tracking_laser_gauge_increase=float(rm_data.get("tracking_laser_gauge_increase", 0.10)),
            tracking_laser_gauge_decrease=float(rm_data.get("tracking_laser_gauge_decrease", 0.10)),
            small_arm_player_damage=float(rm_data.get("small_arm_player_damage", 0.05)),
            small_arm_gauge_increase=float(rm_data.get("small_arm_gauge_increase", 0.03)),
            small_arm_gauge_decrease=float(rm_data.get("small_arm_gauge_decrease", 0.03)),
            floor_discharge_warning_duration=float(rm_data.get("floor_discharge_warning_duration", 1.2)),
            floor_discharge_damage=float(rm_data.get("floor_discharge_damage", 1.0)),
        )

        return cls(
            map_width=float(data.get("map_width", 1366.0)),
            map_height=float(data.get("map_height", 768.0)),
            core_x=core_x,
            core_y=core_y,
            core_radius=float(data.get("core_radius", 120.0)),
            floor_y=float(data.get("floor_y", 605.0)),
            wall_left=float(data.get("wall_left", 50.0)),
            wall_right=float(data.get("wall_right", 1316.0)),
            safe_zone_x=float(data.get("safe_zone_x", 1150.0)),
            player_width=float(data.get("player_width", 40.0)),
            player_height=float(data.get("player_height", 60.0)),
            player_speed=float(data.get("player_speed", 400.0)),
            player_jump_impulse=player_jump_impulse,
            gravity=float(data.get("gravity", 1800.0)),
            dt=float(data.get("dt", 1.0 / 60.0)),
            classic_laser=classic_laser,
            debris_params=debris_params,
            remastered_params=remastered_params,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize EnvParams to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> EnvParams:
        """Construct EnvParams from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_env_params_dict(data: Dict[str, Any]) -> None:
    """Validate a parameter dictionary against the Draft 2020-12 schema.

    Raises:
        jsonschema.ValidationError: If data does not strictly satisfy the schema.
    """
    jsonschema.validate(instance=data, schema=LOTUS_PHASE1_SCHEMA)


def validate_env_params(params: EnvParams) -> None:
    """Validate an EnvParams instance against the Draft 2020-12 schema.

    Raises:
        jsonschema.ValidationError: If params serialized dictionary fails validation.
    """
    validate_env_params_dict(params.to_dict())


# =============================================================================
# Schema → Simulation Adapter (WZ SSOT → lotus_phase1.EnvParams bridge)
# =============================================================================

def to_sim_params(
    wz_params: EnvParams,
    *,
    wall_inset: float = 50.0,
    core_radius_shrink: float = 50.0,
    laser_margin: float = 2.0,
    mode: int = 0,
    player_max_hp: float = 100.0,
    player_duck_h: float = 35.0,
    invincible_duration: float = 1.0,
    max_steps_in_episode: int = 3600,
    debris_spawn_prob: float = 0.15,
    debris_player_eff_radius: float = 25.0,
    boss_max_hp: float = 1000.0,
    boss_shield_max: float = 200.0,
    boss_w: float = 160.0,
) -> Dict[str, Any]:
    """Convert WZ-sourced schema.EnvParams to lotus_phase1.EnvParams constructor kwargs.

    Applies collision padding / normalization rules:
    - ``wall_left = wz.wall_left + wall_inset``  (player-radius clearance)
    - ``wall_right = wz.wall_right - wall_inset``
    - ``core_radius = wz.core_radius - core_radius_shrink``  (inner clearance)
    - ``laser_half_thickness = wz.beam_thickness / 2.0 + laser_margin``
    - ``debris_min_vy / max_vy`` derived from debris type fall speeds
    - ``gauge_gain_rate`` resolved from difficulty enum

    Returns a dict of kwargs that can be unpacked into
    ``lotus_phase1.EnvParams(**to_sim_params(wz))``.
    """
    rm = wz_params.remastered_params

    # Resolve gauge rate from difficulty
    gauge_rate = getattr(rm.gauge_natural_rates, rm.difficulty, rm.gauge_natural_rates.hard)

    # Derive debris velocity range from type definitions
    debris_types = wz_params.debris_params.types
    if debris_types:
        fall_speeds = [float(t.fall_speed) for t in debris_types]
        debris_min_vy = min(fall_speeds)
        debris_max_vy = max(fall_speeds)
    else:
        debris_min_vy = 200.0
        debris_max_vy = 450.0

    return {
        "screen_width": float(wz_params.map_width),
        "screen_height": float(wz_params.map_height),
        "core_x": float(wz_params.core_x),
        "core_y": float(wz_params.core_y),
        "core_radius": float(wz_params.core_radius) - core_radius_shrink,
        "floor_y": float(wz_params.floor_y),
        "wall_left": float(wz_params.wall_left) + wall_inset,
        "wall_right": float(wz_params.wall_right) - wall_inset,
        "safe_zone_x": float(wz_params.safe_zone_x),
        # Player
        "player_w": float(wz_params.player_width),
        "player_h": float(wz_params.player_height),
        "player_duck_h": player_duck_h,
        "player_speed": float(wz_params.player_speed),
        "jump_velocity": float(wz_params.player_jump_impulse),
        "gravity": float(wz_params.gravity),
        "dt": float(wz_params.dt),
        "player_max_hp": player_max_hp,
        "invincible_duration": invincible_duration,
        # Classic Laser
        "laser_omega": float(wz_params.classic_laser.angular_velocity),
        "laser_half_thickness": float(wz_params.classic_laser.beam_thickness) / 2.0 + laser_margin,
        "laser_damage": float(wz_params.classic_laser.damage),
        "laser_max_length": 1200.0,
        # Debris
        "debris_spawn_prob": debris_spawn_prob,
        "debris_min_vy": debris_min_vy,
        "debris_max_vy": debris_max_vy,
        "debris_player_eff_radius": debris_player_eff_radius,
        # Episode
        "max_steps_in_episode": max_steps_in_episode,
        # Remastered
        "mode": mode,
        "gauge_gain_rate": float(gauge_rate),
        "overload_duration": float(rm.overload_duration),
        "boss_max_hp": boss_max_hp,
        "boss_shield_max": boss_shield_max,
        "boss_w": boss_w,
        "electric_field_damage": float(rm.electric_field_damage),
        "tracking_laser_player_dmg": float(rm.tracking_laser_player_damage),
        "tracking_laser_gauge_gain": float(rm.tracking_laser_gauge_increase),
        "tracking_laser_gauge_reduction": float(rm.tracking_laser_gauge_decrease),
        "arm_slam_player_dmg": float(rm.small_arm_player_damage),
        "arm_slam_gauge_gain": float(rm.small_arm_gauge_increase),
        "arm_slam_gauge_reduction": float(rm.small_arm_gauge_decrease),
    }

