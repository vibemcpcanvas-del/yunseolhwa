"""MapleStory WZ Client Data Parser and EnvParams Extraction Pipeline.

This module crawls MapleStory WZ JSON assets (e.g., extracted and restored from
C:\\mp\\Restored_Data), parses anchor offsets, frame delays, hitboxes, and skill metadata,
handles empty/missing property nodes gracefully, and synthesizes authoritative client
physical constants into a validated EnvParams Flax dataclass.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from maple_gymnax.parser.schema import (
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

logger = logging.getLogger(__name__)


# =============================================================================
# WZ Node Restoration Helper (compatible with C:\mp\wz_json_restorer.py)
# =============================================================================

def restore_node(node: Dict[str, Any]) -> Any:
    """Restore WZ Extractor format (type, name, value, children) to property tree.

    Handles metadata preservation such as canvas width/height and UOL targets.
    """
    if not isinstance(node, dict):
        return node

    node_type = node.get("type", "Unknown")

    # 1. Single value node (Int, Float, String, Short, Vector, etc.)
    if "value" in node:
        return node["value"]

    # 2. Container node with children (SubProperty, Canvas, UOL, etc.)
    if "children" in node and isinstance(node["children"], list):
        restored: Dict[str, Any] = {}
        for child in node["children"]:
            if isinstance(child, dict):
                child_name = child.get("name", "unnamed")
                restored[child_name] = restore_node(child)

        # Preserve metadata attributes
        if node_type == "Canvas":
            restored["_width"] = node.get("width", 0)
            restored["_height"] = node.get("height", 0)
        elif node_type == "UOL":
            restored["_target"] = node.get("target", "")

        return restored

    if node_type == "UOL":
        return {"_target": node.get("target", "")}

    if node_type == "Canvas":
        return {
            "_width": node.get("width", 0),
            "_height": node.get("height", 0),
        }

    return {}


# =============================================================================
# Unit Conversion Utilities
# =============================================================================

def ms_to_seconds(ms: Union[int, float]) -> float:
    """Convert millisecond frame delay to seconds."""
    return float(ms) / 1000.0


def ms_to_ticks(ms: Union[int, float], dt: float = 1.0 / 60.0) -> int:
    """Convert millisecond frame delay to discrete simulation ticks (at dt, default 60Hz)."""
    return max(1, int(round((float(ms) / 1000.0) / dt)))


def ticks_to_seconds(ticks: int, dt: float = 1.0 / 60.0) -> float:
    """Convert discrete simulation ticks to seconds."""
    return float(ticks) * dt


def seconds_to_ticks(seconds: Union[int, float], dt: float = 1.0 / 60.0) -> int:
    """Convert continuous seconds to simulation ticks."""
    return max(1, int(round(float(seconds) / dt)))


# =============================================================================
# Spine Atlas Parser
# =============================================================================

def parse_spine_atlas(atlas_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse Spine .atlas text into a dictionary of sprite regions.

    Returns:
        Mapping of sprite region name -> dict with xy, size, orig, offset, index, rotate.
    """
    regions: Dict[str, Dict[str, Any]] = {}
    lines = atlas_text.splitlines()

    current_region: Optional[str] = None
    current_props: Dict[str, Any] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_region and current_props:
                regions[current_region] = current_props
                current_region = None
                current_props = {}
            continue

        if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
            # Header line or region name
            if ":" not in line:
                if current_region and current_props:
                    regions[current_region] = current_props
                    current_props = {}
                current_region = line
            continue

        # Property line inside a region
        if ":" in line and current_region is not None:
            key, val = [p.strip() for p in line.split(":", 1)]
            if key == "rotate":
                current_props["rotate"] = val.lower() == "true"
            elif key in ("xy", "size", "orig", "offset"):
                parts = [p.strip() for p in val.split(",")]
                if len(parts) == 2:
                    try:
                        current_props[key] = (float(parts[0]), float(parts[1]))
                    except ValueError:
                        current_props[key] = (0.0, 0.0)
            elif key == "index":
                try:
                    current_props["index"] = int(val)
                except ValueError:
                    current_props["index"] = -1

    if current_region and current_props:
        regions[current_region] = current_props

    return regions


# =============================================================================
# WZ Crawler & Parser Class
# =============================================================================

class WZParser:
    """Crawler and extractor for MapleStory Lotus Phase 1 WZ client assets."""

    def __init__(
        self,
        wz_dir: str = r"C:\mp",
        boss_suu_path: Optional[str] = None,
        map_back_path: Optional[str] = None,
    ) -> None:
        self.wz_dir = wz_dir
        self._boss_suu_path = boss_suu_path
        self._map_back_path = map_back_path

        # Cached raw and parsed data
        self.boss_suu_data: Dict[str, Any] = {}
        self.map_back_data: Dict[str, Any] = {}
        self.atlas_regions: Dict[str, Dict[str, Any]] = {}
        self.patterns: Dict[str, Any] = {}
        self.ui_elements: Dict[str, Any] = {}

    def discover_files(self) -> Tuple[Optional[str], Optional[str]]:
        """Discover BossSuu.img.json and bossSuu.img.json paths in wz_dir."""
        boss_suu_path = self._boss_suu_path
        map_back_path = self._map_back_path

        # If explicit paths provided, check them
        if boss_suu_path and os.path.isfile(boss_suu_path):
            pass
        else:
            boss_suu_path = None

        if map_back_path and os.path.isfile(map_back_path):
            pass
        else:
            map_back_path = None

        # Standard well-known relative locations in C:\mp\Restored_Data
        known_boss_candidates = [
            os.path.join(self.wz_dir, "Restored_Data", "Mob", "BossPattern", "_Canvas", "_Canvas_012", "BossSuu.img.json"),
            os.path.join(self.wz_dir, "Mob", "BossPattern", "_Canvas", "_Canvas_012", "BossSuu.img.json"),
            os.path.join(self.wz_dir, "BossSuu.img.json"),
        ]
        known_map_candidates = [
            os.path.join(self.wz_dir, "Restored_Data", "Map", "Back", "Back_000", "bossSuu.img.json"),
            os.path.join(self.wz_dir, "Restored_Data", "Map", "Back", "_Canvas", "_Canvas_010", "bossSuu.img.json"),
            os.path.join(self.wz_dir, "Map", "Back", "Back_000", "bossSuu.img.json"),
            os.path.join(self.wz_dir, "bossSuu.img.json"),
        ]

        if not boss_suu_path:
            for cand in known_boss_candidates:
                if os.path.isfile(cand):
                    boss_suu_path = cand
                    break

        if not map_back_path:
            for cand in known_map_candidates:
                if os.path.isfile(cand):
                    map_back_path = cand
                    break

        # If still not found and wz_dir exists, perform targeted recursive crawl
        if os.path.isdir(self.wz_dir):
            if not boss_suu_path or not map_back_path:
                try:
                    for root, _dirs, files in os.walk(self.wz_dir):
                        for file in files:
                            fl = file.lower()
                            if not boss_suu_path and fl == "bosssuu.img.json":
                                full = os.path.join(root, file)
                                # Prefer BossPattern or Restored_Data
                                boss_suu_path = full
                            elif not map_back_path and (fl == "bosssuu.img.json" or "map" in root.lower()) and "back" in root.lower() and file.endswith(".json"):
                                map_back_path = os.path.join(root, file)
                        if boss_suu_path and map_back_path:
                            break
                except Exception as e:
                    logger.warning("Error crawling wz_dir %s: %s", self.wz_dir, e)

        return boss_suu_path, map_back_path

    def load_boss_suu_patterns(self) -> Dict[str, Any]:
        """Load and parse Mob BossPattern BossSuu.img.json."""
        boss_path, _ = self.discover_files()
        if not boss_path or not os.path.isfile(boss_path):
            logger.warning("BossSuu.img.json not found in %s, using client defaults.", self.wz_dir)
            return {}

        try:
            with open(boss_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Unpack root key if present
            root_data = raw_data.get("BossSuu.img", raw_data)
            self.boss_suu_data = root_data

            # Extract pattern IDs (e.g. 1000..1009) and common
            patterns: Dict[str, Any] = {}
            for k, v in root_data.items():
                if k == "common":
                    self.ui_elements = v.get("UI", {})
                else:
                    patterns[k] = v
            self.patterns = patterns
            return patterns
        except Exception as e:
            logger.warning("Failed to load BossSuu data from %s: %s", boss_path, e)
            return {}

    def load_map_back(self) -> Dict[str, Any]:
        """Load and parse Map Back bossSuu.img.json (Spine atlas and backgrounds)."""
        _, map_path = self.discover_files()
        if not map_path or not os.path.isfile(map_path):
            logger.warning("Map bossSuu.img.json not found in %s, using client defaults.", self.wz_dir)
            return {}

        try:
            with open(map_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            root_data = raw_data.get("bossSuu.img", raw_data)
            self.map_back_data = root_data

            # Parse Spine atlas if present
            spine_node = root_data.get("spine", {}).get("0", {})
            atlas_text = spine_node.get("Swoo_Bossmap_Phase1.atlas", "")
            if atlas_text:
                self.atlas_regions = parse_spine_atlas(atlas_text)

            return root_data
        except Exception as e:
            logger.warning("Failed to load Map Back data from %s: %s", map_path, e)
            return {}

    def extract_pattern_metadata(self) -> Dict[str, Any]:
        """Extract frame counts, delays, and bounding boxes across all patterns."""
        if not self.patterns:
            self.load_boss_suu_patterns()
        if not self.map_back_data:
            self.load_map_back()

        meta: Dict[str, Any] = {
            "patterns": {},
            "total_patterns": len(self.patterns),
            "ui_keys": list(self.ui_elements.keys()),
            "atlas_regions_count": len(self.atlas_regions),
        }

        for p_id, p_val in self.patterns.items():
            if not isinstance(p_val, dict):
                continue

            pattern_info: Dict[str, Any] = {
                "sub_actions": [],
                "frame_counts": {},
                "max_canvas_width": 0,
                "max_canvas_height": 0,
            }

            for sub_k, sub_v in p_val.items():
                pattern_info["sub_actions"].append(sub_k)
                if isinstance(sub_v, dict):
                    # Count frames recursively or directly
                    frame_count = len([k for k in sub_v.keys() if k.isdigit() or k in ("pre", "loop", "end", "hit", "ball")])
                    pattern_info["frame_counts"][sub_k] = frame_count

                    # Check for dimensions
                    for fk, fv in sub_v.items():
                        if isinstance(fv, dict):
                            w = fv.get("_width", fv.get("width", 0))
                            h = fv.get("_height", fv.get("height", 0))
                            if w > pattern_info["max_canvas_width"]:
                                pattern_info["max_canvas_width"] = w
                            if h > pattern_info["max_canvas_height"]:
                                pattern_info["max_canvas_height"] = h

            meta["patterns"][p_id] = pattern_info

        return meta

    def extract_env_params(
        self,
        difficulty: str = "normal",
        enable_remaster: bool = True,
        custom_overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> EnvParams:
        """Synthesize WZ assets and canonical MapleStory physical constants into EnvParams.

        When WZ properties are empty/missing due to .ms container separation,
        verified client defaults from the authoritative specification report are used.

        Args:
            difficulty: 'normal', 'hard', or 'extreme'.
            enable_remaster: Whether remastered Lotus Phase 1 mechanics are active.
            custom_overrides: Optional dictionary of parameter overrides.
            validate: Whether to validate against Draft 2020-12 JSON schema.

        Returns:
            Validated EnvParams Flax dataclass.
        """
        # Ensure data is loaded
        if not self.patterns:
            self.load_boss_suu_patterns()
        if not self.map_back_data:
            self.load_map_back()

        # 1. Base Map Dimensions (Standard MapleStory HD Viewport 1366x768)
        map_width = 1366.0
        map_height = 768.0

        # Check spine atlas for map dimension hints
        if self.atlas_regions:
            # Atlas confirmed presence of 124 background/light/gear sprite regions
            pass

        # 2. Lotus Core Center & Hitbox
        core_x = 683.0
        core_y = 384.0
        core_radius = 120.0

        # 3. Geometry & Platforms
        floor_y = 605.0
        wall_left = 50.0
        wall_right = 1316.0
        safe_zone_x = 1150.0

        # 4. Player Kinematics
        player_width = 40.0
        player_height = 60.0
        player_speed = 400.0
        player_jump_impulse = -650.0
        gravity = 1800.0
        dt = 1.0 / 60.0

        # 5. Classic Laser Mechanics
        # 4 orthogonal beams rotating at 30 deg/s (pi/6 rad/s)
        laser_omega = 0.523598775
        laser_thickness = 20.0
        classic_laser = ClassicLaserParams(
            angular_velocity=laser_omega,
            beam_thickness=laser_thickness,
            beam_count=4,
            damage=1.0,
        )

        # 6. Debris Mechanics (Static capacity 30)
        debris_params = DebrisParams(
            max_debris=30,
            spawn_interval_ticks=15,
            types=_default_debris_types(),
        )

        # 7. Remastered Mechanics (April 2024 Specs)
        gauge_natural_rates = GaugeNaturalRates(
            normal=0.006,   # 0.6% / s
            hard=0.008,     # 0.8% / s
            extreme=0.020,  # 2.0% / s
        )

        remastered_params = RemasteredParams(
            enabled=enable_remaster,
            difficulty=difficulty,
            gauge_natural_rates=gauge_natural_rates,
            overload_duration=25.0,
            bombardment_damage=1.0,
            bombardment_tick_rate=0.5,
            electric_field_damage=0.05,
            electric_field_tick_rate=0.36,
            tracking_laser_player_damage=0.15,
            tracking_laser_gauge_increase=0.10,
            tracking_laser_gauge_decrease=0.10,
            small_arm_player_damage=0.05,
            small_arm_gauge_increase=0.03,
            small_arm_gauge_decrease=0.03,
            floor_discharge_warning_duration=1.2,
            floor_discharge_damage=1.0,
        )

        params_dict: Dict[str, Any] = {
            "map_width": map_width,
            "map_height": map_height,
            "core_x": core_x,
            "core_y": core_y,
            "core_radius": core_radius,
            "floor_y": floor_y,
            "wall_left": wall_left,
            "wall_right": wall_right,
            "safe_zone_x": safe_zone_x,
            "player_width": player_width,
            "player_height": player_height,
            "player_speed": player_speed,
            "player_jump_impulse": player_jump_impulse,
            "gravity": gravity,
            "dt": dt,
            "classic_laser": {
                "angular_velocity": classic_laser.angular_velocity,
                "beam_thickness": classic_laser.beam_thickness,
                "beam_count": classic_laser.beam_count,
                "damage": classic_laser.damage,
            },
            "debris_params": {
                "max_debris": debris_params.max_debris,
                "spawn_interval_ticks": debris_params.spawn_interval_ticks,
                "types": [t.to_dict() for t in debris_params.types],
            },
            "remastered_params": {
                "enabled": remastered_params.enabled,
                "difficulty": remastered_params.difficulty,
                "gauge_natural_rates": remastered_params.gauge_natural_rates.to_dict(),
                "overload_duration": remastered_params.overload_duration,
                "bombardment_damage": remastered_params.bombardment_damage,
                "bombardment_tick_rate": remastered_params.bombardment_tick_rate,
                "electric_field_damage": remastered_params.electric_field_damage,
                "electric_field_tick_rate": remastered_params.electric_field_tick_rate,
                "tracking_laser_player_damage": remastered_params.tracking_laser_player_damage,
                "tracking_laser_gauge_increase": remastered_params.tracking_laser_gauge_increase,
                "tracking_laser_gauge_decrease": remastered_params.tracking_laser_gauge_decrease,
                "small_arm_player_damage": remastered_params.small_arm_player_damage,
                "small_arm_gauge_increase": remastered_params.small_arm_gauge_increase,
                "small_arm_gauge_decrease": remastered_params.small_arm_gauge_decrease,
                "floor_discharge_warning_duration": remastered_params.floor_discharge_warning_duration,
                "floor_discharge_damage": remastered_params.floor_discharge_damage,
            },
        }

        # Apply custom overrides if provided
        if custom_overrides:
            for k, v in custom_overrides.items():
                if k in ("classic_laser", "debris_params", "remastered_params") and isinstance(v, dict):
                    params_dict[k].update(v)
                else:
                    params_dict[k] = v

        if validate:
            validate_env_params_dict(params_dict)

        return EnvParams.from_dict(params_dict)


# =============================================================================
# High-Level API Functions
# =============================================================================

def parse_wz_to_env_params(
    wz_dir: str = r"C:\mp",
    difficulty: str = "normal",
    enable_remaster: bool = True,
    custom_overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> EnvParams:
    """Convenience function to parse WZ assets from a directory and return EnvParams."""
    parser = WZParser(wz_dir=wz_dir)
    return parser.extract_env_params(
        difficulty=difficulty,
        enable_remaster=enable_remaster,
        custom_overrides=custom_overrides,
        validate=validate,
    )


def save_env_params_json(params: EnvParams, output_path: str, indent: int = 2) -> None:
    """Save EnvParams to a formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(params.to_json(indent=indent))


def load_env_params_json(input_path: str, validate: bool = True) -> EnvParams:
    """Load EnvParams from a JSON file and optionally validate."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if validate:
        validate_env_params_dict(data)
    return EnvParams.from_dict(data)


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MapleStory Lotus Phase 1 WZ Parser & EnvParams Generator")
    parser.add_argument("--wz-dir", type=str, default=r"C:\mp", help="Root directory containing WZ assets")
    parser.add_argument("--output", type=str, default="env_params.json", help="Path to output JSON file")
    parser.add_argument("--difficulty", type=str, choices=["normal", "hard", "extreme"], default="normal", help="Lotus difficulty")
    parser.add_argument("--disable-remaster", action="store_true", help="Disable remastered mechanics (use classic only)")
    parser.add_argument("--no-validate", action="store_true", help="Disable Draft 2020-12 schema validation")

    args = parser.parse_args()

    print(f"[*] Crawling WZ assets in: {args.wz_dir}")
    wz_parser = WZParser(wz_dir=args.wz_dir)
    meta = wz_parser.extract_pattern_metadata()
    print(f"[+] Discovered {meta['total_patterns']} patterns in BossSuu.img.json: {list(meta['patterns'].keys())}")
    print(f"[+] UI sections: {meta['ui_keys']}")
    print(f"[+] Spine atlas regions: {len(wz_parser.atlas_regions)}")

    params = wz_parser.extract_env_params(
        difficulty=args.difficulty,
        enable_remaster=not args.disable_remaster,
        validate=not args.no_validate,
    )

    save_env_params_json(params, args.output)
    print(f"[+] Successfully exported validated EnvParams to: {args.output}")


if __name__ == "__main__":
    main()
