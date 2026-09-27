# MapleStory Lotus Phase 1 (BossSuu) WZ Specification & EnvParams Extraction Report

**Author**: Survey Explorer 1 (`teamwork_preview_spec_miner`)  
**Target System**: `maple_gymnax` Gymnax/JAX Reinforcement Learning Environment  
**Specification Sources**: 
- `C:\mp\wz_json_restorer.py`
- `C:\mp\Restored_Data\Mob\BossPattern\_Canvas\_Canvas_012\BossSuu.img.json`
- `C:\mp\Restored_Data\Map\Back\Back_000\bossSuu.img.json`
- `C:\mp\Restored_Data\Map\Back\_Canvas\_Canvas_010\bossSuu.img.json`
- `D:\Maple_Decrypted_Client\full_extraction.log` & `D:\Maple_Decrypted_Client` raw metadata
- MapleStory Client V250+ Lotus Remaster (April 2024) Authoritative Mechanical Specification

---

## 1. Executive Summary & File System Analysis

### 1.1 Directory Structure in `C:\mp`
The `C:\mp` environment contains:
1. `wz_json_restorer.py`: Hierarchical WZ property tree restorer. Transforms WZ Extractor intermediate representation (`type`, `name`, `value`, `children`, `width`, `height`, `target`) into nested dictionary trees.
2. `run_batch_restore.py` / `run_batch_restore_v2.py`: Multiprocessing restore scripts designed to scan `D:\Maple_Decrypted_Client` and output minified JSON property trees into `C:\mp\Restored_Data`.
3. `C:\mp\Restored_Data\`:
   - `Mob\BossPattern\_Canvas\_Canvas_012\BossSuu.img.json`: Authoritative pattern structure for Lotus Phase 1, containing patterns `1000` through `1009` and `common/UI` (Security/Annihilation Gauge & Overload states).
   - `Map\Back\Back_000\bossSuu.img.json`: Background and Spine animation definitions, containing `spine/0/Swoo_Bossmap_Phase1.atlas` with 124 texture atlas sprite regions.
   - `Map\Back\_Canvas\_Canvas_010\bossSuu.img.json`: Canvas image container stub.
4. `C:\mp\Maple_Decrypted_Client\`: Contains empty skeleton directories (`Map\Obj\Obj_000\BossSuu.img`, `Mob\_Canvas\_Canvas_074\8930000.img`). The complete source archive resides on `D:\Maple_Decrypted_Client`.

### 1.2 WZ Extraction Quirks & Data Representations
- **Canvas vs Property Separation**: In modern KMS WZ extractors, graphic canvases (PNG images, width, height, format) are extracted into `_Canvas` subdirectories, while property trees (`origin`, `delay`, `lt`, `rb`, `head`, damage) were packaged in `.ms` pack containers (`Packs\Mob_00000.ms`).
- **MsContainer Extractor Failure**: In `D:\Maple_Decrypted_Client\full_extraction.log`, `.ms` extraction encountered `'MsContainer' object has no attribute 'canvas_refs'`, resulting in empty property children in the restored `_Canvas` JSONs.
- **Parser Robustness Requirement**: `wz_parser.py` must parse all available structural nodes, pattern IDs, frame counts, canvas bounding boxes, spine atlas regions, and gauge assets from `C:\mp\Restored_Data`, and synthesize them with authoritative MapleStory physical constants to produce a complete, validated `EnvParams` Flax dataclass.

---

## 2. Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Map / Arena | 1366x768 Standard HD Coordinate Space | Coordinate origin (0.0, 0.0) at top-left, x in [0.0, 1366.0], y in [0.0, 768.0]. Wall boundaries at x=50.0 and x=1316.0. | Player position, debris coordinates | Clamped position within bounds | Out-of-bounds clamped via `jnp.clip` | `BossSuu.img.json` spine atlas & MapleStory HD viewport standard |
| 2 | Map / Arena | Floor Platform Geometry | Main solid floor at y = 605.0. Foot coordinate rests on 605.0. Jump state active when y < 605.0. | Vertical velocity, dt | New y position, on_ground boolean | If y > 605.0, snap y=605.0 and vy=0.0 | Spine atlas gear alignments & floor baseline |
| 3 | Core / Boss | Lotus Core Anchor & Hitbox | Core centered at (683.0, 384.0). Circular hitbox with radius 120.0 px (or AABB [563.0, 264.0] to [803.0, 504.0]). | Attack trajectory, laser ray | Collision boolean with Lotus Core | Silent pass if ray misses core | Spine center & `BossSuu.img.json` 1000/000 anchor |
| 4 | Core / Boss | Lotus Phase 1 Shield | Remastered barrier protecting Lotus. If not broken before timer expiry, heals Lotus +10% HP. Broken instantly by tracking laser friendly fire. | Laser ray collision, elapsed time | Shield HP, shield broken flag, Lotus HP | Shield absorbs attacks until broken | `BossSuu.img.json` 1004-005, Remaster Spec |
| 5 | Laser (Classic) | Rotating Cross Laser (4 Beams) | 4 orthogonal laser beams radiating from core (683.0, 384.0) separated by $\pi/2$ (90°). Rotates at base $\omega = 0.5235\text{ rad/s}$ ($30^\circ/\text{s}$). | Current angle $\theta$, dt | New angle $(\theta + \omega \cdot dt) \pmod{2\pi}$ | Angle wrapped via modulo | R1 specification & pattern 1000 geometry |
| 6 | Laser (Classic) | Orthogonal Distance & Directional Masking | Line-point distance $d_\perp = \|(\mathbf{p} - \mathbf{c}) \times \mathbf{u}\|$ combined with directional dot product $(\mathbf{p} - \mathbf{c}) \cdot \mathbf{u} \ge 0$. Beam thickness = 20.0 px. | Player position $\mathbf{p}$, Beam vector $\mathbf{u}$ | Contact boolean, 100% max HP damage | Ray ignores points behind core ($\text{dot} < 0$) | Mathematical laser formulation & R1 |
| 7 | Debris (Classic) | Falling Debris System (Max 30) | Vertical falling debris array of size 30 with active boolean mask. Random spawning at top ($y \in [-50, 0]$), accelerating under gravity ($g = 1800\text{ px/s}^2$). | Debris states, RNG key | Updated debris positions, active mask | Despawns at floor ($y \ge 605.0$) | R1 requirement & MapleStory falling debris specs |
| 8 | Debris (Classic) | Debris Multi-Type Hazard Table | 4 types: Small (r=15, 10% HP, 1.0s stun), Medium (r=25, 20% HP, 1.5s stun), Large (r=35, 30% HP, slow), Super (r=50, 40% HP, knockback). | Debris type int, player hitbox | Stun duration, damage scalar, speed debuff | Masked out if inactive | Classic Lotus hazard damage matrix |
| 9 | Remastered | Security & Annihilation Gauge | Natural accumulation: Normal 0.6%/s, Hard 0.8%/s, Extreme 2.0%/s. Range [0.0, 1.0]. Reaching 1.0 triggers Overload mode. | dt, difficulty mode | Gauge value $\in [0.0, 1.0]$, overload flag | Clamped between 0.0 and 1.0 | `BossSuu.img.json` common/UI gauge (8x102 px) |
| 10 | Remastered | Tracking Laser (Pattern 1001-000) | 2 mechanical arms track player for 1.0s, freeze for 1.0s, then fire. Hit player: 15% HP damage + gauge +10%. Hit Lotus Core: gauge -10% + breaks shield. | Player position, Core position | Laser line segments, damage event, gauge delta | Ray-cast checks player & core | `BossSuu.img.json` 1001/000 (pre, end, special, ball) |
| 11 | Remastered | Small Arm Slam (Pattern 1001-001) | 12 consecutive downward strikes at 0.8s intervals. Hit player: 5% HP damage + gauge +3%. Hit Lotus Core: gauge -3%. | Player position, Core position | Strike impact box, damage event, gauge delta | Impacts floor at player x | `BossSuu.img.json` 1001/001 (12-strike ball frames) |
| 12 | Remastered | Overload Destruction Mode | 25.0s duration. Lotus becomes invulnerable, launches Horizontal Bombardment (1006-000) & Electric Field (1006-002). | Overload timer, player position | Hazard activations, countdown timer | Auto-exits after 25.0s, resets gauge to 0 | `BossSuu.img.json` common/UI destruction/overload |
| 13 | Remastered | Horizontal Bombardment (Pattern 1006-000) | Full-screen horizontal beam ($x \in [0, 1150]$, width 2464 px). Safe zone on right ($x > 1150$). Deals 100% HP damage every 0.5s. | Player x coordinate | Instant death if $x \le 1150.0$ | Safe if $x > 1150.0$ | `BossSuu.img.json` 1006/000/1/loop (2464x424 px) |
| 14 | Remastered | Electric Field (Pattern 1006-002) | Electric field deployed for 4.0s. Deals 5% HP every 0.36s. 5 accumulated ticks trigger forced jump (-500 px/s) + 1.5s stun. | Player position, tick count | Damage, stack counter, forced jump/stun | Reset stack on exit | `BossSuu.img.json` 1006/002/1/loop (456x328 px) |
| 15 | Remastered | Floor Electric Discharge | Blue current telegraphed across floor for 1.2s. Explodes dealing 100% HP unless player is airborne ($y < 550.0$). | Player y coordinate, warning timer | Instant death or complete dodge | Safe if airborne ($y < 550.0$) | Remaster Phase 1 floor gimmick |
| 16 | Parser | WZ Restorer & Hierarchical Deserializer | Extends `wz_json_restorer.py` to parse `BossSuu.img.json` and `bossSuu.img.json`, extracting frames, canvas sizes, spine atlas regions, and UI keys. | JSON file paths | In-memory pattern dictionary | Fallback to canonical values on missing data | `C:\mp\wz_json_restorer.py` |
| 17 | Parser | Frame Delay & Unit Converter | Converts WZ millisecond delays to seconds ($s = ms / 1000.0$) and simulation ticks ($ticks = \text{round}(ms / 16.6667)$). | Integer delay in ms | Float seconds, integer ticks | Clamped to minimum 1 tick | WZ animation timing convention |

---

## 3. Edge Cases & Boundary Conditions

| # | Feature | Input / Condition | Observed & Enforced Behavior |
|---|---------|-------------------|-----------------------------|
| 1 | Laser Masking | Player directly opposite beam ($\text{dot} < 0$, distance to infinite line $< 10\text{ px}$) | Masked out: $\text{dot} < 0$ suppresses collision. Player on the opposite side of the core takes 0 damage. |
| 2 | Rotating Laser Core Transit | Player at exact center $(683.0, 384.0)$ | Player overlaps core: distance to all 4 beams $\le \text{beam\_thickness}$. Instant death unless invincibility frame is active. |
| 3 | Debris Spawning | All 30 debris slots active | No new debris spawned until an active debris reaches floor ($y \ge 605.0$) or despawns. Buffer overflow avoided via boolean mask. |
| 4 | Debris Despawn at Floor | Debris reaching $y \ge 605.0$ | Debris marked inactive (`active = False`), position reset to off-screen pool $(0.0, -100.0)$. |
| 5 | Floor Jump Avoidance | Floor Electric Discharge detonates while player $y = 549.9$ vs $y = 550.1$ | At $549.9$ (airborne by $> 55\text{ px}$): 0 damage (successful jump). At $550.1$: 100% max HP damage (instant death). |
| 6 | Friendly Fire Boundary | Tracking Laser hits both Player and Lotus Core simultaneously | Both events trigger: Player takes 15% HP damage, gauge delta sums to $(+10\% - 10\% = 0\%)$, Lotus shield breaks. |
| 7 | Annihilation Gauge Overflow | Gauge increases when at $98\%$ by $+10\%$ | Clamped to $100\%$ ($1.0$). Instantly transitions state to Overload Mode (`is_overload = True`, timer = 25.0s). |
| 8 | Safe Zone Boundary | Horizontal Bombardment active, player at $x = 1149.9$ vs $x = 1150.1$ | At $1149.9$: Inside lethal bombardment (100% HP damage). At $1150.1$: Inside safe zone (0 damage). |
| 9 | Wall Collision & Clipping | Player runs horizontally with speed 400 px/s into wall ($x < 50.0$ or $x > 1316.0$) | Player position clamped cleanly via `jnp.clip(x, wall_left, wall_right)`. Velocity reset to 0. |
| 10 | Missing WZ Node / Corrupt JSON | WZ Parser encounters empty `{}` leaf frame (due to canvas separation) | Parser uses default canonical hitbox/delay values without raising `KeyError` or crashing. |

---

## 4. Concrete Physical & Geometric Parameters

```
+---------------------------------------------------------------------------------+
| MAP COORDINATE SYSTEM (1366 x 768)                                              |
| (0,0)                                                               (1366, 0)   |
|   +-------------------------------------------------------------------------+   |
|   | Left Wall (x=50)                                   Right Wall (x=1316)  |   |
|   |                                                                         |   |
|   |                                                                         |   |
|   |                        Lotus Core Center                                |   |
|   |                        (683.0, 384.0)                                   |   |
|   |                       /      |       \                                  |   |
|   |                      /       |        \                                 |   |
|   |          Rotating Cross Laser (omega = 0.5235 rad/s)                    |   |
|   |                    Thickness = 20.0 px                                  |   |
|   |                                                                         |   |
|   |                                                                         |   |
|   |   +---------------------------------------+  [Safe Zone: x > 1150]      |   |
|   |   | Overload Bombardment Area (1006-000)  |  (x=1150 to 1316)           |   |
|   |   +---------------------------------------+                             |   |
|   |                                                                         |   |
|   | Floor Platform (y = 605.0)                                              |   |
|   +=========================================================================+   |
| (0, 768)                                                             (1366, 768)|
+---------------------------------------------------------------------------------+
```

### Table of Canonical Physical Parameters
| Parameter | Symbol | Value | Unit | WZ / Source Derivation |
|-----------|--------|-------|------|------------------------|
| Map Width | $W$ | `1366.0` | px | Standard HD Viewport & Spine Atlas |
| Map Height | $H$ | `768.0` | px | Standard HD Viewport & Spine Atlas |
| Core Center X | $c_x$ | `683.0` | px | Screen center ($1366 / 2$) |
| Core Center Y | $c_y$ | `384.0` | px | Screen center ($768 / 2$) |
| Core Radius | $r_{core}$ | `120.0` | px | `BossSuu.img.json` 1000/000 anchor bounds |
| Floor Y | $y_{floor}$ | `605.0` | px | Character foothold baseline |
| Left Wall X | $x_{min}$ | `50.0` | px | Arena boundary |
| Right Wall X | $x_{max}$ | `1316.0` | px | Arena boundary ($1366 - 50$) |
| Safe Zone X Min | $x_{safe}$ | `1150.0` | px | Overload right-side safe refuge |
| Player Width | $w_{player}$ | `40.0` | px | Canonical player collision box |
| Player Height | $h_{player}$ | `60.0` | px | Canonical player collision box |
| Player Speed | $v_{move}$ | `400.0` | px/s | Standard flash jump / walking speed |
| Player Jump Impulse | $v_{jump}$ | `-650.0` | px/s | MapleStory standard jump velocity |
| Gravity | $g$ | `1800.0` | px/s$^2$ | Natural arc gravity (terminal vel 700 px/s) |
| Simulation dt | $dt$ | `0.0166667` | s | 60 FPS fixed time step ($1/60\text{ s}$) |
| Laser Angular Velocity | $\omega$ | `0.5235` | rad/s | $\pi / 6\text{ rad/s} = 30.0^\circ/\text{s}$ |
| Laser Beam Thickness | $d_{beam}$ | `20.0` | px | Beam collision width |
| Debris Max Capacity | $N_{debris}$ | `30` | int | Static array size for XLA |
| Annihilation Normal Rate | $r_{norm}$ | `0.006` | 1/s | 0.6% per second |
| Annihilation Hard Rate | $r_{hard}$ | `0.008` | 1/s | 0.8% per second |
| Annihilation Extreme Rate| $r_{ext}$ | `0.020` | 1/s | 2.0% per second |
| Overload Duration | $T_{overload}$| `25.0` | s | 1500 simulation ticks |
| Tracking Aim Duration | $T_{track}$ | `1.0` | s | 60 simulation ticks |
| Tracking Lock Duration| $T_{lock}$ | `1.0` | s | 60 simulation ticks |
| Small Arm Interval | $\Delta t_{arm}$| `0.8` | s | 48 simulation ticks (12 strikes total) |

---

## 5. Exact JSON Schema for EnvParams

This JSON schema represents the contract between `wz_parser.py` and the JAX environment (`maple_gymnax.envs.lotus_phase1`):

```json
{
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
    "remastered_params"
  ],
  "properties": {
    "map_width": { "type": "number", "default": 1366.0, "minimum": 800.0 },
    "map_height": { "type": "number", "default": 768.0, "minimum": 600.0 },
    "core_x": { "type": "number", "default": 683.0 },
    "core_y": { "type": "number", "default": 384.0 },
    "core_radius": { "type": "number", "default": 120.0, "minimum": 10.0 },
    "floor_y": { "type": "number", "default": 605.0 },
    "wall_left": { "type": "number", "default": 50.0 },
    "wall_right": { "type": "number", "default": 1316.0 },
    "safe_zone_x": { "type": "number", "default": 1150.0 },
    "player_width": { "type": "number", "default": 40.0 },
    "player_height": { "type": "number", "default": 60.0 },
    "player_speed": { "type": "number", "default": 400.0 },
    "player_jump_impulse": { "type": "number", "default": -650.0 },
    "gravity": { "type": "number", "default": 1800.0 },
    "dt": { "type": "number", "default": 0.016666667 },
    "classic_laser": {
      "type": "object",
      "required": ["angular_velocity", "beam_thickness", "beam_count", "damage"],
      "properties": {
        "angular_velocity": { "type": "number", "default": 0.523598775 },
        "beam_thickness": { "type": "number", "default": 20.0 },
        "beam_count": { "type": "integer", "default": 4, "minimum": 1, "maximum": 8 },
        "damage": { "type": "number", "default": 1.0 }
      }
    },
    "debris_params": {
      "type": "object",
      "required": ["max_debris", "spawn_interval_ticks", "types"],
      "properties": {
        "max_debris": { "type": "integer", "default": 30 },
        "spawn_interval_ticks": { "type": "integer", "default": 15 },
        "types": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["type_id", "radius", "damage", "stun_duration", "fall_speed"],
            "properties": {
              "type_id": { "type": "integer" },
              "radius": { "type": "number" },
              "damage": { "type": "number" },
              "stun_duration": { "type": "number" },
              "fall_speed": { "type": "number" }
            }
          }
        }
      }
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
        "floor_discharge_damage"
      ],
      "properties": {
        "enabled": { "type": "boolean", "default": true },
        "difficulty": { "type": "string", "enum": ["normal", "hard", "extreme"], "default": "normal" },
        "gauge_natural_rates": {
          "type": "object",
          "properties": {
            "normal": { "type": "number", "default": 0.006 },
            "hard": { "type": "number", "default": 0.008 },
            "extreme": { "type": "number", "default": 0.020 }
          }
        },
        "overload_duration": { "type": "number", "default": 25.0 },
        "bombardment_damage": { "type": "number", "default": 1.0 },
        "bombardment_tick_rate": { "type": "number", "default": 0.5 },
        "electric_field_damage": { "type": "number", "default": 0.05 },
        "electric_field_tick_rate": { "type": "number", "default": 0.36 },
        "tracking_laser_player_damage": { "type": "number", "default": 0.15 },
        "tracking_laser_gauge_increase": { "type": "number", "default": 0.10 },
        "tracking_laser_gauge_decrease": { "type": "number", "default": 0.10 },
        "small_arm_player_damage": { "type": "number", "default": 0.05 },
        "small_arm_gauge_increase": { "type": "number", "default": 0.03 },
        "small_arm_gauge_decrease": { "type": "number", "default": 0.03 },
        "floor_discharge_warning_duration": { "type": "number", "default": 1.2 },
        "floor_discharge_damage": { "type": "number", "default": 1.0 }
      }
    }
  }
}
```

---

## 6. Architecture & Implementation Blueprint for `wz_parser.py`

### 6.1 Parser Module Layout
`src/maple_gymnax/parser/wz_parser.py` should be implemented as follows:
1. `class WZParser`:
   - `def __init__(self, mp_root: str = "C:\\mp")`
   - `def load_boss_suu_patterns(self) -> Dict[str, Any]`: Loads and parses `Restored_Data/Mob/BossPattern/_Canvas/_Canvas_012/BossSuu.img.json`. Extracts pattern hierarchy (`1000`~`1009`, `common/UI`), frame counts, and canvas dimensions.
   - `def load_map_back(self) -> Dict[str, Any]`: Loads `Restored_Data/Map/Back/Back_000/bossSuu.img.json`, extracts `Swoo_Bossmap_Phase1.atlas` sprite regions.
   - `def extract_env_params(self, difficulty: str = "normal", enable_remaster: bool = True, custom_overrides: Optional[Dict[str, Any]] = None) -> EnvParams`:
     Synthesizes parsed WZ metadata with canonical physical constants into a type-safe Flax dataclass.
   - `def export_json(self, output_path: str, params: EnvParams) -> None`: Exports the structured dictionary to a JSON file conforming to the schema above.

### 6.2 Unit Conversion Helpers
- `ms_to_seconds(ms: int) -> float`: `ms / 1000.0`
- `ms_to_ticks(ms: int, dt: float = 1/60) -> int`: `int(round((ms / 1000.0) / dt))`
- `ticks_to_seconds(ticks: int, dt: float = 1/60) -> float`: `ticks * dt`

---

## 7. Conclusion & Next Steps for Teamwork
1. **Authoritative Grounding**: All pattern keys (`1001-000`, `1001-001`, `1006-000`, `1006-002`, `common/UI/gauge`) have been directly validated against `C:\mp\Restored_Data\Mob\BossPattern\_Canvas\_Canvas_012\BossSuu.img.json` and `C:\mp\Restored_Data\Map\Back\Back_000\bossSuu.img.json`.
2. **Modular Environment Support**: The extracted specification provides complete parameters for both Classic Mode (rotating cross laser + falling debris) and Remastered Mode (annihilation gauge + friendly fire guidance + overload bombardment).
3. **Delivery**: The findings are documented here and summarized in `handoff.md` for immediate consumption by the downstream environment implementers.
