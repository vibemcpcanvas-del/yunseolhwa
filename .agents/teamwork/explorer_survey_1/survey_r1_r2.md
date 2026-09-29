# Technical Survey & Architecture Report: R1 & R2
## Autonomous Micro-Movement & Threat-Gated Evasion: Eliminating Jump-Spam Local Minima via Action Cost, Jitter Regularization, and Tap-Dodging Corridor Rewards

**Subagent**: `explorer_survey_1` (Teamwork Explorer)  
**Parent**: `orchestrator_3` (ID: `e8d54a3b-63f5-4fbd-92da-90a91af57a97`)  
**Target Codebase**: `src/maple_gymnax/envs/lotus_phase1.py`, `src/maple_gymnax/envs/common.py`  
**Timestamp**: 2026-09-29T11:35:00Z  
**Branch**: `feat/debris-threat-obs`  
**Mode**: Read-Only Survey & Specification  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Jump-Spam Local Minimum Artifact
In zero-cost action spaces, Reinforcement Learning agents exploring continuous survival environments frequently collapse into high-frequency, high-energy periodic actions—the canonical **"bunny-hop" local minimum**. In the Maple Gymnax Lotus Phase 1 environment:
- Rollout telemetry reveals that the current PPO policy is trapped in an **11-second / 680-step survival plateau**.
- Action distribution is heavily distorted: **53.7% of all actions are `JUMP_RIGHT` (Action 5 / 6)**.
- While jumping temporarily lifts the player hitbox over the floor, it severely degrades horizontal controllability (ballistic parabolic trajectory), blinds the agent to vertical clearance constraints, and causes catastrophic collisions with medium and large falling debris (averaging **7.7 debris hits/episode**).

### 1.2 Objective & Solution Architecture
To shatter this 680-step plateau and foster human-like grounded micro-movement (**칼무빙 / tap-dodging**), Requirements **R1** and **R2** establish a three-tiered regularized control architecture:
1. **Action Effort & Energy Penalty (R1)**: Discrete jump actions (`JUMP`, `JUMP_LEFT`, `JUMP_RIGHT`) incur an explicit, continuous energy penalty of $r_{\text{action\_jump}} = -0.05$ per tick, while ground actions (`NOOP`, `LEFT`, `RIGHT`, `DOWN`) remain zero-cost ($0.0$).
2. **Action Switching Jitter Regularization (R1)**: Directional action chatter is penalized at $r_{\text{jitter}} = -0.02 \cdot \mathbb{I}(a_t \ne a_{t-1})$, incentivizing coherent 5–10 frame micro-taps rather than single-tick alternating noise.
3. **Hazard-Gated Anti-Jump Penalty (R2)**: Jumping while high-threat debris ($r \ge 24\text{px}$) is within the overhead danger cone ($|\Delta x| < 45\text{px}$, $\Delta y \in [0, 180\text{px}]$) incurs an amplified penalty of $r_{\text{airborne\_hazard}} = -0.35$.
4. **Tap-Dodging Grounded Clearance Bonus (R2)**: Grounded lateral movement that increases horizontal separation from overhead debris ($dx_{\text{next}} > dx_{\text{prev}}$) receives a positive evasion reward shaping bonus of $r_{\text{tap\_dodge}} = +0.25$.
5. **Continuous Overhead Potential Rescaling (R2)**: The smooth Gaussian overhead repulsion potential $\phi_{\text{debris}}$ is scaled from $-0.05$ to $-0.30$, creating a steep spatial gradient away from the fall corridor.

Under this formulation, the instantaneous reward delta between jumping and grounded micro-dodging under an overhead threat is:
$$\Delta r = r_{\text{tap\_dodge}} - (r_{\text{action\_jump}} + r_{\text{airborne\_hazard}}) = (+0.25) - (-0.05 - 0.35) = +0.65 \text{ per tick}$$
This $+0.65$ margin creates an overwhelming policy gradient against airborne spam and guarantees grounded evasion dominance.

---

## 2. Deep Inspection of Current Codebase State

### 2.1 File: `src/maple_gymnax/envs/lotus_phase1.py`

#### 2.1.1 Environment Parameters (`EnvParams`) — Lines 38–136
- Marked with `@flax.struct.dataclass`.
- Stores immutable kinematic limits, dimensions, and damage values:
  - `player_w: float = 40.0`, `player_h: float = 60.0`, `player_duck_h: float = 35.0`
  - `player_speed: float = 400.0`, `jump_velocity: float = -650.0`, `gravity: float = 1800.0`
  - `dt: float = 1.0 / 60.0`, `floor_y: float = 605.0`
  - `mode: int = flax.struct.field(pytree_node=False, default=0)` (0: Classic, 1: Remastered, 2: Hybrid)
- **R1/R2 Extension Opportunity**: New reward hyperparameters can be added with default values to `EnvParams` so they remain configurable across sweeps:
  - `r_action_jump_cost: float = -0.05`
  - `r_jitter_cost: float = -0.02`
  - `r_airborne_hazard_cost: float = -0.35`
  - `r_tap_dodge_bonus: float = 0.25`
  - `debris_repel_scale: float = -0.30`

#### 2.1.2 Environment Dynamic State (`EnvState`) — Lines 138–177
- Marked with `@flax.struct.dataclass`.
- Fields without defaults (Lines 142–163):
  - `player_x`, `player_y`, `player_vx`, `player_vy`, `player_hp`, `player_on_ground`, `invincible_timer`, `laser_angle`, `debris_x`, `debris_y`, `debris_vy`, `debris_radius`, `debris_damage`, `debris_active`, `debris_type`, `time`.
- Fields with defaults (Lines 166–176):
  - `security_gauge: float = 0.0`, `is_overload: bool = False`, `overload_timer: float = 0.0`, `boss_hp: float = 1000.0`, `boss_shield: float = 200.0`, `shield_active: bool = True`, `tracking_laser_timer: float = 0.0`, `tracking_laser_lock_x: float = 683.0`, `tracking_laser_state: int = 0`, `electric_floor_warning: float = 0.0`, `electric_floor_active: bool = False`.
- **CRITICAL OBSERVATION**: `last_action` is currently **MISSING** from `EnvState`.
  - Because Python and Flax require fields with default values to follow fields without default values, `last_action: int = 0` must be declared with a default value (`= 0`) among the default fields (e.g. line 164 or 177).
  - Placing `last_action: int = 0` ensures complete backward compatibility with all existing callers and test harnesses that instantiate `EnvState(...)` using keyword arguments without passing `last_action`.

#### 2.1.3 Player Kinematics Integration (`_step_player_kinematics`) — Lines 183–216
- Computes `vx`, `current_h`, `is_jump_action` via `decode_action(action, ...)`.
- Computes jump eligibility: `can_jump = state.player_on_ground & is_jump_action`.
- Integrates velocity and coordinates:
  - `vy_start = jnp.where(can_jump, params.jump_velocity, state.player_vy)`
  - `vy_next = vy_start + params.gravity * params.dt`
  - `px_cand = state.player_x + vx * params.dt`
  - `py_cand = state.player_y + vy_next * params.dt`
  - Floor landing detection: `hits_floor = py_cand >= params.floor_y`
  - Ground contact status: `on_ground_next = hits_floor`
- Returns: `(px_next, py_next, vx, vy_final, on_ground_next, p_center_x, p_center_y, current_h)`.
- **Key Observation for R2**:
  - `on_ground_next` is immediately available from `_step_player_kinematics`.
  - If the agent is grounded and jumps, `on_ground_next` becomes `False` on that exact tick!
  - `vx` indicates lateral velocity ($-400.0$, $0.0$, or $+400.0$).
  - `state.player_x` represents position before the step, and `px_next` represents position after the step.

#### 2.1.4 Falling Debris Mechanics (`_step_falling_debris`) — Lines 242–288
- Padded static arrays of capacity `MAX_DEBRIS = 30`.
- Debris types, radii, and damages:
  - Type 0: `radii[0] = 16.0`, `damages[0] = 10.0` (Small)
  - Type 1: `radii[1] = 24.0`, `damages[1] = 20.0` (Medium)
  - Type 2: `radii[2] = 36.0`, `damages[2] = 30.0` (Large)
- Arrays returned: `cur_deb_x`, `deb_y_next`, `cur_deb_vy`, `cur_deb_r`, `cur_deb_dmg`, `deb_act_next`, `cur_deb_t`.
- **Key Observation for R2**:
  - `cur_deb_r >= 24.0` identifies high-threat debris (Types 1 and 2).
  - `cur_deb_x` is the horizontal position of debris.
  - `deb_y_next` is the vertical position of debris.
  - `deb_act_next` is the boolean activity mask.

#### 2.1.5 Current Reward Function (`_compute_reward`) — Lines 471–594
- Currently takes 17 arguments:
  ```python
  def _compute_reward(
      took_hit: chex.Array,
      total_dmg: chex.Array,
      laser_hits_boss: chex.Array,
      laser_hits_player: chex.Array,
      shield_shatter: chex.Array,
      triggers_overload: chex.Array,
      gauge_final: chex.Array,
      is_overload_next: chex.Array,
      px_next: chex.Array,
      py_next: chex.Array,
      hp_next: chex.Array,
      tl_state: chex.Array,
      tl_lock_x: chex.Array,
      debris_x: chex.Array,
      debris_y: chex.Array,
      debris_radius: chex.Array,
      debris_active: chex.Array,
      params: EnvParams,
  ) -> chex.Array:
  ```
- **CURRENT GAPS IDENTIFIED**:
  1. `action` and `last_action` are **NOT** passed to `_compute_reward()`. Cannot compute `r_action_jump` or `r_jitter`.
  2. `on_ground_next` is **NOT** passed to `_compute_reward()`. Cannot determine airborne status for `r_airborne_hazard` or grounded status for `r_tap_dodge`.
  3. `state.player_x` (or `vx`) is **NOT** passed to `_compute_reward()`. Cannot determine whether horizontal motion expands separation from debris center.
  4. Overhead Gaussian repulsion potential (lines 560–564) is currently scaled with `-0.05`:
     ```python
     r_debris_repel = jnp.where(
         is_remastered_or_hybrid,
         -0.05 * jnp.sum(overhead_weight),
         0.0,
     )
     ```
     Requirement R2 specifies scaling this from `-0.05` to `-0.30`.

#### 2.1.6 Environment Step (`step_env`) — Lines 607–728
- Lines 659–687 construct `state_next = EnvState(...)`.
  - Currently omits `last_action`. Must pass `last_action=action`.
- Lines 690–710 call `_compute_reward(...)`.
  - Must pass `action`, `state.last_action`, `on_ground_next`, and `state.player_x` (or `vx`).
- Lines 715–727 construct `info` dict.
  - Can include telemetry fields such as `is_jump_action`, `r_action_jump`, `r_jitter`, `r_airborne_hazard`, `r_tap_dodge` for live monitoring.

#### 2.1.7 Environment Reset (`reset_env`) — Lines 730–767
- Lines 738–766 construct initial `EnvState(...)`.
  - Must pass `last_action=0` (or allow dataclass default `0`).

---

### 2.2 File: `src/maple_gymnax/envs/common.py`

#### 2.2.1 Action Space Definition Mismatch Analysis
In `src/maple_gymnax/envs/common.py` (lines 30–45):
```python
ACTION_NOOP: int = 0
ACTION_LEFT: int = 1
ACTION_RIGHT: int = 2
ACTION_JUMP: int = 3
ACTION_JUMP_LEFT: int = 4
ACTION_JUMP_RIGHT: int = 5
ACTION_DUCK: int = 6
```
However, the Authoritative User Request (2026-09-29T11:28:08Z) and Task Objective state:
```
Action space and enum/indices:
0 NOOP
1 LEFT
2 RIGHT
3 DOWN
4 JUMP
5 JUMP_LEFT
6 JUMP_RIGHT
```
And Requirements R1 specifies:
- Assign explicit action cost `r_action_jump = -0.05` to discrete actions `4 (JUMP)`, `5 (JUMP_LEFT)`, and `6 (JUMP_RIGHT)`.
- Maintain zero cost `r_action_ground = 0.0` for actions `0 (NOOP)`, `1 (LEFT)`, `2 (RIGHT)`, `3 (DOWN)`.

#### Why the Re-ordered Action Space is Structurally Superior:
Under the target mapping:
$$\text{Ground Actions } \mathcal{A}_{\text{ground}} = \{0, 1, 2, 3\} \quad (\text{NOOP, LEFT, RIGHT, DOWN})$$
$$\text{Airborne/Jump Actions } \mathcal{A}_{\text{jump}} = \{4, 5, 6\} \quad (\text{JUMP, JUMP\_LEFT, JUMP\_RIGHT})$$
This cleanly partitions the action space into contiguous intervals:
- Jumping check is simply: `action >= 4` or `(action >= ACTION_JUMP)`!
- Ground check is simply: `action < 4` or `(action <= ACTION_DOWN)`!

#### Downstream Codebase & Test Impact Assessment:
We audited all occurrences of action constants and integer literals across the entire repository:
1. `src/maple_gymnax/envs/common.py`:
   - Redefine:
     ```python
     ACTION_NOOP: int = 0
     ACTION_LEFT: int = 1
     ACTION_RIGHT: int = 2
     ACTION_DOWN: int = 3
     ACTION_DUCK: int = 3  # Backward compatibility alias
     ACTION_JUMP: int = 4
     ACTION_JUMP_LEFT: int = 5
     ACTION_JUMP_RIGHT: int = 6

     NOOP: int = ACTION_NOOP
     LEFT: int = ACTION_LEFT
     RIGHT: int = ACTION_RIGHT
     DOWN: int = ACTION_DOWN
     DUCK: int = ACTION_DOWN
     JUMP: int = ACTION_JUMP
     JUMP_LEFT: int = ACTION_JUMP_LEFT
     JUMP_RIGHT: int = ACTION_JUMP_RIGHT
     ```
   - In `decode_action`:
     ```python
     is_left = (action == ACTION_LEFT) | (action == ACTION_JUMP_LEFT)
     is_right = (action == ACTION_RIGHT) | (action == ACTION_JUMP_RIGHT)
     vx = jnp.where(is_left, -player_speed, jnp.where(is_right, player_speed, 0.0))

     is_duck = action == ACTION_DOWN
     current_h = jnp.where(is_duck, duck_h, standard_h)

     is_jump_action = action >= ACTION_JUMP
     ```
2. `src/maple_gymnax/envs/__init__.py`:
   - Export `ACTION_DOWN` and `DOWN` alongside `ACTION_DUCK` and `DUCK`.
3. `tests/test_lotus_phase1.py`:
   - Imports symbolic constants `ACTION_NOOP`, `ACTION_LEFT`, `ACTION_RIGHT`, `ACTION_JUMP`, `ACTION_DUCK`.
   - **Zero regressions**: tests referencing constants will automatically track the updated integers.
4. `tests/e2e/test_tier1_features.py`:
   - Lines 379 and 1468 use integer literal `3` expecting jump (`step_env(key, state, 3, p)`).
   - Line 781 uses integer literal `6` expecting duck (`step_env(key, state, 6, p)`).
   - **Resolution**: Replace raw integer literals in `test_tier1_features.py` with `ACTION_JUMP` and `ACTION_DOWN` (or `4` and `3`).

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Requirement R1: Action Cost & Energy Regularization Engine

#### 3.1.1 Discrete Action Cost
For any incoming action $a_t \in \{0, \dots, 6\}$:
$$r_{\text{action\_jump}}(a_t) = \begin{cases} -0.05 & \text{if } a_t \in \{4, 5, 6\} \quad (\text{JUMP, JUMP\_LEFT, JUMP\_RIGHT}) \\ 0.0 & \text{if } a_t \in \{0, 1, 2, 3\} \quad (\text{NOOP, LEFT, RIGHT, DOWN}) \end{cases}$$

Pure JAX branch-free implementation:
```python
is_jump_action = (action == 4) | (action == 5) | (action == 6)  # or action >= 4
r_action_jump = jnp.where(is_jump_action, params.r_action_jump_cost, 0.0)
```

#### 3.1.2 Action Switching Jitter Regularization
Directional micro-taps require holding an intentional action for 5–10 frames. Alternating back-and-forth between disparate directions on every frame causes policy instability and tracking failure.
$$r_{\text{jitter}}(a_t, a_{t-1}) = -0.02 \cdot \mathbb{I}(a_t \ne a_{t-1})$$

Pure JAX branch-free implementation:
```python
is_action_switched = action != last_action
r_jitter = jnp.where(is_action_switched, params.r_jitter_cost, 0.0)
```

#### 3.1.3 State Lifecycle Tracking
- At episode initialization (`reset_env`):
  `last_action = 0` (NOOP).
- At each simulation tick $t \to t+1$ (`step_env`):
  `state_next = state.replace(..., last_action=action)` or instantiated in `EnvState(..., last_action=action)`.
- During auto-reset in `LogWrapper`:
  `auto_env_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, next_env_state)`
  Automatically resets `last_action` to 0 on terminal boundaries without special casing.

---

### 3.2 Requirement R2: Tap-Dodging Hazard Corridor & Overhead Repulsion

#### 3.2.1 High-Threat Overhead Hazard Condition
Falling debris has radii $r \in \{16, 24, 36\}\text{px}$. High-threat debris is defined as $r \ge 24\text{px}$ (Types 1 and 2, which inflict 20% to 30% HP damage).
The hazard corridor extends vertically $180\text{px}$ above the player and laterally $45\text{px}$ from the player's center:
$$\Delta x_i = |px_{\text{next}} - \text{debris\_x}[i]|$$
$$\Delta y_i = py_{\text{next}} - \text{debris\_y}[i]$$
$$\text{is\_threat\_overhead}_i = \text{debris\_active}[i] \land (r_i \ge 24.0) \land (\Delta x_i < 45.0) \land (\Delta y_i \ge 0.0) \land (\Delta y_i \le 180.0)$$

Vectorized JAX evaluation across all $N=30$ static debris slots:
```python
dx_deb_next = jnp.abs(px_next - debris_x)
dy_deb = py_next - debris_y

is_threat_overhead = (
    debris_active
    & (debris_radius >= 24.0)
    & (dx_deb_next < 45.0)
    & (dy_deb >= 0.0)
    & (dy_deb <= 180.0)
)
has_overhead_threat = jnp.any(is_threat_overhead)
```

#### 3.2.2 Hazard-Gated Anti-Jump Penalty
When high-threat debris is directly falling overhead, jumping into the descending particle is suicidal. If the agent is airborne (`~player_on_ground`) while `has_overhead_threat` is active:
$$r_{\text{airborne\_hazard}} = -0.35 \cdot \mathbb{I}(\text{has\_overhead\_threat} \land \neg \text{player\_on\_ground})$$

Pure JAX branch-free implementation (following Rule 11 boolean typing):
```python
is_airborne = jnp.logical_not(on_ground_next)
r_airborne_hazard = jnp.where(
    has_overhead_threat & is_airborne,
    params.r_airborne_hazard_cost,  # -0.35
    0.0,
)
```

#### 3.2.3 Tap-Dodging Clearance Incentive
When under an overhead threat corridor, the optimal human behavior is grounded tap-dodging (**칼무빙**): maintaining grounded footing while moving laterally away from the particle centerline.

**Mathematical Definition of "Moving Horizontally Away from Debris Center"**:
Let $x_{\text{prev}} = \text{state.player\_x}$ and $x_{\text{next}} = px_{\text{next}}$.
Horizontal distance before step: $dx_{\text{prev}} = |x_{\text{prev}} - \text{debris\_x}[i]|$.
Horizontal distance after step: $dx_{\text{next}} = |x_{\text{next}} - \text{debris\_x}[i]|$.
The player is moving away if and only if:
$$dx_{\text{next}} > dx_{\text{prev}}$$

Equivalently, in terms of displacement $v_x = x_{\text{next}} - x_{\text{prev}}$:
- If player is to the right of debris ($x_{\text{next}} > \text{debris\_x}[i]$), moving away requires $v_x > 0$.
- If player is to the left of debris ($x_{\text{next}} < \text{debris\_x}[i]$), moving away requires $v_x < 0$.
Thus:
$$(x_{\text{next}} - \text{debris\_x}[i]) \cdot (x_{\text{next}} - x_{\text{prev}}) > 0 \iff dx_{\text{next}} > dx_{\text{prev}}$$

**Clearance Boundary Condition**:
Line 97 notes that grounded movement expanding lateral separation achieves safety when $|\Delta x| > r + 15\text{px}$ (e.g. $39\text{px}$ for $r=24$, $51\text{px}$ for $r=36$).
When evaluating whether the agent was in the danger cone or dodging within it:
```python
dx_deb_prev = jnp.abs(state_player_x - debris_x)
moving_away = is_threat_overhead & (dx_deb_next > dx_deb_prev)
can_tap_dodge = on_ground_next & jnp.any(moving_away)
r_tap_dodge = jnp.where(can_tap_dodge, params.r_tap_dodge_bonus, 0.0)  # +0.25
```
Notice:
- `can_tap_dodge` is True **only** if the player is grounded (`on_ground_next == True`).
- It applies **only** when high-threat debris is in the overhead cone.
- It rewards **only** lateral displacement that increases separation from the hazard.
- It prevents reward hacking: standing still ($v_x = 0$) or moving into the debris yields $0.0$.

#### 3.2.4 Continuous Overhead Gaussian Repulsion Potential
In addition to the discrete gating rules, the continuous spatial potential $\phi_{\text{debris}}$ provides smooth gradients for policy improvement across the entire lateral continuum:
$$\phi_{\text{debris}} = -0.30 \cdot \sum_{i=1}^{N} \mathbb{I}(\text{is\_overhead}_i) \cdot \left(\frac{r_i}{36.0}\right) \exp\left(-\frac{1}{2} \left(\frac{\Delta x_i}{45.0}\right)^2\right)$$

In the current code (line 562), the coefficient is `-0.05`. We update this to `-0.30`:
```python
overhead_weight = jnp.where(
    is_overhead,
    (debris_radius / 36.0) * jnp.exp(-0.5 * (dx_deb_next / 45.0) ** 2),
    0.0,
)
r_debris_repel = jnp.where(
    is_remastered_or_hybrid,
    params.debris_repel_scale * jnp.sum(overhead_weight),  # -0.30
    0.0,
)
```

---

## 4. Concrete Code Modification Specifications

### 4.1 Modifications to `src/maple_gymnax/envs/common.py`

#### Target 1: Action Space Enum & Constants (Lines 30–45)
```python
<<<<
ACTION_NOOP: int = 0
ACTION_LEFT: int = 1
ACTION_RIGHT: int = 2
ACTION_JUMP: int = 3
ACTION_JUMP_LEFT: int = 4
ACTION_JUMP_RIGHT: int = 5
ACTION_DUCK: int = 6

# Canonical Short Aliases
NOOP: int = ACTION_NOOP
LEFT: int = ACTION_LEFT
RIGHT: int = ACTION_RIGHT
JUMP: int = ACTION_JUMP
JUMP_LEFT: int = ACTION_JUMP_LEFT
JUMP_RIGHT: int = ACTION_JUMP_RIGHT
DUCK: int = ACTION_DUCK
====
ACTION_NOOP: int = 0
ACTION_LEFT: int = 1
ACTION_RIGHT: int = 2
ACTION_DOWN: int = 3
ACTION_DUCK: int = 3  # Backward-compatible alias
ACTION_JUMP: int = 4
ACTION_JUMP_LEFT: int = 5
ACTION_JUMP_RIGHT: int = 6

# Canonical Short Aliases
NOOP: int = ACTION_NOOP
LEFT: int = ACTION_LEFT
RIGHT: int = ACTION_RIGHT
DOWN: int = ACTION_DOWN
DUCK: int = ACTION_DOWN
JUMP: int = ACTION_JUMP
JUMP_LEFT: int = ACTION_JUMP_LEFT
JUMP_RIGHT: int = ACTION_JUMP_RIGHT
>>>>
```

#### Target 2: Action Decoding Logic (Lines 185–209)
```python
<<<<
    is_left = (action == ACTION_LEFT) | (action == ACTION_JUMP_LEFT)
    is_right = (action == ACTION_RIGHT) | (action == ACTION_JUMP_RIGHT)
    vx = jnp.where(is_left, -player_speed, jnp.where(is_right, player_speed, 0.0))

    is_duck = action == ACTION_DUCK
    current_h = jnp.where(is_duck, duck_h, standard_h)

    is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)

    return vx, current_h, is_jump_action
====
    is_left = (action == ACTION_LEFT) | (action == ACTION_JUMP_LEFT)
    is_right = (action == ACTION_RIGHT) | (action == ACTION_JUMP_RIGHT)
    vx = jnp.where(is_left, -player_speed, jnp.where(is_right, player_speed, 0.0))

    is_duck = action == ACTION_DOWN
    current_h = jnp.where(is_duck, duck_h, standard_h)

    is_jump_action = (action == ACTION_JUMP) | (action == ACTION_JUMP_LEFT) | (action == ACTION_JUMP_RIGHT)

    return vx, current_h, is_jump_action
>>>>
```

---

### 4.2 Modifications to `src/maple_gymnax/envs/lotus_phase1.py`

#### Target 1: `EnvParams` Reward Hyperparameters (Insert around Line 92)
```python
    # Remastered Lotus Mechanics
    ...
    arm_slam_gauge_reduction: float = 0.03

    # R1 & R2: Action Cost, Regularization & Threat-Gated Evasion Shaping
    r_action_jump_cost: float = -0.05
    r_jitter_cost: float = -0.02
    r_airborne_hazard_cost: float = -0.35
    r_tap_dodge_bonus: float = 0.25
    debris_repel_scale: float = -0.30
```

#### Target 2: `EnvState` Field Definition (Line 164)
```python
<<<<
    # Temporal & Episode Tracking
    time: int

    # Remastered Gimmick States (April 2024 Remake)
====
    # Temporal & Episode Tracking
    time: int
    last_action: int = 0

    # Remastered Gimmick States (April 2024 Remake)
>>>>
```

#### Target 3: `_compute_reward` Signature and Implementation (Lines 471–594)
```python
def _compute_reward(
    took_hit: chex.Array,
    total_dmg: chex.Array,
    laser_hits_boss: chex.Array,
    laser_hits_player: chex.Array,
    shield_shatter: chex.Array,
    triggers_overload: chex.Array,
    gauge_final: chex.Array,
    is_overload_next: chex.Array,
    px_next: chex.Array,
    py_next: chex.Array,
    hp_next: chex.Array,
    tl_state: chex.Array,
    tl_lock_x: chex.Array,
    debris_x: chex.Array,
    debris_y: chex.Array,
    debris_radius: chex.Array,
    debris_active: chex.Array,
    action: Union[int, chex.Array],
    last_action: Union[int, chex.Array],
    on_ground_next: chex.Array,
    state_player_x: chex.Array,
    params: EnvParams,
) -> chex.Array:
    is_remastered_or_hybrid = params.mode != MODE_CLASSIC

    # --- Core Survival (Damage-Proportional Penalty - Rule 17) ---
    r_base = 0.03
    r_hit = jnp.where(took_hit, -1.5 * total_dmg, 0.0)
    r_death = jnp.where(hp_next <= 0.0, -70.0, 0.0)
    r_hp = 0.02 * (hp_next / params.player_max_hp)

    # --- R1: Action Cost & Energy Regularization Engine ---
    is_jump_action = (action == 4) | (action == 5) | (action == 6)
    r_action_jump = jnp.where(is_jump_action, params.r_action_jump_cost, 0.0)
    r_jitter = jnp.where(action != last_action, params.r_jitter_cost, 0.0)

    # --- Clean Friendly Fire Redirection (Bait & Dodge) ---
    clean_boss_hit = laser_hits_boss & (~laser_hits_player)
    dirty_boss_hit = laser_hits_boss & laser_hits_player
    r_boss_hit = jnp.where(
        clean_boss_hit,
        70.0,
        jnp.where(dirty_boss_hit, 15.0, 0.0),
    )
    r_shield = jnp.where(shield_shatter, 30.0, 0.0)

    # Friendly fire self-damage penalty
    is_firing = tl_state == 2
    r_self_laser = jnp.where(
        is_firing & laser_hits_player & is_remastered_or_hybrid,
        -15.0,
        0.0,
    )

    # --- Gauge & Overload Management ---
    r_overload = jnp.where(triggers_overload, -20.0, 0.0)
    r_gauge = jnp.where(is_remastered_or_hybrid, -0.08 * gauge_final, 0.0)

    # Safe zone positioning during overload
    in_safe_zone = px_next >= params.safe_zone_x
    r_safe_zone = jnp.where(
        is_overload_next & is_remastered_or_hybrid,
        jnp.where(in_safe_zone, 0.3, -0.3),
        0.0,
    )

    # --- Anti-Wall-Camping Penalty ---
    wall_margin = 220.0
    near_left_wall = px_next < (params.wall_left + wall_margin)
    near_right_wall = px_next > (params.wall_right - wall_margin)
    near_any_wall = near_left_wall | near_right_wall
    r_wall = jnp.where(
        near_any_wall & (~is_overload_next) & is_remastered_or_hybrid,
        -0.30,
        0.0,
    )

    # --- R2: Tap-Dodging Hazard Corridor & Overhead Repulsion ---
    dx_deb_next = jnp.abs(px_next - debris_x)
    dx_deb_prev = jnp.abs(state_player_x - debris_x)
    dy_deb = py_next - debris_y
    is_overhead = (
        debris_active
        & (debris_radius >= 24.0)
        & (dy_deb > 0.0)
        & (dy_deb < 180.0)
    )
    is_threat_overhead = is_overhead & (dx_deb_next < 45.0)
    has_overhead_threat = jnp.any(is_threat_overhead)

    # Anti-jump penalty in overhead danger corridor
    is_airborne = jnp.logical_not(on_ground_next)
    r_airborne_hazard = jnp.where(
        has_overhead_threat & is_airborne,
        params.r_airborne_hazard_cost,
        0.0,
    )

    # Grounded clearance bonus: moving horizontally away from debris center
    moving_away = is_threat_overhead & (dx_deb_next > dx_deb_prev)
    can_tap_dodge = on_ground_next & jnp.any(moving_away)
    r_tap_dodge = jnp.where(can_tap_dodge, params.r_tap_dodge_bonus, 0.0)

    # Continuous overhead Gaussian repulsion potential (scaled to -0.30)
    overhead_weight = jnp.where(
        is_overhead,
        (debris_radius / 36.0) * jnp.exp(-0.5 * (dx_deb_next / 45.0) ** 2),
        0.0,
    )
    r_debris_repel = jnp.where(
        is_remastered_or_hybrid,
        params.debris_repel_scale * jnp.sum(overhead_weight),
        0.0,
    )

    # --- Potential-Based Tracking & Evasion Shaping ---
    is_tracking = tl_state == 1
    dist_to_boss = jnp.abs(tl_lock_x - params.core_x)
    phi_tracking = jnp.exp(-0.5 * (dist_to_boss / (params.boss_w / 2.0)) ** 2)
    r_bait = jnp.where(
        is_tracking & is_remastered_or_hybrid,
        0.12 * phi_tracking,
        0.0,
    )

    dist_from_beam = jnp.abs(px_next - tl_lock_x)
    safe_dist = params.player_w / 2.0
    phi_dodge = jnp.clip((dist_from_beam - safe_dist) / 30.0, 0.0, 1.0)
    r_dodge = jnp.where(
        is_firing & is_remastered_or_hybrid,
        0.20 * phi_dodge,
        0.0,
    )

    return (
        r_base + r_hit + r_death + r_hp
        + r_action_jump + r_jitter
        + r_airborne_hazard + r_tap_dodge
        + r_boss_hit + r_shield + r_self_laser
        + r_overload + r_gauge + r_safe_zone
        + r_wall + r_bait + r_dodge + r_debris_repel
    )
```

#### Target 4: `step_env` Updates (Lines 658–710)
```python
<<<<
        # 7. Next Immutable State
        state_next = EnvState(
            player_x=px_next,
            player_y=py_next,
            player_vx=vx,
            player_vy=vy_final,
            player_hp=hp_next,
            player_on_ground=on_ground_next,
            invincible_timer=inv_timer_next,
            laser_angle=laser_angle_next,
            debris_x=cur_deb_x,
            debris_y=deb_y_next,
            debris_vy=cur_deb_vy,
            debris_radius=cur_deb_r,
            debris_damage=cur_deb_dmg,
            debris_active=deb_act_next,
            debris_type=cur_deb_t,
            time=time_next,
            security_gauge=rm["gauge_final"],
            is_overload=rm["is_overload_next"],
            overload_timer=rm["overload_timer_next"],
            boss_hp=state.boss_hp,
            boss_shield=rm["boss_shield_next"],
            shield_active=rm["shield_active_next"],
            tracking_laser_timer=rm["tl_timer_next"],
            tracking_laser_lock_x=rm["tl_lock_x_next"],
            tracking_laser_state=rm["tl_state_next"],
            electric_floor_warning=rm["electric_floor_warning_next"],
            electric_floor_active=rm["electric_floor_active_next"],
        )

        # 8. Aligned Shaped Reward (v4: Debris-Aware Clean Bait & Dodge)
        gated_boss_hit = rm["laser_hits_boss"] & rm["just_entered_firing"]
        reward = _compute_reward(
            took_hit,
            total_dmg,
            gated_boss_hit,
            rm["laser_hits_player"],
            rm["shield_shatter"],
            rm["triggers_overload"],
            rm["gauge_final"],
            rm["is_overload_next"],
            px_next,
            py_next,
            hp_next,
            rm["tl_state_next"],
            rm["tl_lock_x_next"],
            cur_deb_x,
            deb_y_next,
            cur_deb_r,
            deb_act_next,
            params,
        )
====
        # 7. Next Immutable State
        state_next = EnvState(
            player_x=px_next,
            player_y=py_next,
            player_vx=vx,
            player_vy=vy_final,
            player_hp=hp_next,
            player_on_ground=on_ground_next,
            invincible_timer=inv_timer_next,
            laser_angle=laser_angle_next,
            debris_x=cur_deb_x,
            debris_y=deb_y_next,
            debris_vy=cur_deb_vy,
            debris_radius=cur_deb_r,
            debris_damage=cur_deb_dmg,
            debris_active=deb_act_next,
            debris_type=cur_deb_t,
            time=time_next,
            last_action=action,
            security_gauge=rm["gauge_final"],
            is_overload=rm["is_overload_next"],
            overload_timer=rm["overload_timer_next"],
            boss_hp=state.boss_hp,
            boss_shield=rm["boss_shield_next"],
            shield_active=rm["shield_active_next"],
            tracking_laser_timer=rm["tl_timer_next"],
            tracking_laser_lock_x=rm["tl_lock_x_next"],
            tracking_laser_state=rm["tl_state_next"],
            electric_floor_warning=rm["electric_floor_warning_next"],
            electric_floor_active=rm["electric_floor_active_next"],
        )

        # 8. Aligned Shaped Reward (v5: Threat-Gated Evasion & Micro-Movement)
        gated_boss_hit = rm["laser_hits_boss"] & rm["just_entered_firing"]
        reward = _compute_reward(
            took_hit,
            total_dmg,
            gated_boss_hit,
            rm["laser_hits_player"],
            rm["shield_shatter"],
            rm["triggers_overload"],
            rm["gauge_final"],
            rm["is_overload_next"],
            px_next,
            py_next,
            hp_next,
            rm["tl_state_next"],
            rm["tl_lock_x_next"],
            cur_deb_x,
            deb_y_next,
            cur_deb_r,
            deb_act_next,
            action,
            state.last_action,
            on_ground_next,
            state.player_x,
            params,
        )
>>>>
```

#### Target 5: `reset_env` Updates (Lines 754–766)
```python
<<<<
            time=0,
            security_gauge=0.0,
====
            time=0,
            last_action=0,
            security_gauge=0.0,
>>>>
```

---

## 5. Verification & Invalidation Analysis

### 5.1 XLA JIT & Branch-Free Invariant Verification
1. **No Python dynamic control flow**: All conditional logic uses `jnp.where` with pre-computed boolean masks (`is_threat_overhead`, `can_tap_dodge`, `is_jump_action`).
2. **Static Shapes**: All debris arrays retain compile-time static dimensions `(30,)`. `jnp.any` produces scalar boolean tensors compatible with `jnp.where`.
3. **Boolean Typing (Rule 11)**: `is_airborne` is computed via `jnp.logical_not(on_ground_next)`, avoiding Python `~` bitwise inversion integer traps.
4. **PyTree Alignment**: `last_action` is a standard scalar leaf in `EnvState`. It is compatible with `jax.vmap`, `jax.lax.scan`, `LogWrapper`, and Orbax checkpointing.

### 5.2 Decoupled Physics Assertions (Rule 8 Compliance)
Tests verifying physical mechanics (kinematics, collisions, damage, death) must assert strictly on `state` variables (`player_hp`, `player_vy`, `player_on_ground`, `info["debris_hit"]`), never on shaped reward floats.
Tests asserting on reward shaping (e.g. `test_reward_shaping_gauge_and_safe_zone_alignment`) should add exact analytical assertions for the newly introduced terms:
- Jitter delta: `abs((r_same - r_switch) - 0.02) < 1e-4`
- Jump action cost: `abs((r_ground - r_jump) - 0.05) < 1e-4`
- Overhead hazard penalty: `abs((r_grounded - r_airborne) - 0.35) < 1e-4`
- Tap-dodge bonus: `abs((r_dodge - r_stationary) - 0.25) < 1e-4`

---

## 6. Synthesis & Recommendations for Implementer Subagent

| Requirement | Target File | Recommended Action |
| :--- | :--- | :--- |
| **Action Space Re-ordering** | `src/maple_gymnax/envs/common.py` | Remap actions: 0: NOOP, 1: LEFT, 2: RIGHT, 3: DOWN/DUCK, 4: JUMP, 5: JUMP_LEFT, 6: JUMP_RIGHT. Provide `ACTION_DOWN` alongside `ACTION_DUCK = 3` alias. |
| **Module Export** | `src/maple_gymnax/envs/__init__.py` | Export `ACTION_DOWN` and `DOWN`. |
| **EnvParams Extension** | `src/maple_gymnax/envs/lotus_phase1.py` | Add hyperparameters: `r_action_jump_cost = -0.05`, `r_jitter_cost = -0.02`, `r_airborne_hazard_cost = -0.35`, `r_tap_dodge_bonus = 0.25`, `debris_repel_scale = -0.30`. |
| **EnvState Extension** | `src/maple_gymnax/envs/lotus_phase1.py` | Add `last_action: int = 0` to `EnvState`. |
| **Reward Engine Overhaul** | `src/maple_gymnax/envs/lotus_phase1.py` | Pass `action`, `state.last_action`, `on_ground_next`, `state.player_x` into `_compute_reward`. Implement `r_action_jump`, `r_jitter`, `r_airborne_hazard`, `r_tap_dodge`, and scale Gaussian potential from `-0.05` to `-0.30`. |
| **Lifecycle Integration** | `src/maple_gymnax/envs/lotus_phase1.py` | Initialize `last_action=0` in `reset_env` and assign `last_action=action` in `step_env`. |
| **Unit Test Coverage** | `tests/test_lotus_phase1.py` | Add targeted tests for R1 action cost, jitter, and R2 overhead hazard anti-jump and tap-dodge clearance bonuses. |
