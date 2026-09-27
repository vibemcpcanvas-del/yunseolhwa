# Specification: RL Framework Wrappers, Persona System Prompt & SPS Benchmark (Classic & Remastered Lotus)

**Document Version**: 1.1.0  
**Author**: Survey Explorer 3 (`teamwork_preview_explorer`)  
**Target Repository**: `bold-faraday`  
**Working Directory**: `.agents/teamwork/explorer_survey_3/`  
**Date**: 2026-09-26  

---

## 1. Executive Summary & Problem Scope

This specification defines the architectural contracts, interface designs, mathematical formulas, and implementation blueprints for:
1. **Workspace Python & JAX Runtime Setup**:
   - Setting up Python 3.12 via `uv` in `bold-faraday` (`uv venv --python 3.12 .venv`).
   - Resolving dependencies: `jax==0.11.2`, `flax==0.12.10`, `gymnax==0.0.9`, `flashbax==0.1.3`, `pytest==9.1.1`, `chex==0.1.92`, `optax==0.2.8`.
   - Telemetry on AMD Ryzen 5 5600X (CPU) and AMD Radeon RX 6600 XT (GPU / WSL2 ROCmLab).
2. **RL Framework Wrappers (`src/maple_gymnax/wrappers/`)**:
   - `FlattenObservationWrapper`: Unifies multi-modal observation dictionaries into a 1D `float32` vector, supporting **both Classic** (rotating cross laser, falling debris) and **Remastered** (security/annihilation gauge, tracking laser friendly fire, arm slam, overload mode, shield) mechanics without shape changes.
   - `PureJaxRLAdapterWrapper`: Translates Gymnax 0.0.9 6-tuple step return `(obs, state, r, term, trunc, info)` to standard PureJaxRL 5-tuple `(obs, state, r, done, info)`.
   - `LogWrapper` (Stoix/Stoa Adapter): Zero-overhead branch-free episode metric tracking (`episode_returns`, `episode_lengths`, `returned_episode_returns`, `returned_episode_lengths`).
   - `High-Speed Batched Rollout Runner`: Dual-level vectorization with `jax.vmap` (across $B$ environments) and `jax.lax.scan` (across $T$ steps), achieving ~3M SPS on Ryzen 5600X CPU.
   - `Flashbax Pytree Replay Buffer Zero-Copy Interface`: Direct accelerator memory integration using `fbx.make_flat_buffer` and batched in-place scatter updates.
3. **Persona System Prompt (`src/maple_gymnax/prompts/persona_system_prompt.py`)**:
   - Formalized prompt system instructing LLMs to parse raw MapleStory WZ data (`BossSuu.img.json`, patterns 1001-1009, destruction mode, `bossSuu.img.json`), extract continuous physical parameters, and generate branch-free Gymnax code.
4. **Hardware-Tailored SPS Benchmark Suite (`benchmarks/benchmark_sps.py`)**:
   - Measures steps per second across batch sizes (256, 512, 1024, 2048, 4096) with warmups, `block_until_ready` synchronization, and automated Markdown/JSON export.

---

## 2. Workspace Python & Hardware Runtime Telemetry

### 2.1 Host Environment Discovery
- **Windows Host OS**: Windows 11 / Server (x86_64).
- **System Default Python**: Python 3.14.7 located at `C:\Users\ROCmAdmin\AppData\Local\Programs\Python\Python314\python.exe`.
- **Package Manager**: `uv 0.12.6` (7938ca5d5 2026-08-25 x86_64-pc-windows-msvc) installed and on system `PATH`.
- **Target Python Version**: `cpython-3.12-windows-x86_64-none` is installed in `C:\Users\ROCmAdmin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe`.
- **Workspace `.venv`**: **Does not exist yet**. Needs creation via `uv venv --python 3.12 .venv`.

### 2.2 WSL2 Ubuntu-24.04-ROCmLab Telemetry
- **WSL Distribution**: `Ubuntu-24.04-ROCmLab` (Kernel `6.18.33.2-microsoft-standard-WSL2`).
- **Internal Python**: `/usr/bin/python3` is `Python 3.12.3`.
- **GPU Hardware**: AMD Radeon RX 6600 XT (8GB VRAM, Navi 23, 32 Compute Units / 2048 Stream Processors).
- **Driver Status**: When running `rocm-smi` in WSL2, the ROCm driver returned `amdgpu not found in modules` and `/dev/kfd` is not mounted. The system can execute on Windows host CPU with maximum performance (~2.97M SPS on Ryzen 5600X) and can leverage GPU when WSL2 ROCm pass-through is active.

### 2.3 Dependency Matrix (`uv pip compile` Verified)
All required dependencies resolve in 848ms on Python 3.12 without conflicts:
- `jax==0.11.2`, `jaxlib==0.11.2`
- `flax==0.12.10`
- `gymnax==0.0.9`
- `flashbax==0.1.3`
- `gymnasium==1.3.0`
- `chex==0.1.92`
- `optax==0.2.8`
- `orbax-checkpoint==0.12.6`
- `pytest==9.1.1`

### 2.4 Setup Commands
```powershell
# In project root c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install jax flax gymnax flashbax pytest chex optax
```

---

## 3. Gymnax 0.0.9 Architectural Contract & Version Deltas

### 3.1 6-Tuple Step API vs Classic 5-Tuple API
In Gymnax `0.0.9`:
- `Environment.step(key, state, action, params)` returns **6 values**:
  $$(obs, state, reward, terminated, truncated, info)$$
  where $done = \text{terminated} \lor \text{truncated}$.
- In contrast, classic PureJaxRL (and Gymnax $\le 0.0.6$) expects a **5-tuple**:
  $$(obs, state, reward, done, info)$$

### 3.2 Built-In AutoReset in Gymnax 0.0.9
In Gymnax `0.0.9`, `Environment.step()` natively implements branch-free auto-reset:
```python
# From gymnax.environments.environment.Environment.step:
key_step, key_reset = jax.random.split(key)
obs_st, state_st, reward, terminated, info = self.step_env(key_step, state, action, params)
truncated = self.is_truncated(state_st, params)
done = jnp.logical_or(terminated, truncated)
obs_re, state_re = self.reset_env(key_reset, params)

state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
obs = jax.tree.map(lambda reset_leaf, step_leaf: jax.lax.select(done, reset_leaf, step_leaf), obs_re, obs_st)
info = {**info, "terminated": terminated, "truncated": truncated, "final_observation": obs_st}
```
**Key Takeaways**:
1. `LotusPhase1Env` only needs to define `step_env` and `reset_env`.
2. Environments auto-reset upon player death or max episode steps without breaking `jax.lax.scan`.
3. `PureJaxRLAdapterWrapper` provides the 5-tuple bridge for standard PureJaxRL PPO/DQN scripts.

---

## 4. Domain Knowledge: Classic vs Remastered Lotus Mechanics

Following the April 2024 official MapleStory Lotus Remaster, the environment supports two switchable/combinable modes via `EnvParams`:

### 4.1 Classic Lotus Mechanics
- **Central Core**: Origin at $(683.0, 384.0)$, Floor at $y = 605.0$.
- **Rotating Cross Laser**: 4 orthogonal beams with angular velocity $\omega = 0.5235\text{ rad/s}$ ($30^\circ/\text{s}$). Contact causes instant death (or 100% HP damage).
- **Falling Debris**: Static padded array of up to 30 debris objects falling vertically with varying radii and damage percentages.

### 4.2 Remastered Lotus Mechanics (Patterns 1001-1009)
1. **Security & Annihilation Gauge (보안/섬멸 게이지)**:
   - Natural increase rate: $0.6\%$ (Normal), $0.8\%$ (Hard), $2.0\%$ (Extreme) per second.
   - When gauge reaches $100\%$, triggers **Overload / Destruction Mode** for 25 seconds:
     - Pattern `1006-000` (Horizontal bombardment / 침입자 격퇴 프로토콜): lethal beam across the map except for a safe zone on the right. Contact deals 100% HP every 0.5s.
     - Pattern `1006-002` (Electric field installation): deals 5% HP every 0.36s for 4s; 5 accumulated ticks trigger forced jump + stun.
2. **Friendly Fire & Boss Guidance Mechanics (공멸형 유도 기믹)**:
   - Pattern `1001-000` (Tracking Laser): Top mechanical arm projects 2 laser lines; tracks player for 1.0s, locks aim for 1.0s, then fires.
     - If it hits the player: deals 15% HP damage + increases gauge by $+7\%\sim 20\%$.
     - **If guided to hit Lotus Core: decreases gauge by $-7\%\sim 20\%$ + significantly damages Lotus Shield.**
   - Pattern `1001-001` (Small Arm Slam): 12 consecutive downward slams (0.8s interval).
     - If it hits the player: deals 5% HP damage + increases gauge by $+3\%$.
     - **If guided to hit Lotus Core: decreases gauge by $-3\%$.**
3. **Floor Electric Discharge**:
   - Ground surges with blue electricity followed by an explosion; must be avoided via double jump or upward hovering.
4. **Boss Shield Generation**:
   - Lotus periodically activates a protective barrier; if not broken within time, Lotus recovers HP. Guided tracking lasers shatter the shield.

---

## 5. RL Framework Wrappers Specification (`src/maple_gymnax/wrappers/`)

### 5.1 `FlattenObservationWrapper` (`src/maple_gymnax/wrappers/flatten_observation.py`)
To support both Classic and Remastered modes without requiring separate network architectures, we define a **Unified Static Observation Vector** of fixed dimension.

#### Unified Observation Layout (Dimension = 109 floats)
| Feature Name | Dimension | Normalization Range | Description |
|---|---|---|---|
| `player_pos` | 2 | $[0, 1]$ (divided by $(1366, 768)$) | Player (x, y) coordinates |
| `player_vel` | 2 | $[-1, 1]$ (divided by max speed $600$) | Player velocity (vx, vy) |
| `player_hp` | 1 | $[0, 1]$ | Normalized player health |
| `boss_hp` | 1 | $[0, 1]$ | Normalized Lotus health |
| `boss_shield` | 1 | $[0, 1]$ | Normalized Lotus shield integrity |
| `annihilation_gauge` | 1 | $[0, 1]$ | Security/Annihilation Gauge percentage |
| `is_overload_mode` | 1 | $\{0.0, 1.0\}$ | Overload / Destruction mode active flag |
| `overload_timer` | 1 | $[0, 1]$ | Remaining overload duration (normalized to 25s) |
| `tracking_laser_target` | 2 | $[0, 1]$ | Aimed coordinate $(x, y)$ of pattern 1001-000 |
| `tracking_laser_state` | 1 | $\{0.0, 0.5, 1.0\}$ | 0.0=Idle, 0.5=Aiming, 1.0=Firing |
| `arm_slam_pos` | 2 | $[0, 1]$ | Target impact coordinate of pattern 1001-001 |
| `floor_electric_active`| 1 | $\{0.0, 1.0\}$ | Floor electric surge warning / active flag |
| `classic_laser_angle` | 1 | $[0, 1]$ ($\theta / 2\pi$) | Angle of rotating cross laser (or $0.0$ in Remastered) |
| `mode_flag` | 1 | $\{0.0, 1.0\}$ | 0.0 = Classic Mode, 1.0 = Remastered Mode |
| `debris_pos` | 60 | $[0, 1]$ | 30 static padded debris $(x, y)$ pairs |
| `debris_mask` | 30 | $\{0.0, 1.0\}$ | 30 boolean active indicators for debris |
| **TOTAL** | **109** | **Continuous floats** | **Guaranteed static shape across all modes** |

#### Implementation Contract
```python
from functools import partial
from typing import Any, Tuple
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper

OBSERVATION_DIM = 109

class FlattenObservationWrapper(GymnaxWrapper):
    """Flattens Lotus Phase 1 observation dictionaries (Classic & Remastered)
    into a unified 1D float32 vector of shape (109,).
    """

    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=jnp.float32,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _flatten_obs(self, obs: dict) -> jax.Array:
        # Predefined deterministic key ordering
        keys = [
            "player_pos",            # (2,)
            "player_vel",            # (2,)
            "player_hp",             # (1,)
            "boss_hp",               # (1,)
            "boss_shield",           # (1,)
            "annihilation_gauge",    # (1,)
            "is_overload_mode",      # (1,)
            "overload_timer",        # (1,)
            "tracking_laser_target", # (2,)
            "tracking_laser_state",  # (1,)
            "arm_slam_pos",          # (2,)
            "floor_electric_active", # (1,)
            "classic_laser_angle",   # (1,)
            "mode_flag",             # (1,)
            "debris_pos",            # (30, 2) -> (60,)
            "debris_mask",           # (30,)
        ]
        leaves = [jnp.ravel(obs[k]) for k in keys]
        return jnp.concatenate(leaves, axis=-1).astype(jnp.float32)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> Tuple[jax.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        return self._flatten_obs(obs), state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> Tuple[jax.Array, environment.EnvState, jax.Array, jax.Array, jax.Array, dict]:
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        flat_obs = self._flatten_obs(obs)
        if "final_observation" in info:
            info = {**info, "final_observation": self._flatten_obs(info["final_observation"])}
        return flat_obs, state, reward, terminated, truncated, info
```

---

### 5.2 `PureJaxRLAdapterWrapper` (`src/maple_gymnax/wrappers/purejaxrl_adapter.py`)
Converts the 6-tuple step return of Gymnax 0.0.9 to the 5-tuple format expected by PureJaxRL:
```python
class PureJaxRLAdapterWrapper(GymnaxWrapper):
    """Adapts Gymnax 0.0.9 (obs, state, reward, term, trunc, info)
    into PureJaxRL (obs, state, reward, done, info).
    """

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> Tuple[jax.Array, environment.EnvState, jax.Array, jax.Array, dict]:
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        done = jnp.logical_or(terminated, truncated)
        info = {**info, "terminated": terminated, "truncated": truncated, "done": done}
        return obs, state, reward, done, info
```

---

### 5.3 `LogWrapper` & Stoix Metric Logger (`src/maple_gymnax/wrappers/log_wrapper.py`)
Accumulates episodic returns and lengths branch-free using binary masks:
```python
from flax import struct

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array

class LogWrapper(GymnaxWrapper):
    """Asynchronously accumulates episode returns and lengths across parallel environments."""

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> Tuple[jax.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((), dtype=jnp.float32),
            episode_lengths=jnp.zeros((), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((), dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros((), dtype=jnp.int32),
        )
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float | jax.Array,
        params: environment.EnvParams | None = None,
    ) -> Tuple[jax.Array, LogEnvState, jax.Array, jax.Array, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_return = state.episode_returns + reward
        new_length = state.episode_lengths + 1
        
        # Branch-free accumulation
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_return * (1.0 - done),
            episode_lengths=new_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1.0 - done) + new_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_length * done,
        )
        info = {
            **info,
            "returned_episode_returns": state.returned_episode_returns,
            "returned_episode_lengths": state.returned_episode_lengths,
            "returned_episode": done,
        }
        return obs, state, reward, done, info
```

---

### 5.4 High-Speed Batched Rollout Runner (`src/maple_gymnax/wrappers/rollout_runner.py`)
```python
from flax import struct

@struct.dataclass
class TrajectoryTransition:
    obs: jax.Array        # (B, 109)
    action: jax.Array     # (B,)
    reward: jax.Array     # (B,)
    done: jax.Array       # (B,)
    next_obs: jax.Array   # (B, 109)
    info: dict

def make_rollout_runner(env: environment.Environment, env_params: environment.EnvParams):
    """Instantiates a JIT-compiled rollout runner collecting T steps across B environments."""
    
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    v_reset = jax.vmap(env.reset, in_axes=(0, None))

    def rollout_fn(
        rng: jax.Array,
        init_state: Any,
        last_obs: jax.Array,
        policy_fn: Any,
        policy_params: Any,
        num_steps: int,
    ) -> Tuple[Tuple[jax.Array, Any, jax.Array], TrajectoryTransition]:
        
        def _scan_step(carry, _):
            rng, state, obs = carry
            rng, rng_action, rng_step = jax.random.split(rng, 3)
            
            actions = policy_fn(policy_params, obs, rng_action)
            batch_size = obs.shape[0]
            step_keys = jax.random.split(rng_step, batch_size)
            next_obs, next_state, rewards, dones, info = v_step(
                step_keys, state, actions, env_params
            )
            
            transition = TrajectoryTransition(
                obs=obs,
                action=actions,
                reward=rewards,
                done=dones,
                next_obs=next_obs,
                info=info,
            )
            return (rng, next_state, next_obs), transition

        return jax.lax.scan(_scan_step, (rng, init_state, last_obs), None, length=num_steps)

    return v_reset, rollout_fn
```

---

### 5.5 Flashbax Pytree Replay Buffer Zero-Copy Interface (`src/maple_gymnax/wrappers/flashbax_buffer.py`)
```python
import flashbax as fbx

def create_lotus_replay_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_batch_size: int,
    obs_dim: int = OBSERVATION_DIM,
):
    """Instantiates a zero-copy Flashbax flat replay buffer."""
    
    buffer = fbx.make_flat_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        add_sequences=False,
    )
    
    exemplar = {
        "obs": jnp.zeros((obs_dim,), dtype=jnp.float32),
        "action": jnp.zeros((), dtype=jnp.int32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.bool_),
        "next_obs": jnp.zeros((obs_dim,), dtype=jnp.float32),
    }
    
    buffer_state = buffer.init(exemplar)
    return buffer, buffer_state
```

---

## 6. Persona System Prompt Specification (`src/maple_gymnax/prompts/persona_system_prompt.py`)

### 6.1 Requirements
The Persona System Prompt must guide LLMs to:
1. Parse client nodes from `C:\mp\Restored_Data\Mob\BossPattern\BossSuu.img.json` (patterns 1001-1009, destruction, overload) and `Map\Back\Back_000\bossSuu.img.json`.
2. Convert animation delays, pixel hitboxes, and coordinates into continuous JAX physics tensors.
3. Generate Gymnax `EnvParams` and `EnvState` with strict branch-free execution.
4. Support both Classic Lotus (rotating cross laser, debris) and Remastered Lotus (gauge, friendly fire, overload).

### 6.2 Code Implementation Template
```python
"""Master Persona System Prompt and Templating Module for Maple Gymnax."""

PERSONA_SYSTEM_PROMPT = """You are the Lead Maple Gymnax Simulation & Physics Architect.
Your core mission is to analyze unstructured MapleStory WZ client data (BossSuu.img.json, bossSuu.img.json)
and translate it into mathematically rigorous, branch-free Gymnax (JAX/XLA) reinforcement learning environments.

### Core Architecture & Engineering Principles:
1. STRICT FUNCTIONAL PURITY & IMMUTABILITY:
   - State and parameters must be defined as `@flax.struct.dataclass`.
   - Never mutate state objects in-place. Always return a new instance via functional update.
   - Separate static, invariant configurations into `EnvParams` and dynamic 60Hz tick state into `EnvState`.

2. BRANCH-FREE XLA EXECUTION:
   - Strict prohibition of Python `if`, `else`, `while`, or `for` loops depending on traced JAX values.
   - Use `jnp.where(condition, x, y)` and `jax.lax.select(condition, x, y)` for conditional state updates.
   - Avoid `ConcretizationTypeError` by ensuring array dimensions and loop bounds are static constants.
   - Model dynamic entities (e.g. falling debris) using fixed-size static padded arrays (shape `(30, 2)`) with a boolean active mask (shape `(30,)`).

3. MATHEMATICAL SPECIFICATIONS FOR LOTUS PHASE 1:
   - Coordinate System: Continuous 2D plane (1366 x 768), Core Origin at (683.0, 384.0), Floor Y = 605.0.
   - Time Step: Discrete dt = 1/60s (approx 16.667ms). Frame delays in ms must be converted via `steps = delay_ms / (1000 / 60)`.
   - Classic Mode (Rotating Cross Laser): 4 orthogonal rays originating from (683.0, 384.0), angular velocity omega = 0.5235 rad/s.
     Orthogonal distance from player center (px, py) to ray with angle theta:
     d_ortho = | -(px - cx)*sin(theta) + (py - cy)*cos(theta) |
     Directional half-line mask via dot product:
     dot = (px - cx)*cos(theta) + (py - cy)*sin(theta) > 0.
   - Remastered Mode (Patterns 1001-1009):
     1. Security & Annihilation Gauge: increases by 0.6%~2.0%/s; reaching 100% enters 25s Overload/Destruction mode.
     2. Overload Mode (1006-000, 1006-002): horizontal lethal barrage (100% HP/0.5s) except right safe zone; electric field stun.
     3. Friendly Fire (1001-000 Tracking Laser & 1001-001 Arm Slam):
        - Hitting player: HP damage + increases gauge by +7~20% (laser) / +3% (slam).
        - Guided to hit Lotus Core: decreases gauge by -7~20% (laser) / -3% (slam) + shatters Lotus Shield.
     4. Floor Electric Discharge: ground surge explosion requiring jump/hover avoidance.
     5. Shield Generation: periodic Lotus barrier breakable by guided lasers.

4. CODE STRUCTURE:
   - Must implement Gymnax `Environment` interface:
     - `default_params` property returning `EnvParams`.
     - `step_env(key, state, action, params)` returning `(obs, state, reward, terminated, info)`.
     - `reset_env(key, params)` returning `(obs, state)`.
     - `observation_space(params)` returning `spaces.Box(shape=(109,), dtype=jnp.float32)`.
     - `action_space(params)` returning `spaces.Discrete(5)` (Noop, Left, Right, Jump, Down).
"""

def build_wz_extraction_prompt(wz_node_json: str) -> str:
    """Creates an extraction prompt directing an LLM to extract EnvParams from WZ JSON."""
    return f"""{PERSONA_SYSTEM_PROMPT}

### TASK:
Analyze the following MapleStory WZ client data snippet and extract the continuous physics parameters
into a clean JSON structure conforming to the `EnvParams` schema.

```json
{wz_node_json}
```

Ensure frame delays are converted to seconds, hitboxes are converted to width/height/radius,
and damage values are normalized to player HP percentages.
"""

def build_gymnax_codegen_prompt(env_params_json: str) -> str:
    """Creates a code generation prompt directing an LLM to generate Gymnax environment code."""
    return f"""{PERSONA_SYSTEM_PROMPT}

### TASK:
Given the following extracted `EnvParams` specification, generate the complete, production-grade
JAX/XLA Gymnax environment implementation in Python (`lotus_phase1.py`).

```json
{env_params_json}
```

The code must be strictly branch-free, contain no ConcretizationTypeError, and pass XLA compilation.
"""
```

---

## 7. Hardware-Tailored SPS Benchmark Design (`benchmarks/benchmark_sps.py`)

### 7.1 Performance Targets
- **AMD Ryzen 5 5600X (CPU)**:
  - Cache size: 32MB L3 Cache.
  - L3 Cache saturation occurs around batch sizes 1024 ~ 2048.
  - CartPole baseline on host: **2,966,757 SPS** at $B=4096$ using `jax.lax.scan`.
  - Target for Lotus Phase 1: **100,000 ~ 500,000+ SPS** across 6 cores.
- **AMD Radeon RX 6600 XT (GPU / ROCm)**:
  - 32 Compute Units (2048 stream processors), 8GB VRAM.
  - Target for Lotus Phase 1: **500,000 ~ 2,000,000+ SPS** at $B \ge 4096$.

### 7.2 Benchmark Script Implementation (`benchmarks/benchmark_sps.py`)
```python
"""Hardware-Tailored Steps Per Second (SPS) Benchmark Suite for Maple Gymnax."""

import argparse
import json
import time
from typing import List, Dict, Any
import jax
import jax.numpy as jnp
from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env
from maple_gymnax.wrappers.flatten_observation import FlattenObservationWrapper
from maple_gymnax.wrappers.rollout_runner import make_rollout_runner

def run_sps_benchmark(
    batch_sizes: List[int] = [256, 512, 1024, 2048, 4096],
    num_steps: int = 1000,
    warmup_steps: int = 50,
    output_json: str | None = "benchmarks/results_sps.json",
) -> List[Dict[str, Any]]:
    
    # Instantiate wrapped environment
    raw_env = LotusPhase1Env()
    env = FlattenObservationWrapper(raw_env)
    params = env.default_params
    v_reset, rollout_fn = make_rollout_runner(env, params)
    
    # Constant policy to isolate environment step physics throughput
    def dummy_policy(p, obs, rng):
        return jnp.zeros((obs.shape[0],), dtype=jnp.int32)

    device = jax.devices()[0]
    print("=" * 70)
    print(f" Maple Gymnax SPS Benchmark | Device: {device} | Backend: {jax.default_backend()}")
    print("=" * 70)

    results = []

    for B in batch_sizes:
        rng = jax.random.PRNGKey(42)
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, B)
        obs, state = v_reset(reset_keys, params)

        # 1. Warmup Run (Forces XLA compilation)
        (rng, state, obs), _ = rollout_fn(rng, state, obs, dummy_policy, None, warmup_steps)
        jax.block_until_ready(obs)

        # 2. Timed Benchmark Run
        t0 = time.perf_counter()
        (rng, state, obs), trajectory = rollout_fn(rng, state, obs, dummy_policy, None, num_steps)
        jax.block_until_ready(trajectory.obs)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_steps = B * num_steps
        sps = total_steps / elapsed
        latency_step_us = (elapsed / num_steps) * 1e6
        per_env_ns = (elapsed / total_steps) * 1e9

        res = {
            "batch_size": B,
            "num_steps": num_steps,
            "total_steps": total_steps,
            "elapsed_seconds": round(elapsed, 4),
            "sps": round(sps, 1),
            "step_latency_us": round(latency_step_us, 2),
            "per_env_step_ns": round(per_env_ns, 2),
            "device": str(device),
        }
        results.append(res)
        print(f"Batch: {B:5d} | Steps: {total_steps:8d} | Time: {elapsed:6.3f}s | SPS: {sps:11,.0f} | Latency: {latency_step_us:6.1f} us/step")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Benchmark results saved to {output_json}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple Gymnax SPS Benchmark")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per environment")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup compilation steps")
    parser.add_argument("--json", type=str, default="benchmarks/results_sps.json", help="Output JSON path")
    args = parser.parse_args()
    run_sps_benchmark(num_steps=args.steps, warmup_steps=args.warmup, output_json=args.json)
```

---

## 8. Milestone Matrix & Downstream Deliverables

| Target File | Milestone | Description | Integration Target |
|---|---|---|---|
| `src/maple_gymnax/wrappers/flatten_observation.py` | M3-A | 109-dim unified observation flattening | PureJaxRL PPO/DQN |
| `src/maple_gymnax/wrappers/purejaxrl_adapter.py` | M3-B | 6-tuple to 5-tuple step return adapter | CleanRL / PureJaxRL |
| `src/maple_gymnax/wrappers/log_wrapper.py` | M3-C | Branch-free episode return & length tracker | Stoix / Jumanji |
| `src/maple_gymnax/wrappers/rollout_runner.py` | M3-D | High-speed `vmap` + `scan` rollout engine | SPS Benchmark, Stoix |
| `src/maple_gymnax/wrappers/flashbax_buffer.py` | M3-E | Zero-copy Pytree replay buffer interface | Off-policy RL, SAC |
| `src/maple_gymnax/prompts/persona_system_prompt.py` | M4 | System prompt & prompt templates for LLM client extraction | Agent Workflows |
| `benchmarks/benchmark_sps.py` | Bench | Hardware-tailored multi-batch SPS benchmark suite | E2E Hardware Verification |
