---
trigger: model_decision
description: Best practices for implementing high-throughput JAX/Gymnax RL environments, Flax dataclass static metadata, pure functional wrappers, and reward design.
---

# JAX / Gymnax RL Environment Guidelines

When implementing or modifying JAX-accelerated simulation environments:

1. **Flax Dataclass Static Metadata**:
   - Any parameter that dictates tensor shapes, array capacities, or discrete environment modes MUST be marked with `flax.struct.field(pytree_node=False)`.
   - Never attempt dynamic array sizing or Python `bool()` conversions on PyTree leaves inside JIT-compiled functions.

2. **Branch-Free Pure Vectorization**:
   - Use `jnp.where` or `jax.lax.select` instead of Python `if/else` for runtime simulation states.
   - Padded static arrays with boolean activity masks (e.g. `MAX_DEBRIS = 30`) must be used for dynamic particle/hazard pools.

3. **RL Wrappers & Buffer Contracts**:
   - Wrappers MUST inherit from `gymnax.environments.environment.Environment` or conform strictly to its interface.
   - Never use mutable Python structures (`list`, `dict` mutations) or Python `int()` conversions inside step/sampling functions.
   - Replay buffers must adhere to `flashbax` pure functional state patterns (`init`, `add`, `sample`, `can_sample`).
   - `LogWrapper` must implement branch-free auto-reset on `done=True` to prevent dead-step accumulation during `jax.lax.scan`.

4. **Reward Design & Anti-Reward-Hacking**:
   - Avoid proximity-based reward shaping near lethal hazards to prevent reward hacking.
   - Shape rewards around game-theoretic survival, resource suppression (e.g. boss rage/security gauges), and safe-zone positioning during lethal phases.
   - **Passive Survival Dominance Check**: Ensure base survival reward per step (`r_base + r_hp`) does not dominate the total expected reward budget to the point where risk-avoidant camping (wall-hugging, corner-hiding) becomes the globally optimal strategy. The per-step passive income should be << single gimmick event reward.
   - **Boundary Exploitation Audit**: When hazard spawn zones have bounded coordinate ranges (e.g. `debris x ∈ [wall_left+50, wall_right-50]`), verify that arena boundaries outside the spawn range do not create zero-risk camping spots. Add explicit wall-proximity penalties or expand spawn ranges to cover the full playable area.
   - **Gimmick Reward Scaling**: For multi-step gimmick chains (e.g. lure tracking laser → dodge → laser hits boss), ensure the total gimmick reward substantially exceeds the worst-case damage penalty for a failed attempt. A ratio of ≥ 0.5:1 (gimmick reward : damage penalty) is recommended to make exploration NPV-positive.
   - **Contextual Penalty Gating**: Wall/boundary penalties must be disabled during phases where boundary-camping IS the correct strategy (e.g. Overload safe-zone at x≥1150).

5. **Pure Inlined Modular Helpers**:
   - Keep `step_env` concise by extracting discrete physical steps into pure, undecorated Python functions.
   - Pure functions called inside `step_env` are seamlessly inlined by XLA without runtime tracing overhead.

6. **Orbax & Host Checkpointing Invariants**:
   - Reserve `jax.debug.callback` strictly for non-blocking host telemetry and metric printing; do not dispatch async disk checkpointing inside callback worker threads.
   - Always save Orbax checkpoints with `force=True` to allow safe overwriting.
   - Explicitly call `checkpointer.wait_until_finished()` and `checkpointer.close()` before process exit to prevent interpreter shutdown race conditions.

7. **Host Visualization & Zero-Copy State Batching**:
   - Never access individual fields of JAX `EnvState` DeviceArrays inside rendering loops (e.g., `float(state.player_x)` or `state.debris_x[i]`).
   - Use `jax.device_get(state)` once per simulation tick to batch-transfer the entire state PyTree to CPU memory before passing it to host renderers (such as Pygame, OpenCV, or Matplotlib).
   - Convert static hazard buffers (e.g. `debris_x`, `debris_active`) to `np.asarray()` during host state unpacking to allow vectorized CPU masking (`np.where(active)[0]`) with zero accelerator sync stalls.

8. **Physical Contract Decoupling & Test Pre-Assertions**:
   - **Decouple Physics from Reward Scale**: Tests verifying core simulation physics (collisions, lethal hits, knockbacks, invincibility, death, ground contacts) MUST NEVER assert against shaped reward values (e.g. `assert reward < -50.0`). Assert strictly against physical contracts (`done`, `hp`, `invincible_timer`, `info["..._hit"]`) so that reward hyperparameter tuning never causes false test regressions.
   - **Gimmick Pre-Assertion Invariants**: When testing conditional defense mechanics (invincibility frames, shields, dodge buffers), always pre-assert that the attack actually made contact (e.g. `assert info["laser_hit"] == True`) before verifying HP preservation. This prevents false-positive test passes where the hazard simply missed the agent.
   - **Exact Analytical Reward Assertions**: When testing reward shaping formulas, do not rely solely on loose inequalities (`r_low > r_high`). Directly assert that the reward delta matches the analytical formula within float tolerance (e.g. `assert abs((r_low - r_high) - coef * delta) < 1e-4`).
   - **Directional Ray Masking Completeness**: Tests validating directional raycast dot products (e.g. beam-behind masking) must verify both the safe quadrant (masked, no hit) and the active beam path (unmasked, hit == True) without leaving unused dummy states.

9. **Chunked Outer Scan & Unbuffered Telemetry**:
   - **Chunked Outer Scan Pattern**: For long-horizon RL training (hundreds to thousands of updates), do not compile the entire run into a single monolithic `jax.lax.scan`. Decouple into `(init_fn, update_chunk_fn)` via `make_train_step` and run an outer Python loop over chunks (e.g. 50 updates). Synchronize via `jax.block_until_ready` and persist Orbax checkpoints at the end of every chunk to protect against OOMs, hardware crashes, and process interruption.
   - **Unbuffered Host Telemetry**: All console prints inside asynchronous host callbacks (`jax.debug.callback`) MUST explicitly specify `flush=True` (e.g. `print(..., flush=True)`). This guarantees immediate progress visibility across Windows background tasks, piped CLI runners, and subagent process monitoring without buffering delays.

10. **Colab Accelerator Portability & Cloud Burst Invariants (TPU/GPU Hybrid Scaling)**:
   - **Physical Simulation `float32` Guarantee**: `EnvParams` and `EnvState` coordinate calculations, SAT collision projections, and jump velocities must remain strictly in `float32` across all accelerators (TPU/GPU/CPU) to prevent sub-pixel collision rounding drift.
   - **Actor-Critic Hybrid Precision**: Policy and Value networks should use `dtype=bfloat16` compute with `param_dtype=float32` and `float32` logit/value output heads to maximize TPU Matrix Multiply Units (MXUs) and Tensor Core throughput without probability distribution or GAE instability.
   - **Ephemeral Preemption Protection**: Cloud accelerator jobs dispatched via ephemeral runtimes (such as Google Colab CLI) must utilize live chunk callbacks (`--on_checkpoint_cmd`) to stream intermediate Orbax weights to host persistence storage at every chunk completion, protecting against VM quota timeouts and spot preemption.

11. **Single-Event Reward Gating, JAX Boolean Typing, and Trajectory Alignment**:
   - **Single-Event Gating for Duration Hazards**: Event rewards (e.g., boss guidance hits, shield breaks) and one-time state mutations (gauge deltas) from multi-tick duration hazards (such as 30-tick laser firing) must be gated by state transition flags (`just_entered_firing` / `state == 1 & timer <= 0`) rather than persistent active state conditions (`state == 2`). This prevents sustained multi-tick reward hacking.
   - **JAX Boolean Array Typing Invariants**: Never apply bitwise inversion `~` directly to raw Python boolean expressions (e.g. `~(mode != 0)`). In Python, `~True` yields integer `-2`, which is truthy in NumPy and JAX `where` conditions. Always wrap conditions with `jnp.bool_(...)` or use `jnp.logical_not(...)`.
   - **Rollout Trajectory Alignment**: In `jax.lax.scan` trajectory unrolls (`RolloutRunner`), derive `init_obs` directly from `initial_state` (never calling a disconnected `reset`), and record `prev_obs` alongside `action` in transition tuples $(s_t, a_t, r_t, d_t)$ so that the first observation matches the evaluated state without off-by-one misalignment.

12. **Modern JAX & Gymnax v1.0 Ecosystem Interop**:
   - **Gymnax v1.0 Step Return Unpacking**: Modern Gymnax follows the Gymnasium v1.0 API where `Environment.step` returns a 6-tuple `(obs, state, reward, terminated, truncated, info)`. Custom functional wrappers (`FlattenObservationWrapper`, `LogWrapper`, `PureJaxRLAdapterWrapper`) must dynamically inspect `len(out)` and support both 5-tuple (`obs, state, reward, done, info`) and 6-tuple returns with `done = jnp.logical_or(terminated, truncated)` to prevent `ValueError: too many values to unpack (expected 5)`.
   - **Legacy Dependency Downgrade Prevention (`--no-deps gymnax`)**: When installing legacy RL environment packages on cloud runtimes, always upgrade `jax[cuda12]` / `jax[tpu]` and core libraries (`flax`, `optax`, `chex`) first, and install `gymnax` with `--no-deps` to prevent its legacy `jax<0.7` PyPI metadata from silently downgrading `jaxlib` and causing PJRT C-API version mismatch aborts (`Unexpected PJRT_Plugin_Attributes_Args size: expected 32, got 24`).
   - **Flax Tracers Backward Compatibility**: In JAX 0.11+, `jax.core.get_opaque_trace_state` was moved to `jax.extend.core.get_opaque_trace_state`. Always inject a compatibility shim `if not hasattr(jax.core, 'get_opaque_trace_state'): jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state` to support existing Flax tracer code without requiring full framework refactoring.
   - **Quota Error Detection String Rigor**: Automated cloud orchestrators monitoring stdout for quota exhaustion (such as HTTP 503) must never search for bare status numbers (like `"503"`) inside training metric lines, which cause false-positive interruptions on negative reward or loss values (e.g. `Return: -503.79`). Use exact string signatures (`"503 Server Error"`, `"503 Service Unavailable"`) and filter out `"[Update"` metric lines.

13. **Resumable Checkpoints, Config Fingerprinting, and Ephemeral VM Preemption Recovery**:
   - **Full Deep PyTree Checkpointing**: Fault-tolerant checkpointing across VM preemption must persist the entire `RunnerState` (`train_state` with optimizer parameters and step count, inner `env_state`, `last_obs`, and PRNG key) rather than policy weights alone. Explicitly synchronize via `checkpointer.wait_until_finished()` before writing `manifest.json`.
   - **Orbax Typed PyTree Restoration (`target`)**: When restoring typed Flax `TrainState` and Gymnax `EnvState` tuples via `StandardCheckpointer.restore()`, pass an initialized template structure (`target={"runner": template_state, "update": 0}`) to ensure restored arrays populate typed NamedTuples/dataclasses rather than plain Python dictionaries.
   - **Strict Configuration Fingerprinting**: Maintain schema-versioned metadata in `manifest.json`. Enforce strict equality on `HARD_KEYS` (`mode`, `num_envs`, `num_steps`, `gamma`, `gae_lambda`, `clip_eps`, `ent_coef`, `vf_coef`, `max_grad_norm`, `num_minibatches`, `update_epochs`). Permit precision downcasts (`SOFT_KEYS`) only with explicit consent (`--allow_precision_loss`). Allow training budget expansion via `EXTENSIBLE_KEYS` (`num_updates`).
   - **True Host ACKs over Remote Streaming**: Streaming bridges must enforce `Content-Length` matches, verify end-to-end SHA256 checksums, catch archive extraction faults, and handle idempotent duplicate transfers (200) vs hash collisions (409 Conflict). Never mark an update step as verified for resumption based solely on unverified stdout log lines.
   - **Tesla T4 Tensor Core Precision Adaptation**: Because Tesla T4 GPUs (Turing architecture) lack native `bfloat16` hardware acceleration, cloud dispatchers targeting T4 must automatically adapt `bfloat16` compute to `float16` while preserving `float32` physical simulation, preventing severe software emulation slowdowns.

