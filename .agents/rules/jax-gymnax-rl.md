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
