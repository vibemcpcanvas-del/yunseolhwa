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

5. **Pure Inlined Modular Helpers**:
   - Keep `step_env` concise by extracting discrete physical steps into pure, undecorated Python functions.
   - Pure functions called inside `step_env` are seamlessly inlined by XLA without runtime tracing overhead.
