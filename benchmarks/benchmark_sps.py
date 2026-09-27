"""Hardware-calibrated throughput (SPS) benchmark for Maple Gymnax Lotus Phase 1."""

import argparse
import sys
import time
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.wrappers.flatten_obs import FlattenObservationWrapper


def run_benchmark(
    batch_sizes: List[int] = (256, 512, 1024, 2048, 4096),
    num_steps: int = 100,
    warmup_steps: int = 10,
    mode: int = 1,
) -> Dict[int, Dict[str, float]]:
    """Measures simulation steps per second (SPS) across batched environments under XLA compilation."""
    env = LotusPhase1Env()
    params = EnvParams(mode=mode)

    step_fn = env.step_env

    @jax.jit
    def _step_batch(keys, states, actions):
        return jax.vmap(step_fn, in_axes=(0, 0, 0, None))(keys, states, actions, params)

    results: Dict[int, Dict[str, float]] = {}

    print(f"=== Running Maple Gymnax SPS Benchmark (Mode={mode}, Steps={num_steps}) ===")
    print(f"JAX Default Backend: {jax.default_backend()} | Devices: {jax.devices()}")

    for b in batch_sizes:
        master_key = jax.random.PRNGKey(42)
        reset_keys = jax.random.split(master_key, b)
        _, init_states = jax.vmap(env.reset_env, in_axes=(0, None))(reset_keys, params)

        # Warmup JIT compilation
        warmup_keys = jax.random.split(master_key, b)
        actions = jnp.zeros(b, dtype=jnp.int32)
        _, states, _, _, _ = _step_batch(warmup_keys, init_states, actions)
        jax.block_until_ready(states.player_x)

        for _ in range(warmup_steps):
            k = jax.random.split(master_key, b)
            _, states, _, _, _ = _step_batch(k, states, actions)
        jax.block_until_ready(states.player_x)

        # Timed Execution
        t0 = time.perf_counter()
        curr_states = states
        for _ in range(num_steps):
            k = jax.random.split(master_key, b)
            _, curr_states, _, _, _ = _step_batch(k, curr_states, actions)
        jax.block_until_ready(curr_states.player_x)
        elapsed = time.perf_counter() - t0

        total_sim_steps = b * num_steps
        sps = total_sim_steps / max(elapsed, 1e-6)
        step_latency_us = (elapsed / total_sim_steps) * 1e6

        results[b] = {
            "sps": sps,
            "elapsed_sec": elapsed,
            "latency_us": step_latency_us,
        }
        print(f"Batch {b:5d}: {sps:10,.1f} SPS | Elapsed: {elapsed:6.3f}s | Latency: {step_latency_us:6.2f} us/step")

    return results


def print_markdown_table(results: Dict[int, Dict[str, float]]):
    """Outputs results in standard GitHub Markdown format."""
    print("\n### Benchmark Summary Table")
    print("| Batch Size | SPS (Steps/sec) | Total Time (s) | Per-Step Latency (us) |")
    print("| :--- | :--- | :--- | :--- |")
    for b, metrics in results.items():
        print(f"| {b:,} | {metrics['sps']:,.1f} | {metrics['elapsed_sec']:.3f} | {metrics['latency_us']:.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple Gymnax Lotus Phase 1 Throughput Benchmark")
    parser.add_argument("--steps", type=int, default=100, help="Steps to roll out per batch")
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1, 2], help="Environment mode (0=Classic, 1=Remaster, 2=Hybrid)")
    args = parser.parse_args()

    res = run_benchmark(num_steps=args.steps, mode=args.mode)
    print_markdown_table(res)
