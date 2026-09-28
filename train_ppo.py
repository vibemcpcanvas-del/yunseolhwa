"""High-Performance PPO Training Script for MapleStory Lotus Phase 1.

Runs 4096 parallel environments on JAX/XLA, compiling the entire training loop
into a single JIT kernel via jax.lax.scan. Supports Classic (mode=0) and
Remastered (mode=1) mechanics, Flax Actor-Critic MLP, GAE/PPO loss,
asynchronous console metric logging via jax.debug.callback, and Orbax checkpointing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import chex
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

# JAX 0.11+ backward compatibility shim for older Flax/tracers
if not hasattr(jax.core, "get_opaque_trace_state"):
    try:
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    except Exception:
        pass

from maple_gymnax.envs.lotus_phase1 import LotusPhase1Env, EnvParams
from maple_gymnax.wrappers.log_wrapper import LogWrapper, LogEnvState
from maple_gymnax.wrappers.flatten_obs import FlattenObservationWrapper
from maple_gymnax.wrappers.rollout_runner import RolloutRunner


# ---------------------------------------------------------------------------
# 1. PPO Configuration
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class PPOConfig:
    """Hyperparameters and configuration for PPO training on Lotus Phase 1."""
    # Environment Settings
    mode: int = 0  # 0: Classic, 1: Remastered, 2: Hybrid
    num_envs: int = 2048
    num_steps: int = 64  # Rollout trajectory length per update
    num_updates: int = 300
    seed: int = 42

    # PPO Algorithmic Hyperparameters
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 2
    anneal_lr: bool = True

    # Logging and Checkpointing
    log_interval: int = 1
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    dtype: str = "float32"
    param_dtype: str = "float32"
    on_checkpoint_cmd: Optional[str] = None


# ---------------------------------------------------------------------------
# 2. Actor-Critic Neural Network Architecture
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Two-headed Actor-Critic MLP with separate representation pathways and hybrid precision support."""
    action_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x = x.astype(self.dtype)
        # Actor Network (Policy Head)
        actor = nn.Dense(
            256,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(
            256,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(actor)
        actor = nn.tanh(actor)
        actor_logits = nn.Dense(
            self.action_dim,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(actor)

        # Critic Network (Value Head)
        critic = nn.Dense(
            256,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(
            256,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nn.initializers.constant(0.0),
        )(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(
            1,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(critic)

        return actor_logits, jnp.squeeze(value, axis=-1)



# ---------------------------------------------------------------------------
# 3. Transitions and State Containers
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict[str, chex.Array]


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: LogEnvState
    last_obs: chex.Array
    rng: chex.PRNGKey


# ---------------------------------------------------------------------------
# 4. GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def _calculate_gae(
    traj_batch: Transition,
    last_val: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[chex.Array, chex.Array]:
    """Pure functional reverse scan calculating Generalized Advantage Estimates."""
    def _scan_fn(carry, transition: Transition):
        gae, next_val = carry
        delta = transition.reward + gamma * next_val * (1.0 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1.0 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _scan_fn,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


# ---------------------------------------------------------------------------
# 5. Host-side Console Logging Callback
# ---------------------------------------------------------------------------

def _log_callback(
    update_step: int,
    mean_return: float,
    mean_length: float,
    survival_rate: float,
    mean_gauge: float,
    actor_loss: float,
    critic_loss: float,
    entropy: float,
    sps: float,
) -> None:
    """Prints training progress on host console without halting device computation."""
    print(
        f"[Update {int(update_step):5d}] "
        f"Return: {float(mean_return):8.2f} | "
        f"Length: {float(mean_length):6.1f} | "
        f"Survival: {float(survival_rate) * 100.0:5.1f}% | "
        f"Gauge: {float(mean_gauge):6.4f} | "
        f"Loss(A/C/Ent): {float(actor_loss):.3f}/{float(critic_loss):.3f}/{float(entropy):.3f} | "
        f"SPS: {float(sps):10,.0f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 6. PPO Training Kernel Builder (Single JIT via jax.lax.scan)
# ---------------------------------------------------------------------------

def make_train_step(
    config: PPOConfig,
) -> Tuple[
    Callable[[chex.PRNGKey], RunnerState],
    Callable[[RunnerState, chex.Array], Tuple[RunnerState, Dict[str, Any]]],
]:
    """Decouples initialization and chunked update scan for outer Python loop checkpointing."""
    env = LotusPhase1Env()
    env_params = EnvParams(mode=config.mode)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    obs_dim = 142 if env_params.is_remastered else 130
    action_dim = 7

    # Learning rate schedule
    if config.anneal_lr:
        total_steps = config.num_updates * config.update_epochs * config.num_minibatches
        lr_schedule = optax.linear_schedule(
            init_value=config.lr,
            end_value=0.0,
            transition_steps=total_steps,
        )
    else:
        lr_schedule = config.lr

    compute_dtype = jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32
    param_dtype = jnp.float32
    network = ActorCritic(action_dim=action_dim, dtype=compute_dtype, param_dtype=param_dtype)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )

    def init_fn(rng: chex.PRNGKey) -> RunnerState:
        rng, rng_model, rng_env = jax.random.split(rng, 3)

        # 1. Initialize Network & Optimizer
        init_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        network_params = network.init(rng_model, init_obs)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 2. Vectorized Environment Reset
        reset_keys = jax.random.split(rng_env, config.num_envs)
        init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

        return RunnerState(
            train_state=train_state,
            env_state=init_env_state,
            last_obs=init_obs,
            rng=rng,
        )

    # 3. Outer Update Step (Single update out of NUM_UPDATES)
    def _update_step(runner_state: RunnerState, update_idx: chex.Array):
        # ---------------------------------------------------------------
        # 3A. Trajectory Rollout (Inner scan of length num_steps)
        # ---------------------------------------------------------------
        def _env_step(carry, _):
            r_state, step_env_state, last_obs, step_rng = carry
            step_rng, rng_action, rng_step = jax.random.split(step_rng, 3)

            # Predict policy logits and state value
            logits, value = network.apply(r_state.params, last_obs)
            action = jax.random.categorical(rng_action, logits)
            log_prob = jax.nn.log_softmax(logits)
            act_log_prob = jnp.take_along_axis(log_prob, action[..., None], axis=-1).squeeze(-1)

            # Vectorized environment transition
            env_step_keys = jax.random.split(rng_step, config.num_envs)
            next_obs, next_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(env_step_keys, step_env_state, action, env_params)

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=act_log_prob,
                obs=last_obs,
                info=info,
            )
            return (r_state, next_env_state, next_obs, step_rng), transition

        (cur_train_state, next_env_state, next_obs, cur_rng), traj_batch = jax.lax.scan(
            _env_step,
            (runner_state.train_state, runner_state.env_state, runner_state.last_obs, runner_state.rng),
            None,
            length=config.num_steps,
        )

        # ---------------------------------------------------------------
        # 3B. Advantage Computation via GAE
        # ---------------------------------------------------------------
        _, last_val = network.apply(cur_train_state.params, next_obs)
        advantages, targets = _calculate_gae(
            traj_batch, last_val, config.gamma, config.gae_lambda
        )

        # ---------------------------------------------------------------
        # 3C. Mini-batch Optimization across Epochs
        # ---------------------------------------------------------------
        def _update_epoch(epoch_carry, _):
            t_state, e_rng = epoch_carry
            e_rng, perm_rng = jax.random.split(e_rng)

            batch_size = config.num_envs * config.num_steps
            minibatch_size = batch_size // config.num_minibatches

            # Flatten trajectories across (num_steps, num_envs)
            flat_obs = traj_batch.obs.reshape((-1, obs_dim))
            flat_action = traj_batch.action.reshape((-1,))
            flat_log_prob = traj_batch.log_prob.reshape((-1,))
            flat_advantages = advantages.reshape((-1,))
            flat_targets = targets.reshape((-1,))

            # Standardize advantages
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            permutation = jax.random.permutation(perm_rng, batch_size)
            batch = (
                flat_obs[permutation],
                flat_action[permutation],
                flat_log_prob[permutation],
                flat_advantages[permutation],
                flat_targets[permutation],
            )

            # Reshape into minibatches
            minibatches = jax.tree.map(
                lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]),
                batch,
            )

            def _update_minibatch(t_state, mb):
                mb_obs, mb_action, mb_log_prob, mb_gae, mb_target = mb

                def _loss_fn(params):
                    logits, value = network.apply(params, mb_obs)
                    log_probs = jax.nn.log_softmax(logits)
                    probs = jax.nn.softmax(logits)

                    # Clipped surrogate objective
                    cur_log_prob = jnp.take_along_axis(log_probs, mb_action[..., None], axis=-1).squeeze(-1)
                    ratio = jnp.exp(cur_log_prob - mb_log_prob)
                    surr1 = ratio * mb_gae
                    surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * mb_gae
                    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                    # Value function loss
                    critic_loss = 0.5 * jnp.mean((value - mb_target) ** 2)

                    # Policy entropy bonus
                    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

                    total_loss = actor_loss + config.vf_coef * critic_loss - config.ent_coef * entropy
                    return total_loss, (actor_loss, critic_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (actor_loss, critic_loss, entropy)), grads = grad_fn(t_state.params)
                t_state = t_state.apply_gradients(grads=grads)
                return t_state, (total_loss, actor_loss, critic_loss, entropy)

            t_state, losses = jax.lax.scan(_update_minibatch, t_state, minibatches)
            return (t_state, e_rng), losses

        (new_train_state, post_rng), epoch_losses = jax.lax.scan(
            _update_epoch,
            (cur_train_state, cur_rng),
            None,
            length=config.update_epochs,
        )

        # Unpack mean losses across epochs and minibatches
        mean_losses = jax.tree.map(jnp.mean, epoch_losses)
        total_loss, actor_loss, critic_loss, entropy = mean_losses

        # ---------------------------------------------------------------
        # 3D. Metric Extraction & Asynchronous Host Callback
        # ---------------------------------------------------------------
        done_mask = traj_batch.info["returned_episode"].astype(jnp.float32)
        num_dones = jnp.sum(done_mask)
        has_dones = num_dones > 0

        # Fallback to mean reward if no episode terminated within the window
        mean_return = jnp.where(
            has_dones,
            jnp.sum(traj_batch.info["returned_episode_returns"] * done_mask) / jnp.maximum(num_dones, 1.0),
            jnp.mean(traj_batch.reward) * config.num_steps,
        )
        mean_length = jnp.where(
            has_dones,
            jnp.sum(traj_batch.info["returned_episode_lengths"] * done_mask) / jnp.maximum(num_dones, 1.0),
            config.num_steps,
        )
        survival_rate = jnp.where(
            has_dones,
            jnp.sum((traj_batch.info["returned_episode_lengths"] >= env_params.max_steps_in_episode).astype(jnp.float32) * done_mask)
            / jnp.maximum(num_dones, 1.0),
            0.0,
        )
        mean_gauge = jnp.mean(traj_batch.info.get("security_gauge", jnp.zeros(1)))
        sps = (config.num_envs * config.num_steps) / (env_params.dt * config.num_steps)

        # Conditional callback triggered every log_interval updates
        should_log = (jnp.mod(update_idx + 1, config.log_interval) == 0) | (update_idx == 0)
        jax.lax.cond(
            should_log,
            lambda: jax.debug.callback(
                _log_callback,
                update_idx + 1,
                mean_return,
                mean_length,
                survival_rate,
                mean_gauge,
                actor_loss,
                critic_loss,
                entropy,
                sps,
            ),
            lambda: None,
        )

        next_runner_state = RunnerState(
            train_state=new_train_state,
            env_state=next_env_state,
            last_obs=next_obs,
            rng=post_rng,
        )

        metrics = {
            "mean_return": mean_return,
            "mean_length": mean_length,
            "survival_rate": survival_rate,
            "mean_gauge": mean_gauge,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }
        return next_runner_state, metrics

    def update_chunk_fn(
        runner_state: RunnerState, update_indices: chex.Array
    ) -> Tuple[RunnerState, Dict[str, Any]]:
        """Executes a chunk of update steps via jax.lax.scan."""
        return jax.lax.scan(_update_step, runner_state, update_indices)

    return init_fn, update_chunk_fn


def make_train(config: PPOConfig) -> Callable[[chex.PRNGKey], Tuple[RunnerState, Dict[str, Any]]]:
    """Constructs monolithic PPO training function (backwards-compatible with tests)."""
    init_fn, update_chunk_fn = make_train_step(config)

    def train(rng: chex.PRNGKey) -> Tuple[RunnerState, Dict[str, Any]]:
        runner_state = init_fn(rng)
        return update_chunk_fn(runner_state, jnp.arange(config.num_updates))

    return train



# ---------------------------------------------------------------------------
# 7. Checkpoint & Evaluation Utilities
# ---------------------------------------------------------------------------

def save_checkpoint_orbax(params: Any, config: PPOConfig, step: int) -> str:
    """Persists model PyTree weights using Orbax StandardCheckpointer."""
    ckpt_dir = os.path.abspath(os.path.join(config.checkpoint_dir, f"mode_{config.mode}"))
    os.makedirs(ckpt_dir, exist_ok=True)
    step_dir = os.path.join(ckpt_dir, f"step_{step}")

    # Remove existing checkpoint if directory exists
    if os.path.exists(step_dir):
        shutil.rmtree(step_dir)
    os.makedirs(step_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(step_dir, params, force=True)
    checkpointer.wait_until_finished()
    checkpointer.close()

    # Save hyperparameter metadata alongside checkpoint
    meta_path = os.path.join(step_dir, "config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses_asdict(config), f, indent=2)

    print(f"[Orbax] Checkpoint successfully saved to: {step_dir}")
    return step_dir


def load_checkpoint_orbax(step_dir: str) -> Any:
    """Restores model PyTree weights from an Orbax checkpoint."""
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(step_dir)
    print(f"[Orbax] Restored weights from: {step_dir}")
    return restored


def dataclasses_asdict(obj: Any) -> Dict[str, Any]:
    """Helper to serialize dataclass to dict."""
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def evaluate_policy(
    params: Any,
    config: PPOConfig,
    num_eval_steps: int = 1200,
    rng: Optional[chex.PRNGKey] = None,
) -> Dict[str, float]:
    """Evaluates the trained actor-critic policy via RolloutRunner."""
    if rng is None:
        rng = jax.random.PRNGKey(config.seed + 1000)

    env = LotusPhase1Env()
    env_params = EnvParams(mode=config.mode)
    wrapped = FlattenObservationWrapper(env)

    obs_dim = 142 if env_params.is_remastered else 130
    network = ActorCritic(action_dim=7)

    def policy_fn(obs: chex.Array, state: Any, key: chex.PRNGKey) -> chex.Array:
        # Deterministic / greedy action selection
        logits, _ = network.apply(params, obs)
        return jnp.argmax(logits, axis=-1)

    runner = RolloutRunner(wrapped, env_params, policy_fn=policy_fn)

    k_reset, k_run = jax.random.split(rng)
    init_obs, init_state = wrapped.reset(k_reset, env_params)

    # Run deterministic rollout via JIT-compiled RolloutRunner.run
    final_state, traj = jax.jit(runner.run, static_argnums=(2,))(k_run, init_state, num_eval_steps)

    total_reward = float(jnp.sum(traj["reward"]))
    survival_steps = int(final_state.time)
    survived = bool(final_state.player_hp > 0.0)

    print(
        f"[Evaluation via RolloutRunner] Mode: {config.mode} | "
        f"Total Reward: {total_reward:.2f} | "
        f"Steps: {survival_steps}/{num_eval_steps} | "
        f"Survived: {survived} | Final HP: {float(final_state.player_hp):.1f}"
    )

    return {
        "reward": total_reward,
        "steps": survival_steps,
        "survived": survived,
        "final_hp": float(final_state.player_hp),
    }


# ---------------------------------------------------------------------------
# 8. Command-Line Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> PPOConfig:
    """Parses command line arguments into PPOConfig."""
    parser = argparse.ArgumentParser(
        description="High-Speed PPO Trainer for MapleStory Lotus Phase 1 (Classic & Remastered)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment & Simulation Modes
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2],
                        help="Environment Mode: 0=Classic, 1=Remastered, 2=Hybrid")
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="Number of vectorized parallel environments")
    parser.add_argument("--num_steps", type=int, default=64,
                        help="Rollout steps collected per env per update")
    parser.add_argument("--num_updates", type=int, default=300,
                        help="Total PPO update iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master PRNG seed")

    # PPO Algorithmic Parameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE discount lambda")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm clipping")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Number of minibatches per epoch")
    parser.add_argument("--update_epochs", type=int, default=2, help="PPO update epochs per batch")
    parser.add_argument("--no_anneal_lr", action="store_true", help="Disable linear learning rate decay")

    # Logging, Checkpointing, and Evaluation
    parser.add_argument("--log_interval", type=int, default=1, help="Update interval for console logging")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Checkpoint frequency (chunk size for outer python loop)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Orbax save directory")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                        help="Neural network compute precision (float32 or bfloat16)")
    parser.add_argument("--on_checkpoint_cmd", type=str, default=None,
                        help="Command/script to execute on checkpoint commit (supports {step} and {step_dir})")
    parser.add_argument("--eval_steps", type=int, default=1200, help="Evaluation rollout steps")

    args = parser.parse_args()

    return PPOConfig(
        mode=args.mode,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        anneal_lr=not args.no_anneal_lr,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        dtype=args.dtype,
        on_checkpoint_cmd=args.on_checkpoint_cmd,
    )


def main():
    config = parse_args()
    mode_name = "Classic" if config.mode == 0 else ("Remastered" if config.mode == 1 else "Hybrid")

    print("=" * 76)
    print(f"MapleStory Lotus Phase 1 PPO Training ({mode_name} Mode, mode={config.mode})")
    print(f"Hardware Backend: {jax.default_backend()} | Devices: {jax.devices()}")
    print(f"Parallel Envs: {config.num_envs:,} | Rollout Steps: {config.num_steps} | Updates: {config.num_updates}")
    print(f"Precision: compute={config.dtype}, param={config.param_dtype} | Epochs/Batch: {config.update_epochs}")
    print(f"Chunk Size (Checkpoint Interval): {config.checkpoint_interval}")
    print(f"Total Step Budget: {config.num_envs * config.num_steps * config.num_updates:,} environment steps")
    print("=" * 76)

    # Initialize decoupled step kernels
    init_fn, update_chunk_fn = make_train_step(config)
    jitted_init = jax.jit(init_fn)
    jitted_chunk = jax.jit(update_chunk_fn)

    master_rng = jax.random.PRNGKey(config.seed)
    rng_init, rng_eval = jax.random.split(master_rng)

    print("\n[XLA] Initializing runner state and JIT-compiling chunk kernel...")
    t0 = time.perf_counter()
    runner_state = jitted_init(rng_init)
    jax.block_until_ready(runner_state.train_state.params)

    chunk_size = config.checkpoint_interval
    num_updates = config.num_updates

    # Outer Python Loop: executes in chunk_size increments and automatically saves Orbax checkpoints
    current_step = 0
    while current_step < num_updates:
        this_chunk = min(chunk_size, num_updates - current_step)
        chunk_indices = jnp.arange(current_step, current_step + this_chunk)

        runner_state, chunk_metrics = jitted_chunk(runner_state, chunk_indices)
        jax.block_until_ready(runner_state.train_state.params)
        current_step += this_chunk

        # Automatic checkpoint saving at every chunk completion
        step_dir = save_checkpoint_orbax(runner_state.train_state.params, config, current_step)
        print(f"[Loop Progress] Completed {current_step}/{num_updates} updates (Checkpoint saved)", flush=True)

        if config.on_checkpoint_cmd:
            try:
                cmd_to_run = config.on_checkpoint_cmd.format(step=current_step, step_dir=step_dir)
                subprocess.run(cmd_to_run, shell=True, check=False)
            except Exception as cb_err:
                print(f"[!] on_checkpoint_cmd warning: {cb_err}", flush=True)

    total_time = time.perf_counter() - t0
    total_sim_steps = config.num_envs * config.num_steps * config.num_updates
    overall_sps = total_sim_steps / max(total_time, 1e-6)

    print("\n" + "=" * 76)
    print(f"[Training Complete] Total Time: {total_time:.2f}s | Throughput: {overall_sps:10,.0f} SPS")
    print("=" * 76)

    # Run post-training evaluation via RolloutRunner
    print("\n[Evaluation] Running post-training evaluation via RolloutRunner...")
    evaluate_policy(runner_state.train_state.params, config, num_eval_steps=1200, rng=rng_eval)


if __name__ == "__main__":
    main()

