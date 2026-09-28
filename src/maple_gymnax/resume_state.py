"""Resumable Checkpoint State Serialization and Config Fingerprint Verification.

Provides deep PyTree checkpointing of the entire JAX PPO RunnerState
(train_state, opt_state, env_state, last_obs, PRNG key) alongside
strict configuration fingerprinting across Google Colab VM preemption and restarts.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# Configuration Fingerprint Keys
HARD_KEYS = (
    "mode", "num_envs", "num_steps", "gamma", "gae_lambda",
    "clip_eps", "ent_coef", "vf_coef", "max_grad_norm",
    "num_minibatches", "update_epochs",
)

SOFT_KEYS = ("dtype", "param_dtype")

EXTENSIBLE_KEYS = ("num_updates",)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Serializes PPOConfig to clean dictionary."""
    if hasattr(config, "__dataclass_fields__"):
        return {k: getattr(config, k) for k in config.__dataclass_fields__}
    if hasattr(config, "_asdict"):
        return config._asdict()
    return dict(config)


def save_resume_state(
    step_dir: str,
    runner_state: Any,
    update: int,
    config: Any,
) -> str:
    """Persists entire RunnerState and manifest.json into step_dir/_resume_state.

    Guarantees completion via checkpointer.wait_until_finished().
    """
    step_dir = os.path.abspath(step_dir)
    resume_dir = os.path.abspath(os.path.join(step_dir, "_resume_state"))
    if os.path.exists(resume_dir):
        shutil.rmtree(resume_dir)
    os.makedirs(resume_dir, exist_ok=True)

    # 1. Save Full Runner State PyTree
    checkpointer = ocp.StandardCheckpointer()
    payload = {
        "runner": runner_state,
        "update": int(update),
    }
    checkpointer.save(resume_dir, payload, force=True)
    checkpointer.wait_until_finished()
    checkpointer.close()

    # 2. Write Manifest JSON
    cfg_dict = _config_to_dict(config)
    manifest = {
        "schema_version": 1,
        "mode": int(getattr(config, "mode", 0)),
        "update": int(update),
        "hard_config": {k: cfg_dict.get(k) for k in HARD_KEYS if k in cfg_dict},
        "soft_config": {k: str(cfg_dict.get(k)) for k in SOFT_KEYS if k in cfg_dict},
        "extensible_config": {k: cfg_dict.get(k) for k in EXTENSIBLE_KEYS if k in cfg_dict},
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(resume_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ResumeState] Full RunnerState saved: {resume_dir} (Update {update})", flush=True)
    return resume_dir


def load_resume_state(
    resume_path: str,
    current_config: Any,
    target_runner_state: Optional[Any] = None,
    allow_precision_loss: bool = False,
) -> Tuple[Any, int, Any]:
    """Restores full RunnerState and verifies configuration fingerprint.

    Args:
        resume_path: Path to step_dir or step_dir/_resume_state.
        current_config: Target PPOConfig.
        target_runner_state: Optional template RunnerState PyTree structure for Orbax.
        allow_precision_loss: Whether to permit downcasting (float32 -> float16/bfloat16).

    Returns:
        (restored_runner_state, restored_update, updated_config)
    """
    resume_path = os.path.abspath(resume_path)
    if os.path.basename(resume_path) != "_resume_state":
        inner = os.path.join(resume_path, "_resume_state")
        if os.path.exists(inner):
            resume_path = inner

    manifest_path = os.path.join(resume_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest.json in resume directory: {resume_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cur_dict = _config_to_dict(current_config)

    # 1. HARD_KEYS Fingerprint Check
    hard_saved = manifest.get("hard_config", {})
    mismatches = []
    for k in HARD_KEYS:
        if k in hard_saved and k in cur_dict:
            val_saved = hard_saved[k]
            val_cur = cur_dict[k]
            if val_saved != val_cur:
                mismatches.append(f"{k}: saved={val_saved} vs current={val_cur}")

    if mismatches:
        raise ValueError(
            f"[ResumeState Error] Cannot resume due to HARD_KEYS mismatch:\n  "
            + "\n  ".join(mismatches)
        )

    # 2. SOFT_KEYS Migration Check
    soft_saved = manifest.get("soft_config", {})
    old_dtype = str(soft_saved.get("dtype", cur_dict.get("dtype")))
    new_dtype = str(cur_dict.get("dtype"))
    if old_dtype != new_dtype:
        is_downcast = (old_dtype == "float32" and new_dtype in ("float16", "bfloat16"))
        if is_downcast and not allow_precision_loss:
            raise ValueError(
                f"[ResumeState Error] Precision downcast from {old_dtype} to {new_dtype} "
                f"requires --allow-precision-loss flag."
            )
        print(f"[MIGRATE] dtype {old_dtype} -> {new_dtype} (Migration allowed)", flush=True)

    # 3. EXTENSIBLE_KEYS Extension Check
    ext_saved = manifest.get("extensible_config", {})
    saved_num_updates = ext_saved.get("num_updates", manifest.get("update", 0))
    current_num_updates = getattr(current_config, "num_updates", saved_num_updates)
    if current_num_updates > saved_num_updates:
        print(
            f"[EXTEND] target updates {saved_num_updates} -> {current_num_updates} "
            f"(Extending training budget)", flush=True
        )

    # If target_runner_state is not passed, attempt to construct template
    if target_runner_state is None:
        try:
            from train_ppo import make_train_step
            init_fn, _ = make_train_step(current_config)
            target_runner_state = init_fn(jax.random.PRNGKey(getattr(current_config, "seed", 0)))
        except Exception:
            pass

    # 4. Restore PyTree using Orbax
    checkpointer = ocp.StandardCheckpointer()
    if target_runner_state is not None:
        target = {"runner": target_runner_state, "update": 0}
        restored = checkpointer.restore(resume_path, target=target)
    else:
        restored = checkpointer.restore(resume_path)
    checkpointer.close()

    runner_state = restored["runner"]
    restored_update = int(restored.get("update", manifest.get("update", 0)))

    # Handle dtype casting if migrated
    if old_dtype != new_dtype:
        target_jnp_dtype = jnp.float16 if new_dtype == "float16" else (
            jnp.bfloat16 if new_dtype == "bfloat16" else jnp.float32
        )
        # Cast train_state params PyTree leaves
        def _cast_leaf(x):
            if isinstance(x, (jnp.ndarray, chex.Array)) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(target_jnp_dtype)
            return x

        new_params = jax.tree.map(_cast_leaf, runner_state.train_state.params)
        new_train_state = runner_state.train_state.replace(params=new_params)
        runner_state = runner_state._replace(train_state=new_train_state)

    print(
        f"[ResumeState] Successfully restored RunnerState from {resume_path} "
        f"(Starting at update {restored_update})", flush=True
    )
    return runner_state, restored_update, current_config
