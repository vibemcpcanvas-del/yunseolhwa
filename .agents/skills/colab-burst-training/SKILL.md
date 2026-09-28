---
name: colab-burst-training
description: >-
  Runbooks and operational workflows for high-throughput cloud burst RL training
  on Google Colab TPU and GPU using 5-account pool auto-rotation and live tunnel streaming.
---

# Colab Cloud Burst Training Runbook

## Overview
This skill provides automated workflows for offloading JAX/Gymnax reinforcement learning workloads
from slow local CPUs to Google Colab accelerators (TPU v5e1/v6e1/v2-8 or GPU Tesla T4/A100/L4).
It leverages:
1. **5-Account Combo Pool Auto-Rotation**: Automatically handles quota exhaustion (`503 Quota Exhausted`) by rotating accounts via WSL.
2. **Hybrid Precision Scaling**: Keeps physics in `float32` while executing neural networks in `bfloat16` to scale to 16,384 concurrent environments.
3. **Cloudflare Quick Tunnel Live Sync**: Streams intermediate Orbax checkpoints back to the host workstation every chunk (e.g. 50 updates) to protect against spot VM preemption.

---

## Quick Invocations

### 1. Launch Maximum GPU Training (Tesla T4 / 65,536 Envs / 10-min Time Checkpoints)
```bash
python run_colab_train.py --accelerator gpu --gpu_type T4 --mode 1 --num_envs 65536 --num_updates 50000 --checkpoint_interval_seconds 600.0 --chunk_size 200
```

### 2. Launch Standard GPU Training (Tesla T4 / 16,384 Envs)
```bash
python run_colab_train.py --accelerator gpu --gpu_type T4 --mode 1 --num_envs 16384 --num_updates 10000 --checkpoint_interval_seconds 600.0
```

### 3. Check Account Quotas
```bash
python cloud/colab_account_manager.py list
```

### 4. Emergency Teardown / Session Sweep
```bash
python cloud/colab_account_manager.py stop-all jax_train
```

---

## Architecture & Verification Invariants

- **Physics Simulation Precision**: `EnvParams` and `EnvState` coordinate calculations, SAT collision projections, and velocities MUST remain in `float32`.
- **Policy Network Precision**: Dense layers run with `dtype=bfloat16`, `param_dtype=float32`, emitting logits in `float32`.
- **Selective Packaging**: Code packaging strictly bundles `src/`, `train_ppo.py`, and `pyproject.toml` (~0.32 MB) to eliminate upload overhead.
- **Local Persistence**: Downloaded checkpoints are immediately uncompressed to `checkpoints/mode_<M>/step_<N>` on the host machine.
