"""Master Cloud Orchestrator for Maple-Gymnax Colab Training (TPU & GPU Support).

Consolidates:
  1. Packaging JAX project source code and training configs
  2. Starting Cloudflare Quick Tunnel data exchange bridge
  3. 5-Account Colab Combo Pool Auto-Rotation on Quota Exhaustion
  4. Executing remote TPU (v5e/v2-8) or GPU (T4/A100) PPO training session
  5. Live streaming of intermediate Orbax checkpoints at every chunk commit
  6. Final packaging and safe teardown
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cloud import colab_account_manager as cam
from cloud.colab_tunnel_bridge import JaxTunnelTransferManager

COLAB_BAT = os.path.join(PROJECT_ROOT, "cloud", "colab.bat")
TRAIN_JOB_SCRIPT = os.path.join(PROJECT_ROOT, "cloud", "colab_jax_train_job.py")
UPLOAD_TAR_LOCAL = os.path.join(PROJECT_ROOT, "cloud", "upload_jax.tar")
DOWNLOAD_TAR_LOCAL = os.path.join(PROJECT_ROOT, "cloud", "jax_checkpoints_downloaded.tar")
SESSION_NAME = "jax_train"


def to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    if drive:
        return f"/mnt/{drive[0].lower()}" + rest.replace('\\', '/')
    return win_path.replace('\\', '/')


def resolve_gpu_dtype(gpu_type: str, requested_dtype: str) -> str:
    """Resolves GPU compute dtype, automatically falling back to float16 on Tesla T4."""
    if gpu_type.upper() == "T4" and requested_dtype.lower() == "bfloat16":
        print("[WARN] Tesla T4 lacks native bfloat16 hardware; falling back to float16 for peak Tensor Core throughput.", flush=True)
        return "float16"
    return requested_dtype


def package_jax_codebase(output_tar: str, resume_step_dir: Optional[str] = None) -> int:
    """Packages src/, train_ppo.py, pyproject.toml and optional resume checkpoint into an uploadable tarball."""
    print(f"[*] Packaging project codebase into {output_tar}...", flush=True)
    os.makedirs(os.path.dirname(output_tar), exist_ok=True)
    with tarfile.open(output_tar, "w") as tar:
        # Add src directory
        src_dir = os.path.join(PROJECT_ROOT, "src")
        if os.path.exists(src_dir):
            tar.add(src_dir, arcname="src")

        # Add train_ppo.py
        train_file = os.path.join(PROJECT_ROOT, "train_ppo.py")
        if os.path.exists(train_file):
            tar.add(train_file, arcname="train_ppo.py")

        # Add eval_checkpoint.py
        eval_file = os.path.join(PROJECT_ROOT, "eval_checkpoint.py")
        if os.path.exists(eval_file):
            tar.add(eval_file, arcname="eval_checkpoint.py")

        # Add pyproject.toml
        pyproject_file = os.path.join(PROJECT_ROOT, "pyproject.toml")
        if os.path.exists(pyproject_file):
            tar.add(pyproject_file, arcname="pyproject.toml")

        # Include verified resume checkpoint if supplied
        if resume_step_dir and os.path.exists(resume_step_dir):
            print(f"[*] Bundling verified resume checkpoint from {resume_step_dir} into archive...", flush=True)
            tar.add(resume_step_dir, arcname="resume_checkpoint")

    size = os.path.getsize(output_tar)
    print(f"[SUCCESS] Packaged codebase ({size / (1024*1024):.2f} MB)", flush=True)
    return size


class CloudJaxManager:
    """High-level runner managing Colab accounts, tunnel, and JAX training jobs."""

    def __init__(self):
        self.tunnel_manager: Optional[JaxTunnelTransferManager] = None
        self.last_verified_step: int = 0
        self._is_terminating: bool = False
        atexit.register(self.stop)
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            pass

    def get_verified_resume_dir(self, mode: int) -> Optional[str]:
        """Locates the latest verified checkpoint directory on the host workstation."""
        if self.last_verified_step <= 0:
            return None
        candidate = os.path.join(PROJECT_ROOT, "checkpoints", f"mode_{mode}", f"step_{self.last_verified_step}")
        manifest = os.path.join(candidate, "_resume_state", "manifest.json")
        if os.path.exists(manifest):
            return candidate
        return None

    def _handle_signal(self, sig, frame):
        print(f"\n[Safe Teardown] Signal {sig} received. Terminating cloud manager...", flush=True)
        self.stop()
        sys.exit(0)

    def stop_session(self, session_name: str = SESSION_NAME):
        """Terminates remote Colab session using colab.bat stop."""
        if not os.path.exists(COLAB_BAT):
            return
        print(f"[*] Stopping remote Colab session '{session_name}'...", flush=True)
        try:
            res = subprocess.run(
                [COLAB_BAT, "stop", "-s", session_name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', timeout=20
            )
            print(f"    -> {res.stdout.strip() if res.stdout else 'OK'}", flush=True)
        except Exception as e:
            print(f"[!] colab stop warning: {e}", flush=True)

    def stop(self):
        if self._is_terminating:
            return
        self._is_terminating = True
        print("[*] Initiating CloudJaxManager shutdown...", flush=True)
        if self.tunnel_manager:
            try:
                self.tunnel_manager.stop()
            except Exception:
                pass
            self.tunnel_manager = None
        self.stop_session(SESSION_NAME)
        self._is_terminating = False

    def run_training(
        self,
        mode: int = 1,
        num_envs: Optional[int] = None,
        num_steps: int = 64,
        num_updates: int = 3000,
        checkpoint_interval: int = 100,
        seed: int = 42,
        accelerator: str = "tpu",
        tpu_type: str = "v5e1",
        gpu_type: str = "T4",
        dtype: str = "bfloat16"
    ) -> bool:
        """Executes full training pipeline on Colab with auto-rotation."""
        # Auto-configure num_envs if omitted
        if num_envs is None:
            num_envs = 16384 if accelerator == "tpu" else 4096

        if accelerator == "gpu":
            dtype = resolve_gpu_dtype(gpu_type, dtype)

        print("=" * 68, flush=True)
        print(" 🎮 Maple-Gymnax Cloud Training Orchestrator (1M+ SPS TPU/GPU Target)", flush=True)
        print("=" * 68, flush=True)

        # Locate verified resume checkpoint if resuming after rotation or available
        resume_step_dir = self.get_verified_resume_dir(mode)
        if resume_step_dir:
            print(f"[RESUME] Found verified checkpoint step_{self.last_verified_step} for VM resumption.", flush=True)

        # 1. Package codebase
        package_jax_codebase(UPLOAD_TAR_LOCAL, resume_step_dir=resume_step_dir)

        # 2. Check accounts in pool
        reg = cam.load_registry()
        active_acc = cam.get_active_account(reg) or {}
        acc_id = active_acc.get("id", "account_1")
        acc_name = active_acc.get("name", "Unknown")

        print(f"[*] Active Colab Account: {acc_name} ({acc_id})", flush=True)
        print(f"[*] Target Accelerator: {accelerator.upper()} (Envs: {num_envs:,} | Precision: {dtype})", flush=True)

        # 3. Start tunnel
        self.tunnel_manager = JaxTunnelTransferManager(
            data_tar_path=UPLOAD_TAR_LOCAL,
            script_path=TRAIN_JOB_SCRIPT,
            output_checkpoints_tar=DOWNLOAD_TAR_LOCAL,
            extract_target_dir=os.path.join(PROJECT_ROOT, "checkpoints")
        )
        if self.last_verified_step > 0:
            self.tunnel_manager.verified_steps.add(self.last_verified_step)
        tunnel_url = self.tunnel_manager.start()

        # 4. Prepare accelerator CLI arguments
        if accelerator == "tpu":
            accel_args = ["--tpu", tpu_type]
        else:
            accel_args = ["--gpu", gpu_type]

        extra_remote_args = ["--log_interval", "20"]
        if accelerator == "gpu":
            extra_remote_args.append("--require_gpu")

        wsl_script = to_wsl_path(TRAIN_JOB_SCRIPT)
        cmd = [
            COLAB_BAT, "run"
        ] + accel_args + [
            "--session", SESSION_NAME,
            "--timeout", "36000",
            wsl_script,
            "--tunnel", tunnel_url,
            "--mode", str(mode),
            "--num_envs", str(num_envs),
            "--num_steps", str(num_steps),
            "--num_updates", str(num_updates),
            "--checkpoint_interval", str(checkpoint_interval),
            "--seed", str(seed),
            "--dtype", str(dtype),
        ] + extra_remote_args

        print(f"[*] Dispatching Colab {accelerator.upper()} job on account '{acc_name}'...", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1)

        quota_exhausted = False
        try:
            for line in iter(proc.stdout.readline, ''):
                clean = line.strip()
                if not clean:
                    continue
                print(f"[colab-jax] {clean}", flush=True)

                # Avoid false positives from training metric values like -503.79
                is_training_metric = "[Update" in clean
                if not is_training_metric and any(err in clean for err in [
                    "Quota exceeded", "Rate limit", "ResourceExhausted",
                    "Subscription required", "pro subscription",
                    "cannot assign", "unsupported accelerator",
                    "503 Server Error", "503 Service Unavailable", "status code 503"
                ]):
                    quota_exhausted = True
                    print(f"\n[QUOTA] Quota/tier limitation on account '{acc_name}'. Triggering account rotation...", flush=True)
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    break

            if not quota_exhausted:
                proc.wait()

            if quota_exhausted:
                if self.tunnel_manager:
                    last_step = self.tunnel_manager.get_last_verified_step()
                    if last_step > 0:
                        self.last_verified_step = max(self.last_verified_step, last_step)
                        print(f"[RESUME] Preserving verified step {self.last_verified_step} for rotation.", flush=True)
                self.stop()
                next_acc = cam.switch_next_account()
                if next_acc:
                    print(f"[COMBO] Switched to next account '{next_acc.get('name')}'. Retrying on {next_acc.get('name')}...", flush=True)
                    return self.run_training(mode, num_envs, num_steps, num_updates, checkpoint_interval, seed, accelerator, tpu_type, gpu_type, dtype)
                else:
                    if accelerator == "tpu":
                        print("[FALLBACK] TPU unavailable across account pool. Falling back to GPU (T4)...", flush=True)
                        cam.reset_all_quotas()
                        gpu_dtype = resolve_gpu_dtype("T4", dtype)
                        return self.run_training(mode, 4096, num_steps, num_updates, checkpoint_interval, seed, "gpu", tpu_type, "T4", gpu_dtype)
                    print("[ERROR] All Colab accounts in combo pool are exhausted.", flush=True)
                    return False

            if proc.returncode == 0:
                print("\n[SUCCESS] Colab training run completed successfully!", flush=True)
                return True
            else:
                print(f"\n[!] Colab job failed with exit code {proc.returncode}", flush=True)
                return False

        except Exception as e:
            print(f"[ERROR] Session exception: {e}", flush=True)
            return False
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Maple-Gymnax Cloud Training Runner")
    parser.add_argument("--accelerator", type=str, default="tpu", choices=["tpu", "gpu"], help="Accelerator target (tpu or gpu)")
    parser.add_argument("--tpu_type", type=str, default="v5e1", help="TPU type (v5e1, v6e1)")
    parser.add_argument("--gpu_type", type=str, default="T4", help="GPU accelerator (T4, A100, L4)")
    parser.add_argument("--mode", type=int, default=1, help="0: Classic, 1: Remastered, 2: Hybrid")
    parser.add_argument("--num_envs", type=int, default=None, help="Parallel environments (default: 16384 on TPU, 4096 on GPU)")
    parser.add_argument("--num_steps", type=int, default=64, help="Rollout steps")
    parser.add_argument("--num_updates", type=int, default=3000, help="Updates count")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Compute precision")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    args = parser.parse_args()

    mgr = CloudJaxManager()
    ok = mgr.run_training(
        mode=args.mode,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        accelerator=args.accelerator,
        tpu_type=args.tpu_type,
        gpu_type=args.gpu_type,
        dtype=args.dtype
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
