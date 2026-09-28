"""Colab Remote High-Throughput GPU/TPU Training Job for Maple-Gymnax PPO.

Runs inside Google Colab (Linux + CUDA Tesla T4 / A100 / L4 or TPU v5e / v2-8).
Downloads project package via Cloudflare tunnel, installs dependencies, executes
train_ppo.py with massive parallelism (num_envs=4096~16384) and hybrid precision (bfloat16/float32),
streams intermediate Orbax checkpoints back to the host at every chunk, packages final artifacts,
and auto-terminates the VM to protect compute quota.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# 1. Colab Auto-Termination Watchdog (Protects GPU/TPU Quota)
# ---------------------------------------------------------------------------
def schedule_auto_unassign(delay_seconds: int = 14400, reason: str = "Timeout"):
    """Self-terminates the Colab VM after delay_seconds to protect compute quota."""
    def _unassign():
        print(f"\n[WATCHDOG] Auto-termination triggered ({reason}, after {delay_seconds}s). Releasing Colab VM...", flush=True)
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            try:
                os.system("kill -9 -1")
            except Exception:
                pass

    t = threading.Timer(delay_seconds, _unassign)
    t.daemon = True
    t.start()
    return t


schedule_auto_unassign(14400, "Safety 4h execution limit")


# ---------------------------------------------------------------------------
# 2. CLI Arguments & Paths
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Colab JAX Train Job")
parser.add_argument("--tunnel", type=str, default=None, help="Cloudflare tunnel URL for data exchange")
parser.add_argument("--mode", type=int, default=1, help="Gimmick mode: 0 (Classic), 1 (Remastered), 2 (Hybrid)")
parser.add_argument("--num_envs", type=int, default=16384, help="Number of parallel environments (16,384 on TPU, 4,096 on GPU)")
parser.add_argument("--num_steps", type=int, default=64, help="Rollout steps per update")
parser.add_argument("--num_updates", type=int, default=300, help="Total PPO updates")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Checkpoint interval in updates")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Compute precision")
parser.add_argument("--log_interval", type=int, default=20, help="Console logging interval")
parser.add_argument("--require_gpu", action="store_true", help="Enforce GPU backend requirement")
parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
args, _ = parser.parse_known_args()

TUNNEL_URL = args.tunnel or os.environ.get("TUNNEL_URL", None)
WORKSPACE_DIR = "/content/workspace"
UPLOAD_TAR_PATH = "/content/upload_jax.tar"
OUTPUT_TAR_PATH = "/content/jax_checkpoints.tar"


# ---------------------------------------------------------------------------
# 3. Hardware Accelerator & JAX Environment Verification
# ---------------------------------------------------------------------------
print("=" * 68, flush=True)
print(" 🚀 Colab JAX / Gymnax PPO Accelerator Job (TPU / GPU High-Throughput)", flush=True)
print("=" * 68, flush=True)

# Hardware Platform Detection & Accelerator Setup
is_tpu = os.path.exists("/dev/accel0") or bool(os.environ.get("TPU_NAME")) or bool(os.environ.get("COLAB_TPU_ADDR"))

if is_tpu:
    print("[*] JAX Acceleration: Cloud TPU detected (/dev/accel0)", flush=True)
    try:
        print("[*] Installing/upgrading matching jax[tpu] runtime for Cloud TPU...", flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]",
            "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        ], check=False)
    except Exception:
        pass
else:
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        print(f"[*] JAX Acceleration: GPU ({smi.strip()})", flush=True)
        print("[*] Ensuring JAX with CUDA 12 support is up-to-date...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"], check=False)
    except Exception:
        print("[*] JAX Acceleration: CPU runtime", flush=True)

# Install / upgrade required core packages so flax/optax/chex match modern JAX
core_pkgs = ["flax", "optax", "orbax-checkpoint", "chex", "flashbax"]
print(f"[*] Ensuring core JAX RL packages are up-to-date: {core_pkgs}...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + core_pkgs, check=True)

# Install gymnax with --no-deps to prevent legacy jax<0.7 from downgrading jax/jaxlib
try:
    __import__("gymnax")
except ImportError:
    print("[*] Installing gymnax (--no-deps) on Colab VM...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "gymnax"], check=True)

print(f"[*] Target Precision: {args.dtype} (Hybrid)", flush=True)


# ---------------------------------------------------------------------------
# 4. Download and Extract Repository Codebase
# ---------------------------------------------------------------------------
os.makedirs(WORKSPACE_DIR, exist_ok=True)
if TUNNEL_URL:
    print(f"[*] Downloading codebase from {TUNNEL_URL}/data ...", flush=True)
    ret = os.system(f"curl -L -f -s --connect-timeout 30 --retry 3 {TUNNEL_URL}/data -o {UPLOAD_TAR_PATH}")
    if ret != 0 or not os.path.exists(UPLOAD_TAR_PATH) or os.path.getsize(UPLOAD_TAR_PATH) == 0:
        try:
            with urllib.request.urlopen(f"{TUNNEL_URL}/data", timeout=60) as resp, open(UPLOAD_TAR_PATH, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=8 * 1024 * 1024)
        except Exception as e:
            print(f"[!] urllib download failed: {e}", flush=True)

    if os.path.exists(UPLOAD_TAR_PATH):
        print(f"[SUCCESS] Downloaded codebase: {os.path.getsize(UPLOAD_TAR_PATH)/(1024*1024):.2f} MB", flush=True)
        with tarfile.open(UPLOAD_TAR_PATH, "r") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=WORKSPACE_DIR, filter="data")
            else:
                tar.extractall(path=WORKSPACE_DIR)
        print(f"[SUCCESS] Codebase extracted into {WORKSPACE_DIR}", flush=True)
    else:
        print("[!] Failed to obtain codebase archive. Aborting.", flush=True)
        sys.exit(1)
else:
    print("[!] No tunnel URL provided. Working with existing files in current directory.", flush=True)


# ---------------------------------------------------------------------------
# 5. Live Intermediate Checkpoint Push Helper
# ---------------------------------------------------------------------------
push_helper_script = "/content/push_checkpoint.py"
push_helper_code = '''import hashlib, os, sys, tarfile, urllib.request

step = sys.argv[1]
step_dir = sys.argv[2]
tunnel_url = sys.argv[3]

if not os.path.exists(step_dir):
    sys.exit(0)

tar_path = f"/tmp/{step}.tar"
try:
    with tarfile.open(tar_path, "w") as tar:
        tar.add(step_dir, arcname=os.path.basename(step_dir))

    with open(tar_path, "rb") as f:
        data = f.read()

    sha256_hash = hashlib.sha256(data).hexdigest()

    req = urllib.request.Request(
        f"{tunnel_url}/upload_checkpoint?step={step}&sha256={sha256_hash}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status == 200:
            print(f"[LiveSync] Intermediate checkpoint '{step}' synced to host workstation (SHA256 verified).", flush=True)
except Exception as e:
    print(f"[LiveSync Warning] Checkpoint streaming error: {e}", flush=True)
    sys.exit(1)
finally:
    if os.path.exists(tar_path):
        try: os.remove(tar_path)
        except Exception: pass
'''
with open(push_helper_script, "w", encoding="utf-8") as f:
    f.write(push_helper_code)


# ---------------------------------------------------------------------------
# 6. Execute train_ppo.py
# ---------------------------------------------------------------------------
os.chdir(WORKSPACE_DIR)
src_dir = os.path.join(WORKSPACE_DIR, "src")
sys.path.insert(0, src_dir)

sub_env = os.environ.copy()
sub_env["PYTHONPATH"] = f"{src_dir}:{sub_env.get('PYTHONPATH', '')}"

train_cmd = [
    sys.executable, "train_ppo.py",
    "--mode", str(args.mode),
    "--num_envs", str(args.num_envs),
    "--num_steps", str(args.num_steps),
    "--num_updates", str(args.num_updates),
    "--checkpoint_interval", str(args.checkpoint_interval),
    "--log_interval", str(args.log_interval),
    "--seed", str(args.seed),
    "--dtype", str(args.dtype),
]

if args.require_gpu:
    train_cmd.append("--require_gpu")

resume_checkpoint_dir = os.path.join(WORKSPACE_DIR, "resume_checkpoint")
resume_manifest = os.path.join(resume_checkpoint_dir, "_resume_state", "manifest.json")
if os.path.exists(resume_manifest):
    print(f"[*] Verified resume state found at {resume_checkpoint_dir}. Resuming training!", flush=True)
    train_cmd.extend(["--resume_from", resume_checkpoint_dir, "--allow_precision_loss"])
else:
    print("[*] No verified resume state found; starting fresh.", flush=True)

if TUNNEL_URL:
    callback_arg = f"{sys.executable} {push_helper_script} step_{{step}} {{step_dir}} {TUNNEL_URL}"
    train_cmd.extend(["--on_checkpoint_cmd", callback_arg])

print(f"\n[*] Executing JAX PPO Training: {' '.join(train_cmd)}\n", flush=True)
proc = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=sub_env)

for line in iter(proc.stdout.readline, ''):
    clean = line.strip()
    if clean:
        print(f"[colab-jax] {clean}", flush=True)

proc.wait()
if proc.returncode != 0:
    print(f"\n[!] Training process exited with code {proc.returncode}", flush=True)
    schedule_auto_unassign(60, "Training failure shutdown")
    sys.exit(proc.returncode)

print("\n[SUCCESS] JAX PPO Training completed successfully!", flush=True)


# ---------------------------------------------------------------------------
# 7. Package and Upload Final Checkpoints Archive
# ---------------------------------------------------------------------------
checkpoints_dir = os.path.join(WORKSPACE_DIR, "checkpoints")
if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
    print(f"[*] Packaging final checkpoints from {checkpoints_dir} into {OUTPUT_TAR_PATH}...", flush=True)
    with tarfile.open(OUTPUT_TAR_PATH, "w") as tar:
        tar.add(checkpoints_dir, arcname="checkpoints")
    tar_size = os.path.getsize(OUTPUT_TAR_PATH)
    print(f"[SUCCESS] Checkpoints packaged: {tar_size / (1024*1024):.1f} MB", flush=True)

    if TUNNEL_URL and tar_size > 0:
        print(f"[*] Uploading final checkpoint archive back to local workstation via {TUNNEL_URL}...", flush=True)
        chunk_size = 10 * 1024 * 1024
        total_chunks = (tar_size + chunk_size - 1) // chunk_size
        upload_success = True

        with open(OUTPUT_TAR_PATH, "rb") as f:
            for idx in range(total_chunks):
                data = f.read(chunk_size)
                url = f"{TUNNEL_URL}/upload_chunk?chunk={idx}&total={total_chunks}"
                uploaded = False
                for attempt in range(5):
                    try:
                        req = urllib.request.Request(
                            url, data=data, method="POST",
                            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}
                        )
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            if resp.status == 200:
                                uploaded = True
                                break
                    except Exception as e:
                        time.sleep(2)
                if not uploaded:
                    upload_success = False
                    print(f"[!] Critical: Failed to upload chunk {idx+1}/{total_chunks}", flush=True)
                    break
                pct = int(((idx + 1) / total_chunks) * 100)
                print(f"    [Checkpoints Transferred] {idx+1}/{total_chunks} chunks ({pct}%)...", flush=True)

        if upload_success:
            print("\n[SUCCESS] Final checkpoint transfer complete!", flush=True)
        else:
            print("\n[!] Checkpoint transfer encountered errors.", flush=True)
else:
    print("[!] No checkpoints directory found after training run.", flush=True)

# Schedule auto-shutdown in 2 minutes
print("[*] Releasing Colab VM in 2 minutes to preserve GPU/TPU quota...", flush=True)
schedule_auto_unassign(120, "Post-job shutdown")
