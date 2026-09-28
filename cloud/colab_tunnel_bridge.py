"""Cloudflare Quick Tunnel and Codebase / Checkpoint Transfer Bridge for JAX."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
from typing import Dict, Optional, Set

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED_EXE = os.path.join(PROJECT_ROOT, "bin", "cloudflared.exe")
if not os.path.exists(CLOUDFLARED_EXE):
    fallback = r"C:\rtc\bin\cloudflared.exe"
    if os.path.exists(fallback):
        CLOUDFLARED_EXE = fallback


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def extract_checkpoints_archive(tar_path: str, target_dir: Optional[str] = None) -> None:
    """Extracts downloaded Orbax checkpoints tar into the project's checkpoints directory.

    Raises exceptions on extraction failure to allow caller to handle integrity errors.
    """
    if target_dir is None:
        target_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n[*] Extracting Orbax checkpoints from {tar_path} into {target_dir}...", flush=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    print(f"[SUCCESS] Checkpoints extracted to {target_dir} successfully!\n", flush=True)


class JaxTunnelTransferManager:
    """Manages local HTTP streaming server and Cloudflare tunnel for JAX training on Colab."""

    def __init__(
        self,
        data_tar_path: str,
        script_path: str,
        output_checkpoints_tar: str,
        extract_target_dir: Optional[str] = None,
        port: Optional[int] = None
    ):
        self.data_tar_path = data_tar_path
        self.script_path = script_path
        self.output_checkpoints_tar = output_checkpoints_tar
        self.extract_target_dir = extract_target_dir or os.path.join(PROJECT_ROOT, "checkpoints")
        self.port = port or get_free_port()
        self.tunnel_url: Optional[str] = None
        self.tunnel_proc: Optional[subprocess.Popen] = None
        self.httpd: Optional[http.server.ThreadingHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.checkpoints_received: bool = False
        self.checkpoints_bytes: int = 0
        self.verified_steps: Set[int] = set()
        self.checkpoint_hashes: Dict[str, str] = {}

    def get_last_verified_step(self) -> int:
        """Returns the highest verified checkpoint update step number."""
        return max(self.verified_steps) if self.verified_steps else 0

    def is_step_verified(self, step: int) -> bool:
        """Returns True if the specified update step has been verified on the host."""
        return step in self.verified_steps

    def start(self, enable_tunnel: bool = True) -> str:
        manager = self

        class TransferHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"OK")
                    return

                if self.path == "/data":
                    if not os.path.exists(manager.data_tar_path):
                        self.send_error(404, "Data tar not found")
                        return
                    size = os.path.getsize(manager.data_tar_path)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-tar")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    print(f"\n[STREAM] Streaming upload_jax.tar ({size / (1024*1024):.1f} MB) to Colab VM...", flush=True)
                    chunk_sz = 8 * 1024 * 1024
                    sent = 0
                    with open(manager.data_tar_path, "rb") as f:
                        while True:
                            buf = f.read(chunk_sz)
                            if not buf:
                                break
                            self.wfile.write(buf)
                            sent += len(buf)
                    print("[STREAM] upload_jax.tar transfer completed successfully!", flush=True)

                elif self.path == "/script":
                    if not os.path.exists(manager.script_path):
                        self.send_error(404, "Script not found")
                        return
                    with open(manager.script_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/x-python")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.startswith("/upload_chunk"):
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    chunk_idx = int(query.get("chunk", [0])[0])
                    total_chunks = int(query.get("total", [1])[0])
                    expected_sha256 = query.get("sha256", [None])[0]

                    content_len_hdr = self.headers.get("Content-Length")
                    if not content_len_hdr:
                        self.send_error(400, "Missing Content-Length header")
                        return
                    try:
                        content_len = int(content_len_hdr)
                    except ValueError:
                        self.send_error(400, "Invalid Content-Length header")
                        return

                    chunks_dir = manager.output_checkpoints_tar + ".chunks"
                    os.makedirs(chunks_dir, exist_ok=True)
                    chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx:05d}.part")

                    hasher = hashlib.sha256()
                    received_bytes = 0
                    try:
                        with open(chunk_file, "wb") as f:
                            remaining = content_len
                            while remaining > 0:
                                sz = min(remaining, 1024 * 1024)
                                buf = self.rfile.read(sz)
                                if not buf:
                                    break
                                f.write(buf)
                                hasher.update(buf)
                                remaining -= len(buf)
                                received_bytes += len(buf)
                    except Exception as io_err:
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                            except Exception:
                                pass
                        self.send_error(500, f"IO error while writing chunk: {io_err}")
                        return

                    if received_bytes != content_len:
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                            except Exception:
                                pass
                        self.send_error(400, f"Incomplete chunk {chunk_idx}: received {received_bytes}/{content_len}")
                        return

                    computed_hash = hasher.hexdigest()
                    if expected_sha256 and computed_hash.lower() != expected_sha256.lower():
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                            except Exception:
                                pass
                        self.send_error(400, f"Chunk SHA256 mismatch: {computed_hash} vs expected {expected_sha256}")
                        return

                    pct = int(((chunk_idx + 1) / total_chunks) * 100)
                    print(f"[STREAM] Received checkpoint chunk {chunk_idx+1}/{total_chunks} ({pct}%)...", flush=True)

                    if chunk_idx + 1 == total_chunks:
                        all_present = all(os.path.exists(os.path.join(chunks_dir, f"chunk_{i:05d}.part")) for i in range(total_chunks))
                        if all_present:
                            print(f"\n[*] Assembling {total_chunks} chunks into final checkpoint archive...", flush=True)
                            os.makedirs(os.path.dirname(os.path.abspath(manager.output_checkpoints_tar)), exist_ok=True)
                            temp_out = manager.output_checkpoints_tar + ".assembling"
                            with open(temp_out, "wb") as out_f:
                                for i in range(total_chunks):
                                    cpath = os.path.join(chunks_dir, f"chunk_{i:05d}.part")
                                    with open(cpath, "rb") as in_f:
                                        shutil.copyfileobj(in_f, out_f, length=4 * 1024 * 1024)
                                    try:
                                        os.remove(cpath)
                                    except Exception:
                                        pass
                            try:
                                os.rmdir(chunks_dir)
                            except Exception:
                                pass

                            if os.path.exists(manager.output_checkpoints_tar):
                                try:
                                    os.remove(manager.output_checkpoints_tar)
                                except Exception:
                                    pass
                            os.replace(temp_out, manager.output_checkpoints_tar)
                            manager.checkpoints_bytes = os.path.getsize(manager.output_checkpoints_tar)
                            manager.checkpoints_received = True
                            print(f"[STREAM] All chunks assembled! ({manager.checkpoints_bytes / (1024*1024):.1f} MB)", flush=True)
                            try:
                                extract_checkpoints_archive(manager.output_checkpoints_tar, manager.extract_target_dir)
                            except Exception as ext_err:
                                self.send_error(500, f"Final archive extraction failed: {ext_err}")
                                return

                    try:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"CHUNK_OK")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        pass

                elif self.path.startswith("/upload_checkpoint"):
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    step_name = query.get("step", ["unknown"])[0]
                    expected_sha256 = query.get("sha256", [None])[0]

                    content_len_hdr = self.headers.get("Content-Length")
                    if not content_len_hdr:
                        self.send_error(400, "Missing Content-Length header")
                        return
                    try:
                        content_len = int(content_len_hdr)
                    except ValueError:
                        self.send_error(400, "Invalid Content-Length header")
                        return

                    print(f"\n[STREAM] Receiving live intermediate checkpoint '{step_name}' ({content_len / (1024*1024):.2f} MB)...", flush=True)
                    temp_step_tar = os.path.join(manager.extract_target_dir, f"{step_name}_{time.time_ns()}.tar")
                    hasher = hashlib.sha256()
                    received_bytes = 0
                    try:
                        with open(temp_step_tar, "wb") as f:
                            remaining = content_len
                            while remaining > 0:
                                sz = min(remaining, 1024 * 1024)
                                buf = self.rfile.read(sz)
                                if not buf:
                                    break
                                f.write(buf)
                                hasher.update(buf)
                                remaining -= len(buf)
                                received_bytes += len(buf)
                    except Exception as io_err:
                        if os.path.exists(temp_step_tar):
                            try:
                                os.remove(temp_step_tar)
                            except Exception:
                                pass
                        self.send_error(500, f"IO error while receiving checkpoint: {io_err}")
                        return

                    # (a) Content-Length match check
                    if received_bytes != content_len:
                        if os.path.exists(temp_step_tar):
                            try:
                                os.remove(temp_step_tar)
                            except Exception:
                                pass
                        self.send_error(400, f"Incomplete checkpoint: received {received_bytes}/{content_len}")
                        return

                    computed_hash = hasher.hexdigest()

                    # (b) SHA256 query parameter check
                    if expected_sha256 and computed_hash.lower() != expected_sha256.lower():
                        if os.path.exists(temp_step_tar):
                            try:
                                os.remove(temp_step_tar)
                            except Exception:
                                pass
                        self.send_error(400, f"SHA256 mismatch: {computed_hash} vs expected {expected_sha256}")
                        return

                    # (c) Idempotency and Conflict check
                    if step_name in manager.checkpoint_hashes:
                        prev_hash = manager.checkpoint_hashes[step_name]
                        if prev_hash.lower() == computed_hash.lower():
                            # Idempotent re-upload of identical checkpoint
                            if os.path.exists(temp_step_tar):
                                try:
                                    os.remove(temp_step_tar)
                                except Exception:
                                    pass
                            try:
                                self.send_response(200)
                                self.end_headers()
                                self.wfile.write(b"CHECKPOINT_OK_IDEMPOTENT")
                            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                                pass
                            return
                        else:
                            # Conflict: different hash for the same step identifier
                            if os.path.exists(temp_step_tar):
                                try:
                                    os.remove(temp_step_tar)
                                except Exception:
                                    pass
                            self.send_error(409, f"Conflict: step '{step_name}' already exists with different content hash")
                            return

                    # (d) Extract and verify
                    try:
                        extract_checkpoints_archive(temp_step_tar, manager.extract_target_dir)
                    except Exception as ext_err:
                        self.send_error(500, f"Checkpoint archive extraction failed: {ext_err}")
                        return
                    finally:
                        if os.path.exists(temp_step_tar):
                            try:
                                os.remove(temp_step_tar)
                            except Exception:
                                pass

                    # (e) Record verified step
                    step_num = None
                    try:
                        m = re.search(r"(\d+)", step_name)
                        if m:
                            step_num = int(m.group(1))
                    except Exception:
                        pass

                    if step_num is not None:
                        manager.verified_steps.add(step_num)
                    manager.checkpoint_hashes[step_name] = computed_hash

                    try:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"CHECKPOINT_OK")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        pass
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), TransferHandler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[*] Local transfer server listening on 127.0.0.1:{self.port}", flush=True)

        if not enable_tunnel:
            self.tunnel_url = f"http://127.0.0.1:{self.port}"
            return self.tunnel_url

        if not os.path.exists(CLOUDFLARED_EXE):
            raise FileNotFoundError(f"cloudflared.exe not found at {CLOUDFLARED_EXE}")

        try:
            subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception:
            pass

        print("[*] Spawning high-speed Cloudflare Quick Tunnel...", flush=True)
        cmd = [CLOUDFLARED_EXE, "tunnel", "--url", f"http://127.0.0.1:{self.port}"]
        self.tunnel_proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')

        start = time.time()
        captured_stderr = []
        for line in iter(self.tunnel_proc.stderr.readline, ''):
            captured_stderr.append(line.strip())
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if m:
                self.tunnel_url = m.group(0)
                break
            if time.time() - start > 45:
                break

        if not self.tunnel_url:
            self.stop()
            err_summary = " | ".join([l for l in captured_stderr if l][-3:]) if captured_stderr else "No output from cloudflared"
            raise TimeoutError(f"Failed to establish Cloudflare tunnel within 45s. ({err_summary})")

        print(f"[SUCCESS] Cloudflare tunnel established: {self.tunnel_url}", flush=True)
        return self.tunnel_url

    def start_server_only(self) -> str:
        """Starts only the local HTTP transfer server without spawning Cloudflare tunnel."""
        return self.start(enable_tunnel=False)

    def stop(self) -> None:
        if self.tunnel_proc:
            try:
                self.tunnel_proc.terminate()
                self.tunnel_proc.wait(timeout=3)
            except Exception:
                try:
                    self.tunnel_proc.kill()
                except Exception:
                    pass
            self.tunnel_proc = None

        try:
            subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None
        print("[*] Local transfer server and tunnel closed.", flush=True)
