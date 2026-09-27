"""Cloudflare Quick Tunnel and Codebase / Checkpoint Transfer Bridge for JAX."""

from __future__ import annotations

import http.server
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
from typing import Optional

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
    """Extracts downloaded Orbax checkpoints tar into the project's checkpoints directory."""
    try:
        if target_dir is None:
            target_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(target_dir, exist_ok=True)
        print(f"\n[*] Extracting Orbax checkpoints from {tar_path} into {target_dir}...", flush=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=target_dir)
        print(f"[SUCCESS] Checkpoints extracted to {target_dir} successfully!\n", flush=True)
    except Exception as e:
        print(f"[!] Extraction error: {e}", flush=True)


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

    def start(self) -> str:
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
                    content_len = int(self.headers.get("Content-Length", 0))

                    chunks_dir = manager.output_checkpoints_tar + ".chunks"
                    os.makedirs(chunks_dir, exist_ok=True)
                    chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx:05d}.part")

                    with open(chunk_file, "wb") as f:
                        remaining = content_len
                        while remaining > 0:
                            sz = min(remaining, 1024 * 1024)
                            buf = self.rfile.read(sz)
                            if not buf:
                                break
                            f.write(buf)
                            remaining -= len(buf)

                    actual_len = os.path.getsize(chunk_file) if os.path.exists(chunk_file) else 0
                    if actual_len != content_len:
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                            except Exception:
                                pass
                        self.send_error(500, f"Incomplete chunk {chunk_idx}: {actual_len}/{content_len}")
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
                            extract_checkpoints_archive(manager.output_checkpoints_tar, manager.extract_target_dir)

                    try:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"CHUNK_OK")
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        pass
                elif self.path.startswith("/upload_checkpoint"):
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    step_name = query.get("step", ["unknown"])[0]
                    content_len = int(self.headers.get("Content-Length", 0))
                    print(f"\n[STREAM] Receiving live intermediate checkpoint '{step_name}' ({content_len / (1024*1024):.2f} MB)...", flush=True)
                    temp_step_tar = os.path.join(manager.extract_target_dir, f"{step_name}.tar")
                    with open(temp_step_tar, "wb") as f:
                        remaining = content_len
                        while remaining > 0:
                            sz = min(remaining, 1024 * 1024)
                            buf = self.rfile.read(sz)
                            if not buf:
                                break
                            f.write(buf)
                            remaining -= len(buf)
                    extract_checkpoints_archive(temp_step_tar, manager.extract_target_dir)
                    try:
                        if os.path.exists(temp_step_tar):
                            os.remove(temp_step_tar)
                    except Exception:
                        pass
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
