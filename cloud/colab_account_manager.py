"""Colab 5-Account Combo Pool Manager for JAX Reinforcement Learning.

Provides automatic quota rotation, account switching, session teardown,
and account health tracking via WSL Ubuntu-24.04-ROCmLab.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = os.path.join(PROJECT_ROOT, "cloud", "colab_accounts.json")
RTC_REGISTRY_FILE = r"C:\rtc\cloud\colab_accounts.json"

WSL_DISTRO = "Ubuntu-24.04-ROCmLab"
WSL_CONFIG_DIR = "/home/rocmuser/.config/colab-cli"
WSL_ACCOUNTS_DIR = f"{WSL_CONFIG_DIR}/accounts"


def run_wsl(cmd_str: str, check: bool = True) -> str:
    """Execute bash command in target WSL distro."""
    cmd = ["wsl", "-d", WSL_DISTRO, "bash", "-c", cmd_str]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    if check and res.returncode != 0:
        raise RuntimeError(f"WSL command failed ({res.returncode}): {res.stdout}")
    return res.stdout.strip()


def load_registry() -> Dict[str, Any]:
    """Load or initialize account registry."""
    target_file = REGISTRY_FILE
    if not os.path.exists(target_file):
        if os.path.exists(RTC_REGISTRY_FILE):
            try:
                with open(RTC_REGISTRY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                save_registry(data)
                return data
            except Exception:
                pass

        reg = {
            "active_id": "account_1",
            "accounts": [
                {
                    "id": "account_1",
                    "name": "구글 메인 계정",
                    "status": "READY",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_error": None
                }
            ]
        }
        try:
            run_wsl(f"mkdir -p {WSL_ACCOUNTS_DIR}/account_1")
            run_wsl(f"if [ -f {WSL_CONFIG_DIR}/token.json ]; then cp {WSL_CONFIG_DIR}/token.json {WSL_ACCOUNTS_DIR}/account_1/token.json; fi")
        except Exception:
            pass
        save_registry(reg)
        return reg

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active_id": None, "accounts": []}


def save_registry(reg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def list_accounts() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    reg = load_registry()
    return reg.get("accounts", []), reg.get("active_id")


def switch_account(account_id: str) -> Dict[str, Any]:
    """Switch active Colab account by copying token in WSL."""
    reg = load_registry()
    acc = next((a for a in reg.get("accounts", []) if a["id"] == account_id), None)
    if not acc:
        raise ValueError(f"Account ID '{account_id}' not found in registry.")

    # In WSL, copy target account token to active token.json
    run_wsl(f"if [ -f {WSL_ACCOUNTS_DIR}/{account_id}/token.json ]; then cp {WSL_ACCOUNTS_DIR}/{account_id}/token.json {WSL_CONFIG_DIR}/token.json; else echo 'Token not found'; exit 1; fi")

    reg["active_id"] = account_id
    save_registry(reg)
    print(f"[SUCCESS] Switched active Colab account to: {acc['name']} ({account_id})")
    return acc


def mark_status(account_id: str, status: str, error_msg: Optional[str] = None) -> None:
    """Update status (e.g. 'READY', 'QUOTA_EXHAUSTED') of an account."""
    reg = load_registry()
    for acc in reg.get("accounts", []):
        if acc["id"] == account_id:
            acc["status"] = status
            acc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if error_msg:
                acc["last_error"] = str(error_msg)
            elif status == "READY":
                acc["last_error"] = None
    save_registry(reg)


def get_next_ready_account(exclude_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Return the next available account with 'READY' status."""
    exclude_ids = exclude_ids or set()
    reg = load_registry()
    for acc in reg.get("accounts", []):
        if acc["id"] not in exclude_ids and acc.get("status") == "READY":
            return acc
    return None


def get_active_account(reg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return the currently active account dictionary or None."""
    if reg is None:
        reg = load_registry()
    active_id = reg.get("active_id")
    for acc in reg.get("accounts", []):
        if acc["id"] == active_id:
            return acc
    accounts = reg.get("accounts", [])
    return accounts[0] if accounts else None


def switch_next_account(exclude_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Mark current account quota exhausted and switch to next ready account."""
    reg = load_registry()
    curr_id = reg.get("active_id")
    excludes = set(exclude_ids or [])
    if curr_id:
        excludes.add(curr_id)
        mark_status(curr_id, "QUOTA_EXHAUSTED", "Quota exhausted")
    next_acc = get_next_ready_account(exclude_ids=excludes)
    if next_acc:
        switch_account(next_acc["id"])
        return next_acc
    return None


def reset_all_quotas() -> None:
    """Reset all accounts to READY status (e.g. daily refresh)."""
    reg = load_registry()
    for acc in reg.get("accounts", []):
        acc["status"] = "READY"
        acc["last_error"] = None
        acc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_registry(reg)
    print("[SUCCESS] All accounts in the pool reset to 'READY'.")


def stop_all_active_sessions(session_name: str = "jax_train") -> int:
    """Scan all accounts in the pool and terminate any active sessions to protect quota."""
    reg = load_registry()
    accounts = reg.get("accounts", [])
    curr_active = reg.get("active_id")
    stopped_count = 0
    print(f"\n[*] [전체 계정 세션 전수 회수] 총 {len(accounts)}개 계정 점검 시작...")

    for acc in accounts:
        try:
            switch_account(acc["id"])
            cmd = ["wsl", "-d", WSL_DISTRO, "/home/rocmuser/.local/bin/colab", "sessions"]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
            out = res.stdout or ""
            if "No active sessions" not in out and out.strip():
                print(f"[*] {acc['name']} ({acc['id']}): 활성 세션 감지됨. 세션 종료 요청...")
                stop_cmd = ["wsl", "-d", WSL_DISTRO, "/home/rocmuser/.local/bin/colab", "stop", "-s", session_name]
                s_res = subprocess.run(stop_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15)
                print(f"    -> {s_res.stdout.strip()}")
                stopped_count += 1
            else:
                print(f"[*] {acc['name']} ({acc['id']}): 활성 세션 없음 (안전)")
        except Exception as e:
            print(f"[!] {acc['name']} 점검 중 오류: {e}")

    if curr_active:
        switch_account(curr_active)

    print(f"[SUCCESS] ✅ Colab 세션 전수 회수 점검 완료. (종료된 세션: {stopped_count}개)\n")
    return stopped_count


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "list":
            accs, active = list_accounts()
            print("\n=== Google Colab Account Combo Pool (JAX) ===")
            for a in accs:
                tag = "★ [ACTIVE]" if a["id"] == active else "  [IDLE]"
                print(f"{tag} {a['id']}: {a['name']} | Status: {a['status']} | Last Error: {a['last_error']}")
            print("============================================\n")
        elif cmd == "switch" and len(sys.argv) > 2:
            switch_account(sys.argv[2])
        elif cmd == "reset-quotas":
            reset_all_quotas()
        elif cmd == "stop-all":
            sess = sys.argv[2] if len(sys.argv) > 2 else "jax_train"
            stop_all_active_sessions(sess)
        else:
            print("Usage: python colab_account_manager.py [list | switch <id> | reset-quotas | stop-all <session_name>]")
    else:
        accs, active = list_accounts()
        print("\n=== Google Colab Account Combo Pool (JAX) ===")
        for a in accs:
            tag = "★ [ACTIVE]" if a["id"] == active else "  [IDLE]"
            print(f"{tag} {a['id']}: {a['name']} | Status: {a['status']} | Last Error: {a['last_error']}")
        print("============================================\n")
