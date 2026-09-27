"""Colab CLI Windows-to-WSL Bridge.

Translates Windows paths and options into WSL commands targeting the Colab CLI:
wsl -d Ubuntu-24.04-ROCmLab /home/rocmuser/.local/bin/colab ...
"""

import sys
import os
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def to_wsl(arg: str) -> str:
    if not isinstance(arg, str):
        return arg
    # If path has Windows drive letter (e.g. C:\... or C:/...)
    if len(arg) >= 2 and arg[1] == ':' and arg[0].isalpha():
        abs_p = os.path.abspath(arg)
        drive, rest = os.path.splitdrive(abs_p)
        if drive:
            return f"/mnt/{drive[0].lower()}" + rest.replace('\\', '/')
    # If it contains backslashes or is an existing path
    if '\\' in arg or os.path.exists(arg):
        abs_p = os.path.abspath(arg)
        drive, rest = os.path.splitdrive(abs_p)
        if drive:
            return f"/mnt/{drive[0].lower()}" + rest.replace('\\', '/')
        return arg.replace('\\', '/')
    return arg


if __name__ == "__main__":
    raw_args = sys.argv[1:]

    # Translate legacy 'list' command into 'ls'
    if raw_args and raw_args[0] == "list":
        raw_args[0] = "ls"

    # Translate legacy 'start' command into 'run'
    if raw_args and raw_args[0] == "start":
        new_args = ["run", "--gpu", "T4", "--timeout", "36000"]
        i = 1
        script_to_run = None
        cmd_to_run = None
        extra = []
        while i < len(raw_args):
            a = raw_args[i]
            if a in ("-s", "--session") and i + 1 < len(raw_args):
                new_args.extend(["--session", raw_args[i + 1]])
                i += 2
            elif a in ("-f", "--file") and i + 1 < len(raw_args):
                script_to_run = raw_args[i + 1]
                i += 2
            elif a in ("-c", "--command") and i + 1 < len(raw_args):
                cmd_to_run = raw_args[i + 1]
                i += 2
            else:
                extra.append(a)
                i += 1

        if cmd_to_run:
            temp_dir = os.environ.get("TEMP", "C:\\temp")
            runner_path = os.path.join(temp_dir, "colab_cmd_runner.py")
            os.makedirs(os.path.dirname(runner_path), exist_ok=True)
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(cmd_to_run)
            script_to_run = runner_path

        if script_to_run:
            new_args.append(to_wsl(script_to_run))
        new_args.extend([to_wsl(e) for e in extra])
        converted_args = new_args
    else:
        converted_args = [to_wsl(a) for a in raw_args]

    cmd = ["wsl", "-d", "Ubuntu-24.04-ROCmLab", "/home/rocmuser/.local/bin/colab"] + converted_args
    try:
        res = subprocess.run(cmd)
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
