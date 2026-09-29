@echo off
title Lotus Phase 1 Step 35000 Policy Viewer
cd /d "%~dp0"
echo [*] Launching Lotus Phase 1 60 FPS Pygame Viewer (Mode 1 Remastered)...
echo [*] Controls: Space (Pause), R (Reset), 1-4 (Speed), Esc (Exit)
.venv\Scripts\python.exe view_policy.py --checkpoint_path checkpoints/step_35000 --mode 1
pause
