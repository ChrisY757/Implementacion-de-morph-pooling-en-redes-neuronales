@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" gui_experimentos.py
) else (
    python gui_experimentos.py
)
if errorlevel 1 pause
