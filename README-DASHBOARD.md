# SENTRA AI

## Start on Windows
1. Extract this ZIP.
2. Open the extracted `sentra-ai` folder in File Explorer.
3. Right-click `Start-SENTRA.ps1` and choose **Run with PowerShell**, or open PowerShell in this folder and run:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   `.\Start-SENTRA.ps1`
4. Dashboard: http://127.0.0.1:5500/dashboard.html
5. API docs: http://127.0.0.1:8001/docs

The launcher creates `.venv` if needed, installs requirements, starts the API and dashboard server, and opens the dashboard.

## Important
- This package contains the dashboard frontend and the project source.
- The MVSA dataset is not bundled. Add the officially obtained dataset under `data/raw/` and follow `docs/DATASET_IMPORT.md`.
- Model architectures being registered does not mean they are trained. Prediction requires a compatible trained run/checkpoint.
- First-time dependency installation requires internet access.
