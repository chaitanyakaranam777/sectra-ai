$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
$VenvPython = Join-Path $Project ".venv\Scripts\python.exe"
Write-Host "`nSENTRA AI - SETUP AND LAUNCH" -ForegroundColor Cyan
if (!(Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    py -3 -m venv (Join-Path $Project ".venv")
    $VenvPython = Join-Path $Project ".venv\Scripts\python.exe"
}
Write-Host "Checking dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install -r (Join-Path $Project "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the error above." }
$env:SENTRA_CORS_ORIGINS = "http://127.0.0.1:5500,http://localhost:5500"
$api = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
if (!$api) {
    Start-Process -FilePath $VenvPython -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8001" -WorkingDirectory $Project -RedirectStandardOutput (Join-Path $Project "api-output.log") -RedirectStandardError (Join-Path $Project "api-error.log")
}
$web = Get-NetTCPConnection -LocalPort 5500 -State Listen -ErrorAction SilentlyContinue
if (!$web) {
    Start-Process -FilePath $VenvPython -ArgumentList "-m http.server 5500 --bind 127.0.0.1" -WorkingDirectory $Project -RedirectStandardOutput (Join-Path $Project "web-output.log") -RedirectStandardError (Join-Path $Project "web-error.log")
}
Start-Sleep -Seconds 6
try {
    $h = Invoke-RestMethod "http://127.0.0.1:8001/api/health"
    Write-Host "API ONLINE | Device: $($h.device) | Database: $($h.database)" -ForegroundColor Green
} catch { Write-Host "API did not respond. Check api-error.log." -ForegroundColor Red; throw }
Start-Process "http://127.0.0.1:5500/dashboard.html"
Write-Host "Dashboard opened: http://127.0.0.1:5500/dashboard.html" -ForegroundColor Green
