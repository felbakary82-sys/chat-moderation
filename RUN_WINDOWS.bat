@echo off
REM Double-click this file. It installs everything, trains all models (first time only) and starts the app.
REM Everything is also written to setup_log.txt in this folder.
cd /d "%~dp0"
set "LOG=%~dp0setup_log.txt"
echo ==== setup started %date% %time% ==== > "%LOG%"
echo ============================================================
echo  SWE015 Chat Moderation - automatic setup and run
echo  (a log is written to setup_log.txt)
echo ============================================================

set "PY="
for %%V in (312 311 310) do (
  if not defined PY if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set "PY=%LocalAppData%\Programs\Python\Python%%V\python.exe"
  if not defined PY if exist "%ProgramFiles%\Python%%V\python.exe" set "PY=%ProgramFiles%\Python%%V\python.exe"
)
if not defined PY (
  echo [PROBLEM] Python 3.12 was not found in the usual folders. >> "%LOG%"
  echo.
  echo  [PROBLEM] Python 3.12 was not found. Install python-3.12.10-amd64.exe first.
  pause
  exit /b 1
)
echo Using Python: "%PY%" >> "%LOG%"
echo [1/4] Using Python: %PY%
"%PY%" --version >> "%LOG%" 2>&1

if not exist ".venv\Scripts\python.exe" (
  echo [2/4] Creating virtual environment...
  "%PY%" -m venv .venv >> "%LOG%" 2>&1
)
if not exist ".venv\Scripts\python.exe" goto fail
set "VPY=%~dp0.venv\Scripts\python.exe"

echo [3/4] Installing libraries - this takes 5-15 minutes, please wait...
"%VPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%VPY%" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 goto fail
echo Libraries installed. >> "%LOG%"

if not exist "models\bert_int8\model_int8.pt" (
  echo [4/4] Training all models - BERT takes about 1 hour. Do not close this window.
  "%VPY%" train_all.py
) else (
  echo [4/4] Models already trained - skipping training.
)
if not exist "models\bert_int8\model_int8.pt" goto fail

echo.
echo  Starting the app... open http://127.0.0.1:7860 in your browser.
echo Starting app >> "%LOG%"
"%VPY%" app.py
pause
exit /b 0

:fail
echo FAILED >> "%LOG%"
echo.
echo  [ERROR] Something failed. Tell Claude - it can read setup_log.txt.
pause
exit /b 1
