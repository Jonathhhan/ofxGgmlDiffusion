@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-pixel-gan-model.ps1" %*
exit /b %ERRORLEVEL%
