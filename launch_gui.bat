@echo off
setlocal
cd /d "%~dp0"
echo Launching NSFW Image Audit Studio...
if exist "publish\NsfwAuditGui.exe" (
    start "" "publish\NsfwAuditGui.exe"
) else (
    start "" "NsfwAuditGui\bin\Release\net9.0-windows\win-x64\NsfwAuditGui.exe"
)
