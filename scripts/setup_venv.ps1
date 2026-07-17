param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in .venv"
    & $Python -m venv .venv
}

Write-Host "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing project requirements"
& $VenvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Run the pipeline with:"
Write-Host ".\.venv\Scripts\python.exe scripts\run_hpx_hades.py --vnir2-root-dir `"E:\HADES_HPX_04-2026\VNIR2\Measurement`" --root-mask-dir `"E:\HADES_HPX_04-2026\ROOT2_analysis`""
