# TrainLab Optimize installer for Windows (PowerShell 5.1 or 7).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://trainlab.ai/install.ps1 | iex"
#
# Installs uv (Astral's Python tool manager) if it's missing, then installs trainlab into its
# own isolated environment with its own Python. Nothing else on the machine changes: no system
# Python, no admin rights, no packages in your projects.
# To remove it: uv tool uninstall trainlab-optimize
#
# Options (environment variables):
#   TRAINLAB_PACKAGE   what to install (default trainlab-optimize from PyPI; also a wheel file,
#                      a URL, or git+https://... for pre-releases)
#   TRAINLAB_PYTHON    Python version for trainlab's environment (default 3.12)

$ErrorActionPreference = "Stop"
$Pkg = if ($env:TRAINLAB_PACKAGE) { $env:TRAINLAB_PACKAGE } else { "trainlab-optimize" }
$Py = if ($env:TRAINLAB_PYTHON) { $env:TRAINLAB_PYTHON } else { "3.12" }

function Fail($msg) { Write-Host "trainlab install: $msg" -ForegroundColor Red; exit 1 }

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($c in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$Uv = Find-Uv
if (-not $Uv) {
    Write-Host "Installing uv, the Python tool manager trainlab is installed with ..."
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) { Fail "downloading uv failed. Check the internet connection or proxy and run this again." }
    $Uv = Find-Uv
    if (-not $Uv) { Fail "uv was installed but can't be found. Open a new terminal and run this again." }
}

Write-Host "Installing trainlab-optimize (Python $Py, isolated) ..."
& $Uv tool install --python $Py --force $Pkg
if ($LASTEXITCODE -ne 0) { Fail "the install failed (see above)." }
& $Uv tool update-shell *> $null

$Bin = (& $Uv tool dir --bin 2>$null)
if (-not $Bin) { $Bin = "$env:USERPROFILE\.local\bin" }
$Tl = Join-Path $Bin "trainlab.exe"
if (-not (Test-Path $Tl)) { Fail "installed, but trainlab.exe can't be found in $Bin." }

Write-Host ""
Write-Host ("Installed " + (& $Tl --version) + ".")
if (-not (($env:Path -split ";") -contains $Bin)) {
    $env:Path = "$Bin;$env:Path"          # this window; new terminals get it from update-shell
    Write-Host "Open a new terminal if 'trainlab' isn't found there."
}
Write-Host ""
Write-Host "Next, in your project folder:"
Write-Host "  trainlab ui                                  # the app in your browser"
Write-Host "  trainlab run . --usage billing.csv --open    # 1. where the money goes"
Write-Host "  trainlab trace -- python app.py              # 2. exact cost per line, no code change"
Write-Host "  trainlab trace --capture -- python app.py    # 3. then: trainlab experiment run --max-cost 2 --open"
