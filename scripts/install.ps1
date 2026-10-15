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

# Native programs (uv) report progress and notes on stderr. Windows PowerShell 5.1 turns
# redirected stderr into error records, which "Stop" would make fatal, so this script checks
# exit codes instead of relying on $ErrorActionPreference.
$ErrorActionPreference = "Continue"
$Pkg = if ($env:TRAINLAB_PACKAGE) { $env:TRAINLAB_PACKAGE } else { "trainlab-optimize" }
$Py = if ($env:TRAINLAB_PYTHON) { $env:TRAINLAB_PYTHON } else { "3.12" }

function Fail($msg) {
    Write-Host "trainlab install: $msg" -ForegroundColor Red
    # exit would close the window when run through "irm ... | iex"; throw stops only the script.
    throw "trainlab install failed"
}

# Runs a native command, hides all its output, and returns its exit code.
function Invoke-Quiet([scriptblock]$Cmd) {
    $null = & $Cmd 2>&1
    return $LASTEXITCODE
}

# Runs a native command and returns its stdout as one trimmed string (stderr discarded).
function Get-Output([scriptblock]$Cmd) {
    $out = & $Cmd 2>&1 | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }
    if ($LASTEXITCODE -ne 0) { return $null }
    return (($out | Out-String).Trim())
}

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
$null = Invoke-Quiet { & $Uv tool update-shell }          # adds uv's bin folder to PATH if needed

$Bin = Get-Output { & $Uv tool dir --bin }
if (-not $Bin) { $Bin = "$env:USERPROFILE\.local\bin" }
$Tl = Join-Path $Bin "trainlab.exe"
if (-not (Test-Path $Tl)) { Fail "installed, but trainlab.exe can't be found in $Bin." }

$Version = Get-Output { & $Tl --version }
Write-Host ""
Write-Host "Installed $Version." -ForegroundColor Green
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
