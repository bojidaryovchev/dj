<#
    DJ Crate launcher.

    Gets the machine from "nothing installed" to "helper running":
      1. Python        (via winget, if missing)
      2. JS runtime    (Node via winget, if neither Node nor Deno is present)
      3. config.json   (created from the example, with a fresh token)
      4. yt-dlp/ffmpeg (bootstrap.py, into bin/)
      5. starts the server

    Everything here is idempotent -- on a normal day it falls straight
    through to step 5.

    Use -Yes to install prerequisites without prompting.
#>

[CmdletBinding()]
param(
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Write-Step($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "    $msg" -ForegroundColor DarkGray }

# winget installs land in the registry PATH, not this already-running shell.
function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = ($machine, $user | Where-Object { $_ }) -join ';'
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# "python" on Windows is often the Microsoft Store stub, which resolves but
# does not run. Only trust it if it actually reports a version.
function Test-RealPython {
    if (-not (Test-Command 'python')) { return $false }
    try {
        $v = & python --version 2>&1
        return ($LASTEXITCODE -eq 0 -and "$v" -match 'Python 3')
    } catch {
        return $false
    }
}

function Confirm-Install($what) {
    if ($Yes) { return $true }
    Write-Host ""
    Write-Warn "$what is not installed."
    $answer = Read-Host "    Install it now with winget? [Y/n]"
    return ($answer -eq '' -or $answer -match '^[Yy]')
}

function Install-WingetPackage($id, $label) {
    if (-not (Test-Command 'winget')) {
        throw "$label is missing and winget is unavailable. Install $label manually, then re-run."
    }
    Write-Step "Installing $label via winget (this can take a few minutes)..."
    & winget install --id $id --exact --source winget `
        --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install $label (exit $LASTEXITCODE)."
    }
    Update-PathFromRegistry
}

# --- 1. Python ----------------------------------------------------------
Write-Step "Checking Python"
if (-not (Test-RealPython)) {
    if (-not (Confirm-Install 'Python 3')) { throw "Python is required. Aborting." }
    Install-WingetPackage 'Python.Python.3.13' 'Python 3.13'
    if (-not (Test-RealPython)) {
        throw "Python installed but is not on PATH yet. Close this window, open a new one, and re-run."
    }
}
Write-Ok (& python --version 2>&1)

# --- 2. JavaScript runtime ---------------------------------------------
# YouTube gates playback behind JS challenges yt-dlp must execute. Without a
# runtime some formats silently go missing -- a worse rip with no error.
Write-Step "Checking JavaScript runtime"
if (-not (Test-Command 'node') -and -not (Test-Command 'deno')) {
    if (Confirm-Install 'A JavaScript runtime (Node.js)') {
        Install-WingetPackage 'OpenJS.NodeJS.LTS' 'Node.js LTS'
    } else {
        Write-Warn "Continuing without one. Some YouTube formats will be missing."
    }
}
if (Test-Command 'node')      { Write-Ok "node $(& node --version)" }
elseif (Test-Command 'deno')  { Write-Ok (& deno --version | Select-Object -First 1) }

# --- 3. config.json -----------------------------------------------------
Write-Step "Checking config"
if (-not (Test-Path 'config.json')) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = ($bytes | ForEach-Object { $_.ToString('x2') }) -join ''

    $cfg = Get-Content 'config.example.json' -Raw | ConvertFrom-Json
    $cfg.token = $token
    $json = $cfg | ConvertTo-Json -Depth 5

    # Out-File -Encoding utf8 emits a BOM on PowerShell 5.1, which Python's
    # json.load rejects outright. Write UTF-8 without one.
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot 'config.json'), $json, $utf8NoBom)

    Write-Host ""
    Write-Host "    Created config.json with a new token. Paste this into the" -ForegroundColor Green
    Write-Host "    extension's options page:" -ForegroundColor Green
    Write-Host ""
    Write-Host "        $token" -ForegroundColor White
    Write-Host ""
} else {
    Write-Ok "config.json present"
}

# --- 4. yt-dlp + ffmpeg -------------------------------------------------
Write-Step "Checking yt-dlp and ffmpeg"
& python -u bootstrap.py
if ($LASTEXITCODE -ne 0) { throw "bootstrap.py failed (exit $LASTEXITCODE)." }

# --- 5. run -------------------------------------------------------------
Write-Host ""
& python -u server.py
