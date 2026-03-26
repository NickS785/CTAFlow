param(
    [string]$GatewayUrl = "https://localhost:5000",
    [string]$ConId = "269460054",
    [string]$AccountId = "",
    [string]$Ticker = "NG",
    [string]$VerifySsl = "false",
    [string]$Period = "2d",
    [string]$BarSize = "1min",
    [string]$Timezone = "America/Chicago",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        $resolved = [System.IO.Path]::GetFullPath($candidate)
        if (Test-Path $resolved) {
            $PythonExe = $resolved
            break
        }
    }
}

if (-not $PythonExe) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $PythonExe = $pythonCmd.Source
    }
}

if (-not $PythonExe) {
    throw "No Python interpreter found. Pass -PythonExe explicitly."
}

Write-Host "Using Python: $PythonExe"

$env:CTAFLOW_IBKR_GATEWAY_URL = $GatewayUrl
$env:CTAFLOW_IBKR_CONID = $ConId
$env:CTAFLOW_IBKR_ACCOUNT_ID = $AccountId
$env:CTAFLOW_IBKR_TICKER = $Ticker
$env:CTAFLOW_IBKR_VERIFY_SSL = $VerifySsl
$env:CTAFLOW_IBKR_PERIOD = $Period
$env:CTAFLOW_IBKR_BAR_SIZE = $BarSize
$env:CTAFLOW_IBKR_TZ = $Timezone
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

& $PythonExe -c "import sys; print(sys.executable)"
& $PythonExe -m pytest tests/test_ibkr_client.py -q
