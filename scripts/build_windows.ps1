param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到虚拟环境：$Python"
}

if (-not $SkipInstall) {
    & $Python -m pip install -e "$Root" pyinstaller
}

Push-Location $Root
try {
    & $Python -m PyInstaller --noconfirm --clean packaging\CompanyEventMonitor.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 构建失败"
    }
    Copy-Item -LiteralPath "packaging\README.txt" -Destination "dist\CompanyEventMonitor\README.txt" -Force
    $Zip = Join-Path $Root "dist\CompanyEventMonitor-Windows-x64.zip"
    if (Test-Path -LiteralPath $Zip) {
        Remove-Item -LiteralPath $Zip -Force
    }
    Compress-Archive -Path "dist\CompanyEventMonitor\*" -DestinationPath $Zip -CompressionLevel Optimal
    Write-Host "构建完成：$Zip"
}
finally {
    Pop-Location
}
