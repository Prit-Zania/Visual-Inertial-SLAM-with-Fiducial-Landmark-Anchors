# Build script for OnePlus SLAM Capture Mobile APK
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$toolsRoot = "$scriptDir\..\..\..\_android_build_tools"

$env:JAVA_HOME = "$toolsRoot\jdk-17\jdk-17.0.21.0.2+1"
$env:ANDROID_HOME = "$toolsRoot\android-sdk"
$env:ANDROID_SDK_ROOT = "$toolsRoot\android-sdk"
$env:PATH = "$env:JAVA_HOME\bin;$toolsRoot\android-sdk\platform-tools;$env:PATH"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Building OnePlus SLAM Capture Mobile APK " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$androidDir = "$scriptDir\android"

Push-Location $androidDir
try {
    .\gradlew.bat assembleDebug
} finally {
    Pop-Location
}

$apkSource = "$androidDir\app\build\outputs\apk\debug\app-debug.apk"
if (Test-Path $apkSource) {
    $target = "$scriptDir\OnePlus_SLAM_Capture-debug.apk"
    Copy-Item $apkSource $target -Force
    Write-Host "`nAPK Built Successfully!" -ForegroundColor Green
    Write-Host "Output APK: $target" -ForegroundColor Yellow

    $devices = (& "$toolsRoot\android-sdk\platform-tools\adb.exe" devices | Select-String "device$")
    if ($devices) {
        Write-Host "`nFound connected Android device. Installing APK..." -ForegroundColor Cyan
        & "$toolsRoot\android-sdk\platform-tools\adb.exe" install -r $target
    } else {
        Write-Host "`nTip: To install on your OnePlus phone via USB Debugging, run:" -ForegroundColor Cyan
        Write-Host "& `"$toolsRoot\android-sdk\platform-tools\adb.exe`" install -r `"$target`"" -ForegroundColor White
    }
}
