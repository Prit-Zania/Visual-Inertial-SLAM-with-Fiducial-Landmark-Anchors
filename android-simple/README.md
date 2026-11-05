# OnePlus Nord CE Sensor Studio APK

This lightweight project app records every Android `SensorManager` sensor exposed by the OnePlus Nord CE phone, including separate wake-up and non-wakeup variants, one-shot trigger sensors, accelerometer, gyroscope, magnetic-field/compass, rotation, light, proximity, RGB and vendor sensors. The session inventory stores registration success, reporting mode, rate limits, power, range, resolution and vendor information. It also records rear-camera JPEG frames at about 5 FPS with an updated modern dark AMOLED UI.

Data is always written to the app's external files directory as `sessions/<id>/capture.jsonl`. If a laptop IP is entered, the same newline-delimited JSON records are sent over TCP port 8765.

Build in Ubuntu/WSL with `bash build_apk.sh`. Install with:

`adb install -r OnePlusNordCE-Sensor-Capture-debug.apk`

This debug APK is for project testing and high-rate SLAM sensor streaming.
