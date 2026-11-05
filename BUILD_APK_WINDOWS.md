# Build the OnePlus 12R APK

1. Install Android Studio on Windows and open this folder:
   `android`
2. Let Android Studio install Android SDK Platform 35 and Build Tools.
3. In a terminal inside `android`, run:
   `gradlew.bat assembleDebug`
4. APK output:
   `android\app\build\outputs\apk\debug\app-debug.apk`

Install on the phone with USB debugging enabled:
`adb install -r android\app\build\outputs\apk\debug\app-debug.apk`

The app requests camera and sensor access at runtime and records synchronized phone data for the laptop SLAM pipeline.
