#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$project_dir/build"
framework_jar="/usr/share/java/com.android.android-23.jar"
framework_res="/usr/share/android-framework-res/framework-res.apk"
tools_dir="/usr/lib/android-sdk/build-tools/debian"

rm -rf "$build_dir"
mkdir -p "$build_dir/classes"

"$tools_dir/aapt" package -f -M "$project_dir/AndroidManifest.xml" -I "$framework_res" -F "$build_dir/resources.apk"

javac -source 7 -target 7 -encoding UTF-8 \
  -bootclasspath "$framework_jar" \
  -d "$build_dir/classes" \
  "$project_dir/src/org/oneplus/sensorsimple/MainActivity.java"

"$tools_dir/dx" --dex --output="$build_dir/classes.dex" "$build_dir/classes"
cp "$build_dir/resources.apk" "$build_dir/unsigned.apk"
cd "$build_dir"
"$tools_dir/aapt" add unsigned.apk classes.dex
"$tools_dir/zipalign" -f 4 unsigned.apk aligned.apk

keystore="$project_dir/project-debug.keystore"
if [ ! -f "$keystore" ]; then
  keytool -genkeypair -noprompt -keystore "$keystore" -storepass android -keypass android \
    -alias projectkey -dname "CN=OnePlus Sensor Project,O=Student Project,C=IN" \
    -keyalg RSA -keysize 2048 -validity 3650
fi

"$tools_dir/apksigner" sign --ks "$keystore" --ks-pass pass:android --key-pass pass:android \
  --out "$project_dir/OnePlusNordCE-Sensor-Capture-debug.apk" aligned.apk
cp "$project_dir/OnePlusNordCE-Sensor-Capture-debug.apk" "$project_dir/OnePlus12R-Sensor-Capture-debug.apk"
"$tools_dir/apksigner" verify --verbose "$project_dir/OnePlusNordCE-Sensor-Capture-debug.apk"
