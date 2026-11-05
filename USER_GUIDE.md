# User Guide: OnePlus Visual-Inertial SLAM & Web Application

Welcome to the **IE415 Visual-Inertial SLAM with Fiducial Landmark Anchors** project. This guide explains how to install the Android app, connect it to the laptop web application over Wi-Fi, record datasets, run SLAM, and inspect results.

---

## Quick Start (3 Steps)

### Step 1: Launch the Laptop Web Application
In PowerShell inside this project folder:
```powershell
.\Start-Web-App.ps1
```
This starts the web server and automatically opens **`http://127.0.0.1:8080`** in your browser.

### Step 2: Install the App on your OnePlus Phone
You have the pre-built Android APK ready in this folder:
`OnePlusNordCE4-Sensor-Capture-debug.apk`

- **Via USB & ADB**:
  ```powershell
  adb install -r OnePlusNordCE4-Sensor-Capture-debug.apk
  ```
- **Or via Direct Transfer**:
  Copy `OnePlusNordCE4-Sensor-Capture-debug.apk` to your phone via USB or Google Drive, tap it in your phone's File Manager, and tap **Install**.

### Step 3: Connect & Stream Over Wi-Fi
1. Ensure your phone and laptop are on the same Wi-Fi network (or connect your phone to your laptop's mobile hotspot).
2. Look at the top banner of the Web Application on your laptop:
   - **Laptop Wi-Fi IP**: (e.g. `192.168.1.42`)
   - **Port**: `8765`
   - **Pairing Token**: (e.g. `e9a1`)
3. Open the **Sensor & SLAM Studio** app on your phone.
4. Enter the **Laptop Wi-Fi IP**, **Port**, and **Token**.
5. Tap **START STREAM & LOG**:
   - Hold the phone still for 2 seconds (the estimator uses this stationary window to calibrate initial gravity and gyro bias).
   - Walk through your room past your printed ArUco markers.
6. Tap **STOP** when finished.
7. Return to the web app: click **↻ Refresh** in the session bar. Your new session appears in the dropdown!
8. Click **⚡ Run MSCKF SLAM** to compute your 3D trajectory and room map!

---

## Features & Tab Guide

### Tab 1: 🗺️ 3D SLAM & Map Visualizer
- **Interactive 3D Viewport**: Built with Three.js (WebGL). Orbit, rotate, pan, and zoom around your 3D room.
- **Trace Overlays**:
  - **Cyan Solid Line**: MSCKF SLAM Trajectory with camera heading frustums.
  - **Amber Dashed Line**: Pure VIO Baseline (shows accumulated dead-reckoning drift).
  - **Green Line**: Ground Truth reference.
  - **ArUco Markers**: 3D markers with IDs (`#0`, `#1`, etc.) placed at their exact positions and wall orientations.
- **Camera Presets**: Click `3D Iso`, `Top X-Y`, `Side X-Z`, or `Reset View`.

### Tab 2: 📊 Baseline vs SLAM Scorecard
- Side-by-side comparative table satisfying Section 8 & 9 of the course syllabus:
  - **Translational RMSE (ATE)**
  - **Median Position Error**
  - **Endpoint Drift ($e_p$)**
  - **Drift per Metre Travelled (%)**
  - **Relative Pose Error (1s RPE)**
  - **Orientation RMSE (degrees)**
  - **Normalized Estimation Error Squared (NEES)**
- Click **📉 Run Pure VIO Baseline** to run the baseline without ArUco markers on any recording.

### Tab 3: 🧪 Course Benchmark Experiments
- Pre-configured execution of all 5 mandatory course experiments from Section 8:
  1. **Scenario 1**: Simple L-Shaped Path past 4 markers.
  2. **Scenario 2**: Marker-Loss Path (2.0s blackout segment testing VIO drift and recovery).
  3. **Scenario 3**: Closed-Loop Path returning to origin with loop closure.
  4. **Scenario 4**: Repeat Run in the same room using pre-built marker map.
  5. **Scenario 5**: Difficult aggressive motion with high sensor noise.
- Click **▶ Run Scenario** or **🚀 Run All 5 Experiments** to execute in the background.

### Tab 4: 🎬 Video & Landmark Inspector
- Scrub through all recorded camera preview frames.
- Shows detected ArUco marker IDs and corner coordinates in real time.
- Playback controls: Play/Pause, speed adjustment ($0.5\times, 1.0\times, 2.0\times$).

### Tab 5: 📈 Phone Sensors Analytics
- Select any of the 20+ sensors exposed by your OnePlus hardware (`Accelerometer`, `Gyroscope`, `Magnetometer`, `Gravity`, `Rotation Vector`).
- Real-time multi-dimensional time series canvas plots.
- Click **📥 Download Sensor CSV** to export raw sensor readings for MATLAB/Python.

### Tab 6: 📡 Live Wi-Fi Stream & Pairing
- Live streaming console with real-time frame previews.
- Network log console with timestamps and packet gap tracking.
- If Wi-Fi is unavailable during testing, retrieve `capture.jsonl` from phone via USB cable and drag-and-drop into this tab.

### Tab 7: 🖨️ ArUco Marker Print Studio
- Interactive generator for standard `DICT_6X6_250` ArUco markers (IDs 0 to 7).
- Click **Download SVG** or **🖨️ Print Marker Sheet** to print markers on A4 paper at 100% scale (square size: 120 mm). Place markers on walls around your room.

---

## Running Automated Tests & Demos

To verify the complete test suite:
```powershell
$env:PYTHONPATH="."
python -m unittest discover -s tests -v
```
All 21 unit tests verify MSCKF propagation, camera clone Jacobians, left-nullspace projection, ArUco corner gating, and pose graph optimization.

