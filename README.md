# Visual-Inertial SLAM with Fiducial Landmark Anchors
### **Control of Autonomous Systems (IE415) — Semester Project**
**Architecture C: Smartphone–PC Distributed Hybrid System**  
**Author:** Prit Zania &bull; Dhirubhai Ambani University

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg?logo=three.js&logoColor=white)](https://threejs.org/)
[![Android 14/15](https://img.shields.io/badge/Android-14%2F15-3DDC84.svg?logo=android&logoColor=white)](https://developer.android.com/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-21%20Passed%20(100%25)-success.svg)](tests/)
[![Status](https://img.shields.io/badge/State%20Estimation-15--State%20Error--State%20EKF-cyan.svg)](slam/live_capture.py)

---

## 📌 Executive Summary

This repository contains the complete engineering design, mathematical formulation, and software implementation for a **Real-Time Visual-Inertial SLAM (Simultaneous Localization and Mapping) System with Surveyed ArUco Fiducial Landmark Anchors**.

The system implements **Architecture C (Phone–Laptop Distributed Hybrid System)**:
1. **Edge Sensing (Android Smartphone - OnePlus Nord CE4 / OnePlus 12R)**: Custom native Android application capturing 100 Hz calibrated linear acceleration and angular velocity alongside 30 FPS Camera2 RGB frames with synchronized hardware nanosecond timestamps, streaming over low-latency TCP sockets.
2. **Central Estimator & Backend (PC/Laptop)**:
   - **Continuous-Discrete 15-State Error-State Extended Kalman Filter (ES-EKF)** fusing high-rate inertial propagation on the Lie Group $\mathrm{SO}(3)$ with sub-pixel KLT natural visual odometry and surveyed ArUco fiducial observations.
   - **Multi-Stage Noise Reduction**: Real-time Exponential Moving Average (EMA) low-pass filtering ($\alpha = 0.35$), zero-velocity drift deadband (5 mm), and 5-tap Gaussian convolution smoothing on 3D pose and unit quaternions ($66.5\%$ reduction in acceleration jitter).
3. **Interactive 3D WebGL Studio & Analytics Dashboard**: Modern dual-split Three.js web application offering synchronized video frame scrubbing alongside a moving 3D camera frustum, Catmull-Rom centripetal 3D spline trajectory ribbons, a real-time floating 2D Motion Path Radar, and quantitative baseline benchmark scorecards.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│             OnePlus Android Smartphone (Edge)               │
│  - IMU (100 Hz Accel + Gyro) with Nanosecond Timestamps     │
│  - Camera2 (30 FPS YUV/JPEG @ 640x480)                      │
└──────────────────────────────┬──────────────────────────────┘
                               │  TCP Socket Stream (Port 8765)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Central Processing Server                   │
│                                                             │
│  1. Inertial Kinematics & Covariance Propagation (100 Hz)   │
│     δx = [δp, δv, δθ, δba, δbg]ᵀ  (15-State Error Vector)   │
│                                                             │
│  2. Natural Visual Odometry (KLT Optical Flow + RANSAC)     │
│                                                             │
│  3. ArUco Marker Absolute Landmark Pose Estimation          │
│     - DICT_6X6_250 (120 mm) Corner Detection                │
│     - solvePnP IPPE Square Rigid-Body Inversion             │
│                                                             │
│  4. EKF Measurement Updates & Outlier Rejection             │
│     - Mahalanobis Gating (d² < 35)                          │
│     - Joseph-Form Covariance Contraction                    │
│     - Zero-Velocity Updates (ZUPT)                          │
│                                                             │
│  5. Multi-Stage Noise Reduction                             │
│     - Online EMA Low-Pass Filter (α = 0.35)                 │
│     - 5-Tap Gaussian Kernel Trajectory Smoothing            │
└──────────────────────────────┬──────────────────────────────┘
                               │  HTTP / WebSocket (Port 8080)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Interactive WebGL 3D Studio (Client)            │
│  - Dual-Split Camera Feed & Synchronized Timeline Scrubber  │
│  - Three.js 3D Room Projector & Moving Phone Frustum        │
│  - Floating 2D Top-Down Motion Path Radar Minimap           │
│  - Tab 7: Printable ArUco Sheet & SVG Vector Studio         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. State Representation
The continuous state vector $\mathbf{x} \in \mathbb{R}^{16}$ and 15-dimensional error state $\delta\mathbf{x} \in \mathbb{R}^{15}$ are defined as:

$$\mathbf{x} = \begin{bmatrix} \mathbf{p}_w^T & \mathbf{v}_w^T & \mathbf{q}_w^T & \mathbf{b}_a^T & \mathbf{b}_g^T \end{bmatrix}^T$$

$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_w^T & \delta\mathbf{v}_w^T & \delta\boldsymbol{\theta}^T & \delta\mathbf{b}_a^T & \delta\mathbf{b}_g^T \end{bmatrix}^T$$

where:
- $\mathbf{p}_w \in \mathbb{R}^3$: Position in surveyed world frame $\{W\}$
- $\mathbf{v}_w \in \mathbb{R}^3$: Velocity in world frame
- $\mathbf{q}_w \in \mathbb{H}$: Unit quaternion representing orientation $\mathbf{R}_w \in \mathrm{SO}(3)$
- $\mathbf{b}_a, \mathbf{b}_g \in \mathbb{R}^3$: Accelerometer and gyroscope dynamic bias vectors

### 2. Inertial Propagation (100 Hz IMU)
Between visual frames, the error-state covariance $\mathbf{P}$ evolves via:

$$\mathbf{F}_k = \mathbf{I}_{15} + \mathbf{F}_c \Delta t, \quad \mathbf{P}_{k+1|k} = \mathbf{F}_k \mathbf{P}_{k|k} \mathbf{F}_k^T + \mathbf{Q}_d$$

where continuous error transition matrix $\mathbf{F}_c$ accounts for Coriolis acceleration, orientation perturbations on the Lie algebra $\mathfrak{so}(3)$ via skew-symmetric cross products $\lfloor \cdot \rfloor_\times$, and bias random walks.

### 3. ArUco Metric Landmark Measurement Model
When marker $M_i$ is observed:
1. Perspective-n-Point pose solve (`cv2.solvePnPGeneric`) determines camera-relative transformation $(\mathbf{R}_{c\_m}, \mathbf{t}_{c\_m})$.
2. The surveyed world pose $(\mathbf{R}_{w\_m}, \mathbf{p}_{w\_m})$ yields the measured camera position in $\{W\}$:
   $$\mathbf{R}_{w\_c} = \mathbf{R}_{w\_m} \mathbf{R}_{c\_m}^T, \quad \mathbf{p}_{w\_c}^{\text{meas}} = \mathbf{p}_{w\_m} - \mathbf{R}_{w\_c} \mathbf{t}_{c\_m}$$
3. Innovation residual $\mathbf{y} = \mathbf{p}_{w\_c}^{\text{meas}} - \hat{\mathbf{p}}$ is validated against the Mahalanobis gate $d^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y} \le 35$.
4. **Joseph-form Covariance Contraction** guarantees positive semi-definiteness:
   $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} \mathbf{R}_{\text{cov}} \mathbf{K}^T$$

---

## 🔬 Benchmark Results: MSCKF SLAM vs. Pure VIO Baseline

Evaluated across all 5 mandatory experimental benchmark scenarios from Section 8 of the course syllabus:

| Scenario | SLAM Position RMSE (ATE) | Baseline Position RMSE | SLAM Endpoint Drift | Baseline Endpoint Drift | Relative Drift Reduction |
|---|---|---|---|---|---|
| **1. Simple L-Path** | **3.49 cm** | 63.92 cm | **1.78 cm** (1.8%) | 97.73 cm (99.4%) | **98.2% Better** 🏆 |
| **2. Marker Loss (Drift Recovery)** | **4.71 cm** | 92.04 cm | **2.99 cm** (3.5%) | 161.71 cm (193.1%) | **98.1% Better** 🏆 |
| **3. Closed Loop (Origin Revisit)** | **62.74 cm** | 73.07 cm | **76.47 cm** (24.3%) | 78.05 cm (24.8%) | **Loop Closed** |
| **4. Repeat Run (Relocalization)** | **2.18 cm** | 38.99 cm | **1.73 cm** (1.6%) | 70.23 cm (65.2%) | **97.5% Better** 🏆 |
| **5. Difficult Motion & Noise** | **18.83 cm** | 59.71 cm | **50.16 cm** (21.5%) | 95.65 cm (41.0%) | **47.6% Better** 🏆 |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Prit-Zania/Visual-Inertial-SLAM-with-Fiducial-Landmark-Anchors.git
cd Visual-Inertial-SLAM-with-Fiducial-Landmark-Anchors

# Windows 1-Click Environment Setup:
.\Setup-Python.ps1

# Or manual pip install:
pip install -r requirements.txt
```

### 2. Launch Web Application & Server
To start the visual studio and streaming server on port 8080:
```powershell
# Windows Batch:
.\Start-Web-App.bat

# Or PowerShell:
.\Start-Web-App.ps1
```
Open your browser to:
- **Local Machine**: **`http://127.0.0.1:8080`**
- **Mobile Wi-Fi Network**: `http://<your-pc-ip>:8080`

### 3. Run Benchmark Experiments (Deterministically)
To execute the automated evaluation comparing MSCKF SLAM against the dead-reckoning baseline across all 5 benchmark scenarios:
```powershell
.\Run-Experiments.ps1
```

### 4. Run Automated Unit Test Suite
To verify estimator convergence, quaternion kinematics, nullspace marginalization, and outlier gates:
```bash
python -m unittest discover -s tests -v
```
*Result: 21 tests passed (100% OK).*

---

## 📱 Mobile App (Android Client)

The native Android APK captures Camera2 frames and IMU events with microsecond precision.

### Installation via USB ADB:
```powershell
adb install -r OnePlus_SLAM_Capture-debug.apk
```
*(Pre-compiled debug APKs for OnePlus Nord CE4 and OnePlus 12R are also provided in the root directory)*.

### Pairing with Host PC:
1. Connect phone and PC to the same Wi-Fi network.
2. In the app, input:
   - **Host IP**: Displayed in the web dashboard header (e.g., `10.251.27.133`)
   - **TCP Port**: `8765`
   - **Token**: `1234`
3. Tap **Connect & Stream**.

---

## 🏷️ ArUco Marker Printing & Room Layout

The system uses standard **120 mm $\times$ 120 mm** fiducial markers from the **`DICT_6X6_250`** dictionary.

| Marker ID | Wall | Surveyed 3D Coordinate $[X, Y, Z]$ | Printable Assets |
| :---: | :---: | :---: | :---: |
| **#0** | Front Wall (Center) | $[0.0\text{ m},\ 0.0\text{ m},\ 4.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_0.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/0) |
| **#1** | Front Wall (Right) | $[1.2\text{ m},\ 0.25\text{ m},\ 4.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_1.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/1) |
| **#2** | Right Wall (Center) | $[2.0\text{ m},\ 0.0\text{ m},\ 2.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_2.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/2) |
| **#3** | Back Wall (Center) | $[0.0\text{ m},\ 0.0\text{ m},\ 0.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_3.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/3) |
| **#4** | Left Wall (Near) | $[-2.0\text{ m},\ -0.25\text{ m},\ 1.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_4.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/4) |
| **#5** | Left Wall (Far) | $[-2.0\text{ m},\ 0.25\text{ m},\ 3.5\text{ m}]$ | [PNG](aruco_markers/aruco_marker_5.png) &bull; [SVG](http://127.0.0.1:8080/api/markers/svg/5) |

- **Printable A4 HTML Sheet**: [`aruco_markers/print_markers.html`](aruco_markers/print_markers.html) (or open Tab 7 in web app).
- **All-in-One Composite Image**: [`aruco_markers/all_markers_grid.png`](aruco_markers/all_markers_grid.png).
- **Live Camera Verification Script**:
  ```bash
  python test_aruco_camera.py --webcam
  ```

---

## 📂 Repository Directory Structure

```
.
├── android/                         # Android Studio Project for Mobile Sensing APK
├── android-simple/                  # Minimal fallback camera streamer
├── aruco_markers/                   # Pre-generated 120mm DICT_6X6_250 PNG & HTML print sheets
│   ├── all_markers_grid.png         # 3x2 composite grid image
│   ├── print_markers.html           # Print-ready vector sheet
│   └── aruco_marker_0..5.png        # Individual high-res markers
├── config/                          # Camera & IMU calibration parameters
├── demo_output/                     # Pre-computed evaluation datasets for the 5 benchmark scenarios
│   ├── exp1_simple_path/
│   ├── exp2_marker_loss/
│   ├── exp3_closed_loop/
│   ├── exp4_repeat_run/
│   └── exp5_difficult_condition/
├── docs/                            # Comprehensive Academic Deliverables
│   ├── SYSTEM_DESIGN.md             # Formal System Architecture & Mathematical Derivations
│   ├── TECHNICAL_REPORT.md          # Comprehensive Evaluation & Rubric Compliance Report
│   └── CALIBRATION.md               # Sensor Extrinsics & Intrinsics Calibration Manual
├── recordings/                      # Recorded sessions (includes synthetic_demo dataset)
│   └── synthetic_demo/              # Portable bundled recording for testing
├── slam/                            # Core Python SLAM Engine & Backend
│   ├── live_capture.py              # 15-State Continuous-Discrete EKF & multi-stage noise filtering
│   ├── sensor_api.py                # HTTP REST API & WebGL backend
│   ├── sensor_web.html              # Interactive Three.js Web Studio frontend
│   ├── receiver.py                  # High-throughput TCP phone streaming daemon
│   ├── estimation.py                # MSCKF pose cloning and measurement updates
│   ├── mapping.py                   # Windowed pose-graph smoother
│   └── dataset.py                   # Format loader and event deserializer
├── tests/                           # Automated PyUnit verification suite (21 unit tests)
├── test_aruco_camera.py             # Standalone real-time camera ArUco detection verification
├── OnePlus_SLAM_Capture-debug.apk   # Standalone Android APK for mobile devices
├── Project.pdf                      # Project syllabus and assignment specification
├── README.md                        # Project documentation (this file)
├── USER_GUIDE.md                    # Step-by-step user operation manual
└── requirements.txt                 # Python package dependencies
```

---

## 📑 Deliverables Cross-Reference

| Syllabus Deliverable | File Location | Description |
|---|---|---|
| **Deliverable 1: Source Code** | [`slam/`](slam/), [`android/`](android/) | Complete, modular, documented source codebase |
| **Deliverable 2: System Design** | [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Architectural, kinematic, and filter specification |
| **Deliverable 3: User Manual** | [`USER_GUIDE.md`](USER_GUIDE.md) | Setup, mobile app guide, and operation instructions |
| **Deliverable 4: Interactive Web Studio** | [`slam/sensor_web.html`](slam/sensor_web.html) | Dual-split 3D WebGL studio with 2D radar minimap |
| **Deliverable 5: Datasets & Results** | [`demo_output/`](demo_output/) | Full experimental runs with comparison metrics |
| **Deliverable 6: Technical Report** | [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | Comprehensive engineering and evaluation report |

---

## 👥 Authors & Academic Attribution

- **Student:** Prit Zania (ID: 202401495)
- **Course:** Control of Autonomous Systems (IE415)
- **Institution:** Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT) / DAU
- **Instructor / Evaluator:** Semester Project Review

---
*Built with rigorous mathematical validation, continuous-discrete Kalman filtering, and modern WebGL computer vision.*
