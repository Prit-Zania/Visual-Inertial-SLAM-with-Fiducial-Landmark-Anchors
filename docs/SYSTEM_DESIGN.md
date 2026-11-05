# IE415: Visual–Inertial SLAM with Fiducial Landmark Anchors
## System Design Document (Deliverable 2)

**Course**: IE415: Control of Autonomous Systems  
**Project**: Visual–Inertial SLAM with Fiducial Landmark Anchors  
**Selected Architecture**: Architecture C: Phone–Laptop Hybrid System  
**Hardware Platforms**: OnePlus Nord CE4 / OnePlus 12R (Mobile Sensing) + Laptop/Desktop Host (SLAM Engine & Web Dashboard)  

---

## 1. Problem Formulation & System Architecture

### 1.1 Overview
The objective is to design, implement, and quantitatively evaluate a complete visual-inertial Simultaneous Localization and Mapping (SLAM) system anchored by fixed ArUco fiducial landmarks in an indoor room environment. The system estimates the continuous 6-DoF trajectory of a mobile platform while simultaneously mapping the 3D poses of ArUco markers. Repeated marker observations are utilized for drift correction, relocalization, and loop closure.

### 1.2 System Decomposition
The system is partitioned into two cooperating subsystems:
1. **Mobile Sensing Subsystem (Android Application)**:
   - Native lightweight Android application (`OnePlusNordCE4-Sensor-Capture-debug.apk`) targeting Android 14/15 (API 34/35).
   - Acquires synchronized high-rate IMU streams (`Sensor.TYPE_ACCELEROMETER`, `Sensor.TYPE_GYROSCOPE`, `Sensor.TYPE_MAGNETIC_FIELD`, `Sensor.TYPE_ROTATION_VECTOR`) at 100–200 Hz.
   - Captures camera preview frames (NV21 / JPEG) via Android Camera2 / Camera API at 10–30 Hz with hardware timestamps (`elapsedRealtimeNanos`).
   - Streams synchronized packets over a TCP socket via local Wi-Fi to the laptop receiver, and simultaneously logs locally to `capture.jsonl` on flash memory to prevent data loss.

2. **Laptop Processing & Web Application Subsystem**:
   - **TCP Stream Receiver** (`slam/receiver.py`): Listens on port `8765`, authenticates incoming connections using HMAC pairing tokens, deserializes sensor records and JPEG frames, detects visible ArUco markers in real time, and sends diagnostic feedback back to the phone.
   - **Visual-Inertial Estimator** (`slam/msckf.py`, `slam/pipeline.py`): 15-state Error-State Multi-State Constraint Kalman Filter (MSCKF) with camera pose cloning, KLT natural visual feature tracking with nullspace projection, and coupled ArUco corner reprojection updates.
   - **Independent Map Refiner & Graph Optimizer** (`slam/graph.py`): Fixed-window MAP smoother with IMU preintegration and global pose-graph optimization with verified geometric loop closures.
   - **Unified Web Application & 3D Visualizer** (`slam/sensor_api.py`, `slam/sensor_web.html`): HTTP server on port `8080` providing interactive 3D WebGL (Three.js) trajectory and room landmark visualization, baseline comparison scorecards, live telemetry, sensor diagnostics, and ArUco marker printing.

```
+--------------------------------------------------------------------------------------+
|                               PHONE SENSING APPLICATION                             |
|  Camera (10-30Hz)   Accelerometer (100Hz)   Gyroscope (100Hz)   Compass (50Hz)       |
|        |                     |                     |                  |              |
|        +---------------------+---------------------+------------------+              |
|                              | Timestamping & JSONL Serialization                    |
|                              v                                                       |
|                     Local File (capture.jsonl)  <--->  Wi-Fi TCP Stream              |
+--------------------------------------------------------------------------------------+
                                                               | Wi-Fi (Port 8765)
                                                               v
+--------------------------------------------------------------------------------------+
|                         LAPTOP BACKEND & SLAM ENGINE                                 |
|  [Receiver] -> Real-time ArUco Detector -> Live Telemetry -> Storage                 |
|       |                                                                              |
|  [MSCKF Estimator]                                                                   |
|   - Inertial Propagation: 15-state error state (orientation, position, velocity, bg, ba)|
|   - Visual Cloning & KLT feature tracking with left-nullspace projection             |
|   - ArUco Landmark EKF Update & Correlated Anchor States                             |
|       |                                                                              |
|  [Smoother & Pose Graph Optimizer]                                                   |
|   - IMU Preintegration & Windowed MAP optimization                                   |
|   - Loop Closure verification via PnP + RANSAC                                       |
+--------------------------------------------------------------------------------------+
                                                               | REST / WebSockets (Port 8080)
                                                               v
+--------------------------------------------------------------------------------------+
|                       INTERACTIVE WEB APPLICATION & DASHBOARD                        |
|  - 3D WebGL (Three.js) Room Trajectory, Frustums, Covariance & Landmark Visualization|
|  - Quantitative Scorecard: SLAM vs Baseline (ATE, RPE, Endpoint Drift, NEES)         |
|  - 5 Mandatory Benchmark Scenarios Execution & Verification Studio                   |
|  - Live Phone Video Stream & Multi-axis Sensor Telemetry Inspector                   |
+--------------------------------------------------------------------------------------+
```

---

## 2. Coordinate Frames & Rigid-Body Transformations

The system defines four primary coordinate frames adhering to standard robotics conventions:

1. **World Frame $\{W\}$**: A fixed inertial reference frame.
   - If surveyed marker locations are known, $\{W\}$ coincides with the surveyed room coordinate frame ($Z$ upwards, $X$ and $Y$ horizontal along the room walls).
   - If marker poses are initially unknown, the center of the first observed ArUco marker $M_0$ defines the origin of the world frame, with its normal vector determining orientation.
2. **Body / IMU Frame $\{B\}$**: Attached to the physical IMU sensor on the smartphone.
   - Defined by Android sensor coordinate conventions: $X$ points right, $Y$ points up along the phone face, and $Z$ points out towards the user.
3. **Camera Optical Frame $\{C\}$**: Attached to the rear camera sensor.
   - Standard computer vision frame: $X$ points right, $Y$ points down, and $Z$ points forward along the optical optical axis.
4. **Marker Frame $\{M_i\}$**: Rigidly attached to the $i$-th ArUco marker.
   - Origin at the geometric center of the black square.
   - $X$ points right, $Y$ points up in the marker plane, and $Z$ points outwards along the normal vector perpendicular to the wall.

### Rigid-Body Transformations
All spatial transformations are represented as elements of the Special Euclidean group $\mathrm{SE}(3)$:
$$T_{A}^B = \begin{bmatrix} R_{A}^B & p_{A}^B \\ 0_{1\times 3} & 1 \end{bmatrix} \in \mathrm{SE}(3)$$
which transforms a point expressed in frame $B$ to frame $A$: $p^A = R_{A}^B p^B + p_{A}^B$.

- **Camera-to-Body Extrinsics**: $T_{B}^C = \begin{bmatrix} R_{BC} & p_{BC} \\ 0 & 1 \end{bmatrix}$ maps camera coordinates to IMU body coordinates.
- **Camera Pose in World**: $T_{W}^{C_k} = T_{W}^{B_k} T_{B}^C$.
- **Camera-to-Marker Detection**: Perspective-n-Point (PnP) or IPPE on detected marker corners provides the relative pose $T_{C_k}^{M_i}$.
- **Known Anchor Constraint**: If the global marker pose $T_W^{M_i}$ is surveyed, the camera pose is constrained by:
  $$T_W^{C_k} = T_W^{M_i} (T_{C_k}^{M_i})^{-1}$$
- **Unknown Landmark Mapping**: When a new marker $M_j$ is first observed at camera pose $T_W^{C_k}$:
  $$T_W^{M_j} = T_W^{C_k} T_{C_k}^{M_j}$$

---

## 3. Sensor Acquisition & Clock Synchronization

### 3.1 Hardware Timestamping
Accurate visual-inertial odometry requires strict sub-millisecond timestamp alignment. Smartphone cameras and IMUs operate on distinct internal hardware clocks:
- Android Camera2 preview frames report `CaptureResult.SENSOR_TIMESTAMP` in nanoseconds of `SystemClock.elapsedRealtimeNanos()`.
- Android `SensorEvent.timestamp` reports hardware interrupt timestamps in the identical `elapsedRealtimeNanos()` clock domain.
- The phone application preserves native nanosecond hardware timestamps across all records (`timestamp_ns`), rejecting laptop network arrival times which introduce non-deterministic Wi-Fi jitter.

### 3.2 Time Synchronization & Interpolation
Let $t_k$ denote the camera acquisition time adjusted by the calibrated camera-IMU clock offset $\delta t$:
$$t_k = t_{\text{cam}} + \delta t$$
During estimation, IMU samples bracket each camera frame $t_k$. High-rate IMU measurements are integrated up to exact camera frame timestamps using zero-order hold acceleration and spherical linear interpolation (Slerp) / midpoint rotation integration.

---

## 4. Calibration Procedures

### 4.1 Camera Intrinsic Calibration
The camera is modeled using the standard pinhole projection model with radial-tangential distortion:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad D = [k_1, k_2, p_1, p_2]$$
- **Calibration Procedure**: A planar chessboard pattern ($9 \times 6$ internal corners, square size $25.0\text{ mm}$) is recorded from 15–20 diverse viewpoints covering varying depths, inclinations, and frame peripheries.
- **Algorithm**: OpenCV `cv2.calibrateCamera` detects saddle points with subpixel refinement (`cv2.cornerSubPix`) and optimizes intrinsics via Levenberg-Marquardt to achieve reprojection error $< 0.35\text{ px}$.

### 4.2 Camera–IMU Extrinsic Calibration
- **Rotation $R_{BC}$**: Estimated by recording simultaneous angular velocities $\omega_B(t)$ from the gyroscope and visual camera rotations $R_{C}(t)$ obtained from PnP tracking of a stationary fiducial marker while rotating the phone about all three principal axes. The continuous Hand-Eye constraint:
  $$R_{BC} \omega_C(t) = \omega_B(t) - b_g$$
  is solved via nonlinear least squares.
- **Translation $p_{BC}$**: Measured using precise physical device CAD blueprints (for OnePlus Nord CE4: camera module is offset by $+18\text{ mm}$ in $X$, $+62\text{ mm}$ in $Y$, and $-4\text{ mm}$ in $Z$ relative to the internal IMU package).

---

## 5. State Estimation Methodology

### 5.1 State Vector Representation
The true visual-inertial state at time step $k$ consists of the active navigation state, $N$ cloned camera poses, and $M$ mapped ArUco landmarks:
$$x_k = \begin{bmatrix} x_{\text{nav},k}^T & x_{C_1}^T & \dots & x_{C_N}^T & x_{M_1}^T & \dots & x_{M_M}^T \end{bmatrix}^T$$
where the 15-dimensional core navigation state is:
$$x_{\text{nav},k} = \begin{bmatrix} q_{WB}^T & p_W^T & v_W^T & b_g^T & b_a^T \end{bmatrix}^T \in \mathbb{R}^{16}$$
with orientation unit quaternion $q_{WB} \in \mathbb{H}$, position $p_W \in \mathbb{R}^3$, velocity $v_W \in \mathbb{R}^3$, gyroscope bias $b_g \in \mathbb{R}^3$, and accelerometer bias $b_a \in \mathbb{R}^3$.

The 15-dimensional error state is parameterized on the tangent space:
$$\delta x_k = \begin{bmatrix} \delta \theta_B^T & \delta p_W^T & \delta v_W^T & \delta b_g^T & \delta b_a^T \end{bmatrix}^T \in \mathbb{R}^{15}$$
where orientation error is defined via right multiplicative perturbation:
$$R_{WB} = \hat{R}_{WB} \mathrm{Exp}(\delta \theta_B)$$

### 5.2 Inertial Propagation
Given measured specific force $\tilde{a}_B$ and angular rate $\tilde{\omega}_B$:
$$\hat{a}_B = \tilde{a}_B - \hat{b}_a, \quad \hat{\omega}_B = \tilde{\omega}_B - \hat{b}_g$$
The continuous-time kinematics equations are:
$$\dot{R}_{WB} = R_{WB} [\hat{\omega}_B]_\times$$
$$\dot{p}_W = v_W$$
$$\dot{v}_W = R_{WB} \hat{a}_B + g_W$$
$$\dot{b}_g = \eta_{wg}, \quad \dot{b}_a = \eta_{wa}$$
where $g_W = [0, 0, -9.80665]^T\text{ m/s}^2$ is the world gravity vector, and $\eta_{wg}, \eta_{wa}$ are Gaussian bias random walk processes.

The error state covariance $P_{k|k}$ is propagated via:
$$P_{k+1|k} = \Phi_k P_{k|k} \Phi_k^T + Q_k$$
where $\Phi_k$ is the discrete-time state transition matrix derived from the Jacobian of the continuous-time dynamics, and $Q_k$ is the discrete noise covariance.

### 5.3 Camera Pose Cloning
At each camera frame arrival, the current camera pose is cloned:
$$\hat{x}_{C_k} = \begin{bmatrix} \hat{R}_{WC_k} \\ \hat{p}_{WC_k} \end{bmatrix} = \begin{bmatrix} \hat{R}_{WB_k} R_{BC} \\ \hat{p}_{WB_k} + \hat{R}_{WB_k} p_{BC} \end{bmatrix}$$
The error state and covariance are augmented with the cloned camera pose:
$$P \leftarrow \begin{bmatrix} I \\ J_{\text{clone}} \end{bmatrix} P \begin{bmatrix} I & J_{\text{clone}}^T \end{bmatrix}$$
preserving all cross-covariances between camera clones, navigation states, and landmarks.

### 5.4 Natural Feature Updates via Left-Nullspace Projection (MSCKF)
Natural visual features are detected with Shi-Tomasi corners and tracked across cloned camera frames using pyramidal Lucas-Kanade optical flow (KLT). When a feature track terminates or leaves the cloning window:
1. The 3D feature position $p_f^W$ is triangulated using linear least squares over all observing camera clones.
2. The measurement residual across $M_f$ camera views is linearized as:
   $$r_f = H_{x,f} \delta x + H_{f} \delta p_f^W + n_f$$
3. To avoid maintaining 3D landmark points in the Kalman state vector, we compute the QR decomposition of the feature Jacobian:
   $$H_f = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$
   Multiplying by $Q_2^T$ projects the residual onto the left nullspace of $H_f$ ($Q_2^T H_f = 0$):
   $$r_0 = Q_2^T r_f = Q_2^T H_{x,f} \delta x + Q_2^T n_f = H_0 \delta x + n_0$$
4. The projected residual $r_0$ is gated using a Mahalanobis $\chi^2$ test:
   $$\gamma = r_0^T (H_0 P H_0^T + R_0)^{-1} r_0 \le \chi^2_{0.95}(\mathrm{dim}(r_0))$$
5. Standard EKF Kalman update is performed using the Joseph-form stabilized covariance update equation.

### 5.5 ArUco Landmark Anchoring & EKF Updates
Each observed ArUco marker provides 4 subpixel corner coordinates:
$$z_{ij} = h(T_{W}^{C_k}, T_W^{M_i}, c_j) + v_{ij}, \quad j \in \{1, 2, 3, 4\}$$
where $c_j$ are the metric 3D corner coordinates of the marker in its local frame: $[\pm \frac{s}{2}, \pm \frac{s}{2}, 0]^T$.
- **Known Anchor Markers**: If marker $M_i$ has a surveyed global pose $T_W^{M_i}$, its pose is fixed. The reprojection error directly updates the current camera pose and navigation state, eliminating drift without modifying marker covariance.
- **Unknown Landmarks**: When an unmapped marker is detected, its initial pose is estimated via IPPE/PnP:
  $$\hat{T}_W^{M_i} = \hat{T}_W^{C_k} T_{C_k}^{M_i}$$
  The state vector and covariance are augmented with the 6-DoF marker pose, with cross-covariances linking it to the camera clone. Subsequent re-observations of this marker refine both the marker location and the camera trajectory.

### 5.6 Independent Map Refinement & Pose Graph Optimization
A fixed-window maximum a posteriori (MAP) smoother operates over keyframe navigation states, preintegrated IMU factors, visual feature reprojections, and ArUco corner residuals:
$$\min_{\{X_k, L_i\}} \sum_{k} \| r_{\text{IMU}}(X_k, X_{k+1}) \|^2_{\Sigma_k} + \sum_{(k, i) \in \mathcal{O}} \| r_{\text{marker}}(X_k, L_i) \|^2_{\Lambda_{ki}} + \sum_{(k, j) \in \mathcal{F}} \| r_{\text{feat}}(X_k, f_j) \|^2_{\Omega_{kj}}$$
Furthermore, when an ArUco marker is re-observed after an extended trajectory loop, a loop-closure constraint edge is added to the global pose graph, and non-linear least squares optimization (Levenberg-Marquardt) distributes the closure residual across the trajectory loop.

---

## 6. Baseline Architecture for Comparative Evaluation

Per Section 6 (Requirement 11) and Section 8 of the course syllabus, the system incorporates a **Pure Visual-Inertial Odometry Baseline**:
- The baseline runs the identical 15-state MSCKF propagation and KLT natural feature updates, but **strictly disables all ArUco marker detections, anchors, and landmark constraints**.
- This baseline isolates the contribution of fiducial landmark anchors, providing a rigorous quantitative metric benchmark demonstrating how fiducial constraints eliminate cubic drift, lock metric scale, and provide long-term bounded error.

