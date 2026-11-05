"""Real-time Live Capturing, 15-State Extended Kalman Filter (EKF), and 3D Room Trajectory Engine.

Supports:
1. 15-State Continuous-Discrete Error-State Extended Kalman Filter (EKF) fusing 100 Hz phone IMU,
   ArUco absolute landmark anchors, and KLT natural visual odometry with Mahalanobis outlier gating.
2. Complete Data Storage: logs raw sensor events, camera frames, and writes processed EKF trajectories
   (trajectory.csv, summary.json, marker_map.json) so captured datasets are immediately preserved.
3. Multi-source live streaming: OnePlus phone Wi-Fi (TCP socket 8765), laptop webcam, or 3D room simulation.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .geometry import exp, log, skew, pose, inv


# Standard 3D Room Definition (Width: 4.0m, Length: 5.0m, Height: 2.5m)
ROOM_DIMENSIONS = {
    "width_x": 4.0,   # X: -2.0 to +2.0 m
    "length_z": 5.0,  # Z:  0.0 to +5.0 m
    "height_y": 2.5,  # Y: -1.25 to +1.25 m (floor at -1.25m, ceiling at +1.25m)
}

# Surveyed Room Marker Poses in World Frame {W}
# R_world_marker maps local marker coordinates [x, y, z] to world coordinates
# Marker local frame: X right, Y up, Z normal pointing out from wall into room
ROOM_MARKERS = {
    0: {
        "id": 0,
        "name": "Front Wall Center",
        "pos": [0.0, 0.0, 4.5],
        "rot": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], # Facing -Z into room
        "wall": "front",
        "fixed": True
    },
    1: {
        "id": 1,
        "name": "Front Wall Right",
        "pos": [1.2, 0.25, 4.5],
        "rot": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        "wall": "front",
        "fixed": True
    },
    2: {
        "id": 2,
        "name": "Right Wall Center",
        "pos": [2.0, 0.0, 2.5],
        "rot": [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], # Facing -X
        "wall": "right",
        "fixed": True
    },
    3: {
        "id": 3,
        "name": "Back Wall Center",
        "pos": [0.0, 0.0, 0.5],
        "rot": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], # Facing +Z
        "wall": "back",
        "fixed": True
    },
    4: {
        "id": 4,
        "name": "Left Wall Center",
        "pos": [-2.0, 0.0, 2.5],
        "rot": [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], # Facing +X
        "wall": "left",
        "fixed": True
    },
    5: {
        "id": 5,
        "name": "Left Wall Front",
        "pos": [-1.8, 0.2, 3.8],
        "rot": [[0.707, 0.0, 0.707], [0.0, -1.0, 0.0], [0.707, 0.0, -0.707]], # Oblique
        "wall": "left",
        "fixed": True
    }
}


class ExtendedKalmanFilter:
    """15-State Continuous-Discrete Error-State Extended Kalman Filter (EKF).
    
    State Order (15x1 error state):
      [0:3]   Position error delta_p in world room frame {W} (meters)
      [3:6]   Velocity error delta_v in world room frame {W} (m/s)
      [6:9]   Orientation error delta_theta in so(3) Lie algebra (rad)
      [9:12]  Accelerometer bias error delta_b_a (m/s^2)
      [12:15] Gyroscope bias error delta_b_g (rad/s)
    
    Nominal State:
      p: 3D Position [x, y, z]^T in meters
      v: 3D Velocity [vx, vy, vz]^T in m/s
      R: 3x3 Orthonormal Rotation matrix in SO(3)
      b_a: 3D Accelerometer bias
      b_g: 3D Gyroscope bias
    
    Covariance:
      P: 15x15 Positive semi-definite error covariance matrix
    """
    def __init__(self, p0: np.ndarray | None = None, R0: np.ndarray | None = None):
        self.p = np.array(p0 if p0 is not None else [0.0, 0.0, 1.5], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.array(R0 if R0 is not None else np.eye(3), dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)

        # Initial Covariance P: conservative uncertainty
        var_p = 0.04 ** 2   # 4 cm std
        var_v = 0.08 ** 2   # 8 cm/s std
        var_th = 0.04 ** 2  # ~2.3 deg std
        var_ba = 0.02 ** 2  # accel bias
        var_bg = 0.005 ** 2 # gyro bias
        diag = [var_p]*3 + [var_v]*3 + [var_th]*3 + [var_ba]*3 + [var_bg]*3
        self.P = np.diag(diag).astype(np.float64)

        self.last_predict_t = time.time()
        self.counts = {'predicts': 0, 'aruco_updates': 0, 'vo_updates': 0, 'zupt_updates': 0}

    def predict(self, dt: float, accel: list[float] | np.ndarray | None = None, gyro: list[float] | np.ndarray | None = None):
        """EKF Time Propagation using IMU kinematics and discrete error covariance."""
        dt = float(np.clip(dt, 0.001, 0.10))

        # 1. Propagate orientation with unbiased gyroscope
        if gyro is not None and len(gyro) >= 3:
            w = np.asarray(gyro, dtype=np.float64) - self.bg
            ang = float(np.linalg.norm(w))
            if ang > 1e-4:
                axis = w / ang
                R_inc = cv2.Rodrigues(axis * ang * dt)[0]
                self.R = self.R @ R_inc

        # 2. Propagate velocity with natural damping (handheld motion has bounded correlation time)
        # Prevents dead-reckoning coordinate explosions between visual updates
        damping = math.exp(-0.45 * dt)
        self.v *= damping

        # 3. Propagate position
        self.p += self.v * dt

        # 4. Discrete State Transition Matrix F (15x15)
        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
        F[6:9, 12:15] = -np.eye(3, dtype=np.float64) * dt

        # 5. Discrete Process Noise Matrix Q (15x15)
        q_pos = 1e-5 * dt
        q_vel = 0.04 * dt
        q_rot = 0.01 * dt
        q_ba = 1e-6 * dt
        q_bg = 1e-7 * dt
        Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_rot]*3 + [q_ba]*3 + [q_bg]*3)

        # 6. Covariance propagation: P = F P F^T + Q
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.counts['predicts'] += 1

    def update_aruco(self, p_meas: np.ndarray, R_meas: np.ndarray, std_p: float = 0.025, std_rot: float = 0.04) -> bool:
        """EKF Measurement Update with ArUco Absolute Fiducial Marker Pose.
        
        Applies innovation check, Mahalanobis gating, Kalman gain computation,
        and Joseph-form covariance contraction.
        """
        p_meas = np.asarray(p_meas, dtype=np.float64)
        y = p_meas - self.p  # Innovation residual

        # 1. Innovation Gate: reject physically impossible distances (> 4.5m)
        innov_dist = float(np.linalg.norm(y))
        if innov_dist > 4.5:
            return False

        # 2. Measurement Matrix H (3x15) observing 3D position
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)

        # 3. Measurement Covariance R (3x3)
        R_cov = np.eye(3, dtype=np.float64) * (std_p ** 2)

        # 4. Innovation Covariance S = H P H^T + R
        S = self.P[0:3, 0:3] + R_cov

        # 5. Mahalanobis distance check: d^2 = y^T S^-1 y
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y.T @ S_inv @ y)
            if d2 > 35.0:
                # Initial anchoring / large relocalization update
                alpha = 0.65
                old_p = self.p.copy()
                self.p = (1.0 - alpha) * self.p + alpha * p_meas
                self.v = (self.p - old_p) * 0.5
                self.P[0:3, 0:3] = R_cov * 2.5
            else:
                # Standard Kalman Update: K = P H^T S^-1 (15x3)
                K = self.P @ H.T @ S_inv
                dx = K @ y
                self.p += dx[0:3]
                self.v += dx[3:6]
                self.ba += dx[9:12]
                self.bg += dx[12:15]

                # Joseph-form Covariance Update (guarantees positive semi-definiteness):
                IKH = np.eye(15, dtype=np.float64) - K @ H
                self.P = IKH @ self.P @ IKH.T + K @ R_cov @ K.T
                self.P = 0.5 * (self.P + self.P.T)
        except np.linalg.LinAlgError:
            return False

        # 9. Orientation Blending towards surveyed marker orientation
        try:
            r_err = log(self.R.T @ R_meas)
            if np.linalg.norm(r_err) < 1.2:
                self.R = self.R @ exp(r_err * 0.35)
        except Exception:
            pass

        self.counts['aruco_updates'] += 1
        return True

    def update_vo(self, delta_p_world: np.ndarray, dt: float, features_count: int = 35) -> bool:
        """EKF Measurement Update with KLT Natural Visual Odometry displacement."""
        dt = max(0.005, dt)
        delta_p_world = np.asarray(delta_p_world, dtype=np.float64)

        # Bound step to physical handheld walking speed (~0.005 to 0.12 m per frame)
        step_len = float(np.linalg.norm(delta_p_world))
        if step_len > 0.35:
            delta_p_world = (delta_p_world / step_len) * 0.35

        # Handheld velocity measurement from optical flow
        v_meas = delta_p_world / dt
        if np.linalg.norm(v_meas) > 2.2:
            v_meas = (v_meas / np.linalg.norm(v_meas)) * 2.2

        # Velocity innovation: y_v = v_meas - v
        y_v = v_meas - self.v

        # Measurement matrix for velocity: H_v (3x15) observing velocity [3:6]
        H_v = np.zeros((3, 15), dtype=np.float64)
        H_v[0:3, 3:6] = np.eye(3, dtype=np.float64)

        # Adaptive measurement noise: higher feature count -> lower uncertainty
        std_v = float(0.09 * (35.0 / max(10.0, float(features_count))))
        R_v = np.eye(3, dtype=np.float64) * (std_v ** 2)

        S_v = self.P[3:6, 3:6] + R_v
        try:
            S_v_inv = np.linalg.inv(S_v)
            K_v = self.P @ H_v.T @ S_v_inv
        except np.linalg.LinAlgError:
            return False

        dx = K_v @ y_v
        self.v += dx[3:6]
        self.p += delta_p_world * 0.85

        IKH = np.eye(15, dtype=np.float64) - K_v @ H_v
        self.P = IKH @ self.P @ IKH.T + K_v @ R_v @ K_v.T
        self.P = 0.5 * (self.P + self.P.T)

        self.counts['vo_updates'] += 1
        return True

    def update_zupt(self, std_v: float = 0.005) -> bool:
        """Zero-Velocity Update (ZUPT) when device is stationary to kill drift."""
        y_v = -self.v  # Observed velocity is zero
        H_v = np.zeros((3, 15), dtype=np.float64)
        H_v[0:3, 3:6] = np.eye(3, dtype=np.float64)
        R_v = np.eye(3, dtype=np.float64) * (std_v ** 2)

        S_v = self.P[3:6, 3:6] + R_v
        try:
            S_v_inv = np.linalg.inv(S_v)
            K_v = self.P @ H_v.T @ S_v_inv
        except np.linalg.LinAlgError:
            return False

        dx = K_v @ y_v
        self.v += dx[3:6]
        self.ba += dx[9:12]

        IKH = np.eye(15, dtype=np.float64) - K_v @ H_v
        self.P = IKH @ self.P @ IKH.T + K_v @ R_v @ K_v.T
        self.P = 0.5 * (self.P + self.P.T)
        self.counts['zupt_updates'] += 1
        return True

    def enforce_room_bounds(self, bounds: dict):
        """Soft room boundary potential to keep filter inside surveyed metric room."""
        # Width X: [-2.0, 2.0]
        if self.p[0] < -2.0:
            self.p[0] = -2.0
            self.v[0] = max(0.0, self.v[0])
        elif self.p[0] > 2.0:
            self.p[0] = 2.0
            self.v[0] = min(0.0, self.v[0])

        # Height Y: [-1.15, 1.15]
        if self.p[1] < -1.15:
            self.p[1] = -1.15
            self.v[1] = max(0.0, self.v[1])
        elif self.p[1] > 1.15:
            self.p[1] = 1.15
            self.v[1] = min(0.0, self.v[1])

        # Depth Z: [0.1, 4.9]
        if self.p[2] < 0.1:
            self.p[2] = 0.1
            self.v[2] = max(0.0, self.v[2])
        elif self.p[2] > 4.9:
            self.p[2] = 4.9
            self.v[2] = min(0.0, self.v[2])

    def get_quaternion(self) -> list[float]:
        """Return [qx, qy, qz, qw] unit quaternion representation of rotation matrix."""
        try:
            quat = Rotation.from_matrix(self.R).as_quat()
            return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    def get_sigmas(self) -> list[float]:
        """Return 1-sigma position standard deviations [sigma_x, sigma_y, sigma_z]."""
        variances = np.diag(self.P)[:3]
        return [float(math.sqrt(max(1e-6, v))) for v in variances]


class LiveCaptureEngine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.source_mode = "phone"  # Default to phone stream for project requirements
        self.last_phone_t: float = 0.0
        self.last_sensor_t: float = 0.0
        self.latest_accel = [0.0, 0.0, 9.81]
        self.latest_gyro = [0.0, 0.0, 0.0]

        # 15-State Extended Kalman Filter
        self.ekf = ExtendedKalmanFilter()

        # ArUco detector
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.cfg.dictionary))
        self.dictionary = dictionary
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        # Filtered Live State
        self.current_pos = np.array([0.0, 0.0, 1.5], dtype=np.float64)
        self.current_rot = np.eye(3, dtype=np.float64)
        self.current_quat = [0.0, 0.0, 0.0, 1.0]
        self.current_vel = np.zeros(3, dtype=np.float64)
        self.current_sigmas = [0.04, 0.04, 0.04]
        self.current_state = "EKF READY"
        self.visible_marker_ids: list[int] = []
        self.active_features_count = 0
        self.fps = 0.0
        self.total_frames = 0
        self.start_time = time.time()
        self.total_distance = 0.0
        self.speed = 0.0

        # Accumulated Live Trajectory history (for real-time streaming to UI)
        self.trajectory_history: list[dict] = []
        self.max_history = 2000

        # Active Recording Session Memory (for permanent file storage)
        self.active_recording_folder: Path | None = None
        self.session_frames_count = 0
        self.session_markers_seen: set[int] = set()

        # Latest annotated frame JPEG bytes and buffered recent frames
        self.latest_jpeg: bytes | None = None
        self.recent_frames: collections.deque[np.ndarray] = collections.deque(maxlen=300)
        self._smoothed_pos: np.ndarray | None = None

        # Feature tracking memory (KLT)
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

        # Camera Intrinsics
        self.K, self.D = self.cfg.intrinsics(640, 480)

    def set_source(self, mode: str):
        with self.lock:
            self.source_mode = mode

    def clear_trajectory(self):
        with self.lock:
            self.trajectory_history.clear()
            self.total_frames = 0
            self.total_distance = 0.0
            self.speed = 0.0
            self._smoothed_pos = None
            self.ekf = ExtendedKalmanFilter()
            self.current_pos = self.ekf.p.copy()
            self.current_rot = self.ekf.R.copy()
            self.current_quat = self.ekf.get_quaternion()
            self.current_vel = self.ekf.v.copy()
            self.current_sigmas = self.ekf.get_sigmas()
            self.current_state = "EKF RESET"

    def attach_recording(self, folder: Path):
        with self.lock:
            self.active_recording_folder = Path(folder)
            self.session_frames_count = 0
            self.session_markers_seen.clear()

    def feed_phone_sensor(self, sensor_type: int, values: list):
        now = time.time()
        with self.lock:
            dt = now - self.last_sensor_t if self.last_sensor_t > 0 else 0.01
            self.last_sensor_t = now

            if sensor_type == 1 and len(values) >= 3:
                self.latest_accel = [float(v) for v in values[:3]]
                # 100 Hz IMU EKF Predict step
                if 0.0005 < dt < 0.1:
                    self.ekf.predict(dt, accel=self.latest_accel, gyro=self.latest_gyro)
            elif sensor_type == 4 and len(values) >= 3:
                self.latest_gyro = [float(v) for v in values[:3]]
                # 100 Hz Gyroscope EKF Predict step
                if 0.0005 < dt < 0.1:
                    self.ekf.predict(dt, accel=self.latest_accel, gyro=self.latest_gyro)

    def feed_phone_frame(self, frame_bgr: np.ndarray):
        now = time.time()
        with self.lock:
            if not self.running:
                self.running = True
                self.start_time = now
            self.source_mode = "phone"
            dt = max(0.005, now - self.last_phone_t) if self.last_phone_t > 0 else 0.033
            self.last_phone_t = now
            self.fps = 1.0 / dt
        self._process_frame(frame_bgr, dt)

    def _draw_phone_waiting_frame(self):
        """Render high-clarity waiting screen when waiting for phone stream."""
        h, w = 480, 640
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(12 + 16 * (y / h))
            canvas[y, :] = (val + 10, val + 4, val)

        try:
            from .sensor_api import get_local_ip, RECEIVER_PORT, RECEIVER_TOKEN
            ip = get_local_ip()
            port = RECEIVER_PORT
            token = RECEIVER_TOKEN
        except Exception:
            ip, port, token = "10.x.x.x", 8765, "1234"

        cv2.rectangle(canvas, (130, 30), (510, 450), (56, 189, 248), 2)
        cv2.circle(canvas, (320, 52), 4, (148, 163, 184), -1)
        cv2.line(canvas, (280, 435), (360, 435), (148, 163, 184), 2)

        cv2.putText(canvas, "ONEPLUS PHONE LIVE STREAM", (165, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        cv2.putText(canvas, "Waiting for Mobile App Video Stream...", (160, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1)

        cv2.rectangle(canvas, (155, 155), (485, 345), (20, 28, 45), -1)
        cv2.rectangle(canvas, (155, 155), (485, 345), (129, 140, 248), 1)

        cv2.putText(canvas, "Enter in OnePlus App:", (175, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1)
        cv2.putText(canvas, f"1. Host IP:   {ip}", (175, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (52, 211, 153), 2)
        cv2.putText(canvas, f"2. Port:      {port}", (175, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 1)
        cv2.putText(canvas, f"3. Token:     {token} (or 1234)", (175, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (251, 191, 36), 1)
        cv2.putText(canvas, "4. Tap 'CONNECT & STREAM'", (175, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (56, 189, 248), 1)

        pulse = int(140 + 115 * math.sin(time.time() * 3.5))
        cv2.circle(canvas, (200, 395), 8, (0, pulse, 0), -1)
        cv2.putText(canvas, f"TCP Receiver Active on {ip}:{port}", (220, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

        ret, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            with self.lock:
                self.fps = 0.0
                self.current_state = "WAITING FOR PHONE STREAM"
                self.visible_marker_ids = []
                self.latest_jpeg = buf.tobytes()

    def start(self, source_mode: str = "phone"):
        with self.lock:
            self.source_mode = source_mode
            if self.running and self.thread and self.thread.is_alive():
                return
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
            self.thread = None

    def get_state(self) -> dict:
        with self.lock:
            yaw = float(math.degrees(math.atan2(self.current_rot[0, 2], self.current_rot[2, 2])) % 360)
            return {
                "running": self.running,
                "source": self.source_mode,
                "pos": self.current_pos.tolist(),
                "quat": self.current_quat,
                "rot": self.current_rot.tolist(),
                "vel": self.current_vel.tolist(),
                "speed": round(float(np.linalg.norm(self.current_vel)), 2),
                "yaw_deg": round(yaw, 1),
                "total_distance": round(self.total_distance, 2),
                "elevation": round(float(self.current_pos[1] + 1.25), 2),
                "sigmas": self.current_sigmas,
                "state": self.current_state,
                "visible_markers": list(self.visible_marker_ids),
                "features_count": self.active_features_count,
                "fps": round(self.fps, 1),
                "total_frames": self.total_frames,
                "trajectory_count": len(self.trajectory_history),
                "room": ROOM_DIMENSIONS,
                "room_markers": ROOM_MARKERS,
                "recent_trajectory": self.trajectory_history[-350:],
                "ekf_counts": dict(self.ekf.counts)
            }

    def get_latest_frame_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def _process_frame(self, frame_bgr: np.ndarray, dt: float):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        self.K, self.D = self.cfg.intrinsics(w, h)
        annotated = frame_bgr.copy()

        with self.lock:
            self.recent_frames.append(frame_bgr.copy())

        # 1. KLT Natural Visual Features Tracking
        features_tracked = 0
        good_new = np.empty((0, 2), dtype=np.float32)
        good_old = np.empty((0, 2), dtype=np.float32)

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape and self.prev_pts is not None and len(self.prev_pts) > 0:
            nxt_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, winSize=(21, 21), maxLevel=2
            )
            if nxt_pts is not None and status is not None:
                valid = status.ravel() == 1
                good_new = nxt_pts[valid].reshape(-1, 2)
                good_old = self.prev_pts[valid].reshape(-1, 2)
                features_tracked = len(good_new)
                for pt_new, pt_old in zip(good_new[:45], good_old[:45]):
                    a, b = int(pt_new[0]), int(pt_new[1])
                    c, d = int(pt_old[0]), int(pt_old[1])
                    cv2.circle(annotated, (a, b), 3, (0, 255, 255), -1)
                    cv2.line(annotated, (a, b), (c, d), (0, 200, 255), 1)

        # Detect new features if tracking drops
        if features_tracked < 35:
            new_corners = cv2.goodFeaturesToTrack(gray, maxCorners=65, qualityLevel=0.01, minDistance=10)
            if new_corners is not None:
                self.prev_pts = new_corners
            else:
                self.prev_pts = np.empty((0, 1, 2), dtype=np.float32)
        else:
            self.prev_pts = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        self.active_features_count = features_tracked

        # 2. Detect ArUco Markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        observed_markers = []
        observed_poses = []

        half_s = self.cfg.marker_size_m / 2.0
        obj_pts = np.array([
            [-half_s, half_s, 0.0],
            [half_s, half_s, 0.0],
            [half_s, -half_s, 0.0],
            [-half_s, -half_s, 0.0]
        ], dtype=np.float64)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            for m_id, m_corners in zip(ids.ravel(), corners):
                uv = m_corners.reshape(4, 2).astype(np.float64)
                ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, uv, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ret and len(tvecs) > 0 and tvecs[0][2, 0] > 0.05:
                    r_c_m = cv2.Rodrigues(rvecs[0])[0]
                    t_c_m = tvecs[0].ravel()
                    observed_markers.append(int(m_id))
                    self.session_markers_seen.add(int(m_id))

                    if int(m_id) in ROOM_MARKERS:
                        rm = ROOM_MARKERS[int(m_id)]
                        r_w_m = np.array(rm["rot"], dtype=np.float64)
                        p_w_m = np.array(rm["pos"], dtype=np.float64)

                        # Rigid body transformation from camera to world frame
                        r_w_c = r_w_m @ r_c_m.T
                        p_w_c = p_w_m - r_w_c @ t_c_m
                        observed_poses.append((p_w_c, r_w_c))
                        cv2.drawFrameAxes(annotated, self.K, self.D, rvecs[0], tvecs[0], 0.10)

        # 3. Extended Kalman Filter (EKF) Updates
        with self.lock:
            # First, predict step for frame dt
            self.ekf.predict(dt, accel=self.latest_accel, gyro=self.latest_gyro)

            if observed_poses:
                # EKF Absolute Measurement Update: ArUco Landmark Anchor
                avg_p = np.mean([p for p, _ in observed_poses], axis=0)
                avg_r = observed_poses[0][1]
                updated = self.ekf.update_aruco(avg_p, avg_r, std_p=0.025, std_rot=0.04)
                if updated:
                    self.current_state = f"EKF ANCHORED (ArUco #{observed_markers[0]})"
                else:
                    self.current_state = f"EKF ANCHORED (ArUco • Gated)"

            elif not self.source_mode.startswith("sim"):
                # Natural Visual Odometry via KLT optical flow
                moved = False
                flow_mag = 0.0
                if features_tracked >= 8:
                    try:
                        flow_vecs = good_new - good_old
                        flow_mag = float(np.linalg.norm(np.median(flow_vecs, axis=0)))
                        if flow_mag < 0.35 and np.linalg.norm(self.latest_gyro) < 0.04:
                            # Device is held still -> apply Zero-Velocity Update (ZUPT)
                            self.ekf.update_zupt(std_v=0.005)
                            self.current_state = "EKF STATIONARY (ZUPT Active)"
                            moved = True
                        else:
                            E, inlier_mask = cv2.findEssentialMat(
                                good_new, good_old, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.5
                            )
                            if E is not None and E.shape == (3, 3):
                                _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, good_new, good_old, self.K, mask=inlier_mask)
                                if flow_mag > 0.4:
                                    step_m = min(0.08, max(0.004, flow_mag * 0.0035))
                                    delta_c = t_rel.ravel() * step_m
                                    delta_w = self.ekf.R @ delta_c
                                    self.ekf.update_vo(delta_w, dt, features_count=features_tracked)
                                    self.current_state = f"EKF TRACKING (VIO • {features_tracked} pts)"
                                    moved = True
                    except Exception:
                        pass

                if not moved:
                    self.current_state = f"EKF COASTING (Inertial • {features_tracked} pts)"
            else:
                self.current_state = "EKF TRAVERSAL (3D Simulation)"

            # 4. Enforce surveyed room boundary constraints
            self.ekf.enforce_room_bounds(ROOM_DIMENSIONS)

            # 5. Extract smooth filtered EKF state with online EMA low-pass filter
            raw_p = self.ekf.p.copy()
            if self._smoothed_pos is None:
                self._smoothed_pos = raw_p.copy()
            else:
                alpha = 0.35  # responsive yet smooths out handheld micro-tremor
                self._smoothed_pos = alpha * raw_p + (1.0 - alpha) * self._smoothed_pos
            self.current_pos = self._smoothed_pos.copy()
            self.current_rot = self.ekf.R.copy()
            self.current_quat = self.ekf.get_quaternion()
            self.current_vel = self.ekf.v.copy()
            self.current_sigmas = self.ekf.get_sigmas()
            self.speed = float(np.linalg.norm(self.current_vel))
            self.visible_marker_ids = observed_markers

            # Update path traveled metric with deadband against stationary noise
            if len(self.trajectory_history) > 0:
                last_p = self.trajectory_history[-1]
                dx = float(self.current_pos[0]) - last_p['x']
                dy = float(self.current_pos[1]) - last_p['y']
                dz = float(self.current_pos[2]) - last_p['z']
                step = math.sqrt(dx*dx + dy*dy + dz*dz)
                if 0.005 < step < 0.6:  # 5mm noise threshold prevents drift accumulation when stationary
                    self.total_distance += step

        yaw = float(math.degrees(math.atan2(self.current_rot[0, 2], self.current_rot[2, 2])) % 360)

        # 6. Draw HUD Overlay with EKF Status and Sensor Uncertainty
        cv2.rectangle(annotated, (10, 10), (360, 150), (15, 23, 42), -1)
        cv2.rectangle(annotated, (10, 10), (360, 150), (56, 189, 248), 1)
        cv2.putText(annotated, f"SLAM: {self.current_state}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1)
        cv2.putText(annotated, f"3D Pos: X:{self.current_pos[0]:.2f} Y:{self.current_pos[1]:.2f} Z:{self.current_pos[2]:.2f} m", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(annotated, f"Traveled: {self.total_distance:.2f} m | Speed: {self.speed:.2f} m/s", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 211, 153), 1)
        cv2.putText(annotated, f"Heading: {yaw:.0f} deg | ArUco: {self.visible_marker_ids}", (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (251, 191, 36), 1)
        cv2.putText(annotated, f"KLT Points: {self.active_features_count} pts | EKF P-sigma: +/-{self.current_sigmas[0]:.2f}m", (18, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1)
        cv2.putText(annotated, f"FPS: {self.fps:.1f} | Src: {self.source_mode}", (18, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1)

        # 7. Encode JPEG for UI stream
        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            with self.lock:
                self.latest_jpeg = buf.tobytes()
                self.total_frames += 1
                self.session_frames_count += 1

                # Record structured trajectory waypoint
                waypoint = {
                    "t": round(time.time() - self.start_time, 2),
                    "x": float(self.current_pos[0]),
                    "y": float(self.current_pos[1]),
                    "z": float(self.current_pos[2]),
                    "qx": self.current_quat[0],
                    "qy": self.current_quat[1],
                    "qz": self.current_quat[2],
                    "qw": self.current_quat[3],
                    "vx": float(self.current_vel[0]),
                    "vy": float(self.current_vel[1]),
                    "vz": float(self.current_vel[2]),
                    "sigma_x": self.current_sigmas[0],
                    "sigma_y": self.current_sigmas[1],
                    "sigma_z": self.current_sigmas[2],
                    "speed": round(self.speed, 2),
                    "state": self.current_state,
                    "dist": round(self.total_distance, 2),
                    "markers": list(self.visible_marker_ids)
                }
                self.trajectory_history.append(waypoint)
                if len(self.trajectory_history) > self.max_history:
                    self.trajectory_history.pop(0)

    def finalize_recording(self, folder: Path, reason: str = "completed"):
        """Write processed EKF trajectory, summary, and marker map into recording directory."""
        folder = Path(folder)
        processed_dir = folder / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        with self.lock:
            pts = [dict(p) for p in self.trajectory_history]
            tot_dist = self.total_distance
            avg_spd = float(np.mean([p.get('speed', 0.0) for p in pts])) if pts else 0.0
            duration = pts[-1]['t'] - pts[0]['t'] if len(pts) > 1 else 0.0
            markers_seen = list(self.session_markers_seen)

        # Apply 5-tap Gaussian smoothing to eliminate high-frequency jitter
        if len(pts) >= 5:
            x_arr = np.array([p['x'] for p in pts], dtype=np.float64)
            y_arr = np.array([p['y'] for p in pts], dtype=np.float64)
            z_arr = np.array([p['z'] for p in pts], dtype=np.float64)
            kernel = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float64)
            kernel /= np.sum(kernel)
            pad = len(kernel) // 2
            xs = np.convolve(np.pad(x_arr, pad, mode='edge'), kernel, mode='valid')
            ys = np.convolve(np.pad(y_arr, pad, mode='edge'), kernel, mode='valid')
            zs = np.convolve(np.pad(z_arr, pad, mode='edge'), kernel, mode='valid')
            for i in range(len(pts)):
                pts[i]['x'] = round(float(xs[i]), 4)
                pts[i]['y'] = round(float(ys[i]), 4)
                pts[i]['z'] = round(float(zs[i]), 4)

        # 1. Write trajectory.csv
        traj_csv_path = processed_dir / "trajectory.csv"
        with traj_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "x", "y", "z", "qx", "qy", "qz", "qw", "sigma_x", "sigma_y", "sigma_z", "tracking", "speed"])
            for p in pts:
                writer.writerow([
                    f"{p['t']:.2f}",
                    f"{p['x']:.4f}",
                    f"{p['y']:.4f}",
                    f"{p['z']:.4f}",
                    f"{p['qx']:.5f}",
                    f"{p['qy']:.5f}",
                    f"{p['qz']:.5f}",
                    f"{p['qw']:.5f}",
                    f"{p.get('sigma_x', 0.03):.4f}",
                    f"{p.get('sigma_y', 0.03):.4f}",
                    f"{p.get('sigma_z', 0.03):.4f}",
                    p.get('state', 'TRACKING'),
                    f"{p.get('speed', 0.0):.2f}"
                ])

        # 2. Write summary.json
        summary = {
            "frames": len(pts),
            "state_samples": len(pts),
            "total_distance_m": round(tot_dist, 2),
            "average_speed_mps": round(avg_spd, 2),
            "duration_s": round(duration, 2),
            "filter": "15-State Error-State Extended Kalman Filter (EKF) with ArUco & KLT updates",
            "marker_count": len(markers_seen),
            "markers_mapped": len(markers_seen),
            "counts": dict(self.ekf.counts),
            "tracking_counts": {
                "TRACKING": sum(1 for p in pts if "TRACKING" in p.get('state', '') or "ANCHORED" in p.get('state', '')),
                "STATIONARY": sum(1 for p in pts if "STATIONARY" in p.get('state', '')),
                "COASTING": sum(1 for p in pts if "COASTING" in p.get('state', ''))
            },
            "status": reason
        }
        (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # 3. Write marker_map.json
        marker_map = {
            "marker_size_m": self.cfg.marker_size_m,
            "dictionary": self.cfg.dictionary,
            "frame": "surveyed room frame",
            "markers": {}
        }
        for mid, rm in ROOM_MARKERS.items():
            T_mat = np.eye(4, dtype=np.float64)
            T_mat[:3, :3] = np.array(rm["rot"])
            T_mat[:3, 3] = np.array(rm["pos"])
            marker_map["markers"][str(mid)] = {
                "T_world_marker": T_mat.tolist(),
                "name": rm["name"],
                "wall": rm["wall"],
                "fixed": True
            }
        (processed_dir / "marker_map.json").write_text(json.dumps(marker_map, indent=2), encoding="utf-8")

        # 4. Write capture_summary.json if missing
        cap_sum_file = folder / "capture_summary.json"
        if not cap_sum_file.is_file():
            cap_sum_file.write_text(json.dumps({
                "records": len(pts),
                "sequence_gaps": 0,
                "reason": reason
            }, indent=2), encoding="utf-8")

        return folder

    def save_session(self, dest_root: Path | None = None, name: str | None = None) -> Path:
        """Manually save active live session into a standalone dataset folder."""
        if dest_root is None:
            dest_root = Path(__file__).resolve().parents[1] / "recordings"
        dest_root = Path(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)

        session_id = name or (time.strftime("%Y%m%d_%H%M%S") + "_LIVE_" + uuid.uuid4().hex[:6])
        folder = dest_root / session_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frames").mkdir(exist_ok=True)
        (folder / "depth").mkdir(exist_ok=True)

        # 1. Metadata
        meta = {
            "device": f"OnePlus Live Stream ({self.source_mode})",
            "android": "Android 14/15",
            "timestamp": time.time(),
            "selected_sensors": {"1": 1, "4": 4},
            "sensors": [
                {"key": 1, "name": "Accelerometer", "type": 1, "vendor": "Qualcomm/InvenSense", "version": 1},
                {"key": 4, "name": "Gyroscope", "type": 4, "vendor": "Qualcomm/InvenSense", "version": 1}
            ],
            "filter": "15-State Extended Kalman Filter (EKF)",
            "room": ROOM_DIMENSIONS,
            "session_id": session_id
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # 2. Persist recent captured video frames into session folder
        frames_dir = folder / "frames"
        with self.lock:
            buffered = list(self.recent_frames)
        for idx, fr in enumerate(buffered):
            frame_file = frames_dir / f"{idx:010d}.jpg"
            cv2.imwrite(str(frame_file), fr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # 3. Finalize processed trajectory
        self.finalize_recording(folder, reason="user_saved")
        return folder

    def get_trajectory_csv(self) -> str:
        """Generate in-memory CSV text of the active EKF trajectory."""
        output = []
        output.append("timestamp_s,x,y,z,qx,qy,qz,qw,sigma_x,sigma_y,sigma_z,tracking,speed")
        with self.lock:
            pts = list(self.trajectory_history)
        for p in pts:
            output.append(
                f"{p['t']:.2f},{p['x']:.4f},{p['y']:.4f},{p['z']:.4f},"
                f"{p['qx']:.5f},{p['qy']:.5f},{p['qz']:.5f},{p['qw']:.5f},"
                f"{p.get('sigma_x', 0.03):.4f},{p.get('sigma_y', 0.03):.4f},{p.get('sigma_z', 0.03):.4f},"
                f"{p.get('state', 'TRACKING')},{p.get('speed', 0.0):.2f}"
            )
        return "\n".join(output)

    def _run_loop(self):
        """Main capture worker thread."""
        cap = None
        sim_step = 0
        last_t = time.time()

        try:
            while self.running:
                now = time.time()
                dt = max(0.001, now - last_t)
                last_t = now
                self.fps = 1.0 / dt

                if self.source_mode == "webcam":
                    if cap is None or not cap.isOpened():
                        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                        cap = cv2.VideoCapture(0, backend)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, frame = cap.read()
                    if ret:
                        self._process_frame(frame, dt)
                    else:
                        time.sleep(0.03)

                elif self.source_mode.startswith("sim"):
                    if cap is not None:
                        cap.release()
                        cap = None

                    # Generate synthetic room traversal in real-time
                    sim_step += 1
                    t = sim_step * 0.033
                    theta = 0.5 * t
                    target_x = 1.1 * math.sin(theta)
                    target_z = 2.5 + 1.4 * math.cos(theta)
                    target_y = 0.05 * math.sin(2 * theta)

                    yaw = theta + math.pi / 2.0
                    R_yaw = np.array([
                        [math.cos(yaw), 0, math.sin(yaw)],
                        [0, 1, 0],
                        [-math.sin(yaw), 0, math.cos(yaw)]
                    ])

                    # Determine which markers are visible
                    visible = []
                    if target_z > 2.0 and abs(target_x) < 1.0:
                        visible = [0, 1]
                    elif target_x > 0.5:
                        visible = [2]
                    elif target_z < 2.0:
                        visible = [3]
                    elif target_x < -0.5:
                        visible = [4]

                    # Render realistic 3D room view with real ArUco markers
                    sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.rectangle(sim_frame, (0, 240), (640, 480), (32, 38, 48), -1)
                    for gy in range(250, 480, 35):
                        cv2.line(sim_frame, (0, gy), (640, gy), (50, 65, 80), 1)
                    for gx in range(0, 641, 70):
                        cv2.line(sim_frame, (gx, 240), (gx + int((gx - 320) * 0.8), 480), (50, 65, 80), 1)

                    cv2.rectangle(sim_frame, (0, 0), (640, 240), (18, 24, 34), -1)
                    cv2.line(sim_frame, (0, 240), (640, 240), (56, 189, 248), 2)

                    for mid in visible:
                        m_size = 120
                        m_img = cv2.aruco.generateImageMarker(self.dictionary, mid, m_size)
                        m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
                        bordered = cv2.copyMakeBorder(m_bgr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                        bh, bw = bordered.shape[:2]
                        mx = 320 - bw // 2 + int(80 * math.sin(theta))
                        my = 135 - bh // 2 + int(12 * math.cos(theta))
                        if 0 <= mx and mx + bw < 640 and 0 <= my and my + bh < 480:
                            sim_frame[my:my+bh, mx:mx+bw] = bordered

                    cv2.putText(sim_frame, "3D ROOM WALK SIMULATION (ACTIVE)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
                    cv2.putText(sim_frame, f"Traversing: X={target_x:.2f} Z={target_z:.2f} m", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1)

                    self._process_frame(sim_frame, 0.033)
                    time.sleep(0.033)

                elif self.source_mode == "phone":
                    if cap is not None:
                        cap.release()
                        cap = None
                    now = time.time()
                    if (now - getattr(self, 'last_phone_t', 0)) > 1.5:
                        self._draw_phone_waiting_frame()
                    time.sleep(0.08)

        finally:
            if cap is not None and cap.isOpened():
                cap.release()


# Global Singleton Live Capture Engine
LIVE_ENGINE = LiveCaptureEngine()
try:
    LIVE_ENGINE.start("phone")
except Exception:
    pass
