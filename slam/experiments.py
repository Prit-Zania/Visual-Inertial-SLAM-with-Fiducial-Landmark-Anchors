"""Benchmark experiment suite implementing all 5 mandatory scenarios from IE415 Section 8:
1. Simple path: straight / L-shaped trajectory through marked room
2. Marker-loss path: measurable segment with markers occluded/absent to evaluate drift
3. Closed-loop path: return to start and re-observe markers for loop closure
4. Repeat run: relocalize in the same room using a pre-constructed marker map
5. Difficult condition: rapid motion, higher noise, and sparse markers
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import uuid
import cv2
import numpy as np

from .config import Config
from .geometry import pose, inv
from .evaluate import evaluate
from .report import compare, plot, confidence
from .pipeline import process


EXPERIMENT_NAMES = [
    'exp1_simple_path',
    'exp2_marker_loss',
    'exp3_closed_loop',
    'exp4_repeat_run',
    'exp5_difficult_condition'
]

EXPERIMENT_DESCRIPTIONS = {
    'exp1_simple_path': 'Scenario 1: Simple L-shaped trajectory past 4 fixed ArUco markers.',
    'exp2_marker_loss': 'Scenario 2: Marker-loss path with 3.5s occluded segment testing VIO drift and recovery.',
    'exp3_closed_loop': 'Scenario 3: Closed-loop trajectory returning to start with loop-closure re-observation.',
    'exp4_repeat_run': 'Scenario 4: Repeat run in the same room using the pre-constructed marker map from Scenario 3.',
    'exp5_difficult_condition': 'Scenario 5: Difficult condition with rapid angular velocity, high noise, and sparse markers.'
}


def _create_marker_patch(marker_id: int, size_px: int = 120) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


def _render_markers_into_frame(
    image: np.ndarray,
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    k_matrix: np.ndarray,
    marker_poses: dict[int, np.ndarray],
    marker_size_m: float,
    marker_images: dict[int, np.ndarray],
    visible_marker_ids: set[int] | None = None
):
    """Render perspective-correct ArUco markers into a camera frame."""
    half = marker_size_m / 2.0
    square_local = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0]
    ])

    # Camera frame world transform: T_w_c = pose(camera_rot, camera_pos)
    # Point in camera frame: p_c = R_c_w * (p_w - camera_pos)
    r_c_w = camera_rot.T

    for mid, t_w_m in marker_poses.items():
        if visible_marker_ids is not None and mid not in visible_marker_ids:
            continue

        r_w_m = t_w_m[:3, :3]
        p_w_m = t_w_m[:3, 3]

        # 3D corners in world frame
        corners_w = (r_w_m @ square_local.T).T + p_w_m

        # Transform to camera frame
        corners_c = (r_c_w @ (corners_w - camera_pos).T).T

        # Check if all corners are in front of camera
        if np.any(corners_c[:, 2] <= 0.25):
            continue

        # Project to pixels
        uv = (corners_c @ k_matrix.T)
        uv = uv[:, :2] / uv[:, 2:3]

        # Check within frame bounds
        if np.any(uv[:, 0] < -20) or np.any(uv[:, 0] > 660) or np.any(uv[:, 1] < -20) or np.any(uv[:, 1] > 500):
            # Partially or fully out of bounds
            if np.all(uv[:, 0] < 0) or np.all(uv[:, 0] > 640) or np.all(uv[:, 1] < 0) or np.all(uv[:, 1] > 480):
                continue

        center = uv.mean(0)
        quiet = center + (uv - center) * 1.35
        cv2.fillConvexPoly(image, np.round(quiet).astype(np.int32), 255)

        tag = marker_images.get(mid)
        if tag is None:
            tag = _create_marker_patch(mid, 120)
            marker_images[mid] = tag

        h = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [119, 0], [119, 119], [0, 119]]),
            uv.astype(np.float32)
        )
        warped = cv2.warpPerspective(tag, h, (640, 480), borderValue=255, flags=cv2.INTER_NEAREST)
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, np.round(uv).astype(np.int32), 255)
        image[mask > 0] = warped[mask > 0]


def generate_experiment_data(scenario: str, root_dir: Path, prior_map_path: Path | None = None) -> Path:
    """Generate deterministic sensor and camera recordings for a specific scenario."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    frames_dir = root / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        T_body_camera=np.eye(4).tolist(),
        calibration_verified=True,
        max_features=70,
        graph_window=6
    )
    (root / 'config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')

    metadata = {
        'kind': 'hello',
        'protocol': 'oneplus-msckf-v1',
        'session_id': str(uuid.uuid4()),
        'synthetic': True,
        'scenario': scenario,
        'description': EXPERIMENT_DESCRIPTIONS.get(scenario, scenario),
        'selected_sensors': {'1': 1, '4': 4, '2': 2}
    }
    (root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    rng = np.random.default_rng(42 if scenario != 'exp5_difficult_condition' else 99)
    k = np.array(cfg.camera_matrix)

    # 3D natural feature points in room
    points = rng.uniform([-2.5, -1.8, 1.5], [2.5, 1.8, 6.0], (240, 3))
    patches = rng.integers(30, 230, (240, 7, 7), dtype=np.uint8)

    # Marker dictionary
    marker_images = {}
    for mid in range(8):
        marker_images[mid] = _create_marker_patch(mid, 120)

    # Define standard room ArUco marker poses
    # Room layout: 4 walls with markers at z=2.0, z=4.0, x=-1.5, x=1.5
    room_markers = {
        0: pose(np.diag([1., -1., -1.]), [0.0, 0.0, 2.5]),            # Front wall center
        1: pose(np.diag([1., -1., -1.]), [0.8, -0.2, 3.2]),          # Front wall right
        2: pose(np.diag([-1., 1., -1.]), [-0.7, 0.3, 3.8]),         # Oblique wall
        3: pose(np.diag([1., -1., -1.]), [0.0, 0.5, 4.5]),           # Back wall
        4: pose(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), [1.5, 0.0, 2.8]),  # Right wall
        5: pose(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [-1.5, 0.0, 2.8])  # Left wall
    }

    # Save surveyed marker map
    surveyed_map = {
        'marker_size_m': cfg.marker_size_m,
        'dictionary': 'DICT_6X6_250',
        'markers': {str(k_id): {'T_world_marker': m_pose.tolist()} for k_id, m_pose in room_markers.items()}
    }
    (root / 'surveyed_map.json').write_text(json.dumps(surveyed_map, indent=2), encoding='utf-8')

    # Configure trajectory based on scenario
    # Note: All realistic scenarios begin with a 1.6s stationary initialization period
    # where the user holds the phone steady so the estimator can compute gravity and biases.
    init_t = 1.6

    if scenario == 'exp1_simple_path':
        # Straight out, turn right (L-shaped), duration 6.0s
        duration = 6.0
        accel_noise = 0.012
        gyro_noise = 0.0004
        def trajectory_func(t):
            if t < init_t:
                return np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), {0, 1}
            u = t - init_t
            move_dur = duration - init_t
            if u < move_dur * 0.5:
                s = u / (move_dur * 0.5)
                p = np.array([0.0, 0.0, 0.5 * s])
                v = np.array([0.0, 0.0, 0.5 / (move_dur * 0.5)])
                a = np.zeros(3)
                r = np.eye(3)
                omega = np.zeros(3)
            else:
                s = (u - move_dur * 0.5) / (move_dur * 0.5)
                theta = (np.pi / 4.0) * s
                p = np.array([0.45 * s, 0.0, 0.5 + 0.3 * s])
                v = np.array([0.45 / (move_dur * 0.5), 0.0, 0.3 / (move_dur * 0.5)])
                a = np.zeros(3)
                r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                omega = np.array([0.0, (np.pi / 4.0) / (move_dur * 0.5), 0.0])
            return p, v, a, r, omega, {0, 1, 4}

    elif scenario == 'exp2_marker_loss':
        # Straight path where camera pans away from markers between 2.6s and 4.8s
        duration = 5.8
        accel_noise = 0.018
        gyro_noise = 0.0006
        def trajectory_func(t):
            if t < init_t:
                return np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), {0, 1}
            u = t - init_t
            move_dur = duration - init_t
            s = u / move_dur
            p = np.array([0.35 * np.sin(np.pi * s), 0.0, 0.75 * s])
            v = np.array([0.35 * (np.pi / move_dur) * np.cos(np.pi * s), 0.0, 0.75 / move_dur])
            a = np.array([-0.35 * (np.pi / move_dur)**2 * np.sin(np.pi * s), 0.0, 0.0])
            r = np.eye(3)
            omega = np.zeros(3)
            # Occlusion window: NO markers visible between 2.6s and 4.6s (2 seconds of pure drift!)
            if 2.6 <= t <= 4.6:
                visible = set()  # Marker dropout segment
            else:
                visible = {0, 1}
            return p, v, a, r, omega, visible

    elif scenario == 'exp3_closed_loop':
        # Closed oval loop returning to start (0,0,0) with loop closure re-observation
        duration = 6.4
        accel_noise = 0.014
        gyro_noise = 0.0004
        def trajectory_func(t):
            if t < init_t:
                return np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), {0, 1}
            u = t - init_t
            move_dur = duration - init_t
            theta = 2.0 * np.pi * (u / move_dur)
            rx, rz = 0.45, 0.55
            p = np.array([rx * np.sin(theta), 0.0, rz * (1.0 - np.cos(theta))])
            v = np.array([rx * (2.0 * np.pi / move_dur) * np.cos(theta), 0.0, rz * (2.0 * np.pi / move_dur) * np.sin(theta)])
            a = np.array([-rx * (2.0 * np.pi / move_dur)**2 * np.sin(theta), 0.0, rz * (2.0 * np.pi / move_dur)**2 * np.cos(theta)])
            yaw = 0.35 * np.sin(theta)
            r = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
            omega = np.array([0.0, 0.35 * (2.0 * np.pi / move_dur) * np.cos(theta), 0.0])
            # Markers visible around room, including marker 0 at start and return!
            visible = {0, 1, 3}
            return p, v, a, r, omega, visible

    elif scenario == 'exp4_repeat_run':
        # Different exploration trajectory in the same room utilizing pre-existing marker map
        duration = 5.5
        accel_noise = 0.015
        gyro_noise = 0.0005
        def trajectory_func(t):
            if t < init_t:
                return np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), {0, 2}
            u = t - init_t
            move_dur = duration - init_t
            s = u / move_dur
            p = np.array([-0.25 * np.sin(1.5 * np.pi * s), 0.04 * np.sin(2 * np.pi * s), 0.7 * s])
            v = np.array([-0.25 * 1.5 * np.pi / move_dur * np.cos(1.5 * np.pi * s), 0.0, 0.7 / move_dur])
            a = np.zeros(3)
            r = np.eye(3)
            omega = np.zeros(3)
            visible = {0, 2, 5}
            return p, v, a, r, omega, visible

    else:  # exp5_difficult_condition
        # Rapid aggressive turns, high noise, oblique angles
        duration = 5.0
        accel_noise = 0.035
        gyro_noise = 0.002
        def trajectory_func(t):
            if t < init_t:
                return np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), {1}
            u = t - init_t
            move_dur = duration - init_t
            omega_w = 2.0 * np.pi / move_dur
            theta = omega_w * u
            p = np.array([0.25 * np.sin(theta), 0.05 * np.cos(2 * theta), 0.45 * (1.0 - np.cos(theta))])
            v = np.array([0.25 * omega_w * np.cos(theta), -0.10 * omega_w * np.sin(2 * theta), 0.45 * omega_w * np.sin(theta)])
            a = np.array([-0.25 * omega_w**2 * np.sin(theta), -0.20 * omega_w**2 * np.cos(2 * theta), 0.45 * omega_w**2 * np.cos(theta)])
            r = np.array([[np.cos(theta * 0.4), 0, np.sin(theta * 0.4)], [0, 1, 0], [-np.sin(theta * 0.4), 0, np.cos(theta * 0.4)]])
            omega = np.array([0.0, omega_w * 0.4, 0.0])
            visible = {1} if int(u * 2) % 2 == 0 else {4}
            return p, v, a, r, omega, visible

    # Generate events and ground truth
    events = [{'kind': 'camera_info', 'timestamp_source': 1, 'width': 640, 'height': 480, 'depth_capability': False}]
    truth = []
    frame_idx = 0
    seq = 0
    dt_imu = 0.01  # 100 Hz IMU
    total_steps = int(duration / dt_imu)

    for i in range(total_steps + 1):
        t = i * dt_imu
        p, v, a_true, r_mat, omega_true, visible_markers = trajectory_func(t)

        # Accelerometer measurement: a_meas = R_c_w * (a_true + [0, 0, 9.80665]) + noise
        gravity = np.array([0.0, 0.0, 9.80665])
        a_body = r_mat.T @ (a_true + gravity) + rng.normal(0, accel_noise, 3)
        omega_body = omega_true + rng.normal(0, gyro_noise, 3)

        seq += 1
        events.append({
            'kind': 'sensor', 'seq': seq, 'sensor_type': 1, 'sensor_key': 1, 'accuracy': 3,
            'timestamp_ns': int(t * 1e9), 'values': a_body.tolist()
        })
        seq += 1
        events.append({
            'kind': 'sensor', 'seq': seq, 'sensor_type': 4, 'sensor_key': 4, 'accuracy': 3,
            'timestamp_ns': int(t * 1e9), 'values': omega_body.tolist()
        })

        # Orientation quaternion from rotation matrix
        # scipy Rotation or manual
        tr = np.trace(r_mat)
        if tr > 0:
            s_q = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * s_q
            qx = (r_mat[2, 1] - r_mat[1, 2]) / s_q
            qy = (r_mat[0, 2] - r_mat[2, 0]) / s_q
            qz = (r_mat[1, 0] - r_mat[0, 1]) / s_q
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        truth.append([t, p[0], p[1], p[2], qx, qy, qz, qw])

        # Camera frame at 10 Hz (every 10 IMU steps)
        if i % 10 == 0:
            frame_idx += 1
            image = np.full((480, 640), 242, np.uint8)

            # Draw natural background feature patches
            pc = (r_mat.T @ (points - p).T).T
            valid_depth = pc[:, 2] > 0.3
            pixels = (pc[valid_depth] @ k.T)
            pixels = pixels[:, :2] / pixels[:, 2:3]
            for patch, (px, py) in zip(patches[valid_depth], pixels):
                ix, iy = int(round(px)), int(round(py))
                if 8 < ix < 632 and 8 < iy < 472:
                    image[iy-3:iy+4, ix-3:ix+4] = patch

            # Render ArUco markers into frame
            _render_markers_into_frame(
                image=image,
                camera_pos=p,
                camera_rot=r_mat,
                k_matrix=k,
                marker_poses=room_markers,
                marker_size_m=cfg.marker_size_m,
                marker_images=marker_images,
                visible_marker_ids=visible_markers
            )

            # Save JPEG frame
            filename = f'frames/{frame_idx:05d}.jpg'
            cv2.imwrite(str(root / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 96])

            seq += 1
            events.append({
                'kind': 'frame', 'seq': seq, 'timestamp_ns': int(t * 1e9),
                'width': 640, 'height': 480, 'file': filename
            })

    # Write events.jsonl
    with (root / 'events.jsonl').open('w', encoding='utf-8') as out:
        for ev in events:
            out.write(json.dumps(ev) + '\n')

    # Write ground_truth.csv
    with (root / 'ground_truth.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['timestamp_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows(truth)

    return root


def run_experiment(scenario: str, output_base: Path = Path('demo_output')) -> dict:
    """Generate, run SLAM + baseline, and evaluate a scenario."""
    folder = output_base / scenario
    print(f"\n=======================================================")
    print(f" Running Experiment: {scenario}")
    print(f" {EXPERIMENT_DESCRIPTIONS.get(scenario, '')}")
    print(f"=======================================================")

    prior_map = None
    if scenario == 'exp4_repeat_run':
        # Try to use map from exp3_closed_loop if it exists
        candidate_map = output_base / 'exp3_closed_loop' / 'processed' / 'marker_map.json'
        if candidate_map.is_file():
            prior_map = str(candidate_map)
            print(f" Using pre-built map from {candidate_map}")

    # Step 1: Generate synthetic sensor & camera dataset
    print(f"1. Generating sensor data & camera frames in {folder}...")
    generate_experiment_data(scenario, folder, prior_map_path=Path(prior_map) if prior_map else None)

    cfg_file = folder / 'config.json'
    map_to_use = prior_map or (str(folder / 'surveyed_map.json') if (folder / 'surveyed_map.json').is_file() else None)

    # Step 2: Run Full MSCKF SLAM with ArUco Landmark Constraints
    print("2. Running MSCKF SLAM with ArUco landmark constraints...")
    t0 = time.monotonic()
    process(folder, str(cfg_file), prior_map=map_to_use, use_markers=True, smooth=True, allow_uncalibrated=True, output=folder / 'processed')
    slam_time = time.monotonic() - t0
    print(f"   Completed in {slam_time:.2f}s")

    # Step 3: Run Baseline (Pure Visual-Inertial Odometry without ArUco)
    print("3. Running Pure Visual-Inertial Odometry baseline (no ArUco)...")
    t0 = time.monotonic()
    process(folder, str(cfg_file), prior_map=None, use_markers=False, smooth=False, allow_uncalibrated=True, output=folder / 'baseline')
    baseline_time = time.monotonic() - t0
    print(f"   Completed in {baseline_time:.2f}s")

    # Step 4: Quantitative Evaluation
    print("4. Evaluating trajectories against ground truth...")
    report = compare(folder)

    # Attach scenario metadata to comparison
    report['scenario'] = scenario
    report['description'] = EXPERIMENT_DESCRIPTIONS.get(scenario, '')
    report['slam_runtime_s'] = slam_time
    report['baseline_runtime_s'] = baseline_time

    # Save final comparison
    (folder / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    # Print summary
    print(f"\n--- {scenario} Results ---")
    p_res = report.get('processed', {})
    b_res = report.get('baseline', {})
    print(f"  MSCKF + ArUco SLAM:")
    print(f"    Position RMSE:     {p_res.get('position_rmse_m', 0.0)*100:.2f} cm")
    print(f"    Endpoint Drift:    {p_res.get('endpoint_error_m', 0.0)*100:.2f} cm")
    print(f"    Drift per metre:   {p_res.get('endpoint_error_per_m', 0.0)*100:.2f} %")
    print(f"  Baseline (Pure VIO):")
    print(f"    Position RMSE:     {b_res.get('position_rmse_m', 0.0)*100:.2f} cm")
    print(f"    Endpoint Drift:    {b_res.get('endpoint_error_m', 0.0)*100:.2f} cm")
    print(f"    Drift per metre:   {b_res.get('endpoint_error_per_m', 0.0)*100:.2f} %")

    return report


def run_all_experiments(output_base: Path = Path('demo_output')) -> dict:
    """Run all 5 required course experiments and compile a master scorecard."""
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    master_scorecard = {}

    for name in EXPERIMENT_NAMES:
        try:
            res = run_experiment(name, output_base)
            master_scorecard[name] = res
        except Exception as e:
            print(f"Error in {name}: {e}")
            master_scorecard[name] = {'error': str(e)}

    # Save master scorecard
    scorecard_path = output_base / 'master_benchmark_scorecard.json'
    scorecard_path.write_text(json.dumps(master_scorecard, indent=2), encoding='utf-8')
    print(f"\nMaster benchmark scorecard saved to {scorecard_path}")
    return master_scorecard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IE415 SLAM Course Benchmark Experiments")
    parser.add_argument('--scenario', choices=EXPERIMENT_NAMES + ['all'], default='all', help='Scenario to run')
    parser.add_argument('--output', default='demo_output', help='Base directory for outputs')
    args = parser.parse_args()

    if args.scenario == 'all':
        run_all_experiments(Path(args.output))
    else:
        run_experiment(args.scenario, Path(args.output))
