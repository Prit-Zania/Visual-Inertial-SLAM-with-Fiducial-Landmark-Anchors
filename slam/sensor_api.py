"""Unified HTTP API and interactive Web Dashboard Server for OnePlus SLAM & sensors."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name('sensor_web.html')

# Global receiver & live state
RECEIVER_STOP = None
RECEIVER_THREAD = None
RECEIVER_TOKEN = "1234"
RECEIVER_PORT = 8765
RECEIVER_STATUS = "Receiver active on port 8765."
RECEIVER_LOGS = []
PROCESSING_ACTIVE = False
PROCESSING_STATUS = ""
ACTIVE_RECORDING_FOLDER = None
LATEST_FRAME_JPEG = None
LATEST_TELEMETRY = {"records": 0, "frames": 0, "markers": [], "imu_status": "idle", "updated_at": 0}

# In-memory cache for fast sensor queries
_SESSION_CACHE = {}


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def start_phone_receiver(root_folder: Path, port: int = 8765, token: str = "1234"):
    global RECEIVER_STOP, RECEIVER_THREAD, RECEIVER_TOKEN, RECEIVER_PORT, RECEIVER_STATUS, RECEIVER_LOGS
    if RECEIVER_THREAD and RECEIVER_THREAD.is_alive():
        return
    RECEIVER_PORT = port
    RECEIVER_TOKEN = token
    RECEIVER_STOP = threading.Event()
    RECEIVER_LOGS = []

    def log_status(msg):
        global RECEIVER_STATUS, RECEIVER_LOGS
        RECEIVER_STATUS = msg
        RECEIVER_LOGS.append(f'[{time.strftime("%H:%M:%S")}] {msg}')
        if len(RECEIVER_LOGS) > 60:
            RECEIVER_LOGS.pop(0)

    def run_srv():
        from .receiver import serve
        try:
            log_status(f'Listening on 0.0.0.0:{RECEIVER_PORT}; pairing token: {RECEIVER_TOKEN}')
            serve('0.0.0.0', RECEIVER_PORT, RECEIVER_TOKEN, root_folder, log_status, RECEIVER_STOP)
        except Exception as err:
            log_status(f'Receiver stopped: {err}')

    RECEIVER_THREAD = threading.Thread(target=run_srv, daemon=True)
    RECEIVER_THREAD.start()


def _json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def _ensure_session_processed(folder: Path) -> bool:
    """Auto-generate EKF processed trajectory and summary for recorded sessions if missing."""
    try:
        processed_dir = folder / 'processed'
        if (processed_dir / 'trajectory.csv').is_file():
            return True
        frames_dir = folder / 'frames'
        if not frames_dir.is_dir():
            return False
        frames = sorted(frames_dir.glob('*.jpg'))
        if not frames:
            return False
        if len(frames) > 250:
            stride = max(1, len(frames) // 200)
            frames = frames[::stride]

        from .live_capture import ExtendedKalmanFilter, ROOM_MARKERS
        from .config import Config

        processed_dir.mkdir(parents=True, exist_ok=True)
        cfg = Config()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        detector = cv2.aruco.ArucoDetector(dictionary)
        ekf = ExtendedKalmanFilter()
        half_s = cfg.marker_size_m / 2.0
        obj_pts = np.array([[-half_s, half_s, 0], [half_s, half_s, 0], [half_s, -half_s, 0], [-half_s, -half_s, 0]], float)

        rows = []
        t = 0.0
        tot_dist = 0.0
        prev_p = None
        markers_seen = set()

        for f in frames:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            K, D = cfg.intrinsics(w, h)
            ekf.predict(0.033)
            corners, ids, _ = detector.detectMarkers(gray)
            state = "TRACKING"
            if ids is not None and len(ids) > 0:
                for mid, mc in zip(ids.ravel(), corners):
                    m_int = int(mid)
                    markers_seen.add(m_int)
                    if m_int in ROOM_MARKERS:
                        ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, mc.reshape(4, 2).astype(float), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if ret and len(tvecs) > 0:
                            rm = ROOM_MARKERS[m_int]
                            r_w_m = np.array(rm['rot'], float)
                            p_w_m = np.array(rm['pos'], float)
                            r_c_m = cv2.Rodrigues(rvecs[0])[0]
                            t_c_m = tvecs[0].ravel()
                            r_w_c = r_w_m @ r_c_m.T
                            p_w_c = p_w_m - r_w_c @ t_c_m
                            ekf.update_aruco(p_w_c, r_w_c)
                            state = f"ANCHORED (ArUco #{m_int})"
                            break
            ekf.enforce_room_bounds({'width_x': 4.0, 'length_z': 5.0, 'height_y': 2.5})
            q = ekf.get_quaternion()
            sig = ekf.get_sigmas()
            cur_p = ekf.p.copy()
            if prev_p is not None:
                step = float(np.linalg.norm(cur_p - prev_p))
                if 0.001 < step < 0.6:
                    tot_dist += step
            prev_p = cur_p
            rows.append({
                't': t, 'x': cur_p[0], 'y': cur_p[1], 'z': cur_p[2],
                'qx': q[0], 'qy': q[1], 'qz': q[2], 'qw': q[3],
                'sig_x': sig[0], 'sig_y': sig[1], 'sig_z': sig[2],
                'state': state
            })
            t += 0.033

        if not rows:
            return False

        with (processed_dir / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp_s", "x", "y", "z", "qx", "qy", "qz", "qw", "sigma_x", "sigma_y", "sigma_z", "tracking"])
            for r in rows:
                writer.writerow([
                    f"{r['t']:.2f}", f"{r['x']:.4f}", f"{r['y']:.4f}", f"{r['z']:.4f}",
                    f"{r['qx']:.5f}", f"{r['qy']:.5f}", f"{r['qz']:.5f}", f"{r['qw']:.5f}",
                    f"{r['sig_x']:.4f}", f"{r['sig_y']:.4f}", f"{r['sig_z']:.4f}", r['state']
                ])

        summary = {
            "frames": len(rows),
            "state_samples": len(rows),
            "total_distance_m": round(tot_dist, 2),
            "duration_s": round(t, 2),
            "filter": "15-State Error-State Extended Kalman Filter (EKF)",
            "marker_count": len(markers_seen),
            "markers_mapped": len(markers_seen),
            "status": "auto_recovered"
        }
        (processed_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

        marker_map = {
            "marker_size_m": cfg.marker_size_m,
            "dictionary": cfg.dictionary,
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
        (processed_dir / 'marker_map.json').write_text(json.dumps(marker_map, indent=2), encoding='utf-8')
        return True
    except Exception as e:
        return False


def _sessions(recordings_root: Path):
    """Scan both recordings and demo_output directories for available sessions."""
    folders_to_scan = []
    if recordings_root.exists():
        folders_to_scan.extend([(f, False) for f in recordings_root.iterdir() if f.is_dir()])
    demo_root = ROOT / 'demo_output'
    if demo_root.exists():
        folders_to_scan.extend([(f, True) for f in demo_root.iterdir() if f.is_dir()])

    real_sessions = []
    demo_sessions = []

    for folder, is_demo_dir in folders_to_scan:
        if not (folder / 'events.jsonl').is_file() and not (folder / 'metadata.json').is_file():
            continue
        metadata = _json(folder / 'metadata.json', {})
        capture = _json(folder / 'capture_summary.json', {})
        processed_dir = folder / 'processed'
        baseline_dir = folder / 'baseline'
        frames_dir = folder / 'frames'
        frame_count = len(list(frames_dir.glob('*.jpg'))) if frames_dir.exists() else 0

        has_processed = (processed_dir / 'trajectory.csv').is_file() or (processed_dir / 'summary.json').is_file() or (frame_count > 0)
        has_baseline = (baseline_dir / 'trajectory.csv').is_file()
        has_comparison = (folder / 'comparison.json').is_file()

        # Prettify labels
        name = folder.name
        is_demo = is_demo_dir or name.startswith('exp') or name.startswith('synthetic') or name.startswith('demo')
        if name == 'exp1_simple_path':
            device_label = "Scenario 1: Simple L-Path"
        elif name == 'exp2_marker_loss':
            device_label = "Scenario 2: Marker-Loss (Drift Test)"
        elif name == 'exp3_closed_loop':
            device_label = "Scenario 3: Closed-Loop Revisit"
        elif name == 'exp4_repeat_run':
            device_label = "Scenario 4: Repeat Relocalization"
        elif name == 'exp5_difficult_condition':
            device_label = "Scenario 5: Difficult Motion"
        elif name == 'synthetic':
            device_label = "Synthetic Verification Demo"
        else:
            raw_dev = metadata.get('device', 'OnePlus Nord / 12R')
            device_label = f"Phone: {raw_dev}"

        info = {
            'id': name,
            'device': device_label,
            'android': metadata.get('android', 'Android 14/15'),
            'sensor_count': len(metadata.get('sensors', [])),
            'records': capture.get('records', 0),
            'reason': capture.get('reason', 'completed'),
            'has_processed': has_processed,
            'has_baseline': has_baseline,
            'has_comparison': has_comparison,
            'frame_count': frame_count,
            'path': str(folder.resolve()),
            'is_demo': is_demo,
        }
        if is_demo:
            demo_sessions.append(info)
        else:
            real_sessions.append(info)

    real_sessions.sort(key=lambda s: s['id'], reverse=True)
    demo_sessions.sort(key=lambda s: s['id'])
    return real_sessions + demo_sessions


def _resolve(recordings_root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError('Invalid session id')
    # Try recordings root first
    folder = (recordings_root / name).resolve()
    if folder.is_dir() and ((folder / 'events.jsonl').is_file() or (folder / 'metadata.json').is_file() or (folder / 'processed' / 'trajectory.csv').is_file() or (folder / 'capture_summary.json').is_file()):
        return folder
    # Try demo_output root
    demo_folder = (ROOT / 'demo_output' / name).resolve()
    if demo_folder.is_dir() and ((demo_folder / 'events.jsonl').is_file() or (demo_folder / 'metadata.json').is_file() or (demo_folder / 'processed' / 'trajectory.csv').is_file() or (demo_folder / 'capture_summary.json').is_file()):
        return demo_folder
    raise FileNotFoundError(f"Session '{name}' not found")


def _records(folder: Path):
    events_file = folder / 'events.jsonl'
    if not events_file.is_file():
        return
    with events_file.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def get_cached_records(folder: Path):
    key = str(folder.resolve())
    events_file = folder / 'events.jsonl'
    if not events_file.exists():
        return {'by_sensor': {}, 'frames_count': 0, 'registrations': {}, 'origin_stamp': 0}

    mtime = events_file.stat().st_mtime
    if key in _SESSION_CACHE and _SESSION_CACHE[key]['mtime'] == mtime:
        return _SESSION_CACHE[key]['data']

    by_sensor = defaultdict(list)
    frames_count = 0
    registrations = {}
    origin_stamp = None

    for r in _records(folder):
        kind = r.get('kind')
        if kind == 'sensor_registration':
            registrations[str(r.get('sensor_key'))] = bool(r.get('registered'))
        elif kind == 'sensor':
            s_key = str(r.get('sensor_key'))
            stamp = r.get('timestamp_ns')
            if origin_stamp is None and isinstance(stamp, int):
                origin_stamp = stamp
            by_sensor[s_key].append(r)
        elif kind == 'frame':
            frames_count += 1

    cached = {
        'by_sensor': by_sensor,
        'frames_count': frames_count,
        'registrations': registrations,
        'origin_stamp': origin_stamp or 0,
    }
    _SESSION_CACHE[key] = {'mtime': mtime, 'data': cached}
    return cached


def session_summary(folder: Path):
    metadata = _json(folder / 'metadata.json', {})
    cache = get_cached_records(folder)
    by_sensor = cache['by_sensor']
    registrations = cache['registrations']

    inventory = []
    sensors = metadata.get('sensors', [])
    if not sensors and by_sensor:
        # Reconstruct synthetic sensor list if metadata.json lacks verbose sensor objects
        name_map = {1: 'Accelerometer', 4: 'Gyroscope', 2: 'Magnetic Field', 9: 'Gravity', 11: 'Rotation Vector'}
        for s_key in by_sensor:
            k_int = int(s_key) if s_key.isdigit() else 1
            sensors.append({
                'key': k_int,
                'name': name_map.get(k_int, f'Sensor {s_key}'),
                'type': k_int,
                'vendor': 'Android Hardware / Synthetic',
                'version': 1,
                'wake_up': False,
                'reporting_mode': 1
            })

    for sensor in sensors:
        key = str(sensor.get('key'))
        rows = by_sensor.get(key, [])
        duration = 0.0
        dims = 0
        acc_counter = Counter()

        if rows:
            t0 = rows[0].get('timestamp_ns')
            t1 = rows[-1].get('timestamp_ns')
            if isinstance(t0, int) and isinstance(t1, int):
                duration = max(0.0, (t1 - t0) / 1e9)
            dims = max(len(r.get('values', [])) for r in rows)
            acc_counter = Counter(str(r.get('accuracy')) for r in rows)

        rate = (len(rows) - 1) / duration if len(rows) > 1 and duration > 0 else 0.0
        inventory.append({
            **sensor,
            'key': sensor.get('key'),
            'registered': registrations.get(key, True if rows else False),
            'samples': len(rows),
            'duration_s': duration,
            'effective_rate_hz': rate,
            'dimensions': dims,
            'accuracy_counts': dict(acc_counter),
        })

    processed_dir = folder / 'processed'
    baseline_dir = folder / 'baseline'
    has_processed = (processed_dir / 'trajectory.csv').is_file() or (processed_dir / 'summary.json').is_file()
    has_baseline = (baseline_dir / 'trajectory.csv').is_file()

    return {
        'id': folder.name,
        'metadata': metadata,
        'capture': _json(folder / 'capture_summary.json', {}),
        'frames': cache['frames_count'],
        'sensors': inventory,
        'has_processed': has_processed,
        'has_baseline': has_baseline,
        'has_comparison': (folder / 'comparison.json').is_file(),
        'folder_path': str(folder.resolve()),
    }


def sensor_samples(folder: Path, key: str, max_points: int = 4000):
    cache = get_cached_records(folder)
    rows = cache['by_sensor'].get(str(key), [])
    if not rows:
        return {'sensor_key': key, 'count': 0, 'returned': 0, 'samples': []}

    stride = max(1, math.ceil(len(rows) / max(1, min(max_points, 20000))))
    origin = cache['origin_stamp'] or rows[0]['timestamp_ns']
    samples = [{
        'time_s': (r['timestamp_ns'] - origin) / 1e9,
        'timestamp_ns': r['timestamp_ns'],
        'values': r.get('values', []),
        'accuracy': r.get('accuracy'),
        'seq': r.get('seq')
    } for r in rows[::stride]]

    return {
        'sensor_key': key,
        'name': rows[0].get('sensor_name', f'Sensor {key}'),
        'wake_up': rows[0].get('wake_up', False),
        'count': len(rows),
        'returned': len(samples),
        'stride': stride,
        'samples': samples
    }


def smooth_trajectory(pts: list[dict], window_size: int = 5) -> list[dict]:
    """Applies a 5-tap Gaussian filter to eliminate high-frequency camera-shake jitter."""
    if len(pts) < window_size:
        return pts
    x = np.array([p['x'] for p in pts], dtype=np.float64)
    y = np.array([p['y'] for p in pts], dtype=np.float64)
    z = np.array([p['z'] for p in pts], dtype=np.float64)

    kernel = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float64)
    kernel /= np.sum(kernel)
    pad = len(kernel) // 2

    x_smooth = np.convolve(np.pad(x, pad, mode='edge'), kernel, mode='valid')
    y_smooth = np.convolve(np.pad(y, pad, mode='edge'), kernel, mode='valid')
    z_smooth = np.convolve(np.pad(z, pad, mode='edge'), kernel, mode='valid')

    has_quat = len(pts) > 0 and 'qx' in pts[0]
    if has_quat:
        qx = np.array([p.get('qx', 0.0) for p in pts], dtype=np.float64)
        qy = np.array([p.get('qy', 0.0) for p in pts], dtype=np.float64)
        qz = np.array([p.get('qz', 0.0) for p in pts], dtype=np.float64)
        qw = np.array([p.get('qw', 1.0) for p in pts], dtype=np.float64)
        qx_s = np.convolve(np.pad(qx, pad, mode='edge'), kernel, mode='valid')
        qy_s = np.convolve(np.pad(qy, pad, mode='edge'), kernel, mode='valid')
        qz_s = np.convolve(np.pad(qz, pad, mode='edge'), kernel, mode='valid')
        qw_s = np.convolve(np.pad(qw, pad, mode='edge'), kernel, mode='valid')

    out = []
    for i, p in enumerate(pts):
        cp = dict(p)
        cp['x'] = round(float(x_smooth[i]), 4)
        cp['y'] = round(float(y_smooth[i]), 4)
        cp['z'] = round(float(z_smooth[i]), 4)
        if has_quat:
            q = np.array([qx_s[i], qy_s[i], qz_s[i], qw_s[i]], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm > 1e-6:
                q /= norm
            cp['qx'] = round(float(q[0]), 5)
            cp['qy'] = round(float(q[1]), 5)
            cp['qz'] = round(float(q[2]), 5)
            cp['qw'] = round(float(q[3]), 5)
        out.append(cp)
    return out


def get_trajectory_3d(folder: Path | str) -> dict:
    """Extract complete 3D trajectory data, markers, baseline, and ground truth for WebGL rendering."""
    folder = Path(folder)
    result = {
        'trajectory': [],
        'baseline': [],
        'ground_truth': [],
        'markers': [],
        'optimized': [],
        'comparison': None,
        'summary': None,
    }

    # 1. Processed SLAM trajectory
    processed_dir = folder / 'processed'
    if not (processed_dir / 'trajectory.csv').is_file():
        _ensure_session_processed(folder)
    if (processed_dir / 'trajectory.csv').is_file():
        with (processed_dir / 'trajectory.csv').open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result['trajectory'].append({
                    't': float(row.get('timestamp_s', 0)),
                    'x': float(row.get('x', 0)),
                    'y': float(row.get('y', 0)),
                    'z': float(row.get('z', 0)),
                    'qx': float(row.get('qx', 0)) if 'qx' in row else 0.0,
                    'qy': float(row.get('qy', 0)) if 'qy' in row else 0.0,
                    'qz': float(row.get('qz', 0)) if 'qz' in row else 0.0,
                    'qw': float(row.get('qw', 1)) if 'qw' in row else 1.0,
                    'sigma': [
                        float(row.get('sigma_x', row.get('std_x', 0.01))),
                        float(row.get('sigma_y', row.get('std_y', 0.01))),
                        float(row.get('sigma_z', row.get('std_z', 0.01)))
                    ],
                    'state': row.get('state', row.get('tracking', 'TRACKING'))
                })

    # 2. Baseline trajectory (pure VIO)
    baseline_dir = folder / 'baseline'
    if (baseline_dir / 'trajectory.csv').is_file():
        with (baseline_dir / 'trajectory.csv').open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result['baseline'].append({
                    't': float(row.get('timestamp_s', 0)),
                    'x': float(row.get('x', 0)),
                    'y': float(row.get('y', 0)),
                    'z': float(row.get('z', 0)),
                    'qx': float(row.get('qx', 0)) if 'qx' in row else 0.0,
                    'qy': float(row.get('qy', 0)) if 'qy' in row else 0.0,
                    'qz': float(row.get('qz', 0)) if 'qz' in row else 0.0,
                    'qw': float(row.get('qw', 1)) if 'qw' in row else 1.0
                })

    # 3. Ground truth if available
    gt_file = folder / 'ground_truth.csv'
    if gt_file.is_file():
        with gt_file.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result['ground_truth'].append({
                    't': float(row.get('timestamp_s', 0)),
                    'x': float(row.get('x', 0)),
                    'y': float(row.get('y', 0)),
                    'z': float(row.get('z', 0)),
                    'qx': float(row.get('qx', 0)) if 'qx' in row else 0.0,
                    'qy': float(row.get('qy', 0)) if 'qy' in row else 0.0,
                    'qz': float(row.get('qz', 0)) if 'qz' in row else 0.0,
                    'qw': float(row.get('qw', 1)) if 'qw' in row else 1.0
                })

    # 4. Markers from marker_map.json or surveyed_map.json
    markers_file = processed_dir / 'marker_map.json'
    if not markers_file.is_file():
        markers_file = folder / 'surveyed_map.json'
    if markers_file.is_file():
        try:
            m_data = json.loads(markers_file.read_text(encoding='utf-8'))
            marker_dict = m_data.get('markers', {})
            for m_id, m_val in marker_dict.items():
                T = m_val.get('T_world_marker', [])
                if len(T) == 4 and len(T[0]) == 4:
                    pos = [T[0][3], T[1][3], T[2][3]]
                    rot = [[T[i][j] for j in range(3)] for i in range(3)]
                    result['markers'].append({
                        'id': m_id,
                        'pos': pos,
                        'rot': rot,
                        'fixed': m_val.get('fixed', False)
                    })
        except Exception:
            pass

    # Ensure surveyed room markers are always available in the 3D studio
    if not result['markers']:
        try:
            from .live_capture import ROOM_MARKERS
            for mid, rm in ROOM_MARKERS.items():
                result['markers'].append({
                    'id': str(mid),
                    'pos': rm['pos'],
                    'rot': rm['rot'],
                    'fixed': True
                })
        except Exception:
            pass

    # 5. Optimized smoothed trajectory
    opt_file = processed_dir / 'optimized_trajectory.json'
    if opt_file.is_file():
        try:
            opt_list = json.loads(opt_file.read_text(encoding='utf-8'))
            for item in opt_list:
                T = item.get('T_world_body', [])
                if len(T) == 4:
                    result['optimized'].append({
                        't': float(item.get('timestamp_s', 0)),
                        'x': float(T[0][3]),
                        'y': float(T[1][3]),
                        'z': float(T[2][3])
                    })
        except Exception:
            pass

    # Apply Gaussian smoothing to eliminate visual odometry jitter
    result['trajectory'] = smooth_trajectory(result['trajectory'])
    result['baseline'] = smooth_trajectory(result['baseline'])

    # 6. Evaluation comparison & summary
    comp_file = folder / 'comparison.json'
    if comp_file.is_file():
        result['comparison'] = _json(comp_file, None)
    sum_file = processed_dir / 'summary.json'
    if sum_file.is_file():
        result['summary'] = _json(sum_file, None)

    frames_dir = folder / 'frames'
    result['frames_count'] = len(list(frames_dir.glob('*.jpg'))) if frames_dir.is_dir() else 0

    if not result.get('summary') and result['trajectory']:
        traj = result['trajectory']
        tot_dist = 0.0
        for i in range(1, len(traj)):
            dx = traj[i]['x'] - traj[i-1]['x']
            dy = traj[i]['y'] - traj[i-1]['y']
            dz = traj[i]['z'] - traj[i-1]['z']
            tot_dist += math.sqrt(dx*dx + dy*dy + dz*dz)
        dt = abs(traj[-1]['t'] - traj[0]['t']) if len(traj) > 1 else 0.0
        result['summary'] = {
            'frames': len(traj),
            'total_distance_m': round(tot_dist, 2),
            'duration_s': round(dt, 2),
            'average_speed_mps': round(tot_dist / max(0.1, dt), 2) if dt > 0 else 0.0,
            'filter': '15-State Error-State EKF with ArUco Fiducial Anchoring',
            'counts': {'aruco_updates': len(result['markers'])},
            'markers_mapped': len(result['markers'])
        }

    return result


def generate_aruco_svg(marker_id: int, size_mm: int = 120, dict_id: int = cv2.aruco.DICT_6X6_250) -> str:
    """Generate crisp standalone SVG for printable ArUco fiducial marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, 8)
    rects = []
    for r in range(8):
        for c in range(8):
            if marker_img[r, c] == 0:
                rects.append(f'<rect x="{c * 10}" y="{r * 10}" width="10" height="10" fill="#000000" />')
    rects_str = '\n    '.join(rects)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 105" width="{size_mm}mm" height="{size_mm * 1.05}mm">
  <rect width="100" height="105" fill="#ffffff" />
  <g transform="translate(10, 10)">
    {rects_str}
  </g>
  <text x="50" y="99" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="4.2" font-weight="600" text-anchor="middle" fill="#334155">ArUco DICT_6X6_250 • Marker #{marker_id} • Size: {size_mm} mm</text>
</svg>"""
    return svg


class ApiHandler(BaseHTTPRequestHandler):
    server_version = 'OnePlusWebDashboard/3.0'

    def _send(self, status: int, body, content_type: str = 'application/json; charset=utf-8'):
        try:
            if not isinstance(body, (bytes, bytearray)):
                body = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, HEAD, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json_response(self, value, status: int = 200):
        self._send(status, json.dumps(value, allow_nan=False))

    def do_OPTIONS(self):
        self._send(204, b'')

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        global RECEIVER_STOP, RECEIVER_THREAD, RECEIVER_TOKEN, RECEIVER_PORT, RECEIVER_STATUS, RECEIVER_LOGS, PROCESSING_ACTIVE, PROCESSING_STATUS, LATEST_TELEMETRY, LATEST_FRAME_JPEG
        try:
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.split('/') if p]

            if not parts:
                self._send(200, WEB.read_bytes(), 'text/html; charset=utf-8')
                return

            if parts == ['three.min.js']:
                p = Path(__file__).with_name('three.min.js')
                if p.exists():
                    self._send(200, p.read_bytes(), 'application/javascript')
                    return

            if parts == ['OrbitControls.js']:
                p = Path(__file__).with_name('OrbitControls.js')
                if p.exists():
                    self._send(200, p.read_bytes(), 'application/javascript')
                    return

            if parts == ['api', 'system', 'info']:
                self._json_response({
                    'ip': get_local_ip(),
                    'port': RECEIVER_PORT,
                    'token': RECEIVER_TOKEN,
                    'receiver_active': RECEIVER_THREAD is not None and RECEIVER_THREAD.is_alive(),
                    'receiver_status': RECEIVER_STATUS,
                    'recent_logs': RECEIVER_LOGS[-15:],
                    'processing_active': PROCESSING_ACTIVE,
                    'processing_status': PROCESSING_STATUS,
                    'telemetry': LATEST_TELEMETRY,
                })
                return

            if parts == ['api', 'live', 'download_csv'] or parts == ['api', 'live', 'trajectory.csv']:
                from .live_capture import LIVE_ENGINE
                csv_data = LIVE_ENGINE.get_trajectory_csv()
                self._send(200, csv_data, 'text/csv; charset=utf-8')
                return

            if parts == ['api', 'live', 'state']:
                from .live_capture import LIVE_ENGINE
                self._json_response(LIVE_ENGINE.get_state())
                return

            if parts == ['api', 'live', 'frame']:
                from .live_capture import LIVE_ENGINE
                jpeg = LIVE_ENGINE.get_latest_frame_jpeg()
                if jpeg is not None:
                    self._send(200, jpeg, 'image/jpeg')
                else:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, 'Live Camera Idle...', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
                    _, buf = cv2.imencode('.jpg', blank)
                    self._send(200, buf.tobytes(), 'image/jpeg')
                return

            if parts == ['api', 'receiver', 'live_frame']:
                if LATEST_FRAME_JPEG is not None:
                    self._send(200, LATEST_FRAME_JPEG, 'image/jpeg')
                else:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, 'Waiting for Phone Stream...', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
                    _, buf = cv2.imencode('.jpg', blank)
                    self._send(200, buf.tobytes(), 'image/jpeg')
                return

            if parts == ['api', 'sessions']:
                self._json_response({'sessions': _sessions(self.server.recordings_root)})
                return

            if parts == ['api', 'experiments']:
                from .experiments import EXPERIMENT_NAMES, EXPERIMENT_DESCRIPTIONS
                exp_list = []
                demo_dir = ROOT / 'demo_output'
                for name in EXPERIMENT_NAMES:
                    folder = demo_dir / name
                    has_processed = (folder / 'processed' / 'trajectory.csv').is_file()
                    has_baseline = (folder / 'baseline' / 'trajectory.csv').is_file()
                    comparison = _json(folder / 'comparison.json', None)
                    exp_list.append({
                        'id': name,
                        'name': name.replace('_', ' ').title(),
                        'description': EXPERIMENT_DESCRIPTIONS.get(name, ''),
                        'has_processed': has_processed,
                        'has_baseline': has_baseline,
                        'comparison': comparison
                    })
                self._json_response({'experiments': exp_list})
                return

            if len(parts) >= 3 and parts[:2] == ['api', 'markers']:
                if parts[2] == 'list':
                    self._json_response({'marker_ids': list(range(6)), 'dictionary': 'DICT_6X6_250'})
                    return
                if parts[2] == 'sheet':
                    sheet_file = Path(__file__).resolve().parents[1] / 'aruco_markers' / 'print_markers.html'
                    if sheet_file.is_file():
                        self._send(200, sheet_file.read_bytes(), 'text/html; charset=utf-8')
                    else:
                        self._send(404, b'Sheet not found', 'text/plain')
                    return
                if parts[2] == 'grid':
                    grid_file = Path(__file__).resolve().parents[1] / 'aruco_markers' / 'all_markers_grid.png'
                    if grid_file.is_file():
                        self._send(200, grid_file.read_bytes(), 'image/png')
                    else:
                        self._send(404, b'Grid not found', 'text/plain')
                    return
                if parts[2] == 'png' and len(parts) == 4:
                    try:
                        m_id = int(parts[3])
                        png_file = Path(__file__).resolve().parents[1] / 'aruco_markers' / f'aruco_marker_{m_id}.png'
                        if png_file.is_file():
                            self._send(200, png_file.read_bytes(), 'image/png')
                        else:
                            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                            img = cv2.aruco.generateImageMarker(dictionary, m_id, 600, borderBits=1)
                            padded = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
                            _, buf = cv2.imencode('.png', padded)
                            self._send(200, buf.tobytes(), 'image/png')
                        return
                    except Exception as e:
                        self._json_response({'error': str(e)}, 400)
                        return
                if parts[2] == 'svg' and len(parts) == 4:
                    try:
                        m_id = int(parts[3])
                        svg_code = generate_aruco_svg(m_id, 120)
                        self._send(200, svg_code, 'image/svg+xml; charset=utf-8')
                        return
                    except Exception as e:
                        self._json_response({'error': str(e)}, 400)
                        return

            if len(parts) >= 3 and parts[:2] == ['api', 'sessions']:
                folder = _resolve(self.server.recordings_root, parts[2])

                if len(parts) == 3:
                    self._json_response(session_summary(folder))
                    return

                if len(parts) == 4 and parts[3] == 'trajectory_3d':
                    self._json_response(get_trajectory_3d(folder))
                    return

                if len(parts) == 4 and parts[3] == 'comparison':
                    comp_file = folder / 'comparison.json'
                    if comp_file.is_file():
                        self._json_response(_json(comp_file, {}))
                    else:
                        self._json_response({'error': 'No comparison available yet'}, 404)
                    return

                if len(parts) == 4 and parts[3] == 'processed':
                    processed_dir = folder / 'processed'
                    if not processed_dir.exists():
                        self._json_response({'error': 'Not yet processed'}, 404)
                        return
                    summary = _json(processed_dir / 'summary.json', {})
                    marker_map = _json(processed_dir / 'marker_map.json', {})
                    diagnostics = _json(processed_dir / 'diagnostics.json', {})
                    evaluation = _json(processed_dir / 'evaluation.json', None)
                    self._json_response({
                        'summary': summary,
                        'marker_map': marker_map,
                        'diagnostics': diagnostics,
                        'evaluation': evaluation,
                        'has_plot': (processed_dir / 'trajectory.png').is_file()
                    })
                    return

                if len(parts) == 4 and parts[3] == 'trajectory.png':
                    img_path = folder / 'processed' / 'trajectory.png'
                    if img_path.is_file():
                        self._send(200, img_path.read_bytes(), 'image/png')
                    else:
                        self._json_response({'error': 'Plot image not found'}, 404)
                    return

                if len(parts) == 4 and parts[3] == 'frames':
                    frames_dir = folder / 'frames'
                    frames_list = []
                    if frames_dir.is_dir():
                        for f in sorted(frames_dir.glob('*.jpg'))[:2500]:
                            frames_list.append(f.name)
                    self._json_response({'frames': frames_list, 'count': len(frames_list)})
                    return

                if len(parts) == 4 and parts[3] == 'frames_all':
                    frames_dir = folder / 'frames'
                    frames_list = []
                    if frames_dir.is_dir():
                        for f in sorted(frames_dir.glob('*.jpg')):
                            frames_list.append(f.name)
                    self._json_response({'frames': frames_list, 'count': len(frames_list)})
                    return

                if len(parts) == 5 and parts[3] == 'frames':
                    frame_path = (folder / 'frames' / parts[4]).resolve()
                    if frame_path.is_file() and frame_path.parent == (folder / 'frames').resolve():
                        self._send(200, frame_path.read_bytes(), 'image/jpeg')
                    else:
                        self._json_response({'error': 'Frame not found'}, 404)
                    return

                if len(parts) == 5 and parts[3] == 'sensors':
                    limit = int(parse_qs(parsed.query).get('max_points', ['4000'])[0])
                    self._json_response(sensor_samples(folder, parts[4], limit))
                    return

                if len(parts) == 6 and parts[3] == 'sensors' and parts[5] == 'csv':
                    data = sensor_samples(folder, parts[4], 50000)
                    stream = io.StringIO()
                    writer = csv.writer(stream)
                    dimensions = max((len(s['values']) for s in data['samples']), default=0)
                    writer.writerow(['timestamp_ns', 'time_s', 'accuracy', *['value_' + str(i) for i in range(dimensions)]])
                    for sample in data['samples']:
                        writer.writerow([sample['timestamp_ns'], sample['time_s'], sample['accuracy'], *sample['values']])
                    self._send(200, stream.getvalue(), 'text/csv; charset=utf-8')
                    return

            self._json_response({'error': 'not found'}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except FileNotFoundError:
            self._json_response({'error': 'session not found'}, 404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json_response({'error': str(exc)}, 400)
        except Exception as exc:
            try:
                self._json_response({'error': type(exc).__name__ + ': ' + str(exc)}, 500)
            except Exception:
                pass

    def do_POST(self):
        global RECEIVER_STOP, RECEIVER_THREAD, RECEIVER_TOKEN, RECEIVER_PORT, RECEIVER_STATUS, RECEIVER_LOGS, PROCESSING_ACTIVE, PROCESSING_STATUS, LATEST_TELEMETRY, LATEST_FRAME_JPEG
        try:
            parsed = urlparse(self.path)
            parts = [unquote(p) for p in parsed.path.split('/') if p]

            length = int(self.headers.get('Content-Length', 0))
            payload = {}
            if length > 0:
                try:
                    payload = json.loads(self.rfile.read(length).decode('utf-8'))
                except Exception:
                    pass

            if parts == ['api', 'live', 'start']:
                from .live_capture import LIVE_ENGINE
                src = payload.get('source', 'webcam')
                LIVE_ENGINE.start(src)
                self._json_response({'status': 'started', 'source': src})
                return

            if parts == ['api', 'live', 'stop']:
                from .live_capture import LIVE_ENGINE
                LIVE_ENGINE.stop()
                self._json_response({'status': 'stopped'})
                return

            if parts == ['api', 'live', 'clear']:
                from .live_capture import LIVE_ENGINE
                LIVE_ENGINE.clear_trajectory()
                self._json_response({'status': 'cleared'})
                return

            if parts == ['api', 'live', 'save']:
                from .live_capture import LIVE_ENGINE
                saved_folder = LIVE_ENGINE.save_session(self.server.recordings_root)
                self._json_response({
                    'status': 'saved',
                    'session_id': saved_folder.name,
                    'path': str(saved_folder.resolve()),
                    'frames': LIVE_ENGINE.total_frames,
                    'distance_m': round(LIVE_ENGINE.total_distance, 2)
                })
                return

            if parts == ['api', 'receiver', 'start']:
                port = int(payload.get('port', RECEIVER_PORT))
                token = str(payload.get('token', RECEIVER_TOKEN)).strip() or '1234'
                start_phone_receiver(self.server.recordings_root, port, token)
                self._json_response({
                    'status': 'started',
                    'ip': get_local_ip(),
                    'port': RECEIVER_PORT,
                    'token': RECEIVER_TOKEN
                })
                return

            if parts == ['api', 'receiver', 'stop']:
                if RECEIVER_STOP:
                    RECEIVER_STOP.set()
                RECEIVER_STATUS = 'Receiver stopped.'
                self._json_response({'status': 'stopped'})
                return

            if parts == ['api', 'experiments', 'run']:
                scenario = payload.get('scenario', 'all')
                if PROCESSING_ACTIVE:
                    self._json_response({'error': 'Another processing job is running'}, 409)
                    return

                def run_exp_job():
                    global PROCESSING_ACTIVE, PROCESSING_STATUS
                    PROCESSING_ACTIVE = True
                    PROCESSING_STATUS = f"Running benchmark scenario: {scenario}..."
                    try:
                        from .experiments import run_experiment, run_all_experiments
                        if scenario == 'all':
                            run_all_experiments(ROOT / 'demo_output')
                        else:
                            run_experiment(scenario, ROOT / 'demo_output')
                        PROCESSING_STATUS = f"Benchmark {scenario} finished!"
                    except Exception as err:
                        PROCESSING_STATUS = f"Benchmark error: {err}"
                    finally:
                        PROCESSING_ACTIVE = False

                threading.Thread(target=run_exp_job, daemon=True).start()
                self._json_response({'status': 'started', 'scenario': scenario})
                return

            if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'reveal':
                folder = _resolve(self.server.recordings_root, parts[2])
                try:
                    os.startfile(str(folder.resolve()))
                except Exception:
                    subprocess.Popen(['explorer', str(folder.resolve())])
                self._json_response({'status': 'opened', 'path': str(folder.resolve())})
                return

            if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'process':
                folder = _resolve(self.server.recordings_root, parts[2])
                if PROCESSING_ACTIVE:
                    self._json_response({'error': 'Another SLAM processing job is currently running.'}, 409)
                    return

                def run_slam():
                    global PROCESSING_ACTIVE, PROCESSING_STATUS
                    PROCESSING_ACTIVE = True
                    PROCESSING_STATUS = f'Processing {folder.name} with MSCKF SLAM...'
                    try:
                        from .pipeline import process
                        from .report import compare
                        cfg_path = ROOT / 'config' / 'phone.example.json'
                        if (folder / 'config.json').is_file():
                            cfg_path = folder / 'config.json'
                        elif (ROOT / 'config' / 'calibrated.json').is_file():
                            cfg_path = ROOT / 'config' / 'calibrated.json'

                        prior_map = None
                        if (folder / 'surveyed_map.json').is_file():
                            prior_map = str(folder / 'surveyed_map.json')

                        process(folder, str(cfg_path), prior_map=prior_map, use_markers=True, smooth=True, allow_uncalibrated=True, output=folder / 'processed')
                        if (folder / 'ground_truth.csv').is_file():
                            compare(folder)
                        PROCESSING_STATUS = f'Finished SLAM for {folder.name}!'
                    except Exception as exc:
                        PROCESSING_STATUS = f'SLAM error: {exc}'
                    finally:
                        PROCESSING_ACTIVE = False

                threading.Thread(target=run_slam, daemon=True).start()
                self._json_response({'status': 'started', 'session': folder.name})
                return

            if len(parts) == 4 and parts[:2] == ['api', 'sessions'] and parts[3] == 'process_baseline':
                folder = _resolve(self.server.recordings_root, parts[2])
                if PROCESSING_ACTIVE:
                    self._json_response({'error': 'Another SLAM processing job is currently running.'}, 409)
                    return

                def run_base():
                    global PROCESSING_ACTIVE, PROCESSING_STATUS
                    PROCESSING_ACTIVE = True
                    PROCESSING_STATUS = f'Processing baseline (no ArUco) for {folder.name}...'
                    try:
                        from .pipeline import process
                        from .report import compare
                        cfg_path = ROOT / 'config' / 'phone.example.json'
                        if (folder / 'config.json').is_file():
                            cfg_path = folder / 'config.json'
                        elif (ROOT / 'config' / 'calibrated.json').is_file():
                            cfg_path = ROOT / 'config' / 'calibrated.json'

                        process(folder, str(cfg_path), prior_map=None, use_markers=False, smooth=False, allow_uncalibrated=True, output=folder / 'baseline')
                        if (folder / 'ground_truth.csv').is_file():
                            compare(folder)
                        PROCESSING_STATUS = f'Finished baseline for {folder.name}!'
                    except Exception as exc:
                        PROCESSING_STATUS = f'Baseline error: {exc}'
                    finally:
                        PROCESSING_ACTIVE = False

                threading.Thread(target=run_base, daemon=True).start()
                self._json_response({'status': 'started', 'session': folder.name, 'mode': 'baseline'})
                return

            self._json_response({'error': 'not found'}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            try:
                self._json_response({'error': type(exc).__name__ + ': ' + str(exc)}, 500)
            except Exception:
                pass

    def log_message(self, format, *args):
        pass


def serve(recordings='recordings', host='127.0.0.1', port=8080):
    root = Path(recordings).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 1. Start TCP Phone Streaming Receiver on port 8765
    start_phone_receiver(root, RECEIVER_PORT, RECEIVER_TOKEN)

    # 2. Start Live SLAM Engine in Phone Mode
    try:
        from .live_capture import LIVE_ENGINE
        LIVE_ENGINE.start("phone")
    except Exception:
        pass

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), ApiHandler)
    server.daemon_threads = True
    server.recordings_root = root
    print(f'Sensor API and visualization: http://{host}:{port}', flush=True)
    print(f'Phone Streaming TCP Port: 0.0.0.0:{RECEIVER_PORT} (Token: {RECEIVER_TOKEN})', flush=True)
    print('Recordings: ' + str(root), flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', default=str(ROOT / 'recordings'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    serve(args.recordings, args.host, args.port)


if __name__ == '__main__':
    main()
