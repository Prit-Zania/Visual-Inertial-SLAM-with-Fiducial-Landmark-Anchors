"""ArUco Marker Real-Time Verification and Live Camera Testing Tool.

Usage:
  1. Test against generated marker images:
     python test_aruco_camera.py --test-images

  2. Test with live webcam feed:
     python test_aruco_camera.py --webcam

Press 'q' or 'ESC' to exit the live webcam window.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slam.config import Config
from slam.live_capture import ROOM_MARKERS, ROOM_DIMENSIONS


def test_images():
    """Verify ArUco detection and 3D pose estimation across all generated markers."""
    print("==================================================================")
    print("ArUco Verification Test: Dictionary DICT_6X6_250 (120mm Markers)")
    print("==================================================================")

    cfg = Config()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary)
    K, D = cfg.intrinsics(720, 790)

    half_s = cfg.marker_size_m / 2.0
    obj_pts = np.array([
        [-half_s, half_s, 0.0],
        [half_s, half_s, 0.0],
        [half_s, -half_s, 0.0],
        [-half_s, -half_s, 0.0]
    ], dtype=np.float64)

    markers_dir = Path(__file__).resolve().parent / "aruco_markers"
    if not markers_dir.is_dir():
        print(f"Error: Directory {markers_dir} does not exist.")
        return False

    success_count = 0
    for mid in range(6):
        img_path = markers_dir / f"aruco_marker_{mid}.png"
        if not img_path.is_file():
            print(f"[-] Missing file: {img_path.name}")
            continue

        bgr = cv2.imread(str(img_path))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0 and int(ids.ravel()[0]) == mid:
            uv = corners[0].reshape(4, 2).astype(np.float64)
            ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, uv, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ret:
                t_vec = tvecs[0].ravel()
                dist = float(np.linalg.norm(t_vec))
                rm = ROOM_MARKERS.get(mid, {})
                wall_name = rm.get("name", "Unknown Wall")
                survey_pos = rm.get("pos", [0, 0, 0])
                print(f"[+] Marker #{mid:02d} DETECTED! Distance: {dist:.2f}m | Surveyed {wall_name} at {survey_pos}")
                success_count += 1
            else:
                print(f"[!] Marker #{mid:02d} detected, but solvePnP failed.")
        else:
            print(f"[-] Marker #{mid:02d} was NOT detected in {img_path.name}")

    print("------------------------------------------------------------------")
    print(f"Detection Results: {success_count} / 6 markers detected successfully.")
    return success_count == 6


def run_webcam():
    """Run live webcam ArUco detection window with 3D coordinate axes."""
    print("==================================================================")
    print("Starting Live Webcam ArUco Detection (DICT_6X6_250)")
    print("Point your camera at any ArUco marker #0 to #5.")
    print("Press 'q' or 'ESC' in the preview window to exit.")
    print("==================================================================")

    cfg = Config()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[!] Error: Could not open webcam device (index 0).")
        print("    If you don't have a webcam, use phone streaming or run with --test-images.")
        return

    K, D = cfg.intrinsics(640, 480)
    half_s = cfg.marker_size_m / 2.0
    obj_pts = np.array([
        [-half_s, half_s, 0.0],
        [half_s, half_s, 0.0],
        [half_s, -half_s, 0.0],
        [-half_s, -half_s, 0.0]
    ], dtype=np.float64)

    fps = 0.0
    last_t = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, now - last_t))
            last_t = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # HUD Background
            cv2.rectangle(frame, (10, 10), (450, 90), (15, 23, 42), -1)
            cv2.rectangle(frame, (10, 10), (450, 90), (56, 189, 248), 1)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                detected_list = []
                for m_id, m_corners in zip(ids.ravel(), corners):
                    m_id_int = int(m_id)
                    uv = m_corners.reshape(4, 2).astype(np.float64)
                    ret_pnp, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, uv, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ret_pnp and len(tvecs) > 0:
                        cv2.drawFrameAxes(frame, K, D, rvecs[0], tvecs[0], 0.08)
                        dist = float(np.linalg.norm(tvecs[0]))
                        rm = ROOM_MARKERS.get(m_id_int, {})
                        wall_str = rm.get("wall", "unknown").upper()
                        detected_list.append(f"#{m_id_int} ({wall_str} @ {dist:.2f}m)")

                cv2.putText(frame, f"STATUS: ANCHORED (ArUco Visible)", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52, 211, 153), 2)
                cv2.putText(frame, f"Markers: {', '.join(detected_list)}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1)
                cv2.putText(frame, f"FPS: {fps:.1f} | Resolution: 640x480", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1)
            else:
                cv2.putText(frame, "STATUS: SEARCHING FOR ARUCO MARKERS...", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1)
                cv2.putText(frame, "Show marker #0-#5 (DICT_6X6_250) to camera", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1)
                cv2.putText(frame, f"FPS: {fps:.1f} | Press 'q' to exit", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1)

            cv2.imshow("ArUco Live Camera Detection - Visual-Inertial SLAM", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Verify ArUco Markers")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam detection window")
    parser.add_argument("--test-images", action="store_true", help="Test detection across all generated marker files")

    args = parser.parse_args()
    if args.webcam:
        run_webcam()
    else:
        # Default runs image test
        test_images()
