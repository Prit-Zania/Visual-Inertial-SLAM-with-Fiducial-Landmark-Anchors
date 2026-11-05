# Estimator and data design

## State and frames

The navigation state is `(R_WB, p_WB, v_WB, b_g, b_a)`. The 15-vector error is ordered `(delta_theta_B, delta_p_W, delta_v_W, delta_bg, delta_ba)`. Rotation errors multiply on the right. Each camera clone and unknown marker adds six errors `(delta_theta, delta_p_W)` to the covariance, with full cross-correlations.

`T_WC = T_WB T_BC`. A marker observation gives `T_CM`. A surveyed marker supplies `T_WC = T_WM inverse(T_CM)`. A new marker is initialized with `T_WM = T_WC T_CM`. Augmentation includes the navigation Jacobian and observation uncertainty. The first observation is not immediately consumed again as an independent update.

When the first marker defines the world frame, the navigation states, camera clones, gravity and saved earlier poses are transformed together. Known map coordinates remain in their supplied frame. World orientation is arbitrary in the marker-origin case; gravity need not equal world negative Z after that transformation.

## Inertial prediction

The filter integrates bias-corrected angular rate on SO(3), uses a midpoint rotation for specific force, and integrates position/velocity with gravity. Covariance is propagated through the linearized 15-state model, including sensor noise and bias random walks. The first stationary window estimates gyro bias, gravity alignment and only the observable radial component of accelerometer bias. It does not claim full accelerometer calibration from one pose.

A stale or missing IMU measurement is not treated as a new measurement. Missing acceleration uses constant velocity with enlarged uncertainty; fresh gyro can still propagate rotation. Missing gyro increases uncertainty and leaves visual/marker updates to constrain rotation. If neither camera nor suitable constraints are available long enough, status becomes LOST.

## MSCKF visual update

1. Clone the camera pose at each image time. Covariance augmentation preserves its correlation with the navigation state and all landmarks.
2. Track natural corners using pyramidal KLT, a backward-flow check and RANSAC. ArUco image areas are excluded from new feature detection to reduce duplicate information.
3. Finish a track when lost or when its oldest clone must leave the window. Triangulate only with sufficient parallax and positive depth.
4. Form `r = H_x delta_x + H_f delta_feature + noise`.
5. Project through the left nullspace of `H_f`, eliminating the feature position from the update.
6. Apply per-track and stacked innovation gating, orthogonal compression when useful, and a Joseph-form covariance update.
7. Inject corrections, reset the right-error covariance, then marginalize the expired clone by retaining the remaining covariance submatrix.

Camera clone first estimates are retained for feature Jacobians. This is not a complete observability-constrained FEJ implementation for every state and propagation term. There is no claim of formal consistency equivalent to a mature research library. Empirical covariance coverage must be checked.

## ArUco and reference maps

PnP/IPPE validates the marker geometry and supplies initialization/relocalization poses. The filter's repeated marker measurement uses all four normalized corner observations and a reprojection model. Learned markers remain correlated with navigation. Fixed surveyed anchors constrain the room frame. Imported learned markers carry their previous covariance as independent run-to-run priors; cross-marker correlations from an earlier run are not serialized, so reuse remains an approximation.

A rejected fixed-marker observation may initiate relocalization only after two temporally adjacent, geometrically consistent poses while tracking is lost. Relocalization resets camera clones and learned marker correlations. The reinitialization is explicitly logged. A single noisy detection is not silently treated as a successful correction.

## Independent map refinement

The bounded optimizer uses IMU preintegration, bias evolution, raw normalized visual tracks and raw marker corners. Navigation states, marker poses and triangulated feature points are optimized with robust loss. The oldest state anchors the window. This implementation fixes the boundary instead of maintaining a Schur-complement marginalization prior. The window is an educational MAP refinement, not a claim of full fixed-lag smoothing consistency.

A final pose graph uses relative local estimates, repeated marker corners and geometrically verified visual-loop edges. The relative estimates and marker information can be correlated. The graph is used for robust map refinement, not for a second independent uncertainty estimate. It does not feed estimates back into the MSCKF, avoiding a second Kalman update with the same observations.

Robust optimization needs physical variable scaling: rotations, positions, velocities and biases have different units. The implementation uses explicit scales and strict sparse-solver tolerances. A candidate must converge and remain within a covariance-based trust region of the original filter estimate; rejected refinements are reported and the earlier estimate remains available.

Visual loops use ORB matching to older keyframes, triangulated 3-D feature positions, and RANSAC PnP with inlier count/ratio and reprojection checks. Marker identity provides an additional loop constraint. This is a bounded reference implementation, not a large-scale place-recognition database.

## Timing, transport and depth

Sensor timestamps and Camera2 image timestamps are preserved. Capture callback arrival times and laptop receive times are saved only for diagnosis. Replay sorts measurements by their acquisition time plus the calibrated camera/IMU offset. Late incompatible frames are rejected. Exposure and rolling-shutter metadata are stored; the current projection model treats a frame as a global exposure and does not compensate rolling shutter.

The phone writes disk records before offering them to the independent network queue. Sequence gaps identify transport loss; the local phone log can be imported into a separate dataset without overwriting a partial network recording. Interrupted/incomplete records are rejected. The TCP pairing token is not encryption; use a trusted local network.

Standard Camera2 autofocus distance is not a laser ranging measurement. Range-related result keys are enumerated for inspection, but no vendor-specific laser interface is assumed. Exposed DEPTH16 frames are optionally saved, with their confidence bits. A range update is enabled only with calibrated depth intrinsics/extrinsics, an explicit verification flag, a close timestamp, valid range/confidence, and a tight depth cluster near the registered marker centre. Uncalibrated focus data never enters the estimator. The IR blaster is logged as a transmitter capability only.

## References and implementation provenance

- MSCKF feature nullspace and EKF concepts: https://docs.openvins.com/update-null.html and https://docs.openvins.com/propagation.html
- VINS-Mono visual–inertial optimization: https://arxiv.org/abs/1708.03852
- Manifold IMU preintegration background: https://gtsam.org/notes/imu-factor/
- OpenCV ArUco/PnP: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- Android Camera2: https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics
- Android sensor timestamps: https://developer.android.com/reference/android/hardware/SensorEvent

Python estimator, graph, protocol, UI and Java recorder are project source. OpenCV supplies vision primitives; SciPy supplies rotations, chi-square distributions and numerical optimization. No OpenVINS/GTSAM/ORB-SLAM3 source is embedded or executed.
