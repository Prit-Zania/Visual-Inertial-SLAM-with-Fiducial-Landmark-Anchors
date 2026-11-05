# Validation and remaining device work

## Software checks

The automated suite exercises:

- Stationary inertial propagation and gyro bias estimation.
- Known constant acceleration and position/velocity integration.
- Projection Jacobians against finite differences.
- Camera clone cross-covariance, marker augmentation and clone marginalization.
- A genuine MSCKF nullspace update that reduces an injected attitude error.
- Marker correction and rejection of a large false measurement.
- IMU preintegration against known motion.
- Camera-loss status and uncertainty growth.
- Missing-gyro visual/marker fallback.
- Compass field and accuracy rejection.
- DEPTH16 confidence decoding and registration gates.
- Pairing, image transfer, sequence handling and persistent recording over a socket.
- Unknown clock-source rejection.
- Synthetic image/IMU replay with and without ArUco constraints.
- First-marker origin and loaded-map covariance.
- Geometric visual-loop verification, including rejection of an unrelated image.
- Window optimization of a controlled inertial problem.
- Rejection of implausible optimizer jumps.

Run `python -m unittest discover -s tests -v` using the project environment. Run `python tools/check_java.py` after installing `requirements-dev.txt` for Java syntax parsing. That parser is not an Android compiler.

`demo_output/synthetic/comparison.json` contains reproducible synthetic trajectory metrics. The synthetic images use simple projected textures and a marker with a known size. The IMU, timestamps and calibration are generated consistently, without real rolling shutter, phone thermal throttling, network delay, optical stabilization or actual sensor bias instability. Numbers from this example must not be reported as measured OnePlus 12R accuracy.

Some optimization windows may be rejected because they do not converge or exceed the filter-based trust region. `optimization.json` records both `converged` and `accepted`. A rejected window does not overwrite the earlier trajectory. Model-based filter confidence and optimizer acceptance are distinct quantities.

The saved development run passed **21 automated tests** and Java parsing for four source files. A hidden desktop smoke test loaded 61 state samples and one marker into the dashboard. The synthetic comparison yielded about **0.0359 m position RMSE with marker constraints** and **0.0030 m for the no-marker baseline**. The baseline is better on this short, nearly ideal-IMU example; the small rendered marker introduces pixel/PnP errors. This is not evidence that markers always improve accuracy, and the project does not claim that result. Longer real recordings and an independent survey are necessary to measure the benefit of anchors and loop closure.

## Not yet verified

- Android SDK compilation and an installable APK: intentionally not attempted after the source-only instruction.
- Camera2 stream combinations, preview sizing, timestamp source and frame rate on the actual OnePlus 12R.
- Device camera/IMU extrinsics, timing, noise parameters and magnetic calibration.
- Whether this phone exposes a usable depth stream or any vendor-specific laser data.
- Real room trajectory accuracy, marker repeatability, thermal/battery behavior and Wi-Fi throughput.

## Required physical experiments

Place 6–10 fixed ArUco markers and record their printed widths, positions and orientations. Use an independent room survey or measured path and record these experiments:

1. Straight, L-shaped or rectangular path.
2. A section with no visible marker.
3. Return to an earlier location and observe the same marker.
4. A new run using the saved map.
5. Poor light, rapid motion, partial occlusion or another difficult condition.

For each, process the same recording with the baseline and with marker constraints. Report position/orientation error where surveyed, endpoint drift, relative displacement error, marker reprojection residuals, tracking-loss rate, relocalization time and runtime. Check how often independent ground-truth errors lie inside the reported uncertainty bounds. Keep rejected measurements and dropout logs; do not remove failed sections to improve the score.

The original handout also requires a real dataset, technical report and demonstration video. Those require physical capture and cannot be replaced by the synthetic regression dataset.
