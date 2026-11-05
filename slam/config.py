import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Config:
    camera_matrix: list = field(default_factory=lambda: [[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    distortion: list = field(default_factory=lambda: [0.] * 5)
    image_width: int = 640
    image_height: int = 480
    # Camera optical frame -> Android IMU body frame. MUST be calibrated for phone runs.
    T_body_camera: list = field(default_factory=lambda: np.diag([1., -1., -1., 1.]).tolist())
    camera_imu_offset_s: float = 0.
    camera_clock_offset_verified: bool = False
    calibration_verified: bool = False
    marker_size_m: float = .12
    dictionary: str = 'DICT_6X6_250'
    gyro_noise: float = .008
    accel_noise: float = .12
    gyro_bias_walk: float = .0002
    accel_bias_walk: float = .003
    pixel_sigma: float = 1.5
    max_clones: int = 8
    max_features: int = 120
    max_markers: int = 32
    imu_max_gap_s: float = .05
    camera_timeout_s: float = 1.
    dead_reckon_limit_s: float = 2.
    position_sigma_limit_m: float = 3.
    initialization_s: float = 1.
    mag_enabled: bool = False
    mag_bias: list = field(default_factory=lambda: [0., 0., 0.])
    mag_matrix: list = field(default_factory=lambda: np.eye(3).tolist())
    mag_reference_world: list | None = None
    mag_sigma_rad: float = .15
    graph_window: int = 8
    graph_keyframe_s: float = .5
    range_enabled: bool = False
    range_calibration_verified: bool = False
    depth_camera_matrix: list | None = None
    T_camera_depth: list | None = None
    accel_matrix: list = field(default_factory=lambda: np.eye(3).tolist())
    accel_offset: list = field(default_factory=lambda: [0.,0.,0.])

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        out = cls(**data)
        k, t = np.asarray(out.camera_matrix), np.asarray(out.T_body_camera)
        if k.shape != (3, 3) or t.shape != (4, 4) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(t)):
            raise ValueError('Invalid camera matrices')
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(t[:3, :3]) < .999:
            raise ValueError('Camera/IMU rotation must be a proper rotation')
        if not np.allclose(t[3], [0, 0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError('Invalid camera transform or focal length')
        if out.marker_size_m <= 0 or not 3 <= out.max_clones <= 20 or out.max_markers < 1:
            raise ValueError('Invalid marker size or window limits')
        for key in ('gyro_noise', 'accel_noise', 'gyro_bias_walk', 'accel_bias_walk', 'pixel_sigma', 'imu_max_gap_s'):
            if not np.isfinite(getattr(out,key)) or getattr(out, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key,shape in (('accel_matrix',(3,3)),('accel_offset',(3,)),('mag_matrix',(3,3)),('mag_bias',(3,))):
            value=np.asarray(getattr(out,key),float)
            if value.shape!=shape or not np.all(np.isfinite(value)):raise ValueError(f'Invalid {key}')
        if not np.isfinite(out.camera_imu_offset_s):raise ValueError('Invalid time offset')
        if out.mag_reference_world is not None:
            ref=np.asarray(out.mag_reference_world,float)
            if ref.shape!=(3,) or not np.all(np.isfinite(ref)) or np.linalg.norm(ref)<1:raise ValueError('Invalid magnetic reference')
        if out.range_enabled and out.range_calibration_verified:
            kd=np.asarray(out.depth_camera_matrix,float);td=np.asarray(out.T_camera_depth,float)
            if kd.shape!=(3,3) or td.shape!=(4,4) or not np.all(np.isfinite(kd)) or not np.all(np.isfinite(td)):
                raise ValueError('Verified depth fusion needs measured intrinsics and extrinsics')
        return out

    def intrinsics(self, width, height):
        k = np.array(self.camera_matrix, float)
        k[0] *= width / self.image_width
        k[1] *= height / self.image_height
        return k, np.array(self.distortion, float)
