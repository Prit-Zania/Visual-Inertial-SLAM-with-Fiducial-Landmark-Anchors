"""Error-state MSCKF with correlated camera clones and persistent marker states.

Error order: [body rotation, position, velocity, gyro bias, accel bias], then
6-D camera clones and 6-D markers. Right rotation errors, additive world positions.
Feature states are eliminated by left-nullspace projection, not stored in P.
"""
from collections import OrderedDict
import numpy as np
from scipy.stats import chi2
from .geometry import exp, log, skew, pose, inv, project, projection_jacobian, triangulate


class MSCKF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.g = np.array([0., 0., -9.80665])
        self.P = np.diag(np.r_[np.ones(3)*.05**2, np.ones(3)*.02**2,
                               np.ones(3)*.05**2, np.ones(3)*.02**2, np.ones(3)*.2**2])
        self.clones = OrderedDict()
        self.fej = {}
        self.markers = OrderedDict()
        self.fixed = {}
        self.t = None
        self.ready = False
        self.last_update = {'accepted': False, 'reason': 'initializing'}
        self.counts = {'visual_updates': 0, 'marker_updates': 0, 'rejected': 0}

    @property
    def T(self):
        return pose(self.R, self.p)

    @property
    def camera(self):
        return self.T @ np.asarray(self.cfg.T_body_camera)

    def initialize(self, t, accelerations, gyros):
        a, w = np.asarray(accelerations), np.asarray(gyros)
        if len(a) < 20 or len(w) < 20:
            raise ValueError('Initialization needs at least 20 stationary IMU samples')
        if np.max(np.std(a, axis=0)) > .25 or np.max(np.std(w, axis=0)) > .03 or np.linalg.norm(np.mean(w, axis=0)) > .15:
            raise ValueError('Hold the phone still during initialization')
        avg = np.mean(a, axis=0)
        if not 8.8 < np.linalg.norm(avg) < 10.8:
            raise ValueError('Accelerometer must contain gravity in m/s^2')
        source = avg / np.linalg.norm(avg)
        target = np.array([0., 0., 1.])
        axis = np.cross(source, target)
        angle = np.arccos(np.clip(source @ target, -1, 1))
        self.R = exp((axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-8 else np.array([1., 0., 0.])) * angle)
        self.bg = np.mean(w, axis=0)
        # One static pose does not identify all accelerometer bias components.
        self.ba = source * (np.linalg.norm(avg) - 9.80665)
        self.t, self.ready = float(t), True

    def propagate(self, t, accel, gyro):
        if not self.ready:
            return False
        dt = float(t - self.t)
        if dt <= 0:
            return False
        if dt > self.cfg.imu_max_gap_s or accel is None or gyro is None:
            # No fabricated stale accelerations. Constant velocity with large process noise.
            self.p += self.v * dt
            if gyro is not None and dt <= self.cfg.imu_max_gap_s:
                self.R = self.R @ exp((np.asarray(gyro)-self.bg)*dt)
            f = np.eye(len(self.P)); f[3:6, 6:9] = np.eye(3)*dt
            self.P = f @ self.P @ f.T
            self.P[:9, :9] += np.diag(np.r_[np.ones(3)*dt*.5, np.ones(3)*dt**2, np.ones(3)*dt])
            self.t = float(t)
            return False
        a, w = np.asarray(accel) - self.ba, np.asarray(gyro) - self.bg
        rmid = self.R @ exp(w*dt/2)
        aw = rmid @ a + self.g
        self.p += self.v*dt + .5*aw*dt**2
        self.v += aw*dt
        self.R = self.R @ exp(w*dt)
        f = np.zeros((15, 15))
        f[:3, :3] = -skew(w); f[:3, 9:12] = -np.eye(3)
        f[3:6, 6:9] = np.eye(3)
        f[6:9, :3] = -rmid @ skew(a); f[6:9, 12:15] = -rmid
        phi = np.eye(len(self.P)); phi[:15, :15] += f*dt + .5*(f@f)*dt**2
        g = np.zeros((15, 12))
        g[:3, :3] = -np.eye(3); g[6:9, 3:6] = -rmid
        g[9:12, 6:9] = np.eye(3); g[12:15, 9:12] = np.eye(3)
        q = np.repeat(np.square([self.cfg.gyro_noise, self.cfg.accel_noise,
                                self.cfg.gyro_bias_walk, self.cfg.accel_bias_walk]), 3)
        qd = g @ np.diag(q) @ g.T
        # Midpoint discretization includes the position/velocity noise cross term.
        half = np.eye(15) + .5*f*dt
        self.P = phi @ self.P @ phi.T
        self.P[:15, :15] += half @ qd @ half.T * dt
        self.P = (self.P + self.P.T)/2
        self.t = float(t)
        return True

    def clone_index(self, key):
        return 15 + 6*list(self.clones).index(key)

    def marker_index(self, key):
        return 15 + 6*len(self.clones) + 6*list(self.markers).index(key)

    def add_clone(self, key):
        tbc = np.asarray(self.cfg.T_body_camera)
        j = np.zeros((6, len(self.P)))
        j[:3, :3] = tbc[:3, :3].T
        j[3:, :3] = -self.R @ skew(tbc[:3, 3]); j[3:, 3:6] = np.eye(3)
        # Insert clones ahead of marker states without dropping cross-covariance.
        at = 15 + 6*len(self.clones)
        n = len(self.P)
        augmented = np.block([[self.P, self.P @ j.T], [j @ self.P, j @ self.P @ j.T + np.eye(6)*1e-12]])
        order = list(range(at)) + list(range(n, n+6)) + list(range(at, n))
        self.P = augmented[np.ix_(order, order)]
        self.clones[key] = self.camera.copy()
        self.fej[key] = self.camera.copy()

    def remove_clone(self, key):
        i = self.clone_index(key)
        keep = np.r_[0:i, i+6:len(self.P)]
        self.P = self.P[np.ix_(keep, keep)]
        del self.clones[key], self.fej[key]

    def update(self, residual, h, noise, label, gate=True):
        r = np.asarray(residual)
        noise = np.asarray(noise)
        s = h @ self.P @ h.T + noise
        try:
            nis = float(r @ np.linalg.solve(s, r))
            threshold = float(chi2.ppf(.99, len(r)))
            self.last_update = {'kind': label, 'nis': nis, 'dof': len(r), 'threshold': threshold, 'accepted': False}
            if not np.isfinite(nis) or (gate and nis > threshold):
                self.counts['rejected'] += 1
                return False
            k = np.linalg.solve(s, h @ self.P).T
        except np.linalg.LinAlgError:
            return False
        dx = k @ r
        eye = np.eye(len(self.P)); a = eye-k@h
        self.P = a @ self.P @ a.T + k @ noise @ k.T
        self.R = self.R @ exp(dx[:3]); self.p += dx[3:6]; self.v += dx[6:9]
        self.bg += dx[9:12]; self.ba += dx[12:15]
        reset = eye.copy(); reset[:3, :3] -= .5*skew(dx[:3])
        for mapping, index_fn in ((self.clones, self.clone_index), (self.markers, self.marker_index)):
            for key, t in mapping.items():
                i = index_fn(key)
                t[:3, :3] = t[:3, :3] @ exp(dx[i:i+3]); t[:3, 3] += dx[i+3:i+6]
                reset[i:i+3, i:i+3] -= .5*skew(dx[i:i+3])
        self.P = reset @ self.P @ reset.T
        self.P = (self.P + self.P.T)/2
        self.last_update['accepted'] = True
        return True

    def feature_update(self, tracks, focal_length):
        sigma = self.cfg.pixel_sigma / focal_length
        hs, rs = [], []
        for obs in tracks:
            obs = [(key, uv) for key, uv in obs if key in self.clones]
            try:
                point = triangulate(obs, self.clones)
                point_fej = triangulate(obs, self.fej)
                h = np.zeros((2*len(obs), len(self.P))); hf = np.zeros((2*len(obs), 3)); residual = []
                for n, (key, uv) in enumerate(obs):
                    prediction, _ = project(self.clones[key], point)
                    _, pc = project(self.fej[key], point_fej)
                    j = projection_jacobian(pc)
                    rcw = self.fej[key][:3, :3].T
                    i = self.clone_index(key)
                    h[2*n:2*n+2, i:i+6] = j @ np.c_[skew(pc), -rcw]
                    hf[2*n:2*n+2] = j @ rcw
                    residual.extend(np.asarray(uv)-prediction)
                u, singular, _ = np.linalg.svd(hf, full_matrices=True)
                rank = np.sum(singular > 1e-8)
                null = u[:, rank:].T
                hh, rr = null @ h, null @ np.asarray(residual)
                s = hh @ self.P @ hh.T + np.eye(len(rr))*sigma**2
                if len(rr) and rr @ np.linalg.solve(s, rr) < chi2.ppf(.99, len(rr)):
                    hs.append(hh); rs.append(rr)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not hs:
            return 0
        h, r = np.vstack(hs), np.concatenate(rs)
        # Orthogonal measurement compression preserves isotropic image noise.
        if len(r) > h.shape[1]:
            q, _ = np.linalg.qr(h, mode='reduced')
            h, r = q.T @ h, q.T @ r
        if self.update(r, h, np.eye(len(r))*sigma**2, 'msckf_features'):
            self.counts['visual_updates'] += 1
            return len(hs)
        return 0

    def add_marker(self, ident, t_cm, sigma_rotation=.04, sigma_position=.03):
        if ident in self.markers or ident in self.fixed or len(self.markers) >= self.cfg.max_markers:
            return False
        tbc = np.asarray(self.cfg.T_body_camera)
        tbm = tbc @ t_cm
        j = np.zeros((6, len(self.P)))
        j[:3, :3] = tbm[:3, :3].T
        j[3:, :3] = -self.R @ skew(tbm[:3, 3]); j[3:, 3:6] = np.eye(3)
        r = np.diag(np.r_[np.ones(3)*sigma_rotation**2, np.ones(3)*sigma_position**2])
        self.P = np.block([[self.P, self.P@j.T], [j@self.P, j@self.P@j.T+r]])
        self.markers[ident] = self.camera @ t_cm
        return True

    def marker_update(self, ident, corners_normalized, focal_length=None):
        tm = self.fixed.get(ident, self.markers.get(ident))
        if tm is None:
            return False
        s = self.cfg.marker_size_m/2
        corners = np.array([[-s,s,0], [s,s,0], [s,-s,0], [-s,-s,0]])
        tbc = np.asarray(self.cfg.T_body_camera)
        tc = self.camera; rcw = tc[:3, :3].T
        jcam = np.zeros((6, len(self.P)))
        jcam[:3,:3] = tbc[:3,:3].T
        jcam[3:,:3] = -self.R @ skew(tbc[:3,3]); jcam[3:,3:6] = np.eye(3)
        h = np.zeros((8,len(self.P))); residual = []
        try:
            for n, corner in enumerate(corners):
                pw = tm[:3,:3] @ corner + tm[:3,3]
                uv, pc = project(tc,pw); j = projection_jacobian(pc)
                h[2*n:2*n+2] = j @ np.c_[skew(pc), -rcw] @ jcam
                if ident in self.markers:
                    i = self.marker_index(ident)
                    h[2*n:2*n+2,i:i+6] = j @ np.c_[-rcw @ tm[:3,:3] @ skew(corner), rcw]
                residual.extend(corners_normalized[n]-uv)
        except ValueError:
            return False
        focal = focal_length or np.mean(np.diag(np.asarray(self.cfg.camera_matrix))[:2])
        accepted = self.update(residual, h, np.eye(8)*(self.cfg.pixel_sigma/focal)**2, 'aruco')
        if accepted:
            self.counts['marker_updates'] += 1
        return accepted

    def range_update(self, ident, depth_m, sigma_m):
        """Calibrated optical-axis depth at the marker centre; never focus distance."""
        tm = self.fixed.get(ident, self.markers.get(ident))
        if tm is None or not .1 < depth_m < 15 or not .001 <= sigma_m <= 2:
            return False
        tc = self.camera; rcw = tc[:3,:3].T
        pc = rcw @ (tm[:3,3]-tc[:3,3])
        tbc = np.asarray(self.cfg.T_body_camera)
        jc = np.zeros((6,len(self.P))); jc[:3,:3] = tbc[:3,:3].T
        jc[3:,:3] = -self.R @ skew(tbc[:3,3]); jc[3:,3:6] = np.eye(3)
        h = (np.c_[skew(pc),-rcw] @ jc)[2:3]
        if ident in self.markers:
            i = self.marker_index(ident); h[0,i+3:i+6] = rcw[2]
        return self.update([depth_m-pc[2]],h,np.array([[sigma_m**2]]),'validated_depth')

    def compass_update(self, magnetic, accuracy):
        reference = self.cfg.mag_reference_world
        if not self.cfg.mag_enabled or reference is None or accuracy < 2:
            return False
        m = np.asarray(self.cfg.mag_matrix) @ (np.asarray(magnetic)-self.cfg.mag_bias)
        ref = np.asarray(reference, float)
        if not 20 < np.linalg.norm(m) < 75 or not .75 < np.linalg.norm(m)/np.linalg.norm(ref) < 1.25:
            return False
        up = -self.g / np.linalg.norm(self.g)
        def error(r):
            measured = r @ m
            a = measured-up*(measured@up); b = ref-up*(ref@up)
            if min(np.linalg.norm(a),np.linalg.norm(b)) < 5:
                raise ValueError('Horizontal magnetic field too weak')
            return np.arctan2(up @ np.cross(a,b), a@b)
        try:
            residual = error(self.R)
            h = np.zeros((1,len(self.P)))
            for i in range(3):
                d = np.zeros(3); d[i] = 1e-6
                # measurement Jacobian = minus derivative of residual
                h[0,i] = -np.arctan2(np.sin(error(self.R@exp(d))-residual),np.cos(error(self.R@exp(d))-residual))/1e-6
            return self.update([residual], h, np.array([[self.cfg.mag_sigma_rad**2]]), 'compass')
        except ValueError:
            return False

    def zero_velocity_update(self):
        h = np.zeros((3,len(self.P))); h[:,6:9] = np.eye(3)
        return self.update(-self.v,h,np.eye(3)*.02**2,'stationary_velocity')

    def rebase(self, transform):
        """Change world coordinates without adding information."""
        r, p = transform[:3,:3], transform[:3,3]
        self.R = r@self.R; self.p = r@self.p+p; self.v = r@self.v; self.g = r@self.g
        j = np.eye(len(self.P)); j[3:6,3:6] = r; j[6:9,6:9] = r
        for mapping,index_fn in ((self.clones,self.clone_index),(self.markers,self.marker_index)):
            for key in mapping:
                mapping[key] = transform @ mapping[key]
                i = index_fn(key); j[i+3:i+6,i+3:i+6] = r
        for key in self.fej:
            self.fej[key] = transform @ self.fej[key]
        for key in self.fixed:
            self.fixed[key] = transform @ self.fixed[key]
        if self.cfg.mag_reference_world is not None:
            self.cfg.mag_reference_world = (r @ self.cfg.mag_reference_world).tolist()
        self.P = j@self.P@j.T

    def status(self):
        return {'position_sigma_m': np.sqrt(np.maximum(0,np.diag(self.P)[3:6])).tolist(),
                'orientation_sigma_rad': np.sqrt(np.maximum(0,np.diag(self.P)[:3])).tolist(),
                'position_covariance': self.P[3:6,3:6].tolist(), 'pose_covariance': self.P[:6,:6].tolist(),
                'last_update': self.last_update.copy(), 'counts': self.counts.copy()}
