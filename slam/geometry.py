"""Transforms T_AB map coordinates in B into A. Rotations use right errors."""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp(v):
    return Rotation.from_rotvec(v).as_matrix()


def log(r):
    return Rotation.from_matrix(r).as_rotvec()


def pose(r=None, p=None):
    t = np.eye(4)
    if r is not None:
        t[:3, :3] = r
    if p is not None:
        t[:3, 3] = p
    return t


def inv(t):
    r = t[:3, :3].T
    return pose(r, -r @ t[:3, 3])


def perturb(t, dx):
    return pose(t[:3, :3] @ exp(dx[:3]), t[:3, 3] + dx[3:6])


def difference(a, b):
    """Residual from predicted b to observed a, [rotation, world translation]."""
    return np.r_[log(b[:3, :3].T @ a[:3, :3]), a[:3, 3] - b[:3, 3]]


def project(t_wc, point):
    pc = t_wc[:3, :3].T @ (point - t_wc[:3, 3])
    if pc[2] <= .05:
        raise ValueError('Point behind or too near camera')
    return pc[:2] / pc[2], pc


def projection_jacobian(pc):
    x, y, z = pc
    return np.array([[1/z, 0, -x/z**2], [0, 1/z, -y/z**2]])


def triangulate(observations, poses, min_parallax_deg=.5):
    """Linear ray intersection; rejects low-parallax / ill-conditioned tracks."""
    rows, rhs, directions = [], [], []
    for key, uv in observations:
        t = poses[key]
        d = t[:3, :3] @ np.r_[uv, 1.]
        d /= np.linalg.norm(d)
        a = np.eye(3) - np.outer(d, d)
        rows.append(a)
        rhs.append(a @ t[:3, 3])
        directions.append(d)
    if len(rows) < 3:
        raise ValueError('Track too short')
    angle = max(np.arccos(np.clip(np.dot(directions[0], d), -1, 1)) for d in directions)
    if angle < np.deg2rad(min_parallax_deg):
        raise ValueError('Insufficient parallax')
    a = np.vstack(rows)
    if np.linalg.cond(a) > 1e5:
        raise ValueError('Ill-conditioned triangulation')
    point = np.linalg.lstsq(a, np.concatenate(rhs), rcond=None)[0]
    for key, _ in observations:
        project(poses[key], point)
    return point
