import numpy as np


class Health:
    def __init__(self,cfg):
        self.cfg = cfg
        self.last = {}
        self.last_constraint = None
        self.failures = {}
        self.active = {}

    def sample(self,name,t,values=None):
        if values is not None and not np.all(np.isfinite(values)):
            self.failures[name] = 'nonfinite'; return False
        if name in self.last and t<=self.last[name]:
            self.failures[name] = 'nonmonotonic'; return False
        self.last[name] = t
        self.failures.pop(name,None)
        return True

    def constraint(self,name,t):
        self.last_constraint = t
        self.active[name] = t

    def status(self,t,estimator):
        detail = {}
        for name,timeout in [('accel',self.cfg.imu_max_gap_s),('gyro',self.cfg.imu_max_gap_s),
                             ('camera',self.cfg.camera_timeout_s),('compass',1.)]:
            age = t-self.last.get(name,-1e20)
            detail[name] = self.failures.get(name, 'ok' if age<=timeout else 'missing_or_stale')
        sigma = float(np.sqrt(np.max(np.diag(estimator.P)[3:6])))
        age = t-self.last_constraint if self.last_constraint is not None else float('inf')
        if not estimator.ready:
            mode = 'INITIALIZING'
        elif age>self.cfg.dead_reckon_limit_s or sigma>self.cfg.position_sigma_limit_m:
            mode = 'LOST'
        elif age>self.cfg.camera_timeout_s or any(detail[x]!='ok' for x in ('accel','gyro','camera')):
            mode = 'DEGRADED'
        else:
            mode = 'TRACKING'
        active={name for name,stamp in self.active.items() if t-stamp<=self.cfg.camera_timeout_s}
        active.update(name for name in ('accel','gyro') if detail[name]=='ok')
        return {'tracking':mode,'sensors':detail,'active_constraints':sorted(active),
                'seconds_since_constraint':age if np.isfinite(age) else None,
                'valid_pose':mode=='TRACKING'}
