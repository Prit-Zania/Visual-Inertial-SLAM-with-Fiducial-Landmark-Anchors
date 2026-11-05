"""Body-frame IMU preintegration for independent batch/sliding-window smoothing."""
import numpy as np
from .geometry import exp,log,skew


def integrate(samples,bg,ba):
    r=np.eye(3); v=np.zeros(3); p=np.zeros(3); duration=0.
    for dt,a,w in samples:
        mid=r@exp((w-bg)*dt/2)
        acceleration=mid@(a-ba)
        p+=v*dt+.5*acceleration*dt**2; v+=acceleration*dt
        r=r@exp((w-bg)*dt); duration+=dt
    return r,v,p,duration


class Preintegrated:
    def __init__(self,samples,bg,ba,cfg):
        self.bg=np.array(bg); self.ba=np.array(ba)
        self.R,self.v,self.p,self.dt=integrate(samples,self.bg,self.ba)
        self.J=np.zeros((9,6))
        for i in range(6):
            b=np.r_[self.bg,self.ba]; b[i]+=1e-5
            r,v,p,_=integrate(samples,b[:3],b[3:])
            self.J[:,i]=np.r_[log(self.R.T@r),v-self.v,p-self.p]/1e-5
        # Discrete covariance in rotation / delta velocity / delta position order.
        self.cov=np.eye(9)*1e-10; rotation=np.eye(3)
        for dt,a,w in samples:
            f=np.eye(9)
            f[:3,:3]=exp(-(w-bg)*dt)
            f[3:6,:3]=-rotation@skew(a-ba)*dt
            f[6:9,:3]=-.5*rotation@skew(a-ba)*dt**2
            f[6:9,3:6]=np.eye(3)*dt
            g=np.zeros((9,6)); g[:3,:3]=np.eye(3)
            g[3:6,3:6]=rotation; g[6:9,3:6]=.5*rotation*dt
            q=np.diag(np.r_[np.ones(3)*cfg.gyro_noise**2,np.ones(3)*cfg.accel_noise**2])*dt
            self.cov=f@self.cov@f.T+g@q@g.T
            rotation=rotation@exp((w-bg)*dt)
        self.sqrt_info=np.linalg.inv(np.linalg.cholesky(self.cov+np.eye(9)*1e-8))

    def residual(self,a,b,gravity):
        # Each state contains R,p,v,bg,ba.
        correction=self.J@np.r_[a['bg']-self.bg,a['ba']-self.ba]
        dr=self.R@exp(correction[:3]); dv=self.v+correction[3:6]; dp=self.p+correction[6:9]
        dt=self.dt; ri=a['R'].T
        error=np.r_[log(dr.T@ri@b['R']),ri@(b['v']-a['v']-gravity*dt)-dv,
                    ri@(b['p']-a['p']-a['v']*dt-.5*gravity*dt**2)-dp]
        return self.sqrt_info@error
