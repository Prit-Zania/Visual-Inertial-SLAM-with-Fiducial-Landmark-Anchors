import unittest
import numpy as np
from slam.config import Config
from slam.geometry import pose,exp,project,skew,projection_jacobian,triangulate
from slam.msckf import MSCKF
from slam.preintegration import Preintegrated
from slam.health import Health


def config():
    return Config(T_body_camera=np.eye(4).tolist(),calibration_verified=True)


class EstimationTests(unittest.TestCase):
    def test_stationary_inertial_state_and_bias(self):
        f=MSCKF(config());bias=np.array([.01,-.02,.005])
        f.initialize(0,np.tile([0,0,9.80665],(100,1)),np.tile(bias,(100,1)))
        for i in range(1,501):f.propagate(i*.005,[0,0,9.80665],bias)
        np.testing.assert_allclose(f.p,0,atol=1e-10)
        np.testing.assert_allclose(f.bg,bias)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(),-1e-10)

    def test_constant_acceleration(self):
        f=MSCKF(config());f.initialize(0,np.tile([0,0,9.80665],(30,1)),np.zeros((30,3)))
        for i in range(1,101):f.propagate(i*.01,[1,0,9.80665],[0,0,0])
        np.testing.assert_allclose(f.p,[.5,0,0],atol=1e-5)
        np.testing.assert_allclose(f.v,[1,0,0],atol=1e-5)

    def test_projection_jacobian_matches_finite_difference(self):
        t=pose(exp([.1,.2,-.1]),[.3,-.1,.5]);point=np.array([1.,.2,4.])
        uv,pc=project(t,point);h=projection_jacobian(pc)@np.c_[skew(pc),-t[:3,:3].T]
        from slam.geometry import perturb
        numerical=np.empty((2,6))
        for i in range(6):
            dx=np.zeros(6);dx[i]=1e-6;numerical[:,i]=(project(perturb(t,dx),point)[0]-uv)/1e-6
        np.testing.assert_allclose(h,numerical,atol=1e-5)

    def test_clone_cross_covariance_and_marginalization(self):
        f=MSCKF(config());f.add_marker(5,pose(p=[0,0,3]));f.add_clone(1);f.add_clone(2)
        self.assertEqual(f.P.shape,(33,33))
        self.assertGreater(np.linalg.norm(f.P[:15,15:21]),0)
        before=f.P[f.marker_index(5):,f.marker_index(5):].copy()
        f.remove_clone(1)
        np.testing.assert_allclose(f.P[f.marker_index(5):,f.marker_index(5):],before)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(),-1e-9)

    def test_msckf_nullspace_feature_update(self):
        f=MSCKF(config());truth={};observations=[]
        for i in range(5):
            f.p=np.array([i*.08,0,0]);truth[i]=f.camera.copy()
            f.add_clone(i)
        # Introduce independent clone attitude uncertainty/error after augmentation.
        f.P+=np.eye(len(f.P))*.01
        f.clones[2][:3,:3]=exp([0,.006,0]);f.fej[2]=f.clones[2].copy()
        rng=np.random.default_rng(42)
        for point in rng.uniform([-1,-1,3],[1,1,5],(20,3)):
            observations.append([(i,project(truth[i],point)[0]) for i in truth])
        before=np.linalg.norm(f.clones[2][:3,:3]-truth[2][:3,:3])
        accepted=f.feature_update(observations,500)
        self.assertGreater(accepted,0)
        self.assertLess(np.linalg.norm(f.clones[2][:3,:3]-truth[2][:3,:3]),before)
        self.assertGreater(np.linalg.eigvalsh(f.P).min(),-1e-9)

    def test_marker_correction_and_reject_outlier(self):
        f=MSCKF(config());f.fixed[7]=pose(p=[0,0,3]);f.p=np.array([.02,0,0]);f.P*=20
        s=f.cfg.marker_size_m/2
        square=np.array([[-s,s,3],[s,s,3],[s,-s,3],[-s,-s,3]])
        uv=np.array([project(np.eye(4),p)[0] for p in square])
        self.assertTrue(f.marker_update(7,uv));self.assertLess(abs(f.p[0]),.02)
        old=f.p.copy();self.assertFalse(f.marker_update(7,uv+2))
        np.testing.assert_allclose(f.p,old)

    def test_preintegration_matches_accelerated_motion(self):
        cfg=config();pre=Preintegrated([(.01,np.array([1,0,9.80665]),np.zeros(3)) for _ in range(100)],np.zeros(3),np.zeros(3),cfg)
        a={'R':np.eye(3),'p':np.zeros(3),'v':np.zeros(3),'bg':np.zeros(3),'ba':np.zeros(3)}
        b={**a,'p':np.array([.5,0,0]),'v':np.array([1.,0,0])}
        np.testing.assert_allclose(pre.residual(a,b,np.array([0,0,-9.80665])),0,atol=1e-6)

    def test_failed_camera_becomes_lost_and_uncertainty_grows(self):
        cfg=config();f=MSCKF(cfg);f.initialize(0,np.tile([0,0,9.80665],(30,1)),np.zeros((30,3)))
        h=Health(cfg);h.constraint('aruco',0);h.sample('camera',0)
        initial=f.P[3,3]
        for i in range(1,301):
            t=i*.01;h.sample('accel',t);h.sample('gyro',t);f.propagate(t,[0,0,9.80665],[0,0,0])
        self.assertEqual(h.status(3,f)['tracking'],'LOST');self.assertGreater(f.P[3,3],initial)

    def test_stale_imu_does_not_integrate_fake_acceleration(self):
        f=MSCKF(config());f.ready=True;f.t=0;before=f.P.copy()
        self.assertFalse(f.propagate(1,None,None));np.testing.assert_allclose(f.p,0)
        self.assertGreater(f.P[3,3],before[3,3])

    def test_compass_rejects_interference(self):
        cfg=config();cfg.mag_enabled=True;cfg.mag_reference_world=[30,0,30]
        f=MSCKF(cfg)
        self.assertFalse(f.compass_update([500,0,20],3))
        self.assertFalse(f.compass_update([30,0,30],0))
        self.assertTrue(f.compass_update([30,0,30],3))


if __name__=='__main__':unittest.main()
