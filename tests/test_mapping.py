import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from slam.config import Config
from slam.geometry import pose,project
from slam.msckf import MSCKF
from slam.graph import Smoother
from slam.loops import VisualLoops
from slam.pipeline import Engine,process
from slam.simulate import generate


class MappingTests(unittest.TestCase):
    def test_visual_loop_requires_geometric_pnp_verification(self):
        cfg=Config(T_body_camera=np.eye(4).tolist());loops=VisualLoops(cfg)
        rng=np.random.default_rng(7);image=rng.integers(0,256,(480,640),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),0)
        kp,desc=loops.orb.detectAndCompute(image,None);xy=np.array([p.pt for p in kp])
        k=np.array(cfg.camera_matrix);rays=(np.linalg.inv(k)@np.c_[xy,np.ones(len(xy))].T).T
        depths=rng.uniform(2,5,len(xy));points={i:rays[i]*depths[i] for i in range(len(xy))}
        loops.history=[{'key':1,'t':0.,'T':np.eye(4),'xy':xy,'desc':desc,'points':points}]
        # Five intervening keyframes are excluded as recent neighbors.
        for i in range(5):loops.history.append({'key':i+2,'t':i+1.,'T':np.eye(4),'xy':xy,'desc':desc,'points':{}})
        found=loops.add(10,7.,image,np.eye(4))
        self.assertEqual(len(found),1);np.testing.assert_allclose(found[0][2],np.eye(4),atol=1e-4)
        unrelated=rng.integers(0,256,image.shape,dtype=np.uint8)
        self.assertEqual(loops.add(11,8.,unrelated,np.eye(4)),[])

    def test_window_optimizes_preintegrated_imu(self):
        cfg=Config(T_body_camera=np.eye(4).tolist(),graph_window=4)
        graph=Smoother(cfg);f=MSCKF(cfg);f.ready=True;f.t=0
        f.fixed[0]=pose(p=[0,0,3])
        for i in range(4):
            f.p=np.array([.01*i,0,0]);f.v=np.zeros(3)
            graph.add(i,i*.1,f,[(.01,np.array([0,0,9.80665]),np.zeros(3))]*10,[])
        graph.optimize_window()
        self.assertTrue(graph.reports[-1]['accepted'])
        self.assertLess(np.linalg.norm(graph.nodes[-1]['p']),.005)

    def test_map_optimizer_rejects_unobservable_jump(self):
        cfg=Config();graph=Smoother(cfg);f=MSCKF(cfg);graph.add(1,0,f,[],[])
        self.assertFalse(graph.within_trust_region([(1,pose(p=[5,0,0]))]))

    def test_loaded_marker_retains_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.json';cov=np.eye(6)*.04
            path.write_text(json.dumps({'marker_size_m':.12,'markers':{'8':{'T_world_marker':pose(p=[0,0,3]).tolist(),'fixed':False,'filter_covariance':cov.tolist()}}}))
            engine=Engine(Config(),{},tmp,path)
            self.assertIn(8,engine.filter.markers);self.assertNotIn(8,engine.filter.fixed)
            np.testing.assert_allclose(engine.filter.P[-6:,-6:],cov,atol=1e-7)

    def test_missing_gyro_uses_visual_marker_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=generate(Path(tmp)/'data');path=folder/'events.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows=[r for r in rows if not(r['kind']=='sensor' and r['sensor_type']==4)]
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            _,summary=process(folder,folder/'config.json',folder/'surveyed_map.json',smooth=False)
            self.assertFalse(summary['imu_initialized'])
            self.assertGreater(summary['counts']['marker_updates'],0)
            self.assertGreater(summary['tracking_counts']['DEGRADED'],0)


if __name__=='__main__':unittest.main()
