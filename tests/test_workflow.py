import base64
import copy
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
import uuid
import cv2
import numpy as np
from slam.dataset import Store,read_record,import_phone,PROTOCOL
from slam.receiver import receive
from slam.simulate import generate
from slam.pipeline import process,Engine
from slam.config import Config
from slam.geometry import pose
from slam.evaluate import evaluate
from slam.depth import marker_depth


class WorkflowTests(unittest.TestCase):
    def test_protocol_rejects_nonfinite_and_truncated(self):
        for data in [b'{"x": NaN}\n',b'{"x":1}']:
            with self.assertRaises(ValueError):read_record(io.BytesIO(data))

    def test_socket_pairing_records_frame_and_sensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=socket.socketpair();a.settimeout(5);errors=[];saved=[]
            def worker():
                try:
                    with b:saved.append(receive(b,'secret',tmp,lambda _:None))
                except Exception as exc:errors.append(exc)
            t=threading.Thread(target=worker);t.start()
            with a,a.makefile('rb') as reader:
                hello={'kind':'hello','protocol':PROTOCOL,'session_id':str(uuid.uuid4()),'token':'secret'}
                a.sendall((json.dumps(hello)+'\n').encode());self.assertEqual(read_record(reader)['kind'],'ready')
                _,data=cv2.imencode('.jpg',np.full((80,80),255,np.uint8))
                records=[{'kind':'sensor','seq':1,'sensor_type':1,'sensor_key':1,'accuracy':3,'timestamp_ns':1,'values':[0,0,9.8]},
                    {'kind':'frame','seq':2,'timestamp_ns':2,'width':80,'height':80,'format':'jpeg','image_b64':base64.b64encode(data).decode()},
                    {'kind':'end','seq':3}]
                for r in records:a.sendall((json.dumps(r)+'\n').encode())
                self.assertEqual(read_record(reader)['kind'],'markers')
            t.join(5);self.assertFalse(t.is_alive());self.assertFalse(errors,errors)
            self.assertTrue((saved[0]/'frames'/'0000000002.jpg').exists())
            self.assertNotIn('secret',(saved[0]/'metadata.json').read_text())

    def test_synthetic_camera_imu_marker_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=generate(Path(tmp)/'dataset')
            output,summary=process(folder,folder/'config.json',folder/'surveyed_map.json',smooth=False)
            self.assertGreater(summary['counts']['visual_updates'],0)
            self.assertGreater(summary['counts']['marker_updates'],0)
            metrics=evaluate(output/'trajectory.csv',folder/'ground_truth.csv')
            self.assertLess(metrics['position_rmse_m'],.10)
            rows=[json.loads(r) for r in (output/'trajectory.jsonl').read_text().splitlines()]
            hidden=[r for r in rows if 2.5<r['timestamp_s']<3.3]
            self.assertTrue(any(r.get('accepted_tracks',0)>0 for r in hidden))
            self.assertTrue(all(np.linalg.eigvalsh(r['position_covariance']).min()>-1e-8 for r in rows))
            _,baseline=process(folder,folder/'config.json',use_markers=False,smooth=False)
            self.assertEqual(baseline['counts']['marker_updates'],0)
            self.assertGreater(baseline['counts']['visual_updates'],0)

    def test_unknown_map_first_marker_is_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=generate(Path(tmp)/'dataset')
            output,_=process(folder,folder/'config.json',smooth=False)
            marker=json.loads((output/'marker_map.json').read_text())['markers']['0']
            np.testing.assert_allclose(marker['T_world_marker'],np.eye(4),atol=1e-8)

    def test_unknown_camera_clock_is_rejected(self):
        cfg=Config(calibration_verified=True);engine=Engine(cfg,{},'.');engine.filter.ready=True;engine.filter.t=0
        engine.camera_info={'timestamp_source':0}
        with self.assertRaisesRegex(ValueError,'clock'):
            engine.consume({'kind':'frame','timestamp_ns':1,'file':'does-not-exist.jpg'})

    def test_depth_confidence_and_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Config(range_enabled=True,range_calibration_verified=True,depth_camera_matrix=[[100,0,10],[0,100,10],[0,0,1]],T_camera_depth=np.eye(4).tolist())
            # DEPTH16 confidence bits 0 mean 100% confidence; units are millimetres.
            (Path(tmp)/'depth.bin').write_bytes(np.full((20,20),2000,dtype='<u2').tobytes())
            event={'file':'depth.bin','width':20,'height':20,'row_stride':40,'pixel_stride':2}
            marker={'corners':np.array([[2,2],[18,2],[18,18],[2,18]],float)}
            measurement=marker_depth(tmp,event,marker,cfg,np.array(cfg.depth_camera_matrix))
            self.assertAlmostEqual(measurement[0],2)
            cfg.range_calibration_verified=False
            self.assertIsNone(marker_depth(tmp,event,marker,cfg,np.array(cfg.depth_camera_matrix)))


if __name__=='__main__':unittest.main()
