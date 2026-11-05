"""Deterministic synthetic data, explicitly not a phone accuracy benchmark."""
import argparse
import csv
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
from .config import Config
from .geometry import pose,inv


def generate(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);(root/'frames').mkdir(exist_ok=True)
    cfg=Config(T_body_camera=np.eye(4).tolist(),calibration_verified=True,max_features=70,graph_window=6)
    (root/'config.json').write_text(json.dumps(vars(cfg),indent=2),encoding='utf-8')
    metadata={'kind':'hello','protocol':'oneplus-msckf-v1','session_id':str(uuid.uuid4()),'synthetic':True,'selected_sensors':{'1':1,'4':4,'2':2}}
    (root/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    tm=pose(np.diag([1.,-1.,-1.]),[0,0,2.])
    (root/'surveyed_map.json').write_text(json.dumps({'marker_size_m':cfg.marker_size_m,'markers':{'0':{'T_world_marker':tm.tolist()}}}),encoding='utf-8')
    rng=np.random.default_rng(17);points=rng.uniform([-1.6,-1.,2.5],[1.6,1.,5.],(180,3))
    patches=rng.integers(30,230,(180,7,7),dtype=np.uint8)
    k=np.array(cfg.camera_matrix);tag=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),0,120)
    events=[{'kind':'camera_info','timestamp_source':1,'width':640,'height':480,'depth_capability':False}]
    truth=[];frame=0;seq=0
    duration=4.5
    for i in range(int(duration*100)+1):
        t=i/100;u=max(0,t-1.5);omega=2*np.pi/3
        p=np.array([.22*(1-np.cos(omega*u)),0.,0.])
        a=np.array([.22*omega**2*np.cos(omega*u) if t>1.5 else 0.,0.,9.80665])
        for typ,val in [(1,a+rng.normal(0,.008,3)),(4,rng.normal(0,.0003,3))]:
            seq+=1;events.append({'kind':'sensor','seq':seq,'sensor_type':typ,'sensor_key':typ,'accuracy':3,
                                'timestamp_ns':int(t*1e9),'values':val.tolist()})
        truth.append([t,*p])
        if i%10:continue
        frame+=1;image=np.full((480,640),245,np.uint8)
        pc=points-p;pixels=(pc@k.T);pixels=pixels[:,:2]/pixels[:,2:]
        for patch,(x,y) in zip(patches,pixels):
            x,y=int(round(x)),int(round(y))
            if 5<x<635 and 5<y<475:image[y-3:y+4,x-3:x+4]=patch
        if not 2.3<t<3.4:
            half=cfg.marker_size_m/2
            square=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]])
            corners=(tm[:3,:3]@square.T).T+tm[:3,3]-p
            uv=(corners@k.T);uv=uv[:,:2]/uv[:,2:]
            # White quiet zone, then the black-bordered marker itself.
            center=uv.mean(0);quiet=center+(uv-center)*1.4
            cv2.fillConvexPoly(image,np.round(quiet).astype(np.int32),255)
            h=cv2.getPerspectiveTransform(np.float32([[0,0],[119,0],[119,119],[0,119]]),uv.astype(np.float32))
            warped=cv2.warpPerspective(tag,h,(640,480),borderValue=255,flags=cv2.INTER_NEAREST)
            mask=np.zeros_like(image);cv2.fillConvexPoly(mask,np.round(uv).astype(np.int32),255)
            image[mask>0]=warped[mask>0]
        filename=f'frames/{frame:05d}.jpg';cv2.imwrite(str(root/filename),image,[cv2.IMWRITE_JPEG_QUALITY,98])
        seq+=1;events.append({'kind':'frame','seq':seq,'timestamp_ns':int(t*1e9),'width':640,'height':480,'file':filename})
    with (root/'events.jsonl').open('w',encoding='utf-8') as out:
        for event in events:out.write(json.dumps(event)+'\n')
    with (root/'ground_truth.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.writer(out);writer.writerow(['timestamp_s','x','y','z']);writer.writerows(truth)
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='demo_output/synthetic');args=p.parse_args();print(generate(args.output))
