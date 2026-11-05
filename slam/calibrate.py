"""Camera calibration and camera/gyro rotation + time-offset fitting.

Translation extrinsics still need a measured transform or a full calibration tool
such as Kalibr. Rotation-only data cannot identify an arbitrary IMU translation.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from .config import Config
from .dataset import events
from .geometry import exp,log
from .vision import Vision


def camera(folder,output,columns,rows,square_m):
    grid=np.zeros((rows*columns,3),np.float32)
    grid[:,:2]=np.mgrid[0:columns,0:rows].T.reshape(-1,2)*square_m
    object_points=[];image_points=[];size=None
    for path in sorted((Path(folder)/'frames').glob('*')):
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:continue
        if size is not None and size!=(image.shape[1],image.shape[0]):raise ValueError('Calibration images must have the same size')
        size=(image.shape[1],image.shape[0])
        found,corners=cv2.findChessboardCornersSB(image,(columns,rows))
        if found:object_points.append(grid);image_points.append(corners)
    if len(image_points)<12:raise ValueError('Need at least 12 clear chessboard views with varied angles and image coverage')
    rms,k,d,r,t=cv2.calibrateCamera(object_points,image_points,size,None,None)
    cfg=Config(camera_matrix=k.tolist(),distortion=d.ravel().tolist(),image_width=size[0],image_height=size[1])
    Path(output).write_text(json.dumps(vars(cfg),indent=2),encoding='utf-8')
    report={'views':len(image_points),'reprojection_rms_px':float(rms),'camera_only':True,
        'next':'Calibrate camera/IMU rotation, translation and clock offset before setting calibration_verified.'}
    Path(str(output)+'.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def rotation_time(folder,config,output,marker_id):
    cfg=Config.load(config);vision=Vision(cfg);records=events(folder)
    metadata=json.loads((Path(folder)/'metadata.json').read_text())
    key=metadata.get('selected_sensors',{}).get('4')
    gyro=[r for r in records if r['kind']=='sensor' and r['sensor_type']==4 and (key is None or r['sensor_key']==key)]
    if len(gyro)<100:raise ValueError('Insufficient gyroscope data')
    gyro.sort(key=lambda r:r['timestamp_ns']);stamps=np.array([r['timestamp_ns']/1e9 for r in gyro]);values=np.array([r['values'][:3] for r in gyro])
    views=[]
    for r in sorted((r for r in records if r['kind']=='frame'),key=lambda r:r['timestamp_ns']):
        gray=cv2.imread(str(Path(folder)/r['file']),cv2.IMREAD_GRAYSCALE)
        for m in vision.markers(gray):
            if m['id']==marker_id and not m['ambiguous']:views.append((r['timestamp_ns']/1e9,m['T_camera_marker'][:3,:3].T))
    pairs=[]
    for a,b in zip(views[:-3],views[3:]):
        if .1<b[0]-a[0]<.7 and a[0]-.15>stamps[0] and b[0]+.15<stamps[-1]:pairs.append((a,b))
    if len(pairs)<15:raise ValueError('Need a longer sequence rotating around multiple axes with an unambiguous fixed marker visible')
    rotations=np.array([log(a[1].T@b[1]) for a,b in pairs])
    if np.linalg.svd(rotations,compute_uv=False)[1]<.15:raise ValueError('Rotation is not sufficiently excited around two or more axes')
    def delta(start,end,bias):
        ts=np.r_[start,stamps[(stamps>start)&(stamps<end)],end];r=np.eye(3)
        for left,right in zip(ts[:-1],ts[1:]):
            middle=(left+right)/2;w=np.array([np.interp(middle,stamps,values[:,i]) for i in range(3)])-bias
            r=r@exp(w*(right-left))
        return r
    base=np.asarray(cfg.T_body_camera)[:3,:3]
    def residual(x):
        rbc=base@exp(x[:3]);result=[]
        for a,b in pairs:
            body=delta(a[0]+x[3],b[0]+x[3],x[4:7]);camera=a[1].T@b[1]
            result.extend(log((body@rbc).T@(rbc@camera)))
        return result
    result=least_squares(residual,np.zeros(7),bounds=(np.r_[-np.ones(3)*np.pi,-.15,-np.ones(3)*.15],np.r_[np.ones(3)*np.pi,.15,np.ones(3)*.15]),loss='huber',f_scale=.02,max_nfev=100)
    transform=np.array(cfg.T_body_camera);transform[:3,:3]=base@exp(result.x[:3]);cfg.T_body_camera=transform.tolist()
    cfg.camera_imu_offset_s=float(result.x[3]);cfg.calibration_verified=False
    Path(output).write_text(json.dumps(vars(cfg),indent=2),encoding='utf-8')
    report={'pairs':len(pairs),'rms_rotation_rad':float(np.sqrt(np.mean(np.square(result.fun)))),'converged':bool(result.success),
            'camera_imu_offset_s':cfg.camera_imu_offset_s,'estimated_gyro_bias':result.x[4:7].tolist(),
            'translation_not_estimated':True,'clock_offset_search_seconds':[-.15,.15]}
    Path(str(output)+'.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    cam=sub.add_parser('camera');cam.add_argument('dataset');cam.add_argument('--output',default='config/camera.json')
    cam.add_argument('--columns',type=int,default=9);cam.add_argument('--rows',type=int,default=6);cam.add_argument('--square-m',type=float,default=.025)
    rt=sub.add_parser('rotation-time');rt.add_argument('dataset');rt.add_argument('--config',required=True);rt.add_argument('--output',default='config/calibrated.json');rt.add_argument('--marker-id',type=int,default=0)
    args=p.parse_args();Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    report=camera(args.dataset,args.output,args.columns,args.rows,args.square_m) if args.command=='camera' else rotation_time(args.dataset,args.config,args.output,args.marker_id)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
