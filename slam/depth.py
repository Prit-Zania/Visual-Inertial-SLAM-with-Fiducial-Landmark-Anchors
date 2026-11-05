"""Optional registered DEPTH16 marker-centre measurement with confidence gating."""
from pathlib import Path
import numpy as np


def marker_depth(folder,event,marker,config,k):
    if not (config.range_enabled and config.range_calibration_verified and config.depth_camera_matrix is not None and config.T_camera_depth is not None):
        return None
    path=(Path(folder)/event['file']).resolve()
    if not path.is_relative_to(Path(folder).resolve()):raise ValueError('Depth path escapes dataset')
    data=path.read_bytes();w,h=event['width'],event['height'];row=event['row_stride'];pixel=event['pixel_stride']
    if pixel!=2 or len(data)<(h-1)*row+w*pixel:return None
    packed=np.ndarray((h,w),dtype='<u2',buffer=data,strides=(row,pixel))
    z=(packed&8191).astype(float)/1000
    bits=packed>>13
    confidence=np.where(bits==0,1.,(bits.astype(float)-1)/7)
    yy,xx=np.indices((h,w));mask=(z>.1)&(z<15)&(confidence>=.7)
    kd=np.array(config.depth_camera_matrix);td=np.array(config.T_camera_depth)
    rays=np.linalg.inv(kd)@np.vstack([xx[mask],yy[mask],np.ones(mask.sum())])
    pc=td[:3,:3]@(rays*z[mask])+td[:3,3:4]
    positive=pc[2]>.1;pc=pc[:,positive]
    if pc.shape[1]<5:return None
    uv=k@pc;uv=uv[:2]/uv[2:]
    center=marker['corners'].mean(0)
    radius=max(2.,np.min(np.linalg.norm(marker['corners']-center,axis=1))*.3)
    values=pc[2,np.linalg.norm(uv.T-center,axis=1)<radius]
    if len(values)<5:return None
    median=float(np.median(values));spread=float(1.4826*np.median(np.abs(values-median)))
    if spread>.10:return None
    return median,max(.02,spread)
