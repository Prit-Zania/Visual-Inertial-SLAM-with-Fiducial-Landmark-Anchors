"""Timestamp-aligned trajectory evaluation; never estimates a scale to hide metric error."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def table(path):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    return np.array([[float(r[k]) for k in ('timestamp_s','x','y','z')] for r in rows])


def evaluate(trajectory,truth,align=False,max_gap=.2):
    estimate=table(trajectory);gt=table(truth)
    if len(gt)<2 or np.any(np.diff(gt[:,0])<=0):raise ValueError('Ground truth must have increasing timestamps')
    mask=(estimate[:,0]>=gt[0,0])&(estimate[:,0]<=gt[-1,0]);estimate=estimate[mask]
    right=np.clip(np.searchsorted(gt[:,0],estimate[:,0]),1,len(gt)-1)
    mask=(gt[right,0]-gt[right-1,0])<=max_gap;estimate=estimate[mask]
    if len(estimate)<3:raise ValueError('Insufficient time overlap / ground truth samples')
    reference=np.column_stack([np.interp(estimate[:,0],gt[:,0],gt[:,i]) for i in range(1,4)])
    points=estimate[:,1:].copy()
    alignment_rotation=np.eye(3)
    if align:
        a=points-points.mean(0);b=reference-reference.mean(0)
        u,_,vt=np.linalg.svd(a.T@b);r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
        points=a@r.T+reference.mean(0)
        alignment_rotation=r
    error=np.linalg.norm(points-reference,axis=1)
    # Relative displacement error over a 1-second interval.
    rpe=[]
    for i,t in enumerate(estimate[:,0]):
        j=np.searchsorted(estimate[:,0],t+1.)
        if j<len(estimate) and abs(estimate[j,0]-t-1.)<.1:rpe.append(np.linalg.norm((points[j]-points[i])-(reference[j]-reference[i])))
    distance=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
    result={'samples':len(error),'alignment':'rigid_SE3_no_scale' if align else 'none',
        'position_rmse_m':float(np.sqrt(np.mean(error**2))),'position_median_m':float(np.median(error)),
        'position_p95_m':float(np.percentile(error,95)),'endpoint_error_m':float(error[-1]),
        'relative_displacement_1s_rmse_m':float(np.sqrt(np.mean(np.square(rpe)))) if rpe else None,
        'reference_distance_m':distance,'endpoint_error_per_m':float(error[-1]/distance) if distance>0 else None}
    with Path(truth).open(newline='',encoding='utf-8') as f:truth_rows=list(csv.DictReader(f))
    with Path(trajectory).open(newline='',encoding='utf-8') as f:estimate_rows=list(csv.DictReader(f))
    qkeys=('qx','qy','qz','qw')
    if all(k in truth_rows[0] and k in estimate_rows[0] for k in qkeys):
        truth_rot=Rotation.from_quat([[float(row[k]) for k in qkeys] for row in truth_rows])
        interpolated=Slerp(gt[:,0],truth_rot)
        times=np.array([float(row['timestamp_s']) for row in estimate_rows]);valid=(times>=gt[0,0])&(times<=gt[-1,0])
        if valid.any():
            rotations=Rotation.from_quat(np.array([[float(row[k]) for k in qkeys] for row in estimate_rows])[valid])
            errors=(interpolated(times[valid]).inv()*Rotation.from_matrix(alignment_rotation)*rotations).magnitude()
            result['orientation_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(errors**2))))
    return result
