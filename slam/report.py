"""Headless plots and quantitative comparison for exported recordings."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from .evaluate import evaluate,table


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    rows=[json.loads(r) for r in (output/'trajectory.jsonl').read_text().splitlines()]
    positions=np.array([np.array(r['T_world_body'])[:3,3] for r in rows]);times=np.array([r['timestamp_s'] for r in rows]);times-=times[0]
    sigma=np.array([r['position_sigma_m'] for r in rows])
    optimized=json.loads((output/'optimized_trajectory.json').read_text())
    markers=json.loads((output/'marker_map.json').read_text())['markers']
    figure=plt.figure(figsize=(12,5.2),layout='constrained')
    ax=figure.add_subplot(121,projection='3d');ax.plot(*positions.T,label='MSCKF',color='#146d9f',linewidth=2)
    if optimized:
        op=np.array([np.array(r['T_world_body'])[:3,3] for r in optimized]);ax.plot(*op.T,label='Map refinement',color='#ce7f16')
    for key,marker in markers.items():
        p=np.array(marker['T_world_marker'])[:3,3];ax.scatter(*p,color='black',marker='s');ax.text(*p,' ID '+key)
    ax.set(xlabel='X / m',ylabel='Y / m',zlabel='Z / m',title='Trajectory and landmark map');ax.legend()
    right=figure.add_subplot(122)
    for i,label in enumerate('XYZ'):right.plot(times,2*sigma[:,i],label=label)
    lost=np.array([r['tracking']=='LOST' for r in rows]);right.fill_between(times,0,float(np.max(2*sigma))+.01,where=lost,color='#c84c4c',alpha=.15,label='Tracking lost')
    right.set(xlabel='Elapsed time / s',ylabel='Approximate 2-sigma uncertainty / m',title='Position uncertainty (MSCKF)');right.grid(alpha=.2);right.legend()
    figure.savefig(output/'trajectory.png',dpi=140);plt.close(figure)


def confidence(output,truth):
    output=Path(output);gt=table(truth);values=[]
    rows=[json.loads(line) for line in (output/'trajectory.jsonl').read_text().splitlines()]
    for row in rows:
        t=row['timestamp_s']
        if t<gt[0,0] or t>gt[-1,0]:continue
        p=np.array(row['T_world_body'])[:3,3];reference=np.array([np.interp(t,gt[:,0],gt[:,i]) for i in range(1,4)])
        error=p-reference;cov=np.array(row['position_covariance'])
        values.append(float(error@np.linalg.solve(cov+np.eye(3)*1e-12,error)))
    return {'position_nees_mean':float(np.mean(values)) if values else None,
            'position_95pct_ellipsoid_coverage':float(np.mean(np.array(values)<=chi2.ppf(.95,3))) if values else None,
            'expected_position_nees_if_model_consistent':3,'samples':len(values),
            'note':'Same coordinate frame required. Synthetic coverage does not validate phone noise calibration.'}


def compare(dataset):
    dataset=Path(dataset);report={}
    for mode in ('processed','baseline'):
        folder=dataset/mode
        if (folder/'trajectory.csv').exists():
            report[mode]=evaluate(folder/'trajectory.csv',dataset/'ground_truth.csv')
            if mode=='processed':report[mode]['uncertainty']=confidence(folder,dataset/'ground_truth.csv')
            plot(folder)
    (dataset/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset');args=p.parse_args();print(json.dumps(compare(args.dataset),indent=2))
