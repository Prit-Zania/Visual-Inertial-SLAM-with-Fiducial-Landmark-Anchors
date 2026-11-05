"""Replay the same timestamped events used by the live receiver."""
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .config import Config
from .dataset import events
from .geometry import inv,pose
from .msckf import MSCKF
from .vision import Vision
from .health import Health
from .graph import Smoother
from .loops import VisualLoops
from .depth import marker_depth


class Engine:
    def __init__(self,cfg,metadata,folder,prior_map=None,use_markers=True):
        self.cfg=cfg;self.folder=Path(folder);self.filter=MSCKF(cfg);self.vision=Vision(cfg)
        self.health=Health(cfg);self.graph=Smoother(cfg);self.loops=VisualLoops(cfg)
        self.selected={int(k):int(v) for k,v in metadata.get('selected_sensors',{}).items()}
        self.samples={'accel':[],'gyro':[]};self.latest={};self.camera_info=None
        self.rows=[];self.detections=[];self.diagnostics=[];self.pending_imu=[];self.frame_id=0
        self.world_anchored=False;self.use_markers=use_markers;self.last_snapshot=-1e20;self.last_frame=None
        self.pending_imu_broken=False;self.depth_events=[]
        self.imu_initialized=False;self.relocalization_candidate=None
        self.start_t=None
        self.loaded_marker_ids=set()
        if prior_map:
            data=json.loads(Path(prior_map).read_text(encoding='utf-8'))
            if abs(data['marker_size_m']-cfg.marker_size_m)>1e-5:raise ValueError('Map marker size differs')
            if data.get('dictionary',cfg.dictionary)!=cfg.dictionary:raise ValueError('Map dictionary differs')
            for key,entry in data['markers'].items():
                ident=int(key);t=np.asarray(entry['T_world_marker'],float)
                if t.shape!=(4,4) or not np.all(np.isfinite(t)) or not np.allclose(t[3],[0,0,0,1]):raise ValueError('Invalid map transform')
                if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(t[:3,:3])<.999:raise ValueError('Invalid map rotation')
                self.loaded_marker_ids.add(ident)
                cov=entry.get('filter_covariance')
                if entry.get('fixed',cov is None):self.filter.fixed[ident]=t
                else:
                    # Independent previous-run landmark prior; retain its uncertainty.
                    covariance=np.asarray(cov,float)
                    if covariance.shape!=(6,6) or not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance).min()<-1e-8:raise ValueError('Invalid map covariance')
                    n=len(self.filter.P);aug=np.zeros((n+6,n+6));aug[:n,:n]=self.filter.P
                    aug[n:,n:]=covariance+np.eye(6)*1e-8;self.filter.P=aug;self.filter.markers[ident]=t

    def rebase(self,transform):
        fixed=self.filter.fixed.copy()
        learned={key:self.filter.markers[key].copy() for key in self.loaded_marker_ids if key in self.filter.markers}
        original_cov=self.filter.P.copy()
        self.filter.rebase(transform)
        self.filter.fixed=fixed
        for key,t in learned.items():
            self.filter.markers[key]=t
            i=self.filter.marker_index(key)
            # Before initial alignment, loaded priors are independent of navigation.
            self.filter.P[i:i+6,:]=0;self.filter.P[:,i:i+6]=0
            self.filter.P[i:i+6,i:i+6]=original_cov[i:i+6,i:i+6]
        self.graph.rebase(transform);self.loops.rebase(transform)
        for row in self.rows:
            t=transform@np.asarray(row['T_world_body']);row['T_world_body']=t.tolist()
            p=np.asarray(row['position_covariance']);r=transform[:3,:3]
            row['position_covariance']=(r@p@r.T).tolist();row['position_sigma_m']=np.sqrt(np.diag(r@p@r.T)).tolist()
            jac=np.eye(6);jac[3:,3:]=r
            row['pose_covariance']=(jac@np.asarray(row['pose_covariance'])@jac.T).tolist()
            row['velocity_mps']=(r@row['velocity_mps']).tolist()
        self.world_anchored=True

    def propagate_to(self,t):
        f=self.filter
        if not f.ready or t<=f.t:return
        a=self.latest.get('accel');w=self.latest.get('gyro')
        valid_a=self.imu_initialized and a is not None and t-a[0]<=self.cfg.imu_max_gap_s
        valid_w=w is not None and t-w[0]<=self.cfg.imu_max_gap_s
        dt=t-f.t
        good=f.propagate(t,a[1] if valid_a else None,w[1] if valid_w else None)
        if good:self.pending_imu.append((dt,a[1].copy(),w[1].copy()))
        elif dt>0:self.pending_imu_broken=True
        if not good:self.diagnostics.append({'t':t,'kind':'imu_gap','duration_s':dt})

    def consume(self,event):
        kind=event['kind']
        if kind=='camera_info':self.camera_info=event;return
        if kind not in ('sensor','frame'):return
        t=event['timestamp_ns']/1e9
        if self.start_t is None:self.start_t=t
        if kind=='sensor':
            typ=event['sensor_type'];name={1:'accel',4:'gyro',2:'compass'}.get(typ)
            if name is None:return
            self.selected.setdefault(typ,event['sensor_key'])
            if event['sensor_key']!=self.selected[typ]:return
            values=event['values'][:3]
            if len(values)<3 or any(v is None for v in values):return
            values=np.array(values,float)
            if name=='accel':values=np.asarray(self.cfg.accel_matrix)@(values-np.asarray(self.cfg.accel_offset))
            if (name=='accel' and np.linalg.norm(values)>120) or (name=='gyro' and np.linalg.norm(values)>35):
                self.health.failures[name]='out_of_range';return
            if not self.health.sample(name,t,values):return
            if name in ('accel','gyro'):
                self.latest[name]=(t,values)
                self.samples[name].append((t,values))
                self.samples[name]=[(stamp,value) for stamp,value in self.samples[name] if t-stamp<=max(2.,self.cfg.initialization_s+.2)]
            f=self.filter
            if not self.imu_initialized:
                a,w=self.samples['accel'],self.samples['gyro']
                if a and w and min(t-a[0][0],t-w[0][0])>=self.cfg.initialization_s:
                    try:
                        if f.ready:
                            temporary=MSCKF(self.cfg);temporary.initialize(t,[v for _,v in a],[v for _,v in w])
                            f.bg=temporary.bg;f.ba=temporary.ba
                            gravity=-f.R@(np.mean([v for _,v in a],axis=0)-f.ba)
                            f.g=gravity/np.linalg.norm(gravity)*9.80665
                        else:f.initialize(t,[v for _,v in a],[v for _,v in w])
                        self.imu_initialized=True
                        self.diagnostics.append({'t':t,'kind':'initialized','gyro_bias':f.bg.tolist(),'accel_bias':f.ba.tolist()})
                    except ValueError:pass
            if not f.ready:
                return
            if name=='gyro':
                self.propagate_to(t)
                if t-self.last_snapshot>=.1:self.snapshot(t,None)
            elif name=='compass':
                accepted=f.compass_update(values,event['accuracy'])
                if accepted:self.health.active['compass']=t
                else:self.health.failures['compass']='disabled_or_untrusted'
                self.diagnostics.append({'t':t,'kind':'compass','accepted':accepted})
            return
        # Frames must be synchronized to the IMU clock; callback arrival time is never used.
        if self.camera_info is None:raise ValueError('Missing Camera2 camera_info record')
        if self.camera_info.get('timestamp_source')!=1 and not self.cfg.camera_clock_offset_verified:
            raise ValueError('Camera clock is UNKNOWN: calibrate its clock offset and set camera_clock_offset_verified')
        t+=self.cfg.camera_imu_offset_s
        if not self.filter.ready:
            # Marker pose permits a degraded visual start even with a missing IMU.
            # Give stationary initialization its first two seconds before using this fallback.
            if t-self.start_t<2.:return
            startup_path=(self.folder/event['file']).resolve()
            if not startup_path.is_relative_to(self.folder.resolve()):raise ValueError('Frame path escapes dataset')
            startup_gray=cv2.imread(str(startup_path),cv2.IMREAD_GRAYSCALE)
            if startup_gray is None:return
            startup_observations=self.vision.markers(startup_gray) if self.use_markers else []
            if not startup_observations:return
            self.filter.ready=True;self.filter.t=t;self.filter.P[:9,:9]+=np.eye(9)*.25
            self.diagnostics.append({'t':t,'kind':'visual_marker_start','imu_initialized':False})
        if t<self.filter.t-1e-6:
            self.diagnostics.append({'t':t,'kind':'late_frame_rejected'});return
        if not self.health.sample('camera',t):return
        self.propagate_to(t)
        path=(self.folder/event['file']).resolve()
        if not path.is_relative_to(self.folder.resolve()):raise ValueError('Frame path escapes dataset')
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('Unreadable camera frame: '+str(path))
        f=self.filter;self.frame_id+=1;key=self.frame_id
        k,_=self.cfg.intrinsics(gray.shape[1],gray.shape[0])
        if self.last_frame is not None and t-self.last_frame>self.cfg.camera_timeout_s:
            self.vision.reset()
            for old in list(f.clones):f.remove_clone(old)
        observations=self.vision.markers(gray) if self.use_markers else []
        if not self.world_anchored and observations:
            reference={**f.fixed,**{k:f.markers[k] for k in self.loaded_marker_ids if k in f.markers}}
            known=[o for o in observations if o['id'] in reference]
            if reference and not known:
                pass
            else:
                first=min(known or observations,key=lambda m:m['reprojection_px'])
                if first['ambiguous']:
                    self.diagnostics.append({'t':t,'kind':'ambiguous_anchor_pose','marker':first['id']})
                target=reference[first['id']] if known else np.eye(4)
                transform=target@inv(f.camera@first['T_camera_marker'])
                self.rebase(transform)
                if not known:f.fixed[first['id']]=np.eye(4)
        for obs in observations:
            ident=obs['id'];accepted=False;initialized=False
            if ident not in f.fixed and ident not in f.markers:
                initialized=f.add_marker(ident,obs['T_camera_marker'],sigma_position=max(.02,obs['reprojection_px']*.02))
                # Its first observation initialized the state: do not update with it twice.
            else:
                accepted=f.marker_update(ident,obs['normalized'],float((k[0,0]+k[1,1])/2))
                if not accepted and ident in f.fixed and self.health.status(t,f)['tracking']=='LOST':
                    candidate=f.fixed[ident]@inv(obs['T_camera_marker'])@inv(np.asarray(self.cfg.T_body_camera))
                    prior=self.relocalization_candidate
                    if prior is not None and prior[0]==ident and 0<t-prior[1]<.5 and np.linalg.norm(candidate[:3,3]-prior[2][:3,3])<.3:
                        from .geometry import log
                        if np.linalg.norm(log(prior[2][:3,:3].T@candidate[:3,:3]))<.25:
                            # A deliberate reinitialization invalidates old clone cross-covariances.
                            # Keep independently fixed anchors; learned markers must be remapped.
                            f.R=candidate[:3,:3].copy();f.p=candidate[:3,3].copy();f.v[:]=0
                            f.clones.clear();f.fej.clear();f.markers.clear()
                            f.P=np.diag(np.r_[np.ones(3)*.08**2,np.ones(3)*.1**2,np.ones(3)*.5**2,
                                              np.ones(3)*.02**2,np.ones(3)*.2**2])
                            self.vision.reset();accepted=True;self.pending_imu_broken=True
                            f.last_update={'kind':'fixed_anchor_reinitialization','accepted':True,'marker':ident}
                            self.diagnostics.append({'t':t,'kind':'relocalized_fixed_anchor','marker':ident})
                    self.relocalization_candidate=(ident,t,candidate)
            if accepted:self.health.constraint('aruco',t)
            if self.depth_events:
                nearest=min(self.depth_events,key=lambda e:abs(e['timestamp_ns']/1e9+self.cfg.camera_imu_offset_s-t))
                if abs(nearest['timestamp_ns']/1e9+self.cfg.camera_imu_offset_s-t)<.025:
                    measurement=marker_depth(self.folder,nearest,obs,self.cfg,k)
                    if measurement is not None:
                        used=f.range_update(ident,*measurement)
                        if used:self.health.active['validated_depth']=t
                        self.diagnostics.append({'t':t,'kind':'depth','marker':ident,'accepted':used,'depth_m':measurement[0],'sigma_m':measurement[1]})
            self.detections.append({'t':t,'frame':key,'id':ident,'accepted':accepted,'initialized':initialized,
                'reprojection_px':obs['reprojection_px'],'ambiguous':obs['ambiguous'],
                'T_camera_marker':obs['T_camera_marker'].tolist(),'corners':obs['corners'].tolist()})
        f.add_clone(key)
        oldest=next(iter(f.clones)) if len(f.clones)>self.cfg.max_clones else None
        tracks,count=self.vision.track(gray,key,observations,oldest)
        used=f.feature_update(tracks,float((k[0,0]+k[1,1])/2))
        if used:self.health.constraint('visual_features',t)
        if oldest is not None:f.remove_clone(oldest)
        self.graph.features.extend(tracks)
        self.graph.add(key,t,f,[] if self.pending_imu_broken else self.pending_imu,observations)
        self.pending_imu=[];self.pending_imu_broken=False
        if key%5==0:self.graph.loops.extend(self.loops.add(key,t,gray,f.camera))
        self.last_frame=t
        self.snapshot(t,key,{'tracked_features':count,'accepted_tracks':used,'visible_markers':[o['id'] for o in observations]})

    def snapshot(self,t,frame,extra=None):
        self.last_snapshot=t;f=self.filter
        row={'timestamp_s':t,'frame':frame,'T_world_body':f.T.tolist(),'velocity_mps':f.v.tolist(),
            'gyro_bias_radps':f.bg.tolist(),'accel_bias_mps2':f.ba.tolist(),**f.status(),**self.health.status(t,f)}
        if extra:row.update(extra)
        self.rows.append(row)

    def save(self,output,smooth=True):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if not self.filter.ready:raise ValueError('IMU initialization failed: record at least 2 stationary seconds with accel and gyro')
        if not self.graph.nodes:raise ValueError('No synchronized frames after initialization')
        if smooth:
            # Replay bounded windows; the final global graph applies marker/visual loop constraints.
            all_nodes=self.graph.nodes
            for end in range(3,len(all_nodes)+1,4):
                self.graph.nodes=all_nodes[:end];self.graph.optimize_window()
            self.graph.nodes=all_nodes;self.graph.optimize_window();self.graph.optimize_global()
        with (output/'trajectory.jsonl').open('w',encoding='utf-8') as stream:
            for row in self.rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
        with (output/'trajectory.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['timestamp_s','x','y','z','qx','qy','qz','qw','sigma_x','sigma_y','sigma_z','tracking'])
            for row in self.rows:
                t=np.asarray(row['T_world_body']);q=Rotation.from_matrix(t[:3,:3]).as_quat()
                writer.writerow([row['timestamp_s'],*t[:3,3],*q,*row['position_sigma_m'],row['tracking']])
        optimized=[{'timestamp_s':n['t'],'frame':n['key'],'T_world_body':pose(n['R'],n['p']).tolist()} for n in self.graph.nodes]
        (output/'optimized_trajectory.json').write_text(json.dumps(optimized,indent=2),encoding='utf-8')
        markers={}
        for ident,t in {**self.filter.markers,**self.filter.fixed}.items():
            cov=None
            if ident in self.filter.markers:
                i=self.filter.marker_index(ident);cov=self.filter.P[i:i+6,i:i+6].tolist()
            markers[str(ident)]={'T_world_marker':t.tolist(),
                                'optimized_T_world_marker':self.graph.markers.get(ident,self.filter.fixed.get(ident,t)).tolist(),
                                'filter_T_world_marker':t.tolist(),'filter_covariance':cov,'fixed':ident in self.filter.fixed}
        (output/'marker_map.json').write_text(json.dumps({'marker_size_m':self.cfg.marker_size_m,
            'dictionary':self.cfg.dictionary,'frame':'first marker or supplied map' if self.world_anchored else 'initial gravity-aligned IMU',
            'markers':markers},indent=2),encoding='utf-8')
        for name,data in [('detections',self.detections),('diagnostics',self.diagnostics),('optimization',self.graph.reports)]:
            (output/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
        summary={'frames':self.frame_id,'state_samples':len(self.rows),'calibration_verified':self.cfg.calibration_verified,
            'imu_initialized':self.imu_initialized,
            'filter':'15-error-state MSCKF + correlated camera clones + ArUco landmark states',
            'counts':self.filter.counts,'marker_count':len(markers),'visual_loops':len(self.graph.loops),
            'optimization_attempts':len(self.graph.reports),
            'optimization_accepted':sum(r.get('accepted',False) for r in self.graph.reports),
            'tracking_counts':{mode:sum(r['tracking']==mode for r in self.rows) for mode in ('INITIALIZING','TRACKING','DEGRADED','LOST')},
            'confidence':'MSCKF covariance is model-based; verify coverage against surveyed ground truth. Optimized poses do not inherit filter covariance.',
            'limitations':['Camera/IMU calibration must be measured on the actual phone.',
                'Fixed-window boundary is held constant; this is an educational smoother, not full marginalization.',
                'No phone hardware validation or APK build has been performed.',
                'Depth is recorded when exposed; range fusion requires explicit calibrated measurements.']}
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (output/'configuration.json').write_text(json.dumps(vars(self.cfg),indent=2),encoding='utf-8')
        return summary


def process(folder,config,prior_map=None,use_markers=True,smooth=True,allow_uncalibrated=False,output=None):
    cfg=Config.load(config)
    if not cfg.calibration_verified and not allow_uncalibrated:
        raise ValueError('Calibration is not verified. Calibrate first, or use --allow-uncalibrated only for exploratory runs.')
    metadata=json.loads((Path(folder)/'metadata.json').read_text(encoding='utf-8'))
    engine=Engine(cfg,metadata,folder,prior_map,use_markers)
    records=events(folder)
    for record in records:
        if record['kind']=='camera_info':engine.consume(record)
    engine.depth_events=[r for r in records if r['kind']=='depth']
    timed=[r for r in records if r['kind'] in ('sensor','frame')]
    timed.sort(key=lambda r:(r['timestamp_ns']/1e9+(cfg.camera_imu_offset_s if r['kind']=='frame' else 0),0 if r['kind']=='sensor' else 1))
    began=time.monotonic()
    for record in timed:engine.consume(record)
    output=Path(output) if output else Path(folder)/('processed' if use_markers else 'baseline')
    summary=engine.save(output,smooth=smooth);summary['runtime_s']=time.monotonic()-began
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    from .report import plot
    plot(output)
    return output,summary
