"""Independent fixed-window MAP smoother and global marker/visual loop pose graph.

Raw IMU, feature and marker observations are used by the local optimizer. Its
answer is NOT fed back into the MSCKF (which already consumed those observations).
The bounded window fixes its boundary; exported covariance is explicitly conditional.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .geometry import pose, inv, exp, log, perturb, difference, project, triangulate
from .preintegration import Preintegrated


class Smoother:
    def __init__(self,cfg):
        self.cfg=cfg; self.nodes=[]; self.markers={}; self.fixed={}; self.imu={}
        self.observations=[]; self.features=[]; self.loops=[]; self.reports=[]
        self.gravity=np.array([0.,0.,-9.80665]); self.tbc=np.asarray(cfg.T_body_camera)
        self.reference={}

    def add(self,key,t,estimator,samples,markers):
        state={'key':key,'t':t,'R':estimator.R.copy(),'p':estimator.p.copy(),'v':estimator.v.copy(),
               'bg':estimator.bg.copy(),'ba':estimator.ba.copy()}
        if self.nodes and samples and abs(sum(s[0] for s in samples)-(t-self.nodes[-1]['t']))<.002:
            if all(0<s[0]<=self.cfg.imu_max_gap_s for s in samples):
                prior=self.nodes[-1]
                self.imu[len(self.nodes)]=Preintegrated(samples,prior['bg'],prior['ba'],self.cfg)
        self.nodes.append(state)
        self.reference[key]=(pose(estimator.R,estimator.p),estimator.P[:6,:6].copy())
        self.gravity=estimator.g.copy()
        for m in markers:
            ident=m['id']
            if ident in estimator.fixed:
                self.fixed[ident]=estimator.fixed[ident].copy()
            elif ident in estimator.markers:
                self.markers.setdefault(ident,estimator.markers[ident].copy())
            else:
                continue
            self.observations.append((key,ident,m['normalized'].copy(),m['T_camera_marker'].copy(),m['reprojection_px']))

    def rebase(self,t):
        for n in self.nodes:
            n['R']=t[:3,:3]@n['R']; n['p']=t[:3,:3]@n['p']+t[:3,3]; n['v']=t[:3,:3]@n['v']
        for mapping in (self.markers,self.fixed):
            for key in mapping: mapping[key]=t@mapping[key]
        self.gravity=t[:3,:3]@self.gravity
        for key,(p,cov) in self.reference.items():
            j=np.eye(6);j[3:,3:]=t[:3,:3]
            self.reference[key]=(t@p,j@cov@j.T)

    def within_trust_region(self,states):
        """Reject unobservable optimization jumps; this is a quality gate, not a new factor."""
        for key,t in states:
            reference,cov=self.reference[key]
            error=difference(t,reference)
            rotation_limit=max(.1,3*np.sqrt(np.max(np.diag(cov)[:3])))
            position_limit=max(.05,3*np.sqrt(np.max(np.diag(cov)[3:])))
            if np.linalg.norm(error[:3])>rotation_limit or np.linalg.norm(error[3:])>position_limit:return False
        return True

    def optimize_window(self):
        if len(self.nodes)<3:
            return
        nodes=self.nodes[-self.cfg.graph_window:]
        ids={n['key']:i for i,n in enumerate(nodes)}
        camera_poses={n['key']:pose(n['R'],n['p'])@self.tbc for n in nodes}
        obs=[o for o in self.observations if o[0] in ids]
        marker_ids=sorted({o[1] for o in obs if o[1] not in self.fixed})
        marker_base={k:self.markers[k].copy() for k in marker_ids}
        features=[]
        for track in self.features:
            local=[(k,uv) for k,uv in track if k in ids]
            if len(local)>=3:
                try: features.append((local,triangulate(local,camera_poses)))
                except (ValueError,np.linalg.LinAlgError): pass
        features=features[-40:]
        # The oldest navigation state anchors this window. Other states are optimized.
        node_offsets={n['key']:15*(i-1) for i,n in enumerate(nodes) if i>0}
        count=15*(len(nodes)-1)
        marker_offsets={k:count+6*i for i,k in enumerate(marker_ids)}
        count+=6*len(marker_ids)
        point_offsets=[count+3*i for i in range(len(features))]
        count+=3*len(features)
        if not count: return
        focal=np.mean(np.diag(np.asarray(self.cfg.camera_matrix))[:2]); pix=self.cfg.pixel_sigma/focal
        def unpack(x):
            states={}
            for n in nodes:
                state={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in n.items()}
                if n['key'] in node_offsets:
                    off=node_offsets[n['key']]; dx=x[off:off+15]
                    state['R']=n['R']@exp(dx[:3]); state['p']=n['p']+dx[3:6]
                    state['v']=n['v']+dx[6:9]; state['bg']=n['bg']+dx[9:12]; state['ba']=n['ba']+dx[12:15]
                states[n['key']]=state
            markers={**self.fixed,**{k:perturb(marker_base[k],x[i:i+6]) for k,i in marker_offsets.items()}}
            points=[p+x[off:off+3] for (_,p),off in zip(features,point_offsets)]
            return states,markers,points
        global_index={n['key']:i for i,n in enumerate(self.nodes)}
        factors=[]
        def node_cols(key):
            return list(range(node_offsets[key],node_offsets[key]+15)) if key in node_offsets else []
        for a,b in zip(nodes[:-1],nodes[1:]):
            pre=self.imu.get(global_index[b['key']]); ak,bk=a['key'],b['key']
            if pre is not None:
                def inertial(s,m,p,ak=ak,bk=bk,pre=pre):
                    dt=max(pre.dt,.001)
                    return np.r_[pre.residual(s[ak],s[bk],self.gravity),
                        (s[bk]['bg']-s[ak]['bg'])/(self.cfg.gyro_bias_walk*np.sqrt(dt)),
                        (s[bk]['ba']-s[ak]['ba'])/(self.cfg.accel_bias_walk*np.sqrt(dt))]
                factors.append((15,node_cols(ak)+node_cols(bk),inertial))
            else:
                # A gap gets only a weak continuity prior, never a fake IMU measurement.
                rel=inv(pose(a['R'],a['p']))@pose(b['R'],b['p'])
                def weak(s,m,p,ak=ak,bk=bk,rel=rel):
                    predicted=inv(pose(s[ak]['R'],s[ak]['p']))@pose(s[bk]['R'],s[bk]['p'])
                    return difference(rel,predicted)/np.r_[np.ones(3),np.ones(3)*2]
                factors.append((6,node_cols(ak)+node_cols(bk),weak))
        half=self.cfg.marker_size_m/2
        square=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]])
        for key,ident,uv,_,_ in obs:
            def marker(s,m,p,key=key,ident=ident,uv=uv):
                tc=pose(s[key]['R'],s[key]['p'])@self.tbc; tm=m[ident]; residual=[]
                for corner,z in zip(square,uv):
                    try: prediction,_=project(tc,tm[:3,:3]@corner+tm[:3,3]); residual.extend((prediction-z)/pix)
                    except ValueError: residual.extend([100.,100.])
                return residual
            cols=node_cols(key)+(list(range(marker_offsets[ident],marker_offsets[ident]+6)) if ident in marker_offsets else [])
            factors.append((8,cols,marker))
        for fidx,(track,_) in enumerate(features):
            for key,uv in track:
                def visual(s,m,p,key=key,uv=uv,fidx=fidx):
                    try:
                        prediction,_=project(pose(s[key]['R'],s[key]['p'])@self.tbc,p[fidx]); return (prediction-uv)/pix
                    except ValueError: return [100.,100.]
                factors.append((2,node_cols(key)+list(range(point_offsets[fidx],point_offsets[fidx]+3)),visual))
        rows=sum(f[0] for f in factors); pattern=lil_matrix((rows,count),dtype=int); row=0
        for n,cols,_ in factors:
            pattern[row:row+n,cols]=1; row+=n
        def residual(x):
            s,m,p=unpack(x)
            return np.concatenate([np.asarray(fn(s,m,p)) for _,_,fn in factors])
        result=least_squares(residual,np.zeros(count),jac_sparsity=pattern.tocsr(),loss='huber',f_scale=2.,
            x_scale=np.r_[np.tile(np.repeat([.1,.1,.5,.01,.1],3),len(nodes)-1),
                          np.ones(6*len(marker_ids))*.1,np.ones(3*len(features))*.5],
            tr_options={'atol':1e-12,'btol':1e-12,'maxiter':2000},
            ftol=1e-7,xtol=1e-7,gtol=1e-5,max_nfev=100)
        states,markers,_=unpack(result.x)
        accepted=bool(result.success and np.all(np.isfinite(result.x)) and result.cost<=.5*np.sum(residual(np.zeros(count))**2)
            and self.within_trust_region([(key,pose(s['R'],s['p'])) for key,s in states.items()]))
        if accepted:
            for node in nodes: node.update(states[node['key']])
            for k in marker_ids: self.markers[k]=markers[k]
        self.reports.append({'kind':'window','end_frame':nodes[-1]['key'],'cost':float(result.cost),
                             'optimality':float(result.optimality),'converged':bool(result.success),'accepted':accepted,
                             'imu_factors':sum(global_index[n['key']] in self.imu for n in nodes[1:]),
                             'feature_tracks':len(features),'marker_observations':len(obs)})

    def optimize_global(self):
        """Pose graph over local-smoothed odometry, marker revisits and verified visual loops."""
        if len(self.nodes)<2: return
        base=[pose(n['R'],n['p']) for n in self.nodes]
        lookup={n['key']:i for i,n in enumerate(self.nodes)}
        mids=sorted(self.markers); offset={k:6*(len(base)-1)+6*i for i,k in enumerate(mids)}
        count=6*(len(base)-1)+6*len(mids)
        def unpack(x):
            poses=[base[0]]+[perturb(base[i],x[6*(i-1):6*i]) for i in range(1,len(base))]
            markers={**self.fixed,**{k:perturb(self.markers[k],x[o:o+6]) for k,o in offset.items()}}
            return poses,markers
        def cols(i): return list(range(6*(i-1),6*i)) if i>0 else []
        factors=[]
        for i in range(1,len(base)):
            measurement=inv(base[i-1])@base[i]
            sigma=np.r_[np.ones(3)*.05,np.ones(3)*.10]
            factors.append((6,cols(i-1)+cols(i),lambda p,m,i=i,z=measurement,s=sigma:difference(z,inv(p[i-1])@p[i])/s))
        half=self.cfg.marker_size_m/2
        square=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]])
        pixel_sigma=self.cfg.pixel_sigma/np.mean(np.diag(np.asarray(self.cfg.camera_matrix))[:2])
        for key,ident,uv,measurement,error in self.observations:
            i=lookup[key]
            if ident not in self.fixed and ident not in offset: continue
            def corners(p,m,i=i,ident=ident,uv=uv):
                residual=[];tc=p[i]@self.tbc;tm=m[ident]
                for corner,observed in zip(square,uv):
                    try: predicted,_=project(tc,tm[:3,:3]@corner+tm[:3,3]);residual.extend((predicted-observed)/pixel_sigma)
                    except ValueError:residual.extend([100.,100.])
                return residual
            factors.append((8,cols(i)+(list(range(offset[ident],offset[ident]+6)) if ident in offset else []),corners))
        for a,b,z in self.loops:
            ia,ib=lookup[a],lookup[b]
            factors.append((6,cols(ia)+cols(ib),lambda p,m,ia=ia,ib=ib,z=z:difference(z,inv(p[ia]@self.tbc)@(p[ib]@self.tbc))/np.r_[np.ones(3)*.05,np.ones(3)*.1]))
        pattern=lil_matrix((sum(f[0] for f in factors),count),dtype=int);row=0
        for size,c,_ in factors: pattern[row:row+size,c]=1;row+=size
        def residual(x):
            p,m=unpack(x);return np.concatenate([f(p,m) for _,_,f in factors])
        result=least_squares(residual,np.zeros(count),jac_sparsity=pattern.tocsr(),loss='huber',f_scale=2.,
            x_scale=.1,tr_options={'atol':1e-10,'btol':1e-10,'maxiter':2000},
            ftol=1e-6,xtol=1e-7,gtol=1e-5,max_nfev=150)
        poses,markers=unpack(result.x)
        accepted=bool(result.success and self.within_trust_region([(n['key'],t) for n,t in zip(self.nodes,poses)]))
        if accepted:
            for n,t in zip(self.nodes,poses): n['R']=t[:3,:3]; n['p']=t[:3,3]
            for k in self.markers: self.markers[k]=markers[k]
        self.reports.append({'kind':'global_pose_graph','cost':float(result.cost),'converged':bool(result.success),
                             'accepted':accepted,
                             'visual_loops':len(self.loops),'marker_constraints':len(self.observations),
                             'note':'Correlated odometry/marker summaries; robust refinement, not an independent uncertainty estimate.'})
