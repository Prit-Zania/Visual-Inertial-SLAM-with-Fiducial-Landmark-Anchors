"""Conservative visual loop candidates verified against triangulated 3-D points."""
import cv2
import numpy as np
from .geometry import pose,inv


class VisualLoops:
    def __init__(self,cfg):
        self.cfg=cfg; self.orb=cv2.ORB_create(nfeatures=700); self.history=[]

    @staticmethod
    def matches(a,b):
        if a is None or b is None or len(a)<2 or len(b)<2: return []
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a,b,k=2)
        return [x[0] for x in pairs if len(x)==2 and x[0].distance<.7*x[1].distance]

    def add(self,key,t,gray,twc):
        k,d=self.cfg.intrinsics(gray.shape[1],gray.shape[0])
        kp,desc=self.orb.detectAndCompute(gray,None)
        xy=np.array([p.pt for p in kp],np.float32).reshape(-1,2)
        current={'key':key,'t':t,'T':twc.copy(),'xy':xy,'desc':desc,'points':{}}
        loops=[]
        for old in self.history[:-5]:
            if t-old['t']<5 or len(old['points'])<20: continue
            matches=[m for m in self.matches(old['desc'],desc) if m.queryIdx in old['points']]
            if len(matches)<20: continue
            points=np.array([old['points'][m.queryIdx] for m in matches],np.float64)
            pixels=np.array([xy[m.trainIdx] for m in matches],np.float64)
            ok,rv,tv,inliers=cv2.solvePnPRansac(points,pixels,k,d,iterationsCount=150,reprojectionError=2.,confidence=.999)
            if not ok or inliers is None or len(inliers)<20 or len(inliers)<.6*len(matches): continue
            t_est=inv(pose(cv2.Rodrigues(rv)[0],tv.ravel()))
            projection,_=cv2.projectPoints(points[inliers.ravel()],rv,tv,k,d)
            rms=np.sqrt(np.mean(np.sum((projection.reshape(-1,2)-pixels[inliers.ravel()])**2,axis=1)))
            if rms<1.5:
                loops.append((old['key'],key,inv(old['T'])@t_est)); break
        if self.history and len(kp):
            prior=self.history[-1]
            matches=self.matches(prior['desc'],desc)
            if len(matches)>=12 and np.linalg.norm(twc[:3,3]-prior['T'][:3,3])>.03:
                a=np.array([prior['xy'][m.queryIdx] for m in matches]); b=np.array([xy[m.trainIdx] for m in matches])
                ua=cv2.undistortPoints(a.reshape(-1,1,2),k,d).reshape(-1,2)
                ub=cv2.undistortPoints(b.reshape(-1,1,2),k,d).reshape(-1,2)
                pa,pb=inv(prior['T'])[:3],inv(twc)[:3]
                xyz=cv2.triangulatePoints(pa,pb,ua.T,ub.T)
                for i,m in enumerate(matches):
                    if abs(xyz[3,i])<1e-8: continue
                    point=xyz[:3,i]/xyz[3,i]
                    ca=pa[:3,:3]@point+pa[:3,3]; cb=pb[:3,:3]@point+pb[:3,3]
                    if min(ca[2],cb[2])>.2 and max(ca[2],cb[2])<20 and np.linalg.norm(ca[:2]/ca[2]-ua[i])<.003 and np.linalg.norm(cb[:2]/cb[2]-ub[i])<.003:
                        current['points'][m.trainIdx]=point
        self.history.append(current)
        self.history=self.history[-300:]
        return loops

    def rebase(self,t):
        for item in self.history:
            item['T']=t@item['T']
            item['points']={k:t[:3,:3]@v+t[:3,3] for k,v in item['points'].items()}
