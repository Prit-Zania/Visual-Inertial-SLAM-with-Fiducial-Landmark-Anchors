"""Pyramidal KLT tracks, geometric outlier rejection, and calibrated ArUco."""
import cv2
import numpy as np
from .geometry import pose


class Vision:
    def __init__(self, cfg):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.previous = None
        self.points = np.empty((0, 1, 2), np.float32)
        self.ids = []
        self.tracks = {}
        self.next_id = 0

    def markers(self, gray):
        k, d = self.cfg.intrinsics(gray.shape[1], gray.shape[0])
        corners, ids, _ = self.detector.detectMarkers(gray)
        observations = []
        if ids is None:
            return observations
        s = self.cfg.marker_size_m/2
        obj = np.array([[-s,s,0], [s,s,0], [s,-s,0], [-s,-s,0]], np.float64)
        for ident, corner in zip(ids.ravel(), corners):
            uv = corner.reshape(4,2).astype(np.float64)
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(obj,uv,k,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
            candidates = []
            for rv,tv in zip(rvecs,tvecs):
                if tv[2,0] <= .05:
                    continue
                reproj,_ = cv2.projectPoints(obj,rv,tv,k,d)
                err = float(np.sqrt(np.mean(np.sum((reproj.reshape(4,2)-uv)**2,axis=1))))
                candidates.append((err,pose(cv2.Rodrigues(rv)[0],tv.ravel())))
            candidates.sort(key=lambda item:item[0])
            if not candidates or candidates[0][0] > 3.0 or abs(cv2.contourArea(uv.astype(np.float32))) < 100:
                continue
            # Report the planar pose ambiguity; corner updates do not depend on the PnP branch.
            ambiguity = len(candidates)>1 and candidates[1][0]-candidates[0][0] < .2
            observations.append({'id':int(ident),'T_camera_marker':candidates[0][1],
                'corners':uv, 'normalized':cv2.undistortPoints(uv.reshape(-1,1,2),k,d).reshape(-1,2),
                'reprojection_px':candidates[0][0], 'ambiguous':ambiguity})
        return observations

    def track(self, gray, key, markers, oldest=None):
        k,d = self.cfg.intrinsics(gray.shape[1],gray.shape[0])
        ended = []; new_points = []; new_ids = []
        if self.previous is not None and len(self.points):
            nxt, valid, _ = cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,
                winSize=(21,21),maxLevel=3)
            if nxt is not None:
                back, backvalid, _ = cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None,winSize=(21,21),maxLevel=3)
                valid = valid.ravel().astype(bool) & backvalid.ravel().astype(bool)
                valid &= np.linalg.norm(back-self.points,axis=2).ravel()<1.
                xy = nxt.reshape(-1,2)
                valid &= (xy[:,0]>5)&(xy[:,0]<gray.shape[1]-5)&(xy[:,1]>5)&(xy[:,1]<gray.shape[0]-5)
                good = np.flatnonzero(valid)
                if len(good)>=12:
                    _, inlier = cv2.findFundamentalMat(self.points[good].reshape(-1,2),xy[good],cv2.FM_RANSAC,1.5,.99)
                    if inlier is not None:
                        valid[good] &= inlier.ravel().astype(bool)
                normalized = cv2.undistortPoints(nxt,k,d).reshape(-1,2)
                for i,ident in enumerate(self.ids):
                    if valid[i]:
                        self.tracks[ident].append((key,normalized[i]))
                        if oldest is not None and any(t[0]==oldest for t in self.tracks[ident]):
                            ended.append(self.tracks.pop(ident))
                            self.next_id += 1; ident = self.next_id
                            self.tracks[ident] = [(key,normalized[i])]
                        new_points.append(xy[i]); new_ids.append(ident)
                    else:
                        ended.append(self.tracks.pop(ident))
            else:
                ended.extend(self.tracks.values()); self.tracks.clear()
        self.points = np.asarray(new_points,np.float32).reshape(-1,1,2)
        self.ids = new_ids
        mask = np.full(gray.shape,255,np.uint8)
        for p in self.points.reshape(-1,2):
            cv2.circle(mask,tuple(np.round(p).astype(int)),12,0,-1)
        # Fiducial pixels are handled by marker measurements; avoid counting them twice.
        for marker in markers:
            cv2.fillConvexPoly(mask,np.round(marker['corners']).astype(np.int32),0)
        need = self.cfg.max_features-len(self.ids)
        if need>0:
            extra = cv2.goodFeaturesToTrack(gray,need,.015,15,mask=mask)
            if extra is not None:
                norm = cv2.undistortPoints(extra,k,d).reshape(-1,2)
                for uv in norm:
                    self.next_id += 1; self.ids.append(self.next_id)
                    self.tracks[self.next_id] = [(key,uv)]
                self.points = np.concatenate([self.points,extra])
        self.previous = gray.copy()
        return [track for track in ended if len(track)>=3], len(self.ids)

    def reset(self):
        self.previous = None; self.points = np.empty((0,1,2),np.float32)
        self.ids = []; self.tracks.clear()
