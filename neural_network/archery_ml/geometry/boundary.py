"""Observed crop silhouette; never an adjustment to canonical coordinates or GT."""
import hashlib

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import _target_color_masks, _ellipse_inliers
from neural_network.archery_ml.geometry.target_templates import CANONICAL_CENTER_PX, CANONICAL_RADIUS_PX


def canonical_display(image, minimum_scoring_zone):
    """Keep holes/occluders inside a measured blue-to-paper outer silhouette.

    Polar samples require blue support inside and neutral paper outside. Missing
    arcs are interpolated only with sufficient angular support. The ellipse is
    an outlier filter, not the final boundary. All limits below govern display
    segmentation only; they cannot change geometry quality/READY or homographies.
    """
    radius=CANONICAL_RADIUS_PX*(11-minimum_scoring_zone)*.1
    yy,xx=np.indices(image.shape[:2])
    dx,dy=xx-CANONICAL_CENTER_PX,yy-CANONICAL_CENTER_PX
    distance=np.hypot(dx,dy)
    alpha=np.clip(radius-distance+.5,0,1)
    info={"status":"MODEL_BOUNDARY","independent_gt":False,"coordinate_space":"canonical_pixels",
          "method":"nominal visible radius","reason":"white scoring zones cannot use the blue edge as outer boundary"}
    if minimum_scoring_zone in {5,6}:
        info.update(status="BOUNDARY_UNCERTAIN",method="blue-to-paper polar transitions",
                    reason="insufficient observed outer-edge support; nominal mask retained")
        blue=_target_color_masks(image)["blue"]>0
        rgb=image.astype(float)
        paper=((rgb.max(axis=2)-rgb.min(axis=2))/np.maximum(1,rgb.max(axis=2))<.28)&(rgb.mean(axis=2)>80)
        angles=np.linspace(0,2*np.pi,720,endpoint=False)
        radial=np.arange(.75*radius,1.25*radius,.5)
        directions=np.column_stack((np.cos(angles),np.sin(angles)))
        points=CANONICAL_CENTER_PX+directions[:,None,:]*radial[None,:,None]
        x,y=np.rint(points).astype(int).transpose(2,0,1)
        inside=(x>=0)&(x<image.shape[1])&(y>=0)&(y<image.shape[0])
        x=np.clip(x,0,image.shape[1]-1);y=np.clip(y,0,image.shape[0]-1)
        b=blue[y,x]&inside;w=paper[y,x]&inside
        transitions=b[:,:-5]&b[:,1:-4]&b[:,2:-3]&w[:,3:-2]&w[:,4:-1]&w[:,5:]
        boundary_radii=radial[2:-3]+.25
        best=np.argmin(np.where(transitions,np.abs(boundary_radii-radius),np.inf),axis=1)
        valid=transitions.any(axis=1)
        observed=boundary_radii[best]
        if valid.sum()>=12:
            edge=CANONICAL_CENTER_PX+directions*observed[:,None]
            try:
                center,axes,angle=cv2.fitEllipse(_ellipse_inliers(edge[valid].astype(np.float32).reshape(-1,1,2)))
                if not np.isfinite([*center,*axes,angle]).all() or min(axes)<=0:
                    raise ValueError("invalid boundary ellipse")
                angle=np.deg2rad(angle);delta=edge-center
                rotated=np.column_stack((delta[:,0]*np.cos(angle)+delta[:,1]*np.sin(angle),
                                         -delta[:,0]*np.sin(angle)+delta[:,1]*np.cos(angle)))
                residual=(np.linalg.norm(rotated/(np.asarray(axes)/2),axis=1)-1)*np.mean(axes)/2
                median=np.median(residual[valid]);mad=np.median(np.abs(residual[valid]-median))
                valid &= np.abs(residual-median)<=max(1.5,4*1.4826*mad)
            except (cv2.error,FloatingPointError,ValueError):
                valid[:]=False
        indices=np.flatnonzero(valid)
        coverage=float(valid.mean())
        gap=float(np.diff(np.r_[indices,indices[0]+720]).max()/2) if len(indices) else 360.
        info.update(angular_coverage=coverage,maximum_unobserved_arc_deg=gap)
        if coverage>=.65 and gap<=30:
            radii=np.interp(np.arange(720),np.r_[indices-720,indices,indices+720],np.tile(observed[valid],3))
            theta=np.mod(np.arctan2(dy,dx),2*np.pi)*720/(2*np.pi)
            limit=np.interp(theta,np.arange(721),np.r_[radii,radii[0]])
            alpha=np.clip(limit-distance+.5,0,1)
            polygon=CANONICAL_CENTER_PX+directions[::4]*radii[::4,None]
            info.update(status="OBSERVED_BOUNDARY",reason="observed blue/paper transitions; short occluded arcs interpolated",
                        polygon_canonical_px=np.round(polygon,3).tolist(),
                        maximum_model_discrepancy_norm=float(np.max(np.abs(radii-radius))/CANONICAL_RADIUS_PX))
    info["mask_sha256"]=hashlib.sha256(np.rint(alpha*255).astype(np.uint8).tobytes()).hexdigest()
    display=np.rint(image*alpha[:,:,None]+128*(1-alpha[:,:,None])).astype(np.uint8)
    return display,info
