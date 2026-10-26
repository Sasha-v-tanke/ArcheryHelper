"""Target candidates and conservative grouping; appearance never establishes ownership."""
from dataclasses import asdict, replace
from itertools import combinations
import hashlib
import json
import math
from datetime import datetime

import cv2
import numpy as np

from neural_network.archery_ml.geometry.detector import (
    _target_color_masks, _face_candidates, _associated_rings, _order_candidates, _Candidate,
)
from neural_network.archery_ml.geometry.target_templates import geometry_template
from neural_network.archery_ml.targets import TargetConfig, TargetFormat, TenRingMode, TripleLayout


def config_for_group(group, minimum_zone=None):
    return TargetConfig(TargetFormat(group["format"]),minimum_zone or group.get("minimum_zone_proposal",5),
                        TenRingMode.RECURVE,1,None if group["layout"] is None else TripleLayout(group["layout"]))


class TargetCandidateDetector:
    def detect(self, image):
        height,width = image.shape[:2]
        factor = min(1.,1600/max(height,width))
        resized = cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA) if factor < 1 else image
        masks = _target_color_masks(resized)
        template = geometry_template(TargetConfig(TargetFormat.SINGLE,5,TenRingMode.RECURVE,1))
        pool = _face_candidates(masks,resized.shape[1],resized.shape[0],apply_area_floor=False)
        results = []
        for candidate in pool:
            rings = _associated_rings(candidate,masks,template)
            centers, red_ratio = [], None
            for points,radius in rings:
                if len(points) < 5:
                    continue
                center,axes,_ = cv2.fitEllipse(np.asarray(points,dtype=np.float32))
                if radius != template.colored_outer_radius_norm:
                    centers.append(math.dist(center,(candidate.ellipse.center_x,candidate.ellipse.center_y)) / candidate.ellipse.minor_axis_px)
                if radius == .4:
                    red_ratio = max(axes)/candidate.ellipse.major_axis_px
            deviation = max(centers,default=1.)
            confidence = max(0.,min(1.,(len(rings)/3) * math.exp(-4*deviation)))
            ellipse = candidate.ellipse
            # Keep weak candidates visible for manual selection, but do not group
            # isolated background colour fragments lacking concentric ring evidence.
            if len(rings) < 2 or deviation > .4:
                continue
            points = cv2.ellipse2Poly((round(ellipse.center_x/factor),round(ellipse.center_y/factor)),
                (max(1,round(ellipse.diameter_x/2/factor)),max(1,round(ellipse.diameter_y/2/factor))),round(ellipse.angle_deg),0,360,5)
            x,y,w,h = cv2.boundingRect(points)
            results.append({"center":[ellipse.center_x/factor,ellipse.center_y/factor],
                "bbox":[x,y,x+w,y+h],"diameter":ellipse.major_axis_px/factor,
                "ellipse":asdict(replace(ellipse,center_x=ellipse.center_x/factor,center_y=ellipse.center_y/factor,
                    diameter_x=ellipse.diameter_x/factor,diameter_y=ellipse.diameter_y/factor)),
                "confidence":confidence,"ring_count":len(rings),"concentricity_proxy":deviation,
                "red_to_blue_radius_ratio":red_ratio,
                "minimum_zone_proposal":6 if red_ratio is not None and red_ratio > .735 else 5,
                "config_source":"ring-ratio proposal, not confirmed TargetConfig","coordinate_space":"upright_rgb_full_image",
                "observed_ring_contours_rgb":[{"radius_label_proposal":radius,"points":(np.asarray(points)/factor).tolist()} for points,radius in rings],
                "ambiguity_reasons":[] if confidence >= .65 else ["weak ring/center evidence"]})
        results.sort(key=lambda r:(r["center"][1],r["center"][0]))
        for index,result in enumerate(results):
            result.update(id=f"f{index}",groups=[])
        return results


class TargetGroupBuilder:
    def build(self, candidates, image=None):
        groups = []
        for candidate in candidates:
            groups.append({"format":"SINGLE","layout":None,"candidate_ids":[candidate["id"]],
                "confidence":candidate["confidence"],"minimum_zone_proposal":1,
                "ambiguity_reasons":["single face may belong to an unselected triple"] if len(candidates)>1 else []})
        for triple in combinations(candidates,3):
            diameters = [c["diameter"] for c in triple]
            if max(diameters)/min(diameters) > 1.6:
                continue
            centers = np.array([c["center"] for c in triple])
            mean_size = float(np.median(diameters))
            distances = [np.linalg.norm(a-b)/mean_size for a,b in combinations(centers,2)]
            if min(distances) < .85 or max(distances) > 4.2:
                continue
            centered = centers-centers.mean(axis=0)
            _,_,vt = np.linalg.svd(centered)
            along,cross = centered@vt[0],centered@vt[1]
            order = np.argsort(along)
            gaps = np.diff(along[order])/mean_size
            vertical = max(abs(cross))/mean_size < .16 and min(gaps) > .85 and max(gaps)/min(gaps)<1.55
            # An upright-paper prior rejects horizontal rows/diagonals formed from
            # several adjacent sheets. It does not establish physical orientation.
            vertical = vertical and abs(vt[0,1]) > .85
            layout = "VERTICAL" if vertical else None
            if not vertical:
                triangle_area = abs(np.linalg.det(np.column_stack((centers[1]-centers[0],centers[2]-centers[0]))))/2
                top_index = int(np.argmin(centers[:,1]))
                base = np.delete(centers,top_index,axis=0)
                upright_triangle = np.ptp(base[:,1])<.35*mean_size and min(base[:,1]-centers[top_index,1])>.5*mean_size
                if max(distances)/min(distances)<1.35 and triangle_area/mean_size**2 > .45 and upright_triangle:
                    layout = "TRIANGULAR"
            if layout is None:
                continue
            # Do not skip an intermediate face or accept a narrow cross-sheet
            # diagonal when another plausible face lies inside the group envelope.
            lo,hi = centers.min(axis=0),centers.max(axis=0)
            members = {c["id"] for c in triple}
            interfering = [c for c in candidates if c["id"] not in members and np.all(np.array(c["center"])>=lo-.15*mean_size) and np.all(np.array(c["center"])<=hi+.15*mean_size)]
            if interfering:
                continue
            if layout == "VERTICAL":
                ordered = sorted(triple,key=lambda c:c["center"][1])
            else:
                top = min(triple,key=lambda c:c["center"][1])
                ordered = [top,*sorted((c for c in triple if c is not top),key=lambda c:c["center"][0])]
            zone_votes = [c["minimum_zone_proposal"] for c in triple]
            gap_evidence = []
            paper_sides = []
            if image is not None:
                for a,b in zip(ordered,ordered[1:]):
                    start,end = np.array(a["center"]),np.array(b["center"])
                    length = np.linalg.norm(end-start)
                    fractions = np.linspace(.55*a["diameter"]/length,1-.55*b["diameter"]/length,24)
                    if fractions[0] >= fractions[-1]:
                        gap_evidence.append(0.)
                        continue
                    points = np.rint(start[None,:]+fractions[:,None]*(end-start)).astype(int)
                    points[:,0] = np.clip(points[:,0],0,image.shape[1]-1)
                    points[:,1] = np.clip(points[:,1],0,image.shape[0]-1)
                    rgb = image[points[:,1],points[:,0]].astype(float)
                    saturation = (rgb.max(axis=1)-rgb.min(axis=1))/np.maximum(1,rgb.max(axis=1))
                    gap_evidence.append(float(np.mean((saturation<.28)&(rgb.mean(axis=1)>80))))
                # The blue disks may almost touch. Test the continuous white
                # margins of the paper instead of demanding a large center gap.
                for sign in (-1,1):
                    votes = []
                    for a,b in zip(ordered,ordered[1:]):
                        start,end = np.array(a["center"]),np.array(b["center"])
                        axis = (end-start)/np.linalg.norm(end-start)
                        normal = np.array([-axis[1],axis[0]])
                        def support(c):
                            e=c["ellipse"];angle=np.deg2rad(e["angle_deg"])
                            return np.hypot(e["diameter_x"]*.5*np.dot(normal,[np.cos(angle),np.sin(angle)]),
                                            e["diameter_y"]*.5*np.dot(normal,[-np.sin(angle),np.cos(angle)]))
                        for fraction in np.linspace(0,1,32):
                            radius = support(a)*(1-fraction)+support(b)*fraction
                            points = np.rint(start+(end-start)*fraction+sign*normal*radius*np.array([1.04,1.10,1.16])[:,None]).astype(int)
                            if np.any(points<0) or np.any(points[:,0]>=image.shape[1]) or np.any(points[:,1]>=image.shape[0]):
                                votes.append(False);continue
                            rgb = image[points[:,1],points[:,0]].astype(float)
                            sat=(rgb.max(axis=1)-rgb.min(axis=1))/np.maximum(1,rgb.max(axis=1))
                            votes.append(bool(np.sum((sat<.28)&(rgb.mean(axis=1)>80))>=2))
                    paper_sides.append(float(np.mean(votes)))
            groups.append({"format":"TRIPLE","layout":layout,"candidate_ids":[c["id"] for c in ordered],
                "confidence":float(np.mean([c["confidence"] for c in triple])),
                "minimum_zone_proposal":max(set(zone_votes),key=zone_votes.count),
                "ambiguity_reasons":["ownership cannot be inferred from rings or arrows"],
                "paper_gap_evidence":gap_evidence,"paper_side_evidence":paper_sides,
                "paper_continuity_supported":bool(paper_sides) and min(paper_sides)>=.65,
                "group_evidence":"size, spacing, collinearity/layout; upright image-axis prior"})
        for index,group in enumerate(groups):
            group["id"] = f"g{index}"
            group["face_indices"] = [{"candidate_id":cid,"face_index":i} for i,cid in enumerate(group["candidate_ids"])]
            for candidate in candidates:
                if candidate["id"] in group["candidate_ids"]:
                    candidate["groups"].append(group["id"])
        return groups


def fingerprint(candidates):
    return hashlib.sha256(json.dumps([(c["id"],[round(v,2) for v in c["center"]],round(c["diameter"],2)) for c in candidates]).encode()).hexdigest()


class TargetGroupSelector:
    def select(self, candidates, groups, previous=None):
        stamp = fingerprint(candidates)
        if previous and previous.get("status") in {"USER_CONFIRMED","AMBIGUOUS","NO_GROUP"} and previous.get("source") == "USER":
            if previous.get("detection_fingerprint") == stamp:
                validate_selection(candidates, previous)
                return previous
            return {"status":"AMBIGUOUS","source":"STALE_SELECTION","candidate_ids":[],
                    "detection_fingerprint":stamp,"previous_selection":previous,"reason":"detections changed; reconfirm selection"}
        triples = [g for g in groups if g["format"] == "TRIPLE"]
        choices = triples or (groups if len(candidates)==1 else [])
        if (len(choices) == 1 and choices[0]["confidence"] >= .65
                and (choices[0]["format"] == "SINGLE" or choices[0].get("paper_continuity_supported"))):
            group = choices[0]
            return {"status":"AUTO_CONFIDENT","source":"UNIQUE_GEOMETRIC_GROUP","group_id":group["id"],
                    "candidate_ids":group["candidate_ids"],"layout":group["layout"],"format":group["format"],
                    "minimum_scoring_zone":group["minimum_zone_proposal"],"physical_face_order_confirmed":False,
                    "detection_fingerprint":stamp,"ownership_confirmed":False}
        return {"status":"AMBIGUOUS" if candidates else "NO_GROUP","source":"AUTOMATIC","candidate_ids":[],
                "detection_fingerprint":stamp,"ownership_confirmed":False,
                "reason":"several equivalent sheets or incomplete grouping; user selection required" if candidates else "no plausible face found"}


def validate_selection(candidates, selection):
    """Apply the same checks to offline JSON and interactive confirmations."""
    ids = selection.get("candidate_ids", [])
    valid = {c["id"] for c in candidates}
    if (not isinstance(ids, list) or any(not isinstance(cid, str) or cid not in valid for cid in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("invalid candidate identities")
    if selection.get("status") == "USER_CONFIRMED":
        if len(ids) not in {1, 3}:
            raise ValueError("select one or three faces")
        expected_format = "SINGLE" if len(ids) == 1 else "TRIPLE"
        if selection.get("format") != expected_format:
            raise ValueError("selected format differs from face count")
        if selection.get("layout") not in ({None} if len(ids) == 1 else {"VERTICAL", "TRIANGULAR"}):
            raise ValueError("invalid selected layout")
    elif ids:
        raise ValueError("unconfirmed selection must not retain candidate identities")
    if selection.get("minimum_scoring_zone") not in {None, 1, 5, 6}:
        raise ValueError("unsupported scoring-zone proposal")


def track_confirmed_group(previous_row,current_row,previous_detection,current_detection,selection,previous_preview,current_preview):
    """Conservative scene correspondence, never a fabricated capture session."""
    if selection.get("status") != "USER_CONFIRMED" or not selection.get("remember"):
        return None
    if previous_row.get("device_model") in {None,"unknown"} or previous_row.get("device_model") != current_row.get("device_model"):
        return None
    try:
        gap = (datetime.strptime(current_row["capture_datetime"],"%Y:%m:%d %H:%M:%S")-
               datetime.strptime(previous_row["capture_datetime"],"%Y:%m:%d %H:%M:%S")).total_seconds()
    except (ValueError,KeyError,TypeError):
        return None
    if not 0<=gap<=1800:
        return None
    def features(image,row,det):
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        sx,sy=image.shape[1]/row["rgb_size"][0],image.shape[0]/row["rgb_size"][1]
        for c in det["candidates"]:
            x,y=c["center"];r=c["diameter"]*.75
            cv2.rectangle(mask,(round((x-r)*sx),round((y-r)*sy)),(round((x+r)*sx),round((y+r)*sy)),0,-1)
        return cv2.ORB_create(nfeatures=2000).detectAndCompute(gray,mask)
    ka,da=features(previous_preview,previous_row,previous_detection)
    kb,db=features(current_preview,current_row,current_detection)
    if da is None or db is None:
        return None
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.7*p[1].distance]
    if len(good)<20:
        return None
    a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(42)
    transform,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.)
    if transform is None or inliers is None or inliers.sum()<20 or inliers.mean()<.6:
        return None
    if not np.isfinite(transform).all() or np.linalg.det(transform[:2,:2])<=0:
        return None
    scale_a=np.array([previous_preview.shape[1]/previous_row["rgb_size"][0],previous_preview.shape[0]/previous_row["rgb_size"][1]])
    scale_b=np.array([current_preview.shape[1]/current_row["rgb_size"][0],current_preview.shape[0]/current_row["rgb_size"][1]])
    source=[next(c for c in previous_detection["candidates"] if c["id"]==cid)["center"] for cid in selection["candidate_ids"]]
    predicted=cv2.perspectiveTransform((np.asarray(source)*scale_a).astype(np.float32).reshape(-1,1,2),transform).reshape(-1,2)/scale_b
    ids=[]
    for point in predicted:
        choices=sorted((np.linalg.norm(point-c["center"])/c["diameter"],c["id"]) for c in current_detection["candidates"])
        if not choices or choices[0][0]>.2 or len(choices)>1 and choices[1][0]-choices[0][0]<.3:
            return None
        ids.append(choices[0][1])
    if len(set(ids))!=len(ids) or not any(set(g["candidate_ids"])==set(ids) for g in current_detection["groups"]):
        return None
    return {**selection,"status":"AUTO_CONFIDENT","source":"VISUAL_TRACKING_FROM_USER","candidate_ids":ids,
        "ownership_confirmed":False,"detection_fingerprint":current_detection["fingerprint"],
        "physical_face_order_confirmed":False,"group_id":"tracked",
        "tracking_evidence":{"reference_original_id":previous_row["original_id"],"background_inliers":int(inliers.sum()),
            "inlier_fraction":float(inliers.mean()),"exif_time_gap_seconds":gap,"capture_session_id":"unknown",
            "note":"visual continuity proposal, not user confirmation or pre-capture provenance"}}
