"""Bounded local colour adaptation and physically ordered measured ring transitions."""
import cv2
import numpy as np

from neural_network.archery_ml.geometry.arc_extraction import balance_ring
from neural_network.archery_ml.geometry.occlusion import boundary_support
from neural_network.archery_ml.geometry.geometry_math import project


def local_color_masks(image, seed_masks):
    """Relax only near observed chromatic seeds, inside fixed hue/chroma limits."""
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    chroma=np.hypot(lab[:,:,1].astype(np.float32)-128,lab[:,:,2].astype(np.float32)-128)
    result={}
    windows={'yellow':((14,43),),'red':((0,14),(164,179)),'blue':((85,140),)}
    for color,windows_for_color in windows.items():
        samples=hsv[:,:,1][seed_masks[color]!=0]
        floor=max(25.,min(55.,float(np.percentile(samples,15))*.65)) if len(samples) else 55.
        hue=np.zeros(hsv.shape[:2],bool)
        for low,high in windows_for_color:
            hue|=(hsv[:,:,0]>=low)&(hsv[:,:,0]<=high)
        allowed=hue&(hsv[:,:,1]>=floor)&(hsv[:,:,2]>=25)&(chroma>=22)
        if color=='blue':
            allowed&=image[:,:,2].astype(np.int16)>image[:,:,0].astype(np.int16)+8
        result[color]=(allowed*255).astype(np.uint8)
    neutral=(hsv[:,:,1]<55)&(chroma<35)
    result['black']=(neutral&(hsv[:,:,2]<100)).astype(np.uint8)*255
    result['white']=(neutral&(hsv[:,:,2]>150)).astype(np.uint8)*255
    result['neutral_outer']=((hsv[:,:,1]<65)&(chroma<40)&(hsv[:,:,2]>=90)).astype(np.uint8)*255
    return result


def collect_ring_evidence(image, masks, template, initial, contour_cache=None):
    center=project([[0.,0.]],np.linalg.inv(initial))[0]
    rings=[];excluded=[]
    boundaries=template.ring_outer_radii
    if template.visible_radius_norm>=.8:
        boundaries=boundaries+(("black",.8),)
    cache={} if contour_cache is None else contour_cache
    for color,radius in boundaries:
        if color not in cache:
            contours,_=cv2.findContours(masks[color],cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            cache[color]=contours
        contours=cache[color]
        if not contours:
            continue
        points=np.concatenate(contours).reshape(-1,2).astype(np.float64)
        mapped=project(points,initial)
        distance=np.linalg.norm(mapped,axis=1)
        near=(distance>radius*.65)&(distance<radius*1.35)
        points=points[near]
        if len(points)<12:
            continue
        outside=(masks['red'] if color=='yellow' else masks['blue'] if color=='red'
                 else masks['white'] if color=='black' else
                 masks['black'] if template.visible_radius_norm>.6 else masks['neutral_outer'])
        good,weights=boundary_support(points,center,masks[color],outside,image)
        # Full-face black/white edges are permitted only at known template
        # radii with the confirmed colour transition; neutral edges alone
        # cannot identify a ring. This excludes arbitrary shadow boundaries.
        excluded.extend(points[~good][::max(1,int((~good).sum()/128))].tolist())
        if good.sum()<12:
            continue
        ring=balance_ring(points[good],weights[good],radius,color,initial)
        if len(np.unique(ring.sectors))>=8:
            rings.append(ring)
    return rings,tuple(excluded[:384])
