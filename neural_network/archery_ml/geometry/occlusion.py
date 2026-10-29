"""Local boundary support: reject line crossings and colour-inconsistent damage edges."""
import numpy as np


def boundary_support(points, center, inside_mask, outside_mask, image):
    height,width=inside_mask.shape
    direction=points-np.asarray(center)
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
    # Both sides must be observed in frame; a crop edge is never ring evidence.
    offset=max(2.,min(5.,min(height,width)*.003))
    inner=np.round(points-offset*direction).astype(int)
    outer=np.round(points+offset*direction).astype(int)
    valid=(inner[:,0]>=1)&(inner[:,0]<width-1)&(inner[:,1]>=1)&(inner[:,1]<height-1)
    valid&=(outer[:,0]>=1)&(outer[:,0]<width-1)&(outer[:,1]>=1)&(outer[:,1]<height-1)
    inner=np.clip(inner,[0,0],[width-1,height-1]);outer=np.clip(outer,[0,0],[width-1,height-1])
    # A hole/shaft has no matching adjacent colour zone on its outward side.
    valid&=(inside_mask[inner[:,1],inner[:,0]]!=0)&(inside_mask[outer[:,1],outer[:,0]]==0)
    if outside_mask is not None:
        valid&=outside_mask[outer[:,1],outer[:,0]]!=0
        far=np.round(points+3*offset*direction).astype(int)
        visible=(far[:,0]>=0)&(far[:,0]<width)&(far[:,1]>=0)&(far[:,1]<height)
        far=np.clip(far,[0,0],[width-1,height-1])
        valid&=visible&(inside_mask[far[:,1],far[:,0]]==0)
    inner_rgb=image[inner[:,1],inner[:,0]].astype(np.float32)
    outer_rgb=image[outer[:,1],outer[:,0]].astype(np.float32)
    gradient=np.linalg.norm(inner_rgb-outer_rgb,axis=1)
    valid&=gradient>=18
    weights=np.clip(gradient/100,.2,1.)
    return valid,weights
