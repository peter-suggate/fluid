"""FVM1 affine material coordinates and GPU-ready optical texture layers.

Geometry never enters this companion: it stores only sampled mapping identities,
base normals, affine UV rows and tangent frames. Texture rows use Blender's UV
origin (bottom row first), with all levels of each layer contiguous.
"""
import math
import numpy as np
from texture_filter import _area_axis


def affine_uv(vertices, uvs):
    p=np.asarray(vertices,dtype=np.float64);uv=np.asarray(uvs,dtype=np.float64)
    e1=p[1]-p[0];e2=p[2]-p[0];a=float(e1@e1);b=float(e1@e2);c=float(e2@e2);det=a*c-b*b
    if det<=max(1e-30,a*c*1e-14):return None
    du=uv[1]-uv[0];dv=uv[2]-uv[0]
    gu=(e1*(c*du[0]-b*dv[0])+e2*(a*dv[0]-b*du[0]))/det
    gv=(e1*(c*du[1]-b*dv[1])+e2*(a*dv[1]-b*du[1]))/det
    uvdet=du[0]*dv[1]-du[1]*dv[0]
    if abs(uvdet)<1e-16:return None
    tangent=(e1*dv[1]-e2*du[1])/uvdet;tangent/=np.linalg.norm(tangent)
    bitangent=(e2*du[0]-e1*dv[0])/uvdet
    normal=np.cross(e1,e2);normal/=np.linalg.norm(normal)
    sign=1. if np.dot(np.cross(normal,tangent),bitangent)>=0 else -1.
    return np.r_[gu,uv[0,0]-gu@p[0]],np.r_[gv,uv[0,1]-gv@p[0]],tangent,sign,normal


def srgb_decode(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def srgb_encode(x):
    x=np.clip(x,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def resize(pixels,size):
    return _area_axis(_area_axis(np.asarray(pixels,dtype=np.float32),size,0),size,1)


def rgba_mips(rgb,alpha,size,normal=False):
    """Encode linear RGB (or mean unit normals) for one sRGB texture array.

    Hardware sRGB decoding occurs before interpolation. Normal means retain
    cancellation through mip construction; the shader normalizes after lookup.
    Alpha is always linear, and carries roughness on color layers.
    """
    rgb=np.asarray(rgb,dtype=np.float32)
    if normal:
        rgb=rgb*2-1;length=np.linalg.norm(rgb,axis=-1,keepdims=True)
        rgb=np.where(length>1e-10,rgb/np.maximum(length,1e-10),np.array([0.,0.,1.]))
    rgb=resize(rgb,size);alpha=resize(np.asarray(alpha,dtype=np.float32)[...,None],size)[...,0]
    levels=[]
    while True:
        encoded=srgb_encode(rgb*.5+.5 if normal else rgb)
        rgba=np.concatenate((encoded,alpha[...,None]),axis=-1)
        levels.append(np.clip(np.rint(rgba*255),0,255).astype(np.uint8).tobytes())
        if rgb.shape[0]==1:break
        n=max(1,rgb.shape[0]//2);rgb=resize(rgb,n);alpha=resize(alpha[...,None],n)[...,0]
    return b''.join(levels)
