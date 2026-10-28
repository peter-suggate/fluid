"""Linear-space texture minification for sampled physical surfaces.

The source cell projects through the triangle's UV Jacobian. Its largest
texel-space singular value selects a conservative isotropic mip footprint.
Normal mips retain mean unit vectors; normalization happens after sampling,
so cancellation is not discarded at intermediate levels.
"""
import bisect
import math
import numpy as np


def uv_cell_metric(vertices, uvs, cell_size):
    e1=np.asarray(vertices[1],dtype=np.float64)-vertices[0]
    e2=np.asarray(vertices[2],dtype=np.float64)-vertices[0]
    a=float(e1@e1);b=float(e1@e2);c=float(e2@e2);det=a*c-b*b
    if det<=max(1e-30,a*c*1e-14):return (0.,0.,0.)
    d1=np.asarray(uvs[1],dtype=np.float64)-uvs[0]
    d2=np.asarray(uvs[2],dtype=np.float64)-uvs[0]
    gu=cell_size*(e1*(c*d1[0]-b*d2[0])+e2*(a*d2[0]-b*d1[0]))/det
    gv=cell_size*(e1*(c*d1[1]-b*d2[1])+e2*(a*d2[1]-b*d1[1]))/det
    return (float(gu@gu),float(gu@gv),float(gv@gv))


def _area_axis(pixels, count, axis):
    """Exact box-area resampling, including non-power-of-two image sizes."""
    src=np.moveaxis(pixels,axis,0);old=src.shape[0]
    if old==count:return pixels
    edges=np.linspace(0,old,count+1);index=np.floor(edges).astype(int)
    cumulative=np.concatenate((np.zeros_like(src[:1]),np.cumsum(src,axis=0,dtype=np.float64)))
    fraction=(edges-index).reshape((-1,)+(1,)*(src.ndim-1))
    integral=cumulative[index]+fraction*src[np.minimum(index,old-1)]
    return np.moveaxis(np.diff(integral,axis=0)/(old/count),0,axis).astype(np.float32)


def _unit(v):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.where(length>1e-10,v/np.maximum(length,1e-10),np.array([0.,0.,1.]))


class TextureMipChain:
    def __init__(self,pixels,normal=False):
        p=np.asarray(pixels,dtype=np.float32)
        if p.ndim!=3 or p.shape[2]!=3 or not np.isfinite(p).all():raise ValueError('Expected finite RGB texture')
        self.normal=normal;self.height,self.width=p.shape[:2]
        if normal:p=_unit(p*2-1).astype(np.float32)
        self.levels=[p];self.scales=[0.]
        while p.shape[0]>1 or p.shape[1]>1:
            p=_area_axis(p,max(1,(p.shape[0]+1)//2),0)
            p=_area_axis(p,max(1,(p.shape[1]+1)//2),1)
            self.levels.append(p)
            self.scales.append(math.log2(max(self.width/p.shape[1],self.height/p.shape[0])))

    def lod(self,metric):
        a=metric[0]*self.width*self.width;b=metric[1]*self.width*self.height;c=metric[2]*self.height*self.height
        eigen=.5*(a+c+math.sqrt(max(0.,(a-c)*(a-c)+4*b*b)))
        return max(0.,.5*math.log2(max(1.,eigen)))

    @staticmethod
    def _bilinear(p,uv):
        h,w,_=p.shape;x=(float(uv[0])%1)*w-.5;y=(float(uv[1])%1)*h-.5
        ix=math.floor(x);iy=math.floor(y);fx=x-ix;fy=y-iy
        return (p[iy%h,ix%w]*(1-fx)*(1-fy)+p[iy%h,(ix+1)%w]*fx*(1-fy)
                +p[(iy+1)%h,ix%w]*(1-fx)*fy+p[(iy+1)%h,(ix+1)%w]*fx*fy)

    def sample(self,uv,lod):
        lo=max(0,min(len(self.levels)-1,bisect.bisect_right(self.scales,lod)-1));hi=min(lo+1,len(self.levels)-1)
        f=0. if hi==lo else min(1.,max(0.,(lod-self.scales[lo])/(self.scales[hi]-self.scales[lo])))
        value=self._bilinear(self.levels[lo],uv)
        if f>0:value=value*(1-f)+self._bilinear(self.levels[hi],uv)*f
        return _unit(value) if self.normal else value
