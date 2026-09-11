"""Compare publication-only ablation against identical accepted density."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/axis-artifacts-2026-09-10')
def crossing(p):
    h=np.full((64,64),np.nan)
    for y in range(47):
        a,b=p[:,y,:],p[:,y+1,:]
        mask=(a<0)&(b>=0)
        h[mask]=(y+.5-a[mask]/(b[mask]-a[mask]))*.05
    return h
def metrics(h):
    lap=h[1:-1,:-2]+h[1:-1,2:]+h[:-2,1:-1]+h[2:,1:-1]-4*h[1:-1,1:-1]
    return dict(laplacian_rms_mm=float(np.sqrt(np.nanmean(lap**2))*1000),max_neighbor_jump_mm=float(max(np.nanmax(abs(np.diff(h,axis=0))),np.nanmax(abs(np.diff(h,axis=1))))*1000))
fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
results=[]
for row,step in enumerate([60,90,100]):
    rho=np.fromfile(root/'base'/f'{step}-density.bin',np.float32).reshape(64,48,64)
    raw=crossing(.5-rho)
    base=crossing(np.fromfile(root/'base'/f'presentation-{step}'/'phi.bin',np.float32).reshape(64,48,64))
    no=crossing(np.fromfile(root/'no-height'/f'presentation-{step}'/'phi.bin',np.float32).reshape(64,48,64))
    identical=(root/'base'/f'{step}-density.bin').read_bytes()==(root/'no-height'/f'{step}-density.bin').read_bytes()
    assert identical, "publication ablation changed accepted density"
    delta=(base-no)*1000
    result=dict(time_s=step/30,density_bit_identical=identical,base=metrics(base),no_height=metrics(no),publication_delta_max_mm=float(np.nanmax(abs(delta))),no_height_vs_raw_max_mm=float(np.nanmax(abs(no-raw))*1000),affected_columns=int(np.sum(abs(delta)>.1)))
    results.append(result)
    for ax,h,title in zip(axes[row],[base,no,delta],['Current publication','Height branch disabled','Difference (mm)']):
        kw=dict(cmap='coolwarm',vmin=-80,vmax=80) if title.startswith('Difference') else dict(cmap='viridis',vmin=min(base.min(),no.min()),vmax=max(base.max(),no.max()))
        im=ax.imshow(h,origin='lower',extent=[-1.6,1.6,-1.6,1.6],**kw)
        ax.set(title=f'{step/30:.2f} s — {title}',xlabel='x (m)',ylabel='z (m)')
        fig.colorbar(im,ax=ax,shrink=.7)
fig.savefig(root/'comparison.png',dpi=150)
(root/'comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
