"""Compare evolved native coarse density with restriction of evolved minmax1.
This measures physical-state divergence, not a renderer/isosurface difference.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('artifacts/coarse-surface-grid-imprint')
w=4; nx,ny,nz=48,32,40; h=.05
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
report={}
for column,case in enumerate(['gravity','conditioning']):
    dirs=[root/f'{case}-fixed1',root/f'{case}-fixed4']
    if not all((d/'summary.json').exists() for d in dirs): continue
    steps=json.loads((dirs[0]/'summary.json').read_text())['steps']
    trace=[]
    for step in range(steps+1):
        filename='initialDensity.bin' if step==0 else f'step-{step}-density.bin'
        fine,coarse=[np.fromfile(d/filename,np.float32).reshape(nz,ny,nx).astype(np.float64) for d in dirs]
        restricted=fine.reshape(nz//w,w,ny//w,w,nx//w,w).mean(axis=(1,3,5))
        native=coarse[::w,::w,::w]
        delta=(native-restricted)
        # Equal-area native column masses, independent of publication.
        fine_height=restricted.sum(axis=1)*w*h
        coarse_height=native.sum(axis=1)*w*h
        height_delta=(coarse_height-fine_height)*1000
        roi=np.s_[2:-2,2:-2]
        trace.append(dict(step=step,time_s=step/60,
            densityRMSError=float(np.sqrt(np.mean(delta[2:-2,:,2:-2]**2))),
            nativeColumnDifferenceRMS_mm=float(np.sqrt(np.mean(height_delta[roi]**2))),
            nativeColumnDifferenceMax_mm=float(np.abs(height_delta[roi]).max())))
    report[case]=trace
    z=5
    xc=(np.arange(nx//w)+.5)*w*h
    axes[0,column].plot(xc[2:-2],fine_height[z,2:-2],'.-',label='evolved minmax1 restricted to width 4',color='#148a9c')
    axes[0,column].plot(xc[2:-2],coarse_height[z,2:-2],'.-',label='evolved width 4',color='#d46636')
    axes[0,column].set(title=f'{case}, {steps/60:.2f} s: native column volume',ylabel='equivalent depth (m)',xlabel='x (m)')
    axes[0,column].legend(fontsize=8)
    axes[1,column].plot([r['time_s'] for r in trace],[r['nativeColumnDifferenceRMS_mm'] for r in trace],'.-',color='#d46636')
    axes[1,column].set(xlabel='time (s)',ylabel='coarse vs restricted fine column RMS (mm)')
    axes[1,column].grid(alpha=.2)
fig.suptitle('Physical-state controls: gravity/transport and conditioning, with topology frozen\nA growing difference here is in density; it cannot be caused by meshing or lighting')
fig.savefig(root/'density-evolution.png',dpi=180)
(root/'density-evolution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
