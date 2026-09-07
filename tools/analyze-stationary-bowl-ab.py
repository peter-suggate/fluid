"""Compare native volume averages separately from published bowl curvature."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/stationary-bowl-2x')
base=root/'baseline'
roi=np.s_[8:-8,8:-8]
def field(folder,step,name):
    dtype=np.float64 if name=='columns' else np.float32
    shape=(40,32,48) if name=='density' else (40,48)
    return np.fromfile(folder/f'{step}-{name}.bin',dtype=dtype).reshape(shape).astype(float)
def restrict(a,w):
    return a.reshape(40//w,w,32//w,w,48//w,w).mean((1,3,5))
def rms(a):return float(np.sqrt(np.mean(a*a)))
summary={}
for folder in sorted(root.glob('*/*')):
    if not (folder/'trace.json').exists():continue
    trace=json.loads((folder/'trace.json').read_text())
    rows=[]
    for row in trace:
        step=row['step'];a=field(base/'fixed1',0,'density');b=field(folder,step,'density')
        delta=restrict(b-a,8)
        row['commonWidth8DensityRMSError']=rms(delta)
        row['commonWidth8DensityMaxError']=float(abs(delta).max())
        row['relativeMassChange']=row['mass']/trace[0]['mass']-1
        rows.append(row)
    summary[str(folder.relative_to(root))]=rows
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
x=(np.arange(48)+.5)*.05-1.2
for arm,label in [('fixed1','A: minmax1'),('adaptive','B: adaptive'),('frozen','B: frozen topology'),('off','B: conditioning off'),('gamma-only','B: gamma only'),('sharpen-only','B: sharpening only')]:
    rows=summary[f'baseline/{arm}']; t=[r['time'] for r in rows]
    axes[0,0].plot(t,[r['heightRMSError_mm'] for r in rows],'.-',label=label)
    axes[0,1].plot(t,[r['commonWidth8DensityRMSError'] for r in rows],'.-',label=label)
    axes[1,0].plot(t,[r['generation'] for r in rows],'.-',label=label)
for arm,label in [('fixed1','A at 2 s'),('adaptive','B at reset')]:
    step=120 if arm=='fixed1' else 0
    axes[1,1].plot(x,field(base/arm,step,'heights')[20],'.-',label=label)
axes[1,1].plot(x,field(base/'adaptive',120,'heights')[20],'.-',label='B at 2 s')
axes[1,1].plot(x,.865+.12*x*x+.084*.025**2,'k--',label='Analytic')
axes[0,0].set(title='Published surface error (interior)',xlabel='Time (s)',ylabel='RMS height error (mm)')
axes[0,1].set(title='Physics: same width-8 control volumes',xlabel='Time (s)',ylabel='RMS density difference from A reset')
axes[1,0].set(title='Topology changes in stationary liquid',xlabel='Time (s)',ylabel='Accepted generation')
axes[1,1].set(title='Center surface section',xlabel='x (m)',ylabel='Height (m)')
for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Stationary bowl · 2× curvature: zero forces and zero initial velocity')
fig.savefig(root/'ab-diagnosis.png',dpi=160)
print(json.dumps({k:v[-1] for k,v in summary.items()},indent=2))
final=root/'regression/adaptive'
if (final/'trace.json').exists():
    rows=json.loads((final/'trace.json').read_text());last=rows[-1]['step']
    cases=[(base/'fixed1',120,'A · minmax1'),(base/'adaptive',120,'B · before (2 s)'),(final,last,f'B · corrected ({last/60:g} s)')]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    X,Z=np.meshgrid(x,(np.arange(40)+.5)*.05-1)
    exact=.865+.12*X**2+.084*Z**2
    for col,(folder,step,label) in enumerate(cases):
        surface=field(folder,step,'heights');err=(surface-exact)*1000
        im=axes[0,col].imshow(err,origin='lower',extent=[-1.2,1.2,-1,1],vmin=-20,vmax=20,cmap='RdBu_r')
        axes[0,col].set(title=label,xlabel='x (m)',ylabel='z (m)')
        fig.colorbar(im,ax=axes[0,col],label='Height error (mm)')
        axes[1,0].plot(x,surface[20],'.-',label=label)
        axes[1,1].plot(x[1:-1],np.diff(surface[20],n=2)/.05**2,'.-',label=label)
        r=summary[str(folder.relative_to(root))]
        axes[1,2].plot([a['time'] for a in r],[a['commonWidth8DensityRMSError'] for a in r],'.-',label=label)
    axes[1,0].plot(x,exact[20],'k--',label='Analytic')
    axes[1,0].set(title='Center surface section',xlabel='x (m)',ylabel='Height (m)')
    axes[1,1].axhline(.24,color='k',linestyle='--',label='Analytic curvature')
    axes[1,1].set(title='Curvature across the center section',xlabel='x (m)',ylabel='Second derivative (1/m)',ylim=(-.5,1))
    axes[1,2].set(title='Native density on identical width-8 volumes',xlabel='Time (s)',ylabel='RMS difference from A reset')
    for ax in axes[1]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Stationary bowl · 2× curvature: before and after corrections')
    fig.savefig(root/'ab-before-after.png',dpi=160)
