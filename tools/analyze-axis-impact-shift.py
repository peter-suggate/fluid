"""Locate submerged depletion after moving only the impact relative to B8 bricks."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('artifacts/axis-artifacts-2026-09-10')
arms=['shift-center','shift-x1'] + (['shift-x4'] if (root/'shift-x4'/'100-density.bin').exists() else [])
configs=[json.loads((root/a/'configuration.json').read_text()) for a in arms]
assert all(c['residentWGSLHash']==configs[0]['residentWGSLHash'] for c in configs)
assert all(c['values']==configs[0]['values'] for c in configs)
def field(arm,step,suffix='density'):
 return np.fromfile(root/arm/f'{step}-{suffix}.bin',np.float32).reshape(64,48,64)
initial=[field(a,0) for a in arms]
# Exact integer translation of the liquid sphere; the stationary pool is uniform.
initial_errors=[float(np.max(abs(rho-np.roll(initial[0],int(c['impactShiftXCells']),axis=2)))) for rho,c in zip(initial,configs)]
assert max(initial_errors)<1e-6,initial_errors
rows=[]
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for col,step in enumerate([30,60,90]):
 for i,arm in enumerate(arms):
  rho=field(arm,step)
  profile=rho[8:24,4:12,:].mean(axis=(0,1))
  # A pair's depression relative to the immediately flanking columns.
  scores=.5*(profile[:-3]+profile[3:]-profile[1:-2]-profile[2:-1])
  edges=np.arange(2,63)
  keep=(edges>=26)&(edges<=38)
  at=np.flatnonzero(keep)[np.argmax(scores[keep])]
  rows.append(dict(step=step,time_s=step/30,arm=arm,strongest_edge=int(edges[at]),
   pair_dip=float(scores[at]),profile=profile[25:40].tolist()))
  axes[0,col].plot(np.arange(25,40)+.5,profile[25:40],'.-',label=arm)
  axes[1,col].plot(edges[keep],scores[keep],'.-',label=arm)
 for ax in axes[:,col]:
  ax.axvline(32,color='gray',linestyle='--',label='B8 boundary' if ax is axes[0,0] else None)
  ax.axvline(33,color='orange',linestyle=':',label='+1-cell impact' if ax is axes[0,0] else None)
  if 'shift-x4' in arms: ax.axvline(36,color='green',linestyle=':',label='+4-cell impact' if ax is axes[0,0] else None)
  ax.grid(alpha=.2);ax.set_xlabel('x (finest-cell coordinates)')
 axes[0,col].set(title=f'{step/30:.1f} s: submerged density',ylabel='Mean density')
 axes[1,col].set(title='Two-column dip',ylabel='Flanks − seam')
axes[0,0].legend(fontsize=8)
fig.savefig(root/'impact-shift.png',dpi=150)
report=dict(initial_translation_max_errors=initial_errors,rows=rows)
(root/'impact-shift.json').write_text(json.dumps(report,indent=2))
print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='profile'} for r in rows]},indent=2))
# Local stage changes at the two candidate planes. All snapshots use native
# density/gamma parity, mapped through the accepted authored-leaf cell ranges.
stages=['transport-velocity-extension','conservative-transport','gamma-diffusion','surface-sharpening','density-capacity-repair','scalar-publication']
audit=[]
for arm in arms:
 for step in [20,40,60]:
  for stage in stages:
   path=root/arm/f'{step}-{stage}-density.bin'
   if not path.exists():continue
   rho=field(arm,step,stage+'-density');gamma=field(arm,step,stage+'-gamma')
   p=rho[8:24,4:12,:].mean((0,1));g=gamma[8:24,4:12,:].mean((0,1))
   audit.append(dict(arm=arm,step=step,stage=stage,density=p[29:36].tolist(),gamma=g[29:36].tolist()))
(root/'impact-shift-stages.json').write_text(json.dumps(audit,indent=2))
