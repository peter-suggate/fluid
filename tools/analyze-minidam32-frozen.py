"""Compare simulation columns and the actual published surface in mini32 captures."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/minidam32-frozen')
out = root/'analysis'; out.mkdir(parents=True,exist_ok=True)
h = .025

def read(arm,step,kind):
    a=np.fromfile(root/arm/f'{step}-{kind}.bin',dtype='<f4')
    return a.reshape(32,32,32,4) if kind=='velocity' else a.reshape(32,32,32)

def surface(arm,step,axis=1):
    phi=read(arm,step,'published-phi')
    result=np.full((32,32),np.nan)
    for q in range(31):
        low=np.take(phi,q,axis=axis); high=np.take(phi,q+1,axis=axis)
        mask=np.isfinite(low)&np.isfinite(high)&(low<0)&(high>=0)
        result[mask]=q+.5-low[mask]/(high[mask]-low[mask])
    return h*result

before='adaptive'; after='adaptive-after'; fine='minmax1-after'
report={'scene':'minimal-power-dam-break-32','cell_size_m':h,'step':4,'time_s':4/30,
        'arms':{'A':'initial adaptive topology frozen','B':'min1/max1, normal sparse residency'},
        'simulation_fields_unchanged_by_surface_fix':True}
for arm,old in [(after,before),(fine,'minmax1-resident')]:
    for step in range(5):
        for kind in ['density','velocity','pressure']:
            assert np.array_equal(read(arm,step,kind),read(old,step,kind)),(arm,step,kind)
    top=surface(arm,0)[:20,:20]
    report[arm]={'reset_height_max_error_m':float(np.max(np.abs(top-29.44*h)))}
report['before_reset_height_max_error_m']=float(np.nanmax(np.abs(surface(before,0)[:20,:20]-29.44*h)))
report['before_reset_height_range_m']=float(np.ptp(surface(before,0)[:20,:20]))
report['frozen_failure']={'step':5,'time_s':5/30,'code':'EMPTY_DEFICIT_STENCIL',
    'donor_center_fine':[23.5,.5,23.5],'donor_density':float(read(before,4,'density')[23,0,23]),
    'donor_velocity_m_s':read(before,4,'velocity')[23,0,23,:3].tolist(),
    'inactive_corner_bricks':[[3,y,3] for y in range(4)]}

fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
x=(np.arange(20)+.5)*h
for arm,label,color,ls in [(before,'Adaptive: before','#aa5836','-'),('minmax1','min1/max1: before','#aa5836','--'),
                         (after,'Adaptive: corrected','#187f83','-'),(fine,'min1/max1: corrected','#187f83','--')]:
    ax[0].plot(x,1000*surface(arm,0)[0,:20],label=label,color=color,ls=ls)
for arm,label,color in [(after,'Frozen adaptive','#aa5836'),(fine,'min1/max1','#187f83')]:
    ax[1].plot(x,1000*surface(arm,4)[0,:20],label=label,color=color)
for a in ax:
    for boundary in [.2,.4]: a.axvline(boundary,color='.7',lw=.7,ls=':')
    a.set(xlabel='x from tank wall (m)',ylabel='Published waterline (mm)')
    a.grid(alpha=.2);a.legend(fontsize=8)
ax[0].set_title('At reset: a flat body acquired a renderer step')
ax[1].set_title('At 0.133 s: fluid-dependent bands remain')
fig.savefig(out/'waterline.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
tops=[surface(a,4) for a in [after,fine]]
columns=[h*read(a,4,'density').sum(axis=1,dtype=np.float64) for a in [after,fine]]
fronts=[surface(a,4,axis=2) for a in [after,fine]] # positive x: rows z, columns y
for row,arm in enumerate([after,fine]):
    for col,(data,title) in enumerate([(tops[row],'Published top surface'),(columns[row],'Column mass / reference density'),(fronts[row],'Published +x front')]):
        extent=[0,.8,0,.8]
        im=axes[row,col].imshow(data,origin='lower',extent=extent,vmin=0,vmax=.8,cmap='viridis',interpolation='nearest')
        axes[row,col].set_title(('A: frozen adaptive — ' if row==0 else 'B: min1/max1 — ')+title,fontsize=10)
        axes[row,col].set(xlabel='y (m)' if col==2 else 'x (m)',ylabel='z (m)')
        for boundary in [.2,.4,.6]:
            axes[row,col].axvline(boundary,color='white',lw=.5,alpha=.5)
            axes[row,col].axhline(boundary,color='white',lw=.5,alpha=.5)
fig.colorbar(im,ax=axes,label='Position or equivalent column depth (m)',shrink=.8)
fig.suptitle('Minidam32 at 0.133 s: published geometry and authoritative mass')
fig.savefig(out/'fields.png',dpi=160);plt.close(fig)

# Compare volume integrals without interpreting raw coarse-cell plateaux as
# renderer errors. Published geometry and finite-volume data are distinct.
for step in range(5):
    a=h*read(after,step,'density').sum(axis=1,dtype=np.float64)
    b=h*read(fine,step,'density').sum(axis=1,dtype=np.float64)
    report.setdefault('checkpoints',[]).append({'step':step,'time_s':step/30,
        'column_rms_gap_m':float(np.sqrt(np.mean((a-b)**2))),
        'maximum_column_gap_m':float(np.max(np.abs(a-b))),
        'mass_error_percent_A':float(100*(a.sum()/ (11776*h)-1)),
        'mass_error_percent_B':float(100*(b.sum()/ (11776*h)-1))})
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
