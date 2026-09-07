"""Plot height derivatives; no cosmetic mesh smoothing or fitted height replacement."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('artifacts/coarse-surface-grid-imprint')
arms = ['fixed1', 'fixed2', 'fixed4', 'mixed', 'adaptive']
colors = ['#148a9c', '#785eb5', '#d46636', '#b24a8d', '#437b35']
fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
reports = {}
for arm, color in zip(arms, colors):
    folder = root / f'bowl-{arm}'
    if not (folder / 'summary.json').exists():
        continue
    reports[arm] = json.loads((folder / 'summary.json').read_text())
    rows = [r for r in json.loads((folder / 'measurements.json').read_text()) if r['z'] == 20]
    x = np.array([r['x'] + .5 for r in rows]) * .05
    for ax, key, scale in zip(axes, ['error_m', 'slope', 'curvature'], [1000, 1, 1]):
        ax.plot(x, [r[key] * scale for r in rows], '.-', color=color, label=arm, linewidth=1.5)
    if arm == 'fixed1':
        axes[1].plot(x, [r['exactSlope'] for r in rows], '--', color='black', label='analytic bowl')
        axes[2].plot(x, [r['exactCurvature'] for r in rows], '--', color='black', label='analytic bowl')
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(ncol=3, fontsize=9)
    ax.set_xlabel('x (m) — central cross-section')
axes[0].set_ylabel('height error (mm)')
axes[1].set_ylabel('surface slope dh/dx')
axes[2].set_ylabel('second height difference / finest h')
fig.suptitle('Static shallow bowl: coarse-grid imprint in published geometry\nZero gravity, frozen topology, same source volume quadrature')
fig.savefig(root / 'derivatives.png', dpi=180)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, arm in zip(axes, ['fixed1', 'fixed4']):
    folder = root / f'bowl-{arm}'
    actual = np.fromfile(folder / 'heights.bin', dtype=np.float32).reshape(40, 48)
    exact = np.fromfile(folder / 'exact.bin', dtype=np.float32).reshape(40, 48)
    error = (actual-exact)[8:-8, 8:-8]*1000
    error -= np.mean(error) # Remove a constant height bias, not spatial detail.
    im = ax.imshow(error, origin='lower', cmap='RdBu_r', vmin=-.3, vmax=.3, extent=[.4, 2, .4, 1.6])
    ax.set(title=f'{arm}: height error minus mean', xlabel='x (m)', ylabel='z (m)')
fig.colorbar(im, ax=axes, label='mm')
fig.savefig(root / 'grid-error.png', dpi=180)
(root / 'comparison.json').write_text(json.dumps(reports, indent=2)+'\n')

# Native density evidence. Compare to native *volume averages*, not point
# samples of a finer shape: blocky occupancy alone is not an error.
fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
density_report = {}
for col, (arm, w) in enumerate([('fixed1', 1), ('fixed4', 4)]):
    folder = root / f'bowl-{arm}'
    rho = np.fromfile(folder/'density.bin', np.float32).reshape(40, 32, 48)
    initial = np.fromfile(folder/'initialDensity.bin', np.float32).reshape(40, 32, 48)
    z0 = (20 // w)*w
    centres = np.arange(w/2, 48, w)
    zc = z0+w/2
    # Analytic area average of a quadratic, independent of voxel quadrature.
    mean = 17.3+.003*((centres-24)**2 + .7*(zc-20)**2 + 1.7*w*w/12)
    mass_height = rho[z0, :, ::w].sum(axis=0, dtype=np.float64)
    keep = (centres>=8)&(centres<40)
    delta = (mass_height-mean)*.05*1000
    density_report[arm] = dict(maxDensityEvolution=float(abs(rho-initial).max()),
        nativeColumnMeanMaxError_mm=float(abs(delta[keep]).max()),
        nativeColumnMeanRMSError_mm=float(np.sqrt(np.mean(delta[keep]**2))))
    density_errors=[]
    for zz in range(8,32,w):
        for xx in range(8,40,w):
            quadrature=(np.arange(64)+.5)*w/64
            hh=17.3+.003*((xx+quadrature[None,:]-24)**2+.7*(zz+quadrature[:,None]-20)**2)
            for yy in range(0,32,w):
                oracle=np.clip((hh-yy)/w,0,1).mean()
                density_errors.append(float(rho[zz,yy,xx]-oracle))
    density_report[arm]['nativeDensityMeanMaxAbsError']=float(np.max(np.abs(density_errors)))
    ax=axes[0,col]
    ax.imshow(rho[20, 14:24, 8:40], origin='lower', extent=[.4,2,.7,1.2], vmin=0,vmax=1,cmap='Blues',aspect='auto')
    x=np.arange(8.5,40)*.05
    ax.plot(x, (17.3+.003*((x/.05-24)**2+.7*.5**2))*.05, color='#df6732',label='analytic surface at slice centre')
    ax.set(title=f'{arm}: accepted density, z=1.025 m',ylabel='y (m)',xlabel='x (m)')
    ax.legend(fontsize=8)
    ax=axes[1,col]
    smooth_x=np.linspace(8,40,400)
    smooth_mean=17.3+.003*((smooth_x-24)**2+.7*(zc-20)**2+1.7*w*w/12)
    ax.plot(smooth_x*.05,smooth_mean*.05,'-',color='black',label='analytic native column volume / area')
    ax.plot(centres[keep]*.05,mass_height[keep]*.05,'o',ms=4,color='#158ba4',label='measured native density integral')
    ax.set(ylabel='equivalent water depth (m)',xlabel='native column centre x (m)')
    ax.legend(fontsize=8)
    ax=axes[2,col]
    ax.plot(centres[keep]*.05,delta[keep],'.-',color='#158ba4')
    ax.set(ylim=(-.002,.002),ylabel='native column volume error (mm)',xlabel='native column centre x (m)')
    ax.grid(alpha=.2)
fig.suptitle('Density check: coarse occupancy is blocky, but its native column volumes are correct\nOrange curve is point geometry; blue cells are volume averages, not surface samples')
fig.savefig(root/'density-check.png',dpi=180)
plt.close(fig)

# Direct production-kernel evaluation versus complete publication, with and
# without binary16 storage. This separates interpolation from page addressing.
folder=root/'bowl-fixed4'
if (folder/'isolated-kernel.bin').exists():
    values=np.fromfile(folder/'isolated-kernel.bin',np.float32).reshape(40,48,4)
    z,x=np.mgrid[:40,:48]
    exact_cells=17.3+.003*((x+.5-24)**2+.7*(z+.5-20)**2)
    low=np.floor(exact_cells-.5)+.5
    curves=[]
    for lo,hi in [(0,1),(2,3)]:
        curves.append((low-values[:,:,lo]/(values[:,:,hi]-values[:,:,lo]))*.05)
    actual=np.fromfile(folder/'heights.bin',np.float32).reshape(40,48)
    roi=np.s_[8:-8,8:-8]
    density_report['kernelIsolation']={
        'publishedVsIsolatedHalfMax_mm':float(abs(actual-curves[1])[roi].max()*1000),
        'binary16HeightEffectMax_mm':float(abs(curves[0]-curves[1])[roi].max()*1000),
        'float32CurvatureRMSError':float(np.sqrt(np.mean((np.diff(curves[0],2,axis=1)[8:-8,7:-7]/.05-.006)**2)))
    }
(root/'density-evidence.json').write_text(json.dumps(density_report,indent=2)+'\n')
print(json.dumps(density_report,indent=2))

fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,(arm,label) in enumerate([('fixed1','Minmax1'),('fixed4','Width 4')]):
    folder=root/f'bowl-{arm}'
    rho=np.fromfile(folder/'density.bin',np.float32).reshape(40,32,48)
    im0=axes[0,col].imshow(rho[20,14:24,8:40],origin='lower',extent=[.4,2,.7,1.2],vmin=0,vmax=1,cmap='Blues',aspect='auto')
    xs=np.linspace(8,40,300)
    axes[0,col].plot(xs*.05,(17.3+.003*((xs-24)**2+.7*.5**2))*.05,color='#df6732',lw=1.5)
    axes[0,col].set(title=f'{label}: accepted density',xlabel='x (m)',ylabel='y (m)')
    actual=np.fromfile(folder/'heights.bin',np.float32).reshape(40,48)
    exact=np.fromfile(folder/'exact.bin',np.float32).reshape(40,48)
    error=(actual-exact)[8:-8,8:-8]*1000
    error-=error.mean()
    im1=axes[1,col].imshow(error,origin='lower',extent=[.4,2,.4,1.6],cmap='RdBu_r',vmin=-.3,vmax=.3,aspect='auto')
    axes[1,col].set(title=f'{label}: published surface error minus mean',xlabel='x (m)',ylabel='z (m)')
fig.colorbar(im0,ax=axes[0,:],label='native cell fill')
fig.colorbar(im1,ax=axes[1,:],label='height error (mm)')
fig.suptitle('Same stationary bowl: density passes its native-volume check; publication exposes the grid')
fig.savefig(root/'density-vs-surface.png',dpi=180)
plt.close(fig)
