"""Analyze the saved Dawn density and publication before/after region edit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('artifacts/bowl-region-edit')
arms = ['control', 'additive', 'split', 'flat-split']
def read(arm, label, name):
    dtype = np.float64 if name == 'columns' else np.float32
    shape = (40, 32, 48) if name in ['density', 'phi'] else (40, 48)
    return np.fromfile(root / arm / f'{label}-{name}.bin', dtype=dtype).reshape(shape).astype(np.float64)

summary = {}
for arm in arms:
    before = {n: read(arm, 'before', n) for n in ['density', 'phi', 'columns', 'heights']}
    rows = {}
    for label in ['after-uniforms', 'step-2', 'step-3', 'step-4', 'step-5']:
        delta = {n: read(arm, label, n) - a for n, a in before.items()}
        assert all(np.isfinite(a).all() for a in delta.values())
        parent = delta['density'].reshape(10, 4, 8, 4, 12, 4).mean((1, 3, 5))
        roi = delta['heights'][8:-8, 8:-8] * 1000
        rows[label] = {
            'density_max_abs_change': float(abs(delta['density']).max()),
            'parent_mean_max_abs_change': float(abs(parent).max()),
            'column_height_max_abs_change_mm': float(abs(delta['columns']).max()*1000),
            'published_height_max_abs_change_mm': float(abs(delta['heights']).max()*1000),
            'published_height_interior_max_abs_change_mm': float(abs(roi).max()),
            'published_height_interior_rms_change_mm': float(np.sqrt(np.mean(roi**2))),
            'published_phi_max_abs_change_cells': float(abs(delta['phi']).max()),
        }
    summary[arm] = rows
for arm in ['additive', 'split']:
    for name in ['density', 'phi', 'heights']:
        assert np.array_equal(read(arm, 'before', name), read('control', 'before', name))
(root / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

x = (np.arange(48)+.5)*.05-1.2
b = read('split', 'before', 'density')[20]
a = read('split', 'step-2', 'density')[20]
fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for panel, data, title in [(ax[0,0], b, 'Bowl density before: width 4'),
                           (ax[0,1], a, 'Bowl density after: width 4 | width 2')]:
    im = panel.imshow(data, origin='lower', extent=[-1.2,1.2,0,1.6], vmin=0, vmax=1, aspect='auto', cmap='Blues')
    panel.set(ylim=(.6,1.15), xlabel='x (m)', ylabel='y (m)', title=title)
    panel.axvline(0, color='orange', ls='--')
fig.colorbar(im, ax=ax[0,:], label='Native density sampled onto fine cells')
for arm, title in [('split','Bowl'), ('flat-split','Flat control')]:
    dh = (read(arm, 'step-2', 'heights')-read(arm,'before','heights'))[20]*1000
    ax[1,0].plot(x, dh, '.-', label=title)
ax[1,0].axvline(0,color='orange',ls='--')
ax[1,0].set(title='Published surface moves despite unchanged density', xlabel='x (m)', ylabel='After − before height (mm)')
ax[1,0].legend()
delta=(read('split','step-2','heights')-read('split','before','heights'))*1000
im=ax[1,1].imshow(delta,origin='lower',extent=[-1.2,1.2,-1,1],cmap='RdBu_r',vmin=-6.1,vmax=6.1,aspect='auto')
ax[1,1].axvline(0,color='orange',ls='--')
ax[1,1].set(title='Bowl: published height change across the surface',xlabel='x (m)',ylabel='z (m)')
fig.colorbar(im,ax=ax[1,1],label='mm')
fig.suptitle('Dawn live refinement: zero forces, zero velocity, density change = 0', fontsize=14)
fig.savefig(root/'before-after.png',dpi=170)
print(json.dumps({a: s['step-2'] for a,s in summary.items()}, indent=2))
