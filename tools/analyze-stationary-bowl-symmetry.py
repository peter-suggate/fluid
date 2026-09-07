"""Compare reflected native fields in gravity-on bowl runs; no analytic dynamics assumed."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs', nargs='+', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
summary = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for run in args.runs:
    config = json.loads((run / 'configuration.json').read_text())
    nx, ny, nz = (config[k] for k in ['nx', 'ny', 'nz'])
    trace = json.loads((run / 'trace.json').read_text())
    rows = []
    for record in trace:
        step = record['step']
        rho = np.fromfile(run / f'{step}-density.bin', dtype='<f4').reshape(nz, ny, nx)
        velocity = np.fromfile(run / f'{step}-velocity.bin', dtype='<f4').reshape(nz, ny, nx, 4)[..., :3]
        height = np.fromfile(run / f'{step}-heights.bin', dtype='<f4').reshape(nz, nx)
        metrics = {}
        for label, axis, component, height_axis in [('x', 2, 0, 1), ('z', 0, 2, 0)]:
            reflected_rho = np.flip(rho, axis)
            reflected_velocity = np.flip(velocity, axis).copy()
            reflected_velocity[..., component] *= -1
            dr = abs(rho-reflected_rho)
            dv = abs(velocity-reflected_velocity).max(axis=-1)
            wet = np.minimum(rho, reflected_rho) >= .5
            metrics[label] = dict(density_max=float(dr.max()), density_rms=float(np.sqrt(np.mean(dr**2))),
                velocity_max_m_s=float(dv.max()), wet_velocity_max_m_s=float(dv[wet].max()),
                height_max_mm=float(1000*abs(height-np.flip(height,height_axis)).max()))
        rows.append(dict(step=step, time=record['time'], symmetry=metrics))
    t = [r['time'] for r in rows]
    for ax, metric in zip(axes.flat, ['density_max', 'density_rms', 'wet_velocity_max_m_s', 'height_max_mm']):
        ax.semilogy(t, [max(1e-9, max(r['symmetry'][a][metric] for a in ['x','z'])) for r in rows], label=run.name)
        ax.set(xlabel='Time (s)', ylabel=metric.replace('_', ' '))
        ax.grid(alpha=.2)
    summary.append(dict(run=str(run.resolve()), configuration=config, measurements=rows))
axes[0, 0].legend(fontsize=8)
fig.suptitle('Minmax1 reflection symmetry • gravity and 1/30 s • gamma diffusion + sharpening on')
fig.savefig(args.output / 'symmetry.png', dpi=160)
(args.output / 'measurements.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps([{ 'run': s['run'], 'last': s['measurements'][-1]} for s in summary], indent=2))
