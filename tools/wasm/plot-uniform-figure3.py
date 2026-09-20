"""Plot Figure 3 regression captures (matplotlib and numpy required)."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path('docs/research/uniform-geometric-figure3-2026-09-20')
fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
for row, (name, label) in enumerate([('prior', 'Before'), ('fixed', 'Source-aware fallback')]):
    with gzip.open(root / f'{name}.json.gz', 'rt') as f:
        run = json.load(f)
    for ax, frame in zip(axes[row], [20, 24, 30]):
        state = next(s for s in run['energySnapshots'] if s['frame'] == frame)
        volume = np.array(state['volume']).reshape(128, 128)
        phi = np.array(state['phi']).reshape(129, 129)
        ax.imshow(volume, origin='lower', extent=[0, 6.4, 0, 6.4], vmin=0, vmax=1, cmap='Blues')
        ax.contour(np.linspace(0, 6.4, 129), np.linspace(0, 6.4, 129), phi,
                   levels=[0], colors=['#d55e00'], linewidths=1)
        ax.set_title(f'{label}: {frame / 30:.2f} s')
        ax.set_aspect('equal')
fig.suptitle('Figure 3 — two extension sweeps in both runs\nConservative volume (blue), visible surface (orange)')
fig.savefig(root / 'comparison.png', dpi=140)
