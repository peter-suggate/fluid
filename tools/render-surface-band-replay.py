"""Composite x-y sections (z = 0) of the surface-band replay.

Usage: python3.11 tools/render-surface-band-replay.py REPLAY_DIR [--steps 3,5,7,8,12,16]
Rows are steps; columns are the solver's native half-density contour and the
replay arms, each drawn over the solver density (grey) with the analytic
reference (yellow, valid before contact only) underneath.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('rb', Path(__file__).with_name('replay-surface-band.py'))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

parser = argparse.ArgumentParser(); parser.add_argument('replay', type=Path)
parser.add_argument('--steps', default='3,5,7,8,12,16'); parser.add_argument('--arms', default='A,C,D')
args = parser.parse_args()
capture = args.replay.parent
cfg = rb.load_configuration(capture)
lattice = rb.Lattice(cfg['dims'], cfg['origin'], cfg['h'])
ppc, scale = 6, 2
points = rb._section_points(lattice, ppc)
W, H = cfg['dims'][0] * ppc, cfg['dims'][1] * ppc
arms = args.arms.split(',')
colours = {'native': (255, 255, 255), 'A': (0, 220, 255), 'B': (255, 120, 0), 'C': (0, 255, 90), 'D': (255, 0, 200)}
rows = []
for step in [int(s) for s in args.steps.split(',')]:
    density = rb.load_step(capture, step, cfg['dims'])['density']
    grey = np.clip(lattice.trilinear(density, points).reshape(W, H), 0, 1)
    center = rb.reference_center(cfg, step)
    ref = np.minimum(np.linalg.norm(points - center, axis=1) - cfg['radius'], points[:, 1] - cfg['pool'])
    contact = center[1] - cfg['radius'] - cfg['pool'] <= 0
    panels = []
    for column in ['native'] + arms:
        img = np.repeat((40 + 150 * grey)[..., None], 3, -1)
        def contour(values, colour):
            v = values.reshape(W, H); edge = np.zeros((W, H), dtype=bool)
            edge[:-1, :] |= (v[:-1, :] > 0) != (v[1:, :] > 0); edge[:, :-1] |= (v[:, :-1] > 0) != (v[:, 1:] > 0)
            img[edge] = colour
        if not contact:
            contour(-ref, (255, 220, 0))
        if column == 'native':
            contour(lattice.trilinear(density, points) - 0.5, colours['native'])
        else:
            phi = np.load(args.replay / f'phi-{column}-step-{step}.npy').astype(float)
            contour(-lattice.trilinear(phi, points), colours[column])
        img = np.transpose(img, (1, 0, 2))[::-1]
        # crop to the band of interest: y from 0.2 m to 1.2 m
        y0 = int((cfg['dims'][1] * cfg['h'] - 1.2) / cfg['h'] * ppc); y1 = int((cfg['dims'][1] * cfg['h'] - 0.2) / cfg['h'] * ppc)
        img = img[y0:y1]
        img[:, -1] = 0; img[-1, :] = 0
        panels.append(img)
    rows.append(np.concatenate(panels, 1))
img = np.concatenate(rows, 0)
img = np.repeat(np.repeat(img, scale, 0), scale, 1)
out = args.replay / 'composite.png'
rb.write_png(out, img)
print(out, img.shape, 'columns: native(white) +', ' '.join(f'{a}({colours[a]})' for a in arms), 'rows: steps', args.steps)
