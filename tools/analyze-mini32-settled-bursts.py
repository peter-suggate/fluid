"""Summarize the probe's raw accepted and stage fields using only stdlib."""
import array
import json
import math
from pathlib import Path

ROOT = Path('artifacts/mini32-settled-bursts')
STAGES = ['transport-velocity-extension', 'conservative-transport',
          'gamma-diffusion', 'surface-sharpening', 'density-capacity-repair',
          'scalar-publication']

def field(folder, step, name):
    result = array.array('f')
    result.frombytes((folder / f'{step}-{name}.bin').read_bytes())
    return result

def index(q):
    x, y, z = q
    return x + 32 * (y + 32 * z)

summary = {'arms': {}, 'event': {}}
for arm in ['base', 'audit', 'no-diffusion-late', 'closed-wall-guard']:
    folder = ROOT / arm
    if not (folder / 'trace.json').exists():
        continue
    trace = json.loads((folder / 'trace.json').read_text())
    samples = []
    for row in trace:
        step = row['step']
        if step < 240 or not (folder / f'{step}-density.bin').exists():
            continue
        rho, velocity, pressure = [field(folder, step, n)
                                   for n in ['density', 'velocity', 'pressure']]
        for x in [0, 31]:
            for z in [0, 31]:
                i = index((x, 0, z))
                samples.append({'rho': rho[i], 'pressure': pressure[i],
                                'speed_m_s': math.sqrt(sum(velocity[4*i+k]**2
                                                          for k in range(3)))})
    summary['arms'][arm] = {
        'sampleCount': len(samples),
        'cornerMinimumDensity': min(s['rho'] for s in samples),
        'cornerMaximumDensity': max(s['rho'] for s in samples),
        'cornerMaximumSpeed_m_s': max(s['speed_m_s'] for s in samples),
        'zeroPressureCornerSamples': sum(s['pressure'] == 0 for s in samples),
        'maximumAcceptedDensity': max(r['max'] for r in trace if r['step'] >= 240),
    }
    i = index((0, 0, 31))
    event = []
    for step in [296, 297, 298]:
        rho, velocity, pressure = [field(folder, step, n)
                                   for n in ['density', 'velocity', 'pressure']]
        event.append({'step': step, 'rho': rho[i], 'pressure': pressure[i],
                      'velocity_m_s': list(velocity[4*i:4*i+3])})
    summary['event'][arm] = event

folder = ROOT / 'audit'
summary['stages'] = []
for step, point in [(297, (0, 0, 31)), (298, (0, 0, 31)),
                    (299, (31, 0, 31)), (300, (31, 0, 0))]:
    for stage in STAGES:
        rho = field(folder, step, stage + '-density')
        gamma = field(folder, step, stage + '-gamma')
        summary['stages'].append({'step': step, 'stage': stage, 'point': point,
                                  'rho': rho[index(point)], 'gamma': gamma[index(point)]})
# The observational copies must not perturb the production trajectory.
a = json.loads((ROOT / 'base/trace.json').read_text())
b = json.loads((ROOT / 'audit/trace.json').read_text())
by_step = {r['step']: r for r in a}
for row in b:
    if row['step'] < 297:
        continue
    assert row == by_step[row['step']], f"stage capture changed step {row['step']}"
summary['stageCaptureMatchesBaseline'] = True
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['arms'], indent=2))
