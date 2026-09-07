"""Compare each half to its matching uniform-resolution control, stage by stage.

The comparison uses identical physical native-cell centres and widths. It reports
raw differences, not a new pass threshold. Run the stage probe for mixed, coarse,
and fine with --seam-width=2 first. Missing stored cells are below the probe's
1e-7 density capture floor; differences at that scale are not evidence of a bug.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=6)
parser.add_argument('--mixed-label', default='')
args = parser.parse_args()
root = Path('artifacts/analytic-motion')

def key(c):
    return tuple(round(v, 10) for v in (*c['position'], c['width']))

receipts = []
for step in range(1, args.steps + 1):
    lanes = {}
    for lane in ('mixed', 'coarse', 'fine'):
        suffix = '-' + args.mixed_label if lane == 'mixed' and args.mixed_label else ''
        directory = root / f'stages-translation-{lane}-width2{suffix}'
        lanes[lane] = json.loads((directory / f'step-{step}.json').read_text())
    for mixed in lanes['mixed']:
        stage = mixed['stage']
        if any(not any(r['stage'] == stage for r in lanes[lane]) for lane in ('coarse', 'fine')):
            continue  # Older controls did not capture the gamma scratch stage.
        actual = {key(c): c for c in mixed['cells']}
        reference = {}
        for lane in ('coarse', 'fine'):
            control = next(r for r in lanes[lane] if r['stage'] == stage)
            reference.update({key(c): c for c in control['cells']
                              if (c['position'][0] < .8) == (lane == 'coarse')})
        samples = []
        for k in actual.keys() | reference.keys():
            a, b = actual.get(k), reference.get(k)
            arho, brho = (a['rho'] if a else 0), (b['rho'] if b else 0)
            # Velocity is compared only where both runs actually store material.
            dv = a['velocity'] - b['velocity'] if a and b else None
            samples.append({'position_m': list(k[:3]), 'width_m': k[3], 'densityDelta': arho - brho,
                            'velocityDelta_m_s': dv, 'mixedDensity': arho, 'controlDensity': brho})
        samples.sort(key=lambda s: abs(s['densityDelta']), reverse=True)
        max_v = max((abs(s['velocityDelta_m_s']) for s in samples if s['velocityDelta_m_s'] is not None), default=0)
        max_constant_v = max(abs(c['velocity'] + .4) for c in mixed['cells'])
        receipt = {'step': step, 'stage': stage,
                   'maxDensityDelta': abs(samples[0]['densityDelta']),
                   'maxVelocityDelta_m_s': max_v, 'maxConstantVelocityError_m_s': max_constant_v,
                   'densityDifferenceL1_m3': sum(abs(s['densityDelta']) * s['width_m']**3 for s in samples),
                   'largestDifferences': samples[:8]}
        receipts.append(receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'largestDifferences'}))
suffix = '-' + args.mixed_label if args.mixed_label else ''
(root / f'paired-seam-stage-comparison{suffix}.json').write_text(json.dumps(receipts, indent=2))
