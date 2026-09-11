#!/usr/bin/env python3
"""Validate same-frame old/current/split replays and summarize timing vs work.

The counting run deliberately instruments helper calls. Its timings are never
reported; helper counts are WGSL calls, not hardware memory transactions.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

WORK_NAMES = [
    'acceptedRows', 'supportedRows', 'rkSubsteps', 'rkSamples', 'terminalSamples',
    'dualVisits', 'regularVisits', 'mixedVisits', 'newtonIterations',
    'transportOwnerCalls', 'compactOwnerCalls', 'cellCenterCalls', 'cellWidthCalls',
    'transportActiveCalls', 'nativeQueries', 'incidenceVisits', 'nativeFallbacks',
    'rowsWithMixed', 'rowsWithNativeFallback', 'fastMissRows', 'cornerVisits',
    'oldSamplerSamples', 'oldSupportQueries', 'oldSubsteps', 'oldSupportedRows',
    'geometryQueries', 'firstMomentChecks', 'cardinalHits', 'unused',
    'mixedFastRejects', 'outsideFastRejects', 'residualFastRejects',
]
KINDS = {
    'prepareSparseCM12AcceptedFaceRows': 'current',
    'prepareABOldFaceRows': 'old',
    'prepareABRegularLoopFaceRows': 'regularAttempt',
    'prepareABRegularFallbackFaceRows': 'fallback',
}

def read_valid(path: Path):
    data = json.loads(path.read_text())
    assert data['diagnostic']['passed'], (path, 'diagnostic failed')
    assert data['pressureCutoverReceiptGate']['passed'], (path, 'pressure receipt failed')
    assert not data['validationErrors'], (path, data['validationErrors'])
    rows = data['abFaceReplay']
    assert len(rows) == 12 and rows[-1]['name'] == 'prepareSparseCM12AcceptedFaceRows'
    for i, row in enumerate(rows):
        kind = KINDS[row['name']]
        if kind in ('current', 'fallback'):
            assert row['changedFaceWords'] == 0 and row['maximumDifference'] == 0
        if kind == 'regularAttempt':
            assert row['changedFaceWords'] == row['fallbackRows']
            assert KINDS[rows[i + 1]['name']] == 'fallback'
    return data, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    timing, timed = read_valid(args.directory / 'old-current-split-timings.json')
    counted, counts = read_valid(args.directory / 'old-current-split-work.json')
    result = {'baselineRevision': '28c95c47', 'oldFaceRevision': 'cd38e8b4',
              'frame': 24, 'timingAndCountingAreSeparateRuns': True,
              'counterMeaning': 'WGSL source calls and branches; not hardware memory transactions',
              'variants': {}}
    for kind in KINDS.values():
        ts = [r for r in timed if KINDS[r['name']] == kind]
        cs = [r for r in counts if KINDS[r['name']] == kind]
        assert len(ts) == len(cs) == 3
        assert all(r['work'] == cs[0]['work'] for r in cs), (kind, 'nonrepeatable work')
        assert all(r['changedFaceWords'] == ts[0]['changedFaceWords'] for r in ts)
        result['variants'][kind] = {
            'mean_ms': mean(r['duration_ms'] for r in ts),
            'samples_ms': [r['duration_ms'] for r in ts],
            'changedFaceWords': ts[0]['changedFaceWords'],
            'fallbackRows': ts[0]['fallbackRows'],
            'work': dict(zip(WORK_NAMES, cs[0]['work'], strict=True)),
        }
    v = result['variants']
    result['splitTotalMean_ms'] = v['regularAttempt']['mean_ms'] + v['fallback']['mean_ms']
    result['fallbackFractionOfSupportedRows'] = v['fallback']['work']['acceptedRows'] / v['current']['work']['supportedRows']
    result['currentOverOldTime'] = v['current']['mean_ms'] / v['old']['mean_ms']
    result['currentOverOldSubsteps'] = v['current']['work']['rkSubsteps'] / v['old']['work']['oldSubsteps']
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
