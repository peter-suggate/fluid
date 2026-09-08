"""CPU-only cross-check of stored measures against the accepted density field.

Usage: python3.11 tools/audit-current-map-measure.py ARM_DIRECTORY --step 6
Uses independent tensor Gauss-2 over every fine support, then Gauss-8 and
Gauss-12 on a bounded set of the largest discrepancies. Their differences
are numerical evidence, not certified quadrature error bounds.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


_spec = importlib.util.spec_from_file_location('current_map_ray_field', Path(__file__).with_name('analyze-current-map-ray-field.py'))
_field = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_field)


def read_measure(directory, metadata):
    count = int(np.prod(metadata['measure']['dimensions']))
    bank = int(metadata['retainedControl'][1])
    first = metadata['measure']['baseWords']-metadata['baseWords']+4*count*bank
    raw = directory/'current-map.bin'
    source = raw.open('rb') if raw.exists() else gzip.open(directory/'current-map.bin.gz', 'rb')
    with source:
        source.seek(4*first)
        values = np.frombuffer(source.read(16*count), dtype='<f4')
    return values.reshape(tuple(metadata['measure']['dimensions'][::-1])+(4,))


def integrate_cell_means(density, origin, h, cells, order, point_budget=32768):
    """Independent fixed tensor rule; no production axis splitting or reuse."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    z, y, x = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    wz, wy, wx = np.meshgrid(weights, weights, weights, indexing='ij')
    offsets = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    weights = (wx*wy*wz).ravel()
    values = []
    chunk = max(1, point_budget//len(offsets))
    for low in range(0, len(cells), chunk):
        group = cells[low:low+chunk]
        points = origin+h*(group[:, None, :]+offsets[None, :, :])
        samples = density(points.reshape(-1, 3)).reshape(len(group), len(offsets))
        values.extend(samples@weights)
    return np.asarray(values)


def audit(directory, step, refinement_count=64):
    path = directory/f'step-{step}'
    config = _field._capture.read_json(directory/'configuration.json')
    metadata = _field._capture.read_json(path/'current-map.json')
    stored = read_measure(path, metadata)
    previous_step, previous = None, None
    for other in sorted(directory.glob('step-*'), key=lambda p: int(p.name.split('-')[1]), reverse=True):
        ordinal = int(other.name.split('-')[1])
        if ordinal >= step or not (other/'current-map.json').exists():
            continue
        other_meta = _field._capture.read_json(other/'current-map.json')
        if other_meta['retainedControl'][1] == metadata['retainedControl'][1]:
            previous_step, previous = ordinal, read_measure(other, other_meta)
            break
    snapshot = _field.MappedSnapshot(path)
    seed = _field.SeedField(config)
    density = _field.DensityAuthority(snapshot, seed)
    z, y, x = np.indices(stored.shape[:3])
    cells = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    coarse = integrate_cell_means(density, seed.origin, seed.h, cells, 2)
    rho = stored[..., 0].ravel().astype(float)
    delta = rho-coarse
    print(f'Step {step}: completed independent Gauss-2 on {len(cells)} supports', file=sys.stderr, flush=True)
    indices = np.argsort(abs(delta))[-refinement_count:][::-1]
    refined = integrate_cell_means(density, seed.origin, seed.h, cells[indices], 8)
    finer = integrate_cell_means(density, seed.origin, seed.h, cells[indices], 12)
    rows = []
    for k, i in enumerate(indices):
        row = dict(fine=cells[i].tolist(), storedMeasure=stored.reshape(-1, 4)[i].astype(float).tolist(),
                   gauss2Mean=float(coarse[i]), gauss8Mean=float(refined[k]), gauss12Mean=float(finer[k]),
                   storedMinusGauss12=float(rho[i]-finer[k]), gauss12MinusGauss8=float(finer[k]-refined[k]))
        if previous is not None:
            row['previousSameBankMeasure'] = previous.reshape(-1, 4)[i].astype(float).tolist()
            row['allFourWordsEqualPreviousSameBank'] = bool(np.array_equal(stored.reshape(-1, 4)[i], previous.reshape(-1, 4)[i]))
        rows.append(row)

    class StoredMeasure:
        def measure(self):
            return stored.astype(float)

    return dict(scope=__doc__.strip(), step=step, acceptedChainCount=None if snapshot.chain is None else len(snapshot.chain),
                previousSameBankStep=previous_step,
                native=_field._capture.native_consistency(path, StoredMeasure(), config),
                independentGauss2Amount_m3=float(coarse.sum()*config['h']**3),
                maximumStoredMeanVsGauss2=float(np.max(abs(delta))), L1StoredMeanVsGauss2=float(np.sum(abs(delta))),
                refinedSupportCount=len(indices), refinedSupports=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--refine-count', type=int, default=64)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.refine_count < 1:
        parser.error('--refine-count must be positive')
    report = audit(args.directory, args.step, args.refine_count)
    output = args.out or args.directory/f'current-map-measure-audit-step-{args.step}.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output)
