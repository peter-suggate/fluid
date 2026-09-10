"""Surface-band gate: the tracker's numbers from one capture, in one command.

Usage: python3.11 tools/surface-band-gate.py CAPTURE_ARM_DIRECTORY [--steps 0,1,...]
         [--band none|phi|PREFIX] [--limits tests/surface-band-limits.json]
         [--criteria C2.4,C3.1] [--out gate.json]

The capture directory comes from tools/capture-retained-visual-ab-dawn.ts
(per step: density.bin, velocity.bin, activity.json, mesh.bin, receipt.json,
and phi.bin once the resident publishes its band). Every metric is the
replay's (tools/replay-surface-band.py imports tools/analyze-current-map-ray-field.py):
half-level roots along 134 antipodal rays from the discrete free-fall centre,
pool heights on a 17x17 grid, the seed-ramp amount model for mass. The mesh
pool column keeps only vertices two cells inside the walls, because the
shipping mesh's wall faces carry vertices below the waterline. Nothing
is re-implemented here, so a number this script prints for a GPU capture is
comparable with the shadow replay's tables in
docs/HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08.md.

--band selects the band phi source: `none`; `phi` reads step-N/phi.bin
(float32, metres, the density.bin layout); any other value is a file prefix,
relative to the capture directory unless absolute, read as PREFIX-step-N.npy
(the replay writes replay-normal/phi-A-step-N.npy for arm A).

--limits evaluates the criteria in tests/surface-band-limits.json against the
per-step records and exits 1 when any evaluated rule fails; --criteria
restricts evaluation to those ids and makes a missing series a failure.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'replay_surface_band', Path(__file__).with_name('replay-surface-band.py'))
_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_replay)
_ray = _replay._ray


def _mm(stats, key):
    return None if stats is None else round(1e3 * float(stats[key]), 3)


def _surface(metrics):
    sphere, pool = metrics['sphere'], metrics['pool']
    return {
        'sphere.rms_mm': _mm(sphere.get('radialError_m'), 'rms'),
        'sphere.max_mm': _mm(sphere.get('radialError_m'), 'maxAbs'),
        'sphere.missing': sphere.get('missingRoots'),
        'pool.rms_mm': _mm(pool.get('heightError_m'), 'rms'),
        'pool.max_mm': _mm(pool.get('heightError_m'), 'maxAbs'),
    }


def load_band(directory, step, band, dims):
    """Band phi in metres on the fine lattice, or None when absent."""
    if band in (None, 'none'):
        return None
    if band == 'phi':
        path = directory / f'step-{step}' / 'phi.bin'
        if not path.exists():
            return None
        nx, ny, nz = dims
        return np.ascontiguousarray(np.fromfile(path, dtype='<f4').astype(float)
                                    .reshape(nz, ny, nx).transpose(2, 1, 0))
    prefix = Path(band)
    if not prefix.is_absolute():
        prefix = directory / prefix
    path = Path(f'{prefix}-step-{step}.npy')
    if not path.exists():
        return None
    phi = np.load(path).astype(float)
    if phi.shape != tuple(dims):
        raise ValueError(f'{path}: shape {phi.shape} is not the lattice {dims}')
    return phi


def evaluate(directory, steps=None, band='none', width=None, directions=134):
    """Per-step flat records for the tracker. Keys are dotted series names."""
    directory = Path(directory)
    cfg = _replay.load_configuration(directory)
    dims, h, dt = cfg['dims'], cfg['h'], cfg['dt']
    w = h if width is None else width
    lattice = _replay.Lattice(dims, cfg['origin'], h)
    rays = _ray.antipodal_directions(directions)
    if steps is None:
        steps = cfg['steps']
    records = []
    for step in steps:
        if step not in cfg['steps']:
            raise ValueError(f'step {step} is not in the capture ({cfg["steps"]})')
        current = _replay.load_step(directory, step, dims)
        density = current['density']
        center = _replay.reference_center(cfg, step)
        gap = float(center[1] - cfg['radius'] - cfg['pool'])
        record = {'step': step, 'time_s': step * dt, 'gap_mm': round(1e3 * gap, 1) if gap > 0 else None,
                  'solver.amount_L': round(1e3 * float(density.sum() * h ** 3), 4)}
        # Solver against the reference geometry (model-free), pre-contact only.
        nid, widths = _replay.native_layout(current['activity'], dims)
        rho_native, cells = _replay.native_means(density, nid, len(widths))
        if gap > 0:
            ref_phi = np.minimum(np.linalg.norm(lattice.centers - center, axis=1) - cfg['radius'],
                                 lattice.centers[:, 1] - cfg['pool']).reshape(dims)
            exact = _replay.amount_field(ref_phi, lattice, w)
            exact_native, _ = _replay.native_means(exact, nid, len(widths))
            interfacial = (exact_native > 1e-3) & (exact_native < 1 - 1e-3)
            volume = cells * h ** 3
            error = _ray.statistics((rho_native - exact_native)[interfacial]) if interfacial.any() else None
            record['solver.fractionError_max'] = None if error is None else round(error['maxAbs'], 4)
            record['solver.massOutsideReference_ml'] = round(1e6 * float(((rho_native * volume)[exact_native <= 1e-3]).sum()), 2)
        # Native density contour and the shipping mesh.
        contour = _surface(_replay.surface_metrics(cfg, lattice, lambda pts: lattice.trilinear(density, pts), step, rays))
        record.update({f'contour.{k}': v for k, v in contour.items()})
        mesh = _replay.mesh_metrics(cfg, current['mesh'], step)
        record['mesh.sphere.rms_mm'] = _mm(mesh.get('radialError_m'), 'rms')
        record['mesh.sphere.max_mm'] = _mm(mesh.get('radialError_m'), 'maxAbs')
        # The pool column uses interior vertices only: the mesh's wall faces
        # carry vertices at pool level minus two cells that are not surface.
        margin = 2 * h
        p = current['mesh'][:, :3]
        interior = ((p[:, 0] > lattice.lower[0] + margin) & (p[:, 0] < lattice.upper[0] - margin)
                    & (p[:, 2] > lattice.lower[2] + margin) & (p[:, 2] < lattice.upper[2] - margin))
        pool_mesh = _replay.mesh_metrics(cfg, current['mesh'][interior], step)
        record['mesh.pool.rms_mm'] = _mm(pool_mesh.get('heightError_m'), 'rms')
        record['mesh.pool.max_mm'] = _mm(pool_mesh.get('heightError_m'), 'maxAbs')
        record['mesh.vertices'] = int(len(current['mesh']))
        # The band, when present.
        phi = load_band(directory, step, band, dims)
        if phi is not None:
            surface = _surface(_replay.surface_metrics(
                cfg, lattice, lambda pts: 0.5 - lattice.trilinear(phi, pts) / h, step, rays))
            record.update({f'band.{k}': v for k, v in surface.items()})
            amount = _replay.amount_field(phi, lattice, w)
            record['band.amount_L'] = round(1e3 * float(amount.sum() * h ** 3), 4)
            record['band.minusSolver_L'] = round(1e3 * float((amount.sum() - density.sum()) * h ** 3), 4)
            liquid = phi < 0
            record['band.liquidCells'] = int(liquid.sum())
        published = current['receipt'].get('surfaceBand')
        if isinstance(published, dict):
            for key in ('bodies', 'maxShift_mm', 'fills', 'drains', 'unshiftedBodies'):
                if key in published:
                    record[f'band.{key}'] = published[key]
        records.append(record)
    return dict(capture=str(directory), arm=cfg['arm'], scene=cfg['scene_id'], h=h, dt=dt, dims=list(dims),
                band=band, amountRampWidth_m=w, steps=records)


def _steps_of(rule, available):
    steps = rule.get('steps', 'all')
    if steps == 'all':
        return list(available)
    return [int(s) for s in steps]


def check_limits(report, limits, criteria=None):
    """Evaluate the rules; returns (results, failed)."""
    by_step = {r['step']: r for r in report['steps']}
    results = []
    for rule in limits:
        if criteria is not None and rule['id'] not in criteria:
            continue
        required = criteria is not None
        for step in _steps_of(rule, by_step.keys()):
            record = by_step.get(step)
            value = None if record is None else record.get(rule['series'])
            entry = dict(id=rule['id'], series=rule['series'], step=step, value=value)
            if value is None:
                entry['status'] = 'fail' if required else 'missing'
                entry['reason'] = 'no value'
            else:
                ok = True
                if 'max' in rule and not value <= rule['max']:
                    ok = False
                if 'absMax' in rule and not abs(value) <= rule['absMax']:
                    ok = False
                if 'equals' in rule and value != rule['equals']:
                    ok = False
                entry['status'] = 'pass' if ok else 'fail'
                entry['limit'] = {k: rule[k] for k in ('max', 'absMax', 'equals') if k in rule}
            results.append(entry)
    failed = [r for r in results if r['status'] == 'fail']
    return results, failed


def _line(record):
    def f(key, digits=2):
        v = record.get(key)
        return '   -  ' if v is None else f'{v:6.{digits}f}'
    parts = [f"step {record['step']:2d}", f"gap {f('gap_mm', 0)}",
             f"contour {f('contour.sphere.rms_mm')}/{f('contour.sphere.max_mm', 1)}",
             f"mesh {f('mesh.sphere.rms_mm')}/{f('mesh.sphere.max_mm', 1)}"]
    if 'band.sphere.rms_mm' in record:
        parts.append(f"band {f('band.sphere.rms_mm')}/{f('band.sphere.max_mm', 1)} pool {f('band.pool.rms_mm')}")
    if 'band.minusSolver_L' in record:
        parts.append(f"dM {f('band.minusSolver_L', 3)} L")
    if 'band.bodies' in record:
        parts.append(f"bodies {record['band.bodies']} shift {f('band.maxShift_mm')}")
    if 'solver.massOutsideReference_ml' in record:
        parts.append(f"outside {f('solver.massOutsideReference_ml', 0)} ml frac {f('solver.fractionError_max', 3)}")
    return '  '.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--steps', help='Comma-separated captured steps (default: every captured step)')
    parser.add_argument('--band', default='none', help='none | phi | PREFIX of PREFIX-step-N.npy')
    parser.add_argument('--width', type=float, help='Amount ramp width in metres (default: one fine cell)')
    parser.add_argument('--limits', type=Path, help='tests/surface-band-limits.json')
    parser.add_argument('--criteria', help='Comma-separated criterion ids to evaluate (missing series fail)')
    parser.add_argument('--out', type=Path, help='Write the report as JSON')
    args = parser.parse_args(argv)
    steps = None if args.steps is None else [int(s) for s in args.steps.split(',')]
    report = evaluate(args.directory, steps, args.band, args.width)
    for record in report['steps']:
        print(_line(record))
    status = 0
    if args.limits:
        limits = json.loads(args.limits.read_text())
        criteria = None if args.criteria is None else set(args.criteria.split(','))
        results, failed = check_limits(report, limits, criteria)
        report['limits'] = results
        for r in results:
            if r['status'] != 'pass':
                print(f"{r['status'].upper():7s} {r['id']} {r['series']} step {r['step']}: {r['value']} {r.get('limit', r.get('reason'))}")
        passed = sum(1 for r in results if r['status'] == 'pass')
        print(f"limits: {passed} pass, {len(failed)} fail, {sum(1 for r in results if r['status'] == 'missing')} missing")
        status = 1 if failed else 0
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
    return status


if __name__ == '__main__':
    sys.exit(main())
