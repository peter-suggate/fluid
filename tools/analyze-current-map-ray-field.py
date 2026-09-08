"""Read-only half-density surface QA, independent of presentation triangles.

Usage: python3.11 tools/analyze-current-map-ray-field.py CAPTURE_ARM_DIRECTORY
Requires numpy. Replays accepted float32 spline coefficients in float64; this
is an offline field check, not a bitwise replay of shader arithmetic. Analytic
free fall is only a reference. No surface or simulation values are modified.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


_spec = importlib.util.spec_from_file_location(
    'current_map_capture', Path(__file__).with_name('analyze-current-map-capture.py'))
_capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_capture)


class MappedSnapshot(_capture.MapSnapshot):
    """Map raw captures or read only required windows from compressed captures."""
    def __init__(self, directory, basename='current-map'):
        self.meta = _capture.read_json(directory / f'{basename}.json')
        self.layout = self.meta['map']
        raw_path = directory / f'{basename}.bin'
        self.raw = np.memmap(raw_path, dtype='<f4', mode='r') if raw_path.exists() else None
        self._compressed = None if self.raw is not None else gzip.open(directory / f'{basename}.bin.gz', 'rb')
        self.origin = np.asarray(self.layout['originFine'], dtype=float)
        self.spacing = self.layout.get('spacingFine', 1.0)
        self.dims = np.asarray(self.layout['nodeDimensions'])
        control = self.meta['retainedControl']
        if control is None:
            raise ValueError('Checkpoint has no accepted retained control bank')
        self.bank = int(control[1])
        self.banks = [self.vector_plane(base) for base in self.layout['coefficientBaseWords']]
        self.coefficients = self.banks[self.bank]
        self.velocity = self.vector_plane(self.layout['immutableVelocityBaseWords'])
        schedule = self.layout.get('traceSubstepCountBaseWords')
        self.trace_substeps = None if schedule is None else int(self.read_words(schedule, 1)[0])
        self.chain = None
        if 'chainBaseWords' in self.layout:
            count = int(self.read_words(self.layout['chainCountBaseWords'], 1)[0])
            if not 0 <= count <= self.layout['chainCapacity']:
                raise ValueError('Invalid accepted map chain count')
            self.chain = [self.vector_plane(self.layout['chainBaseWords'] + 3*self.layout['nodeCount']*i)
                          for i in range(count)]
        if self._compressed is not None:
            self._compressed.close()

    def read_words(self, base, count):
        first = base-self.meta['baseWords']
        if self.raw is not None:
            return self.raw[first:first+count]
        self._compressed.seek(4*first)
        return np.frombuffer(self._compressed.read(4*count), dtype='<f4')

    def vector_plane(self, base):
        return self.read_words(base, 3*self.layout['nodeCount']).reshape(tuple(self.dims[::-1])+(3,))


def evaluate_increment_many(snapshot, points, coefficients):
    """Same cubic cardinal basis and physical derivatives as MapSnapshot."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    local = (points-snapshot.origin)/snapshot.spacing
    active = np.all((local >= 0) & (local <= snapshot.dims-1), axis=1)
    result = points.copy()
    jacobians = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
    local = local[active]
    if not len(local):
        return result, jacobians
    cell = np.minimum(np.floor(local).astype(int), snapshot.dims-2)
    t = local-cell
    b = [_capture.basis(t[:, axis]) for axis in range(3)]
    d = [_capture.derivative(t[:, axis]) for axis in range(3)]
    displacement = np.zeros((len(local), 3))
    gradient = np.zeros((len(local), 3, 3))
    for z in range(4):
        for y in range(4):
            for x in range(4):
                q = cell+[x, y, z]-1
                valid = np.all((q >= 0) & (q < snapshot.dims), axis=1)
                q = q[valid]
                value = coefficients[q[:, 2], q[:, 1], q[:, 0]].astype(float)
                indices = [x, y, z]
                weight = b[0][x]*b[1][y]*b[2][z]
                displacement[valid] += value*weight[valid, None]
                for axis in range(3):
                    weight = np.ones(len(local))
                    for a in range(3):
                        weight *= (d if a == axis else b)[a][indices[a]]
                    gradient[valid, :, axis] += value*weight[valid, None]/snapshot.spacing
    result[active] += displacement
    jacobians[active] += gradient
    return result, jacobians


def evaluate_many(snapshot, points, bank=None):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if snapshot.chain is None:
        return evaluate_increment_many(snapshot, points, snapshot.banks[snapshot.bank if bank is None else bank])
    mapped = points.copy()
    jacobians = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
    if bank is not None and bank != snapshot.bank:
        mapped, jacobians = evaluate_increment_many(snapshot, mapped, snapshot.banks[bank])
    for coefficients in reversed(snapshot.chain):
        mapped, local_jacobians = evaluate_increment_many(snapshot, mapped, coefficients)
        jacobians = np.einsum('nij,njk->nik', local_jacobians, jacobians)
    return mapped, jacobians


def frozen_velocity_many(snapshot, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    local = (points-snapshot.origin-.5)/snapshot.spacing
    cells = np.floor(local).astype(int)
    t = local-cells
    values = np.zeros(points.shape)
    for corner in range(8):
        offset = np.array([corner & 1, (corner >> 1) & 1, corner >> 2])
        q = cells+offset
        valid = np.all((q >= 0) & (q < snapshot.dims), axis=1)
        indices = q[valid]
        weights = np.prod(np.where(offset, t, 1-t), axis=1)
        values[valid] += weights[valid, None]*snapshot.velocity[indices[:, 2], indices[:, 1], indices[:, 0]]
    dimensions = np.asarray(snapshot.layout['dimensions'])
    values[:, 0] *= np.clip(2*np.minimum(points[:, 0], dimensions[0]-points[:, 0]), -1, 1)
    values[:, 2] *= np.clip(2*np.minimum(points[:, 2], dimensions[2]-points[:, 2]), -1, 1)
    values[:, 1] *= np.clip(2*points[:, 1], -1, 1)
    outside = np.maximum(0, np.maximum(-points, points-dimensions)).max(axis=1)
    blend = np.clip((outside-4)/8, 0, 1)
    return values*(1-blend*blend*(3-2*blend))[:, None]


def packed(values):
    # Source primitives and frame h enter the shader through float32 buffers.
    return np.asarray(values, dtype=np.float32).astype(float)


class SeedField:
    """Replay supported source primitives; never infer seed geometry from rho."""
    def __init__(self, config):
        scene = config['scene']
        fluid, container = scene['fluid'], scene['container']
        if fluid.get('initialBrickSeeds_m'):
            raise ValueError('Ray QA has no source compiler for authored brick seeds')
        if fluid['initialCondition'] != 'tank-fill':
            raise ValueError('Ray QA currently requires an authored tank-fill source')
        if scene.get('rigidBodies') or scene.get('terrain'):
            raise ValueError('Ray QA currently requires no moving rigid bodies or terrain')
        self.h = float(packed(config['h']))
        self.width = float(packed(scene['voxelDomain']['finestCellSize_m']))
        self.origin = packed(config['origin'])
        self.dimensions = np.asarray(config['dimensions'])
        for solid in scene.get('solidVoxels', []):
            lo = np.maximum(solid['minimum'], 0)
            hi = np.minimum(solid['maximumExclusive'], self.dimensions)
            if solid['operation'] != 'clear' and np.all(hi > lo):
                raise ValueError('Ray QA currently requires no authored solids inside the tank')
        self.lower = packed([-container['width_m']/2, 0, -container['depth_m']/2])
        self.upper = packed([container['width_m']/2, container['height_m'], container['depth_m']/2])
        self.support_upper = self.lower+self.dimensions*self.h
        self.primitives = []
        height = fluid.get('initialHeightField')
        if height:
            if height['kind'] != 'quadratic':
                raise ValueError('Unsupported authored height field')
            self.primitives.append(('height', packed([height['center_m']['x'], height['baseHeight_m'], height['center_m']['z']]),
                                    packed([height['curvatureX_mInv'], 0, height['curvatureZ_mInv']])))
        elif container['fillFraction'] > 0:
            self.primitives.append(('height', packed([0, container['height_m']*container['fillFraction'], 0]), np.zeros(3)))
        for volume in fluid.get('initialLiquidVolumes', []):
            if volume['shape'] == 'sphere':
                self.primitives.append(('ellipsoid', packed([volume['center_m'][a] for a in 'xyz']), packed([volume['radius_m']]*3)))
            elif volume['shape'] == 'box':
                self.primitives.append(('box', packed([volume['min_m'][a] for a in 'xyz']), packed([volume['max_m'][a] for a in 'xyz'])))
            else:
                raise ValueError('Unsupported authored liquid primitive')

    def phi(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        phi = np.full(len(points), 1e6)
        for kind, a, b in self.primitives:
            if kind == 'ellipsoid':
                value = .5*np.min(b)*(np.sum(((points-a)/b)**2, axis=1)-1)
            elif kind == 'height':
                delta = points-a
                value = delta[:, 1]-b[0]*delta[:, 0]**2-b[2]*delta[:, 2]**2
            else:
                value = np.full(len(points), -1e6)
                for axis in range(3):
                    if a[axis] > self.lower[axis]:
                        value = np.maximum(value, a[axis]-points[:, axis])
                    if b[axis] < self.upper[axis]:
                        value = np.maximum(value, points[:, axis]-b[axis])
            phi = np.minimum(phi, value)
        phi[np.any((points < self.lower) | (points > self.support_upper), axis=1)] = self.width
        return phi

    def density(self, points):
        return np.clip(.5-self.phi(points)/self.width, 0, 1)


class DensityAuthority:
    def __init__(self, snapshot, seed, bank=None):
        self.snapshot, self.seed, self.bank = snapshot, seed, bank

    def __call__(self, world_points):
        fine = (np.asarray(world_points)-self.seed.origin)/self.seed.h
        mapped, jacobians = evaluate_many(self.snapshot, fine, self.bank)
        density = self.seed.density(self.seed.origin+self.seed.h*mapped)*np.linalg.det(jacobians)
        density[np.any((fine < 0) | (fine > self.seed.dimensions), axis=1)] = 0
        if not np.all(np.isfinite(density)):
            raise ValueError('Nonfinite accepted density in ray query')
        return density


def antipodal_directions(count=134):
    """Cardinal axes plus a deterministic hemisphere and its exact antipodes."""
    if count < 6 or count % 2:
        raise ValueError('Direction count must be even and at least six')
    n = (count-6)//2
    z = (np.arange(n)+.5)/max(n, 1)
    angle = np.arange(n)*(np.pi*(3-np.sqrt(5)))
    radial = np.sqrt(1-z*z)
    hemisphere = np.column_stack([radial*np.cos(angle), radial*np.sin(angle), z])
    half = np.concatenate([np.eye(3), hemisphere])
    return np.concatenate([half, -half])


def descending_roots(density, starts, directions, maximum, scan_samples=65, tolerance=1e-8):
    """Find every sampled water-to-air bracket, then bisect each independently.

    Multiple or absent crossings stay explicit. The scan cannot prove absence
    of a feature narrower than its spacing; the report retains that spacing.
    """
    starts, directions = np.asarray(starts), np.asarray(directions)
    maximum = np.broadcast_to(maximum, (len(starts),))
    if scan_samples < 2 or tolerance <= 0 or np.any(maximum <= 0):
        raise ValueError('Root scan requires positive extent/tolerance and at least two samples')
    distances = maximum[:, None]*np.linspace(0, 1, scan_samples)
    points = starts[:, None, :]+distances[:, :, None]*directions[:, None, :]
    sampled = density(points.reshape(-1, 3)).reshape(len(starts), scan_samples)
    residual = sampled-.5
    descending = (residual[:, :-1] > 0) & (residual[:, 1:] <= 0)
    ascending = (residual[:, :-1] < 0) & (residual[:, 1:] >= 0)
    rows, columns = np.nonzero(descending)
    lo, hi = distances[rows, columns].copy(), distances[rows, columns+1].copy()
    while len(lo) and np.max(hi-lo) > tolerance:
        mid = (lo+hi)/2
        wet = density(starts[rows]+mid[:, None]*directions[rows]) > .5
        lo = np.where(wet, mid, lo)
        hi = np.where(wet, hi, mid)
    crossings = (lo+hi)/2
    final_density = density(starts[rows]+crossings[:, None]*directions[rows]) if len(rows) else np.zeros(0)
    counts = descending.sum(axis=1)
    roots = np.full(len(starts), np.nan)
    unique = counts[rows] == 1
    roots[rows[unique]] = crossings[unique]
    return dict(roots=roots, descending_counts=counts, ascending_counts=ascending.sum(axis=1),
                all_roots=[crossings[rows == i].tolist() for i in range(len(starts))],
                density_at_start=sampled[:, 0], density_at_end=sampled[:, -1],
                maximum_scan_spacing=float(np.max(maximum)/(scan_samples-1)),
                maximum_bracket_width=float(np.max(hi-lo)) if len(lo) else None,
                maximum_density_residual=float(np.max(np.abs(final_density-.5))) if len(rows) else None)


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return None if not len(values) else dict(_capture.errors(values), maxAbs=float(np.max(np.abs(values))))


def crossing_report(result):
    counts = result['descending_counts']
    return dict(sampledRays=int(len(counts)), uniqueRoots=int(np.sum(counts == 1)),
                missingRoots=int(np.sum(counts == 0)), multipleRoots=int(np.sum(counts > 1)),
                ascendingCrossings=int(np.sum(result['ascending_counts'])),
                maximumScanSpacing_m=result['maximum_scan_spacing'],
                maximumFinalBracketWidth_m=result['maximum_bracket_width'],
                maximumRootDensityResidual=result['maximum_density_residual'])


def sphere_metrics(density, center, radius, pool, lower, upper, directions, scan_samples, tolerance):
    center = np.asarray(center)
    gap = center[1]-radius-pool
    if gap <= 0:
        return dict(referenceScope='Reference sphere/pool contact; free-fall radial comparison unavailable',
                    expectedCenter_m=center.tolist(), expectedGap_m=float(gap))
    # End each ray in the sphere side of the reference air-gap separator and
    # inside the tank. This excludes the pool's separate liquid component.
    maximum = np.full(len(directions), 2*radius, dtype=float)
    separator = .5*(pool+center[1]-radius)
    downward = directions[:, 1] < 0
    maximum[downward] = np.minimum(maximum[downward], (center[1]-separator)/-directions[downward, 1])
    for axis in range(3):
        nonzero = directions[:, axis] != 0
        boundary = np.where(directions[:, axis] > 0, upper[axis], lower[axis])
        distance = np.zeros(len(directions))
        distance[nonzero] = (boundary[nonzero]-center[axis])/directions[nonzero, axis]
        maximum[nonzero] = np.minimum(maximum[nonzero], distance[nonzero])
    roots = descending_roots(density, np.broadcast_to(center, directions.shape), directions, maximum, scan_samples, tolerance)
    half = len(directions)//2
    chords = roots['roots'][:half]+roots['roots'][half:]
    errors = roots['roots']-radius
    samples = [dict(direction=direction.tolist(), radius_m=float(r) if np.isfinite(r) else None,
                    radialError_m=float(error) if np.isfinite(error) else None,
                    descendingRoots_m=all_roots)
               for direction, r, error, all_roots in zip(directions, roots['roots'], errors, roots['all_roots'])]
    return dict(crossing_report(roots), expectedCenter_m=center.tolist(), expectedRadius_m=radius,
                expectedGap_m=float(gap), radialError_m=statistics(errors),
                referenceCenteredChordError_m=statistics(chords-2*radius), samples=samples)


def pool_metrics(density, pool, sphere_center, radius, lower, upper, h, grid_count, scan_samples, tolerance):
    gap = sphere_center[1]-radius-pool
    if gap <= 0:
        return dict(referenceScope='Reference sphere/pool contact; separate flat-pool comparison unavailable')
    # Half-fine-cell inset samples the wall-adjacent open fluid as well as the
    # interior. No region near the falling sphere is omitted.
    x = np.linspace(lower[0]+.5*h, upper[0]-.5*h, grid_count)
    z = np.linspace(lower[2]+.5*h, upper[2]-.5*h, grid_count)
    xx, zz = np.meshgrid(x, z)
    low_y = max(lower[1], pool-2*h)
    high_y = min(pool+2*h, .5*(pool+sphere_center[1]-radius), upper[1])
    starts = np.column_stack([xx.ravel(), np.full(xx.size, low_y), zz.ravel()])
    directions = np.broadcast_to([0., 1., 0.], starts.shape)
    roots = descending_roots(density, starts, directions, high_y-low_y, scan_samples, tolerance)
    heights = low_y+roots['roots']
    samples = [dict(x_m=float(point[0]), z_m=float(point[2]), height_m=float(height) if np.isfinite(height) else None,
                    heightError_m=float(height-pool) if np.isfinite(height) else None,
                    descendingHeights_m=[low_y+r for r in all_roots])
               for point, height, all_roots in zip(starts, heights, roots['all_roots'])]
    return dict(crossing_report(roots), expectedHeight_m=pool, heightError_m=statistics(heights-pool),
                gridDimensions=[grid_count, grid_count], samples=samples)


def analyze(directory, steps=None, direction_count=134, grid_count=17, scan_samples=65, tolerance=1e-8):
    config = _capture.read_json(directory/'configuration.json')
    seed = SeedField(config)
    scene, dt = config['scene'], config['dt']
    spheres = [v for v in scene['fluid'].get('initialLiquidVolumes', []) if v['shape'] == 'sphere']
    if len(spheres) != 1 or scene['fluid'].get('initialHeightField'):
        raise ValueError('Sphere/flat-pool reference currently requires exactly one authored sphere and a flat tank fill')
    sphere = spheres[0]
    initial = np.asarray([sphere['center_m'][a] for a in 'xyz'])
    initial_velocity = np.asarray([scene['fluid'].get('initialVelocity_m_s', {}).get(a, 0) for a in 'xyz'])
    gravity = np.asarray([scene['fluid']['gravity_m_s2'][a] for a in 'xyz'])
    radius = sphere['radius_m']
    pool = scene['container']['height_m']*scene['container']['fillFraction']
    directions = antipodal_directions(direction_count)
    report = dict(scope='Accepted current-map half-density roots. Seed authoring and discrete gravity are QA references only; no mesh fitting or geometry changes.',
                  arithmetic='Stored float32 coefficients and packed float32 source parameters evaluated in float64; not a bitwise shader replay.',
                  sampling='Finite ray/grid sampling, not a global surface error certificate. All sampled crossings and root brackets are retained.',
                  referenceCenter='initialCenter + initialVelocity*dt*n + gravity*dt^2*n*(n-1)/2',
                  rootTolerance_m=tolerance, scanSamples=scan_samples, frames=[])
    for path in sorted(directory.glob('step-*'), key=lambda p: int(p.name.split('-')[1])):
        step = int(path.name.split('-')[1])
        if steps is not None and step not in steps:
            continue
        if not (path/'current-map.json').exists() or not ((path/'current-map.bin').exists() or (path/'current-map.bin.gz').exists()):
            continue
        snapshot = MappedSnapshot(path)
        density = DensityAuthority(snapshot, seed)
        center = initial+initial_velocity*dt*step+gravity*dt*dt*step*(step-1)/2
        report['frames'].append(dict(step=step, time_s=dt*step, acceptedBank=snapshot.bank,
            acceptedChainCount=None if snapshot.chain is None else len(snapshot.chain), mapSpacingFine=snapshot.spacing,
            sphere=sphere_metrics(density, center, radius, pool, seed.lower, seed.support_upper, directions, scan_samples, tolerance),
            pool=pool_metrics(density, pool, center, radius, seed.lower, seed.support_upper, seed.h, grid_count, scan_samples, tolerance)))
        print(f'Analyzed accepted half-density field at step {step}', file=sys.stderr, flush=True)
    if not report['frames']:
        raise ValueError('No complete accepted checkpoints matched the requested steps')
    return report


def check_limits(report, sphere_limit=None, pool_limit=None):
    """Apply caller-supplied physical limits without changing the measurement."""
    violations = []
    compared = 0
    for frame in report['frames']:
        for field, key, limit in [('sphere', 'radialError_m', sphere_limit), ('pool', 'heightError_m', pool_limit)]:
            if limit is None:
                continue
            if not np.isfinite(limit) or limit < 0:
                raise ValueError('Acceptance limits must be finite and nonnegative')
            metrics = frame[field]
            if 'referenceScope' in metrics:
                continue
            compared += 1
            if metrics['uniqueRoots'] != metrics['sampledRays']:
                violations.append(dict(step=frame['step'], field=field, reason='Missing or multiple sampled surface roots'))
            value = metrics[key]
            if value is None or value['maxAbs'] > limit:
                violations.append(dict(step=frame['step'], field=field, reason='Maximum absolute surface error exceeds supplied limit',
                                       measured_m=None if value is None else value['maxAbs'], limit_m=limit))
    if not compared:
        violations.append(dict(reason='No precontact fields were compared against supplied limits'))
    return dict(sphereRadialLimit_m=sphere_limit, poolHeightLimit_m=pool_limit,
                comparedFields=compared, passed=not violations, violations=violations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--steps', help='Comma-separated captured step numbers')
    parser.add_argument('--directions', type=int, default=134)
    parser.add_argument('--pool-grid', type=int, default=17)
    parser.add_argument('--scan-samples', type=int, default=65)
    parser.add_argument('--root-tolerance-m', type=float, default=1e-8)
    parser.add_argument('--max-sphere-radial-error-m', type=float, help='Optional explicit acceptance limit; there is no default')
    parser.add_argument('--max-pool-height-error-m', type=float, help='Optional explicit acceptance limit; there is no default')
    args = parser.parse_args()
    if args.pool_grid < 2:
        parser.error('--pool-grid must be at least two')
    report = analyze(args.directory, None if args.steps is None else set(map(int, args.steps.split(','))),
                     args.directions, args.pool_grid, args.scan_samples, args.root_tolerance_m)
    if args.max_sphere_radial_error_m is not None or args.max_pool_height_error_m is not None:
        report['acceptance'] = check_limits(report, args.max_sphere_radial_error_m, args.max_pool_height_error_m)
    output = args.out or args.directory/'current-map-ray-field-analysis.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output)
    if 'acceptance' in report and not report['acceptance']['passed']:
        sys.exit(1)
