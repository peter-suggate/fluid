"""Read-only QA of published meshes and current-map authority checkpoints.

Usage: python3.11 tools/analyze-current-map-capture.py CAPTURE_ARM_DIRECTORY
Requires numpy. JSON output compares actual shipping triangles to a discrete
free-fall reference; no geometry is remeshed, fitted, smoothed or projected.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def errors(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), max=float(values.max()),
                rms=float(np.sqrt(np.mean(values ** 2))), range=float(np.ptp(values)))


def basis(t):
    return np.array([(1-t)**3, 4-6*t*t+3*t**3, 1+3*t+3*t*t-3*t**3, t**3])/6


def derivative(t):
    return np.array([-.5*(1-t)**2, 1.5*t*t-2*t, -1.5*t*t+t+.5, .5*t*t])


class MapSnapshot:
    def __init__(self, directory):
        self.meta = read_json(directory / 'current-map.json')
        self.layout = self.meta['map']
        raw_path = directory / 'current-map.bin'
        self.raw = (np.fromfile(raw_path, dtype='<f4') if raw_path.exists() else
                    np.frombuffer(gzip.decompress((directory / 'current-map.bin.gz').read_bytes()), dtype='<f4'))
        self.origin = np.array(self.layout['originFine'])
        self.spacing = self.layout.get('spacingFine',1.0)
        schedule_at = self.layout.get('traceSubstepCountBaseWords')
        self.trace_substeps = None if schedule_at is None else int(self.raw[schedule_at-self.meta['baseWords']])
        self.dims = np.array(self.layout['nodeDimensions'])
        control = self.meta['retainedControl']
        if control is None:
            raise ValueError('Checkpoint has no accepted retained control bank')
        self.bank = int(control[1])
        self.banks = [self.vector_plane(base) for base in self.layout['coefficientBaseWords']]
        self.coefficients = self.banks[self.bank]
        self.velocity = self.vector_plane(self.layout['immutableVelocityBaseWords'])
        self.chain = None
        if 'chainBaseWords' in self.layout:
            count = int(self.raw[self.layout['chainCountBaseWords']-self.meta['baseWords']])
            if not 0 <= count <= self.layout['chainCapacity']: raise ValueError('Invalid accepted map chain count')
            self.chain = [self.vector_plane(self.layout['chainBaseWords']+3*self.layout['nodeCount']*i) for i in range(count)]

    def vector_plane(self, base):
        first = base-self.meta['baseWords']
        return self.raw[first:first+3*self.layout['nodeCount']].reshape(tuple(self.dims[::-1])+(3,)).astype(float)

    def evaluate_increment(self, point, coefficients):
        point = np.array(point, dtype=float)
        local = (point-self.origin)/self.spacing
        if np.any(local < 0) or np.any(local > self.dims-1):
            return point, np.eye(3)
        cell = np.minimum(np.floor(local).astype(int), self.dims-2)
        t = local-cell
        b = [basis(v) for v in t]
        d = [derivative(v) for v in t]
        displacement, jacobian = np.zeros(3), np.eye(3)
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    q = cell+np.array([x,y,z])-1
                    if np.any(q < 0) or np.any(q >= self.dims):
                        continue
                    value = coefficients[q[2],q[1],q[0]]
                    displacement += value*b[0][x]*b[1][y]*b[2][z]
                    for axis in range(3):
                        w = [b[a][[x,y,z][a]] if a != axis else d[a][[x,y,z][a]] for a in range(3)]
                        jacobian[:,axis] += value*np.prod(w)/self.spacing
        return point+displacement, jacobian

    def evaluate(self, point, bank=None, chain_count=None):
        # Stored increments are departure maps: newest acts first. Jacobians
        # multiply on the left at each successively mapped physical point.
        if self.chain is None:
            return self.evaluate_increment(point,self.banks[self.bank if bank is None else bank])
        count = len(self.chain) if chain_count is None else chain_count
        mapped,jacobian = np.array(point,dtype=float),np.eye(3)
        if chain_count is None and bank is not None and bank != self.bank:
            mapped,jacobian = self.evaluate_increment(mapped,self.banks[bank])
        for coefficients in reversed(self.chain[:count]):
            mapped,local_jacobian = self.evaluate_increment(mapped,coefficients)
            jacobian = local_jacobian@jacobian
        return mapped,jacobian

    def frozen_velocity(self, point):
        # Production immutable velocity lattice is at native fine-cell centres.
        local = (np.array(point)-self.origin-.5)/self.spacing
        cell, t = np.floor(local).astype(int), local-np.floor(local)
        value = np.zeros(3)
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    offset = np.array([x,y,z]); q = cell+offset
                    if np.all(q >= 0) and np.all(q < self.dims):
                        value += np.prod(np.where(offset,t,1-t))*self.velocity[q[2],q[1],q[0]]
        dimensions = np.array(self.layout['dimensions'])
        point = np.array(point)
        value[0] *= np.clip(2*min(point[0],dimensions[0]-point[0]),-1,1)
        value[2] *= np.clip(2*min(point[2],dimensions[2]-point[2]),-1,1)
        value[1] *= np.clip(2*point[1],-1,1)
        outside = np.maximum(0,np.maximum(-point,point-dimensions)).max()
        t = np.clip((outside-4)/8,0,1)
        return value*(1-t*t*(3-2*t))

    def direct_composition(self, point, dt):
        departure = np.array(point,dtype=float)
        count = self.trace_substeps if self.trace_substeps is not None else max(1,int(np.ceil(np.linalg.norm(self.frozen_velocity(point))*dt)))
        if count < 1: raise ValueError("No valid global trace schedule in completed snapshot")
        subdt = dt/count
        for _ in range(count):
            first = self.frozen_velocity(departure)
            departure -= subdt*self.frozen_velocity(departure-.5*subdt*first)
        return self.evaluate(departure,1-self.bank if self.chain is None else None,chain_count=len(self.chain)-1 if self.chain is not None else None)[0]

    def measure(self):
        dims = self.meta['measure']['dimensions']; count = int(np.prod(dims))
        first = self.meta['measure']['baseWords']-self.meta['baseWords']+4*count*self.bank
        return self.raw[first:first+4*count].reshape(tuple(dims[::-1])+(4,)).astype(float)


def mesh_metrics(directory, center, radius, pool):
    triangles = np.fromfile(directory/'mesh.bin', dtype='<f4').reshape(-1,3,8)[:,:,:3].astype(float)
    # Components are separated by the middle of the physical air gap. Refuse
    # a contact frame rather than treating a connected surface as free fall.
    separator = .5*(pool+center[1]-radius)
    if center[1]-radius <= pool:
        return {'referenceScope': 'sphere/pool contact; free-fall mesh comparison unavailable'}
    sphere = triangles[np.min(triangles[:,:,1], axis=1)>separator]
    points = np.unique(sphere.reshape(-1,3),axis=0)
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    top = triangles[(np.max(triangles[:,:,1],axis=1)<separator)
                    &(np.min(triangles[:,:,1],axis=1)>pool-.1)
                    &(np.abs(normals[:,1])>1e-10)]
    top_points = np.unique(top.reshape(-1,3),axis=0)
    volumes = np.einsum('ij,ij->i', sphere[:,0], np.cross(sphere[:,1],sphere[:,2]))/6
    volume = volumes.sum()
    centroid = np.sum(volumes[:,None]*sphere.sum(axis=1)/4,axis=0)/volume
    unique, inverse = np.unique(sphere.reshape(-1,3),axis=0,return_inverse=True)
    faces = inverse.reshape(-1,3)
    edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _, counts = np.unique(edges,axis=0,return_counts=True)
    return dict(sphereTriangles=len(sphere), poolTopTriangles=len(top),
                sphereOpenEdges=int(np.sum(counts==1)), sphereNonmanifoldEdges=int(np.sum(counts>2)),
                expectedCenter_m=center.tolist(), enclosedMeshCenter_m=centroid.tolist(),
                centerError_m=(centroid-center).tolist(), sphereEnclosedVolume_m3=float(volume),
                sphereBounds_m=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
                sphereVertexRadialError_m=errors(np.linalg.norm(points-center,axis=1)-radius),
                sphereCenteredVertexRadialError_m=errors(np.linalg.norm(points-centroid,axis=1)-radius),
                poolHeightError_m=errors(top_points[:,1]-pool),
                poolPeakPoint_m=top_points[np.argmax(top_points[:,1])].tolist())


def native_consistency(directory, snapshot, config):
    nx,ny,nz = config['dimensions']; h = config['h']
    rho = np.fromfile(directory/'density.bin',dtype='<f4').reshape(nz,ny,nx).astype(float)
    fine = snapshot.measure()[:,:,:,0]
    expected = np.zeros_like(fine)
    activity = read_json(directory/'activity.json')
    widths = {}
    coverage = np.zeros_like(fine,dtype=np.uint8)
    for brick in activity['bricks']:
        if not brick['active']: continue
        width = 8*brick['spanBricks']/brick['acceptedResolution']
        if width != int(width): raise ValueError('Non-integral native support width')
        width = int(width); widths[width] = widths.get(width,0)+1
        lower = np.array(brick['coordinate'])*8
        for z in range(lower[2],lower[2]+8*brick['spanBricks'],width):
            for y in range(lower[1],lower[1]+8*brick['spanBricks'],width):
                for x in range(lower[0],lower[0]+8*brick['spanBricks'],width):
                    lo = np.maximum([x,y,z],0); hi = np.minimum(np.array([x,y,z])+width,[nx,ny,nz])
                    if np.any(lo>=hi): continue
                    cut = np.s_[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
                    expected[cut] = fine[cut].sum()/width**3
                    coverage[cut] += 1
    if np.any(coverage>1): raise ValueError('Overlapping accepted native ownership')
    delta = rho-expected
    return dict(nativeWidthActiveLeafHistogram=widths,
                maximumNativeMeanVsAcceptedMeasure=float(np.max(np.abs(delta))),
                L1NativeMeanVsAcceptedMeasure=float(np.sum(np.abs(delta))),
                nativeAmount_m3=float(rho.sum()*h**3), acceptedMeasureAmount_m3=float(fine.sum()*h**3),
                uncoveredMeasureAmount_m3=float(fine[coverage==0].sum()*h**3))


def analyze(directory, points):
    config = read_json(directory/'configuration.json'); scene = config['scene']
    sphere = next(v for v in scene['fluid']['initialLiquidVolumes'] if v['shape']=='sphere')
    initial = np.array([sphere['center_m'][a] for a in 'xyz'])
    gravity = np.array([scene['fluid']['gravity_m_s2'][a] for a in 'xyz'])
    dt,h = config['dt'],config['h']; origin = np.array(config['origin'])
    pool = scene['container']['height_m']*scene['container']['fillFraction']
    result = {'scope': 'Read-only actual published mesh and accepted spatial measure; analytic motion is QA only.',
              'velocityScope': 'Frozen pre-gather VEX from the captured completed step, before that step gravity/projection.',
              'frames':[]}
    for path in sorted(directory.glob('step-*'),key=lambda p:int(p.name.split('-')[1])):
        if not (path/'mesh.bin').exists() or not (path/'density.bin').exists(): continue
        step = int(path.name.split('-')[1]); center = initial+gravity*dt*dt*step*(step-1)/2
        snapshot = MapSnapshot(path)
        probes = []
        for point in points:
            departure,jacobian = snapshot.evaluate(point); det = np.linalg.det(jacobian)
            seed_world = origin+h*departure
            pool_phi = seed_world[1]-pool
            sphere_phi = .5*sphere['radius_m']*(np.sum(((seed_world-initial)/sphere['radius_m'])**2)-1)
            seed = np.clip(.5-min(pool_phi,sphere_phi)/h,0,1)
            probes.append(dict(pointFine=point, physicalPoint_m=(origin+h*np.array(point)).tolist(),
                               frozenVelocity_m_s=(h*snapshot.frozen_velocity(point)).tolist(),
                               departureDisplacement_m=(h*(departure-np.array(point))).tolist(),
                               splineResamplingDisplacementError_m=(h*(departure-snapshot.direct_composition(point,dt))).tolist() if step else None,
                               inverseMapJacobian=jacobian.tolist(), inverseMapDeterminant=float(det),
                               seedDensityAtDeparture=float(seed), transportedPointDensity=float(seed*det)))
        result['frames'].append(dict(step=step,retainedControl=snapshot.meta['retainedControl'],
                                     scalarParity=snapshot.meta['scalarParity'], mapSpacingFine=snapshot.spacing,
                                     traceSubsteps=snapshot.trace_substeps,
                                     acceptedChainCount=len(snapshot.chain) if snapshot.chain is not None else None,
                                     mesh=mesh_metrics(path,center,sphere['radius_m'],pool),
                                     native=native_consistency(path,snapshot,config), probes=probes))
    baseline = result['frames'][0]['mesh']
    for frame in result['frames']:
        mesh = frame['mesh']
        if 'sphereVertexRadialError_m' in mesh:
            mesh['radialRmsChangeFromInitial_m'] = mesh['sphereVertexRadialError_m']['rms']-baseline['sphereVertexRadialError_m']['rms']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--point',action='append',help='Fine-coordinate x,y,z; repeatable')
    args = parser.parse_args()
    points = [list(map(float,p.split(','))) for p in args.point] if args.point else [[15.5,y,15.5] for y in [7.5,8,8.5,9.5]]
    if any(len(p)!=3 for p in points): parser.error('Each point requires three coordinates')
    report = analyze(args.directory,points)
    output = args.out or args.directory/'current-map-analysis.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)
