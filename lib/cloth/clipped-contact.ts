import type { ClothCollider } from './contact';
import {clothRodEllipsoidAxes,clothRodLocalVector,clothRodContains,clothRodNormal,type ClothRodCapsuleGeometry} from './rod-geometry';
import { supportClippedRod, convexClothDistance } from './rod-convex';
import type { ClothVector } from './types';
import { cross3, dot3, norm3, scale3, sub3 } from './topology';
import { quaternionInverseRotate, quaternionMultiply, quaternionRotate } from '../core/rigid-body';
const add = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export function clippedRodDistanceLowerBound(shape:ClothRodCapsuleGeometry,p:ClothVector):number{
 const {radii,length}=clothRodEllipsoidAxes(shape),q=clothRodLocalVector(shape,sub3(p,shape.a)),z=q[2]-Math.max(0,Math.min(length/radii[2],q[2])),distance=Math.hypot(q[0],q[1],z)-1,lower=distance*(distance<0?Math.max(...radii):Math.min(...radii));
 return Math.max(lower,...shape.planes.map(plane=>dot3(plane.normal,p)-plane.offset_m));
}
/** Euclidean projection onto the exact swept ellipsoid, using the monotone
 * ellipsoid Lagrange multiplier equation rather than radial rescaling. */
function projectUncutRod(shape:ClothRodCapsuleGeometry,p:ClothVector):ClothVector{
 const {d1,d2,t,radii,length}=clothRodEllipsoidAxes(shape),q=sub3(p,shape.a),z=Math.max(0,Math.min(length,dot3(q,t))),center=add(shape.a,scale3(t,z)),delta=sub3(p,center),basis=[d1,d2,t],v=basis.map(axis=>dot3(delta,axis)),r2=radii.map(r=>r*r);
 if(v.reduce((sum,value,i)=>sum+value*value/r2[i]!,0)<=1)return p;
 const value=(lambda:number)=>v.reduce((sum,x,i)=>sum+r2[i]!*x*x/(r2[i]!+lambda)**2,0);let lo=0,hi=Math.max(...r2);
 while(value(hi)>1)hi*=2;
 for(let i=0;i<80;i++){const mid=(lo+hi)/2;if(value(mid)>1)lo=mid;else hi=mid;}
 return [0,1,2].map(a=>center[a]!+basis.reduce((sum,axis,i)=>sum+axis[a]!*r2[i]!*v[i]!/(r2[i]!+hi),0)) as unknown as ClothVector;
}
/** Interior closest-boundary solve. The secular root can reach the smallest
 * squared radius when its coordinate is zero; retain that singular solution
 * instead of choosing an arbitrary GJK normal inside the volume. */
function closestRodInteriorBoundary(shape:ClothRodCapsuleGeometry,p:ClothVector):{point:ClothVector;normal:ClothVector;distance:number}{
 const {d1,d2,t,radii,length}=clothRodEllipsoidAxes(shape),q=sub3(p,shape.a),z=Math.max(0,Math.min(length,dot3(q,t))),center=add(shape.a,scale3(t,z)),basis=[d1,d2,t],v=basis.map(axis=>dot3(sub3(p,center),axis)),r2=radii.map(r=>r*r),minimum=Math.min(...r2),small=r2.map(r=>Math.abs(r-minimum)<=minimum*1e-14);
 const lowerValue=v.reduce((sum,x,i)=>small[i]?(Math.abs(x)>1e-30?Infinity:sum):sum+r2[i]!*x*x/(r2[i]!-minimum)**2,0),coordinates=[0,0,0];
 if(lowerValue>1){let lo=-minimum,hi=0;for(let k=0;k<100;k++){const mid=(lo+hi)/2,value=v.reduce((sum,x,i)=>sum+r2[i]!*x*x/(r2[i]!+mid)**2,0);if(value>1)lo=mid;else hi=mid;}for(let i=0;i<3;i++)coordinates[i]=r2[i]!*v[i]!/(r2[i]!+hi);}
 else{let remaining=1;for(let i=0;i<3;i++)if(!small[i]){coordinates[i]=r2[i]!*v[i]!/(r2[i]!-minimum);remaining-=coordinates[i]!**2/r2[i]!;}coordinates[small.indexOf(true)]=Math.sqrt(Math.max(0,remaining)*minimum);}
 let point=[0,1,2].map(a=>center[a]!+basis.reduce((sum,axis,i)=>sum+axis[a]!*coordinates[i]!,0)) as unknown as ClothVector,normal=clothRodNormal(shape,point),distance=norm3(sub3(point,p));
 for(const plane of shape.planes){const d=plane.offset_m-dot3(plane.normal,p);if(d<distance){distance=d;normal=plane.normal;point=add(p,scale3(normal,d));}}
 return {point,normal,distance};
}
function projectRod(shape: ClothRodCapsuleGeometry, p: ClothVector): ClothVector { let x = p; const corrections: ClothVector[] = Array.from({ length: shape.planes.length + 1 }, () => [0, 0, 0]); for (let iteration = 0; iteration < 256; iteration++) {
    const before = x;
    for (let j = 0; j < corrections.length; j++) {
        const q = add(x, corrections[j]!);
        let projected: ClothVector;
        if (j === 0) {
            projected = projectUncutRod(shape,q);
        }
        else {
            const plane = shape.planes[j - 1]!, distance = Math.max(0, dot3(plane.normal, q) - plane.offset_m);
            projected = sub3(q, scale3(plane.normal, distance));
        }
        corrections[j] = sub3(q, projected);
        x = projected;
    }
    if (norm3(sub3(x, before)) < 1e-12 && clippedRodDistanceLowerBound(shape, x) < 1e-10)
        return x;
} throw new Error('Clipped rod projection did not converge'); }
function colliderSupport(c: ClothCollider, beforeEnd: number): ((d: ClothVector) => ClothVector) | undefined {
    const shift = scale3(c.velocity_m_s ?? [0, 0, 0], -beforeEnd);
    if (c.kind === 'sphere')
        return d => add(add(c.center_m, shift), scale3(d, c.radius_m / norm3(d)));
    if (c.kind === 'box')
        return d => [...(d.map((v, a) => (v >= 0 ? c.maximum_m[a]! : c.minimum_m[a]!) + shift[a]!))] as unknown as ClothVector;
    if (c.kind !== 'rigid' || c.body.description.fluidBoundary === 'interior' || c.body.description.shape === 'cup'||(c.kind==='rigid'&&!!c.body.description.cuts?.length))
        return undefined;
    const b = c.body, dimensions = b.description.dimensions_m, omega = b.angularVelocity_rad_s, speed = Math.hypot(omega.x, omega.y, omega.z), half = -.5 * speed * beforeEnd, s = speed ? Math.sin(half) / speed : 0, q = quaternionMultiply({ w: Math.cos(half), x: omega.x * s, y: omega.y * s, z: omega.z * s }, b.orientation), center: ClothVector = [b.position_m.x - beforeEnd * b.linearVelocity_m_s.x, b.position_m.y - beforeEnd * b.linearVelocity_m_s.y, b.position_m.z - beforeEnd * b.linearVelocity_m_s.z];
    return direction => {
        const d = quaternionInverseRotate(q, { x: direction[0], y: direction[1], z: direction[2] }), length = Math.hypot(d.x, d.y, d.z);
        let p;
        switch (b.description.shape) {
            case 'sphere':
                p = { x: d.x * dimensions.x / length, y: d.y * dimensions.x / length, z: d.z * dimensions.x / length };
                break;
            case 'box':
                p = { x: Math.sign(d.x) * dimensions.x / 2, y: Math.sign(d.y) * dimensions.y / 2, z: Math.sign(d.z) * dimensions.z / 2 };
                break;
            case 'capsule':
                p = { x: d.x * dimensions.x / length, y: d.y * dimensions.x / length + Math.sign(d.y) * dimensions.y / 2, z: d.z * dimensions.x / length };
                break;
            case 'rounded-plate': p={x:Math.sign(d.x)*dimensions.x+d.x*dimensions.y/length,y:d.y*dimensions.y/length,z:Math.sign(d.z)*dimensions.z+d.z*dimensions.y/length};break;
            case 'cylinder': {
                const radial = Math.hypot(d.x, d.z);
                p = { x: radial ? d.x * dimensions.x / radial : 0, y: Math.sign(d.y) * dimensions.y / 2, z: radial ? d.z * dimensions.x / radial : 0 };
                break;
            }
            default: throw new Error('Unsupported convex rigid support');
        }
        const world = quaternionRotate(q, p);
        return add(center, [world.x, world.y, world.z]);
    };
}
export interface ClippedRodColliderSample {
    readonly distance_m: number;
    readonly lowerBound_m: number;
    readonly point: ClothVector;
    readonly normal: ClothVector;
}
/** Same analytic clipped volume as rendering and picking. General sampled
 * fields use Lipschitz branch-and-bound, not an uncut endpoint sphere. */
export function sampleClippedRodCollider(shape: ClothRodCapsuleGeometry, collider: ClothCollider, beforeEnd: number, sample: (c: ClothCollider, p: ClothVector, t: number) => {
    distance_m: number;
    normal: ClothVector;
}): ClippedRodColliderSample {
    if (collider.kind === 'plane') {
        const normal = scale3(collider.normal, 1 / norm3(collider.normal)), point = supportClippedRod(shape, scale3(normal, -1)), distance = sample(collider, point, beforeEnd).distance_m;
        return { distance_m: distance, lowerBound_m: distance, point, normal };
    }
    if (collider.kind === 'sphere' || (collider.kind === 'rigid' && collider.body.description.shape === 'sphere' && collider.body.description.fluidBoundary !== 'interior')) {
        const center: ClothVector = collider.kind === 'sphere' ? sub3(collider.center_m, scale3(collider.velocity_m_s ?? [0, 0, 0], beforeEnd)) : [collider.body.position_m.x - beforeEnd * collider.body.linearVelocity_m_s.x, collider.body.position_m.y - beforeEnd * collider.body.linearVelocity_m_s.y, collider.body.position_m.z - beforeEnd * collider.body.linearVelocity_m_s.z], radius = collider.kind === 'sphere' ? collider.radius_m : collider.body.description.dimensions_m.x;
        if(clothRodContains(shape,center,0)){
            const boundary=closestRodInteriorBoundary(shape,center),distance=-boundary.distance-radius;
            return {distance_m:distance,lowerBound_m:distance,point:boundary.point,normal:scale3(boundary.normal,-1)};
        }
        if (!shape.planes.length && (shape.radiusB_m??shape.radius_m)===shape.radius_m) {
            const e = sub3(shape.b, shape.a), t = Math.max(0, Math.min(1, dot3(sub3(center, shape.a), e) / dot3(e, e))), q = add(shape.a, scale3(e, t)), delta = sub3(q, center), distance = norm3(delta), fallback = cross3(e, Math.abs(e[0]) < .8 * norm3(e) ? [1, 0, 0] : [0, 1, 0]), normal = distance > 1e-14 ? scale3(delta, 1 / distance) : scale3(fallback, 1 / norm3(fallback)), point = sub3(q, scale3(normal, shape.radius_m)), gap = distance - shape.radius_m - radius;
            return { distance_m: gap, lowerBound_m: gap, point, normal };
        }
        const result = convexClothDistance(d => supportClippedRod(shape, d), () => center, sub3(shape.a, center), 1e-11);
        return { distance_m: result.distance_m - radius, lowerBound_m: result.lowerBound_m - radius, point: result.a, normal: result.normal };
    }
    const support = colliderSupport(collider, beforeEnd);
    if (support) {
        const result = convexClothDistance(d => supportClippedRod(shape, d), support, sub3(shape.a, shape.b));
        if (result.distance_m > 1e-9)
            return { distance_m: result.distance_m, lowerBound_m: result.lowerBound_m, point: result.a, normal: result.normal };
        const point = supportClippedRod(shape, scale3(result.normal, -1)), s = sample(collider, point, beforeEnd);
        if (s.distance_m <= 1e-9)
            return { distance_m: s.distance_m, lowerBound_m: s.distance_m, point, normal: s.normal };
    }
    type Cell = {
        lo: ClothVector;
        hi: ClothVector;
        lower: number;
    };
    const heap: Cell[] = [];
    const push = (cell: Cell) => { heap.push(cell); let i = heap.length - 1; while (i) {
        const p = (i - 1) >>> 1;
        if (heap[p]!.lower <= cell.lower)
            break;
        heap[i] = heap[p]!;
        i = p;
    } heap[i] = cell; };
    const pop = () => { const first = heap[0]!, last = heap.pop()!; if (heap.length) {
        let i = 0;
        while (i * 2 + 1 < heap.length) {
            let child = i * 2 + 1;
            if (child + 1 < heap.length && heap[child + 1]!.lower < heap[child]!.lower)
                child++;
            if (heap[child]!.lower >= last.lower)
                break;
            heap[i] = heap[child]!;
            i = child;
        }
        heap[i] = last;
    } return first; };
    const lo: [
        number,
        number,
        number
    ] = [0, 0, 0], hi: [
        number,
        number,
        number
    ] = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
        const d: [
            number,
            number,
            number
        ] = [0, 0, 0];
        d[a] = 1;
        hi[a] = supportClippedRod(shape, d)[a]!;
        d[a] = -1;
        lo[a] = supportClippedRod(shape, d)[a]!;
    }
    let best = Infinity, point = shape.a, normal: ClothVector = [0, 1, 0];
    const insert = (lo: ClothVector, hi: ClothVector) => { const center: ClothVector = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], radius = .5 * norm3(sub3(hi, lo)); if (clippedRodDistanceLowerBound(shape, center) > radius)
        return; const q = projectRod(shape, center), value = sample(collider, q, beforeEnd); if (value.distance_m < best) {
        best = value.distance_m;
        point = q;
        normal = value.normal;
    } const candidate=supportClippedRod(shape,scale3(value.normal,-1)),surface=sample(collider,candidate,beforeEnd);if(surface.distance_m<best){best=surface.distance_m;point=candidate;normal=surface.normal;} const lower = Math.max(sample(collider, center, beforeEnd).distance_m - (collider.kind==="sampled"?(collider.spatialLipschitz??2):1) * radius,collider.kind==="sampled"?(collider.distanceLowerBound?.(lo,hi,beforeEnd)??-Infinity):-Infinity); if (lower < best)
        push({ lo, hi, lower }); };
    insert(lo, hi);
    for (let i = 0; i < 16384 && heap.length && best > 1e-9; i++) {
        const cell = pop();
        if (best - cell.lower < 1e-9 || cell.lower > 1e-9)
            return { distance_m: best, lowerBound_m: cell.lower, point, normal };
        let axis = 0;
        for (let a = 1; a < 3; a++)
            if (cell.hi[a]! - cell.lo[a]! > cell.hi[axis]! - cell.lo[axis]!)
                axis = a;
        const mid = (cell.lo[axis]! + cell.hi[axis]!) / 2, a = [...cell.hi] as [
            number,
            number,
            number
        ], b = [...cell.lo] as [
            number,
            number,
            number
        ];
        a[axis] = mid;
        b[axis] = mid;
        insert(cell.lo, a);
        insert(b, cell.hi);
    }
    if (best > 1e-9 && (heap[0]?.lower ?? best) <= 1e-9)
        throw new Error('Clipped rod/sampled-field contact bound did not converge');
    return { distance_m: best, lowerBound_m: Math.min(best, heap[0]?.lower ?? best), point, normal };
}
