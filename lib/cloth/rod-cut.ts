import type { ClothCutBox } from './edit';
import type { ClothVector } from './types';
import {clothRodEllipsoidAxes,type ClothRodCapsuleGeometry} from './rod-geometry';
import {convexClothDistance,supportClippedRod} from './rod-convex';
export interface RodBoxRetainedInterval {
    readonly start: number;
    readonly end: number;
    readonly startNormal?: ClothVector;
    readonly endNormal?: ClothVector;
}
function segmentBoxDistance(a: ClothVector, b: ClothVector, box: ClothCutBox): number {
    const delta = b.map((v, i) => v - a[i]!), cuts = [0, 1];
    for (let k = 0; k < 3; k++)
        if (Math.abs(delta[k]!) > 1e-20)
            for (const plane of [box.minimum_m[k]!, box.maximum_m[k]!]) {
                const t = (plane - a[k]!) / delta[k]!;
                if (t > 0 && t < 1)
                    cuts.push(t);
            }
    cuts.sort((a, b) => a - b);
    const distance = (t: number) => Math.hypot(...a.map((v, k) => { const p = v + t * delta[k]!; return Math.max(box.minimum_m[k]! - p, 0, p - box.maximum_m[k]!); }));
    let best = Infinity;
    for (let i = 0; i < cuts.length - 1; i++) {
        const lo = cuts[i]!, hi = cuts[i + 1]!, mid = (lo + hi) / 2;
        let aa = 0, bb = 0;
        for (let k = 0; k < 3; k++) {
            const p = a[k]! + mid * delta[k]!;
            const boundary = p < box.minimum_m[k]! ? box.minimum_m[k]! : p > box.maximum_m[k]! ? box.maximum_m[k]! : undefined;
            if (boundary !== undefined) {
                aa += delta[k]! ** 2;
                bb += delta[k]! * (a[k]! - boundary);
            }
        }
        best = Math.min(best, distance(lo), distance(hi), distance(aa ? Math.max(lo, Math.min(hi, -bb / aa)) : mid));
    }
    return best;
}
/** Exact through-section cuts. Grazing removes a non-codimensional fragment;
 * it must use the later analytic CSG/section representation, never radius loss. */
export function clipRodThroughBox(a: ClothVector, b: ClothVector, radius: number, box: ClothCutBox, geometry?:ClothRodCapsuleGeometry): readonly RodBoxRetainedInterval[] {
    const delta = b.map((v, i) => v - a[i]!), length = Math.hypot(...delta), tangent = delta.map(v => v / length);
    const shape=geometry??{a,b,radius_m:radius,planes:[]},axes=clothRodEllipsoidAxes(shape);
    const intersects=()=>{
      if((shape.radiusB_m??radius)===radius)return segmentBoxDistance(a,b,box)<radius-1e-12;
      const distance=convexClothDistance(d=>supportClippedRod(shape,d),d=>d.map((v,k)=>v>=0?box.maximum_m[k]!:box.minimum_m[k]!) as unknown as ClothVector,[1,0,0]);
      if(!distance.converged)throw new Error('Elliptical yarn/box intersection did not converge');
      return distance.distance_m<1e-10;
    };
    let lo = -Infinity, hi = Infinity, enterAxis = -1, exitAxis = -1;
    for (let k = 0; k < 3; k++) {
        const d = delta[k]!;
        if (Math.abs(d) < 1e-20) {
            if (a[k]! < box.minimum_m[k]! || a[k]! > box.maximum_m[k]!) {
                if (intersects())
                    throw new Error('Grazing yarn cuts require analytic section CSG');
                return [{ start: 0, end: 1 }];
            }
            continue;
        }
        let t0 = (box.minimum_m[k]! - a[k]!) / d, t1 = (box.maximum_m[k]! - a[k]!) / d;
        if (t0 > t1)
            [t0, t1] = [t1, t0];
        if (t0 > lo) {
            lo = t0;
            enterAxis = k;
        }
        if (t1 < hi) {
            hi = t1;
            exitAxis = k;
        }
    }
    if (lo > hi || hi < 0 || lo > 1) {
        if (intersects())
            throw new Error('Grazing yarn cuts require analytic section CSG');
        return [{ start: 0, end: 1 }];
    }
    const fullSection = (t: number, axis: number) => { if (axis < 0)
        return false; for (let k = 0; k < 3; k++) {
        if (k === axis)
            continue;
        const p = a[k]! + t * delta[k]!, extent = Math.hypot(axes.radii[0]*(axes.d1[k]!-tangent[k]!/tangent[axis]!*axes.d1[axis]!),axes.radii[1]*(axes.d2[k]!-tangent[k]!/tangent[axis]!*axes.d2[axis]!));
        if (p - extent < box.minimum_m[k]! - 1e-10 || p + extent > box.maximum_m[k]! + 1e-10)
            return false;
    } return true; };
    const sphereInside = (p: ClothVector) => p.every((v, k) => {const extent=Math.hypot(axes.radii[0]*axes.d1[k]!,axes.radii[1]*axes.d2[k]!,axes.radii[2]*axes.t[k]!);return v-extent>=box.minimum_m[k]!-1e-10&&v+extent<=box.maximum_m[k]!+1e-10;});
    if ((lo >= 0 ? !fullSection(lo, enterAxis) : !sphereInside(a)) || (hi <= 1 ? !fullSection(hi, exitAxis) : !sphereInside(b)))
        throw new Error('Grazing yarn cuts require analytic section CSG');
    const normal = (axis: number, sign: number): ClothVector => [0, 1, 2].map(k => k === axis ? sign : 0) as unknown as ClothVector, result: RodBoxRetainedInterval[] = [];
    if (lo > 1e-12)
        result.push({ start: 0, end: Math.min(1, lo), endNormal: normal(enterAxis, Math.sign(delta[enterAxis]!)) });
    if (hi < 1 - 1e-12)
        result.push({ start: Math.max(0, hi), end: 1, startNormal: normal(exitAxis, -Math.sign(delta[exitAxis]!)) });
    return result;
}
