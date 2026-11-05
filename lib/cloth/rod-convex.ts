import type { ClothVector } from './types';
import type { ClothRodCapsuleGeometry, ClothRodClipPlane } from './rod-geometry';
import { clothRodPlaneContains,clothRodEllipsoidSupportOffset,clothRodLineRoots } from './rod-geometry';
import { cross3, dot3, norm3, scale3, sub3 } from './topology';
const add = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mix = (a: ClothVector, b: ClothVector, t: number): ClothVector => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function capsuleSupport(shape: ClothRodCapsuleGeometry, direction: ClothVector): ClothVector { const length = norm3(direction); if (length < 1e-30)
    return mix(shape.a, shape.b, .5); const endpoint = dot3(direction, sub3(shape.b, shape.a)) >= 0 ? shape.b : shape.a; return add(endpoint, clothRodEllipsoidSupportOffset(shape,direction)); }
function clippedSupport(shape: ClothRodCapsuleGeometry, direction: ClothVector, plane: ClothRodClipPlane): ClothVector {
    let lower = 0, upper = Math.max(1, norm3(direction)), a = capsuleSupport(shape, direction);
    if (dot3(plane.normal, a) <= plane.offset_m)
        return a;
    let b = capsuleSupport(shape, sub3(direction, scale3(plane.normal, upper)));
    for (let i = 0; i < 80 && dot3(plane.normal, b) > plane.offset_m; i++) {
        upper *= 2;
        b = capsuleSupport(shape, sub3(direction, scale3(plane.normal, upper)));
    }
    for (let i = 0; i < 80; i++) {
        const mid = (lower + upper) / 2, p = capsuleSupport(shape, sub3(direction, scale3(plane.normal, mid)));
        if (dot3(plane.normal, p) > plane.offset_m) {
            lower = mid;
            a = p;
        }
        else {
            upper = mid;
            b = p;
        }
    }
    const fa = dot3(plane.normal, a) - plane.offset_m, fb = dot3(plane.normal, b) - plane.offset_m;
    return Math.abs(fa - fb) > 1e-30 ? mix(a, b, Math.max(0, Math.min(1, fa / (fa - fb)))) : b;
}
function lineCapsuleRoots(shape:ClothRodCapsuleGeometry,origin:ClothVector,direction:ClothVector):number[]{return clothRodLineRoots(shape,origin,direction);}
/** Convex support of a capsule intersected with at most two material end
 * planes. Active-set cases are the original surface, one cap, or their line. */
export function supportClippedRod(shape: ClothRodCapsuleGeometry, direction: ClothVector): ClothVector {
    if (!shape.planes.length)
        return capsuleSupport(shape, direction);
    const candidates = [capsuleSupport(shape, direction), ...shape.planes.map(p => clippedSupport(shape, direction, p))];
    if (shape.planes.length === 2) {
        const [p, q] = shape.planes, d = cross3(p!.normal, q!.normal), d2 = dot3(d, d);
        if (d2 > 1e-20) {
            const origin = scale3(add(scale3(cross3(q!.normal, d), p!.offset_m), scale3(cross3(d, p!.normal), q!.offset_m)), 1 / d2), tangent = scale3(d, 1 / Math.sqrt(d2));
            for (const t of lineCapsuleRoots(shape, origin, tangent))
                candidates.push(add(origin, scale3(tangent, t)));
        }
    }
    const valid = candidates.filter(p => clothRodPlaneContains(shape.planes, p, 1e-8));
    if (!valid.length)
        throw new Error('Clipped rod has no convex support');
    return valid.reduce((best, p) => dot3(p, direction) > dot3(best, direction) ? p : best);
}
export interface ConvexSupportPoint {
    readonly a: ClothVector;
    readonly b: ClothVector;
    readonly w: ClothVector;
}
export interface ConvexDistance {
    readonly distance_m: number;
    readonly lowerBound_m: number;
    readonly a: ClothVector;
    readonly b: ClothVector;
    readonly normal: ClothVector;
    readonly converged: boolean;
}
function linearSolve(matrix: number[][], rhs: number[]): number[] | undefined { const n = rhs.length, a = matrix.map((row, i) => [...row, rhs[i]!]); for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++)
        if (Math.abs(a[i]![k]!) > Math.abs(a[pivot]![k]!))
            pivot = i;
    if (Math.abs(a[pivot]![k]!) < 1e-25)
        return undefined;
    [a[k], a[pivot]] = [a[pivot]!, a[k]!];
    const scale = a[k]![k]!;
    for (let j = k; j <= n; j++)
        a[k]![j]! /= scale;
    for (let i = 0; i < n; i++)
        if (i !== k) {
            const scale = a[i]![k]!;
            for (let j = k; j <= n; j++)
                a[i]![j]! -= scale * a[k]![j]!;
        }
} return a.map(row => row[n]!); }
function closestSimplex(simplex: ConvexSupportPoint[]) {
    let best = Infinity, result: {
        points: ConvexSupportPoint[];
        weights: number[];
        point: ClothVector;
    } | undefined;
    for (let mask = 1; mask < (1 << simplex.length); mask++) {
        const points = simplex.filter((_, i) => mask & (1 << i));
        if (points.length > 4)
            continue;
        const base = points[0]!.w, edges = points.slice(1).map(p => sub3(p.w, base));
        let weights = [1];
        if (edges.length) {
            const coefficients = linearSolve(edges.map(a => edges.map(b => dot3(a, b))), edges.map(a => -dot3(a, base)));
            if (!coefficients)
                continue;
            weights = [1 - coefficients.reduce((a, b) => a + b, 0), ...coefficients];
        }
        if (weights.some(w => w < -1e-10))
            continue;
        const point: [
            number,
            number,
            number
        ] = [0, 0, 0];
        for (let j = 0; j < points.length; j++)
            for (let a = 0; a < 3; a++)
                point[a]! += weights[j]! * points[j]!.w[a]!;
        const distance = dot3(point, point);
        if (distance < best) {
            best = distance;
            result = { points, weights, point };
        }
    }
    if (!result)
        throw new Error('Degenerate convex contact simplex');
    return result;
}
/** GJK distance with witness points and a separating-plane lower bound for
 * conservative advancement. No mesh tessellation or rounded replacement. */
export function convexClothDistance(supportA: (d: ClothVector) => ClothVector, supportB: (d: ClothVector) => ClothVector, initial: ClothVector = [1, 0, 0], tolerance = 1e-9): ConvexDistance {
    const support = (d: ClothVector): ConvexSupportPoint => { const a = supportA(d), b = supportB(scale3(d, -1)); return { a, b, w: sub3(a, b) }; };
    let simplex = [support(norm3(initial) > 1e-20 ? initial : [1, 0, 0])], closest = closestSimplex(simplex), normal: ClothVector = [1, 0, 0], lower = 0, converged = false;
    for (let i = 0; i < 80; i++) {
        const distance = norm3(closest.point);
        if (distance < tolerance) {
            converged = true;
            break;
        }
        normal = scale3(closest.point, 1 / distance);
        const next = support(scale3(normal, -1));
        lower = Math.max(0, dot3(normal, next.w));
        if (distance - lower < tolerance) {
            converged = true;
            break;
        }
        simplex = closest.points.filter((_, j) => closest.weights[j]! > 1e-12);
        if (simplex.some(p => norm3(sub3(p.w, next.w)) < 1e-14)) {
            converged = distance - lower < tolerance * 10;
            break;
        }
        simplex.push(next);
        closest = closestSimplex(simplex);
    }
    const a: [
        number,
        number,
        number
    ] = [0, 0, 0], b: [
        number,
        number,
        number
    ] = [0, 0, 0];
    for (let j = 0; j < closest.points.length; j++)
        for (let k = 0; k < 3; k++) {
            a[k]! += closest.weights[j]! * closest.points[j]!.a[k]!;
            b[k]! += closest.weights[j]! * closest.points[j]!.b[k]!;
        }
    return { distance_m: norm3(closest.point), lowerBound_m: lower, a, b, normal, converged };
}
