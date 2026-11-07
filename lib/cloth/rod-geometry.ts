import type { ClothDocument, ClothSurfacePublication, ClothTopologyData, ClothVector } from './types';
import { cross3, dot3, norm3, point3, scale3, sub3, transportDirector } from './topology';
export interface ClothRodClipPlane {
    readonly normal: ClothVector;
    readonly offset_m: number;
}
export interface ClothRodCapsuleGeometry {
    readonly a: ClothVector;
    readonly b: ClothVector;
    readonly radius_m: number;
    /** Optional second material radius and frame; omitted is the circular ABI. */
    readonly radiusB_m?: number;
    readonly director1?: ClothVector;
    readonly director2?: ClothVector;
    readonly planes: readonly ClothRodClipPlane[];
}
export type ClothRodFramePublication = Pick<ClothSurfacePublication, 'rodTwist_rad' | 'rodDirectors' | 'rodTangents'>;
const references = new WeakMap<ClothTopologyData, {
    directors: Float64Array;
    tangents: Float64Array;
}>();
function referenceFrames(data: ClothTopologyData) {
    let result = references.get(data);
    if (result)
        return result;
    const count = data.rodIndices.length - data.rodOffsets.length + 1, directors = new Float64Array(count * 3), tangents = new Float64Array(count * 3);
    for (let r = 0; r < data.rodOffsets.length - 1; r++) {
        let priorT: ClothVector = [1, 0, 0], priorD: ClothVector = [0, 1, 0];
        for (let j = data.rodOffsets[r]!; j < data.rodOffsets[r + 1]! - 1; j++) {
            const s = j - r, e = sub3(point3(data.restPositions_m, data.rodIndices[j + 1]!), point3(data.restPositions_m, data.rodIndices[j]!)), t = scale3(e, 1 / norm3(e)), d = transportDirector(priorD, priorT, t);
            directors.set(d, 3 * s);
            tangents.set(t, 3 * s);
            priorT = t;
            priorD = d;
        }
    }
    if (data.restRodDirectors)
        directors.set(data.restRodDirectors);
    result = { directors, tangents };
    references.set(data, result);
    return result;
}
export function clothRodMaterialFrame(data: ClothTopologyData, positions: ArrayLike<number>, rod: number, segment: number, publication?: ClothRodFramePublication) {
    const j = data.rodOffsets[rod]! + segment, s = j - rod, a = point3(positions, data.rodIndices[j]!), b = point3(positions, data.rodIndices[j + 1]!), edge = sub3(b, a), t = scale3(edge, 1 / norm3(edge)), defaults = referenceFrames(data);
    const director = point3(publication?.rodDirectors ?? data.initialRodDirectors ?? defaults.directors, s), priorT = point3(publication?.rodTangents ?? data.initialRodTangents ?? defaults.tangents, s), d = transportDirector(director, priorT, t), theta = publication?.rodTwist_rad?.[s] ?? data.initialTwist_rad?.[s] ?? 0, q = cross3(t, d), c = Math.cos(theta), sin = Math.sin(theta), m1: ClothVector = [c * d[0] + sin * q[0], c * d[1] + sin * q[1], c * d[2] + sin * q[2]];
    return { a, b, t, d1: m1, d2: cross3(t, m1), segment: s };
}
/** Exact through-cut cylinder ends. Extend into the removed region before
 * clipping so an oblique plane does not round away untouched tube material. */
export function clothRodCapsuleGeometry(document: Pick<ClothDocument, 'materials'>, data: ClothTopologyData, positions: ArrayLike<number>, rod: number, segment: number, publication?: ClothRodFramePublication): ClothRodCapsuleGeometry {
    const frame = clothRodMaterialFrame(data, positions, rod, segment, publication), material=document.materials[data.rodMaterials?.[rod] ?? 0]!;
    const clip = clothRodSegmentClipNormals(data, rod, segment);
    return clothRodClippedCapsule(frame, material.rodRadius_m, material.rodRadiusB_m ?? material.rodRadius_m, clip.start, clip.end);
}
/** A segment's material-frame end-plane normals (d1, d2, t components): the
 * rod's start plane on its first segment, its end plane on its last, each
 * undefined where that end is uncut. */
export function clothRodSegmentClipNormals(data: ClothTopologyData, rod: number, segment: number): {start?: ClothVector; end?: ClothVector} {
    const last = data.rodOffsets[rod + 1]! - data.rodOffsets[rod]! - 2, pick = (normals: readonly number[] | undefined, on: boolean) => {if (!normals || !on) return undefined; const local = point3(normals, rod); return norm3(local) === 0 ? undefined : local;};
    return {start: pick(data.rodStartClipNormals, segment === 0), end: pick(data.rodEndClipNormals, segment === last)};
}
/** The clipped volume of a segment from its ends and material frame (t from a
 * to b, d1, d2) and its local end-plane normals: the ellipsoid-swept segment,
 * extended past each cut end into the removed region, intersected with each
 * end's half-space through that end vertex. */
export function clothRodClippedCapsule(frame: {readonly a: ClothVector; readonly b: ClothVector; readonly t: ClothVector; readonly d1: ClothVector; readonly d2: ClothVector}, radius: number, radiusB: number, start?: ClothVector, end?: ClothVector): ClothRodCapsuleGeometry {
    const planes: ClothRodClipPlane[] = [];
    let a = frame.a, b = frame.b;
    for (const [isStart, local] of [[true, start], [false, end]] as const) {
        if (!local)
            continue;
        const normal: ClothVector = [local[0] * frame.d1[0] + local[1] * frame.d2[0] + local[2] * frame.t[0], local[0] * frame.d1[1] + local[1] * frame.d2[1] + local[2] * frame.t[1], local[0] * frame.d1[2] + local[1] * frame.d2[2] + local[2] * frame.t[2]], axial = dot3(normal, frame.t), anchor = isStart ? frame.a : frame.b;
        if ((isStart ? -axial : axial) < 1e-8)
            throw new Error('Rod end clipping plane must point out of its segment');
        const extension = Math.max(radius,radiusB) / Math.abs(axial), e = scale3(frame.t, isStart ? -extension : extension);
        if (isStart)
            a = [a[0] + e[0], a[1] + e[1], a[2] + e[2]];
        else
            b = [b[0] + e[0], b[1] + e[1], b[2] + e[2]];
        planes.push({ normal, offset_m: dot3(normal, anchor) });
    }
    return { a, b, radius_m: radius,radiusB_m:radiusB,director1:frame.d1,director2:frame.d2, planes };
}
export function clothRodPlaneContains(planes: readonly ClothRodClipPlane[], point: ClothVector, tolerance = 1e-10): boolean { return planes.every(plane => dot3(plane.normal, point) <= plane.offset_m + tolerance); }

/** Explicit shared closure adaptation: segment plus material-axis ellipsoid.
 * Axial radius sqrt(a*b) preserves the circular capsule limit. The released
 * OpenGL line preview defines no volumetric endpoint-cap law. */
export function clothRodEllipsoidAxes(shape:ClothRodCapsuleGeometry){
 const edge=sub3(shape.b,shape.a),length=norm3(edge),t=length>1e-20?scale3(edge,1/length):[0,0,1] as ClothVector;
 const d1=shape.director1??transportDirector([0,1,0],[1,0,0],t),d2=shape.director2??cross3(t,d1);
 const a=shape.radius_m,b=shape.radiusB_m??a,c=Math.sqrt(a*b);
 return {d1,d2,t,radii:[a,b,c] as ClothVector,length};
}
export function clothRodBoundingRadius(shape:ClothRodCapsuleGeometry):number{return Math.max(shape.radius_m,shape.radiusB_m??shape.radius_m);}
export function clothRodEllipsoidSupportOffset(shape:ClothRodCapsuleGeometry,direction:ClothVector):ClothVector{
 const {d1,d2,t,radii}=clothRodEllipsoidAxes(shape),basis=[d1,d2,t],q=basis.map((v,i)=>dot3(v,direction)*radii[i]!),length=Math.hypot(...q);
 if(length<1e-30)return [0,0,0];
 return [0,1,2].map(k=>basis.reduce((sum,v,i)=>sum+v[k]!*radii[i]!*q[i]!/length,0)) as unknown as ClothVector;
}
export function clothRodLocalVector(shape:ClothRodCapsuleGeometry,v:ClothVector):ClothVector{
 const {d1,d2,t,radii}=clothRodEllipsoidAxes(shape);return [dot3(v,d1)/radii[0],dot3(v,d2)/radii[1],dot3(v,t)/radii[2]];
}
export function clothRodNormal(shape:ClothRodCapsuleGeometry,p:ClothVector):ClothVector{
 const {d1,d2,t,radii,length}=clothRodEllipsoidAxes(shape),q=sub3(p,shape.a),z=Math.max(0,Math.min(length,dot3(q,t))),v=sub3(q,scale3(t,z)),basis=[d1,d2,t];
 const n=[0,1,2].map(k=>basis.reduce((sum,b,i)=>sum+b[k]!*dot3(v,b)/radii[i]!**2,0)) as unknown as ClothVector,l=norm3(n);return l?scale3(n,1/l):d1;
}
export function clothRodContains(shape:ClothRodCapsuleGeometry,p:ClothVector,tolerance=1e-10):boolean{
 const {length,radii}=clothRodEllipsoidAxes(shape),q=clothRodLocalVector(shape,sub3(p,shape.a)),z=q[2]-Math.max(0,Math.min(length/radii[2],q[2]));
 return Math.hypot(q[0],q[1],z)<=1+tolerance/Math.min(...radii)&&clothRodPlaneContains(shape.planes,p,tolerance);
}
/** All uncut line/surface roots, retaining the caller's parameterization. */
export function clothRodLineRoots(shape:ClothRodCapsuleGeometry,origin:ClothVector,direction:ClothVector):number[]{
 const {length,radii}=clothRodEllipsoidAxes(shape),L=length/radii[2],o=clothRodLocalVector(shape,sub3(origin,shape.a)),d=clothRodLocalVector(shape,direction),roots:number[]=[];
 const quadratic=(a:number,b:number,c:number,accept:(t:number)=>boolean)=>{const h=b*b-a*c;if(a>1e-30&&h>=0)for(const t of [(-b-Math.sqrt(h))/a,(-b+Math.sqrt(h))/a])if(accept(t))roots.push(t);};
 quadratic(d[0]**2+d[1]**2,o[0]*d[0]+o[1]*d[1],o[0]**2+o[1]**2-1,t=>o[2]+t*d[2]>=0&&o[2]+t*d[2]<=L);
 for(const cap of [0,L]){const z=o[2]-cap;quadratic(dot3(d,d),o[0]*d[0]+o[1]*d[1]+z*d[2],o[0]**2+o[1]**2+z*z-1,t=>cap===0?o[2]+t*d[2]<=0:o[2]+t*d[2]>=L);}
 return roots;
}
