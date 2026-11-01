import { sculptCylinderDistance, sculptCylinderWGSL } from "./sculpt-cut-cylinder";
import { SCULPT_ARENA_WORDS, sculptSubstrateWGSL } from "./svo-sculpt-substrate";
import type { SolidWorldVoxelPatch } from "../../../core/solid-world";
import { ellipsoidBounds } from "../../../core/solid-patch/ellipsoid-bounds";
import { solidPatchKindId, SOLID_PATCH_BOX_KIND, SOLID_PATCH_ELLIPSOID_KIND, SOLID_PATCH_CYLINDER_KIND } from "../../../core/solid-patch/kind";

type V3 = readonly [number, number, number];
export interface SvoSculptLattice { readonly origin_m: V3; readonly cellSize_m: V3 }
export const SVO_SCULPT_CUT_MAXIMUM_RECORDS = 4096;
export const SVO_SCULPT_CUT_HEADER_WORDS = 4;
export const SVO_SCULPT_CUT_RECORD_WORDS = 8;
export const SVO_SCULPT_MATERIAL_OFFSET = 8192;
export const SVO_SCULPT_OWNER_MASK_WORDS = 2048;
export const SVO_SCULPT_OWNER_MASK_OFFSET_WORDS = SVO_SCULPT_CUT_HEADER_WORDS + SVO_SCULPT_CUT_MAXIMUM_RECORDS * SVO_SCULPT_CUT_RECORD_WORDS;
export const SVO_SCULPT_CUT_ARENA_WORDS = SCULPT_ARENA_WORDS;

// Immutable authored patch objects are shared across accepted prefixes. Packing
// their physical operands once avoids repeatedly decoding old rounded stamps.
const packedCutRecords = new WeakMap<SolidWorldVoxelPatch, Map<string, Uint32Array>>();
const packedCutHistories = new WeakMap<readonly SolidWorldVoxelPatch[], Map<string, { length: number; words: Uint32Array<ArrayBuffer> }>>();
const latestPackedCuts = new Map<string, { patches: WeakRef<readonly SolidWorldVoxelPatch[]>; length: number; words: Uint32Array<ArrayBuffer> }>();

function rememberPackedCuts(patches: readonly SolidWorldVoxelPatch[], latticeKey: string, words: Uint32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> {
  let byLattice = packedCutHistories.get(patches);
  if (!byLattice) { byLattice = new Map(); packedCutHistories.set(patches, byLattice); }
  byLattice.set(latticeKey, { length: patches.length, words });
  latestPackedCuts.delete(latticeKey);
  latestPackedCuts.set(latticeKey, { patches: new WeakRef(patches), length: patches.length, words });
  if (latestPackedCuts.size > 8) latestPackedCuts.delete(latestPackedCuts.keys().next().value!);
  return words;
}

/** Explicit sculpt strokes only: legacy voxel clears must not erase scenery.
 * This is the save/undo command stream; the GPU bakes it into sparse pages. */
export function packSvoSculptCuts(patches: readonly SolidWorldVoxelPatch[], lattice: SvoSculptLattice): Uint32Array<ArrayBuffer> {
  const latticeKey = `${lattice.origin_m.join(",")}:${lattice.cellSize_m.join(",")}`;
  const held = packedCutHistories.get(patches)?.get(latticeKey);
  if (held?.length === patches.length) return held.words;
  const previous = latestPackedCuts.get(latticeKey), previousPatches = previous?.patches.deref();
  let prefix = 0, retained = 0;
  if (previousPatches) while (prefix < Math.min(previous!.length, previousPatches.length, patches.length) && previousPatches[prefix] === patches[prefix]) {
    const patch = patches[prefix++]!; if (patch.operation === "clear" && patch.affectsScenery === true) retained++;
  }
  const cuts = patches.slice(prefix).filter(p => p.operation === "clear" && p.affectsScenery === true);
  if (!lattice.origin_m.every(Number.isFinite) || !lattice.cellSize_m.every(v => Number.isFinite(v) && v > 0)) throw new RangeError("Scenery sculpt lattice must be finite and positive");
  // Paint/fill appends and undo of those commands leave the baked cut field
  // unchanged. Preserve its identity so collision caches need no rebuild.
  if (previous && cuts.length === 0 && retained === previous.words[0]) return rememberPackedCuts(patches, latticeKey, previous.words);
  const words = new Uint32Array(SVO_SCULPT_CUT_HEADER_WORDS + (retained + cuts.length) * SVO_SCULPT_CUT_RECORD_WORDS + 4);
  const floats = new Float32Array(words.buffer); words[0] = retained + cuts.length; floats.set(lattice.cellSize_m, 1);
  // Trailer keeps the analytic record ABI intact while carrying the exact
  // voxel-grid origin across worker/publication boundaries.
  floats.set(lattice.origin_m, words.length - 4);
  if (retained) words.set(previous!.words.subarray(4, 4 + retained * 8), 4);
  cuts.forEach((patch, i) => {
    const at = SVO_SCULPT_CUT_HEADER_WORDS + (retained + i) * SVO_SCULPT_CUT_RECORD_WORDS;
    const cached = packedCutRecords.get(patch)?.get(latticeKey);
    if (cached) { words.set(cached, at); return; }
    const kind = solidPatchKindId(patch);
    if (kind !== SOLID_PATCH_BOX_KIND && kind !== SOLID_PATCH_ELLIPSOID_KIND && kind !== SOLID_PATCH_CYLINDER_KIND) throw new RangeError(`Scenery sculpt does not support ${kind} cuts`);
    const bounds = kind === SOLID_PATCH_ELLIPSOID_KIND ? ellipsoidBounds(patch) : patch;
    for (let a = 0; a < 3; a++) {
      const lo = lattice.origin_m[a]! + bounds.minimum[a]! * lattice.cellSize_m[a]!;
      const hi = lattice.origin_m[a]! + bounds.maximumExclusive[a]! * lattice.cellSize_m[a]!;
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo) || !(Math.fround(hi) > Math.fround(lo))) throw new RangeError("Scenery sculpt bounds must be finite and nonempty");
      floats[at + a] = lo; floats[at + 4 + a] = hi;
    }
    if (kind === SOLID_PATCH_CYLINDER_KIND && ![0, 1, 2].includes(patch.axis ?? 1)) throw new RangeError("Invalid sculpt cylinder axis");
    words[at + 3] = kind === SOLID_PATCH_BOX_KIND ? 1 : kind === SOLID_PATCH_ELLIPSOID_KIND ? 2 : 3 + (patch.axis ?? 1);
    let byLattice = packedCutRecords.get(patch);
    if (!byLattice) { byLattice = new Map(); packedCutRecords.set(patch, byLattice); }
    byLattice.set(latticeKey, words.slice(at, at + SVO_SCULPT_CUT_RECORD_WORDS));
  });
  return rememberPackedCuts(patches, latticeKey, words);
}

/** Conservative bounds test for source instances affected by a carve. */
export function intersectsSvoSculptCuts(words: Uint32Array, minimum: V3, maximum: V3): boolean {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  for (let i = 0; i < words[0]!; i++) {
    const at = SVO_SCULPT_CUT_HEADER_WORDS + i * SVO_SCULPT_CUT_RECORD_WORDS;
    if ([0, 1, 2].every(a => maximum[a]! >= f[at + a]! && minimum[a]! <= f[at + 4 + a]!)) return true;
  }
  return false;
}

/** Shared native-instance bounds for publication, invalidation and voxelization.
 * Rotating the local box is substantially tighter than a sphere for flat paving
 * and tall foliage: a ground stroke must not withdraw a distant canopy's native
 * mesh. Float32 inputs and padding enclose the packed GPU primitive as well. */
export function intersectsSvoSculptPrimitiveBounds(words: Uint32Array, center: V3, localExtent: V3,
  orientation?: readonly [number, number, number, number]): boolean {
  const c = center.map(Math.fround), e = localExtent.map(Math.fround);
  const q = orientation ?? [0, 0, 0, 1];
  const magnitude = Math.hypot(...q);
  if (!(magnitude > 1e-12) || !Number.isFinite(magnitude)) throw new RangeError("Scenery sculpt orientation must be finite and nonzero");
  const [x, y, z, w] = q.map(v => Math.fround(v / magnitude)) as [number, number, number, number];
  const extent = [
    Math.abs(1 - 2 * (y * y + z * z)) * e[0]! + Math.abs(2 * (x * y - z * w)) * e[1]! + Math.abs(2 * (x * z + y * w)) * e[2]!,
    Math.abs(2 * (x * y + z * w)) * e[0]! + Math.abs(1 - 2 * (x * x + z * z)) * e[1]! + Math.abs(2 * (y * z - x * w)) * e[2]!,
    Math.abs(2 * (x * z - y * w)) * e[0]! + Math.abs(2 * (y * z + x * w)) * e[1]! + Math.abs(1 - 2 * (x * x + y * y)) * e[2]!,
  ];
  const padding = Math.max(1e-5, Math.hypot(...e) * 1e-5);
  return intersectsSvoSculptCuts(words, c.map((v, a) => v - extent[a]! - padding) as unknown as V3,
    c.map((v, a) => v + extent[a]! + padding) as unknown as V3);
}

/** Distance to the cut union; its negation is the retained-solid field. */
export function sampleSvoSculptCuts(words: Uint32Array, point: V3): { distance: number; normal: V3 } {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  let distance = 1e20, nearest = -1;
  // Only the winning cut needs a normal. Avoid array allocation and normal
  // normalization for every historical stamp at every accepted ray hit.
  for (let i = 0; i < words[0]!; i++) {
    const at = SVO_SCULPT_CUT_HEADER_WORDS + i * SVO_SCULPT_CUT_RECORD_WORDS;
    const rx = (f[at + 4]! - f[at]!) * .5, ry = (f[at + 5]! - f[at + 1]!) * .5, rz = (f[at + 6]! - f[at + 2]!) * .5;
    const x = point[0] - (f[at + 4]! + f[at]!) * .5;
    const y = point[1] - (f[at + 5]! + f[at + 1]!) * .5;
    const z = point[2] - (f[at + 6]! + f[at + 2]!) * .5;
    let d: number;
    if (words[at + 3]! >= 3) d = sculptCylinderDistance(words[at + 3]!, [x, y, z], [rx, ry, rz]);
    else if (words[at + 3] === 2) d = (Math.hypot(x / rx, y / ry, z / rz) - 1) * Math.min(rx, ry, rz);
    else {
      const qx = Math.abs(x) - rx, qy = Math.abs(y) - ry, qz = Math.abs(z) - rz;
      d = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
    }
    if (d < distance) { distance = d; nearest = at; }
  }
  if (nearest < 0) return { distance, normal: [0, 0, 0] };
  const r = [0, 1, 2].map(a => (f[nearest + 4 + a]! - f[nearest + a]!) * .5);
  const p = [0, 1, 2].map(a => point[a]! - (f[nearest + 4 + a]! + f[nearest + a]!) * .5);
  let n: number[];
  if (words[nearest + 3]! >= 3) n = p.map((_, a) => {
    const lo = [...p], hi = [...p]; lo[a]! -= 1e-5; hi[a]! += 1e-5;
    return sculptCylinderDistance(words[nearest + 3]!, hi, r) - sculptCylinderDistance(words[nearest + 3]!, lo, r);
  });
  else if (words[nearest + 3] === 2) n = p.map((v, a) => v / (r[a]! * r[a]!));
  else {
    const q = p.map((v, a) => Math.abs(v) - r[a]!);
    const outside = q.map(v => Math.max(v, 0));
    const axis = q[0]! >= q[1]! && q[0]! >= q[2]! ? 0 : q[1]! >= q[2]! ? 1 : 2;
    n = Math.hypot(...outside) > 0 ? outside.map((v, a) => v * (p[a]! < 0 ? -1 : 1))
      : p.map((v, a) => a === axis ? (v < 0 ? -1 : 1) : 0);
  }
  const magnitude = Math.hypot(...n);
  return { distance, normal: magnitude > 1e-12 ? n.map(v => v / magnitude) as unknown as V3 : [0, 0, 0] };
}

/** `load` reads an arena-relative u32. No bindings are introduced here. */
export function svoSculptCutsWGSL(load: string): string {
  return /* wgsl */ `
${sculptCylinderWGSL}
struct SvoSculptCutSample { distance:f32, normal:vec3f }
${sculptSubstrateWGSL(load)}
fn svoSculptCutCount()->u32{return min(${load}(0u),${SVO_SCULPT_CUT_MAXIMUM_RECORDS}u);}
fn svoSculptCutValue(index:u32,world:vec3f)->SvoSculptCutSample{
  let at=4u+index*8u;
  let lo=bitcast<vec3f>(vec3u(${load}(at),${load}(at+1u),${load}(at+2u)));
  let hi=bitcast<vec3f>(vec3u(${load}(at+4u),${load}(at+5u),${load}(at+6u)));
  let r=max(.5*(hi-lo),vec3f(1e-20));let p=world-.5*(hi+lo);
  var d=0.;var normal=vec3f(0.);
  if(${load}(at+3u)>=3u){
    let kind=${load}(at+3u);d=sculptCylinderDistance(kind,p,r);
    for(var a=0u;a<3u;a++){var offset=vec3f(0);offset[a]=1e-5;
      normal[a]=sculptCylinderDistance(kind,p+offset,r)-sculptCylinderDistance(kind,p-offset,r);}
  }else if(${load}(at+3u)==2u){
    d=(length(p/r)-1.)*min(r.x,min(r.y,r.z));normal=p/(r*r);
  }else{
    let q=abs(p)-r;let outside=max(q,vec3f(0));let magnitude=length(outside);
    d=magnitude+min(max(q.x,max(q.y,q.z)),0.);
    let direction=select(vec3f(1),vec3f(-1),p<vec3f(0));
    if(magnitude>0.){normal=outside*direction;}else{
      var axis=0u;if(q.y>q.x){axis=1u;}if(q.z>q[axis]){axis=2u;}normal[axis]=direction[axis];
    }
  }
  let magnitude=length(normal);if(magnitude>1e-12){normal/=magnitude;}
  return SvoSculptCutSample(d,normal);
}
fn svoSculptCutRecordDistance(index:u32,world:vec3f)->f32{
  let at=4u+index*8u;
  let lo=bitcast<vec3f>(vec3u(${load}(at),${load}(at+1u),${load}(at+2u)));
  let hi=bitcast<vec3f>(vec3u(${load}(at+4u),${load}(at+5u),${load}(at+6u)));
  let r=max(.5*(hi-lo),vec3f(1e-20));let p=world-.5*(hi+lo);
  if(${load}(at+3u)>=3u){return sculptCylinderDistance(${load}(at+3u),p,r);}
  if(${load}(at+3u)==2u){return (length(p/r)-1.)*min(r.x,min(r.y,r.z));}
  let q=abs(p)-r;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.);
}
fn svoSculptCutSample(world:vec3f)->SvoSculptCutSample{
  if(svoSculptCutCount()==0u){return SvoSculptCutSample(1e20,vec3f(0));}
  if(svoSculptBaked()){return sculptFieldSample(world);}
  var nearest=0xffffffffu;var distance=1e20;
  for(var i=0u;i<svoSculptCutCount();i+=1u){let d=svoSculptCutRecordDistance(i,world);if(d<distance){distance=d;nearest=i;}}
  if(nearest==0xffffffffu){return SvoSculptCutSample(distance,vec3f(0));}
  return svoSculptCutValue(nearest,world);
}
fn svoSculptCutDistance(world:vec3f)->f32{
  if(svoSculptCutCount()==0u){return 1e20;}
  if(svoSculptBaked()){return sculptFieldTrilinear(world).distance;}
  var distance=1e20;
  for(var i=0u;i<svoSculptCutCount();i+=1u){distance=min(distance,svoSculptCutRecordDistance(i,world));}
  return distance;
}
fn svoSculptApplyDistance(distance:f32,world:vec3f)->f32{return max(distance,-svoSculptCutDistance(world));}
`;
}
