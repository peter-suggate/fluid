import type { SolidWorldVoxelPatch } from "../../../core/solid-world";
import { solidPatchKindId, SOLID_PATCH_BOX_KIND, SOLID_PATCH_ELLIPSOID_KIND } from "../../../core/solid-patch/kind";

type V3 = readonly [number, number, number];
export interface SvoSculptLattice { readonly origin_m: V3; readonly cellSize_m: V3 }
export const SVO_SCULPT_CUT_MAXIMUM_RECORDS = 4096;
export const SVO_SCULPT_CUT_HEADER_WORDS = 4;
export const SVO_SCULPT_CUT_RECORD_WORDS = 8;
export const SVO_SCULPT_MATERIAL_OFFSET = 8192;
export const SVO_SCULPT_OWNER_MASK_WORDS = 2048;
export const SVO_SCULPT_OWNER_MASK_OFFSET_WORDS = SVO_SCULPT_CUT_HEADER_WORDS + SVO_SCULPT_CUT_MAXIMUM_RECORDS * SVO_SCULPT_CUT_RECORD_WORDS;
export const SVO_SCULPT_CUT_ARENA_WORDS = SVO_SCULPT_OWNER_MASK_OFFSET_WORDS + SVO_SCULPT_OWNER_MASK_WORDS;

/** Explicit sculpt strokes only: legacy voxel clears must not erase scenery. */
export function packSvoSculptCuts(patches: readonly SolidWorldVoxelPatch[], lattice: SvoSculptLattice): Uint32Array<ArrayBuffer> {
  const cuts = patches.filter(p => p.operation === "clear" && (p as SolidWorldVoxelPatch & { affectsScenery?: boolean }).affectsScenery === true);
  if (cuts.length > SVO_SCULPT_CUT_MAXIMUM_RECORDS) throw new RangeError(`Scenery sculpt capacity reached (${SVO_SCULPT_CUT_MAXIMUM_RECORDS} cuts); undo cuts before adding more`);
  if (!lattice.origin_m.every(Number.isFinite) || !lattice.cellSize_m.every(v => Number.isFinite(v) && v > 0)) throw new RangeError("Scenery sculpt lattice must be finite and positive");
  const words = new Uint32Array(SVO_SCULPT_CUT_HEADER_WORDS + cuts.length * SVO_SCULPT_CUT_RECORD_WORDS);
  const floats = new Float32Array(words.buffer); words[0] = cuts.length;
  cuts.forEach((patch, i) => {
    const kind = solidPatchKindId(patch);
    if (kind !== SOLID_PATCH_BOX_KIND && kind !== SOLID_PATCH_ELLIPSOID_KIND) throw new RangeError(`Scenery sculpt does not support ${kind} cuts`);
    const at = SVO_SCULPT_CUT_HEADER_WORDS + i * SVO_SCULPT_CUT_RECORD_WORDS;
    for (let a = 0; a < 3; a++) {
      const lo = lattice.origin_m[a]! + patch.minimum[a]! * lattice.cellSize_m[a]!;
      const hi = lattice.origin_m[a]! + patch.maximumExclusive[a]! * lattice.cellSize_m[a]!;
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo) || !(Math.fround(hi) > Math.fround(lo))) throw new RangeError("Scenery sculpt bounds must be finite and nonempty");
      floats[at + a] = lo; floats[at + 4 + a] = hi;
    }
    words[at + 3] = kind === SOLID_PATCH_BOX_KIND ? 1 : 2;
  });
  return words;
}

/** Conservative bounds test for replacing native imported draws with carved voxel geometry. */
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
  let distance = 1e20, normal: V3 = [0, 0, 0];
  for (let i = 0; i < words[0]!; i++) {
    const at = SVO_SCULPT_CUT_HEADER_WORDS + i * SVO_SCULPT_CUT_RECORD_WORDS;
    const r = [0, 1, 2].map(a => (f[at + 4 + a]! - f[at + a]!) * .5);
    const p = [0, 1, 2].map(a => point[a]! - (f[at + 4 + a]! + f[at + a]!) * .5);
    let d: number, n: number[];
    if (words[at + 3] === 2) {
      d = (Math.hypot(...p.map((v, a) => v / r[a]!)) - 1) * Math.min(...r);
      n = p.map((v, a) => v / (r[a]! * r[a]!));
    } else {
      const q = p.map((v, a) => Math.abs(v) - r[a]!);
      const outside = q.map(v => Math.max(v, 0)), length = Math.hypot(...outside);
      d = length + Math.min(Math.max(...q), 0);
      const axis = q[0]! >= q[1]! && q[0]! >= q[2]! ? 0 : q[1]! >= q[2]! ? 1 : 2;
      n = length > 0 ? outside.map((v, a) => v * (p[a]! < 0 ? -1 : 1))
        : p.map((v, a) => a === axis ? (v < 0 ? -1 : 1) : 0);
    }
    if (d < distance) {
      distance = d; const magnitude = Math.hypot(...n);
      normal = magnitude > 1e-12 ? n.map(v => v / magnitude) as unknown as V3 : [0, 0, 0];
    }
  }
  return { distance, normal };
}

/** `load` reads an arena-relative u32. No bindings are introduced here. */
export function svoSculptCutsWGSL(load: string): string {
  return /* wgsl */ `
struct SvoSculptCutSample { distance:f32, normal:vec3f }
fn svoSculptCutCount()->u32{return min(${load}(0u),${SVO_SCULPT_CUT_MAXIMUM_RECORDS}u);}
fn svoSculptCutValue(index:u32,world:vec3f)->SvoSculptCutSample{
  let at=4u+index*8u;
  let lo=bitcast<vec3f>(vec3u(${load}(at),${load}(at+1u),${load}(at+2u)));
  let hi=bitcast<vec3f>(vec3u(${load}(at+4u),${load}(at+5u),${load}(at+6u)));
  let r=max(.5*(hi-lo),vec3f(1e-20));let p=world-.5*(hi+lo);
  var d=0.;var normal=vec3f(0.);
  if(${load}(at+3u)==2u){
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
  if(${load}(at+3u)==2u){return (length(p/r)-1.)*min(r.x,min(r.y,r.z));}
  let q=abs(p)-r;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.);
}
fn svoSculptCutSample(world:vec3f)->SvoSculptCutSample{
  var nearest=0xffffffffu;var distance=1e20;
  for(var i=0u;i<svoSculptCutCount();i+=1u){let d=svoSculptCutRecordDistance(i,world);if(d<distance){distance=d;nearest=i;}}
  if(nearest==0xffffffffu){return SvoSculptCutSample(distance,vec3f(0));}
  return svoSculptCutValue(nearest,world);
}
fn svoSculptCutDistance(world:vec3f)->f32{
  var distance=1e20;
  for(var i=0u;i<svoSculptCutCount();i+=1u){distance=min(distance,svoSculptCutRecordDistance(i,world));}
  return distance;
}
fn svoSculptApplyDistance(distance:f32,world:vec3f)->f32{return max(distance,-svoSculptCutDistance(world));}
`;
}
