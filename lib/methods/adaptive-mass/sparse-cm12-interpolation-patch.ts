/** A topology-only rectangular-section dual cell. No field values are cached.
 * Experimental compiler/evaluator; resident transport does not consume it yet.
 * Corner order is x + 2*y + 4*z, matching TransportStencil.
 */
export type PatchPoint = readonly [number, number, number];
export interface CM12InterpolationPatch {
  readonly axis: number;
  readonly lower: PatchPoint;
  readonly upper: PatchPoint;
  readonly lowerSpan: PatchPoint;
  readonly upperSpan: PatchPoint;
}

/** Recognizes boxes, axis-aligned wedges, pyramids and rectangular frusta.
 * Unsupported geometry is explicit: the caller must not silently substitute
 * regular-grid weights. Classification belongs to topology compilation.
 */
export function compileCM12InterpolationPatch(nodes: readonly PatchPoint[]): CM12InterpolationPatch | null {
  if (nodes.length !== 8 || nodes.some(p => p.some(v => !Number.isFinite(v)))) return null;
  for (let axis = 0; axis < 3; axis++) {
    const bit = 1 << axis, lower = nodes[0]!, upper = nodes[bit]!;
    if (upper[axis]! <= lower[axis]!) continue;
    const lowerSpan: [number, number, number] = [0, 0, 0];
    const upperSpan: [number, number, number] = [0, 0, 0];
    for (let tangent = 0; tangent < 3; tangent++) if (tangent !== axis) {
      lowerSpan[tangent] = nodes[1 << tangent]![tangent]! - lower[tangent]!;
      upperSpan[tangent] = nodes[bit | (1 << tangent)]![tangent]! - upper[tangent]!;
    }
    if (lowerSpan.some(v => v < 0) || upperSpan.some(v => v < 0)) continue;
    let exact = true;
    for (let corner = 0; corner < 8; corner++) {
      const origin = corner & bit ? upper : lower;
      const span = corner & bit ? upperSpan : lowerSpan;
      for (let k = 0; k < 3; k++) {
        if (nodes[corner]![k] !== origin[k]! + (corner & (1 << k) ? span[k]! : 0)) exact = false;
      }
    }
    if (exact) return { axis, lower: [...lower], upper: [...upper], lowerSpan, upperSpan };
  }
  return null;
}

/** Query-local weights. null means outside this patch, including outside the
 * collapsed side. Zero-width coordinates use 1/2: repeated donors aggregate
 * identically without choosing an arbitrary fine-child direction.
 */
export function evaluateCM12InterpolationPatch(p: CM12InterpolationPatch, point: PatchPoint): number[] | null {
  const t = (point[p.axis]! - p.lower[p.axis]!) / (p.upper[p.axis]! - p.lower[p.axis]!);
  if (!(t >= 0 && t <= 1)) return null;
  const local = [.5, .5, .5]; local[p.axis] = t;
  for (let k = 0; k < 3; k++) if (k !== p.axis) {
    const span = (1-t)*p.lowerSpan[k]! + t*p.upperSpan[k]!;
    const offset = point[k]! - p.lower[k]! - t*(p.upper[k]! - p.lower[k]!);
    if (span === 0) { if (offset !== 0) return null; }
    else { local[k] = offset/span; if (!(local[k]! >= 0 && local[k]! <= 1)) return null; }
  }
  return Array.from({length:8}, (_, corner) => local.reduce((w, u, axis) => w * (corner & (1 << axis) ? u : 1-u), 1));
}

/** Shared with GPU tests. Descriptor is four vec4s (64 bytes); donor IDs can
 * remain in the topology-owned stencil record. No search or Newton loop.
 */
export const CM12_INTERPOLATION_PATCH_WGSL = /* wgsl */ `
struct CM12InterpolationPatch {
  lower:vec3f, axis:u32,
  upper:vec3f, reserved:u32,
  lowerSpan:vec3f, reserved1:u32,
  upperSpan:vec3f, reserved2:u32,
}
struct CM12PatchWeights { weights:array<f32,8>, inside:bool }
fn cm12EvaluateInterpolationPatch(shape:CM12InterpolationPatch,point:vec3f)->CM12PatchWeights{
  var result:CM12PatchWeights;result.inside=false;
  let axis=shape.axis;
  let t=(point[axis]-shape.lower[axis])/(shape.upper[axis]-shape.lower[axis]);
  if(t<0.0||t>1.0){return result;}
  var local=vec3f(0.5);local[axis]=t;
  for(var k=0u;k<3u;k+=1u){if(k==axis){continue;}
    let span=(1.0-t)*shape.lowerSpan[k]+t*shape.upperSpan[k];
    let offset=(point[k]-shape.lower[k])-t*(shape.upper[k]-shape.lower[k]);
    if(span==0.0){if(offset!=0.0){return result;}}
    else{local[k]=offset/span;if(local[k]<0.0||local[k]>1.0){return result;}}
  }
  for(var corner=0u;corner<8u;corner+=1u){
    let upper=vec3<bool>((corner&1u)!=0u,(corner&2u)!=0u,(corner&4u)!=0u);
    let f=select(vec3f(1.0)-local,local,upper);
    result.weights[corner]=f.x*f.y*f.z;
  }
  result.inside=true;return result;
}
`;

export function packCM12InterpolationPatches(patches: readonly CM12InterpolationPatch[]): ArrayBuffer {
  const buffer = new ArrayBuffer(64*patches.length);
  const f = new Float32Array(buffer), u = new Uint32Array(buffer);
  patches.forEach((p, i) => {
    f.set(p.lower, 16*i); u[16*i+3] = p.axis;
    f.set(p.upper, 16*i+4); f.set(p.lowerSpan, 16*i+8); f.set(p.upperSpan, 16*i+12);
  });
  return buffer;
}
