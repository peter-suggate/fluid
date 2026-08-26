/**
 * Canonical finite planar-boundary geometry shared by sparse rendering and CM12.
 *
 * A patch is a closed oriented slab rather than an infinitesimally thin plane.
 * Keeping the authored half-thickness is what lets a coarse spatial terminal
 * retain the silhouette and contact location of the finest voxel description.
 */

export type PlanarBoundaryVec3 = readonly [number, number, number];

export interface PlanarBoundaryPatch {
  readonly center_m: PlanarBoundaryVec3;
  /** Unit outward normal; also the slab's thin axis. */
  readonly normal: PlanarBoundaryVec3;
  /** Unit in-plane axis. */
  readonly tangentU: PlanarBoundaryVec3;
  /** Unit in-plane axis; normal x tangentU, modulo authored handedness. */
  readonly tangentV: PlanarBoundaryVec3;
  readonly halfExtentU_m: number;
  readonly halfExtentV_m: number;
  readonly halfThickness_m: number;
  readonly materialId: number;
  readonly ownerId: number;
}

export interface PlanarBoundaryRayHit {
  /** Nearest physical slab face inside the requested ray interval. */
  readonly tHit_m: number;
  readonly tEnter_m: number;
  readonly tExit_m: number;
  readonly normal: PlanarBoundaryVec3;
  /** 0/1 are the finite in-plane edges; 2 is either broad slab face. */
  readonly featureAxis: 0 | 1 | 2;
}

/** Four vec4 words: center/thickness, normal/U extent, U/V extent, V/identity. */
export const PLANAR_BOUNDARY_PATCH_WORDS = 16;
export const PLANAR_BOUNDARY_PATCH_BYTES = PLANAR_BOUNDARY_PATCH_WORDS * 4;
/** The next-smallest extent must decisively exceed slab thickness. */
export const PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO = 8;

const dot = (a: PlanarBoundaryVec3, b: PlanarBoundaryVec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const length = (value: PlanarBoundaryVec3): number => Math.hypot(...value);

function finiteVec3(value: PlanarBoundaryVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

/** Validate the exact orthonormal frame expected by both CPU and GPU readers. */
export function validatePlanarBoundaryPatch(patch: PlanarBoundaryPatch): void {
  finiteVec3(patch.center_m, "Planar boundary center");
  finiteVec3(patch.normal, "Planar boundary normal");
  finiteVec3(patch.tangentU, "Planar boundary tangent U");
  finiteVec3(patch.tangentV, "Planar boundary tangent V");
  const tolerance = 2e-5;
  for (const [label, axis] of [["normal", patch.normal], ["tangent U", patch.tangentU],
    ["tangent V", patch.tangentV]] as const) {
    if (Math.abs(length(axis) - 1) > tolerance) {
      throw new RangeError(`Planar boundary ${label} must be unit length`);
    }
  }
  if (Math.abs(dot(patch.normal, patch.tangentU)) > tolerance
    || Math.abs(dot(patch.normal, patch.tangentV)) > tolerance
    || Math.abs(dot(patch.tangentU, patch.tangentV)) > tolerance) {
    throw new RangeError("Planar boundary axes must be mutually orthogonal");
  }
  for (const [label, value] of [["half extent U", patch.halfExtentU_m],
    ["half extent V", patch.halfExtentV_m], ["half thickness", patch.halfThickness_m]] as const) {
    if (!(value > 0) || !Number.isFinite(value)) {
      throw new RangeError(`Planar boundary ${label} must be positive and finite`);
    }
  }
  for (const [label, value] of [["material ID", patch.materialId], ["owner ID", patch.ownerId]] as const) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff) {
      throw new RangeError(`Planar boundary ${label} must fit uint16`);
    }
  }
}

export function packPlanarBoundaryPatches(
  patches: readonly PlanarBoundaryPatch[],
): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(patches.length * PLANAR_BOUNDARY_PATCH_BYTES));
  const floats = new Float32Array(words.buffer);
  patches.forEach((patch, index) => {
    validatePlanarBoundaryPatch(patch);
    const base = index * PLANAR_BOUNDARY_PATCH_WORDS;
    floats.set(patch.center_m, base); floats[base + 3] = patch.halfThickness_m;
    floats.set(patch.normal, base + 4); floats[base + 7] = patch.halfExtentU_m;
    floats.set(patch.tangentU, base + 8); floats[base + 11] = patch.halfExtentV_m;
    floats.set(patch.tangentV, base + 12);
    words[base + 15] = ((patch.ownerId << 16) | patch.materialId) >>> 0;
  });
  return words;
}

/** Exact ray/finite-slab intersection. Direction need not be normalized. */
export function intersectPlanarBoundaryPatch(
  patch: PlanarBoundaryPatch,
  origin_m: PlanarBoundaryVec3,
  direction: PlanarBoundaryVec3,
  tMin_m = 0,
  tMax_m = Number.POSITIVE_INFINITY,
): PlanarBoundaryRayHit | null {
  validatePlanarBoundaryPatch(patch);
  finiteVec3(origin_m, "Planar boundary ray origin");
  finiteVec3(direction, "Planar boundary ray direction");
  if (direction.every((component) => component === 0)) {
    throw new RangeError("Planar boundary ray direction must be non-zero");
  }
  if (!Number.isFinite(tMin_m) || tMax_m < tMin_m
    || !(Number.isFinite(tMax_m) || tMax_m === Number.POSITIVE_INFINITY)) {
    throw new RangeError("Planar boundary ray interval is invalid");
  }
  const relative = origin_m.map((value, axis) => value - patch.center_m[axis]) as unknown as PlanarBoundaryVec3;
  const axes = [patch.tangentU, patch.tangentV, patch.normal] as const;
  const extents = [patch.halfExtentU_m, patch.halfExtentV_m, patch.halfThickness_m] as const;
  let enter = Number.NEGATIVE_INFINITY, exit = Number.POSITIVE_INFINITY;
  let enterAxis = -1, enterSign = 0, exitAxis = -1, exitSign = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const localOrigin = dot(relative, axes[axis]);
    const localDirection = dot(direction, axes[axis]);
    if (Math.abs(localDirection) < 1e-12) {
      if (Math.abs(localOrigin) > extents[axis]) return null;
      continue;
    }
    let near = (-extents[axis] - localOrigin) / localDirection;
    let far = (extents[axis] - localOrigin) / localDirection;
    let sign = -1;
    if (near > far) { [near, far] = [far, near]; sign = 1; }
    if (near > enter) { enter = near; enterAxis = axis; enterSign = sign; }
    if (far < exit) { exit = far; exitAxis = axis; exitSign = -sign; }
    if (exit < enter) return null;
  }
  const entering = enter >= tMin_m;
  const tHit_m = entering ? enter : exit;
  if (tHit_m < tMin_m || tHit_m > tMax_m) return null;
  const featureAxis = entering ? enterAxis : exitAxis;
  const sign = entering ? enterSign : exitSign;
  if (featureAxis < 0) return null;
  const axis = axes[featureAxis];
  return {
    tHit_m,
    tEnter_m: Math.max(enter, tMin_m),
    tExit_m: Math.min(exit, tMax_m),
    normal: [axis[0] * sign, axis[1] * sign, axis[2] * sign],
    featureAxis: featureAxis as 0 | 1 | 2,
  };
}

/** WGSL decoder/intersector for the packed record above. */
export const planarBoundaryWGSL = /* wgsl */ `
struct PlanarBoundaryPatch {
  centerThickness: vec4f,
  normalExtentU: vec4f,
  tangentUExtentV: vec4f,
  tangentVIdentity: vec4u,
}
struct PlanarBoundaryHit { valid:u32,tHit:f32,tEnter:f32,tExit:f32,normal:vec3f,featureAxis:u32 }
fn planarBoundaryIdentity(boundary:PlanarBoundaryPatch)->u32{return boundary.tangentVIdentity.w;}
fn planarBoundaryTangentV(boundary:PlanarBoundaryPatch)->vec3f{return bitcast<vec3f>(boundary.tangentVIdentity.xyz);}
fn intersectPlanarBoundary(boundary:PlanarBoundaryPatch,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->PlanarBoundaryHit{
  let relative=ro-boundary.centerThickness.xyz;
  let tangentV=planarBoundaryTangentV(boundary);
  let axes=mat3x3f(boundary.tangentUExtentV.xyz,tangentV,boundary.normalExtentU.xyz);
  let extents=vec3f(boundary.normalExtentU.w,boundary.tangentUExtentV.w,boundary.centerThickness.w);
  var enter=-3.402823e38;var exit=3.402823e38;var enterAxis=0u;var enterSign=0.0;var exitAxis=0u;var exitSign=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    let localOrigin=dot(relative,axes[axis]);let localDirection=dot(rd,axes[axis]);
    if(abs(localDirection)<1e-12){if(abs(localOrigin)>extents[axis]){return PlanarBoundaryHit(0u,0.0,0.0,0.0,vec3f(0.0),0u);}continue;}
    var near=(-extents[axis]-localOrigin)/localDirection;var far=(extents[axis]-localOrigin)/localDirection;var sign=-1.0;
    if(near>far){let swap=near;near=far;far=swap;sign=1.0;}
    if(near>enter){enter=near;enterAxis=axis;enterSign=sign;}
    if(far<exit){exit=far;exitAxis=axis;exitSign=-sign;}
    if(exit<enter){return PlanarBoundaryHit(0u,0.0,0.0,0.0,vec3f(0.0),0u);}
  }
  let entering=enter>=tMin;let tHit=select(exit,enter,entering);
  if(tHit<tMin||tHit>tMax){return PlanarBoundaryHit(0u,0.0,0.0,0.0,vec3f(0.0),0u);}
  let normal=select(axes[exitAxis]*exitSign,axes[enterAxis]*enterSign,entering);
  let featureAxis=select(exitAxis,enterAxis,entering);
  return PlanarBoundaryHit(1u,tHit,max(enter,tMin),min(exit,tMax),normal,featureAxis);
}`;
