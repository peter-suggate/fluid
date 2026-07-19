import type { Quaternion } from "./model";
import { SVO_MEDIA, type SvoMediaBoundary, type SvoMediumKind } from "./svo-media";
import { packMaterialOwner, SPARSE_BRICK_NO_OWNER, unpackMaterialOwner } from "./sparse-brick-octree";
import { beerLambert, dielectricFresnel, GLASS_OPTICS, type LinearRgb } from "./webgpu-lighting";
import type { SvoAabb, SvoVec3 } from "./webgpu-svo-traversal";

/** Five host-shareable vec4 lanes. */
export const SVO_THIN_GLASS_RECORD_STRIDE_BYTES = 80;
export const SVO_THIN_GLASS_RECORD_WORDS = SVO_THIN_GLASS_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;

export const SVO_THIN_GLASS_FEATURES = Object.freeze({
  face: 0,
  edgeX: 1,
  edgeY: 2,
  corner: 3,
} as const);

export const SVO_THIN_GLASS_FLAGS = Object.freeze({
  twoSided: 1,
  analyticFinalHitRequired: 2,
} as const);

export const SVO_THIN_GLASS_REFINEMENT = Object.freeze({
  maximumIterations: 6,
  distanceToleranceCells: 1e-3,
  analyticFinalHitRequired: true,
} as const);

export interface SvoThinGlassPane {
  paneId: number;
  materialId: number;
  ownerId?: number;
  center_m: SvoVec3;
  /** Local pane axes are X/Y and the authored outward normal is +Z. */
  orientation?: Quaternion;
  halfExtent_m: readonly [x: number, y: number];
  thickness_m: number;
  indexOfRefraction?: number;
  absorption_mInv: LinearRgb;
  /** Finite pane edge/corner tie tolerance in world metres. */
  edgeEpsilon_m?: number;
  /** Caps grazing optical distance through the collapsed sheet. */
  maximumOpticalPath_m?: number;
}

export interface SvoThinGlassRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  tMin_m?: number;
  tMax_m: number;
}

export interface SvoThinGlassIntersectionOptions {
  /** Rays with a smaller absolute normal cosine are treated as parallel. */
  minimumRayCosine?: number;
  /** Reject self-hits at or below this world-space ray distance. */
  surfaceEpsilon_m?: number;
}

export interface SvoThinGlassHit {
  t_m: number;
  position_m: SvoVec3;
  /** Always faces against the incident ray. */
  geometricNormal: SvoVec3;
  /** Stable authored +Z normal, independent of ray side. */
  authoredNormal: SvoVec3;
  frontFacing: boolean;
  uv: readonly [number, number];
  featureId: number;
  paneId: number;
  materialId: number;
  ownerId: number;
  normalCosine: number;
  opticalPath_m: number;
}

export interface SvoThinGlassBounds {
  exact_m: SvoAabb;
  conservative_m: SvoAabb;
  samplingPadding_m: number;
  refinementTolerance_m: number;
  maximumRefinementIterations: number;
  analyticFinalHitRequired: true;
}

export interface SvoThinGlassOptics {
  fresnel: number;
  opticalPath_m: number;
  absorptionTint: LinearRgb;
  netTransmittance: LinearRgb;
}

const DEFAULT_EDGE_EPSILON_M = 1e-6;
const DEFAULT_MINIMUM_RAY_COSINE = 1e-6;

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function positive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be finite and positive`);
  return value;
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function canonicalOrientation(input: Quaternion | undefined): Quaternion {
  const value = input ?? { w: 1, x: 0, y: 0, z: 0 };
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) throw new RangeError("Thin-glass orientation must be finite");
  const length = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(length > 1e-12)) throw new RangeError("Thin-glass orientation must be non-zero");
  return { w: value.w / length, x: value.x / length, y: value.y / length, z: value.z / length };
}

/** Validate and fill all bounded thin-pane defaults. */
export function canonicalSvoThinGlassPane(input: SvoThinGlassPane): Required<SvoThinGlassPane> {
  const paneId = uint32(input.paneId, "Thin-glass pane ID");
  if (!Number.isInteger(input.materialId) || input.materialId < 1 || input.materialId > 0xffff) {
    throw new RangeError("Thin-glass material ID must be a nonzero uint16");
  }
  const ownerId = input.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) throw new RangeError("Thin-glass owner ID must fit uint16");
  finiteVec3(input.center_m, "Thin-glass center");
  positive(input.halfExtent_m[0], "Thin-glass X half extent");
  positive(input.halfExtent_m[1], "Thin-glass Y half extent");
  positive(input.thickness_m, "Thin-glass thickness");
  const indexOfRefraction = positive(input.indexOfRefraction ?? GLASS_OPTICS.indexOfRefraction, "Thin-glass IOR");
  if (indexOfRefraction < 1) throw new RangeError("Thin-glass IOR must be at least one");
  if (input.absorption_mInv.some((channel) => !Number.isFinite(channel) || channel < 0)) {
    throw new RangeError("Thin-glass absorption must contain finite non-negative channels");
  }
  const edgeEpsilon_m = input.edgeEpsilon_m ?? DEFAULT_EDGE_EPSILON_M;
  positive(edgeEpsilon_m, "Thin-glass edge epsilon");
  if (edgeEpsilon_m > Math.min(...input.halfExtent_m) * 0.25) {
    throw new RangeError("Thin-glass edge epsilon is too large for the finite pane");
  }
  const maximumOpticalPath_m = input.maximumOpticalPath_m ?? input.thickness_m * 64;
  positive(maximumOpticalPath_m, "Thin-glass maximum optical path");
  if (maximumOpticalPath_m < input.thickness_m) {
    throw new RangeError("Thin-glass maximum optical path cannot be below its thickness");
  }
  return {
    paneId,
    materialId: input.materialId,
    ownerId,
    center_m: [...input.center_m],
    orientation: canonicalOrientation(input.orientation),
    halfExtent_m: [...input.halfExtent_m],
    thickness_m: input.thickness_m,
    indexOfRefraction,
    absorption_mInv: [...input.absorption_mInv],
    edgeEpsilon_m,
    maximumOpticalPath_m,
  };
}

/** Pack the stable analytic-pane ABI without renderer-specific bindings. */
export function packSvoThinGlassPanes(panes: readonly SvoThinGlassPane[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(panes.length * SVO_THIN_GLASS_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  panes.forEach((input, index) => {
    const pane = canonicalSvoThinGlassPane(input);
    const base = index * SVO_THIN_GLASS_RECORD_WORDS;
    floats.set([pane.center_m[0], pane.center_m[1], pane.center_m[2], pane.thickness_m], base);
    floats.set([pane.halfExtent_m[0], pane.halfExtent_m[1], pane.indexOfRefraction, pane.edgeEpsilon_m], base + 4);
    floats.set([pane.orientation.x, pane.orientation.y, pane.orientation.z, pane.orientation.w], base + 8);
    floats.set([pane.absorption_mInv[0], pane.absorption_mInv[1], pane.absorption_mInv[2], pane.maximumOpticalPath_m], base + 12);
    words.set([
      pane.paneId,
      packMaterialOwner(pane.materialId, pane.ownerId),
      SVO_THIN_GLASS_FLAGS.twoSided | SVO_THIN_GLASS_FLAGS.analyticFinalHitRequired,
      0,
    ], base + 16);
  });
  return words;
}

/** Deterministic unpack mirror for capture and CPU/GPU parity tests. */
export function unpackSvoThinGlassPanes(packed: Uint32Array): Required<SvoThinGlassPane>[] {
  if (packed.length % SVO_THIN_GLASS_RECORD_WORDS !== 0) throw new RangeError("Packed thin-glass data has a partial record");
  const words = new Uint32Array(packed);
  const floats = new Float32Array(words.buffer);
  const result: Required<SvoThinGlassPane>[] = [];
  for (let base = 0; base < words.length; base += SVO_THIN_GLASS_RECORD_WORDS) {
    const identity = unpackMaterialOwner(words[base + 17]);
    result.push(canonicalSvoThinGlassPane({
      paneId: words[base + 16], materialId: identity.materialId, ownerId: identity.ownerId,
      center_m: [floats[base], floats[base + 1], floats[base + 2]], thickness_m: floats[base + 3],
      halfExtent_m: [floats[base + 4], floats[base + 5]], indexOfRefraction: floats[base + 6], edgeEpsilon_m: floats[base + 7],
      orientation: { x: floats[base + 8], y: floats[base + 9], z: floats[base + 10], w: floats[base + 11] },
      absorption_mInv: [floats[base + 12], floats[base + 13], floats[base + 14]], maximumOpticalPath_m: floats[base + 15],
    }));
  }
  return result;
}

function rotate(q: Quaternion, value: SvoVec3): SvoVec3 {
  const tx = 2 * (q.y * value[2] - q.z * value[1]);
  const ty = 2 * (q.z * value[0] - q.x * value[2]);
  const tz = 2 * (q.x * value[1] - q.y * value[0]);
  return [
    value[0] + q.w * tx + q.y * tz - q.z * ty,
    value[1] + q.w * ty + q.z * tx - q.x * tz,
    value[2] + q.w * tz + q.x * ty - q.y * tx,
  ];
}

function inverseRotate(q: Quaternion, value: SvoVec3): SvoVec3 {
  return rotate({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, value);
}

function normalized(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function dot(left: SvoVec3, right: SvoVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

/** Exact two-sided intersection of an oriented finite analytic pane. */
export function intersectSvoThinGlassPane(
  input: SvoThinGlassPane,
  ray: SvoThinGlassRay,
  options: SvoThinGlassIntersectionOptions = {},
): SvoThinGlassHit | null {
  const pane = canonicalSvoThinGlassPane(input);
  finiteVec3(ray.origin_m, "Thin-glass ray origin");
  const direction = normalized(ray.direction, "Thin-glass ray direction");
  const tMin = ray.tMin_m ?? 0;
  if (!Number.isFinite(tMin) || !Number.isFinite(ray.tMax_m) || ray.tMax_m < tMin) {
    throw new RangeError("Thin-glass ray interval must be finite and non-decreasing");
  }
  const minimumRayCosine = positive(options.minimumRayCosine ?? DEFAULT_MINIMUM_RAY_COSINE, "Thin-glass minimum ray cosine");
  if (minimumRayCosine > 1) throw new RangeError("Thin-glass minimum ray cosine cannot exceed one");
  const surfaceEpsilon = options.surfaceEpsilon_m ?? pane.edgeEpsilon_m;
  if (!Number.isFinite(surfaceEpsilon) || surfaceEpsilon < 0) throw new RangeError("Thin-glass surface epsilon must be finite and non-negative");
  const relative: SvoVec3 = [
    ray.origin_m[0] - pane.center_m[0], ray.origin_m[1] - pane.center_m[1], ray.origin_m[2] - pane.center_m[2],
  ];
  const localOrigin = inverseRotate(pane.orientation, relative);
  const localDirection = inverseRotate(pane.orientation, direction);
  const normalCosine = Math.abs(localDirection[2]);
  if (normalCosine < minimumRayCosine) return null;
  const t_m = -localOrigin[2] / localDirection[2];
  if (!(t_m > Math.max(tMin, surfaceEpsilon)) || t_m > ray.tMax_m) return null;
  const x = localOrigin[0] + localDirection[0] * t_m;
  const y = localOrigin[1] + localDirection[1] * t_m;
  if (Math.abs(x) > pane.halfExtent_m[0] + pane.edgeEpsilon_m
      || Math.abs(y) > pane.halfExtent_m[1] + pane.edgeEpsilon_m) return null;
  const atX = pane.halfExtent_m[0] - Math.abs(x) <= pane.edgeEpsilon_m;
  const atY = pane.halfExtent_m[1] - Math.abs(y) <= pane.edgeEpsilon_m;
  const featureId = atX && atY ? SVO_THIN_GLASS_FEATURES.corner
    : atX ? SVO_THIN_GLASS_FEATURES.edgeX
      : atY ? SVO_THIN_GLASS_FEATURES.edgeY : SVO_THIN_GLASS_FEATURES.face;
  const authoredNormal = normalized(rotate(pane.orientation, [0, 0, 1]), "Thin-glass authored normal");
  const frontFacing = dot(direction, authoredNormal) < 0;
  const geometricNormal: SvoVec3 = frontFacing ? authoredNormal : authoredNormal.map(
    (component) => component === 0 ? 0 : -component,
  ) as [number, number, number];
  return {
    t_m,
    position_m: [
      ray.origin_m[0] + direction[0] * t_m,
      ray.origin_m[1] + direction[1] * t_m,
      ray.origin_m[2] + direction[2] * t_m,
    ],
    geometricNormal,
    authoredNormal,
    frontFacing,
    uv: [
      Math.min(1, Math.max(-1, x / pane.halfExtent_m[0])),
      Math.min(1, Math.max(-1, y / pane.halfExtent_m[1])),
    ],
    featureId,
    paneId: pane.paneId,
    materialId: pane.materialId,
    ownerId: pane.ownerId,
    normalCosine,
    opticalPath_m: Math.min(pane.maximumOpticalPath_m, pane.thickness_m / Math.max(normalCosine, minimumRayCosine)),
  };
}

/** Conservative SVO candidate bounds plus the mandatory analytic final-hit rule. */
export function svoThinGlassBounds(input: SvoThinGlassPane, cellSize_m: SvoVec3): SvoThinGlassBounds {
  const pane = canonicalSvoThinGlassPane(input);
  finiteVec3(cellSize_m, "Thin-glass bounds cell size");
  cellSize_m.forEach((value) => positive(value, "Thin-glass bounds cell size"));
  const axisX = rotate(pane.orientation, [1, 0, 0]);
  const axisY = rotate(pane.orientation, [0, 1, 0]);
  const axisZ = rotate(pane.orientation, [0, 0, 1]);
  const halfThickness = 0.5 * pane.thickness_m;
  const extent = [0, 1, 2].map((axis) =>
    Math.abs(axisX[axis]) * pane.halfExtent_m[0]
    + Math.abs(axisY[axis]) * pane.halfExtent_m[1]
    + Math.abs(axisZ[axis]) * halfThickness) as [number, number, number];
  const exact_m: SvoAabb = {
    minimum: pane.center_m.map((value, axis) => value - extent[axis]) as [number, number, number],
    maximum: pane.center_m.map((value, axis) => value + extent[axis]) as [number, number, number],
  };
  const samplingPadding_m = 0.5 * Math.hypot(...cellSize_m) + pane.edgeEpsilon_m;
  const conservative_m: SvoAabb = {
    minimum: exact_m.minimum.map((value) => value - samplingPadding_m) as [number, number, number],
    maximum: exact_m.maximum.map((value) => value + samplingPadding_m) as [number, number, number],
  };
  return {
    exact_m,
    conservative_m,
    samplingPadding_m,
    refinementTolerance_m: SVO_THIN_GLASS_REFINEMENT.distanceToleranceCells * Math.min(...cellSize_m),
    maximumRefinementIterations: SVO_THIN_GLASS_REFINEMENT.maximumIterations,
    analyticFinalHitRequired: true,
  };
}

/** Two-interface thin-sheet Fresnel and bounded Beer-Lambert absorption. */
export function evaluateSvoThinGlassOptics(
  input: SvoThinGlassPane,
  hit: Pick<SvoThinGlassHit, "normalCosine" | "opticalPath_m">,
  incidentMedium: SvoMediumKind,
): SvoThinGlassOptics {
  const pane = canonicalSvoThinGlassPane(input);
  const mediumIor = SVO_MEDIA[incidentMedium].indexOfRefraction;
  const f0 = ((mediumIor - pane.indexOfRefraction) / (mediumIor + pane.indexOfRefraction)) ** 2;
  const fresnel = dielectricFresnel(hit.normalCosine, f0);
  const opticalPath_m = Math.min(pane.maximumOpticalPath_m, Math.max(pane.thickness_m, hit.opticalPath_m));
  const absorptionTint = beerLambert(pane.absorption_mInv, opticalPath_m);
  const interfaceTransmission = (1 - fresnel) ** 2;
  return {
    fresnel,
    opticalPath_m,
    absorptionTint,
    netTransmittance: absorptionTint.map((channel) => channel * interfaceTransmission) as [number, number, number],
  };
}

/** Adapter into `svo-media.ts`; exits/entries still resolve atomically there. */
export function svoThinGlassMediaBoundary(
  input: SvoThinGlassPane,
  hit: SvoThinGlassHit,
): SvoMediaBoundary {
  const pane = canonicalSvoThinGlassPane(input);
  const absorptionTint = beerLambert(pane.absorption_mInv, hit.opticalPath_m);
  return {
    t_m: hit.t_m,
    medium: "glass",
    geometricNormal: hit.authoredNormal,
    thinWall: true,
    thinWallTint: absorptionTint,
    thinWallIor: pane.indexOfRefraction,
    boundaryId: pane.paneId,
  };
}

/** Binding-free GPU mirror of the packed pane and exact finite intersection. */
export const svoThinGlassWGSL = /* wgsl */ `
const SVO_THIN_GLASS_FEATURE_FACE:u32=0u;const SVO_THIN_GLASS_FEATURE_EDGE_X:u32=1u;const SVO_THIN_GLASS_FEATURE_EDGE_Y:u32=2u;const SVO_THIN_GLASS_FEATURE_CORNER:u32=3u;
const SVO_THIN_GLASS_MAX_REFINEMENT:u32=6u;const SVO_THIN_GLASS_DISTANCE_TOLERANCE_CELLS:f32=1e-3;
struct SvoThinGlassRecord{centerThickness:vec4f,extentIorEpsilon:vec4f,orientation:vec4f,absorptionPath:vec4f,identity:vec4u}
struct SvoThinGlassHit{valid:u32,featureId:u32,frontFacing:u32,_padding:u32,t_m:f32,normalCosine:f32,opticalPath_m:f32,_padding2:f32,position_m:vec3f,_padding3:f32,geometricNormal:vec3f,_padding4:f32,authoredNormal:vec3f,_padding5:f32,uv:vec2f,_padding6:vec2f}
struct SvoThinGlassOptics{fresnel:f32,opticalPath_m:f32,_padding:vec2f,absorptionTint:vec3f,_padding2:f32,netTransmittance:vec3f,_padding3:f32}
fn svoThinGlassRotate(q:vec4f,v:vec3f)->vec3f{let t=2.0*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);}
fn svoThinGlassInverseRotate(q:vec4f,v:vec3f)->vec3f{return svoThinGlassRotate(vec4f(-q.xyz,q.w),v);}
fn svoThinGlassMiss()->SvoThinGlassHit{return SvoThinGlassHit(0u,0u,0u,0u,0.0,0.0,0.0,0.0,vec3f(0.0),0.0,vec3f(0.0,1.0,0.0),0.0,vec3f(0.0,1.0,0.0),0.0,vec2f(0.0),vec2f(0.0));}
fn svoThinGlassIntersect(record:SvoThinGlassRecord,rayOrigin_m:vec3f,rayDirectionIn:vec3f,tMin_m:f32,tMax_m:f32,minimumRayCosine:f32,surfaceEpsilon_m:f32)->SvoThinGlassHit{let rayDirection=normalize(rayDirectionIn);let localOrigin=svoThinGlassInverseRotate(record.orientation,rayOrigin_m-record.centerThickness.xyz);let localDirection=svoThinGlassInverseRotate(record.orientation,rayDirection);let normalCosine=abs(localDirection.z);if(normalCosine<minimumRayCosine){return svoThinGlassMiss();}let t_m=-localOrigin.z/localDirection.z;if(t_m<=max(tMin_m,surfaceEpsilon_m)||t_m>tMax_m){return svoThinGlassMiss();}let local=localOrigin+localDirection*t_m;let extent=record.extentIorEpsilon.xy;let epsilon=record.extentIorEpsilon.w;if(any(abs(local.xy)>extent+vec2f(epsilon))){return svoThinGlassMiss();}let atX=extent.x-abs(local.x)<=epsilon;let atY=extent.y-abs(local.y)<=epsilon;var feature=SVO_THIN_GLASS_FEATURE_FACE;if(atX&&atY){feature=SVO_THIN_GLASS_FEATURE_CORNER;}else if(atX){feature=SVO_THIN_GLASS_FEATURE_EDGE_X;}else if(atY){feature=SVO_THIN_GLASS_FEATURE_EDGE_Y;}let authored=normalize(svoThinGlassRotate(record.orientation,vec3f(0.0,0.0,1.0)));let front=dot(rayDirection,authored)<0.0;let normal=select(-authored,authored,front);let opticalPath=min(record.absorptionPath.w,record.centerThickness.w/max(normalCosine,minimumRayCosine));return SvoThinGlassHit(1u,feature,select(0u,1u,front),0u,t_m,normalCosine,opticalPath,0.0,rayOrigin_m+rayDirection*t_m,0.0,normal,0.0,authored,0.0,clamp(local.xy/extent,vec2f(-1.0),vec2f(1.0)),vec2f(0.0));}
fn svoThinGlassBounds(record:SvoThinGlassRecord,samplingPadding_m:f32)->mat2x3f{let axisX=svoThinGlassRotate(record.orientation,vec3f(1.0,0.0,0.0));let axisY=svoThinGlassRotate(record.orientation,vec3f(0.0,1.0,0.0));let axisZ=svoThinGlassRotate(record.orientation,vec3f(0.0,0.0,1.0));let extent=abs(axisX)*record.extentIorEpsilon.x+abs(axisY)*record.extentIorEpsilon.y+abs(axisZ)*0.5*record.centerThickness.w+vec3f(max(samplingPadding_m,0.0));return mat2x3f(record.centerThickness.xyz-extent,record.centerThickness.xyz+extent);}
fn svoThinGlassOptics(record:SvoThinGlassRecord,hit:SvoThinGlassHit,incidentIor:f32)->SvoThinGlassOptics{let ratio=(incidentIor-record.extentIorEpsilon.z)/(incidentIor+record.extentIorEpsilon.z);let f0=ratio*ratio;let fresnel=f0+(1.0-f0)*pow(1.0-clamp(hit.normalCosine,0.0,1.0),5.0);let path=min(record.absorptionPath.w,max(record.centerThickness.w,hit.opticalPath_m));let tint=exp(-max(record.absorptionPath.xyz,vec3f(0.0))*path);return SvoThinGlassOptics(fresnel,path,vec2f(0.0),tint,0.0,tint*pow(1.0-fresnel,2.0),0.0);}
fn svoThinGlassMaterialId(record:SvoThinGlassRecord)->u32{return record.identity.y&0xffffu;}fn svoThinGlassOwnerId(record:SvoThinGlassRecord)->u32{return record.identity.y>>16u;}fn svoThinGlassPaneId(record:SvoThinGlassRecord)->u32{return record.identity.x;}
`;
