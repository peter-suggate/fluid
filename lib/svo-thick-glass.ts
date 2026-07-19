import type { Quaternion } from "./model";
import { SVO_MEDIA, type SvoMediaBoundary, type SvoMediumKind } from "./svo-media";
import { packMaterialOwner, SPARSE_BRICK_NO_OWNER, unpackMaterialOwner } from "./sparse-brick-octree";
import { beerLambert, dielectricFresnel, GLASS_OPTICS, type LinearRgb } from "./webgpu-lighting";
import type { SvoAabb, SvoVec3 } from "./webgpu-svo-traversal";

/** Five host-shareable vec4 lanes, deliberately separate from thin panes. */
export const SVO_THICK_GLASS_RECORD_STRIDE_BYTES = 80;
export const SVO_THICK_GLASS_RECORD_WORDS = SVO_THICK_GLASS_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;

export const SVO_THICK_GLASS_FLAGS = Object.freeze({
  sphere: 1,
  ellipsoid: 2,
  analyticFinalHitRequired: 4,
} as const);

export const SVO_THICK_GLASS_QUERY_STATUS = Object.freeze({
  miss: 0,
  hit: 1,
  invalid: 2,
  stale: 3,
} as const);

export type SvoThickGlassShape = "sphere" | "ellipsoid";

export interface SvoThickGlassVolume {
  glassId: number;
  materialId: number;
  ownerId?: number;
  revision: number;
  shape: SvoThickGlassShape;
  center_m: SvoVec3;
  radii_m: SvoVec3;
  orientation?: Quaternion;
  indexOfRefraction?: number;
  absorption_mInv: LinearRgb;
  surfaceEpsilon_m?: number;
  maximumOpticalPath_m?: number;
}

export interface SvoThickGlassRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  tMin_m?: number;
  tMax_m: number;
}

export interface SvoThickGlassSurfaceHit {
  t_m: number;
  position_m: SvoVec3;
  /** Outward from the bounded glass volume. */
  geometricNormal: SvoVec3;
  frontFacing: boolean;
  glassId: number;
  materialId: number;
  ownerId: number;
  revision: number;
}

export interface SvoThickGlassInterval {
  entry?: SvoThickGlassSurfaceHit;
  exit: SvoThickGlassSurfaceHit;
  first: SvoThickGlassSurfaceHit;
  insideAtStart: boolean;
  tangent: boolean;
  opticalPath_m: number;
}

export type SvoThickGlassQueryResult =
  | { status: "hit"; interval: SvoThickGlassInterval }
  | { status: "miss" }
  | { status: "invalid"; reason: string }
  | { status: "stale"; expectedRevision: number; actualRevision: number };

export interface SvoThickGlassInterfaceOptics {
  from: SvoMediumKind;
  to: SvoMediumKind;
  fromIor: number;
  toIor: number;
  reflectedDirection: SvoVec3;
  refractedDirection?: SvoVec3;
  fresnel: number;
  totalInternalReflection: boolean;
}

export interface SvoThickGlassMediumHandoff {
  boundary: SvoMediaBoundary;
  optics: SvoThickGlassInterfaceOptics;
  absorption_mInv: LinearRgb;
  indexOfRefraction: number;
  opticalPath_m: number;
  absorptionTint: LinearRgb;
}

const Q_IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };

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
  const value = input ?? Q_IDENTITY;
  if (![value.w, value.x, value.y, value.z].every(Number.isFinite)) {
    throw new RangeError("Thick-glass orientation must be finite");
  }
  const length = Math.hypot(value.w, value.x, value.y, value.z);
  if (!(length > 1e-12)) throw new RangeError("Thick-glass orientation must be non-zero");
  return { w: value.w / length, x: value.x / length, y: value.y / length, z: value.z / length };
}

export function canonicalSvoThickGlassVolume(input: SvoThickGlassVolume): Required<SvoThickGlassVolume> {
  const glassId = uint32(input.glassId, "Thick-glass ID");
  if (glassId === 0) throw new RangeError("Thick-glass ID zero is reserved");
  if (!Number.isInteger(input.materialId) || input.materialId < 1 || input.materialId > 0xffff) {
    throw new RangeError("Thick-glass material ID must be a nonzero uint16");
  }
  const ownerId = input.ownerId ?? SPARSE_BRICK_NO_OWNER;
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) {
    throw new RangeError("Thick-glass owner ID must fit uint16");
  }
  const revision = uint32(input.revision, "Thick-glass revision");
  if (revision === 0) throw new RangeError("Thick-glass revision zero is unpublished");
  finiteVec3(input.center_m, "Thick-glass center");
  finiteVec3(input.radii_m, "Thick-glass radii");
  input.radii_m.forEach((radius) => positive(radius, "Thick-glass radius"));
  if (input.shape === "sphere") {
    const tolerance = Math.max(...input.radii_m) * 1e-6;
    if (Math.max(...input.radii_m) - Math.min(...input.radii_m) > tolerance) {
      throw new RangeError("Thick-glass sphere requires equal radii");
    }
  } else if (input.shape !== "ellipsoid") throw new RangeError("Unsupported thick-glass shape");
  const indexOfRefraction = positive(input.indexOfRefraction ?? GLASS_OPTICS.indexOfRefraction, "Thick-glass IOR");
  if (indexOfRefraction < 1) throw new RangeError("Thick-glass IOR must be at least one");
  if (input.absorption_mInv.some((channel) => !Number.isFinite(channel) || channel < 0)) {
    throw new RangeError("Thick-glass absorption must contain finite non-negative channels");
  }
  const surfaceEpsilon_m = positive(input.surfaceEpsilon_m ?? Math.min(...input.radii_m) * 1e-5, "Thick-glass surface epsilon");
  if (surfaceEpsilon_m >= Math.min(...input.radii_m)) {
    throw new RangeError("Thick-glass surface epsilon must be smaller than every radius");
  }
  const maximumOpticalPath_m = positive(
    input.maximumOpticalPath_m ?? 2 * Math.max(...input.radii_m),
    "Thick-glass maximum optical path",
  );
  return {
    glassId,
    materialId: input.materialId,
    ownerId,
    revision,
    shape: input.shape,
    center_m: [...input.center_m],
    radii_m: [...input.radii_m],
    orientation: canonicalOrientation(input.orientation),
    indexOfRefraction,
    absorption_mInv: [...input.absorption_mInv],
    surfaceEpsilon_m,
    maximumOpticalPath_m,
  };
}

export function packSvoThickGlassVolumes(volumes: readonly SvoThickGlassVolume[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(volumes.length * SVO_THICK_GLASS_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  volumes.forEach((input, index) => {
    const volume = canonicalSvoThickGlassVolume(input);
    const base = index * SVO_THICK_GLASS_RECORD_WORDS;
    floats.set([...volume.center_m, volume.radii_m[0]], base);
    floats.set([volume.radii_m[1], volume.radii_m[2], volume.indexOfRefraction, volume.surfaceEpsilon_m], base + 4);
    floats.set([volume.orientation.x, volume.orientation.y, volume.orientation.z, volume.orientation.w], base + 8);
    floats.set([...volume.absorption_mInv, volume.maximumOpticalPath_m], base + 12);
    words.set([
      volume.glassId,
      packMaterialOwner(volume.materialId, volume.ownerId),
      volume.revision,
      (volume.shape === "sphere" ? SVO_THICK_GLASS_FLAGS.sphere : SVO_THICK_GLASS_FLAGS.ellipsoid)
        | SVO_THICK_GLASS_FLAGS.analyticFinalHitRequired,
    ], base + 16);
  });
  return words;
}

export function unpackSvoThickGlassVolumes(packed: Uint32Array): Required<SvoThickGlassVolume>[] {
  if (packed.length % SVO_THICK_GLASS_RECORD_WORDS !== 0) throw new RangeError("Packed thick-glass data has a partial record");
  const words = new Uint32Array(packed);
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const result: Required<SvoThickGlassVolume>[] = [];
  for (let base = 0; base < words.length; base += SVO_THICK_GLASS_RECORD_WORDS) {
    const identity = unpackMaterialOwner(words[base + 17]);
    const shapeFlags = words[base + 19] & (SVO_THICK_GLASS_FLAGS.sphere | SVO_THICK_GLASS_FLAGS.ellipsoid);
    if (shapeFlags !== SVO_THICK_GLASS_FLAGS.sphere && shapeFlags !== SVO_THICK_GLASS_FLAGS.ellipsoid) {
      throw new RangeError("Packed thick-glass data has invalid shape flags");
    }
    result.push(canonicalSvoThickGlassVolume({
      glassId: words[base + 16],
      materialId: identity.materialId,
      ownerId: identity.ownerId,
      revision: words[base + 18],
      shape: shapeFlags === SVO_THICK_GLASS_FLAGS.sphere ? "sphere" : "ellipsoid",
      center_m: [floats[base], floats[base + 1], floats[base + 2]],
      radii_m: [floats[base + 3], floats[base + 4], floats[base + 5]],
      indexOfRefraction: floats[base + 6],
      surfaceEpsilon_m: floats[base + 7],
      orientation: { x: floats[base + 8], y: floats[base + 9], z: floats[base + 10], w: floats[base + 11] },
      absorption_mInv: [floats[base + 12], floats[base + 13], floats[base + 14]],
      maximumOpticalPath_m: floats[base + 15],
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

function normalize(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function dot(left: SvoVec3, right: SvoVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function map3(value: SvoVec3, transform: (component: number, axis: number) => number): SvoVec3 {
  return [transform(value[0], 0), transform(value[1], 1), transform(value[2], 2)];
}

function surfaceHit(
  volume: Required<SvoThickGlassVolume>,
  ray: Required<Pick<SvoThickGlassRay, "origin_m" | "direction">>,
  t_m: number,
): SvoThickGlassSurfaceHit {
  const position_m = map3(ray.origin_m, (value, axis) => value + ray.direction[axis] * t_m);
  const local = inverseRotate(volume.orientation, map3(position_m, (value, axis) => value - volume.center_m[axis]));
  const localGradient = map3(local, (value, axis) => value / (volume.radii_m[axis] ** 2));
  const geometricNormal = normalize(rotate(volume.orientation, localGradient), "Thick-glass normal");
  return {
    t_m,
    position_m,
    geometricNormal,
    frontFacing: dot(ray.direction, geometricNormal) < 0,
    glassId: volume.glassId,
    materialId: volume.materialId,
    ownerId: volume.ownerId,
    revision: volume.revision,
  };
}

/** Exact constant-work entry/exit interval for an oriented sphere or ellipsoid. */
export function intersectSvoThickGlassVolume(
  input: SvoThickGlassVolume,
  inputRay: SvoThickGlassRay,
): SvoThickGlassInterval | null {
  const volume = canonicalSvoThickGlassVolume(input);
  finiteVec3(inputRay.origin_m, "Thick-glass ray origin");
  const direction = normalize(inputRay.direction, "Thick-glass ray direction");
  const tMin_m = inputRay.tMin_m ?? 0;
  if (!Number.isFinite(tMin_m) || !Number.isFinite(inputRay.tMax_m) || inputRay.tMax_m < tMin_m) {
    throw new RangeError("Thick-glass ray interval must be finite and non-decreasing");
  }
  const relative = map3(inputRay.origin_m, (value, axis) => value - volume.center_m[axis]);
  const localOrigin = inverseRotate(volume.orientation, relative);
  const localDirection = inverseRotate(volume.orientation, direction);
  const scaledOrigin = map3(localOrigin, (value, axis) => value / volume.radii_m[axis]);
  const scaledDirection = map3(localDirection, (value, axis) => value / volume.radii_m[axis]);
  const a = dot(scaledDirection, scaledDirection);
  const halfB = dot(scaledOrigin, scaledDirection);
  const c = dot(scaledOrigin, scaledOrigin) - 1;
  const discriminant = halfB * halfB - a * c;
  if (discriminant < 0) return null;
  const root = Math.sqrt(Math.max(0, discriminant));
  const nearT = (-halfB - root) / a;
  const farT = (-halfB + root) / a;
  const minimumForward = Math.max(tMin_m, volume.surfaceEpsilon_m);
  if (!(farT > minimumForward) || nearT > inputRay.tMax_m) return null;
  const insideAtStart = c < -volume.surfaceEpsilon_m / Math.max(...volume.radii_m);
  const tangent = root <= volume.surfaceEpsilon_m * Math.sqrt(a);
  const ray = { origin_m: [...inputRay.origin_m] as SvoVec3, direction };
  const exit = surfaceHit(volume, ray, farT);
  const entry = nearT > minimumForward ? surfaceHit(volume, ray, nearT) : undefined;
  const first = entry ?? exit;
  if (first.t_m > inputRay.tMax_m) return null;
  const opticalPath_m = tangent ? 0 : Math.min(
    volume.maximumOpticalPath_m,
    Math.max(0, farT - Math.max(nearT, tMin_m, 0)),
  );
  return { entry, exit, first, insideAtStart, tangent, opticalPath_m };
}

/** Fail-closed adapter for publication generation checks and malformed records. */
export function querySvoThickGlassVolume(
  input: SvoThickGlassVolume,
  ray: SvoThickGlassRay,
  expectedRevision: number,
): SvoThickGlassQueryResult {
  if (!Number.isInteger(expectedRevision) || expectedRevision < 1 || expectedRevision > 0xffff_ffff) {
    return { status: "invalid", reason: "Expected thick-glass revision must be a positive uint32" };
  }
  try {
    const volume = canonicalSvoThickGlassVolume(input);
    if (volume.revision !== expectedRevision) {
      return { status: "stale", expectedRevision, actualRevision: volume.revision };
    }
    const interval = intersectSvoThickGlassVolume(volume, ray);
    return interval ? { status: "hit", interval } : { status: "miss" };
  } catch (error) {
    return { status: "invalid", reason: error instanceof Error ? error.message : String(error) };
  }
}

function reflect(direction: SvoVec3, normal: SvoVec3): SvoVec3 {
  const scale = 2 * dot(direction, normal);
  return normalize(map3(direction, (value, axis) => value - scale * normal[axis]), "Thick-glass reflection");
}

export function evaluateSvoThickGlassInterface(
  input: SvoThickGlassVolume,
  hit: SvoThickGlassSurfaceHit,
  incidentDirection: SvoVec3,
  from: SvoMediumKind,
  to: SvoMediumKind,
): SvoThickGlassInterfaceOptics {
  const volume = canonicalSvoThickGlassVolume(input);
  const direction = normalize(incidentDirection, "Thick-glass incident direction");
  let normal = normalize(hit.geometricNormal, "Thick-glass geometric normal");
  if (dot(direction, normal) > 0) normal = map3(normal, (value) => -value);
  const fromIor = from === "glass" ? volume.indexOfRefraction : SVO_MEDIA[from].indexOfRefraction;
  const toIor = to === "glass" ? volume.indexOfRefraction : SVO_MEDIA[to].indexOfRefraction;
  const eta = fromIor / toIor;
  const cosine = Math.min(1, Math.max(0, -dot(direction, normal)));
  const discriminant = 1 - eta * eta * (1 - cosine * cosine);
  const reflectedDirection = reflect(direction, normal);
  const f0 = ((fromIor - toIor) / (fromIor + toIor)) ** 2;
  if (discriminant < 0) return {
    from, to, fromIor, toIor, reflectedDirection,
    fresnel: 1,
    totalInternalReflection: true,
  };
  const refractedDirection = normalize(map3(direction, (value, axis) =>
    eta * value + (eta * cosine - Math.sqrt(Math.max(0, discriminant))) * normal[axis]),
  "Thick-glass refraction");
  return {
    from, to, fromIor, toIor, reflectedDirection, refractedDirection,
    fresnel: dielectricFresnel(cosine, f0),
    totalInternalReflection: false,
  };
}

/** One-interface handoff; the production binder remains intentionally absent. */
export function svoThickGlassMediumHandoff(
  input: SvoThickGlassVolume,
  interval: SvoThickGlassInterval,
  incidentDirection: SvoVec3,
  from: SvoMediumKind,
  to: SvoMediumKind,
): SvoThickGlassMediumHandoff {
  const volume = canonicalSvoThickGlassVolume(input);
  const surface = interval.first;
  return {
    boundary: {
      t_m: surface.t_m,
      medium: "glass",
      geometricNormal: surface.geometricNormal,
      thinWall: false,
      boundaryId: volume.glassId,
    },
    optics: evaluateSvoThickGlassInterface(volume, surface, incidentDirection, from, to),
    absorption_mInv: volume.absorption_mInv,
    indexOfRefraction: volume.indexOfRefraction,
    opticalPath_m: interval.opticalPath_m,
    absorptionTint: beerLambert(volume.absorption_mInv, interval.opticalPath_m),
  };
}

export function svoThickGlassBounds(input: SvoThickGlassVolume): SvoAabb {
  const volume = canonicalSvoThickGlassVolume(input);
  const axes = [
    rotate(volume.orientation, [1, 0, 0]),
    rotate(volume.orientation, [0, 1, 0]),
    rotate(volume.orientation, [0, 0, 1]),
  ];
  const extent: SvoVec3 = [0, 1, 2].map((axis) => axes.reduce(
    (sum, basis, basisIndex) => sum + Math.abs(basis[axis]) * volume.radii_m[basisIndex],
    0,
  )) as [number, number, number];
  return {
    minimum: map3(volume.center_m, (value, axis) => value - extent[axis]),
    maximum: map3(volume.center_m, (value, axis) => value + extent[axis]),
  };
}

/** Binding-free constant-work GPU mirror; publication binding is intentionally deferred. */
export const svoThickGlassWGSL = /* wgsl */ `
const SVO_THICK_GLASS_MISS:u32=0u;const SVO_THICK_GLASS_HIT:u32=1u;const SVO_THICK_GLASS_INVALID:u32=2u;const SVO_THICK_GLASS_STALE:u32=3u;
const SVO_THICK_GLASS_SPHERE:u32=1u;const SVO_THICK_GLASS_ELLIPSOID:u32=2u;const SVO_THICK_GLASS_ANALYTIC_FINAL:u32=4u;
struct SvoThickGlassRecord{centerRadiusX:vec4f,radiiYzIorEpsilon:vec4f,orientation:vec4f,absorptionPath:vec4f,identity:vec4u}
struct SvoThickGlassSurface{t_m:f32,frontFacing:u32,_padding:vec2u,position_m:vec3f,_padding2:f32,normal:vec3f,_padding3:f32}
struct SvoThickGlassInterval{status:u32,insideAtStart:u32,tangent:u32,hasEntry:u32,opticalPath_m:f32,_padding:vec3f,entry:SvoThickGlassSurface,exit:SvoThickGlassSurface}
struct SvoThickGlassInterface{reflectedDirection:vec3f,fresnel:f32,refractedDirection:vec3f,totalInternalReflection:u32,absorptionTint:vec3f,_padding:f32}
fn svoThickGlassFinite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn svoThickGlassFinite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
fn svoThickGlassRotate(q:vec4f,v:vec3f)->vec3f{let t=2.0*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);}
fn svoThickGlassInverseRotate(q:vec4f,v:vec3f)->vec3f{return svoThickGlassRotate(vec4f(-q.xyz,q.w),v);}
fn svoThickGlassEmpty(status:u32)->SvoThickGlassInterval{let surface=SvoThickGlassSurface(0.0,0u,vec2u(0u),vec3f(0.0),0.0,vec3f(0.0,1.0,0.0),0.0);return SvoThickGlassInterval(status,0u,0u,0u,0.0,vec3f(0.0),surface,surface);}
fn svoThickGlassSurfaceAt(record:SvoThickGlassRecord,origin:vec3f,direction:vec3f,t_m:f32)->SvoThickGlassSurface{let position=origin+direction*t_m;let q=normalize(record.orientation);let local=svoThickGlassInverseRotate(q,position-record.centerRadiusX.xyz);let radii=vec3f(record.centerRadiusX.w,record.radiiYzIorEpsilon.xy);let gradient=local/(radii*radii);let normal=normalize(svoThickGlassRotate(q,gradient));return SvoThickGlassSurface(t_m,select(0u,1u,dot(direction,normal)<0.0),vec2u(0u),position,0.0,normal,0.0);}
fn svoThickGlassIntersect(record:SvoThickGlassRecord,origin:vec3f,directionIn:vec3f,tMin_m:f32,tMax_m:f32,expectedRevision:u32)->SvoThickGlassInterval{let radii=vec3f(record.centerRadiusX.w,record.radiiYzIorEpsilon.xy);let shape=record.identity.w&3u;let qLength=dot(record.orientation,record.orientation);let sphereMismatch=shape==SVO_THICK_GLASS_SPHERE&&(max(max(radii.x,radii.y),radii.z)-min(min(radii.x,radii.y),radii.z)>max(max(radii.x,radii.y),radii.z)*1e-6);if(expectedRevision==0u||record.identity.z!=expectedRevision){return svoThickGlassEmpty(select(SVO_THICK_GLASS_STALE,SVO_THICK_GLASS_INVALID,expectedRevision==0u));}if((shape!=SVO_THICK_GLASS_SPHERE&&shape!=SVO_THICK_GLASS_ELLIPSOID)||sphereMismatch||(record.identity.w&SVO_THICK_GLASS_ANALYTIC_FINAL)==0u||record.identity.x==0u||record.identity.y==0u||any(radii<=vec3f(0.0))||record.radiiYzIorEpsilon.z<1.0||record.radiiYzIorEpsilon.w<=0.0||record.absorptionPath.w<=0.0||any(record.absorptionPath.xyz<vec3f(0.0))||!svoThickGlassFinite3(record.centerRadiusX.xyz)||!svoThickGlassFinite3(radii)||!svoThickGlassFinite(qLength)||qLength<1e-12||!svoThickGlassFinite3(directionIn)||dot(directionIn,directionIn)<1e-18||!svoThickGlassFinite(tMin_m)||!svoThickGlassFinite(tMax_m)||tMax_m<tMin_m){return svoThickGlassEmpty(SVO_THICK_GLASS_INVALID);}let direction=normalize(directionIn);let q=record.orientation*inverseSqrt(qLength);let localOrigin=svoThickGlassInverseRotate(q,origin-record.centerRadiusX.xyz);let localDirection=svoThickGlassInverseRotate(q,direction);let scaledOrigin=localOrigin/radii;let scaledDirection=localDirection/radii;let a=dot(scaledDirection,scaledDirection);let halfB=dot(scaledOrigin,scaledDirection);let c=dot(scaledOrigin,scaledOrigin)-1.0;let discriminant=halfB*halfB-a*c;if(discriminant<0.0){return svoThickGlassEmpty(SVO_THICK_GLASS_MISS);}let root=sqrt(max(0.0,discriminant));let nearT=(-halfB-root)/a;let farT=(-halfB+root)/a;let minimumForward=max(tMin_m,record.radiiYzIorEpsilon.w);if(farT<=minimumForward||nearT>tMax_m){return svoThickGlassEmpty(SVO_THICK_GLASS_MISS);}let entry=svoThickGlassSurfaceAt(record,origin,direction,nearT);let exit=svoThickGlassSurfaceAt(record,origin,direction,farT);let hasEntry=nearT>minimumForward;let firstT=select(farT,nearT,hasEntry);if(firstT>tMax_m){return svoThickGlassEmpty(SVO_THICK_GLASS_MISS);}let tangent=root<=record.radiiYzIorEpsilon.w*sqrt(a);let path=select(min(record.absorptionPath.w,max(0.0,farT-max(max(nearT,tMin_m),0.0))),0.0,tangent);return SvoThickGlassInterval(SVO_THICK_GLASS_HIT,select(0u,1u,c<0.0),select(0u,1u,tangent),select(0u,1u,hasEntry),path,vec3f(0.0),entry,exit);}
fn svoThickGlassInterface(record:SvoThickGlassRecord,surface:SvoThickGlassSurface,incidentIn:vec3f,fromIor:f32,toIor:f32,opticalPath_m:f32)->SvoThickGlassInterface{let incident=normalize(incidentIn);var normal=surface.normal;if(dot(incident,normal)>0.0){normal=-normal;}let eta=fromIor/toIor;let cosine=clamp(-dot(incident,normal),0.0,1.0);let discriminant=1.0-eta*eta*(1.0-cosine*cosine);let reflected=normalize(incident-2.0*dot(incident,normal)*normal);let ratio=(fromIor-toIor)/(fromIor+toIor);let f0=ratio*ratio;let tint=exp(-max(record.absorptionPath.xyz,vec3f(0.0))*max(opticalPath_m,0.0));if(discriminant<0.0){return SvoThickGlassInterface(reflected,1.0,vec3f(0.0),1u,tint,0.0);}let refracted=normalize(eta*incident+(eta*cosine-sqrt(max(0.0,discriminant)))*normal);let fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);return SvoThickGlassInterface(reflected,fresnel,refracted,0u,tint,0.0);}
fn svoThickGlassMaterialId(record:SvoThickGlassRecord)->u32{return record.identity.y&0xffffu;}fn svoThickGlassOwnerId(record:SvoThickGlassRecord)->u32{return record.identity.y>>16u;}fn svoThickGlassId(record:SvoThickGlassRecord)->u32{return record.identity.x;}
`;
