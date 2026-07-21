import { encodeSvoTemporalNormal, packSvoTemporalHitKey, type SvoTemporalHitKey } from "./svo-temporal-history";
import type { SvoVec3 } from "./webgpu-svo-traversal";

export type SvoGBufferFormat = "rgba16float" | "rgba16uint" | "rgba32uint" | "depth32float";

/** Core color attachments meet WebGPU's baseline 32-byte/sample limit exactly. */
export const SVO_GBUFFER_COLOR_ATTACHMENT_COUNT = 3;
export const SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE = 32;
export const SVO_GBUFFER_BYTES_PER_PIXEL = 36;
export const SVO_GBUFFER_DEBUG_SIDECAR_BYTES_PER_PIXEL = 16;

/** Fixed production order. The debug sidecar is never a simultaneous core MRT. */
export const SVO_GBUFFER_LAYOUT = Object.freeze({
  radianceDepth: {
    location: 0, format: "rgba16float" as SvoGBufferFormat, bytes: 8,
    encoding: "scene-linear HDR RGB; W is linear metres along the normalized primary ray, zero on miss",
  },
  packedSurface: {
    location: 1, format: "rgba32uint" as SvoGBufferFormat, bytes: 16,
    encoding: "two oct8 normals, exact local generation, signed 10:10:10 velocity plus motion kind, packed source/flags/failure/feature",
  },
  identityMedia: {
    location: 2, format: "rgba16uint" as SvoGBufferFormat, bytes: 8,
    encoding: "exact material, owner, medium-before, medium-after uint16 IDs",
  },
  hardwareDepth: {
    format: "depth32float" as SvoGBufferFormat, bytes: 4,
    encoding: "reversed-Z device depth; clear 0 and compare greater",
  },
  debugSidecar: {
    format: "rgba32uint" as SvoGBufferFormat, bytes: 16,
    encoding: "reserved optional storage record for diagnostic-only passes",
  },
} as const);

export const SVO_GBUFFER_MAX_VELOCITY_M_S = 64;
export const SVO_GBUFFER_PRECISION = Object.freeze({
  halfMaximumFinite: 65_504,
  halfMinimumSubnormal: 2 ** -24,
  halfRelativeTolerance: 2 ** -10,
  maximumLinearDepth_m: 65_504,
  maximumNormalAngularError_deg: 1.5,
  maximumVelocity_m_s: SVO_GBUFFER_MAX_VELOCITY_M_S,
  maximumVelocityAbsoluteError_m_s: SVO_GBUFFER_MAX_VELOCITY_M_S / (2 * 511),
  identityAndLocalGenerationAreExact: true,
  missLinearDepth_m: 0,
  missHardwareDepth: 0,
} as const);

export const SVO_GBUFFER_FIELD_SOURCES = Object.freeze({
  none: 0, structuralDiscrete: 1,
  analyticPrimitive: 4, terrainHeightfield: 5, rasterFallback: 6,
} as const);
export type SvoGBufferFieldSource = typeof SVO_GBUFFER_FIELD_SOURCES[keyof typeof SVO_GBUFFER_FIELD_SOURCES];

export const SVO_GBUFFER_MOTION_KINDS = Object.freeze({ static: 0, rigid: 1 } as const);
export type SvoGBufferMotionKind = typeof SVO_GBUFFER_MOTION_KINDS[keyof typeof SVO_GBUFFER_MOTION_KINDS];

/** These sixteen bits occupy metadata bits 4..19. */
export const SVO_GBUFFER_FLAGS = Object.freeze({
  validSurface: 1 << 0, miss: 1 << 1, depthValid: 1 << 2,
  geometricNormalValid: 1 << 3, shadingNormalValid: 1 << 4,
  motionValid: 1 << 5, mediaValid: 1 << 6,
  hardFeature: 1 << 8,
  workExhausted: 1 << 12, invalidField: 1 << 13,
  staleGeneration: 1 << 14, nonresident: 1 << 15,
} as const);

export const SVO_GBUFFER_FAILURES = Object.freeze({
  none: 0, noIntersection: 1, workExhausted: 2, invalidField: 3,
  staleGeneration: 4, nonresident: 5, invalidRay: 6, mediaStack: 7,
} as const);
export type SvoGBufferFailure = typeof SVO_GBUFFER_FAILURES[keyof typeof SVO_GBUFFER_FAILURES];

/** Four metadata bits are available to preserve the authored local feature. */
export const SVO_GBUFFER_FEATURES = Object.freeze({ smooth: 0, boxFaceX: 1, boxFaceY: 2, boxFaceZ: 3 } as const);

export interface SvoGBufferHit {
  status: "hit";
  radianceLinear: SvoVec3;
  depth_m: number;
  geometricNormal: SvoVec3;
  shadingNormal: SvoVec3;
  materialId: number;
  ownerId: number;
  mediumBefore: number;
  mediumAfter: number;
  velocity_m_s: SvoVec3;
  motionKind: SvoGBufferMotionKind;
  motionValid: boolean;
  fieldSource: SvoGBufferFieldSource;
  localTopologyGeneration: number;
  featureId: number;
  /** Diagnostic flags only; the packer supplies canonical validity bits. */
  additionalFlags?: number;
}

export interface SvoGBufferMiss {
  status: "miss";
  radianceLinear: SvoVec3;
  fieldSource?: SvoGBufferFieldSource;
  localTopologyGeneration?: number;
  failure?: SvoGBufferFailure;
  additionalFlags?: number;
}
export type SvoGBufferPixel = SvoGBufferHit | SvoGBufferMiss;

/** Exact CPU mirror of three core MRTs and the optional non-MRT sidecar. */
export interface SvoPackedGBufferPixel {
  radianceDepth: Uint16Array<ArrayBuffer>;
  packedSurface: Uint32Array<ArrayBuffer>;
  identityMedia: Uint16Array<ArrayBuffer>;
  debugSidecar: Uint32Array<ArrayBuffer>;
}

const UINT16_MAX = 0xffff;
const UINT32_MAX = 0xffff_ffff;

function uint(value: number, maximum: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > maximum) throw new RangeError(`${label} must be an integer from 0 to ${maximum}`);
  return value;
}

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) throw new RangeError(`${label} must contain three finite components`);
}

function normalized(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const magnitude = Math.hypot(...value);
  if (!(magnitude > 1e-12)) throw new RangeError(`${label} must have nonzero length`);
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function roundToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

/** IEEE-754 binary16, round-to-nearest-even, saturating finite overflow. */
export function encodeSvoGBufferFloat16(value: number): number {
  if (Number.isNaN(value)) throw new RangeError("G-buffer half value must not be NaN");
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const magnitude = Math.min(Math.abs(value), SVO_GBUFFER_PRECISION.halfMaximumFinite);
  if (magnitude < SVO_GBUFFER_PRECISION.halfMinimumSubnormal / 2) return sign;
  if (magnitude < 2 ** -14) return sign | Math.min(0x3ff, roundToEven(magnitude / 2 ** -24));
  let exponent = Math.floor(Math.log2(magnitude));
  let mantissa = roundToEven((magnitude / 2 ** exponent - 1) * 1024);
  if (mantissa === 1024) { exponent += 1; mantissa = 0; }
  if (exponent > 15) return sign | 0x7bff;
  return sign | (exponent + 15 << 10) | mantissa;
}

export function decodeSvoGBufferFloat16(bits: number): number {
  const word = uint(bits, UINT16_MAX, "G-buffer half bits");
  const sign = (word & 0x8000) === 0 ? 1 : -1;
  const exponent = word >>> 10 & 0x1f;
  const mantissa = word & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function signNotZero(value: number): number { return value < 0 ? -1 : 1; }

function octCoordinates(normalInput: SvoVec3): readonly [number, number] {
  const normal = normalized(normalInput, "G-buffer normal");
  const inverseL1 = 1 / (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]));
  let x = normal[0] * inverseL1;
  let y = normal[1] * inverseL1;
  if (normal[2] < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * signNotZero(oldX);
    y = (1 - Math.abs(oldX)) * signNotZero(y);
  }
  return [x, y];
}

/** One normal occupies two signed-normalized bytes. */
export function encodeSvoGBufferNormalOct8(normal: SvoVec3): number {
  const encoded = octCoordinates(normal);
  const x = Math.round(Math.max(-1, Math.min(1, encoded[0])) * 127);
  const y = Math.round(Math.max(-1, Math.min(1, encoded[1])) * 127);
  return (y & 0xff) << 8 | x & 0xff;
}

function signed8(value: number): number { const byte = value & 0xff; return byte >= 0x80 ? byte - 0x100 : byte; }

export function decodeSvoGBufferNormalOct8(packed: number): SvoVec3 {
  const word = uint(packed, UINT16_MAX, "Packed oct8 normal");
  let x = signed8(word) / 127;
  let y = signed8(word >>> 8) / 127;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const adjustment = -z;
    x += x < 0 ? adjustment : -adjustment;
    y += y < 0 ? adjustment : -adjustment;
  }
  return normalized([x, y, z], "Decoded G-buffer normal");
}

function encodeSigned10(value: number): number {
  return Math.round(Math.max(-1, Math.min(1, value / SVO_GBUFFER_MAX_VELOCITY_M_S)) * 511) & 0x3ff;
}

function decodeSigned10(value: number): number {
  const bits = value & 0x3ff;
  return (bits >= 0x200 ? bits - 0x400 : bits) / 511 * SVO_GBUFFER_MAX_VELOCITY_M_S;
}

/** Fixed +/-64 m/s signed 10:10:10 velocity; top two bits retain motion kind. */
export function packSvoGBufferVelocity(velocity_m_s: SvoVec3, motionKind: SvoGBufferMotionKind): number {
  finiteVec3(velocity_m_s, "G-buffer velocity");
  return (encodeSigned10(velocity_m_s[0]) | encodeSigned10(velocity_m_s[1]) << 10
    | encodeSigned10(velocity_m_s[2]) << 20 | uint(motionKind, 3, "Motion kind") << 30) >>> 0;
}

export function unpackSvoGBufferVelocity(packed: number): { velocity_m_s: SvoVec3; motionKind: SvoGBufferMotionKind } {
  const word = uint(packed, UINT32_MAX, "Packed G-buffer velocity");
  return {
    velocity_m_s: [decodeSigned10(word), decodeSigned10(word >>> 10), decodeSigned10(word >>> 20)],
    motionKind: (word >>> 30) as SvoGBufferMotionKind,
  };
}

function failureFlag(failure: SvoGBufferFailure): number {
  if (failure === SVO_GBUFFER_FAILURES.workExhausted) return SVO_GBUFFER_FLAGS.workExhausted;
  if (failure === SVO_GBUFFER_FAILURES.invalidField) return SVO_GBUFFER_FLAGS.invalidField;
  if (failure === SVO_GBUFFER_FAILURES.staleGeneration) return SVO_GBUFFER_FLAGS.staleGeneration;
  if (failure === SVO_GBUFFER_FAILURES.nonresident) return SVO_GBUFFER_FLAGS.nonresident;
  return 0;
}

function packMetadata(source: number, flags: number, failure: number, featureId: number): number {
  return (uint(source, 15, "G-buffer field source") | uint(flags, UINT16_MAX, "G-buffer flags") << 4
    | uint(failure, 255, "G-buffer failure") << 20 | uint(featureId, 15, "Feature ID") << 28) >>> 0;
}

function unpackMetadata(word: number) {
  return {
    fieldSource: (word & 0xf) as SvoGBufferFieldSource,
    flags: word >>> 4 & UINT16_MAX,
    failure: (word >>> 20 & 0xff) as SvoGBufferFailure,
    featureId: word >>> 28,
  };
}

export function packSvoGBufferPixel(pixel: SvoGBufferPixel): SvoPackedGBufferPixel {
  finiteVec3(pixel.radianceLinear, "G-buffer radiance");
  if (pixel.radianceLinear.some((channel) => channel < 0)) throw new RangeError("G-buffer radiance must be non-negative");
  const radianceDepth = new Uint16Array(4);
  radianceDepth.set(pixel.radianceLinear.map(encodeSvoGBufferFloat16));
  const packedSurface = new Uint32Array(4);
  const identityMedia = new Uint16Array(4);
  const debugSidecar = new Uint32Array(4);

  if (pixel.status === "miss") {
    const failure = uint(pixel.failure ?? SVO_GBUFFER_FAILURES.noIntersection, 255, "G-buffer failure") as SvoGBufferFailure;
    const flags = SVO_GBUFFER_FLAGS.miss | failureFlag(failure) | (pixel.additionalFlags ?? 0);
    packedSurface[1] = uint(pixel.localTopologyGeneration ?? 0, UINT32_MAX, "Local topology generation");
    packedSurface[3] = packMetadata(pixel.fieldSource ?? SVO_GBUFFER_FIELD_SOURCES.none, flags, failure, 0);
    return { radianceDepth, packedSurface, identityMedia, debugSidecar };
  }

  if (!Number.isFinite(pixel.depth_m) || pixel.depth_m < 0 || pixel.depth_m > SVO_GBUFFER_PRECISION.maximumLinearDepth_m) {
    throw new RangeError(`G-buffer depth must be finite and from zero to ${SVO_GBUFFER_PRECISION.maximumLinearDepth_m} metres`);
  }
  radianceDepth[3] = encodeSvoGBufferFloat16(pixel.depth_m);
  packedSurface[0] = encodeSvoGBufferNormalOct8(pixel.geometricNormal)
    | encodeSvoGBufferNormalOct8(pixel.shadingNormal) << 16;
  packedSurface[1] = uint(pixel.localTopologyGeneration, UINT32_MAX, "Local topology generation");
  packedSurface[2] = packSvoGBufferVelocity(pixel.velocity_m_s, pixel.motionKind);
  identityMedia.set([
    uint(pixel.materialId, UINT16_MAX, "Material ID"), uint(pixel.ownerId, UINT16_MAX, "Owner ID"),
    uint(pixel.mediumBefore, UINT16_MAX, "Medium-before ID"), uint(pixel.mediumAfter, UINT16_MAX, "Medium-after ID"),
  ]);
  const canonicalFlags = SVO_GBUFFER_FLAGS.validSurface | SVO_GBUFFER_FLAGS.depthValid
    | SVO_GBUFFER_FLAGS.geometricNormalValid | SVO_GBUFFER_FLAGS.shadingNormalValid | SVO_GBUFFER_FLAGS.mediaValid
    | (pixel.motionValid ? SVO_GBUFFER_FLAGS.motionValid : 0)
    | (pixel.featureId !== SVO_GBUFFER_FEATURES.smooth ? SVO_GBUFFER_FLAGS.hardFeature : 0);
  packedSurface[3] = packMetadata(pixel.fieldSource, canonicalFlags | (pixel.additionalFlags ?? 0), SVO_GBUFFER_FAILURES.none, pixel.featureId);
  return { radianceDepth, packedSurface, identityMedia, debugSidecar };
}

export function unpackSvoGBufferPixel(packed: SvoPackedGBufferPixel): SvoGBufferPixel {
  if (packed.radianceDepth.length !== 4 || packed.packedSurface.length !== 4
      || packed.identityMedia.length !== 4 || packed.debugSidecar.length !== 4) {
    throw new RangeError("Packed G-buffer attachment lengths do not match the ABI");
  }
  const radianceLinear = packed.radianceDepth.slice(0, 3).map(decodeSvoGBufferFloat16) as unknown as SvoVec3;
  const metadata = unpackMetadata(packed.packedSurface[3]);
  if ((metadata.flags & SVO_GBUFFER_FLAGS.validSurface) === 0) {
    return {
      status: "miss", radianceLinear, fieldSource: metadata.fieldSource,
      localTopologyGeneration: packed.packedSurface[1], failure: metadata.failure,
      additionalFlags: metadata.flags,
    };
  }
  const velocity = unpackSvoGBufferVelocity(packed.packedSurface[2]);
  return {
    status: "hit", radianceLinear, depth_m: decodeSvoGBufferFloat16(packed.radianceDepth[3]),
    geometricNormal: decodeSvoGBufferNormalOct8(packed.packedSurface[0] & UINT16_MAX),
    shadingNormal: decodeSvoGBufferNormalOct8(packed.packedSurface[0] >>> 16),
    materialId: packed.identityMedia[0], ownerId: packed.identityMedia[1],
    mediumBefore: packed.identityMedia[2], mediumAfter: packed.identityMedia[3],
    velocity_m_s: velocity.velocity_m_s, motionKind: velocity.motionKind,
    motionValid: (metadata.flags & SVO_GBUFFER_FLAGS.motionValid) !== 0,
    fieldSource: metadata.fieldSource, localTopologyGeneration: packed.packedSurface[1],
    featureId: metadata.featureId, additionalFlags: metadata.flags,
  };
}

export function reconstructSvoGBufferWorldPosition(origin_m: SvoVec3, rayDirection: SvoVec3, depth_m: number): SvoVec3 {
  finiteVec3(origin_m, "Ray origin");
  const direction = normalized(rayDirection, "Ray direction");
  if (!Number.isFinite(depth_m) || depth_m < 0) throw new RangeError("Reconstruction depth must be non-negative and finite");
  return [origin_m[0] + direction[0] * depth_m, origin_m[1] + direction[1] * depth_m, origin_m[2] + direction[2] * depth_m];
}

export interface SvoGBufferFeatureNormal { normal: SvoVec3; featureId: number }

/** Stable X -> Y -> Z ties retain one authored face at box edges and corners. */
export function svoGBufferHardBoxFeatureNormal(localPosition_m: SvoVec3, halfExtents_m: SvoVec3): SvoGBufferFeatureNormal {
  finiteVec3(localPosition_m, "Box position");
  finiteVec3(halfExtents_m, "Box half extents");
  if (halfExtents_m.some((extent) => extent <= 0)) throw new RangeError("Box half extents must be positive");
  const q = localPosition_m.map((value, axis) => Math.abs(value) - halfExtents_m[axis]) as [number, number, number];
  let axis = 0;
  if (q[1] > q[0]) axis = 1;
  if (q[2] > q[axis]) axis = 2;
  const normal: [number, number, number] = [0, 0, 0];
  normal[axis] = signNotZero(localPosition_m[axis]);
  return { normal, featureId: SVO_GBUFFER_FEATURES.boxFaceX + axis };
}

/** Existing six-word/24-byte temporal key, built from quantized core MRT values. */
export function makeSvoGBufferTemporalKey(pixel: SvoGBufferHit): Uint32Array<ArrayBuffer> {
  const quantized = unpackSvoGBufferPixel(packSvoGBufferPixel(pixel));
  if (quantized.status !== "hit") throw new Error("A temporal key requires a valid G-buffer hit");
  const temporal: SvoTemporalHitKey = {
    depth_m: quantized.depth_m, geometricNormal: quantized.geometricNormal, shadingNormal: quantized.shadingNormal,
    materialId: quantized.materialId, ownerId: quantized.ownerId,
    mediumBefore: quantized.mediumBefore, mediumAfter: quantized.mediumAfter,
    localTopologyGeneration: quantized.localTopologyGeneration,
  };
  return packSvoTemporalHitKey(temporal);
}

export function makeSvoGBufferTemporalNormalWord(normal: SvoVec3): number { return encodeSvoTemporalNormal(normal); }

/** Binding-free three-MRT fragment output and matching decode/key helpers. */
export const svoGBufferWGSL = /* wgsl */ `
const SVO_GBUFFER_VALID_SURFACE:u32=1u;const SVO_GBUFFER_MISS:u32=2u;const SVO_GBUFFER_DEPTH_VALID:u32=4u;
const SVO_GBUFFER_GEOMETRIC_NORMAL_VALID:u32=8u;const SVO_GBUFFER_SHADING_NORMAL_VALID:u32=16u;
const SVO_GBUFFER_MOTION_VALID:u32=32u;const SVO_GBUFFER_MEDIA_VALID:u32=64u;const SVO_GBUFFER_HARD_FEATURE:u32=256u;
const SVO_GBUFFER_FIELD_NONE:u32=0u;const SVO_GBUFFER_FEATURE_SMOOTH:u32=0u;const SVO_GBUFFER_FEATURE_BOX_X:u32=1u;
const SVO_GBUFFER_MAX_VELOCITY_M_S:f32=64.0;
struct SvoGBufferTargets{@location(0) radianceDepth:vec4f,@location(1) packedSurface:vec4u,@location(2) identityMedia:vec4u}
struct SvoGBufferFeatureNormal{normal:vec3f,featureId:u32}
struct SvoGBufferTemporalKey{depth_m:f32,geometricNormalOct:u32,shadingNormalOct:u32,materialOwner:u32,media:u32,localTopologyGeneration:u32}
fn svoGBufferSignNotZero(value:f32)->f32{return select(1.0,-1.0,value<0.0);}
fn svoGBufferOctCoordinates(normalIn:vec3f)->vec2f{let normal=normalize(normalIn);var result=normal.xy/(abs(normal.x)+abs(normal.y)+abs(normal.z));if(normal.z<0.0){let old=result;result=vec2f((1.0-abs(old.y))*svoGBufferSignNotZero(old.x),(1.0-abs(old.x))*svoGBufferSignNotZero(old.y));}return result;}
fn svoGBufferPackNormalOct8(normal:vec3f)->u32{let encoded=svoGBufferOctCoordinates(normal);let x=i32(round(clamp(encoded.x,-1.0,1.0)*127.0));let y=i32(round(clamp(encoded.y,-1.0,1.0)*127.0));return ((u32(y)&0xffu)<<8u)|(u32(x)&0xffu);}
fn svoGBufferUnpackNormalOct8(packed:u32)->vec3f{var result=vec3f(f32(i32(packed<<24u)>>24)/127.0,f32(i32(packed<<16u)>>24)/127.0,0.0);result.z=1.0-abs(result.x)-abs(result.y);if(result.z<0.0){let adjustment=-result.z;result.x+=select(-adjustment,adjustment,result.x<0.0);result.y+=select(-adjustment,adjustment,result.y<0.0);}return normalize(result);}
fn svoGBufferPackVelocity(velocity_m_s:vec3f,motionKind:u32)->u32{let scaled=vec3i(round(clamp(velocity_m_s/SVO_GBUFFER_MAX_VELOCITY_M_S,vec3f(-1.0),vec3f(1.0))*511.0));return (u32(scaled.x)&0x3ffu)|((u32(scaled.y)&0x3ffu)<<10u)|((u32(scaled.z)&0x3ffu)<<20u)|((motionKind&3u)<<30u);}
fn svoGBufferMetadata(fieldSource:u32,flags:u32,failure:u32,featureId:u32)->u32{return (fieldSource&15u)|((flags&0xffffu)<<4u)|((failure&255u)<<20u)|((featureId&15u)<<28u);}
fn svoGBufferReconstructWorld(origin_m:vec3f,rayDirection:vec3f,linearDepth_m:f32)->vec3f{return origin_m+normalize(rayDirection)*linearDepth_m;}
fn svoGBufferHardBoxFeatureNormal(localPosition_m:vec3f,halfExtents_m:vec3f)->SvoGBufferFeatureNormal{let q=abs(localPosition_m)-halfExtents_m;var axis=0u;if(q.y>q.x){axis=1u;}if(q.z>q[axis]){axis=2u;}var normal=vec3f(0.0);normal[axis]=svoGBufferSignNotZero(localPosition_m[axis]);return SvoGBufferFeatureNormal(normal,SVO_GBUFFER_FEATURE_BOX_X+axis);}
fn svoGBufferPackTemporalNormal(normal:vec3f)->u32{let encoded=svoGBufferOctCoordinates(normal);let x=i32(round(clamp(encoded.x,-1.0,1.0)*32767.0));let y=i32(round(clamp(encoded.y,-1.0,1.0)*32767.0));return ((u32(y)&0xffffu)<<16u)|(u32(x)&0xffffu);}
fn svoGBufferTemporalKey(targets:SvoGBufferTargets)->SvoGBufferTemporalKey{let normals=targets.packedSurface.x;let geometric=svoGBufferUnpackNormalOct8(normals&0xffffu);let shading=svoGBufferUnpackNormalOct8(normals>>16u);return SvoGBufferTemporalKey(targets.radianceDepth.w,svoGBufferPackTemporalNormal(geometric),svoGBufferPackTemporalNormal(shading),(targets.identityMedia.y<<16u)|(targets.identityMedia.x&0xffffu),(targets.identityMedia.w<<16u)|(targets.identityMedia.z&0xffffu),targets.packedSurface.y);}
fn svoGBufferMiss(radianceLinear:vec3f,fieldSource:u32,localGeneration:u32,failure:u32,flags:u32)->SvoGBufferTargets{return SvoGBufferTargets(vec4f(max(radianceLinear,vec3f(0.0)),0.0),vec4u(0u,localGeneration,0u,svoGBufferMetadata(fieldSource,flags|SVO_GBUFFER_MISS,failure,0u)),vec4u(0u));}
fn svoGBufferSurface(radianceLinear:vec3f,linearDepth_m:f32,geometricNormal:vec3f,shadingNormal:vec3f,identityMedia:vec4u,velocity_m_s:vec3f,motionKind:u32,fieldSource:u32,localGeneration:u32,flags:u32,featureId:u32)->SvoGBufferTargets{let canonical=SVO_GBUFFER_VALID_SURFACE|SVO_GBUFFER_DEPTH_VALID|SVO_GBUFFER_GEOMETRIC_NORMAL_VALID|SVO_GBUFFER_SHADING_NORMAL_VALID|SVO_GBUFFER_MEDIA_VALID;let normals=svoGBufferPackNormalOct8(geometricNormal)|(svoGBufferPackNormalOct8(shadingNormal)<<16u);return SvoGBufferTargets(vec4f(max(radianceLinear,vec3f(0.0)),linearDepth_m),vec4u(normals,localGeneration,svoGBufferPackVelocity(velocity_m_s,motionKind),svoGBufferMetadata(fieldSource,flags|canonical,0u,featureId)),identityMedia);}
`;
