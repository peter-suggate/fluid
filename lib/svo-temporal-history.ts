import type { SvoVec3 } from "./webgpu-svo-traversal";

/** depth, geometric normal, shading normal, material/owner, media, local generation. */
export const SVO_TEMPORAL_HIT_KEY_WORDS = 6;

export const SVO_TEMPORAL_HISTORY_THRESHOLDS = Object.freeze({
  minimumDepthTolerance_m: 0.002,
  depthTolerance_cell: 0.2,
  minimumReprojectionTolerance_m: 0.001,
  reprojectionTolerance_cell: 0.15,
  minimumGeometricNormalDot: Math.cos(20 * Math.PI / 180),
  minimumShadingNormalDot: Math.cos(35 * Math.PI / 180),
  staticMotionTolerance_m: 1e-6,
  maximumRigidMotion_m: 0.5,
  maximumRigidMotion_cell: 2,
  maximumFluidMotion_m: 0.25,
  maximumFluidMotion_cell: 1,
  maximumHistorySamples: 255,
  pausedConvergenceSamples: 64,
} as const);

export interface SvoTemporalHitKey {
  depth_m: number;
  geometricNormal: SvoVec3;
  shadingNormal: SvoVec3;
  materialId: number;
  ownerId: number;
  mediumBefore: number;
  mediumAfter: number;
  /** Generation sampled at this hit's topology locality, not a scene-global revision. */
  localTopologyGeneration: number;
}

export type SvoTemporalMotionKind = "static" | "rigid" | "fluid";

export interface SvoTemporalReprojection {
  cellSize_m: number;
  deltaTime_s: number;
  velocity_m_s: SvoVec3;
  motionKind: SvoTemporalMotionKind;
  /** False when the previous pixel lies outside the viewport or behind the camera. */
  reprojectionValid: boolean;
  /** False when a rigid transform or fluid velocity needed by reprojection was unavailable/stale. */
  motionValid: boolean;
  /** World-space error between predicted and reconstructed previous positions. */
  reprojectionError_m: number;
}

export type SvoTemporalHistoryRejectionReason =
  | "accepted"
  | "invalid-current-hit"
  | "invalid-previous-hit"
  | "invalid-reprojection"
  | "invalid-motion"
  | "excessive-motion"
  | "topology-generation-change"
  | "material-change"
  | "owner-change"
  | "medium-change"
  | "disocclusion"
  | "geometric-normal-change"
  | "shading-normal-change";

export interface SvoTemporalHistoryDecision {
  accepted: boolean;
  reason: SvoTemporalHistoryRejectionReason;
  depthDelta_m: number;
  depthTolerance_m: number;
  reprojectionTolerance_m: number;
  motionDistance_m: number;
  motionLimit_m: number;
}

export interface SvoTemporalConvergenceCounters {
  /** Number of samples represented by history, saturated for compact GPU storage. */
  sampleCount: number;
  /** Consecutive accepted paused frames; resets whenever animation resumes or history rejects. */
  pausedStableFrames: number;
  convergedWhilePaused: boolean;
}

const UINT16_MAX = 0xffff;
const UINT32_MAX = 0xffff_ffff;
const NORMAL_EPSILON = 1e-12;

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function normalized(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const length = Math.hypot(...value);
  if (!(length > NORMAL_EPSILON)) throw new RangeError(`${label} must have nonzero length`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function uint16(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT16_MAX) throw new RangeError(`${label} must fit uint16`);
  return value;
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function signNotZero(value: number): number {
  return value < 0 ? -1 : 1;
}

/** Pack a unit vector into two signed-normalized 16-bit octahedral coordinates. */
export function encodeSvoTemporalNormal(normalInput: SvoVec3): number {
  const normal = normalized(normalInput, "Temporal normal");
  const inverseL1 = 1 / (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]));
  let x = normal[0] * inverseL1;
  let y = normal[1] * inverseL1;
  if (normal[2] < 0) {
    const previousX = x;
    x = (1 - Math.abs(y)) * signNotZero(previousX);
    y = (1 - Math.abs(previousX)) * signNotZero(y);
  }
  const encodedX = Math.round(Math.max(-1, Math.min(1, x)) * 32767);
  const encodedY = Math.round(Math.max(-1, Math.min(1, y)) * 32767);
  return ((encodedY & UINT16_MAX) << 16 | (encodedX & UINT16_MAX)) >>> 0;
}

function signed16(word: number): number {
  const value = word & UINT16_MAX;
  return value >= 0x8000 ? value - 0x1_0000 : value;
}

/** Decode the compact normal used by current/previous temporal keys. */
export function decodeSvoTemporalNormal(packed: number): SvoVec3 {
  uint32(packed, "Packed temporal normal");
  let x = signed16(packed) / 32767;
  let y = signed16(packed >>> 16) / 32767;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const adjustment = -z;
    x += x >= 0 ? -adjustment : adjustment;
    y += y >= 0 ? -adjustment : adjustment;
  }
  return normalized([x, y, z], "Decoded temporal normal");
}

/** Pack a validated current/previous key into six words (24 bytes). */
export function packSvoTemporalHitKey(key: SvoTemporalHitKey): Uint32Array<ArrayBuffer> {
  if (!Number.isFinite(key.depth_m) || key.depth_m < 0) throw new RangeError("Temporal depth must be non-negative and finite");
  const buffer = new ArrayBuffer(SVO_TEMPORAL_HIT_KEY_WORDS * Uint32Array.BYTES_PER_ELEMENT);
  const words = new Uint32Array(buffer);
  new Float32Array(buffer)[0] = key.depth_m;
  words[1] = encodeSvoTemporalNormal(key.geometricNormal);
  words[2] = encodeSvoTemporalNormal(key.shadingNormal);
  words[3] = uint16(key.materialId, "Temporal material ID") | (uint16(key.ownerId, "Temporal owner ID") << 16);
  words[4] = uint16(key.mediumBefore, "Temporal entry medium") | (uint16(key.mediumAfter, "Temporal exit medium") << 16);
  words[5] = uint32(key.localTopologyGeneration, "Temporal local topology generation");
  return words;
}

/** Unpack one six-word temporal key. The copied input prevents buffer-offset aliasing. */
export function unpackSvoTemporalHitKey(packed: Uint32Array): SvoTemporalHitKey {
  if (packed.length !== SVO_TEMPORAL_HIT_KEY_WORDS) throw new RangeError(`Temporal hit key must contain ${SVO_TEMPORAL_HIT_KEY_WORDS} words`);
  const words = new Uint32Array(packed);
  const depth_m = new Float32Array(words.buffer)[0];
  if (!Number.isFinite(depth_m) || depth_m < 0) throw new RangeError("Packed temporal depth must be non-negative and finite");
  return {
    depth_m,
    geometricNormal: decodeSvoTemporalNormal(words[1]),
    shadingNormal: decodeSvoTemporalNormal(words[2]),
    materialId: words[3] & UINT16_MAX,
    ownerId: words[3] >>> 16,
    mediumBefore: words[4] & UINT16_MAX,
    mediumAfter: words[4] >>> 16,
    localTopologyGeneration: words[5],
  };
}

function usableHit(key: SvoTemporalHitKey): boolean {
  if (!Number.isFinite(key.depth_m) || key.depth_m < 0) return false;
  try {
    normalized(key.geometricNormal, "Temporal geometric normal");
    normalized(key.shadingNormal, "Temporal shading normal");
    uint16(key.materialId, "Temporal material ID");
    uint16(key.ownerId, "Temporal owner ID");
    uint16(key.mediumBefore, "Temporal entry medium");
    uint16(key.mediumAfter, "Temporal exit medium");
    uint32(key.localTopologyGeneration, "Temporal local topology generation");
    return true;
  } catch {
    return false;
  }
}

function dotNormalized(a: SvoVec3, b: SvoVec3): number {
  const first = normalized(a, "Temporal normal");
  const second = normalized(b, "Temporal normal");
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

function validateReprojection(input: SvoTemporalReprojection): void {
  if (!Number.isFinite(input.cellSize_m) || input.cellSize_m <= 0) throw new RangeError("Temporal cell size must be finite and positive");
  if (!Number.isFinite(input.deltaTime_s) || input.deltaTime_s < 0) throw new RangeError("Temporal delta time must be non-negative and finite");
  finiteVec3(input.velocity_m_s, "Temporal velocity");
  if (!Number.isFinite(input.reprojectionError_m) || input.reprojectionError_m < 0) {
    throw new RangeError("Temporal reprojection error must be non-negative and finite");
  }
}

function motionLimit(input: SvoTemporalReprojection): number {
  const thresholds = SVO_TEMPORAL_HISTORY_THRESHOLDS;
  if (input.motionKind === "static") return thresholds.staticMotionTolerance_m;
  if (input.motionKind === "rigid") return Math.min(thresholds.maximumRigidMotion_m, thresholds.maximumRigidMotion_cell * input.cellSize_m);
  return Math.min(thresholds.maximumFluidMotion_m, thresholds.maximumFluidMotion_cell * input.cellSize_m);
}

function decision(
  reason: SvoTemporalHistoryRejectionReason,
  metrics: Omit<SvoTemporalHistoryDecision, "accepted" | "reason">,
): SvoTemporalHistoryDecision {
  return { accepted: reason === "accepted", reason, ...metrics };
}

/**
 * Deterministic history acceptance contract. Previous depth is assumed to have
 * already been sampled at the motion-reprojected coordinate.
 */
export function evaluateSvoTemporalHistory(
  current: SvoTemporalHitKey,
  previous: SvoTemporalHitKey,
  reprojection: SvoTemporalReprojection,
): SvoTemporalHistoryDecision {
  validateReprojection(reprojection);
  const thresholds = SVO_TEMPORAL_HISTORY_THRESHOLDS;
  const depthDelta_m = Math.abs(current.depth_m - previous.depth_m);
  const depthTolerance_m = Math.max(thresholds.minimumDepthTolerance_m, thresholds.depthTolerance_cell * reprojection.cellSize_m);
  const reprojectionTolerance_m = Math.max(
    thresholds.minimumReprojectionTolerance_m,
    thresholds.reprojectionTolerance_cell * reprojection.cellSize_m,
  );
  const motionDistance_m = Math.hypot(...reprojection.velocity_m_s) * reprojection.deltaTime_s;
  const motionLimit_m = motionLimit(reprojection);
  const metrics = { depthDelta_m, depthTolerance_m, reprojectionTolerance_m, motionDistance_m, motionLimit_m };

  if (!usableHit(current)) return decision("invalid-current-hit", metrics);
  if (!usableHit(previous)) return decision("invalid-previous-hit", metrics);
  if (!reprojection.reprojectionValid || reprojection.reprojectionError_m > reprojectionTolerance_m) {
    return decision("invalid-reprojection", metrics);
  }
  if (!reprojection.motionValid) return decision("invalid-motion", metrics);
  if (motionDistance_m > motionLimit_m) return decision("excessive-motion", metrics);
  if (current.localTopologyGeneration !== previous.localTopologyGeneration) return decision("topology-generation-change", metrics);
  if (current.materialId !== previous.materialId) return decision("material-change", metrics);
  if (current.ownerId !== previous.ownerId) return decision("owner-change", metrics);
  if (current.mediumBefore !== previous.mediumBefore || current.mediumAfter !== previous.mediumAfter) {
    return decision("medium-change", metrics);
  }
  if (depthDelta_m > depthTolerance_m) return decision("disocclusion", metrics);
  if (dotNormalized(current.geometricNormal, previous.geometricNormal) < thresholds.minimumGeometricNormalDot) {
    return decision("geometric-normal-change", metrics);
  }
  if (dotNormalized(current.shadingNormal, previous.shadingNormal) < thresholds.minimumShadingNormalDot) {
    return decision("shading-normal-change", metrics);
  }
  return decision("accepted", metrics);
}

/** Advance compact accumulation counters without tying convergence to wall-clock timing. */
export function advanceSvoTemporalConvergence(
  previous: SvoTemporalConvergenceCounters,
  historyAccepted: boolean,
  paused: boolean,
): SvoTemporalConvergenceCounters {
  const thresholds = SVO_TEMPORAL_HISTORY_THRESHOLDS;
  const sampleCount = historyAccepted
    ? Math.min(thresholds.maximumHistorySamples, Math.max(1, previous.sampleCount) + 1)
    : 1;
  const pausedStableFrames = paused
    ? historyAccepted ? Math.min(thresholds.maximumHistorySamples, Math.max(1, previous.pausedStableFrames) + 1) : 1
    : 0;
  return {
    sampleCount,
    pausedStableFrames,
    convergedWhilePaused: pausedStableFrames >= thresholds.pausedConvergenceSamples,
  };
}

/** Binding-free WGSL mirror for inclusion in a future temporal resolve shader. */
export const svoTemporalHistoryWGSL = /* wgsl */ `
const SVO_TEMPORAL_REASON_ACCEPTED:u32=0u;
const SVO_TEMPORAL_REASON_INVALID_CURRENT:u32=1u;
const SVO_TEMPORAL_REASON_INVALID_PREVIOUS:u32=2u;
const SVO_TEMPORAL_REASON_INVALID_REPROJECTION:u32=3u;
const SVO_TEMPORAL_REASON_INVALID_MOTION:u32=4u;
const SVO_TEMPORAL_REASON_EXCESSIVE_MOTION:u32=5u;
const SVO_TEMPORAL_REASON_TOPOLOGY_CHANGE:u32=6u;
const SVO_TEMPORAL_REASON_MATERIAL_CHANGE:u32=7u;
const SVO_TEMPORAL_REASON_OWNER_CHANGE:u32=8u;
const SVO_TEMPORAL_REASON_MEDIUM_CHANGE:u32=9u;
const SVO_TEMPORAL_REASON_DISOCCLUSION:u32=10u;
const SVO_TEMPORAL_REASON_GEOMETRIC_NORMAL:u32=11u;
const SVO_TEMPORAL_REASON_SHADING_NORMAL:u32=12u;
const SVO_TEMPORAL_MOTION_STATIC:u32=0u;
const SVO_TEMPORAL_MOTION_RIGID:u32=1u;
const SVO_TEMPORAL_MOTION_FLUID:u32=2u;
const SVO_TEMPORAL_MIN_DEPTH_M:f32=0.002;
const SVO_TEMPORAL_DEPTH_CELL:f32=0.2;
const SVO_TEMPORAL_MIN_REPROJECT_M:f32=0.001;
const SVO_TEMPORAL_REPROJECT_CELL:f32=0.15;
const SVO_TEMPORAL_GEOMETRIC_DOT:f32=0.9396926208;
const SVO_TEMPORAL_SHADING_DOT:f32=0.8191520443;
const SVO_TEMPORAL_STATIC_MOTION_M:f32=0.000001;
const SVO_TEMPORAL_RIGID_MOTION_M:f32=0.5;
const SVO_TEMPORAL_RIGID_MOTION_CELL:f32=2.0;
const SVO_TEMPORAL_FLUID_MOTION_M:f32=0.25;
const SVO_TEMPORAL_FLUID_MOTION_CELL:f32=1.0;

struct SvoTemporalHitKey {
  depth_m:f32,
  geometricNormalOct:u32,
  shadingNormalOct:u32,
  materialOwner:u32,
  media:u32,
  localTopologyGeneration:u32,
}

fn svoTemporalSignNotZero(value:f32)->f32{return select(1.0,-1.0,value<0.0);}
fn svoTemporalDecodeNormal(packed:u32)->vec3f{
  let rawX=i32(packed<<16u)>>16;let rawY=i32(packed)>>16;
  var result=vec3f(f32(rawX)/32767.0,f32(rawY)/32767.0,0.0);
  result.z=1.0-abs(result.x)-abs(result.y);
  if(result.z<0.0){let adjustment=-result.z;result.x+=select(-adjustment,adjustment,result.x<0.0);result.y+=select(-adjustment,adjustment,result.y<0.0);}
  return normalize(result);
}
fn svoTemporalUsableHit(key:SvoTemporalHitKey)->bool{
  return key.depth_m>=0.0&&key.depth_m<3.402823e38&&key.depth_m==key.depth_m;
}
fn svoTemporalMotionLimit(cellSize_m:f32,motionKind:u32)->f32{
  if(motionKind==SVO_TEMPORAL_MOTION_STATIC){return SVO_TEMPORAL_STATIC_MOTION_M;}
  if(motionKind==SVO_TEMPORAL_MOTION_RIGID){return min(SVO_TEMPORAL_RIGID_MOTION_M,SVO_TEMPORAL_RIGID_MOTION_CELL*cellSize_m);}
  return min(SVO_TEMPORAL_FLUID_MOTION_M,SVO_TEMPORAL_FLUID_MOTION_CELL*cellSize_m);
}
fn svoTemporalHistoryReason(
  current:SvoTemporalHitKey,previous:SvoTemporalHitKey,cellSize_m:f32,deltaTime_s:f32,
  velocity_m_s:vec3f,motionKind:u32,reprojectionValid:bool,motionValid:bool,reprojectionError_m:f32
)->u32{
  if(!svoTemporalUsableHit(current)){return SVO_TEMPORAL_REASON_INVALID_CURRENT;}
  if(!svoTemporalUsableHit(previous)){return SVO_TEMPORAL_REASON_INVALID_PREVIOUS;}
  let reprojectionTolerance_m=max(SVO_TEMPORAL_MIN_REPROJECT_M,SVO_TEMPORAL_REPROJECT_CELL*cellSize_m);
  if(!reprojectionValid||reprojectionError_m>reprojectionTolerance_m){return SVO_TEMPORAL_REASON_INVALID_REPROJECTION;}
  if(!motionValid){return SVO_TEMPORAL_REASON_INVALID_MOTION;}
  if(length(velocity_m_s)*deltaTime_s>svoTemporalMotionLimit(cellSize_m,motionKind)){return SVO_TEMPORAL_REASON_EXCESSIVE_MOTION;}
  if(current.localTopologyGeneration!=previous.localTopologyGeneration){return SVO_TEMPORAL_REASON_TOPOLOGY_CHANGE;}
  if((current.materialOwner&0xffffu)!=(previous.materialOwner&0xffffu)){return SVO_TEMPORAL_REASON_MATERIAL_CHANGE;}
  if((current.materialOwner>>16u)!=(previous.materialOwner>>16u)){return SVO_TEMPORAL_REASON_OWNER_CHANGE;}
  if(current.media!=previous.media){return SVO_TEMPORAL_REASON_MEDIUM_CHANGE;}
  let depthTolerance_m=max(SVO_TEMPORAL_MIN_DEPTH_M,SVO_TEMPORAL_DEPTH_CELL*cellSize_m);
  if(abs(current.depth_m-previous.depth_m)>depthTolerance_m){return SVO_TEMPORAL_REASON_DISOCCLUSION;}
  if(dot(svoTemporalDecodeNormal(current.geometricNormalOct),svoTemporalDecodeNormal(previous.geometricNormalOct))<SVO_TEMPORAL_GEOMETRIC_DOT){return SVO_TEMPORAL_REASON_GEOMETRIC_NORMAL;}
  if(dot(svoTemporalDecodeNormal(current.shadingNormalOct),svoTemporalDecodeNormal(previous.shadingNormalOct))<SVO_TEMPORAL_SHADING_DOT){return SVO_TEMPORAL_REASON_SHADING_NORMAL;}
  return SVO_TEMPORAL_REASON_ACCEPTED;
}
`;
