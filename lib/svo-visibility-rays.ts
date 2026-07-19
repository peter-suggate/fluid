import type { SvoVec3 } from "./webgpu-svo-traversal";

export const SVO_VISIBILITY_LIMITS = Object.freeze({
  nodeVisits: 256,
  leafVisits: 64,
  workItems: 2_048,
  intersections: 8,
} as const);

export interface SvoVisibilityRayInput {
  surfacePosition_m: SvoVec3;
  geometricNormal: SvoVec3;
  directionToLight: SvoVec3;
  maximumLightDistance_m: number;
  cellSize_m: SvoVec3;
}

export interface SvoVisibilityBiasOptions {
  /** Fraction of the geometric-normal projected cell width. */
  originBiasCells?: number;
}

export interface SvoVisibilityRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  tMin_m: 0;
  /** Distance from the biased origin to the original maximum-light plane. */
  tMax_m: number;
  originBias_m: number;
}

export interface SvoVisibilityBudgets {
  maximumNodeVisits?: number;
  maximumLeafVisits?: number;
  maximumWorkItems?: number;
  maximumIntersections?: number;
}

export interface SvoVisibilityTraceOptions extends SvoVisibilityBudgets {
  /** False treats every hit as opaque, which is the conservative default. */
  allowTransmission?: boolean;
  /** Terminate when every remaining RGB component is at or below this value. */
  minimumTransmittance?: number;
  /** Strict forward progress after a transmissive surface, in world metres. */
  continuationBias_m?: number;
}

export interface SvoVisibilityWorkCounts {
  nodeVisits: number;
  leafVisits: number;
  workItems: number;
}

export interface SvoVisibilityTotals extends SvoVisibilityWorkCounts {
  intersections: number;
}

export interface SvoVisibilityRemainingBudget extends SvoVisibilityWorkCounts {
  intersections: number;
}

export interface SvoVisibilityHit {
  t_m: number;
  /** Opaque always terminates visibility, regardless of transmission values. */
  opaque: boolean;
  /** Per-surface RGB throughput. Required for a transmissive hit. */
  transmittance?: SvoVec3;
  materialId?: number;
  ownerId?: number;
}

export type SvoVisibilityStep =
  | { status: "hit"; counts: SvoVisibilityWorkCounts; hit: SvoVisibilityHit }
  | { status: "miss"; counts: SvoVisibilityWorkCounts }
  | { status: "exhausted"; counts: SvoVisibilityWorkCounts; reason?: string }
  | { status: "invalid"; counts: SvoVisibilityWorkCounts; reason: string };

export interface SvoVisibilityStepQuery {
  ray: SvoVisibilityRay;
  tMin_m: number;
  remaining: SvoVisibilityRemainingBudget;
}

/** Adapter point for packed SVO traversal plus leaf/primitive intersection. */
export type SvoVisibilityStepSource = (query: SvoVisibilityStepQuery) => SvoVisibilityStep;

export type SvoVisibilityTraceResult =
  | {
    status: "visible";
    transmittance: SvoVec3;
    counts: SvoVisibilityTotals;
  }
  | {
    status: "occluded";
    transmittance: readonly [0, 0, 0];
    counts: SvoVisibilityTotals;
    blocker: SvoVisibilityHit;
  }
  | {
    status: "exhausted";
    /** Fail-closed visibility for direct lighting. */
    transmittance: readonly [0, 0, 0];
    counts: SvoVisibilityTotals;
    exhaustedBy: "nodes" | "leaves" | "work" | "intersections" | "source";
    reason?: string;
  }
  | {
    status: "invalid";
    /** Invalid traversal never becomes an unshadowed light leak. */
    transmittance: readonly [0, 0, 0];
    counts: SvoVisibilityTotals;
    reason: string;
  };

const DEFAULT_ORIGIN_BIAS_CELLS = 1e-3;
const DEFAULT_MINIMUM_TRANSMITTANCE = 1e-3;

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function length3(value: SvoVec3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalized(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const magnitude = length3(value);
  if (!(magnitude > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be finite and non-negative`);
  return value;
}

function boundedPositiveInteger(value: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  }
  return value;
}

/**
 * Offset a secondary ray along the geometric normal. The scale is the
 * anisotropic cell width projected onto that normal, rather than a fixed world
 * epsilon. The normal side follows the light ray's hemisphere.
 */
export function createBiasedSvoVisibilityRay(
  input: SvoVisibilityRayInput,
  options: SvoVisibilityBiasOptions = {},
): SvoVisibilityRay {
  finiteVec3(input.surfacePosition_m, "Visibility surface position");
  finiteVec3(input.cellSize_m, "Visibility cell size");
  input.cellSize_m.forEach((size, axis) => {
    if (!(size > 0)) throw new RangeError(`Visibility cell size axis ${axis} must be positive`);
  });
  const normal = normalized(input.geometricNormal, "Visibility geometric normal");
  const direction = normalized(input.directionToLight, "Visibility light direction");
  const maximumDistance = finiteNonNegative(input.maximumLightDistance_m, "Maximum visibility light distance");
  const biasCells = finiteNonNegative(options.originBiasCells ?? DEFAULT_ORIGIN_BIAS_CELLS, "Visibility origin bias");
  const projectedCellWidth = Math.abs(normal[0]) * input.cellSize_m[0]
    + Math.abs(normal[1]) * input.cellSize_m[1]
    + Math.abs(normal[2]) * input.cellSize_m[2];
  const originBias_m = biasCells * projectedCellWidth;
  const side = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2] >= 0 ? 1 : -1;
  const offset: SvoVec3 = [
    side * normal[0] * originBias_m,
    side * normal[1] * originBias_m,
    side * normal[2] * originBias_m,
  ];
  const advanceTowardLight = offset[0] * direction[0] + offset[1] * direction[1] + offset[2] * direction[2];
  return {
    origin_m: [
      input.surfacePosition_m[0] + offset[0],
      input.surfacePosition_m[1] + offset[1],
      input.surfacePosition_m[2] + offset[2],
    ],
    direction,
    tMin_m: 0,
    tMax_m: Math.max(0, maximumDistance - advanceTowardLight),
    originBias_m,
  };
}

function visibilityBudgets(options: SvoVisibilityBudgets): SvoVisibilityRemainingBudget {
  return {
    nodeVisits: boundedPositiveInteger(
      options.maximumNodeVisits ?? SVO_VISIBILITY_LIMITS.nodeVisits,
      SVO_VISIBILITY_LIMITS.nodeVisits,
      "Visibility node budget",
    ),
    leafVisits: boundedPositiveInteger(
      options.maximumLeafVisits ?? SVO_VISIBILITY_LIMITS.leafVisits,
      SVO_VISIBILITY_LIMITS.leafVisits,
      "Visibility leaf budget",
    ),
    workItems: boundedPositiveInteger(
      options.maximumWorkItems ?? SVO_VISIBILITY_LIMITS.workItems,
      SVO_VISIBILITY_LIMITS.workItems,
      "Visibility work budget",
    ),
    intersections: boundedPositiveInteger(
      options.maximumIntersections ?? SVO_VISIBILITY_LIMITS.intersections,
      SVO_VISIBILITY_LIMITS.intersections,
      "Visibility intersection budget",
    ),
  };
}

function validCounts(counts: SvoVisibilityWorkCounts): boolean {
  return [counts.nodeVisits, counts.leafVisits, counts.workItems].every(
    (value) => Number.isSafeInteger(value) && value >= 0,
  );
}

function remaining(budget: SvoVisibilityRemainingBudget, totals: SvoVisibilityTotals): SvoVisibilityRemainingBudget {
  return {
    nodeVisits: budget.nodeVisits - totals.nodeVisits,
    leafVisits: budget.leafVisits - totals.leafVisits,
    workItems: budget.workItems - totals.workItems,
    intersections: budget.intersections - totals.intersections,
  };
}

function exhaustedDimension(
  totals: SvoVisibilityTotals,
  delta: SvoVisibilityWorkCounts,
  budget: SvoVisibilityRemainingBudget,
): "nodes" | "leaves" | "work" | undefined {
  if (totals.nodeVisits + delta.nodeVisits > budget.nodeVisits) return "nodes";
  if (totals.leafVisits + delta.leafVisits > budget.leafVisits) return "leaves";
  if (totals.workItems + delta.workItems > budget.workItems) return "work";
  return undefined;
}

function addCounts(totals: SvoVisibilityTotals, delta: SvoVisibilityWorkCounts): SvoVisibilityTotals {
  return {
    nodeVisits: totals.nodeVisits + delta.nodeVisits,
    leafVisits: totals.leafVisits + delta.leafVisits,
    workItems: totals.workItems + delta.workItems,
    intersections: totals.intersections,
  };
}

function zero(): readonly [0, 0, 0] {
  return [0, 0, 0];
}

function validTransmittance(value: SvoVec3 | undefined): value is SvoVec3 {
  return value !== undefined && value.every((component) => Number.isFinite(component) && component >= 0 && component <= 1);
}

/**
 * Trace conservative direct-light visibility through a renderer-supplied
 * nearest-hit adapter. All loops and resource-independent work are bounded.
 */
export function traceSvoVisibilityRay(
  ray: SvoVisibilityRay,
  source: SvoVisibilityStepSource,
  options: SvoVisibilityTraceOptions = {},
): SvoVisibilityTraceResult {
  finiteVec3(ray.origin_m, "Visibility ray origin");
  const direction = normalized(ray.direction, "Visibility ray direction");
  if (Math.abs(length3(direction) - 1) > 1e-9) throw new RangeError("Visibility ray direction normalization failed");
  if (ray.tMin_m !== 0 || !Number.isFinite(ray.tMax_m) || ray.tMax_m < 0) {
    throw new RangeError("Visibility ray must use a finite [0, tMax] interval");
  }
  finiteNonNegative(ray.originBias_m, "Visibility ray origin bias");
  const budget = visibilityBudgets(options);
  const minimumTransmittance = finiteNonNegative(
    options.minimumTransmittance ?? DEFAULT_MINIMUM_TRANSMITTANCE,
    "Minimum visibility transmittance",
  );
  if (minimumTransmittance > 1) throw new RangeError("Minimum visibility transmittance must not exceed one");
  const continuationBias = finiteNonNegative(
    options.continuationBias_m ?? Math.max(ray.originBias_m, 1e-6),
    "Visibility continuation bias",
  );
  const allowTransmission = options.allowTransmission ?? false;
  let totals: SvoVisibilityTotals = { nodeVisits: 0, leafVisits: 0, workItems: 0, intersections: 0 };
  let transmittance: [number, number, number] = [1, 1, 1];
  let cursor = ray.tMin_m;

  for (let interaction = 0; interaction < budget.intersections; interaction += 1) {
    if (cursor >= ray.tMax_m) return { status: "visible", transmittance, counts: totals };
    const step = source({ ray, tMin_m: cursor, remaining: remaining(budget, totals) });
    if (!validCounts(step.counts)) {
      return { status: "invalid", transmittance: zero(), counts: totals, reason: "Visibility source returned invalid work counts" };
    }
    const exhaustedBy = exhaustedDimension(totals, step.counts, budget);
    if (exhaustedBy) {
      const counts = addCounts(totals, {
        nodeVisits: Math.min(step.counts.nodeVisits, budget.nodeVisits - totals.nodeVisits),
        leafVisits: Math.min(step.counts.leafVisits, budget.leafVisits - totals.leafVisits),
        workItems: Math.min(step.counts.workItems, budget.workItems - totals.workItems),
      });
      return { status: "exhausted", transmittance: zero(), counts, exhaustedBy };
    }
    totals = addCounts(totals, step.counts);
    if (step.status === "invalid") {
      return { status: "invalid", transmittance: zero(), counts: totals, reason: step.reason };
    }
    if (step.status === "exhausted") {
      return { status: "exhausted", transmittance: zero(), counts: totals, exhaustedBy: "source", reason: step.reason };
    }
    if (step.status === "miss") return { status: "visible", transmittance, counts: totals };

    const hit = step.hit;
    if (!Number.isFinite(hit.t_m) || hit.t_m < cursor) {
      return { status: "invalid", transmittance: zero(), counts: totals, reason: "Visibility hit is behind the active ray interval" };
    }
    // A source may return its nearest candidate without clipping to the light.
    // Such a candidate cannot shadow this finite light segment.
    if (hit.t_m > ray.tMax_m) return { status: "visible", transmittance, counts: totals };
    totals = { ...totals, intersections: totals.intersections + 1 };
    if (hit.opaque || !allowTransmission) {
      return { status: "occluded", transmittance: zero(), counts: totals, blocker: hit };
    }
    if (!validTransmittance(hit.transmittance)) {
      return { status: "invalid", transmittance: zero(), counts: totals, reason: "Transmissive visibility hit has invalid RGB attenuation" };
    }
    transmittance = [
      transmittance[0] * hit.transmittance[0],
      transmittance[1] * hit.transmittance[1],
      transmittance[2] * hit.transmittance[2],
    ];
    if (Math.max(...transmittance) <= minimumTransmittance) {
      return { status: "occluded", transmittance: zero(), counts: totals, blocker: hit };
    }
    cursor = hit.t_m + continuationBias;
  }
  return { status: "exhausted", transmittance: zero(), counts: totals, exhaustedBy: "intersections" };
}

/**
 * Binding-free WGSL. The including renderer implements
 * `svoVisibilityNext(ray, tMin_m, remaining) -> SvoVisibilityStep` by composing
 * its SVO traversal, leaf DDA, and analytic primitive helpers.
 */
export const svoVisibilityRaysWGSL = /* wgsl */ `
const SVO_VIS_STATUS_VISIBLE:u32 = 0u;
const SVO_VIS_STATUS_OCCLUDED:u32 = 1u;
const SVO_VIS_STATUS_EXHAUSTED:u32 = 2u;
const SVO_VIS_STATUS_INVALID:u32 = 3u;
const SVO_VIS_STEP_MISS:u32 = 0u;
const SVO_VIS_STEP_HIT:u32 = 1u;
const SVO_VIS_STEP_EXHAUSTED:u32 = 2u;
const SVO_VIS_STEP_INVALID:u32 = 3u;
const SVO_VIS_MAX_NODES:u32 = 256u;
const SVO_VIS_MAX_LEAVES:u32 = 64u;
const SVO_VIS_MAX_WORK:u32 = 2048u;
const SVO_VIS_MAX_INTERSECTIONS:u32 = 8u;

struct SvoVisibilityRay { origin_m:vec3f, tMax_m:f32, direction:vec3f, originBias_m:f32 }
struct SvoVisibilityBudget { nodeVisits:u32, leafVisits:u32, workItems:u32, intersections:u32 }
struct SvoVisibilityStep {
  status:u32, nodeVisits:u32, leafVisits:u32, workItems:u32,
  t_m:f32, opaque:u32, transmittance:vec3f, _padding:u32,
}
struct SvoVisibilityResult {
  status:u32, nodeVisits:u32, leafVisits:u32, workItems:u32,
  intersections:u32, transmittance:vec3f,
}

fn svoVisibilityFinite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}

fn svoBiasedVisibilityRay(surfacePosition_m:vec3f,geometricNormalIn:vec3f,directionToLightIn:vec3f,maximumLightDistance_m:f32,cellSize_m:vec3f,biasCells:f32)->SvoVisibilityRay {
  let geometricNormal=normalize(geometricNormalIn);let directionToLight=normalize(directionToLightIn);
  let projectedCellWidth=dot(abs(geometricNormal),cellSize_m);let originBias_m=max(biasCells,0.0)*projectedCellWidth;
  let side=select(-1.0,1.0,dot(geometricNormal,directionToLight)>=0.0);let offset=side*geometricNormal*originBias_m;
  return SvoVisibilityRay(surfacePosition_m+offset,max(0.0,maximumLightDistance_m-dot(offset,directionToLight)),directionToLight,originBias_m);
}

fn svoVisibilityFail(status:u32,used:SvoVisibilityBudget)->SvoVisibilityResult {
  return SvoVisibilityResult(status,used.nodeVisits,used.leafVisits,used.workItems,used.intersections,vec3f(0.0));
}

// Requires the including renderer's svoVisibilityNext adapter described above.
fn svoTraceVisibility(ray:SvoVisibilityRay,requested:SvoVisibilityBudget,allowTransmission:bool,minimumTransmittance:f32,continuationBias_m:f32)->SvoVisibilityResult {
  let limit=SvoVisibilityBudget(min(max(requested.nodeVisits,1u),SVO_VIS_MAX_NODES),min(max(requested.leafVisits,1u),SVO_VIS_MAX_LEAVES),min(max(requested.workItems,1u),SVO_VIS_MAX_WORK),min(max(requested.intersections,1u),SVO_VIS_MAX_INTERSECTIONS));
  var used=SvoVisibilityBudget(0u,0u,0u,0u);var throughput=vec3f(1.0);var cursor=0.0;
  for(var interaction=0u;interaction<SVO_VIS_MAX_INTERSECTIONS;interaction+=1u){
    if(interaction>=limit.intersections){return svoVisibilityFail(SVO_VIS_STATUS_EXHAUSTED,used);}
    if(cursor>=ray.tMax_m){return SvoVisibilityResult(SVO_VIS_STATUS_VISIBLE,used.nodeVisits,used.leafVisits,used.workItems,used.intersections,throughput);}
    let remaining=SvoVisibilityBudget(limit.nodeVisits-used.nodeVisits,limit.leafVisits-used.leafVisits,limit.workItems-used.workItems,limit.intersections-used.intersections);
    let step=svoVisibilityNext(ray,cursor,remaining);
    if(step.nodeVisits>remaining.nodeVisits||step.leafVisits>remaining.leafVisits||step.workItems>remaining.workItems){return svoVisibilityFail(SVO_VIS_STATUS_EXHAUSTED,used);}
    used.nodeVisits+=step.nodeVisits;used.leafVisits+=step.leafVisits;used.workItems+=step.workItems;
    if(step.status==SVO_VIS_STEP_MISS){return SvoVisibilityResult(SVO_VIS_STATUS_VISIBLE,used.nodeVisits,used.leafVisits,used.workItems,used.intersections,throughput);}
    if(step.status==SVO_VIS_STEP_EXHAUSTED){return svoVisibilityFail(SVO_VIS_STATUS_EXHAUSTED,used);}
    if(step.status!=SVO_VIS_STEP_HIT||!svoVisibilityFinite(step.t_m)||step.t_m<cursor){return svoVisibilityFail(SVO_VIS_STATUS_INVALID,used);}
    if(step.t_m>ray.tMax_m){return SvoVisibilityResult(SVO_VIS_STATUS_VISIBLE,used.nodeVisits,used.leafVisits,used.workItems,used.intersections,throughput);}
    used.intersections+=1u;
    if(step.opaque!=0u||!allowTransmission){return svoVisibilityFail(SVO_VIS_STATUS_OCCLUDED,used);}
    if(any(step.transmittance<vec3f(0.0))||any(step.transmittance>vec3f(1.0))||any(step.transmittance!=step.transmittance)){return svoVisibilityFail(SVO_VIS_STATUS_INVALID,used);}
    throughput*=step.transmittance;if(max(throughput.x,max(throughput.y,throughput.z))<=minimumTransmittance){return svoVisibilityFail(SVO_VIS_STATUS_OCCLUDED,used);}
    cursor=step.t_m+max(continuationBias_m,max(ray.originBias_m,1e-6));
  }
  return svoVisibilityFail(SVO_VIS_STATUS_EXHAUSTED,used);
}
`;
