import {
  resolveSvoMediumBoundaryGroup,
  evaluateSvoDielectricTransition,
  type SvoMediaBoundary,
  type SvoMediumKind,
} from "./svo-media";
import type { SvoStructuralFluidVisibilityResult } from "./svo-fluid-structural-visibility";
import { GLASS_OPTICS, WATER_OPTICS, type LinearRgb } from "./webgpu-lighting";
import type { SvoVec3 } from "./webgpu-svo-traversal";

/** The direct path remains opt-in until its image and performance gates pass. */
export const DEFAULT_SVO_FLUID_MEDIA_HANDOFF = "legacy-water" as const;
export type SvoFluidMediaHandoffRequest = "legacy-water" | "direct-structural-media";

export interface SvoFluidRenderOwnership {
  effective: SvoFluidMediaHandoffRequest;
  directStructuralMedia: boolean;
  legacyExtraction: boolean;
  legacyInterfaces: boolean;
  legacyComposite: boolean;
  fallbackReason?: "direct-media-not-validated";
}

/**
 * One ownership decision controls all four producer/consumer switches. A caller
 * can therefore never enable the direct fluid path while also extracting or
 * compositing the legacy water surface.
 */
export function resolveSvoFluidRenderOwnership(
  requested: SvoFluidMediaHandoffRequest = DEFAULT_SVO_FLUID_MEDIA_HANDOFF,
  endToEndOpticsValidated = false,
): SvoFluidRenderOwnership {
  if (requested === "direct-structural-media" && endToEndOpticsValidated) {
    return {
      effective: requested,
      directStructuralMedia: true,
      legacyExtraction: false,
      legacyInterfaces: false,
      legacyComposite: false,
    };
  }
  return {
    effective: "legacy-water",
    directStructuralMedia: false,
    legacyExtraction: true,
    legacyInterfaces: true,
    legacyComposite: true,
    ...(requested === "direct-structural-media" ? { fallbackReason: "direct-media-not-validated" as const } : {}),
  };
}

export const SVO_FLUID_MEDIA_PATH_LIMITS = Object.freeze({
  boundaryQueries: 16,
  transitions: 8,
  reflections: 4,
  transmissions: 8,
} as const);

interface SvoFluidMediaWaterStepBase {
  insideFluidAtStart: boolean;
  completeGeneration: number;
  coarseFluidRevision: number;
  steps: number;
  nodeVisits: number;
}

export type SvoFluidMediaWaterStep = SvoFluidMediaWaterStepBase & (
  | { status: "hit"; t_m: number; normal: SvoVec3; boundaryId?: number }
  | { status: "miss" }
  | { status: "invalid-field"; reason?: string }
  | { status: "stale-generation"; reason?: string }
  | { status: "nonresident"; reason?: string }
  | { status: "work-exhausted"; reason?: string }
);

export interface SvoFluidMediaQuery {
  ray: SvoFluidMediaRay;
  queryIndex: number;
  expectedCompleteGeneration?: number;
  currentMedium: SvoMediumKind;
}

export type SvoFluidMediaWaterSource = (query: SvoFluidMediaQuery) => SvoFluidMediaWaterStep;

export type SvoFluidMediaSceneStep =
  | { status: "hit"; boundaries: readonly SvoMediaBoundary[] }
  | { status: "miss" }
  | { status: "invalid"; reason: string }
  | { status: "work-exhausted"; reason?: string };

export type SvoFluidMediaSceneSource = (query: SvoFluidMediaQuery) => SvoFluidMediaSceneStep;

export interface SvoFluidMediaRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  maximumDistance_m: number;
}

export interface SvoFluidMediaPathOptions {
  initialMedium?: "air" | "water";
  maximumBoundaryQueries?: number;
  maximumTransitions?: number;
  maximumReflections?: number;
  maximumTransmissions?: number;
  coincidentBoundaryEpsilon_m?: number;
  continuationEpsilon_m?: number;
  /** Scene-linear radiance approached by bounded single scattering. */
  waterInscatterLinear?: LinearRgb;
}

export interface SvoFluidMediaPathCounts {
  boundaryQueries: number;
  transitions: number;
  reflections: number;
  transmissions: number;
  structuralSteps: number;
  topologyNodeVisits: number;
}

interface SvoFluidMediaPathBase {
  throughput: LinearRgb;
  inscatteredRadiance: LinearRgb;
  distance_m: number;
  waterThickness_m: number;
  direction: SvoVec3;
  currentMedium: SvoMediumKind;
  completeGeneration?: number;
  coarseFluidRevision?: number;
  entryNormal?: SvoVec3;
  exitNormal?: SvoVec3;
  counts: SvoFluidMediaPathCounts;
}

export type SvoFluidMediaPathResult =
  | (SvoFluidMediaPathBase & { status: "water-segment"; boundaryId?: number })
  | (SvoFluidMediaPathBase & { status: "opaque-contact"; boundary: SvoMediaBoundary; submerged: boolean })
  | (SvoFluidMediaPathBase & { status: "no-water" | "range-ended-in-water" })
  | (SvoFluidMediaPathBase & {
    status: "invalid";
    failure: "invalid-ray" | "invalid-field" | "stale-generation" | "nonresident" | "media-stack";
    reason?: string;
  })
  | (SvoFluidMediaPathBase & {
    status: "work-exhausted";
    exhaustedBy: "queries" | "transitions" | "reflections" | "transmissions" | "source";
    reason?: string;
  });

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function normalize(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const magnitude = Math.hypot(...value);
  if (!(magnitude > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function pointAt(ray: SvoFluidMediaRay, t_m: number): SvoVec3 {
  return [
    ray.origin_m[0] + ray.direction[0] * t_m,
    ray.origin_m[1] + ray.direction[1] * t_m,
    ray.origin_m[2] + ray.direction[2] * t_m,
  ];
}

function bounded(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) {
    throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  }
  return result;
}

function zeroBase(
  counts: SvoFluidMediaPathCounts,
  direction: SvoVec3,
  currentMedium: SvoMediumKind,
): SvoFluidMediaPathBase {
  return {
    throughput: [0, 0, 0], inscatteredRadiance: [0, 0, 0], distance_m: 0, waterThickness_m: 0,
    direction, currentMedium, counts,
  };
}

function waterSegmentOptics(
  throughput: LinearRgb,
  radiance: LinearRgb,
  distance_m: number,
  inscatter: LinearRgb,
): { throughput: LinearRgb; radiance: LinearRgb } {
  const transmittance = WATER_OPTICS.absorption.map((coefficient) => Math.exp(-coefficient * distance_m)) as [number, number, number];
  const scattered = WATER_OPTICS.scatter.map((coefficient) => 1 - Math.exp(-coefficient * distance_m)) as [number, number, number];
  return {
    throughput: throughput.map((channel, index) => channel * transmittance[index]) as [number, number, number],
    radiance: radiance.map((channel, index) => channel + throughput[index] * inscatter[index] * scattered[index]) as [number, number, number],
  };
}

function nearestSceneDistance(step: SvoFluidMediaSceneStep): number {
  return step.status === "hit" ? Math.min(...step.boundaries.map((boundary) => boundary.t_m)) : Number.POSITIVE_INFINITY;
}

/** Convert the authoritative packed structural visibility result without turning a bad field into empty air. */
export function adaptSvoStructuralFluidVisibilityToMediaStep(
  result: SvoStructuralFluidVisibilityResult,
  boundaryId?: number,
): SvoFluidMediaWaterStep {
  const common = {
    insideFluidAtStart: result.insideFluidAtStart,
    completeGeneration: result.diagnostics.completeGeneration,
    coarseFluidRevision: result.diagnostics.coarseFluidRevision,
    steps: result.steps,
    nodeVisits: result.diagnostics.topologyNodeVisits,
  };
  if (result.status === "hit") return { ...common, status: "hit", t_m: result.t_m, normal: result.normal, boundaryId };
  if (result.status === "miss") return { ...common, status: "miss" };
  if (result.status === "work-exhausted") return { ...common, status: "work-exhausted", reason: result.diagnostics.failureReason };
  if (result.diagnostics.failureReason === "generation-mismatch") {
    return { ...common, status: "stale-generation", reason: result.diagnostics.failureReason };
  }
  if (result.diagnostics.failureReason === "nonresident-leaf") {
    return { ...common, status: "nonresident", reason: result.diagnostics.failureReason };
  }
  return { ...common, status: "invalid-field", reason: result.diagnostics.failureReason };
}

/**
 * Bounded transmission path for the authoritative structural coarse level set.
 * It follows one refracted path (plus TIR continuations), not a recursive path
 * tree. Bad publication/residency/work status returns black fail-closed state.
 */
export function traceSvoStructuralFluidMediaPath(
  inputRay: SvoFluidMediaRay,
  waterSource: SvoFluidMediaWaterSource,
  sceneSource: SvoFluidMediaSceneSource = () => ({ status: "miss" }),
  options: SvoFluidMediaPathOptions = {},
): SvoFluidMediaPathResult {
  finiteVec3(inputRay.origin_m, "Fluid media ray origin");
  let direction = normalize(inputRay.direction, "Fluid media ray direction");
  if (!Number.isFinite(inputRay.maximumDistance_m) || inputRay.maximumDistance_m < 0) {
    throw new RangeError("Fluid media ray maximum distance must be finite and non-negative");
  }
  const limits = {
    queries: bounded(options.maximumBoundaryQueries, SVO_FLUID_MEDIA_PATH_LIMITS.boundaryQueries, SVO_FLUID_MEDIA_PATH_LIMITS.boundaryQueries, "Fluid media query budget"),
    transitions: bounded(options.maximumTransitions, SVO_FLUID_MEDIA_PATH_LIMITS.transitions, SVO_FLUID_MEDIA_PATH_LIMITS.transitions, "Fluid media transition budget"),
    reflections: bounded(options.maximumReflections, SVO_FLUID_MEDIA_PATH_LIMITS.reflections, SVO_FLUID_MEDIA_PATH_LIMITS.reflections, "Fluid media reflection budget"),
    transmissions: bounded(options.maximumTransmissions, SVO_FLUID_MEDIA_PATH_LIMITS.transmissions, SVO_FLUID_MEDIA_PATH_LIMITS.transmissions, "Fluid media transmission budget"),
  };
  const coincidence = options.coincidentBoundaryEpsilon_m ?? 1e-5;
  const continuation = options.continuationEpsilon_m ?? coincidence;
  if (!Number.isFinite(coincidence) || coincidence < 0 || !Number.isFinite(continuation) || continuation < 0) {
    throw new RangeError("Fluid media epsilons must be finite and non-negative");
  }
  const inscatter = options.waterInscatterLinear ?? [0.16, 0.32, 0.38];
  if (inscatter.some((channel) => !Number.isFinite(channel) || channel < 0)) {
    throw new RangeError("Water in-scattering must contain finite non-negative channels");
  }
  let currentMedium: SvoMediumKind = options.initialMedium ?? "air";
  let ray: SvoFluidMediaRay = { origin_m: [...inputRay.origin_m], direction, maximumDistance_m: inputRay.maximumDistance_m };
  let throughput: LinearRgb = [1, 1, 1];
  let radiance: LinearRgb = [0, 0, 0];
  let distance_m = 0;
  let waterThickness_m = 0;
  let completeGeneration: number | undefined;
  let coarseFluidRevision: number | undefined;
  let entryNormal: SvoVec3 | undefined;
  let exitNormal: SvoVec3 | undefined;
  let enteredWater = currentMedium === "water";
  const counts: SvoFluidMediaPathCounts = {
    boundaryQueries: 0, transitions: 0, reflections: 0, transmissions: 0,
    structuralSteps: 0, topologyNodeVisits: 0,
  };
  const base = (): SvoFluidMediaPathBase => ({ throughput, inscatteredRadiance: radiance, distance_m, waterThickness_m,
    direction, currentMedium, completeGeneration, coarseFluidRevision, entryNormal, exitNormal, counts });
  const failed = (failure: "invalid-field" | "stale-generation" | "nonresident", reason?: string): SvoFluidMediaPathResult => ({
    ...zeroBase(counts, direction, currentMedium), completeGeneration, coarseFluidRevision,
    status: "invalid", failure, reason,
  });

  for (let queryIndex = 0; queryIndex < limits.queries; queryIndex += 1) {
    if (ray.maximumDistance_m <= 0) return { ...base(), status: currentMedium === "water" ? "range-ended-in-water" : "no-water" };
    counts.boundaryQueries += 1;
    const query: SvoFluidMediaQuery = { ray, queryIndex, expectedCompleteGeneration: completeGeneration, currentMedium };
    const water = waterSource(query);
    counts.structuralSteps += water.steps;
    counts.topologyNodeVisits += water.nodeVisits;
    if (water.status === "work-exhausted") return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "source", reason: water.reason };
    if (water.status === "stale-generation") return failed("stale-generation", water.reason);
    if (water.status === "nonresident") return failed("nonresident", water.reason);
    if (water.status === "invalid-field") return failed("invalid-field", water.reason);
    if (!(water.completeGeneration > 0) || !(water.coarseFluidRevision > 0)) return failed("invalid-field", "unpublished structural fluid field");
    if (completeGeneration === undefined) {
      completeGeneration = water.completeGeneration;
      coarseFluidRevision = water.coarseFluidRevision;
      if (queryIndex === 0 && water.insideFluidAtStart !== (currentMedium === "water")) {
        return { ...zeroBase(counts, direction, currentMedium), completeGeneration, coarseFluidRevision,
          status: "invalid", failure: "media-stack", reason: "initial medium disagrees with structural phi sign" };
      }
    } else if (water.completeGeneration !== completeGeneration || water.coarseFluidRevision !== coarseFluidRevision) {
      return failed("stale-generation", "structural fluid publication changed during the media path");
    }

    const scene = sceneSource(query);
    if (scene.status === "invalid") return { ...zeroBase(counts, direction, currentMedium), status: "invalid", failure: "invalid-field", reason: scene.reason };
    if (scene.status === "work-exhausted") return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "source", reason: scene.reason };
    const waterDistance = water.status === "hit" ? water.t_m : Number.POSITIVE_INFINITY;
    const sceneDistance = nearestSceneDistance(scene);
    const hitDistance = Math.min(waterDistance, sceneDistance);
    if (!Number.isFinite(hitDistance) || hitDistance > ray.maximumDistance_m) {
      if (currentMedium === "water") {
        const optical = waterSegmentOptics(throughput, radiance, ray.maximumDistance_m, inscatter);
        throughput = optical.throughput; radiance = optical.radiance; waterThickness_m += ray.maximumDistance_m;
      }
      distance_m += ray.maximumDistance_m;
      return { ...base(), status: currentMedium === "water" ? "range-ended-in-water" : "no-water" };
    }
    if (hitDistance < 0) return failed("invalid-field", "negative boundary distance");
    if (currentMedium === "water") {
      const optical = waterSegmentOptics(throughput, radiance, hitDistance, inscatter);
      throughput = optical.throughput; radiance = optical.radiance; waterThickness_m += hitDistance;
    }
    distance_m += hitDistance;
    const hitPoint = pointAt(ray, hitDistance);

    const coincidentScene = scene.status === "hit"
      ? scene.boundaries.filter((boundary) => Math.abs(boundary.t_m - hitDistance) <= coincidence) : [];
    const opaque = coincidentScene.find((boundary) => boundary.medium === "opaque");
    if (opaque) return { ...base(), status: "opaque-contact", boundary: opaque, submerged: currentMedium === "water" };

    const boundaries: SvoMediaBoundary[] = [...coincidentScene];
    if (water.status === "hit" && Math.abs(water.t_m - hitDistance) <= coincidence) {
      boundaries.push({ t_m: hitDistance, medium: "water", geometricNormal: normalize(water.normal, "Structural water boundary normal"), boundaryId: water.boundaryId });
    }
    if (boundaries.length === 0) return failed("invalid-field", "nearest scene boundary was not returned in its coincidence group");
    let resolved;
    try {
      resolved = resolveSvoMediumBoundaryGroup(currentMedium === "water" ? ["air", "water"] : ["air"], boundaries, direction, coincidence);
    } catch (error) {
      return { ...zeroBase(counts, direction, currentMedium), completeGeneration, coarseFluidRevision,
        status: "invalid", failure: "media-stack", reason: error instanceof Error ? error.message : String(error) };
    }
    for (const wall of resolved.thinWalls) {
      if (counts.transitions >= limits.transitions) return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "transitions" };
      if (counts.transmissions >= limits.transmissions) return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "transmissions" };
      counts.transitions += 1; counts.transmissions += 1;
      const ior = wall.thinWallIor ?? GLASS_OPTICS.indexOfRefraction;
      const cosine = Math.abs(direction[0] * wall.geometricNormal[0] + direction[1] * wall.geometricNormal[1] + direction[2] * wall.geometricNormal[2]);
      const f0 = ((ior - (currentMedium === "water" ? WATER_OPTICS.indexOfRefraction : 1)) / (ior + (currentMedium === "water" ? WATER_OPTICS.indexOfRefraction : 1))) ** 2;
      const fresnel = f0 + (1 - f0) * (1 - Math.min(1, Math.max(0, cosine))) ** 5;
      const tint = wall.thinWallTint ?? GLASS_OPTICS.tint;
      throughput = throughput.map((channel, index) => channel * (1 - fresnel) ** 2 * tint[index]) as [number, number, number];
    }
    if (resolved.from !== resolved.to) {
      if (counts.transitions >= limits.transitions) return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "transitions" };
      counts.transitions += 1;
      const transition = evaluateSvoDielectricTransition(direction, resolved.interfaceNormal as SvoVec3, resolved.from, resolved.to);
      if (transition.totalInternalReflection) {
        if (counts.reflections >= limits.reflections) return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "reflections" };
        counts.reflections += 1;
        direction = transition.reflectedDirection;
      } else {
        if (counts.transmissions >= limits.transmissions) return { ...zeroBase(counts, direction, currentMedium), status: "work-exhausted", exhaustedBy: "transmissions" };
        counts.transmissions += 1;
        throughput = throughput.map((channel) => channel * (1 - transition.fresnel)) as [number, number, number];
        direction = transition.refractedDirection as SvoVec3;
        if (resolved.from !== "water" && resolved.to === "water") {
          entryNormal = normalize(resolved.interfaceNormal as SvoVec3, "Water entry normal");
          enteredWater = true;
        }
        if (resolved.from === "water" && resolved.to !== "water") {
          exitNormal = normalize(resolved.interfaceNormal as SvoVec3, "Water exit normal");
          currentMedium = resolved.to;
          return { ...base(), status: "water-segment", boundaryId: water.status === "hit" ? water.boundaryId : undefined };
        }
        currentMedium = resolved.to;
      }
    }
    const remaining = Math.max(0, ray.maximumDistance_m - hitDistance - continuation);
    ray = {
      origin_m: [hitPoint[0] + direction[0] * continuation, hitPoint[1] + direction[1] * continuation, hitPoint[2] + direction[2] * continuation],
      direction,
      maximumDistance_m: remaining,
    };
  }
  return { ...zeroBase(counts, direction, currentMedium), completeGeneration, coarseFluidRevision,
    status: "work-exhausted", exhaustedBy: "queries", reason: enteredWater ? "water path did not terminate" : "boundary path did not terminate" };
}

/** Binding-free GPU seam. Consumers provide the two bounded nearest-boundary callbacks. */
export const svoFluidMediaPathWGSL = /* wgsl */ `
const SVO_FLUID_MEDIA_MISS:u32=0u;const SVO_FLUID_MEDIA_HIT:u32=1u;const SVO_FLUID_MEDIA_INVALID:u32=2u;const SVO_FLUID_MEDIA_EXHAUSTED:u32=3u;const SVO_FLUID_MEDIA_STALE:u32=4u;const SVO_FLUID_MEDIA_NONRESIDENT:u32=5u;
const SVO_FLUID_MEDIA_SEGMENT:u32=1u;const SVO_FLUID_MEDIA_OPAQUE:u32=2u;const SVO_FLUID_MEDIA_RANGE_END:u32=3u;const SVO_FLUID_MEDIA_RESULT_INVALID:u32=4u;const SVO_FLUID_MEDIA_RESULT_EXHAUSTED:u32=5u;
const SVO_FLUID_MEDIA_MAX_QUERIES:u32=16u;const SVO_FLUID_MEDIA_MAX_TRANSITIONS:u32=8u;const SVO_FLUID_MEDIA_MAX_REFLECTIONS:u32=4u;const SVO_FLUID_MEDIA_MAX_TRANSMISSIONS:u32=8u;
struct SvoFluidMediaRay{origin_m:vec3f,maximumDistance_m:f32,direction:vec3f,_padding:f32}
struct SvoFluidMediaWaterStep{status:u32,insideAtStart:u32,generation:u32,revision:u32,t_m:f32,steps:u32,nodeVisits:u32,boundaryId:u32,normal:vec3f,_padding:f32}
struct SvoFluidMediaSceneStep{status:u32,medium:u32,flags:u32,boundaryId:u32,t_m:f32,_padding:vec3f,normal:vec3f,ior:f32,tint:vec3f,_padding2:f32}
struct SvoFluidMediaPathResult{status:u32,failure:u32,currentMedium:u32,generation:u32,throughput:vec3f,distance_m:f32,inscatter:vec3f,waterThickness_m:f32,direction:vec3f,revision:u32,entryNormal:vec3f,queries:u32,exitNormal:vec3f,transitions:u32,counts:vec4u}
fn svoFluidMediaBeer(throughput:vec3f,distance_m:f32)->vec3f{return throughput*exp(-vec3f(${WATER_OPTICS.absorption.join(",")})*max(distance_m,0.0));}
fn svoFluidMediaScatter(radiance:vec3f,throughput:vec3f,distance_m:f32,source:vec3f)->vec3f{return radiance+throughput*source*(vec3f(1.0)-exp(-vec3f(${WATER_OPTICS.scatter.join(",")})*max(distance_m,0.0)));}
fn svoFluidMediaRefract(directionIn:vec3f,normalIn:vec3f,fromIor:f32,toIor:f32)->vec4f{let direction=normalize(directionIn);var normal=normalize(normalIn);if(dot(direction,normal)>0.0){normal=-normal;}let eta=fromIor/toIor;let cosine=clamp(-dot(direction,normal),0.0,1.0);let k=1.0-eta*eta*(1.0-cosine*cosine);if(k<0.0){return vec4f(reflect(direction,normal),-1.0);}return vec4f(normalize(eta*direction+(eta*cosine-sqrt(max(k,0.0)))*normal),cosine);}
// Fixed single-path loop: callbacks must use authoritative structural publication/residency status.
fn svoTraceStructuralFluidMedia(rayIn:SvoFluidMediaRay,initialMedium:u32,expectedGeneration:u32,coincidentEpsilon_m:f32,continuationEpsilon_m:f32,inscatterSource:vec3f)->SvoFluidMediaPathResult{
  var ray=rayIn;var medium=initialMedium;var generation=expectedGeneration;var revision=0u;var throughput=vec3f(1.0);var radiance=vec3f(0.0);var distance_m=0.0;var waterThickness_m=0.0;var entryNormal=vec3f(0.0);var exitNormal=vec3f(0.0);var transitions=0u;var reflections=0u;var transmissions=0u;var structuralSteps=0u;var topologyVisits=0u;
  for(var query=0u;query<SVO_FLUID_MEDIA_MAX_QUERIES;query+=1u){let water=svoFluidMediaQueryWater(ray,generation);let scene=svoFluidMediaQueryScene(ray);structuralSteps+=water.steps;topologyVisits+=water.nodeVisits;if(water.status==SVO_FLUID_MEDIA_EXHAUSTED||scene.status==SVO_FLUID_MEDIA_EXHAUSTED){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_EXHAUSTED,3u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}if(water.status==SVO_FLUID_MEDIA_INVALID||water.status==SVO_FLUID_MEDIA_STALE||water.status==SVO_FLUID_MEDIA_NONRESIDENT||scene.status==SVO_FLUID_MEDIA_INVALID){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_INVALID,water.status,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}if(generation==0u){generation=water.generation;revision=water.revision;}if(generation==0u||revision==0u||(query==0u&&water.insideAtStart!=select(0u,1u,medium==1u))||water.generation!=generation||water.revision!=revision){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_INVALID,SVO_FLUID_MEDIA_STALE,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}let waterT=select(1e30,water.t_m,water.status==SVO_FLUID_MEDIA_HIT);let sceneT=select(1e30,scene.t_m,scene.status==SVO_FLUID_MEDIA_HIT);let hitT=min(waterT,sceneT);let hitPoint=ray.origin_m+ray.direction*hitT;if(hitT>ray.maximumDistance_m){if(medium==1u){radiance=svoFluidMediaScatter(radiance,throughput,ray.maximumDistance_m,inscatterSource);throughput=svoFluidMediaBeer(throughput,ray.maximumDistance_m);waterThickness_m+=ray.maximumDistance_m;}return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RANGE_END,0u,medium,generation,throughput,distance_m+ray.maximumDistance_m,radiance,waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}if(medium==1u){radiance=svoFluidMediaScatter(radiance,throughput,hitT,inscatterSource);throughput=svoFluidMediaBeer(throughput,hitT);waterThickness_m+=hitT;}distance_m+=hitT;if(scene.status==SVO_FLUID_MEDIA_HIT&&scene.medium==3u&&sceneT<=waterT+coincidentEpsilon_m){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_OPAQUE,0u,medium,generation,throughput,distance_m,radiance,waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}if(water.status!=SVO_FLUID_MEDIA_HIT||waterT>sceneT+coincidentEpsilon_m){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_INVALID,6u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}if(transitions>=SVO_FLUID_MEDIA_MAX_TRANSITIONS){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_EXHAUSTED,4u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}transitions+=1u;var targetMedium=select(1u,0u,medium==1u);if(scene.status==SVO_FLUID_MEDIA_HIT&&scene.medium==2u&&abs(sceneT-waterT)<=coincidentEpsilon_m){targetMedium=2u;}let fromIor=select(select(1.0,${GLASS_OPTICS.indexOfRefraction},medium==2u),${WATER_OPTICS.indexOfRefraction},medium==1u);let toIor=select(select(1.0,${GLASS_OPTICS.indexOfRefraction},targetMedium==2u),${WATER_OPTICS.indexOfRefraction},targetMedium==1u);let optical=svoFluidMediaRefract(ray.direction,water.normal,fromIor,toIor);if(optical.w<0.0){if(reflections>=SVO_FLUID_MEDIA_MAX_REFLECTIONS){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_EXHAUSTED,5u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}reflections+=1u;ray.direction=optical.xyz;}else{if(transmissions>=SVO_FLUID_MEDIA_MAX_TRANSMISSIONS){return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_EXHAUSTED,6u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}transmissions+=1u;let ratio=(fromIor-toIor)/(fromIor+toIor);let fresnel=ratio*ratio+(1.0-ratio*ratio)*pow(1.0-optical.w,5.0);throughput*=1.0-fresnel;ray.direction=optical.xyz;if(medium!=1u&&targetMedium==1u){entryNormal=water.normal;}if(medium==1u&&targetMedium!=1u){exitNormal=water.normal;medium=targetMedium;return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_SEGMENT,0u,medium,generation,throughput,distance_m,radiance,waterThickness_m,ray.direction,revision,entryNormal,query+1u,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));}medium=targetMedium;}ray.origin_m=hitPoint+ray.direction*continuationEpsilon_m;ray.maximumDistance_m=max(0.0,ray.maximumDistance_m-hitT-continuationEpsilon_m);
  }return SvoFluidMediaPathResult(SVO_FLUID_MEDIA_RESULT_EXHAUSTED,1u,medium,generation,vec3f(0.0),distance_m,vec3f(0.0),waterThickness_m,ray.direction,revision,entryNormal,SVO_FLUID_MEDIA_MAX_QUERIES,exitNormal,transitions,vec4u(reflections,transmissions,structuralSteps,topologyVisits));
}
`;
