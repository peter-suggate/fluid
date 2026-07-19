import {
  beerLambert,
  dielectricFresnel,
  GLASS_OPTICS,
  WATER_OPTICS,
  type LinearRgb,
} from "./webgpu-lighting";
import type { SvoVec3 } from "./webgpu-svo-traversal";

export type SvoMediumKind = "air" | "water" | "glass";

export const SVO_MEDIA_LIMITS = Object.freeze({
  boundaryQueries: 16,
  transitions: 8,
  reflections: 4,
  transmissions: 8,
} as const);

export const SVO_MEDIA = Object.freeze({
  air: { indexOfRefraction: 1, absorption_mInv: [0, 0, 0] as LinearRgb },
  water: { indexOfRefraction: WATER_OPTICS.indexOfRefraction, absorption_mInv: WATER_OPTICS.absorption },
  glass: { indexOfRefraction: GLASS_OPTICS.indexOfRefraction, absorption_mInv: [0, 0, 0] as LinearRgb },
} as const);

export interface SvoDielectricTransition {
  reflectedDirection: SvoVec3;
  refractedDirection?: SvoVec3;
  fresnel: number;
  totalInternalReflection: boolean;
  from: SvoMediumKind;
  to: SvoMediumKind;
}

export interface SvoMediaBoundary {
  t_m: number;
  /** Opaque terminates immediately and never enters the medium stack. */
  medium: SvoMediumKind | "opaque";
  /** Outward from the bounded volume. */
  geometricNormal: SvoVec3;
  /** Collapsed two-interface glass sheet; it does not alter the medium stack. */
  thinWall?: boolean;
  thinWallTint?: LinearRgb;
  /** Defaults to the canonical glass IOR; authored panes may override it. */
  thinWallIor?: number;
  boundaryId?: number;
}

export interface SvoResolvedBoundaryGroup {
  opaque?: SvoMediaBoundary;
  from: SvoMediumKind;
  to: SvoMediumKind;
  nextStack: readonly SvoMediumKind[];
  interfaceNormal?: SvoVec3;
  thinWalls: readonly SvoMediaBoundary[];
}

export interface SvoMediaRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  maximumDistance_m: number;
}

export type SvoMediaBoundaryStep =
  | { status: "hit"; boundaries: readonly SvoMediaBoundary[] }
  | { status: "miss" }
  | { status: "exhausted"; reason?: string }
  | { status: "invalid"; reason: string };

export interface SvoMediaBoundaryQuery {
  ray: SvoMediaRay;
  mediaStack: readonly SvoMediumKind[];
  queryIndex: number;
}

export type SvoMediaBoundarySource = (query: SvoMediaBoundaryQuery) => SvoMediaBoundaryStep;

export interface SvoMediaTraceOptions {
  initialMediaStack?: readonly SvoMediumKind[];
  maximumBoundaryQueries?: number;
  maximumTransitions?: number;
  maximumReflections?: number;
  maximumTransmissions?: number;
  coincidentBoundaryEpsilon_m?: number;
  continuationEpsilon_m?: number;
}

export interface SvoMediaTraceCounts {
  boundaryQueries: number;
  transitions: number;
  reflections: number;
  transmissions: number;
}

interface SvoMediaTraceBase {
  throughput: LinearRgb;
  counts: SvoMediaTraceCounts;
  distance_m: number;
  mediaStack: readonly SvoMediumKind[];
  direction: SvoVec3;
}

export type SvoMediaTraceResult =
  | (SvoMediaTraceBase & { status: "escaped" })
  | (SvoMediaTraceBase & { status: "opaque"; boundary: SvoMediaBoundary; mediumBefore: SvoMediumKind })
  | (SvoMediaTraceBase & {
    status: "exhausted";
    exhaustedBy: "queries" | "transitions" | "reflections" | "transmissions" | "source";
    reason?: string;
  })
  | (SvoMediaTraceBase & { status: "invalid"; reason: string });

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function length3(value: SvoVec3): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalize(value: SvoVec3, label: string): SvoVec3 {
  finiteVec3(value, label);
  const magnitude = length3(value);
  if (!(magnitude > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function dot(left: SvoVec3, right: SvoVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function mediumDefinition(kind: SvoMediumKind) {
  return SVO_MEDIA[kind];
}

function reflect(direction: SvoVec3, normal: SvoVec3): SvoVec3 {
  const scale = 2 * dot(direction, normal);
  return normalize([
    direction[0] - scale * normal[0],
    direction[1] - scale * normal[1],
    direction[2] - scale * normal[2],
  ], "Reflected media direction");
}

/** Exact Snell direction plus Schlick dielectric Fresnel using shared constants. */
export function evaluateSvoDielectricTransition(
  incidentDirection: SvoVec3,
  geometricNormal: SvoVec3,
  from: SvoMediumKind,
  to: SvoMediumKind,
): SvoDielectricTransition {
  const incident = normalize(incidentDirection, "Media incident direction");
  let normal = normalize(geometricNormal, "Media geometric normal");
  if (dot(incident, normal) > 0) normal = [-normal[0], -normal[1], -normal[2]];
  const fromIor = mediumDefinition(from).indexOfRefraction;
  const toIor = mediumDefinition(to).indexOfRefraction;
  const eta = fromIor / toIor;
  const cosine = Math.min(1, Math.max(0, -dot(incident, normal)));
  const discriminant = 1 - eta * eta * (1 - cosine * cosine);
  const reflectedDirection = reflect(incident, normal);
  const f0 = ((fromIor - toIor) / (fromIor + toIor)) ** 2;
  if (discriminant < 0) {
    return { reflectedDirection, fresnel: 1, totalInternalReflection: true, from, to };
  }
  const refractedDirection = normalize([
    eta * incident[0] + (eta * cosine - Math.sqrt(Math.max(0, discriminant))) * normal[0],
    eta * incident[1] + (eta * cosine - Math.sqrt(Math.max(0, discriminant))) * normal[1],
    eta * incident[2] + (eta * cosine - Math.sqrt(Math.max(0, discriminant))) * normal[2],
  ], "Refracted media direction");
  return {
    reflectedDirection,
    refractedDirection,
    fresnel: dielectricFresnel(cosine, f0),
    totalInternalReflection: false,
    from,
    to,
  };
}

/** Apply the canonical absorption coefficients to a path segment. */
export function attenuateSvoMedium(
  throughput: LinearRgb,
  medium: SvoMediumKind,
  distance_m: number,
): [number, number, number] {
  if (!Number.isFinite(distance_m) || distance_m < 0) throw new RangeError("Media absorption distance must be finite and non-negative");
  if (throughput.some((channel) => !Number.isFinite(channel) || channel < 0)) {
    throw new RangeError("Media throughput must contain finite non-negative channels");
  }
  const attenuation = beerLambert(mediumDefinition(medium).absorption_mInv, distance_m);
  return [throughput[0] * attenuation[0], throughput[1] * attenuation[1], throughput[2] * attenuation[2]];
}

function crossing(boundary: SvoMediaBoundary, direction: SvoVec3): "enter" | "exit" | "tangent" {
  const facing = dot(direction, boundary.geometricNormal);
  if (Math.abs(facing) <= 1e-10) return "tangent";
  return facing < 0 ? "enter" : "exit";
}

function canonicalStack(input: readonly SvoMediumKind[]): SvoMediumKind[] {
  if (input.length < 1 || input[0] !== "air") throw new RangeError("Media stack must start with air");
  if (input.length > 4) throw new RangeError("Media stack supports at most four nested media");
  for (let index = 1; index < input.length; index += 1) {
    if (input[index] === "air") throw new RangeError("Air may only be the root media stack entry");
  }
  return [...input];
}

/**
 * Resolve coincident volume boundaries atomically: opaque first, then all
 * exits, then all entries. Thus a water exit coincident with a glass entry is
 * one water-to-glass interface, not two interfaces separated by fake air.
 */
export function resolveSvoMediumBoundaryGroup(
  currentStackInput: readonly SvoMediumKind[],
  boundaries: readonly SvoMediaBoundary[],
  directionInput: SvoVec3,
  coincidentEpsilon_m = 1e-5,
): SvoResolvedBoundaryGroup {
  const currentStack = canonicalStack(currentStackInput);
  const direction = normalize(directionInput, "Media boundary direction");
  if (!Number.isFinite(coincidentEpsilon_m) || coincidentEpsilon_m < 0) {
    throw new RangeError("Coincident media epsilon must be finite and non-negative");
  }
  if (boundaries.length === 0) throw new RangeError("Media boundary group must not be empty");
  const firstDistance = Math.min(...boundaries.map((boundary) => boundary.t_m));
  if (!Number.isFinite(firstDistance) || firstDistance < 0) throw new RangeError("Media boundary distance must be finite and non-negative");
  for (const boundary of boundaries) {
    if (!Number.isFinite(boundary.t_m) || boundary.t_m < 0 || Math.abs(boundary.t_m - firstDistance) > coincidentEpsilon_m) {
      throw new RangeError("Media boundary group exceeds the coincidence epsilon");
    }
    normalize(boundary.geometricNormal, "Media boundary geometric normal");
    if (boundary.thinWall && boundary.medium !== "glass") {
      throw new RangeError("Only glass boundaries may use thin-wall media semantics");
    }
  }
  const opaque = boundaries.find((boundary) => boundary.medium === "opaque");
  const thinWalls = boundaries.filter((boundary) => boundary.medium === "glass" && boundary.thinWall);
  const volumes = boundaries.filter((boundary): boundary is SvoMediaBoundary & { medium: SvoMediumKind } =>
    boundary.medium !== "opaque" && !boundary.thinWall);
  const exits = volumes.filter((boundary) => crossing(boundary, direction) === "exit");
  const entries = volumes.filter((boundary) => crossing(boundary, direction) === "enter");
  if (exits.length + entries.length !== volumes.length) throw new RangeError("Tangent media boundary has no entry/exit ownership");

  const nextStack = [...currentStack];
  const from = nextStack.at(-1) as SvoMediumKind;
  const pendingExits = [...exits];
  while (pendingExits.length > 0) {
    const exitIndex = pendingExits.findIndex((boundary) => boundary.medium === nextStack.at(-1));
    if (nextStack.length === 1 || exitIndex < 0) {
      throw new RangeError(`Cannot resolve coincident exit from media ${nextStack.at(-1)}`);
    }
    pendingExits.splice(exitIndex, 1);
    nextStack.pop();
  }
  for (const boundary of entries) {
    if (nextStack.length >= 4) throw new RangeError("Media boundary entry exceeds the stack capacity");
    nextStack.push(boundary.medium);
  }
  return {
    opaque,
    from,
    to: nextStack.at(-1) as SvoMediumKind,
    nextStack,
    interfaceNormal: volumes[0]?.geometricNormal,
    thinWalls,
  };
}

function bounded(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) {
    throw new RangeError(`${label} must be an integer from 1 to ${maximum}`);
  }
  return result;
}

function failBase(
  throughput: LinearRgb,
  counts: SvoMediaTraceCounts,
  distance_m: number,
  mediaStack: readonly SvoMediumKind[],
  direction: SvoVec3,
): SvoMediaTraceBase {
  return { throughput, counts, distance_m, mediaStack, direction };
}

function thinTint(boundary: SvoMediaBoundary): LinearRgb {
  const tint = boundary.thinWallTint ?? GLASS_OPTICS.tint;
  if (tint.some((channel) => !Number.isFinite(channel) || channel < 0 || channel > 1)) {
    throw new RangeError("Thin-wall tint must contain finite channels from zero to one");
  }
  return tint;
}

function thinWallFresnel(boundary: SvoMediaBoundary, direction: SvoVec3, currentMedium: SvoMediumKind): number {
  const glassIor = boundary.thinWallIor ?? GLASS_OPTICS.indexOfRefraction;
  if (!Number.isFinite(glassIor) || glassIor <= 0) throw new RangeError("Thin-wall IOR must be finite and positive");
  const cosine = Math.min(1, Math.max(0, Math.abs(dot(direction, normalize(boundary.geometricNormal, "Thin-wall normal")))));
  const currentIor = mediumDefinition(currentMedium).indexOfRefraction;
  const f0 = ((currentIor - glassIor) / (currentIor + glassIor)) ** 2;
  return dielectricFresnel(cosine, f0);
}

/**
 * Bounded CPU oracle for one secondary media path. The boundary source may
 * compose SVO traversal, fluid roots, and primitive hits without prescribing
 * any renderer resource layout.
 */
export function traceSvoMediaRay(
  inputRay: SvoMediaRay,
  source: SvoMediaBoundarySource,
  options: SvoMediaTraceOptions = {},
): SvoMediaTraceResult {
  finiteVec3(inputRay.origin_m, "Media ray origin");
  let direction = normalize(inputRay.direction, "Media ray direction");
  if (!Number.isFinite(inputRay.maximumDistance_m) || inputRay.maximumDistance_m < 0) {
    throw new RangeError("Media ray maximum distance must be finite and non-negative");
  }
  const limits = {
    boundaryQueries: bounded(options.maximumBoundaryQueries, SVO_MEDIA_LIMITS.boundaryQueries, SVO_MEDIA_LIMITS.boundaryQueries, "Media boundary-query budget"),
    transitions: bounded(options.maximumTransitions, SVO_MEDIA_LIMITS.transitions, SVO_MEDIA_LIMITS.transitions, "Media transition budget"),
    reflections: bounded(options.maximumReflections, SVO_MEDIA_LIMITS.reflections, SVO_MEDIA_LIMITS.reflections, "Media reflection budget"),
    transmissions: bounded(options.maximumTransmissions, SVO_MEDIA_LIMITS.transmissions, SVO_MEDIA_LIMITS.transmissions, "Media transmission budget"),
  };
  const coincidentEpsilon = options.coincidentBoundaryEpsilon_m ?? 1e-5;
  const continuationEpsilon = options.continuationEpsilon_m ?? coincidentEpsilon;
  if (!Number.isFinite(coincidentEpsilon) || coincidentEpsilon < 0
      || !Number.isFinite(continuationEpsilon) || continuationEpsilon < 0) {
    throw new RangeError("Media epsilons must be finite and non-negative");
  }
  let mediaStack = canonicalStack(options.initialMediaStack ?? ["air"]);
  let ray: SvoMediaRay = { origin_m: [...inputRay.origin_m], direction, maximumDistance_m: inputRay.maximumDistance_m };
  let throughput: [number, number, number] = [1, 1, 1];
  let distance_m = 0;
  const counts: SvoMediaTraceCounts = { boundaryQueries: 0, transitions: 0, reflections: 0, transmissions: 0 };

  for (let queryIndex = 0; queryIndex < limits.boundaryQueries; queryIndex += 1) {
    if (ray.maximumDistance_m <= 0) return { status: "escaped", ...failBase(throughput, counts, distance_m, mediaStack, direction) };
    counts.boundaryQueries += 1;
    const step = source({ ray, mediaStack, queryIndex });
    if (step.status === "invalid") return { status: "invalid", reason: step.reason, ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
    if (step.status === "exhausted") return { status: "exhausted", exhaustedBy: "source", reason: step.reason, ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
    const currentMedium = mediaStack.at(-1) as SvoMediumKind;
    if (step.status === "miss") {
      throughput = attenuateSvoMedium(throughput, currentMedium, ray.maximumDistance_m);
      distance_m += ray.maximumDistance_m;
      return { status: "escaped", ...failBase(throughput, counts, distance_m, mediaStack, direction) };
    }
    let resolved: SvoResolvedBoundaryGroup;
    try {
      resolved = resolveSvoMediumBoundaryGroup(mediaStack, step.boundaries, direction, coincidentEpsilon);
    } catch (error) {
      return { status: "invalid", reason: error instanceof Error ? error.message : String(error), ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
    }
    const hitDistance = Math.min(...step.boundaries.map((boundary) => boundary.t_m));
    if (hitDistance > ray.maximumDistance_m) {
      throughput = attenuateSvoMedium(throughput, currentMedium, ray.maximumDistance_m);
      distance_m += ray.maximumDistance_m;
      return { status: "escaped", ...failBase(throughput, counts, distance_m, mediaStack, direction) };
    }
    throughput = attenuateSvoMedium(throughput, currentMedium, hitDistance);
    distance_m += hitDistance;
    const hitPoint: SvoVec3 = [
      ray.origin_m[0] + direction[0] * hitDistance,
      ray.origin_m[1] + direction[1] * hitDistance,
      ray.origin_m[2] + direction[2] * hitDistance,
    ];
    if (resolved.opaque) {
      return {
        status: "opaque", boundary: resolved.opaque, mediumBefore: currentMedium,
        ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction),
      };
    }

    for (const wall of resolved.thinWalls) {
      if (counts.transitions >= limits.transitions) return { status: "exhausted", exhaustedBy: "transitions", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
      if (counts.transmissions >= limits.transmissions) return { status: "exhausted", exhaustedBy: "transmissions", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
      counts.transitions += 1;
      counts.transmissions += 1;
      const sheetTransmission = (1 - thinWallFresnel(wall, direction, currentMedium)) ** 2;
      const tint = thinTint(wall);
      throughput = [
        throughput[0] * sheetTransmission * tint[0],
        throughput[1] * sheetTransmission * tint[1],
        throughput[2] * sheetTransmission * tint[2],
      ];
      // A collapsed pair of parallel interfaces has no net Snell deflection.
    }

    if (resolved.from !== resolved.to) {
      if (counts.transitions >= limits.transitions) return { status: "exhausted", exhaustedBy: "transitions", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
      counts.transitions += 1;
      const transition = evaluateSvoDielectricTransition(direction, resolved.interfaceNormal as SvoVec3, resolved.from, resolved.to);
      if (transition.totalInternalReflection) {
        if (counts.reflections >= limits.reflections) return { status: "exhausted", exhaustedBy: "reflections", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
        counts.reflections += 1;
        direction = transition.reflectedDirection;
        // TIR never commits the proposed exit stack.
      } else {
        if (counts.transmissions >= limits.transmissions) return { status: "exhausted", exhaustedBy: "transmissions", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
        counts.transmissions += 1;
        throughput = throughput.map((channel) => channel * (1 - transition.fresnel)) as [number, number, number];
        direction = transition.refractedDirection as SvoVec3;
        mediaStack = [...resolved.nextStack];
      }
    } else {
      mediaStack = [...resolved.nextStack];
    }

    const remainingDistance = Math.max(0, ray.maximumDistance_m - hitDistance - continuationEpsilon);
    ray = {
      origin_m: [
        hitPoint[0] + direction[0] * continuationEpsilon,
        hitPoint[1] + direction[1] * continuationEpsilon,
        hitPoint[2] + direction[2] * continuationEpsilon,
      ],
      direction,
      maximumDistance_m: remainingDistance,
    };
  }
  return { status: "exhausted", exhaustedBy: "queries", ...failBase([0, 0, 0], counts, distance_m, mediaStack, direction) };
}

/**
 * Binding-free WGSL. The including renderer supplies
 * `svoMediaNext(ray, mediaStack, stackSize) -> SvoMediaStep` by composing its
 * existing SVO/fluid/primitive visibility helpers.
 */
export const svoMediaWGSL = /* wgsl */ `
const SVO_MEDIUM_AIR:u32=0u;const SVO_MEDIUM_WATER:u32=1u;const SVO_MEDIUM_GLASS:u32=2u;const SVO_MEDIUM_OPAQUE:u32=3u;
const SVO_MEDIA_ESCAPED:u32=0u;const SVO_MEDIA_OPAQUE:u32=1u;const SVO_MEDIA_EXHAUSTED:u32=2u;const SVO_MEDIA_INVALID:u32=3u;
const SVO_MEDIA_STEP_MISS:u32=0u;const SVO_MEDIA_STEP_HIT:u32=1u;const SVO_MEDIA_STEP_EXHAUSTED:u32=2u;const SVO_MEDIA_STEP_INVALID:u32=3u;
const SVO_MEDIA_THIN_WALL:u32=1u;const SVO_MEDIA_MAX_QUERIES:u32=16u;const SVO_MEDIA_MAX_TRANSITIONS:u32=8u;const SVO_MEDIA_MAX_REFLECTIONS:u32=4u;const SVO_MEDIA_MAX_TRANSMISSIONS:u32=8u;const SVO_MEDIA_MAX_COINCIDENT:u32=4u;
struct SvoMediaRay{origin_m:vec3f,maximumDistance_m:f32,direction:vec3f,_padding:f32}
struct SvoMediaBoundary{t_m:f32,medium:u32,flags:u32,_padding:u32,geometricNormal:vec3f,_padding2:f32,tint:vec3f,ior:f32}
struct SvoMediaStep{status:u32,count:u32,_padding:vec2u,boundaries:array<SvoMediaBoundary,4>}
struct SvoMediaBudget{queries:u32,transitions:u32,reflections:u32,transmissions:u32}
struct SvoMediaResult{status:u32,currentMedium:u32,stackSize:u32,_padding:u32,counts:SvoMediaBudget,throughput:vec3f,distance_m:f32,direction:vec3f,_padding2:f32}
struct SvoMediaRefraction{reflected:vec3f,refracted:vec3f,fresnel:f32,tir:u32}
fn svoMediaIor(medium:u32)->f32{if(medium==SVO_MEDIUM_WATER){return ${WATER_OPTICS.indexOfRefraction};}if(medium==SVO_MEDIUM_GLASS){return ${GLASS_OPTICS.indexOfRefraction};}return 1.0;}
fn svoMediaAbsorption(medium:u32)->vec3f{if(medium==SVO_MEDIUM_WATER){return vec3f(${WATER_OPTICS.absorption.join(",")});}return vec3f(0.0);}
fn svoMediaBeer(throughput:vec3f,medium:u32,distance_m:f32)->vec3f{return throughput*exp(-svoMediaAbsorption(medium)*max(distance_m,0.0));}
fn svoMediaRefract(incidentIn:vec3f,normalIn:vec3f,mediumFrom:u32,mediumTo:u32)->SvoMediaRefraction{let incident=normalize(incidentIn);var normal=normalize(normalIn);if(dot(incident,normal)>0.0){normal=-normal;}let eta=svoMediaIor(mediumFrom)/svoMediaIor(mediumTo);let cosine=clamp(-dot(incident,normal),0.0,1.0);let k=1.0-eta*eta*(1.0-cosine*cosine);let reflected=reflect(incident,normal);let ratio=(svoMediaIor(mediumFrom)-svoMediaIor(mediumTo))/(svoMediaIor(mediumFrom)+svoMediaIor(mediumTo));let f0=ratio*ratio;let fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);if(k<0.0){return SvoMediaRefraction(reflected,vec3f(0.0),1.0,1u);}return SvoMediaRefraction(reflected,normalize(eta*incident+(eta*cosine-sqrt(max(k,0.0)))*normal),fresnel,0u);}
fn svoMediaFail(status:u32,medium:u32,stackSize:u32,counts:SvoMediaBudget,distance_m:f32,direction:vec3f)->SvoMediaResult{return SvoMediaResult(status,medium,stackSize,0u,counts,vec3f(0.0),distance_m,direction,0.0);}
// Fixed-loop oracle; requires the including renderer's svoMediaNext adapter.
fn svoTraceMedia(rayIn:SvoMediaRay,initialMedium:u32,requested:SvoMediaBudget,coincidentEpsilon_m:f32,continuationEpsilon_m:f32)->SvoMediaResult{
  let limit=SvoMediaBudget(min(max(requested.queries,1u),SVO_MEDIA_MAX_QUERIES),min(max(requested.transitions,1u),SVO_MEDIA_MAX_TRANSITIONS),min(max(requested.reflections,1u),SVO_MEDIA_MAX_REFLECTIONS),min(max(requested.transmissions,1u),SVO_MEDIA_MAX_TRANSMISSIONS));var counts=SvoMediaBudget(0u,0u,0u,0u);var stack:array<u32,4>;stack[0]=SVO_MEDIUM_AIR;var stackSize=1u;if(initialMedium!=SVO_MEDIUM_AIR){stack[1]=initialMedium;stackSize=2u;}var ray=rayIn;var throughput=vec3f(1.0);var distance_m=0.0;
  for(var query=0u;query<SVO_MEDIA_MAX_QUERIES;query+=1u){if(query>=limit.queries){return svoMediaFail(SVO_MEDIA_EXHAUSTED,stack[stackSize-1u],stackSize,counts,distance_m,ray.direction);}counts.queries+=1u;let step=svoMediaNext(ray,stack,stackSize);if(step.status==SVO_MEDIA_STEP_MISS){throughput=svoMediaBeer(throughput,stack[stackSize-1u],ray.maximumDistance_m);return SvoMediaResult(SVO_MEDIA_ESCAPED,stack[stackSize-1u],stackSize,0u,counts,throughput,distance_m+ray.maximumDistance_m,ray.direction,0.0);}if(step.status==SVO_MEDIA_STEP_EXHAUSTED){return svoMediaFail(SVO_MEDIA_EXHAUSTED,stack[stackSize-1u],stackSize,counts,distance_m,ray.direction);}if(step.status!=SVO_MEDIA_STEP_HIT||step.count==0u||step.count>SVO_MEDIA_MAX_COINCIDENT){return svoMediaFail(SVO_MEDIA_INVALID,stack[stackSize-1u],stackSize,counts,distance_m,ray.direction);}var hitT=step.boundaries[0].t_m;for(var i=1u;i<SVO_MEDIA_MAX_COINCIDENT;i+=1u){if(i>=step.count){break;}hitT=min(hitT,step.boundaries[i].t_m);}if(hitT<0.0||hitT>ray.maximumDistance_m){return svoMediaFail(SVO_MEDIA_INVALID,stack[stackSize-1u],stackSize,counts,distance_m,ray.direction);}let incidentDirection=ray.direction;let before=stack[stackSize-1u];throughput=svoMediaBeer(throughput,before,hitT);distance_m+=hitT;
    for(var i=0u;i<SVO_MEDIA_MAX_COINCIDENT;i+=1u){if(i>=step.count){break;}let boundary=step.boundaries[i];if(abs(boundary.t_m-hitT)>coincidentEpsilon_m||(boundary.flags&SVO_MEDIA_THIN_WALL)!=0u&&boundary.medium!=SVO_MEDIUM_GLASS){return svoMediaFail(SVO_MEDIA_INVALID,before,stackSize,counts,distance_m,ray.direction);}if(boundary.medium<SVO_MEDIUM_OPAQUE&&(boundary.flags&SVO_MEDIA_THIN_WALL)==0u&&abs(dot(ray.direction,boundary.geometricNormal))<=1e-10){return svoMediaFail(SVO_MEDIA_INVALID,before,stackSize,counts,distance_m,ray.direction);}if(boundary.medium==SVO_MEDIUM_OPAQUE){return svoMediaFail(SVO_MEDIA_OPAQUE,before,stackSize,counts,distance_m,ray.direction);}}
    var priorStack:array<u32,4>;for(var i=0u;i<4u;i+=1u){priorStack[i]=stack[i];}let priorStackSize=stackSize;var interfaceNormal=vec3f(0.0);var hasInterface=false;
    // Exits precede entries, preventing a coincident water/glass pair from creating fake air.
    for(var i=0u;i<SVO_MEDIA_MAX_COINCIDENT;i+=1u){if(i>=step.count){break;}let b=step.boundaries[i];if(b.medium<SVO_MEDIUM_OPAQUE&&(b.flags&SVO_MEDIA_THIN_WALL)==0u&&dot(ray.direction,b.geometricNormal)>1e-10){if(stackSize<=1u||stack[stackSize-1u]!=b.medium){return svoMediaFail(SVO_MEDIA_INVALID,before,stackSize,counts,distance_m,ray.direction);}stackSize-=1u;if(!hasInterface){interfaceNormal=b.geometricNormal;hasInterface=true;}}}
    for(var i=0u;i<SVO_MEDIA_MAX_COINCIDENT;i+=1u){if(i>=step.count){break;}let b=step.boundaries[i];if(b.medium<SVO_MEDIUM_OPAQUE&&(b.flags&SVO_MEDIA_THIN_WALL)==0u&&dot(ray.direction,b.geometricNormal)<-1e-10){if(stackSize>=4u){return svoMediaFail(SVO_MEDIA_INVALID,before,stackSize,counts,distance_m,ray.direction);}stack[stackSize]=b.medium;stackSize+=1u;if(!hasInterface){interfaceNormal=b.geometricNormal;hasInterface=true;}}}
    for(var i=0u;i<SVO_MEDIA_MAX_COINCIDENT;i+=1u){if(i>=step.count){break;}let b=step.boundaries[i];if((b.flags&SVO_MEDIA_THIN_WALL)!=0u){if(counts.transitions>=limit.transitions||counts.transmissions>=limit.transmissions){return svoMediaFail(SVO_MEDIA_EXHAUSTED,before,stackSize,counts,distance_m,ray.direction);}counts.transitions+=1u;counts.transmissions+=1u;let glassIor=select(${GLASS_OPTICS.indexOfRefraction},b.ior,b.ior>0.0);let cosine=clamp(abs(dot(normalize(ray.direction),normalize(b.geometricNormal))),0.0,1.0);let ratio=(svoMediaIor(before)-glassIor)/(svoMediaIor(before)+glassIor);let f0=ratio*ratio;let fresnel=f0+(1.0-f0)*pow(1.0-cosine,5.0);throughput*=pow(1.0-fresnel,2.0)*clamp(b.tint,vec3f(0.0),vec3f(1.0));}}
    let after=stack[stackSize-1u];if(before!=after){if(counts.transitions>=limit.transitions){return svoMediaFail(SVO_MEDIA_EXHAUSTED,before,stackSize,counts,distance_m,ray.direction);}counts.transitions+=1u;let optical=svoMediaRefract(ray.direction,interfaceNormal,before,after);if(optical.tir!=0u){if(counts.reflections>=limit.reflections){return svoMediaFail(SVO_MEDIA_EXHAUSTED,before,stackSize,counts,distance_m,ray.direction);}counts.reflections+=1u;ray.direction=optical.reflected;stackSize=priorStackSize;for(var i=0u;i<4u;i+=1u){stack[i]=priorStack[i];}}else{if(counts.transmissions>=limit.transmissions){return svoMediaFail(SVO_MEDIA_EXHAUSTED,before,stackSize,counts,distance_m,ray.direction);}counts.transmissions+=1u;throughput*=1.0-optical.fresnel;ray.direction=optical.refracted;}}
    let point=ray.origin_m+incidentDirection*hitT;let epsilon=max(continuationEpsilon_m,0.0);ray.origin_m=point+ray.direction*epsilon;ray.maximumDistance_m=max(0.0,ray.maximumDistance_m-hitT-epsilon);
  }return svoMediaFail(SVO_MEDIA_EXHAUSTED,stack[stackSize-1u],stackSize,counts,distance_m,ray.direction);
}
`;
