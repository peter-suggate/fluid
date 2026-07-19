import { intersectSvoPrimitive, type SvoPrimitiveDescriptor, type SvoPrimitiveRayHit } from "./svo-primitive-abi";
import {
  traceSvoStructuralCoarseFluid,
  type SvoStructuralVisibilityOptions,
} from "./svo-fluid-structural-visibility";
import type { SvoStructuralFluidPackedFixture } from "./svo-fluid-structural-sampling";
import { intersectSvoThinGlassPane, type SvoThinGlassPane } from "./svo-thin-glass";
import { SPARSE_BRICK_NO_OWNER } from "./sparse-brick-octree";
import type { SvoMediumKind } from "./svo-media";
import type { TerrainDescription } from "./terrain";
import { intersectSvoTerrainHeightfield } from "./webgpu-svo-dry-scene";
import { intersectSvoRayAabb, type SvoVec3 } from "./webgpu-svo-traversal";

export const SVO_PICKING_DEFAULT_COINCIDENCE_EPSILON_M = 1e-5;
export const SVO_PICKING_DEFAULT_MAXIMUM_ANALYTIC_TESTS = 4_096;
export const SVO_PICKING_MAXIMUM_ANALYTIC_TESTS = 65_536;

export const SVO_PICKING_SOURCES = Object.freeze({
  primitive: 0,
  terrain: 1,
  thinGlass: 2,
  structuralCoarseFluid: 3,
} as const);

export type SvoPickingSource = keyof typeof SVO_PICKING_SOURCES;
export type SvoPickingBoundaryMedium = SvoMediumKind | "opaque";

export interface SvoPickingRay {
  origin_m: SvoVec3;
  direction: SvoVec3;
  tMin_m?: number;
  tMax_m: number;
}

export interface SvoPickingTerrainSource {
  description: TerrainDescription;
  sceneScale_m: number;
  materialId: number;
  localTopologyGeneration?: number;
  normalEpsilon_m?: number;
}

export interface SvoPickingFluidSource {
  source: SvoStructuralFluidPackedFixture;
  materialId: number;
  options?: SvoStructuralVisibilityOptions;
}

export interface SvoPickingScene {
  primitives?: readonly SvoPrimitiveDescriptor[];
  /** Static analytic records normally use generation zero. */
  primitiveLocalTopologyGeneration?: number | ((hit: SvoPrimitiveRayHit) => number);
  terrain?: SvoPickingTerrainSource;
  thinGlass?: readonly SvoThinGlassPane[];
  thinGlassLocalTopologyGeneration?: number;
  structuralFluid?: SvoPickingFluidSource;
  /** Existing body IDs in owner-index order. Environment owners follow this range. */
  rigidBodyIds?: readonly string[];
  incidentMedium?: SvoMediumKind;
}

export interface SvoPickingOptions {
  coincidenceEpsilon_m?: number;
  maximumAnalyticTests?: number;
}

export interface SvoPickingCandidate {
  distance_m: number;
  position_m: SvoVec3;
  geometricNormal: SvoVec3;
  materialId: number;
  ownerId: number;
  mediumBefore: SvoMediumKind;
  mediumAfter: SvoPickingBoundaryMedium;
  boundaryMedium: SvoPickingBoundaryMedium;
  source: SvoPickingSource;
  featureId: number;
  localTopologyGeneration: number;
  /** Stable input order within one source; lower wins exact same-source ties. */
  sourceIndex: number;
}

export type SvoPickingInteraction =
  | { kind: "rigid"; rigidBodyIndex: number; rigidBodyId: string }
  | { kind: "none"; reason: "fluid" | "terrain" | "no-owner" | "environment-owner" };

export interface SvoPickingWork {
  analyticTests: number;
  primitiveTests: number;
  glassTests: number;
  terrainHeightEvaluations: number;
  fluidSteps: number;
  fluidTopologyNodeVisits: number;
}

export type SvoPickingFailureReason =
  | "analytic-work-exhausted"
  | "fluid-work-exhausted"
  | "fluid-invalid-field"
  | "fluid-stale-generation"
  | "fluid-nonresident";

export type SvoPickingResult =
  | { status: "hit"; hit: SvoPickingCandidate; interaction: SvoPickingInteraction; work: SvoPickingWork }
  | { status: "miss"; work: SvoPickingWork }
  | { status: "work-exhausted"; reason: SvoPickingFailureReason; work: SvoPickingWork }
  | { status: "invalid"; reason: SvoPickingFailureReason; work: SvoPickingWork };

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function normalized(value: SvoVec3): SvoVec3 {
  const magnitude = Math.hypot(...value);
  if (!(magnitude > 1e-12)) throw new RangeError("Picking ray direction must be non-zero");
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function uint16(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff) throw new RangeError(`${label} must fit uint16`);
  return value;
}

function canonicalRay(ray: SvoPickingRay) {
  finiteVec3(ray.origin_m, "Picking ray origin");
  finiteVec3(ray.direction, "Picking ray direction");
  const direction = normalized(ray.direction);
  const tMin_m = ray.tMin_m ?? 0;
  if (!Number.isFinite(tMin_m) || tMin_m < 0 || !Number.isFinite(ray.tMax_m) || ray.tMax_m < tMin_m) {
    throw new RangeError("Picking ray interval must be finite, non-negative, and ordered");
  }
  return { origin_m: [...ray.origin_m] as SvoVec3, direction, tMin_m, tMax_m: ray.tMax_m };
}

function settings(options: SvoPickingOptions) {
  const coincidenceEpsilon_m = options.coincidenceEpsilon_m ?? SVO_PICKING_DEFAULT_COINCIDENCE_EPSILON_M;
  if (!Number.isFinite(coincidenceEpsilon_m) || coincidenceEpsilon_m < 0) {
    throw new RangeError("Picking coincidence epsilon must be finite and non-negative");
  }
  const maximumAnalyticTests = options.maximumAnalyticTests ?? SVO_PICKING_DEFAULT_MAXIMUM_ANALYTIC_TESTS;
  if (!Number.isInteger(maximumAnalyticTests) || maximumAnalyticTests < 1 || maximumAnalyticTests > SVO_PICKING_MAXIMUM_ANALYTIC_TESTS) {
    throw new RangeError(`Picking analytic-test cap must be from 1 to ${SVO_PICKING_MAXIMUM_ANALYTIC_TESTS}`);
  }
  return { coincidenceEpsilon_m, maximumAnalyticTests };
}

function sourcePriority(source: SvoPickingSource): number { return SVO_PICKING_SOURCES[source]; }

/** Deterministic nearest selection with opaque/exact authored coincidence priority. */
export function resolveNearestSvoPickingCandidate(
  candidates: readonly SvoPickingCandidate[],
  coincidenceEpsilon_m = SVO_PICKING_DEFAULT_COINCIDENCE_EPSILON_M,
): SvoPickingCandidate | undefined {
  if (!Number.isFinite(coincidenceEpsilon_m) || coincidenceEpsilon_m < 0) throw new RangeError("Picking coincidence epsilon must be finite and non-negative");
  let best: SvoPickingCandidate | undefined;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate.distance_m) || candidate.distance_m < 0) throw new RangeError("Picking candidate distance must be non-negative and finite");
    if (!best || candidate.distance_m < best.distance_m - coincidenceEpsilon_m) { best = candidate; continue; }
    if (Math.abs(candidate.distance_m - best.distance_m) > coincidenceEpsilon_m) continue;
    const priority = sourcePriority(candidate.source) - sourcePriority(best.source);
    if (priority < 0 || priority === 0 && candidate.sourceIndex < best.sourceIndex) best = candidate;
  }
  return best;
}

/** Only scene owner indices inside the existing rigid-body range are interactive. */
export function svoPickingInteractionForHit(
  hit: SvoPickingCandidate,
  rigidBodyIds: readonly string[],
): SvoPickingInteraction {
  if (hit.source === "structuralCoarseFluid") return { kind: "none", reason: "fluid" };
  if (hit.source === "terrain") return { kind: "none", reason: "terrain" };
  if (hit.ownerId === SPARSE_BRICK_NO_OWNER) return { kind: "none", reason: "no-owner" };
  if (hit.ownerId >= rigidBodyIds.length) return { kind: "none", reason: "environment-owner" };
  return { kind: "rigid", rigidBodyIndex: hit.ownerId, rigidBodyId: rigidBodyIds[hit.ownerId] };
}

function emptyWork(): SvoPickingWork {
  return { analyticTests: 0, primitiveTests: 0, glassTests: 0, terrainHeightEvaluations: 0, fluidSteps: 0, fluidTopologyNodeVisits: 0 };
}

function primitiveGeneration(scene: SvoPickingScene, hit: SvoPrimitiveRayHit): number {
  const value = typeof scene.primitiveLocalTopologyGeneration === "function"
    ? scene.primitiveLocalTopologyGeneration(hit) : scene.primitiveLocalTopologyGeneration ?? 0;
  return uint32(value, "Primitive local topology generation");
}

function primitiveCandidate(
  hit: SvoPrimitiveRayHit,
  sourceIndex: number,
  scene: SvoPickingScene,
  incidentMedium: SvoMediumKind,
): SvoPickingCandidate {
  return {
    distance_m: hit.t_m,
    position_m: [hit.position_m.x, hit.position_m.y, hit.position_m.z],
    geometricNormal: [hit.normal.x, hit.normal.y, hit.normal.z],
    materialId: uint16(hit.materialId, "Primitive material ID"),
    ownerId: uint16(hit.ownerId, "Primitive owner ID"),
    mediumBefore: incidentMedium,
    mediumAfter: "opaque",
    boundaryMedium: "opaque",
    source: "primitive",
    featureId: uint32(hit.featureId, "Primitive feature ID"),
    localTopologyGeneration: primitiveGeneration(scene, hit),
    sourceIndex,
  };
}

function fluidBounds(source: SvoStructuralFluidPackedFixture) {
  const minimum = source.domain.worldOrigin_m;
  const maximum = minimum.map((origin, axis) => origin + source.domain.dimensionsCells[axis] * source.domain.cellSize_m[axis]) as [number, number, number];
  return { minimum, maximum };
}

function fluidFailureReason(reason: string | undefined): SvoPickingFailureReason {
  if (reason === "generation-mismatch" || reason === "publication-incomplete") return "fluid-stale-generation";
  if (reason === "nonresident-leaf" || reason === "missing-leaf") return "fluid-nonresident";
  return "fluid-invalid-field";
}

/**
 * Compose exact analytic sources and the continuous structural coarse field.
 * An invalid fluid source fails closed because an unseen fluid hit could be in
 * front of every otherwise valid analytic candidate.
 */
export function pickSvoScene(
  scene: SvoPickingScene,
  rayInput: SvoPickingRay,
  options: SvoPickingOptions = {},
): SvoPickingResult {
  const ray = canonicalRay(rayInput);
  const config = settings(options);
  const incidentMedium = scene.incidentMedium ?? "air";
  const primitiveCount = scene.primitives?.length ?? 0;
  const glassCount = scene.thinGlass?.length ?? 0;
  const analyticTests = primitiveCount + glassCount + (scene.terrain ? 1 : 0);
  const work = emptyWork();
  if (analyticTests > config.maximumAnalyticTests) {
    work.analyticTests = config.maximumAnalyticTests;
    return { status: "work-exhausted", reason: "analytic-work-exhausted", work };
  }

  const candidates: SvoPickingCandidate[] = [];
  scene.primitives?.forEach((primitive, sourceIndex) => {
    work.analyticTests += 1;
    work.primitiveTests += 1;
    if (primitive.kind === "terrain-heightfield") return;
    const hit = intersectSvoPrimitive(primitive, {
      origin_m: { x: ray.origin_m[0], y: ray.origin_m[1], z: ray.origin_m[2] },
      direction: { x: ray.direction[0], y: ray.direction[1], z: ray.direction[2] },
      tMin_m: ray.tMin_m,
      tMax_m: ray.tMax_m,
    });
    if (hit) candidates.push(primitiveCandidate(hit, sourceIndex, scene, incidentMedium));
  });

  if (scene.terrain) {
    work.analyticTests += 1;
    const terrain = scene.terrain;
    const hit = intersectSvoTerrainHeightfield(
      terrain.description,
      { x: ray.origin_m[0], y: ray.origin_m[1], z: ray.origin_m[2] },
      { x: ray.direction[0], y: ray.direction[1], z: ray.direction[2] },
      terrain.sceneScale_m,
      terrain.normalEpsilon_m,
    );
    if (hit) {
      work.terrainHeightEvaluations = hit.heightEvaluations;
      if (hit.t_m >= ray.tMin_m && hit.t_m <= ray.tMax_m) candidates.push({
        distance_m: hit.t_m,
        position_m: [hit.position_m.x, hit.position_m.y, hit.position_m.z],
        geometricNormal: [hit.normal.x, hit.normal.y, hit.normal.z],
        materialId: uint16(terrain.materialId, "Terrain material ID"),
        ownerId: SPARSE_BRICK_NO_OWNER,
        mediumBefore: incidentMedium,
        mediumAfter: "opaque",
        boundaryMedium: "opaque",
        source: "terrain",
        featureId: 0,
        localTopologyGeneration: uint32(terrain.localTopologyGeneration ?? 0, "Terrain local topology generation"),
        sourceIndex: 0,
      });
    }
  }

  scene.thinGlass?.forEach((pane, sourceIndex) => {
    work.analyticTests += 1;
    work.glassTests += 1;
    const hit = intersectSvoThinGlassPane(pane, {
      origin_m: ray.origin_m, direction: ray.direction, tMin_m: ray.tMin_m, tMax_m: ray.tMax_m,
    });
    if (!hit) return;
    candidates.push({
      distance_m: hit.t_m,
      position_m: hit.position_m,
      geometricNormal: hit.geometricNormal,
      materialId: uint16(hit.materialId, "Thin-glass material ID"),
      ownerId: uint16(hit.ownerId, "Thin-glass owner ID"),
      mediumBefore: incidentMedium,
      mediumAfter: incidentMedium,
      boundaryMedium: "glass",
      source: "thinGlass",
      featureId: uint32(hit.featureId, "Thin-glass feature ID"),
      localTopologyGeneration: uint32(scene.thinGlassLocalTopologyGeneration ?? 0, "Thin-glass local topology generation"),
      sourceIndex,
    });
  });

  if (scene.structuralFluid) {
    const fluid = scene.structuralFluid;
    const interval = intersectSvoRayAabb(
      { origin: ray.origin_m, direction: ray.direction, tMin: ray.tMin_m, tMax: ray.tMax_m },
      fluidBounds(fluid.source),
    );
    if (interval) {
      const cellEpsilon = 1e-6 * Math.min(...fluid.source.domain.cellSize_m);
      const tMin_m = Math.min(interval.tExit, interval.tEnter + cellEpsilon);
      const tMax_m = Math.max(tMin_m, interval.tExit - cellEpsilon);
      if (tMax_m > tMin_m) {
        const result = traceSvoStructuralCoarseFluid(fluid.source, {
          origin_m: ray.origin_m, direction: ray.direction, tMin_m, tMax_m,
        }, fluid.options);
        work.fluidSteps = result.steps;
        work.fluidTopologyNodeVisits = result.diagnostics.topologyNodeVisits;
        if (result.status === "work-exhausted") {
          return { status: "work-exhausted", reason: "fluid-work-exhausted", work };
        }
        if (result.status === "invalid-field") {
          return { status: "invalid", reason: fluidFailureReason(result.diagnostics.failureReason), work };
        }
        if (result.status === "hit") {
          candidates.push({
            distance_m: result.t_m,
            position_m: result.position_m,
            geometricNormal: result.normal,
            materialId: uint16(fluid.materialId, "Fluid material ID"),
            ownerId: SPARSE_BRICK_NO_OWNER,
            mediumBefore: result.insideFluidAtStart ? "water" : incidentMedium,
            mediumAfter: result.insideFluidAtStart ? incidentMedium : "water",
            boundaryMedium: "water",
            source: "structuralCoarseFluid",
            featureId: 0,
            localTopologyGeneration: uint32(result.diagnostics.completeGeneration, "Fluid local topology generation"),
            sourceIndex: 0,
          });
        }
      }
    }
  }

  const hit = resolveNearestSvoPickingCandidate(candidates, config.coincidenceEpsilon_m);
  if (!hit) return { status: "miss", work };
  return { status: "hit", hit, interaction: svoPickingInteractionForHit(hit, scene.rigidBodyIds ?? []), work };
}

/** Binding-free candidate/tie/owner adapter for the future GPU picking pass. */
export const svoPickingWGSL = /* wgsl */ `
const SVO_PICK_STATUS_MISS:u32=0u;const SVO_PICK_STATUS_HIT:u32=1u;const SVO_PICK_STATUS_EXHAUSTED:u32=2u;const SVO_PICK_STATUS_INVALID:u32=3u;
const SVO_PICK_SOURCE_PRIMITIVE:u32=0u;const SVO_PICK_SOURCE_TERRAIN:u32=1u;const SVO_PICK_SOURCE_THIN_GLASS:u32=2u;const SVO_PICK_SOURCE_FLUID_COARSE:u32=3u;
const SVO_PICK_NO_OWNER:u32=0xffffu;const SVO_PICK_DEFAULT_COINCIDENCE_EPSILON_M:f32=1e-5;
struct SvoPickingCandidate{valid:u32,source:u32,sourceIndex:u32,featureId:u32,distance_m:f32,_padding:vec3f,position_m:vec3f,materialId:u32,geometricNormal:vec3f,ownerId:u32,mediumBefore:u32,mediumAfter:u32,localTopologyGeneration:u32,_padding2:u32}
struct SvoPickingResult{status:u32,failure:u32,interactionOwner:u32,_padding:u32,hit:SvoPickingCandidate}
fn svoPickingMissCandidate()->SvoPickingCandidate{return SvoPickingCandidate(0u,0u,0u,0u,0.0,vec3f(0.0),vec3f(0.0),0u,vec3f(0.0,1.0,0.0),SVO_PICK_NO_OWNER,0u,0u,0u,0u);}
fn svoPickingPrefer(candidate:SvoPickingCandidate,best:SvoPickingCandidate,epsilon_m:f32)->bool{if(candidate.valid==0u){return false;}if(best.valid==0u||candidate.distance_m<best.distance_m-epsilon_m){return true;}if(abs(candidate.distance_m-best.distance_m)>epsilon_m){return false;}return candidate.source<best.source||(candidate.source==best.source&&candidate.sourceIndex<best.sourceIndex);}
fn svoPickingRigidOwner(hit:SvoPickingCandidate,rigidOwnerCount:u32)->u32{if(hit.source==SVO_PICK_SOURCE_FLUID_COARSE||hit.source==SVO_PICK_SOURCE_TERRAIN||hit.ownerId==SVO_PICK_NO_OWNER||hit.ownerId>=rigidOwnerCount){return SVO_PICK_NO_OWNER;}return hit.ownerId;}
fn svoPickingComplete(best:SvoPickingCandidate,rigidOwnerCount:u32)->SvoPickingResult{if(best.valid==0u){return SvoPickingResult(SVO_PICK_STATUS_MISS,0u,SVO_PICK_NO_OWNER,0u,best);}return SvoPickingResult(SVO_PICK_STATUS_HIT,0u,svoPickingRigidOwner(best,rigidOwnerCount),0u,best);}
fn svoPickingFail(status:u32,failure:u32)->SvoPickingResult{return SvoPickingResult(status,failure,SVO_PICK_NO_OWNER,0u,svoPickingMissCandidate());}
`;
