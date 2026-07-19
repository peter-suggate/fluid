import type { SvoVec3 } from "./webgpu-svo-traversal";

export type SvoRenderLayer = "static" | "dynamic" | "fluid";
export type SvoRenderResidencyState = "active" | "core" | "halo" | "retired";
export type SvoRenderBrickCoordinate = readonly [number, number, number];

export const SVO_RENDER_RESIDENCY_LIMITS = Object.freeze({
  maximumChangedRegions: 4_096,
  maximumBrickRequests: 1_048_576,
  defaultHaloBricks: 1,
  defaultRetireAfterFrames: 3,
} as const);

export const SVO_RENDER_PUBLICATION_STAGES = Object.freeze({
  residency: 1 << 0,
  staticPayload: 1 << 1,
  dynamicPayload: 1 << 2,
  fluidPayload: 1 << 3,
  materialOwner: 1 << 4,
  summaries: 1 << 5,
} as const);

export interface SvoRenderBounds {
  minimum_m: SvoVec3;
  maximum_m: SvoVec3;
}

export interface SvoRenderBrickDomain {
  origin_m: SvoVec3;
  brickSize_m: SvoVec3;
  dimensionsBricks: SvoRenderBrickCoordinate;
}

export interface SvoRenderLayerRevisions {
  static: number;
  dynamic: number;
  fluid: number;
}

export interface SvoRenderRevisionBounds {
  static?: readonly SvoRenderBounds[];
  dynamic?: readonly SvoRenderBounds[];
  fluid?: readonly SvoRenderBounds[];
}

export interface SvoRenderRigidBoundsChange {
  previousBounds_m: SvoRenderBounds;
  currentBounds_m: SvoRenderBounds;
  padding_m?: number;
}

export interface SvoRenderFluidPreactivation {
  currentBounds_m: SvoRenderBounds;
  velocity_m_s: SvoVec3;
  deltaTime_s: number;
  padding_m?: number;
}

export interface SvoRenderBrickRequest {
  coordinate: SvoRenderBrickCoordinate;
  key: string;
  state: Exclude<SvoRenderResidencyState, "retired">;
  layers: readonly SvoRenderLayer[];
  causes: readonly string[];
}

export interface SvoRenderDirtyRequestInput {
  domain: SvoRenderBrickDomain;
  previousRevisions: SvoRenderLayerRevisions;
  currentRevisions: SvoRenderLayerRevisions;
  /** Undefined bounds for a changed revision conservatively dirties the complete domain. */
  changedBounds_m?: SvoRenderRevisionBounds;
  rigidChanges?: readonly SvoRenderRigidBoundsChange[];
  fluidPreactivation?: readonly SvoRenderFluidPreactivation[];
  haloBricks?: number;
  maximumRequests?: number;
}

export interface SvoRenderDirtyRequestPlan {
  requests: readonly SvoRenderBrickRequest[];
  changedLayers: readonly SvoRenderLayer[];
  completeDomainLayers: readonly SvoRenderLayer[];
}

export interface SvoRenderResidentBrick {
  coordinate: SvoRenderBrickCoordinate;
  key: string;
  state: SvoRenderResidencyState;
  layers: readonly SvoRenderLayer[];
  retiredFrames: number;
}

export interface SvoRenderResidencyPlanInput {
  desiredRequests: readonly SvoRenderBrickRequest[];
  previousResidents?: readonly SvoRenderResidentBrick[];
  capacity: number;
  retireAfterFrames?: number;
  /** Required invariant: every unallocated detail request can sample coarse data. */
  coarseCoverageComplete: boolean;
}

export interface SvoRenderResidencyPlan {
  residents: readonly SvoRenderResidentBrick[];
  unallocatedRequests: readonly SvoRenderBrickRequest[];
  capacity: number;
  coreCount: number;
  activeCount: number;
  haloCount: number;
  retiredCount: number;
  overflowCount: number;
  coverage: "fine-complete" | "coarse-fallback" | "incomplete";
  publishable: boolean;
}

export interface SvoRenderPublishedGeneration {
  completeGeneration: number;
  revisions: SvoRenderLayerRevisions;
}

export interface SvoRenderGenerationCandidate {
  targetGeneration: number;
  revisions: SvoRenderLayerRevisions;
  requiredStages: number;
  completedStages: number;
  payloadWritesComplete: boolean;
  residency: Pick<SvoRenderResidencyPlan, "publishable" | "coverage">;
}

export interface SvoRenderPublicationDecision {
  published: boolean;
  reason: "published" | "generation-order" | "incomplete-stages" | "incomplete-payload" | "incomplete-coverage";
  visible: SvoRenderPublishedGeneration;
}

const UINT32_MAX = 0xffff_ffff;
const LAYERS: readonly SvoRenderLayer[] = ["static", "dynamic", "fluid"];
const STATE_PRIORITY: Readonly<Record<SvoRenderResidencyState, number>> = Object.freeze({
  retired: 0,
  halo: 1,
  active: 2,
  core: 3,
});

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

function positiveSafeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function nonNegativeSafeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${label} must be a non-negative safe integer`);
  return value;
}

function validateBounds(bounds: SvoRenderBounds, label = "Render bounds"): void {
  finiteVec3(bounds.minimum_m, `${label} minimum`);
  finiteVec3(bounds.maximum_m, `${label} maximum`);
  if (bounds.maximum_m.some((value, axis) => value < bounds.minimum_m[axis])) {
    throw new RangeError(`${label} maximum must not be below its minimum`);
  }
}

function validateDomain(domain: SvoRenderBrickDomain): void {
  finiteVec3(domain.origin_m, "Render domain origin");
  finiteVec3(domain.brickSize_m, "Render brick size");
  if (domain.brickSize_m.some((component) => component <= 0)) throw new RangeError("Render brick size must be positive");
  if (domain.dimensionsBricks.some((component) => !Number.isSafeInteger(component) || component < 1)) {
    throw new RangeError("Render brick dimensions must be positive safe integers");
  }
}

function validateRevisions(revisions: SvoRenderLayerRevisions, label: string): void {
  for (const layer of LAYERS) uint32(revisions[layer], `${label} ${layer} revision`);
}

function coordinateKey(coordinate: SvoRenderBrickCoordinate): string {
  return `${coordinate[0]},${coordinate[1]},${coordinate[2]}`;
}

function compareCoordinates(a: SvoRenderBrickCoordinate, b: SvoRenderBrickCoordinate): number {
  return a[2] - b[2] || a[1] - b[1] || a[0] - b[0];
}

function domainBounds(domain: SvoRenderBrickDomain): SvoRenderBounds {
  return {
    minimum_m: [...domain.origin_m] as unknown as SvoVec3,
    maximum_m: [
      domain.origin_m[0] + domain.brickSize_m[0] * domain.dimensionsBricks[0],
      domain.origin_m[1] + domain.brickSize_m[1] * domain.dimensionsBricks[1],
      domain.origin_m[2] + domain.brickSize_m[2] * domain.dimensionsBricks[2],
    ],
  };
}

/** Conservative union of old/new rigid support, including optional SDF/filter padding. */
export function sweptSvoRenderBounds(
  previous: SvoRenderBounds,
  current: SvoRenderBounds,
  padding_m = 0,
): SvoRenderBounds {
  validateBounds(previous, "Previous rigid bounds");
  validateBounds(current, "Current rigid bounds");
  if (!Number.isFinite(padding_m) || padding_m < 0) throw new RangeError("Rigid bounds padding must be non-negative and finite");
  return {
    minimum_m: [
      Math.min(previous.minimum_m[0], current.minimum_m[0]) - padding_m,
      Math.min(previous.minimum_m[1], current.minimum_m[1]) - padding_m,
      Math.min(previous.minimum_m[2], current.minimum_m[2]) - padding_m,
    ],
    maximum_m: [
      Math.max(previous.maximum_m[0], current.maximum_m[0]) + padding_m,
      Math.max(previous.maximum_m[1], current.maximum_m[1]) + padding_m,
      Math.max(previous.maximum_m[2], current.maximum_m[2]) + padding_m,
    ],
  };
}

/** Current fluid bounds swept forward by published velocity for preactivation. */
export function velocitySweptSvoRenderBounds(input: SvoRenderFluidPreactivation): SvoRenderBounds {
  validateBounds(input.currentBounds_m, "Current fluid bounds");
  finiteVec3(input.velocity_m_s, "Fluid preactivation velocity");
  if (!Number.isFinite(input.deltaTime_s) || input.deltaTime_s < 0) throw new RangeError("Fluid preactivation delta time must be non-negative and finite");
  const padding_m = input.padding_m ?? 0;
  if (!Number.isFinite(padding_m) || padding_m < 0) throw new RangeError("Fluid preactivation padding must be non-negative and finite");
  const offset = input.velocity_m_s.map((component) => component * input.deltaTime_s) as [number, number, number];
  return sweptSvoRenderBounds(input.currentBounds_m, {
    minimum_m: input.currentBounds_m.minimum_m.map((component, axis) => component + offset[axis]) as unknown as SvoVec3,
    maximum_m: input.currentBounds_m.maximum_m.map((component, axis) => component + offset[axis]) as unknown as SvoVec3,
  }, padding_m);
}

function bricksForBounds(domain: SvoRenderBrickDomain, bounds: SvoRenderBounds): SvoRenderBrickCoordinate[] {
  validateBounds(bounds);
  const first = [0, 0, 0];
  const last = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const localMinimum = (bounds.minimum_m[axis] - domain.origin_m[axis]) / domain.brickSize_m[axis];
    const localMaximum = (bounds.maximum_m[axis] - domain.origin_m[axis]) / domain.brickSize_m[axis];
    first[axis] = Math.max(0, Math.floor(localMinimum));
    const inclusiveLast = localMaximum === localMinimum ? Math.floor(localMaximum) : Math.ceil(localMaximum) - 1;
    last[axis] = Math.min(domain.dimensionsBricks[axis] - 1, inclusiveLast);
    if (first[axis] >= domain.dimensionsBricks[axis] || last[axis] < 0 || last[axis] < first[axis]) return [];
  }
  const result: SvoRenderBrickCoordinate[] = [];
  for (let z = first[2]; z <= last[2]; z += 1) {
    for (let y = first[1]; y <= last[1]; y += 1) {
      for (let x = first[0]; x <= last[0]; x += 1) result.push([x, y, z]);
    }
  }
  return result;
}

function expandedBrickBounds(domain: SvoRenderBrickDomain, bounds: SvoRenderBounds, haloBricks: number): SvoRenderBounds {
  return {
    minimum_m: bounds.minimum_m.map((component, axis) => component - haloBricks * domain.brickSize_m[axis]) as unknown as SvoVec3,
    maximum_m: bounds.maximum_m.map((component, axis) => component + haloBricks * domain.brickSize_m[axis]) as unknown as SvoVec3,
  };
}

interface MutableRequest {
  coordinate: SvoRenderBrickCoordinate;
  state: Exclude<SvoRenderResidencyState, "retired">;
  layers: Set<SvoRenderLayer>;
  causes: Set<string>;
}

function mergeRequest(
  requests: Map<string, MutableRequest>,
  coordinate: SvoRenderBrickCoordinate,
  state: Exclude<SvoRenderResidencyState, "retired">,
  layer: SvoRenderLayer,
  cause: string,
): void {
  const key = coordinateKey(coordinate);
  const existing = requests.get(key);
  if (existing) {
    if (STATE_PRIORITY[state] > STATE_PRIORITY[existing.state]) existing.state = state;
    existing.layers.add(layer);
    existing.causes.add(cause);
    return;
  }
  requests.set(key, { coordinate, state, layers: new Set([layer]), causes: new Set([cause]) });
}

function addBoundsRequests(
  requests: Map<string, MutableRequest>,
  domain: SvoRenderBrickDomain,
  bounds: SvoRenderBounds,
  state: Exclude<SvoRenderResidencyState, "retired">,
  layer: SvoRenderLayer,
  cause: string,
  maximumRequests: number,
): void {
  for (const coordinate of bricksForBounds(domain, bounds)) {
    mergeRequest(requests, coordinate, state, layer, cause);
    if (requests.size > maximumRequests) throw new RangeError(`Render dirty brick requests exceed the ${maximumRequests} request cap`);
  }
}

function immutableRequests(requests: Map<string, MutableRequest>): SvoRenderBrickRequest[] {
  return [...requests].map(([key, request]) => ({
    coordinate: request.coordinate,
    key,
    state: request.state,
    layers: LAYERS.filter((layer) => request.layers.has(layer)),
    causes: [...request.causes].sort(),
  })).sort((a, b) => compareCoordinates(a.coordinate, b.coordinate));
}

/**
 * Build renderer payload work only when a layer revision changed. Static and
 * dynamic regions request active bricks; fluid surfaces request cores, with a
 * velocity-swept active preactivation band and lower-priority stencil halo.
 */
export function buildSvoRenderDirtyBrickRequests(input: SvoRenderDirtyRequestInput): SvoRenderDirtyRequestPlan {
  validateDomain(input.domain);
  validateRevisions(input.previousRevisions, "Previous");
  validateRevisions(input.currentRevisions, "Current");
  const maximumRequests = positiveSafeInteger(input.maximumRequests ?? SVO_RENDER_RESIDENCY_LIMITS.maximumBrickRequests, "Maximum render requests");
  if (maximumRequests > SVO_RENDER_RESIDENCY_LIMITS.maximumBrickRequests) {
    throw new RangeError(`Maximum render requests may not exceed ${SVO_RENDER_RESIDENCY_LIMITS.maximumBrickRequests}`);
  }
  const haloBricks = nonNegativeSafeInteger(input.haloBricks ?? SVO_RENDER_RESIDENCY_LIMITS.defaultHaloBricks, "Render halo bricks");
  const regionCount = LAYERS.reduce((sum, layer) => sum + (input.changedBounds_m?.[layer]?.length ?? 0), 0)
    + (input.rigidChanges?.length ?? 0) + (input.fluidPreactivation?.length ?? 0);
  if (regionCount > SVO_RENDER_RESIDENCY_LIMITS.maximumChangedRegions) {
    throw new RangeError(`Render changed regions exceed the ${SVO_RENDER_RESIDENCY_LIMITS.maximumChangedRegions} region cap`);
  }

  const requests = new Map<string, MutableRequest>();
  const changedLayers = LAYERS.filter((layer) => input.previousRevisions[layer] !== input.currentRevisions[layer]);
  const completeDomainLayers: SvoRenderLayer[] = [];
  for (const layer of changedLayers) {
    const regions = input.changedBounds_m?.[layer];
    if (regions === undefined) {
      completeDomainLayers.push(layer);
      addBoundsRequests(requests, input.domain, domainBounds(input.domain), layer === "fluid" ? "core" : "active", layer, `${layer}-revision-full-domain`, maximumRequests);
      continue;
    }
    for (const bounds of regions) {
      addBoundsRequests(requests, input.domain, bounds, layer === "fluid" ? "core" : "active", layer, `${layer}-revision-bounds`, maximumRequests);
    }
  }

  if (changedLayers.includes("dynamic")) {
    for (const change of input.rigidChanges ?? []) {
      addBoundsRequests(requests, input.domain, sweptSvoRenderBounds(
        change.previousBounds_m, change.currentBounds_m, change.padding_m,
      ), "active", "dynamic", "rigid-swept-bounds", maximumRequests);
    }
  }
  if (changedLayers.includes("fluid")) {
    for (const fluid of input.fluidPreactivation ?? []) {
      addBoundsRequests(requests, input.domain, fluid.currentBounds_m, "core", "fluid", "fluid-surface-core", maximumRequests);
      const swept = velocitySweptSvoRenderBounds(fluid);
      addBoundsRequests(requests, input.domain, swept, "active", "fluid", "fluid-velocity-preactivation", maximumRequests);
      if (haloBricks > 0) {
        addBoundsRequests(requests, input.domain, expandedBrickBounds(input.domain, swept, haloBricks), "halo", "fluid", "fluid-support-halo", maximumRequests);
      }
    }
  }
  return { requests: immutableRequests(requests), changedLayers, completeDomainLayers };
}

function validateRequest(request: SvoRenderBrickRequest): void {
  if (request.key !== coordinateKey(request.coordinate)) throw new RangeError(`Render request key ${request.key} does not match its coordinate`);
  if (request.coordinate.some((component) => !Number.isSafeInteger(component) || component < 0)) throw new RangeError("Render request coordinate must be non-negative integers");
}

function mergedDesiredRequests(inputs: readonly SvoRenderBrickRequest[]): SvoRenderBrickRequest[] {
  const requests = new Map<string, MutableRequest>();
  for (const input of inputs) {
    validateRequest(input);
    for (const layer of input.layers) {
      if (!LAYERS.includes(layer)) throw new RangeError(`Unknown render layer ${String(layer)}`);
      mergeRequest(requests, input.coordinate, input.state, layer, input.causes.join("+") || "desired");
    }
  }
  return immutableRequests(requests);
}

/**
 * Allocate desired detail in strict core -> active -> halo order, then retain
 * stale bricks as retired only when spare capacity permits. Overflow never
 * compromises coverage when the caller supplies the mandatory coarse field.
 */
export function planSvoRenderResidency(input: SvoRenderResidencyPlanInput): SvoRenderResidencyPlan {
  const capacity = nonNegativeSafeInteger(input.capacity, "Render residency capacity");
  const retireAfterFrames = positiveSafeInteger(
    input.retireAfterFrames ?? SVO_RENDER_RESIDENCY_LIMITS.defaultRetireAfterFrames,
    "Render retirement hysteresis",
  );
  const desired = mergedDesiredRequests(input.desiredRequests);
  const desiredKeys = new Set(desired.map((request) => request.key));
  const previous = new Map<string, SvoRenderResidentBrick>();
  for (const resident of input.previousResidents ?? []) {
    if (resident.key !== coordinateKey(resident.coordinate)) throw new RangeError(`Resident key ${resident.key} does not match its coordinate`);
    nonNegativeSafeInteger(resident.retiredFrames, "Resident retirement age");
    if (previous.has(resident.key)) throw new RangeError(`Duplicate previous resident ${resident.key}`);
    previous.set(resident.key, resident);
  }

  const retained: SvoRenderResidentBrick[] = [];
  for (const resident of previous.values()) {
    if (desiredKeys.has(resident.key)) continue;
    const retiredFrames = resident.state === "retired" ? resident.retiredFrames + 1 : 1;
    if (retiredFrames < retireAfterFrames) retained.push({ ...resident, state: "retired", retiredFrames });
  }
  retained.sort((a, b) => a.retiredFrames - b.retiredFrames || compareCoordinates(a.coordinate, b.coordinate));

  const orderedDesired = [...desired].sort((a, b) => STATE_PRIORITY[b.state] - STATE_PRIORITY[a.state]
    || compareCoordinates(a.coordinate, b.coordinate));
  const selectedDesired = orderedDesired.slice(0, capacity);
  const unallocatedRequests = orderedDesired.slice(capacity);
  const residents: SvoRenderResidentBrick[] = selectedDesired.map((request) => ({
    coordinate: request.coordinate,
    key: request.key,
    state: request.state,
    layers: request.layers,
    retiredFrames: 0,
  }));
  for (const stale of retained) {
    if (residents.length >= capacity) break;
    residents.push(stale);
  }
  residents.sort((a, b) => STATE_PRIORITY[b.state] - STATE_PRIORITY[a.state] || compareCoordinates(a.coordinate, b.coordinate));
  const count = (state: SvoRenderResidencyState) => residents.filter((resident) => resident.state === state).length;
  const overflowCount = unallocatedRequests.length;
  const coverage = !input.coarseCoverageComplete ? "incomplete"
    : overflowCount > 0 ? "coarse-fallback" : "fine-complete";
  return {
    residents,
    unallocatedRequests,
    capacity,
    coreCount: count("core"),
    activeCount: count("active"),
    haloCount: count("halo"),
    retiredCount: count("retired"),
    overflowCount,
    coverage,
    publishable: input.coarseCoverageComplete,
  };
}

/** Publish an all-or-old snapshot: failed candidates leave every visible revision untouched. */
export function gateSvoRenderGenerationPublication(
  visible: SvoRenderPublishedGeneration,
  candidate: SvoRenderGenerationCandidate,
): SvoRenderPublicationDecision {
  uint32(visible.completeGeneration, "Visible complete generation");
  validateRevisions(visible.revisions, "Visible");
  uint32(candidate.targetGeneration, "Candidate generation");
  validateRevisions(candidate.revisions, "Candidate");
  uint32(candidate.requiredStages, "Required publication stages");
  uint32(candidate.completedStages, "Completed publication stages");
  if (candidate.targetGeneration !== ((visible.completeGeneration + 1) >>> 0)) {
    return { published: false, reason: "generation-order", visible };
  }
  if ((candidate.completedStages & candidate.requiredStages) !== candidate.requiredStages) {
    return { published: false, reason: "incomplete-stages", visible };
  }
  if (!candidate.payloadWritesComplete) return { published: false, reason: "incomplete-payload", visible };
  if (!candidate.residency.publishable || candidate.residency.coverage === "incomplete") {
    return { published: false, reason: "incomplete-coverage", visible };
  }
  return {
    published: true,
    reason: "published",
    visible: { completeGeneration: candidate.targetGeneration, revisions: { ...candidate.revisions } },
  };
}

/** Binding-free mirror for the final atomic publication pass. */
export const svoRenderGenerationGateWGSL = /* wgsl */ `
struct SvoRenderGenerationGate {
  visibleGeneration:u32,
  targetGeneration:u32,
  requiredStages:u32,
  completedStages:u32,
  payloadWritesComplete:u32,
  coarseCoverageComplete:u32,
  padding0:u32,
  padding1:u32,
}
fn svoRenderGenerationReady(gate:SvoRenderGenerationGate)->bool{
  let ordered=gate.targetGeneration==gate.visibleGeneration+1u;
  let stages=(gate.completedStages&gate.requiredStages)==gate.requiredStages;
  return ordered&&stages&&gate.payloadWritesComplete!=0u&&gate.coarseCoverageComplete!=0u;
}
`;
