/**
 * Binding-neutral sampling contract for consumers of the unified octree.
 *
 * A caller resolves the leaf containing a query point and gathers the two
 * bounded incidence slabs that surround it. The routines below then perform
 * the same resolution-aware reconstruction for rendering, particles,
 * diagnostics, and face transport. No dense 3D texture is part
 * of this ABI.
 *
 * It sits in core rather than beside the octree because its callers are the
 * renderer and the water pipeline: a consumer that had to import the solver to
 * learn how to read its publication would drag the whole adaptive stack into
 * the render graph, which is the coupling the method split exists to remove.
 */

import type {
  CoarseLevelSetConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "./levelset-consumer-abi";

export const OCTREE_CONSUMER_INVALID = 0xffff_ffff;
export const OCTREE_CONSUMER_MAX_FACE_CANDIDATES = 48;

export type OctreeConsumerPoint = readonly [number, number, number];

export interface OctreeConsumerFaceSample {
  readonly origin: OctreeConsumerPoint;
  readonly axis: 0 | 1 | 2;
  readonly span: number;
  readonly normalVelocity: number;
}

function finitePoint(point: OctreeConsumerPoint, label: string): void {
  if (point.some((value) => !Number.isFinite(value))) throw new RangeError(`${label} must be finite`);
}

function component(value: OctreeConsumerPoint, axis: number): number {
  return value[axis];
}

export function octreeConsumerFaceCentre(face: OctreeConsumerFaceSample): [number, number, number] {
  finitePoint(face.origin, "Octree face origin");
  if (face.axis < 0 || face.axis > 2 || !Number.isInteger(face.axis)) throw new RangeError("Octree face axis must be 0, 1, or 2");
  if (!Number.isFinite(face.span) || face.span <= 0) throw new RangeError("Octree face span must be positive");
  const centre: [number, number, number] = [...face.origin];
  centre[(face.axis + 1) % 3] += face.span * 0.5;
  centre[(face.axis + 2) % 3] += face.span * 0.5;
  return centre;
}

/**
 * Samples one staggered component from a bounded canonical-face neighbourhood.
 * Coordinates are in finest-cell units. Weights form a span-aware tent
 * product: on a uniform lattice this is exact per-axis staggered (MAC)
 * interpolation that reproduces each face's own sample; across 2:1
 * transitions it degrades to a normalized partition-of-unity blend. It
 * matches the WGSL and U3 transport sampler.
 */
export function sampleOctreeFaceComponent(
  pointFine: OctreeConsumerPoint,
  axis: 0 | 1 | 2,
  candidates: readonly OctreeConsumerFaceSample[],
  defaultValue = 0,
): number {
  finitePoint(pointFine, "Octree velocity query");
  if (candidates.length > OCTREE_CONSUMER_MAX_FACE_CANDIDATES) {
    throw new RangeError(`Octree velocity query exceeds the ${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}-face 2:1 bound`);
  }
  let weighted = 0;
  let weights = 0;
  let nearest = defaultValue;
  let nearestDistanceSquared = Number.POSITIVE_INFINITY;
  for (const face of candidates) {
    if (face.axis !== axis) continue;
    const centre = octreeConsumerFaceCentre(face);
    const dx = pointFine[0] - centre[0];
    const dy = pointFine[1] - centre[1];
    const dz = pointFine[2] - centre[2];
    const distanceSquared = dx * dx + dy * dy + dz * dz;
    if (distanceSquared < nearestDistanceSquared) {
      nearestDistanceSquared = distanceSquared;
      nearest = face.normalVelocity;
    }
    const support = Math.max(1, face.span);
    const weight = Math.max(0, 1 - Math.abs(dx) / support)
      * Math.max(0, 1 - Math.abs(dy) / support)
      * Math.max(0, 1 - Math.abs(dz) / support);
    weighted += weight * face.normalVelocity;
    weights += weight;
  }
  return weights > 0 ? weighted / weights : nearest;
}

export function sampleOctreeFaceVelocity(
  pointFine: OctreeConsumerPoint,
  candidates: readonly OctreeConsumerFaceSample[],
  defaultValue: OctreeConsumerPoint = [0, 0, 0],
): [number, number, number] {
  return [0, 1, 2].map((axis) => sampleOctreeFaceComponent(
    pointFine, axis as 0 | 1 | 2, candidates, component(defaultValue, axis),
  )) as [number, number, number];
}

export interface GlobalFineLevelSetConsumerSource {
  readonly kind: "global-fine-levelset-sampling";
  readonly metadata: GPUBufferBinding;
  readonly worklist: GPUBufferBinding;
  readonly samples: GPUBufferBinding;
  readonly coarsePhiDirectory?: GPUBufferBinding;
  readonly coarsePhiRowCapacity?: number;
  /** GPU transaction that published the selected A/B fine slot. */
  readonly topologyControl?: GPUBufferBinding;
  readonly sampleDimensions: readonly [number, number, number];
  readonly brickDimensions: readonly [number, number, number];
  readonly brickResolution: 4 | 8 | 16;
  readonly samplesPerBrick: number;
  readonly pageCapacity: number;
  readonly fineFactor: 1 | 4 | 8;
  readonly fineCellWidth: number;
  readonly domainOrigin: readonly [number, number, number];
  readonly generation: number;
}

export function validateCoarseLevelSetConsumerSource(source: CoarseLevelSetConsumerSource): void {
  if (source.kind !== "coarse-levelset-sampling") throw new RangeError("Coarse source kind is invalid");
  for (const value of source.sampleDimensions) {
    if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff) {
      throw new RangeError("Coarse sample dimensions must be positive 16-bit integers");
    }
  }
  if (!Number.isSafeInteger(source.rowCapacity) || source.rowCapacity < 1) {
    throw new RangeError("Coarse phi row capacity must be a positive integer");
  }
  if (!Number.isSafeInteger(source.generation) || source.generation < 1) {
    throw new RangeError("Coarse phi generation must be a positive integer");
  }
  if (!Number.isFinite(source.physicalCellSize) || source.physicalCellSize <= 0
    || source.domainOrigin.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Coarse phi physical coordinates must be finite with positive spacing");
  }
}

/** CPU mirror of the renderer's fail-closed Section-5 publication gate.
 * Aanjaneya et al. (2017), Section 5 rebuilds the fine SPGrid after every
 * advection and then uses it to correct the background-octree level set. A
 * rejected rebuild cannot authorize that correction with stale fine data. */
export function globalFineCoarseGenerationPairIsValid(
  fineGeneration: number,
  coarseGeneration: number,
  topologyControl: ArrayLike<number> | undefined,
): boolean {
  if (!topologyControl || topologyControl.length < 8) return false;
  const mask = 0x3fff_ffff;
  const fine = fineGeneration & mask, coarse = coarseGeneration & mask;
  const flags = Number(topologyControl[0]) >>> 0;
  const published = topologyControl[4] === 1, rolledBack = topologyControl[5] === 1;
  const reason = Number(topologyControl[7]) >>> 0;
  const cleanCurrent = flags === 0 && published && !rolledBack && reason === 0
    && coarse === fine;
  return cleanCurrent;
}

/** Validates the indexable Section-5 fine-SPGrid ABI without reading GPU data. */
export function validateGlobalFineLevelSetConsumerSource(source: GlobalFineLevelSetConsumerSource): void {
  if (source.kind !== "global-fine-levelset-sampling") throw new RangeError("Global fine source kind is invalid");
  const positiveInteger = (value: number, label: string) => {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  };
  source.sampleDimensions.forEach((value) => positiveInteger(value, "Global fine sample dimension"));
  if (source.sampleDimensions.some((value) => value > 0xffff)) {
    throw new RangeError("Global fine sample dimensions exceed the renderer's 16-bit cube-coordinate ABI");
  }
  source.brickDimensions.forEach((value) => positiveInteger(value, "Global fine brick dimension"));
  if (source.brickResolution !== 4 && source.brickResolution !== 8
    && source.brickResolution !== 16) {
    throw new RangeError("Global fine brick resolution must be 4, 8, or 16");
  }
  if (source.fineFactor !== 1 && source.fineFactor !== 4 && source.fineFactor !== 8) {
    throw new RangeError("Global fine factor must be 1, 4, or 8");
  }
  if (source.samplesPerBrick !== source.brickResolution ** 3) {
    throw new RangeError("Global fine sample stride must equal brickResolution cubed");
  }
  const expectedBricks = source.sampleDimensions.map((value) => Math.ceil(value / source.brickResolution));
  if (source.brickDimensions.some((value, axis) => value !== expectedBricks[axis])) {
    throw new RangeError("Global fine brick dimensions do not index the complete logical sample lattice");
  }
  // Zero pages is a *published* state, not a malformed source. The compact
  // Sparse CM12 publication sizes its page list from accepted ownership, so a
  // scene with no accepted fluid — the first rung of the complexity ladder, or
  // any domain before its first seed — legitimately publishes none. Every ABI
  // guard in the shaders admits a page only when `id < pageCapacity`, so a zero
  // capacity reads as "no page resolves" rather than as an out-of-range read.
  // Refusing it here took the whole renderer down on an empty scene.
  if (!Number.isSafeInteger(source.pageCapacity) || source.pageCapacity < 0) {
    throw new RangeError("Global fine page capacity must be a non-negative integer");
  }
  positiveInteger(source.generation, "Global fine generation");
  if (!Number.isFinite(source.fineCellWidth) || source.fineCellWidth <= 0
    || source.domainOrigin.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Global fine physical coordinates must be finite with positive spacing");
  }
  // Compact sparse publications may place their finite physical page pool in
  // a larger signed address lattice. `domainOrigin` translates that lattice;
  // no dense logical-domain table or texture is implied by a non-zero value.
  if (Boolean(source.coarsePhiDirectory) !== Boolean(source.coarsePhiRowCapacity)) {
    throw new RangeError("Global fine compact-coarse directory and row capacity must be provided together");
  }
  if (source.coarsePhiDirectory && !source.topologyControl) {
    throw new RangeError("Global fine compact-coarse directory requires current-slot topology provenance");
  }
  if (source.coarsePhiRowCapacity !== undefined) {
    positiveInteger(source.coarsePhiRowCapacity, "Compact coarse phi row capacity");
  }
}

export function createGlobalFineLevelSetConsumerSource(source: WebGPUFineLevelSetBrickSource): GlobalFineLevelSetConsumerSource {
  const plan = source.plan;
  const consumer: GlobalFineLevelSetConsumerSource = { kind: "global-fine-levelset-sampling", metadata: { buffer: source.metadata },
    worklist: { buffer: source.worklist }, samples: { buffer: source.samples },
    ...(source.coarsePhiDirectory ? { coarsePhiDirectory: { buffer: source.coarsePhiDirectory } } : {}),
    ...(source.coarsePhiRowCapacity ? { coarsePhiRowCapacity: source.coarsePhiRowCapacity } : {}),
    ...(source.topologyControl ? { topologyControl: { buffer: source.topologyControl } } : {}),
    sampleDimensions: plan.sampleDimensions, brickDimensions: plan.brickDimensions,
    brickResolution: plan.brickResolution, samplesPerBrick: plan.samplesPerBrick,
    pageCapacity: plan.maximumResidentBricks, fineFactor: plan.fineFactor,
    fineCellWidth: plan.fineCellWidth, domainOrigin: plan.domainOrigin, generation: source.generation };
  validateGlobalFineLevelSetConsumerSource(consumer);
  return consumer;
}

export interface OctreeConsumerTrafficInputs {
  readonly finestCellCount: number;
  readonly velocityQueries: number;
  readonly phiQueries: number;
  readonly averageFaceCandidatesPerVelocityQuery: number;
  readonly legacyPublicationBytes?: number;
}

export interface OctreeConsumerTrafficPlan {
  readonly densePersistentBytes: number;
  readonly adaptivePersistentBytes: number;
  readonly persistentBytesAvoided: number;
  readonly denseFieldReadBytes: number;
  readonly adaptiveFieldReadBytesUpperBound: number;
  readonly estimatedFieldReadReduction: number;
}

/**
 * Conservative uncached traffic model. It intentionally charges every face
 * candidate as an incidence index plus a full 24-byte face record. Hardware
 * cache locality can improve the adaptive number; this plan never assumes it.
 */
export function planOctreeConsumerTraffic(input: OctreeConsumerTrafficInputs): OctreeConsumerTrafficPlan {
  for (const [name, value] of Object.entries(input)) {
    if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be finite and non-negative`);
  }
  if (!Number.isInteger(input.finestCellCount) || input.finestCellCount < 1) throw new RangeError("finestCellCount must be a positive integer");
  if (input.averageFaceCandidatesPerVelocityQuery > OCTREE_CONSUMER_MAX_FACE_CANDIDATES) {
    throw new RangeError("averageFaceCandidatesPerVelocityQuery exceeds the bounded incidence neighbourhood");
  }
  // One rgba32float velocity texture plus one r32float phi texture/publication.
  const densePersistentBytes = input.finestCellCount * 20 + (input.legacyPublicationBytes ?? 0);
  const adaptivePersistentBytes = 0; // adapters alias simulation-owned buffers
  // Dense trilinear sampling: eight texels. The fine/coarse surface lookup is
  // charged as one directory record plus eight fine values.
  const denseFieldReadBytes = input.velocityQueries * 8 * 16 + input.phiQueries * 8 * 4;
  const adaptiveFieldReadBytesUpperBound = input.velocityQueries
    * (8 + input.averageFaceCandidatesPerVelocityQuery * (4 + 32))
    + input.phiQueries * (16 + 8 * 4);
  return {
    densePersistentBytes,
    adaptivePersistentBytes,
    persistentBytesAvoided: densePersistentBytes,
    denseFieldReadBytes,
    adaptiveFieldReadBytesUpperBound,
    estimatedFieldReadReduction: denseFieldReadBytes / Math.max(1, adaptiveFieldReadBytesUpperBound),
  };
}

/**
 * WGSL library only: it declares no groups, bindings, globals, or entry point.
 * Velocity consumers gather their incidence neighbourhood into the fixed array
 * and can include this source unchanged.
 */
export const octreeConsumerSamplingWGSL = /* wgsl */ `
const OCTREE_CONSUMER_MAX_FACES = 48u;
struct OctreeConsumerFaceSample { originX:u32, originY:u32, originZ:u32, axisSpan:u32, normalVelocity:f32, pad:u32 }
fn octreeConsumerOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn octreeConsumerAxis(face:OctreeConsumerFaceSample)->u32{return face.axisSpan&3u;}
fn octreeConsumerSpan(face:OctreeConsumerFaceSample)->u32{return face.axisSpan>>2u;}
fn octreeConsumerCentre(face:OctreeConsumerFaceSample)->vec3f{let axis=octreeConsumerAxis(face);var p=vec3f(vec3u(face.originX,face.originY,face.originZ));let span=0.5*f32(octreeConsumerSpan(face));p[(axis+1u)%3u]+=span;p[(axis+2u)%3u]+=span;return p;}
fn octreeConsumerComponent(point:vec3f,axis:u32,candidates:array<OctreeConsumerFaceSample,48>,count:u32,defaultValue:f32)->f32{
  var weighted=0.0;var weights=0.0;var nearest=defaultValue;var nearestD2=3.402823e38;
  for(var i=0u;i<min(count,OCTREE_CONSUMER_MAX_FACES);i+=1u){let face=candidates[i];if(octreeConsumerAxis(face)!=axis){continue;}let delta=point-octreeConsumerCentre(face);let d2=dot(delta,delta);if(d2<nearestD2){nearestD2=d2;nearest=face.normalVelocity;}let support=max(1.0,f32(octreeConsumerSpan(face)));let tent=max(vec3f(0.0),vec3f(1.0)-abs(delta)/support);let weight=tent.x*tent.y*tent.z;weighted+=weight*face.normalVelocity;weights+=weight;}
  return select(nearest,weighted/weights,weights>0.0);
}
fn octreeConsumerVelocity(point:vec3f,candidates:array<OctreeConsumerFaceSample,48>,count:u32,defaultValue:vec3f)->vec3f{return vec3f(octreeConsumerComponent(point,0u,candidates,count,defaultValue.x),octreeConsumerComponent(point,1u,candidates,count,defaultValue.y),octreeConsumerComponent(point,2u,candidates,count,defaultValue.z));}
`;
