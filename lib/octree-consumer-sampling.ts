/**
 * Binding-neutral sampling contract for consumers of the unified octree.
 *
 * A caller resolves the leaf containing a query point and gathers the two
 * bounded incidence slabs that surround it. The routines below then perform
 * the same resolution-aware reconstruction for rendering, particles,
 * diagnostics, and face transport. No dense 3D compatibility texture is part
 * of this ABI.
 */

import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";
import { validateOctreeSurfacePageSource, type OctreeSurfacePageSource } from "./webgpu-octree-surface-pages";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";

export const OCTREE_CONSUMER_INVALID = 0xffff_ffff;
export const OCTREE_CONSUMER_MAX_FACE_CANDIDATES = 48;
export const OCTREE_CONSUMER_SURFACE_PAGE_RESOLUTION = 2;
export const OCTREE_CONSUMER_SURFACE_PAGE_SAMPLES = 8;

export type OctreeConsumerPoint = readonly [number, number, number];

export interface OctreeConsumerFaceSample {
  readonly origin: OctreeConsumerPoint;
  readonly axis: 0 | 1 | 2;
  readonly span: number;
  readonly normalVelocity: number;
}

export interface OctreeConsumerSurfaceLeaf {
  readonly origin: OctreeConsumerPoint;
  readonly size: number;
  /** Phi at the leaf centre followed by its finest-coordinate gradient. */
  readonly phiGradient: readonly [number, number, number, number];
  /** X-major 2³ or explicitly refined 4³ page, or undefined outside the resident band. */
  readonly phiPage?: ArrayLike<number>;
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
 * Coordinates are in finest-cell units. This is Shepard reconstruction with
 * a span-aware singularity clamp, matching the WGSL and U3 transport sampler.
 */
export function sampleOctreeFaceComponent(
  pointFine: OctreeConsumerPoint,
  axis: 0 | 1 | 2,
  candidates: readonly OctreeConsumerFaceSample[],
  fallback = 0,
): number {
  finitePoint(pointFine, "Octree velocity query");
  if (candidates.length > OCTREE_CONSUMER_MAX_FACE_CANDIDATES) {
    throw new RangeError(`Octree velocity query exceeds the ${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}-face 2:1 bound`);
  }
  let weighted = 0;
  let weights = 0;
  let nearest = fallback;
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
    const weight = 1 / Math.max(0.0625 * support * support, distanceSquared);
    weighted += weight * face.normalVelocity;
    weights += weight;
  }
  return weights > 0 ? weighted / weights : nearest;
}

export function sampleOctreeFaceVelocity(
  pointFine: OctreeConsumerPoint,
  candidates: readonly OctreeConsumerFaceSample[],
  fallback: OctreeConsumerPoint = [0, 0, 0],
): [number, number, number] {
  return [0, 1, 2].map((axis) => sampleOctreeFaceComponent(
    pointFine, axis as 0 | 1 | 2, candidates, component(fallback, axis),
  )) as [number, number, number];
}

function fallbackPhi(pointFine: OctreeConsumerPoint, leaf: OctreeConsumerSurfaceLeaf): number {
  const centre = leaf.origin.map((value) => value + leaf.size * 0.5) as [number, number, number];
  return leaf.phiGradient[0]
    + leaf.phiGradient[1] * (pointFine[0] - centre[0])
    + leaf.phiGradient[2] * (pointFine[1] - centre[1])
    + leaf.phiGradient[3] * (pointFine[2] - centre[2]);
}

function mix(a: number, b: number, t: number): number { return a + (b - a) * t; }

/** Trilinear 2- or 4-cubed page lookup with an affine bulk fallback. */
export function sampleOctreeSurfacePhi(
  pointFine: OctreeConsumerPoint,
  leaf: OctreeConsumerSurfaceLeaf,
): number {
  finitePoint(pointFine, "Octree phi query");
  finitePoint(leaf.origin, "Octree surface leaf origin");
  if (!Number.isFinite(leaf.size) || leaf.size <= 0) throw new RangeError("Octree surface leaf size must be positive");
  if (leaf.phiGradient.some((value) => !Number.isFinite(value))) throw new RangeError("Octree surface leaf phi plane must be finite");
  const page = leaf.phiPage;
  if (page === undefined) return fallbackPhi(pointFine, leaf);
  const resolution = page.length === 8 ? 2 : page.length === 64 ? 4 : 0;
  if (resolution === 0) throw new RangeError("Octree phi page must contain 8 or 64 samples");
  const grid = pointFine.map((value, axis) => Math.max(0, Math.min(resolution - 1,
    (value - leaf.origin[axis]) / leaf.size * resolution - 0.5,
  ))) as [number, number, number];
  const a = grid.map(Math.floor) as [number, number, number];
  const b = a.map((value) => Math.min(resolution - 1, value + 1)) as [number, number, number];
  const t = grid.map((value, axis) => value - a[axis]) as [number, number, number];
  const at = (x: number, y: number, z: number) => Number(page[x + resolution * (y + resolution * z)]);
  const z0 = mix(mix(at(a[0], a[1], a[2]), at(b[0], a[1], a[2]), t[0]),
    mix(at(a[0], b[1], a[2]), at(b[0], b[1], a[2]), t[0]), t[1]);
  const z1 = mix(mix(at(a[0], a[1], b[2]), at(b[0], a[1], b[2]), t[0]),
    mix(at(a[0], b[1], b[2]), at(b[0], b[1], b[2]), t[0]), t[1]);
  return mix(z0, z1, t[2]);
}

export interface UnifiedOctreeConsumerSource {
  readonly kind: "unified-octree-sampling";
  readonly leaves: GPUBufferBinding;
  readonly faces: GPUBufferBinding;
  readonly incidence: GPUBufferBinding;
  readonly faceControl: GPUBufferBinding;
  readonly surfaceArena: GPUBufferBinding;
  readonly surfaceParams: GPUBufferBinding;
  readonly surfaceDispatch: { readonly buffer: GPUBuffer; readonly offsetBytes: number };
  readonly faceCapacity: number;
  readonly leafCapacity: number;
  readonly pageCapacity: number;
  readonly pageResolution: 2 | 4;
  readonly generation: number;
}

export interface GlobalFineLevelSetConsumerSource {
  readonly kind: "global-fine-levelset-sampling";
  readonly hash: GPUBufferBinding;
  readonly metadata: GPUBufferBinding;
  readonly worklist: GPUBufferBinding;
  readonly flags: GPUBufferBinding;
  readonly phi: GPUBufferBinding;
  readonly coarsePhiDirectory?: GPUBufferBinding;
  readonly coarsePhiHashCapacity?: number;
  /** GPU transaction that published the selected A/B fine slot. */
  readonly topologyControl?: GPUBufferBinding;
  readonly sampleDimensions: readonly [number, number, number];
  readonly brickDimensions: readonly [number, number, number];
  readonly brickResolution: 4 | 8;
  readonly samplesPerBrick: number;
  readonly hashCapacity: number;
  readonly maximumHashProbes: number;
  readonly pageCapacity: number;
  readonly fineFactor: 4 | 8;
  readonly fineCellWidth: number;
  readonly domainOrigin: readonly [number, number, number];
  readonly generation: number;
}

/** CPU mirror of the renderer's fail-closed current-publication epoch gate.
 * A rejected transaction may preserve the last physical field by retagging it,
 * but it is not a new render publication and must retain the prior mesh. */
export function globalFineCoarseGenerationPairIsValid(
  fineGeneration: number,
  coarseGeneration: number,
  topologyControl: ArrayLike<number> | undefined,
): boolean {
  if (!topologyControl || topologyControl.length < 8) return false;
  const mask = 0x3fff_ffff;
  const fine = fineGeneration & mask, coarse = coarseGeneration & mask;
  return coarse === fine && topologyControl[0] === 0 && topologyControl[4] === 1
    && topologyControl[5] === 0 && topologyControl[7] === 0;
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
  if (source.brickResolution !== 4 && source.brickResolution !== 8) {
    throw new RangeError("Global fine brick resolution must be 4 or 8");
  }
  if (source.fineFactor !== 4 && source.fineFactor !== 8) {
    throw new RangeError("Global fine factor must be 4 or 8");
  }
  if (source.samplesPerBrick !== source.brickResolution ** 3) {
    throw new RangeError("Global fine sample stride must equal brickResolution cubed");
  }
  const expectedBricks = source.sampleDimensions.map((value) => Math.ceil(value / source.brickResolution));
  if (source.brickDimensions.some((value, axis) => value !== expectedBricks[axis])) {
    throw new RangeError("Global fine brick dimensions do not index the complete logical sample lattice");
  }
  positiveInteger(source.hashCapacity, "Global fine hash capacity");
  if ((source.hashCapacity & (source.hashCapacity - 1)) !== 0) {
    throw new RangeError("Global fine hash capacity must be a power of two");
  }
  positiveInteger(source.maximumHashProbes, "Global fine maximum hash probes");
  positiveInteger(source.pageCapacity, "Global fine page capacity");
  positiveInteger(source.generation, "Global fine generation");
  if (!Number.isFinite(source.fineCellWidth) || source.fineCellWidth <= 0
    || source.domainOrigin.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Global fine physical coordinates must be finite with positive spacing");
  }
  // The current compact coarse directory and tetra emitter both use the
  // fluid-domain-local frame. Do not pretend that an untranslated directory
  // can safely serve a fine lattice whose configured origin is elsewhere.
  if (source.domainOrigin.some((value) => value !== 0)) {
    throw new RangeError("Global fine renderer currently requires a zero domain origin");
  }
  if (Boolean(source.coarsePhiDirectory) !== Boolean(source.coarsePhiHashCapacity)) {
    throw new RangeError("Global fine compact-coarse directory and hash capacity must be provided together");
  }
  if (source.coarsePhiDirectory && !source.topologyControl) {
    throw new RangeError("Global fine compact-coarse directory requires current-slot topology provenance");
  }
  if (source.coarsePhiHashCapacity !== undefined) {
    positiveInteger(source.coarsePhiHashCapacity, "Compact coarse phi hash capacity");
    if ((source.coarsePhiHashCapacity & (source.coarsePhiHashCapacity - 1)) !== 0) {
      throw new RangeError("Compact coarse phi hash capacity must be a power of two");
    }
  }
}

export function createGlobalFineLevelSetConsumerSource(source: WebGPUFineLevelSetBrickSource): GlobalFineLevelSetConsumerSource {
  const plan = source.plan;
  const consumer: GlobalFineLevelSetConsumerSource = { kind: "global-fine-levelset-sampling", hash: { buffer: source.hash }, metadata: { buffer: source.metadata },
    worklist: { buffer: source.worklist }, flags: { buffer: source.flags }, phi: { buffer: source.phi },
    ...(source.coarsePhiDirectory ? { coarsePhiDirectory: { buffer: source.coarsePhiDirectory } } : {}),
    ...(source.coarsePhiHashCapacity ? { coarsePhiHashCapacity: source.coarsePhiHashCapacity } : {}),
    ...(source.topologyControl ? { topologyControl: { buffer: source.topologyControl } } : {}),
    sampleDimensions: plan.sampleDimensions, brickDimensions: plan.brickDimensions,
    brickResolution: plan.brickResolution, samplesPerBrick: plan.samplesPerBrick,
    hashCapacity: plan.hashCapacity, maximumHashProbes: plan.maximumHashProbes,
    pageCapacity: plan.maximumResidentBricks, fineFactor: plan.fineFactor,
    fineCellWidth: plan.fineCellWidth, domainOrigin: plan.domainOrigin, generation: source.generation };
  validateGlobalFineLevelSetConsumerSource(consumer);
  return consumer;
}

/** Adapts the U2/U3 face arena and U4 surface pages without copying either. */
export function createUnifiedOctreeConsumerSource(
  face: OctreeFaceMirrorSource,
  surface: OctreeSurfacePageSource,
  generation = 0,
): UnifiedOctreeConsumerSource {
  validateOctreeSurfacePageSource(surface);
  if (face.plan.rowCapacity !== surface.plan.leafCapacity) {
    throw new RangeError("Unified octree face and surface leaf capacities must match");
  }
  if (!Number.isSafeInteger(generation) || generation < 0) throw new RangeError("Unified octree generation must be non-negative");
  return {
    kind: "unified-octree-sampling",
    leaves: surface.leaves,
    faces: { buffer: face.faces },
    incidence: { buffer: face.incidence },
    faceControl: { buffer: face.control },
    surfaceArena: surface.arena,
    surfaceParams: surface.params,
    surfaceDispatch: { buffer: surface.activePages.indirectBuffer, offsetBytes: surface.activePages.indirectOffsetBytes },
    faceCapacity: face.plan.faceCapacity,
    leafCapacity: face.plan.rowCapacity,
    pageCapacity: surface.plan.pageCapacity,
    pageResolution: surface.plan.pageResolution,
    generation,
  };
}

export function validateUnifiedOctreeConsumerSource(source: UnifiedOctreeConsumerSource): void {
  if (source.pageResolution !== 2 && source.pageResolution !== 4) {
    throw new RangeError("Unified octree consumer page resolution must be 2 or 4");
  }
  if (!Number.isSafeInteger(source.leafCapacity) || source.leafCapacity < 1
    || !Number.isSafeInteger(source.pageCapacity) || source.pageCapacity < 1
    || source.pageCapacity > source.leafCapacity) {
    throw new RangeError("Unified octree consumer capacities are invalid");
  }
}

export type UnifiedOctreeConsumerRole = "renderer" | "particles" | "diagnostics";

export interface UnifiedOctreeConsumerAdapter {
  readonly representation: "direct-adaptive-octree";
  readonly role: UnifiedOctreeConsumerRole;
  readonly source: UnifiedOctreeConsumerSource;
  /** The consumer resolves an owner row, then gathers at most two incidence slabs. */
  readonly query: "owner-row-plus-bounded-incidence";
  /** Diagnostics materialization is transient; renderer and particles never materialize. */
  readonly materialization: "none" | "transient-output-only";
}

/** Creates zero-allocation views over one source for the three remaining consumers. */
export function createUnifiedOctreeConsumerAdapters(source: UnifiedOctreeConsumerSource): Readonly<{
  renderer: UnifiedOctreeConsumerAdapter;
  particles: UnifiedOctreeConsumerAdapter;
  diagnostics: UnifiedOctreeConsumerAdapter;
}> {
  const adapter = (role: UnifiedOctreeConsumerRole): UnifiedOctreeConsumerAdapter => ({
    representation: "direct-adaptive-octree",
    role,
    source,
    query: "owner-row-plus-bounded-incidence",
    materialization: role === "diagnostics" ? "transient-output-only" : "none",
  });
  return { renderer: adapter("renderer"), particles: adapter("particles"), diagnostics: adapter("diagnostics") };
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
  // Dense trilinear sampling: eight texels. Adaptive phi: leaf record/table + eight page values.
  const denseFieldReadBytes = input.velocityQueries * 8 * 16 + input.phiQueries * 8 * 4;
  const adaptiveFieldReadBytesUpperBound = input.velocityQueries
    * (8 + input.averageFaceCandidatesPerVelocityQuery * (4 + 32))
    + input.phiQueries * (4 + 48 + 8 * 4);
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
 * Consumers gather their incidence neighbourhood into the fixed array and can
 * include this source unchanged in render, particle, or diagnostic shaders.
 */
export const octreeConsumerSamplingWGSL = /* wgsl */ `
const OCTREE_CONSUMER_MAX_FACES = 48u;
struct OctreeConsumerFaceSample { originX:u32, originY:u32, originZ:u32, axisSpan:u32, normalVelocity:f32, pad:u32 }
struct OctreeConsumerSurfaceLeaf { originX:u32, originY:u32, originZ:u32, size:u32, phiGradient:vec4f }
fn octreeConsumerOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn octreeConsumerAxis(face:OctreeConsumerFaceSample)->u32{return face.axisSpan&3u;}
fn octreeConsumerSpan(face:OctreeConsumerFaceSample)->u32{return face.axisSpan>>2u;}
fn octreeConsumerCentre(face:OctreeConsumerFaceSample)->vec3f{let axis=octreeConsumerAxis(face);var p=vec3f(vec3u(face.originX,face.originY,face.originZ));let span=0.5*f32(octreeConsumerSpan(face));p[(axis+1u)%3u]+=span;p[(axis+2u)%3u]+=span;return p;}
fn octreeConsumerComponent(point:vec3f,axis:u32,candidates:array<OctreeConsumerFaceSample,48>,count:u32,fallback:f32)->f32{
  var weighted=0.0;var weights=0.0;var nearest=fallback;var nearestD2=3.402823e38;
  for(var i=0u;i<min(count,OCTREE_CONSUMER_MAX_FACES);i+=1u){let face=candidates[i];if(octreeConsumerAxis(face)!=axis){continue;}let delta=point-octreeConsumerCentre(face);let d2=dot(delta,delta);if(d2<nearestD2){nearestD2=d2;nearest=face.normalVelocity;}let support=max(1.0,f32(octreeConsumerSpan(face)));let weight=1.0/max(0.0625*support*support,d2);weighted+=weight*face.normalVelocity;weights+=weight;}
  return select(nearest,weighted/weights,weights>0.0);
}
fn octreeConsumerVelocity(point:vec3f,candidates:array<OctreeConsumerFaceSample,48>,count:u32,fallback:vec3f)->vec3f{return vec3f(octreeConsumerComponent(point,0u,candidates,count,fallback.x),octreeConsumerComponent(point,1u,candidates,count,fallback.y),octreeConsumerComponent(point,2u,candidates,count,fallback.z));}
fn octreeConsumerFallbackPhi(point:vec3f,leaf:OctreeConsumerSurfaceLeaf)->f32{let centre=vec3f(vec3u(leaf.originX,leaf.originY,leaf.originZ))+vec3f(0.5*f32(leaf.size));return leaf.phiGradient.x+dot(leaf.phiGradient.yzw,point-centre);}
fn octreeConsumerPageIndex(q:vec3u,resolution:u32)->u32{return q.x+resolution*(q.y+resolution*q.z);}
// The including shader supplies octreeConsumerPageLoad(base,index), normally
// as one storage-buffer read. This keeps the shared ABI genuinely 2^3/4^3
// dynamic instead of smuggling every page through a fixed 64-value parameter.
fn octreeConsumerPhi(point:vec3f,leaf:OctreeConsumerSurfaceLeaf,pageBase:u32,pageResolution:u32,hasPage:bool)->f32{
  if(!hasPage||(pageResolution!=2u&&pageResolution!=4u)){return octreeConsumerFallbackPhi(point,leaf);}let r=pageResolution;let origin=vec3f(vec3u(leaf.originX,leaf.originY,leaf.originZ));let grid=clamp((point-origin)/f32(leaf.size)*f32(r)-vec3f(0.5),vec3f(0.0),vec3f(f32(r-1u)));let a=vec3u(floor(grid));let b=min(a+vec3u(1u),vec3u(r-1u));let t=fract(grid);
  let c000=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(a,r));let c100=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(b.x,a.y,a.z),r));let c010=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(a.x,b.y,a.z),r));let c110=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(b.x,b.y,a.z),r));let c001=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(a.x,a.y,b.z),r));let c101=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(b.x,a.y,b.z),r));let c011=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(vec3u(a.x,b.y,b.z),r));let c111=octreeConsumerPageLoad(pageBase,octreeConsumerPageIndex(b,r));return mix(mix(mix(c000,c100,t.x),mix(c010,c110,t.x),t.y),mix(mix(c001,c101,t.x),mix(c011,c111,t.x),t.y),t.z);
}
`;
