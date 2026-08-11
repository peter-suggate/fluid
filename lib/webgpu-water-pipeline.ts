import { environmentShaderLibrary } from "./webgpu-environments";
import { advancePresentationClock, frameInterval_ms } from "./frame-pacing";
import type { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import {
  GLASS_OPTICS,
  packWaterSceneOptics,
  resolveDisplayGrade,
  resolveWaterKeyLight,
  resolveWaterOptics,
  unifiedDisplayTransferShaderLibrary,
  unifiedLightingShaderLibrary,
  WATER_OPTICS,
  WATER_SCENE_OPTICS_BYTES,
  WATER_SCENE_OPTICS_FLOATS,
  WATER_SCENE_OPTICS_RECEIVER_FLOAT_OFFSET,
  waterSceneOpticsShaderLibrary,
  type WaterOpticsAuthoring,
  type DisplayGradeAuthoring,
} from "./webgpu-lighting";
import { terrainContentStamp, terrainHeightAt, type TerrainDescription } from "./terrain";
import { cameraApertureShaderLibrary } from "./webgpu-camera";
import {
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "./webgpu-octree-power-coarse-levelset";
import {
  validateCoarseLevelSetConsumerSource,
  validateGlobalFineLevelSetConsumerSource,
  type CoarseLevelSetConsumerSource,
  type GlobalFineLevelSetConsumerSource,
} from "./octree-consumer-sampling";
import { globalFineClassifiedEmitShader, globalFineClassifiedIndirectScanShader } from "./webgpu-water-global-fine-tetra";
import { globalFineSurfaceClassificationShader } from "./webgpu-water-global-fine-classify";
import type { GPUTimestampPhase } from "./performance-trace";
import type { FrameBandPartitioner } from "./webgpu-frame-band-sampler";
import {
  disabledRenderStagesEqual,
  NO_DISABLED_RENDER_STAGES,
  type DisabledRenderStages,
} from "./render-stage-switches";

/**
 * Rasterized water presentation for the WebGPU renderer.
 *
 * The solver already owns the liquid volume.  This pipeline keeps that data on
 * the GPU and turns its 0.5 isosurface into triangles every frame.  The result
 * is rendered as a nearest front/back interval plus one depth-peeled interval,
 * which preserves transmission through folded sheets without scanning the
 * volume once per screen pixel.
 */

export function shouldUpdateWaterSurface(extractedRevision: number, latestRevision: number, lastExtractionAt_ms: number, now_ms: number) {
  return extractedRevision < 0
    || (latestRevision !== extractedRevision && now_ms - lastExtractionAt_ms + 0.5 >= frameInterval_ms());
}

/** Serializable caustic-receiver cache key shared with regression tests. */
export function causticReceiverContentKey(
  terrain: TerrainDescription | undefined,
  width: number,
  depth: number,
  contentStamp = terrainContentStamp(terrain),
): string {
  return terrain && width > 0 && depth > 0 ? `${width}x${depth}:${contentStamp}` : "";
}

/** Raster/body depth separation that activates the local implicit resolver. */
export const CONTACT_RESOLVE_BAND_CELLS = 1.5;

/**
 * Resolve one filtered sample from the additive caustic target.
 *
 * RGB is deposited energy and alpha is deposited coverage, so filtering across
 * a covered/uncovered edge produces premultiplied RGB.  Restore the uncovered
 * share to the neutral modulation (one) without normalizing genuine overlaps:
 * alpha above one means multiple bundles really did land on the same texel.
 * This CPU mirror keeps that distinction explicit for regression tests.
 */
export function resolvePremultipliedCausticSample(
  sampled: readonly [number, number, number, number],
  strength = 1,
): readonly [number, number, number] {
  const alpha = Math.max(0, sampled[3]);
  if (alpha <= 1e-4) return [1, 1, 1];
  const coverage = Math.min(alpha, 1);
  const resolvedStrength = Math.min(1, Math.max(0, strength));
  const resolveChannel = (channel: number) => {
    const deposited = Math.max(channel, 0) / Math.max(coverage, 1e-4);
    const modulation = 1 + (deposited - 1) * coverage;
    return 1 + (modulation - 1) * resolvedStrength;
  };
  return [resolveChannel(sampled[0]), resolveChannel(sampled[1]), resolveChannel(sampled[2])];
}

/** Shared disabled storage must satisfy the compact coarse-directory ABI. */
export const WATER_DISABLED_STORAGE_BYTES = Math.max(
  64,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES
    + OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
);

/** CPU mirror of the shader gate, kept explicit for regression tests. */
export function shouldResolveRigidContact(frontDepth: number, rigidDepth: number, cellSize: number, bodyCount: number) {
  return bodyCount > 0
    && Number.isFinite(frontDepth)
    && Number.isFinite(rigidDepth)
    && rigidDepth < 1e19
    && Math.abs(rigidDepth - frontDepth) <= CONTACT_RESOLVE_BAND_CELLS * Math.max(cellSize, 0);
}

export interface SurfaceExtractionDispatchPlan {
  mode: "full-volume" | "restricted-band";
  full?: [number, number, number];
  band?: [number, number, number];
  tallSides?: [number, number, number];
  walls?: [number, number, number];
  bandCubeRows?: number;
}

export interface RasterWaterEncodeResult {
  surfaceUpdated: boolean;
  /** True only when this extraction copied a fresh, source-matched diagnostic receipt. */
  surfaceDiagnosticsCaptured: boolean;
}

export interface WaterSurfacePresentationDiagnostics {
  /** Presentation geometry only; this does not confer simulation authority on a fallback field. */
  readonly surfaceGeometrySource: WaterSurfaceGeometrySource;
  readonly globalFineAttached: boolean;
  /** Generation of the global-fine source captured with this queue-fenced diagnostic. */
  readonly globalFineAttachedGeneration?: number;
  /** GPU-written generation whose zero crossing produced the retained raster mesh, when that mesh is global-fine. */
  readonly meshPublicationGeneration?: number;
  readonly globalFineCrossingPublished: boolean;
  readonly presentationFallbackActive: boolean;
  /** Per-pipeline frames rendered after a source receipt was available. */
  readonly sourceFrameCounts?: Readonly<Record<WaterSurfaceGeometrySource, number>>;
}

export interface WaterRenderDiagnostics extends WaterSurfacePresentationDiagnostics {
  readonly vertexCount: number;
  readonly activeCubeCount: number;
  readonly vertexAllocator: number;
  readonly globalFineAuthorityLatch: number;
}

export type WaterSurfaceGeometrySource =
  | "global-fine-coarse"
  | "compact-coarse"
  | "retained-previous"
  | "empty"
  | "volume";

/**
 * Decodes the renderer-private transaction words. `authorityLatch` and the
 * GPU-written mesh publication generation trail the four WebGPU indirect-draw
 * arguments, so the required draw `firstInstance` remains zero on devices
 * without the optional indirect-first-instance feature. The latch is
 * presentation evidence only; it never makes a presentation field
 * authoritative for simulation.
 */
export function waterSurfaceGeometrySource(
  globalFineAttached: boolean,
  vertexCount: number,
  authorityLatch: number,
  coarseAttached = false,
): WaterSurfaceGeometrySource {
  if (!globalFineAttached && !coarseAttached) return vertexCount > 0 ? "volume" : "empty";
  if (authorityLatch !== 0) return coarseAttached && !globalFineAttached
    ? "compact-coarse" : "global-fine-coarse";
  if (coarseAttached) return vertexCount > 0 ? "retained-previous" : "empty";
  return vertexCount > 0 ? "retained-previous" : "empty";
}

/**
 * Encodes a complete replacement for the analytic dry-scene pass.
 *
 * A successful replacement may resolve into a different sampled texture (for
 * example temporal ping-pong history). The water composite consumes that view
 * directly, avoiding a full-frame alias-breaking copy back into `target`.
 */
export interface DrySceneReplacementResult {
  readonly encoded: true;
  readonly sampledTargetView: GPUTextureView;
}

export type RenderPathTracePhase = (phase: GPUTimestampPhase) => void;

export type DrySceneReplacementEncoder = (
  encoder: GPUCommandEncoder,
  target: GPUTexture | GPUTextureView,
  tracePhase?: RenderPathTracePhase,
) => DrySceneReplacementResult | false;

/** What to put behind raster water when no dry-scene encoder is requested. */
export type RasterWaterBackgroundMode = "require-dry-scene" | "clear";

/**
 * Restricted tall cells cannot contain a free surface below their cubic band.
 * The interior can therefore follow that band, while a separate perimeter
 * dispatch retains the full-height tank-side interfaces needed by the optical
 * composite. Two adjacent base steps can meet across a cube diagonal.
 */
export function surfaceExtractionDispatchPlan(
  nx: number,
  ny: number,
  nz: number,
  packedNy: number,
  restrictedTallCell: boolean,
  maximumNeighborDelta: number
): SurfaceExtractionDispatchPlan {
  if (!restrictedTallCell) {
    return { mode: "full-volume", full: [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)] };
  }
  const bandCubeRows = Math.min(ny + 1, Math.max(1, packedNy + 2 * Math.ceil(Math.max(0, maximumNeighborDelta)) - 1));
  const perimeterCubes = 2 * (nx + 1) + 2 * Math.max(0, nz - 1);
  return {
    mode: "restricted-band",
    band: [Math.ceil(Math.max(0, nx - 1) / 4), Math.ceil(bandCubeRows / 4), Math.ceil(Math.max(0, nz - 1) / 4)],
    tallSides: [Math.ceil(Math.max(0, nx - 1) / 8), Math.ceil(Math.max(0, nz - 1) / 8), 1],
    walls: [Math.ceil(perimeterCubes * (ny + 1) / 64), 1, 1],
    bandCubeRows
  };
}

/**
 * Extraction is split into two GPU stages so the full-lattice sweep stays
 * lean. Classification kernels only load a cube's eight corners and append
 * surface-crossing cubes to a worklist; the triangle-emitting polygonise
 * kernel then runs over just those cubes via an indirect dispatch. Keeping
 * the heavy emission code out of the sweep kernels preserves their occupancy,
 * which is what hides the latency of the classification texture loads.
 */
export const EXTRACTION_POLYGONISE_WORKGROUP = 64;

/** Vertex capacity from grid surface area (32 bytes per vertex, 64 MiB cap).
 *
 * Marching tetrahedra emits more vertices per crossing cube than marching
 * cubes, and a ratio-two transported surface can retain transient sheets whose
 * area is several times the tank footprint. 80 vertices per lattice-area unit
 * covers the mini dam-break's folded front through its two-second acceptance
 * window while the byte ceiling remains absolute.
 */
export function surfaceVertexCapacity(nx: number, ny: number, nz: number) {
  const area = nx * ny + nx * nz + ny * nz;
  return Math.max(262_144, Math.min(2_097_152, area * 80));
}

/**
 * A surface-crossing cube always emits at least one triangle (three
 * vertices), so a worklist of capacity/3 entries can only clip on fields
 * that would clip the vertex buffer as well.
 */
export function activeCubeCapacity(maxVertices: number) {
  return Math.ceil(maxVertices / 3);
}

/** Bounded two-dimensional dispatch over every physical fine-brick sample. */
export function globalFineSurfaceDispatch(pageCapacity: number, samplesPerBrick: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 1
    || !Number.isSafeInteger(samplesPerBrick) || samplesPerBrick < 1) {
    throw new RangeError("Global fine extraction capacities must be positive integers");
  }
  const groups = Math.ceil(pageCapacity * samplesPerBrick / 256);
  const x = Math.min(65_535, groups);
  const y = Math.ceil(groups / 65_535);
  if (y > 65_535) throw new RangeError("Global fine extraction exceeds the WebGPU dispatch limit");
  return [x, y, 1] as const;
}

/** One cooperative workgroup per compact coarse-phi row. */
export function globalFineCoarseSurfaceDispatch(rowCapacity: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Global coarse extraction capacity must be a positive integer");
  }
  const x = Math.min(65_535, rowCapacity);
  const y = Math.ceil(rowCapacity / 65_535);
  if (y > 65_535) throw new RangeError("Global coarse extraction exceeds the WebGPU dispatch limit");
  return [x, y, 1] as const;
}

/** One invocation per factor-1 compact-coarse lattice cube. The dense
 * complement is authoritative in this mode, so scanning the lattice gives
 * dry-side and wet-side faces the same unique lower-anchor owner. */
export function compactCoarseSurfaceDispatch(
  sampleDimensions: readonly [number, number, number],
): readonly [number, number, number] {
  if (sampleDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Compact coarse extraction dimensions must be positive integers");
  }
  const cubes = sampleDimensions.reduce((product, value) => product * (value + 1), 1);
  const groups = Math.ceil(cubes / 256);
  const x = Math.min(65_535, groups);
  const y = Math.ceil(groups / 65_535);
  if (y > 65_535) throw new RangeError("Compact coarse extraction exceeds the WebGPU dispatch limit");
  return [x, y, 1] as const;
}

export const surfaceExtractionShader = /* wgsl */ `
struct Uniforms {
  viewport: vec4f,
  cameraPosition: vec4f,
  cameraTarget: vec4f,
  container: vec4f,
  options: vec4f,
  gridInfo: vec4f,
  debug: vec4f,
}
struct SurfaceVertex { position: vec4f, normal: vec4f }
struct IndirectArgs {
  vertexCount: atomic<u32>,
  instanceCount: u32,
  firstVertex: u32,
  firstInstance: u32,
  activeCubeCount: atomic<u32>,
  vertexAllocator: atomic<u32>,
  globalFineAuthorityLatch: atomic<u32>,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var columnBases: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<SurfaceVertex>;
@group(0) @binding(4) var<storage, read_write> drawArgs: IndirectArgs;
@group(0) @binding(5) var<storage, read_write> activeCubes: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> globalCubeValues: array<vec4f>;
struct SparseParams {
  coarseDims: vec4u,
  fineDims: vec4u,
  brickDims: vec4u,
  settings: vec4f,
  cellAndDt: vec4f,
  sizing: vec4f,
  physical: vec4f,
}
@group(0) @binding(7) var<storage, read> sparsePageTable: array<u32>;
@group(0) @binding(8) var<storage, read> sparseActivePages: array<u32>;
@group(0) @binding(9) var<storage, read> sparsePhi: array<f32>;
@group(0) @binding(10) var<uniform> sparseParams: SparseParams;
@group(0) @binding(11) var<storage, read> sparseControl: array<u32>;
@group(0) @binding(12) var<storage, read> sparseStates: array<u32>;
override countOnly = false;
override sparseField = false;

const SPARSE_INVALID: u32 = 0xffffffffu;
const SPARSE_CORE: u32 = 2u;

fn sparseOverflow() -> bool {
  return arrayLength(&sparseControl) > 2u && sparseControl[2] != 0u;
}

fn sparseFineDimensions() -> vec3u { return sparseParams.fineDims.xyz; }
fn sparsePayloadIndex(q: vec3u) -> u32 {
  if (any(q >= sparseParams.fineDims.xyz)) { return SPARSE_INVALID; }
  let brickSize = sparseParams.fineDims.w;
  let page = q / brickSize;
  let pageIndex = page.x + sparseParams.brickDims.x * (page.y + sparseParams.brickDims.y * page.z);
  if (pageIndex >= arrayLength(&sparsePageTable)) { return SPARSE_INVALID; }
  let slot = sparsePageTable[pageIndex];
  if (slot == SPARSE_INVALID || slot >= u32(sparseParams.sizing.w)) { return SPARSE_INVALID; }
  let local = q % brickSize;
  let localIndex = local.x + brickSize * (local.y + brickSize * local.z);
  return slot * brickSize * brickSize * brickSize + localIndex;
}
fn sparseCorePageAt(q: vec3u) -> bool {
  if (any(q >= sparseParams.fineDims.xyz)) { return false; }
  let page = q / sparseParams.fineDims.w;
  let pageIndex = page.x + sparseParams.brickDims.x * (page.y + sparseParams.brickDims.y * page.z);
  return pageIndex < arrayLength(&sparseStates)
    && (sparseStates[pageIndex] & SPARSE_CORE) != 0u
    && sparsePayloadIndex(q) != SPARSE_INVALID;
}
fn coarsePhiAtFine(position: vec3f) -> f32 {
  let factor = f32(sparseParams.coarseDims.w);
  let p = clamp((position + vec3f(0.5)) / factor - vec3f(0.5), vec3f(0.0), vec3f(sparseParams.coarseDims.xyz - vec3u(1)));
  let a = vec3i(floor(p)); let b = min(a + vec3i(1), vec3i(sparseParams.coarseDims.xyz) - vec3i(1)); let t = fract(p);
  let p000=textureLoad(volume,vec3i(a.x,a.y,a.z),0).x;let p100=textureLoad(volume,vec3i(b.x,a.y,a.z),0).x;
  let p010=textureLoad(volume,vec3i(a.x,b.y,a.z),0).x;let p110=textureLoad(volume,vec3i(b.x,b.y,a.z),0).x;
  let p001=textureLoad(volume,vec3i(a.x,a.y,b.z),0).x;let p101=textureLoad(volume,vec3i(b.x,a.y,b.z),0).x;
  let p011=textureLoad(volume,vec3i(a.x,b.y,b.z),0).x;let p111=textureLoad(volume,vec3i(b.x,b.y,b.z),0).x;
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);
}
fn sparsePhiAt(cell: vec3i) -> f32 {
  if (any(cell < vec3i(0)) || any(cell >= vec3i(sparseParams.fineDims.xyz))) { return coarsePhiAtFine(vec3f(cell)); }
  let payload = sparsePayloadIndex(vec3u(cell));
  if (payload == SPARSE_INVALID || payload >= arrayLength(&sparsePhi)) { return coarsePhiAtFine(vec3f(cell)); }
  return sparsePhi[payload];
}
// Level-set fields become a smooth occupancy whose 0.5 contour is phi = 0.
// The band spans four cells so no corner of a surface-crossing cube saturates
// (the cube diagonal is under two cells); a saturated corner biases the linear
// crossing estimate and extracts as cell-pitch lattice artifacts.
fn occupancyFromPhi(phi: f32) -> f32 {
  let samplesY = select(u.gridInfo.y, f32(sparseParams.fineDims.y), sparseField);
  let band = 4.0 * u.container.y / max(samplesY, 1.0);
  return clamp(0.5 - phi / band, 0.0, 1.0);
}

fn fieldCell(cell: vec3i) -> f32 {
  let dims = vec3i(u.gridInfo.xyz);
  if (any(cell < vec3i(0)) || any(cell >= dims)) { return 0.0; }
  let mode = u.gridInfo.w;
  if (mode < 1.5) { return textureLoad(volume, cell, 0).x; }
  if (mode > 2.5) { return occupancyFromPhi(textureLoad(volume, cell, 0).x); }
  let base = i32(round(textureLoad(columnBases, cell.xz, 0).x));
  if (cell.y < base && base > 0) {
    let t = clamp(f32(cell.y) / f32(max(base - 1, 1)), 0.0, 1.0);
    return occupancyFromPhi(mix(textureLoad(volume, vec3i(cell.x, 0, cell.z), 0).x, textureLoad(volume, vec3i(cell.x, 1, cell.z), 0).x, t));
  }
  let packedY = 2 + cell.y - base;
  let stored = vec3i(textureDimensions(volume));
  if (packedY < 2 || packedY >= stored.y) { return 0.0; }
  return occupancyFromPhi(textureLoad(volume, vec3i(cell.x, packedY, cell.z), 0).x);
}

fn columnBaseAt(x: i32, z: i32) -> i32 {
  return i32(round(textureLoad(columnBases, vec2i(x, z), 0).x));
}

// The virtual lattice has one zero-valued layer on every tank boundary.  It
// closes the liquid mesh at glass/floor contacts, so a camera ray always has a
// usable exit interface as well as a free-surface entry interface.
fn latticeValue(p: vec3i) -> f32 {
  let dims = select(vec3i(u.gridInfo.xyz), vec3i(sparseParams.fineDims.xyz), sparseField);
  // Side/top boundaries are optical interfaces. The floor is a solid contact,
  // not a water-air surface: extend the bottom cell value to y=0 so extraction
  // cannot create a large horizontal sheet across the tank base.
  if (p.x <= 0 || p.z <= 0 || p.x >= dims.x + 1 || p.z >= dims.z + 1 || p.y >= dims.y + 1) { return 0.0; }
  let cell = vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1);
  if (sparseField) { return occupancyFromPhi(sparsePhiAt(cell)); }
  return fieldCell(cell);
}

fn latticeWorld(p: vec3f, dims:vec3f) -> vec3f {
  let local = clamp((p - vec3f(0.5)) / dims, vec3f(0.0), vec3f(1.0));
  return vec3f(-0.5 * u.container.x, 0.0, -0.5 * u.container.z) + local * u.container.xyz;
}

// Analytic gradient of the cube's trilinear reconstruction. The eight scalar
// values were already loaded for classification, so this replaces the former
// 48 additional volume loads performed for every emitted vertex normal.
fn surfaceNormal(lattice: vec3f, cubeBase: vec3f, cubeScale: f32, value: ptr<function, array<f32, 8>>, dims:vec3f) -> vec3f {
  let f = clamp((lattice - cubeBase) / max(cubeScale, 1.0), vec3f(0.0), vec3f(1.0));
  let dx0 = mix((*value)[1] - (*value)[0], (*value)[2] - (*value)[3], f.y);
  let dx1 = mix((*value)[5] - (*value)[4], (*value)[6] - (*value)[7], f.y);
  let dy0 = mix((*value)[3] - (*value)[0], (*value)[2] - (*value)[1], f.x);
  let dy1 = mix((*value)[7] - (*value)[4], (*value)[6] - (*value)[5], f.x);
  let lower = mix(mix((*value)[0], (*value)[1], f.x), mix((*value)[3], (*value)[2], f.x), f.y);
  let upper = mix(mix((*value)[4], (*value)[5], f.x), mix((*value)[7], (*value)[6], f.x), f.y);
  let dx = mix(dx0, dx1, f.z);
  let dy = mix(dy0, dy1, f.z);
  let dz = upper - lower;
  let scaled = vec3f(dx * dims.x / u.container.x, dy * dims.y / u.container.y, dz * dims.z / u.container.z);
  if (length(scaled) > 1e-5) { return -normalize(scaled); }
  return vec3f(0.0, 1.0, 0.0);
}

// The cube's corner values travel by pointer: WGSL passes arrays by value, and
// the former copies at up to 24 crossings per cube dominated this kernel's
// stack footprint.
fn crossing(a: vec3f, b: vec3f, va: f32, vb: f32, cubeBase: vec3f, cubeScale: f32, cubeValue: ptr<function, array<f32, 8>>, dims:vec3f) -> SurfaceVertex {
  let denominator = vb - va;
  var t = 0.5;
  if (abs(denominator) > 1e-6) { t = clamp((0.5 - va) / denominator, 0.02, 0.98); }
  let lattice = mix(a, b, t);
  return SurfaceVertex(vec4f(latticeWorld(lattice,dims), 1.0), vec4f(surfaceNormal(lattice, cubeBase, cubeScale, cubeValue,dims), 0.0));
}

// Slots for the current thread's reserved vertex block. Reservation happens
// once per workgroup in polygoniseMain; emission never touches a global
// counter, replacing the former per-triangle compare-exchange loop that
// serialized every triangle in the dispatch on a single cache line.
var<private> emitSlot: u32 = 0u;
var<private> emitLimit: u32 = 0u;

fn emitTriangle(a: SurfaceVertex, b: SurfaceVertex, c: SurfaceVertex) {
  let first = emitSlot;
  emitSlot = first + 3u;
  if (first + 3u > emitLimit) { return; }
  let geometric = cross(b.position.xyz - a.position.xyz, c.position.xyz - a.position.xyz);
  let outward = normalize(a.normal.xyz + b.normal.xyz + c.normal.xyz);
  vertices[first] = a;
  if (dot(geometric, outward) >= 0.0) {
    vertices[first + 1u] = b; vertices[first + 2u] = c;
  } else {
    vertices[first + 1u] = c; vertices[first + 2u] = b;
  }
}

fn polygoniseTetra(p: array<vec3f, 4>, v: array<f32, 4>, cubeBase: vec3f, cubeScale: f32, cubeValue: ptr<function, array<f32, 8>>, dims:vec3f) {
  var inside = array<i32, 4>();
  var outside = array<i32, 4>();
  var ni = 0; var no = 0;
  for (var i = 0; i < 4; i += 1) {
    if (v[i] >= 0.5) { inside[ni] = i; ni += 1; }
    else { outside[no] = i; no += 1; }
  }
  if (ni == 0 || ni == 4) { return; }
  if (ni == 1) {
    let a = inside[0];
    emitTriangle(crossing(p[a], p[outside[0]], v[a], v[outside[0]], cubeBase, cubeScale, cubeValue,dims), crossing(p[a], p[outside[1]], v[a], v[outside[1]], cubeBase, cubeScale, cubeValue,dims), crossing(p[a], p[outside[2]], v[a], v[outside[2]], cubeBase, cubeScale, cubeValue,dims));
  } else if (ni == 3) {
    let a = outside[0];
    emitTriangle(crossing(p[a], p[inside[0]], v[a], v[inside[0]], cubeBase, cubeScale, cubeValue,dims), crossing(p[a], p[inside[2]], v[a], v[inside[2]], cubeBase, cubeScale, cubeValue,dims), crossing(p[a], p[inside[1]], v[a], v[inside[1]], cubeBase, cubeScale, cubeValue,dims));
  } else {
    let i0 = inside[0]; let i1 = inside[1]; let o0 = outside[0]; let o1 = outside[1];
    let p00 = crossing(p[i0], p[o0], v[i0], v[o0], cubeBase, cubeScale, cubeValue,dims);
    let p01 = crossing(p[i0], p[o1], v[i0], v[o1], cubeBase, cubeScale, cubeValue,dims);
    let p10 = crossing(p[i1], p[o0], v[i1], v[o0], cubeBase, cubeScale, cubeValue,dims);
    let p11 = crossing(p[i1], p[o1], v[i1], v[o1], cubeBase, cubeScale, cubeValue,dims);
    emitTriangle(p00, p10, p11); emitTriangle(p00, p11, p01);
  }
}

fn loadCubeCornersScaled(base: vec3i, scale: i32) -> array<f32, 8> {
  let offsets = array<vec3i, 8>(
    vec3i(0,0,0), vec3i(1,0,0), vec3i(1,1,0), vec3i(0,1,0),
    vec3i(0,0,1), vec3i(1,0,1), vec3i(1,1,1), vec3i(0,1,1)
  );
  var value = array<f32, 8>();
  for (var i = 0; i < 8; i += 1) { value[i] = latticeValue(base + offsets[i] * scale); }
  return value;
}
fn loadCubeCorners(base: vec3i) -> array<f32, 8> { return loadCubeCornersScaled(base, 1); }

// Must classify vertices exactly as polygoniseTetra does: the polygonise pass
// writes into per-thread blocks sized by this count, so a mismatch corrupts a
// neighbouring thread's triangles.
fn tetraTriangleCount(v0: f32, v1: f32, v2: f32, v3: f32) -> u32 {
  var inside = 0u;
  if (v0 >= 0.5) { inside += 1u; }
  if (v1 >= 0.5) { inside += 1u; }
  if (v2 >= 0.5) { inside += 1u; }
  if (v3 >= 0.5) { inside += 1u; }
  if (inside == 0u || inside == 4u) { return 0u; }
  if (inside == 2u) { return 2u; }
  return 1u;
}

fn cubeTriangleCount(value: ptr<function, array<f32, 8>>) -> u32 {
  let tetra = array<vec4i, 6>(vec4i(0,1,2,6), vec4i(0,2,3,6), vec4i(0,3,7,6), vec4i(0,7,4,6), vec4i(0,4,5,6), vec4i(0,5,1,6));
  var triangles = 0u;
  for (var t = 0; t < 6; t += 1) {
    let ids = tetra[t];
    triangles += tetraTriangleCount((*value)[ids.x], (*value)[ids.y], (*value)[ids.z], (*value)[ids.w]);
  }
  return triangles;
}

// The sweep kernels stop here: eight corner loads, a min/max test, and one
// worklist append per *surface* cube. Emission code is confined to
// polygoniseMain so the register footprint of the full-lattice scan stays
// small enough for the occupancy that hides the load latency.
fn classifyCubeScaled(base: vec3i, scale: u32) {
  let fieldDims = select(vec3u(u.gridInfo.xyz), sparseParams.fineDims.xyz, sparseField);
  let cubeDims = fieldDims + vec3u(1);
  if (any(base < vec3i(0)) || any(vec3u(base) >= cubeDims)) { return; }
  var value = loadCubeCornersScaled(base, i32(scale));
  var minimum = 1.0; var maximum = 0.0;
  for (var i = 0; i < 8; i += 1) {
    minimum = min(minimum, value[i]); maximum = max(maximum, value[i]);
  }
  if (minimum >= 0.5 || maximum < 0.5) { return; }
  if (countOnly) {
    // The benchmark's uncapped equivalence count. Counting whole cubes here
    // keeps it exact regardless of the production worklist capacity.
    atomicAdd(&drawArgs.vertexCount, 3u * cubeTriangleCount(&value));
    return;
  }
  let slot = atomicAdd(&drawArgs.activeCubeCount, 1u);
  if (slot < arrayLength(&activeCubes)) {
    activeCubes[slot] = vec2u(u32(base.x) | (u32(base.z) << 16u), u32(base.y) | (scale << 16u));
  }
}
fn classifyCube(base: vec3i) { classifyCubeScaled(base, 1u); }

var<workgroup> workgroupVertexTotal: atomic<u32>;
var<workgroup> workgroupBaseSlot: u32;

// One thread per surface-crossing cube from the classify worklist. Threads
// combine their exact vertex counts in workgroup memory, thread 0 performs
// the workgroup's only two global atomics (block allocation and the indirect
// draw count), and each thread then emits into its private slice.
@compute @workgroup_size(${EXTRACTION_POLYGONISE_WORKGROUP})
fn polygoniseMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let activeTotal = min(atomicLoad(&drawArgs.activeCubeCount), arrayLength(&activeCubes));
  // Normal reconstruction needs the selected lattice dimensions as well as
  // the cube-local samples; keep this sixth tetra argument at every LOD.
  let fieldDimensions=select(u.gridInfo.xyz,vec3f(sparseParams.fineDims.xyz),sparseField);
  var base = vec3i(0);
  var cubeScale = 1u;
  var value = array<f32, 8>();
  var vertexCount = 0u;
  var validCube = false;
  if (gid.x < activeTotal) {
    validCube = true;
    let packedCube = activeCubes[gid.x];
    base = vec3i(i32(packedCube.x & 0xffffu), i32(packedCube.y & 0xffffu), i32(packedCube.x >> 16u));
    cubeScale = max(1u, packedCube.y >> 16u);
    if (validCube) {
      value = loadCubeCornersScaled(base, i32(cubeScale));
      vertexCount = 3u * cubeTriangleCount(&value);
    }
  }
  let localOffset = atomicAdd(&workgroupVertexTotal, vertexCount);
  workgroupBarrier();
  let capacity = arrayLength(&vertices);
  // Slots stay triangle-aligned, so clipping drops whole triangles and the
  // indirect draw count can never exceed the allocation.
  let usableCapacity = capacity - capacity % 3u;
  if (localIndex == 0u) {
    let total = atomicLoad(&workgroupVertexTotal);
    let blockStart = atomicAdd(&drawArgs.vertexAllocator, total);
    workgroupBaseSlot = blockStart;
    let fitted = u32(clamp(i32(usableCapacity) - i32(blockStart), 0, i32(total)));
    if (fitted > 0u) { atomicAdd(&drawArgs.vertexCount, fitted); }
  }
  workgroupBarrier();
  if (vertexCount == 0u) { return; }
  emitSlot = workgroupBaseSlot + localOffset;
  emitLimit = usableCapacity;
  let offsets = array<vec3i, 8>(
    vec3i(0,0,0), vec3i(1,0,0), vec3i(1,1,0), vec3i(0,1,0),
    vec3i(0,0,1), vec3i(1,0,1), vec3i(1,1,1), vec3i(0,1,1)
  );
  var p = array<vec3f, 8>();
  for (var i = 0; i < 8; i += 1) { p[i] = vec3f(base + offsets[i] * i32(cubeScale)); }
  // Six tetrahedra sharing cube diagonal 0-6.  Unlike a lookup-table
  // marching-cubes implementation this has no ambiguous saddle cases.
  let tetra = array<vec4i, 6>(vec4i(0,1,2,6), vec4i(0,2,3,6), vec4i(0,3,7,6), vec4i(0,7,4,6), vec4i(0,4,5,6), vec4i(0,5,1,6));
  for (var t = 0; t < 6; t += 1) {
    let ids = tetra[t];
    polygoniseTetra(array<vec3f,4>(p[ids.x],p[ids.y],p[ids.z],p[ids.w]), array<f32,4>(value[ids.x],value[ids.y],value[ids.z],value[ids.w]), vec3f(base), f32(cubeScale), &value,fieldDimensions);
  }
}

@compute @workgroup_size(4, 4, 4)
fn extractMain(@builtin(global_invocation_id) gid: vec3u) {
  classifyCube(vec3i(gid));
}

// Coarse extraction remains complete outside detail cores. A fine support
// halo deliberately overlaps the coarse mesh around every core: the coarse
// and fine cell-centred lattices do not share vertices, so handing ownership
// off at the outer edge of any resident page can leave a visible T-junction.
// Keeping coarse cubes through the halo gives the depth pass continuous
// coverage while the core still receives the independently transported detail.
@compute @workgroup_size(4, 4, 4)
fn extractHybridCoarseMain(@builtin(global_invocation_id) gid: vec3u) {
  let base=vec3i(gid);
  if (!sparseOverflow()) {
    let coarseCell=clamp(base-vec3i(1),vec3i(0),vec3i(u.gridInfo.xyz)-vec3i(1));
    let factor=i32(sparseParams.coarseDims.w);
    let fineCenter=vec3u(coarseCell*factor+vec3i(factor/2));
    if (sparseCorePageAt(fineCenter)) { return; }
  }
  classifyCube(base);
}

@compute @workgroup_size(1)
fn resetSurfaceWorklistMain() {
  atomicStore(&drawArgs.activeCubeCount,0u);
}

// One invocation per resident fine voxel. A lattice cube with base b is owned
// by fine cell clamp(b - 1, 0, dims - 1), so every ordinary cube has one base
// at q + 1 and a cell on a low domain boundary additionally owns base 0. The
// Cartesian product is important: it includes wall edges, floor strips, and
// triple corners as well as face interiors. The former face-only clauses left
// optical pinholes wherever a sparse detail core reached two tank walls.
@compute @workgroup_size(256)
fn extractSparseMain(@builtin(global_invocation_id) gid: vec3u) {
  if (sparseOverflow()) { return; }
  let brickSize = sparseParams.fineDims.w;
  let voxelsPerPage = brickSize * brickSize * brickSize;
  let stream = gid.x + gid.y * sparseActivePages[1] * 256u;
  let activeIndex = stream / voxelsPerPage;
  if (activeIndex >= sparseActivePages[0] || 4u + activeIndex >= arrayLength(&sparseActivePages)) { return; }
  let pageIndex = sparseActivePages[4u + activeIndex];
  if (pageIndex >= sparseParams.brickDims.w) { return; }
  let page = vec3u(pageIndex % sparseParams.brickDims.x,
    (pageIndex / sparseParams.brickDims.x) % sparseParams.brickDims.y,
    pageIndex / (sparseParams.brickDims.x * sparseParams.brickDims.y));
  let localIndex = stream - activeIndex * voxelsPerPage;
  let local = vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize, localIndex / (brickSize * brickSize));
  let q = page * brickSize + local;
  let dims = sparseParams.fineDims.xyz;
  if (any(q >= dims)) { return; }
  let xBases = array<i32, 2>(i32(q.x + 1u), 0);
  let yBases = array<i32, 2>(i32(q.y + 1u), 0);
  let zBases = array<i32, 2>(i32(q.z + 1u), 0);
  let xCount = select(1u, 2u, q.x == 0u);
  let yCount = select(1u, 2u, q.y == 0u);
  let zCount = select(1u, 2u, q.z == 0u);
  for (var zIndex = 0u; zIndex < zCount; zIndex += 1u) {
    for (var yIndex = 0u; yIndex < yCount; yIndex += 1u) {
      for (var xIndex = 0u; xIndex < xCount; xIndex += 1u) {
        classifyCube(vec3i(xBases[xIndex], yBases[yIndex], zBases[zIndex]));
      }
    }
  }
}

// Interior cubes follow the per-column cubic band instead of traversing the
// full virtual height. The dispatch includes the configured diagonal base
// delta; this local bound handles the exact four bases that touch each cube.
@compute @workgroup_size(4, 4, 4)
fn extractBandMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3i(u.gridInfo.xyz);
  if (gid.x >= u32(max(0, dims.x - 1)) || gid.z >= u32(max(0, dims.z - 1))) { return; }
  let x = i32(gid.x) + 1;
  let z = i32(gid.z) + 1;
  let b00 = columnBaseAt(x - 1, z - 1);
  let b10 = columnBaseAt(x, z - 1);
  let b01 = columnBaseAt(x - 1, z);
  let b11 = columnBaseAt(x, z);
  let minimumBase = min(min(b00, b10), min(b01, b11));
  let maximumBase = max(max(b00, b10), max(b01, b11));
  let regularLayers = i32(textureDimensions(volume).y) - 2;
  let y = minimumBase + i32(gid.y);
  if (y > dims.y || y > maximumBase + regularLayers) { return; }
  classifyCube(vec3i(x, y, z));
}

// A rigid-body clearance can lift a column base above a shallow free surface.
// Its aggregate tall fraction can then classify differently from a neighbour.
// One thread per interior x/z cube expands only those sparse vertical sides;
// ordinary wet/wet and dry/dry tall regions return after four texture loads.
@compute @workgroup_size(8, 8, 1)
fn extractTallSidesMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3i(u.gridInfo.xyz);
  if (gid.x >= u32(max(0, dims.x - 1)) || gid.y >= u32(max(0, dims.z - 1))) { return; }
  let x = i32(gid.x) + 1;
  let z = i32(gid.y) + 1;
  let b00 = columnBaseAt(x - 1, z - 1);
  let b10 = columnBaseAt(x, z - 1);
  let b01 = columnBaseAt(x - 1, z);
  let b11 = columnBaseAt(x, z);
  // Column bases come from GPU solver output; a corrupted value must not turn
  // this per-thread loop into a watchdog-length stall.
  let minimumBase = min(min(min(b00, b10), min(b01, b11)), dims.y);
  if (minimumBase <= 0) { return; }
  for (var y = 0; y < minimumBase; y += 1) { classifyCube(vec3i(x, y, z)); }
}

// The virtual lattice closes liquid against the four tank sides. Those wall
// strips extend below the free-surface band, so enumerate their unique
// perimeter cubes at full height without restoring a full-volume dispatch.
@compute @workgroup_size(64, 1, 1)
fn extractWallMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3u(u.gridInfo.xyz);
  let firstPair = 2u * (dims.x + 1u);
  let wallCount = firstPair + 2u * max(0u, dims.z - 1u);
  let total = wallCount * (dims.y + 1u);
  if (gid.x >= total) { return; }
  let wall = gid.x % wallCount;
  let y = gid.x / wallCount;
  var x = 0u;
  var z = 0u;
  if (wall < dims.x + 1u) {
    x = wall;
  } else if (wall < firstPair) {
    x = wall - (dims.x + 1u);
    z = dims.z;
  } else if (wall < firstPair + dims.z - 1u) {
    z = wall - firstPair + 1u;
  } else {
    x = dims.x;
    z = wall - (firstPair + dims.z - 1u) + 1u;
  }
  classifyCube(vec3i(i32(x), i32(y), i32(z)));
}
`;

// Sizes the polygonise indirect dispatch from the classify worklist. Kept in
// its own module and bind group so the indirect-args buffer is never bound
// while it is consumed by dispatchWorkgroupsIndirect (WebGPU forbids a
// writable-storage binding and indirect use in the same dispatch scope).
export const extractionPrepareShader = /* wgsl */ `
struct IndirectArgs { vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32, activeCubeCount: u32 }
struct DispatchArgs { x: u32, y: u32, z: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: IndirectArgs;
@group(0) @binding(1) var<storage, read> activeCubes: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> dispatchArgs: DispatchArgs;
@compute @workgroup_size(1)
fn prepareMain() {
  let activeTotal = min(drawArgs.activeCubeCount, arrayLength(&activeCubes));
  dispatchArgs = DispatchArgs((activeTotal + ${EXTRACTION_POLYGONISE_WORKGROUP - 1}u) / ${EXTRACTION_POLYGONISE_WORKGROUP}u, 1u, 1u);
}
`;

export const WATER_INTERFACE_CULL_MODES = Object.freeze({
  front: "back" as GPUCullMode,
  back: "front" as GPUCullMode,
});

export const surfaceRasterShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f }
struct SurfaceVertex { position:vec4f, normal:vec4f }
override interfaceCoverageExpansionPixels:f32=0.0;
override peelBehindFirstExit:f32=0.0;
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage,read> vertices: array<SurfaceVertex>;
@group(1) @binding(0) var firstBackPosition:texture_2d<f32>;
${cameraApertureShaderLibrary("u")}
struct Out { @builtin(position) clip:vec4f, @location(0) world:vec3f, @location(1) normal:vec3f }
fn project(world:vec3f)->vec4f {
  let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0))); let up=normalize(cross(right,forward));
  let relative=world-u.cameraPosition.xyz; let depth=max(dot(relative,forward),0.001);
  let aspect=u.viewport.x/max(u.viewport.y,1.0);
  let aperture=cameraTanHalfFov();
  let ndc=vec2f(dot(relative,right)/(depth*aspect*aperture),dot(relative,up)/(depth*aperture));
  return vec4f(ndc*depth,clamp(depth/50.0,0.0,1.0)*depth,depth);
}
@vertex fn surfaceVertex(@builtin(vertex_index) index:u32)->Out {
  let v=vertices[index]; var o:Out; o.clip=project(v.position.xyz);o.world=v.position.xyz;o.normal=normalize(v.normal.xyz);
  // A closed liquid/wall silhouette is shared by a front-facing free-surface
  // triangle and a back-facing wall triangle. The raster top-left rule can
  // otherwise give their exact shared edge to the back pass alone at one
  // pixel. Expand only front-facing triangle coverage by one conservative
  // raster pixel; the 0.75-pixel margin still left a reproducible back-only
  // wall-corner sample in the reverse Dawn view at t=0.368 s. Back faces
  // remain culled, so actual holes are still visible to the strict
  // back-without-front smoke oracle.
  if(interfaceCoverageExpansionPixels>0.0){
    let first=index-index%3u;let c0=project(vertices[first].position.xyz);let c1=project(vertices[first+1u].position.xyz);let c2=project(vertices[first+2u].position.xyz);
    let center=(c0.xy/c0.w+c1.xy/c1.w+c2.xy/c2.w)/3.0;var ndc=o.clip.xy/o.clip.w;let radial=ndc-center;
    if(dot(radial,radial)>1e-12){ndc+=normalize(radial)*interfaceCoverageExpansionPixels*vec2f(2.0/max(u.viewport.x,1.0),2.0/max(u.viewport.y,1.0));o.clip.x=ndc.x*o.clip.w;o.clip.y=ndc.y*o.clip.w;}
  }
  return o;
}
struct SurfaceOut { @location(0) position:vec4f, @location(1) normal:vec4f }
@fragment fn surfaceFragment(input:Out)->SurfaceOut {
  if(peelBehindFirstExit>.5){
    let firstBack=textureLoad(firstBackPosition,vec2i(input.clip.xy),0);
    if(firstBack.a<.5){discard;}
    let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
    let firstExitDepth=dot(firstBack.xyz-u.cameraPosition.xyz,forward);
    let candidateDepth=dot(input.world-u.cameraPosition.xyz,forward);
    let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));
    if(candidateDepth<=firstExitDepth+max(.0005,.04*cellSize)){discard;}
  }
  var o:SurfaceOut;o.position=vec4f(input.world,1.0);o.normal=vec4f(normalize(input.normal),1.0);return o;
}
`;

/**
 * Edge of the square caustic map, which is also the receiver lattice's.
 *
 * The map is an orthographic plan projection of the whole container, so one
 * texel is `container.x / 384` by `container.z / 384` — 4.7 by 3.1 mm on the
 * hero garden, against a global-fine surface sampled at 6.25 mm. Sampling the
 * receiver at exactly the map's own resolution is the only choice that needs no
 * argument: a coarser receiver would move a caustic further than the map can
 * represent, and a finer one would resolve relief the splat cannot land on.
 */
export const CAUSTIC_MAP_RESOLUTION = 384;

/**
 * Refracted caustics, as the ray bundle each surface triangle carries.
 *
 * Four things were wrong with the projection this replaces, and they compound:
 * the light direction was the literal `[-0.45, 0.86, 0.28]` default rather than
 * the scene's; the receiver was a plane at `y = 0.006` regardless of what the
 * ground under the water actually was; the deposited energy was
 * `.012 + .045 * n.y^2`, which is a function of *tilt alone* and therefore
 * cannot form a filament however the surface curves; and — decisively — nothing
 * ever sampled the result, so none of it was visible either way.
 *
 * **The energy term.** A caustic is a Jacobian: the brightness at a receiver
 * point is the ratio between the cross-section a refracted bundle presents to
 * the light and the footprint it lands on. Where the surface is convex the
 * bundle spreads and the ratio falls below one; where it is concave the bundle
 * converges and the ratio runs away, which is the filament. The standard cheap
 * form differences the refracted landing map across neighbouring surface
 * samples; this shader uses the mesh's own triangle as that stencil, so the
 * differencing vectors are the triangle's two edges `(v1 - v0, v2 - v0)` and
 * the ratio comes out *exactly* rather than to first order:
 *
 *   numerator   = |dot(0.5 * cross(e1, e2), L)|   — the bundle's cross-section
 *                                                   perpendicular to the light
 *   denominator = 0.5 * |f1.x * f2.z - f1.z * f2.x| — the plan area its three
 *                                                   landing points enclose
 *
 * where `f_i` are the landing-point edges. No normals are consulted for the
 * energy at all: the per-vertex shading normals only steer each corner's
 * refraction, and the geometric normal that carries the flux is the cross
 * product. That is what makes this term survive a mesh whose vertex normals are
 * noisy, which a marching-tetrahedra surface's always are.
 *
 * **What is stored.** Not radiance — the *ratio to the illumination the dry
 * pass already assumed*. The SVO lights the pond floor with the unrefracted key,
 * so the correction the composite must apply is exactly
 * `refracted / flat-surface`, which is 1 on still water. Dividing by the flat
 * reference here rather than in the consumer also makes the additive blend mean
 * the right thing: two bundles landing on one texel sum to a ratio above one,
 * which is a caustic, and an uncovered texel keeps alpha at zero and is left
 * alone rather than going black.
 *
 * The light's own passage through the water *is* included, because nothing else
 * accounts for it: the dry pass lights the basin floor as if the pond were not
 * there. So a still hero pond deposits `exp(-absorption * pathLength)` rather
 * than 1, which is the second half of what makes it read teal.
 */
export const causticShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f }
struct SurfaceVertex { position:vec4f, normal:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> vertices:array<SurfaceVertex>;
${waterSceneOpticsShaderLibrary(0, 2, 3)}
${unifiedLightingShaderLibrary}
struct Out { @builtin(position) clip:vec4f, @location(0) @interpolate(flat) energy:vec3f, @location(1) @interpolate(flat) covered:f32 }

// Where a refracted ray meets the receiver, as (distance, converged).
//
// A fixed-point iteration on t = (origin.y - h(origin + d t)) / -d.y rather
// than a march, because a march that resolved a texel would need ~50
// heightfield fetches and this runs once per vertex per triangle -- three
// times over, since a vertex shader cannot hoist per-triangle work. Five
// evaluations is what the same accuracy costs here.
//
// The iteration contracts exactly when |grad h| * |d.xz| < |d.y|: the receiver
// is flatter under this ray than the ray is steep. That is not a limitation to
// apologise for, it is the condition under which a single landing point exists
// at all. The pond's inner face falls 155 mm in 35 mm of run -- a slope of 4.4
// -- so rays aimed at the wall correctly fail to converge and their bundles are
// dropped, instead of being deposited at whichever point the iteration happened
// to stop on. The test is the last step's horizontal movement against one map
// texel.
fn causticShadingNormal(stored:vec3f,geometric:vec3f)->vec3f{
  let n=normalize(stored);
  return select(-n,n,dot(n,geometric)>=0.0);
}

fn causticLanding(origin:vec3f,direction:vec3f)->vec2f{
  let descent=max(-direction.y,1e-3);
  let ceiling=4.0*u.container.y;
  var t=clamp((origin.y-waterReceiverHeight(origin.x,origin.z))/descent,0.0,ceiling);
  var previous=t;
  for(var iteration=0;iteration<4;iteration+=1){
    previous=t;
    let p=origin+direction*t;
    t=clamp((origin.y-waterReceiverHeight(p.x,p.z))/descent,0.0,ceiling);
  }
  let texel=max(u.container.x,u.container.z)/${CAUSTIC_MAP_RESOLUTION}.0;
  return vec2f(t,select(0.0,1.0,abs(t-previous)*length(direction.xz)<=texel));
}

@vertex fn causticVertex(@builtin(vertex_index) index:u32)->Out {
  let first=index-index%3u;
  let p0=vertices[first].position.xyz;
  let p1=vertices[first+1u].position.xyz;
  let p2=vertices[first+2u].position.xyz;
  // The scene's key, unconditionally — not the environment preset's. The map is
  // a *correction* to what the dry pass already put on the floor, and the dry
  // pass lights from buildSvoSceneLights, whose un-authored default is the
  // same [-0.45, 0.86, 0.28] this resolves to. Falling back to the environment's
  // sun here would divide by a reference the receiver was never lit with.
  let towardLight=waterAuthoredKeyDirection();
  // Twice the triangle's area normal. Its sign is the mesh's winding, which the
  // extraction orients outward, so a downward-facing face has a negative dot
  // with an upward light and is rejected below rather than mirrored.
  let areaNormal=.5*cross(p1-p0,p2-p0);
  let crossSection=dot(areaNormal,towardLight);
  var o:Out;
  o.clip=vec4f(2.0,2.0,0.0,1.0);o.energy=vec3f(0.0);o.covered=0.0;
  // A face the light does not see refracts nothing, and a face whose own area
  // has collapsed carries no bundle to divide by.
  if(crossSection<=1e-9){return o;}
  let geometric=normalize(areaNormal);
  let eta=1.0/waterIndexOfRefraction();
  // Each corner refracts through its own shading normal so the landing triangle
  // follows the surface's curvature rather than the tessellation's facets, which
  // is the whole source of the Jacobian's variation. The stored normals are the
  // level set's gradient and are not orientation-guaranteed — the composite
  // flips them against the view ray for the same reason — so each is put on the
  // winding's side before it is refracted through.
  let d0=refract(-towardLight,causticShadingNormal(vertices[first].normal.xyz,geometric),eta);
  let d1=refract(-towardLight,causticShadingNormal(vertices[first+1u].normal.xyz,geometric),eta);
  let d2=refract(-towardLight,causticShadingNormal(vertices[first+2u].normal.xyz,geometric),eta);
  if(max(max(d0.y,d1.y),d2.y)>-.02){return o;}
  let l0=causticLanding(p0,d0);
  let l1=causticLanding(p1,d1);
  let l2=causticLanding(p2,d2);
  if(l0.y+l1.y+l2.y<2.5){return o;}
  let q0=p0+d0*l0.x;let q1=p1+d1*l1.x;let q2=p2+d2*l2.x;
  let f1=q1-q0;let f2=q2-q0;
  let footprint=.5*abs(f1.x*f2.z-f1.z*f2.x);
  // The flat-water reference the dry pass already applied to this floor: an
  // unrefracted key arriving on level ground at the same Fresnel geometry.
  let flatTransmission=1.0-unifiedDielectricFresnel(max(towardLight.y,1e-3),waterFresnelF0());
  let reference=max(towardLight.y,1e-3)*flatTransmission;
  let transmission=1.0-unifiedDielectricFresnel(clamp(dot(geometric,towardLight),0.0,1.0),waterFresnelF0());
  let travel=(l0.x+l1.x+l2.x)/3.0;
  let concentration=crossSection*transmission/(max(footprint,1e-9)*reference);
  // A bundle converging onto a point is a singularity in the continuum and a
  // firefly on a 384-texel map. Eight times the still-water level is about as
  // bright as a real caustic filament gets before the receiver's own resolution
  // is what is being measured.
  o.energy=min(vec3f(8.0),vec3f(concentration))*unifiedBeerLambert(waterAbsorption(),travel);
  o.covered=1.0;
  var landing=q0;
  if(index==first+1u){landing=q1;}else if(index==first+2u){landing=q2;}
  o.clip=vec4f(2.0*landing.x/u.container.x,2.0*landing.z/u.container.z,0.0,1.0);
  return o;
}
@fragment fn causticFragment(input:Out)->@location(0) vec4f{
  if(input.covered<.5){discard;}
  return vec4f(input.energy,1.0);
}
`;

export const compositeShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16> }
struct BodyGPU { positionRadius:vec4f, halfSizeShape:vec4f, orientation:vec4f, colorSelected:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var sceneTexture:texture_2d<f32>;
@group(0) @binding(2) var frontPosition:texture_2d<f32>;
@group(0) @binding(3) var frontNormal:texture_2d<f32>;
@group(0) @binding(4) var backPosition:texture_2d<f32>;
@group(0) @binding(5) var backNormal:texture_2d<f32>;
@group(0) @binding(6) var linearSampler:sampler;
@group(0) @binding(7) var<storage,read> bodies:array<BodyGPU,12>;
@group(0) @binding(8) var liquidField:texture_3d<f32>;
@group(0) @binding(9) var tallCellBases:texture_2d<f32>;
@group(0) @binding(10) var rearFrontPosition:texture_2d<f32>;
@group(0) @binding(11) var rearFrontNormal:texture_2d<f32>;
@group(0) @binding(12) var rearBackPosition:texture_2d<f32>;
@group(0) @binding(13) var rearBackNormal:texture_2d<f32>;
@group(0) @binding(14) var causticMap:texture_2d<f32>;
${waterSceneOpticsShaderLibrary(0, 15, 16)}
${cameraApertureShaderLibrary("u")}
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
fn project(world:vec3f)->vec2f{let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(f,vec3f(0,1,0)));let up=normalize(cross(r,f));let q=world-u.cameraPosition.xyz;let d=max(dot(q,f),1e-4);let aperture=cameraTanHalfFov();let ndc=vec2f(dot(q,r)/(d*u.viewport.x/max(u.viewport.y,1.0)*aperture),dot(q,up)/(d*aperture));return vec2f(ndc.x*.5+.5,.5-ndc.y*.5);}
fn safeSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{return textureSampleLevel(texture,linearSampler,clamp(uv,vec2f(.001),vec2f(.999)),0);}
// Interface targets carry a binary validity mask in alpha. Their RGB is not
// useful outside that mask, so bilinear filtering makes a boundary sample
// premultiplied by its fractional coverage. Divide that coverage back out;
// this preserves smooth interpolation among valid surface samples without
// pulling world positions toward zero or normals toward the clear value.
fn safeInterfaceSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{
  let sampled=safeSample(texture,uv);
  if(sampled.a<=1e-4){return vec4f(0.0);}
  return vec4f(sampled.rgb/sampled.a,sampled.a);
}
fn cameraRay(textureUV:vec2f)->vec3f{let ndc=vec2f(textureUV.x*2.0-1.0,1.0-textureUV.y*2.0);let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let aperture=cameraTanHalfFov();return normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*aperture+up*ndc.y*aperture);}
fn boxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let near3=min(a,b);let far3=max(a,b);return vec2f(max(max(near3.x,near3.y),near3.z),min(min(far3.x,far3.y),far3.z));}
${environmentShaderLibrary}
${unifiedLightingShaderLibrary}
${unifiedDisplayTransferShaderLibrary}
// The water's key light. A document that authors one wins — that is the only
// way the highlight on the water can agree with the set the SVO lit from the
// same record — and one that does not keeps the environment preset's sun, which
// is byte-identical to the authored default. See resolveWaterKeyLight.
fn waterKeyDirection()->vec3f{return select(environmentLightDirection(),waterAuthoredKeyDirection(),waterKeyAuthored());}
fn waterKeyColor()->vec3f{return select(environmentLightColor(),waterKeyRadiance(),waterKeyAuthored());}
// The two silhouette tints in the water shading below predate authorable optics:
// they are the same in-scattering seen at a grazing angle, tuned by eye against
// the clean-water table. Scaling them by the scene's departure from that table
// leaves every un-authoring scene byte-identical and lets an authored pond
// carry its own hue all the way out to the rim instead of ending in a fixed
// pale turquoise that contradicts its body.
fn waterTintScale()->vec3f{return waterScatter()/vec3f(${WATER_OPTICS.scatter.join(",")});}
fn qrot(q:vec4f,v:vec3f)->vec3f{let a=cross(q.yzw,v);return v+2.0*(q.x*a+cross(q.yzw,a));}
fn qinv(q:vec4f,v:vec3f)->vec3f{return qrot(vec4f(q.x,-q.yzw),v);}
struct RigidHit { t:f32, n:vec3f }
fn sphereRigidHit(ro:vec3f,rd:vec3f,center:vec3f,radius:f32)->RigidHit{
  let oc=ro-center;let b=dot(oc,rd);let discriminant=b*b-dot(oc,oc)+radius*radius;
  if(discriminant<0.0){return RigidHit(1e20,vec3f(0,1,0));}
  let root=sqrt(discriminant);var t=-b-root;if(t<=1e-4){t=-b+root;}
  if(t<=1e-4){return RigidHit(1e20,vec3f(0,1,0));}
  return RigidHit(t,normalize(ro+rd*t-center));
}
fn cylinderRigidHit(ro:vec3f,rd:vec3f,radius:f32,halfHeight:f32,capped:bool)->RigidHit{
  var best=RigidHit(1e20,vec3f(0,1,0));let a=dot(rd.xz,rd.xz);
  if(a>1e-8){let b=dot(ro.xz,rd.xz);let c=dot(ro.xz,ro.xz)-radius*radius;let discriminant=b*b-a*c;
    if(discriminant>=0.0){let root=sqrt(discriminant);var t=(-b-root)/a;if(t<=1e-4){t=(-b+root)/a;}let y=ro.y+rd.y*t;
      if(t>1e-4&&abs(y)<=halfHeight){let p=ro+rd*t;best=RigidHit(t,normalize(vec3f(p.x,0,p.z)));}}}
  if(capped&&abs(rd.y)>1e-8){for(var side=-1.0;side<=1.0;side+=2.0){let t=(side*halfHeight-ro.y)/rd.y;let p=ro+rd*t;
    if(t>1e-4&&t<best.t&&dot(p.xz,p.xz)<=radius*radius){best=RigidHit(t,vec3f(0,side,0));}}}
  return best;
}
fn bodyRigidHit(ro:vec3f,rd:vec3f,body:BodyGPU)->RigidHit{
  let o=qinv(body.orientation,ro-body.positionRadius.xyz);let d=qinv(body.orientation,rd);let shape=i32(round(body.halfSizeShape.w));var hit=RigidHit(1e20,vec3f(0,1,0));
  if(shape==0){hit=sphereRigidHit(o,d,vec3f(0),body.halfSizeShape.x);}
  else if(shape==1){let interval=boxHit(o,d,-body.halfSizeShape.xyz,body.halfSizeShape.xyz);var t=interval.x;if(t<=1e-4){t=interval.y;}
    if(t>1e-4&&interval.x<=interval.y){let p=o+d*t;let q=abs(p/max(body.halfSizeShape.xyz,vec3f(1e-5)));var n=vec3f(0,0,sign(p.z));
      if(q.x>=q.y&&q.x>=q.z){n=vec3f(sign(p.x),0,0);}else if(q.y>=q.z){n=vec3f(0,sign(p.y),0);}hit=RigidHit(t,n);}}
  else if(shape==2){hit=cylinderRigidHit(o,d,body.halfSizeShape.x,body.halfSizeShape.y,false);let upper=sphereRigidHit(o,d,vec3f(0,body.halfSizeShape.y,0),body.halfSizeShape.x);let lower=sphereRigidHit(o,d,vec3f(0,-body.halfSizeShape.y,0),body.halfSizeShape.x);if(upper.t<hit.t){hit=upper;}if(lower.t<hit.t){hit=lower;}}
  else{hit=cylinderRigidHit(o,d,body.halfSizeShape.x,body.halfSizeShape.y,true);}
  return RigidHit(hit.t,normalize(qrot(body.orientation,hit.n)));
}
fn nearestRigid(ro:vec3f,rd:vec3f)->RigidHit{var best=RigidHit(1e20,vec3f(0,1,0));for(var i=0u;i<12u;i+=1u){if(i>=u32(round(u.options.z))){break;}let hit=bodyRigidHit(ro,rd,bodies[i]);if(hit.t<best.t){best=hit;}}return best;}

// The raster mesh is the fast global solution. Only pixels whose analytic
// rigid depth lies in this narrow band evaluate the resident implicit field.
fn contactOccupancyFromPhi(phi:f32)->f32{let band=4.0*u.container.y/max(u.gridInfo.y,1.0);return clamp(0.5-phi/band,0.0,1.0);}
fn contactFieldCell(cell:vec3i)->f32{
  let dims=vec3i(u.gridInfo.xyz);if(any(cell<vec3i(0))||any(cell>=dims)){return 0.0;}let mode=u.gridInfo.w;
  if(mode<1.5){return textureLoad(liquidField,cell,0).x;}if(mode>2.5){return contactOccupancyFromPhi(textureLoad(liquidField,cell,0).x);}
  let base=i32(round(textureLoad(tallCellBases,cell.xz,0).x));
  if(cell.y<base&&base>0){let t=clamp(f32(cell.y)/f32(max(base-1,1)),0.0,1.0);return contactOccupancyFromPhi(mix(textureLoad(liquidField,vec3i(cell.x,0,cell.z),0).x,textureLoad(liquidField,vec3i(cell.x,1,cell.z),0).x,t));}
  let packedY=2+cell.y-base;let stored=vec3i(textureDimensions(liquidField));if(packedY<2||packedY>=stored.y){return 0.0;}return contactOccupancyFromPhi(textureLoad(liquidField,vec3i(cell.x,packedY,cell.z),0).x);
}
fn contactFluidValue(world:vec3f)->f32{
  let dims=vec3i(u.gridInfo.xyz);let boundsMin=vec3f(-0.5*u.container.x,0,-0.5*u.container.z);let uvw=clamp((world-boundsMin)/u.container.xyz,vec3f(0),vec3f(1));
  let q=clamp(uvw*vec3f(dims)-vec3f(0.5),vec3f(0),vec3f(dims-vec3i(1)));let base=vec3i(floor(q));let f=fract(q);
  let c000=contactFieldCell(base);let c100=contactFieldCell(base+vec3i(1,0,0));let c010=contactFieldCell(base+vec3i(0,1,0));let c110=contactFieldCell(base+vec3i(1,1,0));
  let c001=contactFieldCell(base+vec3i(0,0,1));let c101=contactFieldCell(base+vec3i(1,0,1));let c011=contactFieldCell(base+vec3i(0,1,1));let c111=contactFieldCell(base+vec3i(1,1,1));
  return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
struct ContactSurface { point:vec3f, normal:vec3f, valid:bool }
fn refineContactSurface(ro:vec3f,rd:vec3f,rasterT:f32,cellSize:f32)->ContactSurface{
  let radius=1.35*cellSize;let lo=max(1e-4,rasterT-radius);let hi=rasterT+radius;var t=rasterT;let initialError=abs(contactFluidValue(ro+rd*t)-0.5);
  let epsilon=max(2e-4,0.18*cellSize);
  for(var iteration=0;iteration<4;iteration+=1){let point=ro+rd*t;let value=contactFluidValue(point)-0.5;let derivative=(contactFluidValue(point+rd*epsilon)-contactFluidValue(point-rd*epsilon))/(2.0*epsilon);if(abs(derivative)<1e-5){break;}t=clamp(t-value/derivative,lo,hi);}
  let point=ro+rd*t;let e=max(3e-4,0.3*cellSize);let gradient=vec3f(contactFluidValue(point+vec3f(e,0,0))-contactFluidValue(point-vec3f(e,0,0)),contactFluidValue(point+vec3f(0,e,0))-contactFluidValue(point-vec3f(0,e,0)),contactFluidValue(point+vec3f(0,0,e))-contactFluidValue(point-vec3f(0,0,e)))/(2.0*e);
  let normal=select(-rd,-normalize(gradient),length(gradient)>1e-5);return ContactSurface(point,normal,initialError<0.42&&abs(contactFluidValue(point)-0.5)<0.12);
}
fn boxNormal(point:vec3f,center:vec3f,halfSize:vec3f)->vec3f{
  let q=abs((point-center)/max(halfSize,vec3f(1e-5)));
  if(q.x>=q.y&&q.x>=q.z){return vec3f(sign(point.x-center.x),0,0);}
  if(q.y>=q.z){return vec3f(0,sign(point.y-center.y),0);}
  return vec3f(0,0,sign(point.z-center.z));
}
// The compact SVO G-buffer uses zero linear depth on a miss, while the raster
// compatibility pass retains its historical half-float maximum sentinel.
fn resolvedDrySceneDepth(encodedDepth:f32)->f32{return select(65504.0,encodedDepth,encodedDepth>0.0);}
// The caustic consumer. The map holds the ratio between the illumination the
// refracting surface actually delivers to a receiver point and the flat-water
// illumination the dry pass already applied there, so this is a modulation of
// the transmitted radiance rather than an addition to it: an uncovered texel
// (alpha zero) leaves the dry pass's own answer alone, which is what a
// screen-space additive overlay could never do.
//
// The receiver point is reconstructed from the dry scene's own linear depth
// rather than assumed, so the caustic lands on the sculpted basin the renderer
// actually drew. resolvedDrySceneDepth turns a miss into the far sentinel.
fn causticModulation(textureUV:vec2f)->vec3f{
  let strength=waterCausticStrength();
  if(strength<=0.0){return vec3f(1.0);}
  let depth=resolvedDrySceneDepth(safeSample(sceneTexture,textureUV).a);
  if(!(depth>0.0)||depth>60000.0){return vec3f(1.0);}
  let world=u.cameraPosition.xyz+cameraRay(textureUV)*depth;
  let mapUV=vec2f(.5+world.x/max(u.container.x,1e-4),.5-world.z/max(u.container.z,1e-4));
  if(any(mapUV<vec2f(0.0))||any(mapUV>vec2f(1.0))){return vec3f(1.0);}
  let sampled=textureSampleLevel(causticMap,linearSampler,mapUV,0.0);
  if(sampled.a<=1e-4){return vec3f(1.0);}
  // The additive caustic target is premultiplied at filtered coverage edges:
  // RGB and alpha both approach zero between tiny landing triangles. Restore
  // the uncovered share to neutral illumination instead of mistaking that
  // fractional RGB for a fully covered, nearly black caustic. Alpha above one
  // remains unnormalized because it represents real overlapping ray bundles.
  let coverage=clamp(sampled.a,0.0,1.0);
  let deposited=max(sampled.rgb,vec3f(0.0))/max(coverage,1e-4);
  let modulation=mix(vec3f(1.0),deposited,coverage);
  return mix(vec3f(1.0),modulation,strength);
}
fn compositeFrontGlass(color:vec3f,ro:vec3f,rd:vec3f,sceneDepth:f32)->vec3f{
  // The garden pond has no vessel: nothing to composite in front of the water.
  if(environmentIndex()==7){return color;}
  let size=u.container.xyz;let mn=vec3f(-size.x*.5,0,-size.z*.5);let mx=vec3f(size.x*.5,size.y,size.z*.5);let hit=boxHit(ro,rd,mn,mx);
  if(hit.x>hit.y||hit.y<=0.0){return color;}
  let glassT=select(hit.x,hit.y,hit.x<=1e-4);
  if(glassT<=1e-4||glassT>resolvedDrySceneDepth(sceneDepth)+.001){return color;}
  let center=(mn+mx)*.5;let halfSize=size*.5;let point=ro+rd*glassT;let normal=boxNormal(point,center,halfSize);
  let q=abs((point-center)/max(halfSize,vec3f(1e-5)));
  let edgeCoordinate=max(max(min(q.x,q.y),min(q.x,q.z)),min(q.y,q.z));
  let outerEdge=smoothstep(.955,.998,edgeCoordinate);
  let innerEdge=smoothstep(.91,.975,edgeCoordinate)*(1.0-outerEdge);
  let cosine=clamp(abs(dot(-rd,normal)),0.0,1.0);let fresnel=unifiedDielectricFresnel(cosine,${GLASS_OPTICS.fresnelF0.toFixed(2)});
  let paneAlpha=.008+.065*fresnel;let edgeAlpha=.52*outerEdge+.10*innerEdge;
  let glassTint=vec3f(${GLASS_OPTICS.tint.join(",")});var result=mix(color,color*vec3f(.985,1.0,.998)+glassTint*.035,paneAlpha+edgeAlpha);
  let light=environmentLightDirection();let glint=unifiedSpecularLobe(normal,-rd,light,240.0);
  result+=environmentLightColor()*(glint*(.18+.82*outerEdge)+fresnel*outerEdge*.16);
  return result;
}
// The first depth-tested pair describes only the nearest connected water
// interval. A breaking sheet can leave another interval behind it, so the
// interface raster peels one more front/back pair after the first exit. Shade
// that pair before the foreground interval consumes the transmitted radiance.
fn compositeRearWater(textureUV:vec2f,dryColor:vec3f)->vec3f{
  let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let rd=cameraRay(textureUV);
  let front=safeInterfaceSample(rearFrontPosition,textureUV);if(front.a<.5){return dryColor;}
  let scene=safeSample(sceneTexture,textureUV);let frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));
  if(resolvedDrySceneDepth(scene.a)+max(.0015,.18*cellSize)<frontDepth){return dryColor;}
  var n=normalize(safeInterfaceSample(rearFrontNormal,textureUV).xyz);if(dot(n,rd)>0.0){n=-n;}
  var inside=refract(rd,n,1.0/waterIndexOfRefraction());if(length(inside)<1e-5){inside=reflect(rd,n);}
  var exitUV=textureUV;var back=vec4f(0);var exitN=vec3f(0,-1,0);
  for(var iteration=0;iteration<3;iteration+=1){back=safeInterfaceSample(rearBackPosition,exitUV);if(back.a<.5){break;}let backDepth=dot(back.xyz-ro,forward);let frontPlane=dot(front.xyz-ro,forward);let travel=max(0.0,(backDepth-frontPlane)/max(dot(inside,forward),.001));exitUV=project(front.xyz+inside*travel);exitN=normalize(safeInterfaceSample(rearBackNormal,exitUV).xyz);}
  let refinedBack=safeInterfaceSample(rearBackPosition,exitUV);if(refinedBack.a<.5){return dryColor;}back=refinedBack;exitN=normalize(safeInterfaceSample(rearBackNormal,exitUV).xyz);
  let thickness=length(back.xyz-front.xyz);if(thickness<1e-4){return dryColor;}
  if(dot(exitN,inside)<0.0){exitN=-exitN;}var outgoing=refract(inside,-exitN,waterIndexOfRefraction());let tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}
  let backgroundUV=project(back.xyz+outgoing*(.55+.45*thickness));let transmitted=safeSample(sceneTexture,backgroundUV).rgb*causticModulation(backgroundUV);
  let refracted=unifiedAbsorbingTransmission(transmitted,waterAbsorption(),waterScatter(),thickness);
  let reflectedDir=reflect(rd,n);var reflected=environmentLight(reflectedDir);let ssr=safeSample(sceneTexture,project(front.xyz+reflectedDir*.8));reflected=mix(reflected,ssr.rgb,select(0.0,.32,ssr.a>0.0&&ssr.a<60000.0));
  let cosine=clamp(dot(-rd,n),0.0,1.0);let fresnel=unifiedDielectricFresnel(cosine,waterFresnelF0());var water=mix(refracted,reflected,fresnel);if(tir){water=mix(water,environmentLight(outgoing),.88);}
  water+=waterKeyColor()*unifiedSpecularLobe(n,-rd,waterKeyDirection(),180.0)*1.4;
  water+=vec3f(.018,.10,.085)*waterTintScale()*(1.0-exp(-thickness*2.4));water+=vec3f(.08,.18,.15)*waterTintScale()*pow(1.0-cosine,3.0)*.15;
  return water;
}
// Scenery is geometry, not a screen-space overlay: every frond, batten and
// blade that used to be painted here in NDC is now an analytic primitive in
// the scene's own scenery graph, so it parallaxes, occludes and takes light like the rest
// of the world. Only the lens falloff remains, which belongs to the camera.
fn finish(color:vec3f,ndc:vec2f)->vec4f{let c=color*(1.0-.08*dot(ndc*.55,ndc*.55));return vec4f(unifiedDisplayGradeBalanced(c,waterDisplayExposure(),waterDisplayToneCurve(),waterDisplayWhiteBalance()),1);}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  // Full-screen interpolated UV has Y=1 at the top of the render target,
  // while sampled WebGPU textures have Y=0 there. The shared legacy upscaler
  // performs the same conversion for the final target; all raster-path
  // intermediate reads and world projections must do it here as well.
  let ndc=input.uv*2.0-1.0;let textureUV=vec2f(input.uv.x,1.0-input.uv.y);let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let aperture=cameraTanHalfFov();let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*aperture+up*ndc.y*aperture);
  let scene=safeSample(sceneTexture,textureUV);var front=safeInterfaceSample(frontPosition,textureUV);if(front.a<.5){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}var frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));let depthEpsilon=max(.0015,.18*cellSize);
  var n=normalize(safeInterfaceSample(frontNormal,textureUV).xyz);let rigidFront=nearestRigid(ro,rd);let contactBand=${CONTACT_RESOLVE_BAND_CELLS.toFixed(1)}*cellSize;
  if(u.gridInfo.w>.5&&rigidFront.t<1e19&&abs(rigidFront.t-frontDepth)<=contactBand){let contact=refineContactSurface(ro,rd,frontDepth,cellSize);if(contact.valid){front=vec4f(contact.point,1);frontDepth=dot(contact.point-ro,rd);n=contact.normal;}if(rigidFront.t<=frontDepth+max(3e-4,.03*cellSize)){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}}
  if(resolvedDrySceneDepth(scene.a)+depthEpsilon<frontDepth){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}
  if(dot(n,rd)>0.0){n=-n;}let etaIn=1.0/waterIndexOfRefraction();var inside=refract(rd,n,etaIn);if(length(inside)<1e-5){inside=reflect(rd,n);}
  var exitUV=textureUV;var back=vec4f(0);var exitN=vec3f(0,-1,0);
  for(var iteration=0;iteration<3;iteration+=1){back=safeInterfaceSample(backPosition,exitUV);if(back.a<.5){break;}let backDepth=dot(back.xyz-ro,forward);let frontPlane=dot(front.xyz-ro,forward);let travel=max(0.0,(backDepth-frontPlane)/max(dot(inside,forward),.001));exitUV=project(front.xyz+inside*travel);exitN=normalize(safeInterfaceSample(backNormal,exitUV).xyz);}
  let refinedBack=safeInterfaceSample(backPosition,exitUV);if(refinedBack.a>.5){back=refinedBack;exitN=normalize(safeInterfaceSample(backNormal,exitUV).xyz);}
  var exitPoint=back.xyz;var thickness=length(exitPoint-front.xyz);let meshExitValid=back.a>=.5&&thickness>=1e-4;let innerStep=max(.0005,cellSize*.08);let innerOrigin=front.xyz+inside*innerStep;let rigidExit=nearestRigid(innerOrigin,inside);var opaqueSolidExit=false;
  if(rigidExit.t<1e19&&(!meshExitValid||rigidExit.t+innerStep<thickness)){opaqueSolidExit=true;exitPoint=innerOrigin+inside*rigidExit.t;thickness=length(exitPoint-front.xyz);}
  else if(!meshExitValid){
    // Solid contacts are not extracted as fake water-air sheets. When the
    // refracted ray reaches the floor (or a mesh exit is temporarily missing),
    // terminate it analytically at the tank boundary instead.
    let boundsMin=vec3f(-u.container.x*.5,0,-u.container.z*.5);let boundsMax=vec3f(u.container.x*.5,u.container.y,u.container.z*.5);let tankExit=boxHit(innerOrigin,inside,boundsMin,boundsMax);let travel=max(.002,tankExit.y);
    thickness=length(innerOrigin-front.xyz)+travel;exitPoint=innerOrigin+inside*travel;exitN=boxNormal(exitPoint,(boundsMin+boundsMax)*.5,u.container.xyz*.5);
  }
  var outgoing=inside;var tir=false;var backgroundUV=project(exitPoint);
  if(!opaqueSolidExit){if(dot(exitN,inside)<0.0){exitN=-exitN;}outgoing=refract(inside,-exitN,waterIndexOfRefraction());tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}backgroundUV=project(exitPoint+outgoing*(.55+.45*thickness));}
  // The modulation goes on the *dry* term rather than on what comes back:
  // where a rear water interval exists it shades its own receiver and applies
  // its own caustic, and multiplying that result again would count the floor's
  // concentration twice through two layers of water.
  let transmittedScene=compositeRearWater(backgroundUV,safeSample(sceneTexture,backgroundUV).rgb*causticModulation(backgroundUV));
  // Absorption is the scene's, not the renderer's: the same clean-water rate
  // that turns a metre of tank blue leaves a hand's breadth of pond colourless.
  // A small in-scattering term keeps thick regions luminous instead of turning
  // into opaque ink, and has to grow with the absorption it accompanies.
  let refracted=unifiedAbsorbingTransmission(transmittedScene,waterAbsorption(),waterScatter(),thickness);let reflectedDir=reflect(rd,n);var reflected=environmentLight(reflectedDir);
  let ssrUV=project(front.xyz+reflectedDir*.8);let ssr=safeSample(sceneTexture,ssrUV);reflected=mix(reflected,ssr.rgb,select(0.0,.32,ssr.a>0.0&&ssr.a<60000.0));
  let cosine=clamp(dot(-rd,n),0.0,1.0);let fresnel=unifiedDielectricFresnel(cosine,waterFresnelF0());var water=mix(refracted,reflected,fresnel);
  if(tir){water=mix(water,environmentLight(outgoing),.88);}
  water+=waterKeyColor()*unifiedSpecularLobe(n,-rd,waterKeyDirection(),180.0)*1.4;
  // Thin forward-scattering highlight at silhouettes, plus a restrained
  // turquoise body tint that grows only with actual optical thickness.
  water+=vec3f(.018,.10,.085)*waterTintScale()*(1.0-exp(-thickness*2.4));water+=vec3f(.08,.18,.15)*waterTintScale()*pow(1.0-cosine,3.0)*.15;
  return finish(compositeFrontGlass(water,ro,rd,scene.a),ndc);
}
`;

async function checkedModule(device: GPUDevice, label: string, code: string) {
  const shaderModule = device.createShaderModule({ label, code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length) throw new Error(`${label}:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
  return shaderModule;
}

/**
 * What the water pipeline needs from the document.
 *
 * One call rather than three setters, because the three facts are consumed by
 * one uniform and a scene change moves all of them together. `terrain` is the
 * caustic receiver: the water pipeline resamples it onto its own lattice rather
 * than taking a texture, so the CPU mirror stays `terrainHeightAt` and a grid
 * and an analytic heightfield reach the shader through the same path.
 */
export interface WaterSceneOpticsInput {
  readonly optics?: WaterOpticsAuthoring;
  readonly grade?: DisplayGradeAuthoring;
  readonly directional?: {
    readonly direction?: readonly [number, number, number];
    readonly colorLinear?: readonly [number, number, number];
    readonly intensity?: number;
  };
  readonly terrain?: TerrainDescription;
  /** Main-thread content identity retained across the structured-clone seam. */
  readonly terrainContentStamp?: string;
  /** The plan the caustic map projects onto, in metres. */
  readonly container?: { readonly width_m: number; readonly depth_m: number };
}

export class RasterWaterPipeline {
  private extractPipeline?: GPUComputePipeline;
  private extractBandPipeline?: GPUComputePipeline;
  private extractTallSidesPipeline?: GPUComputePipeline;
  private extractWallPipeline?: GPUComputePipeline;
  private extractGlobalFinePipeline?: GPUComputePipeline;
  private extractGlobalCoarsePipeline?: GPUComputePipeline;
  private preparePipeline?: GPUComputePipeline;
  private polygonisePipeline?: GPUComputePipeline;
  private polygoniseGlobalFineScanPipeline?: GPUComputePipeline;
  private polygoniseGlobalFineEmitPipeline?: GPUComputePipeline;
  private surfaceFrontPipeline?: GPURenderPipeline;
  private surfaceBackPipeline?: GPURenderPipeline;
  private surfaceRearFrontPipeline?: GPURenderPipeline;
  private surfaceRearBackPipeline?: GPURenderPipeline;
  private causticPipeline?: GPURenderPipeline;
  private compositePipeline?: GPURenderPipeline;
  private extractLayout?: GPUBindGroupLayout;
  private globalExtractLayout?: GPUBindGroupLayout;
  private globalPolygoniseLayout?: GPUBindGroupLayout;
  private globalPolygoniseEmitLayout?: GPUBindGroupLayout;
  private prepareLayout?: GPUBindGroupLayout;
  private causticLayout?: GPUBindGroupLayout;
  private surfaceLayout?: GPUBindGroupLayout;
  private surfacePeelLayout?: GPUBindGroupLayout;
  private compositeLayout?: GPUBindGroupLayout;
  /** A clear fluid-only background is immutable and needs one full-frame clear per attachment lifetime. */
  private clearBackgroundEncoded = false;
  private sampler?: GPUSampler;
  private vertexBuffer?: GPUBuffer;
  private indirectBuffer?: GPUBuffer;
  /**
   * GPU-resident source for the per-frame indirect-header reset.
   *
   * The reset used to be two `queue.writeBuffer` calls per frame into a buffer
   * the GPU otherwise wholly owns — a host staging round trip for 24 bytes, and
   * one applied at submit time rather than in encoder order, so any earlier
   * pass in the same command buffer would have read the already-clobbered
   * header. Both patterns live here instead and are copied buffer-to-buffer in
   * the encoder, which is both free of host traffic and correctly ordered.
   */
  private indirectResetTemplate?: GPUBuffer;
  private activeCubeBuffer?: GPUBuffer;
  private globalCubeValues?: GPUBuffer;
  private globalCubeOffsets?: GPUBuffer;
  private polygoniseDispatchBuffer?: GPUBuffer;
  private extractBindGroup?: GPUBindGroup;
  private globalExtractBindGroup?: GPUBindGroup;
  private globalPolygoniseBindGroup?: GPUBindGroup;
  private globalPolygoniseEmitBindGroup?: GPUBindGroup;
  private prepareBindGroup?: GPUBindGroup;
  private causticBindGroup?: GPUBindGroup;
  private surfaceBindGroup?: GPUBindGroup;
  private surfaceUnpeeledBindGroup?: GPUBindGroup;
  private surfacePeelBindGroup?: GPUBindGroup;
  private compositeBindGroup?: GPUBindGroup;
  private compositeBindGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  private sceneTexture?: GPUTexture;
  private sceneTextureView?: GPUTextureView;
  private frontPosition?: GPUTexture;
  private frontNormal?: GPUTexture;
  private frontDepth?: GPUTexture;
  private backPosition?: GPUTexture;
  private backNormal?: GPUTexture;
  private backDepth?: GPUTexture;
  private rearFrontPosition?: GPUTexture;
  private rearFrontNormal?: GPUTexture;
  private rearFrontDepth?: GPUTexture;
  private rearBackPosition?: GPUTexture;
  private rearBackNormal?: GPUTexture;
  private rearBackDepth?: GPUTexture;
  private causticTexture?: GPUTexture;
  private causticReceiver?: GPUTexture;
  private waterSceneOpticsBuffer?: GPUBuffer;
  private readonly waterSceneOptics = new Float32Array(WATER_SCENE_OPTICS_FLOATS);
  private waterSceneOpticsDirty = true;
  private causticStrength = 0;
  private receiverKey = "";
  private readonly receiverStampByTerrain = new WeakMap<TerrainDescription, string>();
  private geometryKey = "";
  private targetKey = "";
  private volume?: GPUTexture;
  private columnBases?: GPUTexture;
  private extractedRevision = -1;
  private lastExtractionAt_ms = -Infinity;
  private causticsValid = false;
  private sceneHasFluid = true;
  private dryInterfaceClearsEncoded = false;
  private disabledStages: DisabledRenderStages = NO_DISABLED_RENDER_STAGES;
  private secondaryParticles?: SecondaryParticleRenderPipeline;
  private globalFineLevelSet?: GlobalFineLevelSetConsumerSource;
  private coarseLevelSet?: CoarseLevelSetConsumerSource;
  private globalFineRenderParams?: GPUBuffer;
  private fallbackSparsePageTable?: GPUBuffer;
  private fallbackSparseActivePages?: GPUBuffer;
  private fallbackSparsePhi?: GPUBuffer;
  private fallbackSparseParams?: GPUBuffer;
  private fallbackSparseControl?: GPUBuffer;
  private surfaceDiagnosticReadback?: GPUBuffer;
  private surfaceDiagnosticPending = false;
  private surfaceDiagnosticCompletion?: Promise<WaterRenderDiagnostics | undefined>;
  private lastSurfaceDiagnostics?: WaterRenderDiagnostics;
  private lastSurfaceDiagnosticEncodeAt_ms = -Infinity;
  // An extraction can land while the bounded diagnostics readback is still
  // throttled (or an older receipt is in flight). Keep the replacement receipt
  // armed after the mesh revision stops changing so a paused UI cannot retain
  // an obsolete "empty" result beside newly published geometry.
  private surfaceDiagnosticsDirty = false;
  private pendingSurfaceDiagnosticGlobalFine = false;
  private pendingSurfaceDiagnosticCoarse = false;
  private pendingSurfaceDiagnosticGlobalFineGeneration?: number;
  private readonly surfaceSourceFrameCounts: Record<WaterSurfaceGeometrySource, number> = {
    "global-fine-coarse": 0,
    "compact-coarse": 0,
    "retained-previous": 0,
    empty: 0,
    volume: 0,
  };

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer
  ) {
    // Clean water and the default key until a document says otherwise, so a
    // caller that never sets scene optics gets the frozen table verbatim.
    this.setSceneOptics({});
  }

  /**
   * Adopt the document's medium, key light and caustic receiver.
   *
   * Idempotent and cheap: the uniform is only written when a lane actually
   * changes, and the receiver is only resampled when the heightfield identity
   * or the container does. Both are per-scene facts arriving on a per-frame
   * call.
   */
  setSceneOptics(input: WaterSceneOpticsInput) {
    const packed = packWaterSceneOptics(
      resolveWaterOptics(input.optics),
      resolveWaterKeyLight(input.directional),
      resolveDisplayGrade(input.grade),
    );
    this.causticStrength = packed[15];
    for (let index = 0; index < packed.length; index += 1) {
      // Receiver lanes are pipeline-owned and may already contain the sampled
      // heightfield. The document packet carries placeholders there only to
      // preserve the WGSL layout; a per-frame scene adoption must not erase
      // them before updateCausticReceiver's identity fast path returns.
      if (index >= WATER_SCENE_OPTICS_RECEIVER_FLOAT_OFFSET
          && index < WATER_SCENE_OPTICS_RECEIVER_FLOAT_OFFSET + 8) continue;
      if (this.waterSceneOptics[index] === packed[index]) continue;
      this.waterSceneOptics[index] = packed[index];
      this.waterSceneOpticsDirty = true;
    }
    this.updateCausticReceiver(
      input.terrain,
      input.container?.width_m ?? 0,
      input.container?.depth_m ?? 0,
      input.terrainContentStamp,
    );
    this.flushSceneOptics();
  }

  private flushSceneOptics() {
    if (!this.waterSceneOpticsBuffer || !this.waterSceneOpticsDirty) return;
    this.device.queue.writeBuffer(this.waterSceneOpticsBuffer, 0, this.waterSceneOptics);
    this.waterSceneOpticsDirty = false;
  }

  /**
   * Resample the scene's ground onto the caustic map's own lattice.
   *
   * `terrainHeightAt` rather than the grid's samples, because an analytic
   * terrain and a sculpted grid must reach the same shader through the same
   * path, and because the CPU function is the one the stones, the solver and
   * the renderer all already agree on. The lattice is container-aligned and
   * square in *samples*, not in metres: it matches the map it feeds, so one map
   * texel gets one receiver sample.
   */
  private updateCausticReceiver(
    terrain: TerrainDescription | undefined,
    width: number,
    depth: number,
    publishedStamp?: string,
  ) {
    // The published stamp is computed and memoized before the document crosses
    // into the worker. Direct/headless callers retain the same identity-memo
    // behavior locally, while the key itself remains purely content-based.
    let contentStamp = publishedStamp;
    if (contentStamp === undefined && terrain) {
      contentStamp = this.receiverStampByTerrain.get(terrain);
      if (contentStamp === undefined) {
        contentStamp = terrainContentStamp(terrain);
        this.receiverStampByTerrain.set(terrain, contentStamp);
      }
    }
    const key = causticReceiverContentKey(terrain, width, depth, contentStamp);
    if (key === this.receiverKey) return;
    this.receiverKey = key;
    if (!this.causticReceiver) return;
    const size = CAUSTIC_MAP_RESOLUTION;
    const spacing = Math.max(Math.max(width, depth) / (size - 1), 1e-6);
    const originX = -0.5 * width, originZ = -0.5 * depth;
    const heights = new Float32Array(size * size);
    if (terrain && width > 0 && depth > 0) {
      for (let row = 0; row < size; row += 1) {
        const z = originZ + row * spacing;
        for (let column = 0; column < size; column += 1) {
          heights[row * size + column] = terrainHeightAt(terrain, originX + column * spacing, z);
        }
      }
    }
    this.device.queue.writeTexture({ texture: this.causticReceiver }, heights, { bytesPerRow: size * 4, rowsPerImage: size }, { width: size, height: size });
    // A single spacing on both axes, so the shader's mirror of
    // `sampleTerrainGrid` needs no per-axis case. The lattice therefore covers
    // at least the container and overhangs the shorter axis, which costs
    // nothing: the overhang samples ground the map can never address.
    const lanes = terrain && width > 0 && depth > 0
      ? [originX, originZ, spacing, 0, size, size, 0, 0]
      : [0, 0, 1, 0, 0, 0, 0, 0];
    for (let index = 0; index < lanes.length; index += 1) {
      const slot = WATER_SCENE_OPTICS_RECEIVER_FLOAT_OFFSET + index;
      if (this.waterSceneOptics[slot] === lanes[index]) continue;
      this.waterSceneOptics[slot] = lanes[index];
      this.waterSceneOpticsDirty = true;
    }
    this.rebuildBindGroups();
  }

  async initialize(onProgress:(label:string,completed:number,total:number)=>void=()=>{}) {
    const [extract, globalClassify, globalScan, globalEmitAll, prepare, surface, caustic, composite] = await Promise.all([
      checkedModule(this.device, "Water isosurface extraction", surfaceExtractionShader),
      checkedModule(this.device, "Global fine water classification", globalFineSurfaceClassificationShader),
      checkedModule(this.device, "Classified global fine scan", globalFineClassifiedIndirectScanShader),
      checkedModule(this.device, "Classified global fine tetrahedra", globalFineClassifiedEmitShader),
      checkedModule(this.device, "Water extraction dispatch prepare", extractionPrepareShader),
      checkedModule(this.device, "Water interface raster", surfaceRasterShader),
      checkedModule(this.device, "Water caustic projection", causticShader),
      checkedModule(this.device, "Water optical composite", compositeShader)
    ]);
    this.extractLayout = this.device.createBindGroupLayout({ label: "Water extraction bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ,{ binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
      ,{ binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    this.globalExtractLayout = this.device.createBindGroupLayout({ label: "Global fine water classification bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.globalPolygoniseLayout = this.device.createBindGroupLayout({ label: "Global fine water polygonise bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    // Emission consumes the dispatch buffer through INDIRECT, so it must not
    // inherit the scan layout's writable-storage declaration for binding 11.
    // The distinct group is the WebGPU usage-scope barrier between the GPU
    // authored count and the exact-sized indirect launch.
    this.globalPolygoniseEmitLayout = this.device.createBindGroupLayout({ label: "Global fine water emit bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.prepareLayout = this.device.createBindGroupLayout({ label: "Water extraction prepare bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    this.surfaceLayout = this.device.createBindGroupLayout({ label: "Water surface bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
    ] });
    this.surfacePeelLayout = this.device.createBindGroupLayout({ label: "Water rear-interface peel binding", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }
    ] });
    // Binding 14 is the caustic map the projection pass writes and this pass
    // finally reads — the consumer whose absence made the whole caustic path a
    // no-op. 15 and 16 are the scene's own optics and caustic receiver.
    this.compositeLayout = this.device.createBindGroupLayout({ label: "Water composite bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ...[1,2,3,4,5,10,11,12,13,14].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } })),
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      { binding: 15, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }
    ] });
    this.causticLayout = this.device.createBindGroupLayout({ label: "Water caustic projection bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } }
    ] });
    const extractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.extractLayout] });
    const globalExtractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.globalExtractLayout] });
    const total=16;let completed=0;
    const compute=async(label:string,descriptor:GPUComputePipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createComputePipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    const render=async(label:string,descriptor:GPURenderPipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createRenderPipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    this.extractPipeline = await compute("Classifying liquid surface cubes",{ label: "Classify liquid surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractMain" } });
    this.extractBandPipeline = await compute("Classifying restricted water band",{ label: "Classify restricted water band", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractBandMain" } });
    this.extractTallSidesPipeline = await compute("Classifying tall-cell interfaces",{ label: "Classify tall-cell side interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractTallSidesMain" } });
    this.extractWallPipeline = await compute("Classifying water wall interfaces",{ label: "Classify water wall interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractWallMain" } });
    this.extractGlobalFinePipeline = await compute("Classifying global fine surface bricks",{ label: "Classify global fine surface bricks", layout: globalExtractionPipelineLayout, compute: { module: globalClassify, entryPoint: "extractGlobalFineMain" } });
    this.extractGlobalCoarsePipeline = await compute("Classifying compact coarse cells",{ label: "Classify compact coarse fallback", layout: globalExtractionPipelineLayout, compute: { module: globalClassify, entryPoint: "extractGlobalCoarseMain" } });
    this.polygonisePipeline = await compute("Building water surface mesh",{ label: "Polygonise surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain" } });
    const globalPolygonScanLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.globalPolygoniseLayout]});
    const globalPolygonEmitLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.globalPolygoniseEmitLayout]});
    this.polygoniseGlobalFineScanPipeline=await compute("Scanning global fine water mesh",{label:"Scan classified global fine triangles",layout:globalPolygonScanLayout,compute:{module:globalScan,entryPoint:"scanGlobalFineTriangles"}});
    this.polygoniseGlobalFineEmitPipeline=await compute("Emitting six global fine tetrahedra",{label:"Emit classified global fine tetrahedra",layout:globalPolygonEmitLayout,compute:{module:globalEmitAll,entryPoint:"emitGlobalFineTetrahedra"}});
    this.preparePipeline = await compute("Preparing surface dispatch",{ label: "Prepare polygonise dispatch", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareLayout] }), compute: { module: prepare, entryPoint: "prepareMain" } });
    this.polygoniseDispatchBuffer = this.device.createBuffer({ label: "Water polygonise dispatch arguments", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const surfacePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.surfaceLayout, this.surfacePeelLayout] });
    const causticPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.causticLayout] });
    const surfaceDescriptor = (label: string, cullMode: GPUCullMode, coverageExpansionPixels = 0, peel = false): GPURenderPipelineDescriptor => ({
      label, layout: surfacePipelineLayout, vertex: { module: surface, entryPoint: "surfaceVertex", constants: { interfaceCoverageExpansionPixels: coverageExpansionPixels } },
      fragment: { module: surface, entryPoint: "surfaceFragment", constants: { peelBehindFirstExit: peel ? 1 : 0 }, targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", frontFace: "ccw", cullMode },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    this.surfaceFrontPipeline = await render("Rendering front water interfaces",surfaceDescriptor("Raster water front interfaces", WATER_INTERFACE_CULL_MODES.front,1.0));
    this.surfaceBackPipeline = await render("Rendering back water interfaces",surfaceDescriptor("Raster water back interfaces", WATER_INTERFACE_CULL_MODES.back));
    this.surfaceRearFrontPipeline = await render("Peeling rear front water interfaces",surfaceDescriptor("Raster water rear front interfaces", WATER_INTERFACE_CULL_MODES.front,0,true));
    this.surfaceRearBackPipeline = await render("Peeling rear back water interfaces",surfaceDescriptor("Raster water rear back interfaces", WATER_INTERFACE_CULL_MODES.back,0,true));
    this.causticPipeline = await render("Projecting water caustics",{
      label: "Project refracted caustics", layout: causticPipelineLayout, vertex: { module: caustic, entryPoint: "causticVertex" },
      fragment: { module: caustic, entryPoint: "causticFragment", targets: [{ format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    const compositePipelineLayout=this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] });
    const compositeDescriptor:GPURenderPipelineDescriptor={ label:"Composite layered water optics", layout: compositePipelineLayout, vertex: { module: composite, entryPoint: "vertexMain" }, fragment: { module: composite, entryPoint: "fragmentMain", targets: [{ format: this.targetFormat }] }, primitive: { topology: "triangle-list" } };
    this.compositePipeline = await render("Compositing water optics",compositeDescriptor);
    this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.fallbackSparsePageTable = this.device.createBuffer({ label: "Water sparse-page fallback", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fallbackSparseActivePages = this.device.createBuffer({ label: "Water sparse-active fallback", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.fallbackSparsePhi = this.device.createBuffer({ label: "Water sparse-phi fallback", size: 4, usage: GPUBufferUsage.STORAGE });
    this.fallbackSparseParams = this.device.createBuffer({ label: "Water sparse-params fallback", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.globalFineRenderParams = this.device.createBuffer({ label: "Water global fine parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fallbackSparseControl = this.device.createBuffer({ label: "Water disabled storage binding", size: WATER_DISABLED_STORAGE_BYTES, usage: GPUBufferUsage.STORAGE });
    this.waterSceneOpticsBuffer = this.device.createBuffer({ label: "Water scene optics and caustic receiver", size: WATER_SCENE_OPTICS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Allocated unconditionally: the composite declares the receiver whether or
    // not the scene has ground, and a bind group with a missing entry is not a
    // bind group. A container-less scene leaves it zeroed and the uniform's
    // receiver size at zero, which is what `waterReceiverPresent` reads.
    this.causticReceiver = this.device.createTexture({
      label: "Caustic receiver heights", size: [CAUSTIC_MAP_RESOLUTION, CAUSTIC_MAP_RESOLUTION], format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.waterSceneOpticsDirty = true;
    this.receiverKey = "";
    this.flushSceneOptics();
    this.rebuildBindGroups();
  }

  setVolume(texture: GPUTexture, columnBases: GPUTexture) {
    if (this.volume === texture && this.columnBases === columnBases) return;
    this.volume = texture; this.columnBases = columnBases; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  /** Selects row-independent global fine bricks without synthesizing leaf ownership. */
  setGlobalFineLevelSet(source: GlobalFineLevelSetConsumerSource | undefined) {
    if (source) validateGlobalFineLevelSetConsumerSource(source);
    const previous = this.globalFineLevelSet;
    if (previous === source) return;
    const sameBindings = previous && source
      && previous.metadata.buffer === source.metadata.buffer
      && previous.worklist.buffer === source.worklist.buffer
      && previous.samples.buffer === source.samples.buffer
      && previous.coarsePhiDirectory?.buffer === source.coarsePhiDirectory?.buffer
      && previous.coarsePhiRowCapacity === source.coarsePhiRowCapacity
      && previous.topologyControl?.buffer === source.topologyControl?.buffer;
    if (sameBindings && previous.generation === source.generation) return;
    this.globalFineLevelSet = source;
    this.writeCompactRenderParams();
    this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false;
    // Keep same-shaped geometry alive across A/B source publication. The next
    // encode still calls ensureGeometry(), so a genuine dimension change
    // reallocates; clearing the key here destroyed A before B could prove its
    // tags and defeated the fail-closed retained-mesh contract.
    if (!sameBindings) this.rebuildBindGroups();
  }

  /** Selects the moving compact-octree surface without enabling a fine band. */
  setCoarseLevelSet(source: CoarseLevelSetConsumerSource | undefined) {
    if (source) validateCoarseLevelSetConsumerSource(source);
    const previous = this.coarseLevelSet;
    if (previous === source) return;
    const sameBindings = previous && source
      && previous.directory.buffer === source.directory.buffer
      && previous.control.buffer === source.control.buffer
      && previous.rowCapacity === source.rowCapacity;
    if (sameBindings && previous.generation === source.generation) return;
    this.coarseLevelSet = source;
    this.writeCompactRenderParams();
    this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false;
    // Factor-one coarse publication rewrites stable directory/control arenas.
    // A new generation changes only the uniform and extraction invalidation;
    // rebuilding every water bind group here added nine host allocations to
    // every presented frame without changing any bound resource identity.
    if (!sameBindings) this.rebuildBindGroups();
  }

  private writeCompactRenderParams() {
    if (!this.globalFineRenderParams) return;
    const fine = this.globalFineLevelSet;
    const coarse = this.coarseLevelSet;
    const source = fine ?? coarse;
    if (!source) return;
    const bytes = new ArrayBuffer(112); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    if (fine) {
      u32.set([...fine.sampleDimensions, fine.brickResolution], 0);
      u32.set([...fine.brickDimensions, fine.samplesPerBrick], 4);
      u32.set([fine.pageCapacity, 7, fine.pageCapacity, fine.generation], 8);
      f32.set([...fine.domainOrigin, fine.fineCellWidth], 12); f32[16] = fine.fineFactor;
    } else {
      const dimensions = coarse!.sampleDimensions;
      u32.set([...dimensions, 4], 0);
      u32.set(dimensions.map((value) => Math.ceil(value / 4)).concat(64), 4);
      // Table state 6 is the renderer-private compact-coarse publication mode.
      u32.set([1, 6, 1, coarse!.generation], 8);
      f32.set([...coarse!.domainOrigin, coarse!.physicalCellSize], 12); f32[16] = 1;
    }
    this.device.queue.writeBuffer(this.globalFineRenderParams, 0, bytes);
  }

  setSecondaryParticles(pipeline: SecondaryParticleRenderPipeline | undefined) {
    this.secondaryParticles = pipeline;
  }

  diagnosticCaptureTexture(stageKey: string) {
    const texture = stageKey === "interfaces" ? this.frontNormal
      : stageKey === "interface-positions" ? this.frontPosition
        : stageKey === "back-interface-positions" ? this.backPosition
      : this.sceneTexture;
    return texture ? { texture, dimensions: [texture.width, texture.height, 1] as [number, number, number] } : undefined;
  }

  /** Smoke-only source for an exact unordered symmetry audit of the emitted mesh. */
  diagnosticSurfaceVertexSource() {
    return this.vertexBuffer && this.activeCubeBuffer && this.globalCubeOffsets
      ? { buffer: this.vertexBuffer, strideBytes: 32,
        classifiedCubes: this.activeCubeBuffer, classifiedOffsets: this.globalCubeOffsets }
      : undefined;
  }

  /** Latest bounded GPU readback proving what surface geometry is presented. */
  get surfaceRenderDiagnostics() { return this.lastSurfaceDiagnostics; }

  /**
   * The dry-scene HDR plane the compositor consumes: RGB is scene-linear
   * radiance and alpha is the linear depth water and spray sort against.
   * Exposed read-only so the render-stage overlay can present the lighting
   * result before compositing, without a copy.
   */
  get drySceneRadianceView(): GPUTextureView | undefined { return this.sceneTextureView; }

  /**
   * Full-rate surface receipts are a Dawn-session tool, not a UI mode.
   *
   * Opening the diagnostics or visual panel used to escalate this to every
   * frame, and a full-rate receipt is not a cheap one: `completeSurfaceDiagnostics`
   * awaits a QUEUE-WIDE `onSubmittedWorkDone` and then maps, so the browser's
   * render path became fully synchronous — per frame — for a panel that reads
   * the value at human rates. The 250 ms cadence now holds in the browser
   * regardless of which panel is open; `FLUID_WATER_DIAGNOSTICS=1` still buys
   * per-capture evidence where a harness genuinely needs it.
   */
  private surfaceDiagnosticsFullRateRequested() {
    return typeof process !== "undefined" && process.env?.FLUID_WATER_DIAGNOSTICS === "1";
  }

  private encodeSurfaceDiagnostics(encoder: GPUCommandEncoder, force = false): boolean {
    if (this.surfaceDiagnosticPending || !this.indirectBuffer) return false;
    const now_ms = performance.now();
    // The normal UI needs failure evidence, not a frame-rate-synchronous
    // telemetry stream. Match the solver's bounded 250 ms readback cadence;
    // explicit diagnostic/Dawn sessions retain per-capture evidence.
    if (!force && !this.surfaceDiagnosticsFullRateRequested()
      && now_ms - this.lastSurfaceDiagnosticEncodeAt_ms < 250) return false;
    this.lastSurfaceDiagnosticEncodeAt_ms = now_ms;
    this.surfaceDiagnosticReadback?.destroy();
    this.surfaceDiagnosticReadback = this.device.createBuffer({ label: "Water render diagnostics", size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.indirectBuffer, 0, this.surfaceDiagnosticReadback, 0, 32);
    this.pendingSurfaceDiagnosticGlobalFine = Boolean(this.globalFineLevelSet);
    this.pendingSurfaceDiagnosticCoarse = Boolean(this.coarseLevelSet);
    this.pendingSurfaceDiagnosticGlobalFineGeneration = this.globalFineLevelSet?.generation;
    this.surfaceDiagnosticPending = true;
    return true;
  }

  /** Called immediately after the frame submission that contains the copies. */
  completeSurfaceDiagnostics(submissionCompletion?: Promise<void>): Promise<WaterRenderDiagnostics | undefined> {
    if (this.surfaceDiagnosticCompletion) return this.surfaceDiagnosticCompletion;
    const readback = this.surfaceDiagnosticReadback;
    if (!readback || !this.surfaceDiagnosticPending) return Promise.resolve(undefined);
    const completion = (submissionCompletion ?? this.device.queue.onSubmittedWorkDone()).then(async () => {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const globalFineAttached = this.pendingSurfaceDiagnosticGlobalFine;
      const coarseAttached = this.pendingSurfaceDiagnosticCoarse;
      const globalFineAttachedGeneration = this.pendingSurfaceDiagnosticGlobalFineGeneration;
      const surfaceGeometrySource = waterSurfaceGeometrySource(
        globalFineAttached, words[0], words[6], coarseAttached,
      );
      const meshPublicationGeneration = (surfaceGeometrySource === "global-fine-coarse"
        || surfaceGeometrySource === "compact-coarse"
        || surfaceGeometrySource === "retained-previous") && words[7] !== 0xffff_ffff
        ? words[7] : undefined;
      this.lastSurfaceDiagnostics = {
        vertexCount: words[0], activeCubeCount: words[4], vertexAllocator: words[5],
        globalFineAuthorityLatch: words[6],
        surfaceGeometrySource,
        globalFineAttached,
        globalFineAttachedGeneration,
        meshPublicationGeneration,
        globalFineCrossingPublished: surfaceGeometrySource === "global-fine-coarse",
        presentationFallbackActive: surfaceGeometrySource === "retained-previous",
        sourceFrameCounts: { ...this.surfaceSourceFrameCounts },
      };
      console.info("Water render diagnostics", JSON.stringify(this.lastSurfaceDiagnostics));
      const result = this.lastSurfaceDiagnostics;
      readback.unmap();
      return result;
    }).catch(() => undefined).finally(() => {
      this.surfaceDiagnosticPending = false;
      if (this.surfaceDiagnosticCompletion === completion) this.surfaceDiagnosticCompletion = undefined;
    });
    this.surfaceDiagnosticCompletion = completion;
    return completion;
  }

  /**
   * Word 0..7 is the compact-surface reset applied at byte 4; word 8..15 is
   * the dense reset applied at byte 0. Written once, copied every frame.
   *
   * Created on demand rather than only alongside the geometry allocation: a
   * reset template that can be missing would be a new way for `encode` to fail
   * closed, and `encode` failing closed is indistinguishable from a solver that
   * never published — it stalls the fenced t=0 raster handoff with no error.
   */
  private createIndirectResetTemplate(): GPUBuffer {
    const template = this.device.createBuffer({ label: "Water indirect header reset patterns", size: 64, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(template, 0, new Uint32Array([
      1, 0, 0, 0, 0xffff_ffff, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0xffff_ffff,
    ]));
    return template;
  }

  private ensureGeometry(nx: number, ny: number, nz: number) {
    const key = `${nx}x${ny}x${nz}`;
    if (key === this.geometryKey) return;
    this.vertexBuffer?.destroy(); this.indirectBuffer?.destroy(); this.indirectResetTemplate?.destroy(); this.activeCubeBuffer?.destroy(); this.globalCubeValues?.destroy(); this.globalCubeOffsets?.destroy();
    // Surface area, not volume, controls the normal case.  The generous factor
    // also covers breaking sheets and entrained blobs while imposing a hard
    // 64 MiB ceiling on adversarial checkerboard fields.
    const maxVertices = surfaceVertexCapacity(nx, ny, nz);
    this.vertexBuffer = this.device.createBuffer({ label: `Extracted water surface (${maxVertices} vertices)`, size: maxVertices * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    // The first 16 bytes are the standard draw-indirect ABI. Renderer-private
    // counters, global-fine authority latch, and GPU-published mesh generation
    // trail it; firstInstance must stay zero unless the optional
    // indirect-first-instance feature is enabled.
    this.indirectBuffer = this.device.createBuffer({ label: "Water indirect draw arguments and extraction counters", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(this.indirectBuffer, 28, new Uint32Array([0xffff_ffff]));
    this.indirectResetTemplate = this.createIndirectResetTemplate();
    this.activeCubeBuffer = this.device.createBuffer({ label: "Water surface cube worklist", size: activeCubeCapacity(maxVertices) * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.globalCubeValues = this.device.createBuffer({ label: "Global fine classified cube values", size: activeCubeCapacity(maxVertices) * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.globalCubeOffsets = this.device.createBuffer({ label: "Global fine tetrahedron offsets", size: activeCubeCapacity(maxVertices) * 6 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.geometryKey = key; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  ensureSize(width: number, height: number) {
    const key = `${width}x${height}`;
    if (key === this.targetKey) return;
    for (const texture of [this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth]) texture?.destroy();
    const sampledTarget = (label: string) => this.device.createTexture({ label, size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.sceneTexture = this.device.createTexture({ label: "Dry scene HDR", size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }); this.sceneTextureView = this.sceneTexture.createView(); this.frontPosition = sampledTarget("Water front positions"); this.frontNormal = sampledTarget("Water front normals"); this.backPosition = sampledTarget("Water back positions"); this.backNormal = sampledTarget("Water back normals"); this.rearFrontPosition = sampledTarget("Water rear front positions"); this.rearFrontNormal = sampledTarget("Water rear front normals"); this.rearBackPosition = sampledTarget("Water rear back positions"); this.rearBackNormal = sampledTarget("Water rear back normals");
    const depth = (label: string) => this.device.createTexture({ label, size: [width,height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.frontDepth = depth("Water front depth"); this.backDepth = depth("Water back depth"); this.rearFrontDepth = depth("Water rear front depth"); this.rearBackDepth = depth("Water rear back depth");
    this.causticTexture?.destroy(); this.causticTexture = this.device.createTexture({ label: "Refracted floor caustics", size: [384,384], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.causticsValid = false;
    this.dryInterfaceClearsEncoded = false;
    this.clearBackgroundEncoded = false;
    this.targetKey = key; this.rebuildBindGroups();
  }

  /**
   * Scene-level fact from the runtime plan. A fluid-less scene has no water
   * geometry to draw, so the per-frame front/back interface passes reduce to
   * their clears; those are encoded once and the passes skipped thereafter.
   */
  setSceneHasFluid(hasFluid: boolean) {
    if (this.sceneHasFluid === hasFluid) return;
    this.sceneHasFluid = hasFluid;
    this.dryInterfaceClearsEncoded = false;
  }

  /**
   * Frame-graph stages this pipeline must not encode. See
   * `render-stage-switches`; only `surface-extraction`, `water-interfaces`,
   * `caustics` and `optical-composite` are its to honour.
   *
   * Re-enabling has to re-run the once-only clears and re-project the caustic
   * map: both are retained precisely because nothing invalidated them, and a
   * withheld frame is an invalidation.
   */
  setDisabledStages(disabled: DisabledRenderStages) {
    if (disabledRenderStagesEqual(this.disabledStages, disabled)) return;
    this.disabledStages = new Set(disabled);
    this.dryInterfaceClearsEncoded = false;
    this.causticsValid = false;
    this.clearBackgroundEncoded = false;
  }

  private rebuildBindGroups() {
    this.compositeBindGroups = new WeakMap();
    const globalFine = this.globalFineLevelSet;
    const coarse = this.coarseLevelSet;
    const coarseDirectory = globalFine?.coarsePhiDirectory ?? coarse?.directory;
    if (this.extractLayout && this.volume && this.columnBases && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.fallbackSparsePageTable && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseParams && this.fallbackSparseControl) this.extractBindGroup = this.device.createBindGroup({ layout: this.extractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 2, resource: this.columnBases.createView() }, { binding: 3, resource: { buffer: this.vertexBuffer } }, { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 7, resource: { buffer: this.fallbackSparsePageTable } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.samples ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: globalFine ? { buffer: this.globalFineRenderParams! } : { buffer: this.fallbackSparseParams } },
      { binding: 11, resource: globalFine?.samples ?? { buffer: this.fallbackSparseControl } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } }
    ] });
    if (this.globalExtractLayout && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalFineRenderParams && this.fallbackSparsePageTable && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseControl) this.globalExtractBindGroup = this.device.createBindGroup({ layout: this.globalExtractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 6, resource: { buffer: this.globalCubeValues } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.samples ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: { buffer: this.globalFineRenderParams } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } },
      { binding: 16, resource: coarseDirectory ?? { buffer: this.fallbackSparseControl } },
      { binding: 17, resource: globalFine?.topologyControl ?? { buffer: this.fallbackSparseControl } },
    ] });
    if (this.globalPolygoniseLayout && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalCubeOffsets && this.polygoniseDispatchBuffer && this.globalFineRenderParams && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseControl) this.globalPolygoniseBindGroup = this.device.createBindGroup({ layout: this.globalPolygoniseLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 3, resource: { buffer: this.vertexBuffer } },
      { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 6, resource: { buffer: this.globalCubeValues } }, { binding: 7, resource: { buffer: this.globalCubeOffsets } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.samples ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: { buffer: this.globalFineRenderParams } },
      { binding: 11, resource: { buffer: this.polygoniseDispatchBuffer } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } },
      { binding: 16, resource: coarseDirectory ?? { buffer: this.fallbackSparseControl } },
    ] });
    if (this.globalPolygoniseEmitLayout && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalCubeOffsets && this.globalFineRenderParams && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseControl) this.globalPolygoniseEmitBindGroup = this.device.createBindGroup({ layout: this.globalPolygoniseEmitLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 3, resource: { buffer: this.vertexBuffer } },
      { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 6, resource: { buffer: this.globalCubeValues } }, { binding: 7, resource: { buffer: this.globalCubeOffsets } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.samples ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: { buffer: this.globalFineRenderParams } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } },
      { binding: 16, resource: coarseDirectory ?? { buffer: this.fallbackSparseControl } },
    ] });
    if (this.prepareLayout && this.indirectBuffer && this.activeCubeBuffer && this.polygoniseDispatchBuffer) this.prepareBindGroup = this.device.createBindGroup({ layout: this.prepareLayout, entries: [
      { binding: 0, resource: { buffer: this.indirectBuffer } }, { binding: 1, resource: { buffer: this.activeCubeBuffer } }, { binding: 2, resource: { buffer: this.polygoniseDispatchBuffer } }
    ] });
    if (this.surfaceLayout && this.vertexBuffer) this.surfaceBindGroup = this.device.createBindGroup({ layout: this.surfaceLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.vertexBuffer } }] });
    if (this.causticLayout && this.vertexBuffer && this.waterSceneOpticsBuffer && this.causticReceiver) this.causticBindGroup = this.device.createBindGroup({ label: "Water caustic projection inputs", layout: this.causticLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.vertexBuffer } },
      { binding: 2, resource: { buffer: this.waterSceneOpticsBuffer } }, { binding: 3, resource: this.causticReceiver.createView() }
    ] });
    if (this.surfacePeelLayout && this.sceneTextureView) this.surfaceUnpeeledBindGroup = this.device.createBindGroup({ label: "Water unpeeled placeholder binding", layout: this.surfacePeelLayout, entries: [{ binding: 0, resource: this.sceneTextureView }] });
    if (this.surfacePeelLayout && this.backPosition) this.surfacePeelBindGroup = this.device.createBindGroup({ layout: this.surfacePeelLayout, entries: [{ binding: 0, resource: this.backPosition.createView() }] });
    this.compositeBindGroup = this.sceneTextureView ? this.compositeBindGroupFor(this.sceneTextureView) : undefined;
  }

  private compositeBindGroupFor(sceneView: GPUTextureView): GPUBindGroup | undefined {
    const cached = this.compositeBindGroups.get(sceneView);
    if (cached) return cached;
    if (!this.compositeLayout || !this.frontPosition || !this.frontNormal || !this.backPosition || !this.backNormal || !this.rearFrontPosition || !this.rearFrontNormal || !this.rearBackPosition || !this.rearBackNormal || !this.sampler || !this.volume || !this.columnBases || !this.causticTexture || !this.waterSceneOpticsBuffer || !this.causticReceiver) return undefined;
    const bindGroup = this.device.createBindGroup({ layout: this.compositeLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: sceneView }, { binding: 2, resource: this.frontPosition.createView() }, { binding: 3, resource: this.frontNormal.createView() }, { binding: 4, resource: this.backPosition.createView() }, { binding: 5, resource: this.backNormal.createView() }, { binding: 6, resource: this.sampler }, { binding: 7, resource: { buffer: this.bodyBuffer } }, { binding: 8, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 9, resource: this.columnBases.createView() }, { binding: 10, resource: this.rearFrontPosition.createView() }, { binding: 11, resource: this.rearFrontNormal.createView() }, { binding: 12, resource: this.rearBackPosition.createView() }, { binding: 13, resource: this.rearBackNormal.createView() }, { binding: 14, resource: this.causticTexture.createView() }, { binding: 15, resource: { buffer: this.waterSceneOpticsBuffer } }, { binding: 16, resource: this.causticReceiver.createView() }
    ] });
    this.compositeBindGroups.set(sceneView, bindGroup);
    return bindGroup;
  }

  encode(encoder: GPUCommandEncoder, output: GPUTexture | GPUTextureView, nx: number, ny: number, nz: number, restrictedTallCell: boolean, maximumNeighborDelta: number, revision: number, drySceneReplacement?: DrySceneReplacementEncoder, traceBoundary?: () => void, tracePhase?: RenderPathTracePhase, forceSurfaceDiagnostics = false, backgroundMode: RasterWaterBackgroundMode = "require-dry-scene", allowSurfaceDiagnostics = true, bandPartitioner?: FrameBandPartitioner): RasterWaterEncodeResult | false {
    // Count only frames whose source has a completed GPU receipt. The
    // diagnostics/visual panels request full-rate receipts, making this an
    // exact source-mode counter while it is being used to judge fidelity.
    if (this.lastSurfaceDiagnostics) {
      this.surfaceSourceFrameCounts[this.lastSurfaceDiagnostics.surfaceGeometrySource] += 1;
    }
    const compactSurface = this.globalFineLevelSet ?? this.coarseLevelSet;
    const geometryDimensions = compactSurface?.sampleDimensions ?? [nx, ny, nz] as const;
    this.ensureGeometry(geometryDimensions[0],geometryDimensions[1],geometryDimensions[2]);
    if (!this.extractPipeline||!this.extractBandPipeline||!this.extractTallSidesPipeline||!this.extractWallPipeline||!this.extractGlobalFinePipeline||!this.extractGlobalCoarsePipeline||!this.preparePipeline||!this.polygonisePipeline||!this.polygoniseGlobalFineScanPipeline||!this.polygoniseGlobalFineEmitPipeline||!this.surfaceFrontPipeline||!this.surfaceBackPipeline||!this.surfaceRearFrontPipeline||!this.surfaceRearBackPipeline||!this.causticPipeline||!this.compositePipeline||!this.extractBindGroup||!this.globalExtractBindGroup||!this.globalPolygoniseBindGroup||!this.globalPolygoniseEmitBindGroup||!this.prepareBindGroup||!this.surfaceBindGroup||!this.causticBindGroup||!this.surfaceUnpeeledBindGroup||!this.surfacePeelBindGroup||!this.compositeBindGroup||!this.indirectBuffer||!this.polygoniseDispatchBuffer||!this.volume||!this.sceneTexture||!this.frontPosition||!this.frontNormal||!this.frontDepth||!this.backPosition||!this.backNormal||!this.backDepth||!this.rearFrontPosition||!this.rearFrontNormal||!this.rearFrontDepth||!this.rearBackPosition||!this.rearBackNormal||!this.rearBackDepth||!this.causticTexture||!this.causticReceiver) return false;
    const now_ms = performance.now();
    // A paused t=0 handoff cannot wait for a new solver revision: reset has
    // already made the current revision the only one that will be presented.
    // Retry extraction until its own diagnostic copy is admitted. This also
    // bypasses the ordinary 250 ms telemetry throttle, but never overwrites a
    // readback still owned by an earlier submission.
    // The extraction chain has its own switch: it is the largest compute block
    // in a wet frame and interface drawing is a separate question. Withheld,
    // the retained mesh keeps drawing (the interfaces read the last extraction)
    // so the delta is classify + scan + emit and nothing downstream. The t=0
    // handoff's forced capture overrides the withhold — a startup gate that can
    // never satisfy its own admission condition would stall the presentation
    // forever behind a diagnostic switch.
    const updateSurface = forceSurfaceDiagnostics
      || (!this.disabledStages.has("surface-extraction")
        && shouldUpdateWaterSurface(this.extractedRevision, revision, this.lastExtractionAt_ms, now_ms));
    let surfaceDiagnosticsCaptured = false;
    // The map follows the mesh: a retained surface deposits the same bundles,
    // so re-projecting it would spend a full pass to write the same texels. A
    // scene that authors zero caustic strength never encodes the pass at all.
    const updateCaustics = this.causticStrength > 0 && !this.disabledStages.has("caustics")
      && (updateSurface || !this.causticsValid);
    if (updateSurface) {
      const indirectReset = this.indirectResetTemplate ??= this.createIndirectResetTemplate();
      if (compactSurface) {
        // Preserve the last published draw count while the GPU validates the
        // next A/B generation. Classification clears the sentinel only after
        // observing finite tagged fine data or a published compact-coarse
        // fallback; an invalid generation therefore retains the previous mesh
        // and its GPU-written publication generation in word 7.
        encoder.copyBufferToBuffer(indirectReset,0,this.indirectBuffer,4,24);
      } else {
        encoder.copyBufferToBuffer(indirectReset,32,this.indirectBuffer,0,32);
      }
      const plan = surfaceExtractionDispatchPlan(nx, ny, nz, this.volume.depthOrArrayLayers, restrictedTallCell, maximumNeighborDelta);
      // Classify appends surface-crossing cubes to the worklist, the prepare
      // kernel sizes the indirect dispatch, and polygonise emits triangles for
      // just those cubes. The writable prepare binding and its later INDIRECT
      // use must occupy distinct WebGPU usage scopes.
      let compute=encoder.beginComputePass({label:"Extract water isosurface"});compute.setBindGroup(0,this.extractBindGroup);
      const prepareAndPolygonise=(pipeline:GPUComputePipeline,group:GPUBindGroup)=>{
        compute.setPipeline(this.preparePipeline!);compute.setBindGroup(0,this.prepareBindGroup!);compute.dispatchWorkgroups(1);
        compute.end();
        compute=encoder.beginComputePass({label:"Polygonise water isosurface"});
        compute.setPipeline(pipeline);compute.setBindGroup(0,group);compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer!,0);
      };
      const globalFine = this.globalFineLevelSet;
      const coarse = this.coarseLevelSet;
      if (globalFine || coarse) {
        compute.setBindGroup(0, this.globalExtractBindGroup);
        if (globalFine) {
          compute.setPipeline(this.extractGlobalFinePipeline);
          compute.dispatchWorkgroups(...globalFineSurfaceDispatch(globalFine.pageCapacity, globalFine.samplesPerBrick));
        }
        if(globalFine?.coarsePhiRowCapacity){compute.setPipeline(this.extractGlobalCoarsePipeline);compute.dispatchWorkgroups(...globalFineCoarseSurfaceDispatch(globalFine.coarsePhiRowCapacity));}
        else if(coarse){compute.setPipeline(this.extractGlobalCoarsePipeline);compute.dispatchWorkgroups(...compactCoarseSurfaceDispatch(coarse.sampleDimensions));}
        compute.end();
        compute=encoder.beginComputePass({label:"Scan classified global fine surface"});
        compute.setBindGroup(0,this.globalPolygoniseBindGroup);
        compute.setPipeline(this.polygoniseGlobalFineScanPipeline);compute.dispatchWorkgroups(1);
        compute.end();
        compute=encoder.beginComputePass({label:"Emit classified global fine surface"});
        compute.setBindGroup(0,this.globalPolygoniseEmitBindGroup);
        compute.setPipeline(this.polygoniseGlobalFineEmitPipeline);compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer,0);
        compute.end();
      } else {
        if (plan.mode === "restricted-band") {
          compute.setPipeline(this.extractBandPipeline); compute.dispatchWorkgroups(...plan.band!);
          compute.setPipeline(this.extractTallSidesPipeline); compute.dispatchWorkgroups(...plan.tallSides!);
          compute.setPipeline(this.extractWallPipeline); compute.dispatchWorkgroups(...plan.walls!);
        } else {
          compute.setPipeline(this.extractPipeline); compute.dispatchWorkgroups(...plan.full!);
        }
        prepareAndPolygonise(this.polygonisePipeline,this.extractBindGroup);
        compute.end();
      }
      this.surfaceDiagnosticsDirty = true;
      this.extractedRevision = revision; this.lastExtractionAt_ms = advancePresentationClock(this.lastExtractionAt_ms, now_ms);
      tracePhase?.({ id: "surface-extraction", label: "Water surface extraction" });
    }
    // Diagnostics have their own bounded cadence. Retry the copy independently
    // of surface extraction: the solver revision may remain unchanged forever
    // after a paused/manual step, while the most recent extraction still needs
    // to displace a throttled generation-transition receipt.
    if (allowSurfaceDiagnostics && this.surfaceDiagnosticsDirty) {
      surfaceDiagnosticsCaptured = this.encodeSurfaceDiagnostics(encoder, forceSurfaceDiagnostics);
      if (surfaceDiagnosticsCaptured) this.surfaceDiagnosticsDirty = false;
    }
    if (updateCaustics) {
      const caustic=encoder.beginRenderPass({label:"Water caustics",colorAttachments:[{view:this.causticTexture.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});caustic.setPipeline(this.causticPipeline);caustic.setBindGroup(0,this.causticBindGroup);caustic.drawIndirect(this.indirectBuffer,0);caustic.end();
      this.causticsValid = true;
      tracePhase?.({ id: "water-caustics", label: "Water caustic map" });
    }
    traceBoundary?.();
    // Everything above — extraction, diagnostics, caustics — is the
    // water-surface band; the dry-scene replacement crosses its own internal
    // boundaries, so this scope's encoder must resynchronize afterwards.
    if (bandPartitioner) encoder = bandPartitioner.boundary("water-surface");
    const sparseSceneResult = drySceneReplacement?.(encoder, this.sceneTexture, tracePhase) ?? false;
    if (bandPartitioner) encoder = bandPartitioner.current;
    if (sparseSceneResult) {
      // A later switch to fluid-only must clear imagery left by this frame.
      this.clearBackgroundEncoded = false;
    } else if (backgroundMode === "clear") {
      // No geometry, shader, draw, or recurring full-frame write. The clear is
      // retained until a dry-scene encoder writes the attachment or it resizes.
      if (!this.clearBackgroundEncoded) {
        encoder.beginRenderPass({label:"Fluid-only clear background",colorAttachments:[{
          view:this.sceneTextureView!,clearValue:{r:.01,g:.025,b:.024,a:65504},loadOp:"clear",storeOp:"store"
        }]}).end();
        this.clearBackgroundEncoded = true;
        tracePhase?.({ id: "dry-scene", label: "Fluid-only background clear" });
      }
    } else {
      this.clearBackgroundEncoded = false;
      // The live sparse scene is the only dry-scene authority. Its absence is
      // intentionally visible and contains no scene-like substitute. Alpha is
      // far depth so the independently authoritative water interfaces remain
      // visible while the missing dry scene still fails closed unmistakably.
      encoder.beginRenderPass({label:"SVO dry-scene unavailable",colorAttachments:[{
        view:this.sceneTextureView!,clearValue:{r:.18,g:0,b:.045,a:65504},loadOp:"clear",storeOp:"store"
      }]}).end();
      tracePhase?.({ id: "dry-scene", label: "SVO dry-scene unavailable · fail closed" });
    }
    traceBoundary?.();
    // Water and spray target the same interface attachments and depth state.
    // Encode both draws in one pass per side so spray does not force two extra
    // full-resolution attachment load/store cycles.
    const interfacePass=(label:string,pipeline:GPURenderPipeline,position:GPUTexture,normal:GPUTexture,depth:GPUTexture,side:"front"|"back",particles=true,peel=false)=>{const pass=encoder.beginRenderPass({label,colorAttachments:[{view:position.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"},{view:normal.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});pass.setPipeline(pipeline);pass.setBindGroup(0,this.surfaceBindGroup!);pass.setBindGroup(1,peel?this.surfacePeelBindGroup!:this.surfaceUnpeeledBindGroup!);pass.drawIndirect(this.indirectBuffer!,0);if(particles){this.secondaryParticles?.encodeOpticalInterface(pass,side);}pass.end();};
    // Withheld, a fluid scene takes the same once-only clears a dry one does:
    // the compositor reads these attachments unconditionally, so leaving them
    // holding the last drawn interface would keep compositing water that this
    // frame did not extract.
    const interfacesWithheld = this.disabledStages.has("water-interfaces");
    if (this.sceneHasFluid && !interfacesWithheld) {
      interfacePass("Water + spray front interfaces",this.surfaceFrontPipeline,this.frontPosition,this.frontNormal,this.frontDepth,"front");
      tracePhase?.({ id: "water-front-interface", label: "Water + spray front interface" });
      interfacePass("Water + spray back interfaces",this.surfaceBackPipeline,this.backPosition,this.backNormal,this.backDepth,"back");
      tracePhase?.({ id: "water-back-interface", label: "Water + spray back interface" });
      interfacePass("Water rear front interfaces",this.surfaceRearFrontPipeline,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,"front",false,true);
      interfacePass("Water rear back interfaces",this.surfaceRearBackPipeline,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth,"back",false,true);
      // Their own seam: without it these two peeled passes were charged to the
      // optical composite, the next label to close.
      tracePhase?.({ id: "water-interfaces", label: "Water rear interfaces" });
    } else if (!this.dryInterfaceClearsEncoded) {
      // A fluid-less scene draws no interface geometry. Clear once so the
      // compositor's no-interface input cannot retain a preceding fluid scene.
      const clearPass=(label:string,position:GPUTexture,normal:GPUTexture,depth:GPUTexture)=>{encoder.beginRenderPass({label,colorAttachments:[{view:position.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"},{view:normal.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}}).end();};
      clearPass("Fluid-less scene front interface clear",this.frontPosition,this.frontNormal,this.frontDepth);
      tracePhase?.({ id: "water-front-interface", label: "Fluid-less scene front interface clear" });
      clearPass("Fluid-less scene back interface clear",this.backPosition,this.backNormal,this.backDepth);
      tracePhase?.({ id: "water-back-interface", label: "Fluid-less scene back interface clear" });
      clearPass("Fluid-less scene rear front interface clear",this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth);
      clearPass("Fluid-less scene rear back interface clear",this.rearBackPosition,this.rearBackNormal,this.rearBackDepth);
      tracePhase?.({ id: "water-interfaces", label: "Fluid-less scene rear interface clears" });
      this.dryInterfaceClearsEncoded = true;
    }
    traceBoundary?.();
    const compositeBindGroup = sparseSceneResult ? this.compositeBindGroupFor(sparseSceneResult.sampledTargetView) : this.compositeBindGroup;
    if (!compositeBindGroup) return false;
    const outputView="createView" in output?output.createView():output;const composite=encoder.beginRenderPass({label:"Layered water optical composite",colorAttachments:[{view:outputView,clearValue:{r:.01,g:.025,b:.024,a:1},loadOp:"clear",storeOp:"store"}]});
    // The single final transform every render path shares: withholding it keeps
    // the clear so the presentation target is the background colour rather than
    // a stale composite that would look like the pass still ran.
    if (!this.disabledStages.has("optical-composite")) {
      composite.setPipeline(this.compositePipeline);composite.setBindGroup(0,compositeBindGroup);composite.draw(3);
    }
    composite.end();tracePhase?.({ id: "optical-composite", label: "Layered optical composite" });traceBoundary?.();return { surfaceUpdated: updateSurface, surfaceDiagnosticsCaptured };
  }

  destroy() {
    for (const resource of [this.vertexBuffer,this.indirectBuffer,this.activeCubeBuffer,this.globalCubeValues,this.globalCubeOffsets,this.polygoniseDispatchBuffer,this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth,this.causticTexture,this.causticReceiver,this.waterSceneOpticsBuffer,this.fallbackSparsePageTable,this.fallbackSparseActivePages,this.fallbackSparsePhi,this.fallbackSparseParams,this.globalFineRenderParams,this.fallbackSparseControl,this.surfaceDiagnosticReadback]) { try { resource?.destroy(); } catch { /* device loss */ } }
  }
}
