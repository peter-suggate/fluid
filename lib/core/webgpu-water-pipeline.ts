import { uniformMixedPresentationWGSL } from "../methods/uniform/uniform-mixed-presentation.wgsl";
import type { FluidSurfaceRenderMode } from "../features/surface-display/definition";
import { environmentShaderLibrary } from "./webgpu-environments";
import { advancePresentationClock, frameInterval_ms } from "./frame-pacing";
import type { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import type { GPUSolverInstance } from "./method-contract";
import {
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
import { fluidOnlyRigidSceneShader, rigidBodyRaymarchShaderLibrary } from "./webgpu-rigid-raymarch";
import { cameraApertureShaderLibrary } from "./webgpu-camera";
import {
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "./octree-power-coarse-levelset-sample-abi";
import {
  validateCoarseLevelSetConsumerSource,
  validateGlobalFineLevelSetConsumerSource,
  type GlobalFineLevelSetConsumerSource,
} from "./octree-consumer-sampling";
import type { CoarseLevelSetConsumerSource, DenseLevelSetVolumeConsumerSource } from "./levelset-consumer-abi";
import {
  GLOBAL_FINE_SURFACE_EMIT_LANES,
  globalFineClassifiedEmitShader,
} from "./webgpu-water-global-fine-tetra";
import { parallelSurfaceScanShader, surfaceClassifyDispatchShader, SURFACE_SCAN_BLOCK_SIZE } from "./webgpu-water-surface-scan";
import { globalFineSurfaceClassificationShader } from "./webgpu-water-global-fine-classify";
import { marchingCubesLookupWGSL } from "./marching-cubes-lookup.wgsl";
import type { RenderFrameSeam } from "./render-frame-stages";
import type { FrameBandPartitioner } from "./webgpu-frame-band-sampler";
import {
  disabledRenderStagesEqual,
  NO_DISABLED_RENDER_STAGES,
  type DisabledRenderStages,
} from "../svo/pipeline/render-stage-switches";

type FluidDomain = NonNullable<GPUSolverInstance["fluidDomain"]>;

/**
 * Rasterized water presentation for the WebGPU renderer.
 *
 * The solver already owns the liquid volume.  This pipeline keeps that data on
 * the GPU and turns its 0.5 isosurface into triangles every frame.  The result
 * is rendered as a nearest front/back interval plus one depth-peeled interval,
 * which preserves transmission through folded sheets without scanning the
 * volume once per screen pixel.
 */

export function shouldUpdateWaterSurface(
  extractedRevision: number,
  latestRevision: number,
  lastExtractionAt_ms: number,
  now_ms: number,
  revisionCadenceBypass = false,
) {
  return extractedRevision < 0
    || (latestRevision !== extractedRevision
      && (revisionCadenceBypass
        || now_ms - lastExtractionAt_ms + 0.5 >= frameInterval_ms()));
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

/** The rigid buffer this pipeline is handed holds twelve records; see `bodyBuffer`. */
export const RASTER_WATER_MAXIMUM_BODIES = 12;

/**
 * The fluid-only backdrop.
 *
 * Linear counterpart of the studio's own ground (`--bg`, warmed): the viewport
 * is most of the window, so a teal backdrop here made the whole app read cold
 * no matter what the chrome around it did. Alpha is far depth, the dry-scene
 * attachment's convention (`resolvedDrySceneDepth`).
 */
const FLUID_ONLY_BACKGROUND: GPUColor = Object.freeze({ r: .0194, g: .0145, b: .0097, a: 65504 });

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

/**
 * How this pipeline closes its own seams. Typed to the water pipeline's stages,
 * so it can neither misspell one nor close a stage another encoder owns.
 */
export type RenderPathTraceStage = RenderFrameSeam<"water">;

/**
 * The dry scene, encoded into this pipeline's own background attachment.
 *
 * It takes no seam: the replacement owns a different encoder's stages, and the
 * caller that supplies it already holds the seam for them. Forwarding one
 * through here would be this pipeline handing out a closer for stages it does
 * not own.
 */
export type DrySceneReplacementEncoder = (
  encoder: GPUCommandEncoder,
  target: GPUTexture | GPUTextureView,
) => DrySceneReplacementResult | false;

/** What to put behind raster water when no dry-scene encoder is requested. */
export type RasterWaterBackgroundMode = "require-dry-scene" | "clear";

/** How the extracted liquid surface is presented in the viewport. */

/**
 * Restricted tall cells cannot contain a free surface below their cubic band.
 * The interior can therefore follow that band. Two adjacent base steps can
 * meet across a cube diagonal.
 */
export function surfaceExtractionDispatchPlan(
  nx: number,
  ny: number,
  nz: number,
  packedNy: number,
  restrictedTallCell: boolean,
  maximumNeighborDelta: number,
): SurfaceExtractionDispatchPlan {
  if (!restrictedTallCell) {
    return { mode: "full-volume", full: [Math.ceil((nx + 1) / 4),
      Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)] };
  }
  const bandCubeRows = Math.min(ny + 1, Math.max(1, packedNy + 2 * Math.ceil(Math.max(0, maximumNeighborDelta)) - 1));
  return {
    mode: "restricted-band",
    band: [Math.ceil(Math.max(0, nx - 1) / 4), Math.ceil(bandCubeRows / 4), Math.ceil(Math.max(0, nz - 1) / 4)],
    tallSides: [Math.ceil(Math.max(0, nx - 1) / 8), Math.ceil(Math.max(0, nz - 1) / 8), 1],
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
const EXTRACTION_ORDER_WORKGROUP = 256;
/** The listed-window classify launch: this many workgroups share the list. */
export const EXTRACTION_WINDOW_WORKGROUPS = 4096;

/** The dense classify launch: `full` scans every cube of the (n+1)^3 lattice;
 * `windows` lists the 4^3 cube windows the Uniform Geometric 4h vertex base
 * cannot prove empty and scans only those (collectWindowsMain,
 * extractWindowsMain). */
export type WaterSurfaceClassify = "full" | "windows";
const WATER_SURFACE_CLASSIFY_ENTRY: Record<WaterSurfaceClassify, string> = { full: "extractMain", windows: "extractWindowsMain" };

/** Vertex capacity from grid surface area (32 bytes per vertex, 64 MiB cap). */
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

/**
 * Geometry capacity for a sparse fine source is a function of physical pages,
 * never of the signed address lattice those pages can occupy. Six vertices per
 * physical sample is deliberately generous for an ordinary surface while the
 * established 8/64 MiB bounds still contain pathological fields.
 */
export function globalFineSurfaceVertexCapacity(
  pageCapacity: number,
  samplesPerBrick: number,
): number {
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 1
    || !Number.isSafeInteger(samplesPerBrick) || samplesPerBrick < 1) {
    throw new RangeError("Global fine geometry capacities must be positive integers");
  }
  return Math.max(262_144, Math.min(2_097_152,
    pageCapacity * samplesPerBrick * 6));
}

/**
 * Bounded two-dimensional dispatch over every physical fine-brick sample.
 *
 * An empty publication carries no pages, and the answer for it is a dispatch of
 * no workgroups — `[0, 1, 1]`. The caller skips the encode rather than issuing
 * it, so the empty scene costs nothing and Dawn is not asked to validate a
 * zero-workgroup launch.
 */
export function globalFineSurfaceDispatch(pageCapacity: number, samplesPerBrick: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < 0
    || !Number.isSafeInteger(samplesPerBrick) || samplesPerBrick < 1) {
    throw new RangeError("Global fine extraction capacities must be non-negative and positive integers");
  }
  if (pageCapacity === 0) return [0, 1, 1] as const;
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

/** uniformPhiNormal's 27-sample accumulation around `center`: `sample` is the
 * statement that forms `phi` at the lattice vertex q, `head` declares q. */
const uniformNormalLoopWGSL = (sample: string, head = "let q=clamp(center+vec3i(ox,oy,oz),vec3i(0),dimensions);") => `for(var oz=-1;oz<=1;oz+=1){for(var oy=-1;oy<=1;oy+=1){for(var ox=-1;ox<=1;ox+=1){
    ${head}
    let delta=vec3f(q)-x;
    let weight=exp(-0.5*dot(delta,delta)/(.85*.85));
    var phi=0.0;
    ${sample}
    if(!(abs(phi)<1e10)){return fallback;}
    weightSum+=weight;phiSum+=weight*phi;
    derivativeWeightSum+=weight*delta;phiDerivativeSum+=weight*phi*delta;
  }}}`;
/** The extraction's hook in umVertexValue: a vertex off the tile corners, in
 * a tile with a 4h tile in its stencil, of the cube whose corners are being
 * loaded (uniformCubeMemo). */
export const SURFACE_EXTRACTION_VERTEX_CACHE = "if(cubeMemo!=0u){return uniformCubeVertex(p);}";

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
  meshGeneration: u32,
  // Uniform Geometric: the mixed cubes the listed windows appended from the
  // worklist's end, until orderSurfaceWorklistMain moves them behind the
  // others and the prepare kernel folds them into activeCubeCount. Zero to
  // every other reader: the worklist is [0, activeCubeCount) as it was.
  mixedCubeCount: atomic<u32>,
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
@group(0) @binding(13) var denseVertexPhi: texture_3d<f32>;
// Uniform Geometric: the 4h vertex base ((t+1)^3, texel g = phi at vertex 4g).
// The mixed samplers load a 4h owner's corners from it and an h tile's
// vertices from denseVertexPhi, the detail field (a 1^3 placeholder while
// the solver holds no h-tile capacity: no h tile exists then).
@group(0) @binding(15) var coarseVertexPhi: texture_3d<f32>;
${uniformMixedPresentationWGSL(14,"denseVertexPhi","coarseVertexPhi",undefined,SURFACE_EXTRACTION_VERTEX_CACHE)}
override countOnly = false;
override sparseField = false;
${marchingCubesLookupWGSL}

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
  if(umPresentationEnabled()){return 0.5-umSampleVertex(vec3f(cell)+vec3f(0.5))/(u.container.y/u.gridInfo.y);}
  // Raw loads: a packed field always arrives with its mixed topology, which
  // took the branch above.
  if (mode < 1.5) {
    // An explicitly published contour field is nodal (n+1), independent of
    // the simulation's packed cell volume. Average its corners at the centre.
    if(all(textureDimensions(volume)==vec3u(dims)+1u)){
      var phi=0.0;for(var k=0u;k<8u;k++){phi+=textureLoad(volume,cell+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),0).x;}
      return 0.5-0.125*phi/(u.container.y/u.gridInfo.y);
    }
    return textureLoad(volume, cell, 0).x;
  }
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

// The solver has already published the field to contour: raw rho for the
// ordinary CM12 path, or rho'' after its optional Section 3.8 reconstruction.
// Sample it exactly. A second presentation blur is not contour preserving: a
// one-cell sheet with rho=.6 becomes .3 under [1 2 1]^3 / 64 and vanishes even
// though the solver and the surface-density diagnostic both classify it as
// liquid. Marching cubes owns interpolation between these cell-centred nodes.
fn presentationFieldCell(cell:vec3i)->f32{
  return fieldCell(cell);
}

fn latticeValue(p: vec3i) -> f32 {
  let dims = select(vec3i(u.gridInfo.xyz), vec3i(sparseParams.fineDims.xyz), sparseField);
  if (p.x <= 0 || p.z <= 0 || p.x >= dims.x + 1 || p.z >= dims.z + 1 || p.y >= dims.y + 1) { return 0.0; }
  let cell = vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1);
  if (sparseField) { return occupancyFromPhi(sparsePhiAt(cell)); }
  return presentationFieldCell(cell);
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

// Uniform Geometric: the normal stencil's samples, classified once per cube.
// A crossing of the cube at lattice base b has its stencil centre in
// b-1..b+1, so every sample of every crossing is a nodal value at a lattice
// vertex in b-2..b+2 (clamped to the lattice). Every tile whose closure holds
// one of those is the cube's home tile b/4 or one of its 26 neighbours, and
// the home tile's stencil word already says which of them are h. With that,
// umVertexValue reduces exactly to:
//   - a vertex off the tile corners with a 4h tile around it is that tile's
//     trilinear interpolant of its eight corners. Every 4h tile around it
//     gives the same terms (the weights off the vertex's own face or edge are
//     zero) and umVertexSum8 is invariant under the axis flips that relate
//     their slots, so the value is formed in the cube's own base block: the
//     27 base vertices of the at most two tiles an axis that hold b-2..b+2,
//     loaded once per cube;
//   - a vertex off the tile corners with only h tiles around it is stored;
//   - a tile corner needs its own tile's stencil and stays umVertexValue,
//     except where every tile of the base block is 4h: then it is the base.
// The arithmetic is umVertexFrom4's, term for term, so the normals are the
// reference's bit for bit. Off, every sample is umVertexValue: the reference
// the extraction lane compares against.
override uniformNormalGather = true;
// A cube's nodal values, formed once. A mixed (class 3) cube asks
// umVertexValue for the same few vertices again and again, each time through
// the incident-tile walk: its h corner cells read 64 of the 27 at b-1..b+1,
// and every crossing's normal 27 of the 125 at b-2..b+2. Each is formed the
// first time it is asked for and kept for the rest of the cube
// (uniformMemoVertex). The corner cells keep umSampleVertex's own arithmetic:
// only umVertexValue's answer comes from here, through the sampler's
// cacheLookup. An all-h (class 1) cube's are all stored. Off, every read
// walks: the reference.
override uniformCubeMemo = true;
// The normal's 27 samples by one literal loop per cube class. Off, the
// single loop that branches per sample: the reference.
override uniformNormalLoops = true;
var<private> normalCube: u32 = 0u;       // 0 reference, 1 all h, 2 all 4h, 3 mixed
var<private> normalCubeBase: vec3i;
var<private> normalBase: array<f32, 27>;
var<private> normalBaseLow: vec3i;       // the base block's first and last tile
var<private> normalBaseHigh: vec3i;
var<private> normalHome: vec3i;
var<private> normalFine: u32;            // the home stencil: a bit per neighbour, set when h
fn uniformNormalCube(base: vec3i) {
  normalCube = 0u;
  if (!(uniformNormalGather || uniformCubeMemo) || !umPresentationEnabled() || u.gridInfo.w >= 1.5) { return; }
  let n = vec3i(umDimensions());
  if (any(n != vec3i(u.gridInfo.xyz))) { return; }
  normalCubeBase = base;
  let home = min(base, n - vec3i(1)) / 4;
  let stencil = umTileStencil(umTileAt(vec3u(home)));
  normalHome = home; normalFine = stencil.x;
  if ((stencil.x >> 27u) == 1u) { normalCube = 1u; return; }
  // Every in-domain tile around a vertex of b-2..b+2 is h (bits r0..r1 an
  // axis of the stencil): each sample is stored, as in the all-h cube, with
  // no part in what the home tile's other neighbours are. A band of h tiles
  // has such a stencil at every tile of its rim.
  {
    let t = vec3i(umTileDimensions());
    let r0 = vec3u(max((max(base - vec3i(2), vec3i(0)) - vec3i(1)) >> vec3u(2u), vec3i(0)) - home + vec3i(1));
    let r1 = vec3u(min(min(base + vec3i(2), n) >> vec3u(2u), t - vec3i(1)) - home + vec3i(1));
    let m = ((vec3u(2u) << r1) - vec3u(1u)) & ~((vec3u(1u) << r0) - vec3u(1u));
    let rows = m.x * ((m.y & 1u) | ((m.y & 2u) << 2u) | ((m.y & 4u) << 4u));
    let mask = rows * ((m.z & 1u) | ((m.z & 2u) << 8u) | ((m.z & 4u) << 16u));
    if ((stencil.x & mask) == mask) { normalCube = 1u; return; }
  }
  // Tiles g0..g1 hold b-2..b+2 in their closure: one or two an axis.
  let g0 = max(base - vec3i(2), vec3i(0)) / 4;
  let g1 = (min(base + vec3i(2), n) + vec3i(3)) / 4 - vec3i(1);
  var coarse = true;
  for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
    let r = vec3u(min(g0 + vec3i(umCorner(k, 2u)), g1) - home + vec3i(1));
    if (((stencil.x >> (r.x + 3u * (r.y + 3u * r.z))) & 1u) != 0u) { coarse = false; }
  }
  let t = vec3i(umTileDimensions());
  // Opaque bound: 27 loads, not 27 unrolled load sites.
  for (var k = 0u; k < min(arrayLength(&umTopology), 27u); k += 1u) {
    let c = vec3i(umCorner(k, 3u));
    if (any(c > g1 - g0 + vec3i(1))) { continue; }
    normalBase[k] = textureLoad(coarseVertexPhi, min(g0 + c, t), 0).x;
  }
  normalBaseLow = g0; normalBaseHigh = g1;
  normalCube = select(3u, 2u, coarse);
}
// The stencil bits of the tiles low..high an axis, as offsets from the home
// tile (-1..1).
fn uniformStencilBox(low: vec3i, high: vec3i) -> u32 {
  let m = ((vec3u(2u) << vec3u(high + vec3i(1))) - vec3u(1u)) & ~((vec3u(1u) << vec3u(low + vec3i(1))) - vec3u(1u));
  let rows = m.x * ((m.y & 1u) | ((m.y & 2u) << 2u) | ((m.y & 4u) << 4u));
  return rows * ((m.z & 1u) | ((m.z & 2u) << 8u) | ((m.z & 4u) << 16u));
}
// Whether uniformNormalCube classes the cube at base mixed, from its home
// stencil alone (a listed window carries it): no all-h stencil, a 4h tile
// around a vertex of b-2..b+2, and an h tile among g0..g1. It orders the
// worklist and decides no value.
fn uniformCubeMixed(base: vec3i, fine: u32) -> bool {
  if ((fine >> 27u) == 1u) { return false; }
  let n = vec3i(umDimensions());
  let t = vec3i(umTileDimensions());
  let home = min(base, n - vec3i(1)) / 4;
  let low = max(base - vec3i(2), vec3i(0));
  let high = min(base + vec3i(2), n);
  let around = uniformStencilBox(max((low - vec3i(1)) >> vec3u(2u), vec3i(0)) - home, min(high >> vec3u(2u), t - vec3i(1)) - home);
  if ((fine & around) == around) { return false; }
  return (fine & uniformStencilBox(low / 4 - home, (high + vec3i(3)) / 4 - vec3i(1) - home)) != 0u;
}
// The trilinear interpolant of the base block's tile at q: umVertexFrom4.
fn uniformNormalBase(q: vec3i) -> f32 {
  let tile = clamp(q / 4, normalBaseLow, normalBaseHigh);
  let f = vec3f(q - 4 * tile) / 4.0;
  let r = vec3u(tile - normalBaseLow);
  let i = r.x + 3u * (r.y + 3u * r.z);
  // Slots 0..3 and 4..7 of umVertexFrom4's values, weight (x*y)*z as there;
  // a zero weight contributes no term, as there.
  let xy = vec4f(1.0 - f.x, f.x, 1.0 - f.x, f.x) * vec4f(1.0 - f.y, 1.0 - f.y, f.y, f.y);
  let lowWeight = xy * (1.0 - f.z);
  let highWeight = xy * f.z;
  let low = select(vec4f(0.0),
    lowWeight * vec4f(normalBase[i], normalBase[i + 1u], normalBase[i + 3u], normalBase[i + 4u]),
    lowWeight > vec4f(0.0));
  let high = select(vec4f(0.0),
    highWeight * vec4f(normalBase[i + 9u], normalBase[i + 10u], normalBase[i + 12u], normalBase[i + 13u]),
    highWeight > vec4f(0.0));
  // umVertexSum8: ((v0+v5)+(v1+v4))+((v2+v7)+(v3+v6)).
  let pair = low + high.yxwz;
  return (pair.x + pair.y) + (pair.z + pair.w);
}
// Whether a tile whose closure holds q is 4h (q is in the cube's b-2..b+2).
// The in-domain tiles around q as a mask of the home stencil: an axis holds
// q's own tile (below the upper wall) and, on a tile plane, the one before
// it. No loop: a mixed cube asks this of each of its samples.
fn uniformNormalWide(q: vec3i) -> bool {
  let t = vec3i(umTileDimensions());
  let a = q / 4;
  let r = vec3u(clamp(a - normalHome + vec3i(1), vec3i(0), vec3i(2)));
  let on = (q & vec3i(3)) == vec3i(0);
  let m = select(vec3u(0u), vec3u(1u) << r, a < t) | select(vec3u(0u), (vec3u(1u) << r) >> vec3u(1u), on & (a > vec3i(0)));
  let rows = m.x * ((m.y & 1u) | ((m.y & 2u) << 2u) | ((m.y & 4u) << 4u));
  let mask = rows * ((m.z & 1u) | ((m.z & 2u) << 8u) | ((m.z & 4u) << 16u));
  return (~normalFine & mask) != 0u;
}

// umVertexValue at a tile corner: the stencil of the tile the corner opens
// decides the field (all h: the detail field; otherwise the base).
fn uniformCornerValue(q: vec3i) -> f32 {
  let tile = umTileAt(vec3u(min(q, vec3i(umDimensions()) - vec3i(1))) / 4u);
  if (umTileMaximumWidth(tile) == 1u) { return umLoadFineVertex(vec3u(q)); }
  return umLoadCoarseVertex(vec3u(q));
}
// Whether a tile is h: a tile within one of the classified home tile.
fn uniformTileFine(tile: vec3i) -> bool {
  let r = vec3u(tile - normalHome + vec3i(1));
  return ((normalFine >> (r.x + 3u * (r.y + 3u * r.z))) & 1u) != 0u;
}

// Uniform Geometric: a cell-centre contour value from its eight inputs, in
// umSampleVertex's own arithmetic (fieldCell is the reference).
//   kind 0: a 4h cell, from its tile's eight corners: the regular sample of
//           a 4h owner (a 4h owner is as wide as any stencil);
//   kind 1: an h cell of a tile whose whole stencil is h, from its own eight
//           vertices as stored: the regular sample of a unit owner;
//   kind 2: an h cell of a tile with a 4h neighbour, from umVertexValue of
//           its eight vertices: the general sample.
// A cell centre is half a cell into a unit owner, so every weight there is
// 1/8; the two h kinds keep their own operand order.
fn uniformCellValue(cell: vec3i, kind: u32, x: array<f32, 8>) -> f32 {
  let t = select(vec3f(0.5), (vec3f(cell) + vec3f(0.5) - vec3f(4 * (cell / 4))) / 4.0, kind == 0u);
  var values: array<f32, 8>;
  for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
    let w = select(vec3f(1.0) - t, t, umCorner(k, 2u) != vec3u(0u));
    if (kind == 2u) { values[k] = (w.x * w.y * w.z) * x[k]; } else { values[k] = x[k] * w.x * w.y * w.z; }
  }
  return 0.5 - umVertexSum8(values) / (u.container.y / u.gridInfo.y);
}

// umVertexValue of a vertex off the tile corners in the classified cube's
// b-2..b+2, formed once per cube: the interpolant of a 4h tile around it, or
// stored where every tile around it is h (uniformNormalCube's reduction).
var<private> cubeMemo: u32 = 0u;          // the cube's class while its corners load
var<private> cubeMemoHave: array<u32, 4>; // a bit per vertex of b-2..b+2
var<private> cubeMemoValue: array<f32, 125>;
fn uniformMemoVertex(q: vec3i) -> f32 {
  let d = vec3u(q - normalCubeBase + vec3i(2));
  let k = d.x + 5u * (d.y + 5u * d.z);
  let bit = 1u << (k & 31u);
  if ((cubeMemoHave[k >> 5u] & bit) != 0u) { return cubeMemoValue[k]; }
  var value = 0.0;
  if (uniformNormalWide(q)) { value = uniformNormalBase(q); } else { value = umLoadFineVertex(vec3u(q)); }
  cubeMemoValue[k] = value;
  cubeMemoHave[k >> 5u] |= bit;
  return value;
}
// The sampler's cacheLookup while a cube's corners load.
fn uniformCubeVertex(p: vec3u) -> f32 {
  if (cubeMemo == 1u) { return umLoadFineVertex(p); }
  return uniformMemoVertex(vec3i(p));
}

// Uniform Geometric publishes an (n+1)^3 nodal signed distance. Reconstruct
// its gradient at the world-space crossing, rather than differentiating each
// cube's eight contour values independently. Adjacent cubes then give a shared
// crossing the same optical normal, including in the caustic projection.
fn uniformPhiNormal(lattice:vec3f, fallback:vec3f) -> vec3f {
  let dimensions=vec3i(u.gridInfo.xyz);
  // The mixed lattice is the 4h base's; a raw nodal field's is its texture's.
  var vertexDimensions=textureDimensions(denseVertexPhi);
  if(umPresentationEnabled()){vertexDimensions=umDimensions()+vec3u(1u);}
  if(u.gridInfo.w>=1.5 || any(vec3i(vertexDimensions)!=dimensions+vec3i(1))){return fallback;}
  // The x/z halo closes liquid against the tank. Those triangles are boundary
  // faces, not phi's free surface: its gradient can point upward while their
  // geometric normal must point into the wall. Retain the contour normal in
  // this half-cell closure strip and at the floor contact.
  if(lattice.x<=1.0 || lattice.z<=1.0 || lattice.x>=f32(dimensions.x)
    || lattice.z>=f32(dimensions.z) || lattice.y<=1.0){return fallback;}
  let x=lattice-vec3f(0.5);
  let center=vec3i(round(x));
  // The cube's classification covers the stencil of every centre in b-1..b+1.
  let gather=select(0u,normalCube,uniformNormalGather&&all(abs(center-normalCubeBase)<=vec3i(1)));
  var weightSum=0.0;var phiSum=0.0;
  var derivativeWeightSum=vec3f(0.0);var phiDerivativeSum=vec3f(0.0);
  if(uniformNormalLoops&&gather==1u){
  ${uniformNormalLoopWGSL("if(all((q&vec3i(3))==vec3i(0))){phi=umVertexValue(vec3u(q));}else{phi=umLoadFineVertex(vec3u(q));}")}
  }else if(uniformNormalLoops&&gather==2u){
  ${uniformNormalLoopWGSL("phi=uniformNormalBase(q);")}
  }else if(uniformNormalLoops&&gather==3u){
  ${uniformNormalLoopWGSL("if(all((q&vec3i(3))==vec3i(0))){phi=umVertexValue(vec3u(q));}else if(uniformCubeMemo){phi=uniformMemoVertex(q);}else if(uniformNormalWide(q)){phi=uniformNormalBase(q);}else{phi=umLoadFineVertex(vec3u(q));}")}
  }else{
  ${uniformNormalLoopWGSL(`let tileCorner=all((q&vec3i(3))==vec3i(0));
    if(uniformCubeMemo&&gather==3u&&!tileCorner){phi=uniformMemoVertex(q);}
    else if(gather==2u||(gather==3u&&!tileCorner&&uniformNormalWide(q))){phi=uniformNormalBase(q);}
    else if(gather!=0u&&!tileCorner){phi=umLoadFineVertex(vec3u(q));}
    else if(umPresentationEnabled()){phi=umVertexValue(vec3u(q));}else{phi=textureLoad(denseVertexPhi,q,0).x;}`)}
  }
  let derivative=phiDerivativeSum*weightSum-phiSum*derivativeWeightSum;
  let cell=u.container.xyz/max(u.gridInfo.xyz,vec3f(1.0));
  let gradient=derivative/max(cell,vec3f(1e-6));
  return select(fallback,normalize(gradient),weightSum>1e-8&&length(gradient)>1e-8);
}

// The cube's corner values travel by pointer because WGSL passes arrays by
// value. Lorensen--Cline interpolation is the unmodified linear 0.5 crossing;
// in particular, crossings at cube corners must not be displaced inward.
fn crossing(a: vec3f, b: vec3f, va: f32, vb: f32, cubeBase: vec3f, cubeScale: f32, cubeValue: ptr<function, array<f32, 8>>, dims:vec3f) -> SurfaceVertex {
  let denominator = vb - va;
  var t = 0.5;
  if (abs(denominator) > 1e-20) { t = clamp((0.5 - va) / denominator, 0.0, 1.0); }
  let lattice = mix(a, b, t);
  return SurfaceVertex(vec4f(latticeWorld(lattice,dims), 1.0),
    vec4f(uniformPhiNormal(lattice,surfaceNormal(lattice, cubeBase, cubeScale, cubeValue,dims)), 0.0));
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

fn cubeEdgeVertex(edgeId:u32,p:ptr<function,array<vec3f,8>>,value:ptr<function,array<f32,8>>,cubeBase:vec3f,cubeScale:f32,dims:vec3f)->SurfaceVertex{
  let a=MC_EDGE_A[edgeId];let b=MC_EDGE_B[edgeId];
  return crossing((*p)[a],(*p)[b],(*value)[a],(*value)[b],cubeBase,cubeScale,value,dims);
}

// A crossing is evaluated once per cube and reused by every triangle of the
// cube that meets at it, in the same emission order. Off, every triangle
// corner evaluates its own: the reference the extraction lane compares against.
override shareCubeVertices = true;
fn polygoniseCube(p:ptr<function,array<vec3f,8>>,value:ptr<function,array<f32,8>>,cubeBase:vec3f,cubeScale:f32,dims:vec3f){
  let cubeCase=mcCase(value);let indexCount=mcIndexCount(cubeCase);
  if(shareCubeVertices){
    var edge:array<SurfaceVertex,12>;var corner:array<SurfaceVertex,3>;var have=0u;
    for(var index=0u;index<indexCount;index+=1u){
      let id=mcEdge(cubeCase,index);
      if((have&(1u<<id))==0u){edge[id]=cubeEdgeVertex(id,p,value,cubeBase,cubeScale,dims);have|=1u<<id;}
      corner[index%3u]=edge[id];
      if(index%3u==2u){emitTriangle(corner[0],corner[1],corner[2]);}
    }
    return;
  }
  for(var index=0u;index<indexCount;index+=3u){
    emitTriangle(
      cubeEdgeVertex(mcEdge(cubeCase,index),p,value,cubeBase,cubeScale,dims),
      cubeEdgeVertex(mcEdge(cubeCase,index+1u),p,value,cubeBase,cubeScale,dims),
      cubeEdgeVertex(mcEdge(cubeCase,index+2u),p,value,cubeBase,cubeScale,dims));
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

// Must classify vertices exactly as polygoniseCube does: the polygonise pass
// writes into per-thread blocks sized by this count, so a mismatch corrupts a
// neighbouring thread's triangles.
fn cubeTriangleCount(value: ptr<function, array<f32, 8>>) -> u32 {
  return mcIndexCount(mcCase(value))/3u;
}

// The sweep kernels stop here: eight corner loads, a min/max test, and one
// worklist append per *surface* cube. Emission code is confined to
// polygoniseMain so the register footprint of the full-lattice scan stays
// small enough for the occupancy that hides the load latency.
fn cubeHoldsSurface(value: ptr<function, array<f32, 8>>) -> bool {
  var minimum = 1.0; var maximum = 0.0;
  for (var i = 0; i < 8; i += 1) {
    minimum = min(minimum, (*value)[i]); maximum = max(maximum, (*value)[i]);
  }
  return !(minimum >= 0.5 || maximum < 0.5);
}
// mixed: a Uniform Geometric cube that polygoniseMain will class mixed. It
// is appended from the worklist's end, so the worklist polygoniseMain reads
// holds the mixed cubes together: a lane group pays for every class its
// lanes take, and a mixed cube among the others cost them its samples too.
fn appendSurfaceCube(base: vec3i, scale: u32, value: ptr<function, array<f32, 8>>, mixed: bool) {
  if (countOnly) {
    // The benchmark's uncapped equivalence count. Counting whole cubes here
    // keeps it exact regardless of the production worklist capacity.
    atomicAdd(&drawArgs.vertexCount, 3u * cubeTriangleCount(value));
    return;
  }
  let cube = vec2u(u32(base.x) | (u32(base.z) << 16u), u32(base.y) | (scale << 16u));
  let capacity = arrayLength(&activeCubes);
  if (mixed) {
    let slot = atomicAdd(&drawArgs.mixedCubeCount, 1u);
    if (slot < capacity) { activeCubes[capacity - 1u - slot] = cube; }
    return;
  }
  let slot = atomicAdd(&drawArgs.activeCubeCount, 1u);
  if (slot < capacity) { activeCubes[slot] = cube; }
}
fn classifyCubeValues(base: vec3i, scale: u32, value: ptr<function, array<f32, 8>>) {
  if (cubeHoldsSurface(value)) { appendSurfaceCube(base, scale, value, false); }
}
fn classifyCubeScaled(base: vec3i, scale: u32) {
  let fieldDims = select(vec3u(u.gridInfo.xyz), sparseParams.fineDims.xyz, sparseField);
  let cubeDims = fieldDims + vec3u(1);
  if (any(base < vec3i(0)) || any(vec3u(base) >= cubeDims)) { return; }
  var value = loadCubeCornersScaled(base, i32(scale));
  classifyCubeValues(base, scale, &value);
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
  udrInit();
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
    // Uniform Geometric: one classification serves the corner values here
    // and every crossing's normal stencil below.
    if (cubeScale == 1u && !sparseField) { uniformNormalCube(base); }
    if (uniformCubeMemo && (normalCube == 1u || normalCube == 3u)) { cubeMemo = normalCube; }
    value = loadCubeCornersScaled(base, i32(cubeScale));
    cubeMemo = 0u;
    vertexCount = 3u * cubeTriangleCount(&value);
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
  // Section 3.8: the rendered surface is the classic marching-cubes mesh of
  // the density field's 0.5 isocontour.
  polygoniseCube(&p,&value,vec3f(base),f32(cubeScale),fieldDimensions);
}

@compute @workgroup_size(4, 4, 4)
fn extractMain(@builtin(global_invocation_id) gid: vec3u) {
  udrInit();
  classifyCube(vec3i(gid));
}

// Uniform Geometric: an exact test that a window of cubes holds no surface,
// from the 4h vertex base alone. Window g in [0, t]^3 is the cube bases
// 4g..4g+3 on each axis; their corner cells 4g-1..4g+3 lie in tiles g-1 and
// g. A cell of a 4h tile is sampled from its tile's eight corners only (a 4h
// owner is as wide as any stencil, so umSampleVertex takes its regular path),
// so while those eight tiles are 4h, every corner value is 0.5 - s/h with s
// in the hull of their 27 corner vertices g-1..g+1, or the wall halo's 0:
//   - all 27 above a margin: every corner is below 0.5 (the margin, 1e-6 h,
//     is far above the rounding of the interpolation and of 0.5 - s/h; the
//     halo's 0 is below 0.5 too), so no cube crosses;
//   - all 27 at or below zero and no corner in the x/z/upper halo: every
//     term of the interpolation is non-positive, so every corner is at least
//     0.5 (sign-exact) and no cube crosses.
// A NaN fails both tests. A tile outside the lattice holds only clamped or
// halo cells; the floor is clamped, not a halo.
fn uniformWindowQuiet(g: vec3u) -> bool {
  if (!umPresentationEnabled()) { return false; }
  let t = umTileDimensions();
  if (any(4u * t != vec3u(u.gridInfo.xyz))) { return false; }
  let low = vec3i(g) - vec3i(1);
  for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
    let tile = low + vec3i(umCorner(k, 2u));
    if (any(tile < vec3i(0)) || any(tile >= vec3i(t))) { continue; }
    if (umTileWidth(umTileAt(vec3u(tile))) != 4u) { return false; }
  }
  let margin = 1e-6 * u.container.y / u.gridInfo.y;
  var air = true;
  var liquid = g.x > 0u && g.z > 0u && all(g < t);
  // Opaque bound (27): a literal lets Metal unroll the loads.
  for (var k = 0u; k < min(arrayLength(&umTopology), 27u); k += 1u) {
    let vertex = clamp(low + vec3i(umCorner(k, 3u)), vec3i(0), vec3i(t));
    let phi = textureLoad(coarseVertexPhi, vertex, 0).x;
    air = air && phi > margin;
    liquid = liquid && phi <= 0.0;
  }
  return air || liquid;
}

// Two launches. The first runs one lane per window and lists the windows
// that may hold surface; the second classifies only the listed windows'
// cubes, so the scan is t^3 plus the surface rather than n^3.
struct SurfaceWindows { count: atomic<u32>, windows: array<u32> }
@group(0) @binding(16) var<storage, read_write> surfaceWindows: SurfaceWindows;
@compute @workgroup_size(4, 4, 4)
fn collectWindowsMain(@builtin(global_invocation_id) gid: vec3u) {
  udrInit();
  // ceil((n + 1) / 4) windows on an axis: t + 1 on a lattice of whole tiles.
  if (any(gid >= (vec3u(u.gridInfo.xyz) + vec3u(4u)) / 4u) || uniformWindowQuiet(gid)) { return; }
  let slot = atomicAdd(&surfaceWindows.count, 1u);
  if (slot < arrayLength(&surfaceWindows.windows)) { surfaceWindows.windows[slot] = gid.x | (gid.y << 10u) | (gid.z << 20u); }
}
// A fixed launch of EXTRACTION_WINDOW_WORKGROUPS workgroups, each a
// contiguous share of the list, a window at a time: its 64 lanes one cube
// each as the full scan's are. (One workgroup per slot of the (t + 1)^3
// lattice paid 7 ns for every slot the list did not fill: 0.26 ms at 128^3,
// 1.6 at 256^3, at any detail. Fewer workgroups than this starve the launch
// where most windows are listed; striding the list measures the same but
// for 0.3 ms more at 256^3 Full.)
//
// The workgroup shares the window's samples. Its 64 cubes read the 125
// lattice values 4g..4g+4, whose cells 4g-1..4g+3 read the 216 vertices
// 4g-1..4g+4 and, where a tile is 4h, the 27 base vertices g-1..g+1: each
// is formed once in workgroup memory (every cube forming its own eight
// corners loads each vertex about nineteen times over). Lane 0 reads the
// list slot and classifies the window from the stencil words of its eight
// tiles; the lanes then fill the vertices, the cells and their cubes, a
// barrier between. The values are fieldCell's expressions over the same
// inputs; the full scan (extractMain), every cube through latticeValue, is
// the reference, and the extraction lane holds the two worklists equal.
//
// Measured and not kept: the list's quiet test extended to an all-h window
// (its 216 stored vertices), and the list as a word per window in lattice
// order. Both are exact; the first takes 0.13 ms off classify at 128^3 Full,
// and either changes the worklist's order, which costs polygonise 0.2 ms at
// partial detail: it is fastest behind this list.
//   x: the window (WINDOW_NONE past the list)
//   y: the stencil word of the home tile min(g, t-1): its neighbourhood
//      holds every tile around the window's vertices
//   z: a bit per tile g-1..g, set when its whole stencil is h
//   w: 0 the reference per cube (no mixed lattice), 1 every tile of the
//      window has an all-h stencil (every vertex is stored: direct loads),
//      2 every tile is 4h (the base alone), 3 mixed
var<workgroup> windowHeader: vec4u;
var<workgroup> windowRange: vec2u;
var<workgroup> windowBase: array<f32, 27>;
var<workgroup> windowVertex: array<f32, 216>;
var<workgroup> windowCell: array<f32, 125>;
const WINDOW_NONE = 0xffffffffu;
// A listed window whose every nodal input lies on one side of the contour
// holds no surface (uniformWindowQuiet's two tests, on the values the vertex
// phase has just formed: every cell of the window is a convex combination of
// them): its lanes skip the cells and the cubes.
// bit 0: a value not above the air margin; bit 1: one not at or below zero.
var<workgroup> windowSigns: atomic<u32>;
fn uniformWindowSign(phi: f32) -> u32 {
  let margin = 1e-6 * u.container.y / u.gridInfo.y;
  return select(1u, 0u, phi > margin) | select(2u, 0u, phi <= 0.0);
}
fn uniformWindowHeader(slot: u32) -> vec4u {
  if (slot >= min(atomicLoad(&surfaceWindows.count), arrayLength(&surfaceWindows.windows))) { return vec4u(WINDOW_NONE, 0u, 0u, 0u); }
  let window = surfaceWindows.windows[slot];
  if (!umPresentationEnabled() || sparseField) { return vec4u(window, 0u, 0u, 0u); }
  let t = vec3i(umTileDimensions());
  if (any(4 * t != vec3i(u.gridInfo.xyz))) { return vec4u(window, 0u, 0u, 0u); }
  let g = vec3i(vec3u(window & 1023u, (window >> 10u) & 1023u, window >> 20u));
  let home = min(g, t - vec3i(1));
  let word = umTopology[2u * umTileCount() + 2u * umTileAt(vec3u(home))];
  var fine = 0u; var allFine = true; var anyFine = false;
  for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
    let tile = g - vec3i(1) + vec3i(umCorner(k, 2u));
    if (any(tile < vec3i(0)) || any(tile >= t)) { continue; }
    if ((umTopology[2u * umTileCount() + 2u * umTileAt(vec3u(tile))] >> 27u) == 1u) { fine |= 1u << k; } else { allFine = false; }
    let r = vec3u(tile - home + vec3i(1));
    if (((word >> (r.x + 3u * (r.y + 3u * r.z))) & 1u) != 0u) { anyFine = true; }
  }
  return vec4u(window, word, fine, select(select(2u, 3u, anyFine), 1u, allFine));
}
// The in-domain tiles whose closure holds vertex q of the window: x, whether
// one is h; y, one that is 4h (its index plus one), or zero. The tiles as
// uniformNormalWide's mask of the home stencil; any 4h one serves (the
// interpolant of a shared face or edge is the same from either side).
fn uniformWindowIncident(q: vec3i) -> vec2u {
  let t = vec3i(umTileDimensions());
  let a = q / 4;
  let r = vec3u(clamp(a - normalHome + vec3i(1), vec3i(0), vec3i(2)));
  let on = (q & vec3i(3)) == vec3i(0);
  let m = select(vec3u(0u), vec3u(1u) << r, a < t) | select(vec3u(0u), (vec3u(1u) << r) >> vec3u(1u), on & (a > vec3i(0)));
  let rows = m.x * ((m.y & 1u) | ((m.y & 2u) << 2u) | ((m.y & 4u) << 4u));
  let mask = rows * ((m.z & 1u) | ((m.z & 2u) << 8u) | ((m.z & 4u) << 16u));
  let wide = ~normalFine & mask;
  var incident = vec2u(select(0u, 1u, (normalFine & mask) != 0u), 0u);
  if (wide != 0u) {
    let i = firstTrailingBit(wide);
    incident.y = umTileAt(vec3u(normalHome + vec3i(vec3u(i % 3u, (i / 3u) % 3u, i / 9u)) - vec3i(1))) + 1u;
  }
  return incident;
}
// One base slot for each of the first 27 lanes: the tile corners g-1..g+1.
// Returns the lane's sign bits.
fn uniformWindowBaseValues(g: vec3i, lane: u32) -> u32 {
  if (lane >= 27u) { return 0u; }
  let corner = g - vec3i(1) + vec3i(umCorner(lane, 3u));
  if (any(corner < vec3i(0)) || any(corner > vec3i(umTileDimensions()))) { return 0u; }
  let phi = textureLoad(coarseVertexPhi, corner, 0).x;
  windowBase[lane] = phi;
  return uniformWindowSign(phi);
}
// umVertexFrom4 of the 4h tile at q, its corners read from the window's base
// (the same texels, loaded once a window rather than eight times a vertex)
// and its eight terms formed as two vectors: slots 0..3 and 4..7 of
// umVertexFrom4's values, weight (x*y)*z as there; a zero weight contributes
// no term, as there; the sum is umVertexSum8's. A corner with a positive
// weight lies in g-1..g+1: the tile's closure holds q, so a corner at g+2
// belongs to a tile opening at q's plane and weighs nothing; its step
// rereads the slot before it.
fn uniformWindowFrom4(q: vec3i, g: vec3i, tile: u32) -> f32 {
  let a = vec3i(umTileCoord(tile));
  let f = vec3f(q - 4 * a) / 4.0;
  let r = vec3u(a - g + vec3i(1));
  let s = select(vec3u(1u, 3u, 9u), vec3u(0u), r == vec3u(2u));
  let i = r.x + 3u * (r.y + 3u * r.z);
  let j = i + s.z;
  let xy = vec4f(1.0 - f.x, f.x, 1.0 - f.x, f.x) * vec4f(1.0 - f.y, 1.0 - f.y, f.y, f.y);
  let lowWeight = xy * (1.0 - f.z);
  let highWeight = xy * f.z;
  let low = select(vec4f(0.0),
    lowWeight * vec4f(windowBase[i], windowBase[i + s.x], windowBase[i + s.y], windowBase[i + s.x + s.y]),
    lowWeight > vec4f(0.0));
  let high = select(vec4f(0.0),
    highWeight * vec4f(windowBase[j], windowBase[j + s.x], windowBase[j + s.y], windowBase[j + s.x + s.y]),
    highWeight > vec4f(0.0));
  // umVertexSum8: ((v0+v5)+(v1+v4))+((v2+v7)+(v3+v6)).
  let pair = low + high.yxwz;
  return (pair.x + pair.y) + (pair.z + pair.w);
}
// Four vertex slots a lane. An all-h window's are all stored: a literal loop
// of direct loads. Otherwise a vertex off the tile corners is umVertexValue:
// the interpolant of a 4h tile around it (from the base, which the lanes
// loaded before the barrier), or stored where every tile around it is h. A
// tile corner is left as stored (uniformCellValue's kind 1; kind 2 forms its
// own). A vertex no h tile holds is read by no h cell and stays unwritten.
// Returns the lane's sign bits.
fn uniformWindowVertices(g: vec3i, scan: u32, lane: u32) -> u32 {
  let t = vec3i(umTileDimensions());
  var signs = 0u;
  if (scan == 1u) {
    for (var m = 0u; m < 4u; m += 1u) {
      let slot = 4u * lane + m;
      let q = 4 * g - vec3i(1) + vec3i(umCorner(slot, 6u));
      if (slot < 216u && all(q >= vec3i(0)) && all(q <= 4 * t)) {
        let phi = umLoadFineVertex(vec3u(q));
        windowVertex[slot] = phi; signs |= uniformWindowSign(phi);
      }
    }
  } else {
    for (var m = 0u; m < min(arrayLength(&umTopology), 4u); m += 1u) {
      let slot = 4u * lane + m;
      if (slot >= 216u) { break; }
      let q = 4 * g - vec3i(1) + vec3i(umCorner(slot, 6u));
      if (any(q < vec3i(0)) || any(q > 4 * t)) { continue; }
      let incident = uniformWindowIncident(q);
      if (incident.x == 0u) { continue; }
      var phi = 0.0;
      if (incident.y != 0u && any((q & vec3i(3)) != vec3i(0))) { phi = uniformWindowFrom4(q, g, incident.y - 1u); }
      else { phi = umLoadFineVertex(vec3u(q)); }
      windowVertex[slot] = phi; signs |= uniformWindowSign(phi);
    }
  }
  return signs;
}
// Two lattice values a lane of an all-h window: latticeValue(4g + slot), a
// cell from its eight stored vertices in the regular sample's arithmetic at
// a cell centre (every weight 1/8, as x*.5*.5*.5, summed as umVertexSum8).
fn uniformWindowCellsFine(g: vec3i, lane: u32) {
  let n = vec3i(u.gridInfo.xyz);
  for (var m = 0u; m < 2u; m += 1u) {
    let slot = 2u * lane + m;
    if (slot < 125u) {
      let p = 4 * g + vec3i(umCorner(slot, 5u));
      var value = 0.0;
      // latticeValue's wall halo.
      if (!(p.x <= 0 || p.z <= 0 || p.x >= n.x + 1 || p.z >= n.z + 1 || p.y >= n.y + 1)) {
        let d = vec3u(vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1) - 4 * g + vec3i(1));
        let i = d.x + 6u * (d.y + 6u * d.z);
        let low = vec4f(windowVertex[i], windowVertex[i + 1u], windowVertex[i + 6u], windowVertex[i + 7u]) * 0.5 * 0.5 * 0.5;
        let high = vec4f(windowVertex[i + 36u], windowVertex[i + 37u], windowVertex[i + 42u], windowVertex[i + 43u]) * 0.5 * 0.5 * 0.5;
        // umVertexSum8: ((v0+v5)+(v1+v4))+((v2+v7)+(v3+v6)).
        let pair = low + high.yxwz;
        value = 0.5 - ((pair.x + pair.y) + (pair.z + pair.w)) / (u.container.y / u.gridInfo.y);
      }
      windowCell[slot] = value;
    }
  }
}
// Two lattice values a lane: latticeValue(4g + slot), each cell by its kind
// (uniformCellValue).
fn uniformWindowCells(g: vec3i, header: vec4u, lane: u32) {
  let n = vec3i(u.gridInfo.xyz);
  for (var m = 0u; m < min(arrayLength(&umTopology), 2u); m += 1u) {
    let slot = 2u * lane + m;
    if (slot >= 125u) { break; }
    let p = 4 * g + vec3i(umCorner(slot, 5u));
    // latticeValue's wall halo.
    if (p.x <= 0 || p.z <= 0 || p.x >= n.x + 1 || p.z >= n.z + 1 || p.y >= n.y + 1) { windowCell[slot] = 0.0; continue; }
    let cell = vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1);
    let tile = cell / 4;
    let e = vec3u(tile - g + vec3i(1));
    var x: array<f32, 8>;
    var kind = 0u;
    if (uniformTileFine(tile)) {
      kind = select(2u, 1u, ((header.z >> (e.x + 2u * (e.y + 2u * e.z))) & 1u) != 0u);
      for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
        let q = cell + vec3i(umCorner(k, 2u));
        let d = vec3u(q - 4 * g + vec3i(1));
        if (kind == 2u && all((q & vec3i(3)) == vec3i(0))) { x[k] = uniformCornerValue(q); }
        else { x[k] = windowVertex[d.x + 6u * (d.y + 6u * d.z)]; }
      }
    } else {
      for (var k = 0u; k < umPresentationLoopBound(); k += 1u) {
        let c = e + umCorner(k, 2u);
        x[k] = windowBase[c.x + 3u * (c.y + 3u * c.z)];
      }
    }
    windowCell[slot] = uniformCellValue(cell, kind, x);
  }
}
// A fixed launch shares the window list out: workgroup w takes the slots
// [w * each, (w + 1) * each), so the launch costs what the list holds,
// whatever the lattice could.
@compute @workgroup_size(4, 4, 4)
fn extractWindowsMain(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_id) lane: vec3u, @builtin(local_invocation_index) index: u32) {
  udrInit();
  if (index == 0u) {
    let count = min(atomicLoad(&surfaceWindows.count), arrayLength(&surfaceWindows.windows));
    windowRange = min(vec2u(group.x, group.x + 1u) * ((count + groups.x - 1u) / groups.x), vec2u(count));
  }
  let range = workgroupUniformLoad(&windowRange);
  for (var slot = range.x; slot < range.y; slot += 1u) {
    if (index == 0u) { atomicStore(&windowSigns, 0u); windowHeader = uniformWindowHeader(slot); }
    let header = workgroupUniformLoad(&windowHeader);
    let g = vec3i(vec3u(header.x & 1023u, (header.x >> 10u) & 1023u, header.x >> 20u));
    var signs = 0u;
    if (header.w != 0u) { normalHome = min(g, vec3i(umTileDimensions()) - vec3i(1)); normalFine = header.y; }
    if (header.w > 1u) { signs = uniformWindowBaseValues(g, index); }
    // A mixed window's interpolated vertices read the base just loaded.
    if (header.w == 3u) { workgroupBarrier(); }
    if (header.w == 1u || header.w == 3u) { signs |= uniformWindowVertices(g, header.w, index); }
    if (signs != 0u) { atomicOr(&windowSigns, signs); }
    workgroupBarrier();
    var quiet = false;
    if (header.w != 0u) {
      let signs = atomicLoad(&windowSigns);
      // Air, or liquid with no corner in the wall halo.
      quiet = (signs & 1u) == 0u || ((signs & 2u) == 0u && g.x > 0 && g.z > 0 && all(g < vec3i(umTileDimensions())));
    }
    if (header.w != 0u && !quiet) {
      if (header.w == 1u) { uniformWindowCellsFine(g, index); }
      else { uniformWindowCells(g, header, index); }
    }
    workgroupBarrier();
    let base = 4 * g + vec3i(lane);
    if (header.w == 0u) { classifyCube(base); }
    else if (!quiet && all(vec3u(base) < vec3u(u.gridInfo.xyz) + vec3u(1u))) {
      // The cube's corners in marching-cubes order.
      let i = lane.x + 5u * (lane.y + 5u * lane.z);
      var value = array<f32, 8>(windowCell[i], windowCell[i + 1u], windowCell[i + 6u], windowCell[i + 5u],
        windowCell[i + 25u], windowCell[i + 26u], windowCell[i + 31u], windowCell[i + 30u]);
      if (cubeHoldsSurface(&value)) { appendSurfaceCube(base, 1u, &value, uniformCubeMixed(base, header.y)); }
    }
  }
}

// Coarse extraction remains complete outside detail cores. A fine support
// halo deliberately overlaps the coarse mesh around every core: the coarse
// and fine cell-centred lattices do not share vertices, so handing ownership
// off at the outer edge of any resident page can leave a visible T-junction.
// Keeping coarse cubes through the halo gives the depth pass continuous
// coverage while the core still receives the independently transported detail.
@compute @workgroup_size(4, 4, 4)
fn extractHybridCoarseMain(@builtin(global_invocation_id) gid: vec3u) {
  udrInit();
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

// The listed windows' mixed cubes move from the worklist's end to behind the
// others and the count takes them: [0, activeCubeCount) is the worklist every
// reader has, in class order, so a polygonise workgroup takes one class path
// (bar the one on the boundary). One workgroup: the count waits on the moves.
@compute @workgroup_size(${EXTRACTION_ORDER_WORKGROUP})
fn orderSurfaceWorklistMain(@builtin(local_invocation_index) index: u32) {
  let capacity = arrayLength(&activeCubes);
  let mixed = atomicLoad(&drawArgs.mixedCubeCount);
  let front = min(atomicLoad(&drawArgs.activeCubeCount), capacity);
  let back = min(mixed, capacity - front);
  // The cubes already in [front, front + back) stay.
  let moved = min(back, capacity - front - back);
  for (var k = index; k < moved; k += ${EXTRACTION_ORDER_WORKGROUP}u) { activeCubes[front + k] = activeCubes[capacity - 1u - k]; }
  storageBarrier();
  if (index == 0u) { atomicAdd(&drawArgs.activeCubeCount, mixed); atomicStore(&drawArgs.mixedCubeCount, 0u); }
}

// One invocation per resident fine voxel. A lattice cube with base b is owned
// by fine cell clamp(b - 1, 0, dims - 1), so every ordinary cube has one base
// at q + 1 and a cell on a low domain boundary additionally owns base 0. The
// Cartesian product is important: it includes wall edges, floor strips, and
// triple corners as well as face interiors. The former face-only clauses left
// optical pinholes wherever a sparse detail core reached two domain edges.
@compute @workgroup_size(256)
fn extractSparseMain(@builtin(global_invocation_id) gid: vec3u) {
  udrInit();
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
  udrInit();
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
  udrInit();
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
struct Out { @builtin(position) clip:vec4f, @location(0) world:vec3f, @location(1) normal:vec3f, @location(2) film:f32 }
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
  let v=vertices[index]; var o:Out; o.clip=project(v.position.xyz);o.world=v.position.xyz;o.normal=normalize(v.normal.xyz);o.film=v.normal.w;
  // A closed liquid/wall silhouette is shared by a front-facing free-surface
  // triangle and a back-facing wall triangle. The raster top-left rule can
  // otherwise give their exact shared edge to the back pass alone at one
  // pixel. Expand only front-facing triangle coverage by one conservative
  // raster pixel; the 0.75-pixel margin still left a reproducible back-only
  // wall-corner sample in the reverse Dawn view at t=0.368 s. Back faces
  // remain culled, so actual holes are still visible to the strict
  // back-without-front smoke oracle.
  if(interfaceCoverageExpansionPixels>0.0&&v.normal.w<0.5){
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
  // Store the continuous wall-film thickness in normal alpha. RGB is
  // premultiplied by the same value so filtered coverage recovery still
  // returns a unit normal. Position alpha remains ordinary coverage.
  let encodedFilm=1.0+max(input.film,0.0);
  var o:SurfaceOut;o.position=vec4f(input.world,1.0);o.normal=vec4f(normalize(input.normal)*encodedFilm,encodedFilm);return o;
}
`;

/**
 * Direct view of the extracted triangle topology.
 *
 * WebGPU has no polygon line mode, so each non-indexed triangle carries an
 * analytic barycentric coordinate and the fragment stage keeps a one-pixel
 * neighbourhood of its edges. The dry-scene linear depth remains authoritative
 * for occlusion, which keeps back-wall and buried liquid triangles from showing
 * through the tank while leaving genuine holes unmistakable.
 */
export const surfaceWireframeShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f }
struct SurfaceVertex { position:vec4f, normal:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> vertices:array<SurfaceVertex>;
@group(1) @binding(0) var dryScene:texture_2d<f32>;
${cameraApertureShaderLibrary("u")}
struct Out {
  @builtin(position) clip:vec4f,
  @location(0) world:vec3f,
  @location(1) barycentric:vec3f,
}
fn project(world:vec3f)->vec4f {
  let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  let up=normalize(cross(right,forward));
  let relative=world-u.cameraPosition.xyz;
  let depth=max(dot(relative,forward),0.001);
  let aspect=u.viewport.x/max(u.viewport.y,1.0);
  let aperture=cameraTanHalfFov();
  let ndc=vec2f(dot(relative,right)/(depth*aspect*aperture),dot(relative,up)/(depth*aperture));
  return vec4f(ndc*depth,clamp(depth/50.0,0.0,1.0)*depth,depth);
}
@vertex fn wireVertex(@builtin(vertex_index) index:u32)->Out {
  let v=vertices[index];
  let corner=index%3u;
  var o:Out;
  o.clip=project(v.position.xyz);
  o.world=v.position.xyz;
  o.barycentric=select(select(vec3f(0.0,0.0,1.0),vec3f(0.0,1.0,0.0),corner==1u),vec3f(1.0,0.0,0.0),corner==0u);
  return o;
}
@fragment fn wireFragment(input:Out)->@location(0) vec4f {
  let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
  let surfaceDepth=dot(input.world-u.cameraPosition.xyz,forward);
  let encodedDryDepth=textureLoad(dryScene,vec2i(input.clip.xy),0).a;
  let dryDepth=select(65504.0,encodedDryDepth,encodedDryDepth>0.0);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));
  if(dryDepth+max(.0015,.18*cellSize)<surfaceDepth){discard;}
  let edge=min(input.barycentric.x,min(input.barycentric.y,input.barycentric.z));
  let width=max(fwidth(edge),1e-5);
  let coverage=1.0-smoothstep(.05*width,.55*width,edge);
  if(coverage<=.01){discard;}
  return vec4f(vec3f(.06,.86,.68),.82*coverage);
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
override wireframeOnly:f32=0.0;
override simpleSurface:f32=0.0;
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
// The mixed topology of Uniform Geometric's nodal level set (its tail is the
// detail table of a packed field); one word for every other method. With it
// the contact band samples phi as the extraction does, from the 4h vertex
// base (19) and the detail field (18: any texture while the solver holds no
// h-tile capacity), and liquidField is not read.
@group(0) @binding(17) var<storage,read> liquidTopology:array<u32>;
@group(0) @binding(18) var contactDetailPhi:texture_3d<f32>;
@group(0) @binding(19) var contactCoarsePhi:texture_3d<f32>;
${uniformMixedPresentationWGSL(17,"contactDetailPhi","contactCoarsePhi","liquidTopology")}
${waterSceneOpticsShaderLibrary(0, 15, 16)}
${cameraApertureShaderLibrary("u")}
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
fn project(world:vec3f)->vec2f{let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(f,vec3f(0,1,0)));let up=normalize(cross(r,f));let q=world-u.cameraPosition.xyz;let d=max(dot(q,f),1e-4);let aperture=cameraTanHalfFov();let ndc=vec2f(dot(q,r)/(d*u.viewport.x/max(u.viewport.y,1.0)*aperture),dot(q,up)/(d*aperture));return vec2f(ndc.x*.5+.5,.5-ndc.y*.5);}
fn safeSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{return textureSampleLevel(texture,linearSampler,clamp(uv,vec2f(.001),vec2f(.999)),0);}
// Interface positions deliberately live in unfilterable rgba32float targets.
// A half-float world coordinate has millimetre-scale ULPs at ordinary camera
// distances; subtracting two such independently rounded coordinates turns a
// sub-cell film thickness into visible contour bands. Preserve the full world
// position and reproduce bilinear filtering explicitly with textureLoad.
fn safePositionSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{
  let size=vec2i(textureDimensions(texture));
  let p=clamp(uv,vec2f(.001),vec2f(.999))*vec2f(size)-vec2f(.5);
  let base=vec2i(floor(p));let f=fract(p);let hi=size-vec2i(1);
  let s00=textureLoad(texture,clamp(base,vec2i(0),hi),0);
  let s10=textureLoad(texture,clamp(base+vec2i(1,0),vec2i(0),hi),0);
  let s01=textureLoad(texture,clamp(base+vec2i(0,1),vec2i(0),hi),0);
  let s11=textureLoad(texture,clamp(base+vec2i(1,1),vec2i(0),hi),0);
  let sampled=mix(mix(s00,s10,f.x),mix(s01,s11,f.x),f.y);
  if(sampled.a<=1e-4){return vec4f(0.0);}
  return vec4f(sampled.rgb/sampled.a,sampled.a);
}
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
fn recoveredWallFilm(positionSample:vec4f,normalSample:vec4f)->f32{
  return max(0.0,normalSample.a/max(positionSample.a,1e-4)-1.0);
}
fn cameraRay(textureUV:vec2f)->vec3f{let ndc=vec2f(textureUV.x*2.0-1.0,1.0-textureUV.y*2.0);let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let aperture=cameraTanHalfFov();return normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*aperture+up*ndc.y*aperture);}
${rigidBodyRaymarchShaderLibrary}
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

// The raster mesh is the fast global solution. Only pixels whose analytic
// rigid depth lies in this narrow band evaluate the resident implicit field.
fn contactOccupancyFromPhi(phi:f32)->f32{let band=4.0*u.container.y/max(u.gridInfo.y,1.0);return clamp(0.5-phi/band,0.0,1.0);}
fn contactFieldCell(cell:vec3i)->f32{
  let dims=vec3i(u.gridInfo.xyz);if(any(cell<vec3i(0))||any(cell>=dims)){return 0.0;}let mode=u.gridInfo.w;
  if(mode<1.5){
    if(all(textureDimensions(liquidField)==vec3u(dims)+1u)){
      var phi=0.0;for(var k=0u;k<8u;k++){phi+=textureLoad(liquidField,cell+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),0).x;}
      return 0.5-0.125*phi/(u.container.y/u.gridInfo.y);
    }
    return textureLoad(liquidField,cell,0).x;
  }if(mode>2.5){return contactOccupancyFromPhi(textureLoad(liquidField,cell,0).x);}
  let base=i32(round(textureLoad(tallCellBases,cell.xz,0).x));
  if(cell.y<base&&base>0){let t=clamp(f32(cell.y)/f32(max(base-1,1)),0.0,1.0);return contactOccupancyFromPhi(mix(textureLoad(liquidField,vec3i(cell.x,0,cell.z),0).x,textureLoad(liquidField,vec3i(cell.x,1,cell.z),0).x,t));}
  let packedY=2+cell.y-base;let stored=vec3i(textureDimensions(liquidField));if(packedY<2||packedY>=stored.y){return 0.0;}return contactOccupancyFromPhi(textureLoad(liquidField,vec3i(cell.x,packedY,cell.z),0).x);
}
fn contactFluidValue(world:vec3f)->f32{
  let dims=vec3i(u.gridInfo.xyz);let boundsMin=vec3f(-0.5*u.container.x,0,-0.5*u.container.z);let uvw=clamp((world-boundsMin)/u.container.xyz,vec3f(0),vec3f(1));
  if(umPresentationEnabled()){
    // Uniform Geometric: the nodal level set itself, on the scale the
    // extraction contours (0.5 - phi / h, its 0.5 crossing at phi = 0). The
    // cell field is centre phi there, canonical only at owner origins.
    return 0.5-umSampleVertex(uvw*vec3f(umDimensions()))/(u.container.y/max(u.gridInfo.y,1.0));
  }
  let q=clamp(uvw*vec3f(dims)-vec3f(0.5),vec3f(0),vec3f(dims-vec3i(1)));let base=vec3i(floor(q));let f=fract(q);
  let c000=contactFieldCell(base);let c100=contactFieldCell(base+vec3i(1,0,0));let c010=contactFieldCell(base+vec3i(0,1,0));let c110=contactFieldCell(base+vec3i(1,1,0));
  let c001=contactFieldCell(base+vec3i(0,0,1));let c101=contactFieldCell(base+vec3i(1,0,1));let c011=contactFieldCell(base+vec3i(0,1,1));let c111=contactFieldCell(base+vec3i(1,1,1));
  return mix(mix(mix(c000,c100,f.x),mix(c010,c110,f.x),f.y),mix(mix(c001,c101,f.x),mix(c011,c111,f.x),f.y),f.z);
}
struct ContactSurface { point:vec3f, normal:vec3f, valid:bool }
// A loop bound the compiler cannot fold (the lattice is never empty): the
// field has one evaluation site below, and Metal must not unroll it into one
// inlined sampler per stage and tap.
fn contactBound(n:u32)->u32{return n+u32(max(-u.gridInfo.x,0.0));}
// Stage 0 measures the raster crossing, stages 1..4 are Newton steps along
// the ray (value and central difference), stage 5 takes the gradient and the
// final value at the refined point.
fn refineContactSurface(ro:vec3f,rd:vec3f,rasterT:f32,cellSize:f32)->ContactSurface{
  let radius=1.35*cellSize;let lo=max(1e-4,rasterT-radius);let hi=rasterT+radius;var t=rasterT;
  let epsilon=max(2e-4,0.18*cellSize);let e=max(3e-4,0.3*cellSize);
  var initialError=0.0;var finalValue=0.0;var gradient=vec3f(0.0);var point=ro+rd*t;var settled=false;
  for(var stage=0u;stage<contactBound(6u);stage+=1u){
    let newton=stage>=1u&&stage<=4u;
    if(newton&&settled){continue;}
    point=ro+rd*t;
    var taps:array<vec3f,7>;var count=1u;taps[0]=point;
    if(newton){taps[1]=point+rd*epsilon;taps[2]=point-rd*epsilon;count=3u;}
    if(stage==5u){
      taps[1]=point+vec3f(e,0,0);taps[2]=point-vec3f(e,0,0);taps[3]=point+vec3f(0,e,0);taps[4]=point-vec3f(0,e,0);
      taps[5]=point+vec3f(0,0,e);taps[6]=point-vec3f(0,0,e);count=7u;
    }
    var value:array<f32,7>;
    for(var k=0u;k<min(count,contactBound(7u));k+=1u){value[k]=contactFluidValue(taps[k]);}
    if(stage==0u){initialError=abs(value[0]-0.5);}
    else if(newton){
      let derivative=(value[1]-value[2])/(2.0*epsilon);
      if(abs(derivative)<1e-5){settled=true;}else{t=clamp(t-(value[0]-0.5)/derivative,lo,hi);}
    }else{
      gradient=vec3f(value[1]-value[2],value[3]-value[4],value[5]-value[6])/(2.0*e);finalValue=value[0];
    }
  }
  let normal=select(-rd,-normalize(gradient),length(gradient)>1e-5);return ContactSurface(point,normal,initialError<0.42&&abs(finalValue-0.5)<0.12);
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
// A back face that lies on the dry scene's own solid is a liquid/basin contact,
// not a water-to-air interface. The mesh is extracted at solver-cell pitch, so
// refracting out through its facets displaced and tinted the refined floor per
// coarse facet (concentric terraces in a pond), and light does not leave an
// opaque basin anyway. No extraction path receives solid occupancy, so the
// contact is recognised here: the refined dry surface along the exit pixel's
// ray lies no more than ~a solver cell behind the back face, measured along the
// back normal so grazing views of a flat floor classify like head-on ones. A
// dry surface in front of the back face (a coarse contact buried in the solid)
// is a contact too; a sky miss never is.
fn backIsSolidContact(back:vec3f,exitN:vec3f,exitUV:vec2f,cellSize:f32)->bool{
  let dry=resolvedDrySceneDepth(safeSample(sceneTexture,exitUV).a);if(dry>60000.0){return false;}
  let ray=back-u.cameraPosition.xyz;let backDistance=length(ray);
  return (dry-backDistance)*abs(dot(ray/max(backDistance,1e-6),exitN))<=1.25*cellSize;
}
// Screen-space march of the refracted ray against the refined dry depth, as in
// SSR: eight linear steps over reach, then four bisections of the bracketing
// step; twelve dry-scene taps in all. Returns (uv, path length, valid). A
// crossing whose overshoot survives bisection is the ray passing behind a
// foreground occluder's silhouette rather than landing on a surface, and is
// refused like a march that leaves the screen or never crosses.
fn solidTerminatedHit(origin:vec3f,direction:vec3f,reach:f32)->vec4f{
  let ro=u.cameraPosition.xyz;let step=reach/8.0;var lo=0.0;var hi=-1.0;var overshoot=0.0;
  for(var i=1;i<=8;i+=1){let t=step*f32(i);let p=origin+direction*t;let uv=project(p);if(any(uv<vec2f(0.0))||any(uv>vec2f(1.0))){break;}let behind=length(p-ro)-resolvedDrySceneDepth(safeSample(sceneTexture,uv).a);if(behind>=0.0){hi=t;overshoot=behind;break;}lo=t;}
  if(hi<0.0){return vec4f(0.0);}
  for(var i=0;i<4;i+=1){let mid=.5*(lo+hi);let p=origin+direction*mid;let behind=length(p-ro)-resolvedDrySceneDepth(safeSample(sceneTexture,project(p)).a);if(behind>=0.0){hi=mid;overshoot=behind;}else{lo=mid;}}
  return vec4f(project(origin+direction*hi),hi,select(0.0,1.0,overshoot<=step));
}
// The first depth-tested pair describes only the nearest connected water
// interval. A breaking sheet can leave another interval behind it, so the
// interface raster peels one more front/back pair after the first exit. Shade
// that pair before the foreground interval consumes the transmitted radiance.
fn compositeRearWater(textureUV:vec2f,dryColor:vec3f)->vec3f{
  let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let rd=cameraRay(textureUV);
  let front=safePositionSample(rearFrontPosition,textureUV);if(front.a<.5){return dryColor;}
  let scene=safeSample(sceneTexture,textureUV);let frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));
  if(resolvedDrySceneDepth(scene.a)+max(.0015,.18*cellSize)<frontDepth){return dryColor;}
  let frontNormalSample=safeInterfaceSample(rearFrontNormal,textureUV);let filmDensity=recoveredWallFilm(front,frontNormalSample);
  var n=normalize(frontNormalSample.xyz);if(dot(n,rd)>0.0){n=-n;}
  var inside=refract(rd,n,1.0/waterIndexOfRefraction());if(length(inside)<1e-5){inside=reflect(rd,n);}
  var exitUV=textureUV;var back=vec4f(0);var exitN=vec3f(0,-1,0);
  for(var iteration=0;iteration<3;iteration+=1){back=safePositionSample(rearBackPosition,exitUV);if(back.a<.5){break;}let backNormalSample=safeInterfaceSample(rearBackNormal,exitUV);let backDepth=dot(back.xyz-ro,forward);let frontPlane=dot(front.xyz-ro,forward);let travel=max(0.0,(backDepth-frontPlane)/max(dot(inside,forward),.001));exitUV=project(front.xyz+inside*travel);exitN=normalize(backNormalSample.xyz);}
  let refinedBack=safePositionSample(rearBackPosition,exitUV);let refinedBackNormal=safeInterfaceSample(rearBackNormal,exitUV);if(refinedBack.a<.5){return dryColor;}back=refinedBack;exitN=normalize(refinedBackNormal.xyz);
  var thickness=length(back.xyz-front.xyz);if(thickness<1e-4){return dryColor;}
  let thinBoundaryFilm=filmDensity>1e-4;
  var outgoing=inside;var tir=false;var transmitted=dryColor;
  // A basin contact ends the light at the refined solid (backIsSolidContact).
  // Fail-safe when the march finds no surface: the unrefracted pixel, i.e.
  // dryColor, over its own dry depth, bounded by the march reach.
  if(backIsSolidContact(back.xyz,exitN,exitUV,cellSize)){let reach=1.5*thickness+4.0*cellSize;let hit=solidTerminatedHit(front.xyz,inside,reach);if(hit.w>.5){thickness=hit.z;transmitted=safeSample(sceneTexture,hit.xy).rgb*causticModulation(hit.xy);}else{thickness=clamp(resolvedDrySceneDepth(scene.a)-frontDepth,1e-4,reach);}}
  else{if(dot(exitN,inside)<0.0){exitN=-exitN;}outgoing=refract(inside,-exitN,waterIndexOfRefraction());tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}
  let backgroundUV=project(back.xyz+outgoing*(.55+.45*thickness));transmitted=safeSample(sceneTexture,backgroundUV).rgb*causticModulation(backgroundUV);}
  let refracted=unifiedAbsorbingTransmission(transmitted,waterAbsorption(),waterScatter(),thickness);
  let reflectedDir=reflect(rd,n);var reflected=environmentLight(reflectedDir);let ssr=safeSample(sceneTexture,project(front.xyz+reflectedDir*.8));reflected=mix(reflected,ssr.rgb,select(0.0,.32,ssr.a>0.0&&ssr.a<60000.0));
  let cosine=clamp(dot(-rd,n),0.0,1.0);let fresnel=unifiedDielectricFresnel(cosine,waterFresnelF0());var water=mix(refracted,reflected,fresnel);if(tir){water=mix(water,environmentLight(outgoing),.88);}
  water+=waterKeyColor()*unifiedSpecularLobe(n,-rd,waterKeyDirection(),180.0)*1.4;
  water+=vec3f(.018,.10,.085)*waterTintScale()*(1.0-exp(-thickness*2.4));water+=vec3f(.08,.18,.15)*waterTintScale()*pow(1.0-cosine,3.0)*.15;
  // The augmented wall value tends continuously to zero, but a rasterized
  // Fresnel lobe does not: its last covered pixel is still fully reflective.
  // Fade only the last few percent of represented film mass so the true dry
  // edge is antialiased in scalar-field space. Bulk water never takes this
  // branch; the response reaches the ordinary water answer before the film
  // hands ownership back at the half-cell isovalue.
  if(thinBoundaryFilm){water=mix(dryColor,water,smoothstep(.015,.25,filmDensity));}
  return water;
}
// Scenery is geometry, not a screen-space overlay: every frond, batten and
// blade that used to be painted here in NDC is now an analytic primitive in
// the scene's own scenery graph, so it parallaxes, occludes and takes light like the rest
// of the world. Only the lens falloff remains, which belongs to the camera.
fn finish(color:vec3f,ndc:vec2f)->vec4f{let c=color*(1.0-.08*dot(ndc*.55,ndc*.55));return vec4f(unifiedDisplayGradeBalanced(c,waterDisplayExposure(),waterDisplayToneCurve(),waterDisplayWhiteBalance()),1);}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  udrInit();
  // Full-screen interpolated UV has Y=1 at the top of the render target,
  // while sampled WebGPU textures have Y=0 there. The shared legacy upscaler
  // performs the same conversion for the final target; all raster-path
  // intermediate reads and world projections must do it here as well.
  let ndc=input.uv*2.0-1.0;let textureUV=vec2f(input.uv.x,1.0-input.uv.y);let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let aperture=cameraTanHalfFov();let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*aperture+up*ndc.y*aperture);
  let scene=safeSample(sceneTexture,textureUV);if(wireframeOnly>.5){return finish(scene.rgb,ndc);}var front=safePositionSample(frontPosition,textureUV);if(front.a<.5){return finish(scene.rgb,ndc);}var frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));let depthEpsilon=max(.0015,.18*cellSize);
  // Diagnostic material: preserve the actual mesh silhouette and normals, with
  // undistorted scenery showing through. Fixed diffuse lighting exposes folds
  // without environment reflections, specular highlights or optical contact repair.
  if(simpleSurface>.5){
    if(resolvedDrySceneDepth(scene.a)+depthEpsilon<frontDepth){return finish(scene.rgb,ndc);}
    var normal=normalize(safeInterfaceSample(frontNormal,textureUV).xyz);
    if(dot(normal,rd)>0.0){normal=-normal;}
    let diffuse=.38+.62*max(dot(normal,normalize(vec3f(-.45,.8,.35))),0.0);
    let rim=pow(1.0-clamp(dot(normal,-rd),0.0,1.0),2.0);
    let color=vec3f(.12,.48,.64)*diffuse;
    return finish(mix(scene.rgb,color,.58+.12*rim),ndc);
  }
  let frontNormalSample=safeInterfaceSample(frontNormal,textureUV);let filmDensity=recoveredWallFilm(front,frontNormalSample);var n=normalize(frontNormalSample.xyz);let rigidFront=nearestRigid(ro,rd);let contactBand=${CONTACT_RESOLVE_BAND_CELLS.toFixed(1)}*cellSize;
  if(u.gridInfo.w>.5&&rigidFront.t<1e19&&abs(rigidFront.t-frontDepth)<=contactBand){let contact=refineContactSurface(ro,rd,frontDepth,cellSize);if(contact.valid){front=vec4f(contact.point,1);frontDepth=dot(contact.point-ro,rd);n=contact.normal;}if(rigidFront.t<=frontDepth+max(3e-4,.03*cellSize)){return finish(scene.rgb,ndc);}}
  if(resolvedDrySceneDepth(scene.a)+depthEpsilon<frontDepth){return finish(scene.rgb,ndc);}
  if(dot(n,rd)>0.0){n=-n;}let etaIn=1.0/waterIndexOfRefraction();var inside=refract(rd,n,etaIn);if(length(inside)<1e-5){inside=reflect(rd,n);}
  var exitUV=textureUV;var back=vec4f(0);var exitN=vec3f(0,-1,0);
  for(var iteration=0;iteration<3;iteration+=1){back=safePositionSample(backPosition,exitUV);if(back.a<.5){break;}let backNormalSample=safeInterfaceSample(backNormal,exitUV);let backDepth=dot(back.xyz-ro,forward);let frontPlane=dot(front.xyz-ro,forward);let travel=max(0.0,(backDepth-frontPlane)/max(dot(inside,forward),.001));exitUV=project(front.xyz+inside*travel);exitN=normalize(backNormalSample.xyz);}
  let refinedBack=safePositionSample(backPosition,exitUV);let refinedBackNormal=safeInterfaceSample(backNormal,exitUV);if(refinedBack.a>.5){back=refinedBack;exitN=normalize(refinedBackNormal.xyz);}else{back=vec4f(0);}
  var exitPoint=back.xyz;var thickness=length(exitPoint-front.xyz);var meshExitValid=back.a>=.5&&thickness>=1e-4;
  let thinBoundaryFilm=filmDensity>1e-4;
  let innerStep=max(.0005,cellSize*.08);let innerOrigin=front.xyz+inside*innerStep;let rigidExit=nearestRigid(innerOrigin,inside);var opaqueSolidExit=false;
  if(rigidExit.t<1e19&&(!meshExitValid||rigidExit.t+innerStep<thickness)){opaqueSolidExit=true;exitPoint=innerOrigin+inside*rigidExit.t;thickness=length(exitPoint-front.xyz);}
  else if(!meshExitValid){
    // No analytic container fallback: the dry SVO owns solid depth. A missing
    // water back face is conservatively limited to one fluid cell.
    thickness=max(.002,cellSize);exitPoint=innerOrigin+inside*thickness;exitN=-inside;
  }
  var outgoing=inside;var tir=false;var backgroundUV=project(exitPoint);var transmittedScene=vec3f(0.0);
  // A liquid/basin back face (backIsSolidContact) is not refracted out: the
  // refracted ray ends where it meets the refined dry solid, and its path
  // length is the absorbing thickness. Nothing lies behind that opaque exit, so
  // no rear water interval is composited. Fail-safe when the march leaves the
  // screen or never crosses: the unrefracted pixel's colour over its own dry
  // depth, bounded by the march reach.
  if(!opaqueSolidExit&&meshExitValid&&backIsSolidContact(back.xyz,exitN,exitUV,cellSize)){
    let reach=1.5*thickness+4.0*cellSize;let hit=solidTerminatedHit(front.xyz,inside,reach);
    if(hit.w>.5){backgroundUV=hit.xy;thickness=hit.z;}else{backgroundUV=textureUV;thickness=clamp(resolvedDrySceneDepth(scene.a)-frontDepth,1e-4,reach);}
    transmittedScene=safeSample(sceneTexture,backgroundUV).rgb*causticModulation(backgroundUV);
  }else{
  if(!opaqueSolidExit){if(dot(exitN,inside)<0.0){exitN=-exitN;}outgoing=refract(inside,-exitN,waterIndexOfRefraction());tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}backgroundUV=project(exitPoint+outgoing*(.55+.45*thickness));}
  // The modulation goes on the *dry* term rather than on what comes back:
  // where a rear water interval exists it shades its own receiver and applies
  // its own caustic, and multiplying that result again would count the floor's
  // concentration twice through two layers of water.
  transmittedScene=compositeRearWater(backgroundUV,safeSample(sceneTexture,backgroundUV).rgb*causticModulation(backgroundUV));}
  // Absorption is the scene's, not the renderer's: the same clean-water rate
  // that turns a metre of water blue leaves a hand's breadth colourless.
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
  // At a glass/solid contact there is no air-to-water interface carrying the
  // full Fresnel lobe above. Fade that presentation term with represented film
  // mass so rho->0 is optically continuous with the dry pane instead of a
  // high-contrast painted ribbon. A half-cell film retains the ordinary water
  // answer and naturally merges into bulk rendering.
  if(thinBoundaryFilm){water=mix(transmittedScene,water,smoothstep(.015,.25,filmDensity));}
  return finish(water,ndc);
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
  /** Uniform Geometric: one thread per 4h window (extractTilesMain). */
  private collectWindowsPipeline?: GPUComputePipeline;
  private extractWindowsPipeline?: GPUComputePipeline;
  private orderWorklistPipeline?: GPUComputePipeline;
  /** Window list of the `windows` classify: a count word, then one word per
   * window of the lattice. */
  private surfaceWindows?: GPUBuffer;
  private extractionModule?: GPUShaderModule;
  private extractionPipelineLayout?: GPUPipelineLayout;
  /** QA (lanes and benchmarks): count-only classifiers, built on demand. */
  private countOnlyPipelines = new Map<WaterSurfaceClassify, GPUComputePipeline>();
  private extractBandPipeline?: GPUComputePipeline;
  private extractTallSidesPipeline?: GPUComputePipeline;
  private extractGlobalFinePipeline?: GPUComputePipeline;
  private extractGlobalFinePipelinePromise?: Promise<void>;
  private startGlobalSurfaceCompilation?: () => Promise<void>;
  private globalSurfaceCompilationError?: unknown;
  private extractGlobalCoarsePipeline?: GPUComputePipeline;
  /**
   * Compact-coarse extraction is a compatibility path for Losasso/Power
   * publications. Sparse CM12 never dispatches it, so retain its ingredients
   * and compile it only if a source that needs the fallback is attached.
   */
  private globalClassifyModule?: GPUShaderModule;
  private globalExtractionPipelineLayout?: GPUPipelineLayout;
  private extractGlobalCoarsePipelinePromise?: Promise<void>;
  private extractGlobalCoarsePipelineFailed = false;
  private globalCoarseCompilationError?: unknown;
  private preparePipeline?: GPUComputePipeline;
  private polygonisePipeline?: GPUComputePipeline;
  private prepareClassifyPipeline?: GPUComputePipeline;
  private prepareClassifyLayout?: GPUBindGroupLayout;
  private prepareClassifyBindGroup?: GPUBindGroup;
  private classifyDispatchBuffer?: GPUBuffer;
  private surfaceScanBlocks?: GPUBuffer;
  private surfaceScanWorkLayout?: GPUBindGroupLayout;
  private surfaceScanWorkBindGroup?: GPUBindGroup;
  private prepareSurfaceScanPipeline?: GPUComputePipeline;
  private countSurfaceBlocksPipeline?: GPUComputePipeline;
  private addSurfaceBlockOffsetsPipeline?: GPUComputePipeline;
  private polygoniseGlobalFineScanPipeline?: GPUComputePipeline;
  private polygoniseGlobalFineEmitPipeline?: GPUComputePipeline;
  private surfaceFrontPipeline?: GPURenderPipeline;
  private surfaceBackPipeline?: GPURenderPipeline;
  private surfaceRearFrontPipeline?: GPURenderPipeline;
  private surfaceRearBackPipeline?: GPURenderPipeline;
  private surfaceWireframePipeline?: GPURenderPipeline;
  private causticPipeline?: GPURenderPipeline;
  private compositePipeline?: GPURenderPipeline;
  private wireframeCompositePipeline?: GPURenderPipeline;
  private simpleCompositePipeline?: GPURenderPipeline;
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
  private denseNormalPhi?: GPUTexture;
  private mixedOwnership?: GPUBufferBinding;
  /** Uniform Geometric's 4h vertex base; with it the extraction binds no cell field. */
  private coarseVertexPhi?: GPUTexture;
  private levelSetSource?: DenseLevelSetVolumeConsumerSource;
  /** A 1^3 field for a 3D texture binding the bound source never loads. */
  private fallbackField?: GPUTexture;
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
  private wireframeSceneBindGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  private rigidSceneLayout?: GPUBindGroupLayout;
  private rigidScenePipeline?: GPURenderPipeline;
  private rigidSceneBindGroup?: GPUBindGroup;
  private rigidBodyCount = 0;
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
  surfaceExtractionReason = "startup";
  private extractionCount = 0;
  get surfaceExtractionCount(): number { return this.extractionCount; }
  private lastExtractionAt_ms = -Infinity;
  private causticsValid = false;
  private sceneHasFluid = true;
  private dryInterfaceClearsEncoded = false;
  private disabledStages: DisabledRenderStages = NO_DISABLED_RENDER_STAGES;
  private secondaryParticles?: SecondaryParticleRenderPipeline;
  private globalFineLevelSet?: GlobalFineLevelSetConsumerSource;
  private coarseLevelSet?: CoarseLevelSetConsumerSource;
  private fluidDomain?: FluidDomain;
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
    private readonly bodyBuffer: GPUBuffer,
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

  async initialize(
    onProgress:(label:string,completed:number,total:number)=>void=()=>{},
    options: { readonly deferSceneClassifiers?: boolean } = {},
  ) {
    const [extract, globalClassify, globalScan, globalEmitAll, prepare, surface, wireframe, caustic, composite, rigidScene] = await Promise.all([
      checkedModule(this.device, "Water isosurface extraction", surfaceExtractionShader),
      checkedModule(this.device, "Global fine water classification", globalFineSurfaceClassificationShader),
      checkedModule(this.device, "Classified global fine scan", parallelSurfaceScanShader),
      checkedModule(this.device, "Classified global fine adaptive contour", globalFineClassifiedEmitShader),
      checkedModule(this.device, "Water extraction dispatch prepare", extractionPrepareShader),
      checkedModule(this.device, "Water interface raster", surfaceRasterShader),
      checkedModule(this.device, "Water surface wireframe", surfaceWireframeShader),
      checkedModule(this.device, "Water caustic projection", causticShader),
      checkedModule(this.device, "Water optical composite", compositeShader),
      checkedModule(this.device, "Fluid-only rigid bodies", fluidOnlyRigidSceneShader)
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
      ,{ binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ,{ binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 15, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      // The tenth storage buffer of the stage: the ceiling browsers report here.
      ,{ binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
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
    const scanEntries: GPUBindGroupLayoutEntry[] = [
      ...[3, 4, 7, 18].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      ...[5, 6].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ];
    this.surfaceScanWorkLayout = this.device.createBindGroupLayout({ label: "Parallel surface scan work", entries: scanEntries });
    this.globalPolygoniseLayout = this.device.createBindGroupLayout({ label: "Surface scan dispatch writer", entries: [
      ...scanEntries, { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.prepareClassifyLayout = this.device.createBindGroupLayout({ label: "Active surface pages dispatch", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
    // no-op. 15 and 16 are the scene's own optics and caustic receiver; 17 is
    // the mixed topology whose detail table addresses a packed liquid field.
    this.compositeLayout = this.device.createBindGroupLayout({ label: "Water composite bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ...[1,3,5,11,13,14].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } })),
      ...[2,4,10,12].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" as const } })),
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      { binding: 15, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      { binding: 17, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ...[18,19].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } }))
    ] });
    // Binding 3 is the caustic receiver the optics library declares and this
    // pass never reads. It stays in the layout because the library is included
    // whole: the alternative is a second copy of it that differs only by what
    // it leaves out.
    this.rigidSceneLayout = this.device.createBindGroupLayout({ label: "Fluid-only rigid body bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }
    ] });
    this.causticLayout = this.device.createBindGroupLayout({ label: "Water caustic projection bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "unfilterable-float" } }
    ] });
    const extractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.extractLayout] });
    const globalExtractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.globalExtractLayout] });
    this.globalClassifyModule = globalClassify;
    this.globalExtractionPipelineLayout = globalExtractionPipelineLayout;
    const globalPolygonScanLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.globalPolygoniseLayout]});
    const globalPolygonEmitLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.globalPolygoniseEmitLayout]});
    this.polygoniseDispatchBuffer = this.device.createBuffer({ label: "Water polygonise dispatch arguments", size: 24, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.classifyDispatchBuffer = this.device.createBuffer({ label: "Active surface pages indirect dispatch", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const surfacePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.surfaceLayout, this.surfacePeelLayout] });
    const causticPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.causticLayout] });
    const surfaceDescriptor = (label: string, cullMode: GPUCullMode, coverageExpansionPixels = 0, peel = false): GPURenderPipelineDescriptor => ({
      label, layout: surfacePipelineLayout, vertex: { module: surface, entryPoint: "surfaceVertex", constants: { interfaceCoverageExpansionPixels: coverageExpansionPixels } },
      fragment: { module: surface, entryPoint: "surfaceFragment", constants: { peelBehindFirstExit: peel ? 1 : 0 }, targets: [{ format: "rgba32float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", frontFace: "ccw", cullMode },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    const compositePipelineLayout=this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] });
    const compositeDescriptor:GPURenderPipelineDescriptor={ label:"Composite layered water optics", layout: compositePipelineLayout, vertex: { module: composite, entryPoint: "vertexMain" }, fragment: { module: composite, entryPoint: "fragmentMain", targets: [{ format: this.targetFormat }] }, primitive: { topology: "triangle-list" } };

    // Keep adaptive-water specialization off dry-scene startup. The first
    // global fine/coarse source starts this group; encode withholds that source
    // until its complete GPU extraction chain is available.
    const compileGlobalFine = () => this.device.createComputePipelineAsync({
      label: "Classify global fine surface bricks",
      layout: globalExtractionPipelineLayout,
      compute: { module: globalClassify, entryPoint: "extractGlobalFineMain" },
    }).then((pipeline) => {
      this.extractGlobalFinePipeline = pipeline;
      this.extractedRevision = -1; this.surfaceExtractionReason = "fine pipeline ready";
    });
    const globalFineCompilation = options.deferSceneClassifiers ? undefined : compileGlobalFine();
    this.extractGlobalFinePipelinePromise = globalFineCompilation;
    const globalCoarseCompilation = options.deferSceneClassifiers
      ? undefined
      : this.device.createComputePipelineAsync({
        label: "Classify compact coarse fallback",
        layout: globalExtractionPipelineLayout,
        compute: { module: globalClassify, entryPoint: "extractGlobalCoarseMain" },
      }).then((pipeline) => {
        this.extractGlobalCoarsePipeline = pipeline;
      });

    // Feed the remaining independent descriptors to the native slots not
    // already occupied by scene classifiers, so raw headless devices and the
    // managed browser device both stay at or below the proven width of three.
    type CompileJob = { readonly label: string; readonly run: () => Promise<void> };
    const jobs: CompileJob[] = [];
    const compute = (label:string, descriptor:GPUComputePipelineDescriptor,
      accept:(pipeline:GPUComputePipeline)=>void) => jobs.push({ label, run: async () => {
        accept(await this.device.createComputePipelineAsync(descriptor));
      } });
    const render = (label:string, descriptor:GPURenderPipelineDescriptor,
      accept:(pipeline:GPURenderPipeline)=>void) => jobs.push({ label, run: async () => {
        accept(await this.device.createRenderPipelineAsync(descriptor));
      } });

    const scanWorkLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.surfaceScanWorkLayout] });
    const classifyDispatchModule = await checkedModule(this.device, "Active surface pages dispatch", surfaceClassifyDispatchShader);
    compute("Preparing active surface pages", { label: "Prepare active surface pages", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareClassifyLayout] }), compute: { module: classifyDispatchModule, entryPoint: "prepareClassify" } }, pipeline => { this.prepareClassifyPipeline = pipeline; });
    compute("Preparing surface scan", { label: "Prepare surface scan", layout: globalPolygonScanLayout, compute: { module: globalScan, entryPoint: "prepareSurfaceScan" } }, pipeline => { this.prepareSurfaceScanPipeline = pipeline; });
    compute("Counting surface blocks", { label: "Count surface blocks", layout: scanWorkLayout, compute: { module: globalScan, entryPoint: "countSurfaceBlocks" } }, pipeline => { this.countSurfaceBlocksPipeline = pipeline; });
    compute("Scanning surface block totals", { label: "Scan surface block totals", layout: globalPolygonScanLayout, compute: { module: globalScan, entryPoint: "scanSurfaceBlocks" } }, pipeline => { this.polygoniseGlobalFineScanPipeline = pipeline; });
    compute("Adding surface block offsets", { label: "Add surface block offsets", layout: scanWorkLayout, compute: { module: globalScan, entryPoint: "addSurfaceBlockOffsets" } }, pipeline => { this.addSurfaceBlockOffsetsPipeline = pipeline; });
    compute("Emitting adaptive global fine contour",{label:"Emit classified adaptive global fine contour",layout:globalPolygonEmitLayout,compute:{module:globalEmitAll,entryPoint:"emitGlobalFineTetrahedra"}},pipeline=>{this.polygoniseGlobalFineEmitPipeline=pipeline;});
    if (options.deferSceneClassifiers) {
      const globalJobs = jobs.splice(0);
      this.startGlobalSurfaceCompilation = async () => {
        await Promise.all([compileGlobalFine(), ...globalJobs.map(job => job.run())]);
        this.extractedRevision = -1; this.surfaceExtractionReason = "surface pipelines ready";
      };
    }
    compute("Classifying liquid surface cubes",{ label: "Classify liquid surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractMain" } },pipeline=>{this.extractPipeline=pipeline;});
    compute("Listing liquid surface windows",{ label: "List liquid surface windows", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "collectWindowsMain" } },pipeline=>{this.collectWindowsPipeline=pipeline;});
    compute("Classifying liquid surface windows",{ label: "Classify listed liquid surface windows", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractWindowsMain" } },pipeline=>{this.extractWindowsPipeline=pipeline;});
    compute("Ordering the surface worklist",{ label: "Order the liquid surface worklist", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "orderSurfaceWorklistMain" } },pipeline=>{this.orderWorklistPipeline=pipeline;});
    this.extractionModule = extract; this.extractionPipelineLayout = extractionPipelineLayout;
    compute("Classifying restricted water band",{ label: "Classify restricted water band", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractBandMain" } },pipeline=>{this.extractBandPipeline=pipeline;});
    compute("Classifying tall-cell interfaces",{ label: "Classify tall-cell side interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractTallSidesMain" } },pipeline=>{this.extractTallSidesPipeline=pipeline;});
    compute("Building water surface mesh",{ label: "Polygonise surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain" } },pipeline=>{this.polygonisePipeline=pipeline;});
    compute("Preparing surface dispatch",{ label: "Prepare polygonise dispatch", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareLayout] }), compute: { module: prepare, entryPoint: "prepareMain" } },pipeline=>{this.preparePipeline=pipeline;});
    render("Rendering front water interfaces",surfaceDescriptor("Raster water front interfaces", WATER_INTERFACE_CULL_MODES.front),pipeline=>{this.surfaceFrontPipeline=pipeline;});
    render("Rendering back water interfaces",surfaceDescriptor("Raster water back interfaces", WATER_INTERFACE_CULL_MODES.back),pipeline=>{this.surfaceBackPipeline=pipeline;});
    render("Peeling rear front water interfaces",surfaceDescriptor("Raster water rear front interfaces", WATER_INTERFACE_CULL_MODES.front,0,true),pipeline=>{this.surfaceRearFrontPipeline=pipeline;});
    render("Peeling rear back water interfaces",surfaceDescriptor("Raster water rear back interfaces", WATER_INTERFACE_CULL_MODES.back,0,true),pipeline=>{this.surfaceRearBackPipeline=pipeline;});
    render("Rendering water surface wireframe",{
      label:"Raster water surface wireframe",layout:surfacePipelineLayout,
      vertex:{module:wireframe,entryPoint:"wireVertex"},
      fragment:{module:wireframe,entryPoint:"wireFragment",targets:[{format:this.targetFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha"}}}]},
      primitive:{topology:"triangle-list",frontFace:"ccw",cullMode:"back"},
      depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:"equal"},
    },pipeline=>{this.surfaceWireframePipeline=pipeline;});
    render("Projecting water caustics",{
      label: "Project refracted caustics", layout: causticPipelineLayout, vertex: { module: caustic, entryPoint: "causticVertex" },
      fragment: { module: caustic, entryPoint: "causticFragment", targets: [{ format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    },pipeline=>{this.causticPipeline=pipeline;});
    render("Compositing water optics",compositeDescriptor,pipeline=>{this.compositePipeline=pipeline;});
    render("Compositing wireframe background",{
      ...compositeDescriptor,label:"Composite water wireframe background",
      fragment:{module:composite,entryPoint:"fragmentMain",constants:{wireframeOnly:1},targets:[{format:this.targetFormat}]},
    },pipeline=>{this.wireframeCompositePipeline=pipeline;});
    render("Compositing simple translucent water",{
      ...compositeDescriptor,label:"Composite simple translucent water",
      fragment:{module:composite,entryPoint:"fragmentMain",constants:{simpleSurface:1},targets:[{format:this.targetFormat}]},
    },pipeline=>{this.simpleCompositePipeline=pipeline;});
    render("Drawing rigid bodies",{
      label: "Fluid-only rigid bodies", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.rigidSceneLayout] }),
      vertex: { module: rigidScene, entryPoint: "vertexMain" },
      fragment: { module: rigidScene, entryPoint: "fragmentMain", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" }
    },pipeline=>{this.rigidScenePipeline=pipeline;});
    let completed=0;let next=0;const total=jobs.length;
    const compileWorker=async()=>{for(;;){const index=next;next+=1;const job=jobs[index];if(!job)return;
      onProgress(job.label,completed,total);await job.run();completed+=1;onProgress(job.label,completed,total);}};
    const classifierSlots = (globalCoarseCompilation ? 1 : 0) + (globalFineCompilation ? 1 : 0);
    const workerCount = Math.min(Math.max(1, 3 - classifierSlots), total);
    await Promise.all(Array.from({length:workerCount},()=>compileWorker()));
    // Direct/headless consumers retain initialize()'s historical fully-ready
    // contract. The interactive renderer opts into deferral because it owns a
    // retrying frame loop and can overlap this work with solver construction.
    if (!options.deferSceneClassifiers) {
      await Promise.all([globalFineCompilation, globalCoarseCompilation]);
    }
    this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.fallbackSparsePageTable = this.device.createBuffer({ label: "Water sparse-page fallback", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fallbackSparseActivePages = this.device.createBuffer({ label: "Water sparse-active fallback", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.fallbackSparsePhi = this.device.createBuffer({ label: "Water sparse-phi fallback", size: 4, usage: GPUBufferUsage.STORAGE });
    this.fallbackSparseParams = this.device.createBuffer({ label: "Water sparse-params fallback", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.globalFineRenderParams = this.device.createBuffer({ label: "Water global fine parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fallbackSparseControl = this.device.createBuffer({ label: "Water disabled storage binding", size: WATER_DISABLED_STORAGE_BYTES, usage: GPUBufferUsage.STORAGE });
    this.fallbackField = this.device.createTexture({ label: "Water unread field binding", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
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
    this.ensureGlobalCoarsePipeline();
  }

  /** source: the solver's nodal level set, when it publishes one. */
  setVolume(texture: GPUTexture, columnBases: GPUTexture, source?: DenseLevelSetVolumeConsumerSource) {
    const volumeChanged = this.volume !== texture || this.columnBases !== columnBases;
    if (!this.adoptLevelSetSource(source) && !volumeChanged) return;
    this.volume = texture; this.columnBases = columnBases;
    this.extractedRevision = -1; this.surfaceExtractionReason = volumeChanged ? "volume binding changed" : "normal phi binding changed";
    this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  /** Follows the solver's source every frame: its object changes identity
   * when a field it names is re-created or the detail field appears or goes
   * (the solver's h-tile capacity leaving or reaching zero). */
  setDenseLevelSetVolumeSource(source: DenseLevelSetVolumeConsumerSource | undefined) {
    if (!this.adoptLevelSetSource(source)) return;
    this.extractedRevision = -1; this.surfaceExtractionReason = "level-set source changed";
    this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  /** The mixed presentation (ownership and 4h base) binds the detail field
   * only while the solver names one; any other nodal source binds its
   * vertex field raw. */
  private adoptLevelSetSource(source: DenseLevelSetVolumeConsumerSource | undefined): boolean {
    if (source === this.levelSetSource) return false;
    const contourChanged = this.levelSetSource?.contourVertexPhi !== source?.contourVertexPhi;
    this.levelSetSource = source;
    const mixed = source?.mixedOwnership && source.coarseVertexPhi ? source : undefined;
    const detail = mixed ? mixed.detailVertexPhi : source?.vertexPhi, ownership = mixed?.mixedOwnership, coarse = mixed?.coarseVertexPhi;
    if (!contourChanged && this.denseNormalPhi === detail && this.mixedOwnership === ownership && this.coarseVertexPhi === coarse) return false;
    this.denseNormalPhi = detail; this.mixedOwnership = ownership; this.coarseVertexPhi = coarse;
    return true;
  }

  setFluidDomain(domain: FluidDomain | undefined) {
    const previous = this.fluidDomain;
    if (previous && domain
      && previous.origin_m.every((value, axis) => value === domain.origin_m[axis])
      && previous.cellSize_m.every((value, axis) => value === domain.cellSize_m[axis])
      && previous.dimensions.every((value, axis) => value === domain.dimensions[axis])) return;
    if (!previous && !domain) return;
    this.fluidDomain = domain;
    this.writeCompactRenderParams();
    this.extractedRevision = -1; this.surfaceExtractionReason = "domain changed";
  }

  /**
   * Forget the retained water mesh after a live field edit.
   *
   * Solver steps carry their own monotonically increasing revision into
   * `encode`. Editor injections do not: they update the same GPU buffers
   * between steps, so a paused renderer would otherwise keep drawing the mesh
   * extracted before the edit. Invalidating here makes the next presentation
   * extract exactly once from the newly published field.
   */
  invalidateSurface() {
    this.extractedRevision = -1; this.surfaceExtractionReason = "field edit";
    this.lastExtractionAt_ms = -Infinity;
    this.causticsValid = false;
  }

  /** The dense classifier in use: the window scan whenever the mixed
   * presentation is bound (its window test needs the 4h base), the full
   * lattice scan for every other dense source. */
  private get surfaceClassify(): WaterSurfaceClassify {
    return this.surfaceClassifyForQA ?? (this.mixedOwnership && this.coarseVertexPhi ? "windows" : "full");
  }
  private surfaceClassifyForQA?: WaterSurfaceClassify;
  /** QA: force a dense classifier (the lanes and the benchmark compare them). */
  setSurfaceClassifyForQA(classify?: WaterSurfaceClassify) { this.surfaceClassifyForQA = classify; this.invalidateSurface(); }

  /** full: one workgroup per 4^3 cube bases of the (n+1)^3 lattice.
   * windows: one lane per window lists it (the caller has zeroed the list's
   * count word), a fixed launch shares the list out, a workgroup a window at
   * a time, then one workgroup puts the mixed cubes last in the worklist. */
  private dispatchDenseClassify(pass: GPUComputePassEncoder, nx: number, ny: number, nz: number, classify: WaterSurfaceClassify, countOnly: boolean, classifier?: GPUComputePipeline) {
    const windows = [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)] as const;
    const pipeline = classifier ?? (countOnly ? this.countOnlyPipeline(classify)
      : classify === "windows" ? this.extractWindowsPipeline! : this.extractPipeline!);
    if (classify !== "windows") { pass.setPipeline(pipeline); pass.dispatchWorkgroups(...windows); return; }
    pass.setPipeline(this.collectWindowsPipeline!);
    pass.dispatchWorkgroups(Math.ceil(windows[0] / 4), Math.ceil(windows[1] / 4), Math.ceil(windows[2] / 4));
    pass.setPipeline(pipeline); pass.dispatchWorkgroups(Math.min(EXTRACTION_WINDOW_WORKGROUPS, windows[0] * windows[1] * windows[2]));
    if (!countOnly) { pass.setPipeline(this.orderWorklistPipeline!); pass.dispatchWorkgroups(1); }
  }

  private countOnlyPipeline(classify: WaterSurfaceClassify): GPUComputePipeline {
    const pipeline = this.countOnlyPipelines.get(classify);
    if (!pipeline) throw new Error("Count-only extraction needs prepareSurfaceCountForQA()");
    return pipeline;
  }

  /** QA: compile the count-only classifiers (the countOnly override). */
  async prepareSurfaceCountForQA(): Promise<void> {
    if (!this.extractionModule || !this.extractionPipelineLayout) throw new Error("Water extraction is not initialized");
    for (const classify of ["full", "windows"] as const) {
      if (this.countOnlyPipelines.has(classify)) continue;
      this.countOnlyPipelines.set(classify, await this.device.createComputePipelineAsync({ label: `Count liquid surface vertices (${classify})`, layout: this.extractionPipelineLayout,
        compute: { module: this.extractionModule, entryPoint: WATER_SURFACE_CLASSIFY_ENTRY[classify], constants: { countOnly: 1 } } }));
    }
  }

  /** QA: the dense extraction chain alone, with no raster pass and no
   * presentation state (lanes and benchmarks; the renderer uses encode).
   * countOnly: the classifier adds every surface cube's vertex count to the
   * draw count and builds no worklist or mesh (uncapped, so exact).
   * Otherwise classify, prepare and polygonise run as encode runs them.
   * The caller owns the uniform (container and gridInfo) and the source.
   * classifier, polygoniser: a benchmark's own classify (the full-lattice
   * scan, or the listed-window scan after the production window list) and
   * polygonise pipelines on denseExtractionLayoutForQA, dispatched as the
   * production ones are. */
  encodeDenseSurfaceExtractionForQA(encoder: GPUCommandEncoder, nx: number, ny: number, nz: number, classify: WaterSurfaceClassify, countOnly = false, classifier?: GPUComputePipeline, polygoniser?: GPUComputePipeline): void {
    this.ensureGeometry(nx, ny, nz);
    if (!this.extractBindGroup || !this.prepareBindGroup || !this.indirectBuffer || !this.surfaceWindows || !this.preparePipeline || !this.polygonisePipeline || !this.extractPipeline || !this.collectWindowsPipeline || !this.extractWindowsPipeline || !this.orderWorklistPipeline || !this.polygoniseDispatchBuffer)
      throw new Error("Water extraction is not initialized");
    const indirectReset = this.indirectResetTemplate ??= this.createIndirectResetTemplate();
    encoder.copyBufferToBuffer(indirectReset, 32, this.indirectBuffer, 0, 36);
    encoder.copyBufferToBuffer(indirectReset, 32, this.surfaceWindows, 0, 4);
    let pass = encoder.beginComputePass({ label: `Extract water isosurface (${classify}${countOnly ? ", count" : ""})` });
    pass.setBindGroup(0, this.extractBindGroup);
    this.dispatchDenseClassify(pass, nx, ny, nz, classify, countOnly, classifier);
    if (!countOnly) {
      pass.setPipeline(this.preparePipeline); pass.setBindGroup(0, this.prepareBindGroup); pass.dispatchWorkgroups(1);
      pass.end();
      pass = encoder.beginComputePass({ label: "Polygonise water isosurface" });
      pass.setPipeline(polygoniser ?? this.polygonisePipeline); pass.setBindGroup(0, this.extractBindGroup); pass.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
    }
    pass.end();
    this.extractedRevision = -1;
  }

  /** QA: the extraction pipeline layout, for a benchmark's own classifier. */
  get denseExtractionLayoutForQA(): GPUPipelineLayout {
    if (!this.extractionPipelineLayout) throw new Error("Water extraction is not initialized");
    return this.extractionPipelineLayout;
  }

  /** QA: the last extraction's counters, worklist and mesh, read back. */
  async readDenseSurfaceExtractionForQA(): Promise<{ vertexCount: number; activeCubeCount: number; vertexAllocator: number; activeCubes: Uint32Array; vertices: Float32Array }> {
    if (!this.indirectBuffer || !this.activeCubeBuffer || !this.vertexBuffer) throw new Error("Water extraction is not initialized");
    const read = async (source: GPUBuffer, bytes: number) => {
      const staging = this.device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        if (bytes > 0) { const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, staging, 0, bytes); this.device.queue.submit([encoder.finish()]); }
        await staging.mapAsync(GPUMapMode.READ); return staging.getMappedRange().slice(0, bytes);
      } finally { if (staging.mapState === "mapped") staging.unmap(); staging.destroy(); }
    };
    const header = new Uint32Array(await read(this.indirectBuffer, 32));
    const vertexCount = header[0]!, activeCubeCount = header[4]!, vertexAllocator = header[5]!;
    const cubes = Math.min(activeCubeCount, this.activeCubeBuffer.size / 8), vertices = Math.min(vertexCount, this.vertexBuffer.size / 32);
    return { vertexCount, activeCubeCount, vertexAllocator, activeCubes: new Uint32Array(await read(this.activeCubeBuffer, 8 * cubes)), vertices: new Float32Array(await read(this.vertexBuffer, 32 * vertices)) };
  }

  private needsGlobalCoarsePipeline(): boolean {
    return Boolean(this.globalFineLevelSet?.coarsePhiRowCapacity)
      || Boolean(this.coarseLevelSet);
  }

  /**
   * Compile the compact-coarse compatibility classifier on first use.
   *
   * Sparse CM12 publishes complete signed sparse pages, so it has neither a
   * compact coarse-only source nor a coarse-phi complement. Keeping this very
   * expensive Metal specialization out of initialize() removes work that can
   * never contribute to a CM12 frame.
   */
  private ensureGlobalCoarsePipeline() {
    if (!this.needsGlobalCoarsePipeline()
      || this.extractGlobalCoarsePipeline
      || this.extractGlobalCoarsePipelinePromise
      || this.extractGlobalCoarsePipelineFailed
      || !this.globalClassifyModule
      || !this.globalExtractionPipelineLayout) return;
    this.extractGlobalCoarsePipelinePromise = this.device.createComputePipelineAsync({
      label: "Classify compact coarse fallback",
      layout: this.globalExtractionPipelineLayout,
      compute: { module: this.globalClassifyModule, entryPoint: "extractGlobalCoarseMain" },
    }).then((pipeline) => {
      this.extractGlobalCoarsePipeline = pipeline;
      this.extractGlobalCoarsePipelinePromise = undefined;
      // A source may have tried to present while compilation was outstanding.
      // Force the next frame to extract rather than retaining an old mesh.
      this.extractedRevision = -1; this.surfaceExtractionReason = "coarse pipeline ready";
    }).catch((error: unknown) => {
      this.extractGlobalCoarsePipelinePromise = undefined;
      this.extractGlobalCoarsePipelineFailed = true;
      this.globalCoarseCompilationError = error;
      console.error("Failed to compile compact-coarse water classification", error);
    });
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
    if (sameBindings && previous.generation === source.generation
      && previous.surfaceMeshRefinement === source.surfaceMeshRefinement) return;
    this.globalFineLevelSet = source;
    this.writeCompactRenderParams();
    this.extractedRevision = -1; this.surfaceExtractionReason = "fine publication changed"; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false;
    // Keep same-shaped geometry alive across A/B source publication. The next
    // encode still calls ensureGeometry(), so a genuine dimension change
    // reallocates; clearing the key here destroyed A before B could prove its
    // tags and defeated the fail-closed retained-mesh contract.
    if (!sameBindings) this.rebuildBindGroups();
    this.prepareSurfacePipelines();
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
    this.extractedRevision = -1; this.surfaceExtractionReason = "coarse publication changed"; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false;
    // Factor-one coarse publication rewrites stable directory/control arenas.
    // A new generation changes only the uniform and extraction invalidation;
    // rebuilding every water bind group here added nine host allocations to
    // every presented frame without changing any bound resource identity.
    if (!sameBindings) this.rebuildBindGroups();
    this.prepareSurfacePipelines();
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
      f32[27] = fine.surfaceMeshRefinement ?? 0;
    } else {
      const dimensions = coarse!.sampleDimensions;
      u32.set([...dimensions, 4], 0);
      u32.set(dimensions.map((value) => Math.ceil(value / 4)).concat(64), 4);
      // Table state 6 is the renderer-private compact-coarse publication mode.
      u32.set([1, 6, 1, coarse!.generation], 8);
      f32.set([...coarse!.domainOrigin, coarse!.physicalCellSize], 12); f32[16] = 1;
    }
    if (this.fluidDomain) {
      f32.set(this.fluidDomain.cellSize_m, 20);
      const presentationOrigin = fine
        ? this.fluidDomain.origin_m.map((value, axis) =>
          value + fine.domainOrigin[axis]!)
        : this.fluidDomain.origin_m;
      f32.set(presentationOrigin, 24);
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
          : stageKey === "back-interfaces" ? this.backNormal
            : stageKey === "rear-interface-positions" ? this.rearFrontPosition
              : stageKey === "rear-interfaces" ? this.rearFrontNormal
                : stageKey === "rear-back-interface-positions" ? this.rearBackPosition
                  : stageKey === "rear-back-interfaces" ? this.rearBackNormal
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
    const template = this.device.createBuffer({ label: "Water indirect header reset patterns", size: 68, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // The dense pattern's ninth word is the mixed cube count.
    this.device.queue.writeBuffer(template, 0, new Uint32Array([
      1, 0, 0, 0, 0xffff_ffff, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0xffff_ffff, 0,
    ]));
    return template;
  }

  private ensureGeometry(nx: number, ny: number, nz: number,
    sparseMaxVertices?: number) {
    const key = `${nx}x${ny}x${nz}:${sparseMaxVertices ?? "dense"}`;
    if (key === this.geometryKey) return;
    this.vertexBuffer?.destroy(); this.indirectBuffer?.destroy(); this.indirectResetTemplate?.destroy(); this.activeCubeBuffer?.destroy(); this.globalCubeValues?.destroy(); this.globalCubeOffsets?.destroy(); this.surfaceScanBlocks?.destroy(); this.surfaceWindows?.destroy();
    // Surface area, not volume, controls the normal case.  The generous factor
    // also covers breaking sheets and entrained blobs while imposing a hard
    // 64 MiB ceiling on adversarial checkerboard fields.
    const maxVertices = sparseMaxVertices ?? surfaceVertexCapacity(nx, ny, nz);
    this.vertexBuffer = this.device.createBuffer({ label: `Extracted water surface (${maxVertices} vertices)`, size: maxVertices * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    // The first 16 bytes are the standard draw-indirect ABI. Renderer-private
    // counters, global-fine authority latch, GPU-published mesh generation and
    // the listed windows' mixed cube count trail it; firstInstance must stay
    // zero unless the optional indirect-first-instance feature is enabled.
    this.indirectBuffer = this.device.createBuffer({ label: "Water indirect draw arguments and extraction counters", size: 36, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(this.indirectBuffer, 28, new Uint32Array([0xffff_ffff]));
    this.indirectResetTemplate = this.createIndirectResetTemplate();
    this.activeCubeBuffer = this.device.createBuffer({ label: "Water surface cube worklist", size: activeCubeCapacity(maxVertices) * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.globalCubeValues = this.device.createBuffer({ label: "Global fine classified cube values", size: activeCubeCapacity(maxVertices) * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.globalCubeOffsets = this.device.createBuffer({ label: "Global fine contour offsets", size: activeCubeCapacity(maxVertices) * GLOBAL_FINE_SURFACE_EMIT_LANES * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // One word per 4^3 cube window of a dense (n+1)^3 lattice, after the
    // count; a sparse source never scans by window.
    const windowWords = sparseMaxVertices === undefined ? Math.ceil((nx + 1) / 4) * Math.ceil((ny + 1) / 4) * Math.ceil((nz + 1) / 4) : 1;
    this.surfaceWindows = this.device.createBuffer({ label: "Water surface window list", size: 4 * (1 + windowWords), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.surfaceScanBlocks = this.device.createBuffer({ label: "Surface scan block totals", size: Math.max(4, Math.ceil(activeCubeCapacity(maxVertices) / SURFACE_SCAN_BLOCK_SIZE) * 4), usage: GPUBufferUsage.STORAGE });
    this.geometryKey = key; this.extractedRevision = -1; this.surfaceExtractionReason = "geometry allocation changed"; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  ensureSize(width: number, height: number) {
    const key = `${width}x${height}`;
    if (key === this.targetKey) return;
    for (const texture of [this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth]) texture?.destroy();
    const sampledTarget = (label: string, format: GPUTextureFormat = "rgba16float") => this.device.createTexture({ label, size: [width,height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.sceneTexture = this.device.createTexture({ label: "Dry scene HDR", size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }); this.sceneTextureView = this.sceneTexture.createView(); this.frontPosition = sampledTarget("Water front positions", "rgba32float"); this.frontNormal = sampledTarget("Water front normals"); this.backPosition = sampledTarget("Water back positions", "rgba32float"); this.backNormal = sampledTarget("Water back normals"); this.rearFrontPosition = sampledTarget("Water rear front positions", "rgba32float"); this.rearFrontNormal = sampledTarget("Water rear front normals"); this.rearBackPosition = sampledTarget("Water rear back positions", "rgba32float"); this.rearBackNormal = sampledTarget("Water rear back normals");
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
   * How many bodies the dry-scene attachment has to account for.
   *
   * Only the fluid-only path reads it. A `full-scene` presentation draws the
   * roster through the SVO rigid raster and never asks. Zero returns the
   * attachment to its once-only clear, which is why this invalidates it:
   * removing the last body must repaint the background the body was over.
   */
  setRigidBodyCount(count: number) {
    const bodies = Math.max(0, Math.min(RASTER_WATER_MAXIMUM_BODIES, Math.floor(count)));
    if (this.rigidBodyCount === bodies) return;
    this.rigidBodyCount = bodies;
    this.clearBackgroundEncoded = false;
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
    this.wireframeSceneBindGroups = new WeakMap();
    const globalFine = this.globalFineLevelSet;
    const coarse = this.coarseLevelSet;
    const coarseDirectory = globalFine?.coarsePhiDirectory ?? coarse?.directory;
    // The mixed presentation contours the nodal phi alone: the 4h base, and
    // the detail field while the solver names one. Its extraction binds no
    // cell field, and no h-sized texture at all while no tile can be at h.
    const mixedSurface = Boolean(this.mixedOwnership && this.coarseVertexPhi);
    const contourField = this.levelSetSource?.contourVertexPhi ? this.levelSetSource.vertexPhi : this.volume;
    if (this.extractLayout && this.volume && this.columnBases && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.fallbackSparsePageTable && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseParams && this.fallbackSparseControl && this.fallbackField && this.surfaceWindows) this.extractBindGroup = this.device.createBindGroup({ layout: this.extractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: (mixedSurface ? this.fallbackField : contourField!).createView({ dimension: "3d" }) }, { binding: 2, resource: this.columnBases.createView() }, { binding: 3, resource: { buffer: this.vertexBuffer } }, { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 7, resource: { buffer: this.fallbackSparsePageTable } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.samples ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: globalFine ? { buffer: this.globalFineRenderParams! } : { buffer: this.fallbackSparseParams } },
      { binding: 11, resource: globalFine?.samples ?? { buffer: this.fallbackSparseControl } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } },
      { binding: 13, resource: (this.denseNormalPhi ?? (mixedSurface ? this.fallbackField : this.volume)).createView({ dimension: "3d" }) },
      { binding: 14, resource: this.mixedOwnership ?? { buffer: this.fallbackSparseControl, size: 4 } },
      { binding: 15, resource: (this.coarseVertexPhi ?? this.fallbackField).createView({ dimension: "3d" }) },
      { binding: 16, resource: { buffer: this.surfaceWindows } }
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
    if (this.prepareClassifyLayout && this.classifyDispatchBuffer && this.globalFineRenderParams && this.fallbackSparseActivePages) this.prepareClassifyBindGroup = this.device.createBindGroup({ layout: this.prepareClassifyLayout, entries: [
      { binding: 0, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 1, resource: { buffer: this.globalFineRenderParams } },
      { binding: 2, resource: { buffer: this.classifyDispatchBuffer } },
    ] });
    if (this.globalPolygoniseLayout && this.surfaceScanWorkLayout && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalCubeOffsets && this.polygoniseDispatchBuffer && this.globalFineRenderParams && this.surfaceScanBlocks) {
      const entries: GPUBindGroupEntry[] = [
        { binding: 3, resource: { buffer: this.vertexBuffer } },
        { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
        { binding: 6, resource: { buffer: this.globalCubeValues } }, { binding: 7, resource: { buffer: this.globalCubeOffsets } },
        { binding: 10, resource: { buffer: this.globalFineRenderParams } },
        { binding: 18, resource: { buffer: this.surfaceScanBlocks } },
      ];
      this.surfaceScanWorkBindGroup = this.device.createBindGroup({ layout: this.surfaceScanWorkLayout, entries });
      this.globalPolygoniseBindGroup = this.device.createBindGroup({ layout: this.globalPolygoniseLayout, entries: [
        ...entries, { binding: 11, resource: { buffer: this.polygoniseDispatchBuffer } },
      ] });
    }
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
    if (this.rigidSceneLayout && this.waterSceneOpticsBuffer && this.causticReceiver) this.rigidSceneBindGroup = this.device.createBindGroup({ label: "Fluid-only rigid body inputs", layout: this.rigidSceneLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } },
      { binding: 2, resource: { buffer: this.waterSceneOpticsBuffer } }, { binding: 3, resource: this.causticReceiver.createView() }
    ] });
    this.compositeBindGroup = this.sceneTextureView ? this.compositeBindGroupFor(this.sceneTextureView) : undefined;
  }

  private compositeBindGroupFor(sceneView: GPUTextureView): GPUBindGroup | undefined {
    const cached = this.compositeBindGroups.get(sceneView);
    if (cached) return cached;
    if (!this.compositeLayout || !this.frontPosition || !this.frontNormal || !this.backPosition || !this.backNormal || !this.rearFrontPosition || !this.rearFrontNormal || !this.rearBackPosition || !this.rearBackNormal || !this.sampler || !this.volume || !this.columnBases || !this.causticTexture || !this.waterSceneOpticsBuffer || !this.causticReceiver || !this.fallbackSparseControl || !this.fallbackField) return undefined;
    // Under the mixed presentation the contact band samples the nodal level
    // set (17-19) and the cell field at 8 is not read.
    const mixedSurface = Boolean(this.mixedOwnership && this.coarseVertexPhi);
    const contourField = this.levelSetSource?.contourVertexPhi ? this.levelSetSource.vertexPhi : this.volume;
    const bindGroup = this.device.createBindGroup({ layout: this.compositeLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: sceneView }, { binding: 2, resource: this.frontPosition.createView() }, { binding: 3, resource: this.frontNormal.createView() }, { binding: 4, resource: this.backPosition.createView() }, { binding: 5, resource: this.backNormal.createView() }, { binding: 6, resource: this.sampler }, { binding: 7, resource: { buffer: this.bodyBuffer } }, { binding: 8, resource: (mixedSurface ? this.fallbackField : contourField!).createView({ dimension: "3d" }) }, { binding: 9, resource: this.columnBases.createView() }, { binding: 10, resource: this.rearFrontPosition.createView() }, { binding: 11, resource: this.rearFrontNormal.createView() }, { binding: 12, resource: this.rearBackPosition.createView() }, { binding: 13, resource: this.rearBackNormal.createView() }, { binding: 14, resource: this.causticTexture.createView() }, { binding: 15, resource: { buffer: this.waterSceneOpticsBuffer } }, { binding: 16, resource: this.causticReceiver.createView() },
      { binding: 17, resource: mixedSurface ? this.mixedOwnership! : { buffer: this.fallbackSparseControl, size: 4 } },
      { binding: 18, resource: ((mixedSurface ? this.denseNormalPhi : undefined) ?? this.fallbackField).createView({ dimension: "3d" }) },
      { binding: 19, resource: (this.coarseVertexPhi ?? this.fallbackField).createView({ dimension: "3d" }) }
    ] });
    this.compositeBindGroups.set(sceneView, bindGroup);
    return bindGroup;
  }

  private wireframeSceneBindGroupFor(sceneView: GPUTextureView): GPUBindGroup | undefined {
    const cached = this.wireframeSceneBindGroups.get(sceneView);
    if (cached) return cached;
    if (!this.surfacePeelLayout) return undefined;
    const bindGroup = this.device.createBindGroup({
      label: "Water wireframe dry-scene depth",
      layout: this.surfacePeelLayout,
      entries: [{ binding: 0, resource: sceneView }],
    });
    this.wireframeSceneBindGroups.set(sceneView, bindGroup);
    return bindGroup;
  }

  /** Start source-specific compilation before opening a presentation encoder.
   * Pending specialization is a retryable state; compilation errors still fail.
   */
  prepareSurfacePipelines(): boolean {
    this.ensureGlobalCoarsePipeline();
    const needsGlobalSurface = Boolean(this.globalFineLevelSet || this.coarseLevelSet);
    if (needsGlobalSurface && this.startGlobalSurfaceCompilation) {
      const start = this.startGlobalSurfaceCompilation;
      this.startGlobalSurfaceCompilation = undefined;
      this.extractGlobalFinePipelinePromise = start().catch((error: unknown) => {
        this.globalSurfaceCompilationError = error;
      });
    }
    if (needsGlobalSurface && this.globalSurfaceCompilationError) throw this.globalSurfaceCompilationError;
    if (this.needsGlobalCoarsePipeline() && this.globalCoarseCompilationError) {
      throw this.globalCoarseCompilationError;
    }
    return !needsGlobalSurface || Boolean(
      this.prepareClassifyPipeline && this.prepareSurfaceScanPipeline && this.countSurfaceBlocksPipeline
      && this.addSurfaceBlockOffsetsPipeline && this.polygoniseGlobalFineScanPipeline && this.polygoniseGlobalFineEmitPipeline
      && (!this.globalFineLevelSet || this.extractGlobalFinePipeline)
      && (!this.needsGlobalCoarsePipeline() || this.extractGlobalCoarsePipeline));
  }

  encode(encoder: GPUCommandEncoder, output: GPUTexture | GPUTextureView, nx: number, ny: number, nz: number, restrictedTallCell: boolean, maximumNeighborDelta: number, revision: number, drySceneReplacement?: DrySceneReplacementEncoder, tracePhase?: RenderPathTraceStage, forceSurfaceDiagnostics = false, backgroundMode: RasterWaterBackgroundMode = "require-dry-scene", allowSurfaceDiagnostics = true, bandPartitioner?: FrameBandPartitioner, revisionCadenceBypass = false, surfaceRenderMode: FluidSurfaceRenderMode = "shaded"): RasterWaterEncodeResult | false {
    // Count only frames whose source has a completed GPU receipt. The
    // diagnostics/visual panels request full-rate receipts, making this an
    // exact source-mode counter while it is being used to judge fidelity.
    if (this.lastSurfaceDiagnostics) {
      this.surfaceSourceFrameCounts[this.lastSurfaceDiagnostics.surfaceGeometrySource] += 1;
    }
    const compactSurface = this.globalFineLevelSet ?? this.coarseLevelSet;
    const geometryDimensions = compactSurface?.sampleDimensions ?? [nx, ny, nz] as const;
    const sparseGeometryCapacity = this.globalFineLevelSet
      ? globalFineSurfaceVertexCapacity(this.globalFineLevelSet.pageCapacity,
        this.globalFineLevelSet.samplesPerBrick)
      : undefined;
    this.ensureGeometry(geometryDimensions[0], geometryDimensions[1],
      geometryDimensions[2], sparseGeometryCapacity);
    if (!this.prepareSurfacePipelines()) return false;
    const globalFinePipeline = this.extractGlobalFinePipeline;
    const globalCoarsePipeline = this.extractGlobalCoarsePipeline;
    if (!this.extractPipeline||!this.extractWindowsPipeline||!this.orderWorklistPipeline||!this.collectWindowsPipeline||!this.extractBandPipeline||!this.extractTallSidesPipeline||(Boolean(this.globalFineLevelSet)&&!globalFinePipeline)||(this.needsGlobalCoarsePipeline()&&!globalCoarsePipeline)||!this.preparePipeline||!this.polygonisePipeline||!this.surfaceFrontPipeline||!this.surfaceBackPipeline||!this.surfaceRearFrontPipeline||!this.surfaceRearBackPipeline||!this.surfaceWireframePipeline||!this.causticPipeline||!this.compositePipeline||!this.wireframeCompositePipeline||!this.simpleCompositePipeline||!this.extractBindGroup||!this.globalExtractBindGroup||!this.globalPolygoniseBindGroup||!this.globalPolygoniseEmitBindGroup||!this.prepareBindGroup||!this.surfaceBindGroup||!this.causticBindGroup||!this.surfaceUnpeeledBindGroup||!this.surfacePeelBindGroup||!this.compositeBindGroup||!this.indirectBuffer||!this.polygoniseDispatchBuffer||!this.volume||!this.sceneTexture||!this.frontPosition||!this.frontNormal||!this.frontDepth||!this.backPosition||!this.backNormal||!this.backDepth||!this.rearFrontPosition||!this.rearFrontNormal||!this.rearFrontDepth||!this.rearBackPosition||!this.rearBackNormal||!this.rearBackDepth||!this.causticTexture||!this.causticReceiver) return false;
    const now_ms = performance.now();
    // A paused t=0 handoff cannot wait for a new solver revision: reset has
    // already made the current revision the only one that will be presented.
    // Retry extraction until its own diagnostic copy is admitted. This also
    // bypasses the ordinary 250 ms telemetry throttle, but never overwrites a
    // readback still owned by an earlier submission.
    // The extraction chain has its own switch, independent of interface drawing. Withheld,
    // the retained mesh keeps drawing (the interfaces read the last extraction)
    // so the delta is classify + scan + emit and nothing downstream. The t=0
    // handoff's forced capture overrides the withhold — a startup gate that can
    // never satisfy its own admission condition would stall the presentation
    // forever behind a diagnostic switch.
    const updateSurface = forceSurfaceDiagnostics
      || (!this.disabledStages.has("surface-extraction")
        && shouldUpdateWaterSurface(this.extractedRevision, revision,
          this.lastExtractionAt_ms, now_ms, revisionCadenceBypass));
    let surfaceDiagnosticsCaptured = false;
    // The map follows the mesh: a retained surface deposits the same bundles,
    // so re-projecting it would spend a full pass to write the same texels. A
    // scene that authors zero caustic strength never encodes the pass at all.
    const updateCaustics = this.causticStrength > 0 && !this.disabledStages.has("caustics")
      && (updateSurface || !this.causticsValid);
    if (updateSurface) {
      this.extractionCount += 1;
      if (forceSurfaceDiagnostics) this.surfaceExtractionReason = "forced receipt";
      else if (this.extractedRevision >= 0) this.surfaceExtractionReason = `revision ${this.extractedRevision} → ${revision}`;
      const indirectReset = this.indirectResetTemplate ??= this.createIndirectResetTemplate();
      if (compactSurface) {
        // Preserve the last published draw count while the GPU validates the
        // next A/B generation. Classification clears the sentinel only after
        // observing finite tagged fine data or a published compact-coarse
        // fallback; an invalid generation therefore retains the previous mesh
        // and its GPU-written publication generation in word 7.
        encoder.copyBufferToBuffer(indirectReset,0,this.indirectBuffer,4,24);
      } else {
        encoder.copyBufferToBuffer(indirectReset,32,this.indirectBuffer,0,36);
        if (this.surfaceWindows) encoder.copyBufferToBuffer(indirectReset,32,this.surfaceWindows,0,4);
      }
      const plan = surfaceExtractionDispatchPlan(nx, ny, nz,
        this.volume.depthOrArrayLayers, restrictedTallCell, maximumNeighborDelta);
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
          compute.setPipeline(this.prepareClassifyPipeline!);
          compute.setBindGroup(0, this.prepareClassifyBindGroup!);
          compute.dispatchWorkgroups(1);
          compute.end();
          compute = encoder.beginComputePass({ label: "Classify active water pages" });
          compute.setBindGroup(0, this.globalExtractBindGroup);
          compute.setPipeline(globalFinePipeline!);
          compute.dispatchWorkgroupsIndirect(this.classifyDispatchBuffer!, 0);
        }
        if(globalFine?.coarsePhiRowCapacity){compute.setPipeline(globalCoarsePipeline!);compute.dispatchWorkgroups(...globalFineCoarseSurfaceDispatch(globalFine.coarsePhiRowCapacity));}
        else if(coarse){compute.setPipeline(globalCoarsePipeline!);compute.dispatchWorkgroups(...compactCoarseSurfaceDispatch(coarse.sampleDimensions));}
        compute.end();
        const scanPass = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup, indirect: boolean) => {
          const pass = encoder.beginComputePass({ label });
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          if (indirect) pass.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer!, 12);
          else pass.dispatchWorkgroups(1);
          pass.end();
        };
        scanPass("Prepare water surface scan", this.prepareSurfaceScanPipeline!, this.globalPolygoniseBindGroup, false);
        scanPass("Count water surface blocks", this.countSurfaceBlocksPipeline!, this.surfaceScanWorkBindGroup!, true);
        scanPass("Scan water surface block totals", this.polygoniseGlobalFineScanPipeline!, this.globalPolygoniseBindGroup, false);
        scanPass("Add water surface block offsets", this.addSurfaceBlockOffsetsPipeline!, this.surfaceScanWorkBindGroup!, true);
        compute=encoder.beginComputePass({label:"Emit classified global fine surface"});
        compute.setBindGroup(0,this.globalPolygoniseEmitBindGroup);
        compute.setPipeline(this.polygoniseGlobalFineEmitPipeline!);compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer,0);
        compute.end();
      } else {
        if (plan.mode === "restricted-band") {
          compute.setPipeline(this.extractBandPipeline); compute.dispatchWorkgroups(...plan.band!);
          compute.setPipeline(this.extractTallSidesPipeline); compute.dispatchWorkgroups(...plan.tallSides!);
        } else {
          this.dispatchDenseClassify(compute, nx, ny, nz, this.surfaceClassify, false);
        }
        prepareAndPolygonise(this.polygonisePipeline,this.extractBindGroup);
        compute.end();
      }
      this.surfaceDiagnosticsDirty = true;
      this.extractedRevision = revision; this.lastExtractionAt_ms = advancePresentationClock(this.lastExtractionAt_ms, now_ms);
      tracePhase?.("surface-extraction");
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
      tracePhase?.("caustics");
    }
    // Everything above — extraction, diagnostics, caustics — is the
    // water-surface band; the dry-scene replacement crosses its own internal
    // boundaries, so this scope's encoder must resynchronize afterwards.
    if (bandPartitioner) encoder = bandPartitioner.boundary("water-surface");
    const sparseSceneResult = drySceneReplacement?.(encoder, this.sceneTexture) ?? false;
    if (bandPartitioner) encoder = bandPartitioner.current;
    if (sparseSceneResult) {
      // A later switch to fluid-only must clear imagery left by this frame.
      this.clearBackgroundEncoded = false;
    } else if (backgroundMode === "clear") {
      // A fluid-only scene runs no SVO world, so `webgpu-svo-rigid-raster` --
      // the only other thing that draws a body -- never encodes, and the
      // composite reads the roster for contact and for the interior exit but
      // never shades one. Without this pass a body is invisible until it is
      // behind water, which is not a thing you can drop something into.
      //
      // It writes what the SVO dry scene writes, so nothing downstream changes:
      // radiance in RGB, ray distance in alpha. Bodies therefore sort against
      // the water through `resolvedDrySceneDepth` and refract through the
      // ordinary scene-texture read.
      if (this.rigidBodyCount > 0 && this.rigidScenePipeline && this.rigidSceneBindGroup) {
        const pass = encoder.beginRenderPass({label:"Fluid-only rigid bodies",colorAttachments:[{
          view:this.sceneTextureView!,clearValue:FLUID_ONLY_BACKGROUND,loadOp:"clear",storeOp:"store"
        }]});
        pass.setPipeline(this.rigidScenePipeline);
        pass.setBindGroup(0,this.rigidSceneBindGroup);
        pass.draw(3);
        pass.end();
        // Bodies move, so this attachment is authored every frame. The retained
        // clear below is no longer what is on it.
        this.clearBackgroundEncoded = false;
        tracePhase?.("fluid-only-rigid-bodies");
      } else if (!this.clearBackgroundEncoded) {
        // No geometry, shader, draw, or recurring full-frame write. The clear is
        // retained until a dry-scene encoder writes the attachment or it resizes.
        encoder.beginRenderPass({label:"Fluid-only clear background",colorAttachments:[{
          view:this.sceneTextureView!,clearValue:FLUID_ONLY_BACKGROUND,loadOp:"clear",storeOp:"store"
        }]}).end();
        this.clearBackgroundEncoded = true;
        tracePhase?.("fluid-only-background");
      }
    } else {
      this.clearBackgroundEncoded = false;
      // The live sparse scene is the only dry-scene authority, and until it is
      // attached there is no scene-like substitute: the attachment is the
      // studio's own ground and nothing else. It used to be a maroon fault
      // clear, which made every ordinary cold load — the half minute the
      // presentation takes to compile — look like a failure. Absence is not a
      // fault here; the activity mark says what is being prepared, and a real
      // failure is reported by the viewport's alert, not by a colour. Alpha is
      // far depth so the independently authoritative water interfaces remain
      // visible meanwhile.
      encoder.beginRenderPass({label:"SVO dry-scene unavailable",colorAttachments:[{
        view:this.sceneTextureView!,clearValue:FLUID_ONLY_BACKGROUND,loadOp:"clear",storeOp:"store"
      }]}).end();
      tracePhase?.("dry-scene-unavailable");
    }
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
      tracePhase?.("water-front-interface");
      interfacePass("Water + spray back interfaces",this.surfaceBackPipeline,this.backPosition,this.backNormal,this.backDepth,"back");
      tracePhase?.("water-back-interface");
      interfacePass("Water rear front interfaces",this.surfaceRearFrontPipeline,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,"front",false,true);
      interfacePass("Water rear back interfaces",this.surfaceRearBackPipeline,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth,"back",false,true);
      // Their own seam: without it these two peeled passes were charged to the
      // optical composite, the next label to close.
      tracePhase?.("water-rear-interfaces");
    } else if (!this.dryInterfaceClearsEncoded) {
      // A fluid-less scene draws no interface geometry. Clear once so the
      // compositor's no-interface input cannot retain a preceding fluid scene.
      const clearPass=(label:string,position:GPUTexture,normal:GPUTexture,depth:GPUTexture)=>{encoder.beginRenderPass({label,colorAttachments:[{view:position.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"},{view:normal.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}}).end();};
      clearPass("Fluid-less scene front interface clear",this.frontPosition,this.frontNormal,this.frontDepth);
      tracePhase?.("water-front-interface");
      clearPass("Fluid-less scene back interface clear",this.backPosition,this.backNormal,this.backDepth);
      tracePhase?.("water-back-interface");
      clearPass("Fluid-less scene rear front interface clear",this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth);
      clearPass("Fluid-less scene rear back interface clear",this.rearBackPosition,this.rearBackNormal,this.rearBackDepth);
      tracePhase?.("water-rear-interfaces");
      this.dryInterfaceClearsEncoded = true;
    }
    const compositeBindGroup = sparseSceneResult ? this.compositeBindGroupFor(sparseSceneResult.sampledTargetView) : this.compositeBindGroup;
    if (!compositeBindGroup) return false;
    const outputView="createView" in output?output.createView():output;const composite=encoder.beginRenderPass({label:"Layered water optical composite",colorAttachments:[{view:outputView,clearValue:{r:.01,g:.025,b:.024,a:1},loadOp:"clear",storeOp:"store"}]});
    // The single final transform every render path shares: withholding it keeps
    // the clear so the presentation target is the background colour rather than
    // a stale composite that would look like the pass still ran.
    if (!this.disabledStages.has("optical-composite")) {
      composite.setPipeline(surfaceRenderMode === "wireframe" ? this.wireframeCompositePipeline : surfaceRenderMode === "simple" ? this.simpleCompositePipeline : this.compositePipeline);composite.setBindGroup(0,compositeBindGroup);composite.draw(3);
    }
    composite.end();
    if (surfaceRenderMode === "wireframe" && !this.disabledStages.has("optical-composite")) {
      const sceneView = sparseSceneResult ? sparseSceneResult.sampledTargetView : this.sceneTextureView!;
      const wireframeSceneBindGroup = this.wireframeSceneBindGroupFor(sceneView);
      if (!wireframeSceneBindGroup) return false;
      const wireframePass=encoder.beginRenderPass({label:"Water surface wireframe overlay",colorAttachments:[{view:outputView,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:this.frontDepth.createView(),depthLoadOp:"load",depthStoreOp:"store"}});
      wireframePass.setPipeline(this.surfaceWireframePipeline);
      wireframePass.setBindGroup(0,this.surfaceBindGroup);
      wireframePass.setBindGroup(1,wireframeSceneBindGroup);
      wireframePass.drawIndirect(this.indirectBuffer,0);
      wireframePass.end();
    }
    tracePhase?.("optical-composite");return { surfaceUpdated: updateSurface, surfaceDiagnosticsCaptured };
  }

  destroy() {
    for (const resource of [this.vertexBuffer,this.indirectBuffer,this.activeCubeBuffer,this.globalCubeValues,this.globalCubeOffsets,this.surfaceScanBlocks,this.classifyDispatchBuffer,this.polygoniseDispatchBuffer,this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.rearFrontPosition,this.rearFrontNormal,this.rearFrontDepth,this.rearBackPosition,this.rearBackNormal,this.rearBackDepth,this.causticTexture,this.causticReceiver,this.waterSceneOpticsBuffer,this.fallbackSparsePageTable,this.fallbackSparseActivePages,this.fallbackSparsePhi,this.fallbackSparseParams,this.globalFineRenderParams,this.fallbackSparseControl,this.fallbackField,this.surfaceWindows,this.surfaceDiagnosticReadback]) { try { resource?.destroy(); } catch { /* device loss */ } }
  }
}
