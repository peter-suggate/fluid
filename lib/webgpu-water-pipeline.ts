import { environmentShaderLibrary } from "./webgpu-environments";
import { advancePresentationClock, frameInterval_ms } from "./frame-pacing";
import type { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";
import {
  GLASS_OPTICS,
  unifiedDisplayTransferShaderLibrary,
  unifiedLightingShaderLibrary,
  WATER_OPTICS,
} from "./webgpu-lighting";
import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import {
  SPARSE_SURFACE_CONTROL_BYTES,
  type SparseSurfaceBandGPUSource,
} from "./webgpu-sparse-surface-band";
import { OCTREE_SURFACE_LEAF_RECORD_BYTES } from "./webgpu-octree-surface-pages";
import {
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_ENTRY_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_SAMPLE_HEADER_BYTES,
} from "./webgpu-octree-power-coarse-levelset";
import {
  validateGlobalFineLevelSetConsumerSource,
  validateUnifiedOctreeConsumerSource,
  type GlobalFineLevelSetConsumerSource,
  type UnifiedOctreeConsumerSource,
} from "./octree-consumer-sampling";
import { globalFineClassifiedCountShader, globalFineClassifiedEmitShader, globalFineClassifiedEmitShaders, globalFineClassifiedScanShader } from "./webgpu-water-global-fine-tetra";
import { globalFineSurfaceClassificationShader } from "./webgpu-water-global-fine-classify";

/**
 * Rasterized water presentation for the WebGPU renderer.
 *
 * The solver already owns the liquid volume.  This pipeline keeps that data on
 * the GPU and turns its 0.5 isosurface into triangles every frame.  The result
 * is then rendered twice (front and back interfaces), which gives the optical
 * composite enough information for two-interface refraction without scanning
 * the volume once per screen pixel.
 */

export function shouldUpdateWaterSurface(extractedRevision: number, latestRevision: number, lastExtractionAt_ms: number, now_ms: number) {
  return extractedRevision < 0
    || (latestRevision !== extractedRevision && now_ms - lastExtractionAt_ms + 0.5 >= frameInterval_ms());
}

/** Raster/body depth separation that activates the local implicit resolver. */
export const CONTACT_RESOLVE_BAND_CELLS = 1.5;

/** Shared disabled storage must satisfy every struct it can stand in for. */
export const WATER_SPARSE_FALLBACK_BYTES = Math.max(
  SPARSE_SURFACE_CONTROL_BYTES,
  OCTREE_SURFACE_LEAF_RECORD_BYTES,
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

export type SurfaceExtractionRepresentation = "adaptive-octree" | "sparse-band" | "dense-texture";

export function surfaceExtractionRepresentation(hasAdaptiveOctree: boolean, hasSparseBand: boolean): SurfaceExtractionRepresentation {
  return hasAdaptiveOctree ? "adaptive-octree" : hasSparseBand ? "sparse-band" : "dense-texture";
}

export type TimestampRange = { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number };
export interface RasterWaterTimestampRanges {
  extraction: TimestampRange;
  caustics: TimestampRange;
  scene: TimestampRange;
  frontInterfaces: TimestampRange;
  backInterfaces: TimestampRange;
  sprayFront: TimestampRange;
  sprayBack: TimestampRange;
  composite: TimestampRange;
}

export interface RasterWaterEncodeResult {
  surfaceUpdated: boolean;
  causticsUpdated: boolean;
  sprayRendered: boolean;
}

export interface WaterSurfacePresentationDiagnostics {
  /** Presentation geometry only; this does not confer simulation authority on a fallback field. */
  readonly surfaceGeometrySource: WaterSurfaceGeometrySource;
  readonly globalFineAttached: boolean;
  /** Generation of the global-fine source captured with this queue-fenced diagnostic. */
  readonly globalFineAttachedGeneration?: number;
  /** Generation whose zero crossing produced the retained raster mesh, when that mesh is global-fine. */
  readonly meshPublicationGeneration?: number;
  readonly globalFineCrossingPublished: boolean;
  readonly presentationFallbackActive: boolean;
}

export interface AdaptiveWaterRenderDiagnostics extends WaterSurfacePresentationDiagnostics {
  readonly leafCapacity: number;
  readonly pageCapacity: number;
  readonly pageResolution: number;
  readonly samplesPerPage: number;
  readonly surfaceFreePages: number;
  readonly surfaceAllocatedPages: number;
  readonly surfaceCandidatePages: number;
  readonly surfaceActivePages: number;
  readonly surfaceOverflow: number;
  readonly finestResidentPages: number;
  readonly coarseResidentPages: number;
  readonly maximumResidentLeafSize: number;
  readonly surfaceDispatch: readonly [number, number, number];
  readonly vertexCount: number;
  readonly activeCubeCount: number;
  readonly vertexAllocator: number;
  readonly globalFineAuthorityLatch: number;
}

export type WaterSurfaceGeometrySource =
  | "global-fine-coarse"
  | "adaptive-fallback"
  | "retained-previous"
  | "empty"
  | "adaptive-octree";

/**
 * Decodes the renderer-private transaction words. `authorityLatch` trails the
 * four WebGPU indirect-draw arguments, so the required draw `firstInstance`
 * remains zero on devices without the optional indirect-first-instance
 * feature. The latch is presentation evidence only; it never makes a fallback
 * field authoritative for simulation.
 */
export function waterSurfaceGeometrySource(
  globalFineAttached: boolean,
  vertexCount: number,
  authorityLatch: number,
  vertexAllocator: number,
): WaterSurfaceGeometrySource {
  if (!globalFineAttached) return vertexCount > 0 ? "adaptive-octree" : "empty";
  if (authorityLatch !== 0) return "global-fine-coarse";
  if (vertexAllocator !== 0xffff_ffff) return "adaptive-fallback";
  return vertexCount > 0 ? "retained-previous" : "empty";
}

/** Adaptive geometry may seed global presentation only before any mesh exists. */
export function globalFineFallbackMaySeedRenderer(vertexCount: number, authorityLatch: number): boolean {
  return vertexCount === 0 && authorityLatch === 0;
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

export type DrySceneReplacementEncoder = (
  encoder: GPUCommandEncoder,
  target: GPUTexture | GPUTextureView,
  timestampWrites?: TimestampRange
) => DrySceneReplacementResult | false;

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
 * area is several times the tank footprint. 64 vertices per lattice-area unit
 * covers that normal fine-band case while the byte ceiling remains absolute.
 */
export function surfaceVertexCapacity(nx: number, ny: number, nz: number) {
  const area = nx * ny + nx * nz + ny * nz;
  return Math.max(262_144, Math.min(2_097_152, area * 64));
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
struct AdaptiveLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct AdaptiveParams { shape:vec4u, offsets0:vec4u, offsets1:vec4u, offsets2:vec4u, cellDt:vec4f, spare0:vec4u, spare1:vec4u, spare2:vec4u }
@group(0) @binding(13) var<storage, read> adaptiveLeaves: array<AdaptiveLeaf>;
@group(0) @binding(14) var<storage, read> adaptiveArena: array<u32>;
@group(0) @binding(15) var<uniform> adaptiveParams: AdaptiveParams;
override countOnly = false;
override sparseField = false;
override adaptiveField = false;
override globalFineField = false;
override globalFineFallback = false;

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
fn globalHash(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(sparseParams.brickDims.x-1u);}
fn globalLookup(key:u32)->u32{let start=globalHash(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=sparseParams.brickDims.y){break;}
 let slot=(start+probe)&(sparseParams.brickDims.x-1u);let stored=sparsePageTable[slot*2u];if(stored==key){let id=sparsePageTable[slot*2u+1u];
  if(id<sparseParams.brickDims.z&&sparseStates[id*10u+1u]==key&&sparseStates[id*10u+2u]==sparseParams.brickDims.w){return id;}return SPARSE_INVALID;}
 if(stored==SPARSE_INVALID){return SPARSE_INVALID;}}return SPARSE_INVALID;}
fn globalCoarsePhi(q:vec3i)->f32{let factor=max(1.0,sparseParams.cellAndDt.x);let p=clamp((vec3f(q)+vec3f(0.5))/factor-vec3f(0.5),vec3f(0),u.gridInfo.xyz-vec3f(1));
 let a=vec3i(floor(p));let b=min(a+vec3i(1),vec3i(u.gridInfo.xyz)-vec3i(1));let t=fract(p);
 let p000=textureLoad(volume,vec3i(a.x,a.y,a.z),0).x;let p100=textureLoad(volume,vec3i(b.x,a.y,a.z),0).x;let p010=textureLoad(volume,vec3i(a.x,b.y,a.z),0).x;let p110=textureLoad(volume,vec3i(b.x,b.y,a.z),0).x;
 let p001=textureLoad(volume,vec3i(a.x,a.y,b.z),0).x;let p101=textureLoad(volume,vec3i(b.x,a.y,b.z),0).x;let p011=textureLoad(volume,vec3i(a.x,b.y,b.z),0).x;let p111=textureLoad(volume,b,0).x;
 return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);}
fn globalPhiAt(cell:vec3i)->f32{if(any(cell<vec3i(0))||any(cell>=vec3i(sparseParams.coarseDims.xyz))){return globalCoarsePhi(cell);}
 let q=vec3u(cell);let r=sparseParams.coarseDims.w;let brick=q/r;let local=q-brick*r;let key=brick.x+sparseParams.fineDims.x*(brick.y+sparseParams.fineDims.y*brick.z);let id=globalLookup(key);
 if(id==SPARSE_INVALID){return globalCoarsePhi(cell);}let localIndex=local.x+r*(local.y+r*local.z);let index=id*sparseParams.fineDims.w+localIndex;
 if(index>=arrayLength(&sparsePhi)||(sparseControl[index]&1u)==0u){return globalCoarsePhi(cell);}return sparsePhi[index];}

fn adaptiveOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn adaptiveLeafOrigin(leaf:AdaptiveLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn adaptiveFallback(leaf:AdaptiveLeaf,p:vec3f)->f32{
  let c=vec3f(adaptiveLeafOrigin(leaf))+vec3f(0.5*f32(leaf.size));
  let physicalGradient=leaf.phiGradient.yzw/max(adaptiveParams.cellDt.xyz,vec3f(1e-9));
  let boundedCellGradient=physicalGradient/max(1.0,length(physicalGradient))*adaptiveParams.cellDt.xyz;
  return leaf.phiGradient.x+dot(boundedCellGradient,p-c);
}
fn adaptivePageSlot(row:u32)->u32{return adaptiveArena[adaptiveParams.offsets0.x+row];}
fn adaptiveHash(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn adaptiveContains(leaf:AdaptiveLeaf,p:vec3f)->bool{let o=vec3f(adaptiveLeafOrigin(leaf));return all(p>=o)&&all(p<o+vec3f(f32(leaf.size)));}
fn adaptiveResidentRow(p:vec3f,fallbackRow:u32)->u32{
  let h=max(1u,adaptiveLeaves[fallbackRow].size);let q=vec3u(max(vec3f(0.0),floor(p/f32(h))));let mask=adaptiveParams.offsets2.y-1u;var slot=adaptiveHash(q)&mask;
  for(var probe=0u;probe<16u;probe+=1u){let encoded=adaptiveArena[adaptiveParams.offsets1.y+slot];if(encoded==0u){break;}let row=encoded-1u;if(row<adaptiveParams.shape.x&&adaptiveContains(adaptiveLeaves[row],p)){return row;}slot=(slot+1u)&mask;}
  return fallbackRow;
}
fn adaptiveLoad(slot:u32,resolution:u32,q:vec3u)->f32{return bitcast<f32>(adaptiveArena[adaptiveParams.offsets1.z+slot*adaptiveParams.shape.w+q.x+resolution*(q.y+resolution*q.z)]);}
fn adaptivePagePhi(row:u32,p:vec3f)->f32{
  let leaf=adaptiveLeaves[row];let slot=adaptivePageSlot(row);if(slot==SPARSE_INVALID||slot>=adaptiveParams.shape.y||!adaptiveContains(leaf,p)){return adaptiveFallback(leaf,p);}
  let resolution=adaptiveParams.shape.z;let origin=vec3f(adaptiveLeafOrigin(leaf));let grid=clamp((p-origin)/f32(leaf.size)*f32(resolution)-vec3f(0.5),vec3f(0.0),vec3f(f32(resolution-1u)));
  let a=vec3u(floor(grid));let b=min(a+vec3u(1u),vec3u(resolution-1u));let t=fract(grid);
  let c000=adaptiveLoad(slot,resolution,a);let c100=adaptiveLoad(slot,resolution,vec3u(b.x,a.y,a.z));let c010=adaptiveLoad(slot,resolution,vec3u(a.x,b.y,a.z));let c110=adaptiveLoad(slot,resolution,vec3u(b.x,b.y,a.z));
  let c001=adaptiveLoad(slot,resolution,vec3u(a.x,a.y,b.z));let c101=adaptiveLoad(slot,resolution,vec3u(b.x,a.y,b.z));let c011=adaptiveLoad(slot,resolution,vec3u(a.x,b.y,b.z));let c111=adaptiveLoad(slot,resolution,b);
  return mix(mix(mix(c000,c100,t.x),mix(c010,c110,t.x),t.y),mix(mix(c001,c101,t.x),mix(c011,c111,t.x),t.y),t.z);
}
var<private> adaptiveOwnerRow:u32=0u;
fn adaptivePhiAt(cell:vec3i)->f32{
  let p=(vec3f(cell)+vec3f(0.5))/f32(adaptiveParams.shape.z);let row=adaptiveResidentRow(p,adaptiveOwnerRow);return adaptivePagePhi(row,p);
}

// Level-set fields become a smooth occupancy whose 0.5 contour is phi = 0.
// The band spans four cells so no corner of a surface-crossing cube saturates
// (the cube diagonal is under two cells); a saturated corner biases the linear
// crossing estimate and extracts as cell-pitch lattice artifacts.
fn occupancyFromPhi(phi: f32) -> f32 {
  let samplesY = select(select(select(u.gridInfo.y, f32(sparseParams.fineDims.y), sparseField), u.gridInfo.y*f32(adaptiveParams.shape.z), adaptiveField),f32(sparseParams.coarseDims.y),globalFineField);
  let band = 4.0 * u.container.y / max(samplesY, 1.0);
  return clamp(0.5 - phi / band, 0.0, 1.0);
}

fn fieldCell(cell: vec3i) -> f32 {
  if(globalFineField){return occupancyFromPhi(globalPhiAt(cell));}
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
  let dims = select(select(select(vec3i(u.gridInfo.xyz), vec3i(sparseParams.fineDims.xyz), sparseField), vec3i(u.gridInfo.xyz)*i32(adaptiveParams.shape.z), adaptiveField),vec3i(sparseParams.coarseDims.xyz),globalFineField);
  // Side/top boundaries are optical interfaces. The floor is a solid contact,
  // not a water-air surface: extend the bottom cell value to y=0 so extraction
  // cannot create a large horizontal sheet across the tank base.
  if (p.x <= 0 || p.z <= 0 || p.x >= dims.x + 1 || p.z >= dims.z + 1 || p.y >= dims.y + 1) { return 0.0; }
  let cell = vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1);
  if (adaptiveField) { return occupancyFromPhi(adaptivePhiAt(cell)); }
  if (globalFineField) { return occupancyFromPhi(globalPhiAt(cell)); }
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

// Global fine entry points deliberately do not reach the adaptive leaf ABI.
// This keeps the coarse fallback texture plus global hash/payload resources
// below Metal's storage-binding limit instead of relying on override folding
// during bind-group reflection.
fn globalLatticeValue(p:vec3i)->f32{
  let dims=vec3i(sparseParams.coarseDims.xyz);
  if(p.x<=0||p.z<=0||p.x>=dims.x+1||p.z>=dims.z+1||p.y>=dims.y+1){return 0.0;}
  let band=4.0*u.container.y/max(f32(sparseParams.coarseDims.y),1.0);
  return clamp(0.5-globalPhiAt(vec3i(p.x-1,max(p.y-1,0),p.z-1))/band,0.0,1.0);
}
fn loadGlobalCubeCorners(base:vec3i)->array<f32,8>{
  let offsets=array<vec3i,8>(vec3i(0,0,0),vec3i(1,0,0),vec3i(1,1,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(1,0,1),vec3i(1,1,1),vec3i(0,1,1));
  var value=array<f32,8>();for(var i=0;i<8;i+=1){value[i]=globalLatticeValue(base+offsets[i]);}return value;
}
fn classifyGlobalCube(base:vec3i){
  let cubeDims=sparseParams.coarseDims.xyz+vec3u(1u);if(any(base<vec3i(0))||any(vec3u(base)>=cubeDims)){return;}
  var value=loadGlobalCubeCorners(base);var minimum=1.0;var maximum=0.0;
  for(var i=0;i<8;i+=1){minimum=min(minimum,value[i]);maximum=max(maximum,value[i]);}
  if(minimum>=0.5||maximum<0.5){return;}let slot=atomicAdd(&drawArgs.activeCubeCount,1u);
  if(slot<arrayLength(&activeCubes)&&slot*2u+1u<arrayLength(&globalCubeValues)){activeCubes[slot]=vec2u(u32(base.x)|(u32(base.z)<<16u),u32(base.y)|(1u<<16u));globalCubeValues[slot*2u]=vec4f(value[0],value[1],value[2],value[3]);globalCubeValues[slot*2u+1u]=vec4f(value[4],value[5],value[6],value[7]);}
}
// The sweep kernels stop here: eight corner loads, a min/max test, and one
// worklist append per *surface* cube. Emission code is confined to
// polygoniseMain so the register footprint of the full-lattice scan stays
// small enough for the occupancy that hides the load latency.
fn classifyCubeScaled(base: vec3i, scale: u32) {
  // Adaptive pages form a pageResolution-times finer virtual lattice.  Using
  // the coarse solver dimensions here clips almost every page-owned cube
  // before it reaches the worklist, leaving a valid-but-empty water draw.
  let fieldDims = select(select(
    select(vec3u(u.gridInfo.xyz), sparseParams.fineDims.xyz, sparseField),
    vec3u(u.gridInfo.xyz) * adaptiveParams.shape.z,
    adaptiveField,
  ),sparseParams.coarseDims.xyz,globalFineField);
  let cubeDims = fieldDims + vec3u(1);
  if (any(base < vec3i(0)) || any(vec3u(base) >= cubeDims)) { return; }
  var value = loadCubeCornersScaled(base, i32(scale));
  var minimum = 1.0; var maximum = 0.0;
  for (var i = 0; i < 8; i += 1) {
    minimum = min(minimum, value[i]); maximum = max(maximum, value[i]);
  }
  if (minimum >= 0.5 || maximum < 0.5) { return; }
  if (globalFineFallback) {
    var accepted = false;
    for (var attempt = 0u; attempt < 32u; attempt += 1u) {
      let claim = atomicCompareExchangeWeak(&drawArgs.vertexAllocator, SPARSE_INVALID, 0u);
      if (claim.exchanged) { atomicStore(&drawArgs.vertexCount, 0u); accepted = true; break; }
      if (claim.old_value == 0u) { accepted = true; break; }
      if (claim.old_value != SPARSE_INVALID) { break; }
    }
    if (!accepted) { return; }
  }
  if (countOnly) {
    // The benchmark's uncapped equivalence count. Counting whole cubes here
    // keeps it exact regardless of the production worklist capacity.
    atomicAdd(&drawArgs.vertexCount, 3u * cubeTriangleCount(&value));
    return;
  }
  let slot = atomicAdd(&drawArgs.activeCubeCount, 1u);
  if (slot < arrayLength(&activeCubes)) {
    if (adaptiveField) {
      // Adaptive coordinates need at most 13 bits (1024 coarse cells times
      // pageResolution <= 4). Reuse the six high bits from x/z and the
      // nineteen high bits from y/scale to carry a 24-bit owner row plus the
      // resident-page scale flag. The hot worklist therefore stays 8 bytes.
      let coordinateMask = 0x1fffu;
      let owner = adaptiveOwnerRow & 0x00ffffffu;
      activeCubes[slot] = vec2u(
        (u32(base.x) & coordinateMask) | ((u32(base.z) & coordinateMask) << 13u) | ((owner & 0x3fu) << 26u),
        (u32(base.y) & coordinateMask) | ((owner >> 6u) << 13u) | (select(0u, 1u, scale == 1u) << 31u),
      );
    } else {
      activeCubes[slot] = vec2u(u32(base.x) | (u32(base.z) << 16u), u32(base.y) | (scale << 16u));
    }
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
  // Both polygonisers are dispatched for a sparse surface. Exactly one owns
  // the shared worklist: fine pages normally, complete coarse extraction if
  // the bounded allocator reported that any required page was unavailable.
  let activeTotal = min(atomicLoad(&drawArgs.activeCubeCount), arrayLength(&activeCubes));
  // Normal reconstruction needs the selected lattice dimensions as well as
  // the cube-local samples; keep this sixth tetra argument at every LOD.
  let fieldDimensions=select(select(select(u.gridInfo.xyz,vec3f(sparseParams.fineDims.xyz),sparseField),u.gridInfo.xyz*f32(adaptiveParams.shape.z),adaptiveField),vec3f(sparseParams.coarseDims.xyz),globalFineField);
  var base = vec3i(0);
  var cubeScale = 1u;
  var value = array<f32, 8>();
  var vertexCount = 0u;
  var validCube = false;
  if (gid.x < activeTotal) {
    validCube = true;
    let packedCube = activeCubes[gid.x];
    if (adaptiveField) {
      let coordinateMask = 0x1fffu;
      base = vec3i(i32(packedCube.x & coordinateMask), i32(packedCube.y & coordinateMask), i32((packedCube.x >> 13u) & coordinateMask));
      adaptiveOwnerRow = (packedCube.x >> 26u) | (((packedCube.y >> 13u) & 0x3ffffu) << 6u);
      validCube = adaptiveOwnerRow < arrayLength(&adaptiveLeaves);
      if (validCube) { cubeScale = select(max(1u, adaptiveLeaves[adaptiveOwnerRow].size * adaptiveParams.shape.z), 1u, (packedCube.y & 0x80000000u) != 0u); }
    } else {
      base = vec3i(i32(packedCube.x & 0xffffu), i32(packedCube.y & 0xffffu), i32(packedCube.x >> 16u));
      cubeScale = max(1u, packedCube.y >> 16u);
    }
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

// One invocation per sample in each resident leaf-attached page. Each sample
// owns the lattice cube above it, so adjacent leaves neither duplicate cubes
// nor require a dense page table over the finest domain.
@compute @workgroup_size(256)
fn extractAdaptiveMain(@builtin(global_invocation_id) gid:vec3u) {
  // Global dispatches precede both fallback classifiers. vertexCount therefore
  // remains zero throughout fallback classification and changes only in the
  // later polygonise dispatch; the allocator CAS below still chooses one seed.
  if(globalFineFallback&&(atomicLoad(&drawArgs.globalFineAuthorityLatch)!=0u||atomicLoad(&drawArgs.vertexCount)!=0u)){return;}
  if(adaptiveArena[3]!=0u){return;}
  let stream=gid.x+gid.y*65535u*256u;let samples=adaptiveParams.shape.w;let activeCount=adaptiveArena[adaptiveParams.offsets1.x];let item=stream/samples;
  if(item>=activeCount){return;}let row=adaptiveArena[adaptiveParams.offsets1.x+4u+item];if(row>=adaptiveParams.shape.x){return;}
  let leaf=adaptiveLeaves[row];if(leaf.size!=1u){return;}adaptiveOwnerRow=row;
  let localIndex=stream-item*samples;let resolution=adaptiveParams.shape.z;let local=vec3u(localIndex%resolution,(localIndex/resolution)%resolution,localIndex/(resolution*resolution));
  let q=adaptiveLeafOrigin(leaf)*resolution+local;let xBases=array<i32,2>(i32(q.x+1u),0);let yBases=array<i32,2>(i32(q.y+1u),0);let zBases=array<i32,2>(i32(q.z+1u),0);
  let xCount=select(1u,2u,q.x==0u);let yCount=select(1u,2u,q.y==0u);let zCount=select(1u,2u,q.z==0u);
  for(var zi=0u;zi<zCount;zi+=1u){for(var yi=0u;yi<yCount;yi+=1u){for(var xi=0u;xi<xCount;xi+=1u){classifyCube(vec3i(xBases[xi],yBases[yi],zBases[zi]));}}}
}

@compute @workgroup_size(256)
fn extractGlobalFineMain(@builtin(global_invocation_id) gid:vec3u){
  if(sparseActivePages[1u]!=sparseParams.brickDims.w){return;}
  let stream=gid.x+gid.y*65535u*256u;let samples=sparseParams.fineDims.w;let activeCount=min(sparseActivePages[0],sparseParams.brickDims.z);let item=stream/samples;
  if(item>=activeCount||5u+item>=arrayLength(&sparseActivePages)){return;}let id=sparseActivePages[5u+item];
  if(id>=sparseParams.brickDims.z||id*10u+2u>=arrayLength(&sparseStates)||sparseStates[id*10u+2u]!=sparseParams.brickDims.w){return;}
  let key=sparseStates[id*10u+1u];let brickDims=max(sparseParams.fineDims.xyz,vec3u(1u));let xy=max(1u,brickDims.x*brickDims.y);let bz=key/xy;let rem=key-bz*xy;let by=rem/brickDims.x;let bx=rem-by*brickDims.x;
  let localIndex=stream-item*samples;let r=max(1u,sparseParams.coarseDims.w);let local=vec3u(localIndex%r,(localIndex/r)%r,localIndex/max(1u,r*r));let q=vec3u(bx,by,bz)*r+local;
  if(any(q>=sparseParams.coarseDims.xyz)){return;}let xBases=array<i32,2>(i32(q.x+1u),0);let yBases=array<i32,2>(i32(q.y+1u),0);let zBases=array<i32,2>(i32(q.z+1u),0);
  let xCount=select(1u,2u,q.x==0u);let yCount=select(1u,2u,q.y==0u);let zCount=select(1u,2u,q.z==0u);
  for(var zi=0u;zi<zCount;zi+=1u){for(var yi=0u;yi<yCount;yi+=1u){for(var xi=0u;xi<xCount;xi+=1u){classifyGlobalCube(vec3i(xBases[xi],yBases[yi],zBases[zi]));}}}
}

// Every live leaf has an affine phi plane even when it does not own a detail
// page. CORE/HALO are residency hints produced by the page adapter, not a
// prerequisite for presentation: classify the compact nonresident rows and
// let the eight-corner sign test discard leaves that do not cross phi=0.
// Resident pages continue through extractAdaptiveMain at their finer spacing.
@compute @workgroup_size(256)
fn extractAdaptiveLeafMain(@builtin(global_invocation_id) gid:vec3u) {
  if(globalFineFallback&&(atomicLoad(&drawArgs.globalFineAuthorityLatch)!=0u||atomicLoad(&drawArgs.vertexCount)!=0u)){return;}
  let row=gid.x;if(row>=adaptiveParams.shape.x||row>=arrayLength(&adaptiveLeaves)){return;}
  let leaf=adaptiveLeaves[row];if(leaf.size==0u||(leaf.flags&32u)==0u){return;}
  let page=adaptivePageSlot(row);if(page!=SPARSE_INVALID&&page<adaptiveParams.shape.y){return;}
  adaptiveOwnerRow=row;let resolution=adaptiveParams.shape.z;
  classifyCubeScaled(vec3i(adaptiveLeafOrigin(leaf)*resolution),max(1u,leaf.size*resolution));
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
struct IndirectArgs { vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32, activeCubeCount: u32, vertexAllocator: u32, globalFineAuthorityLatch: u32 }
struct DispatchArgs { x: u32, y: u32, z: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: IndirectArgs;
@group(0) @binding(1) var<storage, read> activeCubes: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> dispatchArgs: DispatchArgs;
@compute @workgroup_size(1)
fn prepareMain() {
  if (drawArgs.globalFineAuthorityLatch != 0u) {
    dispatchArgs = DispatchArgs(0u, 1u, 1u);
    return;
  }
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
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage,read> vertices: array<SurfaceVertex>;
struct Out { @builtin(position) clip:vec4f, @location(0) world:vec3f, @location(1) normal:vec3f }
fn project(world:vec3f)->vec4f {
  let forward=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0))); let up=normalize(cross(right,forward));
  let relative=world-u.cameraPosition.xyz; let depth=max(dot(relative,forward),0.001);
  let aspect=u.viewport.x/max(u.viewport.y,1.0);
  let ndc=vec2f(dot(relative,right)/(depth*aspect*${CAMERA_TAN_HALF_FOV}),dot(relative,up)/(depth*${CAMERA_TAN_HALF_FOV}));
  return vec4f(ndc*depth,clamp(depth/50.0,0.0,1.0)*depth,depth);
}
@vertex fn surfaceVertex(@builtin(vertex_index) index:u32)->Out {
  let v=vertices[index]; var o:Out; o.clip=project(v.position.xyz);o.world=v.position.xyz;o.normal=normalize(v.normal.xyz);return o;
}
struct SurfaceOut { @location(0) position:vec4f, @location(1) normal:vec4f }
@fragment fn surfaceFragment(input:Out)->SurfaceOut {
  var o:SurfaceOut;o.position=vec4f(input.world,1.0);o.normal=vec4f(normalize(input.normal),1.0);return o;
}
`;

export const causticShader = /* wgsl */ `
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f }
struct SurfaceVertex { position:vec4f, normal:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> vertices:array<SurfaceVertex>;
struct Out { @builtin(position) clip:vec4f, @location(0) energy:f32 }
@vertex fn causticVertex(@builtin(vertex_index) index:u32)->Out {
  let v=vertices[index];let n=normalize(v.normal.xyz);let towardLight=normalize(vec3f(-0.45,0.86,0.28));
  let transmitted=refract(-towardLight,n,0.75019);
  let downward=transmitted.y<-.02;
  let distance=select(0.0,clamp((.006-v.position.y)/min(transmitted.y,-.02),0.0,u.container.y*2.0),downward);
  var floorPoint=v.position.xyz+transmitted*distance;
  floorPoint.x=clamp(floorPoint.x,-u.container.x*.499,u.container.x*.499);
  floorPoint.z=clamp(floorPoint.z,-u.container.z*.499,u.container.z*.499);
  var o:Out;o.clip=vec4f(2.0*floorPoint.x/u.container.x,2.0*floorPoint.z/u.container.z,0.0,1.0);
  let topFacing=smoothstep(.18,.62,n.y);o.energy=select(0.0,(.012+.045*n.y*n.y)*topFacing,downward);return o;
}
@fragment fn causticFragment(input:Out)->@location(0) vec4f{if(input.energy<.0005){discard;}return vec4f(input.energy*vec3f(0.63,0.96,0.86),input.energy);}
`;

export const sceneShader = /* wgsl */ `
const ENABLE_CAUSTICS = false;
struct Uniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f, options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f, terrainMeta:vec4f, terrainFeatures:array<vec4f,16> }
struct BodyGPU { positionRadius:vec4f, halfSizeShape:vec4f, orientation:vec4f, colorSelected:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> bodies:array<BodyGPU,12>;
@group(0) @binding(2) var caustics:texture_2d<f32>;
@group(0) @binding(3) var linearSampler:sampler;
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
fn boxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let n=min(a,b);let f=max(a,b);return vec2f(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
${environmentShaderLibrary}
${unifiedLightingShaderLibrary}
fn qrot(q:vec4f,v:vec3f)->vec3f{let a=cross(q.yzw,v);return v+2.0*(q.x*a+cross(q.yzw,a));}
fn qinv(q:vec4f,v:vec3f)->vec3f{return qrot(vec4f(q.x,-q.yzw),v);}
struct Hit{t:f32,n:vec3f,color:vec3f,selected:f32}
fn bodyHit(ro:vec3f,rd:vec3f,b:BodyGPU)->Hit{
  let o=qinv(b.orientation,ro-b.positionRadius.xyz);let d=qinv(b.orientation,rd);let shape=i32(round(b.halfSizeShape.w));var t=1e20;var n=vec3f(0,1,0);
  if(shape==0){let radius=b.halfSizeShape.x;let h=dot(o,d);let disc=h*h-dot(o,o)+radius*radius;if(disc>=0.0){t=-h-sqrt(disc);if(t<=1e-4){t=-h+sqrt(disc);}if(t>1e-4){n=normalize(o+d*t);}else{t=1e20;}}}
  else if(shape==1){let h=boxHit(o,d,-b.halfSizeShape.xyz,b.halfSizeShape.xyz);t=select(h.x,h.y,h.x<=1e-4);if(t>1e-4&&h.x<=h.y){let p=o+d*t;let q=abs(p/max(b.halfSizeShape.xyz,vec3f(1e-5)));if(q.x>=q.y&&q.x>=q.z){n=vec3f(sign(p.x),0,0);}else if(q.y>=q.z){n=vec3f(0,sign(p.y),0);}else{n=vec3f(0,0,sign(p.z));}}else{t=1e20;}}
  else {let radius=b.halfSizeShape.x;let hh=b.halfSizeShape.y;let a=dot(d.xz,d.xz);let bb=dot(o.xz,d.xz);let cc=dot(o.xz,o.xz)-radius*radius;if(a>1e-7&&bb*bb-a*cc>=0.0){t=(-bb-sqrt(bb*bb-a*cc))/a;let y=o.y+d.y*t;if(t<=1e-4||abs(y)>hh){t=1e20;}else{let p=o+d*t;n=normalize(vec3f(p.x,0,p.z));}}if(shape==2){let capY=select(-hh,hh,d.y<0.0);let center=vec3f(0,capY,0);let oc=o-center;let h=dot(oc,d);let disc=h*h-dot(oc,oc)+radius*radius;if(disc>=0.0){let st=-h-sqrt(disc);if(st>1e-4&&st<t){t=st;n=normalize(oc+d*t);}}}else if(abs(d.y)>1e-7){for(var side=-1.0;side<=1.0;side+=2.0){let ct=(side*hh-o.y)/d.y;let cp=o+d*ct;if(ct>1e-4&&ct<t&&dot(cp.xz,cp.xz)<=radius*radius){t=ct;n=vec3f(0,side,0);}}}}
  return Hit(t,qrot(b.orientation,n),b.colorSelected.xyz,b.colorSelected.w);
}
fn nearestBody(ro:vec3f,rd:vec3f)->Hit{var best=Hit(1e20,vec3f(0,1,0),vec3f(.7),0);for(var i=0u;i<12u;i+=1u){if(i>=u32(round(u.options.z))){break;}let h=bodyHit(ro,rd,bodies[i]);if(h.t<best.t){best=h;}}return best;}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  let ndc=input.uv*2.0-1.0;let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*.72+up*ndc.y*.72);
  let room=sampleEnvironment(ro,rd);var color=room.color;var nearest=room.depth;let light=environmentLightDirection();
  let rigid=nearestBody(ro,rd);if(rigid.t<nearest){let material=unifiedMaterial(rigid.color,1.0,rigid.selected*vec3f(.12,.42,.32),.16,vec3f(.04),0.0,vec3f(.18,.34,.31),1.0);let lighting=unifiedLightingInput(rigid.n,-rd,light,environmentLightColor());color=shadeUnifiedSurface(material,lighting);nearest=rigid.t;}
  // Rear seams belong in the dry scene so they are refracted by the water.
  // The near glass pane and its edges are composited after the water below.
  // The garden pond sits in open ground: there is no glass tank to glow.
  if(environmentIndex()!=7){let size=u.container.xyz;let mn=vec3f(-size.x*.5,0,-size.z*.5);let mx=vec3f(size.x*.5,size.y,size.z*.5);let tank=boxHit(ro,rd,mn,mx);if(tank.x<=tank.y&&tank.y>0.0){let p=ro+rd*tank.y;let local=abs((p-(mn+mx)*.5)/(size*.5));let edge=max(max(min(local.x,local.y),min(local.x,local.z)),min(local.y,local.z));let edgeGlow=smoothstep(.955,.998,edge);color+=vec3f(.11,.25,.23)*edgeGlow*.42;}}
  let vignette=1.0-.16*dot(ndc*.58,ndc*.58);return vec4f(color*vignette,nearest);
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
override adaptiveSurface=false;
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
fn project(world:vec3f)->vec2f{let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(f,vec3f(0,1,0)));let up=normalize(cross(r,f));let q=world-u.cameraPosition.xyz;let d=max(dot(q,f),1e-4);let ndc=vec2f(dot(q,r)/(d*u.viewport.x/max(u.viewport.y,1.0)*.72),dot(q,up)/(d*.72));return vec2f(ndc.x*.5+.5,.5-ndc.y*.5);}
fn safeSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{return textureSampleLevel(texture,linearSampler,clamp(uv,vec2f(.001),vec2f(.999)),0);}
fn boxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let near3=min(a,b);let far3=max(a,b);return vec2f(max(max(near3.x,near3.y),near3.z),min(min(far3.x,far3.y),far3.z));}
${environmentShaderLibrary}
${unifiedLightingShaderLibrary}
${unifiedDisplayTransferShaderLibrary}
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
fn finish(color:vec3f,ndc:vec2f)->vec4f{let c=environmentForeground(color,ndc)*(1.0-.08*dot(ndc*.55,ndc*.55));return vec4f(unifiedDisplayTransfer(c),1);}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  // Full-screen interpolated UV has Y=1 at the top of the render target,
  // while sampled WebGPU textures have Y=0 there. The shared legacy upscaler
  // performs the same conversion for the final target; all raster-path
  // intermediate reads and world projections must do it here as well.
  let ndc=input.uv*2.0-1.0;let textureUV=vec2f(input.uv.x,1.0-input.uv.y);let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*.72+up*ndc.y*.72);
  let scene=safeSample(sceneTexture,textureUV);var front=safeSample(frontPosition,textureUV);if(front.a<.5){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}var frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));let depthEpsilon=max(.0015,.18*cellSize);
  var n=normalize(safeSample(frontNormal,textureUV).xyz);let rigidFront=nearestRigid(ro,rd);let contactBand=${CONTACT_RESOLVE_BAND_CELLS.toFixed(1)}*cellSize;
  if(!adaptiveSurface&&u.gridInfo.w>.5&&rigidFront.t<1e19&&abs(rigidFront.t-frontDepth)<=contactBand){let contact=refineContactSurface(ro,rd,frontDepth,cellSize);if(contact.valid){front=vec4f(contact.point,1);frontDepth=dot(contact.point-ro,rd);n=contact.normal;}if(rigidFront.t<=frontDepth+max(3e-4,.03*cellSize)){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}}
  if(resolvedDrySceneDepth(scene.a)+depthEpsilon<frontDepth){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}
  if(dot(n,rd)>0.0){n=-n;}let etaIn=1.0/${WATER_OPTICS.indexOfRefraction.toFixed(3)};var inside=refract(rd,n,etaIn);if(length(inside)<1e-5){inside=reflect(rd,n);}
  var exitUV=textureUV;var back=vec4f(0);var exitN=vec3f(0,-1,0);
  for(var iteration=0;iteration<3;iteration+=1){back=safeSample(backPosition,exitUV);if(back.a<.5){break;}let backDepth=dot(back.xyz-ro,forward);let frontPlane=dot(front.xyz-ro,forward);let travel=max(0.0,(backDepth-frontPlane)/max(dot(inside,forward),.001));exitUV=project(front.xyz+inside*travel);exitN=normalize(safeSample(backNormal,exitUV).xyz);}
  let refinedBack=safeSample(backPosition,exitUV);if(refinedBack.a>.5){back=refinedBack;exitN=normalize(safeSample(backNormal,exitUV).xyz);}
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
  if(!opaqueSolidExit){if(dot(exitN,inside)<0.0){exitN=-exitN;}outgoing=refract(inside,-exitN,${WATER_OPTICS.indexOfRefraction.toFixed(3)});tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}backgroundUV=project(exitPoint+outgoing*(.55+.45*thickness));}
  let transmittedScene=safeSample(sceneTexture,backgroundUV).rgb;
  // Clean water: red is attenuated first.  A small in-scattering term keeps
  // thick regions luminous instead of turning into opaque ink.
  let refracted=unifiedAbsorbingTransmission(transmittedScene,vec3f(${WATER_OPTICS.absorption.join(",")}),vec3f(${WATER_OPTICS.scatter.join(",")}),thickness);let reflectedDir=reflect(rd,n);var reflected=environmentLight(reflectedDir);
  let ssrUV=project(front.xyz+reflectedDir*.8);let ssr=safeSample(sceneTexture,ssrUV);reflected=mix(reflected,ssr.rgb,select(0.0,.32,ssr.a>0.0&&ssr.a<60000.0));
  let cosine=clamp(dot(-rd,n),0.0,1.0);let fresnel=unifiedDielectricFresnel(cosine,${WATER_OPTICS.fresnelF0});var water=mix(refracted,reflected,fresnel);
  if(tir){water=mix(water,environmentLight(outgoing),.88);}
  let light=environmentLightDirection();water+=environmentLightColor()*unifiedSpecularLobe(n,-rd,light,180.0)*1.4;
  // Thin forward-scattering highlight at silhouettes, plus a restrained
  // turquoise body tint that grows only with actual optical thickness.
  water+=vec3f(.018,.10,.085)*(1.0-exp(-thickness*2.4));water+=vec3f(.08,.18,.15)*pow(1.0-cosine,3.0)*.15;
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

export class RasterWaterPipeline {
  /** Temporarily disabled while the projected map is being retuned. */
  private readonly causticsEnabled = false;
  private extractPipeline?: GPUComputePipeline;
  private extractBandPipeline?: GPUComputePipeline;
  private extractTallSidesPipeline?: GPUComputePipeline;
  private extractWallPipeline?: GPUComputePipeline;
  private extractSparsePipeline?: GPUComputePipeline;
  private extractGlobalFinePipeline?: GPUComputePipeline;
  private extractGlobalCoarsePipeline?: GPUComputePipeline;
  private extractAdaptivePipeline?: GPUComputePipeline;
  private extractAdaptiveLeafPipeline?: GPUComputePipeline;
  private extractAdaptiveFallbackPipeline?: GPUComputePipeline;
  private extractAdaptiveLeafFallbackPipeline?: GPUComputePipeline;
  private extractHybridCoarsePipeline?: GPUComputePipeline;
  private resetSurfaceWorklistPipeline?: GPUComputePipeline;
  private preparePipeline?: GPUComputePipeline;
  private polygonisePipeline?: GPUComputePipeline;
  private polygoniseSparsePipeline?: GPUComputePipeline;
  private polygoniseGlobalFineCountPipeline?: GPUComputePipeline;
  private polygoniseGlobalFineScanPipeline?: GPUComputePipeline;
  private polygoniseGlobalFineEmitPipelines?: GPUComputePipeline[];
  private polygoniseGlobalFineEmitPipeline?: GPUComputePipeline;
  private globalFineEmitWorkgroups = 1;
  private polygoniseAdaptivePipeline?: GPUComputePipeline;
  private surfaceFrontPipeline?: GPURenderPipeline;
  private surfaceBackPipeline?: GPURenderPipeline;
  private causticPipeline?: GPURenderPipeline;
  private scenePipeline?: GPURenderPipeline;
  private compositePipeline?: GPURenderPipeline;
  private adaptiveCompositePipeline?: GPURenderPipeline;
  private extractLayout?: GPUBindGroupLayout;
  private globalExtractLayout?: GPUBindGroupLayout;
  private globalPolygoniseLayout?: GPUBindGroupLayout;
  private prepareLayout?: GPUBindGroupLayout;
  private surfaceLayout?: GPUBindGroupLayout;
  private sceneLayout?: GPUBindGroupLayout;
  private compositeLayout?: GPUBindGroupLayout;
  private sampler?: GPUSampler;
  private vertexBuffer?: GPUBuffer;
  private indirectBuffer?: GPUBuffer;
  private activeCubeBuffer?: GPUBuffer;
  private globalCubeValues?: GPUBuffer;
  private globalCubeOffsets?: GPUBuffer;
  private polygoniseDispatchBuffer?: GPUBuffer;
  private extractBindGroup?: GPUBindGroup;
  private globalExtractBindGroup?: GPUBindGroup;
  private globalPolygoniseBindGroup?: GPUBindGroup;
  private prepareBindGroup?: GPUBindGroup;
  private surfaceBindGroup?: GPUBindGroup;
  private sceneBindGroup?: GPUBindGroup;
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
  private causticTexture?: GPUTexture;
  private geometryKey = "";
  private targetKey = "";
  private volume?: GPUTexture;
  private columnBases?: GPUTexture;
  private extractedRevision = -1;
  private lastExtractionAt_ms = -Infinity;
  private causticsValid = false;
  private secondaryParticles?: SecondaryParticleRenderPipeline;
  private sparseSurface?: SparseSurfaceBandGPUSource;
  private adaptiveOctree?: UnifiedOctreeConsumerSource;
  private globalFineLevelSet?: GlobalFineLevelSetConsumerSource;
  private globalFineRenderParams?: GPUBuffer;
  private fallbackSparsePageTable?: GPUBuffer;
  private fallbackSparseActivePages?: GPUBuffer;
  private fallbackSparsePhi?: GPUBuffer;
  private fallbackSparseParams?: GPUBuffer;
  private fallbackSparseControl?: GPUBuffer;
  private adaptiveDiagnosticReadback?: GPUBuffer;
  private adaptiveDiagnosticPending = false;
  private adaptiveDiagnosticCompletion?: Promise<AdaptiveWaterRenderDiagnostics | undefined>;
  private lastAdaptiveDiagnostics?: AdaptiveWaterRenderDiagnostics;
  private pendingAdaptiveDiagnosticShape?: readonly [number, number, number, number];
  private pendingAdaptiveDiagnosticGlobalFine = false;
  private pendingAdaptiveDiagnosticGlobalFineGeneration?: number;
  private lastRasterMeshPublicationGeneration?: number;

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer
  ) {}

  async initialize(onProgress:(label:string,completed:number,total:number)=>void=()=>{}) {
    const [extract, globalClassify, globalCount, globalScan, globalEmit, globalEmitAll, prepare, surface, caustic, scene, composite] = await Promise.all([
      checkedModule(this.device, "Water isosurface extraction", surfaceExtractionShader),
      checkedModule(this.device, "Global fine water classification", globalFineSurfaceClassificationShader),
      checkedModule(this.device, "Classified global fine count", globalFineClassifiedCountShader),
      checkedModule(this.device, "Classified global fine scan", globalFineClassifiedScanShader),
      Promise.all(globalFineClassifiedEmitShaders.map((source, index) =>
        checkedModule(this.device, `Classified global fine tetrahedron ${index}`, source))),
      checkedModule(this.device, "Classified global fine tetrahedra", globalFineClassifiedEmitShader),
      checkedModule(this.device, "Water extraction dispatch prepare", extractionPrepareShader),
      checkedModule(this.device, "Water interface raster", surfaceRasterShader),
      checkedModule(this.device, "Water caustic projection", causticShader),
      checkedModule(this.device, "Water background scene", sceneShader),
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
      ,{ binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
      ,{ binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
    ] });
    this.globalExtractLayout = this.device.createBindGroupLayout({ label: "Global fine water classification bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    this.prepareLayout = this.device.createBindGroupLayout({ label: "Water extraction prepare bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    this.surfaceLayout = this.device.createBindGroupLayout({ label: "Water surface bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
    ] });
    this.sceneLayout = this.device.createBindGroupLayout({ label: "Water scene bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } }
    ] });
    this.compositeLayout = this.device.createBindGroupLayout({ label: "Water composite bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ...[1,2,3,4,5].map((binding) => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } })),
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } }
    ] });
    const extractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.extractLayout] });
    const globalExtractionPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.globalExtractLayout] });
    const total=31;let completed=0;
    const compute=async(label:string,descriptor:GPUComputePipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createComputePipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    const render=async(label:string,descriptor:GPURenderPipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createRenderPipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    this.extractPipeline = await compute("Classifying liquid surface cubes",{ label: "Classify liquid surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractMain" } });
    this.extractBandPipeline = await compute("Classifying restricted water band",{ label: "Classify restricted water band", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractBandMain" } });
    this.extractTallSidesPipeline = await compute("Classifying tall-cell interfaces",{ label: "Classify tall-cell side interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractTallSidesMain" } });
    this.extractWallPipeline = await compute("Classifying water wall interfaces",{ label: "Classify water wall interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractWallMain" } });
    this.extractSparsePipeline = await compute("Classifying sparse fine surface pages",{ label: "Classify sparse fine surface pages", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractSparseMain", constants: { sparseField: 1 } } });
    this.extractGlobalFinePipeline = await compute("Classifying global fine surface bricks",{ label: "Classify global fine surface bricks", layout: globalExtractionPipelineLayout, compute: { module: globalClassify, entryPoint: "extractGlobalFineMain" } });
    this.extractGlobalCoarsePipeline = await compute("Classifying compact coarse surface leaves",{ label: "Classify compact coarse fallback", layout: globalExtractionPipelineLayout, compute: { module: globalClassify, entryPoint: "extractGlobalCoarseMain" } });
    this.extractAdaptivePipeline = await compute("Classifying adaptive octree surface pages",{ label: "Classify adaptive octree surface pages", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractAdaptiveMain", constants: { adaptiveField: 1 } } });
    this.extractAdaptiveLeafPipeline = await compute("Classifying adaptive octree surface leaves",{ label: "Classify adaptive octree surface leaves", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractAdaptiveLeafMain", constants: { adaptiveField: 1 } } });
    this.extractAdaptiveFallbackPipeline = await compute("Classifying adaptive fallback surface pages",{ label: "Classify adaptive fallback surface pages", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractAdaptiveMain", constants: { adaptiveField: 1, globalFineFallback: 1 } } });
    this.extractAdaptiveLeafFallbackPipeline = await compute("Classifying adaptive fallback surface leaves",{ label: "Classify adaptive fallback surface leaves", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractAdaptiveLeafMain", constants: { adaptiveField: 1, globalFineFallback: 1 } } });
    this.extractHybridCoarsePipeline = await compute("Classifying coarse surface outside detail patches",{ label: "Classify hybrid coarse surface", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractHybridCoarseMain" } });
    this.resetSurfaceWorklistPipeline = await compute("Preparing fine detail worklist",{ label: "Reset surface worklist between hierarchy levels", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "resetSurfaceWorklistMain" } });
    this.polygonisePipeline = await compute("Building water surface mesh",{ label: "Polygonise surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain" } });
    this.polygoniseSparsePipeline = await compute("Building sparse fine water mesh",{ label: "Polygonise sparse fine surface", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain", constants: { sparseField: 1 } } });
    const globalPolygonLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.globalPolygoniseLayout]});
    this.polygoniseGlobalFineCountPipeline=await compute("Counting global fine water mesh",{label:"Count classified global fine triangles",layout:globalPolygonLayout,compute:{module:globalCount,entryPoint:"countGlobalFineTriangles"}});
    this.polygoniseGlobalFineScanPipeline=await compute("Scanning global fine water mesh",{label:"Scan classified global fine triangles",layout:globalPolygonLayout,compute:{module:globalScan,entryPoint:"scanGlobalFineTriangles"}});
    this.polygoniseGlobalFineEmitPipelines=[];
    for(let tetrahedron=0;tetrahedron<6;tetrahedron+=1)this.polygoniseGlobalFineEmitPipelines.push(await compute(`Emitting global fine tetrahedron ${tetrahedron+1}/6`,{label:`Emit classified global fine tetrahedron ${tetrahedron}`,layout:globalPolygonLayout,compute:{module:globalEmit[tetrahedron],entryPoint:`emitGlobalFineTetra${tetrahedron}`}}));
    this.polygoniseGlobalFineEmitPipeline=await compute("Emitting six global fine tetrahedra",{label:"Emit classified global fine tetrahedra",layout:globalPolygonLayout,compute:{module:globalEmitAll,entryPoint:"emitGlobalFineTetrahedra"}});
    this.polygoniseAdaptivePipeline = await compute("Building adaptive octree water mesh",{ label: "Polygonise adaptive octree surface", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain", constants: { adaptiveField: 1 } } });
    this.preparePipeline = await compute("Preparing surface dispatch",{ label: "Prepare polygonise dispatch", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareLayout] }), compute: { module: prepare, entryPoint: "prepareMain" } });
    this.polygoniseDispatchBuffer = this.device.createBuffer({ label: "Water polygonise dispatch arguments", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const surfacePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.surfaceLayout] });
    const surfaceDescriptor = (label: string, cullMode: GPUCullMode): GPURenderPipelineDescriptor => ({
      label, layout: surfacePipelineLayout, vertex: { module: surface, entryPoint: "surfaceVertex" },
      fragment: { module: surface, entryPoint: "surfaceFragment", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", frontFace: "ccw", cullMode },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    this.surfaceFrontPipeline = await render("Rendering front water interfaces",surfaceDescriptor("Raster water front interfaces", WATER_INTERFACE_CULL_MODES.front));
    this.surfaceBackPipeline = await render("Rendering back water interfaces",surfaceDescriptor("Raster water back interfaces", WATER_INTERFACE_CULL_MODES.back));
    this.causticPipeline = await render("Projecting water caustics",{
      label: "Project refracted caustics", layout: surfacePipelineLayout, vertex: { module: caustic, entryPoint: "causticVertex" },
      fragment: { module: caustic, entryPoint: "causticFragment", targets: [{ format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.scenePipeline = await render("Rendering the dry scene",{ label: "Render dry scene for water refraction", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout] }), vertex: { module: scene, entryPoint: "vertexMain" }, fragment: { module: scene, entryPoint: "fragmentMain", targets: [{ format: "rgba16float" }] }, primitive: { topology: "triangle-list" } });
    const compositePipelineLayout=this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] });
    const compositeDescriptor=(label:string,adaptiveSurface:number):GPURenderPipelineDescriptor=>({ label, layout: compositePipelineLayout, vertex: { module: composite, entryPoint: "vertexMain" }, fragment: { module: composite, entryPoint: "fragmentMain", constants:{adaptiveSurface}, targets: [{ format: this.targetFormat }] }, primitive: { topology: "triangle-list" } });
    this.compositePipeline = await render("Compositing water optics",compositeDescriptor("Composite two-interface water optics",0));
    this.adaptiveCompositePipeline = await render("Compositing adaptive water optics",compositeDescriptor("Composite adaptive two-interface water optics",1));
    this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    this.fallbackSparsePageTable = this.device.createBuffer({ label: "Water sparse-page fallback", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fallbackSparseActivePages = this.device.createBuffer({ label: "Water sparse-active fallback", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.fallbackSparsePhi = this.device.createBuffer({ label: "Water sparse-phi fallback", size: 4, usage: GPUBufferUsage.STORAGE });
    this.fallbackSparseParams = this.device.createBuffer({ label: "Water sparse-params fallback", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.globalFineRenderParams = this.device.createBuffer({ label: "Water global fine parameters", size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // This disabled source also occupies the adaptive-leaf binding. WebGPU
    // validates the complete 64-byte leaf/control minimum before overrides.
    this.fallbackSparseControl = this.device.createBuffer({ label: "Water sparse-control fallback", size: WATER_SPARSE_FALLBACK_BYTES, usage: GPUBufferUsage.STORAGE });
    this.rebuildBindGroups();
  }

  setVolume(texture: GPUTexture, columnBases: GPUTexture) {
    if (this.volume === texture && this.columnBases === columnBases) return;
    this.volume = texture; this.columnBases = columnBases; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  setSparseSurface(source: SparseSurfaceBandGPUSource | undefined) {
    if (this.sparseSurface === source) return;
    this.sparseSurface = source;
    this.extractedRevision = -1;
    this.lastExtractionAt_ms = -Infinity;
    this.causticsValid = false;
    this.geometryKey = "";
    this.rebuildBindGroups();
  }

  /** Selects the authoritative leaf-page surface without copying or publishing a 3D texture. */
  setAdaptiveOctree(source: UnifiedOctreeConsumerSource | undefined) {
    if (source) validateUnifiedOctreeConsumerSource(source);
    if (source && source.leafCapacity > 0x01000000) throw new RangeError("Adaptive water extraction supports at most 2^24 leaf rows");
    if (this.adaptiveOctree === source) return;
    this.adaptiveOctree = source;
    this.extractedRevision = -1;
    this.lastExtractionAt_ms = -Infinity;
    this.causticsValid = false;
    this.geometryKey = "";
    this.rebuildBindGroups();
  }

  /** Selects row-independent global fine bricks without synthesizing leaf ownership. */
  setGlobalFineLevelSet(source: GlobalFineLevelSetConsumerSource | undefined) {
    if (source) validateGlobalFineLevelSetConsumerSource(source);
    const previous = this.globalFineLevelSet;
    if (previous === source || (previous && source
      && previous.generation === source.generation
      && previous.hash.buffer === source.hash.buffer
      && previous.metadata.buffer === source.metadata.buffer
      && previous.worklist.buffer === source.worklist.buffer
      && previous.flags.buffer === source.flags.buffer
      && previous.phi.buffer === source.phi.buffer
      && previous.coarsePhiDirectory?.buffer === source.coarsePhiDirectory?.buffer
      && previous.coarsePhiHashCapacity === source.coarsePhiHashCapacity
      && previous.topologyControl?.buffer === source.topologyControl?.buffer)) return;
    this.globalFineLevelSet = source;
    if (source && this.globalFineRenderParams) {
      const bytes = new ArrayBuffer(112); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
      u32.set([...source.sampleDimensions, source.brickResolution], 0);
      u32.set([...source.brickDimensions, source.samplesPerBrick], 4);
      u32.set([source.hashCapacity, source.maximumHashProbes, source.pageCapacity, source.generation], 8);
      f32.set([...source.domainOrigin, source.fineCellWidth], 12); f32[16] = source.fineFactor;
      this.device.queue.writeBuffer(this.globalFineRenderParams, 0, bytes);
    }
    this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false;
    // Keep same-shaped geometry alive across A/B source publication. The next
    // encode still calls ensureGeometry(), so a genuine dimension change
    // reallocates; clearing the key here destroyed A before B could prove its
    // tags and defeated the fail-closed retained-mesh contract.
    this.rebuildBindGroups();
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

  /** Latest opt-in (`?waterdiag=1`) GPU readback for sparse presentation debugging. */
  get adaptiveRenderDiagnostics() { return this.lastAdaptiveDiagnostics; }

  private adaptiveDiagnosticsEnabled() {
    if (typeof location !== "undefined") {
      const query = new URLSearchParams(location.search);
      if (query.get("waterdiag") === "1" || query.get("diagnostics") === "1" || query.get("panel") === "diagnostics" || query.get("panel") === "visual") return true;
    }
    return typeof process !== "undefined" && process.env?.FLUID_WATER_DIAGNOSTICS === "1";
  }

  /** Whether this session requested the bounded presentation readback. */
  get adaptiveDiagnosticsReadbackEnabled() { return this.adaptiveDiagnosticsEnabled(); }

  private encodeAdaptiveDiagnostics(encoder: GPUCommandEncoder, adaptive: UnifiedOctreeConsumerSource) {
    if (!this.adaptiveDiagnosticsEnabled() || this.adaptiveDiagnosticPending || !this.indirectBuffer) return;
    this.adaptiveDiagnosticReadback?.destroy();
    this.adaptiveDiagnosticReadback = this.device.createBuffer({ label: "Adaptive water render diagnostics", size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const activeOffsetBytes = (16 + 2 * adaptive.leafCapacity + 2 * adaptive.pageCapacity) * 4;
    encoder.copyBufferToBuffer(adaptive.surfaceArena.buffer, adaptive.surfaceArena.offset ?? 0, this.adaptiveDiagnosticReadback, 0, 64);
    encoder.copyBufferToBuffer(adaptive.surfaceArena.buffer, (adaptive.surfaceArena.offset ?? 0) + activeOffsetBytes, this.adaptiveDiagnosticReadback, 64, 16);
    encoder.copyBufferToBuffer(this.indirectBuffer, 0, this.adaptiveDiagnosticReadback, 80, 28);
    this.pendingAdaptiveDiagnosticShape = [adaptive.leafCapacity, adaptive.pageCapacity, adaptive.pageResolution, adaptive.pageResolution ** 3];
    this.pendingAdaptiveDiagnosticGlobalFine = Boolean(this.globalFineLevelSet);
    this.pendingAdaptiveDiagnosticGlobalFineGeneration = this.globalFineLevelSet?.generation;
    this.adaptiveDiagnosticPending = true;
  }

  /** Called immediately after the frame submission that contains the copies. */
  completeAdaptiveDiagnostics(): Promise<AdaptiveWaterRenderDiagnostics | undefined> {
    if (this.adaptiveDiagnosticCompletion) return this.adaptiveDiagnosticCompletion;
    const readback = this.adaptiveDiagnosticReadback;
    if (!readback || !this.adaptiveDiagnosticPending) return Promise.resolve(undefined);
    const completion = this.device.queue.onSubmittedWorkDone().then(async () => {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const shape = this.pendingAdaptiveDiagnosticShape ?? [0, 0, 0, 0];
      const groups = Math.max(1, Math.ceil(words[16] * shape[3] / 256));
      const globalFineAttached = this.pendingAdaptiveDiagnosticGlobalFine;
      const globalFineAttachedGeneration = this.pendingAdaptiveDiagnosticGlobalFineGeneration;
      const surfaceGeometrySource = waterSurfaceGeometrySource(globalFineAttached, words[20], words[26], words[25]);
      if (surfaceGeometrySource === "global-fine-coarse") {
        this.lastRasterMeshPublicationGeneration = globalFineAttachedGeneration;
      } else if (surfaceGeometrySource !== "retained-previous") {
        this.lastRasterMeshPublicationGeneration = undefined;
      }
      this.lastAdaptiveDiagnostics = {
        leafCapacity: shape[0], pageCapacity: shape[1], pageResolution: shape[2], samplesPerPage: shape[3],
        surfaceFreePages: words[0], surfaceAllocatedPages: words[4], surfaceCandidatePages: words[12], surfaceActivePages: words[16], surfaceOverflow: words[3],
        finestResidentPages: words[8], coarseResidentPages: words[9], maximumResidentLeafSize: words[10],
        surfaceDispatch: [Math.min(65_535, groups), Math.max(1, Math.ceil(groups / 65_535)), 1],
        vertexCount: words[20], activeCubeCount: words[24], vertexAllocator: words[25],
        globalFineAuthorityLatch: words[26],
        surfaceGeometrySource,
        globalFineAttached,
        globalFineAttachedGeneration,
        meshPublicationGeneration: this.lastRasterMeshPublicationGeneration,
        globalFineCrossingPublished: surfaceGeometrySource === "global-fine-coarse",
        presentationFallbackActive: surfaceGeometrySource === "adaptive-fallback" || surfaceGeometrySource === "retained-previous",
      };
      console.info("Adaptive water diagnostics", JSON.stringify(this.lastAdaptiveDiagnostics));
      const result = this.lastAdaptiveDiagnostics;
      readback.unmap();
      return result;
    }).catch(() => undefined).finally(() => {
      this.adaptiveDiagnosticPending = false;
      if (this.adaptiveDiagnosticCompletion === completion) this.adaptiveDiagnosticCompletion = undefined;
    });
    this.adaptiveDiagnosticCompletion = completion;
    return completion;
  }

  private ensureGeometry(nx: number, ny: number, nz: number) {
    const key = `${nx}x${ny}x${nz}`;
    if (key === this.geometryKey) return;
    this.vertexBuffer?.destroy(); this.indirectBuffer?.destroy(); this.activeCubeBuffer?.destroy(); this.globalCubeValues?.destroy(); this.globalCubeOffsets?.destroy();
    // Surface area, not volume, controls the normal case.  The generous factor
    // also covers breaking sheets and entrained blobs while imposing a hard
    // 64 MiB ceiling on adversarial checkerboard fields.
    const maxVertices = surfaceVertexCapacity(nx, ny, nz);
    this.vertexBuffer = this.device.createBuffer({ label: `Extracted water surface (${maxVertices} vertices)`, size: maxVertices * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // The first 16 bytes are the standard draw-indirect ABI. Renderer-private
    // counters and the global-fine authority latch trail it; firstInstance must
    // stay zero unless the optional indirect-first-instance feature is enabled.
    this.indirectBuffer = this.device.createBuffer({ label: "Water indirect draw arguments and extraction counters", size: 28, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.activeCubeBuffer = this.device.createBuffer({ label: "Water surface cube worklist", size: activeCubeCapacity(maxVertices) * 8, usage: GPUBufferUsage.STORAGE });
    this.globalCubeValues = this.device.createBuffer({ label: "Global fine classified cube values", size: activeCubeCapacity(maxVertices) * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.globalCubeOffsets = this.device.createBuffer({ label: "Global fine tetrahedron offsets", size: activeCubeCapacity(maxVertices) * 6 * 4, usage: GPUBufferUsage.STORAGE });
    this.globalFineEmitWorkgroups=Math.ceil(activeCubeCapacity(maxVertices)/64);
    this.lastRasterMeshPublicationGeneration = undefined;
    this.geometryKey = key; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  ensureSize(width: number, height: number) {
    const key = `${width}x${height}`;
    if (key === this.targetKey) return;
    for (const texture of [this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth]) texture?.destroy();
    const sampledTarget = (label: string) => this.device.createTexture({ label, size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.sceneTexture = this.device.createTexture({ label: "Dry scene HDR", size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }); this.sceneTextureView = this.sceneTexture.createView(); this.frontPosition = sampledTarget("Water front positions"); this.frontNormal = sampledTarget("Water front normals"); this.backPosition = sampledTarget("Water back positions"); this.backNormal = sampledTarget("Water back normals");
    const depth = (label: string) => this.device.createTexture({ label, size: [width,height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.frontDepth = depth("Water front depth"); this.backDepth = depth("Water back depth");
    this.causticTexture?.destroy(); this.causticTexture = this.device.createTexture({ label: "Refracted floor caustics", size: [384,384], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.causticsValid = false;
    this.targetKey = key; this.rebuildBindGroups();
  }

  private rebuildBindGroups() {
    this.compositeBindGroups = new WeakMap();
    const sparse = this.sparseSurface;
    const adaptive = this.adaptiveOctree;
    const globalFine = this.globalFineLevelSet;
    if (this.extractLayout && this.volume && this.columnBases && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.fallbackSparsePageTable && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseParams && this.fallbackSparseControl) this.extractBindGroup = this.device.createBindGroup({ layout: this.extractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 2, resource: this.columnBases.createView() }, { binding: 3, resource: { buffer: this.vertexBuffer } }, { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 7, resource: globalFine?.hash ?? sparse?.pageTable ?? { buffer: this.fallbackSparsePageTable } },
      { binding: 8, resource: globalFine?.worklist ?? sparse?.activePages ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.phi ?? sparse?.phi ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: globalFine ? { buffer: this.globalFineRenderParams! } : sparse?.params ?? { buffer: this.fallbackSparseParams } },
      { binding: 11, resource: globalFine?.flags ?? sparse?.control ?? { buffer: this.fallbackSparseControl } },
      { binding: 12, resource: globalFine?.metadata ?? sparse?.states ?? { buffer: this.fallbackSparseControl } }
      ,{ binding: 13, resource: adaptive?.leaves ?? { buffer: this.fallbackSparseControl } }
      ,{ binding: 14, resource: adaptive?.surfaceArena ?? { buffer: this.fallbackSparseControl } }
      ,{ binding: 15, resource: adaptive?.surfaceParams ?? { buffer: this.fallbackSparseParams } }
    ] });
    if (this.globalExtractLayout && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalFineRenderParams && this.fallbackSparsePageTable && this.fallbackSparseActivePages && this.fallbackSparsePhi && this.fallbackSparseControl) this.globalExtractBindGroup = this.device.createBindGroup({ layout: this.globalExtractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } },
      { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 6, resource: { buffer: this.globalCubeValues } },
      { binding: 7, resource: globalFine?.hash ?? { buffer: this.fallbackSparsePageTable } },
      { binding: 8, resource: globalFine?.worklist ?? { buffer: this.fallbackSparseActivePages } },
      { binding: 9, resource: globalFine?.phi ?? { buffer: this.fallbackSparsePhi } },
      { binding: 10, resource: { buffer: this.globalFineRenderParams } },
      { binding: 11, resource: globalFine?.flags ?? { buffer: this.fallbackSparseControl } },
      { binding: 12, resource: globalFine?.metadata ?? { buffer: this.fallbackSparseControl } },
      { binding: 16, resource: globalFine?.coarsePhiDirectory ?? { buffer: this.fallbackSparseControl } },
      { binding: 17, resource: globalFine?.topologyControl ?? { buffer: this.fallbackSparseControl } },
    ] });
    if (this.globalPolygoniseLayout && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.globalCubeValues && this.globalCubeOffsets && this.globalFineRenderParams) this.globalPolygoniseBindGroup = this.device.createBindGroup({ layout: this.globalPolygoniseLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 3, resource: { buffer: this.vertexBuffer } },
      { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } },
      { binding: 6, resource: { buffer: this.globalCubeValues } }, { binding: 7, resource: { buffer: this.globalCubeOffsets } },
      { binding: 10, resource: { buffer: this.globalFineRenderParams } },
    ] });
    if (this.prepareLayout && this.indirectBuffer && this.activeCubeBuffer && this.polygoniseDispatchBuffer) this.prepareBindGroup = this.device.createBindGroup({ layout: this.prepareLayout, entries: [
      { binding: 0, resource: { buffer: this.indirectBuffer } }, { binding: 1, resource: { buffer: this.activeCubeBuffer } }, { binding: 2, resource: { buffer: this.polygoniseDispatchBuffer } }
    ] });
    if (this.surfaceLayout && this.vertexBuffer) this.surfaceBindGroup = this.device.createBindGroup({ layout: this.surfaceLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.vertexBuffer } }] });
    if (this.sceneLayout && this.causticTexture && this.sampler) this.sceneBindGroup = this.device.createBindGroup({ layout: this.sceneLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } }, { binding: 2, resource: this.causticTexture.createView() }, { binding: 3, resource: this.sampler }] });
    this.compositeBindGroup = this.sceneTextureView ? this.compositeBindGroupFor(this.sceneTextureView) : undefined;
  }

  private compositeBindGroupFor(sceneView: GPUTextureView): GPUBindGroup | undefined {
    const cached = this.compositeBindGroups.get(sceneView);
    if (cached) return cached;
    if (!this.compositeLayout || !this.frontPosition || !this.frontNormal || !this.backPosition || !this.backNormal || !this.sampler || !this.volume || !this.columnBases) return undefined;
    const bindGroup = this.device.createBindGroup({ layout: this.compositeLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: sceneView }, { binding: 2, resource: this.frontPosition.createView() }, { binding: 3, resource: this.frontNormal.createView() }, { binding: 4, resource: this.backPosition.createView() }, { binding: 5, resource: this.backNormal.createView() }, { binding: 6, resource: this.sampler }, { binding: 7, resource: { buffer: this.bodyBuffer } }, { binding: 8, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 9, resource: this.columnBases.createView() }
    ] });
    this.compositeBindGroups.set(sceneView, bindGroup);
    return bindGroup;
  }

  encode(encoder: GPUCommandEncoder, output: GPUTexture | GPUTextureView, nx: number, ny: number, nz: number, restrictedTallCell: boolean, maximumNeighborDelta: number, revision: number, timestamps?: RasterWaterTimestampRanges, drySceneReplacement?: DrySceneReplacementEncoder): RasterWaterEncodeResult | false {
    const geometryDimensions = this.globalFineLevelSet?.sampleDimensions ?? (this.adaptiveOctree
      ? [nx * this.adaptiveOctree.pageResolution, ny * this.adaptiveOctree.pageResolution, nz * this.adaptiveOctree.pageResolution] as const
      : this.sparseSurface?.fineDimensions ?? [nx, ny, nz] as const);
    this.ensureGeometry(geometryDimensions[0],geometryDimensions[1],geometryDimensions[2]);
    if (!this.extractPipeline||!this.extractBandPipeline||!this.extractTallSidesPipeline||!this.extractWallPipeline||!this.extractSparsePipeline||!this.extractGlobalFinePipeline||!this.extractGlobalCoarsePipeline||!this.extractAdaptivePipeline||!this.extractAdaptiveLeafPipeline||!this.extractAdaptiveFallbackPipeline||!this.extractAdaptiveLeafFallbackPipeline||!this.extractHybridCoarsePipeline||!this.resetSurfaceWorklistPipeline||!this.preparePipeline||!this.polygonisePipeline||!this.polygoniseSparsePipeline||!this.polygoniseGlobalFineCountPipeline||!this.polygoniseGlobalFineScanPipeline||this.polygoniseGlobalFineEmitPipelines?.length!==6||!this.polygoniseGlobalFineEmitPipeline||!this.polygoniseAdaptivePipeline||!this.surfaceFrontPipeline||!this.surfaceBackPipeline||!this.causticPipeline||!this.scenePipeline||!this.compositePipeline||!this.adaptiveCompositePipeline||!this.extractBindGroup||!this.globalExtractBindGroup||!this.globalPolygoniseBindGroup||!this.prepareBindGroup||!this.surfaceBindGroup||!this.sceneBindGroup||!this.compositeBindGroup||!this.indirectBuffer||!this.polygoniseDispatchBuffer||!this.volume||!this.sceneTexture||!this.frontPosition||!this.frontNormal||!this.frontDepth||!this.backPosition||!this.backNormal||!this.backDepth||!this.causticTexture) return false;
    const now_ms = performance.now();
    // Rendering follows the newest available solver revision, but extraction
    // follows the fixed presentation cadence. Unchanged solver revisions
    // retain the existing mesh, so pausing does not create redundant work.
    const updateSurface = shouldUpdateWaterSurface(this.extractedRevision, revision, this.lastExtractionAt_ms, now_ms);
    const updateCaustics = this.causticsEnabled && (updateSurface || !this.causticsValid);
    if (updateSurface) {
      if (this.globalFineLevelSet) {
        // Preserve the last published draw count while the GPU validates the
        // next A/B generation. Classification clears the sentinel only after
        // observing finite tagged fine data or a published compact-coarse
        // fallback; an invalid generation therefore retains the previous mesh.
        this.device.queue.writeBuffer(this.indirectBuffer,4,new Uint32Array([1,0,0,0,0xffff_ffff,0]));
      } else {
        this.device.queue.writeBuffer(this.indirectBuffer,0,new Uint32Array([0,1,0,0,0,0,0]));
      }
      const plan = surfaceExtractionDispatchPlan(nx, ny, nz, this.volume.depthOrArrayLayers, restrictedTallCell, maximumNeighborDelta);
      // Classify appends surface-crossing cubes to the worklist, the prepare
      // kernel sizes the indirect dispatch, and polygonise emits triangles for
      // just those cubes. Dispatches in one pass order their storage writes.
      const compute=encoder.beginComputePass({label:"Extract water isosurface",...(timestamps?{timestampWrites:timestamps.extraction}:{})});compute.setBindGroup(0,this.extractBindGroup);
      const adaptive = this.adaptiveOctree;
      const globalFine = this.globalFineLevelSet;
      if (globalFine) {
        compute.setBindGroup(0, this.globalExtractBindGroup);
        compute.setPipeline(this.extractGlobalFinePipeline);
        compute.dispatchWorkgroups(...globalFineSurfaceDispatch(globalFine.pageCapacity, globalFine.samplesPerBrick));
        if(globalFine.coarsePhiHashCapacity){compute.setPipeline(this.extractGlobalCoarsePipeline);compute.dispatchWorkgroups(Math.ceil(globalFine.coarsePhiHashCapacity/256));}
        compute.setBindGroup(0,this.globalPolygoniseBindGroup);
        compute.setPipeline(this.polygoniseGlobalFineScanPipeline);compute.dispatchWorkgroups(1);
        compute.setPipeline(this.polygoniseGlobalFineEmitPipeline);compute.dispatchWorkgroups(this.globalFineEmitWorkgroups,6);
        // A newly attached renderer has no old mesh to retain. If neither the
        // tagged fine generation nor compact coarse authority found a crossing,
        // let the existing leaf-page representation publish the fallback mesh.
        // The trailing renderer-private word is a GPU-only authority latch set
        // by global crossings; the draw ABI's firstInstance remains zero.
        // validation errors remain visible and no readback steers rendering.
        if(adaptive){
          compute.setBindGroup(0,this.extractBindGroup);
          compute.setPipeline(this.extractAdaptiveLeafFallbackPipeline);compute.dispatchWorkgroups(Math.ceil(adaptive.leafCapacity/256));
          compute.setPipeline(this.extractAdaptiveFallbackPipeline);compute.dispatchWorkgroupsIndirect(adaptive.surfaceDispatch.buffer,adaptive.surfaceDispatch.offsetBytes);
          compute.setPipeline(this.preparePipeline);compute.setBindGroup(0,this.prepareBindGroup);compute.dispatchWorkgroups(1);
          compute.setPipeline(this.polygoniseAdaptivePipeline);compute.setBindGroup(0,this.extractBindGroup);compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer,0);
        }
      } else if (surfaceExtractionRepresentation(Boolean(adaptive), Boolean(this.sparseSurface)) === "adaptive-octree" && adaptive) {
        compute.setPipeline(this.extractAdaptiveLeafPipeline);
        compute.dispatchWorkgroups(Math.ceil(adaptive.leafCapacity / 256));
        compute.setPipeline(this.extractAdaptivePipeline);
        compute.dispatchWorkgroupsIndirect(adaptive.surfaceDispatch.buffer, adaptive.surfaceDispatch.offsetBytes);
        compute.setPipeline(this.preparePipeline); compute.setBindGroup(0, this.prepareBindGroup); compute.dispatchWorkgroups(1);
        compute.setPipeline(this.polygoniseAdaptivePipeline); compute.setBindGroup(0, this.extractBindGroup); compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
      } else if (this.sparseSurface) {
        // Level 0: retain the complete coarse surface except inside fine detail
        // cores. Fine halo pages overlap this mesh to make the LOD handoff
        // watertight even though the cell-centred lattices do not share vertices.
        compute.setPipeline(this.extractHybridCoarsePipeline);
        compute.dispatchWorkgroups(Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4));
        compute.setPipeline(this.preparePipeline); compute.setBindGroup(0, this.prepareBindGroup); compute.dispatchWorkgroups(1);
        compute.setPipeline(this.polygonisePipeline); compute.setBindGroup(0, this.extractBindGroup); compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
        // Level 1: reuse the compact worklist while retaining the shared vertex
        // allocator and indirect draw count produced by the coarse pass.
        compute.setPipeline(this.resetSurfaceWorklistPipeline); compute.dispatchWorkgroups(1);
        compute.setPipeline(this.extractSparsePipeline);
        compute.dispatchWorkgroupsIndirect(this.sparseSurface.activePages.buffer, (this.sparseSurface.activePages.offset ?? 0) + 4);
        compute.setPipeline(this.preparePipeline); compute.setBindGroup(0, this.prepareBindGroup); compute.dispatchWorkgroups(1);
        compute.setPipeline(this.polygoniseSparsePipeline); compute.setBindGroup(0, this.extractBindGroup); compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
      } else if (plan.mode === "restricted-band") {
        compute.setPipeline(this.extractBandPipeline); compute.dispatchWorkgroups(...plan.band!);
        compute.setPipeline(this.extractTallSidesPipeline); compute.dispatchWorkgroups(...plan.tallSides!);
        compute.setPipeline(this.extractWallPipeline); compute.dispatchWorkgroups(...plan.walls!);
      } else {
        compute.setPipeline(this.extractPipeline); compute.dispatchWorkgroups(...plan.full!);
      }
      if (!this.sparseSurface && !this.adaptiveOctree) {
        compute.setPipeline(this.preparePipeline); compute.setBindGroup(0, this.prepareBindGroup); compute.dispatchWorkgroups(1);
        compute.setPipeline(this.polygonisePipeline); compute.setBindGroup(0, this.extractBindGroup); compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
      }
      compute.end();
      if (adaptive) this.encodeAdaptiveDiagnostics(encoder, adaptive);
      this.extractedRevision = revision; this.lastExtractionAt_ms = advancePresentationClock(this.lastExtractionAt_ms, now_ms);
    }
    if (updateCaustics) {
      const caustic=encoder.beginRenderPass({label:"Water caustics",colorAttachments:[{view:this.causticTexture.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],...(timestamps?{timestampWrites:timestamps.caustics}:{})});caustic.setPipeline(this.causticPipeline);caustic.setBindGroup(0,this.surfaceBindGroup);caustic.drawIndirect(this.indirectBuffer,0);caustic.end();
      this.causticsValid = true;
    }
    const sparseSceneResult = drySceneReplacement?.(encoder, this.sceneTexture, timestamps?.scene) ?? false;
    if (!sparseSceneResult) {
      const scene=encoder.beginRenderPass({label:"Dry scene",colorAttachments:[{view:this.sceneTextureView!,clearValue:{r:0,g:0,b:0,a:65504},loadOp:"clear",storeOp:"store"}],...(timestamps?{timestampWrites:timestamps.scene}:{})});scene.setPipeline(this.scenePipeline);scene.setBindGroup(0,this.sceneBindGroup);scene.draw(3);scene.end();
    }
    // Water and spray target the same interface attachments and depth state.
    // Encode both draws in one pass per side so spray does not force two extra
    // full-resolution attachment load/store cycles.
    const interfacePass=(label:string,pipeline:GPURenderPipeline,position:GPUTexture,normal:GPUTexture,depth:GPUTexture,side:"front"|"back",timestampWrites?:TimestampRange)=>{const pass=encoder.beginRenderPass({label,colorAttachments:[{view:position.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"},{view:normal.createView(),clearValue:{r:0,g:1,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"},...(timestampWrites?{timestampWrites}:{})});pass.setPipeline(pipeline);pass.setBindGroup(0,this.surfaceBindGroup!);pass.drawIndirect(this.indirectBuffer!,0);this.secondaryParticles?.encodeOpticalInterface(pass,side);pass.end();};
    interfacePass("Water + spray front interfaces",this.surfaceFrontPipeline,this.frontPosition,this.frontNormal,this.frontDepth,"front",timestamps?.frontInterfaces);interfacePass("Water + spray back interfaces",this.surfaceBackPipeline,this.backPosition,this.backNormal,this.backDepth,"back",timestamps?.backInterfaces);
    const compositeBindGroup = sparseSceneResult ? this.compositeBindGroupFor(sparseSceneResult.sampledTargetView) : this.compositeBindGroup;
    if (!compositeBindGroup) return false;
    const outputView="createView" in output?output.createView():output;const composite=encoder.beginRenderPass({label:"Two-interface water composite",colorAttachments:[{view:outputView,clearValue:{r:.01,g:.025,b:.024,a:1},loadOp:"clear",storeOp:"store"}],...(timestamps?{timestampWrites:timestamps.composite}:{})});composite.setPipeline(this.adaptiveOctree&&!this.globalFineLevelSet?this.adaptiveCompositePipeline:this.compositePipeline);composite.setBindGroup(0,compositeBindGroup);composite.draw(3);composite.end();return { surfaceUpdated: updateSurface, causticsUpdated: updateCaustics, sprayRendered: false };
  }

  destroy() {
    for (const resource of [this.vertexBuffer,this.indirectBuffer,this.activeCubeBuffer,this.globalCubeValues,this.globalCubeOffsets,this.polygoniseDispatchBuffer,this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.causticTexture,this.fallbackSparsePageTable,this.fallbackSparseActivePages,this.fallbackSparsePhi,this.fallbackSparseParams,this.globalFineRenderParams,this.fallbackSparseControl,this.adaptiveDiagnosticReadback]) { try { resource?.destroy(); } catch { /* device loss */ } }
  }
}
