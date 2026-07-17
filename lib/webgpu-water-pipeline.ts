import { environmentShaderLibrary } from "./webgpu-environments";
import { advancePresentationClock, frameInterval_ms } from "./frame-pacing";
import type { SecondaryParticleRenderPipeline } from "./webgpu-secondary-particles";

/**
 * Rasterized water presentation for the WebGPU renderer.
 *
 * The solver already owns the liquid volume.  This pipeline keeps that data on
 * the GPU and turns its 0.5 isosurface into triangles every frame.  The result
 * is then rendered twice (front and back interfaces), which gives the optical
 * composite enough information for two-interface refraction without scanning
 * the volume once per screen pixel.
 */

export function shouldUpdateWaterSurface(extractedRevision: number, latestRevision: number, lastExtractionAt_ms: number, now_ms: number, targetFps = 60) {
  return extractedRevision < 0
    || (latestRevision !== extractedRevision && now_ms - lastExtractionAt_ms + 0.5 >= frameInterval_ms(targetFps));
}

/** Raster/body depth separation that activates the local implicit resolver. */
export const CONTACT_RESOLVE_BAND_CELLS = 1.5;

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

type TimestampRange = { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number };
export interface RasterWaterTimestampRanges {
  extraction: TimestampRange;
  scene: TimestampRange;
  frontInterfaces: TimestampRange;
  backInterfaces: TimestampRange;
  composite: TimestampRange;
}

export interface RasterWaterEncodeResult {
  surfaceUpdated: boolean;
}

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

/** Vertex capacity from grid surface area (32 bytes per vertex, 64 MiB cap). */
export function surfaceVertexCapacity(nx: number, ny: number, nz: number) {
  const area = nx * ny + nx * nz + ny * nz;
  return Math.max(262_144, Math.min(2_097_152, area * 32));
}

/**
 * A surface-crossing cube always emits at least one triangle (three
 * vertices), so a worklist of capacity/3 entries can only clip on fields
 * that would clip the vertex buffer as well.
 */
export function activeCubeCapacity(maxVertices: number) {
  return Math.ceil(maxVertices / 3);
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
struct IndirectArgs { vertexCount: atomic<u32>, instanceCount: u32, firstVertex: u32, firstInstance: u32 }
struct ExtractionMeta { activeCubeCount: atomic<u32>, vertexAllocator: atomic<u32> }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var columnBases: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<SurfaceVertex>;
@group(0) @binding(4) var<storage, read_write> drawArgs: IndirectArgs;
@group(0) @binding(5) var<storage, read_write> activeCubes: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> extractionMeta: ExtractionMeta;
override countOnly = false;

// Level-set fields become a smooth occupancy whose 0.5 contour is phi = 0.
// The band spans four cells so no corner of a surface-crossing cube saturates
// (the cube diagonal is under two cells); a saturated corner biases the linear
// crossing estimate and extracts as cell-pitch lattice artifacts.
fn occupancyFromPhi(phi: f32) -> f32 {
  let band = 4.0 * u.container.y / max(u.gridInfo.y, 1.0);
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
  let dims = vec3i(u.gridInfo.xyz);
  // Side/top boundaries are optical interfaces. The floor is a solid contact,
  // not a water-air surface: extend the bottom cell value to y=0 so extraction
  // cannot create a large horizontal sheet across the tank base.
  if (p.x <= 0 || p.z <= 0 || p.x >= dims.x + 1 || p.z >= dims.z + 1 || p.y >= dims.y + 1) { return 0.0; }
  return fieldCell(vec3i(p.x - 1, max(p.y - 1, 0), p.z - 1));
}

fn latticeWorld(p: vec3f) -> vec3f {
  let dims = u.gridInfo.xyz;
  let local = clamp((p - vec3f(0.5)) / dims, vec3f(0.0), vec3f(1.0));
  return vec3f(-0.5 * u.container.x, 0.0, -0.5 * u.container.z) + local * u.container.xyz;
}

// Analytic gradient of the cube's trilinear reconstruction. The eight scalar
// values were already loaded for classification, so this replaces the former
// 48 additional volume loads performed for every emitted vertex normal.
fn surfaceNormal(lattice: vec3f, cubeBase: vec3f, value: ptr<function, array<f32, 8>>) -> vec3f {
  let f = clamp(lattice - cubeBase, vec3f(0.0), vec3f(1.0));
  let dx0 = mix((*value)[1] - (*value)[0], (*value)[2] - (*value)[3], f.y);
  let dx1 = mix((*value)[5] - (*value)[4], (*value)[6] - (*value)[7], f.y);
  let dy0 = mix((*value)[3] - (*value)[0], (*value)[2] - (*value)[1], f.x);
  let dy1 = mix((*value)[7] - (*value)[4], (*value)[6] - (*value)[5], f.x);
  let lower = mix(mix((*value)[0], (*value)[1], f.x), mix((*value)[3], (*value)[2], f.x), f.y);
  let upper = mix(mix((*value)[4], (*value)[5], f.x), mix((*value)[7], (*value)[6], f.x), f.y);
  let dx = mix(dx0, dx1, f.z);
  let dy = mix(dy0, dy1, f.z);
  let dz = upper - lower;
  let scaled = vec3f(dx * u.gridInfo.x / u.container.x, dy * u.gridInfo.y / u.container.y, dz * u.gridInfo.z / u.container.z);
  if (length(scaled) > 1e-5) { return -normalize(scaled); }
  return vec3f(0.0, 1.0, 0.0);
}

// The cube's corner values travel by pointer: WGSL passes arrays by value, and
// the former copies at up to 24 crossings per cube dominated this kernel's
// stack footprint.
fn crossing(a: vec3f, b: vec3f, va: f32, vb: f32, cubeBase: vec3f, cubeValue: ptr<function, array<f32, 8>>) -> SurfaceVertex {
  let denominator = vb - va;
  var t = 0.5;
  if (abs(denominator) > 1e-6) { t = clamp((0.5 - va) / denominator, 0.02, 0.98); }
  let lattice = mix(a, b, t);
  return SurfaceVertex(vec4f(latticeWorld(lattice), 1.0), vec4f(surfaceNormal(lattice, cubeBase, cubeValue), 0.0));
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

fn polygoniseTetra(p: array<vec3f, 4>, v: array<f32, 4>, cubeBase: vec3f, cubeValue: ptr<function, array<f32, 8>>) {
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
    emitTriangle(crossing(p[a], p[outside[0]], v[a], v[outside[0]], cubeBase, cubeValue), crossing(p[a], p[outside[1]], v[a], v[outside[1]], cubeBase, cubeValue), crossing(p[a], p[outside[2]], v[a], v[outside[2]], cubeBase, cubeValue));
  } else if (ni == 3) {
    let a = outside[0];
    emitTriangle(crossing(p[a], p[inside[0]], v[a], v[inside[0]], cubeBase, cubeValue), crossing(p[a], p[inside[2]], v[a], v[inside[2]], cubeBase, cubeValue), crossing(p[a], p[inside[1]], v[a], v[inside[1]], cubeBase, cubeValue));
  } else {
    let i0 = inside[0]; let i1 = inside[1]; let o0 = outside[0]; let o1 = outside[1];
    let p00 = crossing(p[i0], p[o0], v[i0], v[o0], cubeBase, cubeValue);
    let p01 = crossing(p[i0], p[o1], v[i0], v[o1], cubeBase, cubeValue);
    let p10 = crossing(p[i1], p[o0], v[i1], v[o0], cubeBase, cubeValue);
    let p11 = crossing(p[i1], p[o1], v[i1], v[o1], cubeBase, cubeValue);
    emitTriangle(p00, p10, p11); emitTriangle(p00, p11, p01);
  }
}

fn loadCubeCorners(base: vec3i) -> array<f32, 8> {
  let offsets = array<vec3i, 8>(
    vec3i(0,0,0), vec3i(1,0,0), vec3i(1,1,0), vec3i(0,1,0),
    vec3i(0,0,1), vec3i(1,0,1), vec3i(1,1,1), vec3i(0,1,1)
  );
  var value = array<f32, 8>();
  for (var i = 0; i < 8; i += 1) { value[i] = latticeValue(base + offsets[i]); }
  return value;
}

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
fn classifyCube(base: vec3i) {
  let cubeDims = vec3u(u.gridInfo.xyz) + vec3u(1);
  if (any(base < vec3i(0)) || any(vec3u(base) >= cubeDims)) { return; }
  var value = loadCubeCorners(base);
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
  let slot = atomicAdd(&extractionMeta.activeCubeCount, 1u);
  if (slot < arrayLength(&activeCubes)) {
    activeCubes[slot] = vec2u(u32(base.x) | (u32(base.z) << 16u), u32(base.y));
  }
}

var<workgroup> workgroupVertexTotal: atomic<u32>;
var<workgroup> workgroupBaseSlot: u32;

// One thread per surface-crossing cube from the classify worklist. Threads
// combine their exact vertex counts in workgroup memory, thread 0 performs
// the workgroup's only two global atomics (block allocation and the indirect
// draw count), and each thread then emits into its private slice.
@compute @workgroup_size(${EXTRACTION_POLYGONISE_WORKGROUP})
fn polygoniseMain(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) localIndex: u32) {
  let activeTotal = min(atomicLoad(&extractionMeta.activeCubeCount), arrayLength(&activeCubes));
  var base = vec3i(0);
  var value = array<f32, 8>();
  var vertexCount = 0u;
  if (gid.x < activeTotal) {
    let packedCube = activeCubes[gid.x];
    base = vec3i(i32(packedCube.x & 0xffffu), i32(packedCube.y), i32(packedCube.x >> 16u));
    value = loadCubeCorners(base);
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
    let blockStart = atomicAdd(&extractionMeta.vertexAllocator, total);
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
  for (var i = 0; i < 8; i += 1) { p[i] = vec3f(base + offsets[i]); }
  // Six tetrahedra sharing cube diagonal 0-6.  Unlike a lookup-table
  // marching-cubes implementation this has no ambiguous saddle cases.
  let tetra = array<vec4i, 6>(vec4i(0,1,2,6), vec4i(0,2,3,6), vec4i(0,3,7,6), vec4i(0,7,4,6), vec4i(0,4,5,6), vec4i(0,5,1,6));
  for (var t = 0; t < 6; t += 1) {
    let ids = tetra[t];
    polygoniseTetra(array<vec3f,4>(p[ids.x],p[ids.y],p[ids.z],p[ids.w]), array<f32,4>(value[ids.x],value[ids.y],value[ids.z],value[ids.w]), vec3f(base), &value);
  }
}

@compute @workgroup_size(4, 4, 4)
fn extractMain(@builtin(global_invocation_id) gid: vec3u) {
  classifyCube(vec3i(gid));
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
  let minimumBase = min(min(b00, b10), min(b01, b11));
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
struct ExtractionMeta { activeCubeCount: u32, vertexAllocator: u32 }
struct DispatchArgs { x: u32, y: u32, z: u32 }
@group(0) @binding(0) var<storage, read> extractionMeta: ExtractionMeta;
@group(0) @binding(1) var<storage, read> activeCubes: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> dispatchArgs: DispatchArgs;
@compute @workgroup_size(1)
fn prepareMain() {
  let activeTotal = min(extractionMeta.activeCubeCount, arrayLength(&activeCubes));
  dispatchArgs = DispatchArgs((activeTotal + ${EXTRACTION_POLYGONISE_WORKGROUP - 1}u) / ${EXTRACTION_POLYGONISE_WORKGROUP}u, 1u, 1u);
}
`;

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
  let ndc=vec2f(dot(relative,right)/(depth*aspect*0.72),dot(relative,up)/(depth*0.72));
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
  let rigid=nearestBody(ro,rd);if(rigid.t<nearest){let diffuse=.16+.84*max(dot(rigid.n,light),0.0);let rim=pow(1.0-max(dot(-rd,rigid.n),0.0),3.0);color=rigid.color*diffuse+vec3f(.18,.34,.31)*rim+rigid.selected*vec3f(.12,.42,.32);nearest=rigid.t;}
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
struct VOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index)i:u32)->VOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VOut;o.position=vec4f(p[i],0,1);o.uv=p[i]*.5+.5;return o;}
fn project(world:vec3f)->vec2f{let f=normalize(u.cameraTarget.xyz-u.cameraPosition.xyz);let r=normalize(cross(f,vec3f(0,1,0)));let up=normalize(cross(r,f));let q=world-u.cameraPosition.xyz;let d=max(dot(q,f),1e-4);let ndc=vec2f(dot(q,r)/(d*u.viewport.x/max(u.viewport.y,1.0)*.72),dot(q,up)/(d*.72));return vec2f(ndc.x*.5+.5,.5-ndc.y*.5);}
fn safeSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{return textureSampleLevel(texture,linearSampler,clamp(uv,vec2f(.001),vec2f(.999)),0);}
fn boxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let near3=min(a,b);let far3=max(a,b);return vec2f(max(max(near3.x,near3.y),near3.z),min(min(far3.x,far3.y),far3.z));}
${environmentShaderLibrary}
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
fn compositeFrontGlass(color:vec3f,ro:vec3f,rd:vec3f,sceneDepth:f32)->vec3f{
  // The garden pond has no vessel: nothing to composite in front of the water.
  if(environmentIndex()==7){return color;}
  let size=u.container.xyz;let mn=vec3f(-size.x*.5,0,-size.z*.5);let mx=vec3f(size.x*.5,size.y,size.z*.5);let hit=boxHit(ro,rd,mn,mx);
  if(hit.x>hit.y||hit.y<=0.0){return color;}
  let glassT=select(hit.x,hit.y,hit.x<=1e-4);
  if(glassT<=1e-4||glassT>sceneDepth+.001){return color;}
  let center=(mn+mx)*.5;let halfSize=size*.5;let point=ro+rd*glassT;let normal=boxNormal(point,center,halfSize);
  let q=abs((point-center)/max(halfSize,vec3f(1e-5)));
  let edgeCoordinate=max(max(min(q.x,q.y),min(q.x,q.z)),min(q.y,q.z));
  let outerEdge=smoothstep(.955,.998,edgeCoordinate);
  let innerEdge=smoothstep(.91,.975,edgeCoordinate)*(1.0-outerEdge);
  let cosine=clamp(abs(dot(-rd,normal)),0.0,1.0);let fresnel=.04+.96*pow(1.0-cosine,5.0);
  let paneAlpha=.008+.065*fresnel;let edgeAlpha=.52*outerEdge+.10*innerEdge;
  let glassTint=vec3f(.30,.58,.54);var result=mix(color,color*vec3f(.985,1.0,.998)+glassTint*.035,paneAlpha+edgeAlpha);
  let light=environmentLightDirection();let glint=pow(max(dot(reflect(rd,normal),light),0.0),240.0);
  result+=environmentLightColor()*(glint*(.18+.82*outerEdge)+fresnel*outerEdge*.16);
  return result;
}
fn finish(color:vec3f,ndc:vec2f)->vec4f{var c=environmentForeground(color,ndc)*(1.0-.08*dot(ndc*.55,ndc*.55));c=c/(c+vec3f(1.0));c=pow(max(c,vec3f(0.0)),vec3f(1.0/2.2));return vec4f(c,1);}
@fragment fn fragmentMain(input:VOut)->@location(0) vec4f{
  // Full-screen interpolated UV has Y=1 at the top of the render target,
  // while sampled WebGPU textures have Y=0 there. The shared legacy upscaler
  // performs the same conversion for the final target; all raster-path
  // intermediate reads and world projections must do it here as well.
  let ndc=input.uv*2.0-1.0;let textureUV=vec2f(input.uv.x,1.0-input.uv.y);let ro=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-ro);let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));let rd=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*.72+up*ndc.y*.72);
  let scene=safeSample(sceneTexture,textureUV);var front=safeSample(frontPosition,textureUV);if(front.a<.5){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}var frontDepth=dot(front.xyz-ro,rd);
  let cellSize=min(min(u.container.x/max(u.gridInfo.x,1.0),u.container.y/max(u.gridInfo.y,1.0)),u.container.z/max(u.gridInfo.z,1.0));let depthEpsilon=max(.0015,.18*cellSize);
  var n=normalize(safeSample(frontNormal,textureUV).xyz);let rigidFront=nearestRigid(ro,rd);let contactBand=${CONTACT_RESOLVE_BAND_CELLS.toFixed(1)}*cellSize;
  if(u.gridInfo.w>.5&&rigidFront.t<1e19&&abs(rigidFront.t-frontDepth)<=contactBand){let contact=refineContactSurface(ro,rd,frontDepth,cellSize);if(contact.valid){front=vec4f(contact.point,1);frontDepth=dot(contact.point-ro,rd);n=contact.normal;}if(rigidFront.t<=frontDepth+max(3e-4,.03*cellSize)){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}}
  if(scene.a+depthEpsilon<frontDepth){return finish(compositeFrontGlass(scene.rgb,ro,rd,scene.a),ndc);}
  if(dot(n,rd)>0.0){n=-n;}let etaIn=1.0/1.333;var inside=refract(rd,n,etaIn);if(length(inside)<1e-5){inside=reflect(rd,n);}
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
  if(!opaqueSolidExit){if(dot(exitN,inside)<0.0){exitN=-exitN;}outgoing=refract(inside,-exitN,1.333);tir=length(outgoing)<1e-5;if(tir){outgoing=reflect(inside,-exitN);}backgroundUV=project(exitPoint+outgoing*(.55+.45*thickness));}
  let transmittedScene=safeSample(sceneTexture,backgroundUV).rgb;
  // Clean water: red is attenuated first.  A small in-scattering term keeps
  // thick regions luminous instead of turning into opaque ink.
  let absorption=vec3f(.45,.09,.06);let transmission=exp(-absorption*thickness);let scatter=vec3f(.012,.055,.049)*(vec3f(1.0)-transmission);
  let refracted=transmittedScene*transmission+scatter;let reflectedDir=reflect(rd,n);var reflected=environmentLight(reflectedDir);
  let ssrUV=project(front.xyz+reflectedDir*.8);let ssr=safeSample(sceneTexture,ssrUV);reflected=mix(reflected,ssr.rgb,select(0.0,.32,ssr.a<60000.0));
  let cosine=clamp(dot(-rd,n),0.0,1.0);let fresnel=.02037+(1.0-.02037)*pow(1.0-cosine,5.0);var water=mix(refracted,reflected,fresnel);
  if(tir){water=mix(water,environmentLight(outgoing),.88);}
  let light=environmentLightDirection();water+=environmentLightColor()*pow(max(dot(reflect(rd,n),light),0.0),180.0)*1.4;
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
  private preparePipeline?: GPUComputePipeline;
  private polygonisePipeline?: GPUComputePipeline;
  private surfaceFrontPipeline?: GPURenderPipeline;
  private surfaceBackPipeline?: GPURenderPipeline;
  private causticPipeline?: GPURenderPipeline;
  private scenePipeline?: GPURenderPipeline;
  private compositePipeline?: GPURenderPipeline;
  private extractLayout?: GPUBindGroupLayout;
  private prepareLayout?: GPUBindGroupLayout;
  private surfaceLayout?: GPUBindGroupLayout;
  private sceneLayout?: GPUBindGroupLayout;
  private compositeLayout?: GPUBindGroupLayout;
  private sampler?: GPUSampler;
  private vertexBuffer?: GPUBuffer;
  private indirectBuffer?: GPUBuffer;
  private activeCubeBuffer?: GPUBuffer;
  private extractionMetaBuffer?: GPUBuffer;
  private polygoniseDispatchBuffer?: GPUBuffer;
  private extractBindGroup?: GPUBindGroup;
  private prepareBindGroup?: GPUBindGroup;
  private surfaceBindGroup?: GPUBindGroup;
  private sceneBindGroup?: GPUBindGroup;
  private compositeBindGroup?: GPUBindGroup;
  private sceneTexture?: GPUTexture;
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

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bodyBuffer: GPUBuffer
  ) {}

  async initialize(onProgress:(label:string,completed:number,total:number)=>void=()=>{}) {
    const [extract, prepare, surface, caustic, scene, composite] = await Promise.all([
      checkedModule(this.device, "Water isosurface extraction", surfaceExtractionShader),
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
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
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
    const total=11;let completed=0;
    const compute=async(label:string,descriptor:GPUComputePipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createComputePipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    const render=async(label:string,descriptor:GPURenderPipelineDescriptor)=>{onProgress(label,completed,total);const result=await this.device.createRenderPipelineAsync(descriptor);completed+=1;onProgress(label,completed,total);return result;};
    this.extractPipeline = await compute("Classifying liquid surface cubes",{ label: "Classify liquid surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractMain" } });
    this.extractBandPipeline = await compute("Classifying restricted water band",{ label: "Classify restricted water band", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractBandMain" } });
    this.extractTallSidesPipeline = await compute("Classifying tall-cell interfaces",{ label: "Classify tall-cell side interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractTallSidesMain" } });
    this.extractWallPipeline = await compute("Classifying water wall interfaces",{ label: "Classify water wall interfaces", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "extractWallMain" } });
    this.polygonisePipeline = await compute("Building water surface mesh",{ label: "Polygonise surface cubes", layout: extractionPipelineLayout, compute: { module: extract, entryPoint: "polygoniseMain" } });
    this.preparePipeline = await compute("Preparing surface dispatch",{ label: "Prepare polygonise dispatch", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareLayout] }), compute: { module: prepare, entryPoint: "prepareMain" } });
    this.extractionMetaBuffer = this.device.createBuffer({ label: "Water extraction worklist counters", size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.polygoniseDispatchBuffer = this.device.createBuffer({ label: "Water polygonise dispatch arguments", size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const surfacePipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.surfaceLayout] });
    const surfaceDescriptor = (label: string, cullMode: GPUCullMode): GPURenderPipelineDescriptor => ({
      label, layout: surfacePipelineLayout, vertex: { module: surface, entryPoint: "surfaceVertex" },
      fragment: { module: surface, entryPoint: "surfaceFragment", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
      primitive: { topology: "triangle-list", frontFace: "ccw", cullMode },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    this.surfaceFrontPipeline = await render("Rendering front water interfaces",surfaceDescriptor("Raster water front interfaces", "back"));
    this.surfaceBackPipeline = await render("Rendering back water interfaces",surfaceDescriptor("Raster water back interfaces", "front"));
    this.causticPipeline = await render("Projecting water caustics",{
      label: "Project refracted caustics", layout: surfacePipelineLayout, vertex: { module: caustic, entryPoint: "causticVertex" },
      fragment: { module: caustic, entryPoint: "causticFragment", targets: [{ format: "rgba16float", blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } } }] },
      primitive: { topology: "triangle-list", cullMode: "none" }
    });
    this.scenePipeline = await render("Rendering the dry scene",{ label: "Render dry scene for water refraction", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout] }), vertex: { module: scene, entryPoint: "vertexMain" }, fragment: { module: scene, entryPoint: "fragmentMain", targets: [{ format: "rgba16float" }] }, primitive: { topology: "triangle-list" } });
    this.compositePipeline = await render("Compositing water optics",{ label: "Composite two-interface water optics", layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }), vertex: { module: composite, entryPoint: "vertexMain" }, fragment: { module: composite, entryPoint: "fragmentMain", targets: [{ format: this.targetFormat }] }, primitive: { topology: "triangle-list" } });
    this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  }

  setVolume(texture: GPUTexture, columnBases: GPUTexture) {
    if (this.volume === texture && this.columnBases === columnBases) return;
    this.volume = texture; this.columnBases = columnBases; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  setSecondaryParticles(pipeline: SecondaryParticleRenderPipeline | undefined) {
    this.secondaryParticles = pipeline;
  }

  private ensureGeometry(nx: number, ny: number, nz: number) {
    const key = `${nx}x${ny}x${nz}`;
    if (key === this.geometryKey) return;
    this.vertexBuffer?.destroy(); this.indirectBuffer?.destroy(); this.activeCubeBuffer?.destroy();
    // Surface area, not volume, controls the normal case.  The generous factor
    // also covers breaking sheets and entrained blobs while imposing a hard
    // 64 MiB ceiling on adversarial checkerboard fields.
    const maxVertices = surfaceVertexCapacity(nx, ny, nz);
    this.vertexBuffer = this.device.createBuffer({ label: `Extracted water surface (${maxVertices} vertices)`, size: maxVertices * 32, usage: GPUBufferUsage.STORAGE });
    this.indirectBuffer = this.device.createBuffer({ label: "Water indirect draw arguments", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.activeCubeBuffer = this.device.createBuffer({ label: "Water surface cube worklist", size: activeCubeCapacity(maxVertices) * 8, usage: GPUBufferUsage.STORAGE });
    this.geometryKey = key; this.extractedRevision = -1; this.lastExtractionAt_ms = -Infinity; this.causticsValid = false; this.rebuildBindGroups();
  }

  ensureSize(width: number, height: number) {
    const key = `${width}x${height}`;
    if (key === this.targetKey) return;
    for (const texture of [this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth]) texture?.destroy();
    const sampledTarget = (label: string) => this.device.createTexture({ label, size: [width,height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.sceneTexture = sampledTarget("Dry scene HDR"); this.frontPosition = sampledTarget("Water front positions"); this.frontNormal = sampledTarget("Water front normals"); this.backPosition = sampledTarget("Water back positions"); this.backNormal = sampledTarget("Water back normals");
    const depth = (label: string) => this.device.createTexture({ label, size: [width,height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.frontDepth = depth("Water front depth"); this.backDepth = depth("Water back depth");
    this.causticTexture?.destroy(); this.causticTexture = this.device.createTexture({ label: "Refracted floor caustics", size: [384,384], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.causticsValid = false;
    this.targetKey = key; this.rebuildBindGroups();
  }

  private rebuildBindGroups() {
    if (this.extractLayout && this.volume && this.columnBases && this.vertexBuffer && this.indirectBuffer && this.activeCubeBuffer && this.extractionMetaBuffer) this.extractBindGroup = this.device.createBindGroup({ layout: this.extractLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 2, resource: this.columnBases.createView() }, { binding: 3, resource: { buffer: this.vertexBuffer } }, { binding: 4, resource: { buffer: this.indirectBuffer } }, { binding: 5, resource: { buffer: this.activeCubeBuffer } }, { binding: 6, resource: { buffer: this.extractionMetaBuffer } }
    ] });
    if (this.prepareLayout && this.extractionMetaBuffer && this.activeCubeBuffer && this.polygoniseDispatchBuffer) this.prepareBindGroup = this.device.createBindGroup({ layout: this.prepareLayout, entries: [
      { binding: 0, resource: { buffer: this.extractionMetaBuffer } }, { binding: 1, resource: { buffer: this.activeCubeBuffer } }, { binding: 2, resource: { buffer: this.polygoniseDispatchBuffer } }
    ] });
    if (this.surfaceLayout && this.vertexBuffer) this.surfaceBindGroup = this.device.createBindGroup({ layout: this.surfaceLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.vertexBuffer } }] });
    if (this.sceneLayout && this.causticTexture && this.sampler) this.sceneBindGroup = this.device.createBindGroup({ layout: this.sceneLayout, entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.bodyBuffer } }, { binding: 2, resource: this.causticTexture.createView() }, { binding: 3, resource: this.sampler }] });
    if (this.compositeLayout && this.sceneTexture && this.frontPosition && this.frontNormal && this.backPosition && this.backNormal && this.sampler && this.volume && this.columnBases) this.compositeBindGroup = this.device.createBindGroup({ layout: this.compositeLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: this.sceneTexture.createView() }, { binding: 2, resource: this.frontPosition.createView() }, { binding: 3, resource: this.frontNormal.createView() }, { binding: 4, resource: this.backPosition.createView() }, { binding: 5, resource: this.backNormal.createView() }, { binding: 6, resource: this.sampler }, { binding: 7, resource: { buffer: this.bodyBuffer } }, { binding: 8, resource: this.volume.createView({ dimension: "3d" }) }, { binding: 9, resource: this.columnBases.createView() }
    ] });
  }

  encode(encoder: GPUCommandEncoder, output: GPUTextureView, nx: number, ny: number, nz: number, restrictedTallCell: boolean, maximumNeighborDelta: number, revision: number, targetFps = 60, timestamps?: RasterWaterTimestampRanges): RasterWaterEncodeResult | false {
    this.ensureGeometry(nx,ny,nz);
    if (!this.extractPipeline||!this.extractBandPipeline||!this.extractTallSidesPipeline||!this.extractWallPipeline||!this.preparePipeline||!this.polygonisePipeline||!this.surfaceFrontPipeline||!this.surfaceBackPipeline||!this.causticPipeline||!this.scenePipeline||!this.compositePipeline||!this.extractBindGroup||!this.prepareBindGroup||!this.surfaceBindGroup||!this.sceneBindGroup||!this.compositeBindGroup||!this.indirectBuffer||!this.extractionMetaBuffer||!this.polygoniseDispatchBuffer||!this.volume||!this.sceneTexture||!this.frontPosition||!this.frontNormal||!this.frontDepth||!this.backPosition||!this.backNormal||!this.backDepth||!this.causticTexture) return false;
    const now_ms = performance.now();
    // Rendering follows the newest available solver revision, but extraction
    // follows the selected presentation cadence. Unchanged solver revisions
    // retain the existing mesh, so pausing does not create redundant work.
    const updateSurface = shouldUpdateWaterSurface(this.extractedRevision, revision, this.lastExtractionAt_ms, now_ms, targetFps);
    const updateCaustics = this.causticsEnabled && (updateSurface || !this.causticsValid);
    if (updateSurface) {
      this.device.queue.writeBuffer(this.indirectBuffer,0,new Uint32Array([0,1,0,0]));
      this.device.queue.writeBuffer(this.extractionMetaBuffer,0,new Uint32Array([0,0]));
      const plan = surfaceExtractionDispatchPlan(nx, ny, nz, this.volume.depthOrArrayLayers, restrictedTallCell, maximumNeighborDelta);
      // Classify appends surface-crossing cubes to the worklist, the prepare
      // kernel sizes the indirect dispatch, and polygonise emits triangles for
      // just those cubes. Dispatches in one pass order their storage writes.
      const compute=encoder.beginComputePass({label:"Extract water isosurface",...(timestamps?{timestampWrites:timestamps.extraction}:{})});compute.setBindGroup(0,this.extractBindGroup);
      if (plan.mode === "restricted-band") {
        compute.setPipeline(this.extractBandPipeline); compute.dispatchWorkgroups(...plan.band!);
        compute.setPipeline(this.extractTallSidesPipeline); compute.dispatchWorkgroups(...plan.tallSides!);
        compute.setPipeline(this.extractWallPipeline); compute.dispatchWorkgroups(...plan.walls!);
      } else {
        compute.setPipeline(this.extractPipeline); compute.dispatchWorkgroups(...plan.full!);
      }
      compute.setPipeline(this.preparePipeline); compute.setBindGroup(0, this.prepareBindGroup); compute.dispatchWorkgroups(1);
      compute.setPipeline(this.polygonisePipeline); compute.setBindGroup(0, this.extractBindGroup); compute.dispatchWorkgroupsIndirect(this.polygoniseDispatchBuffer, 0);
      compute.end();
      this.extractedRevision = revision; this.lastExtractionAt_ms = advancePresentationClock(this.lastExtractionAt_ms, now_ms, targetFps);
    }
    if (updateCaustics) {
      const caustic=encoder.beginRenderPass({label:"Water caustics",colorAttachments:[{view:this.causticTexture.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});caustic.setPipeline(this.causticPipeline);caustic.setBindGroup(0,this.surfaceBindGroup);caustic.drawIndirect(this.indirectBuffer,0);caustic.end();
      this.causticsValid = true;
    }
    const scene=encoder.beginRenderPass({label:"Dry scene",colorAttachments:[{view:this.sceneTexture.createView(),clearValue:{r:0,g:0,b:0,a:65504},loadOp:"clear",storeOp:"store"}],...(timestamps?{timestampWrites:timestamps.scene}:{})});scene.setPipeline(this.scenePipeline);scene.setBindGroup(0,this.sceneBindGroup);scene.draw(3);scene.end();
    const interfacePass=(label:string,pipeline:GPURenderPipeline,position:GPUTexture,normal:GPUTexture,depth:GPUTexture,side:"front"|"back",timestampWrites?:TimestampRange)=>{const pass=encoder.beginRenderPass({label,colorAttachments:[{view:position.createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"},{view:normal.createView(),clearValue:{r:0,g:1,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:depth.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"},...(timestampWrites?{timestampWrites}:{})});pass.setPipeline(pipeline);pass.setBindGroup(0,this.surfaceBindGroup!);pass.drawIndirect(this.indirectBuffer!,0);this.secondaryParticles?.encodeOpticalInterface(pass,side);pass.end();};
    interfacePass("Water front interfaces",this.surfaceFrontPipeline,this.frontPosition,this.frontNormal,this.frontDepth,"front",timestamps?.frontInterfaces);interfacePass("Water back interfaces",this.surfaceBackPipeline,this.backPosition,this.backNormal,this.backDepth,"back",timestamps?.backInterfaces);
    const composite=encoder.beginRenderPass({label:"Two-interface water composite",colorAttachments:[{view:output,clearValue:{r:.01,g:.025,b:.024,a:1},loadOp:"clear",storeOp:"store"}],...(timestamps?{timestampWrites:timestamps.composite}:{})});composite.setPipeline(this.compositePipeline);composite.setBindGroup(0,this.compositeBindGroup);composite.draw(3);composite.end();return { surfaceUpdated: updateSurface };
  }

  destroy() {
    for (const resource of [this.vertexBuffer,this.indirectBuffer,this.activeCubeBuffer,this.extractionMetaBuffer,this.polygoniseDispatchBuffer,this.sceneTexture,this.frontPosition,this.frontNormal,this.frontDepth,this.backPosition,this.backNormal,this.backDepth,this.causticTexture]) { try { resource?.destroy(); } catch { /* device loss */ } }
  }
}
