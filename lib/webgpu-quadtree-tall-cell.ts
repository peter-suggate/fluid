import { adaptivePressureCellTopology, buildMlsProjectionRows, buildQuadtree, buildVariationalSystem, populateTallPressureGrid, quadtreeFromPackedCells, quadtreeSizingFromVelocityAndSurface, signedDistanceFromVolume, type QuadtreeGrid, type TallPressureGrid, type TallPressureSample, type VariationalBody, type VariationalSystem } from "./quadtree-tall-cell-grid";
import { damBreakFractions } from "./initial-fluid";
import { insidePrimitive } from "./fluid-rigid-coupling";
import { boundingRadius, quaternionRotate, type RigidBodyState } from "./rigid-body";
import type { SceneDescription, Vec3 } from "./model";
import { WebGPUQuadtreeBuilder, type WebGPUQuadtreeConstructionCache } from "./webgpu-quadtree-builder";

export interface QuadtreeRigidCoupling {
  bodies: RigidBodyState[];
  /** True when a load consumer integrates the bodies; kinematic bodies keep M^-1 = 0. */
  dynamic: boolean;
}

/**
 * The WGSL, bind-group layout, and compiled pipelines are identical for every
 * rebuilt projection; recompiling them per topology rebuild costs tens of
 * milliseconds per step. The first projection fills this cache and every
 * rebuild reuses it.
 */
export interface QuadtreeGPUCache {
  layout: GPUBindGroupLayout;
  module: GPUShaderModule;
  pipelineLayout: GPUPipelineLayout;
  pipelines?: Record<string, GPUComputePipeline>;
  dispatchLayout?: GPUBindGroupLayout;
  dispatchPipeline?: GPUComputePipeline;
  construction?: WebGPUQuadtreeConstructionCache;
}

export interface QuadtreeBodyImpulse {
  bodyId: string;
  impulse_N_s: Vec3;
  angularImpulse_N_m_s: Vec3;
  displacedVolume_m3: number;
}

function solidFieldsFromBodies(scene: SceneDescription, bodies: RigidBodyState[], nx: number, ny: number, nz: number, h: Vec3) {
  if (bodies.length === 0) return undefined;
  const solidFraction = new Float32Array(nx * ny * nz);
  const solidOwner = new Int32Array(nx * ny * nz).fill(-1);
  const halfWidth = scene.container.width_m / 2, halfDepth = scene.container.depth_m / 2;
  // A display nozzle is a filled rigid primitive; its open channel is the
  // prescribed inflow cylinder, which must stay carved out of [A] exactly as
  // the legacy coupling kernel carved its inflow velocity cells.
  const inflow = scene.fluid.inflow;
  const inflowSpeed = inflow ? Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z) : 0;
  const inflowDirection = inflow && inflowSpeed > 0 ? { x: inflow.velocity_m_s.x / inflowSpeed, y: inflow.velocity_m_s.y / inflowSpeed, z: inflow.velocity_m_s.z / inflowSpeed } : undefined;
  const margin = Math.max(h.x, h.y, h.z);
  const insideInflowChannel = (point: Vec3) => {
    if (!inflow || !inflowDirection) return false;
    const dx = point.x - inflow.center_m.x, dy = point.y - inflow.center_m.y, dz = point.z - inflow.center_m.z;
    const along = dx * inflowDirection.x + dy * inflowDirection.y + dz * inflowDirection.z;
    if (Math.abs(along) > inflow.length_m / 2 + margin) return false;
    const rx = dx - along * inflowDirection.x, ry = dy - along * inflowDirection.y, rz = dz - along * inflowDirection.z;
    return Math.hypot(rx, ry, rz) <= inflow.radius_m + margin;
  };
  bodies.forEach((body, owner) => {
    const radius = boundingRadius(body) + Math.max(h.x, h.y, h.z);
    const localX = body.position_m.x + halfWidth, localY = body.position_m.y, localZ = body.position_m.z + halfDepth;
    const x0 = Math.max(0, Math.floor((localX - radius) / h.x)), x1 = Math.min(nx - 1, Math.ceil((localX + radius) / h.x));
    const y0 = Math.max(0, Math.floor((localY - radius) / h.y)), y1 = Math.min(ny - 1, Math.ceil((localY + radius) / h.y));
    const z0 = Math.max(0, Math.floor((localZ - radius) / h.z)), z1 = Math.min(nz - 1, Math.ceil((localZ + radius) / h.z));
    for (let z = z0; z <= z1; z += 1) for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) {
      let inside = 0;
      for (let corner = 0; corner < 8; corner += 1) {
        const point = {
          x: -halfWidth + (x + 0.5 + ((corner & 1) > 0 ? 0.4 : -0.4)) * h.x,
          y: (y + 0.5 + ((corner & 2) > 0 ? 0.4 : -0.4)) * h.y,
          z: -halfDepth + (z + 0.5 + ((corner & 4) > 0 ? 0.4 : -0.4)) * h.z
        };
        if (insidePrimitive(body, point)) inside += 1;
      }
      if (inside === 0) continue;
      if (insideInflowChannel({ x: -halfWidth + (x + 0.5) * h.x, y: (y + 0.5) * h.y, z: -halfDepth + (z + 0.5) * h.z })) continue;
      const index = x + nx * (y + ny * z), fraction = inside / 8;
      if (fraction > solidFraction[index]) { solidFraction[index] = fraction; solidOwner[index] = owner; }
    }
  });
  return { solidFraction, solidOwner };
}

function variationalBodiesFrom(scene: SceneDescription, coupling: QuadtreeRigidCoupling): VariationalBody[] {
  const rho = scene.fluid.density_kg_m3, halfWidth = scene.container.width_m / 2, halfDepth = scene.container.depth_m / 2;
  return coupling.bodies.map((body) => {
    let inverseMass = 0;
    const inverseInertia = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    if (coupling.dynamic) {
      inverseMass = rho * body.inverseMass_kg;
      const q = body.orientation;
      const columns = [quaternionRotate(q, { x: 1, y: 0, z: 0 }), quaternionRotate(q, { x: 0, y: 1, z: 0 }), quaternionRotate(q, { x: 0, y: 0, z: 1 })];
      const invBody = [body.inverseInertiaBody_kg_m2.x, body.inverseInertiaBody_kg_m2.y, body.inverseInertiaBody_kg_m2.z];
      const axes = ["x", "y", "z"] as const;
      for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
        let sum = 0;
        for (let k = 0; k < 3; k += 1) sum += columns[k][axes[row]] * invBody[k] * columns[k][axes[column]];
        inverseInertia[3 * row + column] = rho * sum;
      }
    }
    return {
      position: { x: body.position_m.x + halfWidth, y: body.position_m.y, z: body.position_m.z + halfDepth },
      linearVelocity: body.linearVelocity_m_s,
      angularVelocity: body.angularVelocity_rad_s,
      inverseMass, inverseInertia
    };
  });
}

const INVALID = 0xffffffff;

export interface QuadtreeTallCellProjectionOptions {
  pressureIterations: number;
  relativeTolerance: number;
  adaptivityStrength: number;
  maximumLeafSize: number;
  opticalDepthFraction: number;
}

export interface QuadtreeTallCellProjectionInfo {
  leafCount: number;
  pressureSampleCount: number;
  liquidDofCount: number;
  faceCount: number;
  tallSegmentCount: number;
  ghostFaceCount: number;
  maximumNeighborRatio: number;
  compressionRatio: number;
  maximumFluidScale: number;
  allocatedBytes: number;
  /** GPU queue + compact readback time for Sec. 4.1 construction. */
  gpuConstruction_ms?: number;
  /** Timestamp-query duration of only the GPU sizing/subdivision kernels. */
  gpuConstructionKernel_ms?: number;
  /** CPU time left for tall-cell/variational sparse packing after the tree exists. */
  cpuTopologyPack_ms?: number;
  cpuRedistance_ms?: number;
  cpuQuadtreeDecode_ms?: number;
  cpuTallGrid_ms?: number;
  cpuVariationalAssembly_ms?: number;
  cpuSystemPack_ms?: number;
  cpuResourceUpload_ms?: number;
  topologyReused?: boolean;
  topologyReuseCount?: number;
  pressureIterationsUsed?: number;
  /** Bytes read back for an update (phi + VOF + compact 2D leaves + diagnostics). */
  topologyReadbackBytes?: number;
}

interface ProjectionResources {
  velocityIn: GPUTexture;
  velocityOut: GPUTexture;
  volume: GPUTexture;
}

interface ProjectionFields {
  phi: Float32Array;
  velocity?: Vec3[];
  quadtree?: QuadtreeGrid;
  pressureGrid?: TallPressureGrid;
  topologyWords?: Uint32Array;
}

function pressureTopologyWords(grid: TallPressureGrid) {
  const words = new Uint32Array(3 + grid.quadtree.leaves.length + 2 * grid.segments.length);
  words.set([grid.quadtree.leaves.length, grid.samples.length, grid.segments.length]);
  let cursor = 3;
  for (const leaf of grid.quadtree.leaves) words[cursor++] = leaf.x | (leaf.z << 10) | (leaf.size << 20);
  for (const segment of grid.segments) {
    const bottom = grid.samples[segment.bottomSample], top = grid.samples[segment.topSample];
    words[cursor++] = segment.leaf;
    words[cursor++] = segment.firstY | (segment.lastY << 10) | ((segment.tall ? 1 : 0) << 20) | ((bottom.liquid ? 1 : 0) << 21) | ((top.liquid ? 1 : 0) << 22);
  }
  return words;
}

function sameWords(a: Uint32Array, b: Uint32Array) {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

function uploadLevelSet(device: GPUDevice, texture: GPUTexture, phi: Float32Array, nx: number, ny: number, nz: number) {
  const rowBytes = nx * 4, pitch = Math.ceil(rowBytes / 256) * 256, upload = new Uint8Array(pitch * ny * nz), source = new Uint8Array(phi.buffer, phi.byteOffset, phi.byteLength);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) upload.set(source.subarray(rowBytes * (y + ny * z), rowBytes * (y + ny * z + 1)), pitch * (y + ny * z));
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: pitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
}

function initialFields(scene: SceneDescription, nx: number, ny: number, nz: number) {
  const count = nx * ny * nz, phi = new Float32Array(count), velocity = Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0 }));
  const dam = damBreakFractions(scene.container.fillFraction);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const wet = scene.fluid.initialCondition === "tank-fill"
      ? (y + 0.5) / ny <= scene.container.fillFraction
      : (x + 0.5) / nx <= dam.width && (y + 0.5) / ny <= dam.height && (z + 0.5) / nz <= dam.depth;
    phi[x + nx * (y + ny * z)] = wet ? -1 : 1;
  }
  return { phi, velocity };
}

function initialSizing(scene: SceneDescription, nx: number, nz: number, h: Vec3, bodies?: RigidBodyState[]) {
  const sizing = new Float32Array(nx * nz);
  const sizingBodies = bodies?.map((body) => ({ position_m: body.position_m, dimensions_m: body.description.dimensions_m })) ?? scene.rigidBodies;
  // Rigid geometry is a persistent explicit sizing source, as in the paper's
  // examples. Surface features need no blanket refinement: the dynamic sizing
  // evaluates its curvature/velocity demand over each candidate leaf's whole
  // footprint, so a flat surface genuinely coarsens (the paper's headline
  // deep-water case) while edges, blobs, and droplets always register.
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const worldX = -scene.container.width_m / 2 + (x + 0.5) * h.x, worldZ = -scene.container.depth_m / 2 + (z + 0.5) * h.z;
    for (const body of sizingBodies) {
      const radius = Math.max(body.dimensions_m.x, body.dimensions_m.y, body.dimensions_m.z);
      if (Math.hypot(worldX - body.position_m.x, worldZ - body.position_m.z) <= radius + 2 * Math.max(h.x, h.z)) sizing[x + nx * z] = Math.max(sizing[x + nx * z], 2 / Math.min(h.x, h.z));
    }
  }
  return sizing;
}

function interpolation(samples: TallPressureSample[], y: number) {
  let lower = samples[0], upper = samples[samples.length - 1];
  for (const sample of samples) {
    if (sample.y <= y) lower = sample;
    if (sample.y >= y) { upper = sample; break; }
  }
  if (lower.id === upper.id) return [lower.id, lower.id, 1, 0] as const;
  const weight = (y - lower.y) / Math.max(1, upper.y - lower.y);
  return [lower.id, upper.id, 1 - weight, weight] as const;
}

function bufferWithData(device: GPUDevice, label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE, minimumSize = 4) {
  const size = Math.max(minimumSize, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
  if (data.byteLength) device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

export const quadtreeTallCellProjectionShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f, counts: vec4u, solve: vec4f, coupling: vec4u, couplingCounts: vec4u }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u }
struct Entry { face: u32, coefficient: f32 }
struct Interpolation { nodes: vec2u, weights: vec2f }
struct SolverState { pressure: f32, bestPressure: f32, residual: f32, direction: f32, preconditioned: f32, matrixDirection: f32, diagonal: f32 }
@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var velocityOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var volumeIn: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> faces: array<Face>;
@group(0) @binding(4) var<storage, read> rowOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> rowEntries: array<Entry>;
@group(0) @binding(6) var<storage, read> interpolationIn: array<Interpolation>;
@group(0) @binding(7) var cellProjection: texture_3d<f32>;
@group(0) @binding(8) var<storage, read_write> state: array<SolverState>;
@group(0) @binding(9) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(10) var<storage, read> factorColumns: array<vec2u>;
@group(0) @binding(11) var<storage, read> factorEntries: array<Entry>;
@group(0) @binding(12) var<uniform> params: Params;
@group(0) @binding(13) var factorAux: texture_2d<u32>;
@group(0) @binding(14) var mlsRowIndex: texture_3d<u32>;
@group(0) @binding(15) var levelSetIn: texture_3d<f32>;
var<workgroup> reductionA: array<f32, 256>;
var<workgroup> reductionB: array<f32, 256>;
fn dofs() -> u32 { return params.counts.x; }
fn auxWord(index: u32) -> u32 {
  let texel = index / 4u; let packed = textureLoad(factorAux, vec2i(i32(texel % params.dims.w), i32(texel / params.dims.w)), 0);
  return packed[index % 4u];
}
fn auxEntry(wordOffset: u32, index: u32) -> Entry { return Entry(auxWord(wordOffset + 2u * index), bitcast<f32>(auxWord(wordOffset + 2u * index + 1u))); }
fn faceSpan(face: Face) -> u32 { return face.packed & 0xffffu; }
fn faceAxis(face: Face) -> u32 { return (face.packed >> 16u) & 0x3u; }
fn faceNodeCount(face: Face) -> u32 { return (face.packed >> 18u) & 0x7u; }
fn faceGhost(face: Face) -> bool { return ((face.packed >> 21u) & 0x1u) != 0u; }
fn faceSamplePhi(face: Face, slot: u32) -> f32 {
  let packed = face.sampleCells[slot]; let span = face.sampleSpans[slot];
  let origin = vec2u(packed & 1023u, (packed >> 10u) & 1023u); let y = (packed >> 20u) & 1023u;
  let position = vec2f(origin) + vec2f(f32(span) * 0.5 - 0.5);
  let a = vec2u(floor(position)); let b = min(a + vec2u(1), params.dims.xz - vec2u(1)); let t = fract(position);
  let p00 = textureLoad(levelSetIn, vec3i(i32(a.x), i32(y), i32(a.y)), 0).x;
  let p10 = textureLoad(levelSetIn, vec3i(i32(b.x), i32(y), i32(a.y)), 0).x;
  let p01 = textureLoad(levelSetIn, vec3i(i32(a.x), i32(y), i32(b.y)), 0).x;
  let p11 = textureLoad(levelSetIn, vec3i(i32(b.x), i32(y), i32(b.y)), 0).x;
  return mix(mix(p00, p10, t.x), mix(p01, p11, t.x), t.y);
}
@compute @workgroup_size(128)
fn refreshFaces(@builtin(global_invocation_id) gid: vec3u) {
  let faceId = gid.x; if (faceId >= params.counts.y) { return; }
  var face = faces[faceId]; var all = 0.0; var liquid = 0.0; var allLiquid = true;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let phi = faceSamplePhi(face, slot); let term = face.coefficients[slot] * phi; all += term;
    if (phi < 0.0) { liquid += term; } else { allLiquid = false; }
  }
  var scale = 1.0;
  if (!allLiquid) { scale = select(min(100.0, max(0.0, all / liquid)), 0.0, abs(liquid) < 1e-12); }
  face.weights.y = face.weights.x * scale; faces[faceId] = face;
}
fn faceVelocity(face: Face) -> f32 {
  let axis = faceAxis(face);
  // Every vertical face spans its leaf's full x/z footprint, ghost or not;
  // the horizontal branch below would sample only the corner column.
  if (axis != 1u) {
    var sum = 0.0; var count = 0.0;
    for (var y = face.bounds.z; y < face.bounds.w; y += 1u) {
      for (var transverse = 0u; transverse < faceSpan(face); transverse += 1u) {
        var left = vec3u(face.bounds.x, y, face.bounds.y);
        if (axis == 0u) { left.z += transverse; }
        if (axis == 2u) { left.x += transverse; }
        sum += textureLoad(velocityIn, vec3i(left), 0)[axis]; count += 1.0;
      }
    }
    return sum / max(1.0, count);
  }
  var sum = 0.0; var count = 0.0;
  for (var z = face.bounds.y; z < face.bounds.y + faceSpan(face); z += 1u) {
    for (var x = face.bounds.x; x < face.bounds.x + faceSpan(face); x += 1u) {
      for (var y = face.bounds.z; y < face.bounds.w; y += 1u) {
        sum += textureLoad(velocityIn, vec3i(vec3u(x, y, z)), 0).y; count += 1.0;
      }
    }
  }
  return sum / max(1.0, count);
}
fn faceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu) { result += face.coefficients[slot] * state[node].direction; }
  }
  return result;
}
fn rowProduct(row: u32) -> f32 {
  var sum = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let item = rowEntries[entry]; let face = faces[item.face];
    sum += item.coefficient * face.weights.y * faceGradient(face);
  }
  return sum;
}
fn initializeRow(row: u32) {
  var rhs = 0.0; var diag = 0.0;
  for (var entry = rowOffsets[row]; entry < rowOffsets[row + 1u]; entry += 1u) {
    let item = rowEntries[entry]; let face = faces[item.face];
    // The face flux is A u_fluid (from the staged texture) plus the
    // CPU-integrated (1-A) u_solid constraint flux of moving rigid bodies.
    rhs += item.coefficient * (face.weights.x * faceVelocity(face) + face.solidFlux);
    diag += item.coefficient * item.coefficient * face.weights.y;
  }
  state[row].pressure = 0.0; state[row].bestPressure = 0.0; state[row].residual = rhs; state[row].diagonal = max(diag, 1e-12);
  state[row].preconditioned = 0.0; state[row].direction = 0.0; state[row].matrixDirection = 0.0;
}
@compute @workgroup_size(128)
fn initialize(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { initializeRow(gid.x); } }
fn applyPrecondition(lid: u32, solveActive: bool) {
  if (solveActive) { for (var row = lid; row < dofs(); row += 256u) { state[row].preconditioned = state[row].residual; } }
  storageBarrier(); workgroupBarrier();
  // The IC graph is level-scheduled so every row within a level is
  // independent. One workgroup supplies the required global ordering while
  // evaluating up to 256 triangular rows concurrently.
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y); let rowOffsetsOffset = bitcast<u32>(params.solve.z); let rowEntriesOffset = bitcast<u32>(params.solve.w);
    let range = vec2u(auxWord(levelsOffset + 4u * level), auxWord(levelsOffset + 4u * level + 1u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let row = auxWord(slot); var value = state[row].preconditioned;
      for (var entry = auxWord(rowOffsetsOffset + row); entry < auxWord(rowOffsetsOffset + row + 1u); entry += 1u) {
        let factor = auxEntry(rowEntriesOffset, entry); value -= factor.coefficient * state[factor.face].preconditioned;
      }
      state[row].preconditioned = value * bitcast<f32>(factorColumns[row].y);
    } }
    storageBarrier(); workgroupBarrier();
  }
  for (var level = 0u; level < params.counts.w; level += 1u) {
    let levelsOffset = bitcast<u32>(params.solve.y);
    let range = vec2u(auxWord(levelsOffset + 4u * level + 2u), auxWord(levelsOffset + 4u * level + 3u));
    if (solveActive) { for (var slot = range.x + lid; slot < range.y; slot += 256u) {
      let column = auxWord(slot); var value = state[column].preconditioned;
      for (var entry = factorColumns[column].x; entry < factorColumns[column + 1u].x; entry += 1u) {
        let factor = factorEntries[entry]; value -= factor.coefficient * state[factor.face].preconditioned;
      }
      state[column].preconditioned = value * bitcast<f32>(factorColumns[column].y);
    } }
    storageBarrier(); workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn precondition(@builtin(local_invocation_id) lid: vec3u) {
  applyPrecondition(lid.x, !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]));
}
@compute @workgroup_size(128)
fn startDirection(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < dofs()) { state[gid.x].direction = state[gid.x].preconditioned; }
}
fn reducePair(local: u32, a: f32, b: f32) {
  reductionA[local] = a; reductionB[local] = b; workgroupBarrier();
  var stride = 128u;
  loop { if (local < stride) { reductionA[local] += reductionA[local + stride]; reductionB[local] += reductionB[local + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
}
@compute @workgroup_size(256)
fn reduceInitial(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var bb = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) { rz += state[row].residual * state[row].preconditioned; bb += state[row].residual * state[row].residual; }
  reducePair(lid.x, rz, bb); if (lid.x == 0u) { scalars[0] = reductionA[0]; scalars[2] = reductionB[0]; scalars[3] = max(reductionB[0], 1e-30); scalars[7] = reductionB[0]; }
}
// Uncoupled pressure is a persistent one-workgroup kernel. The sparse IC(0)
// triangular solve already requires one globally ordered workgroup; keeping
// matrix products and reductions in that same dispatch avoids hundreds of
// empty indirect commands after convergence and prevents rebuild readbacks
// from sitting behind a large queue backlog.
@compute @workgroup_size(256)
fn solveUncoupled(@builtin(local_invocation_id) lid: vec3u) {
  for (var row = lid.x; row < dofs(); row += 256u) { initializeRow(row); }
  storageBarrier(); workgroupBarrier();
  applyPrecondition(lid.x, true);
  for (var row = lid.x; row < dofs(); row += 256u) { state[row].direction = state[row].preconditioned; }
  storageBarrier(); workgroupBarrier();
  var initialRz = 0.0; var initialBb = 0.0;
  for (var row = lid.x; row < dofs(); row += 256u) {
    initialRz += state[row].residual * state[row].preconditioned;
    initialBb += state[row].residual * state[row].residual;
  }
  reducePair(lid.x, initialRz, initialBb);
  if (lid.x == 0u) {
    scalars[0] = reductionA[0]; scalars[2] = reductionB[0]; scalars[3] = max(reductionB[0], 1e-30); scalars[7] = reductionB[0];
  }
  storageBarrier(); workgroupBarrier();
  for (var iteration = 0u; iteration < params.counts.z; iteration += 1u) {
    let active = !(scalars[3] > 0.0 && scalars[2] <= params.solve.x * scalars[3]);
    if (active) {
      for (var row = lid.x; row < dofs(); row += 256u) { state[row].matrixDirection = rowProduct(row); }
    }
    storageBarrier(); workgroupBarrier();
    var denominator = 0.0;
    if (active) { for (var row = lid.x; row < dofs(); row += 256u) { denominator += state[row].direction * state[row].matrixDirection; } }
    reducePair(lid.x, denominator, 0.0);
    if (lid.x == 0u) {
      if (active) { scalars[1] = reductionA[0]; scalars[4] = scalars[0] / max(reductionA[0], 1e-30); scalars[9] += 1.0; }
      else { scalars[4] = 0.0; }
    }
    storageBarrier(); workgroupBarrier();
    if (active) { for (var row = lid.x; row < dofs(); row += 256u) {
      state[row].pressure += scalars[4] * state[row].direction;
      state[row].residual -= scalars[4] * state[row].matrixDirection;
    } }
    storageBarrier(); workgroupBarrier();
    applyPrecondition(lid.x, active);
    var rz = 0.0; var rr = 0.0;
    if (active) { for (var row = lid.x; row < dofs(); row += 256u) {
      rz += state[row].residual * state[row].preconditioned;
      rr += state[row].residual * state[row].residual;
    } }
    reducePair(lid.x, rz, rr);
    if (lid.x == 0u && active) {
      scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]);
      scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]);
    }
    storageBarrier(); workgroupBarrier();
    if (active) { for (var row = lid.x; row < dofs(); row += 256u) {
      if (scalars[2] <= scalars[7]) { state[row].bestPressure = state[row].pressure; }
      state[row].direction = state[row].preconditioned + scalars[6] * state[row].direction;
    } }
    if (lid.x == 0u && active) { scalars[0] = scalars[5]; }
    storageBarrier(); workgroupBarrier();
  }
}
@compute @workgroup_size(128)
fn multiply(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs()) { state[gid.x].matrixDirection = rowProduct(gid.x); } }
@compute @workgroup_size(256)
fn reduceDenominator(@builtin(local_invocation_id) lid: vec3u) {
  var value = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { value += state[row].direction * state[row].matrixDirection; }
  reducePair(lid.x, value, 0.0); if (lid.x == 0u) { scalars[1] = reductionA[0]; scalars[4] = select(scalars[0] / max(reductionA[0], 1e-30), 0.0, scalars[2] <= params.solve.x * scalars[3]); }
}
@compute @workgroup_size(128)
fn updateSolution(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row >= dofs()) { return; } let alpha = scalars[4];
  state[row].pressure += alpha * state[row].direction; state[row].residual -= alpha * state[row].matrixDirection; state[row].preconditioned = state[row].residual / state[row].diagonal;
}
@compute @workgroup_size(256)
fn reduceResidual(@builtin(local_invocation_id) lid: vec3u) {
  var rz = 0.0; var rr = 0.0; for (var row = lid.x; row < dofs(); row += 256u) { rz += state[row].residual * state[row].preconditioned; rr += state[row].residual * state[row].residual; }
  reducePair(lid.x, rz, rr); if (lid.x == 0u) { scalars[5] = reductionA[0]; scalars[2] = reductionB[0]; scalars[7] = min(scalars[7], reductionB[0]); scalars[6] = reductionA[0] / max(abs(scalars[0]), 1e-30) * sign(scalars[0]); }
}
@compute @workgroup_size(128)
fn saveBest(@builtin(global_invocation_id) gid: vec3u) { if (gid.x < dofs() && scalars[2] <= scalars[7]) { state[gid.x].bestPressure = state[gid.x].pressure; } }
@compute @workgroup_size(128)
fn updateDirection(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x; if (row < dofs()) { state[row].direction = state[row].preconditioned + scalars[6] * state[row].direction; }
  if (row == 0u) { scalars[0] = scalars[5]; }
}
fn cellIndex(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
// Narita Sec. 4.4: the monolithic body coupling K = [grad]^T [V] (1-[A]) [L] is
// rank six per body. Static rows live in the aux words texture; the per-body
// six-vectors live past the CG scalars (base word 12, stride 8 per body).
fn auxF32(index: u32) -> f32 { return bitcast<f32>(auxWord(index)); }
var<workgroup> coupleScratch: array<f32, 256>;
fn coupleGather(lid: u32, body: u32, usePressure: bool) {
  let offsets = params.coupling.x;
  let start = auxWord(offsets + body); let end = auxWord(offsets + body + 1u);
  let entries = offsets + params.coupling.w + 1u;
  var sums = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var slot = start + lid; slot < end; slot += 256u) {
    let base = entries + slot * 8u; let dof = auxWord(base);
    let value = select(state[dof].direction, state[dof].bestPressure, usePressure);
    for (var component = 0u; component < 6u; component += 1u) { sums[component] += auxF32(base + 2u + component) * value; }
  }
  for (var component = 0u; component < 6u; component += 1u) {
    coupleScratch[lid] = sums[component]; workgroupBarrier();
    var stride = 128u;
    loop { if (lid < stride) { coupleScratch[lid] += coupleScratch[lid + stride]; } workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u; }
    if (lid == 0u) { scalars[12u + body * 8u + component] = coupleScratch[0]; }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn coupleReduce(@builtin(local_invocation_id) lid: vec3u) {
  for (var body = 0u; body < params.coupling.w; body += 1u) { coupleGather(lid.x, body, false); }
  storageBarrier(); workgroupBarrier();
  if (lid.x < params.coupling.w) {
    let body = lid.x; let table = params.coupling.z + body * 12u; let base = 12u + body * 8u;
    let g3 = scalars[base + 3u]; let g4 = scalars[base + 4u]; let g5 = scalars[base + 5u];
    scalars[base] *= auxF32(table); scalars[base + 1u] *= auxF32(table); scalars[base + 2u] *= auxF32(table);
    scalars[base + 3u] = auxF32(table + 1u) * g3 + auxF32(table + 2u) * g4 + auxF32(table + 3u) * g5;
    scalars[base + 4u] = auxF32(table + 4u) * g3 + auxF32(table + 5u) * g4 + auxF32(table + 6u) * g5;
    scalars[base + 5u] = auxF32(table + 7u) * g3 + auxF32(table + 8u) * g4 + auxF32(table + 9u) * g5;
  }
}
@compute @workgroup_size(128)
fn coupleApply(@builtin(global_invocation_id) gid: vec3u) {
  let distinct = params.couplingCounts.x;
  if (gid.x >= distinct) { return; }
  let dofIds = params.coupling.y; let starts = dofIds + distinct; let entries = starts + distinct + 1u;
  let dof = auxWord(dofIds + gid.x);
  var sum = 0.0;
  for (var slot = auxWord(starts + gid.x); slot < auxWord(starts + gid.x + 1u); slot += 1u) {
    let base = entries + slot * 8u; let body = auxWord(base);
    for (var component = 0u; component < 6u; component += 1u) { sum += auxF32(base + 2u + component) * scalars[12u + body * 8u + component]; }
  }
  state[dof].matrixDirection += sum;
}
@compute @workgroup_size(256)
fn coupleImpulse(@builtin(local_invocation_id) lid: vec3u) {
  // Raw K^T p per body; the host converts to impulses via -rho and M^-1.
  for (var body = 0u; body < params.coupling.w; body += 1u) { coupleGather(lid.x, body, true); }
}
fn solvedFaceGradient(face: Face) -> f32 {
  var result = 0.0;
  for (var slot = 0u; slot < faceNodeCount(face); slot += 1u) {
    let node = face.nodes[slot]; if (node != 0xffffffffu) { result += face.coefficients[slot] * state[node].bestPressure; }
  }
  return result;
}
// Ando--Batty MLS sub-face row: the precomputed weights already include the
// conservation shift, so the sub-face corrections average exactly to the
// solved variational face value.
fn mlsRowGradient(slot: u32) -> f32 {
  let base = params.couplingCounts.y;
  let entries = base + params.couplingCounts.z + 1u;
  var result = 0.0;
  for (var cursor = auxWord(base + slot); cursor < auxWord(base + slot + 1u); cursor += 1u) {
    result += bitcast<f32>(auxWord(entries + 2u * cursor + 1u)) * state[auxWord(entries + 2u * cursor)].bestPressure;
  }
  return result;
}
@compute @workgroup_size(4,4,4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; } let id = vec3i(gid);
  var value = textureLoad(velocityIn, id, 0).xyz; let projection = textureLoad(cellProjection, id, 0);
  let ownLiquid = projection.w > 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    var plus = gid; plus[axis] += 1u;
    if (plus[axis] >= params.dims[axis]) { value[axis] = 0.0; continue; }
    let plusProjection = textureLoad(cellProjection, vec3i(plus), 0); let otherLiquid = plusProjection.w > 0.0;
    if (!ownLiquid && !otherLiquid) { value[axis] = 0.0; continue; }
    let packedFace = u32(round(projection[axis]));
    if (packedFace > 0u) {
      let face = faces[packedFace - 1u];
      // A variational face stores the area-average velocity constrained by
      // Eq. (1). Sub-faces with an MLS row get the reconstructed gradient
      // whose face average is exactly the solved value; the rest receive the
      // constant, conservative prolongation.
      let fluidScale = select(0.0, face.weights.y / face.weights.x, face.weights.x > 0.0);
      let mlsSlot = textureLoad(mlsRowIndex, id, 0)[axis];
      if (mlsSlot > 0u) { value[axis] -= fluidScale * mlsRowGradient(mlsSlot - 1u); }
      else { value[axis] -= fluidScale * solvedFaceGradient(face); }
    }
  }
  textureStore(velocityOut, id, vec4f(value, 0.0));
}
`;

// Convergence-driven indirect dispatch arguments live in their own buffer and
// pipeline. That separation is intentional: WebGPU validates each dispatch as
// one usage scope, so the same buffer cannot be a writable storage binding and
// the indirect argument source of that dispatch. A tiny preceding dispatch
// writes the arguments; all pressure kernels can then remain in one pass.
export const quadtreeDispatchShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> scalars: array<f32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn updateDispatch() {
  let keepSolving = !(scalars[3] > 0.0 && scalars[2] <= bitcast<f32>(args[18]) * scalars[3]);
  if (keepSolving) { scalars[9] += 1.0; }
  for (var word = 0u; word < 9u; word += 1u) { args[word] = select(0u, args[9u + word], keepSolving || word % 3u != 0u); }
}
`;

export class WebGPUQuadtreeTallCellProjection {
  readonly info: QuadtreeTallCellProjectionInfo;
  private readonly buffers: GPUBuffer[];
  private readonly bindGroup: GPUBindGroup;
  private pipelines: Record<string, GPUComputePipeline>;
  private readonly shaderModule:GPUShaderModule;
  private readonly pipelineLayout:GPUPipelineLayout;
  private readonly gpuCache: QuadtreeGPUCache;
  private readonly params: GPUBuffer;
  private readonly scalarBuffer: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly dispatchBindGroup: GPUBindGroup;
  private readonly dispatchPipeline: GPUComputePipeline;
  private readonly cellProjection: GPUTexture;
  private readonly cellTopology: GPUTexture;
  private readonly factorAux: GPUTexture;
  private readonly mlsRowIndex: GPUTexture;
  private readonly dofCount: number;
  private readonly iterations: number;
  private readonly levelSetTexture: GPUTexture;
  private readonly topologyWords: Uint32Array;
  private facesDirty = false;
  private lastRelativeResidual?: number;
  levelSetMismatchFraction?: number;
  bodyPressureImpulses?: QuadtreeBodyImpulse[];
  private readonly couplingBodyCount: number;
  private readonly couplingDistinctDofs: number;
  private readonly couplingBodyIndices: number[];
  private readonly displacedVolumes: number[];
  private lastResidualRms?: number;
  private lastInitialResidualRms?: number;

  constructor(private readonly device: GPUDevice, private readonly scene: SceneDescription, private readonly dims: { nx: number; ny: number; nz: number }, private readonly resources: ProjectionResources, private readonly options: QuadtreeTallCellProjectionOptions, fields?: ProjectionFields, private readonly coupling?: QuadtreeRigidCoupling,deferPipelineCompilation=false,cache?:QuadtreeGPUCache) {
    const constructorStartedAt = performance.now();
    const { nx, ny, nz } = dims, h = { x: scene.container.width_m / nx, y: scene.container.height_m / ny, z: scene.container.depth_m / nz };
    const initial: ProjectionFields = fields ?? initialFields(scene, nx, ny, nz);
    if (!fields) initial.phi = signedDistanceFromVolume(Float32Array.from(initial.phi, (value) => value < 0 ? 1 : 0), nx, ny, nz, h);
    this.levelSetTexture = device.createTexture({ label: "Quadtree pressure level set", size: [nx, ny, nz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    uploadLevelSet(device, this.levelSetTexture, initial.phi, nx, ny, nz);
    const explicitSizing = initialSizing(scene, nx, nz, h);
    let quadtree = initial.quadtree;
    if (!quadtree) {
      if (!initial.velocity) throw new Error("Initial quadtree construction needs a velocity field");
      const sizing = quadtreeSizingFromVelocityAndSurface(initial.phi, initial.velocity, nx, ny, nz, h);
      for (let index = 0; index < sizing.length; index += 1) sizing[index] = Math.max(sizing[index], explicitSizing[index]);
      quadtree = buildQuadtree(sizing, nx, nz, { h: Math.min(h.x, h.z), maximumLeafSize: options.maximumLeafSize, adaptivityStrength: options.adaptivityStrength, smoothingDilations: 3 });
    }
    const tallGridStartedAt = performance.now();
    const opticalDepth = Math.max(1, Math.ceil(ny * scene.container.fillFraction * options.opticalDepthFraction));
    const pressureGrid = initial.pressureGrid ?? populateTallPressureGrid(quadtree, initial.phi, ny, h, opticalDepth);
    this.topologyWords = initial.topologyWords ?? pressureTopologyWords(pressureGrid);
    const tallGrid_ms = performance.now() - tallGridStartedAt;
    const variationalStartedAt = performance.now();
    const solidFields = coupling ? solidFieldsFromBodies(scene, coupling.bodies, nx, ny, nz, h) : undefined;
    const variationalBodies = coupling ? variationalBodiesFrom(scene, coupling) : [];
    const system = buildVariationalSystem(pressureGrid, {
      velocity: initial.velocity,
      solidFraction: solidFields?.solidFraction, solidOwner: solidFields?.solidOwner, bodies: variationalBodies
    }, { assembleDense: false });
    const variationalAssembly_ms = performance.now() - variationalStartedAt;
    this.displacedVolumes = (coupling?.bodies ?? []).map(() => 0);
    if (solidFields) {
      const cellVolume = h.x * h.y * h.z;
      for (let index = 0; index < solidFields.solidFraction.length; index += 1) {
        const owner = solidFields.solidOwner[index];
        if (owner >= 0 && initial.phi[index] < 0) this.displacedVolumes[owner] += solidFields.solidFraction[index] * cellVolume;
      }
    }
    this.dofCount = system.liquidSampleIds.length;
    // CG on this operator needs O(sqrt(n)) iterations, and a deep-water column
    // stack in the hundreds of thousands of samples stalls inside a fixed 240
    // budget, so very large systems raise their own floor. Ordinary scenes
    // keep the configured budget (the relative tolerance remains the stop
    // criterion; converged iterations are no-ops).
    const largeSystemFloor = this.dofCount > 50_000 ? Math.min(2048, Math.ceil(3 * Math.sqrt(this.dofCount))) : 1;
    this.iterations = Math.max(1, Math.round(options.pressureIterations), largeSystemFloor);
    const packStartedAt = performance.now();
    const packed = this.packSystem(system, nx, ny, nz, coupling?.dynamic ? variationalBodies : []);
    const systemPack_ms = performance.now() - packStartedAt;
    const uploadStartedAt = performance.now();
    const faces = bufferWithData(device, "Quadtree tall-cell variational faces", packed.faces, GPUBufferUsage.STORAGE, 96);
    const rowOffsets = bufferWithData(device, "Quadtree tall-cell row offsets", packed.rowOffsets);
    const rowEntries = bufferWithData(device, "Quadtree tall-cell row entries", packed.rowEntries, GPUBufferUsage.STORAGE, 8);
    const interpolationBuffer = bufferWithData(device, "Quadtree tall-cell pressure interpolation", packed.interpolation, GPUBufferUsage.STORAGE, 16);
    const factorColumns = bufferWithData(device, "Quadtree tall-cell IC(0) columns", packed.factorColumns, GPUBufferUsage.STORAGE, 8);
    const factorEntries = bufferWithData(device, "Quadtree tall-cell IC(0) entries", packed.factorEntries, GPUBufferUsage.STORAGE, 8);
    const factorAuxTexels = Math.max(1, Math.ceil(packed.factorAuxWords.length / 4)), factorAuxWidth = Math.min(2048, factorAuxTexels), factorAuxHeight = Math.ceil(factorAuxTexels / factorAuxWidth);
    this.factorAux = device.createTexture({ label: "Quadtree tall-cell IC level data", size: [factorAuxWidth, factorAuxHeight], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const factorAuxRowBytes = factorAuxWidth * 16, factorAuxPitch = Math.ceil(factorAuxRowBytes / 256) * 256, factorAuxUpload = new Uint8Array(factorAuxPitch * factorAuxHeight), factorAuxSource = new Uint8Array(packed.factorAuxWords.buffer);
    for (let row = 0; row < factorAuxHeight; row += 1) factorAuxUpload.set(factorAuxSource.subarray(row * factorAuxRowBytes, Math.min(factorAuxSource.length, (row + 1) * factorAuxRowBytes)), row * factorAuxPitch);
    device.queue.writeTexture({ texture: this.factorAux }, factorAuxUpload, { bytesPerRow: factorAuxPitch, rowsPerImage: factorAuxHeight }, { width: factorAuxWidth, height: factorAuxHeight });
    this.cellProjection = device.createTexture({ label: "Quadtree tall-cell projection field", size: [nx, ny, nz], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const projectionRowBytes = nx * 16, projectionPitch = Math.ceil(projectionRowBytes / 256) * 256;
    const projectionBytes = new Uint8Array(projectionPitch * ny * nz), projectionSource = new Uint8Array(packed.cellProjection.buffer);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) projectionBytes.set(projectionSource.subarray(projectionRowBytes * (y + ny * z), projectionRowBytes * (y + ny * z + 1)), projectionPitch * (y + ny * z));
    device.queue.writeTexture({ texture: this.cellProjection }, projectionBytes, { bytesPerRow: projectionPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    this.cellTopology = device.createTexture({ label: "Quadtree tall-cell debug topology", size: [nx, ny, nz], dimension: "3d", format: "rg32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const topologyRowBytes = nx * 8, topologyPitch = Math.ceil(topologyRowBytes / 256) * 256;
    const topologyBytes = new Uint8Array(topologyPitch * ny * nz), topologySource = new Uint8Array(packed.cellTopology.buffer);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) topologyBytes.set(topologySource.subarray(topologyRowBytes * (y + ny * z), topologyRowBytes * (y + ny * z + 1)), topologyPitch * (y + ny * z));
    device.queue.writeTexture({ texture: this.cellTopology }, topologyBytes, { bytesPerRow: topologyPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    this.mlsRowIndex = device.createTexture({ label: "Quadtree tall-cell MLS row index", size: [nx, ny, nz], dimension: "3d", format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const mlsBytes = new Uint8Array(projectionPitch * ny * nz), mlsSource = new Uint8Array(packed.mlsIndex.buffer);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) mlsBytes.set(mlsSource.subarray(projectionRowBytes * (y + ny * z), projectionRowBytes * (y + ny * z + 1)), projectionPitch * (y + ny * z));
    device.queue.writeTexture({ texture: this.mlsRowIndex }, mlsBytes, { bytesPerRow: projectionPitch, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    const state = device.createBuffer({ label: "Quadtree tall-cell PCG state", size: Math.max(28, this.dofCount * 28), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // Words 0..11 are the CG scalars; words 12+ hold the per-body coupling
    // six-vectors (stride 8, up to 12 bodies).
    const scalars = device.createBuffer({ label: "Quadtree tall-cell CG scalars", size: 48 + 12 * 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.scalarBuffer = scalars;
    this.dispatchArgs = device.createBuffer({ label: "Quadtree tall-cell active dispatches", size: 76, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.couplingBodyCount = packed.couplingBodyCount; this.couplingDistinctDofs = packed.couplingDistinctDofs; this.couplingBodyIndices = packed.couplingBodyIndices;
    this.params = device.createBuffer({ label: "Quadtree tall-cell parameters", size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([nx, ny, nz, factorAuxWidth, 0, 0, 0, 0]).buffer);
    device.queue.writeBuffer(this.params, 16, new Float32Array([h.x, h.y, h.z, 0]).buffer);
    device.queue.writeBuffer(this.params, 32, new Uint32Array([this.dofCount, system.faces.length, this.iterations, packed.factorLevelCount]).buffer);
    const solveParams = new ArrayBuffer(16); new Float32Array(solveParams)[0] = options.relativeTolerance ** 2;
    new Uint32Array(solveParams).set([packed.levelsOffset, packed.rowOffsetsOffset, packed.rowEntriesOffset], 1); device.queue.writeBuffer(this.params, 48, solveParams);
    device.queue.writeBuffer(this.params, 64, new Uint32Array([packed.couplingByBodyOffset, packed.couplingByDofOffset, packed.couplingTableOffset, packed.couplingBodyCount]).buffer);
    device.queue.writeBuffer(this.params, 80, new Uint32Array([packed.couplingDistinctDofs, packed.mlsOffsetsBase, packed.mlsRowCount, 0]).buffer);
    const rowGroups = Math.ceil(this.dofCount / 128), couplingGroups = Math.ceil(packed.couplingDistinctDofs / 128);
    const dispatchWords = new Uint32Array(19);
    dispatchWords.set([rowGroups, 1, 1, 1, 1, 1, couplingGroups, 1, 1, rowGroups, 1, 1, 1, 1, 1, couplingGroups, 1, 1]);
    dispatchWords[18] = new Uint32Array(new Float32Array([options.relativeTolerance ** 2]).buffer)[0];
    device.queue.writeBuffer(this.dispatchArgs, 0, dispatchWords);
    let layout: GPUBindGroupLayout;
    if (cache) {
      layout = cache.layout; this.shaderModule = cache.module; this.pipelineLayout = cache.pipelineLayout; this.gpuCache = cache;
    } else {
      layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ...Array.from({ length: 3 }, (_, index) => ({ binding: index + 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
        { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
        { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
      ] });
      this.shaderModule = device.createShaderModule({ label: "Quadtree tall-cell variational PCG", code: quadtreeTallCellProjectionShader });
      void this.shaderModule.getCompilationInfo().then((result) => { for (const message of result.messages) if (message.type === "error") console.error(`Quadtree tall-cell WGSL ${message.lineNum}:${message.linePos} ${message.message}`); }).catch(()=>{/* Device loss is reported by the renderer. */});
      this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      this.gpuCache = { layout, module: this.shaderModule, pipelineLayout: this.pipelineLayout };
    }
    if (!this.gpuCache.dispatchLayout || !this.gpuCache.dispatchPipeline) {
      this.gpuCache.dispatchLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ] });
      const dispatchModule = device.createShaderModule({ label: "Quadtree convergence dispatch", code: quadtreeDispatchShader });
      const dispatchPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.gpuCache.dispatchLayout] });
      this.gpuCache.dispatchPipeline = device.createComputePipeline({ label: "Quadtree convergence dispatch", layout: dispatchPipelineLayout, compute: { module: dispatchModule, entryPoint: "updateDispatch" } });
    }
    this.dispatchPipeline = this.gpuCache.dispatchPipeline;
    this.dispatchBindGroup = device.createBindGroup({ layout: this.gpuCache.dispatchLayout, entries: [
      { binding: 0, resource: { buffer: scalars } }, { binding: 1, resource: { buffer: this.dispatchArgs } }
    ] });
    const names = ["refreshFaces", "solveUncoupled", "initialize", "precondition", "startDirection", "reduceInitial", "multiply", "reduceDenominator", "updateSolution", "reduceResidual", "saveBest", "updateDirection", "project", "coupleReduce", "coupleApply", "coupleImpulse"];
    this.pipelines = this.gpuCache.pipelines ?? (deferPipelineCompilation ? {} : Object.fromEntries(names.map((entryPoint) => [entryPoint, device.createComputePipeline(this.pipelineDescriptor(entryPoint))])));
    if (!this.gpuCache.pipelines && !deferPipelineCompilation) this.gpuCache.pipelines = this.pipelines;
    const all = [faces, rowOffsets, rowEntries, interpolationBuffer, state, scalars, factorColumns, factorEntries];
    this.bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: resources.velocityIn.createView() }, { binding: 1, resource: resources.velocityOut.createView() }, { binding: 2, resource: resources.volume.createView() },
      ...all.slice(0, 4).map((buffer, index) => ({ binding: index + 3, resource: { buffer } })),
      { binding: 7, resource: this.cellProjection.createView() },
      ...all.slice(4, 8).map((buffer, index) => ({ binding: index + 8, resource: { buffer } })), { binding: 12, resource: { buffer: this.params } },
      { binding: 13, resource: this.factorAux.createView() },
      { binding: 14, resource: this.mlsRowIndex.createView() },
      { binding: 15, resource: this.levelSetTexture.createView() }
    ] });
    this.buffers = [...all, this.dispatchArgs];
    const tallSegmentCount = pressureGrid.segments.filter((segment) => segment.tall).length;
    const allocatedBytes = this.buffers.reduce((sum, buffer) => sum + buffer.size, this.params.size + nx * ny * nz * 24);
    const resourceUpload_ms = performance.now() - uploadStartedAt;
    this.info = { leafCount: quadtree.leaves.length, pressureSampleCount: pressureGrid.samples.length, liquidDofCount: this.dofCount, faceCount: system.faces.length, tallSegmentCount, ghostFaceCount: system.faces.filter((face) => face.ghost).length, maximumNeighborRatio: quadtree.maximumNeighborRatio, compressionRatio: this.dofCount / Math.max(1, nx * ny * nz), maximumFluidScale: system.faces.reduce((maximum, face) => Math.max(maximum, face.fluidScale), 0), allocatedBytes, cpuTallGrid_ms: tallGrid_ms, cpuVariationalAssembly_ms: variationalAssembly_ms, cpuSystemPack_ms: systemPack_ms, cpuResourceUpload_ms: resourceUpload_ms, cpuTopologyPack_ms: performance.now() - constructorStartedAt, topologyReused: false, topologyReuseCount: 0 };
  }

  private pipelineDescriptor(entryPoint:string):GPUComputePipelineDescriptor{return{label:`Quadtree tall-cell ${entryPoint}`,layout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  static async createAsync(device:GPUDevice,scene:SceneDescription,dims:{nx:number;ny:number;nz:number},resources:ProjectionResources,options:QuadtreeTallCellProjectionOptions,fields?:ProjectionFields,coupling?:QuadtreeRigidCoupling,onProgress:(label:string,completed:number,total:number)=>void=()=>{},cache?:QuadtreeGPUCache){const projection=new WebGPUQuadtreeTallCellProjection(device,scene,dims,resources,options,fields,coupling,true,cache);await projection.initializePipelines(onProgress);return projection;}
  async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void=()=>{}){
    if(this.gpuCache.pipelines){this.pipelines=this.gpuCache.pipelines;return;}
    const names=["refreshFaces","solveUncoupled","initialize","precondition","startDirection","reduceInitial","multiply","reduceDenominator","updateSolution","reduceResidual","saveBest","updateDirection","project","coupleReduce","coupleApply","coupleImpulse"];const pipelines:Record<string,GPUComputePipeline>={};for(let index=0;index<names.length;index+=1){const entryPoint=names[index];onProgress(`Adaptive pressure · ${entryPoint}`,index,names.length);pipelines[entryPoint]=await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint));onProgress(`Adaptive pressure · ${entryPoint}`,index+1,names.length);}this.pipelines=pipelines;this.gpuCache.pipelines=pipelines;
  }

  private packSystem(system: VariationalSystem, nx: number, ny: number, nz: number, dynamicBodies: VariationalBody[]) {
    // The last two vec4s retain each pressure sample's leaf origin/y and span.
    // They let a reused topology refresh free-surface weights from the current
    // GPU level set without rebuilding or uploading the sparse face graph.
    const faceStride = 24, faces = new ArrayBuffer(system.faces.length * faceStride * 4), faceU32 = new Uint32Array(faces), faceF32 = new Float32Array(faces);
    const incident: Array<Array<{ face: number; coefficient: number }>> = Array.from({ length: system.liquidSampleIds.length }, () => []);
    const cellProjection = new Float32Array(nx * ny * nz * 4), cellTopology = adaptivePressureCellTopology(system.grid);
    const matrixRows: Array<Map<number, number>> = Array.from({ length: system.liquidSampleIds.length }, () => new Map<number, number>());
    system.faces.forEach((face, faceIndex) => {
      const offset = faceIndex * faceStride, nodeCount = face.nodes.length;
      for (let slot = 0; slot < 4; slot += 1) {
        const dof = slot < nodeCount ? system.dofBySample[face.nodes[slot]] : -1;
        faceU32[offset + slot] = dof < 0 ? INVALID : dof;
        faceF32[offset + 4 + slot] = slot < nodeCount ? face.coefficients[slot] : 0;
        if (slot < nodeCount) {
          const sample = system.grid.samples[face.nodes[slot]], leaf = system.grid.quadtree.leaves[sample.leaf];
          faceU32[offset + 16 + slot] = leaf.x | (leaf.z << 10) | (sample.y << 20);
          faceU32[offset + 20 + slot] = leaf.size;
        }
        if (dof >= 0) incident[dof].push({ face: faceIndex, coefficient: face.coefficients[slot] });
      }
      faceU32.set([face.bounds.x, face.bounds.z, face.bounds.y0, face.bounds.y1], offset + 8);
      faceU32[offset + 12] = face.bounds.span | (face.axis << 16) | (nodeCount << 18) | ((face.ghost ? 1 : 0) << 21);
      const va = face.volume * face.openFraction;
      faceF32[offset + 13] = face.volume * face.solidFlux;
      faceF32.set([va, va * face.fluidScale], offset + 14);
      const matrixWeight = va * face.fluidScale;
      const liquidTerms: Array<{ dof: number; coefficient: number }> = [];
      for (let slot = 0; slot < nodeCount; slot += 1) {
        const dof = system.dofBySample[face.nodes[slot]];
        if (dof >= 0) liquidTerms.push({ dof, coefficient: face.coefficients[slot] });
      }
      for (const a of liquidTerms) for (const b of liquidTerms) matrixRows[a.dof].set(b.dof, (matrixRows[a.dof].get(b.dof) ?? 0) + a.coefficient * b.coefficient * matrixWeight);
      if (face.axis === 0 || face.axis === 2) {
        for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) for (let transverse = 0; transverse < face.bounds.span; transverse += 1) {
          const x = face.axis === 0 ? face.bounds.x : face.bounds.x + transverse;
          const z = face.axis === 2 ? face.bounds.z : face.bounds.z + transverse;
          if (x < nx && z < nz) cellProjection[4 * (x + nx * (y + ny * z)) + face.axis] = faceIndex + 1;
        }
      } else {
        const leaf = system.grid.quadtree.leaves[system.grid.samples[face.nodes[0]].leaf];
        for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) cellProjection[4 * (x + nx * (y + ny * z)) + 1] = faceIndex + 1;
      }
    });
    const rowOffsets = new Uint32Array(incident.length + 1), entryCount = incident.reduce((sum, row) => sum + row.length, 0), rowEntries = new ArrayBuffer(Math.max(1, entryCount) * 8), entryU32 = new Uint32Array(rowEntries), entryF32 = new Float32Array(rowEntries);
    let entry = 0; incident.forEach((row, index) => { rowOffsets[index] = entry; for (const item of row) { entryU32[2 * entry] = item.face; entryF32[2 * entry + 1] = item.coefficient; entry += 1; } }); rowOffsets[incident.length] = entry;
    const interpolationBuffer = new ArrayBuffer(nx * ny * nz * 16), interpolationU32 = new Uint32Array(interpolationBuffer), interpolationF32 = new Float32Array(interpolationBuffer);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const leafId = system.grid.quadtree.leafAt[x + nx * z], leaf = system.grid.quadtree.leaves[leafId];
      for (let y = 0; y < ny; y += 1) {
        const [a, b, wa, wb] = interpolation(system.grid.samplesByLeaf[leafId], y), index = x + nx * (y + ny * z), dofA = system.dofBySample[a], dofB = system.dofBySample[b];
        interpolationU32[4 * index] = dofA < 0 ? INVALID : dofA; interpolationU32[4 * index + 1] = dofB < 0 ? INVALID : dofB;
        interpolationF32[4 * index + 2] = wa; interpolationF32[4 * index + 3] = wb;
        const representedPhi = wa * system.grid.samples[a].phi + wb * system.grid.samples[b].phi;
        cellProjection[4 * index + 3] = representedPhi < 0 ? leaf.size : -leaf.size;
      }
    }
    // Bridson's public-domain modified incomplete Cholesky level-zero
    // factorization (omega=0.97, minimum pivot ratio=0.25), generalized from
    // the regular seven-point stencil to this sparse variational matrix.
    const n = matrixRows.length, factorStarts = new Uint32Array(n + 1), factorRows: number[] = [], factorValues: number[] = [];
    const factorPositions: Array<Map<number, number>> = Array.from({ length: n }, () => new Map<number, number>());
    const originalDiagonal = new Float64Array(n), workDiagonal = new Float64Array(n), inverseDiagonal = new Float64Array(n);
    for (let column = 0; column < n; column += 1) {
      factorStarts[column] = factorRows.length;
      const diagonal = matrixRows[column].get(column) ?? 0; originalDiagonal[column] = diagonal; workDiagonal[column] = diagonal;
      for (const row of [...matrixRows[column].keys()].filter((row) => row > column).sort((a, b) => a - b)) {
        factorPositions[column].set(row, factorRows.length); factorRows.push(row); factorValues.push(matrixRows[column].get(row) ?? 0);
      }
    }
    factorStarts[n] = factorRows.length;
    for (let column = 0; column < n; column += 1) {
      if (!(originalDiagonal[column] > 0)) continue;
      const safePivot = Number.isFinite(workDiagonal[column]) && workDiagonal[column] >= 0.25 * originalDiagonal[column] ? workDiagonal[column] : originalDiagonal[column];
      inverseDiagonal[column] = 1 / Math.sqrt(Math.max(safePivot, 1e-30));
      for (let p = factorStarts[column]; p < factorStarts[column + 1]; p += 1) factorValues[p] *= inverseDiagonal[column];
      for (let p = factorStarts[column]; p < factorStarts[column + 1]; p += 1) {
        const targetColumn = factorRows[p], multiplier = factorValues[p]; let missing = 0;
        for (let a = factorStarts[column]; a < factorStarts[column + 1]; a += 1) {
          const targetRow = factorRows[a], source = factorValues[a];
          if (targetRow < targetColumn) {
            if (!matrixRows[targetColumn].has(targetRow)) missing += source;
          } else if (targetRow === targetColumn) {
            workDiagonal[targetColumn] -= multiplier * source;
          } else {
            const destination = factorPositions[targetColumn].get(targetRow);
            if (destination === undefined) missing += source;
            else factorValues[destination] -= multiplier * source;
          }
        }
        // Narita et al. report that Bridson's modified variant performs
        // poorly because this variational operator is not an M-matrix.  Their
        // chosen solver is ICCG with minor changes, i.e. IC(0) without MIC's
        // dropped-fill diagonal compensation.
        workDiagonal[targetColumn] -= 0 * multiplier * missing;
      }
    }
    const factorColumnsBuffer = new ArrayBuffer(Math.max(1, n + 1) * 8), factorColumnsU32 = new Uint32Array(factorColumnsBuffer), factorColumnsF32 = new Float32Array(factorColumnsBuffer);
    for (let column = 0; column <= n; column += 1) factorColumnsU32[2 * column] = factorStarts[column];
    for (let column = 0; column < n; column += 1) factorColumnsF32[2 * column + 1] = inverseDiagonal[column];
    const factorEntriesBuffer = new ArrayBuffer(Math.max(1, factorRows.length) * 8), factorEntriesU32 = new Uint32Array(factorEntriesBuffer), factorEntriesF32 = new Float32Array(factorEntriesBuffer);
    factorRows.forEach((row, index) => { factorEntriesU32[2 * index] = row; factorEntriesF32[2 * index + 1] = factorValues[index]; });
    const rowFactors: Array<Array<{ column: number; value: number }>> = Array.from({ length: n }, () => []);
    for (let column = 0; column < n; column += 1) for (let entryIndex = factorStarts[column]; entryIndex < factorStarts[column + 1]; entryIndex += 1) rowFactors[factorRows[entryIndex]].push({ column, value: factorValues[entryIndex] });
    const forwardLevel = new Uint32Array(n), backwardLevel = new Uint32Array(n);
    for (let row = 0; row < n; row += 1) for (const factor of rowFactors[row]) forwardLevel[row] = Math.max(forwardLevel[row], forwardLevel[factor.column] + 1);
    for (let column = n - 1; column >= 0; column -= 1) for (let entryIndex = factorStarts[column]; entryIndex < factorStarts[column + 1]; entryIndex += 1) backwardLevel[column] = Math.max(backwardLevel[column], backwardLevel[factorRows[entryIndex]] + 1);
    let deepestLevel = 0;
    for (let row = 0; row < n; row += 1) deepestLevel = Math.max(deepestLevel, forwardLevel[row], backwardLevel[row]);
    const levelCount = Math.max(1, 1 + deepestLevel);
    const forwardByLevel: number[][] = Array.from({ length: levelCount }, () => []), backwardByLevel: number[][] = Array.from({ length: levelCount }, () => []);
    for (let row = 0; row < n; row += 1) { forwardByLevel[forwardLevel[row]].push(row); backwardByLevel[backwardLevel[row]].push(row); }
    const schedule = new Uint32Array(Math.max(1, 2 * n)), levels = new Uint32Array(levelCount * 4); let scheduleOffset = 0;
    for (let level = 0; level < levelCount; level += 1) {
      levels[4 * level] = scheduleOffset; schedule.set(forwardByLevel[level], scheduleOffset); scheduleOffset += forwardByLevel[level].length; levels[4 * level + 1] = scheduleOffset;
    }
    for (let level = 0; level < levelCount; level += 1) {
      levels[4 * level + 2] = scheduleOffset; schedule.set(backwardByLevel[level], scheduleOffset); scheduleOffset += backwardByLevel[level].length; levels[4 * level + 3] = scheduleOffset;
    }
    const factorRowOffsets = new Uint32Array(n + 1), factorRowEntriesBuffer = new ArrayBuffer(Math.max(1, factorRows.length) * 8), factorRowEntriesU32 = new Uint32Array(factorRowEntriesBuffer), factorRowEntriesF32 = new Float32Array(factorRowEntriesBuffer);
    let rowEntry = 0;
    for (let row = 0; row < n; row += 1) {
      factorRowOffsets[row] = rowEntry;
      for (const factor of rowFactors[row]) { factorRowEntriesU32[2 * rowEntry] = factor.column; factorRowEntriesF32[2 * rowEntry + 1] = factor.value; rowEntry += 1; }
    }
    factorRowOffsets[n] = rowEntry;
    const levelsOffset = scheduleOffset, rowOffsetsOffset = levelsOffset + levels.length, rowEntriesOffset = rowOffsetsOffset + factorRowOffsets.length;
    // Rank-6 body couplings ride in the same aux-words texture (the storage
    // binding budget is exhausted): a by-body CSR for K^T reductions, a
    // by-DOF CSR for race-free K applications, and the per-body generalized
    // inverse masses (rho/m, then rho R I^-1 R^T row-major).
    const couplings = dynamicBodies.length > 0 ? system.couplings : [];
    const couplingRowCount = couplings.reduce((sum, coupling) => sum + coupling.rows.size, 0);
    const couplingBodyCount = couplings.length;
    const byDof = new Map<number, Array<{ body: number; row: Float64Array }>>();
    couplings.forEach((coupling, slot) => {
      // `slot` is the packed body index used by the WGSL loops and the mass
      // table; empty couplings were filtered out upstream.
      for (const [dof, row] of coupling.rows) {
        let list = byDof.get(dof);
        if (!list) { list = []; byDof.set(dof, list); }
        list.push({ body: slot, row });
      }
    });
    const couplingDistinctDofs = byDof.size;
    const couplingByBodyOffset = rowEntriesOffset + 2 * rowEntry;
    const couplingByDofOffset = couplingByBodyOffset + (couplingBodyCount + 1) + couplingRowCount * 8;
    const couplingTableOffset = couplingByDofOffset + couplingDistinctDofs + (couplingDistinctDofs + 1) + couplingRowCount * 8;
    // MLS pressure-mapping rows (Narita Alg. 1 line 10 via Ando--Batty MLS)
    // follow the couplings in the same aux texture.
    const mlsRows = buildMlsProjectionRows(system);
    const mlsEntryCount = mlsRows.reduce((sum, row) => sum + row.entries.length, 0);
    const mlsOffsetsBase = couplingTableOffset + couplingBodyCount * 12;
    const totalWords = mlsOffsetsBase + (mlsRows.length + 1) + mlsEntryCount * 2;
    const factorAuxWords = new Uint32Array(Math.max(4, totalWords));
    const factorAuxFloats = new Float32Array(factorAuxWords.buffer);
    factorAuxWords.set(schedule.subarray(0, scheduleOffset), 0); factorAuxWords.set(levels, levelsOffset); factorAuxWords.set(factorRowOffsets, rowOffsetsOffset);
    factorAuxWords.set(new Uint32Array(factorRowEntriesBuffer, 0, 2 * rowEntry), rowEntriesOffset);
    if (couplingBodyCount > 0) {
      let entryCursor = 0;
      const entriesBase = couplingByBodyOffset + couplingBodyCount + 1;
      couplings.forEach((coupling, index) => {
        factorAuxWords[couplingByBodyOffset + index] = entryCursor;
        for (const [dof, row] of coupling.rows) {
          const base = entriesBase + entryCursor * 8;
          factorAuxWords[base] = dof;
          for (let component = 0; component < 6; component += 1) factorAuxFloats[base + 2 + component] = row[component];
          entryCursor += 1;
        }
      });
      factorAuxWords[couplingByBodyOffset + couplingBodyCount] = entryCursor;
      const dofIds = [...byDof.keys()];
      const startsBase = couplingByDofOffset + couplingDistinctDofs, dofEntriesBase = startsBase + couplingDistinctDofs + 1;
      let dofCursor = 0;
      dofIds.forEach((dof, index) => {
        factorAuxWords[couplingByDofOffset + index] = dof;
        factorAuxWords[startsBase + index] = dofCursor;
        for (const entry of byDof.get(dof)!) {
          const base = dofEntriesBase + dofCursor * 8;
          factorAuxWords[base] = entry.body;
          for (let component = 0; component < 6; component += 1) factorAuxFloats[base + 2 + component] = entry.row[component];
          dofCursor += 1;
        }
      });
      factorAuxWords[startsBase + couplingDistinctDofs] = dofCursor;
      couplings.forEach((coupling, index) => {
        const body = dynamicBodies[coupling.body], base = couplingTableOffset + index * 12;
        factorAuxFloats[base] = body.inverseMass;
        for (let component = 0; component < 9; component += 1) factorAuxFloats[base + 1 + component] = body.inverseInertia[component];
      });
    }
    const mlsIndex = new Uint32Array(nx * ny * nz * 4);
    {
      const entriesBase = mlsOffsetsBase + mlsRows.length + 1;
      let cursor = 0;
      mlsRows.forEach((row, index) => {
        factorAuxWords[mlsOffsetsBase + index] = cursor;
        for (const [dof, weight] of row.entries) {
          factorAuxWords[entriesBase + 2 * cursor] = dof;
          factorAuxFloats[entriesBase + 2 * cursor + 1] = weight;
          cursor += 1;
        }
        mlsIndex[4 * row.cell + row.axis] = index + 1;
      });
      factorAuxWords[mlsOffsetsBase + mlsRows.length] = cursor;
    }
    return {
      faces: new Uint8Array(faces), rowOffsets, rowEntries: new Uint8Array(rowEntries, 0, entryCount * 8), interpolation: new Uint8Array(interpolationBuffer), cellProjection, cellTopology,
      factorColumns: new Uint8Array(factorColumnsBuffer), factorEntries: new Uint8Array(factorEntriesBuffer, 0, factorRows.length * 8),
      factorAuxWords, factorLevelCount: levelCount, levelsOffset, rowOffsetsOffset, rowEntriesOffset,
      couplingByBodyOffset, couplingByDofOffset, couplingTableOffset, couplingBodyCount, couplingDistinctDofs,
      couplingBodyIndices: couplings.map((coupling) => coupling.body),
      mlsOffsetsBase, mlsRowCount: mlsRows.length, mlsIndex
    };
  }

  encode(encoder: GPUCommandEncoder, nx: number, ny: number, nz: number) {
    encoder.clearBuffer(this.scalarBuffer);
    // One compute pass for the complete solve avoids thousands of pass
    // begin/end transitions. Dispatch boundaries still provide the required
    // storage visibility, while indirect zero-workgroup dispatches preserve
    // the relative-residual early exit without a CPU readback.
    const pass = encoder.beginComputePass();
    const direct = (entry: string, workgroups: number, y = 1, z = 1) => { pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroups(workgroups, y, z); };
    const indirect = (entry: string, offset: number) => { pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, this.bindGroup); pass.dispatchWorkgroupsIndirect(this.dispatchArgs, offset); };
    if (this.facesDirty) { direct("refreshFaces", Math.ceil(this.info.faceCount / 128)); this.facesDirty = false; }
    const coupled = this.couplingBodyCount > 0;
    if (!coupled) {
      direct("solveUncoupled", 1);
      direct("project", Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
      pass.end();
      return;
    }
    direct("initialize", Math.ceil(this.dofCount / 128)); direct("precondition", 1); direct("startDirection", Math.ceil(this.dofCount / 128)); direct("reduceInitial", 1);
    for (let iteration = 0; iteration < this.iterations; iteration += 1) {
      pass.setPipeline(this.dispatchPipeline); pass.setBindGroup(0, this.dispatchBindGroup); pass.dispatchWorkgroups(1);
      indirect("multiply", 0);
      if (coupled) { indirect("coupleReduce", 12); indirect("coupleApply", 24); }
      indirect("reduceDenominator", 12); indirect("updateSolution", 0); indirect("precondition", 12); indirect("reduceResidual", 12); indirect("saveBest", 0); indirect("updateDirection", 0);
    }
    if (coupled) direct("coupleImpulse", 1);
    direct("project", Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    pass.end();
  }

  async rebuildFromState(dt_s: number, bodies?: RigidBodyState[]) {
    const { nx, ny, nz } = this.dims;
    const scalarBytes = 48 + 12 * 8 * 4;
    const h = { x: this.scene.container.width_m / nx, y: this.scene.container.height_m / ny, z: this.scene.container.depth_m / nz };
    const activeBodies = bodies ?? this.coupling?.bodies;
    const builder = new WebGPUQuadtreeBuilder(this.device, this.dims, h, this.options.maximumLeafSize, this.options.adaptivityStrength, 3, this.gpuCache.construction);
    this.gpuCache.construction = builder.cache;
    const built = await builder.build({
      velocity: this.resources.velocityOut, volume: this.resources.volume, levelSet: this.levelSetTexture, dt_s,
      explicitSizing: initialSizing(this.scene, nx, nz, h, activeBodies), diagnosticBuffer: this.scalarBuffer, diagnosticBytes: scalarBytes
    });
    const cpuStartedAt = performance.now();
    // The GPU owns advection, VOF reconciliation, redistancing, vertical
    // sizing, and the complete subdivision/smoothing loop. Sparse CSR/IC
    // packing remains host-side because its graph changes with the tall grid.
    const redistanceStartedAt = performance.now();
    const phi = built.advectedPhi, mismatchFraction = built.mismatchFraction;
    const redistance_ms = performance.now() - redistanceStartedAt;
    const decodeStartedAt = performance.now();
    const quadtree = quadtreeFromPackedCells(built.packedCells, nx, nz);
    const quadtreeDecode_ms = performance.now() - decodeStartedAt;
    const tallGridStartedAt = performance.now();
    const opticalDepth = Math.max(1, Math.ceil(ny * this.scene.container.fillFraction * this.options.opticalDepthFraction));
    const pressureGrid = populateTallPressureGrid(quadtree, phi, ny, h, opticalDepth);
    const topologyWords = pressureTopologyWords(pressureGrid);
    const tallGrid_ms = performance.now() - tallGridStartedAt;
    const nextCoupling = this.coupling ? { bodies: activeBodies ?? this.coupling.bodies, dynamic: this.coupling.dynamic } : undefined;
    const reuseTopology = !this.coupling && sameWords(this.topologyWords, topologyWords);
    let next: WebGPUQuadtreeTallCellProjection;
    if (reuseTopology) {
      const uploadStartedAt = performance.now();
      uploadLevelSet(this.device, this.levelSetTexture, phi, nx, ny, nz);
      this.facesDirty = true;
      this.info.cpuTallGrid_ms = tallGrid_ms;
      this.info.cpuVariationalAssembly_ms = 0;
      this.info.cpuSystemPack_ms = 0;
      this.info.cpuResourceUpload_ms = performance.now() - uploadStartedAt;
      this.info.topologyReused = true;
      this.info.topologyReuseCount = (this.info.topologyReuseCount ?? 0) + 1;
      // Returning the same projection is the cache-hit signal consumed by the
      // owner; it must not retire the still-resident buffers after the swap.
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      next = this;
    } else {
      const fields: ProjectionFields = { phi, quadtree, pressureGrid, topologyWords };
      next = this.gpuCache.pipelines
        ? new WebGPUQuadtreeTallCellProjection(this.device, this.scene, this.dims, this.resources, this.options, fields, nextCoupling, false, this.gpuCache)
        : await WebGPUQuadtreeTallCellProjection.createAsync(this.device, this.scene, this.dims, this.resources, this.options, fields, nextCoupling, undefined, this.gpuCache);
      next.info.cpuTallGrid_ms = tallGrid_ms;
      next.info.topologyReused = false;
      next.info.topologyReuseCount = this.info.topologyReuseCount ?? 0;
    }
    next.levelSetMismatchFraction = mismatchFraction;
    next.info.gpuConstruction_ms = built.gpuWall_ms;
    next.info.gpuConstructionKernel_ms = built.gpuKernel_ms;
    next.info.cpuRedistance_ms = redistance_ms;
    next.info.cpuQuadtreeDecode_ms = quadtreeDecode_ms;
    next.info.cpuTopologyPack_ms = performance.now() - cpuStartedAt;
    next.info.topologyReadbackBytes = nx * ny * nz * 4 + nx * nz * 4 + scalarBytes + 4 + 16;
    const solve = built.diagnostics;
    next.info.pressureIterationsUsed = Math.round(solve[9] ?? 0);
    next.lastRelativeResidual = Math.sqrt(Math.max(0, solve[7]) / Math.max(1e-30, solve[3]));
    next.lastResidualRms = Math.sqrt(Math.max(0, solve[7]) / Math.max(1, this.dofCount));
    next.lastInitialResidualRms = Math.sqrt(Math.max(0, solve[3]) / Math.max(1, this.dofCount));
    if (this.coupling?.dynamic && this.couplingBodyCount > 0) {
      // coupleImpulse leaves K^T p per body past the CG scalars; the physical
      // impulse on the body is -rho K^T p (torque likewise).
      const rho = this.scene.fluid.density_kg_m3;
      next.bodyPressureImpulses = this.couplingBodyIndices.map((bodyIndex, slot) => {
        const base = 12 + slot * 8, body = this.coupling!.bodies[bodyIndex];
        return {
          bodyId: body?.description.id ?? `body-${bodyIndex}`,
          impulse_N_s: { x: -rho * solve[base], y: -rho * solve[base + 1], z: -rho * solve[base + 2] },
          angularImpulse_N_m_s: { x: -rho * solve[base + 3], y: -rho * solve[base + 4], z: -rho * solve[base + 5] },
          displacedVolume_m3: this.displacedVolumes[bodyIndex] ?? 0
        };
      });
    }
    return next;
  }

  get relativeResidual() { return this.lastRelativeResidual; }
  get residualRms() { return this.lastResidualRms; }
  get initialResidualRms() { return this.lastInitialResidualRms; }
  get topologyTexture() { return this.cellTopology; }

  destroy() { for (const buffer of this.buffers) buffer.destroy(); this.params.destroy(); this.cellProjection.destroy(); this.cellTopology.destroy(); this.factorAux.destroy(); this.mlsRowIndex.destroy(); this.levelSetTexture.destroy(); }
}
